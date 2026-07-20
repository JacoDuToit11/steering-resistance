"""Run provenance + experiment tracking (PRINCIPLES §6/§9).

Every pipeline invocation writes results_dir/run_meta.json — the full snapshot
that ties the run to everything needed to reproduce it: resolved config, git
commit (+ code.patch when the tree is dirty), package/hardware versions, input
data hashes, and (on finalize) a hash of every artifact in the run directory.
A compact one-line history accumulates in results_dir/invocations.jsonl.

The run directory stays the source of truth; Weights & Biases (Tracker) and the
HF Hub (steer.hub) are mirrors of it. Tracking is fail-soft by design: a
missing wandb install or an offline box prints a loud warning and the science
run proceeds untracked, never crashes.
"""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from steer.common import append_jsonl

PROVENANCE_FILES = ["run_meta.json", "invocations.jsonl", "config.yaml", "code.patch"]
REPO_ROOT = Path(__file__).resolve().parents[2]  # editable install: the checkout this code runs from
_TRACKED_PACKAGES = ["torch", "transformers", "peft", "accelerate", "datasets", "numpy", "wandb", "huggingface_hub"]


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_info(repo_root: str | Path = REPO_ROOT) -> dict:
    """Commit/branch/dirty state (+ diff text when dirty); all-None outside a repo."""
    root = Path(repo_root)
    commit = _git(["rev-parse", "HEAD"], root)
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None, "remote": None, "diff": None}
    status = _git(["status", "--porcelain"], root) or ""
    dirty = bool(status)
    return {
        "commit": commit,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": dirty,
        "remote": _git(["remote", "get-url", "origin"], root),
        "diff": (_git(["diff", "HEAD"], root) or status) if dirty else None,
    }


def env_info() -> dict:
    """Python/platform/package versions + accelerator, for the manifest."""
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    accel = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            accel = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            accel = "mps"
    except Exception:  # noqa: BLE001 — env capture must never kill a run
        accel = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "accelerator": accel,
        "packages": versions,
    }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_fingerprints(cfg: dict) -> dict:
    """Hashes/identity of the run's data inputs (local files hashed, HF sets named)."""
    fp = {}
    for key in ("concepts_path", "qa_path"):
        p = cfg.get(key)
        if p and Path(p).exists():
            fp[key] = {"path": p, "sha256": sha256_file(p)}
    if cfg.get("qa_source") == "popqa":
        fp["popqa"] = {
            "dataset": "akariasai/PopQA",
            "split": "test",
            "relations": cfg.get("popqa_relations"),
            "max_questions": cfg.get("popqa_max_questions"),
        }
    return fp


def artifact_hashes(results_dir: str | Path) -> dict:
    """{relpath: {sha256, bytes}} for every artifact in the run dir (provenance and wandb files excluded)."""
    results_dir = Path(results_dir)
    out = {}
    for p in sorted(results_dir.rglob("*")):
        rel = p.relative_to(results_dir)
        if not p.is_file() or rel.parts[0] == "wandb" or str(rel) in PROVENANCE_FILES:
            continue
        out[str(rel)] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def snapshot_invocation(cfg: dict, config_path: str | Path | None, stages: list[str]) -> dict:
    """Write results_dir/run_meta.json (+ config copy, + code.patch when dirty) at invocation start.

    Written BEFORE any stage runs so a crashed run still carries full provenance.
    Returns the manifest dict; finalize_invocation() completes it.
    """
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    git = git_info()
    meta = {
        "run": results_dir.name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "stages": stages,
        "config_path": str(config_path) if config_path else None,
        "config": cfg,
        "git": {k: v for k, v in git.items() if k != "diff"},
        "env": env_info(),
        "data": data_fingerprints(cfg),
        "status": "running",
    }
    if git["dirty"]:
        (results_dir / "code.patch").write_text(git["diff"] + "\n")
        print(f"WARNING: git tree is dirty — uncommitted changes saved to {results_dir / 'code.patch'}")
    if config_path and Path(config_path).exists():
        shutil.copyfile(config_path, results_dir / "config.yaml")
    _write_meta(meta, results_dir)
    print(f"provenance -> {results_dir / 'run_meta.json'} (commit {(git['commit'] or 'none')[:12]}{'+dirty' if git['dirty'] else ''})")
    return meta


def _point(summaries: list[dict], model: str, condition: str, alpha: float, outcome: str):
    for s in summaries:
        if s["model"] == model and s["condition"] == condition and s["alpha"] == alpha:
            return s[outcome][0]
    return None


def headline(summaries: list[dict] | None) -> str:
    """Compact one-line M0->M1 result for the run record, best-effort.

    e.g. "clean 100%->94% · steer_heldout@0.8 correct 3%->17%". Empty string if
    the needed conditions aren't present (e.g. an eval-only or vectors-only run).
    """
    if not summaries:
        return ""
    parts = []
    c0, c1 = _point(summaries, "M0", "clean", 0.0, "correct"), _point(summaries, "M1", "clean", 0.0, "correct")
    if c0 is not None and c1 is not None:
        parts.append(f"clean {c0:.0%}->{c1:.0%}")
    alphas = sorted({s["alpha"] for s in summaries if s["condition"] == "steer_heldout"})
    if alphas:
        a = alphas[-1]
        s0, s1 = _point(summaries, "M0", "steer_heldout", a, "correct"), _point(summaries, "M1", "steer_heldout", a, "correct")
        if s0 is not None and s1 is not None:
            parts.append(f"steer_heldout@{a} correct {s0:.0%}->{s1:.0%}")
    return " · ".join(parts)


def finalize_invocation(meta: dict, cfg: dict, status: str, wandb_url: str | None = None, summaries: list[dict] | None = None):
    """Complete the manifest (status/duration/artifact hashes/headline) + append the history line."""
    results_dir = Path(cfg["results_dir"])
    meta["status"] = status
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["wandb_url"] = wandb_url
    meta["headline"] = headline(summaries)
    meta["artifacts"] = artifact_hashes(results_dir)
    _write_meta(meta, results_dir)
    append_jsonl(
        results_dir / "invocations.jsonl",
        {
            "time": meta["started_at"],
            "stages": meta["stages"],
            "commit": (meta["git"]["commit"] or "none")[:12] + ("+dirty" if meta["git"]["dirty"] else ""),
            "status": status,
            "headline": meta["headline"],
            "wandb_url": wandb_url,
        },
    )


def _write_meta(meta: dict, results_dir: Path):
    with open(results_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False, default=str)


class Tracker:
    """Weights & Biases logging that degrades to a no-op.

    Enabled by cfg['wandb_project']; anything that stops wandb from starting
    (not installed, not logged in, no network) prints one loud warning and every
    method becomes a no-op — tracking must never take down a GPU run.
    """

    def __init__(self, run=None):
        self.run = run

    @property
    def url(self) -> str | None:
        return self.run.url if self.run is not None else None

    @classmethod
    def start(cls, cfg: dict, meta: dict, stages: list[str]) -> "Tracker":
        project = cfg.get("wandb_project")
        if not project:
            return cls()
        try:
            import wandb

            run = wandb.init(
                project=project,
                entity=cfg.get("wandb_entity"),
                name=f"{meta['run']}:{'+'.join(stages)}",
                config={**cfg, "git_commit": meta["git"]["commit"], "git_dirty": meta["git"]["dirty"]},
                tags=["steering-resistance", meta["run"]],
                dir=cfg["results_dir"],
            )
            print(f"wandb run -> {run.url}")
            return cls(run)
        except Exception as e:  # noqa: BLE001 — see class docstring
            print(f"WARNING: wandb tracking disabled ({type(e).__name__}: {e}); run continues untracked")
            return cls()

    def log(self, data: dict, step: int | None = None):
        if self.run is not None:
            self.run.log(data, step=step)

    def summary(self, data: dict):
        if self.run is not None:
            self.run.summary.update(data)

    def finish(self, status: str = "success"):
        if self.run is not None:
            self.run.finish(exit_code=0 if status == "success" else 1)


def eval_summary_metrics(summaries: list[dict], model_name: str) -> dict:
    """Flatten analysis.summarize() rows into wandb-summary keys (point estimates)."""
    out = {}
    for s in summaries:
        if s["model"] != model_name:
            continue
        base = f"eval/{s['model']}/{s['condition']}/alpha{s['alpha']}"
        for outcome in ("correct", "steered", "other"):
            out[f"{base}/{outcome}"] = s[outcome][0]
        out[f"{base}/n"] = s["n"]
    return out
