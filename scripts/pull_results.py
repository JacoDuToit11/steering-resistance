"""Pull an experiment's results from the HF Hub and print a clean M0-vs-M1 table.

Auth is read from your environment — NEVER pass the token as an argument or paste
it into a shared session. Either of these makes private repos readable:

    hf auth login                    # once, interactive; token cached locally
    export HF_TOKEN=hf_...            # in your own shell

Then:

    python scripts/pull_results.py JacoDuToit/steer-qwen3b   # one experiment
    python scripts/pull_results.py --author JacoDuToit       # every experiment
"""

import argparse
import csv
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

TAG = "steering-resistance"
# conditions where a "steered" rate is meaningful (an attack concept was injected)
_STEERED_CONDITIONS = {"steer_train", "steer_heldout", "steer_random", "steer_ortho"}


def experiment_repos(author: str) -> list[str]:
    """Model repos under `author` that are our experiments — matched by the
    `steer-` naming convention or the steering-resistance tag. Robust across
    huggingface_hub versions (avoids the removed list_models(tags=...) kwarg)."""
    out = []
    for m in HfApi().list_models(author=author):
        name = m.id.split("/")[-1]
        tags = getattr(m, "tags", None) or []
        if name.startswith("steer-") or TAG in tags:
            out.append(m.id)
    return sorted(out)


def _download(repo_id: str, filename: str) -> Path | None:
    try:
        return Path(hf_hub_download(repo_id, filename, repo_type="model"))
    except Exception:  # noqa: BLE001 — missing file / no access -> handled by caller
        return None


def _pct(row: dict | None, outcome: str) -> str:
    return f"{float(row[outcome + '_pct']) * 100:.0f}%" if row else "—"


def print_capability(repo_id: str):
    """If the run pushed a capability summary, print MMLU/GSM8K M0->M1 + delta."""
    p = _download(repo_id, "run/capability/capability_summary.csv")
    if not p:
        return
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return
    print(f"\n{'capability':<15}{'':>6}   {'M0 -> M1':>20}   {'delta (pp)':>12}")
    print("-" * 60)
    for r in rows:
        m0 = f"{r['m0']}%" if r.get("m0") else "n/a"
        m1 = f"{r['m1']}%" if r.get("m1") else "n/a"
        print(f"{r['benchmark']:<15}{'':>6}   {m0 + ' -> ' + m1:>20}   {r.get('delta_pp', ''):>12}")


def pull_one(repo_id: str):
    print(f"\n=== {repo_id} ===")
    meta_p = _download(repo_id, "run/run_meta.json")
    meta = json.loads(meta_p.read_text()) if meta_p else {}
    if meta:
        g, env = meta.get("git", {}), meta.get("env", {})
        commit = (g.get("commit") or "")[:12] + ("+dirty" if g.get("dirty") else "")
        print(f"model  : {meta.get('config', {}).get('model_id', '?')}")
        print(f"result : {meta.get('headline') or '—'}")
        print(f"run    : {(meta.get('finished_at') or '')[:19]} · {env.get('accelerator', '?')} · commit {commit}")

    csv_p = _download(repo_id, "run/summary.csv")
    if not csv_p:
        print("(no run/summary.csv — is the repo private and your token unset, or did eval not run?)")
        return

    rows = list(csv.DictReader(open(csv_p)))
    by = {}  # (condition, alpha) -> {model_name: row}
    for r in rows:
        by.setdefault((r["condition"], float(r["alpha"])), {})[r["model"]] = r

    print(f"\n{'condition':<15}{'alpha':>6}   {'correct  M0 -> M1':>20}   {'steered  M0 -> M1':>20}")
    print("-" * 68)
    for cond, a in sorted(by, key=lambda k: (k[0], k[1])):
        m0, m1 = by[(cond, a)].get("M0"), by[(cond, a)].get("M1")
        correct = f"{_pct(m0, 'correct'):>6} -> {_pct(m1, 'correct'):<6}"
        steered = f"{_pct(m0, 'steered'):>6} -> {_pct(m1, 'steered'):<6}" if cond in _STEERED_CONDITIONS else ""
        print(f"{cond:<15}{a:>6}   {correct:>20}   {steered:>20}")

    print_capability(repo_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo_id", nargs="?", help="a model repo, e.g. JacoDuToit/steer-qwen3b")
    ap.add_argument("--author", help="pull every steering-resistance model under this HF user")
    args = ap.parse_args()

    if args.author:
        repos = experiment_repos(args.author)
        if not repos:
            print(f"no steer- / '{TAG}' models under '{args.author}' (wrong username, or token can't see them?).")
            return
        for r in repos:
            pull_one(r)
    elif args.repo_id:
        pull_one(args.repo_id)
    else:
        ap.error("give a repo_id (e.g. JacoDuToit/steer-qwen3b) or --author <hf-user>")


if __name__ == "__main__":
    main()
