"""Capability-retention eval (MMLU + GSM8K): does resistance SFT keep the model smart?

Mirrors the capability check in "Steering Awareness" (arXiv:2511.21399, App. E.4):
general-capability benchmarks via EleutherAI's lm-evaluation-harness, comparing
the base model (M0) against the resist adapter (M1 = base + adapter). In the paper
the *detection* LoRA cost a lot of capability (Gemma-2-9B: MMLU 73.9->51.1, GSM8K
82.8->13.0); the question here is whether our *resistance* adapter degrades less.

Paper-exact settings (App. E.4): MMLU 5-shot multiple-choice; GSM8K 8-shot
chain-of-thought; greedy decoding (temperature 0); accuracy on the test split.

  - MMLU      -> harness task `mmlu`      (57 subjects, multiple_choice, acc), 5-shot
  - GSM8K CoT -> harness task `gsm8k_cot` (generate_until, exact_match),       8-shot

--num_fewshot is global per harness run, so we launch ONE lm_eval process per
(model, task). lm-eval is heavy and its deps can clash with the pinned ones, so
by default we run it from a separate interpreter (cfg['lmeval_python'], default
.venv-lmeval); if that doesn't exist we fall back to THIS interpreter, so on
Colab you can just `pip install lm-eval` and run. See docs/CAPABILITY.md.

Usage:
  python scripts/eval_capability.py [configs/qwen3b.yaml] \
      [--models m0,m1] [--tasks mmlu,gsm8k_cot] [--limit N] [--chat-template] [--force]

M0 = base cfg['model_id']; M1 = base + cfg['adapter_dir'] (applied via PEFT).
Writes harness JSON under <results_dir>/capability/<model>/<task>/ and a combined
capability_summary.{md,csv}. Resumable: a (model, task) with results JSON is
skipped unless --force.
"""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from steer.common import load_config

REPO = Path(__file__).resolve().parent.parent
GENERATION_TASKS = {"gsm8k_cot", "gsm8k"}  # need explicit greedy decoding


def capability_dir(cfg: dict) -> Path:
    return REPO / (cfg.get("capability_dir") or f"{cfg['results_dir']}/capability")


def lmeval_python(cfg: dict) -> str:
    """Interpreter to run lm-eval in: config override, else a local .venv-lmeval,
    else THIS interpreter (Colab: just `pip install lm-eval` and go)."""
    if cfg.get("lmeval_python"):
        return cfg["lmeval_python"]
    sep = REPO / ".venv-lmeval/bin/python"
    return str(sep) if sep.exists() else sys.executable


def parse_limit(spec):
    """None | int | 'N' | 'mmlu=15,gsm8k_cot=200' -> None | int | {task: int}.

    'full'/'none'/'all'/'' -> None (run the whole test set; the opt-in that beats
    a fast-subset default set in the config)."""
    if spec is None or isinstance(spec, (int, dict)):
        return spec
    if spec.strip().lower() in ("full", "none", "all", ""):
        return None
    if "=" in spec:
        return {p.split("=")[0].strip(): int(p.split("=")[1]) for p in spec.split(",") if p}
    return int(spec)


def limit_for(limit, task):
    return limit.get(task) if isinstance(limit, dict) else limit


def model_args(cfg: dict, model: str) -> str:
    """lm-eval --model_args for M0 (base) or M1 (base + resist adapter)."""
    args = f"pretrained={cfg['model_id']},dtype=bfloat16"
    if model == "m1":
        adapter = (REPO / cfg["adapter_dir"]).resolve()
        if not adapter.exists():
            sys.exit(f"adapter_dir not found: {adapter} (train M1 first: run.py --stages train)")
        args += f",peft={adapter}"
    return args


def results_json(out_dir: Path):
    """Newest harness results_*.json under out_dir (harness nests one dir per model name)."""
    hits = sorted(glob.glob(str(out_dir / "**" / "results_*.json"), recursive=True))
    return Path(hits[-1]) if hits else None


def run_one(cfg: dict, model: str, task: str, fewshot: int, args, logf) -> Path:
    """Launch one lm_eval process for (model, task); returns its output dir."""
    out_dir = capability_dir(cfg) / model / task
    existing = results_json(out_dir)
    if existing and not args.force:
        print(f"  [skip] {model}/{task}: found {existing.name} (use --force to rerun)")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        lmeval_python(cfg), "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args(cfg, model),
        "--tasks", task,
        "--num_fewshot", str(fewshot),
        "--batch_size", str(cfg.get("capability_batch_size", "auto")),
        "--seed", str(cfg.get("seed", 0)),
        "--output_path", str(out_dir),
    ]
    if task in GENERATION_TASKS:
        cmd += ["--gen_kwargs", "do_sample=False"]  # greedy / temperature 0 (paper)
    if args.chat_template or cfg.get("capability_apply_chat_template", False):
        cmd += ["--apply_chat_template", "--fewshot_as_multiturn"]
    eff_limit = limit_for(args._limit, task)
    if eff_limit is None:
        eff_limit = limit_for(parse_limit(cfg.get("capability_limit")), task)
    if eff_limit is not None:
        cmd += ["--limit", str(eff_limit)]

    print(f"\n=== {model} / {task} ({fewshot}-shot) ===\n  {' '.join(cmd)}")
    logf.write(f"\n\n########## {model} / {task} ##########\n{' '.join(cmd)}\n")
    logf.flush()
    rc = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"  [FAIL] {model}/{task} exited {rc} — see log")
    return out_dir


# metric key we treat as the headline number for each task
PRIMARY = {
    "mmlu": ["acc,none"],
    "gsm8k_cot": ["exact_match,strict-match", "exact_match,flexible-extract"],
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract"],
}


def read_metric(out_dir: Path, task: str):
    """(value, metric_key) headline accuracy for `task`, or (None, None)."""
    rj = results_json(out_dir)
    if not rj:
        return None, None
    res = json.loads(rj.read_text()).get("results", {})
    row = res.get(task) or next((v for k, v in res.items() if k.startswith(task)), {})
    for key in PRIMARY.get(task, []):
        if key in row:
            return row[key], key
    for key, val in row.items():  # fallback: first accuracy-like metric
        if isinstance(val, (int, float)) and ("acc" in key or "exact_match" in key):
            return val, key
    return None, None


def write_summary(cfg, tasks, models, cap_dir):
    import csv as csvmod

    pair = set(models) == {"m0", "m1"}
    # collect once: task -> (metric_key, {model: pct-or-None}, delta_pp-or-None)
    summary = {}
    for task in tasks:
        got = {m: read_metric(cap_dir / m / task, task) for m in models}
        vals = {m: (got[m][0] * 100 if got[m][0] is not None else None) for m in models}
        key = next((got[m][1] for m in models if got[m][1]), "-")
        delta = (vals["m1"] - vals["m0"]) if pair and vals.get("m0") is not None and vals.get("m1") is not None else None
        summary[task] = (key, vals, delta)

    header = f"{'benchmark':<14}{'metric':<26}" + "".join(f"{m.upper():>9}" for m in models) + (f"{'delta':>9}" if pair else "")
    print("\n" + "=" * len(header) + "\n" + header + "\n" + "-" * len(header))
    for task, (key, vals, delta) in summary.items():
        cells = "".join(f"{vals[m]:>8.1f}%" if vals[m] is not None else f"{'n/a':>9}" for m in models)
        line = f"{task:<14}{key:<26}{cells}" + (f"{delta:>+8.1f}" if delta is not None else "")
        print(line)

    md = ["| benchmark | metric | " + " | ".join(m.upper() for m in models) + (" | delta (pp) |" if pair else " |"),
          "|" + "---|" * (len(models) + 2 + (1 if pair else 0))]
    for task, (key, vals, delta) in summary.items():
        cells = [f"{vals[m]:.1f}%" if vals[m] is not None else "n/a" for m in models]
        if pair:
            cells.append(f"{delta:+.1f}" if delta is not None else "n/a")
        md.append("| " + task + " | " + key + " | " + " | ".join(cells) + " |")
    (cap_dir / "capability_summary.md").write_text(
        "# Capability retention: M0 (base) vs M1 (resist SFT)\n\n"
        "arXiv:2511.21399 App. E.4 methodology — lm-eval-harness, MMLU 5-shot MC, "
        "GSM8K 8-shot CoT, greedy, accuracy on the test split.\n\n" + "\n".join(md) + "\n")

    # proper CSV (metric keys like "acc,none" contain commas -> must be quoted)
    with open(cap_dir / "capability_summary.csv", "w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(["benchmark", "metric", *models] + (["delta_pp"] if pair else []))
        for task, (key, vals, delta) in summary.items():
            row = [task, key] + [f"{vals[m]:.2f}" if vals[m] is not None else "" for m in models]
            if pair:
                row.append(f"{delta:+.2f}" if delta is not None else "")
            w.writerow(row)
    print(f"\nsaved -> {cap_dir / 'capability_summary.md'} , {cap_dir / 'capability_summary.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="configs/qwen3b.yaml")
    ap.add_argument("--models", default="m0,m1", help="comma list of m0,m1")
    ap.add_argument("--tasks", default=None, help="override cfg capability_tasks (comma list)")
    ap.add_argument("--limit", default=None,
                    help="per-task cap: int for all, or 'mmlu=15,gsm8k_cot=200'. NOTE mmlu's limit "
                         "is PER-SUBJECT (x57 subjects); 'full' runs the whole set.")
    ap.add_argument("--chat-template", action="store_true", help="instruct-style: chat template + multiturn fewshot")
    ap.add_argument("--force", action="store_true", help="rerun even if results JSON exists")
    args = ap.parse_args()
    args._limit = parse_limit(args.limit)

    cfg = load_config(args.config)
    tasks = dict(cfg.get("capability_tasks", {"mmlu": 5, "gsm8k_cot": 8}))
    if args.tasks:
        tasks = {t: tasks.get(t, 5) for t in args.tasks.split(",")}
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    cap_dir = capability_dir(cfg)
    cap_dir.mkdir(parents=True, exist_ok=True)
    log_path = cap_dir / "lmeval_run.log"
    print(f"lm-eval interp: {lmeval_python(cfg)} | tasks: {tasks} | models: {models}")
    print(f"streaming harness output -> {log_path}")
    with open(log_path, "a") as logf:
        for model in models:
            for task, fewshot in tasks.items():
                run_one(cfg, model, task, fewshot, args, logf)
    write_summary(cfg, tasks, models, cap_dir)


if __name__ == "__main__":
    main()
