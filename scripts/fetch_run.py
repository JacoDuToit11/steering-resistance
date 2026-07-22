"""Restore a run from its HF model repo so you can resume eval WITHOUT retraining.

When a long run dies after training (the adapter is pushed the moment training
ends) but before the eval finishes, this pulls the pieces back into place:
  - the adapter (repo root)          -> cfg['adapter_dir']
  - the run/ directory               -> cfg['results_dir']
    (vectors.pt, eval_questions.json, eval_m0.jsonl, config.yaml, ...)

so the same vectors and held-out split are reused. Then finish the eval cheaply:

    python scripts/fetch_run.py configs/paper_3b.yaml --repo-id JacoDuToit/steer-paper_3b
    python scripts/run.py configs/paper_3b.yaml --stages eval_m1
    python scripts/eval_capability.py configs/paper_3b.yaml
    python scripts/push_to_hub.py configs/paper_3b.yaml --repo-id JacoDuToit/steer-paper_3b
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from huggingface_hub import snapshot_download

from steer.common import load_config

ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the run's configs/*.yaml")
    ap.add_argument("--repo-id", default=None, help="HF model repo (default: cfg hub_repo_id)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = args.repo_id or cfg.get("hub_repo_id")
    if not repo:
        sys.exit("no --repo-id given and no hub_repo_id in config")

    src = Path(snapshot_download(repo, repo_type="model"))
    adapter_dir = Path(cfg["adapter_dir"])
    results_dir = Path(cfg["results_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    n_adapter = 0
    for name in ADAPTER_FILES:
        f = src / name
        if f.exists():
            shutil.copyfile(f, adapter_dir / name)
            n_adapter += 1
    if not (adapter_dir / "adapter_config.json").exists():
        sys.exit(f"no adapter found in {repo} — was training pushed?")

    run_dir = src / "run"
    n_run = 0
    if run_dir.is_dir():
        for f in run_dir.rglob("*"):
            if f.is_file():
                dst = results_dir / f.relative_to(run_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst)
                n_run += 1

    print(f"restored {n_adapter} adapter files -> {adapter_dir}")
    print(f"restored {n_run} run files -> {results_dir}")
    print("\nnow finish the eval (no retrain):")
    print(f"  python scripts/run.py {args.config} --stages eval_m1")
    print(f"  python scripts/eval_capability.py {args.config}")
    print(f"  python scripts/push_to_hub.py {args.config} --repo-id {repo}")


if __name__ == "__main__":
    main()
