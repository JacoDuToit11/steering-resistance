"""Run the full experiment (or selected stages) with ONE model load.

Usage:
    python scripts/run.py configs/smoke_0.5b.yaml
    python scripts/run.py configs/qwen7b_popqa.yaml --stages vectors,data,eval_m0
    python scripts/run.py configs/qwen7b_popqa.yaml --stages train,eval_m1
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from steer.common import load_config
from steer.pipeline import STAGE_ORDER, run_stages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to a configs/*.yaml")
    ap.add_argument(
        "--stages",
        default=",".join(STAGE_ORDER),
        help=f"comma-separated subset of {STAGE_ORDER} (default: all)",
    )
    args = ap.parse_args()

    t0 = time.time()
    run_stages(load_config(args.config), [s.strip() for s in args.stages.split(",") if s.strip()])
    print(f"\npipeline done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
