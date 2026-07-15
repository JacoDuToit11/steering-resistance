"""Push a finished run's adapter + provenance to the Hugging Face Hub.

The pipeline auto-pushes after train and eval_m1 when hub_repo_id is set; this
script is the manual path — first backup of an old run, retry after a failed
upload, or card refresh after re-analysis.

Usage:
    python scripts/push_to_hub.py configs/qwen7b_popqa.yaml
    python scripts/push_to_hub.py configs/qwen7b_popqa.yaml --repo-id you/name
Auth: `hf auth login` once, or HF_TOKEN in the environment.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from steer.common import load_config
from steer.hub import push_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to the run's configs/*.yaml")
    ap.add_argument("--repo-id", default=None, help="override cfg hub_repo_id (user/name)")
    args = ap.parse_args()
    push_run(load_config(args.config), repo_id=args.repo_id)


if __name__ == "__main__":
    main()
