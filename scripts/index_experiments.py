"""Build a cross-experiment index from the Hugging Face Hub.

Each real run pushes one HF model repo (adapter + run/ dir + model card),
tagged `steering-resistance`. This lists every such repo under your account and
renders EXPERIMENTS.md — a single table of every experiment, its result, the
model it produced, and the exact commit. The Hub is the durable source of
truth; this is just a view, so it works from anywhere and survives ephemeral
(Colab) boxes. Run it whenever you want the index refreshed:

    python scripts/index_experiments.py --author <hf-username>
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
TAG = "steering-resistance"


def _meta_for(repo_id: str) -> dict:
    """Best-effort fetch of run/run_meta.json from a model repo (empty on miss)."""
    try:
        p = hf_hub_download(repo_id, "run/run_meta.json", repo_type="model")
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001 — a repo without our run dir just gets a sparse row
        return {}


def build_rows(author: str) -> list[dict]:
    api = HfApi()
    rows = []
    for m in api.list_models(author=author, tags=TAG):
        meta = _meta_for(m.id)
        git = meta.get("git", {})
        rows.append(
            {
                "run": meta.get("run") or m.id.split("/")[-1],
                "date": (meta.get("finished_at") or meta.get("started_at") or "")[:10],
                "model_id": meta.get("config", {}).get("model_id", "—"),
                "result": meta.get("headline") or "—",
                "repo": m.id,
                "commit": (git.get("commit") or "")[:12] + ("+dirty" if git.get("dirty") else ""),
            }
        )
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows


def render_md(rows: list[dict], author: str) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Experiments",
        "",
        f"Auto-generated from the Hugging Face Hub (`{author}`, tag `{TAG}`) by "
        "`scripts/index_experiments.py`. The Hub model repos are the source of "
        "truth — each has the adapter, the full run directory, and a model card.",
        f"\n_Last refreshed: {now} · {len(rows)} experiments._\n",
        "| date | run | base model | result | adapter | commit |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['date']} | {r['run']} | `{r['model_id']}` | {r['result']} | "
            f"[{r['repo']}](https://huggingface.co/{r['repo']}) | `{r['commit']}` |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--author", required=True, help="your HF username")
    ap.add_argument("--out", default=str(REPO_ROOT / "EXPERIMENTS.md"))
    args = ap.parse_args()

    rows = build_rows(args.author)
    if not rows:
        print(f"no models tagged '{TAG}' under '{args.author}' — run an experiment with hub_repo_id set first.")
        sys.exit(0)
    Path(args.out).write_text(render_md(rows, args.author))
    print(f"indexed {len(rows)} experiments -> {args.out}")


if __name__ == "__main__":
    main()
