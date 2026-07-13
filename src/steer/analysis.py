"""Metrics + bootstrap CIs over questions; prints per-(model, condition, alpha) table."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from steer.common import read_jsonl


def bootstrap_ci(values_by_question: dict[str, list[float]], n_resamples: int, seed: int = 0):
    """Bootstrap over questions (cluster resampling): each question's rows stay together."""
    qids = sorted(values_by_question)
    rng = np.random.default_rng(seed)
    point = float(np.mean([v for q in qids for v in values_by_question[q]]))
    if len(qids) < 2:
        return point, point, point
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(qids, size=len(qids), replace=True)
        vals = [v for q in sample for v in values_by_question[q]]
        means.append(np.mean(vals))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return point, float(lo), float(hi)


def summarize(paths: list[str | Path], n_resamples: int = 10000) -> list[dict]:
    rows = [r for p in paths for r in read_jsonl(p)]
    groups = defaultdict(lambda: defaultdict(list))  # (model, cond, alpha) -> qid -> outcomes
    for r in rows:
        groups[(r["model"], r["condition"], r["alpha"])][r["question_id"]].append(r["outcome"])

    out = []
    for (model, cond, alpha), by_q in sorted(groups.items()):
        n = sum(len(v) for v in by_q.values())
        summary = {"model": model, "condition": cond, "alpha": alpha, "n": n}
        for outcome in ["correct", "steered", "other"]:
            vals = {q: [1.0 if o == outcome else 0.0 for o in outs] for q, outs in by_q.items()}
            summary[outcome] = bootstrap_ci(vals, n_resamples)
        out.append(summary)
    return out


def print_table(summaries: list[dict]):
    def fmt(t):
        p, lo, hi = t
        return f"{p:.0%} [{lo:.0%},{hi:.0%}]"

    header = f"{'model':<10} {'condition':<15} {'alpha':>5} {'n':>4} | {'correct':<16} {'steered':<16} {'other':<16}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['model']:<10} {s['condition']:<15} {s['alpha']:>5} {s['n']:>4} | "
            f"{fmt(s['correct']):<16} {fmt(s['steered']):<16} {fmt(s['other']):<16}"
        )


def write_summary(summaries: list[dict], out_dir: str | Path) -> tuple[Path, Path]:
    """Persist the summary as summary.csv (spreadsheet) and summary.md (skimmable).

    Returns (csv_path, md_path). Same numbers as print_table, just saved so the
    headline result isn't stdout-only.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / "summary.csv", out_dir / "summary.md"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["model", "condition", "alpha", "n"]
            + [f"{o}_{s}" for o in ("correct", "steered", "other") for s in ("pct", "lo", "hi")]
        )
        for s in summaries:
            row = [s["model"], s["condition"], s["alpha"], s["n"]]
            for o in ("correct", "steered", "other"):
                row += [round(x, 4) for x in s[o]]
            w.writerow(row)

    def fmt(t):
        p, lo, hi = t
        return f"{p:.0%} [{lo:.0%},{hi:.0%}]"

    with open(md_path, "w") as f:
        f.write("| model | condition | alpha | n | correct | steered | other |\n")
        f.write("|---|---|---:|---:|---|---|---|\n")
        for s in summaries:
            f.write(
                f"| {s['model']} | {s['condition']} | {s['alpha']} | {s['n']} | "
                f"{fmt(s['correct'])} | {fmt(s['steered'])} | {fmt(s['other'])} |\n"
            )
    return csv_path, md_path


if __name__ == "__main__":
    # default: every run under results/<run>/ (all runs are namespaced)
    paths = sys.argv[1:] or sorted(Path("results").glob("*/eval_*.jsonl"))
    print_table(summarize([str(p) for p in paths]))
