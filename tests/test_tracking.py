"""Fast unit tests for provenance + tracking logic (no model, no GPU, no network; <2s).

Run:  python tests/test_tracking.py      (or: pytest)
Covers: git/env capture, input+artifact hashing, manifest write/finalize cycle,
model-card generation, and the Tracker no-op guarantee (tracking must never be
able to crash a science run).
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from steer.hub import build_model_card
from steer.tracking import (
    Tracker,
    artifact_hashes,
    data_fingerprints,
    env_info,
    eval_summary_metrics,
    finalize_invocation,
    git_info,
    sha256_file,
    snapshot_invocation,
)


def _tmp_cfg(tmp: Path) -> dict:
    concepts = tmp / "concepts.json"
    concepts.write_text('{"concepts": []}')
    return {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "seed": 0,
        "layer": 8,
        "lora_r": 16,
        "lr": 1e-4,
        "epochs": 1,
        "qa_source": "popqa",
        "popqa_relations": ["capital"],
        "popqa_max_questions": 800,
        "concepts_path": str(concepts),
        "results_dir": str(tmp / "results" / "testrun"),
        "adapter_dir": str(tmp / "results" / "testrun" / "m1_resist_adapter"),
    }


def test_git_info_in_this_repo():
    g = git_info(Path(__file__).resolve().parent.parent)
    assert g["commit"] and len(g["commit"]) == 40
    assert isinstance(g["dirty"], bool)
    assert g["dirty"] == (g["diff"] is not None), "diff present iff dirty"
    g2 = git_info(tempfile.gettempdir())  # not a repo root — must not raise
    assert isinstance(g2, dict)


def test_env_info_versions():
    e = env_info()
    assert e["python"].count(".") >= 1
    assert e["packages"]["torch"], "torch version must be captured"
    assert "wandb" in e["packages"]  # key present even when not installed (None)


def test_sha256_and_fingerprints():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        f = tmp / "x.json"
        f.write_text("hello")
        assert sha256_file(f) == hashlib.sha256(b"hello").hexdigest()
        fp = data_fingerprints(_tmp_cfg(tmp))
        assert fp["concepts_path"]["sha256"] == sha256_file(tmp / "concepts.json")
        assert fp["popqa"]["dataset"] == "akariasai/PopQA"
        assert fp["popqa"]["relations"] == ["capital"]


def test_snapshot_finalize_cycle():
    with tempfile.TemporaryDirectory() as td:
        cfg = _tmp_cfg(Path(td))
        results_dir = Path(cfg["results_dir"])
        meta = snapshot_invocation(cfg, None, ["train", "eval_m1"])
        on_disk = json.loads((results_dir / "run_meta.json").read_text())
        assert on_disk["status"] == "running" and on_disk["stages"] == ["train", "eval_m1"]
        assert on_disk["config"]["lora_r"] == 16, "manifest must embed the resolved config"
        assert "diff" not in on_disk["git"], "diff goes to code.patch, not the manifest"

        (results_dir / "summary.csv").write_text("model,condition\n")  # a fake artifact
        finalize_invocation(meta, cfg, status="success", wandb_url="https://wandb.ai/x/y/z")
        on_disk = json.loads((results_dir / "run_meta.json").read_text())
        assert on_disk["status"] == "success" and on_disk["wandb_url"].startswith("https://wandb.ai")
        assert "summary.csv" in on_disk["artifacts"]
        assert "run_meta.json" not in on_disk["artifacts"], "manifest must not hash itself"
        hist = [json.loads(l) for l in (results_dir / "invocations.jsonl").read_text().splitlines()]
        assert len(hist) == 1 and hist[0]["status"] == "success"


def test_model_card_contents():
    with tempfile.TemporaryDirectory() as td:
        cfg = _tmp_cfg(Path(td))
        meta = snapshot_invocation(cfg, "configs/qwen7b_popqa.yaml", ["train"])
        card = build_model_card(cfg, meta, "| model | correct |\n|---|---|\n| M1 | 95% |")
        assert card.startswith("---\n"), "Hub frontmatter must lead the card"
        assert f"base_model: {cfg['model_id']}" in card
        assert "library_name: peft" in card
        assert "akariasai/PopQA" in card
        assert meta["git"]["commit"][:12] in card, "card must pin the code commit"
        assert "| lora_r | `16` |" in card
        assert "| M1 | 95% |" in card, "eval summary table must be embedded"
        no_eval = build_model_card(cfg, meta, None)
        assert "No eval summary at push time" in no_eval


def test_tracker_is_noop_safe():
    t = Tracker.start({"wandb_project": None}, {"run": "x", "git": {}}, ["train"])
    assert t.url is None
    t.log({"train/loss": 1.0}, step=1)  # all must be silent no-ops
    t.summary({"a": 1})
    t.finish("success")
    # enabled-but-broken (wandb missing / not logged in) must degrade, not raise
    t2 = Tracker.start(
        {"wandb_project": "p", "wandb_entity": None, "results_dir": tempfile.gettempdir()},
        {"run": "x", "git": {"commit": "c", "dirty": False}},
        ["train"],
    )
    t2.log({"train/loss": 1.0})
    t2.finish("crashed")


def test_eval_summary_metrics_flatten():
    summaries = [
        {"model": "M1", "condition": "steer_heldout", "alpha": 0.6, "n": 40,
         "correct": (0.9, 0.8, 0.95), "steered": (0.05, 0.0, 0.1), "other": (0.05, 0.0, 0.1)},
        {"model": "M0", "condition": "clean", "alpha": 0.0, "n": 40,
         "correct": (1.0, 1.0, 1.0), "steered": (0.0, 0.0, 0.0), "other": (0.0, 0.0, 0.0)},
    ]  # fmt: skip
    m = eval_summary_metrics(summaries, "M1")
    assert m["eval/M1/steer_heldout/alpha0.6/correct"] == 0.9
    assert m["eval/M1/steer_heldout/alpha0.6/n"] == 40
    assert not any("/M0/" in k for k in m), "must filter to the requested model"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
