"""Pipeline stages sharing ONE loaded model.

Each numbered script in scripts/ wraps exactly one of these stage functions, and
scripts/run_pipeline.py chains them with a single model load (the model was
previously reloaded from disk by every script — ~6 loads per experiment).

Canonical stage order: vectors -> data -> eval_m0 -> train -> eval_m1.
The order matters: vectors/data/eval_m0 must see the BASE model, train mutates
it in place (PEFT), eval_m1 sees the merged trained model.
"""

import json
import random
import time
from pathlib import Path

from steer.analysis import print_table, summarize, write_summary
from steer.common import load_model_and_tokenizer, set_seed
from steer.data import build_train_examples, clean_filter, load_questions, split_questions
from steer.eval import run_eval
from steer.tracking import Tracker, eval_summary_metrics, finalize_invocation, headline, snapshot_invocation
from steer.train import train_m1
from steer.vectors import build_vectors, efficacy_check, load_concepts, load_vectors, save_vectors

STAGE_ORDER = ["vectors", "data", "eval_m0", "train", "eval_m1"]


def stage_vectors(model, tok, cfg: dict) -> dict:
    """Build CAA vectors + mandatory per-concept efficacy gate; save bundle (drop failures)."""
    concepts_cfg = load_concepts(cfg["concepts_path"])
    print(f"Building CAA vectors at layer {cfg['layer']} ...")
    bundle = build_vectors(model, tok, concepts_cfg, cfg["layer"])
    print(f"mean residual norm at L{cfg['layer']}: {bundle['mean_norm']:.2f}")

    alpha = cfg["efficacy_alpha"]
    print(f"\nEfficacy check (alpha={alpha}, threshold={cfg['efficacy_min_rate']}):")
    rates = efficacy_check(model, tok, bundle, alpha)
    keep, drop = [], []
    for c in concepts_cfg["concepts"]:
        name = c["name"]
        ok = rates[name] >= cfg["efficacy_min_rate"]
        (keep if ok else drop).append(name)
        print(f"  {name:<10} ({c['category']}, {c['split']:<7}): {rates[name]:.0%}  {'OK' if ok else 'DROP'}")

    bundle["efficacy_rates"] = rates
    bundle["efficacy_alpha"] = alpha
    for name in drop:
        del bundle["vectors"][name]
        del bundle["unit_vectors"][name]
    bundle["concepts"] = [c for c in concepts_cfg["concepts"] if c["name"] in keep]

    save_vectors(bundle, cfg["vectors_path"])
    print(f"\nKept {len(keep)}/{len(rates)} vectors -> {cfg['vectors_path']}")
    if drop:
        print("Dropped:", drop)
    return bundle


def stage_data(model, tok, cfg: dict):
    """Clean-filter QA on the base model, split questions, build training examples."""
    rng = random.Random(cfg["seed"])
    questions = load_questions(cfg)
    print(f"{len(questions)} questions ({cfg.get('qa_source', 'json')}); running clean-correctness filter ...")
    kept = clean_filter(model, tok, questions, cfg["max_new_tokens"])
    by_cat = {}
    for q in kept:
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1
    print(f"kept {len(kept)}/{len(questions)} answered correctly clean: {by_cat}")

    train_q, eval_q = split_questions(kept, cfg["eval_question_frac"], rng)
    print(f"split: {len(train_q)} train / {len(eval_q)} eval questions")

    bundle = load_vectors(cfg["vectors_path"])
    train_concepts = [c for c in bundle["concepts"] if c["split"] == "train"]
    examples = build_train_examples(train_q, train_concepts, cfg, rng)
    n_steered = sum(e["steered"] for e in examples)
    n_rel = sum(1 for e in examples if e["pairing"] == "relevant")
    n_alpaca = sum(1 for e in examples if e["category"] == "alpaca")
    n_clean = len(examples) - n_steered - n_alpaca
    print(
        f"{len(examples)} training examples: {n_steered} steered ({n_rel} relevant / "
        f"{n_steered - n_rel} irrelevant), {n_clean} clean, {n_alpaca} alpaca-replay"
    )

    Path(cfg["train_examples_path"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["train_examples_path"], "w") as f:
        json.dump(examples, f, indent=1, ensure_ascii=False)
    with open(cfg["eval_questions_path"], "w") as f:
        json.dump({"train": train_q, "eval": eval_q}, f, indent=1, ensure_ascii=False)
    print(f"wrote {cfg['train_examples_path']} and {cfg['eval_questions_path']}")


def stage_eval(model, tok, cfg: dict, model_name: str, with_m0_comparison: bool = False) -> list[dict]:
    """One eval sweep -> results_dir/eval_<name>.jsonl + printed/saved summary."""
    out = Path(cfg["results_dir"]) / f"eval_{model_name.lower()}.jsonl"
    run_eval(model, tok, cfg, model_name, out)
    paths = [out]
    if with_m0_comparison:
        m0 = Path(cfg["results_dir"]) / "eval_m0.jsonl"
        paths = [p for p in [m0, out] if p.exists()]
    summaries = summarize(paths, cfg["bootstrap_resamples"])
    print_table(summaries)
    csv_path, md_path = write_summary(summaries, cfg["results_dir"])
    print(f"\nsaved summary -> {csv_path} , {md_path}")
    return summaries


def _try_push(cfg: dict, meta: dict):
    """Hub backup, loud-but-nonfatal: a dead network must not kill the GPU run."""
    from steer.hub import push_run

    try:
        meta.setdefault("hub_url", push_run(cfg, meta))
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: hub push FAILED ({type(e).__name__}: {e}) — back up manually: python scripts/push_to_hub.py")


def run_stages(cfg: dict, stages: list[str], config_path: str | None = None):
    """Run the requested stages (canonical order enforced) with one model load."""
    unknown = set(stages) - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; valid: {STAGE_ORDER}")
    stages = [s for s in STAGE_ORDER if s in stages]
    set_seed(cfg["seed"])
    print(f"stages: {stages}")
    meta = snapshot_invocation(cfg, config_path, stages)
    tracker = Tracker.start(cfg, meta, stages)
    meta["wandb_url"] = tracker.url
    status = "crashed"
    last_summaries = None
    try:
        model, tok = load_model_and_tokenizer(cfg)

        trained = None
        for stage in stages:
            t0 = time.time()
            print(f"\n=== stage: {stage} ===")
            if stage == "vectors":
                stage_vectors(model, tok, cfg)
            elif stage == "data":
                stage_data(model, tok, cfg)
            elif stage == "eval_m0":
                last_summaries = stage_eval(model, tok, cfg, "M0")
                tracker.summary(eval_summary_metrics(last_summaries, "M0"))
            elif stage == "train":
                out = train_m1(model, tok, cfg, tracker)
                trained = out["model"]
                tracker.summary({"train/first_batch_loss": out["first_batch_loss"], "train/last_batch_loss": out["last_batch_loss"]})
                if cfg.get("hub_repo_id"):  # PRINCIPLES §9: off-box the moment training ends
                    _try_push(cfg, meta)
            elif stage == "eval_m1":
                if trained is not None:
                    eval_model = trained.merge_and_unload()
                    eval_model.eval()
                else:  # eval_m1 without train in this invocation: load adapter from disk
                    eval_model, tok = load_model_and_tokenizer(cfg, adapter_path=cfg["adapter_dir"])
                last_summaries = stage_eval(eval_model, tok, cfg, "M1", with_m0_comparison=True)
                tracker.summary(eval_summary_metrics(last_summaries, "M1"))
                meta["headline"] = headline(last_summaries)  # so the refreshed card carries the result
                if cfg.get("hub_repo_id"):  # refresh card with eval results
                    meta.pop("hub_url", None)
                    _try_push(cfg, meta)
            print(f"=== stage {stage} done in {time.time() - t0:.0f}s ===")
        status = "success"
    finally:
        finalize_invocation(meta, cfg, status=status, wandb_url=tracker.url, summaries=last_summaries)
        if meta.get("headline"):
            print(f"\nresult: {meta['headline']}")
        if meta.get("hub_url"):
            print(f"model:  {meta['hub_url']}")
        tracker.finish(status)
