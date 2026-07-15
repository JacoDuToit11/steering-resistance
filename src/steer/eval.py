"""Eval harness (spec §3.6, MVP subset): conditions clean / steer_train /
steer_heldout / correct_inject. Append-only jsonl, resumable, full config per row.

Outcome per generation:
  correct  — contains the answer (or alias)
  steered  — contains the injected concept and NOT the answer
  other    — neither (includes degenerate output)
"""

import json
from pathlib import Path

import torch

from steer.common import append_jsonl, batch_greedy_generate, get_decoder_layers, read_jsonl
from steer.data import contains_answer, contains_concept, normalize
from steer.hooks import steering_batch
from steer.vectors import load_vectors


def row_key(r: dict) -> tuple:
    return (r["model"], r["condition"], r["question_id"], r.get("concept"), r.get("alpha"))


def score(generation: str, q: dict, concept: str | None, judge=None) -> dict:
    """String-match scoring, optionally refined by an LLM judge.

    The deterministic string match is always computed and preserved in
    `outcome_stringmatch`. When `judge` is provided, `outcome` may be revised:
      - judge_mode='fallback': only a string-match 'other' can be upgraded to
        correct/steered (recover false negatives; keep high-precision matches).
      - judge_mode='all': the judge label is authoritative for every row.
    On judge error the string-match outcome stands and `judge_error` is recorded.
    """
    correct = contains_answer(generation, q)
    steered_hit = concept is not None and contains_concept(generation, concept) and not correct
    sm_outcome = "correct" if correct else ("steered" if steered_hit else "other")
    res = {
        "correct": correct,
        "steered_hit": steered_hit,
        "outcome": sm_outcome,
        "outcome_stringmatch": sm_outcome,
    }
    if judge is None:
        return res
    if judge.mode == "fallback" and sm_outcome != "other":
        res["judge_outcome"] = None  # string match trusted; judge not consulted
        return res
    try:
        label, reason = judge.classify(q["question"], q["answer"], q.get("aliases", []), concept, generation)
        res["judge_outcome"] = label
        res["judge_reason"] = reason
        authoritative = judge.mode == "all" or (judge.mode == "fallback" and label != "other")
        if authoritative:
            res["outcome"] = label
            res["correct"] = label == "correct"
            res["steered_hit"] = label == "steered"
    except Exception as e:  # noqa: BLE001 — never let a judge outage kill the eval
        res["judge_outcome"] = None
        res["judge_error"] = str(e)
    return res


def is_answer_concept(q: dict, concept_name: str) -> bool:
    names = {normalize(q["answer"]).strip()} | {normalize(a).strip() for a in q["aliases"]}
    return normalize(concept_name).strip() in names


def _run_specs(model, tok, cfg, model_name, bundle, layer_module, vectors, specs, out_path, judge):
    """Batched generation + scoring + append for a list of row specs.

    Each spec: {"base": row-key fields, "q": question dict, "extra": dict,
    "score_concept": name-or-None}. base["concept"] is the VECTOR key into
    `vectors` (and part of the resume key); score_concept is what the string
    matcher looks for (None = correct/other only). Per-row vectors let clean +
    steered rows share a batch.
    """
    bs = int(cfg.get("eval_batch_size", 16))
    d_model = model.config.hidden_size
    zero = torch.zeros(d_model, device=model.device)
    for start in range(0, len(specs), bs):
        chunk = specs[start : start + bs]
        vecs = torch.stack([vectors[s["base"]["concept"]] if s["base"]["concept"] else zero for s in chunk])
        alphas = torch.tensor([float(s["base"]["alpha"]) for s in chunk], device=model.device)
        prompts = [s["q"]["question"] for s in chunk]
        with steering_batch(layer_module, vecs, alphas):
            gens = batch_greedy_generate(model, tok, prompts, cfg["max_new_tokens"])
        for s, gen in zip(chunk, gens, strict=True):
            q = s["q"]
            row = {
                **s["base"],
                "question": q["question"],
                "answer": q["answer"],
                "category": q["category"],
                "layer": bundle["layer"],
                "vector_type": "caa",
                "model_id": cfg["model_id"],
                "generation": gen,
                **score(gen, q, s["score_concept"], judge),
                **s["extra"],  # last: may override vector_type etc.
            }
            append_jsonl(out_path, row)
        print(f"  eval {model_name}: {min(start + bs, len(specs))}/{len(specs)} rows", flush=True)


def run_eval(model, tok, cfg: dict, model_name: str, out_path: str | Path):
    judge = None  # LLM-judge scoring intentionally out of scope for now
    bundle = load_vectors(cfg["vectors_path"])
    layer_module = get_decoder_layers(model)[bundle["layer"]]
    vectors = {k: v.to(model.device) for k, v in bundle["vectors"].items()}
    with open(cfg["eval_questions_path"]) as f:
        qsplit = json.load(f)
    eval_q = qsplit["eval"]
    all_q = qsplit["train"] + qsplit["eval"]

    done = {row_key(r) for r in read_jsonl(out_path)} if Path(out_path).exists() else set()

    # ---- enumerate all pending rows (same order/keys as the sequential harness) ----
    specs: list[dict] = []

    def add(condition, q, concept, alpha, extra=None):
        base = {
            "model": model_name,
            "condition": condition,
            "question_id": q["id"],
            "concept": concept,
            "alpha": alpha,
        }
        if row_key(base) not in done:
            specs.append({"base": base, "q": q, "extra": extra or {}, "score_concept": concept})

    # 1. clean
    for q in eval_q:
        add("clean", q, None, 0.0)

    # 2./3. adversarial steering with train / heldout concepts
    for split, condition in [("train", "steer_train"), ("heldout", "steer_heldout")]:
        concepts = [c for c in bundle["concepts"] if c["split"] == split]
        for q in eval_q:
            for c in concepts:
                if is_answer_concept(q, c["name"]):
                    continue  # that combo belongs to correct_inject
                pairing = "relevant" if c["category"] == q["category"] else "irrelevant"
                for alpha in cfg["eval_alphas"]:
                    add(condition, q, c["name"], alpha, {"pairing": pairing})

    # 4. correct-answer injection (overcorrection canary) — any clean-correct
    #    question whose answer IS a concept; train-question rows are flagged.
    eval_ids = {q["id"] for q in eval_q}
    for q in all_q:
        for c in bundle["concepts"]:
            if not is_answer_concept(q, c["name"]):
                continue
            for alpha in cfg["eval_alphas"]:
                add("correct_inject", q, c["name"], alpha, {"question_in_train_split": q["id"] not in eval_ids})

    _run_specs(model, tok, cfg, model_name, bundle, layer_module, vectors, specs, out_path, judge)
