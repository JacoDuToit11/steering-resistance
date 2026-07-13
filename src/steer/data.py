"""QA filtering, question<->concept pairing, and training-example construction (spec §3.3).

Training targets are the base model's own clean greedy generations (captured
during the clean-correctness filter), so the resist objective is literally
"under steering, produce what you would have produced clean" — no style shift.
"""

import json
import random
import re
from pathlib import Path


def load_qa(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)["questions"]


def load_popqa(cfg: dict) -> list[dict]:
    """Load PopQA (akariasai/PopQA test split) into the QA schema.

    Filters to cfg['popqa_relations'] and remaps each relation to a category via
    cfg['popqa_category_map'] so the existing concept pool stays a meaningful
    relevant/irrelevant split (e.g. capital->city: city concepts are plausible
    wrong capitals, animal concepts are off-topic).
    """
    from datasets import load_dataset

    relations = set(cfg.get("popqa_relations") or [])
    cat_map = cfg.get("popqa_category_map") or {}
    ds = load_dataset("akariasai/PopQA", split="test")

    rows = []
    for r in ds:
        if relations and r["prop"] not in relations:
            continue
        answer = r["obj"]
        aliases = [a for a in json.loads(r["possible_answers"]) if a != answer]
        rows.append(
            {
                "id": f"pop{r['id']}",
                "category": cat_map.get(r["prop"], r["prop"]),
                "question": r["question"],
                "answer": answer,
                "aliases": aliases,
                "popularity": r.get("o_pop"),
            }
        )

    n = cfg.get("popqa_max_questions")
    if n and len(rows) > n:
        rows = sorted(rows, key=lambda x: -(x["popularity"] or 0))[:n]  # most-popular first: 3B likelier to know them
    return rows


def load_questions(cfg: dict) -> list[dict]:
    """Dispatch on cfg['qa_source'] ('json' default, or 'popqa')."""
    if cfg.get("qa_source", "json") == "popqa":
        return load_popqa(cfg)
    return load_qa(cfg["qa_path"])


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def contains_answer(generation: str, q: dict) -> bool:
    gen = normalize(generation)
    return any(f" {normalize(a).strip()} " in f" {gen} " for a in [q["answer"], *q["aliases"]])


def contains_concept(generation: str, concept: str) -> bool:
    gen = normalize(generation)
    return f" {normalize(concept).strip()} " in f" {gen} "


def clean_filter(model, tok, questions: list[dict], max_new_tokens: int, batch_size: int = 32) -> list[dict]:
    """Keep questions the model answers correctly clean; attach the clean generation as `target`."""
    from steer.common import batch_greedy_generate

    kept = []
    for start in range(0, len(questions), batch_size):
        chunk = questions[start : start + batch_size]
        gens = batch_greedy_generate(model, tok, [q["question"] for q in chunk], max_new_tokens)
        for q, gen in zip(chunk, gens):
            if contains_answer(gen, q):
                kept.append({**q, "target": gen.strip()})
    return kept


def split_questions(questions: list[dict], eval_frac: float, rng: random.Random):
    """Stratified-by-category train/eval split."""
    train, evalq = [], []
    by_cat: dict[str, list] = {}
    for q in questions:
        by_cat.setdefault(q["category"], []).append(q)
    for cat_qs in by_cat.values():
        cat_qs = sorted(cat_qs, key=lambda q: q["id"])
        rng.shuffle(cat_qs)
        n_eval = max(1, round(len(cat_qs) * eval_frac))
        evalq += cat_qs[:n_eval]
        train += cat_qs[n_eval:]
    return sorted(train, key=lambda q: q["id"]), sorted(evalq, key=lambda q: q["id"])


def pick_concept(q: dict, concepts: list[dict], relevant_frac: float, rng: random.Random) -> tuple[str, str]:
    """Sample an injected concept for a question. Returns (concept_name, pairing).

    relevant  = same category (a plausible wrong answer)
    irrelevant = different category (anti-shortcut mix, spec §3.3)
    Never picks a concept that IS the correct answer.
    """
    answer_names = {normalize(q["answer"]).strip()} | {normalize(a).strip() for a in q["aliases"]}
    usable = [c for c in concepts if normalize(c["name"]).strip() not in answer_names]
    relevant = [c for c in usable if c["category"] == q["category"]]
    irrelevant = [c for c in usable if c["category"] != q["category"]]
    if relevant and (not irrelevant or rng.random() < relevant_frac):
        return rng.choice(relevant)["name"], "relevant"
    return rng.choice(irrelevant)["name"], "irrelevant"


def build_train_examples(
    train_questions: list[dict],
    train_concepts: list[dict],
    cfg: dict,
    rng: random.Random,
) -> list[dict]:
    """repeats_per_question examples per training question, each independently
    steered (steered_frac, random concept+alpha) or clean."""
    examples = []
    for q in train_questions:
        for _ in range(cfg["repeats_per_question"]):
            ex = {
                "question_id": q["id"],
                "question": q["question"],
                "target": q["target"],
                "category": q["category"],
            }
            if rng.random() < cfg["steered_frac"]:
                concept, pairing = pick_concept(q, train_concepts, cfg["relevant_frac"], rng)
                ex.update(steered=True, concept=concept, pairing=pairing, alpha=rng.choice(cfg["train_alphas"]))
            else:
                ex.update(steered=False, concept=None, pairing=None, alpha=0.0)
            examples.append(ex)
    rng.shuffle(examples)
    return examples
