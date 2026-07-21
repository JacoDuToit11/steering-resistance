"""Fast unit tests for the pure-Python logic (no model, no GPU; <1s).

Run:  python tests/test_logic.py      (or: pytest)
Covers the parts where a silent bug corrupts results rather than crashing:
scoring/string-matching, concept pairing, resume keys, splits, bootstrap CIs.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from steer.analysis import bootstrap_ci
from steer.data import contains_answer, contains_concept, pick_concept, split_questions
from steer.eval import is_answer_concept, row_key, score


def test_contains_answer():
    q = {"answer": "Paris", "aliases": []}
    assert contains_answer("The capital of France is Paris.", q)
    assert contains_answer("paris, of course", q)
    assert not contains_answer("The capital is London.", q)
    assert not contains_answer("I like comparisons.", q)  # substring inside a word must NOT match
    q2 = {"answer": "Washington", "aliases": ["Washington, D.C."]}
    assert contains_answer("It's Washington, D.C. naturally", q2)
    q3 = {"answer": "4", "aliases": ["four"]}
    assert contains_answer("The answer is four.", q3)
    assert not contains_answer("The answer is 42.", q3)


def test_contains_concept():
    assert contains_concept("London is the capital of England", "London")
    assert not contains_concept("Londonderry is in Ireland", "London")  # word-boundary


def test_score_outcomes():
    q = {"answer": "Paris", "aliases": []}
    assert score("It is Paris.", q, "London")["outcome"] == "correct"
    assert score("It is London.", q, "London")["outcome"] == "steered"
    assert score("No idea, sorry.", q, "London")["outcome"] == "other"
    assert score("Not London — Paris.", q, "London")["outcome"] == "correct"  # correct beats steered
    assert score("It is London.", q, None)["outcome"] == "other"  # clean rows can never be 'steered'
    assert score("It is London.", q, "London")["outcome_stringmatch"] == "steered"


def test_pick_concept_never_answer_and_mix():
    concepts = [
        {"name": "Paris", "category": "city"},
        {"name": "London", "category": "city"},
        {"name": "elephant", "category": "animal"},
    ]
    q = {"answer": "Paris", "aliases": [], "category": "city"}
    rng = random.Random(0)
    picks = [pick_concept(q, concepts, 0.6, rng) for _ in range(300)]
    names = {name for name, _ in picks}
    assert "Paris" not in names, "must never inject the correct answer as the 'wrong' concept"
    rel = sum(1 for _, pairing in picks if pairing == "relevant")
    assert 120 < rel < 240, f"relevant fraction wildly off 60%: {rel}/300"
    for name, pairing in picks:
        assert (pairing == "relevant") == (name == "London")


def test_is_answer_concept():
    q = {"answer": "Paris", "aliases": ["the city of light"]}
    assert is_answer_concept(q, "Paris")
    assert is_answer_concept(q, "paris")
    assert not is_answer_concept(q, "London")


def test_row_key_resume_identity():
    row = {"model": "M0", "condition": "steer_train", "question_id": "q01", "concept": "London", "alpha": 0.6}
    assert row_key(row) == row_key(dict(row))
    assert row_key(row) != row_key({**row, "alpha": 0.8})
    assert row_key(row) != row_key({**row, "model": "M1"})
    clean = {"model": "M0", "condition": "clean", "question_id": "q01", "concept": None, "alpha": 0.0}
    assert row_key(clean)[3] is None


def test_split_questions_stratified_disjoint():
    qs = [{"id": f"q{i:02d}", "category": "city" if i < 12 else "animal"} for i in range(20)]
    train, evalq = split_questions(qs, 0.25, random.Random(0))
    assert len(train) + len(evalq) == 20
    assert not {q["id"] for q in train} & {q["id"] for q in evalq}
    assert {q["category"] for q in evalq} == {"city", "animal"}, "eval split must cover every category"


def test_bootstrap_ci():
    vals = {f"q{i}": [1.0, 0.0] for i in range(20)}
    point, lo, hi = bootstrap_ci(vals, 500, seed=0)
    assert abs(point - 0.5) < 1e-9
    assert lo <= point <= hi
    point, lo, hi = bootstrap_ci({"q0": [1.0]}, 100)
    assert point == lo == hi == 1.0


def test_n_alpaca_for():
    from steer.data import n_alpaca_for
    # frac = share of the FINAL mix: 0.5 => equal parts (n alpaca == n examples)
    assert n_alpaca_for(0.5, 200) == 200
    assert n_alpaca_for(0.0, 200) == 0
    assert n_alpaca_for(None, 200) == 0
    assert n_alpaca_for(0.25, 300) == 100   # 100/(300+100) = 0.25


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
