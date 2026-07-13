"""CAA steering-vector construction + per-vector efficacy check (spec §3.2).

v_c = mean(residual at layer L, final token, pos prompts)
    - mean(same, neg prompts), then L2-normalized.

Stored vectors are pre-scaled by the mean residual norm at layer L, so the
injection is simply hidden += alpha * vec (hooks.py) and alpha is comparable
across layers/models.
"""

import json
from pathlib import Path

import torch

from steer.common import get_decoder_layers
from steer.hooks import capture_residual, steering

EFFICACY_QUESTIONS = [
    "What is 7 plus 5?",
    "What color is a ripe banana?",
    "How many days are there in a week?",
    "What do you call frozen water?",
    "Name a primary color.",
    "What season comes after winter?",
    "How many legs does a spider have?",
    "What gas do humans breathe in to survive?",
    "What is the opposite of hot?",
    "How many minutes are in an hour?",
]


def load_concepts(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def final_token_residual(model, tok, layer_module, text: str) -> torch.Tensor:
    ids = tok(text, return_tensors="pt")["input_ids"].to(model.device)
    store = []
    with capture_residual(layer_module, store), torch.no_grad():
        model(ids)
    return store[0][0, -1, :].float().cpu()  # [d_model]


def build_vectors(model, tok, concepts_cfg: dict, layer: int) -> dict:
    """Returns {"vectors": {name: scaled_vec}, "unit_vectors": {...}, "mean_norm": float, "layer": int}."""
    layer_module = get_decoder_layers(model)[layer]
    templates = concepts_cfg["templates"]

    raw = {}
    all_norms = []
    for c in concepts_cfg["concepts"]:
        pos_acts, neg_acts = [], []
        for tpl in templates:
            pos_acts.append(final_token_residual(model, tok, layer_module, tpl.format(c=c["name"])))
            for contrast in c["contrasts"]:
                neg_acts.append(final_token_residual(model, tok, layer_module, tpl.format(c=contrast)))
        pos_mean = torch.stack(pos_acts).mean(0)
        neg_mean = torch.stack(neg_acts).mean(0)
        v = pos_mean - neg_mean
        raw[c["name"]] = v / v.norm()
        all_norms += [a.norm().item() for a in pos_acts + neg_acts]

    mean_norm = float(torch.tensor(all_norms).mean())
    scaled = {name: v * mean_norm for name, v in raw.items()}
    return {"vectors": scaled, "unit_vectors": raw, "mean_norm": mean_norm, "layer": layer}


def efficacy_check(model, tok, vec_bundle: dict, alpha: float, max_new_tokens: int = 40) -> dict:
    """Inject each vector on unrelated questions; rate = fraction of generations
    containing the concept token. One batched generate per concept."""
    from steer.common import batch_greedy_generate

    layer_module = get_decoder_layers(model)[vec_bundle["layer"]]
    rates = {}
    for name, vec in vec_bundle["vectors"].items():
        vec_d = vec.to(model.device)
        with steering(layer_module, vec_d, alpha):
            gens = batch_greedy_generate(model, tok, EFFICACY_QUESTIONS, max_new_tokens)
        rates[name] = sum(name.lower() in g.lower() for g in gens) / len(EFFICACY_QUESTIONS)
    return rates


def save_vectors(bundle: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


def load_vectors(path: str | Path) -> dict:
    return torch.load(path, weights_only=True)
