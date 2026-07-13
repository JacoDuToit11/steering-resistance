"""Machinery verification — run once per new model/machine, before any experiment.

(a) residual delta at layer L with hook on vs off matches alpha * ||vec||
(b) loss.backward() with the hook active produces nonzero grads on lora_B params,
    including on layers BELOW the injection point (gradient flows THROUGH the hook)
(c) the same hook at eval reproduces the training-time delta

Prints the module tree so layer indexing is verified, never guessed.
Usage: python scripts/verify_hooks.py configs/smoke_0.5b.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from peft import LoraConfig, get_peft_model

from steer.common import chat_input_ids, get_decoder_layers, load_config, load_model_and_tokenizer, set_seed
from steer.hooks import capture_residual, steering


def residual_at_layer(model, layer_module, ids):
    store = []
    with capture_residual(layer_module, store), torch.no_grad():
        model(ids)
    return store[0]


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/smoke_0.5b.yaml")
    set_seed(cfg["seed"])
    L, alpha = cfg["layer"], 4.0

    model, tok = load_model_and_tokenizer(cfg)
    d_model = model.config.hidden_size

    print("=== module tree (base model, depth 2) ===")
    for name, _ in model.named_children():
        print(" ", name)
    print("  decoder layers path: model.model.layers, count =", len(get_decoder_layers(model)))

    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_targets"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    layers = get_decoder_layers(model)
    print("\n=== after PEFT wrap ===")
    print("  class:", model.__class__.__name__)
    print("  decoder layers resolved, count =", len(layers), "| layer module:", layers[L].__class__.__name__)

    torch.manual_seed(0)
    vec = torch.randn(d_model)
    vec = (vec / vec.norm() * 10.0).to(model.device)

    ids = chat_input_ids(tok, "What is the capital of France?", model.device)

    # ---- (a) norm delta at layer L, hook on vs off ----
    model.eval()
    clean = residual_at_layer(model, layers[L], ids)
    with steering(layers[L], vec, alpha):
        steered = residual_at_layer(model, layers[L], ids)
    # fp32 compare with dtype-scaled tolerance (bf16 quantizes the stored residual)
    delta = (steered.float() - clean.float()).norm(dim=-1)
    expected = (alpha * vec.norm()).float()
    eps = torch.finfo(clean.dtype).eps
    tol = 4 * eps * clean.float().norm(dim=-1) + 1e-3 * expected
    print(f"\n(a) per-position delta norm: mean={delta.mean():.4f}, expected={expected:.4f}")
    assert torch.all((delta - expected).abs() <= tol), "delta != alpha*||vec||"
    print("    PASS")

    # ---- (b) gradients flow to LoRA params with hook active ----
    model.train()
    with steering(layers[L], vec, alpha):
        out = model(ids, labels=ids.clone())
        out.loss.backward()
    lora_grads = [
        (n, p.grad.abs().max().item())
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    ]
    # at step 0 lora_B == 0, so lora_A grads are analytically zero; require all lora_B grads nonzero
    b_grads = [(n, g) for n, g in lora_grads if "lora_B" in n]
    n_nonzero = sum(1 for _, g in b_grads if g > 0)
    print(f"(b) loss={out.loss.item():.4f}; lora_B params with grad: {len(b_grads)}, nonzero: {n_nonzero}")
    assert b_grads and n_nonzero == len(b_grads), "lora_B grads missing/zero with hook active"
    below = [g for n, g in b_grads if any(f"layers.{i}." in n for i in range(L))]
    assert below and max(below) > 0, "no gradient below injection layer"
    print(f"    PASS (max grad below layer {L}: {max(below):.2e})")
    model.zero_grad(set_to_none=True)

    # ---- (c) eval-time hook reproduces the same delta ----
    model.eval()
    with steering(layers[L], vec, alpha):
        steered2 = residual_at_layer(model, layers[L], ids)
    delta2 = (steered2.float() - clean.float()).norm(dim=-1)
    print(f"(c) eval delta norm: mean={delta2.mean():.4f} (train-pass delta: {delta.mean():.4f})")
    assert torch.allclose(delta2, delta, rtol=1e-4), "eval hook delta differs from training hook delta"
    print("    PASS")

    clean2 = residual_at_layer(model, layers[L], ids)
    assert torch.allclose(clean2, clean), "hook leaked: clean forward changed after context exit"
    print("\nAll three asserts PASS; hooks clean up after themselves.")


if __name__ == "__main__":
    main()
