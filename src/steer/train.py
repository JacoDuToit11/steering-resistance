"""Hand-written PEFT training loop with live steering hooks (spec §3.4/§3.5).

Per-example forward passes (vector varies per example) with gradient
accumulation to the effective batch size. Loss on assistant tokens only.
"""

import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from steer.common import get_decoder_layers, set_seed
from steer.hooks import steering_batch
from steer.vectors import load_vectors


def encode_example(tok, question: str, target: str, max_seq_len: int):
    """Chat-formatted (input_ids, labels) lists with loss only on the assistant turn."""
    msgs_prompt = [{"role": "user", "content": question}]
    msgs_full = msgs_prompt + [{"role": "assistant", "content": target}]
    prompt_ids = tok.apply_chat_template(msgs_prompt, add_generation_prompt=True)
    full_ids = tok.apply_chat_template(msgs_full)
    if not isinstance(prompt_ids, list):  # transformers v5 returns BatchEncoding
        prompt_ids, full_ids = prompt_ids["input_ids"], full_ids["input_ids"]
    full_ids = full_ids[:max_seq_len]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
    return full_ids, labels


def encode_batch(tok, examples: list[dict], max_seq_len: int, device):
    """Right-pad a chunk of examples into (input_ids, attention_mask, labels) tensors."""
    encoded = [encode_example(tok, ex["question"], ex["target"], max_seq_len) for ex in examples]
    max_len = max(len(ids) for ids, _ in encoded)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
    labels = torch.full((len(encoded), max_len), -100, dtype=torch.long)
    for i, (row_ids, row_labels) in enumerate(encoded):
        ids[i, : len(row_ids)] = torch.tensor(row_ids)
        mask[i, : len(row_ids)] = 1
        labels[i, : len(row_labels)] = torch.tensor(row_labels)
    return ids.to(device), mask.to(device), labels.to(device)


def train_m1(model, tok, cfg: dict, tracker=None) -> dict:
    """LoRA-trains `model` in place on the resist objective; returns loss stats."""
    set_seed(cfg["seed"])
    with open(cfg["train_examples_path"]) as f:
        examples = json.load(f)
    bundle = load_vectors(cfg["vectors_path"])

    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_targets"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    layer_module = get_decoder_layers(model)[bundle["layer"]]
    vectors = {k: v.to(model.device) for k, v in bundle["vectors"].items()}

    n_epochs = cfg["epochs"]
    batch_size = int(cfg.get("train_batch_size", 8))
    accum = max(1, cfg["effective_batch_size"] // batch_size)  # micro-batches per optimizer step
    order = list(range(len(examples)))
    rng = random.Random(cfg["seed"])
    d_model = model.config.hidden_size
    zero = torch.zeros(d_model, device=model.device)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["lr"])
    batches_per_epoch = (len(examples) + batch_size - 1) // batch_size
    total_steps = max(1, (batches_per_epoch * n_epochs) // accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    model.train()
    losses = []  # one entry per micro-batch (token-weighted within the batch)
    micro = 0
    for epoch in range(n_epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            chunk = [examples[i] for i in order[start : start + batch_size]]
            ids, mask, labels = encode_batch(tok, chunk, cfg["max_seq_len"], model.device)
            vecs = torch.stack([vectors[ex["concept"]] if ex["steered"] else zero for ex in chunk])
            alphas = torch.tensor(
                [float(ex["alpha"]) if ex["steered"] else 0.0 for ex in chunk], device=model.device
            )
            with steering_batch(layer_module, vecs, alphas):
                loss = model(ids, attention_mask=mask, labels=labels).loss
            (loss / accum).backward()
            losses.append(loss.item())
            micro += 1
            if micro % accum == 0:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                recent = sum(losses[-accum:]) / accum
                print(f"epoch {epoch} step {micro // accum}/{total_steps}: loss {recent:.4f}", flush=True)
                if tracker is not None:
                    tracker.log(
                        {"train/loss": recent, "train/lr": sched.get_last_lr()[0], "train/epoch": epoch},
                        step=micro // accum,
                    )
    if micro % accum:
        opt.step()
        opt.zero_grad(set_to_none=True)

    first = sum(losses[:accum]) / min(accum, len(losses))
    last = sum(losses[-accum:]) / min(accum, len(losses))
    print(f"loss first-batch {first:.4f} -> last-batch {last:.4f}")
    assert last < first, "training loss did not decrease"

    adapter_dir = Path(cfg["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    print(f"adapter saved to {adapter_dir}")
    # `model` (the PEFT wrapper) is returned so a single-load pipeline can
    # merge_and_unload() and go straight to eval without reloading from disk.
    return {"first_batch_loss": first, "last_batch_loss": last, "n_examples": len(examples), "model": model}
