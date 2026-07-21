"""Shared utilities: config loading, model loading, device/dtype selection, seeding."""

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def pick_device(cfg: dict) -> torch.device:
    want = cfg.get("device", "auto")
    if want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(cfg: dict, device: torch.device) -> torch.dtype:
    """Model dtype. Explicit config wins (bfloat16/float16/float32) — needed to
    fit big models on MPS, where 3B fp32 (~12GB) blows a 16GB Mac but bf16 (~6GB)
    fits. `auto` stays conservative: bf16 on cuda, fp32 elsewhere."""
    want = str(cfg.get("dtype", "auto")).lower()
    explicit = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32}
    if want in explicit:
        return explicit[want]
    return torch.bfloat16 if device.type == "cuda" else torch.float32  # auto


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_and_tokenizer(cfg: dict, adapter_path: str | Path | None = None):
    device = pick_device(cfg)
    dtype = pick_dtype(cfg, device)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=dtype)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()  # eval-only: merge so hook paths match base model
    model.to(device)
    model.eval()
    return model, tok


def get_decoder_layers(model):
    """Return the list of decoder layers, robust to PEFT wrapping.

    Verified against Qwen2ForCausalLM: base path is model.model.layers;
    after PeftModel wrapping: model.base_model.model.model.layers.
    """
    m = model
    if m.__class__.__name__.startswith("Peft"):  # PeftModel wraps: base_model.model = HF model
        m = m.base_model.model
    return m.model.layers


def chat_input_ids(tok, user_msg: str, device) -> torch.Tensor:
    """User message -> input_ids tensor with generation prompt, [1, seq]."""
    enc = tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc
    return ids.to(device)


def greedy_generate(model, tok, user_msg: str, max_new_tokens: int = 64) -> str:
    ids = chat_input_ids(tok, user_msg, model.device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


def batch_greedy_generate(model, tok, user_msgs: list[str], max_new_tokens: int = 64) -> list[str]:
    """Greedy-generate for a list of user messages in ONE batched forward.

    Left-pads so all prompts end at the same position (generation starts
    together); pad tokens are attention-masked out. Equivalent to per-row
    greedy_generate up to floating-point batching noise (see README).
    """
    if not user_msgs:
        return []
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": m}], add_generation_prompt=True, tokenize=False
        )
        for m in user_msgs
    ]
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    finally:
        tok.padding_side = old_side
    enc = {k: v.to(model.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    new_tokens = out[:, enc["input_ids"].shape[1] :]
    return tok.batch_decode(new_tokens, skip_special_tokens=True)


def append_jsonl(path: str | Path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
