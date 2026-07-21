# Capability-retention eval

Does resistance training make the model dumber? `scripts/eval_capability.py`
answers it the way "Steering Awareness" (arXiv:2511.21399, App. E.4) does —
EleutherAI's lm-evaluation-harness, **MMLU 5-shot** + **GSM8K 8-shot CoT**,
greedy, comparing the base model (M0) vs the resist adapter (M1 = base + LoRA).

This matters because in the paper the *detection* LoRA wrecked capability
(Gemma-2-9B GSM8K 82.8 -> 13.0). A resistance defense that tanks general ability
isn't a defense worth having — so this is a required check, not a nice-to-have.

## Setup: the lm-eval interpreter

lm-eval pulls in a lot of dependencies that can clash with the pinned ones, so
by default the script runs it from a **separate venv**, falling back to the
current interpreter if that venv is absent.

**Local (recommended — isolated):**
```bash
uv venv --python 3.12 .venv-lmeval
uv pip install --python .venv-lmeval/bin/python -e ".[capability]"
```
(`.[capability]` installs lm-eval plus the repo's own pins so the adapter loads
with the same transformers/peft it was trained under.)

**Colab (simplest — same interpreter):**
```bash
pip install -e ".[capability]"     # no separate venv; the script uses sys.executable
```

Point the script at a specific interpreter with `lmeval_python` in the config if
you want to override the auto-detection.

## Run

```bash
# fast sanity first (tiny subset), then the full paper-scale run
python scripts/eval_capability.py configs/qwen3b.yaml --limit "mmlu=10,gsm8k_cot=40" --chat-template
python scripts/eval_capability.py configs/qwen3b.yaml --chat-template
```

- Needs a trained adapter (`--stages train` first) — M1 = `cfg['model_id']` +
  `cfg['adapter_dir']`.
- `--chat-template` applies the instruct chat template (recommended for the
  Qwen-Instruct models).
- **`mmlu`'s `--limit` is PER-SUBJECT** (MMLU is 57 subjects), so `mmlu=15` ≈ 855
  items (fast) while `mmlu=200` ≈ the whole benchmark. `gsm8k_cot`'s limit is
  absolute. `--limit full` runs everything.
- Resumable: a completed (model, task) is skipped unless `--force`.
- Writes harness JSON under `results/<run>/capability/<model>/<task>/` and a
  combined `capability_summary.{md,csv}`.

## Config knobs (all optional; sensible defaults)

```yaml
capability_tasks: {mmlu: 5, gsm8k_cot: 8}   # task -> num_fewshot
capability_dir: null           # default: <results_dir>/capability
capability_limit: null         # int, or "mmlu=15,gsm8k_cot=200"
capability_batch_size: auto
capability_apply_chat_template: false
lmeval_python: null            # override the interpreter auto-detection
```
