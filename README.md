# steering-resistance

SFT training for **resistance to adversarial activation steering**: fine-tune a
model *while* steering vectors are injected into its activations, rewarding it
for producing its own clean answers — then measure whether resistance holds on
attacks it never saw.

Ground-up rebuild of the resistance pipeline first developed in
[steering-awareness](https://github.com/steering-awareness/steering-awareness)
(group sprint, workshop paper in progress). This repo imports that sprint's
battle-tested core and its methodology — see
[docs/PRINCIPLES.md](docs/PRINCIPLES.md), which every experiment here must
satisfy — and will extend from there.

## How it works

1. **vectors** — one CAA steering vector per concept (difference of mean
   activations, concept vs contrast prompts, at one decoder layer), scaled to
   the layer's residual norm. Gate: a vector must hijack the *base* model on
   ≥30% of unrelated questions or it is discarded.
2. **data** — keep only questions the base model answers correctly clean; the
   target is the model's own clean answer. 70% of examples steered (mixed
   plausible-wrong-answer / off-topic concepts), 30% clean.
3. **train** — LoRA fine-tuning with the injection live in the forward pass;
   gradients flow through the hook into the adapters. No detection signal
   exists anywhere.
4. **eval** — base vs trained model on held-out questions: clean /
   train-concept / held-out-concept / correct-answer-injection (overcorrection
   canary). Outcomes correct/steered/other with bootstrap CIs by question.

## Quickstart

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/python tests/test_logic.py               # <1s, no GPU
.venv/bin/python scripts/verify_hooks.py configs/smoke_0.5b.yaml   # once per machine
.venv/bin/python scripts/run.py configs/smoke_0.5b.yaml            # ~5 min on a laptop
```

The smoke run validates the whole pipeline at 0.5B scale. The reference
experiment (`configs/qwen7b_popqa.yaml`, one GPU-hour) reproduces the original
headline result and is the gate before building anything new on this stack.

`scripts/run.py` runs all stages on one model load; `--stages train,eval_m1`
reruns a subset. Evals are append-only jsonl, self-describing per row, and
resumable. One `results/<run>/` directory per experiment.

## Layout

```
src/steer/
├── hooks.py      injection context managers (single + per-row batched); capture
├── vectors.py    CAA construction + efficacy gate
├── data.py       clean filter, concept pairing, PopQA loader, example building
├── train.py      batched PEFT loop, gradients through the live hook
├── eval.py       eval conditions -> jsonl (string-match scoring)
├── analysis.py   outcome rates, bootstrap CIs, summary tables
├── pipeline.py   stages sharing one model load
└── common.py     config/model/device/generation utilities
```

## Roadmap (rough)

- [ ] Reproduce the 7B PopQA result on this stack (validation gate)
- [ ] Adapter backup discipline (auto-push after training)
- [ ] Model-family abstraction (Llama/Gemma via config only)
- [ ] Adaptive attacker (vectors optimized against the defended model)
- [ ] Mechanism probes (decodability of injected concepts under resistance)
