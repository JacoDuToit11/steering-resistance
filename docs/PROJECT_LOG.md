# Technical AI Safety Project Log — Training Resistance to Activation Steering

_Living document. Latest log entry first. Deep detail lives in the linked repo docs._

---

## Project summary

**Building on** "Steering Awareness: Detecting Activation Steering from Within"
([arXiv:2511.21399](https://arxiv.org/abs/2511.21399)). They show a model can be
*trained to detect* activation steering from within its own forward pass — but
detection does **not** make it *resist*; detection-trained models are, if
anything, more steerable. They flag training an explicit **resistance objective**
as open future work.

**What I'm trying to figure out:** can a model be trained to **resist steering
vectors** — keep producing the correct answer while a vector is injected —
**while retaining general capabilities and coherence** when unsteered? And, if so,
is the resistance *genuine perturbation-robustness* or a *geometric shortcut*
(deleting the subspace the training vectors live in)?

**Why it matters for AI safety:** activation steering is a common safety tool —
used both to *evaluate* models and to *control* them (e.g.
[arXiv:2510.20487](https://arxiv.org/abs/2510.20487) steers a model to be *less
evaluation-aware*). All such uses assume steering is a reliable channel the model
can't resist. If future models can detect **and resist** steering, these
techniques stop working — so it matters whether resistance is trainable, how well
it generalizes, and what it costs.

Full framing: [MOTIVATION.md](MOTIVATION.md) · [LITERATURE.md](LITERATURE.md)
(close reading of the source papers).

---

## Method (one-screen version)

Pipeline (`src/steer/`, one config per experiment):

1. **Attack vectors** — CAA (Contrastive Activation Addition): `v_c = mean(act on
   concept prompts) − mean(act on neutral-baseline prompts)` at one decoder
   layer, scaled to the layer's residual norm. **Efficacy gate:** a vector is
   kept only if it actually hijacks the *base* model on ≥30% of unrelated
   questions — so every "resisted" attack was a proven threat.
2. **Data** — keep only questions the base model already answers correctly clean
   (resistance is undefined otherwise); the training target is the model's *own
   clean answer*. Attack concepts come from a separate authored bank (500
   concepts / 21 categories, paper-faithful), with held-out concepts and whole
   held-out categories for generalization.
3. **Train** — LoRA fine-tuning with the steering vector injected **live during
   the forward pass**; gradients flow *through* the injection into the adapter.
   The model is never told an attack exists (no detection signal). **+50% Alpaca
   instruction-following replay** to preserve general capabilities (paper §3.3).
4. **Evaluate** — base (M0) vs trained (M1) on held-out questions under injection:
   `correct` / `steered` / `other`, bootstrap 95% CIs. Plus **capability
   retention** (MMLU + GSM8K via lm-eval-harness) to check the model stays smart.

Design principles (the self-skeptical measurement rules): [PRINCIPLES.md](PRINCIPLES.md).
Mechanistic hypothesis: [HYPOTHESIS.md](HYPOTHESIS.md).
Capability eval details: [CAPABILITY.md](CAPABILITY.md).

**Reproducibility:** installable package, unit tests + CI, every run writes full
provenance (config, git commit, data hashes) and auto-backs-up the adapter +
results to the HF Hub. `pull_results.py` fetches any run's resistance +
capability tables.

---

## Results so far (headline)

Qwen2.5 models, paper-faithful recipe, MPS/Colab-T4. Steered = fraction hijacked
into the injected wrong answer; lower is better after training.

| setup | resistance (steered, base→trained) | clean cost | capability |
|---|---|---|---|
| **0.5B, paper recipe + Alpaca** | steer_train α0.8: 94→**30%**; heldout α0.8: 92→56% | 100→**100%** | pending (GPU) |
| 0.5B, paper recipe, no Alpaca | steer_train α0.6: 65→6% | 100→100% | GSM8K **16→1.5%** (crashed) |
| 3B, paper recipe + Alpaca | training done; **eval pending** | — | pending |

**Read:** resistance **trains and generalizes** to held-out concepts, but is
**partial and decays with attack strength** — an honest, credible pattern.
Adding Alpaca replay held resistance at **zero clean cost**. The **open
questions**: (1) is it genuine robustness or subspace-nulling (held-out concepts
share the training vectors' subspace geometry, so they can't tell these apart —
needs the random/orthogonalized sweep)? (2) does resistance training damage
reasoning (the GSM8K signal)?

---

## Log

_Newest first._

### 22 Jul 2026 — Paper-faithful data + capability eval + Alpaca; first clean 0.5B result

**What I did:** Built the paper-faithful data recipe — a separate authored attack
bank (500 concepts / 21 categories) with **neutral-baseline contrast** and
held-out members/categories. Added the **capability eval** (MMLU/GSM8K via
lm-eval-harness) and **50% Alpaca replay** (the paper's capability-preservation
mechanism). Ran the 0.5B paper+Alpaca experiment; fixed an MPS memory leak that
was OOM-ing training; built a "resume-eval-from-HF" tool so a run whose training
finished but eval died can be completed without retraining.

**What I expected vs what happened:** Expected Alpaca replay to *rescue* the
GSM8K capability collapse (16→1.5% without it). Got the **resistance** result —
strong (steer_train α0.8 steering down to 30%) at **zero clean cost**, and Alpaca
did *not* weaken resistance despite halving the resist data. But I couldn't get
the capability number: **MMLU/GSM8K via lm-eval is impractically slow on the Mac
(MPS)** — it ran for hours on the first benchmark. So "did Alpaca fix GSM8K?" is
still open; it needs a real GPU (Colab).

**What this changes about my thinking:** (a) The **data recipe matters** —
neutral-baseline contrast + a rich, categorised attack bank gives a much cleaner
result than sibling-contrast handwritten data. (b) Capability benchmarking is a
GPU-only step, and the capability cost may concentrate in *reasoning* (GSM8K) not
*knowledge* (MMLU) — the key thing to watch at 3B, where base math is real.

**What I will do next:** Finish the 3B run (resume the eval from the saved
adapter on Colab) to get 3B resistance + capability. Then the **geometric sweep**
(random / orthogonalized attack directions) — the decisive test of genuine
robustness vs subspace-nulling.

### 15–21 Jul 2026 — Solid, reproducible pipeline

**What I did:** Built the SFT-resistance pipeline as a clean, installable package
with unit tests, CI (lint + tests on every push), full run provenance (config +
git commit + data hashes per run), automatic adapter + results backup to the HF
Hub, and optional W&B tracking. Set up a Colab launcher so a run is one click on
a GPU, and a `pull_results.py` to fetch any run's tables back down.

**What I expected vs what happened:** Expected mostly plumbing; instead spent real
time on reproducibility and environment gotchas (CUDA-driver/torch mismatch on
rented GPUs, a torchao/peft clash on Colab, a token-leak-into-provenance bug).
Each is now fixed and guarded. Nothing gets lost, and every result is traceable
to a commit.

**What this changes about my thinking:** Reproducibility discipline is a
precondition for trusting any result or mechanism claim later — worth the upfront
cost.

**What I will do next:** Run the paper-faithful experiments (data recipe,
capability, Alpaca) on this stack.

### 10 Jul 2026 — Project setup + close reading + hypothesis

**What I did:** Started the project: read the source papers closely (Steering
Awareness, and Endogenous Steering Resistance
[arXiv:2602.06941](https://arxiv.org/abs/2602.06941), which shows large models
*spontaneously* resist *benign* off-topic steering). Wrote the literature
grounding, a first-principles mechanistic hypothesis, and the design principles.

**What I expected vs what happened:** Reading the detection paper's mechanism
(fine-tuning rotates injected vectors onto a shared readout direction, and it's
*geometric* — it fails on directions dissimilar to training) reframed the whole
question. If resistance is the same kind of learned linear operation, I should
**expect it to be subspace-specific too** — strong near the training directions,
weaker far from them. So the real question is not "does resistance train?" but
"**is it genuine robustness or a geometric shortcut?**", and the eval has to be
built so it can't fool itself.

**What this changes about my thinking:** The mechanism question (genuine
self-correction vs subspace-nulling) is the crux, and the geometric-dissimilarity
sweep (random / orthogonalized attacks) is the experiment that decides it — a P0
target once the pipeline is solid.

**What I will do next:** Build the pipeline rigorously, then design the geometric
sweep.
