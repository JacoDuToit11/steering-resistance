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
techniques stop working — so it's important to know whether resistance is
trainable, how well it generalizes, and what it costs.

Full framing: [MOTIVATION.md](MOTIVATION.md) · [LITERATURE.md](LITERATURE.md)
(close reading of the two source papers).

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

| model / setup | resistance (steered, base→trained) | clean cost | capability |
|---|---|---|---|
| **0.5B, paper recipe + Alpaca** | steer_train α0.8: 94→**30%**; heldout α0.8: 92→56% | 100→**100%** | pending (see log) |
| 0.5B, paper recipe, no Alpaca | steer_train α0.6: 65→6% | 100→100% | GSM8K **16→1.5%** (crashed) |
| 3B, handwritten data | steer_train α0.6: 57→29% | 100→100% | — |
| 3B, paper recipe + Alpaca | training done; **eval pending** | — | pending |
| 7B + PopQA (group pilot) | 72–92% → ~0% at all strengths | 100→100% | MMLU +0.9, GSM8K −7 |

**Read:** resistance **trains and generalizes** to held-out concepts, but is
**partial and decays with attack strength** — an honest, credible pattern (the
group pilot's near-perfect 0% is exactly what a geometric shortcut would also
look like, which is why the mechanism question is open). Adding Alpaca replay
held resistance at **zero clean cost**. The **open questions**: (1) is it genuine
robustness or subspace-nulling? (2) does resistance training damage reasoning
(the GSM8K signal)?

---

## Log

_Newest first._

### 22 Jul 2026 — Paper-faithful data + capability + Alpaca; first clean 0.5B result

**What I did:** Ported the paper-faithful data recipe into the solid pipeline —
separate authored attack bank (500 concepts / 21 categories) with
**neutral-baseline contrast** (not sibling contrast), and held-out
members/categories. Added the **capability eval** (MMLU/GSM8K via
lm-eval-harness) and **50% Alpaca replay** (the paper's capability-preservation
trick). Ran the 0.5B paper+Alpaca experiment; fixed an MPS memory leak that
OOM'd training; built a "resume-eval-from-HF" tool so a run whose training
finished but eval died can be completed without retraining.

**What I expected vs what happened:** Expected Alpaca replay to *rescue* the
GSM8K capability collapse (16→1.5% without it). Got the **resistance** result —
strong (steer_train α0.8 down to 30% steered) at **zero clean cost**, and Alpaca
did *not* weaken resistance despite halving the resist data. But I couldn't get
the capability number: **MMLU/GSM8K via lm-eval is impractically slow on the Mac
(MPS)** — it ran for hours on the first benchmark. So "did Alpaca fix GSM8K?" is
still unanswered; it needs a real GPU (Colab).

**What this changes about my thinking:** Two things. (a) The **data recipe
matters more than model size** — paper-recipe 0.5B beat the handwritten 3B on
resistance. (b) Capability benchmarking is a real infra constraint (GPU-only),
and capability cost may concentrate in *reasoning* (GSM8K), not *knowledge*
(MMLU) — the key thing to watch at 3B/7B where base math is real.

**What I will do next:** Finish the 3B run (resume the eval from the saved
adapter on Colab) to get the 3B resistance + capability numbers. Then the
**geometric sweep** (random / orthogonalized attack directions) — the decisive
test of genuine robustness vs subspace-nulling.

### 15–21 Jul 2026 — Ground-up rebuild: solid, reproducible pipeline

**What I did:** Rebuilt the SFT-resistance pipeline from scratch as a clean,
installable package (`steering-resistance`, separate from the group sprint code)
with unit tests, CI (lint + tests on every push), full run provenance
(config + git commit + data hashes per run), automatic adapter + results backup
to the HF Hub, and optional W&B tracking. Validated it reproduces the pilot's
dummy-scale numbers. Set up a Colab launcher so runs are one click on a GPU.

**What I expected vs what happened:** Expected a quick code import; instead spent
real time on reproducibility infra and environment gotchas (CUDA-driver/torch
mismatch on rented GPUs, a torchao/peft clash on Colab, lost-adapter incidents).
The infra investment paid off — nothing gets lost now, every result is traceable
to a commit.

**What this changes about my thinking:** Reproducibility discipline isn't
overhead here — the group sprint lost several trained adapters to dead GPU boxes,
and results were hard to trust. A solid base is a precondition for believing any
mechanism claim.

**What I will do next:** Run the paper-faithful experiments (data recipe,
capability, Alpaca) on this stack.

### 10 Jul 2026 — Close reading + decision to rebuild

**What I did:** Read both source papers closely (Steering Awareness; and
Endogenous Steering Resistance, [arXiv:2602.06941](https://arxiv.org/abs/2602.06941),
which shows large models *spontaneously* resist *benign* off-topic steering).
Wrote up the literature grounding, a first-principles mechanistic hypothesis, and
the design principles. Decided to fork off a clean personal repo rather than
build on the sprint code.

**What I expected vs what happened:** The group pilot's 7B result (steering
72–92% → ~0% *everywhere*) looked almost *too* clean. Realized this is exactly
the signature of **subspace-nulling** (the model deleting the region of
activation space the training vectors occupy) rather than genuine self-correction
— and held-out concepts can't tell the two apart because they share that region.

**What this changes about my thinking:** The interesting question isn't "does
resistance train?" (it clearly does) but **"is it real robustness or a geometric
shortcut?"** — and the measurement has to be built to not fool itself.
Hypothesis: resistance is a *feedforward suppression of the off-manifold injected
component*, which (per the detection paper's own geometry) should be
subspace-specific — so I should *expect* it to weaken on dissimilar directions.

**What I will do next:** Rebuild solid, then design the geometric-dissimilarity
sweep as the decisive experiment.

### 8–9 Jul 2026 — Group sprint pilot (origin of the project)

**What I did:** As part of a small team (2.5-day sprint), built the first
SFT-resistance pipeline and trained a model to answer correctly under live
steering. Got a strong headline on Qwen2.5-7B + PopQA and a first
(dummy-scale) geometric sweep.

**What I expected vs what happened:** Resistance trained startlingly well —
steering success collapsed to ~0% and generalized to held-out concepts at no
clean-accuracy cost. The dummy-scale geometric sweep pointed *away* from pure
subspace-nulling (the trained model recovered even under random directions).

**What this changes about my thinking:** Resistance is trainable and looks
genuine at small scale — promising enough to pursue properly. But the sprint was
fast and the strongest result was suspiciously clean, so it needs rigorous
re-validation before any claim.

**What I will do next:** Rebuild the pipeline rigorously as an individual project
(→ everything above).
