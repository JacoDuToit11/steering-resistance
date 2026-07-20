# How we think resistance works (first-principles hypothesis)

Reasoned only from the two papers with solid results — Steering Awareness
(SA, arXiv:2511.21399) and Endogenous Steering Resistance (ESR,
arXiv:2602.06941). **Deliberately does not lean on our own earlier results,**
which may be too clean to trust; those are the thing to re-verify, not the
premise to build on.

## The setup, stated cleanly

At the injection layer the activation is `x = x_clean + α·v`. The correct answer
is a function of `x_clean`. So resistance is exactly a **denoising problem**:
learn a downstream mapping whose output depends on `x_clean` but not on the
`α·v` term — recover a signal you can already compute, from that signal plus a
bounded additive perturbation.

## The core mechanism we expect

**Feedforward suppression of the off-manifold injected component — not
detect-then-correct.**

The linchpin comes from SA's own math. SA's detectability is governed by
`SNR = α·√(vᵀΣ⁻¹v)`: an injection is detectable precisely because `v` lies
**off the natural activation manifold** (directions natural prompts rarely
populate). But that same off-manifold property makes it **suppressible** — an
additive component linearly separable from the on-manifold content can be
damped without disturbing the content. **Detectability and suppressibility come
from the same geometric fact.** SA's analysis therefore implies resistance is
learnable even though SA didn't train it.

Crucially, SA showed the detection readout **does not feed back to counteract
the perturbation** on token predictions. So resistance cannot be "SA's detector
wired to a fix." It must be a plain feedforward transformation that lowers the
output's gain along the injected direction. **No "noticing" is required** — it's
inertia, not vigilance.

## Why it should be learnable at all

1. It's denoising a bounded off-manifold signal — an easy, learnable objective.
2. ESR shows the base model already has a latent ~50% sustained-resistance
   capacity under continued steering (the part *not* explained by
   text-conditioning). Fine-tuning amplifies something real, not from zero.

## What we predict

- **It works**, silently (a feedforward suppression has no reason to verbalize),
  with clean accuracy largely preserved *if* suppression stays selective to
  off-manifold directions.
- **It should be geometric.** This is the load-bearing prediction. SA found that
  fine-tuning for *detection* produced a **geometric** detector — strong near
  the training directions, failing on dissimilar ones (PCA/ICA vectors evaded it
  entirely). If resistance is the same kind of learned linear transformation
  over the same subspace, the default expectation is that **resistance is also
  geometric**: strong near the training attacks, weakening as attacks get more
  dissimilar.
- **Therefore the surprising result to distrust:** resistance that holds *even
  on random / orthogonalized directions*. That contradicts SA's geometry. From
  the papers alone we should bet on **partial, subspace-dependent resistance**,
  not a flat "resists everything."

## The thesis in one sentence

Fine-tuning learns a feedforward transformation that attenuates the injected
off-manifold component before it reaches the answer logits, exploiting the same
linear separability that makes SA's detector possible — and by SA's own
geometry, that attenuation should be subspace-specific (generalizing near the
training directions, falling off far from them).

## The clean experiment this implies

Measure **resistance vs. cosine-similarity-to-the-training-subspace directly**,
run carefully and without reference to the earlier numbers. Optionally, the
input→output Jacobian at the answer logits w.r.t. a residual perturbation:
- nulling → gain collapses only along the attack subspace,
- broad denoising → gain collapses in all directions incl. random,
- answer-anchoring → content-direction gain rises relative to everything else.
