# What we're doing and why

## The one-line version

Activation steering is increasingly used as a safety tool — to probe, evaluate,
and control models — on the assumption that the model is a passive target. We
show that a model can be **trained to resist steering**: to keep answering
correctly while a steering vector is actively pushing it toward a wrong answer.
If steering can be resisted, safety methods that rely on it cannot be assumed to
work.

## Background: the gap we build on

Activation steering adds a vector to a model's residual stream to change its
behaviour (Turner et al. 2023; Zou et al. 2023). It is a workhorse of
interpretability and safety work, and it is used in evaluations *as if the model
cannot tell it is being steered and cannot do anything about it*.

**Steering Awareness** (Fonseca Rivera & Africa, arXiv:2511.21399) broke the
first half of that assumption: models can be fine-tuned to *detect*, within
their own forward pass, that a steering vector was injected and what concept it
encodes (up to 95.5% on held-out concepts). But detection is not resistance —
their detection-trained models were, if anything, *more* steerable. They named
the open question directly: can a model be trained with an **explicit resistance
objective** to produce correct outputs *despite* injected perturbations?

## Our contribution

We train exactly that objective and measure whether it works and how far it
generalizes.

- **Method:** LoRA fine-tuning with a steering vector injected *live during the
  forward pass*, rewarding the model for reproducing its own clean answer. The
  model is never told an attack is happening — no detection signal anywhere.
- **Result (Qwen2.5-7B):** steering that hijacks the base model 70–90% of the
  time drops to near-zero after training, at no cost to clean accuracy. The
  resistance is **silent** (the model just answers correctly; it never announces
  the attack) and **generalizes** — to concepts, whole categories, and attack
  strengths never seen in training, and even to attack directions with the
  training directions mathematically removed.

## Why it matters

Steering is being proposed both as a way to *evaluate* models (does this
intervention change behaviour?) and as a way to *control* them (steer toward
honesty, away from harm). Both uses assume steering is a stable, reliable
channel into the model.

Our result shows that channel is **trainable-away**: a model can learn to hold
its output steady under steering, silently and generally. That undercuts the
reliability of steering-based safety evaluations — a "no behaviour change under
steering" result may mean the intervention failed, not that the model is safe —
and it means a model's resistance to steering must itself be checked, not
assumed. Combined with Steering Awareness (models can *detect* steering) and
Endogenous Steering Resistance (arXiv:2602.06941; large models sometimes resist
*spontaneously*), the picture is that activation steering should not be treated
as an invisible, dependable intervention.

## Scope, for now

We are keeping this deliberately narrow: the SFT resistance result on factual QA,
made solid and reproducible. Natural extensions — adaptive attackers, other
model families, the dual-use cost to *beneficial* steering, mechanism — are
noted in the roadmap but not part of the core claim yet.
