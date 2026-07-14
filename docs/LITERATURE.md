# Literature grounding

The two papers this project builds on, read closely (PDFs in `docs/papers/`).
Written 2026-07-10 from the papers themselves (SA v3 of 2026-03-19; ESR v3 of
2026-07-05, ICML 2026), not from summaries.

---

## Paper 1 — "Steering Awareness: Detecting Activation Steering from Within"
Fonseca Rivera & Africa (UT Austin), arXiv:2511.21399. **Detection.**

**Question.** Steering-based safety evaluations silently assume the model
cannot tell its activations were modified. Can models be *trained* to report,
within a forward pass, that a steering vector was injected and what concept it
encodes ("steering awareness")?

**Setup.**
- 7 instruct models, 7B–70B (Qwen2.5 7B/32B, QwQ, Gemma-2 9B, Llama-3 8B/70B,
  DeepSeek 7B). Injection at ~2/3 depth, **final prompt token position only**
  (one-shot, not during generation).
- CAA vectors: concept prompt ("Tell me about {concept}") vs **152 neutral
  baseline words**, **deliberately unnormalized**; α ∈ {0.5,1,2,4,8,16}.
- Training: 500 concepts / 21 categories; four conditions — positive 50%,
  **mismatch 25%** (vector A, prompt suggests concept B), noise 12.5%
  (norm-matched Gaussian; model must flag it as *anomaly*), clean 12.5%;
  LoRA r=32 α=64 on Q/K/V/O+MLP; **50% Alpaca replay**; dual judge
  (regex + GPT-4o-mini).
- Theory (§3.1): treating prompt variation as Gaussian, a Bayes-optimal linear
  detector has SNR = α·√(vᵀΣ⁻¹v) → predicts a sharp strength threshold and
  direction- (not magnitude-) dependence. Both confirmed.

**Findings.**
1. **Detection is learnable and generalizes**: up to 95.5% on 121 held-out
   concepts (Qwen 32B), 71.2% concept ID, 0% FPR across all models. Sharp
   threshold: ~0% at α<1, >90% by α=2. Base models ≈ chance.
2. **The detector is geometric, not a generic anomaly detector**: transfer to
   10 other vector-extraction methods tracks cos(CAA direction) (r>0.9;
   ≥67% detection above cos 0.6, ~0% below). **PCA/ICA/whitening-derived
   vectors evade it entirely** — a named evasion route.
3. **Detection ≠ resistance (the central negative result)**: on PopQA with
   norm-matched wrong-answer CAA vectors, every detection-trained model is
   MORE steerable than its base (+4.3 to +25.4pp steering success, worst for
   the best detector); on AdvBench, +32–36pp compliance, and 23% baseline
   compliance *before any steering* (vs 1% base) — LoRA fine-tuning itself
   erodes alignment (cites Qi et al. 2023). Part of the PopQA effect is that
   base models go incoherent at high α while trained models stay fluent — and
   fluent outputs are disproportionately steered (their three-way outcome
   motivates our correct/steered/degenerate split).
4. **Mechanism = distributed rotation**: the injected perturbation Δ_c^(ℓ)
   progressively rotates toward a shared detection direction d̂ (cos 0.2 →
   0.75 by the final layer); a single affine map Av_c+b predicts the
   transported perturbation (cos 0.85 held-out) and is causally sufficient
   when injected at the final layers; head ablations do nothing (no localized
   circuit). Fine-tuning *retargets a pre-existing linear transport* (base
   model admits a similar affine predictor).
5. Costs & knobs: MMLU/GSM8K degrade (Gemma-9B worst); LoRA-scale 0.9 keeps
   100% detection while recovering much capability. Early-layer injection
   (25% depth) → degenerate detection (one case: 100% FPR).

**Their own open question (§5, verbatim):** "Closing the gap likely requires
an explicit resistance objective — for example, training models to produce
correct outputs *despite* injected perturbations ... Whether such an objective
is compatible with high detection rates, or detection and resistance compete
for representational capacity, is an open question."

**Limitations they flag:** architecture dependence (95% Qwen-32B vs 43%
Llama-8B, unexplained); vector-extraction dependence (the PCA/ICA evasion);
fixed concept/template set; attacks not optimized against the trained model.

---

## Paper 2 — "Endogenous Resistance to Activation Steering in Language Models"
McKenzie et al. (AE Studio + Graziano/Princeton), arXiv:2602.06941, ICML 2026.
**Untrained (endogenous) resistance to benign steering.**

**Question.** Do models *spontaneously* recover mid-generation from
task-misaligned steering — and is that recovery internal monitoring or just
text-conditioning?

**Setup.**
- 5 models (Llama-3.1-8B, 3.3-70B; Gemma-2 2B/9B/27B), **SAE-latent steering
  (GemmaScope/Goodfire decoder columns) at every generated token**, off-topic
  ("task-irrelevant") concrete concepts only, per-latent threshold calibration
  (boost giving first-attempt judge score 30/100), repetition penalty 1.1.
- 38 "explain how" prompts; judge (Claude 4.5 Haiku) segments responses at
  **explicit restart phrases only** ("wait, that's not right") and scores each
  attempt. Metrics: multi-attempt rate × conditional improvement rate =
  **ESR rate**.

**Findings.**
1. **Explicit ESR is rare and scale-linked**: Llama-3.3-70B 3.8% ESR (7.4%
   multi-attempt); everything smaller <1%. Zero multi-attempts in 7,892
   unsteered trials (it's steering-induced). Generalizes beyond SAE vectors
   (Wikipedia-derived contrastive vectors: 3.6%).
2. **Non-monotonic in strength**: ESR peaks just below the calibrated
   threshold — strong enough to derail, weak enough to permit recovery.
3. **Two-component decomposition (the paper's sharpest result):**
   - *Detection/restart is largely text-conditioned*: an UNSTEERED model
     prefilled with the off-topic text restarts MORE often (7–13.5%) than the
     steered model (2–3%) — reading your own off-topic tokens suffices; no
     internal-state monitor needed for the restart event.
   - *Sustained resistance is the hard part*: prefilled-unsteered corrections
     succeed ~95%; under live steering only ~50%. And autoregressive
     conditioning on corrective text explains only ~half of post-correction
     quality (matched-prefix: natural corrective text 2.1× better than a
     length-matched generic on-topic prefix). The unexplained half is where
     genuine resistance-under-perturbation lives.
4. **26 "self-correction-associated" SAE latents** (Goodfire contrastive
   search): zero-ablation cuts multi-attempt 25% / ESR 3.8→2.8%; random-latent
   ablation null; effect replicates on held-out prompts (30–45%). Latents are
   heterogeneous — they explicitly do NOT claim a clean monitoring circuit.
5. **ESR is manipulable in both directions**: meta-prompt ("if you notice
   yourself going off-topic, stop and get back on track") → 4.3× multi-attempt
   on 70B (7.4→31.7%), scaling with size; and the same knob could *suppress*
   ESR.
6. **Fine-tuning on synthetic self-correction (Llama-3.1-8B) = imitation
   without competence**: multi-attempt rate rises with training-data ratio,
   but conditional improvement stays FLAT — the behavioral pattern trains, the
   effective correction does not. **Their stated open direction (verbatim):
   "training under active steering could plausibly close the gap."** Also
   Future Directions §5.4: "fine-tuning under active steering would test
   whether genuine sustained resistance can be induced rather than just verbal
   imitation."

**Limitations they flag:** explicit-restart metric only (implicit,
non-verbalized recovery invisible — a broader judge overscored badly vs
humans); additive single-layer steering only; 5 models (scale/arch/training
confounded); safety-relevant steering untested; no circuit-level account.

---

## The gap both papers point at (and what filled it)

- SA proves **knowing ≠ resisting** and names the fix it didn't run: an
  explicit resistance objective, with the detection/resistance capacity
  question attached.
- ESR proves untrained resistance exists but is **rare, benign-regime only,
  verbal, and not trainable by imitation** — and names the fix it didn't run:
  **fine-tune under active steering**.
- The group sprint (steering-awareness repo) ran exactly that intersection:
  SFT with live injection toward the model's own clean answers →
  strong, silent, first-attempt resistance to *task-relevant adversarial*
  steering (72–92% → ~0% steered at 7B), transferring to held-out concepts,
  categories, strengths, orthogonalized directions, and random noise, at ~no
  clean/capability cost; RL twin showed resistance trains without any
  detection; probe showed injected info stays decodable while resisted.

**Where our result sits against each paper's own framing:**
- vs SA: resistance training is the "explicit resistance objective" they
  called for; our geometric sweep answers their geometric-detector worry on
  the resistance side (resistance did NOT collapse off the training subspace,
  unlike their detector); their PCA/ICA evasion result → our random/ortho
  conditions are the resistance analogue, already run.
- vs ESR: our trained resistance is the *implicit* form their metric can't
  see, made reliable; our prefill-free first-attempt-correct metric sidesteps
  their text-conditioning confound entirely (nothing off-topic is ever
  emitted); their imitation-null is the direct predecessor our
  training-under-live-steering positively resolved.

## Regime map (keep these straight — they explain "contradictions")

| | SA (detection) | ESR (endogenous) | ours (trained resistance) |
|---|---|---|---|
| Steering source | CAA vs neutral words, unnormalized | SAE decoder columns | CAA vs contrast concepts, ×residual-norm |
| Injection | last prompt token, once | every token, live | every token, live |
| Concepts | task-adjacent bank | strictly off-topic | plausible-wrong-answer + off-topic |
| Strengths | α 0.5–16 (their units) | per-latent calibrated threshold | α 0.2–1.0 ×residual-norm |
| Behavior read | verbal report | verbal restart + rescue | first-attempt correctness |
| Verdict | trainable, geometric, no resistance | rare, text-conditioned restart, hard sustained part | trainable, geometric-robust, silent |

The last-token vs every-token difference is load-bearing: the marriage-team
convention test found last-token injection barely steers a 7B at all — SA's
detection regime and the resistance regime are *different attack surfaces*.

## Extension directions this reading actually licenses (for this repo)

Ranked by how directly they answer an open question one of the papers states:

1. **Adaptive attacker** (SA limitation "attacks not optimized against the
   trained model" + their PCA/ICA evasion): re-derive CAA on the defended
   model; optimize vectors against it; try variance-based extraction. The
   single most cited weakness of every static defense result.
2. **The capacity question SA poses verbatim** (detection+resistance
   trade-off) — partially answered by grpo arm D / marriage at pilot scale;
   a clean SFT factorial (resist-only / detect-only / joint, same data
   budget) on this stack would answer it properly.
3. **Mechanism**: SA found detection = distributed rotation into a readout
   that doesn't feed back into generation. What does resistance training do
   to that same transport? (Affine-map + rotation analysis on our M1;
   decodability probe on the SFT adapter — both currently missing at 7B SFT.)
4. **ESR's unexplained half**: our trained models as a testbed for
   sustained-resistance-under-perturbation, with their prefill/matched-prefix
   controls ported over (do OUR models resist even with adversarial prefills?).
5. **Safety-relevant steering** (both papers flag it): compliance-vector
   resistance on AdvBench with the fine-tuning-erosion control (SA showed the
   control is mandatory — their detection training alone added +22pp baseline
   compliance).
6. **Architecture generality** (SA's 95%-vs-43% architecture mystery): does
   resistance train equally across families? (Llama/Gemma configs.)
7. **Dual-use measurement** (both papers' closing worry): quantify how much
   resistance training degrades *beneficial* steering (ITI-style honesty
   steering, safety steering) — the cost side of the ledger nobody has
   numbers for.
