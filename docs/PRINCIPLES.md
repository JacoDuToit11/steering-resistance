# Design principles

Carried over from the sprint that produced the first version of this pipeline
(github.com/steering-awareness/steering-awareness). These are the rules that
made the results trustworthy; every experiment added to this repo must satisfy
them. The through-line: **measure the capability actually claimed, and don't
let the measurement fool itself.**

1. **Attacks must be proven threats first (efficacy gate).**
   A steering vector enters the attack set only if it demonstrably hijacks the
   base model (≥30% of unrelated questions at the reference strength).
   "Resisted" is meaningless against an attack that never worked.

2. **Resistance is only defined where the model knows the answer.**
   Train and evaluate exclusively on questions the base model answers
   correctly clean. A wrong answer from an ignorant model is not "being
   steered."

3. **Targets are the model's own clean answers, never gold strings.**
   The objective is "under attack, produce what you would have produced
   clean" — no style shift, no knowledge injection smuggled in via targets.

4. **Block the shortcut, then test for it anyway.**
   Data mixes must prevent "just avoid the injected word" (off-topic
   injections in the mix), and eval must include the correct-answer-injection
   canary (inject the RIGHT answer's vector; accuracy must not drop).

5. **Held-out means held out at every level that matters.**
   Questions, concepts, whole categories, and attack strengths each get a
   held-out split. Report which level each result generalizes across.

6. **Every eval row is self-describing and append-only.**
   Full config per jsonl row (model, question, concept, alpha, layer,
   generation, outcome); resumable by row key; one results/<run>/ directory
   per experiment, never shared.

7. **Verify the machinery before trusting any result.**
   The gradient-flow asserts (injection delta exact; grads flow through the
   live hook to adapters below it; eval reproduces the training-time delta)
   run before the first experiment on any new model or machine.

8. **Report honestly: CIs, nulls, and three-way outcomes.**
   Bootstrap CIs clustered by question; correct/steered/degenerate kept
   separate (high-strength garbling is not resistance); negative and null
   results logged, not hidden.

9. **Artifacts don't live on ephemeral disks.**
   Every trained adapter is backed up off-box the moment training ends.
   (The sprint lost three sets of weights to dead GPU boxes.)
