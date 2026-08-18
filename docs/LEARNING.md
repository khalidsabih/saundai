# DJAI 5.0 learning architecture

DJAI separates perception, technical truth, musical coherence, artistic preference and expert imitation.

## Neural ears vs DJ taste

Beat/downbeat and foundation music encoders provide evidence. They are not professional-DJ preference models. Generic embedding similarity never receives direct authority as transition taste.

## Structured DJ-specific heads

5.0 adds trainable heads for transition family, duration, stem choreography, rendered-transition preference and next-track value. Checkpoints are small structured JSON models with feature normalization and held-out metrics. They are governed by the model registry.

## Technical quality

Automatic render measurements and synthetic corruptions are valid supervision for objective failures such as clipping, silence, discontinuity, phase risk, beat/transient incoherence and bass overload. They are **not** proof of artistic taste.

## Pairwise artistic preference

Taste Lab records A/B/Tie/Neither plus optional reason tags. `Neither` supplies absolute rejection evidence. The rendered DJ Judge is intended to learn from real rendered-transition preferences rather than from self-generated heuristic labels.

## Expert imitation

Expert demonstrations remain a separate learning signal: next track, cues, transition family, duration, tempo movement, stem choreography and set context. This answers “what do skilled DJs tend to do here?” rather than “which rendered result sounds better?”.

## Rights-cleared neural training

`neural-train` defaults to `rights_cleared=true` rows. Production promotion requires held-out performance gates, explicit rights review, deployment-ready blind Arena evidence, explicit governance approval, and a pinned checkpoint hash. Research/reference datasets do not silently become production training data.

```bash
djai neural-train transition_family datasets/transition-family.jsonl -o config/neural/transition-family.json
djai models-promote transition_family_policy advisory
```

## Model authority

```text
shadow     -> observe only
advisory   -> bounded residual after confidence gate
production -> larger bounded residual + explicit production approval
```

A fresh model does not control the brain merely because it fits its training set. Use the Arena and fixed benchmark crate before promotion.

## Active learning

Listening sessions should prioritize close/uncertain alternatives so professional-DJ time is spent where the model has useful uncertainty. The highest-value labels are genuinely different transition families, durations and stem arrangements—not cosmetic variants of one crossfade.

## Curriculum

The intended order is:

```text
neural perception
-> self-supervised musical coherence
-> technical rendered quality
-> transition preference
-> expert imitation
-> set strategy
-> personalization
-> live adaptation
```
