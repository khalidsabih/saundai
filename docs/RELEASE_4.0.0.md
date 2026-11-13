# DJAI 4.0.0 release notes — Professional Brain Core

> **Historical status:** This document records an earlier implementation/release milestone. It is retained for provenance and must not be used as the current runtime/version contract. See [CANONICAL_ARCHITECTURE.md](CANONICAL_ARCHITECTURE.md) for current architecture, authority and schema sources of truth.


DJAI 4.0.0 reorganizes automatic decision making around a professional-DJ-brain loop while preserving the tested deterministic renderer and safety contracts from 3.x.

## Major changes

- Bar-synchronous multi-scale TrackMap.
- Persistent multidimensional SetState.
- Receding-horizon set-policy seeds.
- Diversity-aware transition imagination.
- Immutable TransitionProgram and fingerprint in the execution signature.
- Private rehearsal of actual transition audio before commitment.
- New musical-execution measurements and rendered-transition embeddings.
- Structured Professional DJ Judge with hard safety, artistic/context evidence, rejection memory and uncertainty.
- Retry/expanded-search loop for weak or uncertain candidates.
- Immediate rejected-transition memory.
- TransitionRanker v2 with rendered/musical/context features and `Neither` evidence.
- Separate expert-imitation policy.
- Taste Lab reason tags and active-learning pairing.
- Expert dataset schemas, synthetic technical-failure generator and training curriculum.
- Research/production data provenance audit with default-deny production policy.
- Permanent champion-vs-challenger DJAI Arena.
- Live manifest primary/backup/emergency alternatives.
- Native Python setup hardening and isolated CUE-DETR reference environment.

## Important boundary

This release implements the architecture needed to learn professional taste; it does not bundle invented expert taste. Pairwise preference and imitation models gain authority only from real data and held-out validation.

The current TransitionProgram is a semantic immutable control contract over existing deterministic DSP primitives. Fully generic arbitrary automation execution and crowd-reactive live switching remain future milestones.
