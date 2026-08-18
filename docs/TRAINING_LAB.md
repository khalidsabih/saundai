# SAUND.AI Training Lab

The Training Lab is the internal first-training surface for SAUND.AI. It is intentionally separate from Create and Studio.

## Purpose

The first learned model should answer a narrow question: **between two technically safe transition candidates, which one is more likely to be preferred in context?**

Hard rendered-audio failures remain non-overridable. Training does not grant a model permission to bypass clipping, silence, phase, bass, timing, or other safety gates.

## Open the lab

With the local Compose stack running, open:

```text
http://localhost:3000/lab
```

The local Compose configuration enables the lab by default. Public deployments should set:

```bash
DJAI_TRAINING_LAB_ENABLED=0
```

## Workflow

1. Add or select two tracks.
2. Generate four rehearsed candidates.
3. Listen to the candidates blind as A/B pairs.
4. Choose A, B, No preference, or Neither works. Reason tags are optional.
5. Repeat across diverse track pairs.
6. Freeze an immutable dataset snapshot.
7. Once at least 24 decisive A/B labels exist, train a shadow preference model.
8. Compare cross-validated model accuracy against the blind-position baseline.

A 24-comparison model is an engineering smoke test, not deployment evidence. The Lab marks 100 decisive comparisons as the first meaningful benchmark checkpoint.

## Dataset snapshots

Snapshots are written to:

```text
datasets/training-lab/dataset-*/
```

Each snapshot contains:

- `manifest.json` — versions, counts, hashes, rights state, and immutability contract.
- `tracks.jsonl` — content-addressed track IDs and local source references.
- `candidates.jsonl` — deduplicated transition candidates and rendered measurements.
- `preferences.jsonl` — blind A/B labels with candidate feature payloads.
- `sessions.jsonl` — listening-session provenance.
- `feature_schema.json` — exact ranker feature order and schema hash.
- `pairwise_matrix.npz` — model-ready `X`, `y`, weights, groups, splits, and feature names.

Raw source audio is not copied into the dataset snapshot.

## Shadow models

Training Lab models are written to:

```text
config/training-lab-models/
```

They are marked `shadow_only=true` and `promoted_to_live=false`. Training a model in the Lab does **not** replace:

```text
config/transition-ranker.json
```

Promotion should be a separate, explicit future step after benchmark and listening review.

## Rights state

Local experimentation can proceed without claiming production readiness. For a snapshot to be marked production-training-ready, both the production dataset registry and the local source-audio rights flag must be ready.

Set this only when the tracks used for training are cleared for the intended model-training use:

```bash
SAUND_TRAINING_SOURCE_RIGHTS_CONFIRMED=1
```

The flag is provenance metadata, not a license grant.

## Recommended first dataset

Start with 200–500 diverse rights-cleared tracks over time, but do not wait for that scale before validating the pipeline. The first useful milestone is 24–100 high-quality decisive blind comparisons across many track pairs. Prefer diversity and careful listening over mass labeling.

## SAUND.AI 1.1.1 taste-learning loop

1.1.1 keeps blind human preference and rendered technical judgement as two separate learning heads.

### Global Taste Ranker

The Global Taste Ranker learns pairwise preference from structured candidate features. Decisive A/B votes are the primary supervision. It is written to `config/training-lab-models/` and remains shadow-only; the 1.1.1 professional judge logs what it would have preferred without changing the chosen candidate.

### Rendered DJ Judge

The Rendered DJ Judge learns from the actual rendered outcome representation: human-labeled rendered candidate features/embeddings plus measured synthetic technical failures. Snapshot creation can corrupt the transition section of local blind-session preview audio in memory, measure the result, and store only the training features and provenance. Corrupted WAVs are not retained.

The synthetic failure families are clipping, silence gaps, gain jumps, phase inversion, transient doubling, bass overload, and timing shifts. They establish technical discrimination; blind human comparisons provide subjective DJ-taste evidence.

Snapshots now include `rendered_judge.jsonl` and report separate human/synthetic counts. Validation is grouped by track pair when multiple pairs are available.

### Runtime rehearsal evidence

Real quality-gate rehearsals append structured candidate evidence to `ratings/runtime-candidate-evidence.jsonl` by default. The evidence is audio-free, pseudonymizes local track paths, strips listener identity, and records selected/gate/hard-failure outcome context. It lets later experiments compare what SAUND proposed, rejected, and selected.

```env
DJAI_TASTE_EVIDENCE_CAPTURE=1
DJAI_TASTE_SYNTHETIC_FAILURES=1
DJAI_TASTE_SYNTHETIC_MAX_CANDIDATES=128
```

### Train/status commands

```bash
./scripts/saund.sh training-snapshot --name taste-learning
./scripts/saund.sh training-taste-train
./scripts/saund.sh training-taste-status
```

A combined train operation may legitimately train one head and block the other when its evidence minimum is not met. It never fabricates readiness. The Rendered DJ Judge requires at least two independent track-pair groups for held-out evaluation; the Global Taste Ranker retains the existing minimum of 24 decisive A/B comparisons and at least two track pairs.

If the registry's Rendered DJ Judge is still shadow/disabled, production-unapproved, and not hash-pinned, training may refresh its checkpoint so runtime shadow predictions are visible immediately. Registry authority remains zero. Training refuses to overwrite advisory, production-approved, or pinned checkpoints.

Hard rendered-audio safety remains authoritative for every mode and cannot be overridden by either learned head.

## SAUND.AI 1.1.2 responsive blind generation

Blind-pair generation reuses `.djai-cache/tracks`, the same canonical analysis cache used by normal set planning. If both songs have already been through Create Set, the Lab should report two analysis cache hits instead of re-running full track analysis. The Lab shows an elapsed render indicator while candidate rehearsal/measurement/preview rendering is still synchronous. The listening-session response includes stage timings and cache-hit counts for latency diagnosis.
