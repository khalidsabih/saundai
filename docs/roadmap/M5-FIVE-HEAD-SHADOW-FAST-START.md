# SAUND.AI — five-head shadow v0.1 fast start

**Decision (2026-11-17):** get independently measurable, zero-authority baselines for all five DJ decision heads quickly, using existing training infrastructure. Do not pretrain foundation models from scratch. Do not imply any v0.1 weights exist until checkpoint hashes and held-out results are recorded.

## Fixed rules

1. A baseline is a real checkpoint with a recorded training command, dataset hash, task-specific labels, at least one honest validation metric, explicit source rights, and source-group-disjoint evaluation. A class stub, synthetic-only rehearsal, or imported dataset alone is not a completed baseline.
2. All trained heads stay **shadow**, with authority **0.0**. No production registry writes or live promotion. Synthetic supervision must be tagged and evaluated separately from human/expert evidence.
3. Never train on Ballroom. Ballroom is reserved for independently labeled rhythm testing; SAUND's original 8-WAV pilot has not received benchmark approval. Groove full audio is for separate M1 rhythm development/evaluation only and does not directly supervise DJ policy choices.
4. Use existing `src/djai/training/neural_heads.py`, `src/djai/training/dataset_builder.py`, `src/djai/training/research_synthetic.py`, and `docs/TRAINING_LAB.md`; don't create replacement model architectures.
5. Split by **source mix / source song pair / musical session**, not random candidate row. Keep untouched test sets sealed until the end. Reject train/test leakage, missing provenance, forged positive labels, and unclear rights.
6. Compare against current deterministic SAUND logic on the *same frozen cohort*. Only train heads whose dataset has real task-matching supervision; label-free track collections are not suitable targets.

## Training sequence (fastest useful feedback first)

| Head | First honest supervision and candidate source | v0.1 criterion | Next upgrade |
|---|---|---|---|
| `duration_policy` | UnmixDB exact controlled overlap timings; separately, verified DJ mix transition durations if acquisition and training rights permit | A real numeric duration checkpoint vs fixed heuristic duration baseline; held-out error by source group | Expert real-mix overlap durations conditioned on phrase/energy |
| `transition_family_policy` | Controlled UnmixDB mix/effect labels mapped to a conservative explicit supported-technique taxonomy | Train only on interpretable, quality-checked technique labels. If effect is not technique, mark head blocked rather than invent labels | Expert-labeled real DJ transition families |
| `next_track_policy` | Existing synthetic next-track research snapshot; optionally rights-reviewed ListenBrainz sequence edges. Synthetic model is *research* v0, not expert model | Shadow ranking metrics and frozen-sequence baseline; visibly report synthetic-vs-real label origin | Intent-conditioned expert DJ sets; negatives from the same crate |
| `rendered_dj_judge` | Existing Training Lab blind A/B judgments and measured technical failures | First 24 decisive A/B across 2+ pair groups: engineering smoke test only. 100 decisive: initial meaningful checkpoint, not deployment readiness | Diverse multi-genre blind sessions; separate hard-safety and taste metrics |
| `stem_choreography_policy` | Slakh2100 multitrack audio/MIDI for **stem representations only**. No canonical DJ stem-action labels exist there. Use explicitly synthetic action labels for a research smoke test, not taste quality | Head remains **data-blocked for real supervision** until independent source-aligned, rights-cleared stem-action demonstrations exist; a synthetic stub must be marked as such | Real rendered stem transitions with expert/human action preferences |

## Immediate work that does NOT require another architecture rewrite

1. Freeze five task schemas and evaluation objectives in this table; record corpus source, terms, splits, labels, source mix/session, hashes and status before training.
2. Use the existing importer for **UnmixDB** as the first controlled technical corpus. Verify each asset's actual terms and checksum before training, and reserve source-group holdout. Its automatic mixes are not expert preferences.
3. Use the existing `research-synthetic-validate`, `research-synthetic-import`, and `research-synthetic-train` path if the synthetic corpus is actually present. This may provide a **research-only next-track checkpoint** without claiming DJ expertise.
4. Run existing Training Lab blind preference collection for a first Rendered Judge / Global Taste baseline. A head without enough real labels is **data-blocked**, not silently trained on fabricated ratings.
5. For `transition_family_policy` and `stem_choreography_policy`, build small accurately labeled, rights-cleared demonstration sets. Controlled effects/stems can initialize features, not supply a musical-quality oracle.
6. Report first model scorecards **separately**. Prefer one runnable verified model over five untested checkpoints. Upgrade each only after a comparable improvement.

## Source locations

- Model roster and default shadow authority: `config/model-registry.json`, `src/djai/intelligence/model_registry.py`.
- Five-head trainer: `src/djai/training/neural_heads.py`.
- Existing source catalog: `src/djai/training/source_catalog.py` (UnmixDB, FMA, EDM-CUE, ListenBrainz, Medley2K).
- Existing source normalization/bootstrap trainers: `src/djai/training/dataset_builder.py`.
- Real human preference workflow: `docs/TRAINING_LAB.md`, `src/djai/training/taste_learning.py`.
- Synthetic-only next-track branch: `src/djai/training/research_synthetic.py`.
- Rhythm evaluation scope: `docs/roadmap/M1-BALLROOM-REAL-AUDIO-PILOT.md`, `docs/roadmap/M1-GROOVE-MIDI-METADATA-PILOT.md`.

## Per-head scorecard contract

```json
{
  "model_id": "duration_policy",
  "checkpoint_sha256": null,
  "dataset_snapshot_sha256": null,
  "license_review": "pending",
  "label_source": "not_yet_verified",
  "train_groups": null,
  "validation_groups": null,
  "test_groups_sealed": true,
  "baseline_metric": null,
  "shadow_metric": null,
  "authority": 0.0,
  "status": "not_trained"
}
```

The above is a **template**, not evidence of an existing checkpoint. Store real reproducible values per head. Tasks can enter different readiness states; don't block valid tests waiting for all heads.

**Milestone order remains M0→M6**; starting M5 datasets and shadow experiments early does not authorize bypassing M1 rhythm or M2/M3 safety/quality gates.
