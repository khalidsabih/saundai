# Milestone 1 — Four-track musical mixing evidence

**Goal:** Begin measuring the distance from a technically safe automated set
to an impressive, intentional DJ performance. A continuous rendered WAV does
not by itself establish professional mixing quality.

## Run on the Mac with four tracks from audio/uploads

Run in the project checkout (not via Docker):

```bash
git fetch origin
git merge --no-edit origin/main
python -m pytest tests/test_milestone1_four_track.py tests/test_phase_lock_trace_summary.py tests/test_beat_integrity.py -q

python scripts/milestone1_four_track.py \
  "audio/uploads/AL021_-_Agents_Of_Time_-_Dream_Vision.wav" \
  "audio/uploads/AL016_-_Tale_Of_Us_-_Nova.wav" \
  "audio/uploads/Ben Böhmer - Beyond Beliefs (Official Visualiser).wav" \
  "audio/uploads/AL035 - Mind Against - Walking Away.wav" \
  --journey balanced \
  --output-dir output/milestone1-four-track
```

Use actual filename spelling on your machine. The planner **chooses its own
order**, which may differ from the argument order. All four sources must have
distinct basenames. The script stages them in a persistent `output/milestone1-four-track/crate/` directory using\nhardlinks where possible, without modifying the originals. This keeps the saved\nplan's source paths valid for inspection and rerendering.

## Outputs

- `output/milestone1-four-track/crate/`: retained original audio source references\n- `output/milestone1-four-track/milestone1-report.json`: per-edge technique,
  classification, measured score, beatmatched flag, professional mix family
  and contract data when available
- `output/milestone1-four-track/phase-rejections.jsonl`: rejected phase-lock
  candidate evidence
- `output/milestone1-four-track/balanced/full-set.wav`: finished audio
- `output/milestone1-four-track/balanced/set-plan.json`: authoritative selected
  order and transition programs
- `output/milestone1-four-track/phase16-report.json`: technical safety and
  rehearsal readiness

Existing outputs in this benchmark directory, including the staged crate, are replaced, to avoid confusing
an old passing plan with a new failed run.

## Interpret carefully

- `normal_transition` is **not** proof of an attractive sustained blend.
- `structural_reset` can be an excellent creative choice if justified by the
  actual songs.
- `emergency_escape` is distinct from a deliberate technique.
- `phase_trajectory_patterns_unique` classifies the rejected candidates into
  mostly directional drift, abrupt jumps, irregular motion, low excursion,
  and insufficient trajectory. This is a **heuristic**, not a proven diagnosis.
- The report always sets `musically_certified=false`: professional quality
  requires real listening comparison and independent evidence.
- Phase Integrity gates, default correction limits and production DSP remain
  unchanged. `DJAI_BEAT_GAP_AWARE` stays off unless independently opted into.

## First acceptance checkpoint

1. Tests and CI pass.
2. All four real tracks are planned and three transitions are rendered.
3. The Phase 16 technical result and every fallback/reset are recorded.
4. Rejected phase patterns and offending track pairs are visible.
5. Listen to each of the three transitions and describe: seamless blend,
   intentional creative handoff, unwanted reset, rhythmic mismatch, or other
   audible defect.

A technically passing benchmark is a **baseline**, not completion of the
musical milestone. Subsequent changes should demonstrate an improved listening
experience while keeping every hard safety gate.
