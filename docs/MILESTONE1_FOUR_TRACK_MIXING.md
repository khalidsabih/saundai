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

## AI knowledge and decision audit (read-only)

Before changing beat or mixing heuristics, inspect what the system actually
recorded and which AI models are installed locally. This **does not** rerun
planning, render audio, change safety limits, or promote model authority.

```bash
python scripts/audit_ai_knowledge.py \
  --set-plan output/milestone1-four-track/balanced/set-plan.json \
  --output output/milestone1-four-track/ai-knowledge-report.json
```

The output separates:

1. **Track knowledge:** detected BPM, beat confidence, structural confidence,
   instrument/vocal statistics, embeddings, cue counts and neural evidence.
2. **Model inventory:** registry authority, effective weight, checkpoint
   presence/loadability, checkpoint feature names and available training metrics.
3. **Final decisions:** technique, duration, beatmatching, timings, ranker
   metadata, fallback and selected mix-family evidence.
4. **Unknowns:** saved `set-plan.json` stores only a truncated beat-grid
   preview and only the committed transitions, not all rejected proposals.
   Therefore this report alone cannot prove whether phase drift is real, which
   model originally caused a decision, or whether a mix sounds professional.

Model configuration is **not** proof of model accuracy or active usage. A
registered shadow model can be installed and loadable while remaining unable to
change a final DJ action. Neural beat and embedding adapters need separate
runtime probes if their saved evidence is unavailable.

The next diagnostic decision should follow from this audit: incorrect source
features imply a perception fix; unhelpful predictions imply training/ranking
work; reliable features and choices that cannot pass beat integrity imply a
phase/DSP implementation investigation.

## World model to DJ Brain utilization audit

This observational report traces saved four-track cue times to the musical
phrases in the matching cached world models, with evidence-status fields
that prevent overclaiming causal influence.

```bash
python scripts/audit_world_model_utilization.py \
  --set-plan output/milestone1-four-track/balanced/set-plan.json \
  --world-cache .djai-cache/features \
  --output output/milestone1-four-track/world-model-utilization-report.json
```

The report includes per-track phrase role counts, top transition-ready
phrases, neural evidence, the phrase around each chosen exit and entry cue,
selected fallback type, and the plan's available routing/ranking metadata.

Important: the `world_model/6.0` JSON is a **projection of the canonical
musical timeline**, not a learned world-model checkpoint. The planner
consumes the timeline directly for song-form protection and cue search.
A matching cached filename alone does not prove that the cache bytes
match the benchmark's source audio. Multiple cached matches are flagged
as ambiguous and the phrase join is suppressed. The report does not
claim to explain the complete ranking process because rejected
candidate details are not retained in the saved winning plan.

The source code establishes how signals are wired, whereas the report
establishes their observed values. To prove causality, a separate
instrumented candidate-decision trace and a controlled feature ablation
are required. Do that only after validating source features and timing.

## Opt-in transition decision trace (diagnostics, no mix changes)

The previous audit maps *final* cues to cached phrase context. To inspect
the selection evidence preceding fallback, opt in when running a **new**
four-track planning run:

```bash
export DJAI_DECISION_TRACE=1
# Re-run the same milestone1 four-track command you used for the baseline.
# Afterwards, inspect the newly produced set-plan.json:
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("output/milestone1-four-track/balanced/set-plan.json").read_text())
for i, edge in enumerate(p.get("transitions", []), 1):
    meta = edge.get("plan", {}).get("metadata", {})
    pool = meta.get("decision_trace_sequence_candidates", {})
    rejected = meta.get("decision_trace_rejected_phase", {})
    print(f"Transition {i}: pool={pool.get('total_candidates', 'not captured')} "
          f"phase_rejection={'captured' if rejected else 'not captured'}")
    if rejected:
        print("  rejected:", rejected.get("candidate"))
        print("  phase profile:", rejected.get("phase_integrity"))
PY
```

The opt-in `decision_trace_sequence_candidates` is a bounded snapshot
of the candidates prepared by the sequence planner **before** professional
rehearsal. It does **not** represent all candidates constructed internally
by the professional brain. If the finalization gate rejects a selected
beatmatched plan, `decision_trace_rejected_phase` retains that selected
plan's times, rates and exact available phase-integrity profile.

Only emitted diagnostic metadata changes; no scoring, fallback conditions,
phase thresholds, or audio rendering algorithms are modified.
The saved plan must be **generated again** with the opt-in enabled;
existing JSON cannot retroactively recover discarded candidate details.

## Read-only rejected beat-grid diagnosis (no rerun required)

After a four-track run with `DJAI_DECISION_TRACE=1`, inspect the exact
cached beat timestamps spanning each rejected rhythmic overlap. The
script requires the monolithic analysis caches with **full beat arrays**
under `.djai-cache/tracks`, matched by exact source path.

```bash
python scripts/diagnose_rejected_beat_grids.py \
  --set-plan output/milestone1-four-track/balanced/set-plan.json \
  --cache-root .djai-cache/tracks \
  --output output/milestone1-four-track/beat-grid-diagnostics.json
```

Upload `output/milestone1-four-track/beat-grid-diagnostics.json`
for review. Per deck, the report includes local expected/observed beat
intervals, out-of-window beat gaps, local accumulated interval error,
the nominal playback BPM, and cache-match status.

If status is `missing_or_ambiguous_full_beat_cache`, do not infer
anything about the detector; this tool deliberately refuses to use
the 16-beat previews in `set-plan.json` as full grids.

Previously saved decision traces might omit the precise overlap duration
and initial phase nudge. In that case duration is inferred from bars and
target BPM, and nudge is reported as unknown; a new traced planning run
with the latest version includes both. The diagnostic does not recreate
`_phase_integrity_profile` or prove audio-level beat correctness.

## Controlled detected-grid phase-drift decomposition (read only)

Reconstruct the rejected uncorrected phase trajectories from cached full
beat grids and mathematically decompose each trajectory into the idealized
constant-tempo reference plus separate deck-A and deck-B detected-grid
deviations. It does not adjust audio or bypass any validation gate.

```bash
python scripts/reconstruct_phase_drift.py \
  --set-plan output/milestone1-four-track/balanced/set-plan.json \
  --cache-root .djai-cache/tracks \
  --output output/milestone1-four-track/phase-drift-decomposition.json
```

The output reports `max_abs_replay_difference_ms` and
`replay_matches_logged_1ms` against the original rejected trajectory
where both histories have matching sample counts, plus per-sample
`detected_grid_phase_drift_ms`, `nominal_constant_tempo_drift_ms`,
`deck_a_grid_vs_nominal_change_ms`, and
`deck_b_grid_vs_nominal_change_ms`.

**Interpretation:** if nominal drift is near zero while detected-grid
drift is large, the discrepancy comes from how detected beat positions
or indices deviate from a constant-tempo grid. The report cannot prove
whether this is beat-detector error or *real musical tempo variation*.
The nominal grid is a theoretical counterfactual, not validated
ground-truth beat timing. When old rejected-candidate snapshots omit an
initial nudge, the script explicitly labels its zero-nudge assumption.
Do not conclude a phase implementation defect unless the replay
matches the original rejected phase error. This tool does not model
the bounded warp correction nor provide an audible-quality judgement.

There are no safety-threshold changes, policy changes, or automatic
reanalysis.

## Independent audio tempo validation before changing BPM models

The cached beat-grid decomposition reproduces phase differences, but cannot
establish audible tempo. This separate diagnostic estimates onset periodicity
from the **original audio** near each rejected overlap, without relying on
saved beat timestamps or the existing `librosa.beat.beat_track` output.

```bash
python scripts/validate_overlap_audio_tempo.py \
  --set-plan output/milestone1-four-track/balanced/set-plan.json \
  --cache-root .djai-cache/tracks \
  --output output/milestone1-four-track/independent-audio-tempo.json
```

If source paths from the saved plan are no longer available, supply
`--source-dir audio/uploads` (matching filenames required).

The resulting `independent-audio-tempo.json` contains per-overlap
audio-onset periodicity (wide and overlap-only windows), original cached
grid tempo, and saved global BPM. The audio estimator searches 105–140 BPM
and uses 128-sample hops to improve frequency resolution, but even its
result is only a periodicity estimate, not human-annotated ground truth.
Do not overwrite the production BPM based on one spectral-flux ACF peak.
Look for agreement between multiple independent windows, then validate
the true musical beat positions against audio before modifying the
production analysis contract. Nothing in this script modifies caches,
training authority, renderer, or phase safety.

## Experimental, fail-closed tempo reconciliation

Independent audio checks indicated that the saved 117.45 BPM for Mind
Against diverged from roughly 120 BPM onset periodicity, while Agents
of Time also showed a discrepancy. To test the hypothesis **without
changing default production mixing**, enable the analyzer's opt-in
`DJAI_TEMPO_CONSISTENCY=1` switch during a new four-track run:

```bash
DJAI_TEMPO_CONSISTENCY=1 DJAI_DECISION_TRACE=1 \
python scripts/milestone1_four_track.py \
  "audio/uploads/AL021_-_Agents_Of_Time_-_Dream_Vision.wav" \
  "audio/uploads/AL016_-_Tale_Of_Us_-_Nova.wav" \
  "audio/uploads/Ben Böhmer - Beyond Beliefs (Official Visualiser).wav" \
  "audio/uploads/AL035 - Mind Against - Walking Away.wav" \
  --journey balanced \
  --output-dir output/milestone1-tempo-consistency
```

This **experimental**, DJ-tempo-range-only policy changes the scalar
global BPM only if the whole beat-grid span and three distributed,
independently computed onset-periodicity windows agree within 0.6%,
their normalized ACF strengths are each >=0.5, and the difference from
the original BPM is between 0.8% and 7%. Otherwise it retains the
original BPM. It does **not** interpolate, rewarp or alter original
detected beat timestamps or downbeats. It does not guarantee correct
beat alignment or a musical transition. Changed BPM propagates to
tempo curves, derived bar durations and candidate features on the
fresh analysis.

The switch is OFF by default, and analysis caches use separate
fingerprints for ON/OFF modes. Opt-in results do not overwrite the
default feature-store projections. Analyze the new plan's
`tracks[].analysis.neural_evidence.tempo_consistency` record for
each track's decision and evidence. Do not promote this policy to
default without audio-level and phase-lock regression validation.
All Beat Integrity limits and emergency fallback paths remain unchanged.

## Read-only fine-resolution cue-onset audit

After the opt-in tempo-consistency run, compare the **original cached
beat positions** with the independent 128-hop (~5.8 ms) audio onset
strength peaks near the two rejected rhythmic overlap windows:

```bash
python scripts/audit_cue_onset_precision.py \
  --set-plan output/milestone1-tempo-consistency/balanced/set-plan.json \
  --cache-root .djai-cache/tracks \
  --output output/milestone1-tempo-consistency/cue-onset-precision.json
```

If the saved plan's source files were moved, add
`--source-dir output/milestone1-tempo-consistency/crate`.

For each rejected candidate, the JSON gives every supported local
peak observation within ±75 ms of an existing cached beat and
reports shifts in milliseconds, support coverage and distributions.
No candidate/beat timestamp/downbeat is modified. **Onset peaks can
come from hi-hats or other percussion and are not verified beat
ground truth**, so a measured shift is a diagnostic observation,
not evidence that replacing the beat would improve rendered mixing.
The earlier 25 ms phase-integrity gate is unchanged. Share
`cue-onset-precision.json` for a review of the two remaining failures.

## Read-only beat-sequence counterfactual after cue-onset audit

Given the saved beat-grid caches and the independent onset-alignment audit,
test a hypothetical **conservative** change to individual beat timestamps
without modifying any source analysis, cache, real set plan, or renderer:

```bash
python scripts/counterfactual_beat_sequence.py \
  --set-plan output/milestone1-tempo-consistency/balanced/set-plan.json \
  --cue-audit output/milestone1-tempo-consistency/cue-onset-precision.json \
  --cache-root .djai-cache/tracks \
  --output output/milestone1-tempo-consistency/beat-sequence-counterfactual.json
```

The tool uses existing onset observations within 75 ms of the saved beat
positions but only *proposes* shifts of up to 30 ms with onset support
ratio at least 2.0 and both neighboring intervals within ±10% of the
original transition's expected source-beat period. It never inserts or
removes beat indices. At least 90% cue-onset observation coverage is
required to evaluate a transition. It reports baseline and hypothetical
**uncorrected** phase drift, not a new safety determination. The original
25 ms phase-integrity envelope is unchanged.

Use this report to decide whether small onset-local refinements could
plausibly help, or whether larger interval/indexing anomalies require
independent beat-sequence validation. Onset peaks can belong to hats or
other instruments rather than beats; even a reduced hypothetical drift
is not proof of audio quality or justification to enable production
grid rewriting.

## Independent beat-sequence interpretation diagnostic (read-only)

After the experiment, analyze original cached beat interval anomalies against
independently detected audio onsets. Unlike the nearest-onset audit, onset
events are extracted from the entire audio overlap and are **not**
searched only around each cached beat:

```bash
python scripts/validate_beat_sequence_interpretation.py \
  --set-plan output/milestone1-tempo-consistency/balanced/set-plan.json \
  --cache-root .djai-cache/tracks \
  --output output/milestone1-tempo-consistency/beat-sequence-interpretation.json
```

Add `--source-dir output/milestone1-tempo-consistency/crate` only if
the plan's stored audio paths have moved.

The report identifies short beat intervals (possible extra/misplaced
events), intervals near double the expected beat period (possible
missing beats), and other interval mismatches. Missing-beat hypotheses
are marked `audio_supported_candidate` only when an independently
found onset falls within 55 ms of the gap midpoint. These are
**hypotheses**, not accepted beats: independent onset events can reflect
hats, subdivisions, or syncopation. No beat index, timestamp,
downbeat, cached analysis, planner output, or rendering is changed.
Original 25 ms Beat Integrity limits remain in force.

Share `beat-sequence-interpretation.json` for the next evidence
review; it is not a new phase-integrity pass/fail test.

## Read-only continuous beat-phase coherence (competing explanations)

After generating `beat-sequence-interpretation.json`, evaluate whole-overlap
beat-phase hypotheses against the independently detected audio-onset candidates:

```bash
python scripts/diagnose_continuous_beat_phase.py \
  --set-plan output/milestone1-tempo-consistency/balanced/set-plan.json \
  --sequence-audit output/milestone1-tempo-consistency/beat-sequence-interpretation.json \
  --cache-root .djai-cache/tracks \
  --output output/milestone1-tempo-consistency/continuous-beat-phase.json
```

The diagnostic scores each deck's original event grid, a steady pulse at
the expected source tempo, bounded event displacement, and (only when
interval evidence permits) single missing/extra beat hypotheses. Matching
is one-to-one against the independently detected onset candidates rather
than beat-guided peaks, and scores penalize interval errors, displacement
and index edits. Scores are **uncalibrated heuristic explanations**, not
probabilities, verified kick positions or professional DJ judgments.
A preference never authorizes a beat-grid change. No caches, plans,
renderer, safety limits or beat indices are modified. The output is
`continuous-beat-phase.json` for technical evidence review.

## Read-only cross-deck bounded-phase counterfactual

After `continuous-beat-phase.json` is available, test whether a
steady incoming-deck timing reference can reduce the **actual bounded
cross-deck phase residual** under the same saved cue times, nudge,
overlap duration, playback rates and sample count:

```bash
python scripts/counterfactual_crossdeck_phase.py \
  --set-plan output/milestone1-tempo-consistency/balanced/set-plan.json \
  --coherence output/milestone1-tempo-consistency/continuous-beat-phase.json \
  --cache-root .djai-cache/tracks \
  --output output/milestone1-tempo-consistency/crossdeck-phase-counterfactual.json
```

The script replays the production-default 45 ms maximum phase warp,
18 ms maximum per-sample step, and 25 ms residual threshold. It
checks that the winning *incoming* steady hypothesis has at least
0.10 score advantage over the original grid and at least 0.55
independent onset-support score. For a numerical grid alternative,
**every beat** within its overlap context must move no more than
30 ms, new join intervals must stay within 10% of expected, and
original downbeat anchors must remain within 25 ms. Failed checks
are reported as rejections; no alternative grid is applied.

Note: despite its earlier label, the steady-pulse candidate uses
the **planner's tempo** to fit an original beat-index sequence, not
an independently measured absolute audio beat grid. Even if a
candidate yields a numerical residual below 25 ms, that does not
establish safety, correct musical alignment or rendered audio
quality. It never authorizes production rewriting and leaves
all safety policies unchanged. Upload
`crossdeck-phase-counterfactual.json` for review.

## Read-only beat-grid outlier localization and event-index continuity

To pinpoint the individual timestamps behind the large rejected
steady-grid movement, inspect original cached beat sequences against
the existing *independent* audio-onset candidates:

```bash
python scripts/localize_beat_grid_outliers.py \
  --set-plan output/milestone1-tempo-consistency/balanced/set-plan.json \
  --sequence-audit output/milestone1-tempo-consistency/beat-sequence-interpretation.json \
  --cache-root .djai-cache/tracks \
  --output output/milestone1-tempo-consistency/beat-grid-outliers.json
```

The report flags beats whose fixed-period reference offset exceeds
45 ms or where adjacent intervals deviate by over 12% from the
expected source-beat period. For each suspicious beat it includes
the original and reference timestamps, neighboring intervals,
median phase offsets on each side, independent onset evidence
within 55 ms of both interpretations, and conservative *hypotheses*
for isolated displacement or persistent indexing offset.

The fixed-period reference still comes from planner tempo and
original beat indices. A persistent offset may indicate missing
or extra beats, incorrect tempo, or musical timing changes; this
test cannot distinguish them definitively. No beat timing or index
changes are authorized. The 25 ms phase integrity limit is
unchanged, and renderer quality is not evaluated.

## Read-only piecewise beat-phase continuity and half-beat ambiguity

Inspect stable beat runs on either side of anomalous local intervals,
without treating a constant phase offset as a beat defect:

```bash
python scripts/diagnose_piecewise_beat_continuity.py \
  --set-plan output/milestone1-tempo-consistency/balanced/set-plan.json \
  --sequence-audit output/milestone1-tempo-consistency/beat-sequence-interpretation.json \
  --cache-root .djai-cache/tracks \
  --output output/milestone1-tempo-consistency/piecewise-beat-continuity.json
```

The diagnostic finds runs of at least three beats whose adjacent
intervals are within 8% of the candidate's expected source beat period.
For each bridge between stable runs it reports the *relative* phase
shift from saved beat indices (no arbitrary absolute origin), each
irregular interval, and independent audio-onset coverage for the
cached rhythmic phase versus both half-beat-shifted alternatives.

The audio evidence contains many possible hats and subdivisions;
half-beat ambiguity remains explicit even when one interpretation
has greater onset coverage. Audio onsets were captured inside the
original overlap only, so contextual runs outside the overlap have
limited support. Scores are descriptive, not musical ground truth.
No beat index or timestamp is changed; the 25 ms Beat Integrity
threshold and the production fallback behavior are unchanged.
Share `piecewise-beat-continuity.json` for interpretation.

## Extended-context multi-band rhythmic verification (read-only)

After generating `piecewise-beat-continuity.json`, inspect the source
audio around each stable-run junction for at least four bars on both
sides, where available. This addresses the missing audio evidence
outside the previous narrow overlap.

```bash
python scripts/verify_extended_rhythmic_context.py \
  --set-plan output/milestone1-tempo-consistency/balanced/set-plan.json \
  --piecewise output/milestone1-tempo-consistency/piecewise-beat-continuity.json \
  --cache-root .djai-cache/tracks \
  --output output/milestone1-tempo-consistency/extended-rhythmic-evidence.json
```

The tool decodes the audio around the original cached junction indices
and extracts independent onset candidates from low-pass (below 180 Hz),
high-pass (above 1800 Hz) and full-band signals. It compares one-to-one
onset matches for the original beat positions versus half-beat earlier
and later alternatives, separately before and after each junction.
Four bars of context are sought on each side. A frequency band is a
rough kick- or percussion-weighted proxy, **not** source separation or
a verified quarter-note beat. Sparse evidence, comparable support for
two phases, and potential preference flips remain explicitly uncertain.

If the plan stores paths that have moved, supply `--source-dir` pointing
to the directory containing the original source audio filenames.
Missing audio is reported, not replaced by made-up evidence. Nothing
modifies a beat grid, downbeat, cache, transition, mixer or the
existing 25 ms Beat Integrity threshold.
Upload `extended-rhythmic-evidence.json` for evidence review.

## Independent beat tracking and metrical-accent comparison (read-only)

Use a second, independently initialized dynamic-programming beat tracker on
extended source-audio sections around each saved piecewise discontinuity.
The diagnostic also computes a separate audio-onset autocorrelation tempo
and explores four possible modulo-four bass-accent placements. Neither
method certifies musical downbeat identity. Optional mono excerpts enable
direct human listening to the *original*, uncorrected audio.

```bash
python scripts/audit_independent_beat_downbeat.py \
  --set-plan output/milestone1-tempo-consistency/balanced/set-plan.json \
  --piecewise output/milestone1-tempo-consistency/piecewise-beat-continuity.json \
  --cache-root .djai-cache/tracks \
  --excerpts-dir output/milestone1-tempo-consistency/listening-excerpts \
  --output output/milestone1-tempo-consistency/independent-beat-downbeat.json
```

If saved audio paths have moved, provide `--source-dir` pointing to the
original WAV files. Excerpts are optional and remain on the local Mac;
share them only if listening feedback is wanted. They are resampled mono
extracts, not corrected transition renders. The beat tracker may follow
offbeats or octave-tempo pulses. Modulo-four bass onset accents are
*metrical hypotheses*, not validated downbeats or a 4/4 proof. The tool
cannot safely choose beat-index edits and never modifies the cached
beat grids, mixer, plans, safety limits or fallback policy. Upload
`independent-beat-downbeat.json` for the next evidence review.

## Read-only local cached-vs-independent beat disagreement audit

After the independent beat/downbeat report, align individual cached and
second-method beat events monotonically around saved discontinuities:

```bash
python scripts/compare_local_beat_disagreements.py \
  --set-plan output/milestone1-tempo-consistency/balanced/set-plan.json \
  --independent output/milestone1-tempo-consistency/independent-beat-downbeat.json \
  --cache-root .djai-cache/tracks \
  --output output/milestone1-tempo-consistency/local-beat-disagreements.json
```

The diagnostic uses dynamic programming to align timestamp sequences
without changing event indices, retaining unmatched cached and independently
tracked events. It lists timestamp residuals and event-count differences for
six beats on either side of each saved junction. It particularly flags
differences **greater than 25 ms but still within the prior 70 ms agreement
tolerance**. The 25 ms value here is only a local comparison screen:
it is NOT the measured cross-deck phase-integrity residual or a proposed
correction policy. No beat tracker is presumed ground truth, and
insertion/deletion hypotheses are *not* authorized beat-index edits.

Production grids, audio, cache, cue times, safety limits, fallback
decisions and renders remain unchanged. Upload
`local-beat-disagreements.json` for timestamp-level interpretation.
