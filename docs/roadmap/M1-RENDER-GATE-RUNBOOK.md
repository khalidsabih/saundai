# M1 real-transition render gate (current selected grids)

Status: **offline operator workflow only**. This runbook and `scripts/m1-render-gate.sh` are on the PR #28 research branch; they do not change SAUND's rhythm selection, planner, renderer, or model authority.

## What it does

1. Rejects missing paths, existing output destinations, or missing explicit rights confirmation.
2. Runs focused unit tests for the four-bar timing preflight and the real-render canary.
3. Executes `scripts/m1-real-transition-render-canary.py` on **approved development-only** music pairs, with automatic model downloads disabled.
4. Writes private WAV previews and `canary-report.json` under a new output directory.

**Scope limits:** The canary measures the **currently selected rhythm grids**. It does **not** inject the frozen confidence-0.50 challenger, independently verify transition cue phase, certify subjective mix quality, or open the final 80-track test cohort. The earlier 80-track four-bar preflight is already reported in `M1-BALLROOM-INDEPENDENT-RHYTHM-PILOT.md`; it need not be repeated.

## Manifest contract

Provide a private JSON file (never commit music or approved private manifests):

```json
{
  "schema_version": "saund-m1-render-canary-v1",
  "split": "development",
  "rights_confirmed": true,
  "render_permission_confirmed": true,
  "pairs": [
    {"id": "dev-pair-1", "track_a": "track-a.wav", "track_b": "track-b.wav"}
  ]
}
```

The paths are relative to `SAUND_AUDIO_ROOT`; pairs must contain distinct existing supported audio files. `rights_confirmed` and `render_permission_confirmed` must reflect an actual operator review of local derived-audio rendering rights; independent rhythm evaluation permission alone is insufficient.

## Run on the operator's machine

From a local checkout of the PR #28 branch with SAUND's Python dependencies installed:

```bash
export SAUND_PYTHON="$PWD/.venv/bin/python"
export SAUND_AUDIO_ROOT="/absolute/path/to/authorized/development/audio"
export SAUND_RENDER_MANIFEST="/absolute/path/to/approved-development-pairs.json"
export SAUND_OUTPUT_DIR="/tmp/saund-m1-render-canary/baseline-01"

# Set this only after reviewing the local derived-audio rights.
export SAUND_RENDER_RIGHTS_CONFIRMED=yes

bash scripts/m1-render-gate.sh
```

Use a **new** `SAUND_OUTPUT_DIR` for each run. Missing rights confirmation or preexisting output stops execution. Focused test failures stop the script before rendering.

## Interpret the report

- `actual_renders`: how many pairs reached real rendering; not all are necessarily good.
- `technical_checks_passed`: initial peak/silence/boundary-jump checks passed, **not** proof of safe or musical handoff.
- `abstained_or_blocked`: the guarded planner declined or source grid checks failed.
- `errors`: runtime or per-pair failures requiring investigation.
- `challenger_grid_applied=false`: confirms this is a baseline canary only.
- `human_listening_review_completed=false`: reviews are still required.

For each rendered pair, inspect the private audio and verify musical bar/downbeat alignment, audible phase, bass interactions, continuity, and actual cue positions. Preserve failed/blocked cases and reasons, rather than treating abstention as success.

## Next gate

To run an apples-to-apples challenger comparison, update grid-dependent TrackAnalysis features and musical-timeline/cue evidence **before** compiling a TransitionPlan, then seal/validate the new plan and render it through the same deterministic path. Do not splice timestamps into an existing signed plan. Run cross-genre development canaries before deciding whether to use the untouched final holdout. Keep the frozen confidence threshold and production defaults unchanged until explicit acceptance and rollback requirements are met.

## Safe-planner rejection diagnostics (M1 synthetic pair)

When `saund plan --style safe --tempo-policy safe --no-quality-check` finds no feasible technique, the planner now reports the **four most common hard-failure reasons** and their cue-pair counts. These are existing assessments, not newly relaxed safety thresholds. Counts may overlap because one cue pair can violate multiple constraints. It intentionally does not automatically switch to an emergency cut.

Reproduce with the same synthetic fixture pair from the owner-managed demo-set workspace:

```bash
PYTHONPATH="$PWD/src" DJAI_MODEL_ALLOW_DOWNLOAD=0 \\
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 DJAI_NEURAL_BEATS=1 \\
"$M1_PYTHON" -m djai plan \\
  "$M1_AUDIO_ROOT/$M1_TRACK_A" "$M1_AUDIO_ROOT/$M1_TRACK_B" \\
  --tempo-policy safe --style safe --no-quality-check
```

Save the new `Rejection diagnostics across ...` output to identify whether tempo/meter/grid, phrase/harmony, available bars, or remaining source duration dominates. The focused code regression is `tests/test_decision_planner.py::test_infeasible_planner_reports_hard_constraints_without_forcing`. The change has been inspected in GitHub but the real Mac fixture and Python tests must be executed in the owner environment; no successful local runtime verification is claimed here.

## Owner's 144-pair diagnostic and precise blend-gate follow-up (2026-11-17)

The owner reran the two-track synthetic `demo-set` CLI with `--style safe --tempo-policy safe --no-quality-check`. All **144 examined cue pairs** showed the shared hard failure `The tempo/meter/beat-grid state cannot support a beatmatched blend`. The top-four frequency report also included missing stems, drum-and-bass-only double drop, and bassline overlap on 144 pairs. Those other restrictions apply to individual techniques and **do not establish the shared primary root cause**.

The planner now adds `Blend gate:` evidence with the *specific* failed shared prerequisites (tempo difference, incoming rate range, meter mismatch, local beat-grid confidence, or phase-warp/variable-tempo conflict). This is read-only failure explanation; no threshold, rhythm grid, rendering, or safety change. Counts are cue-pair incidences and may overlap. Re-run the same CLI command above and inspect the new `Blend gate:` section before considering changes. The owner's earlier 144-pair result predates this finer diagnostic; do **not** attribute it to a particular prerequisite without the new output.

Added focused tests in `tests/test_decision_planner.py` for low local beat-grid confidence and incompatible meters. Python tests and the owner audio rerun are pending; no verified production fix or successful render is claimed.

## Confirmed 4-versus-6 meter mismatch on generated demo pair

The owner reran the M1 planner tests successfully (**7 passed**) and retried `01_opening_122bpm_Aminor.wav` into `02_groove_124bpm_Eminor.wav` with `--style safe --tempo-policy safe`. The newly detailed planner gate reports:

```text
Blend gate: estimated meters incompatible or unknown (4 vs 6) (144 pair(s)).
```

The audio producer `src/djai/demo.py:generate_demo_track` explicitly synthesizes four kicks per bar and `bar_seconds = beat_seconds * 4` for **both** tracks. Therefore the incoming six-beat estimate is a **meter-perception mismatch on this generated fixture**, not proof that a 4/4-to-6/4 transition should be rendered.

The estimator `src/djai/analysis.py:_estimate_time_signature` compares periodic accent scores over 3, 4 and 6 beats; `analyze_track` can run that estimation again after accepting a neural beat grid. Neither the precise score margin nor whether the neural grid was selected was captured in the owner's prior CLI output. **Do not change thresholds or override a claimed six-beat meter without that provenance.**

A new read-only audit (`scripts/m1-demo-meter-audit.py`) records the two fixture meters and confidences, median beats between downbeats, and Beat This! availability/acceptance evidence. It exits 2 for a known-generator meter mismatch, writes an immutable JSON report, refuses output overwrite and disables downloads. It does **not** modify production state. Use the actual generated demo files (not sealed Ballroom holdout):

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
"$M1_PYTHON" -m pytest -q tests/test_m1_demo_meter_audit.py tests/test_decision_planner.py
DJAI_NEURAL_BEATS=1 DJAI_MODEL_ALLOW_DOWNLOAD=0 DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
  "$M1_PYTHON" scripts/m1-demo-meter-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-demo-meter-audit-01.json
```

An audit exit code of **2** is expected if the currently observed six-beat estimate persists; inspect the report rather than treating the nonzero status as an infrastructure failure. If the audit shows neural grid accepted, distinguish the effect of re-estimating meter on the accepted neural beat frames from the deterministic estimator. If not accepted, investigate the deterministic onset-phase score and failure on synthetic four-beat accents. A future production fix requires tests on real **3/4 and six-beat** references to avoid correcting synthetic 4/4 at the expense of real non-4/4 music.

## Neural downbeat-density issue: owner demo audit

The owner ran the repository read-only fixture audit after **10 focused tests passed**. The report confirms:

- Opening 122 BPM demo: deterministic grid retained; estimated **4** beats/bar; **128 beats / 32 downbeats**; median downbeat spacing **4.0476 beats**.
- Groove 124 BPM demo: Beat This! grid accepted (`neural_used=true`, confidence **0.9225**); estimated meter **6**; **129 beats / 124 downbeats**; median downbeat spacing **1.0 beat**.
- Both audio files are generated as 4/4 by `src/djai/demo.py`.

This exposes two distinct failures: the incoming neural downbeat sequence is implausibly dense, while `analysis.py` re-estimates meter separately from onset accents after neural acceptance and produces 6. High beat-grid confidence is not proof of correct downbeat phase or meter. Avoid treating the current neural acceptance as safe.

### Opt-in integrity experiment, not a production policy change

`src/djai/intelligence/neural_ears.py` now provides `downbeat_spacing_integrity`, a structural filter for accepted neural grids. When `DJAI_NEURAL_DOWNBEAT_INTEGRITY=1`, the existing accepted-neural rule must also pass basic downbeat ordering and median spacing **at least 2.5 beat periods**. The current demo's one-downbeat-per-beat sequence fails. A rejection preserves the current deterministic *whole grid and BPM*, rather than mixing scalar tempo with a different grid. The flag is **off by default** to preserve the frozen research/production policies; these rules are not yet approved for global promotion.

The spacing criterion is only a preliminary sanity check: it may reject legitimate irregular musical patterns, and it cannot detect a wrong but regularly spaced downbeat phase or wrong meter. Valid 3, 4 and 6 beat groupings pass synthetic unit fixtures, but real-data cross-domain false rejection testing remains essential.

Run code-focused tests and perform **separate** generated-demo audits with unique output paths:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
"$M1_PYTHON" -m pytest -q tests/test_m1_neural_downbeat_integrity.py tests/test_m1_demo_meter_audit.py tests/test_decision_planner.py

DJAI_NEURAL_BEATS=1 DJAI_MODEL_ALLOW_DOWNLOAD=0 DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
  "$M1_PYTHON" scripts/m1-demo-meter-audit.py \
  --audio-root "$M1_AUDIO_ROOT" --output /tmp/saund-m1-meter-baseline-new.json

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_MODEL_ALLOW_DOWNLOAD=0 DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
  "$M1_PYTHON" scripts/m1-demo-meter-audit.py \
  --audio-root "$M1_AUDIO_ROOT" --output /tmp/saund-m1-meter-integrity-trial.json
```

A nonzero exit from the **baseline** mismatch audit is expected (2). Inspect both JSON files, especially `neural_used`, `neural_reason`, `predicted_meter`, `downbeat_count`, and `inferred_beats_per_downbeat_interval`. Do not interpret a successful audit as an audio-quality certification. Do not use the untouched Ballroom final test set for this trial. Only after the structural trial passes should the operator retry the no-force safe planner, and then independently test actual rendering and music quality.

## Owner-verified opt-in integrity outcome — 2026-11-17

The owner ran the focused tests and read-only synthetic fixture audit with `DJAI_NEURAL_DOWNBEAT_INTEGRITY=1`. The provided terminal output reported **14 passing tests** (dot progress) and `all_meters_match_generator=true`. These are owner-run results; no assistant-side audio inference or rendering was performed.

| Demo item | Original selection | Opt-in trial |
|---|---|---|
| Opening 122 BPM | meter 4, 128 beats, 32 downbeats, neural retained as evidence | meter **4**, 128 beats, 32 downbeats, neural `used=false` |
| Groove 124 BPM | meter **6**, 129 beats, 124 downbeats, neural `used=true` (confidence 0.9225) | meter **4**, 126 beats, 31 downbeats, neural `used=false` (reported confidence still 0.9225) |
| Groove median downbeat interval in beat periods | **1.0** | **3.9524** |

The rejected neural result reported `opt-in neural downbeat integrity rejected grid: neural downbeats are too dense relative to beats`. The original deterministic fallback remains internally consistent on the synthetic 4/4 fixture. This establishes a successful **analysis-stage** guard trial on the generated pair, not that the safe planner or renderer has passed. Keep the feature flag **off by default** and the final Ballroom holdout sealed.

### Next operator gate: no-force planning, then separately authorized rendering

After fetching this branch, re-run the same synthetic pair with the explicit opt-in guard enabled:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
PYTHONPATH="$PWD/src" DJAI_NEURAL_BEATS=1 \
DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_MODEL_ALLOW_DOWNLOAD=0 DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
"$M1_PYTHON" -m djai plan \
  "$M1_AUDIO_ROOT/$M1_TRACK_A" "$M1_AUDIO_ROOT/$M1_TRACK_B" \
  --tempo-policy safe --style safe --no-quality-check
```

A successful plan must still satisfy the planner's **existing** hard constraints and automatic safety checks. A failure may reveal a different issue (cue durations, local compatibility, boundary confidence); capture its diagnostic reason and do not force or relax constraints. Only if a valid no-force plan exists should an approved synthetic or other rights-cleared development render proceed. The structural spacing check has not yet been tested for false rejection on diverse 3/4, 6/8, or other real-music cases.

## Owner safe-planner rerun: meter gate cleared; continuous phase safety abstained

After enabling `DJAI_NEURAL_DOWNBEAT_INTEGRITY=1`, the owner reran the generated 4/4 demo pair with `djai plan --style safe --tempo-policy safe --no-quality-check`. The prior `estimated meters incompatible or unknown (4 vs 6)` rejection did **not** recur. Instead the CLI stopped with:

```text
Continuous beat-phase lock is outside the bounded correction envelope; automatic delivery should replan or use a structural reset.
```

This is meaningful progression: the analysis-stage meter inconsistency was fixed for the opt-in synthetic demo trial, but the planner still **cannot safely deliver** the proposed transition. No audio render, human listening verdict, threshold relaxation, forced transition, or production promotion occurred. The new message originates in `src/djai/performance/deck.py:apply_deck_performance`, which sets `requires_confirmation` when a beatmatched plan's continuous phase profile is unsafe. The measured error and exact reason **have not yet been observed** in the pasted CLI output. This is not necessarily proof of audible drift.

### Capture the existing phase-lock trace without changing decisions

SAUND already supports opt-in read-only rejected-plan logging through `DJAI_PHASE_LOCK_TRACE_PATH` and `scripts/summarize_phase_lock_trace.py`. Run the same no-force safe plan with the full current opt-in configuration, saving output to a **new private trace filename**:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
export M1_PHASE_TRACE="/tmp/saund-m1-demo-phase-lock-01.jsonl"
test ! -e "$M1_PHASE_TRACE" || { echo "Choose a new private trace filename" >&2; exit 2; }

PYTHONPATH="$PWD/src" \
DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_MODEL_ALLOW_DOWNLOAD=0 DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
DJAI_PHASE_LOCK_TRACE_PATH="$M1_PHASE_TRACE" \
"$M1_PYTHON" -m djai plan \
  "$M1_AUDIO_ROOT/$M1_TRACK_A" "$M1_AUDIO_ROOT/$M1_TRACK_B" \
  --tempo-policy safe --style safe --no-quality-check

"$M1_PYTHON" scripts/summarize_phase_lock_trace.py "$M1_PHASE_TRACE"
```

The CLI is **expected to exit nonzero** on the recorded unsafe candidate; the summary command can still be run afterwards in an interactive shell. The JSONL trace includes cue positions, overlap duration, beat-grid confidence and interval CV, correction envelope and step limit, sample-by-sample `phase_error_ms` and `residual_ms`, and unmatched-point counts. The summarizer reports unique configurations and directional drift vs abrupt jumps. **Do not interpret a 999 ms insufficient-evidence sentinel as measured residual.**

Before any change to timing behavior, inspect the actual trace: assess whether the issue is steady pulse drift, beat-index discontinuity/missing beats, exhausted incoming source, or insufficient reliable intervals. Neither disabling phase warp nor increasing its correction limits is an approved remedy. Use real audio only with separately approved render rights and keep the final holdout sealed.

## Owner phase-lock trace: 111 rejected overlap configurations

The owner reran the guarded synthetic demo pair with `DJAI_PHASE_LOCK_TRACE_PATH` and used the existing `scripts/summarize_phase_lock_trace.py`. The pasted summary reported:

| Measure | Owner-observed |
|---|---:|
| Raw rejected plan evaluations | 111 |
| Unique cue/rate/overlap configurations | 111 |
| Insufficient-evidence cases | 0 |
| Median measured residual | 210.420 ms |
| P90 measured residual | 465.839 ms |
| Mostly directional trajectories | 83 |
| Irregular trajectories | 28 |
| Reason: correction exceeds bounded phase-lock envelope | 111 |

**Interpretation:** These are measured residuals in SAUND's inferred continuous phase profiles, not verified acoustic drift relative to real kick events. In particular, the baseline `_unwrapped_beat_numbers` uses each detected event's sequential index. A dropped beat may cause a permanent index-origin shift, even when the local beat period is plausible. The `DJAI_BEAT_GAP_AWARE` implementation already exists but is disabled by default. Do not enable or retune it solely from aggregate trace statistics, and do not raise the 25 ms residual safety limit.

### Synthetic beat-index audit to distinguish drift from missing beats

Added `scripts/m1-demo-beat-index-audit.py` and `tests/test_m1_demo_beat_index_audit.py`. The read-only diagnostic compares each estimated beat timestamp against the **known exactly timed kicks** of `src/djai/demo.py`, which generates unswung 4/4 at 122 and 124 BPM. It reports median/max event-time error, missing/doubled nearest generator beat indices and the largest index discontinuity. This ground truth is valid **only on the generated demo**, not real Ballroom music. It does not modify beat predictions, planning, or phase-lock thresholds.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
"$M1_PYTHON" -m pytest -q tests/test_m1_demo_beat_index_audit.py
DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_MODEL_ALLOW_DOWNLOAD=0 DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
"$M1_PYTHON" scripts/m1-demo-beat-index-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-demo-beat-index-01.json
```

Next inspect the resulting track-specific gap counts and event-time errors, then correlate with the original rejected trace's candidate cue spans. If detected beats are dropped or duplicated, study the phase-index problem with synthetic known events before proposing a bounded corrective policy. If event indices are intact, investigate relative phase calculations and tempo-rate assumptions. In either case, **do not force a mix** or relax the phase-integrity stop, and do not score the sealed Ballroom test cohort.

## Owner generator-grounded beat-index result and drift decomposition

The owner ran `tests/test_m1_demo_beat_index_audit.py` (**3 passed**) and evaluated the real generated demo tracks with the opt-in neural downbeat guard enabled. Read-only results:

| Quantity | Opening 122 BPM | Groove 124 BPM |
|---|---:|---:|
| Accepted neural grid | no | no |
| Beat observations | 128 | 126 |
| Median absolute error to nearest generator kick | **210.477 ms** | **27.286 ms** |
| Maximum nearest-kick error | 226.362 ms | 39.216 ms |
| Missing nearest-index jumps | 0 | 0 |
| Duplicate nearest-index jumps | 1 | 0 |
| First observed beat | 0.510839 s | 0.510839 s |

These observations suggest the outgoing deterministic grid is much less accurate against known generated kicks, but **do not prove** that the residual phase-lock rejection is caused only by a dropped beat or only by actual tempo drift. Nearest-grid errors are bounded by half a beat and can mask accumulated drift by wrapping the reference beat index. Also, an observed beat beginning around 0.51 seconds does not automatically mean the first beat was missed: beat estimators can omit onset-adjacent events.

To separate **fixed phase offset** from **accumulating slope**, the existing `scripts/m1-demo-beat-index-audit.py` now reports `sequential_index_drift`: first- and last-third signed errors, ms-per-beat fitted slope, approximate implied BPM, and endpoint displacement anchored to the first nearest generator kick. This is valid only on the constant-tempo generated examples, and the assumption of sequential beats must be cross-checked against the missing/double event report. The reported values do not feed the planner or allow overrides.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
"$M1_PYTHON" -m pytest -q tests/test_m1_demo_beat_index_audit.py
DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_MODEL_ALLOW_DOWNLOAD=0 DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
"$M1_PYTHON" scripts/m1-demo-beat-index-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-demo-beat-index-drift-02.json
```

Before changing an inference model or switching phase-index policy, compare `slope_ms_per_beat` with `first_third_median_signed_ms` / `last_third_median_signed_ms` and the original phase-lock JSONL profiles. Maintain the current fail-closed continuous phase gate (25 ms default) and keep all final-test music sealed.

## Owner drift decomposition: likely local slip, not uniform drift

The owner reran `scripts/m1-demo-beat-index-audit.py` after the sequential-index drift extension. The outgoing generated 122 BPM track reported `end_minus_start_signed_ms=-484.957`, `slope_ms_per_beat=-2.0565`, `first_third_median_signed_ms=-203.762`, and `last_third_median_signed_ms=-217.605`. Its nearest-generator error was ~210 ms median with one duplicate nearest-event index jump. The incoming generated 124 BPM track reported `end_minus_start_signed_ms=4.111`, `slope_ms_per_beat=-0.003`, and `first_third_median_signed_ms=27.424` versus `last_third_median_signed_ms=27.286`, with no nearest-index jumps.

The outgoing **~one-pulse endpoint displacement** and nearly unchanged first/last-third median errors are not a clean signature of sustained linear tempo drift. A short/long local beat interval, reference-index wrap, or beat-index slip could account for much of the global regression slope. The aggregate numbers cannot establish which event occurred. They also cannot certify the full overlap against source audio, and must not be used as a meter/beat correction oracle in production.

The audit now adds `local_interval_diagnostics`: indices and timestamps of observed beat intervals deviating at least **15%** from generator period, nearest-generator index wraps, and signed-error medians for four sequential quarters. This still only reads synthetic known-tempo fixtures and never alters plans or beat grids. To run:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
"$M1_PYTHON" -m pytest -q tests/test_m1_demo_beat_index_audit.py
DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_MODEL_ALLOW_DOWNLOAD=0 DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
"$M1_PYTHON" scripts/m1-demo-beat-index-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-demo-beat-local-03.json
```

Inspect the outgoing track's `local_interval_diagnostics.first_outlier_intervals`, `nearest_index_wraps`, and `signed_sequential_error_quartile_medians_ms`, and cross-check affected positions against the 111 rejected phase-lock cue spans. Do not change `DJAI_DECK_PHASE_RESIDUAL_MAX_MS` or blindly enable `DJAI_BEAT_GAP_AWARE`. The next intervention, if evidence supports one, should retain hard abstention when confidence or phase evidence is insufficient.

## Owner's local beat-interval results — two outgoing compression clusters

The owner ran the read-only generated-demo beat-index audit and **7 focused tests passed**. The outgoing 122 BPM track has five detected intervals at least 15% shorter than the generator's exact 491.803 ms beat period:

- **8.823583–9.636281 s:** two successive 417.959 and 394.739 ms intervals (indices 17–19).
- **58.305306–59.535964 s:** three successive 417.959, 394.740 and 417.959 ms intervals (indices 118–121); the nearest-generator event index repeats at 118→119.
- Signed sequential phase-error quartile medians: **−4.714, −214.545, −212.362, −219.491 ms**. The first cluster aligns with the first sustained shift; the last cluster may affect a short tail segment.

The incoming 124 BPM track has **zero >15% interval outliers** and stable signed-error quartile medians near +27 ms. These figures narrow the suspected timing problem to localized **beat-tracker compression** rather than persistent musical tempo drift. The inferred +/− timing deviations are *relative to synthetic generator truth only*; don't treat them as proof that source audio changes tempo or use them to edit real-music grids.

The diagnostic now additionally emits `local_interval_diagnostics.outlier_clusters`, grouping consecutive anomalous beat intervals and reporting their cumulative departure from the known generator period. This provides a bounded, reproducible target for subsequent phase-drift analysis; it does **not** apply any correction, set `DJAI_BEAT_GAP_AWARE`, or weaken the 25 ms residual envelope.

The owner's second attempt to write `/tmp/saund-m1-demo-beat-local-03.json` failed solely because the report already existed. This is intentional no-overwrite behavior. To examine the new cluster fields, use a **fresh** output filename:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
"$M1_PYTHON" -m pytest -q tests/test_m1_demo_beat_index_audit.py
DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_MODEL_ALLOW_DOWNLOAD=0 DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
"$M1_PYTHON" scripts/m1-demo-beat-index-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-demo-beat-clusters-04.json
```

Do **not** remove the older report just to rerun the audit. Phase-lock planning and real rendering remain blocked/unverified until evidence supports a safe remedy.

## Owner-confirmed beat-compression clusters and phase-overlap correlation

The owner reran the generated demo audit with the opt-in neural downbeat gate and confirmed the two outgoing track clusters:

| Cluster | Window | Interval count | Sum of local interval departures from the generator period |
|---|---|---:|---:|
| Early | 8.823583–9.636281 s | 2 | −170.908 ms |
| Late | 58.305306–59.535964 s | 3 | −244.751 ms |

The incoming track had **no detected interval outlier clusters** at the same 15% synthetic reference criterion. The owner's rerun completed successfully and did not modify production behavior or render a transition.

The next task is to establish whether either cluster intersects the 111 independently traced phase-lock *rejections*. A new **read-only** script, `scripts/m1-correlate-phase-anomaly-overlaps.py`, compares each rejected trace's outgoing and incoming source windows (including actual render rates and incoming phase nudge) with the synthetic beat-index audit's cluster windows. It summarizes counts and median *rejected-only* residuals for overlaps that intersect an anomaly versus those that do not, and preserves a small set of example cue positions. **Even a strong association is not proof that timing compression caused a rejected overlap; the trace contains only rejected candidates, and no acceptance-rate denominator.**

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD

python -m pytest -q tests/test_m1_phase_anomaly_overlap.py

python scripts/m1-correlate-phase-anomaly-overlaps.py \
  --trace /tmp/saund-m1-demo-phase-lock-01.jsonl \
  --beat-audit /tmp/saund-m1-demo-beat-clusters-05.json \
  --output /tmp/saund-m1-phase-anomaly-correlation-01.json
```

Use the locally saved private phase trace and synthetic audit from prior steps; choose a fresh output path if a report already exists. This utility does not call a neural model, evaluate audio or trigger rendering. Compare `groups.crosses_anomaly.rejected_configurations` against `groups.does_not_cross_anomaly.rejected_configurations`. If a substantial number of unsafe overlaps **do not** cross either timing cluster, localized compression alone cannot explain every rejection. Do not change the 25 ms phase residual envelope or silently promote the model on the basis of this analysis. Future experiments should separately validate real 3/4 and six-pulse material and authorized development renders.

## Owner overlap-to-anomaly correlation — localized clusters are insufficient alone

The owner correlated the existing **111 rejected-only** phase-lock trace configurations with the two confirmed outgoing demo beat-compression windows:

| Rejected overlap class | Configurations | Median measured residual |
|---|---:|---:|
| Crosses a detected local beat anomaly | 28 | 396.179 ms |
| Does not cross a detected anomaly | **83** | **210.420 ms** |

All 111 trace rows were eligible; none were skipped. **83/111 (74.8%) rejected configurations do not actually traverse either locally anomalous interval**, so fixing only those two short segments is not a demonstrated complete remedy. The early compression can, however, shift later sequential phase indices even when the corresponding overlap does not include the compressed interval. This is a rejected-only analysis with no acceptance-rate denominator and cannot establish causality.

The trace's continuous phase algorithm subtracts each overlap's **initial phase error**, so an *unchanging* global offset by itself cannot explain within-overlap drift. Other hypotheses still requiring differentiation include index-origin discontinuity, beat-clock mismatch within the overlap, incoming/outgoing render-rate assumptions, and accumulated timing inconsistency.

The read-only `scripts/m1-correlate-phase-anomaly-overlaps.py` now additionally summarizes, separately for crossing and non-crossing rejected configurations: median absolute start-to-end sampled phase change, median largest sample-to-sample change and directional/jump/irregular trajectory categories. These fields use existing rejected JSONL `points` and cannot change planning or safety.

To obtain the newly expanded report on the operator machine, run the focused tests and reuse existing evidence with a fresh output filename:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_phase_anomaly_overlap.py

python scripts/m1-correlate-phase-anomaly-overlaps.py \
  --trace /tmp/saund-m1-demo-phase-lock-01.jsonl \
  --beat-audit /tmp/saund-m1-demo-beat-clusters-05.json \
  --output /tmp/saund-m1-phase-anomaly-trajectory-02.json
```

Compare `groups.crosses_anomaly.phase_trajectory_patterns` with `groups.does_not_cross_anomaly.phase_trajectory_patterns`, plus each group's sampled phase-change metrics, rather than declaring any phase correction authorized. Continue rejecting plans over the existing **25 ms phase residual envelope**. No real transition has been rendered or accepted.

## Owner trajectory results: all non-crossing rejections show directional motion

The owner reran `tests/test_m1_phase_anomaly_overlap.py` (**6 passed**) and the expanded rejected-trace correlation. The **28 anomaly-crossing** rejected configurations were all classified as irregular phase motion, with median **396.179 ms** residual and median largest sampled step **169.698 ms**. The **83 non-crossing** rejected configurations were all classified as mostly directional drift, with median **210.420 ms** residual, median absolute end-to-end phase change **255.420 ms**, and median largest sampled step **46.440 ms**. All 111 rows were measured; none were skipped.

This strongly suggests there are both localized tracker errors and a broader beat-clock/tempo-rate discrepancy. **It is not evidence that the 25 ms phase envelope is too strict.**

Source inspection identifies a specific research hypothesis: `decision._tempo_context` derives incoming render rate via `match_bpm(effective_bpm_a, track_b.bpm)`, while `performance/deck.py:_phase_integrity_profile` measures phase from sampled `track_a.beats_seconds` and `track_b.beats_seconds`. Both generated tracks' scalar estimated BPM was **123.0469**, yet earlier synthetic reference analysis found substantially different beat-clock rates (incoming ~124.0008 BPM; outgoing globally fitted ~122.5123 BPM, itself potentially distorted by local compression). Thus a nominal rate of 1.0 inferred from scalar BPM equality could leave measurable within-overlap drift.

Added `scripts/m1-demo-phase-clock-counterfactual.py` and `tests/test_m1_demo_phase_clock_counterfactual.py`. The read-only script combines the saved generator-grounded beat-index audit with the rejected phase trace and reports planned render rates, fitted output beat-clock BPM for each deck, **predicted uncontrolled drift**, observed start-to-end phase movement, direction agreement, and remaining unexplained movement. This is an explanatory comparison only; it does not swap BPM, build/re-sign a plan, render, bypass a grid or open sealed test audio.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_demo_phase_clock_counterfactual.py

python scripts/m1-demo-phase-clock-counterfactual.py \
  --trace /tmp/saund-m1-demo-phase-lock-01.jsonl \
  --beat-audit /tmp/saund-m1-demo-beat-clusters-05.json \
  --output /tmp/saund-m1-demo-phase-clock-01.json
```

Because the outgoing global fitted BPM can be biased by local compressed beat intervals, this is a **hypothesis test, not a production BPM correction**. Compare predicted and measured change and how much remains unexplained, then consider segment-local rate estimation and independent audio-onset evidence before any controlled proposal. The phase-lock stop, current 25 ms limit, frozen neural policy, and rights constraints remain unchanged.

## Owner global-clock counterfactual: 101 of 111 directional matches

The owner ran the synthetic demo's read-only `m1-demo-phase-clock-counterfactual.py` and observed 111 evaluable, zero skipped trace rows. Both tracks still showed the identical **123.0469 BPM** scalar in SAUND's analyses, but their **whole-track fitted beat clocks** were approximately **122.5123 BPM** (outgoing, biased by local compression) and **124.0008 BPM** (incoming). Existing rejected plans used `render_rate_a=1.0` and `render_rate_b=1.0` in the supplied examples. The global-clock counterfactual reported:

- **101/111** observed phase-change directions matching its prediction.
- Median predicted uncontrolled change **−188.759 ms**, observed **−255.420 ms**.
- Median absolute unexplained change **89.881 ms**.

This supports the diagnosis that scalar BPM equality does **not** establish beat-clock agreement. It is not a proof of a production fix: the outgoing clock fit is contaminated by compressed intervals and the counterfactual does not reproduce the continuous phase warp or all indexing discontinuities.

### Cue-local clock experiment — no planning authority

A new **read-only** `scripts/m1-demo-local-clock-audit.py` analyzes the two known synthetic demo files with the current opt-in neural downbeat integrity configuration, then measures the local beat-period median inside **each actual rejected trace window**. It separately counts discarded >15% local interval outliers; they remain unsafe and are **never silently corrected** in a plan. For each rejection, the report compares the *planned* output rate to cue-local output clocks, predicts uncorrected phase change, and records the remaining unexplained end-to-end change. It cannot replan, render, rescore or change the frozen rhythm policy or phase residual limits.

Run it on the Mac with the existing private JSONL rejection trace; choose a new output filename:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_demo_local_clock_audit.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_MODEL_ALLOW_DOWNLOAD=0 DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-demo-local-clock-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --trace /tmp/saund-m1-demo-phase-lock-01.jsonl \
  --output /tmp/saund-m1-demo-local-clock-01.json
```

Compare `median_absolute_unexplained_ms`, `median_predicted_local_clock_change_ms`, `matching_direction_count`, and `non_evaluable_count` with the global-clock counterfactual. The new results should be compared by cue interval and inspected for any local timing anomalies. Do not treat filtered BPM evidence as authorization to interpolate across a bad beat or force a long transition. Validation must later include independent real audio onset references and both non-4/4 meters before any production proposal.

## Owner local-clock result: median beat period fails to explain within-overlap drift

The owner completed the new cue-local clock audit on the 111 rejected synthetic demo configurations. All **111 were evaluated**, none were skipped. Representative cue windows had clean local interval medians of **123.04689 BPM on both decks**, with 31 outgoing and 32 incoming intervals and zero discarded outliers. The planned rates were both **1.0**. The local median-clock model predicted **0 ms** change, while the measured median rejected phase change was **−255.420 ms**, leaving median absolute unexplained change **255.420 ms** and matching predicted/observed directions in **0 cases**.

This **rejects** a simplistic conclusion that the rejected phase motion is explained by a *difference in cue-local median BPM*. The earlier whole-track regression was contaminated by localized beat compressions and cannot authorize a rate override. Equivalent local *median* periods do **not**, however, prove equivalent accumulated unwrapped phase; endpoint timestamps, fractional beat progress and observed event indices can differ even when median interval lengths match.

### Read-only exact continuous-phase replay

Added `scripts/m1-demo-phase-index-replay.py` with `tests/test_m1_demo_phase_index_replay.py`. It loads the two known generated demo files with the same opt-in neural integrity configuration, reads the existing rejected phase JSONL, and replays the actual `_continuous_beat_position` calculation for the recorded candidate cue positions and sample fractions. It separately reports whole-beat **integer index** and **fractional beat** contributions to each rejection's end-to-end phase movement, and verifies the decomposed position against the imported deck calculation. It does **not** replace `TrackAnalysis`, sign any plan, alter beat timestamps, change `DJAI_BEAT_GAP_AWARE` or touch the default **25 ms** phase integrity envelope.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_demo_phase_index_replay.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_BEAT_GAP_AWARE=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-demo-phase-index-replay.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --trace /tmp/saund-m1-demo-phase-lock-01.jsonl \
  --output /tmp/saund-m1-phase-index-replay-01.json
```

Compare `median_abs_integer_index_change_ms`, `median_abs_fraction_change_ms`, and `median_abs_replay_trace_difference_ms`. A small replay discrepancy would support consistency with the recorded phase implementation; a large discrepancy requires investigating environmental differences or reconstruction assumptions before interpreting the decomposition. An integer index change is *not automatically an actual missing beat*, because the fractional component may cancel it. All findings remain diagnostic and synthetic-only; no transition has been approved for rendering.

## Owner exact phase-position replay: no discrepancy with production phase calculations

The owner executed the read-only `m1-demo-phase-index-replay.py` on the same 111 rejected synthetic demo configurations. All **111 evaluated**, none skipped, and the median absolute replay-versus-original-trace difference was **0.000 ms**. Median absolute integer-index contribution was **487.619 ms** and median absolute fractional-beat contribution **228.328 ms**. A representative no-anomaly-overlap example contained an integer-index change of **−487.619 ms**, offset by **+232.199 ms** fractional movement, for a net **−255.420 ms** phase change. These contributions must be read together; an integer-index component is **not** proof of an undetected or additional beat.

The replay exactly reproduces the deck's current sequential-index continuous-phase computation on the selected beat grids. It does not independently validate those beat grids against actual acoustic kicks and does not authorize any grid correction. No rendered audio, plan approval, or production threshold change has occurred.

### Per-deck beat-clock excess and complete-interval budgets

The same replay diagnostic now reports, for each rejected overlap, separate **outgoing** and **incoming** accumulated beat-position changes in excess of their ideal source-clock advancement. It also reports each deck's complete beat-interval count, summed signed deviation from the plan's expected beat period, maximum absolute deviation, and unusual intervals. This distinguishes which deck contributes to end-to-end phase movement, while avoiding the false assumption that equal median local BPM means equal cumulative phase. Complete intervals alone exclude the partially intersected edge intervals and should be interpreted together with exact replay positions.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_demo_phase_index_replay.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_BEAT_GAP_AWARE=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-demo-phase-index-replay.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --trace /tmp/saund-m1-demo-phase-lock-01.jsonl \
  --output /tmp/saund-m1-phase-index-replay-03.json
```

Inspect `median_abs_outgoing_clock_excess_ms`, `median_abs_incoming_clock_excess_ms`, `median_abs_clock_excess_difference_ms`, and the first example's two `*_complete_interval_budget` objects. The phase-lock gate remains at its unchanged **25 ms** residual limit; the frozen M1 rhythm selection policy is unchanged. The new script output and tests have not been verified on the owner's Mac yet.

## Owner per-deck cumulative interval replay — source of median-BPM discrepancy

The owner submitted a further replay after the cumulative-budget extension. All **111** rejected synthetic configurations were evaluated, none skipped, with **0.000 ms** median absolute difference from the original phase trace. For the representative 15.603806-second overlap starting around **36.176689 s** on the outgoing track and **9.705941 s** on the incoming:

| Deck | Complete intervals | Median interval | Cumulative interval deviation from target | Beat-clock excess |
|---|---:|---:|---:|---:|
| Outgoing | 31 | 487.619 ms | **+139.324 ms** | **−139.324 ms** |
| Incoming | 32 | 487.619 ms | **−116.096 ms** | **+116.096 ms** |

Their excess difference is **−255.420 ms**, exactly matching the measured rejected phase movement. The median absolute excess across configurations was **132.991 ms** outgoing, **139.316 ms** incoming, and **255.420 ms** difference. **No individual complete interval in this representative overlap exceeded the script's 15% anomaly threshold**; maximum absolute single-interval departure was **23.220 ms** on both decks.

This establishes that *equal local median periods do not imply equal total beat advancement*. Arithmetic mean period and accumulated interval deviation capture many small consistent offsets that the median discards. The new `interval_budget` report additionally includes `mean_interval_ms`, `mean_minus_median_ms`, `mean_implied_bpm`, and `median_implied_bpm` to make the discrepancy explicit.

This does **not** authorize replacing the production BPM estimator with a local mean, moving beat indices, modifying a signed plan, or weakening phase safety. Complete-interval budgets exclude partial edge intervals; production still needs independent onset evidence, broader music evaluation, and guarded integration tests.

Rerun the two additional pure-unit tests and report with a **fresh filename**:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_demo_phase_index_replay.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_BEAT_GAP_AWARE=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-demo-phase-index-replay.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --trace /tmp/saund-m1-demo-phase-lock-01.jsonl \
  --output /tmp/saund-m1-phase-mean-median-04.json
```

Compare the mean and median per-deck intervals and the signed cumulative biases. Fail-closed planner behavior and the 25 ms phase residual limit remain unchanged.

## Owner mean-versus-median replay: validated scalar-tempo inconsistency on synthetic demo

The owner ran the extended phase replay on the 111 rejected generated-demo configurations. No rows were skipped, and the median absolute replay-versus-trace discrepancy remained **0.000 ms**. In the representative outgoing 36.176689 s / incoming 9.705941 s overlap:

| Quantity | Outgoing | Incoming |
|---|---:|---:|
| Mean observed beat interval | **492.113 ms** | **483.991 ms** |
| Mean implied BPM | **121.92315** | **123.96926** |
| Median observed beat interval | **487.619 ms** | **487.619 ms** |
| Median implied BPM | **123.04689** | **123.04689** |
| Sum of complete-interval deviations from plan beat period | +139.324 ms | −116.096 ms |

The mean values are close to the known synthetic generator's **122 and 124 BPM**, whereas the common median aligns with the prior scalar estimate **123.0469 BPM**. This provides concrete evidence for a synthetic **scalar BPM versus aggregate beat-grid-clock mismatch**. It does not prove that replacing BPM with a mean is safe on real variable-tempo or irregular music.

### Important existing capability: opt-in onset-backed tempo consistency

Source inspection found `src/djai/analysis.py` initially derives deterministic BPM from the `librosa.beat.beat_track` scalar, with the beat timestamps separately determined by its detected frames. The neural model's BPM replaces that scalar only if its grid is selected. Crucially, SAUND **already has** `src/djai/music/tempo_consistency.py:reconcile_bpm`, enabled only when `DJAI_TEMPO_CONSISTENCY=1`. It retains the original scalar unless the complete grid span and **three independent audio-onset autocorrelation windows** agree, have sufficient confidence, and satisfy conservative dance-tempo checks. It never edits beat timestamps. No new production estimator is justified before this existing experimental path is tested.

The read-only `scripts/m1-demo-meter-audit.py` now includes `grid_span_bpm`, `scalar_to_grid_span_difference_bpm`, and `tempo_consistency_evidence` when available, making the existing reconciler's `decision`, `reason`, and independent onset-window observations visible.

Use separate output filenames for the **unchanged** synthetic guard and **guard plus tempo reconciliation**:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD

python -m pytest -q tests/test_m1_demo_meter_audit.py tests/test_tempo_consistency.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_TEMPO_CONSISTENCY=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-demo-meter-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-meter-tempo-baseline-01.json

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_TEMPO_CONSISTENCY=1 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-demo-meter-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-meter-tempo-reconcile-01.json
```

The key comparison is `tempo_consistency_evidence.decision` and `reason`, alongside `bpm` and `grid_span_bpm` for both tracks. If either track **retains** its original BPM due to insufficient audio agreement, that is a valid fail-closed outcome, not a bug. If both scalars reconcile, repeat the **unforced** safe plan separately with both opt-in flags, then inspect the phase trace. No automatic rendering or safety-rule change is authorized. Keep the Ballroom sealed test set untouched.

## Owner tempo-consistency rerun — materiality gate prevents audio review

The owner ran the guarded generated-demo meter audit with `DJAI_TEMPO_CONSISTENCY=1`. Both tracks correctly remained **4/4**, but both BPM values stayed at **123.0469**. The outgoing full-grid span was **122.95467 BPM** (scalar minus grid **+0.09223 BPM**); the incoming span was **123.99157 BPM** (scalar minus grid **−0.94467 BPM**). Both reconciler decisions were `retain` with `reason=no_material_global_disagreement`. The incoming relative mismatch is approximately **0.768%**, just below the reconciler's strict **0.8%** independent-onset evaluation threshold. No independent `audio_windows` evidence was computed in that original run. The neural checkpoint was not bootstrapped locally; both grids were deterministic (`neural_available=false`), so this run cannot establish neural acceptance behavior.

### New opt-in read-only onset-window shadow inspection

The existing `reconcile_bpm` retains exactly the same materiality threshold, decisions, numerical BPM, and beat-grid timestamps. When **both** `DJAI_TEMPO_CONSISTENCY=1` and `DJAI_TEMPO_SHADOW_AUDIO_WINDOWS=1`, it additionally evaluates the same three distributed audio-onset ACF windows on the **early-retain** branch and records `tempo_consistency.shadow_audio_windows` plus `shadow_only=true`. This is diagnosis only, not authority to select the shadow estimates. It uses the existing audio evidence routine and does not require model downloads.

Run on the two **synthetic development** files with a fresh report filename:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD

python -m pytest -q tests/test_tempo_consistency.py tests/test_m1_demo_meter_audit.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_TEMPO_CONSISTENCY=1 DJAI_TEMPO_SHADOW_AUDIO_WINDOWS=1 \
DJAI_MODEL_ALLOW_DOWNLOAD=0 DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-demo-meter-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-meter-tempo-shadow-02.json
```

Inspect each track's `tempo_consistency_evidence.shadow_audio_windows`, its three BPMs and correlation strengths, and confirm `bpm` remains **123.0469** with `decision=retain`. Weak, missing, or divergent independent windows are grounds to retain/abstain, not to lower a threshold. Even three strong windows would motivate a carefully staged policy design and multi-genre checks, **not** an immediate silent tempo change or an unsafe rendered handoff. Do not touch the sealed Ballroom final test cohort.

## Owner onset-shadow run: three strong windows per synthetic track, no BPM promotion

The owner ran the opt-in `DJAI_TEMPO_SHADOW_AUDIO_WINDOWS=1` audit with the default 0.8% tempo materiality threshold unchanged. Both synthetic tracks retained **123.0469 BPM**, both remain detected 4/4, and no transition was rendered. The neural checkpoint was **not** bootstrapped in this run; both grids were deterministic.

| Synthetic track | Full-grid span BPM | Independent onset-window BPM estimates | Strengths |
|---|---:|---|---|
| Opening (generator 122) | 122.95467 | 122.44625, 122.46173, 122.44389 | 0.9431, 0.9397, 0.9364 |
| Groove (generator 124) | 123.99157 | 123.61149, 123.61791, 123.61261 | 0.9626, 0.9659, 0.9646 |

The independent audio-onset windows are each tightly clustered and strong, and the two tracks' estimates differ. This increases confidence that the shared **123.0469 BPM scalar** is unrepresentative for the synthetic transition; it does **not** establish a universally superior replacement BPM. The audio ACF values differ from the exact known generator references. Since the global disagreements fell below the existing strict materiality threshold, both recorded reconciler decisions remain `retain` with `reason=no_material_global_disagreement`.

Added a separate **offline, read-only** `scripts/m1-shadow-tempo-evidence-assessment.py` with `tests/test_m1_shadow_tempo_evidence_assessment.py`. It reads the already saved meter-audit JSON (no new inference or plan), applies the existing three-window quality/agreement conditions to the shadow evidence, and reports whether the onset and full-grid estimates agree. A hypothetical rate is labeled *not authorized*; it is not applied to production, and cannot authorize a transition that fails the continuous phase-lock gate.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_shadow_tempo_evidence_assessment.py

python scripts/m1-shadow-tempo-evidence-assessment.py \
  --audit /tmp/saund-m1-meter-tempo-shadow-02.json \
  --output /tmp/saund-m1-shadow-tempo-assessment-01.json
```

Inspect `tracks[].max_window_disagreement_pct`, `tracks[].grid_onset_disagreement_pct`, `tracks[].status`, and `comparison` if present. Never silently lower the 0.8% reconciliation gate, apply a hypothetical rate to a signed transition, or weaken the **25 ms** residual safety envelope. Any later guarded policy proposal needs broader music and meter validation plus independent source-onset verification.

## Owner shadow assessment: hypothetical 0.991102 incoming rate is not an approved plan

The owner executed `m1-shadow-tempo-evidence-assessment.py` using the existing synthetic audit. Its two tracks satisfied the existing *independent-window quality and beat-grid agreement tests in shadow only*, but the reconciler **retained** both 123.0469 BPM scalars because the strict 0.8% materiality threshold was not met. The shadow estimates were:

| Fixture | Onset median BPM | Full grid BPM | Onset-vs-grid disagreement | Shadow-only blended BPM |
|---|---:|---:|---:|---:|
| Opening | 122.44625 | 122.95467 | 0.4135% | 122.70046 |
| Groove | 123.61261 | 123.99157 | 0.3056% | 123.80209 |

The research-only suggested B source rate is **0.991102** versus the currently planned **1.0**, with hypothetical tempo difference **1.10163 BPM**. This is **not** an authorized substitution for the planner's BPM or tempo-match rate, and does not establish phase integrity.

### Bounded phase-rate numerical counterfactual on rejected overlaps

Added `scripts/m1-shadow-rate-phase-replay.py` with isolated tests in `tests/test_m1_shadow_rate_phase_replay.py`. It consumes the **saved rejected-phase trace**, synthetic demo audio and the shadow assessment; reuses the existing read-only bounded `phase_profile` routine, holding original cues, overlap duration, outgoing rate and target BPM constant while comparing the existing B rate against 0.991102 on the **unchanged original beat grids**. It reports paired numeric residuals, skipped/incomplete evidence and how many hypothetical profiles happen to fall within the existing 25 ms *numeric* envelope. A numerical pass is **not** a safe/accepted transition: the hypothetical variation has not passed phrase, source-availability, downbeat, timing, rendering or quality authorization.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD

python -m pytest -q tests/test_m1_shadow_rate_phase_replay.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_BEAT_GAP_AWARE=0 DJAI_TEMPO_CONSISTENCY=0 \
DJAI_MODEL_ALLOW_DOWNLOAD=0 DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-shadow-rate-phase-replay.py \
  --trace /tmp/saund-m1-demo-phase-lock-01.jsonl \
  --assessment /tmp/saund-m1-shadow-tempo-assessment-01.json \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-shadow-rate-phase-replay-01.json
```

Inspect `paired_numeric_profiles`, `median_original_numeric_residual_ms`, `median_hypothetical_numeric_residual_ms`, `hypothetical_numeric_below_25ms_count`, and any `missing_phase_coordinate` statuses. The command does not authorize modifying production analysis, planner or rendering. Do not use any hypothetical rate to generate an unsafe transition.
