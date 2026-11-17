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

## Owner shadow-rate replay — reduced error, zero numeric passes

The owner twice provided the same completed read-only rate counterfactual from the generated demo pair. Of **111 rejected trace configurations, 111 paired numerical profiles** were replayed with no skipped cases. The existing incoming rate was **1.000000**, the shadow hypothetical **0.991102**. The median original bounded phase residual was **210.420 ms**, versus median hypothetical **94.797 ms**; the median per-configuration reduction was **138.843 ms**. **Zero of 111** hypothetical profiles satisfied the unchanged **25 ms** numerical phase residual threshold. The generated synthetic transition was not rendered, no new plan created, and no production code or frozen M1 policy promoted.

This establishes that a research-only tempo-rate adjustment reduces phase disagreement but **does not make the rejected transitions numerically safe**. A median improvement must never be conflated with plan feasibility. This specific 0.991102 hypothesis cannot be promoted to an actual deck rate or used to force a blend; further root-cause analysis requires independent beat/onset evidence and source-grid quality checks, not threshold relaxation.

The replay report now explicitly records `hypothetical_numeric_still_exceeds_25ms_count` and `all_paired_profiles_still_fail_25ms_numeric_limit`, with tests preventing accidental presentation of a lower residual as a passing transition. These are read-only summaries and do not affect the authoritative phase gate. The previous report's evidence remains sufficient to conclude **all 111 hypothetical overlaps failed the numeric limit** without another expensive inference run.

To validate the focused reporting logic on the Mac:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_shadow_rate_phase_replay.py
```

Do not rerender or bypass the phase-integrity stop. Only a separate, explicitly safe development plan that passes all existing gates could lead to an authorized synthetic test render.

## Next M1 experiment — independent waveform kick attacks versus detected beats

Following the confirmed **0/111** safe transitions in the shadow-rate-only phase experiment, the next blocker is identifying whether the beat timestamps correspond to real acoustic kicks in the two rights-free synthetic tracks. The new **read-only** `scripts/m1-synthetic-audio-kick-audit.py` measures low-frequency audio energy attacks directly from the WAV (30–155 Hz bandpass, preceding/following short-window energy difference). It uses the generator's **known 122/124 BPM schedule solely to define bounded candidate search windows** and does **not** use the beat tracker output to select attack locations. Missing/weak audio onset evidence is explicitly reported as missing; it is never filled from the generator oracle. It separately aligns the unchanged estimated beat timestamps to measured acoustic events and reports unmatched beats, signed timing deviations, sample examples, detected-event coverage and the audio attack–generator offset.

This is *independently measured waveform timing but not a fully reference-independent onset detector*: the generator schedule narrows the windows. Do not generalize this diagnostic or its oracle to arbitrary music. Unreliable or missing kick attacks cannot authorize changing the production beat grid.

The new `tests/test_m1_synthetic_audio_kick_audit.py` exercises synthetic pulse audio, missing kick abstention, mismatch rejection, invalid grids, and short audio. Run on the operator's Mac with the existing generated demo files:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD

python -m pytest -q tests/test_m1_synthetic_audio_kick_audit.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_TEMPO_CONSISTENCY=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-synthetic-audio-kick-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-synthetic-audio-kick-audit-01.json
```

Check `tracks[].measured_audio_pulses`, `missing_audio_attack_evidence`, `median_audio_attack_minus_generator_ms`, and `grid_alignment_to_independent_audio`. If acoustic events agree with known generated pulses while the beat grid does not, the beat estimator is suspect. If the acoustic attack detector lacks support, diagnose that instead; do not report a false beat-grid failure. No plan, rendering, neural authority, tempo policy, or 25 ms phase-integrity envelope changes in this experiment.

## Owner acoustic-kick audit confirms severe outgoing grid misalignment

The owner ran the read-only synthetic waveform-kick audit. Both tracks had **127/128 measurable acoustic attacks** under the fixed low-frequency envelope detector; the first t=0 pulse failed measurement because the preceding-window analysis is boundary-sensitive. The method uses known synthetic pulse windows, and so does not establish a general-purpose real-music onset detector.

| Generated track | Detected beats | Measured kick attacks | Grid beats farther than 80 ms from measured kick | Median absolute nearest-kick error |
|---|---:|---:|---:|---:|
| Opening 122 BPM | 128 | 127 | **103 / 128** | **219.637 ms** |
| Groove 124 BPM | 126 | 127 | **0 / 126** | **21.780 ms** |

The opening beat tracker first has a clear acoustic miss near **9.241542 s** (112.834 ms early relative to the nearest measured kick), then roughly **209.887 ms early** by 9.636281 s and **234.989 ms early** by 12.561995 s. This is consistent with the earlier early-compression cluster. The incoming 124 BPM grid remains close to acoustic attacks over the evaluated range. Both were deterministic beat grids; `neural_grid_used=false`. None of these measurements gives the synthetic kick oracle authority to change production beat timestamps or sign a transition.

Extended `scripts/m1-synthetic-audio-kick-audit.py` with `outgoing_region_alignment` to segment **nearest measured kick alignment** into the previously identified source windows: before 8.823583 s; 8.823583–9.636281 s; 9.636281–58.305306 s; 58.305306–59.535964 s; and after 59.535964 s. The report separates mismatch counts and median absolute/signed errors. Region selection is from prior detected-grid diagnostics; acoustic attack timestamps are measured separately from the WAV. Nearest-kick alignment does **not** prove a correct musical beat index, and no segment is automatically relabeled or corrected.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_synthetic_audio_kick_audit.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_TEMPO_CONSISTENCY=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-synthetic-audio-kick-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-synthetic-audio-kick-regions-02.json
```

The next engineering decision should be based on the regional mismatch counts, including whether timing error persists between the compressed clusters. Keep the existing 25 ms phase-lock limit, hard rejection of unsafe overlaps, and frozen neural selection policy intact. No transition render has passed the gate.

## Owner regional acoustic replay — long displaced section and restored ending

The owner executed `m1-synthetic-audio-kick-audit.py` with the 5-window outgoing diagnostic. The independently measured WAV kick windows exposed a distinct temporary alignment failure:

| Outgoing source region | Beats | >80ms from measured kick | Median absolute nearest kick error |
|---|---:|---:|---:|
| Before 8.823583 s | 17 | 0 | 22.358 ms |
| 8.823583–9.636281 s | 2 | 1 | 75.895 ms |
| **9.636281–58.305306 s** | **99** | **99** | **222.449 ms** |
| 58.305306–59.535964 s | 3 | 3 | 193.174 ms |
| After 59.535964 s | 7 | 0 | 22.313 ms |

This is a near-half-beat acoustic displacement across almost the entire middle segment, with restored proximity at the end. It **does not yet establish** whether the tracker followed a wrong subdivision, whether its sequential beat indices slipped, or whether the two boundary clusters changed timestamp phase; nearest-kick distance can alias when two neighboring kicks are similarly close. The incoming grid had **0 / 126** beats more than 80 ms from independently measured kick windows. The first t=0 kick's measurement remains unsupported by the preceding/following energy difference method, and is not filled with the generator reference.

Extended the same read-only audit with `outgoing_region_index_associations`. For each original region it reports the nearest **measured** WAV kick's synthetic generator index, the difference to the detected beat event index, and that offset's histogram **only when acoustic timing is within 80 ms**. It also reports nearest-kick duplicate/reverse index steps and first/last examples. Out-of-tolerance associations are *not* counted as reliable index identity and no beat timestamp, neural authority, BPM, signed transition or render safety limit is changed.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_synthetic_audio_kick_audit.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_TEMPO_CONSISTENCY=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-synthetic-audio-kick-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-synthetic-audio-kick-indices-03.json
```

Inspect `tracks[0].outgoing_region_index_associations`, especially `reliable_index_offset_histogram`, `out_of_tolerance_count`, and nearest-kick duplicate steps across the first and second compression regions. The 99 middle mismatches should remain **out of tolerance**, not be misreported as confident index alignment. This is only synthetic diagnostic evidence; the next permitted action depends on independent source-event identity, not forcing a transition through the **25 ms** phase-integrity stop.

## Owner index-association result — 10 focused tests passed; middle remains unassociated

The owner ran `tests/test_m1_synthetic_audio_kick_audit.py` successfully (**10 passed**) and the unchanged-grid acoustic index association experiment. For the outgoing 122 BPM generated track, the **17/17** initial reliably aligned detected beats matched measured acoustic kick indices at a consistent **+1** offset (the grid starts near kick #1, not kick #0). At the first compression region, one beat still associated reliably at **+1**, the next was **112.834 ms early** and exceeded the 80 ms tolerance. All **99/99** middle-section beats were out of acoustic tolerance and therefore **no middle kick-index association is reliable**. The late compression region had **0/3** reliable associations; nearest-kick matching repeated an index, but this is not proof of a physically duplicated beat. After the late cluster, **7/7** beats reliably matched measured kick indices at offset **0**. The incoming track still had zero beats beyond the 80 ms acoustic tolerance.

The index evidence is consistent with a loss and subsequent recovery of rhythmic alignment, but it does not independently establish whether the middle beat tracker followed offbeat hi-hats, shifted to a different metrical phase or lost a pulse count. Crucially, midpoint timestamps are **not measured audio events**: they are mathematical midpoints between pairs of separately measured kick attacks.

### Synthetic half-beat midpoint hypothesis test

Added `scripts/m1-synthetic-offbeat-midpoint-audit.py` and `tests/test_m1_synthetic_offbeat_midpoint_audit.py`. It compares each original detected beat with (a) independently measured WAV kick attacks and (b) **inferred** temporal midpoints between adjacent measured kick events. It reports counts and median distances per known outgoing phase region, plus the whole incoming track. A midpoint is absent if either acoustic neighbor was unmeasured; it never fills missing onset evidence or pretends to have detected an offbeat instrument. A closer inferred midpoint supports a *half-beat displacement hypothesis*, but by itself **does not prove that SAUND is following hi-hats**. A separate actual hi-hat transient detector would be needed for that claim.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_synthetic_offbeat_midpoint_audit.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_TEMPO_CONSISTENCY=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-synthetic-offbeat-midpoint-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-synthetic-offbeat-midpoint-01.json
```

Compare `tracks[0].regions` for `between_compressions`: `closer_to_inferred_midpoint_count`, `median_distance_to_inferred_midpoint_ms`, and `within_inferred_midpoint_tolerance_count` versus corresponding measured kick counts. Synthetic-only timing evidence cannot authorize moving beat positions or override the **25 ms** phase-safety stop, frozen neural policy, rights requirements, or rendering gates.

## Owner midpoint comparison: 99 of 99 middle beats near inferred half-beats

The owner completed the generator-window kick-to-midpoint comparison on the two synthetic demo tracks. Of the **99** outgoing middle-section beats (9.636281–58.305306 s), **99/99** were closer to an **inferred midpoint** between independently measured adjacent kicks than to a kick, and all **99/99** were within 80 ms of that inferred midpoint. Median inferred-midpoint distance was **23.651 ms**, compared with **222.449 ms** median distance to a measured kick. Before the first compression **17/17** aligned nearer kicks; after the second compression **7/7** aligned nearer kicks. The entire incoming track had **126/126** beats nearer measured kicks, with a median kick distance **21.780 ms**. The early compression region had **2/2** beats nearer kicks (one beyond kick tolerance), and the late compression region **2/3** nearer inferred midpoints.

This is strong geometric evidence for temporary **half-beat grid displacement**, not evidence that a particular instrument caused the phase switch. Midpoints are computed from two measured kick times and **are not independently measured hi-hat events**. Nothing in this result approves an oracle beat correction, changes neural selection or justifies widening the 25 ms continuous phase-lock envelope.

### Next research-only evidence: waveform-measured high-frequency offbeat transients

Added `scripts/m1-synthetic-audio-offbeat-attack-audit.py` with `tests/test_m1_synthetic_audio_offbeat_attack_audit.py`. The script analyzes the same rights-free generated WAVs with a **high-frequency bandpass (2.5–9 kHz where sample rate permits)** and before/after energy-rise transient estimator. It selects search windows using the synthetic generator's known offbeat timing, but actual attack positions and evidence presence are measured from the WAV, not supplied by the beat tracker, midpoint interpolation or generator template. The output explicitly separates the number of measurable offbeat high-frequency events from alignment between unchanged detected beat times and those measured events, by the same outgoing source-time regions. This distinguishes **midpoint proximity** from independently verified high-frequency event proximity. Even if the latter matches, causality (e.g. tracker following hats) is not established by timing coincidence alone.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_synthetic_audio_offbeat_attack_audit.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_TEMPO_CONSISTENCY=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-synthetic-audio-offbeat-attack-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-synthetic-audio-offbeat-01.json
```

Inspect `tracks[0].measured_offbeat_events` and `tracks[0].regional_measured_offbeat_alignment`, in particular `between_compressions.within_80ms_count` and `median_absolute_attack_error_ms`. Confirm the detector measures enough actual offbeat WAV transients before drawing a conclusion; weak or missing events must be treated as insufficient evidence, not filled with hypothetical midpoints. No plan or render is produced, no production code or frozen neural policy is changed.

## Owner high-frequency offbeat WAV experiment — confirms actual synthetic offbeat transients

The owner executed `scripts/m1-synthetic-audio-offbeat-attack-audit.py` on the two generated copyright-free tracks. This measurements is **not** generated midpoint interpolation: a high-frequency (2.5–9 kHz) WAV energy-rise estimator identified **127/127** synthetic offbeat attacks on both fixtures, with median attack-time offsets from the expected generator half-beat pulse of **+0.385 ms** (opening) and **+0.091 ms** (incoming).

| Source region | Detected beats within 80 ms of measured high-frequency offbeat attack | Median absolute offbeat attack error |
|---|---:|---:|
| Outgoing before first compression | 0/17 | 220.362 ms |
| Outgoing first compression | 0/2 | 179.716 ms |
| **Outgoing 9.636281–58.305306 s** | **99/99** | **31.882 ms** |
| Outgoing second compression | 2/3 | 47.619 ms |
| Outgoing after second compression | 0/7 | 218.526 ms |
| Incoming entire track | 0/126 | 214.740 ms |

The earlier independent low-frequency kick audit measured **99/99 middle-section outgoing beats beyond 80 ms from kick attacks**. In this high-frequency audit, **all 99 are within 80 ms of actually detected offbeat high-frequency attacks**, versus 0/17 and 0/7 near the beginning and end. Together, these are convergent, source-waveform-based observations of a temporary **kick-to-offbeat rhythmic phase switch** in the deterministic beat grid, followed by re-lock on kicks near the end. The real presence of offbeat high-frequency attacks is now supported; it remains unproven that the internals of librosa's beat tracker *causally chose* those events rather than shifting through its own dynamic programming/tempo phase mechanism.

### M1 next engineering gate — non-oracle integrity detector

Stop trying arbitrary rate adjustments on this pair: a shadow playback-rate counterfactual still failed the 25 ms numeric phase gate on **111/111** rejected configurations. The next candidate should use **audio features independent of the generator's known 122/124 BPM pulse schedule** to detect a sustained kick-to-offbeat inconsistency in the already estimated beat grid. It should run only as a separate research/shadow evaluator initially, and report either (a) supported sustained phase inconsistency, (b) supported stable rhythm, or (c) insufficient/ambiguous evidence. In particular:

- Use onset/energy measurements at grid beats and half-beat-shifted alternatives without assuming the generator's BPM or accepting a known kick template.
- Require extended contiguous support and comparable independent low-/high-band attacks, rather than interpreting every offbeat hi-hat as a wrong beat or using absolute bass energy alone. Test correct kick-aligned beats with audible offbeat hats, true offbeat-locked beats, weak-kick/ambient material, tempo variation, halftime/doubletime, and real 3/4 and 6-pulse material.
- Preserve the existing `TrackAnalysis.beats_seconds`, downbeats, scalar BPM and all production gates. An uncertain detector should abstain, **never silently shift or synthesize beats**. Any eventual opt-in veto needs independent validation and tests showing it cannot green-light or render a transition.
- Keep the M1 Balloom development-validation vs sealed final-test boundaries, frozen neural confidence selection, rights constraints and default 25 ms continuous phase-lock rejection unchanged.

This wave-based synthetic source evidence closes the diagnostic question of whether actual offbeat transients exist. It **does not** close the M1 safe-render gate; no plan has passed, no transition was rendered, and the challenger has not been promoted.

## M1 shadow-only generator-independent two-band phase integrity prototype

Following owner-confirmed measured high-band WAV attacks at **99/99** displaced outgoing middle beats (9.636281–58.305306 s, median absolute timing error **31.882 ms**) and **0/126** incoming kick-aligned beats matching offbeat attacks, the M1 branch now contains a non-oracle shadow prototype:

- `src/djai/music/rhythm_phase_integrity.py`: filters independently into 35–155 Hz low and 2.5–9 kHz high bands, then samples short energy windows on **detected beat timestamps and the midpoint of each adjacent detected-beat pair**. No known generator BPM, synthetic pulse positions, track names, or annotated beats enter the detector. A 16-beat window requires ≥75% paired low/high contrast support, and a sustained run requires at least three overlapping windows. Weak, mixed, unsupported, or silently invalid evidence abstains. Output labels `kick_aligned_with_support`, `sustained_offbeat_phase_inconsistency`, or `insufficient_evidence` are strictly *shadow research labels*, not production track authorization.
- `scripts/m1-shadow-rhythm-phase-integrity.py`: reads the two existing rights-free generated WAV fixtures and the **current unchanged** `analyze_track` beat grids. The two filenames only identify files, not BPM or meter references. It writes the standalone shadow decision, time windows, support counts and runs while explicitly declaring no grid adjustment, new plan, or render.
- `tests/test_rhythm_phase_integrity.py`: focused unit and generated-audio checks for clean kick-vs-hat alignment, half-beat-shifted grid, weak/ambiguous evidence, isolated phase blips, grid bounds, and three-/six-pulse annotation independence. These are limited fixtures; they do not establish generalization to real variable-meter music.

**Critical caveat:** this uses *band-energy concentration*, not a learned kick/hi-hat instrument classifier. Many valid songs can lack a strong kick, have syncopated low-frequency sounds, or emphasize offbeats. The prototype must not be wired to production rejection, neural-grid promotion, beat fixing, tempo reconciliation or rendering approval unless independent multi-genre, variable-tempo, non-4/4 and weak-accent validation demonstrates safe behavior.

Owner Mac command:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_rhythm_phase_integrity.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_TEMPO_CONSISTENCY=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-shadow-rhythm-phase-integrity.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-shadow-rhythm-phase-01.json
```

Inspect each `tracks[].phase_integrity.decision`, `windows[]`, and `sustained_offbeat_runs`. A useful first diagnostic is whether the outgoing track gets sustained offbeat evidence in its middle and the incoming track stays kick-aligned or abstains. **No specific classification is guaranteed until the owner's local run.** The frozen neural confidence policy, 25 ms phase gate, 0.8% BPM reconciliation materiality and all rights restrictions remain unchanged.

## Owner generator-independent two-band detector: correct abstention, high-band rise follow-up

The owner ran the first shadow `m1-shadow-rhythm-phase-integrity.py` report on both synthetic WAVs. The detector **abstained** (`decision=insufficient_evidence`) for both the outgoing 122 BPM and incoming 124 BPM tracks; no sustained offbeat or kick runs were declared. This is an important validation result, not an accepted beat grid or a safe transition.

The outgoing middle windows exhibited strong **low-band** discrepancy: median low beat-versus-midpoint energy ratios often **0.02–0.03**, consistent with low-frequency kick energy being much stronger between the displaced detected beats. But the high-band energy contrast did not support the required paired event evidence. In particular, high-band midpoint/beat energy ratios were often around **0.6–0.8** in the displaced middle and **1.1–1.2** on the incoming track. No 16-beat window achieved the required 12/16 *paired* low/high contrast votes. Crucially, the standalone generator-window-guided audio attack experiment had independently measured 99/99 high-frequency offbeat events near the displaced grid; the generator-independent shadow energy features did **not** provide equally strong phase discrimination, so the research conclusion must remain abstention.

Added **read-only high-frequency transient-rise diagnostics** to `src/djai/music/rhythm_phase_integrity.py`: short-window (6 ms) *before-versus-after* high-band energy increases near each detected beat and midpoint. Each 16-beat shadow window now optionally reports `median_high_rise_mid_to_beat_ratio`, `high_rise_mid_dominant_count`, `high_rise_beat_dominant_count` and `high_rise_diagnostic_only=true`. These new values are **not incorporated into classification or decisions**; they allow targeted inspection of whether attack rise, rather than ongoing energy, distinguishes the phases.

The tests additionally check that adding contradictory high-rise evidence does **not** change an existing classification, and that sustained energy without a new attack does not create an onset. Preserve all original evidence thresholds and the existing 25 ms phase safety envelope.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_rhythm_phase_integrity.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_TEMPO_CONSISTENCY=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-shadow-rhythm-phase-integrity.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-shadow-rhythm-phase-rise-02.json
```

Inspect the high-rise ratios in the outgoing windows spanning **9.64–58.31 s** and compare with the incoming. If contrast is still weak, do not force a decision by reducing thresholds; instead consider independent transient localization and controlled negative controls before a policy proposal. No production grid change, signed transition, or rendering is authorized.

## Owner high-rise waveform follow-up: narrow +/-22 ms windows remain non-discriminative

The owner ran `m1-shadow-rhythm-phase-integrity.py` after adding diagnostic-only 6 ms high-frequency transient-rise sampling. Both tracks again returned `insufficient_evidence`, with zero sustained beat-phase runs; no production change or transition was authorized. Across **all** reported windows, the median `high_rise_mid_to_beat_ratio` was **1.0**, even though individual beat/midpoint dominance counts varied. The outgoing middle low-band beat/midpoint ratios stayed approximately **0.01–0.03**, strongly favoring the opposite beat phase, while the incoming low band favored ordinary kick alignment. This does **not** warrant lowering paired-evidence thresholds.

The new diagnostic readout adds **wider +/-55 ms high-band transient-rise searches** at the same detected beat and between-beat midpoint coordinates, compared with the prior +/-22 ms narrow rise windows. The already independently measured synthetic offbeat attack proximity was a median **31.882 ms** from middle-section outgoing beats, so a fixed narrow search could exclude real transient evidence. This is a hypothesis, not a measured explanation yet. The augmented window report contains `wide_high_rise_beat_evidence_count`, `wide_high_rise_mid_evidence_count`, `wide_high_rise_both_absent_count`, `wide_high_rise_mid_dominant_evidence_count`, `wide_high_rise_beat_dominant_evidence_count`, and `wide_high_rise_median_ratio_on_evidence`. A pair with neither measurable attack must appear as **absent** rather than as positive 1.0-ratio evidence. These fields are **diagnostic only**, excluded from classification and policy; the original classifier, its thresholds, generator independence, and 25 ms transition phase-integrity gate are unchanged.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_rhythm_phase_integrity.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_TEMPO_CONSISTENCY=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-shadow-rhythm-phase-integrity.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-shadow-rhythm-phase-wide-03.json
```

Inspect the wider-window evidence counts across the displaced outgoing middle versus the incoming track. If many windows still lack support or have indistinguishable rise ratios, do not promote this contrast detector or relax safety: the next step is independently localized transients and stronger negative controls. The output remains a shadow-only report and never renders a transition.

## M1 next check — unconditioned event localization (no generator or grid search windows)

The owner ran the wider-window shadow rise experiment. Both generated demo tracks **still abstained** (`insufficient_evidence`). In the outgoing displaced middle, the high-band wide-rise median midpoint/beat ratio was around **1.28–1.33**, versus **0.75–0.80** across the incoming grid; many events were detected on both tested locations. This modest separation is **not a sufficient independent rhythmic phase classification** and must not be used to relax thresholds. The low-band middle contrast remains strong, but cannot alone establish meter/beat truth.

Added `scripts/m1-unconditioned-transient-localization.py` and `tests/test_m1_unconditioned_transient_localization.py`. The script now detects peaks over the **whole WAV independently of the beat grid**, in low (35–155 Hz) and high (2.5–9 kHz) bands, based on local positive filtered-energy rises and acoustic peak prominence. **Only after peaks are located** does it compare those independently observed timestamps to the unchanged beat grid and inter-beat midpoints. It reports match counts, median timing errors, unique matched events and overlaps where a single event is associated with both phases. The known synthetic source-region boundaries are used **only for reporting**, not event localization or classification. No known BPM, generated beat oracle, or generated instrument template enters the algorithm. Results are exploratory: global peaks can be generated by non-drum sounds, and nearby events can be double-associated under an 80 ms tolerance. The algorithm **does not classify or authorize beat fixes**.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_unconditioned_transient_localization.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_TEMPO_CONSISTENCY=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-unconditioned-transient-localization.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-unconditioned-transients-01.json
```

Compare `tracks[0].bands.low.regions` versus `tracks[0].bands.high.regions` for **outgoing_middle** and the intact incoming region. Inspect `localized_global_events`, `beat_within_tolerance_count`, `midpoint_within_tolerance_count`, `unique_*_events_matched`, and `shared_event_count`. Only repeatable audio-event timing with meaningful negative controls should advance toward a further **shadow veto candidate**. Do not reuse the synthetic oracle in production, change the 25 ms overlap envelope, promote the neural challenger, or render rejected overlaps.

PR #28 is explicitly a combined **draft M0 baseline + M1 rhythm-phase research** branch, not its original docs-only M0 draft. Before merging, split or otherwise thoroughly scope-review the combined changes; the M1 render gate remains blocked.

## Owner grid-independent transient audit — robust kick identity still unresolved

The owner executed `scripts/m1-unconditioned-transient-localization.py` against the rights-free synthetic pair, obtaining **256 high-band events on each track** and **459/441 low-band events** on outgoing/incoming. These are globally localized onset peaks, **not kick identities**. The high-band counts are consistent with events on both beat and offbeat positions: for outgoing middle, high-band within-80-ms matches were **99/99 beats and 99/99 midpoints**, with median errors of 34.830 and 33.559 ms; incoming similarly had **125/125 beat and midpoint matches**. Thus high-band presence does not discriminate the phase.

Low-band localization gives the expected regional tendency but includes many secondary events:

| Source | Low-band beat matches | Low-band midpoint matches | Median absolute beat error | Median absolute midpoint error |
|---|---:|---:|---:|---:|
| Outgoing start | 17/17 | 4/17 | 19.410 ms | 84.172 ms |
| **Outgoing middle** | **82/99** | **99/99** | **63.673 ms** | **22.675 ms** |
| Outgoing end | 6/6 | 0/6 | 18.685 ms | 166.712 ms |
| Incoming whole track | 125/125 | 104/125 | 18.503 ms | 27.756 ms |

The fact that 104/125 **correctly aligned incoming midpoints** match a low-frequency peak within 80 ms shows why counting any global low-band onset is too permissive: secondary kick oscillation, bass events, and neighboring peaks can produce false associations. No source-grounded conclusion says those events are true kicks.

### Fixed sensitivity sweep before any threshold selection

Added research-only `scripts/m1-low-band-event-robustness.py` and `tests/test_m1_low_band_event_robustness.py`. The script runs the **same globally localized event detector** using a *predeclared* low-band parameter grid (minimum event spacing 85, 150, 250, 250, 350 ms, paired with threshold fractions 0.12, 0.12, 0.12, 0.30, 0.30). All five settings are reported; **none is selected**. For each, it reports global event counts, unique beat/midpoint event association and regional timing/coverage, always comparing only *after* acoustic peak extraction. It never changes the beat grid, tempo, neural authority, safe plan or 25 ms phase envelope. This sweep explores sensitivity and should **not** be tuned to optimize the two synthetic examples without external negative-control validation.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD

python -m pytest -q tests/test_m1_low_band_event_robustness.py

DJAI_NEURAL_BEATS=1 DJAI_NEURAL_DOWNBEAT_INTEGRITY=1 \
DJAI_TEMPO_CONSISTENCY=0 DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python scripts/m1-low-band-event-robustness.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --output /tmp/saund-m1-low-band-robustness-01.json
```

Next inspect whether event counts approach a plausible number of primary kick attacks and whether **both** the correct incoming beat and incorrect outgoing middle regions retain clean directional separation across multiple predeclared settings. Failing that, remain in diagnostic abstention and prioritize explicit negative-control and instrument-separation tests rather than choose an opportunistic threshold. No production classifier or rendering gate changes are authorized.

## Negative controls: reject premature 250-350 ms event-spacing promotion

The operator executed the fixed five-setting low-band event robustness sweep. The two synthetic tracks showed strong separation at the **three** configurations with minimum spacing **250 or 350 ms**: both returned **128** globally localized low-band events. The outgoing middle had **0/99** beat matches and **99/99** between-grid midpoint matches (median beat/midpoint errors **222.766/22.675 ms**); the incoming had **125/125** beat matches and **0/125** midpoint matches (median errors **18.503/223.129 ms**). The outgoing start and end remained grid-aligned. The original 85/150 ms settings generated hundreds of secondary low-band peaks and many false proximity matches, so apparent success is highly dependent on reasonable event selection.

This does **not** authorize selecting the 250-350 ms spacing as a general-purpose detector. Added `scripts/m1-low-band-negative-controls.py` with `tests/test_m1_low_band_negative_controls.py` to show the implications of such spacing on a **predeclared** deterministic synthetic set: 120 BPM stable pulses, **200 BPM / 300 ms beats**, acceleration, three-pulse accents, weak/noisy kicks, syncopated bass distractors, and silence. All five spacing/threshold settings are reported for every control, with one-to-one post-hoc matching of globally localized peaks against known control-event timestamps. The ground-truth timestamps **never enter the detector**; they are only used to quantify recovery and interference. There is no parameter selection, beat repair, or rendered transition. A focused regression verifies that the **350 ms spacing loses at least some legitimate 300 ms-spaced events**, preventing false blanket approval of that rule.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_low_band_negative_controls.py

python scripts/m1-low-band-negative-controls.py \
  --output /tmp/saund-m1-low-band-negative-controls-01.json
```

Check `cases[].settings[].primary.recall`, `distractor.recall`, and `detected_event_count` for each fixed configuration. A high false-positive / missed-event rate in any control is research evidence to continue abstaining, not a signal to optimize thresholds against the same examples. Extend evaluation on rights-cleared real music and multiple tempo/meter styles before considering any shadow gate promotion. The existing **25 ms continuous phase-integrity limit**, frozen neural selection authority, and no-render-on-unsafe policy remain unchanged.

## Owner negative-control execution: 350 ms loses fast beats; 250 ms remains unvalidated

The owner ran `scripts/m1-low-band-negative-controls.py` against the predeclared seven synthetic cases and five settings. The script reported `selected_configuration=null`, `generator_reference_used_in_detector=false`, `production_changed=false`, and `transition_rendered=false`. The pasted terminal output shows the report, but does **not** include the pytest invocation or its result; do not claim regression tests passed on this evidence alone.

| Synthetic control | 250 ms / 0.30 primary recovery | 350 ms / 0.30 primary recovery | Other observed limitation |
|---|---:|---:|---|
| Stable 120 BPM | 23/23 | 23/23 | — |
| Fast 200 BPM, 300 ms pulses | **38/38** | **17/38 (44.7%)** | 350 ms rejects 21 real pulses |
| Accelerating | **29/29** | **25/29 (86.2%)** | Four real pulses lost at 350 ms |
| Three-pulse accents | 24/24 | 24/24 | Synthetic accents, not a variable-meter corpus |
| Weak/noisy kicks | 23/23 | 23/23 | 250 ms/0.30: 3 unmatched events; 350 ms/0.30: 2 unmatched |
| Syncopated bass | 23/23 | 23/23 | 250 ms/0.30 retains one of 23 distractors; 350 ms/0.30 retains none |
| Silence | No reference pulses; 0 detections | No reference pulses; 0 detections | Null recall correctly represents no ground truth |

The 85 and 150 ms settings generally preserved primary synthetic events but retained substantially more distractor peaks or noise (e.g. weak-kick case 104/61 events for 23 true kicks, and syncopated case 46 events for 23 true kicks plus 23 distractors). The fixed **250 ms minimum spacing** retains all primary synthetic events in these seven cases, and the original 122/124 BPM pair separated correctly, but this is a narrow in-sample finding. It is **not** a recommended or production-ready cutoff, and should not be tuned against these same fixtures.

**Decision:** reject fixed 350 ms event spacing for general use; retain 250 ms only as an unpromoted research comparator. A next shadow experiment may compare predeclared locally tempo-adaptive spacing with that comparator, using *independent audio-derived tempo hypotheses* and controls outside the current synthetic fixtures. Include 240+ BPM, halftime/doubletime, kick omissions, alternating bass emphasis, meter variation, tempo ramps, and rights-cleared real tracks. Any unreliable tempo estimate, conflicting phase evidence or absent events must yield an abstention rather than a beat adjustment. Keep the existing 25 ms planner phase-integrity gate and no-render-on-reject behavior unchanged.

## M1 research-only audio-adaptive low-band event-spacing comparison

The operator's completed seven-control negative-control report showed that fixed 350 ms event spacing retained only **17/38** reference events at 200 BPM and **25/29** under acceleration, versus **38/38** and **29/29** at fixed 250 ms. At fixed 250 ms with 0.30 threshold, the existing controls also retained **23/23** stable 120 BPM, **24/24** three-pulse accent, **23/23** weak-kick, and **23/23** syncopated-bass primary attacks. That restricted sample does not validate a general rhythm integrity policy.

Added `scripts/m1-audio-adaptive-event-spacing.py` and `tests/test_m1_audio_adaptive_event_spacing.py` as a controlled **research-only comparator**, not a production replacement. It estimates one candidate pulse interval from positive **low-band WAV energy-rise autocorrelation** in overlapping six-second audio windows (three-second stride). No analysis beat grid, BPM label, synthetic reference schedule, or generator oracle is used in period estimation. Weak, ambiguous or inconsistent window evidence returns `insufficient_evidence` and the adaptive comparator **abstains**. When supported, exploratory spacing is `clip(0.55 * audio_period_ms, 90, 320)`. The unchanged fixed **250 ms** comparator always runs; no configuration is chosen.

The script uses seven previous controls plus three new synthetic controls: **260 BPM**, **every-fifth kick omitted**, and **alternating offbeat bass distractors**. All reference pulse times enter only **post-hoc one-to-one recovery scoring**. Compare `primary.recall`, `distractor.recall`, event counts, audio-period support and abstention per case; do not infer acceptable musical meter from an ACF pulse. ACF can be confused by half/double-time harmonics, weak kick/bass balance, swing, complex meters, and tempo ramps. A false or confident-looking tempo estimate is not proof of a correct beat, and the eight-second or six-second window selection does not imply real-world validation. The chosen audio periodicity is a **single global comparator**; genuinely local changing-tempo adaptation is future research and is **not** claimed here.

Mac test and research report:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD

python -m pytest -q tests/test_m1_audio_adaptive_event_spacing.py

python scripts/m1-audio-adaptive-event-spacing.py \
  --output /tmp/saund-m1-audio-adaptive-spacing-01.json
```

The six tests include abstention, no oracle/beat-grid input, deterministic extra controls, both strategies with missing evidence, and explicit no-production-change assertions. Tests and report **must be run by the operator** before any performance claims; do not substitute expectations for results. Any unsupported/contradictory evidence remains a no-decision, not a correction or safe planner bypass. No existing production tempo selection, neural authority, 25 ms transition phase gate, or render output is changed.

## Owner audio-adaptive spacing comparison: 260 BPM recovery improves; acceleration abstains

The operator ran `scripts/m1-audio-adaptive-event-spacing.py` on its predeclared **10 synthetic cases**. Report declares `selected_configuration=null`, `beat_grid_used_for_period=false`, `generator_reference_used_for_period=false`, `reference_used_only_for_posthoc_scoring=true`, and no production/render changes. Only execution of the report is shown here; the pasted output does not show a pytest result, so **do not claim the six new tests passed** without separate evidence.

| Synthetic case | Fixed 250 ms primary recovery | Audio-derived adaptive primary recovery | Adaptive pulse/behavior |
|---|---:|---:|---|
| Stable 120 BPM | 23/23 | 23/23 | 500 ms period, 275 ms spacing |
| Fast 200 BPM | 38/38 | 38/38 | 300 ms period, 165 ms spacing |
| **Very fast 260 BPM** | **30/49 (61.2%)** | **49/49 (100%)** | 230 ms measured ACF period, 126.5 ms spacing |
| Three-pulse accents | 24/24 | 24/24 | 480 ms period, 264 ms spacing |
| Weak/noisy kicks | 23/23, 3 unmatched events | 23/23, 2 unmatched events | 500 ms period, 275 ms spacing |
| Syncopated bass | 23/23, 1 unmatched event | 23/23, 0 unmatched events | 500 ms period, 275 ms spacing |
| Missing kicks | 18/18 | 18/18 | 500 ms period, 275 ms spacing |
| Alternating bass | 23/23, 1 unmatched event | 23/23, 0 unmatched events | 500 ms period, 275 ms spacing |
| Accelerating | 29/29 | **abstained** | `weak_or_ambiguous_acf` |
| Silence | 0 events, null recall | **abstained** | `weak_low_band` |

The 260 BPM case demonstrates the *specific* failure of fixed 250 ms spacing: the true pulse separation of approximately 230.8 ms is shorter than the enforced spacing, yielding only 30/49 recovered pulses, whereas audio-derived 126.5 ms spacing retains all 49. The adaptive approach also removed the extra event in each of two controlled bass-distractor cases. Neither result establishes generalizable beat identity or a safe rhythm-grid correction.

**Decision:** Keep adaptive audio-derived spacing as a **shadow research challenger** and fixed 250 ms as an explicitly unpromoted comparator. The current algorithm estimates a **global** ACF period from three six-second windows and refuses strong disagreement; it is **not** a local tempo-tracker and does not solve the accelerating control. Next experiments should add adversarial half-/double-time ambiguity, triplets, shuffled/offbeat bass, complex meter, low-kick/high-bass mixes, different sample rates, and genuinely time-varying tempo. Require independent rights-cleared real-audio validation, stable negative-control behavior and an end-to-end *unforced* safe planner pass before any production use.

No playback rate or beat grid was altered; the frozen neural policy, 25 ms continuous phase-integrity limit, and no-render-on-unsafe rule remain intact.

## M1 adversarial audio-period ambiguity and changing-tempo evaluation (research only)

The owner completed the first 10-case audio-adaptive spacing report. Compared with fixed 250 ms spacing, an audio-derived ACF period retained **49/49** actual generated 260 BPM pulses (versus **30/49** fixed spacing), selected a 126.5 ms shadow event separation from a 230 ms waveform-period hypothesis, and preserved 23/23 primary pulses without the single extra detection for each of the syncopated/alternating-bass controls. But the **accelerating** case returned `weak_or_ambiguous_acf` and explicitly abstained; silence abstained too. This is synthetic evidence only, not an approved tempo/beat interpretation or a production safeguard.

Added `scripts/m1-audio-adaptive-adversarial-controls.py` and `tests/test_m1_audio_adaptive_adversarial_controls.py` to evaluate **eight more deterministic waveform controls** without generator timing in the estimator:

- `halftime_accents_120`: alternating strong/soft primary low-band beats to expose half-time aliasing
- `doubletime_bass_120`: additional strong offbeat bass transient on every beat to expose double-time aliasing
- `halfbeat_bass_120`: additional bass transient on alternating beat midpoints
- `tempo_step_120_to_200`: a genuine source pulse-period change in the middle
- `tempo_ramp_100_to_220`: a continuous pulse acceleration
- `three_beat_accents` and `six_beat_accents`: differing accent cycles without assuming that accent grouping is musical meter
- `low_kick_high_bass`: deliberately weak kick signals competing with stronger low-band offbeat bass

The evaluator calls the **unchanged** audio-only ACF estimator and the fixed 250 ms versus audio-adaptive event locators. After the estimator has returned, it compares its period with the known generated pulse schedule to report a **post-hoc half-/double-time alias flag** (ratio near 0.5 or 2.0). The generator reference is strictly scoring-only: it never feeds the period estimator, peak finder, event-spacing selection, or production beat grid. An ACF period can still sound plausible while corresponding to the wrong musical subdivision. The report does not choose a setting or correct a beat.

Run on the operator's Mac:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD

python -m pytest -q tests/test_m1_audio_adaptive_adversarial_controls.py

python scripts/m1-audio-adaptive-adversarial-controls.py \
  --output /tmp/saund-m1-audio-adaptive-adversarial-01.json
```

Inspect `cases[].audio_periodicity.status`, `hypothetical_period_to_reference_ratio_posthoc`, `potential_half_or_double_time_alias_posthoc`, `strategies[].primary.recall`, `strategies[].distractor.recall`, and abstentions. Treat any confidently wrong half/double estimate, tempo-change overgeneralization, or distractor substitution as a **research gate failure**. These synthetic tests cannot certify real-world musical meter; separately validate with rights-cleared diverse audio before any opt-in policy proposal. Keep PR #28 draft, the frozen neural authority, and the unchanged 25 ms phase-integrity planner stop. No rejected transition may render.

## M1 owner adversarial ACF result: confident wrong-phase event selection blocks promotion

The owner executed `scripts/m1-audio-adaptive-adversarial-controls.py` on all eight adversarial generated-audio controls. Its most consequential counterexample is `low_kick_high_bass`: the waveform-only period estimator reported **supported_audio_periodicity**, **120.0 BPM**, **0.5 s** and median ACF strength **1.0**, yet both fixed 250 ms and the 275 ms audio-adaptive event selector matched **0/35 true kick pulses** and **35/35 stronger offbeat bass distractors**. Therefore **correct estimated periodicity does not imply correct acoustic beat phase**. The post-hoc ACF half-/double-time alias flag was false, and is not a sufficient phase-integrity diagnostic.

The remaining adversarial results showed adaptive **abstention** on `halftime_accents_120`, `doubletime_bass_120`, `halfbeat_bass_120`, `tempo_step_120_to_200`, and `tempo_ramp_100_to_220`; the latter tempo-step case reported `variable_or_conflicting_audio_periodicity` and all others reported `weak_or_ambiguous_acf`. `three_beat_accents` and `six_beat_accents` produced supported 120 BPM and each retained 35/35 primary pulses on both strategies. Those accent cases do **not** independently certify musical meter.

**Engineering gate: failed.** Neither audio-adaptive nor fixed-spacing transient selection may be promoted to a production rhythm integrity detector, used as an automatic beat correction, or allowed to sign a transition. Updating the adversarial report now provides explicit *post-hoc scoring-only* fields: `posthoc_confident_wrong_phase_event_selection` and `posthoc_research_gate` per case, plus `research_gate=failed_confident_wrong_phase_counterexample` and `any_posthoc_confident_wrong_phase_selection=true` in the overall report when a supported period accompanies <=20% true-pulse recall and >=80% distractor recall with adequate reference events. These labels use **known synthetic reference events only after evaluation**; they are *not production-available detections*, and their absence never means the model is validated. Regression tests cover correct-period/wrong-phase triggering and safe abstention.

The next research question is **waveform-only phase ambiguity**: how to identify two competing rhythmic event sequences and decline a single confident source identity when low-frequency bass emphasizes offbeats while weaker kicks retain the primary pulse. Any candidate must be challenged against independent waveforms, syncopation, changing meter, tempo ramps, kick dropouts and rights-cleared real music. A global autocorrelation period by itself is insufficient. Keep the original beat grid unchanged, frozen neural promotion rules, **25 ms planner phase envelope**, and no-render-on-rejected overlap policy.

Re-run on Mac:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_audio_adaptive_adversarial_controls.py
python scripts/m1-audio-adaptive-adversarial-controls.py \
  --output /tmp/saund-m1-audio-adaptive-adversarial-02.json
```

## Owner re-run confirms explicit post-hoc wrong-phase research gate failure

The owner executed the updated `m1-audio-adaptive-adversarial-controls.py` report and confirmed the expected scoring-only safety fields: `any_posthoc_confident_wrong_phase_selection=true`, `research_gate=failed_confident_wrong_phase_counterexample`, `posthoc_gate_is_scoring_only=true`, `selected_configuration=null`, `production_changed=false`, `beat_grid_modified=false`, and `transition_rendered=false`.

For `low_kick_high_bass`, audio autocorrelation reported a supported **120 BPM / 500 ms** periodicity (strength 1.0), but both the fixed 250 ms and audio-adaptive 275 ms selectors recovered **0/35** primary kick events and **35/35** offbeat bass distractors. The case now reports `posthoc_confident_wrong_phase_event_selection=true` and `posthoc_research_gate=fail_wrong_phase_event_selection`, even though the half-/double-time alias flag remains false. This is a confirmed **research gate failure**, not a detected production-safe beat phase.

The five adversarial cases `halftime_accents_120`, `doubletime_bass_120`, `halfbeat_bass_120`, `tempo_step_120_to_200`, and `tempo_ramp_100_to_220` continued to abstain. The three-/six-beat accent cases returned supported 120 BPM and 35/35 synthetic primary-event recovery, without implying meter certification. The supplied evidence confirms the script report only; **pytest execution was not included** in this paste.

**Do not promote.** The explicit wrong-phase check is calculated from known synthetic reference timestamps **after** selection. Real tracks do not supply these labels, so it cannot serve as a runtime phase-integrity guard. The next research requirement is an audio-only ambiguous-phase detector that recognizes competing kick/bass event sequences and abstains without relying on those synthetic labels. No automatic beat rewrite, safe transition authorization, neural policy promotion, or 25 ms planner phase threshold change.

## M1 shadow audio-only competing-phase evidence (no synthetic oracle)

Following the owner's confirmed `low_kick_high_bass` failure — **supported 120 BPM waveform periodicity but 0/35 true kick events and 35/35 strong offbeat bass events chosen** — a standalone research-only `scripts/m1-competing-acoustic-phase-trains.py` evaluates whether the waveform contains **two independent periodic low-band attack trains roughly half an audio-derived pulse period apart**. The goal is **ambiguity detection and abstention**, not selection of a musically correct phase.

The experiment first calls the existing *waveform-only* ACF estimator. If its period is unsupported/ambiguous, it immediately abstains. Otherwise it globally localizes low-band peaks with a short minimum distance (**85 ms**) to retain potential secondary peaks, then uses a circular phase anchor inferred from acoustic peak strengths to check event coverage near an anchor and an opposite half-period phase across complete cycles. It reports rival/primary coverage, relative acoustic rise strength, event counts and the status `competing_acoustic_phase_trains` when a sustained secondary phase is observed. If not, the result is `insufficient_evidence` — **never “correct phase confirmed”**. The report passes neither TrackAnalysis beat-grid timestamps nor synthetic generator beat labels to this detector, and the generated control labels are used only to select controlled WAVs. This is a deliberately limited heuristic, not kick-source separation.

The tests in `tests/test_m1_competing_acoustic_phase_trains.py` cover an explicit secondary event train, a clean single train, weak/sparse second trains, invalid periods and no peaks, lack of mutation, waveform-ACF abstention, and the report's shadow-only authority. The eight prior adversarial cases plus stable 120 BPM, fast 200 BPM, syncopated bass, and silence are included in the Mac run. Critically, **the owner has not yet executed the new tests or report**. The desired first challenge is whether `low_kick_high_bass` raises competing evidence instead of presenting a single phase as validated, without spuriously claiming that the primary kick train is identified.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_competing_acoustic_phase_trains.py

python scripts/m1-competing-acoustic-phase-trains.py \
  --output /tmp/saund-m1-competing-acoustic-phase-trains-01.json
```

This diagnostic **must not be promoted** based on controlled synthetic fixtures. It can miss quiet kick trains, detect false competing patterns from non-drum bass, and cannot resolve correct musical beat identity. Require negative-control sensitivity, rights-cleared variable-meter and syncopation examples, and a subsequent unforced safe-planner evaluation before any operational policy. The production beat grid, neural authority and immutable **25 ms** transition phase-safety requirement remain unchanged; rejected transitions are never rendered.

## Owner competing-acoustic-phase report: critical counterexample now yields waveform-only ambiguity evidence

The owner ran `scripts/m1-competing-acoustic-phase-trains.py` on the 12 synthetic controls. In the prior `low_kick_high_bass` adversarial report, both audio-adaptive and fixed event selectors matched **0/35 kick events and 35/35 offbeat bass events** despite supported **120 BPM** periodicity. The new **audio-only** competing-phase diagnostic, operating on waveform-localized peaks and the audio-derived period without generator labels or a beat grid, reported:

- `low_kick_high_bass`: `competing_acoustic_phase_trains`; **70 localized acoustic events**, **35 complete cycles**, **0.971 anchor event coverage**, **1.000 opposite event coverage**, **0.00983 weaker/stronger median acoustic rise**. Its phase anchor was **0.254 seconds**. The two trains were observed, **not identified as true kicks versus bass**.
- `syncopated_bass`: `competing_acoustic_phase_trains`; 46 global events, 23 complete cycles, **1.000 coverage for each phase**, median weaker/stronger ratio **0.4225**. This is a useful phase-ambiguity control but also shows that competing phase evidence does not automatically imply an unsafe musical grid.
- `fast_200`: supported 200 BPM waveform period, 38 events, 39 cycles, **0.974 anchor coverage and 0.000 opposite coverage**; result `insufficient_evidence`, **not a safe-phase certification**.
- `three_beat_accents` and `six_beat_accents`: supported 120 BPM, 35 events, complete anchor coverage, zero opposite coverage, also `insufficient_evidence`.
- `stable_120`: supported 120 BPM but `too_few_localized_events` with **23 primary events**, because the current heuristic demands at least **24** events (`2 * min_cycles`) even before comparing cycles. This event-count criterion is an explicit research limitation and may be overly conservative.
- `halftime_accents_120`, `doubletime_bass_120`, `halfbeat_bass_120`, `tempo_step_120_to_200`, `tempo_ramp_100_to_220`, and `silence`: all `insufficient_evidence` after the audio-periodicity estimator abstained.

**Interpretation:** This is a meaningful shadow diagnostic result: the known confidently wrong offbeat selection now has independently observed opposing acoustic phase evidence. It remains non-authoritative. The observed **0.00983 weaker/stronger ratio** is only just above the experiment's **0.006** configured floor, so detector sensitivity, waveform noise, and false positives must be tested before generalizing. The clean syncopated case also yields competing evidence; a future production veto must not treat all such evidence as proof of unsafe beat phase. The owner-provided terminal output shows the experiment, not a separate pytest result.

**Decision:** Keep all acoustic-phase conclusions **shadow-only**, with no approved beat identity and no transition authorization. Next validate weaker-rival sensitivity with amplitudes/noise variants and inspect the too-restrictive minimum-event count. Then run rights-cleared non-synthetic audio controls and compare against the actual analyzed grid before considering a veto-only policy. Continue to enforce the existing **25 ms** continuous phase-lock envelope and do not render unsafe plans.

## M1 predeclared acoustic phase-sensitivity sweep and minimum-event guard

The previous owner-run waveform-only competing-phase experiment reported `competing_acoustic_phase_trains` for the catastrophic `low_kick_high_bass` control: 70 events, anchor/opposite coverage **0.971 / 1.000**, and weaker-to-stronger median rise ratio **0.00983**. Clean `syncopated_bass` also triggered ambiguity, while `stable_120` had only 23 localized events and was rejected by the old **24-event minimum** before phase comparison. This is research evidence, not proof of correct kick identity.

Changed **only** the shadow-only `scripts/m1-competing-acoustic-phase-trains.py` event minimum from `2 * min_cycles` to `min_cycles`. A single periodic attack train can legitimately contain only one event per cycle; requiring twice as many should not prevent the instrument-agnostic comparison from observing that there is *no demonstrated rival*. This does **not** turn a clean single-phase report into a "safe beat" classification: the only statuses remain `competing_acoustic_phase_trains` or `insufficient_evidence`.

Added `scripts/m1-competing-phase-sensitivity.py` plus `tests/test_m1_competing_phase_sensitivity.py` to run a **predeclared 24-case synthetic diagnostic matrix**, without modifying production code:

- **15 two-phase controls:** kick amplitudes 0.02, 0.06, 0.12, 0.25 and 0.5 against constant strong offbeat bass amplitude 1.1, crossed with independent deterministic noise standard deviations 0, 0.001, and 0.005.
- **Nine baseline controls:** stable pulse-only, strong offbeat bass-only, and noise-only, each at the same three noise levels.

The fixture generator returns only a waveform/sample rate to the analysis; no generator timestamps or beat grid are passed into the audio-only ACF estimator, globally localized event finder or rival-phase comparator. Inspect `cases[].audio_periodicity.status`, `cases[].phase_evidence.status`, `anchor_event_coverage`, `opposite_event_coverage`, `weaker_to_stronger_median_rise_ratio` and abstentions. Do **not** post-select the weakest kick/noise combination that passes. A detection in bass-only/noise-only controls or a missed quiet rival in mixed tracks reveals the limits of this heuristic; neither outcome certifies musical beat phase. The comparison still cannot distinguish kick identity, and an unsupported ACF period must abstain.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_competing_phase_sensitivity.py tests/test_m1_competing_acoustic_phase_trains.py

python scripts/m1-competing-phase-sensitivity.py \
  --output /tmp/saund-m1-competing-phase-sensitivity-01.json
```

**Status:** New script/tests have been committed but owner Mac run results are still needed. This remains a **shadow-only abstention experiment**; no beat-grid repair, neural authority change, plan/render permission or modification of the **25 ms** continuous phase-integrity requirement is authorized. Require broader audio and independently rights-cleared music before any proposal for production veto integration.

## Owner 24-case competing-phase sensitivity results: 9/15 mixed detections, 0/9 negative flags

The owner executed `scripts/m1-competing-phase-sensitivity.py` on its predeclared 24 synthetic cases. The result is **not** uniform success: the shadow-only two-phase detector reported `competing_acoustic_phase_trains` in **9/15** mixed kick-plus-offbeat-bass examples, covering all three noise conditions at kick amplitudes **0.12, 0.25 and 0.5**. At weaker amplitudes **0.02 and 0.06**, it reported `insufficient_evidence` in **all six** cases. It did not report competing phases in **any of nine** stable-pulse-only, offbeat-bass-only, or noise-only controls. These are measured fixture outcomes, not population rates or a validated confusion matrix.

| Kick amplitude against offbeat bass 1.1 | Noise 0 | Noise 0.001 | Noise 0.005 | Median weaker/stronger event-rise examples |
|---|---|---|---|---|
| 0.02 | insufficient | insufficient | insufficient | 0 or approximately 0.00027 |
| 0.06 | insufficient | insufficient | insufficient | approximately 0.00244-0.00246 |
| 0.12 | competing | competing | competing | approximately 0.00980-0.00983 |
| 0.25 | competing | competing | competing | approximately 0.04261-0.04269 |
| 0.50 | competing | competing | competing | approximately 0.17057-0.17074 |

All 15 mixed waveforms returned supported 120 BPM periodicity; **the missed weak rival phases therefore cannot be explained as ACF abstentions**. At 0.02/no noise, the event finder returned only 35 peaks, 0.000 opposite coverage, and the weak acoustic train was absent from the selected peaks. At 0.02 with noise, it returned 105 peaks and 1.000 opposite coverage but the weaker/stronger ratio was merely 0.00027, well under the research heuristic's fixed 0.006 relative-strength minimum. At 0.06, opposite event coverage was 1.000 yet the ratio 0.00244-0.00246 remained below the minimum. **Event presence alone cannot override inadequate signal evidence.**

The nine single-source/noise-only controls did not trigger `competing_acoustic_phase_trains`, including the strong bass-only sequence; pure noise produced ACF abstention. Importantly, the stable-pulse-only control at noise standard deviation 0.005 produced **0.543 opposite coverage** from 152 globally localized events, although its weak/strong ratio was only 0.00035. This highlights a potential false-ambiguity risk if coverage or strength cutoffs were weakened to rescue the quiet-kick cases. The clean single-phase 120 BPM controls now reach comparison without the old 24-event count gate, but their `insufficient_evidence` output **still does not certify correct beat phase**.

**Decision: still blocked for production.** Do not tune the 0.006 weakness threshold or 0.60 phase-coverage threshold against this same sweep. Prioritize additional independent random seeds, colored/transient noise, low-kick/high-bass timbres, timing jitter, offbeat subdivisions, and rights-cleared real music to estimate robustness beyond these fixtures. A candidate safeguard must separately establish that it can abstain on an unsafe/unknown phase *without* rejecting ordinary syncopated music or incorrectly approving a one-phase bass train. The current detector is advisory/shadow-only; the frozen neural authority, unchanged 25 ms planner phase-integrity envelope and no-render-on-reject rule remain in effect. The pasted report did not include pytest output, so no test-pass claim is made.

## Shadow-only robustness matrix: independently varied noise, seeds, kick timbre and jitter

The previous 24-case weak-phase sensitivity sweep detected competing phase evidence in **9/15** mixed kick-plus-offbeat-bass examples (kick amplitude >=0.12) and in **0/9** single-source/noise-only controls. At kick amplitude 0.02 and 0.06, all six mixed examples were missed despite supported 120 BPM ACF periodicity. A pulse-only noisy control had 54.3% spurious opposite-phase event coverage, close to the 60% coverage threshold, though its rival strength was only 0.00035. These are **narrow deterministic synthetic results**, not real-world sensitivity or specificity estimates.

Added `scripts/m1-competing-phase-robustness.py` and `tests/test_m1_competing_phase_robustness.py`. This is a **predeclared, separate 52-case synthetic robustness matrix**; it does not change the existing detector, its thresholds or the production pipeline. The matrix crosses **two independent random seeds** (101, 907), **white versus low-pass-colored noise**, **87 Hz decaying sinusoid versus 65→125 Hz chirped low-frequency kick**, and **0 versus ±14 ms independent beat/offbeat timing jitter**. Each combination includes a mixed waveform with a weaker kick amplitude **0.06**, a mixed waveform with weaker kick amplitude **0.12** (both against offbeat bass amplitude 1.1), and a **single-pulse-only** negative control. Four additional **noise-only** negative controls cover both noise types and seeds. The noise standard deviation is fixed at 0.002 after per-waveform noise normalization.

The fixture generator returns **only the synthetic waveform and sample rate**. The existing audio-only ACF period estimator, global low-band event locator and competing-phase comparison run without generated reference beat timestamps, known BPM, analyzed beat grid or source labels. The output aggregates `competing_phase_count` and `periodicity_abstain_count` separately for **32 mixed**, **16 pulse-only**, and **4 noise-only** examples. A competing-phase result is ambiguity evidence only; `insufficient_evidence` neither proves the track is safe nor identifies a correct beat. The thresholds **0.006 relative rival strength**, **0.60 per-phase coverage**, the ACF selection rules and the 25 ms planner envelope are unchanged. No setting may be chosen based on this limited matrix.

To run on the Mac evaluator:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD

python -m pytest -q tests/test_m1_competing_phase_robustness.py

python scripts/m1-competing-phase-robustness.py \
  --output /tmp/saund-m1-competing-phase-robustness-01.json
```

Review `summary_by_fixture` and inspect each case's `periodicity.status`, `phase_evidence.status`, `anchor_event_coverage`, `opposite_event_coverage`, and `weaker_to_stronger_median_rise_ratio`. For mixed-case misses, distinguish ambiguous ACF from failed rival-peak localization or relative-strength cutoffs. For pulse-only/noise-only cases, evaluate false competing-phase signals and abstentions, not just event counts. The tests and script have been committed but **owner execution has not yet been observed**. The complete corpus remains synthetic and insufficient for real-music correctness claims. PR #28 stays draft; neural authority, source beat grids, planner decisions, and 25 ms no-unsafe-render gate remain frozen.

## Owner 52-case robustness run: timing drift and faint-event limits

The owner executed `scripts/m1-competing-phase-robustness.py` and supplied its 52-case output. The unmodified research classifier reported **12/32** `competing_acoustic_phase_trains` decisions on actual two-phase synthetic waveforms (**37.5%**), **0/16** on clean pulse-only controls, and **0/4** on noise-only controls. All four noise-only inputs had `insufficient_evidence` audio periodicity; **none** of the 32 two-phase or 16 pulse-only fixtures abstained at the ACF-period stage. The report flags `detector_thresholds_modified=false`, `selected_configuration=null`, `generator_reference_used_in_detector=false`, `beat_grid_used=false`, `production_changed=false`, and `transition_rendered=false`.

Two failure mechanisms are visible in the detailed output. First, mixed-waveform kick amplitude **0.06** often produces weaker/stronger low-band event-rise ratios around **0.0024–0.003**, below the fixed **0.006** research threshold even when both acoustic phases have high per-cycle coverage. Second, the ±14 ms jitter controls can produce an audio-derived global period near **0.5075 s / 118.227 BPM**, rather than the fixture's nominal 0.500 s. Because the phase comparator folds events against a **single global period**, gradual phase drift can reduce measured coverage (examples: **0.565/0.565** or **0.522/0.522** anchor/opposite coverage) below the required 0.60, even with rival-to-primary rise ratios above 0.006. The actual cause is not yet definitively isolated: event timing jitter, period-estimate bias, and anchor estimation may interact. A dedicated oracle-free timing diagnostic is needed to distinguish them.

The pulse-only negative controls did not trigger a false competing-phase label, but several had substantial apparent opposite-phase coverage from noise (e.g. **0.833** on one lowpass/jittered single-phase case); their much smaller rival-rise ratios (~0.00039–0.00049) prevented the false decision. This is evidence **against indiscriminately reducing the rival-strength threshold**. Zero detections on these limited controls is not a statistically established false-positive rate for real music.

**Engineering decision:** Remain shadow-only and **do not promote** the waveform competing-phase detector or select thresholds based on this test. Next isolate phase-drift sensitivity without synthetic pulse timing as input to the detector: examine window-local audio-derived period/phase consistency, compare whether repeated half-period opposing attacks persist across windows, and carry out additional negative controls. Do not use fixture reference times except for post-hoc scoring, infer kick identity from low-frequency strength, correct production beat grids, relax the **25 ms** planner phase-integrity envelope, or render rejected transitions. The supplied report does not include a pytest result.

## M1 research: global-versus-local phase robustness comparison, unchanged gates

The owner-run 52-case waveform robustness matrix reported **12/32** competing-phase detections on known synthetic two-phase waveforms and **0/16** pulse-only plus **0/4** noise-only competing-phase flags. Some jittered cases showed supported audio periodicity approximately **0.5075 s** instead of nominal 0.500 s; cumulative global phase drift and weak transient contrast are plausible, interacting explanations for low phase coverage. These observations are descriptive, not a validated causal diagnosis.

Added `scripts/m1-window-local-phase-robustness.py` and `tests/test_m1_window_local_phase_robustness.py` to test whether **time-local analysis** makes the competing-phase *evidence* more stable without making the detector more permissive. The new script reuses **exactly the same 52 previously predeclared waveform configurations** (seeds 101/907; white/lowpass noise; decaying 87 Hz/chirping 65–125 Hz kicks; 0/±14 ms timing jitter; 0.06/0.12 mixed kick amplitudes; pulse-only and noise-only negative controls). Each generated waveform is assessed once in full and again in **overlapping 8-second segments at 0–8 s and 4–12 s**. Each call runs the **original unchanged waveform-only ACF, low-band peak locator and phase comparator**, including their existing 85 ms peak spacing, 0.02 peak threshold fraction, 42 ms tolerance, 12-minimum-cycle limit, **0.60** per-phase coverage and **0.006** weaker/stronger event strength threshold. No source/reference beat grid or synthetic timing labels enter the estimator or comparator.

The report explicitly separates:
- `global.phase_evidence.status` from `local_windows[].phase_evidence.status`;
- `local_competing_window_count` (0–2), `local_period_range_ms` (descriptive spread of supported local ACF period hypotheses) and `local_status`, which is `recurrent_competing_acoustic_evidence_shadow_only` only if **both overlapping analysis windows** report two acoustic phase trains, otherwise `insufficient_evidence`;
- `summary_by_fixture_jitter`: whole-clip competing count, any local positive count, recurrent local count and number of fixtures where both local ACF windows support periodicity, grouped by fixture variant and nominal jitter.

**Important methodological qualification:** The two overlapping windows are correlated samples of the **same** waveform. A recurrent local result is not two independent observations, not a verified kick identity, not a production decision and never automatically certifies unsafe/safe beat alignment. No numerical threshold is optimized from these fixtures; if local detection degrades, report it without changing parameter values. A lack of competing evidence remains `insufficient_evidence`, not a safe-grid pass.

Run on the owner's Mac:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_window_local_phase_robustness.py

python scripts/m1-window-local-phase-robustness.py \
  --output /tmp/saund-m1-window-local-phase-robustness-01.json
```

The new tests and waveform report have **not yet been run on the owner's Mac**. No neural candidate promotion, beat-grid adjustment, planner approval, production integration or render is permitted. The original **25 ms** continuous phase-integrity hard limit and no-render-on-rejected-transition policy remain in force; PR #28 stays draft.

## Owner global-versus-local comparison: no aggregate recovery, isolated jitter improvement

The owner ran `scripts/m1-window-local-phase-robustness.py` on the same predeclared 52 controlled waveforms. **The 8-second overlapping-window strategy made no aggregate improvement** over the 12-second global strategy under the research rule that **both** local windows must show competing phase trains:

| Fixture type | Cases | Global competing | Recurrent local competing | Any local competing |
|---|---:|---:|---:|---:|
| Two-phase, zero jitter | 16 | 8 | 8 | 8 |
| Two-phase, ±14 ms jitter | 16 | 4 | 4 | 4 |
| Pulse-only, zero jitter | 8 | 0 | 0 | 0 |
| Pulse-only, ±14 ms jitter | 8 | 0 | 0 | 0 |
| Noise-only | 4 | 0 | 0 | 0 |
| **Total** | **52** | **12** | **12** | **12** |

Both 8-second local segments returned **supported audio periodicity** for all 32 mixed and 16 pulse-only waveforms; all four noise-only inputs abstained on both local periods. The limitation remains **acoustic phase localization and evidence strength**, not simply failure to estimate a local period. The overlapping windows are correlated and their agreement cannot be counted as independent confirmation.

One individual jittered example is diagnostically useful: seed **907**, lowpass noise, **87 Hz decaying kick**, **0.12** kick amplitude, **±14 ms** jitter. The whole-clip audio period was **507.5 ms** with **0.565/0.565** coverage and `insufficient_evidence` (rival rise ratio **0.0118**, above the fixed 0.006 strength floor). Its 0–8 s window estimated **505 ms**, saw **0.933/0.800** coverage and ratio **0.01246**; the 4–12 s window estimated **495 ms**, saw **0.938/0.938** coverage and ratio **0.01142**. Both local windows reported `competing_acoustic_phase_trains`. These observations support the **possibility** that shorter windows reduce some global alignment drift, but they do not prove the underlying cause; window ACF period error, jitter and anchor-phase shifts may interact. Crucially, this local rescue does **not** improve overall detection counts because other cases remain missed.

**Conclusion:** The global-vs-local candidate has **no demonstrated aggregate benefit** on this matrix. Keep **12/32** mixed detections and **0/20** negative flags as empirical fixture counts only. Do not relax fixed research coverage (0.60), rival strength (0.006), or 42 ms tolerance based on this evaluation. If pursuing local analysis further, isolate the period bias versus anchor-phase drift with explicitly diagnostic (non-authoritative) per-window observations and independent audio controls, especially quiet kicks and timing jitter. Do not promote the global or local detector, infer the true kick phase, rewrite source beats, or authorize rejected transitions. The 25 ms phase-integrity hard gate and draft status of PR #28 remain unchanged. The pasted report alone does not establish a pytest pass.

## M1 experimental 2x2 waveform-only period-vs-anchor attribution

The owner's global-versus-overlapping-window evaluation showed **12/32** mixed-waveform acoustic phase detections under either strategy, **0/20** negative-control detections, and one individually recovered jittered waveform despite no aggregate improvement. The next bounded diagnostic is intended to distinguish **audio-period estimation drift** from **acoustic phase-anchor drift** without synthetic timing as a decision input.

Added standalone `scripts/m1-period-anchor-attribution.py` and `tests/test_m1_period_anchor_attribution.py` plus an optional `anchor_phase_seconds` argument to the existing **shadow-only** `compare_phase_trains` comparator in `scripts/m1-competing-acoustic-phase-trains.py`. The default `anchor_phase_seconds=None` retains **exactly the original waveform-derived local-anchor behavior**. This change does not affect production music analysis, neural authority or planner policy.

For each of the **same predeclared 52 synthetic waveform configurations** and **the same detected acoustic events in an 8-second local window**, compare a fixed four-arm attribution matrix:

| Research arm | Period candidate | Acoustic anchor |
|---|---|---|
| `global_period_local_anchor` | ACF hypothesis estimated from entire input WAV | inferred from local window's audio peaks |
| `global_period_held_anchor` | same full-WAV ACF hypothesis | held whole-WAV audio-peaks anchor at that candidate period |
| `local_period_local_anchor` | ACF hypothesis estimated from the current window | inferred from current window's audio peaks |
| `local_period_held_anchor` | same window ACF hypothesis | held whole-WAV audio-peaks anchor at that candidate period |

For held-anchor arms, the whole-WAV anchor is **estimated from acoustic peaks at the same period candidate**; the anchor is then translated from global to local time using `(global_anchor - window_start_seconds) % candidate_period`. This is necessary because the local comparator consumes **window-relative event timestamps**. All arms use the same local detected event list, the same original **42 ms timing radius, 12-cycle minimum, 0.60 coverage and 0.006 acoustic-rise ratio**, and the existing waveform-only ACF and 85 ms / 0.02 candidate-peak localization rules. A missing or unsupported audio-derived period or anchor must **abstain**; no synthetic true-pulse timestamps, generated BPM or track analysis grids are supplied to the estimator, comparator or anchor inference. Fixture parameters exist only to synthesize waveform test controls and group post-hoc results.

The output lists per-window `period_delta_local_minus_global_ms`, all four arms' status/coverage/relative strength, and `summary_by_fixture_jitter` with `any_window_competing_count` and `both_windows_competing_count` per arm. Windows 0–8 s and 4–12 s **overlap** and are not independent validation trials. A difference between arms provides only **descriptive attribution evidence**: locked anchors estimated on the entire waveform also share information with local windows; these are not causal interventions on real beats or grounds to select one arm for production. In particular, a "competing" result never identifies which train is the true musical kick, and a lack of competing evidence never establishes a correct beat.

Run the six new tests plus the prior companion suite on Mac:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q \
  tests/test_m1_period_anchor_attribution.py \
  tests/test_m1_competing_acoustic_phase_trains.py

python scripts/m1-period-anchor-attribution.py \
  --output /tmp/saund-m1-period-anchor-attribution-01.json
```

**Execution status:** committed research source, tests and runbook only; owner-run report and pytest evidence are pending. Do not adjust detector thresholds based on this matrix, merge or promote PR #28, touch the sealed Ballroom test, correct any production beat grids, or render unsafe transitions. The original hard **25 ms** continuous phase-integrity gate and frozen neural selection remain intact.

## Owner four-arm period-vs-anchor attribution results: global audio period plus local anchor is stronger on jitter

The owner ran `scripts/m1-period-anchor-attribution.py` on its previously declared **52 synthetic waveform fixtures**. Its `summary_by_fixture_jitter` reports the following **both-overlapping-8-second-windows** counts for competing acoustic phase evidence; the two local windows share source audio and **are not independent trials**:

| Research arm | Two-phase 0 ms jitter (n=16) | Two-phase ±14 ms jitter (n=16) | Pulse-only and noise-only flags (n=20) |
|---|---:|---:|---:|
| `global_period_held_anchor` | 8 | 4 | 0 |
| `global_period_local_anchor` | **8** | **8** | **0** |
| `local_period_held_anchor` | 8 | 4 | 0 |
| `local_period_local_anchor` | 8 | 4 | 0 |

The **global audio-derived period + window-local acoustic anchor** arm yields **16/32** two-phase detections overall (50%) versus **12/32** (37.5%) for each other arm. Under jitter alone the counts are **8/16** against **4/16**. All four arms abstained from competing-phase claims on every **16 pulse-only** and **four noise-only** negative control. These are corpus-specific research counts, not real-music error rates or independently validated efficacy.

The arm decomposition shows that the relationship is **not simply "local estimates are better."** Retaining the full-waveform period but relearning phase alignment within each shorter segment is associated with more supported competing-attack evidence in this fixture. Replacing the period with a locally estimated ACF period (with either anchor) did not yield aggregate improvement. For the seed-907, lowpass, 87 Hz decaying kick, amplitude-0.12, ±14 ms jitter example, the full-waveform ACF period was **507.5 ms**, with local hypotheses **505 ms** (0–8s) and **495 ms** (4–12s). Under the global-period/local-anchor arm the first window yielded anchor/opposite coverage **0.667/0.733**, ratio **0.0102**, and the second **0.800/0.867**, ratio **0.01026**, both flagging competing acoustic trains. This illustrates the combination's behavior but does not establish causal correctness of either period or anchor, or identify any musically correct kick.

**Decision: do not promote or select an arm.** This comparison reuses the same 52 synthetic fixtures repeatedly examined in prior research; despite a predeclared four-arm matrix, the reported gain needs entirely new, independently generated waveform/timbre/jitter controls and rights-cleared real audio before any claim of generalization. Maintain fixed **42 ms event radius, 0.60 per-phase coverage, 0.006 relative-rise threshold**, original audio-only ACF and low-band peak finder. Held global anchors are also inferred from the same waveform used by local windows, so no independent ground truth is available. A two-phase flag is *ambiguity evidence*, not proof of unsafe alignment; no flag remains `insufficient_evidence`, never a safe-beat approval. The pasted artifact shows the experiment report but not a separate pytest result.

PR #28 stays draft. No production beat-grid edit, source override, neural policy promotion, planner approval, relaxed **25 ms** continuous phase gate or rejected-transition render is authorized. The next bounded research step should independently generate new timing disturbances and source timbres before comparing the four arms against real rights-cleared tracks.

## M1 external pretrained beat/downbeat tracker benchmark: Beat This! first, no production integration

### Why this pivot

The waveform-only research discriminates some competing phase trains but cannot identify which musical beat grid is correct, and repeated tuning on a narrow synthetic fixture family risks overfitting. Before writing another custom detector, benchmark the independent pretrained open-source neural tracker **Beat This!** by CPJKU (Foscarin, Schlüter, Widmer, ISMIR 2024) on the **two existing rights-free generated demo WAVs**, without supplying generator beat schedules to inference. Its official package uses the `beat_this` CLI, `beat_this path/to/audio.wav -o output.beats`. The upstream `save_beat_tsv` file format is each timestamp in seconds followed by a tab and a beat number (**1 marks a downbeat**). Official documentation: https://github.com/CPJKU/beat_this ; https://github.com/CPJKU/beat_this/blob/main/README.md ; https://github.com/CPJKU/beat_this/blob/main/beat_this/utils.py .

Upstream's available checkpoints include `final0` (the default, 78 MB) and `small0`; for the initial predeclared run select **final0**, minimal postprocessing (**no `--dbn`**) and CPU `--gpu=-1` on Mac for reproducibility. Pin the package to **beat-this==1.1.0** and record `beat_this --help`, Python/PyTorch versions, model identifier, and checkpoint provenance/checksum in a local experiment log if possible. Upstream's inference will **automatically download a model checkpoint if it is not cached**; this is a user-invoked opt-in network action in a separate experimental environment, not a production or sealed-test dependency.

The upstream repository lists **MIT code and published model weights**, but explicitly cautions that certain source training audio has copyrighted or restricted licensing; a commercial deployment needs review of provenance and terms, not an assumption that an MIT implementation resolves all rights. Do not import Mixxx GPL code or Essentia AGPL code into a proprietary production codebase without a distinct licensing assessment.

### Implementation

Committed `scripts/m1-beat-this-benchmark.py` and `tests/test_m1_beat_this_benchmark.py`. The benchmark itself imports **no Beat This/PyTorch** and downloads no models. It only reads external `.beats` predictions from its `--predictions-root` and the two generated WAVs under `--audio-root`. It optionally runs the existing unmodified SAUND analysis with `--compare-saund` and all existing model-download switches disabled. To prevent leakage, **the predeclared fixture BPM (122/124) is used only to locate independent measurable low-frequency kick attacks after inference**, via `m1-synthetic-audio-kick-audit.py`. Beat This never receives the fixture truth. The benchmark reports one-to-one matching at **80 ms** for the whole recording and the already predeclared outgoing 9.636281–58.305306 s grid-failure window. This is a **research evaluation tolerance**, **not** the 25 ms production planner safety envelope. The parser rejects unparseable, duplicated/nonmonotonic, nonfinite, negative and out-of-duration model timestamps. Beat numbers of 1 are preserved as predicted downbeats, **but ground-truth downbeat correctness is not measured by this kick-only benchmark**. The two generated tracks are not a representative real-music test set.

### Operator Mac commands

Clone state first; keep the existing SAUND `.venv` separate from the third-party tracker to avoid dependency conflicts.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_beat_this_benchmark.py

python3 -m venv /tmp/saund-m1-beat-this-venv
/tmp/saund-m1-beat-this-venv/bin/python -m pip install --upgrade pip
/tmp/saund-m1-beat-this-venv/bin/python -m pip install "torch>=2.0" "beat-this==1.1.0" tqdm einops soxr rotary-embedding-torch

mkdir -p /tmp/saund-m1-beat-this-predictions-01
/tmp/saund-m1-beat-this-venv/bin/beat_this --help
/tmp/saund-m1-beat-this-venv/bin/beat_this "$M1_AUDIO_ROOT/01_opening_122bpm_Aminor.wav" -o /tmp/saund-m1-beat-this-predictions-01/01_opening_122bpm_Aminor.beats --gpu=-1 --model final0
/tmp/saund-m1-beat-this-venv/bin/beat_this "$M1_AUDIO_ROOT/02_groove_124bpm_Eminor.wav" -o /tmp/saund-m1-beat-this-predictions-01/02_groove_124bpm_Eminor.beats --gpu=-1 --model final0

python scripts/m1-beat-this-benchmark.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --predictions-root /tmp/saund-m1-beat-this-predictions-01 \
  --output /tmp/saund-m1-beat-this-comparison-01.json \
  --compare-saund
```

Only invoke the `beat_this` binary after reviewing installation and agreeing to the one-time external checkpoint download. Use **new output paths** for later runs; the benchmark refuses overwrite. If package incompatibilities arise on the Mac, record the error and do not alter production dependencies to force installation.

### Evaluation decision

Compare `beat_this.whole_track` and `beat_this.outgoing_failure_region` to `saund_current` on the two WAVs, emphasizing how many beats the external model places near measured acoustic kicks within the 99-beat displaced middle region. Independently assess predicted downbeat consistency as a later separate step; the kick-window oracle does not label meter/downbeats. If Beat This succeeds, independently validate on **new generated and rights-cleared non-training music** before proposing an opt-in research candidate; if it fails, investigate an ensemble with an independent reference tracker such as Essentia only after license review. No external result can bypass the frozen neural challenger policy or the **25 ms** continuous phase-integrity render gate. PR #28 stays draft; no sealed Ballroom test is touched.

## Owner external Beat This! final0 benchmark: 99/99 outgoing failure region recovered

The owner successfully installed **beat-this==1.1.0** and its required `soundfile==0.14.0` WAV fallback inside the separate experimental Mac venv; the default **final0 checkpoint (~77.3 MB)** downloaded and cached. The first run failed before model inference because `torchaudio` required `torchcodec`, with `soundfile` and `madmom` absent; after the owner installed `soundfile`, both external offline CLI invocations completed without traceback and the read-only comparison emitted `saund-m1-beat-this-external-benchmark-v1`. The command used `--gpu=-1 --model final0` on both files, without altering production dependencies.

**Measured synthetic kick-attack comparison, one-to-one matching at 80 ms:**

| Generated WAV | Existing SAUND whole-track F1 | Beat This! whole-track F1 | Beat This! median matched error | Beat This! beat/downbeat counts |
|---|---:|---:|---:|---:|
| `01_opening_122bpm_Aminor.wav` | **0.1961** (25/127 measured kicks matched out of 128 beat predictions) | **0.9922** (127/127 measured kicks matched out of 129 predictions) | **4.966 ms** | 129 / 46 |
| `02_groove_124bpm_Eminor.wav` | **0.9960** (126/127 matched out of 126 beat predictions) | **0.9922** (127/127 measured kicks matched out of 129 predictions) | **4.921 ms** | 129 / 55 |

**Predeclared outgoing failure region: 9.636281–58.305306 s**, 99 measured audible kick attacks. **Beat This! matches 99/99 with F1=1.0, 0 missed and 0 unmatched model predictions, median absolute matched error 4.966 ms**, whereas the unchanged SAUND grid matches **0/99 (F1 null due to zero precision/recall)** and misses all 99, with 99 unmatched predictions. The findings directly support that the external pretrained tracker solves the identified *kick-alignment* failure on the current generated pair, while baseline SAUND was already effective on the incoming groove. Beat This! was not trained or tuned using the fixture reference timestamps during this run; the reference generator BPM locates audible kicks only for **post-hoc** scoring.

**Do not overinterpret:** 80 ms is an **offline evaluation tolerance**, not a passed 25 ms planner gate. The median reported timing error is calculated over *matched* beats only, and the two extra beat predictions on each whole track remain unmatched; the current report cannot certify every individual beat within 25 ms. Beat This! produced **46 and 55** beat-number-1 predictions (downbeats) on the respective recordings, which is unusually numerous relative to an assumed four-beat pulse cycle; **true downbeat/bar correctness has not been measured**, and we must not invent meter or count correctness from kick labels. Neither tempo-synchronous transitions nor any overlap planner/render were run. It remains only two rights-free generated demo songs, without independent real-music generalization or licensing approval. The supplied operator log contains the comparison JSON, but no separate pytest result. Flags confirm `correct_phase_certified=false`, `ground_truth_downbeat_accuracy_measured=false`, `beat_grid_modified=false`, `neural_policy_promoted=false`, `production_changed=false`, `sealed_ballroom_test_touched=false`, `transition_rendered=false`.

**Engineering recommendation:** Pivot the next M1 phase to an **external/pretrained Beat This! challenger evaluation**, not another local transient threshold sweep. First verify *predicted downbeat labels/spacing and confidence* with an independent, fixture-specific bar/downbeat reference where appropriate and independent rights-cleared nontraining audio; keep comparisons shadow-only without replacing `TrackAnalysis.beats_seconds`. Check phase-error distribution, per-beat worst case, drift across segments and explicitly whether a proposed model grid passes the unchanged production planner 25 ms gate **without force**, but do not render or authorize transitions based on this benchmark. Review upstream model-weight provenance/licensing before any distribution. Draft PR #28 stays unmerged and the frozen neural policy remains unchanged.

## M1 next checkpoint: Beat This! synthetic 4/4 downbeat labels and strict timing tails

Following the owner's external **Beat This! final0** benchmark (99/99 matched measured attacks in the outgoing 9.636281–58.305306 s failure region, and 127/127 measured kick attacks matched on each whole-track clip at an 80 ms **research-only** tolerance), a key remaining uncertainty is correct **downbeat/bar phase** and **worst-case beat residuals** rather than mean match quality. The model produced **46** and **55** predicted downbeat labels on the opening and groove demo respectively. Kick labels alone cannot determine whether those downbeat predictions are musically correct.

The repository generator `src/djai/demo.py::generate_demo_track` explicitly computes `beat_seconds = 60/bpm`, `bar_seconds = beat_seconds * 4`, writes a kick for every one of **four beats per bar**, and begins chord accompaniment at each bar start when enabled. `generate_demo_set(..., bars=32)` generates the original 122 BPM and 124 BPM demo files. Therefore this particular rights-free fixture provides an explicit **synthetic reference of 128 beats and 32 4/4 bar starts per track**. This is valid *post-hoc evaluation* data; **never provide the generator BPM, reference beats, or bar start positions to Beat This! inference or to any production beat identity decision**.

Added `scripts/m1-beat-this-downbeat-timing-audit.py` and `tests/test_m1_beat_this_downbeat_timing_audit.py`. The script reads **only** the already-generated external `.beats` files plus the WAV duration; it does not install or run Beat This!, download model checkpoints, or modify either model's predictions. It uses the strict TSV parser from the previous external benchmark. It compares Beat This! predicted **beat-number-1 downbeats** directly to the generator's known 32 bar starts and predicted beats to known 128 pulse times using **one-to-one matching at 80 ms**. It also reports **median, 95th percentile and maximum matched timing errors**, the count of reference events without a match within **25 ms**, bar-label index intervals (how often successive predicted downbeats are separated by four beat predictions), fixed **eight-second** regional beat scores, and separate scores in the previously identified failing outgoing interval.

**Critical distinction:** The 25 ms value in this script is a **post-hoc beat/downbeat timestamp residual diagnostic**, and the 80 ms matching tolerance is only the research score. Neither is interchangeable with, or evidence of passing, the existing **continuous 25 ms transition-planner phase-integrity hard gate**. A good median or even all timestamps within 25 ms would not authorize rendering a transition.

Mac commands:

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD

python -m pytest -q \
  tests/test_m1_beat_this_downbeat_timing_audit.py \
  tests/test_m1_beat_this_benchmark.py

python scripts/m1-beat-this-downbeat-timing-audit.py \
  --audio-root "$M1_AUDIO_ROOT" \
  --predictions-root /tmp/saund-m1-beat-this-predictions-01 \
  --output /tmp/saund-m1-beat-this-downbeat-timing-01.json
```

The above script does not need the Beat This! experimental Python venv because its `.beats` output files already exist. It uses the existing SAUND evaluator's NumPy/soundfile dependencies. **Mac execution has not yet been observed**; record actual results only after the user runs it. The two known generated tracks are narrow controls and do not measure generalization to independent rights-cleared tracks or commercial recordings. Regardless of outcome, PR #28 remains draft, model policy frozen, sealed Ballroom test untouched, source beat grids unchanged, and **no unsafe transition may render**.

## Owner-run Beat This! downbeat/timing audit: all 256 synthetic beats within 25 ms, excessive bar-start labels

The owner executed `scripts/m1-beat-this-downbeat-timing-audit.py` against the saved **Beat This! final0 external predictions** and both `generate_demo_set` copyright-free WAVs; the report schema is `saund-m1-external-beat-this-bar-and-timing-audit-v1`. This source generator explicitly defines **32 bars of 4/4 = 128 pulses plus 32 bar-start references per track**. The generator truth was used **after** external inference strictly for scoring, not as model input. The report confirms `beat_this_rerun=false`, `model_download=false`, `production_changed=false`, `beat_grid_modified=false`, `planner_25ms_gate_tested=false`, and `transition_rendered=false`.

| Measured fixture result | Opening 122 BPM | Groove 124 BPM |
|---|---:|---:|
| Reference beats | 128 | 128 |
| Beat This predicted beats | 129 | 129 |
| One-to-one matched beats (80 ms) | **128/128** | **128/128** |
| 80 ms beat F1 | **0.9961** | **0.9961** |
| Beat median absolute residual | 9.508 ms | 9.032 ms |
| Beat p95 absolute residual | 18.689 ms | 18.065 ms |
| Beat maximum absolute residual | **20.984 ms** | **20.000 ms** |
| Measured reference beats without a match within 25 ms | **0** | **0** |
| Extra beat predictions | 1 (62.94 s near end) | 1 (61.92 s near end) |
| Reference bar starts | 32 | 32 |
| Predicted downbeat labels | **46** | **55** |
| Matched reference bar starts at 80 ms | **32/32** | **32/32** |
| Extra downbeat labels | **14** | **23** |
| Downbeat F1 at 80 ms | **0.8205** | **0.7356** |
| Non-four-beat intervals between predicted label-1 positions | **26** | **40** |
| Median predicted downbeat-to-downbeat spacing | **1.48 s** | **0.96 s** |

In the **predeclared outgoing 9.636281–58.305306 s failure region**, the report records **99/99 generated beat references matched**, **no beat residual above 25 ms**, maximum matched error **20.984 ms**, median **9.836 ms**, beat F1 **1.0**. The predicted downbeats in this same region match **25/25** reference bar starts, but include **12 extra** predicted bar-start labels (downbeat F1 **0.8065**). The result therefore shows an exceptionally strong solution to the previously diagnosed *beat phase* error on the current synthetic demo pair while exposing an independent *bar-phase/labeling* defect. It is not credible to call the raw downbeat predictions production-ready.

**Important distinction:** The count `matched_outside_25ms=0` applies to one-to-one matched beat timestamps against the **ideal synthetic generator schedule**, and all 128 generated pulse references on each track have such a match. This is **not** a check of SAUND's continuous planner phase-integrity envelope (which includes relationship between two tracks, playback rate/warp and overlap behavior), and it must not be reported as an unforced planner pass. The two whole-track extra beat predictions are still unmatched. The 80 ms benchmark tolerance has never been promoted into the 25 ms planner threshold. Results are exclusively on two known synthetic tracks, and no real-song or sealed Ballroom validation has run. The supplied log includes the JSON execution but no visible separate pytest output.

**Next bounded investigation:** Determine whether the surplus beat-number-1 labels arise in **raw Beat This downbeat probabilities**, the model's downbeat postprocessor or the exported `.beats` beat-number conversion; measure segment/time locations of extra downbeats and their relationship to 4/4 bar structure without assuming 4/4 for arbitrary future audio. Do this with a **read-only offline comparison** that reuses the already cached checkpoint; preserve original `.beats` and do not rewrite/normalize away surplus downbeats just to make synthetic F1 perfect. If needed, benchmark documented Beat This postprocessing modes before considering an explicit conservative bar-phase consistency check. Do not promote Beat This, modify production TrackAnalysis, touch sealed tests, replace the hard **25 ms** phase gate, or render an unsafe transition. PR #28 remains draft.

## M1 Beat This! v1.1.0 origin trace: locate surplus downbeat labels before any correction

The owner-run generated-demo timing audit found **256/256 beats within 25 ms of known synthetic pulse positions** and 37 extra `beat-number=1` labels (14 of 46 on opening; 23 of 55 on groove), despite recovering all 32 known bar starts on each generated song. We must distinguish the **raw downbeat head**, the **minimal peak decoder**, and the **TSV beat-number writer** instead of attempting a hardcoded four-beat suppression.

**Upstream source inspected at the actual installed release tag v1.1.0:**
- `beat_this/model/postprocessor.py`: default `minimal` route pools beats and downbeat **logits independently over 7 frames (±3 at 50 fps)**, takes positive local maxima, merges neighboring peaks within one frame, and **snaps downbeat peaks to the nearest detected beat**, removing identical duplicates. No hardcoded 4/4 meter is imposed. Alternative `--dbn` uses madmom's DBN with `beats_per_bar=[3,4]`, which is **not** the baseline we ran. Source: https://github.com/CPJKU/beat_this/blob/v1.1.0/beat_this/model/postprocessor.py .
- `beat_this/utils.py`: `save_beat_tsv` calls `infer_beat_numbers` to assign number `1` to each postprocessed downbeat; it does not decide downbeat probabilities itself. Source: https://github.com/CPJKU/beat_this/blob/v1.1.0/beat_this/utils.py .
- `beat_this/cli.py`: `--activations` saves the model's **raw, unsigmoided 2×frames logits** to the same output stem with `.npy`, alongside a freshly generated `.beats` file. Source: https://github.com/CPJKU/beat_this/blob/v1.1.0/beat_this/cli.py .

Added `scripts/m1-beat-this-downbeat-origin-trace.py` and `tests/test_m1_beat_this_downbeat_origin_trace.py`. The tracer **does not invoke inference or import PyTorch**: it loads activation-run `.npy`, strictly parses the new `.beats` and the untouched original `.beats`, independently recreates the upstream 50 fps minimal logit peak detection and beat snapping, and compares both decoded events against the new TSV and the original TSV. It uses the generator's known **4/4 bar start times only after inference/decoding to classify extra labels**. No threshold search, post-hoc bar repair, or production change.

**Mac operator steps** (reuse existing separate `beat_this` venv with soundfile installed, cached final0 checkpoint, same WAVs; never overwrite the original `...predictions-01` files):

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_beat_this_downbeat_origin_trace.py

mkdir -p /tmp/saund-m1-beat-this-activations-01
/tmp/saund-m1-beat-this-venv/bin/beat_this \
  "$M1_AUDIO_ROOT/01_opening_122bpm_Aminor.wav" \
  -o /tmp/saund-m1-beat-this-activations-01/01_opening_122bpm_Aminor.beats \
  --gpu=-1 --model final0 --activations
/tmp/saund-m1-beat-this-venv/bin/beat_this \
  "$M1_AUDIO_ROOT/02_groove_124bpm_Eminor.wav" \
  -o /tmp/saund-m1-beat-this-activations-01/02_groove_124bpm_Eminor.beats \
  --gpu=-1 --model final0 --activations

python scripts/m1-beat-this-downbeat-origin-trace.py \
  --original-predictions-root /tmp/saund-m1-beat-this-predictions-01 \
  --activation-predictions-root /tmp/saund-m1-beat-this-activations-01 \
  --output /tmp/saund-m1-beat-this-downbeat-origin-trace-01.json
```

**Interpretation:** If activation-rerun predictions agree with original `.beats`, inference is reproducible. If independent minimal re-decoding agrees with activation-run TSV for beat and downbeat events, the **TSV label export is not fabricating the surplus downbeats**: the extra labels emerge from independent downbeat logit peaks and the permissive minimal decoder. However this does **not** tell us whether raw logits were musically wrong or a stricter *general* decoder can safely suppress spuriously high logits. If either comparison disagrees, investigate package version, output options, 50 fps assumptions, and ordering before any interpretation; do not silently fix the scorer. Do not use the generator 4/4 truth to choose a production decoder, do not modify reference beats, and do not run the sealed test or unblocked rendering. Draft PR #28 and 25 ms continuous phase gate remain unchanged.

## Owner-run Beat This! downbeat-origin trace: source isolated to minimal downbeat peak decoding

On the Mac, all **7 tests passed**. The owner re-ran `beat_this==1.1.0` with `--gpu=-1 --model final0 --activations` for both existing generated 4/4 demo WAVs, saving new .npy logits and .beats exports separately from the original output. The independent read-only `m1-beat-this-downbeat-origin-trace.py` completed successfully.

| Result | Opening 122 BPM | Groove 124 BPM |
|---|---:|---:|
| Raw logit frames, 50 fps | 3148 | 3097 |
| Positive downbeat-logit frames | 59 | 61 |
| Independent minimal decoded beat peaks | 129 | 129 |
| Independent minimal decoded downbeat peaks before snapping | 46 | 55 |
| Decoded downbeats after snapping | 46 | 55 |
| Original vs rerun TSV beat/downbeat sequences | Exact (0 ms difference) | Exact (0 ms difference) |
| Independent minimal decoded beats/downbeats vs rerun TSV | Exact (0 ms difference) | Exact (0 ms difference) |
| Reference 4/4 bar starts (posthoc scoring only) | 32 | 32 |
| Extra predicted downbeat labels | **14** | **23** |

The extra downbeats arise **by the model-logit-plus-default-minimal-peak-decoding stage**, not by the `.beats` exporter: independent decoding reconstructs all 101 combined downbeats exactly, and positive local downbeat peaks number 46/55 *before* they are snapped to beats. This does not distinguish inherently ambiguous raw model scores from overly permissive minimal decoding. No generator reference entered inference or decoding; it was used only to score known bars afterward. The two-track synthetic result does not establish real-music performance.

**Next independent test:** benchmark Beat This! v1.1.0's upstream `--dbn` mode (madmom Dynamic Bayesian Network constrained to candidate 3- or 4-beat bars) on both WAVs with **new separate output paths**, and compare it to the original minimal decoding using existing posthoc 4/4 downbeat/timing auditor. Do not use generator truth to configure or force the decoder; do not silently tune thresholds or rewrite the original TSVs. Python 3.12 may require compatibility work for madmom; if so record the blocker and avoid production dependency changes. No model promotion, sealed test, altered 25 ms planner gate or transition render; PR #28 remains draft.

## M1 fast-track controlled beat-grid preview: parallel acceptance gates (do not auto-promote)

**Objective:** Replace sequential bespoke rhythm-phase research with a short, reversible **Beat This! final0 beat-grid candidate** evaluation. This is a **production-candidate preparation plan**, not release authorization. No edits to the 25 ms continuous phase-integrity envelope, no force-render, no sealed Ballroom test. Keep draft PR #28 unmerged.

**Already demonstrated on the two known generated demo WAVs:** external Beat This! final0 matches all 99 previously displaced outgoing kicks, all 256 generated pulse timestamps within 25 ms (maximum matched residuals 20.984 / 20.000 ms) and all 64 reference bar-start locations; however downbeat outputs have 37 extra labels total (14 + 23). The independent origin trace reproducibly locates those labels in the logit plus minimal downbeat peak decoding, not TSV export. These are **two related synthetic fixtures only**.

**Existing integration discovered:** `src/djai/intelligence/neural_ears.py::predict_beats` already wraps `File2Beats` under `DJAI_NEURAL_BEATS` and `model_registry` authority, and `src/djai/analysis.py` only replaces grid when `NeuralBeatResult.used` is true. `DJAI_NEURAL_BEATS=off` disables this path. This is a potentially reversible integration; **its default checkpoint is `small0`, whereas all positive new evidence is for `final0`**. The default model registry advisory manifest is named small0. Do not treat final0 benchmark evidence as proof that existing small0 production/advisory policy is correct. Do not simply flip `authority=production`. Existing `downbeat_spacing_integrity` is structural (median downbeat interval relative to beat period), not musical validation. `DJAI_NEURAL_DOWNBEAT_INTEGRITY=1` is a trial-only guard, not a substitute for a release test.

**Additional tool:** `scripts/m1-beat-this-independent-regression.py`, with `tests/test_m1_beat_this_independent_regression.py`, compares *already-saved* Beat This `.beats` and optional SAUND `.beats` against independent human/authorized beat and optional downbeat reference timestamp files. The manifest explicitly allows `split=validation` only; `train`, `test` and `sealed` are rejected. It never runs a tracker, downloads a model or edits an output grid. Annotation timestamps are used solely after inference. Reports per-track beats/downbeats, 70ms research F1, unmatched events, per-track matched error maximum, and count of independent references not matched within 25ms (diagnostic only). Rights clearance and annotation quality need operator verification: the tool cannot certify them.

**Operator minimal usage:**
```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q tests/test_m1_beat_this_independent_regression.py
```
Create a manifest at `/tmp/saund-m1-independent/manifest.json` (user-supplied *licensed* WAV and independent annotations; do not fabricate ground truth):
```json
{
  "tracks": [
    {
      "id": "independent-track-01",
      "split": "validation",
      "audio": "audio/track-01.wav",
      "reference_beats": "labels/track-01-beats.txt",
      "reference_downbeats": "labels/track-01-downbeats.txt",
      "beat_this_predictions": "pred/track-01.beats",
      "saund_predictions": "pred/track-01-saund.beats"
    }
  ]
}
```
Prediction generation is **separate** and must run using the pinned **Beat This! final0** checkpoint with no access to reference-label files; export baseline SAUND beats as a TSV with number 1 for actual SAUND downbeat times only, not inferred labels. Do not use generated fixture timing as truth for independent music. Then:
```bash
python scripts/m1-beat-this-independent-regression.py \
  --manifest /tmp/saund-m1-independent/manifest.json \
  --output /tmp/saund-m1-independent-regression-01.json
```
Use new output paths each run. No music/annotation corpus has been supplied in this conversation, so **independent audio validation remains UNRUN**. Use a **predeclared representative** validation set including strong and weak kicks, offbeats, syncopation, varied electronic genres, breakdowns, intros, changes of tempo and meter; do not set a magic success threshold after reading outcomes.

**Parallel release checklist:**
1. **Independent regression (BLOCKED pending real annotated WAVs):** beat timing F1 and per-track regressions against current SAUND, downbeat precision/recall, outliers, tempo drift, and representative hard cases. No sealed data.
2. **Shadow/off/rollback (NOT YET VERIFIED):** exercise `DJAI_NEURAL_BEATS=off` with cached same fixture and compare baseline; test a separately configured experimental final0 checkpoint with model registry unchanged; prove hard fallback on missing checkpoint, absent PyTorch, corrupt result, model timeout/error, untrusted or contradictory downbeats. No silent use of extra bar-start labels.
3. **Deployment/provenance (NOT YET APPROVED):** pin `beat-this==1.1.0`, actual `final0` checkpoint SHA256, CPU speed and peak memory on intended host, compatible `torch`, `torchaudio`, `soundfile` WAV fallback (Mac experiment previously needed soundfile, already installed in separate environment), cache location, offline behavior and explicit network policy; separately review model-weight provenance and training-data rights. The separate Mac benchmark venv is **not** a deployable package or rights signoff.
4. **End-to-end planner (BLOCKED pending candidate wiring/measurements):** unforced planner experiment with final0 candidate beat grids and safe/downbeat-confidence handling, **hard 25ms continuous phase-integrity check unchanged** and comparisons to existing 111/111 blocked profiles; no unsafe render or default production selection. Do not infer an unforced pass from 25ms *posthoc beat-event* error.
5. **Go/no-go (NOT APPROVED):** only after explicit evidence for the above, determine whether a limited, opt-in **beat-analysis-only** preview is justified; keep full autonomous transitions gated separately. Every failure should preserve the current deterministic behavior and be observable. Rollback must work without editing checkpoints or downgrading the planner guard.

**Speed principle:** execute independent validation, rollback/deployment verification and DBN downbeat research as parallel lanes. Do not withhold positive beat-grid findings for yet another bespoke transient detector; conversely do not claim readiness from two synthetic demos alone.

## M1 fast-track existing neural adapter rollback contract — added regression coverage

After the owner confirmed **8/8 independent-regression runner unit tests passed**, a targeted regression suite `tests/test_m1_beat_this_fast_track_rollback.py` was added to validate the **existing** `src/djai/intelligence/neural_ears.py` controls through mocked inference only. This suite tests: `DJAI_NEURAL_BEATS=off` never constructs a model; an explicit missing checkpoint yields unused/unavailable evidence; external inference failure yields unused/unavailable evidence; zero registry authority cannot select a candidate even with strong evidence; the opt-in downbeat-integrity guard accepts regular four-beat spacing but rejects every-two-beat downbeat predictions; and lowering threshold cannot bypass zero authority. **No checkpoint/model run** is performed, nor is the model registry changed. Unit tests have been committed, but the operator must run them on Mac and capture output.

```bash
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q \
  tests/test_m1_beat_this_fast_track_rollback.py \
  tests/test_m1_beat_this_independent_regression.py \
  tests/test_m1_neural_downbeat_integrity.py
```

**Release limitation:** These unit tests prove isolated adapter off/fallback behavior, **not** end-to-end analysis/cache/transition-planner rollback, deployed checkpoint integrity, model timeout protection or correct beat/downbeat selection on independent recordings. The adapter's `DJAI_BEAT_THIS_MODEL` defaults to `small0`; the validated external CLI used `final0`, so a release candidate must explicitly pin the tested checkpoint. The model registry's `advisory` weight does not constitute a final production signoff. Do not modify `DJAI_NEURAL_BEATS`, `DJAI_MODEL_REGISTRY` or neural model authority in a deployed environment based only on unit tests. No 25 ms planner safety limit changes and no unsafe render.

## Owner Mac execution: all 19 M1 fast-track adapter/regression tests passed

On the user's Mac checkout refreshed at commit `8cc5f7a`, the following command completed with **19 passing pytest tests, zero reported failures**:

```bash
python -m pytest -q \
  tests/test_m1_beat_this_fast_track_rollback.py \
  tests/test_m1_beat_this_independent_regression.py \
  tests/test_m1_neural_downbeat_integrity.py
```

The grouping is **7 adapter flag/fallback/rollback tests**, **8 independent regression-runner tests**, and **4 structural downbeat-integrity tests**. This verifies mocked failure paths, off-switch behavior, zero-authority constraints, the research runner's validation-only split handling and structural downbeat checks. It **does not** establish end-to-end deployed rollback, actual `final0` adapter behavior on independent tracks, independent corpus performance, commercial license approval, host performance, or an unforced continuous 25ms planner gate pass. Keep draft PR #28 unmerged and production settings unchanged.

**Fastest next meaningful evidence:** prepare one fixed, independently annotated, rights-cleared validation set and run `scripts/m1-beat-this-independent-regression.py` with the same **final0** checkpoint tested on the generated demos; in parallel check SHA256/runtime/commercial rights and test the full integration's off-switch in a nonproduction run. Do not substitute synthesized BPM/kick references for independent annotations or invoke the sealed Ballroom test.
