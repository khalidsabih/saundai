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
