# SAUND.AI M1 — from investigation to product improvements

**Prepared:** 2026-11-17. **Scope:** M1 beat tracking, meter/downbeat analysis, neural model integration, and automatic DJ transition safety. **Branch:** `docs/m0-baseline-capability-audit-20261116` / draft PR #28. **No merge or production configuration change.**

## Executive decision

The evidence supports **using an existing pretrained Beat This! model as a serious rhythm candidate now**, with conservative selection and transparent rollback. It does **not** support disabling the continuous **25 ms phase-integrity render gate** or promoting arbitrary model downbeat labels into cue/phrase planning. The highest-value engineering improvements are robust temporal BPM estimation, a fail-closed coherent neural bar grid, a pinned local model checksum, and preserved observability of rejected neural candidates.

### Scoreboard — evidence and limitations

| Investigation | Observed result | Product conclusion |
|---|---|---|
| Original M1 synthetic mix | SAUND followed hats and recovered **0/99** kicks in the failed opening section; all **111** initial transition configurations failed the **25 ms continuous phase-integrity gate** | Serious existing rhythm failure; no auto render allowed |
| Acoustic/DSP transient diagnostics | Investigated beat-index aliasing, meter, tempo, midpoint/offbeat kicks, low-band event spacing, competing acoustic phases, 24-case phase sweep, 52-case robustness controls and hybrid global/local period anchors. Promising isolated recoveries did not robustly fix ambiguous/weak-kick cases across controls | **Do not replace production rhythm with those experimental DSP heuristics** |
| External Beat This! `final0` on two generated 4/4 demos | Failed opening kick region **99/99 recovered**, **256/256** known generated pulses matched within 25 ms; 129 predicted beats per file | Strong proof pretrained beat candidate can solve the original demonstrator failure. Event matching does **not** certify continuous render safety |
| Generated downbeat audit and raw-logit origin trace | Generated bar-start ground truth **32/32** found per track, but `final0` predicted **46** and **55** downbeats respectively (**37 surplus**). Identical TSVs and raw-logit re-decoding proved the surplus came from model-logit / permissive minimal downbeat peak extraction rather than TSV export | Hard-guard irregular bar labels, retain model evidence and abstain from replacing production bar grid |
| Existing Mac SAUND adapter | `beat_this` installed in the normal SAUND venv. Default `small0` had no bootstrapped checkpoint and `available=false`; explicit cached `final0` ran from `neural_ears.predict_beats`: `available=true`, 129 beats, 46 downbeats, confidence 0.7579, authority 0.2577, `used=true`, BPM reported 120 on nominal 122 BPM opening | **Confirmed integration exists**, but current gate could accept surplus bar starts and frame-rounded scalar tempo; repair at adapter boundary |
| Early 8-track Ballroom development evaluation of existing neural selection | Selected neural integration beat F1 0.773 vs deterministic 0.754; downbeat F1 0.532 vs 0.221; selected only 4 of 8 neural grids | Existing confidence authority gate misses many useful beat candidates |
| 80-track Ballroom development shadow comparison | Selected SAUND beat F1 **0.792782** vs raw neural **0.975817**; downbeat **0.694564** vs **0.978045**; tempo MAE **22.548** vs **2.558** BPM. 50 accepted / 30 rejected; 29/30 rejected candidates improved beat F1, 1 regressed | Strong evidence for a gated, paired candidate strategy, not unconditional promotion |
| Frozen confidence >=0.50 / <=2 BPM challenger on 80 *disjoint Ballroom validation* tracks | Beat F1 **0.850554 -> 0.952999**; downbeat **0.770713 -> 0.946314**; tempo MAE **17.738 -> 3.507** BPM; 16 additional neural grids accepted; 0 per-track beat/downbeat F1 regressions | Freeze **as a research challenger**; do not silently install this lower confidence threshold in product without cross-domain acceptance |
| Bar and cue preflight on same 80 validation tracks | Simulated four-bar cue preflight failures **22 current -> 12 challenger -> 13 raw neural**; challenger hypothetical cue start near independent downbeat on 76/80. Later audit found phase misses even when bar durations looked regular | Structural regularity and clean F1 cannot guarantee correct **musical downbeat phase** or safe actual mixed audio |
| New offline `final0` pilot on 8 Ballroom development recordings | Mean per-track beat F1 **0.7578 -> 0.9911**, downbeat **0.1914 -> 0.9912**, 0/8 paired regressions, model SHA256 `8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331` | Confirms strong offline inference and annotation agreement; **NOT an unseen-music holdout** |
| Independent music/generalization | Beat This! v1.1.0 `final0` was trained on all paper datasets other than GTZAN, and upstream `beat_this_annotations` includes Ballroom (see upstream READMEs) | Prior Ballroom scores are **training-overlap-contaminated**. Need genuinely unseen, rights-cleared, independently annotated audio |
| Feature flag and rollback | Owner ran **19/19** targeted mocked adapter / regression / downbeat tests successfully on Mac | Adapter selection/fallback basics tested; real full-analysis/render rollback and deployment readiness still unverified |

### Production-code changes in draft PR #28

**1. `src/djai/intelligence/neural_ears.py`: robust multi-beat scalar tempo.**

The original `_estimate_bpm` used the median of adjacent Beat This! timestamps quantized to 20 ms frames. The 122 BPM generated opening returned 120 BPM despite correctly located beat events. The new estimator uses median **up to 8-beat-span** intervals. It rejects nonfinite, zero or reverse time events instead of propagating invalid tempo. This is not an independent reference or BPM oracle and must be tested on variable-tempo tracks.

**2. `src/djai/intelligence/neural_ears.py`: mandatory coherent-bar veto before grid selection.**

`downbeat_bar_integrity` checks finite/strictly increasing event times, downbeats sufficiently near distinct predicted beats, and constant full **3, 4 or 6 beat-event** spacings between predicted downbeats. If this consistency check fails, `NeuralBeatResult.used` is forced false with an explicit `neural bar-grid safety veto` reason; all candidate times remain in `neural_evidence` for investigation. This is an intentionally conservative **default-on and non-bypassable veto** at the adapter selection boundary. It prevents the specific surplus-downbeat failure from replacing the whole beat/downbeat/cue grid.

**Limitations:** This check may reject valid irregular, variable-meter or truncated-bar recordings. Passing does not prove correct musical downbeat **phase**, proper key, genre or cue quality. Because the existing planner depends on a whole consistent beat/downbeat grid, the implementation abstains instead of mixing a neural beat grid with incompatible deterministic downbeats. It does **not** reduce the existing default model confidence threshold, and it does **not** promote neural outputs globally.

**3. `src/djai/intelligence/neural_ears.py`: optional explicit checkpoint hash pin.**

`DJAI_BEAT_THIS_SHA256` may be set to a 64-hex expected fingerprint. The adapter then refuses to run unless its checkpoint points to an existing, matching local file. This is a fail-closed deployment-control mechanism, not evidence of commercial rights approval. Without the variable, existing behavior remains unchanged. No model download implied.

**4. Regression test suite** `tests/test_m1_neural_bar_safety_promotion.py` covers 122 BPM frame rounding, regular 3/4/6-pulse bars, surplus downbeats not detected by the prior median-only guard, vetoed grid selection with retained candidate evidence, pinned checkpoint success/failure, and unchanged normal selection for coherent bars. The tests mock Beat This! inference; owner Mac execution is **pending**.

**5. Everything else preserved.** `analysis.py` still computes cue/bar features after the candidate grid is selected; a veto means it retains the deterministic grid **before** compiling the downstream bar/cue structures. `DJAI_NEURAL_BEATS=off` still disables external inference. `small0` is still the default: `final0` use requires an explicit, local checkpoint in a separately controlled environment. All neural registry authority values, confidence threshold, planner source, render signatures and hard **25 ms phase-integrity gate** remain unchanged. No production deployment or release is authorized by this report.

### What we evaluated but deliberately did NOT promote

- Loosening the old confidence cutoff straight from 0.70 to the research challenger 0.50. Reasons: training-dataset overlap, one exploratory regression, untouched test still sealed, and phase/cue failures even after improved F1.
- Fixing Beat This! downbeats by forcing every fourth beat without an independently identified musical bar phase. Reasons: can create a perfectly regular but musically wrong grid.
- Using synthetic known BPM or generated kick-time truth during inference. Reasons: data leakage and no real-world generalization.
- Replacing only scalar tempo without the corresponding event grid. Reasons: violates BPM-grid coupling and phase assumptions.
- Mutating beats/downbeats on a completed `TrackAnalysis` or signed `TransitionPlan`. Reasons: stale cue/bar/phrase context and invalid signed planner assumptions.
- Forcing a transition or reducing the continuous **25 ms** phase-integrity hard gate. Reasons: the original **111/111** profiles failed; event-level tolerance is not render-phase integrity.

### Immediate operator test (no production settings changed)

```sh
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q \
  tests/test_m1_neural_bar_safety_promotion.py \
  tests/test_m1_beat_this_fast_track_rollback.py \
  tests/test_m1_neural_downbeat_integrity.py \
  tests/test_neural_brain_5.py
```

For an additional read-only **actual checkpoint** smoke test of coherent-bar selection (without making `final0` the default), set `DJAI_BEAT_THIS_CHECKPOINT` and `DJAI_BEAT_THIS_SHA256` **only for one command**. The current 122 BPM synthetic demo is expected to retain the **deterministic** whole grid after the new bar veto despite `available=true`; this is a correct conservative result, not a tracker failure. Test real rights-cleared audio with true meter/downbeat labels separately.

### Release gates still required

1. Rights-cleared **non-training-overlap** independent audio validation with per-track regressions, challenging genres/tempo and meter; checkpoint identical to the pinned candidate.
2. Correct **beat-only versus bar-phase selection policy**, tested end-to-end on full analysis and downstream cues (do not silently splice inconsistent grids).
3. Mac and server **offline** deployment QA: package versions, CPU time, memory, no implicit downloads, checkpoint pinning, artifact provenance, model/weights commercial-use rights.
4. Authorization-controlled rendering on *separately licensed* non-test audio, full cue recomputation, signed `plan_transition(...,force=False)`, actual rendered 25 ms continuous phase-integrity and human audition review, with hard failures unchanged.
5. Shadow -> limited opt-in -> expanded roll-out with feature-flag rollback, audit logs, monitored failures and explicit release owner signoff.

**Recommendation:** Accept the guarded code improvements for testing, not unconditional `final0` production authority. Complete the above gates, then ship the smallest approved beat-analysis feature before full autonomous transitions.

## Owner verification — 33 focused tests passed on Mac

Owner refreshed `/tmp/saund-m1-evaluator` to commit `f8d2745` and ran:

```sh
python -m pytest -q \
  tests/test_m1_neural_bar_safety_promotion.py \
  tests/test_m1_beat_this_fast_track_rollback.py \
  tests/test_m1_neural_downbeat_integrity.py \
  tests/test_neural_brain_5.py
```

Actual terminal output: `................................. [100%]` — **33/33 passed**, no failures reported. This verifies unit-level neural bar veto, robust multi-beat BPM, checkpoint pin behavior, existing off/fallback controls and related neural tests against mocks and fixtures. It does **not** yet verify actual `final0` inference through `analyze_track()` after the changes, downstream cue reconstruction, production deployment or the 25 ms continuous planner gate.

**Next end-to-end validation:** run the entire `analyze_track` pipeline with the exact SHA256-pinned local `final0` checkpoint on the previously generated opening fixture, compare against `DJAI_NEURAL_BEATS=off`, and inspect `neural_evidence['beat_downbeat']` `available`, `used`, `reason`, beat/downbeat event counts and BPM. Expect the known irregular 46-downbeat candidate to be **rejected** for grid replacement while remaining recorded in diagnostics, with complete analysis retaining deterministic selected grid and cues. Use an ephemeral subprocess-only environment; do not export settings globally. Do not render or relax the gate.

## Owner full-analyzer smoke test: final0 available, selected grid correctly retained

Owner ran complete `djai.analysis.analyze_track` in separate subprocesses on `01_opening_122bpm_Aminor.wav`, using exact `final0` cached checkpoint SHA256 `8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331`, with downloads disabled. Results:

| Selection | Neural off | Neural on |
|---|---|---|
| Selected BPM | 123.0469 | 123.0469 |
| Selected beats | 128 | 128 |
| Selected downbeats | 32 | 32 |
| Entry candidates | 12 | 12 |
| Exit candidates | 12 | 12 |
| Neural available | false | true |
| Neural used | false | false |
| Neural estimated BPM | 0 | 120.0 |
| Neural downbeats | 0 | 46 |
| Reason | neural beats disabled | neural beat grid recorded as evidence but deterministic grid retained |

**Interpretation:** The complete analysis pipeline keeps the original selected grid and equal entry/exit candidate counts when `final0` is available but not used. This is valuable real-runtime fallback evidence, not evidence that the new bar-grid veto specifically fired: the reason did **not** include `neural bar-grid safety veto`, suggesting selection did not pass an earlier confidence/authority threshold. The model's full-analysis `confidence`, `agreement`, `authority` and threshold were not printed. The scalar model BPM remains 120.0 for this track despite the robust estimator, so the model's raw timestamps or estimator need inspection before claiming the nominal 122 BPM discrepancy is fixed.

**Next targeted action:** print confidence/agreement/authority alongside bar-integrity check from the saved candidate arrays, without changing live acceptance thresholds. Then design a separately feature-flagged, beat-only benefit path that does not combine unmatched deterministic/neural bar phases or bypass the hard 25 ms gate. No production update or render.

## Owner updated-source full-analysis diagnosis: two independent selection blockers

After explicitly setting `PYTHONPATH=/tmp/saund-m1-evaluator/src` to avoid importing the older installed checkout, the owner ran `analyze_track` with cached Beat This! `final0` on the synthetic opening. Printed diagnostic evidence:

- Neural confidence: **0.6599**; agreement with deterministic beat grid: **0.1938**.
- Model registry authority: **0.0** (below default registry minimum confidence 0.66).
- Neural grid `used=false`; reason: `neural beat grid recorded as evidence but deterministic grid retained`.
- Independent `downbeat_bar_integrity`: **false**, reason `irregular neural bar spacing or surplus downbeat labels`.
- Selected analyzer output remains 123.0469 BPM, **128** beats and **32** downbeats.

This establishes *two* distinct safeguards on this fixture: the early authority/confidence refusal and the structural bar-grid failure. Since selection already failed the earlier threshold, the final `neural bar-grid safety veto` was not the returned `reason`; the separate validator nevertheless confirms the bar defect. The upstream model was available in the preceding successful call. This is not evidence that the new neural beat timestamps are unsafe; it is evidence that the **existing planner cannot be fed the inconsistent full beat+bar candidate**. Do not lower thresholds to force selection or silently substitute downbeat phase. Next product work is an opt-in beat-only diagnostic/perception capability and a separately validated coherent bar-phase selection before making candidate beats authoritative for transition planning.

## M1 implementation: opt-in Beat This! beat-only preview in the actual SAUND analysis response

**Motivation:** The owner confirmed the original `final0` beat candidate is available in full analysis but rejected (confidence 0.6599 / registry authority 0.0) and its 46 candidate downbeats fail bar integrity. This is a real product gap: the model's accurate beat timestamps remained buried in raw neural evidence, while product output continued to show the deterministic beat grid. We must not force the rejected full beat/downbeat grid into planner context.

**Changes committed in draft PR #28:**
- New `src/djai/intelligence/beat_preview.py` provides `beat_preview_from_evidence` with finite and ordered time checks, clip duration bounds, plausible pulse spacing, a multi-beat BPM estimate and explicit abstention on invalid or unstable beats. It never emits predicted bar starts and never claims correct musical phase.
- `TrackAnalysis.to_dict()` includes a separate **`beat_only_preview`** field only when `DJAI_BEAT_ONLY_PREVIEW=1`. It exposes `available`, timestamps, BPM, confidence, refusal reason and two explicit safeguards: `used_for_planning=false`, `downbeats_authoritative=false`. Existing authoritative `beats_seconds`, `downbeats_seconds`, cue and bar structures remain **unchanged** and are never edited. Default preview flag is **off**. `to_dict(include_beats=False)` limits the new beat list to 16 with a `beat_count` for compact responses.
- Six unit tests in `tests/test_m1_beat_only_preview.py` check default-disabled behavior, real output field, compact shape, rollback via feature flag, invalid input abstention and unchanged selected beat/bar/cue fields. Tests were committed; **owner Mac execution is pending**. No tests imply planner safety.
- **Important:** This is a usable public *analysis-response* improvement, **not an automatic beat-grid replacement in DJ cue selection**. SAUND's actual planner will continue using the coherent selected grid until musical downbeat phase can be established and 25 ms continuous safety is passed. Cached analyses may have been computed without neural inference: if `neural_evidence.beat_downbeat.available` is false or missing, preview correctly abstains. When testing the actual local model, call `analyze_track` directly (no cache) so neural evidence is captured.

**Verification with current local `final0` checkpoint — no persistent environment modification:**

```sh
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q \
  tests/test_m1_beat_only_preview.py \
  tests/test_m1_neural_bar_safety_promotion.py \
  tests/test_m1_beat_this_fast_track_rollback.py

PYTHONPATH=/tmp/saund-m1-evaluator/src \
DJAI_BEAT_ONLY_PREVIEW=1 \
DJAI_NEURAL_BEATS=1 \
DJAI_BEAT_THIS_CHECKPOINT="$HOME/.cache/torch/hub/checkpoints/beat_this-final0.ckpt" \
DJAI_BEAT_THIS_SHA256=8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331 \
DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python - <<'PY'
import json
import os
from djai.analysis import analyze_track

path = os.path.join(os.environ["M1_AUDIO_ROOT"], "01_opening_122bpm_Aminor.wav")
a = analyze_track(path)
p = a.to_dict(include_beats=False)["beat_only_preview"]
print(json.dumps({
    "preview_available": p["available"],
    "preview_beat_count": p.get("beat_count", 0),
    "preview_bpm": p["bpm"],
    "preview_reason": p["reason"],
    "preview_used_for_planning": p["used_for_planning"],
    "preview_downbeats_authoritative": p["downbeats_authoritative"],
    "selected_beat_count": len(a.beats_seconds),
    "selected_downbeat_count": len(a.downbeats_seconds),
    "neural_selected": a.neural_evidence["beat_downbeat"]["used"]
}, indent=2))
PY
```

**Expected on the known opening fixture:** preview has 129 available candidate beats at the model's measured pulse tempo, while selected grid stays at 128 beats / 32 downbeats and neural whole-grid `used=false`. Verify actual output before claiming this integration has passed. `DJAI_BEAT_ONLY_PREVIEW=0` (or unset) immediately removes the new response field; it does not download a model or change any planner safety controls.

**Next genuinely musical step**: a separate experimental bar-phase resolver must establish meter and bar-origin from independent audio evidence (not a synthetic truth oracle) and run full `analyze_track`/cue/planner/renderer validation on rights-cleared music. Do not simply snap every fourth pulse to a guessed bar origin, change the model authority threshold, or weaken the continuous 25 ms render gate.

## Owner verification: beat-only preview regression fix passed

Owner fetched PR #28 branch at commit `166bc70` into `/tmp/saund-m1-evaluator` and executed:

```sh
python -m pytest -q \
  tests/test_m1_beat_only_preview.py \
  tests/test_m1_neural_bar_safety_promotion.py \
  tests/test_m1_beat_this_fast_track_rollback.py
```

Terminal result: `........................ [100%]` (**24/24 passed**, no failures). This confirms the isolated long-/short-interval regression fixes alongside previous mocked adapter safety tests in the owner's normal Python environment. The actual offline `final0` checkpoint has **not yet been run against the newly exposed `beat_only_preview` response**; do not claim end-to-end feature validation until that smoke result is observed. The authoritative planner grid and 25 ms phase gate remain unchanged.

## Owner real-model full-analysis verification — beat-only preview works

Owner fetched the branch at `2e2a9ac`, ran the full `djai.analysis.analyze_track()` path against the generated `01_opening_122bpm_Aminor.wav` using the cached locally pinned Beat This! `final0` checkpoint with downloads disabled, `PYTHONPATH=/tmp/saund-m1-evaluator/src` and `DJAI_BEAT_ONLY_PREVIEW=1`. Output:

```json
{
  "preview_available": true,
  "preview_beats": 129,
  "preview_bpm": 121.8274,
  "preview_reason": "neural pulses available for beat-only preview; bar safety not assessed",
  "preview_used_for_planning": false,
  "selected_beats": 128,
  "selected_downbeats": 32,
  "neural_selected": false
}
```

**Result:** First actual, owner-confirmed `final0` → `analyze_track` → `TrackAnalysis.to_dict(include_beats=False)` → `beat_only_preview` end-to-end feature integration **passed**. The robust wide-span preview BPM of **121.8274** is within 0.173 BPM of the generated fixture's nominal 122 BPM; this does not by itself validate multi-genre tempo accuracy. All model timestamps remain non-authoritative for planner/bar/cue computations; the original selected beat/downbeat counts remain 128/32 and neural whole-grid acceptance remains false. The separate librosa pitch-tuning warning is not an inference failure. User-facing product deployment, normal workspace update, rights approval and rendered transition safety remain outstanding. No production setting changed.

## Experimental neural bar-phase resolution for real DJ cues (opt-in only)

**Implemented (not yet executed on owner Mac):** `src/djai/intelligence/neural_bar_phase.py`, `tests/test_m1_neural_bar_phase_trial.py`, and a guarded integration path in `src/djai/analysis.py`. This is the first candidate path where a coherent **neural beat grid plus independently resolved bar grid** can influence the *actual* downstream `_aggregate_bar_features`, `_structural_boundaries`, entry/exit cues, and therefore an unforced planner. It does **not** bypass the 25 ms continuous phase-integrity gate or modify renderer/planner source.

**Opt-in:** `DJAI_NEURAL_BAR_PHASE_TRIAL=1`. The default is `0`, so no behavior change in current installs. The trial uses neural predictions only if the existing registry did *not* already select them, and only if all of the following are satisfied:

1. The existing time-signature estimator is confident (`time_signature_confidence >= 0.80`) and suggests 3, 4 or 6 beats per bar; no assumed 4/4.
2. Candidate neural beats are finite, ordered, within track bounds, cover >=75% of the audio, and have no >35% long / >30% short local pulse discontinuity.
3. The actual acoustic onset envelope has a single dominant accent phase across **all four** separated track regions, with >=22% per-region winner-to-runner-up margin. No known synthetic meter, annotated beat phase or raw permissive neural downbeat labels are used to *derive* the selected origin.
4. At least eight existing deterministic bar landmarks overlap; >=90% agree within min(70 ms, 15% of beat period) with the independently inferred acoustic-phase bar starts.
5. The resolver returns a *complete* candidate beat grid, downbeat grid and BPM. The full analyzer updates these simultaneously **before** generating bar/structure/cue features. It never patches a signed `TransitionPlan` or merges incompatible bar positions.

**Abstention:** A refusal is recorded in `analysis.neural_evidence["bar_phase_trial"]` with a plain-English reason and limited diagnostics; the original selected rhythm grid, cues and BPM are retained. Even a pass is only a conservative engineering indication, **not** certified musical downbeat identity or production safety; requiring agreement with the baseline bar detector intentionally sacrifices recall and may reject the original failing synthetic demo. There is no silent phase snapping to every fourth beat, and no 0.50 confidence promotion.

**Do not globally enable or deploy this trial** until validation on rights-cleared non-training-overlap audio, sample-level transition audits, and commercial-use review. The existing Beat This! upstream model's Ballroom training overlap still disqualifies Ballroom-only production claims.

### Owner commands: first unit and guarded real-model test

```sh
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
python -m pytest -q \
  tests/test_m1_neural_bar_phase_trial.py \
  tests/test_m1_beat_only_preview.py \
  tests/test_m1_neural_bar_safety_promotion.py \
  tests/test_m1_beat_this_fast_track_rollback.py

# No cache; observe whether the phase resolver abstains on the known demo.
PYTHONPATH=/tmp/saund-m1-evaluator/src \
DJAI_BEAT_ONLY_PREVIEW=1 \
DJAI_NEURAL_BAR_PHASE_TRIAL=1 \
DJAI_NEURAL_BEATS=1 \
DJAI_BEAT_THIS_CHECKPOINT="$HOME/.cache/torch/hub/checkpoints/beat_this-final0.ckpt" \
DJAI_BEAT_THIS_SHA256=8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331 \
DJAI_MODEL_ALLOW_DOWNLOAD=0 \
DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \
python - <<'PY'
import json, os
from djai.analysis import analyze_track

a = analyze_track(os.path.join(os.environ["M1_AUDIO_ROOT"], "01_opening_122bpm_Aminor.wav"))
p = a.neural_evidence.get("bar_phase_trial", {})
print(json.dumps({
    "phase_trial": p,
    "selected_bpm": a.bpm,
    "selected_beats": len(a.beats_seconds),
    "selected_bars": len(a.downbeats_seconds),
    "entry_candidates": len(a.entry_candidates),
    "exit_candidates": len(a.exit_candidates),
}, indent=2))
PY
```

**Interpretation:** `phase_trial.eligible=false` means a correct safety abstention. `eligible=true` means the analysis rebuilt its grid-dependent structures from the resolved candidate, **not** that a real DJ transition has been rendered or passed the 25 ms gate. Compare outputs with the same command **without** `DJAI_NEURAL_BAR_PHASE_TRIAL=1` before selecting any test music for an unforced planner/render canary. For any render, use only separately licensed pair audio and `force=False`, with existing hard integrity checks intact. This trial is not enabled by `DJAI_BEAT_ONLY_PREVIEW` alone.
