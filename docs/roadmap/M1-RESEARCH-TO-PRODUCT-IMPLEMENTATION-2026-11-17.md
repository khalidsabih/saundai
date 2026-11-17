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
