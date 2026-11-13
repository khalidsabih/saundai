# Modernization regression triage

Reference: full-suite terminal log shared on 2026-11-13, prior to the
complete-move generator fix; this is **not** a current post-fix suite run.
Canonical branch at start of modernization: `phase-13-15-time-live-venue-hardening`.

## Confirmed after the 1.7 fix

- `tests/test_complete_djmove_generator_1_7.py`: **11 passing**
  locally on Python 3.12, as reported by the operator.
- Warnings: deprecated `aifc`, `audioop`, `sunau`; fake WAV
  fixtures trigger librosa's audioread fallback. They do not fail these tests.

## Latest operator verification (2026-11-13)

### Phase 2 full-suite baseline

The supported Python test suite is now **green on macOS/Python 3.12**.
The operator ran `python -m pytest -q --tb=short` after the Phase 2
contract/runtime fixes and reported **no failures**. The run contains only
known non-failing warnings plus skipped tests.

The fixes that closed the old failure inventory include:

- frontend/version/hash assertions aligned to authoritative runtime contracts;
- Executive lifecycle expectations aligned with prepared recovery;
- Live tempo-release iteration aligned with audible prefetch runway;
- graph worker accounting and sequence-evidence provenance repaired;
- AI Headphones successful rehearsals no longer lose authority because of
  optional selector-result telemetry;
- automatic rescue expectations aligned with the bounded overlap-safety policy;
- transition duration assertions compare rendered time against source time using
  the committed render rate.

No hard-audio-safety threshold was relaxed to obtain the green suite.

### Phase 1 targeted baseline

The following **26 targeted tests passed** on macOS/Python 3.12 after
Phase 1 fixture repairs:

- `tests/test_complete_djmove_generator_1_7.py`
- `tests/test_phases_13_15_reliability.py`
- `tests/test_dj_executive_v5_prepared_recovery.py`
- `tests/test_full_set_benchmark_1_0.py`

Warnings in these tests are non-failing librosa/audioread and Python 3.13
deprecations. The later Phase 2 full-suite run above supersedes the earlier
"not rerun" limitation.

### Phase 16: actual 13-track real-music evidence

Operator ran `phase16-validate-real audio/uploads --journey balanced
--neural-retrieval --graph-workers 4` with
`-o output/phase16-regression-planner-fix` and without `--render`.
Results:

| Measure | Observed |
| --- | ---: |
| Planning complete | 13 tracks / 12 transitions |
| Planning elapsed | 124.5 seconds |
| Analysis cache | 13 of 13 |
| Pair graph cache | 130 hits / 0 misses |
| Best sparse partial | 13 / 13 |
| Broad graph materialized | No |
| Quality pass rate | 12 of 12 |
| Hard failures | 0 |
| Fallback transitions | **11 of 12 (91.67%)** |
| Structural reset rate | 25% |
| Emergency rate | **50%** |
| Story fit | 0.33939 |
| Full set score | 0.659648 |
| Full-set production safe | **False** |
| Phase 16 passed | **False** |

Important distinction: `planning_ready=True` and successful measured
transition checks do not override the full-set benchmark's production-safe
gate. `render=False` did not cause the failure. The code's
`rehearsal_ready` criterion already requires `production_safe` even
without rendering. The Phase 16 synthetic unit fixture previously lacked
`production_safe`; Phase 2 fixes that test fixture rather than weakening
the release gate.

**Quality diagnosis:** the proposed set has repeated deterministic final
escapes, multiple beat-phase-lock warnings, strong-bass or vocal overlap
warnings, and a DJ-intro file in the crate. The last track in the planned
route was `DJ DROP - OPENER SHOW - English (UK) --- (DJ INTRO)`, despite
its intro purpose; investigate track roles and whether that file should
participate in the musical 13-track route at all. Do not infer audio quality
from scores alone without listening.

**Performance diagnosis:** 99.46 seconds of the 124.5-second plan was
compile/rehearsal; route search was 23.39 seconds and graph work was
1.10 seconds, fully cached. Tune/trace compile and fallback selection
before spending effort on graph-worker parallelism.

**Next investigation:** inspect plan report and per-edge execution
metadata, then compare a controlled 12-track music-only crate with the
same cached analysis and a true 13-track crate containing purpose-tagged
intro. Track emergency count, fallback count, same-track order constraints,
end-of-set energy, score and listening evidence; never skip an edge
without validation or relax hard safety thresholds.

## Full-suite failures requiring separate review

| Area | Tests / observations from old log | Current disposition |
| --- | --- | --- |
| Complete-move search | Generator signatures and duplicate widening tiers | Fixed and operator-verified (11 tests pass) |
| V5 recovery | `test_v5_transactions_carry_recovery_receipts_without_changing_primary` failed because nested `tmp_path / "observed"` did not exist | Fixture fixed; targeted suite now passes |
| Full-set benchmarking | Two tests construct `SimpleNamespace` without `metadata`, while production expects `TransitionPlan.metadata` | Fixture fixed; targeted suite now passes |
| Frontend migration | Multiple literal UI-string assertions no longer matched Create/Studio/Account components | Fixed; frontend contract suite green |
| RC1 transport | Multiple tests pinned old frontend transport SHA-256 values | Fixed; behavioral transport/config assertions replace hash pins |
| Transition naturalness | Historical tests expected filtered overlap even when bounded rescue safety selects structural reset | Resolved; unsafe overlap remains prohibited and tests express current safety contract |
| Headphones | Successful mocked rehearsals lost authority because optional selector telemetry was assumed present | Fixed; acceptance/safety thresholds unchanged |
| Live tempo release | Live iteration manufactured hold audio before the real audible prefetch deadline | Fixed; live/render parity test green |
| API timeline schema | Runtime `1.1.1` vs historical `1.0.0` assertion | Fixed; test binds to canonical schema version |
| Phase 16 | Planning ready but rehearsal not ready with `render=False` | Contract clarified; real crate remains production unsafe and requires later quality work |
| Executive lifecycle | V1/V2 histories omitted prepared recovery states | Fixed; tests now express current lifecycle |
| Autonomous performance version | Historical `4.0` assertion vs runtime `4.2` | Fixed; test binds to canonical program version |

## Next verification commands

Run in the repo root with the Python virtualenv activated:

```bash
python -m pytest -q \
  tests/test_complete_djmove_generator_1_7.py \
  tests/test_phases_13_15_reliability.py \
  tests/test_dj_executive_v5_prepared_recovery.py \
  tests/test_full_set_benchmark_1_0.py

python -m pytest -q --tb=short
```

Phase 2 now has an operator-verified green full-suite baseline. Preserve this
as the regression gate while Phase 3 documentation and Phase 4 refactors proceed.
