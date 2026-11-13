# Phase 1 baseline and regression triage

Reference: full-suite terminal log shared on 2026-11-13, prior to the
complete-move generator fix; this is **not** a current post-fix suite run.
Canonical branch at start of modernization: `phase-13-15-time-live-venue-hardening`.

## Confirmed after the 1.7 fix

- `tests/test_complete_djmove_generator_1_7.py`: **11 passing**
  locally on Python 3.12, as reported by the operator.
- Warnings: deprecated `aifc`, `audioop`, `sunau`; fake WAV
  fixtures trigger librosa's audioread fallback. They do not fail these tests.

## Full-suite failures requiring separate review

| Area | Tests / observations from old log | Current disposition |
| --- | --- | --- |
| Complete-move search | Generator signatures and duplicate widening tiers | Fixed and operator-verified (11 tests pass) |
| V5 recovery | `test_v5_transactions_carry_recovery_receipts_without_changing_primary` failed because nested `tmp_path / "observed"` did not exist | Fixture now creates parent directory; rerun needed |
| Full-set benchmarking | Two tests construct `SimpleNamespace` without `metadata`, while production expects `TransitionPlan.metadata` | Fixture now provides metadata; rerun needed |
| Frontend migration | Multiple literal UI-string assertions no longer match Create/Studio/Account components | Inspect intent, replace brittle string checks with behavioral/API tests |
| RC1 transport | Multiple tests pin old `frontend/lib/api.ts` SHA-256 despite intentional API evolution | Verify upload/auth transport compatibility, then replace hash pins |
| Transition naturalness | Rescues return `echo_out_drop_in` or `structural_reset` instead of old `filter_build`/`compact`; one duration constraint fails | Audio-quality and feasibility regression investigation required; do not update expectations blindly |
| Headphones | Two tests report `assert False is True` | Inspect rendered-candidate evidence and critic authority |
| Live tempo release | Live stream duration and finished render duration diverged | High priority: compare source/rate/handoff clocks and rendered output |
| API timeline schema | Runtime `1.1.1` vs test `1.0.0` | Establish migration/version authority; update only after confirming API compatibility |
| Phase 16 | Planning ready but rehearsal not ready with `render=False` | Clarify phase16_passed semantics and test expectations |
| Executive lifecycle | V1 and V2 phase histories differ from historical expected sequences | Compare intentional V5 lifecycle additions against audit requirements |
| Autonomous performance version | Expected `4.0`, got `4.2` | Check canonical wire schema/version contracts |

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

Do not substitute a passing focused suite for an all-green release gate.
Capture the post-fix full-suite failure list before Phase 2 changes.
