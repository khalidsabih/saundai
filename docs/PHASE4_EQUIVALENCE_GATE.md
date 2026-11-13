# Phase 4 equivalence and performance gate

Status: **pending operator measurements**. Code extraction and the supported
Python full-suite gate are confirmed green; that is necessary but not
sufficient to certify before/after output equivalence.

The pre-extraction reference is Git commit
`cb731ea035130590bfa39dddbc460085ebf723d7`, the parent of the first
Phase 4 graph-cache extraction (`390323ad`). Compare against the current
`phase-13-15-time-live-venue-hardening` branch. Do not compare two runs of
the current branch and call that a pre-refactor baseline.

## Fixed-fixture plan and render comparison

On macOS from the normal repository checkout, with the supported virtualenv
activated, create a shared absolute path containing exactly three real,
authorized WAV music files. The *same physical files at the same absolute
paths* must be used in both captures (execution signatures bind source
identity). Prefer representative tracks with compatible tempos and distinct
musical forms. Set the variables below to your actual paths:

```bash
export SAUND_MAIN="$PWD"
export SAUND_AUDIO="/absolute/path/to/three-track-fixture"
export SAUND_BASELINE="$SAUND_MAIN/../saundai-phase4-baseline"

git worktree add --detach "$SAUND_BASELINE" cb731ea035130590bfa39dddbc460085ebf723d7
mkdir -p "$SAUND_MAIN/output/phase4-evidence"

# Invoke the same comparison script from the current checkout in both runs.
# PYTHONPATH points to the code tree being evaluated; do not change the
# audio directory, environment flags, or Python environment between runs.
cd "$SAUND_BASELINE"
PYTHONPATH="$SAUND_BASELINE/src" python "$SAUND_MAIN/scripts/phase4_equivalence.py" capture \
  --audio-dir "$SAUND_AUDIO" \
  --output "$SAUND_MAIN/output/phase4-evidence/baseline.json" --render

cd "$SAUND_MAIN"
PYTHONPATH="$SAUND_MAIN/src" python scripts/phase4_equivalence.py capture \
  --audio-dir "$SAUND_AUDIO" \
  --output "output/phase4-evidence/current.json" --render

python scripts/phase4_equivalence.py compare \
  output/phase4-evidence/baseline.json \
  output/phase4-evidence/current.json
```

The comparison requires identical input-file SHA-256 digests, track order,
transition execution signatures, program fingerprints, bounded transition
metadata, alternative signatures, and decoded rendered PCM hashes. Nonzero
exit signals a mismatch. It prints separate elapsed times for analysis,
planning, and rendering; times are *observations*, not deterministic equality
requirements. Do not weaken exact execution identity or hard safety to make
this comparison pass.

Do not substitute arbitrary files or generated test WAV placeholders for
representative music when assessing musical quality. This gate is an
engineering reproducibility check, not proof of DJ quality or a production
release certificate.

## Planner performance evidence

For a larger controlled crate, run the existing benchmark in each checkout
against the same audio directory and resource configuration:

```bash
PYTHONPATH="$SAUND_BASELINE/src" python "$SAUND_MAIN/scripts/benchmark.py" "$SAUND_AUDIO" \
  --analysis-workers 1 --graph-workers 1 --compare-serial --warm-cache \
  --output "$SAUND_MAIN/output/phase4-evidence/graph-baseline.json"

PYTHONPATH="$SAUND_MAIN/src" python "$SAUND_MAIN/scripts/benchmark.py" "$SAUND_AUDIO" \
  --analysis-workers 1 --graph-workers 1 --compare-serial --warm-cache \
  --output "$SAUND_MAIN/output/phase4-evidence/graph-current.json"
```

This benchmark primarily covers analysis and graph compilation. For the
longer 13-track cold/cached planner gate, reuse the Phase 16 controlled crate
and record crate identity, environment, hardware, graph/analysis cache state,
timing by stage, memory/resources, fallback rate and safety outcome. Keep
cold and warm runs separate and run from the appropriate working directory:
`--warm-cache` does not itself prove both checkouts used identical cache
contents. Do not compare historical Phase 16 output as if it were an exact
before/after refactor A/B run.

## Sign-off checklist

- [x] Supported Python suite green after modular planner/API extraction
- [x] All HTTP endpoint registrations moved to feature routers without
      changing handler implementations
- [ ] Fixed-input old/new execution signatures and program fingerprints match
- [ ] Old/new decoded PCM render hashes match
- [ ] Comparable old/new timings and resource metrics recorded and reviewed
- [ ] Phase 4 exit decision written with paths to the actual evidence

If a mismatch appears, preserve the two reports and investigate the
source/program/render differences instead of updating expected signatures
or weakening the safety gate.
