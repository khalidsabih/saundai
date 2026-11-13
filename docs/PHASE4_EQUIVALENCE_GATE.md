# Phase 4 equivalence and performance gate

Status: **controlled three-track execution and decoded-PCM equivalence PASSED with quality rehearsal enabled; broader performance and release sign-off pending**. The supported Python full-suite gate was previously reported green. Without rehearsal, both branches correctly reject the same 16.1 dB RMS handoff; with the existing measured quality loop enabled, both branches render successfully and produce identical decoded PCM. This fixture-level result does not certify production DJ quality.

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

# Freeze the trained scoring model too; otherwise the two worktrees can
# resolve different config/scoring-model.json files even for identical audio.
export DJAI_CONFIG_DIR="$SAUND_MAIN/config"
export SAUND_CACHE="$SAUND_MAIN/output/phase4-evidence/shared-analysis"

# Invoke the same comparison script from the current checkout in both runs.
# PYTHONPATH points to the code tree being evaluated; do not change the
# audio directory, environment flags, or Python environment between runs.
cd "$SAUND_BASELINE"
PYTHONPATH="$SAUND_BASELINE/src" python "$SAUND_MAIN/scripts/phase4_equivalence.py" capture \
  --audio-dir "$SAUND_AUDIO" --analysis-cache-dir "$SAUND_CACHE" \
  --output "$SAUND_MAIN/output/phase4-evidence/baseline.json" --rehearse --render

cd "$SAUND_MAIN"
PYTHONPATH="$SAUND_MAIN/src" python scripts/phase4_equivalence.py capture \
  --audio-dir "$SAUND_AUDIO" --analysis-cache-dir "$SAUND_CACHE" \
  --output "output/phase4-evidence/current.json" --rehearse --render

python scripts/phase4_equivalence.py compare \
  output/phase4-evidence/baseline.json \
  output/phase4-evidence/current.json
```

The comparison requires identical input-file SHA-256 digests, full analyzed-track
digests, scoring-model fingerprints, track order,
transition execution signatures, program fingerprints, bounded transition
metadata, alternative signatures, and decoded rendered PCM hashes. A failed render still writes a JSON report, but returns nonzero; the
comparison distinguishes a matching safety rejection from a successful PCM
match. Any render rejection keeps the audio-equivalence gate open. Nonzero
exit signals a mismatch or an unresolved safety rejection. It prints separate elapsed times for analysis,
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
- [x] Fixed-input old/new execution signatures and program fingerprints match
      (shared scoring model; both manual route [1,0,2] and automatic routing)
- [x] Old/new decoded PCM render hashes match (rehearsed, safety-clean fixture)
- [ ] Comparable old/new timings and resource metrics recorded and reviewed
- [x] Interim Phase 4 decision documented with paths to the operator evidence
- [x] Fixed-fixture safety-clean render decision recorded
- [ ] Final Phase 4 exit after comparable planner latency/resource benchmark review

If a mismatch appears, preserve the two reports and investigate the
source/program/render differences instead of updating expected signatures
or weakening the safety gate.

## Controlled three-track operator results (November 13, 2026)

The operator used the detached pre-refactor worktree and the current branch
against the same symlinked fixture under `output/phase4-fixture`, the same
`output/phase4-evidence/shared-analysis` cache, and **the same absolute
`DJAI_CONFIG_DIR="$SAUND_MAIN/config"`**. Previously divergent routes and
transition fingerprints were traced to a model-environment mismatch:
baseline had no scoring model, whereas current used fingerprint
`d98eaf8cd7af928d` with learned adjustments. Once both runs used the same
model, the differences vanished. Do not attribute those differences to the
planner extraction.

- Fixed-order evidence: `output/phase4-evidence/baseline-controlled.json`,
  `output/phase4-evidence/current-controlled.json`. Audio hashes, analysis
  digests, model fingerprint, order and transitions **MATCH**.
- Automatic-order evidence: `output/phase4-evidence/baseline-auto.json`,
  `output/phase4-evidence/current-auto.json`. The same fields **MATCH**.
- Render rejection evidence: `output/phase4-evidence/baseline-render.json`,
  `output/phase4-evidence/current-render.json`. Both renderers reject the
  first `echo_out_drop_in` transition at outgoing source exit `121.835102`
  seconds with the same 16.1 dB RMS handoff failure; neither published
  a completed render. Both measured the same RMS levels:
  pre-tail `-16.47702` dB, transition head `-16.01121` dB,
  transition tail `-28.3371` dB, post-head `-12.22134` dB.
  The beginning jump is `0.47` dB and the end jump is `16.12` dB.
- Planning timings are observational only: most recent render capture
  measured baseline `5.491` s and current `4.949` s. They are **not**
  a controlled performance benchmark or an optimization sign-off.

### Rehearsed render equivalence — PASS (November 13, 2026)

The operator used the same source audio, analyzed-track cache, scoring-model
configuration, Python environment, and `--rehearse --render` flags on the
pre-extraction worktree and current branch. The comparison reported:

```text
schema: MATCH
audio_fingerprints: MATCH
analysis_digests: MATCH
scoring_model_fingerprint: MATCH
order: MATCH
transitions: MATCH
rendered_pcm: MATCH
render_status: baseline=passed current=passed
Execution identity equivalence passed.
```

- Evidence: `output/phase4-evidence/baseline-rehearsed.json` and
  `output/phase4-evidence/current-rehearsed.json`; corresponding WAV paths
  use the same stems with `.wav` extensions.
- Operator repeated the current rehearsed render successfully twice.
- Single observed run timings: analysis `0.180` s for both; planning
  baseline `13.937` s vs current `5.614` s; rendering baseline
  `2.449` s vs current `3.111` s. These are **not** controlled performance
  benchmarks. In particular, do not infer a planner speedup or render slowdown
  from one pair of measurements.
- The `--rehearse` flag opts in to the existing measured-candidate quality
  loop, not a changed safety threshold or post-signature render alteration.
  The original non-rehearsed safety rejection remains documented above.

**Decision:** functional Phase 4 extraction equivalence is **verified** for
this controlled three-track fixture, including identical decoded PCM after
passing the original render safety checks. Keep the broader planner
latency/resource benchmark and Phase 6 musical-quality release gate open.
Review which rehearsed candidate was selected and ensure larger real-music
crates meet the same safety expectations before any production-quality claim.
