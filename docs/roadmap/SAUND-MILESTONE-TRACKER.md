# SAUND.AI milestone tracker — M0–M6

**Planning baseline:** `main@1aef9a8d8fbfaac90e2b65d4f9e70120618785bd`, `VERSION 2.0.0-rc.1`. Update only with dated, reproducible evidence. Proposed acceptance thresholds require owner approval before being treated as release gates.

| Milestone | Status | Output | Required decision |
|---|---|---|---|
| M0 Source and operational baseline | **IN PROGRESS** | Capability inventory, pinned revision, model status, end-to-end synthetic render evidence, fixed genre scenarios | Owner accepts complete run evidence and private rights-reviewed scenario inventory |
| M1 Music rhythm/structure | RESEARCH ONLY — PR #27 draft | Independently validated beat/downbeat/tempo/meter/cue evidence and safe fallback | Target proposal: beat F1 >= .90 @70 ms; downbeat F1 >= .85 @100 ms on untouched holdout; no known unsafe grid promotion |
| M2 Multi-genre transitions | NOT STARTED | Supported techniques, compatibility/decline rules, safe execution | All candidates obey hard safety; fixed scenarios have explicit pass/fail or safe decline |
| M3 Rendered transition quality | NOT STARTED | Rendered judge and blind comparisons versus frozen baseline | Proposed >=60% blinded preference among non-ties, with enough judgments to quantify uncertainty, no hard safety regression |
| M4 Set storytelling | NOT STARTED | Intent-based energy arc, phrasing, ending, multi-genre route | Duration/ending requirements met; blinded set-quality improvement |
| M5 Learned DJ intelligence | NOT STARTED | Shadow next-track, transition-family, duration, stems, rendered preference evidence | Holdout value added; production rights and governance gates met before promotion |
| M6 Live reliability | NOT STARTED | On-device performance and recovery certification | Repeated fault/recovery and latency criteria pass without critical playback error |

## M0 tasks (execution order)

| Task | Status | Evidence |
|---|---|---|
| 00. Pin source to main immutable SHA and VERSION | DONE (repository-inspected) | `M0-CAPABILITY-BASELINE.md` |
| 01. Inventory implemented/optional/shadow capabilities and risks | DONE (source-only) | `M0-CAPABILITY-BASELINE.md` |
| 02. Freeze multi-genre scenario definitions | DONE (specification only) | `M0-MULTIGENRE-SUITE.md` |
| 03. Capture actual local models and runtime config | PARTIAL — user-supplied `models-status`; overrides/cache/environment not frozen | `M0-LOCAL-EVIDENCE-2026-11-16.md` |
| 04. Render a pinned five-track synthetic set and inspect WAV + report | PASS for synthetic render and JSON report audit; WAV container inspection still optional | `M0-LOCAL-EVIDENCE-2026-11-16.md`, `M0-RENDER-REPORT-REVIEW-2026-11-16.md` |
| 05. Identify rights-cleared scenario assets and freeze private hashes/splits | PARTIAL — 34 unique hashes, genre candidate map documented; S06/S08–S11 lack established coverage; rights and dev/test splits unassigned | `M0-GENRE-COVERAGE-2026-11-16.md`, private `/tmp/saund-m0-crate-unique.json` |
| 05B. Implement dataset and milestone benchmark registry | VERIFIED — catalog/status commands worked and 5/5 focused tests passed on user's Mac (2026-11-16) | `benchmarks/dataset-registry/catalog.json`, `scripts/dataset-benchmark-registry.py`, `tests/test_dataset_benchmark_registry.py`, `M0-DATASET-BENCHMARK-REGISTRY.md` |
| 05C. Connect M1 reference metric evaluator to registry | VERIFIED — owner's Mac ran 11/11 combined evaluator + registry focused tests (2026-11-16); real annotation import pending | `scripts/m1-rhythm-benchmark.py`, `tests/test_m1_rhythm_benchmark.py`, `M1-RHYTHM-BENCHMARK-RUNBOOK.md` |
| 05D. Import real GiantSteps Tempo and Ballroom reference formats | VERIFIED — owner's Mac passed 18/18 combined importer, evaluator and registry tests (2026-11-16); real dataset/audio acquisition pending | `scripts/m1-import-references.py`, `tests/test_m1_import_references.py`, `M1-REFERENCE-IMPORT-RUNBOOK.md` |
| 05E. Acquire and audit independently published rhythm annotations | VERIFIED — owner audited 664 GiantSteps and 698 Ballroom annotations, 22/22 focused tests passed; matching audio and rights unresolved | `scripts/m1-audit-annotations.py`, `tests/test_m1_audit_annotations.py`, `M1-REAL-REFERENCE-ACQUISITION-2026-11-16.md` |
| 05F. Identify official Ballroom audio and audit pairing/replicas without download | VERIFIED — owner's Mac passed 5/5 focused audio-readiness tests (27/27 cumulatively across M1 tool checks); matching audio and rights review pending | `scripts/m1-audio-readiness.py`, `tests/test_m1_audio_readiness.py`, `M1-AUDIO-PAIRING-PILOT.md` |
| 05G. Prepare bounded official Ballroom real-audio pilot | INSPECTED — official archive integrity passed and eight genre-diverse annotated WAV candidates identified on owner's Mac; no extraction/rights approval or scores yet | `scripts/m1-prepare-ballroom-pilot.py`, `tests/test_m1_prepare_ballroom_pilot.py`, `M1-BALLROOM-REAL-AUDIO-PILOT.md` |
| 06. M0 exit review | BLOCKED by environment/evidence signoff; dataset acquisitions are future milestone work | Explicit milestone verdict and deviations |

## Isolation rule

**Do not merge draft PR #27 or apply its research threshold changes during M0.** Source inspection, baseline documents and evidence gathering do not change any production rhythm gate, frontend/backend, cue, grid, audio file, playback or live model authority.

**Current gate verdict: M0 NOT YET PASSED.** The user supplied a complete 5-track synthetic render, a WAV checksum and `models-status` on 2026-11-16. Model cache downloaded `small0.ckpt` during the run, so network/cache effects must be tracked. Report audit completed: 4/4 exact technical checks passed, but two risky emergency transitions and `production_safe=false` remain recorded. Runtime environment and private rights-cleared scenario manifest remain open; the benchmark registry commands ran successfully and all five focused tests passed in the owner's Python environment. No dataset evaluation runs or champions have been recorded.
