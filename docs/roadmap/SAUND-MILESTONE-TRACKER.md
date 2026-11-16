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
| 06. M0 exit review | BLOCKED by 03–05 | Explicit milestone verdict and deviations |

## Isolation rule

**Do not merge draft PR #27 or apply its research threshold changes during M0.** Source inspection, baseline documents and evidence gathering do not change any production rhythm gate, frontend/backend, cue, grid, audio file, playback or live model authority.

**Current gate verdict: M0 NOT YET PASSED.** The user supplied a complete 5-track synthetic render, a WAV checksum and `models-status` on 2026-11-16. Model cache downloaded `small0.ckpt` during the run, so network/cache effects must be tracked. Report audit completed: 4/4 exact technical checks passed, but two risky emergency transitions and `production_safe=false` remain recorded. Runtime environment and private rights-cleared scenario manifest remain open.
