# SAUND.AI M0 — product baseline and capability inventory

Status: **IN PROGRESS — source audit complete; first local synthetic render completed 2026-11-16; full runtime freeze pending**  
Roadmap: M0 → M1 rhythm → M2 transitions → M3 rendered evaluation → M4 set storytelling → M5 learned DJ intelligence → M6 live reliability.  
Source baseline: `main` at `1aef9a8d8fbfaac90e2b65d4f9e70120618785bd`, version `2.0.0-rc.1`.  
This file documents **source-supported functionality, not demonstrated real-world accuracy**.

## 1. Source-supported capability inventory

| ID | User outcome | Evidence in repository | Source maturity | M0 verification |
| --- | --- | --- | --- | --- |
| CAP-01 | Analyze audio: tempo, beats, downbeats, key, sections, loudness, candidate cues | `src/djai/analysis.py`; `./scripts/saund.sh analyze` | Implemented, correctness unverified | Analyze a synthetic track; record status/errors |
| CAP-02 | Plan a complete DJ set from audio | `src/djai/set_planner.py`, `src/djai/brain/`; `./scripts/saund.sh set` | Implemented, musical quality unverified | Render and preserve a deterministic demo |
| CAP-03 | Select transition parameters and alternatives | `src/djai/planner.py`, complete-move and performance components | Implemented, constrained technique set | Record transition candidates and feasibility |
| CAP-04 | Render transitions and a continuous set | `src/djai/renderer.py`, `src/djai/set_renderer.py` | Implemented; not independently auditioned here | Confirm output audio file is produced |
| CAP-05 | Rehearse and evaluate rendered candidates | `src/djai/judge/`, `src/djai/intelligence/`; README AI Headphones | Implemented bounded scoring, not proof of expert taste | Record evaluation report and fallback use |
| CAP-06 | Use Create/Studio/Live/Lab frontend workflows | README frontend workflow; `frontend/` | Implemented according to repository; runtime not checked | Confirm routes load in local app |
| CAP-07 | Execute a prepared set with bounded future replanning | `src/djai/live.py`, `src/djai/performance/`; README Live | Implemented; real hardware/latency not certified by M0 | Review local readiness; do not start live playback for M0 |
| CAP-08 | Optional vocals/drums/bass/other separation | `src/djai/stems.py`, Demucs extra | Optional, checkpoint/runtime-dependent | Report available vs unavailable, no mandatory download |
| CAP-09 | Track-pair memory, blind listening and training workflows | README Crate Memory, Training Lab; `src/djai/training/` | Implemented infrastructure; learned taste unvalidated | Record availability only |
| CAP-10 | Optional neural beat/downbeat evidence | `src/djai/intelligence/neural_ears.py`, model registry | Experimental evidence; adoption safety under research | Report checkpoint and model authority, never enable automatically |
| CAP-11 | Optional music embeddings (MERT/MusicFM) | `src/djai/intelligence/model_registry.py` | Advisory in default registry; installation unknown | Record actual backend status |
| CAP-12 | Optional neural cue detection (CUE-DETR) | `src/djai/intelligence/cue_detr.py`, README | Research integration, disabled by default | Not required for M0 |
| CAP-13 | Learned DJ heads (next track, duration, family, stem choreography, rendered judge) | `src/djai/intelligence/model_registry.py` | **Shadow by default**: zero authority until trained and governed | Do not assert trained weights are present |
| CAP-14 | Release checking and regression suite | `scripts/release-gate.sh`, `scripts/test.sh` | Infrastructure implemented, latest local pass unknown | Capture test/status evidence; do not claim certification |

### Important truth boundaries

- `Implemented` means repository implementation exists, **not** tested or production-ready on the user's machine.
- Optional dependencies, model checkpoints and local registry overrides may differ from the default repo. A registered model is not necessarily installed, trained, running, or approved.
- The default registry lists Beat This! and the music encoder as **advisory** and five DJ decision heads as **shadow**. Actual authority must be read from the **local** `models-status` output.
- Live is prepared-performance playback with bounded FUTURE replanning, **not** unrestricted crowd-reactive autonomy.
- A full set render proves execution, not that the mix sounds good or works across all genres.
- Draft PR #27 is separate, observational rhythm research. Its experimental gating does **not** establish changes in `main` or authorize a neural-grid replacement.

## 2. Known-issues register

| ID | Priority | Verified observation / gap | Milestone |
| --- | --- | --- | --- |
| R-01 | P0 | On 17 approximate-BPM tracks, the fallback-aware neural full-grid gate reproduced two gate passes rejected by the provisional research screen. No **observed live** adoption or independent beat/downbeat correctness. Results are isolated in draft PR #27. | M1 |
| R-02 | P0 | No independently verified musical beat/downbeat labels on the 17-track personal sample; no accuracy or false-rejection claim justified. | M1 |
| R-03 | P0 | The 2026-11-16 demo completed and WAV checksum was captured; exact runtime config, artifact report and repeatability still need review. | M0 |
| R-04 | P0 | Registry `models-status` reports seven models without registered checkpoint files, but Beat This! weights downloaded during the demo. Local cache, overrides and application readiness remain unconfirmed. | M0 |
| R-05 | P1 | No frozen, rights-cleared multi-genre audio manifest / transition matrix yet. | M0 |
| R-06 | P1 | Real-music rendered set quality and cross-genre performance have not been measured by the current research. | M2–M4 |
| R-07 | P1 | Default DJ decision heads have zero authority; trained production models have not been demonstrated. | M5 |
| R-08 | P1 | End-to-end live playback under real-time constraints and recovery scenarios not certified by this source audit. | M6 |

## 3. Source-of-truth and isolation

- Code snapshot for M0: `main@1aef9a8d8fbfaac90e2b65d4f9e70120618785bd`.
- Version source: `VERSION` → `2.0.0-rc.1`.
- Source policy: **do not change any engine, cue, grid, track, live playback or model authority while capturing M0**.
- Test data: synthetic demo is repeatable from repo; real audio is a separate private, rights-cleared evaluation and **must never be committed**.
- Any fresh `main` changes require a **new baseline revision**; do not silently replace the pinned commit.
- M0 evidence must record success/failure rather than assuming it.

## 4. M0 acceptance criteria

- [x] Capability inventory and truthful status distinctions recorded.
- [x] Known-issues register with destinations M1–M6 recorded.
- [x] Source commit, version and default model policy frozen **on paper**.
- [x] Reproducible local smoke commands specified in `docs/roadmap/M0-BASELINE-RUNBOOK.md`.
- [x] Fixed benchmark/scenario design specified in `docs/roadmap/M0-MULTIGENRE-SUITE.md`.
- [x] User-provided `models-status` recorded (2026-11-16); full runtime environment/checkpoint cache still pending.
- [x] Synthetic five-track end-to-end render completed, WAV existence and SHA-256 captured; JSON report and WAV metadata inspection still pending.
- [ ] App readiness, frontend status and failures recorded, where runtime is available.
- [ ] Rights-cleared real-music reference manifest frozen, without copying private audio.
- [ ] Baseline evidence artifact reviewed and M0 **explicitly accepted**.

**Gate: M0 is not completed** until all required local reproduction and evidence conditions pass. The next task is to inspect the existing render report/metadata and freeze remaining local evidence; no new model training or audio accuracy benchmark is necessary for M0. Evidence: `M0-LOCAL-EVIDENCE-2026-11-16.md`.
