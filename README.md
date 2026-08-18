# SAUND.AI

**Local-first autonomous music-performance engine for planning, rehearsing, and rendering DJ sets.**

SAUND.AI analyzes tracks, builds a musical representation of the crate, plans track order and transitions, renders candidate transitions for evaluation, and produces a continuous set.

The project combines deterministic audio execution with optional learned musical models. Learned models operate only within the authority granted by the model registry and cannot bypass hard technical failures.

> The `VERSION` file is the authoritative product version.

---

## What SAUND.AI does

The implemented pipeline is roughly:

```text
audio tracks
    ↓
track analysis
    ↓
canonical musical timeline
    ↓
set state + planning
    ↓
complete DJ-move candidates
    ↓
transition programs
    ↓
rendered candidate evaluation
    ↓
accept / retry / fallback
    ↓
deterministic final render
```

SAUND can work from ordinary full-track audio. When stem separation is enabled, it can also prepare vocals, drums, bass, and other components with Demucs and use those sources when planning supported transition techniques.

---

## Interfaces

The frontend currently provides four user-facing areas.

### Create

Create is the simplified autonomous workflow.

Choose music and a direction, then SAUND prepares the set using the configured planning, analysis, rehearsal, and rendering pipeline.

### Studio

Studio exposes more of the planning process.

It supports inspecting the current route, comparing alternative planning branches, previewing transitions, and reviewing transition and set information.

### Live

After a set has been prepared, the Live player can play the prepared performance and expose current playback information.

Live currently follows the prepared plan. It is not presented as a crowd-reactive autonomous replanning system.

### Lab

The local Training Lab provides research and learning workflows including:

* blind A/B transition comparison
* optional reason tags
* runtime candidate evidence
* dataset snapshots
* shadow taste-model training
* bootstrap dataset/model status
* held-out bootstrap metrics where available
* model and rights status

Training a model does not automatically give it control over live planning.

---

# Design principles

## Local first

The default Docker application runs on the local machine.

Music analysis, planning, rendering, model inference, ratings, and local training state can remain inside the local workspace.

Some explicit operations can use the network, including downloading pretrained models or acquiring configured external datasets.

---

## Deterministic execution

Committed transitions are represented by `TransitionProgram` objects and executed by the performance engine.

Final set rendering re-measures the exact transition samples before they are written into the completed set.

Hard technical failures can prevent a transition from being committed.

---

## Learned models have bounded authority

SAUND distinguishes between:

* deterministic analysis and execution
* pretrained sensory models
* learned DJ-specific models
* runtime model authority

The model registry supports authority states such as:

```text
shadow
advisory
production
```

A shadow model can produce predictions without changing the final decision.

Advisory models can contribute only within configured bounds and confidence requirements.

DJ-specific neural heads in the default registry start in shadow mode with zero decision weight.

---

## Foundation models are sensors

Pretrained music models are treated as evidence sources rather than automatic DJ taste.

The normal `brain` Docker target includes runtime dependencies for:

* PyTorch
* torchaudio
* transformers
* Beat This!
* Demucs

Model weights are not silently bootstrapped during normal startup.

---

# Quick start

## Requirements

* Docker Desktop
* Docker Compose

Create your local environment:

```bash
cp .env.example .env
```

Make the helper scripts executable:

```bash
chmod +x scripts/*.sh
```

Build the project:

```bash
./scripts/bootstrap.sh
```

`bootstrap.sh` builds the backend, test, CLI, and frontend images and runs the project's test workflow.

Check model state:

```bash
./scripts/saund.sh models-status
```

Explicitly cache the default pretrained rhythm model:

```bash
./scripts/saund.sh models-bootstrap
```

Start the application:

```bash
./scripts/start.sh
```

Open:

```text
SAUND.AI     http://localhost:3000
API docs     http://localhost:8000/docs
API health   http://localhost:8000/api/ready
```

Stop it with:

```bash
./scripts/stop.sh
```

`stop.sh` uses normal `docker compose down` and does not request volume deletion.

---

# Model setup

## Beat This!

SAUND can use Beat This! as neural beat/downbeat evidence.

The default configuration does not silently download its checkpoint during analysis.

Bootstrap it explicitly:

```bash
./scripts/saund.sh models-bootstrap
```

Inspect its state with:

```bash
./scripts/saund.sh models-status
```

Neural rhythm evidence must pass the configured confidence and authority gates before it replaces the deterministic rhythm result.

---

## MERT

MERT support is optional.

Cache it explicitly with:

```bash
./scripts/saund.sh models-bootstrap --mert
```

When a configured foundation embedding backend is unavailable, the project contains a deterministic statistical embedding fallback.

---

## DJ-specific models

The codebase contains trainable components for tasks including:

* transition preference
* transition family
* transition duration
* stem choreography
* rendered-transition preference
* next-track prediction
* state-conditioned taste learning

The existence of these components does **not** mean validated production checkpoints are bundled with the source tree.

The default neural registry keeps the DJ-specific heads in shadow mode until separately trained, evaluated, governed, and promoted.

---

# Bootstrap learned priors

SAUND also contains a bootstrap-learning pipeline for research datasets.

Bootstrap models can provide bounded priors for areas such as:

* cue selection
* transition duration
* next-track adjacency
* sequence generalization

In local development, configured bootstrap models may operate as bounded advisory signals.

Production configuration guards those research priors separately and can force them back to shadow operation unless explicitly approved.

Bootstrap models do not bypass rendered-audio validation or hard failures.

---

# Set planning

SAUND's planner works with more than track order alone.

A complete DJ move can include information such as:

* next track
* exit position
* entry position
* transition duration
* tempo relationship
* beat-phase adjustment
* transition technique
* stem strategy
* structural operations
* projected post-transition state

Planning can evaluate multiple moves ahead using bounded search.

The implementation remains constrained by the available analysis evidence and supported execution techniques.

---

# Rendered transition evaluation

Important candidates can be rendered before final commitment.

The rendered result can be evaluated using technical and musical measurements including areas such as:

* technical quality
* rhythm
* harmony
* vocals
* bass interaction
* naturalness
* set context
* transition structure

Learned rendered-preference models may contribute only if their configured authority permits it.

Hard failures remain feasibility failures regardless of learned preference.

---

# AI Headphones

The project includes an `AI Headphones` decision stage for the authoritative complete-move planner.

When enabled, it renders a bounded group of current serious candidates before committing the current move.

The implementation can expand the number of renders when configured decision conditions justify it.

Future search remains cheaper than the current commit-boundary rehearsal.

AI Headphones is therefore a bounded rendered-candidate evaluation stage, not a claim that SAUND has human hearing.

---

# Crate Memory

SAUND includes persistent local `Crate Memory` backed by SQLite.

It can store aggregate experience for tracks and directional track pairs, including information related to:

* exit bars
* entry bars
* transition length
* tempo relationships
* loop locations
* transition techniques
* stem strategies
* known failure reasons

The planner consumes this information as a bounded adjustment.

Crate Memory does not have authority over hard safety checks or rendered feasibility.

---

# Automatic stems

With the normal `brain` Docker target, Demucs support is installed.

A full track can be separated into:

```text
full master
    ├── vocals
    ├── drums
    ├── bass
    └── other
```

SAUND can reuse cached separation results during later planning.

Stem availability does not guarantee that a stem-based transition will be selected. Stem and layer strategies remain candidate execution choices.

Supplied performance layers are also supported.

---

# Useful CLI commands

Both `saund` and the historical `djai` command invoke the same Python CLI.

The repository helper is:

```bash
./scripts/saund.sh
```

## Analyze a track

```bash
./scripts/saund.sh analyze audio/track.wav
```

## Plan a transition

```bash
./scripts/saund.sh plan audio/a.wav audio/b.wav
```

## Render a transition

```bash
./scripts/saund.sh transition \
  audio/a.wav \
  audio/b.wav \
  -o output/pair.wav
```

## Build a continuous set

```bash
./scripts/saund.sh set audio -o output/full-set.wav
```

## Inspect models

```bash
./scripts/saund.sh models-status
```

## Create a blind listening session

```bash
./scripts/saund.sh listening-session \
  audio/a.wav \
  audio/b.wav \
  --candidates 6
```

## Train the transition preference ranker

```bash
./scripts/saund.sh brain-train
```

## Evaluate the transition preference ranker

```bash
./scripts/saund.sh brain-benchmark
```

---

# Training Lab

With the local Docker configuration, the Training Lab is available at:

```text
http://localhost:3000/lab
```

Useful commands include:

```bash
./scripts/saund.sh training-lab-status
```

Freeze an immutable training snapshot:

```bash
./scripts/saund.sh training-snapshot --name first-train
```

Train the taste-learning shadow heads:

```bash
./scripts/saund.sh training-taste-train
```

Inspect their status:

```bash
./scripts/saund.sh training-taste-status
```

Training Lab snapshots are designed to store structured model-ready evidence rather than copies of raw audio.

Production-training rights are tracked separately from local research readiness.

---

# Dataset Builder

The repository contains tooling for external research-data ingestion and bootstrap training.

Inspect configured sources:

```bash
./scripts/saund.sh dataset-sources
```

Inspect builder state:

```bash
./scripts/saund.sh dataset-builder-status
```

Import a supported source:

```bash
./scripts/saund.sh dataset-import <source> <path>
```

Freeze a bootstrap corpus:

```bash
./scripts/saund.sh dataset-build --name bootstrap-v1
```

Train research bootstrap models:

```bash
./scripts/saund.sh dataset-train-bootstrap
```

Production use of external training data is subject to provenance, privacy, license, and training-rights review.

---

# Optional CUE-DETR support

CUE-DETR integration exists as an optional research/reference environment.

It is not part of the normal `brain` Docker target and is disabled by default.

The project provides a separate setup path for it.

Do not treat an optional cue model as authoritative DJ decision logic without separate validation and rights review.

---

# Native Python development

The Python package requires:

```text
Python >=3.11,<3.14
```

Set up the development environment with:

```bash
./scripts/setup-native.sh
```

Activate it:

```bash
source .venv/bin/activate
```

Then:

```bash
./.venv/bin/saund --version
```

The Python package remains named `djai-engine`, and the `djai` CLI/environment namespace is retained for backward compatibility.

---

# Project structure

```text
src/djai/
├── brain/           planning, DJ state, search and memory
├── intelligence/    model registry, learned ranking and representations
├── judge/           rendered technical and musical evaluation
├── performance/     transition programs and deterministic execution
├── training/        dataset, provenance and training workflows
└── evaluation/      evaluation and Arena tooling

frontend/            Next.js application
scripts/             build, development and release tooling
tests/               backend/product tests
config/              model and runtime configuration
datasets/            dataset metadata and local training workspace
docs/                technical documentation
ratings/             local learning/evidence state
```

---

# Development checks

Run the project test workflow:

```bash
./scripts/test.sh
```

Check frontend source:

```bash
./scripts/check-frontend.sh
```

Validate Compose configuration:

```bash
docker compose config
```

Build the main services:

```bash
docker compose build api frontend
```

Start them:

```bash
docker compose up -d api frontend
```

Check status:

```bash
docker compose ps
```

Follow logs:

```bash
docker compose logs -f api frontend
```

---

# Release checks

The repository includes a maintainer release gate:

```bash
./scripts/release-gate.sh
```

It checks, among other things:

* synchronized version metadata
* Python compilation
* frontend source checks
* production dataset provenance
* privacy and model governance
* clean frontend dependency installation
* dependency audit
* TypeScript
* production frontend build
* Playwright browser tests
* Python tests
* release benchmark requirements
* Docker Compose configuration
* RC certification when applicable

The release gate also requires explicit confirmation of distribution rights for the supplied hero intro media.

Do not set that confirmation merely to make the release gate pass.

---

# Current scope and limitations

SAUND.AI contains substantial autonomous planning and audio-rendering infrastructure, but the repository intentionally does not claim capabilities that the current model state has not earned.

In particular:

* DJ-specific neural heads are shadow/zero-authority by default in the shipped model registry.
* A trained model is not automatically a production-authorized model.
* Foundation models provide sensory evidence; they are not automatically DJ taste.
* Live playback follows a prepared performance rather than performing crowd-reactive live replanning.
* CUE-DETR is optional research functionality, not the main DJ brain.
* Transition execution is limited to the techniques and operations implemented by the deterministic engine.
* External datasets, pretrained models, and derived weights require their own rights and license review.
* Local research readiness is not the same as public/commercial production approval.

---

# Private local state

Do not publish local runtime, user, dataset, or learned state.

Examples include:

```text
.env
accounts/
ratings/
.djai-cache/
datasets/downloads/
datasets/external/
datasets/bootstrap-corpora/
datasets/training-lab/
config/bootstrap-models/
config/training-lab-models/
config/neural/
audio/
output/
```

Use `.env.example` only for non-secret configuration examples.

---

# Documentation

Detailed technical documentation is available under [`docs/`](docs/).

Useful starting points include:

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
* [`docs/AUTOMATIC.md`](docs/AUTOMATIC.md)
* [`docs/LEARNING.md`](docs/LEARNING.md)
* [`docs/MEASUREMENT.md`](docs/MEASUREMENT.md)
* [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)
* [`docs/STEMS.md`](docs/STEMS.md)
* [`docs/TRAINING_LAB.md`](docs/TRAINING_LAB.md)
* [`docs/SECURITY.md`](docs/SECURITY.md)
* [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)

Historical release notes belong in `CHANGELOG.md`, Git releases, or archived development documentation rather than the main README.

---

# License

Project source code is licensed according to [`LICENSE`](LICENSE).

External datasets, audio, media, pretrained checkpoints, and model weights retain their own licenses and rights requirements.

Review those rights separately before redistribution, public deployment, or commercial model training.
