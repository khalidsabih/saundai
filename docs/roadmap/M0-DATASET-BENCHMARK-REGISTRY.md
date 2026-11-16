# M0 Dataset & Benchmark Registry — first implementation

This is a **metadata-first evidence ledger**, not a DJ decision model. It is separate from `scripts/release-benchmark.py` and existing SAUND `dataset-import` / `dataset-build` capabilities. None of these commands downloads audio, learns parameters, enables models, or modifies Live/Play/Beat grid authority.

## Included

- `benchmarks/dataset-registry/catalog.json`: proposed sources (private crate, FMA, GiantSteps Tempo, Ballroom, MTG-Jamendo, unverified DJ transition metadata) with purpose and **explicit rights uncertainty**, plus named metric definitions for M0–M6.
- `scripts/dataset-benchmark-registry.py`: stdlib-only CLI using local SQLite for recorded measurements, source commit, implementation ID, manifest SHA-256 fingerprint, split, dataset ID, and stated safety/rights/human evidence. Outputs remain private in `benchmarks/dataset-registry-results.sqlite3` (already covered by `benchmarks/*.sqlite3` in `.gitignore`).
- `tests/test_dataset_benchmark_registry.py`: repeatability and comparability checks; reject missing rights attestations, cross-manifest comparisons, unsafe test candidates and unreviewed human-evidence gates.

**Do not record synthetic musical metrics as real music scores.** The M0 5-track synthetic render's tech verification was 4/4 and `production_safe=false` in its plan. Do not register that run as a production champion.

## Local command flow

Fetch the M0 branch in a separate worktree (do not switch or alter the currently edited workspace):

```sh
cd "$DJAI_WORKSPACE"
git fetch origin docs/m0-baseline-capability-audit-20261116
git worktree add --detach /tmp/saund-m0-dataset-registry FETCH_HEAD
cd /tmp/saund-m0-dataset-registry
python scripts/dataset-benchmark-registry.py catalog
python scripts/dataset-benchmark-registry.py status
python -m pytest -q tests/test_dataset_benchmark_registry.py
```

Later, **only after rights review and a frozen dataset/split exist**, record one externally measured metric:

```sh
python scripts/dataset-benchmark-registry.py record \
  --benchmark m1_beat_f1 --dataset ballroom --split development \
  --manifest /path/to/private/frozen-ballroom-dev-manifest.json \
  --source-commit "$(git rev-parse HEAD)" \
  --implementation deterministic-baseline --value 0.80 \
  --safety-passed --rights-confirmed
```

`0.80` above is **example syntax, not a measured result**. Do not run as factual evidence until a real evaluator computed the metric and rights were checked. The script records only a SHA-256 of the manifest, not the private path/audio contents.

Comparison is allowed only when benchmark ID, dataset ID, split and manifest fingerprint match:

```sh
python scripts/dataset-benchmark-registry.py compare --incumbent 1 --challenger 2
```

Champion promotion is an **explicit, separate ledger action**, and requires an untouched `test` split, attested rights and safety, a strict improvement over an existing champion and a human-evidence reference if the benchmark defines one. An initial champion requires `--initial`. This DOES NOT alter `model-registry.json`, neural authority or the product release champion. An absolute metric threshold in the catalog is a **proposal**, not a universally applicable statistical certification.

```sh
python scripts/dataset-benchmark-registry.py promote --run-id 1 --initial
```

## Guardrails and current limits

- Catalog entries are **proposals, not downloaded data or approved licenses**. Each specific clip and use case must be reviewed; `--rights-confirmed` is a human attestation and is not a legal review.
- Numeric scores are externally supplied. The tool does not execute audio evaluation, verify metric provenance, prove blind judging or establish holdout independence from reading the manifest; **those are still M1+ integration requirements**.
- The existing SAUND source builders are in `src/djai/training/dataset_builder.py`; the release benchmark is in `src/djai/evaluation/release_benchmark.py`; blind Arena is in `src/djai/evaluation/arena.py`. They remain untouched. Later integration should connect their structured outputs instead of duplicating evaluation logic.
- The current first implementation must pass tests in the user's Python runtime before being marked validated. No tests were executed on the user's local audio or system as part of authoring.
- Freeze one approved development and held-out collection per milestone. Keep artist/recording-group disjoint, no repeated test-driven tuning, and no automatic promotions.

## Acceptance for M0 registry tool

- [x] Registry catalog and acceptance metric definitions committed in isolated draft branch.
- [x] Local SQLite ledger, fingerprinted records, explicit comparison/champion steps implemented.
- [x] Focused tests checked in.
- [ ] Execute focused tests on clean checkout.
- [ ] Demonstrate `catalog` and `status` commands, then connect an actual measured run on approved data.
- [ ] Rights-cleared splits and dataset acquisition separately approved; M0 source and synthetic baseline gates still tracked in `SAUND-MILESTONE-TRACKER.md`.

**Status: CODE PRESENT / LOCAL EXECUTION NOT YET VERIFIED.**
