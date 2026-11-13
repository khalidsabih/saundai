# Real next-track retrieval training

SAUND's retrieval model must learn **what is worth considering** without taking
authority away from deterministic transition preparation, rendered rehearsal, or
hard safety.

This lane keeps three evidence sources in one immutable snapshot:

1. **Research synthetic prior** — useful bootstrap supervision, retained as a
   separate corpus because its `state_*` / `candidate_*` feature namespace is
   not yet identical to the live runtime feature contract.
2. **Phase 16 real rehearsal evidence** — trainable now because rows are produced
   with the same `transition_feature_dict` used by live candidate retrieval.
3. **AotM-2011 human playlist sequencing** — real ordered human choices using MSD
   track identifiers. It is retained as sequencing supervision until those IDs
   are resolved to a feature/audio catalog.

No source in this lane grants production authority. The trained checkpoint is
shadow-only and is used only by the explicit `--neural-retrieval` experiment.

## AotM-2011

Use the public AotM-2011 playlist file from Brian McFee's dataset page:

- Dataset page: https://brianmcfee.net/data/aotm2011.html
- File: `aotm2011_playlists.json.gz`

Keep the downloaded source outside committed code, for example:

```bash
mkdir -p datasets/external
# place aotm2011_playlists.json.gz in datasets/external/
```

The importer reads `filtered_lists`, preserving only contiguous matched MSD
segments. It creates positive ordered pairs A -> B but does **not** treat them as
technical transition-safety labels.

## Build a real-retrieval snapshot

After at least one Phase 16 run has produced a set plan:

```bash
SYNTH_ID="$(cat datasets/research-synthetic/LATEST)"

./scripts/saund.sh real-retrieval-build \
  --synthetic-next-track "datasets/research-synthetic/$SYNTH_ID/next-track.jsonl" \
  --phase16-plan output/phase16-real-music/balanced/set-plan.json \
  --aotm-source datasets/external/aotm2011_playlists.json.gz
```

Repeat `--phase16-plan` for additional real crates/journeys. More independent
plans are substantially more valuable than repeatedly training on one crate.

The snapshot is written under:

```text
datasets/real-retrieval/<dataset-id>/
  manifest.json
  next-track-runtime.jsonl
  phase16-next-track.jsonl
  synthetic-next-track-prior.jsonl
  aotm-sequences.jsonl
```

### Phase 16 labels

Accepted professional transitions use their measured score, but fallback
transitions are deliberately discounted:

- normal professionally accepted transition: measured quality target
- structural reset: capped at 0.55
- other fallback: capped at 0.40
- true emergency/final escape: capped at 0.20
- Executive-blocked transition: 0.0 hard negative
- unselected route alternative: low-weight alternative

This makes the retriever learn route survivability instead of merely learning
that an emergency transition eventually rendered.

## Train the runtime-aligned shadow retriever

```bash
./scripts/saund.sh real-retrieval-train
```

The checkpoint is written under `config/real-retrieval-models/` and
`NEXT_TRACK_REAL_RETRIEVAL_LATEST` points to the newest model.

When present, the experimental runtime retrieval path prefers this checkpoint to
the synthetic-only MLP.

## Evaluate

```bash
./scripts/saund.sh phase16-validate-real audio/uploads \
  --journey balanced \
  --render \
  --neural-retrieval
```

Compare at minimum:

- planning seconds
- graph pairs materialized
- retrieval width actually used
- route completion
- Executive replan count
- fallback rate
- structural reset rate
- true emergency rate
- mean measured transition quality
- story fit
- full-set production-safe result

## Next data milestone

AotM should only enter the runtime MLP after an explicit MSD/audio feature
resolution layer maps its track IDs into the same canonical feature contract as
SAUND tracks. Until then, keeping AotM separate avoids misleading offline
metrics caused by incompatible feature namespaces.
