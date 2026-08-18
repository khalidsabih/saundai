# SAUND.AI release benchmark crates

SAUND 1.9 keeps benchmark **definitions and metrics** in the repository, but it does not ship music.
Benchmark audio must come from tracks you have the right to use.

Canonical scale suites:

- `small`: 7 tracks
- `standard`: 14 tracks
- `large`: 30 tracks
- `stress`: 100 tracks

Specialist suites are also named in the runtime catalog: `vocal-heavy`, `mixed-bpm`, `dnb`, `house`, `techno`, `pop-electronic`, and `difficult-bridges`.
Their exact crate size may vary, but the same crate must be used for champion/challenger comparisons.

A completed set plan can be recorded with:

```bash
python scripts/release-benchmark.py record output/<job>/set-plan.json --suite standard
python scripts/release-benchmark.py compare --suite standard --challenger latest
python scripts/release-benchmark.py promote --suite standard --run-id <id>
```

The SQLite store persists metrics only. Track paths and titles are not copied into the benchmark database.
Human blind-listening results can be attached with `--human-preference challenger|champion|tie|neither`.
