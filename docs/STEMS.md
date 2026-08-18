# Automatic performance layers and stem-aware rendering

## Product default in 4.2

The normal user uploads one full master. DJAI prepares a cached performance bundle automatically:

```text
full master (original upload)
  -> vocals.wav
  -> drums.wav
  -> bass.wav
  -> other.wav
  -> instrumental.wav = drums + bass + other
```

The full master remains the timing and final-master reference. AI-derived layers are performance material, not replacements for the original track.

## Runtime modes

```env
DJAI_AUTO_STEMS=off|background|wait
```

- `off`: deliberate stereo-only mode; no automatic separation.
- `background`: default. Upload queues separation, and `/api/sets/automatic` synchronizes before planning by reusing/waiting for cached material or preparing anything still missing.
- `wait`: retained for explicit callers that want eager preparation semantics; automatic set creation is already a synchronization point in 4.2.

Normal Docker `djai` and `api` services now default to the `stems` image target so Demucs is available without a separate product startup command.

## Source priority

DJAI chooses the cleanest known source for each performance role:

```text
supplied/original stems
  > supplied instrumental/acapella
  > cached AI separation
  > full master only
```

A complete supplied bundle suppresses unnecessary AI separation. A partial supplied bundle can be completed from cached AI material.

## Cache

The expensive Demucs stems remain keyed by the source-audio fingerprint:

```text
.djai-cache/stems/<audio-fingerprint>/
  vocals.wav
  drums.wav
  bass.wav
  other.wav
  instrumental.wav
  manifest.json
```

The manifest records the separation model/device plus integrity/quality proxies. Existing valid 4.1 stem caches can be upgraded in place by deriving `instrumental.wav` and a 4.2 manifest; the expensive separation does not need to rerun.

## Quality confidence

AI separation is not assumed perfect. DJAI records conservative integrity proxies including:

- stem reconstruction similarity against the master;
- residual energy after summing stems;
- timeline/duration consistency;
- silent/missing stem checks;
- peak/clipping risk.

These are integrity proxies, not oracle SDR ground truth. The resulting `quality_confidence` is carried into `PerformanceLayers` and the planner reduces the authority of weak layers.

## Ground-truth-like local evidence

When cached sources are available, cue-local estimates can use the isolated material:

```text
vocal_probability <- vocal stem activity
bass_activity      <- bass stem activity
```

The same sources can be used for real transition rendering.

## Layer-aware techniques

The performance engine supports both true four-stem techniques and the higher-level three-part source contract:

- EQ bass swap;
- filter build;
- double drop;
- stem vocal overlay;
- `layer_vocal_bridge`;
- `layer_instrumental_handoff`.

Every candidate still follows the normal invariant:

```text
plan -> private render -> measure/judge -> immutable execution plan -> final renderer
```

The renderer never invents a layer switch not present in the plan.

## Resource considerations

- Demucs is computationally and memory intensive.
- CPU preparation can be slow for a large library; cached results avoid repeat work.
- Separation artifacts can be audible and are why quality confidence matters.
- Stem caches consume substantial disk space.
- A low-confidence separation must never be treated as equivalent to clean supplied originals.
