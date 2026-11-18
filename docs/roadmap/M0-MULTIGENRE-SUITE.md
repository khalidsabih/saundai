# M0 multi-genre reference suite — fixed specification

**Status:** SPECIFIED, NOT POPULATED OR RUN. Owner approval and a rights-cleared private asset manifest are still required. Frozen source baseline: `main@1aef9a8d8fbfaac90e2b65d4f9e70120618785bd`.

## Twelve scenarios (IDs remain stable across M1–M6)

| ID | Scenario | What must be demonstrated | Milestones |
|---|---|---|---|
| S01 | House → House | stable meter, clean compatible blend | M1–M3 |
| S02 | Techno → Melodic Techno | long-form phrase/cue alignment | M1–M3 |
| S03 | House → Afro House | syncopation and rhythmic compatibility | M1–M3 |
| S04 | Afro House → Pop | genre-change technique choice | M2–M4 |
| S05 | House → Hip-Hop | tempo gap, half/double relationship, safe alternative to stretching | M1–M3 |
| S06 | R&B → vocal Pop | concurrent lead vocals and vocal conflict | M2–M3 |
| S07 | Hip-Hop / Rap → Pop | section-aware cut or supported short technique | M2–M3 |
| S08 | Amapiano / Dancehall / Reggaeton | syncopated, subdivided beat evidence | M1–M2 |
| S09 | Jazz / Rock / live drums | drifting / unquantized tempo and safe abstention | M1–M2 |
| S10 | Ambient / orchestral / sparse intro | no stable beat; choose safe non-beat technique or decline | M1–M3 |
| S11 | Odd meter / meter shift | correct metrical level and bar one, or refuse unsupported behavior | M1–M2 |
| S12 | Multi-genre 20–30 minute set | arc, target duration, no broken joins, coherent ending | M3–M6 |

## Dataset manifest contract (private, not committed)

Each scenario needs a stable ID, audio filename or private asset pointer, SHA-256, rights/provenance status, human-reviewed genre, pair order, expected operational characteristics, development vs test split, and any verified tempo/beat/downbeat annotations. **Never infer musical ground truth from Librosa or Beat This! output.** Freeze the evaluation tracks before tuning. Do not put private music, personal paths, model weights or copyrighted waveform excerpts in the repository.

Proposed starter coverage: at least three independent examples per applicable scenario, plus an untouched holdout set separated by recording, artist, alternate versions, and excerpts. This is a sampling *target*, not a claim of statistical sufficiency. Explicitly record unavailable combinations.

## Measurement and exit decisions

- **M0:** source/revision freeze, successful synthetic render, active model status, audio file size/hash and error/fallback log. **No musical-quality score inferred.**
- **M1:** independently verified BPM metrical level and beat/downbeat event matches; conservative gate failure evidence.
- **M2:** transition feasibility, supported technique, hard safety, and safe-decline counts across S01–S11.
- **M3:** preregistered blind A/B comparisons, ties/neither, audible artifact rates and uncertainty.
- **M4:** set-level timing/energy/ending intent, blinded listening and complete-join reliability.
- **M5:** shadow learned models versus frozen deterministic champion on holdout music.
- **M6:** device-specific live playback and recovery under latency/load/fault scenarios.

Any scenario lacking legal use rights or credible reference data is `not_ready`, never silently skipped or marked passed. A model agreeing with another model is not independent validation.
