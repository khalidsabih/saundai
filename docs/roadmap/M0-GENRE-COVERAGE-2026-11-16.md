# M0-05B — Actual genre-coverage assessment (2026-11-16)

**Evidence:** Owner provided the 34 relative file names from private `/tmp/saund-m0-crate-unique.json`. All 34 SHA-256 file payloads are unique, and validation reported zero structural errors. Filename/title assessment is **provisional**, not audio-derived classification or a rights approval.

## What is genuinely present

**Strong cluster:** Approximately 30 electronic/dance-oriented files, mostly melodic techno, melodic house, progressive house and adjacent styles (Tale Of Us, Agents Of Time, Mathame, Colyn, Innellea, Mind Against, ARTBAT, Anyma, Argy, Massano, Stephan Bodzin, Ben Böhmer, Lane 8, Yotto, etc.). Their specific genre labels must be checked individually; the shared cluster is not proof of conventional house, techno, or genre-balanced sampling.

**Promising outliers to review:**
- `uploads/14 - Dizzy DROS - SLS DIARY _Pt. 1_.mp3`: rap/hip-hop candidate, exact musical characteristics need confirmation.
- `uploads/18. Lmoutchou - Toc Toc 4.mp3`: genre **unknown**, do not infer an Afro House, Amapiano, Dancehall or Reggaeton label from the name.
- `uploads/Libianca_-_People__Official_Video_.wav`: Afrobeats / Afro-pop candidate, **not automatically Afro House**.
- `uploads/Peggy Gou - _It Goes Like_ Nanana - Official Video.wav`: dance-pop/house candidate.
- `final-crate/BICEP_GLUE_(Official_Video).wav`: breakbeat / alternative electronic candidate, useful for rhythm complexity only after listening/verification.
- `final-crate/WhoMadeWho_-_Silence_&_Secrets_(Black_Coffee_Remix)_(Official_Audio).wav`: **Afro House** candidate, separately cataloged as Afro House on Beatport (track `Silence & Secrets — Black Coffee Remix`, released 2022-04-08; external catalog genre is a metadata label, not beat-grid validation).

External source for the last catalog classification: `https://www.beatport.com/track/silence-secrets/16350963`. Rap recording official source: `https://www.youtube.com/watch?v=zDQeFmTYT0I`. Do not use catalog metadata as verified tempo/downbeat labels.

## Fixed S01–S12 scenario status

All are **unassigned** in the local manifest. The following table is an *assignment plan*, not completed/ready scenarios.

| ID | Candidate assignment and observation | Gap / status |
|---|---|---|
| S01 House → House | CamelPhat & Elderbrook `Cola`, Peggy Gou `Nanana`, electronic house candidates | PROVISIONAL; validate actual versions and select comparable independent recordings |
| S02 Techno → Melodic Techno | Tale Of Us `Nova`, Mathame `Skywalking`, Innellea `The World Returns` | PROVISIONAL; verify Techno vs Melodic Techno distinction |
| S03 House → Afro House | CamelPhat `Cola` or another verified house recording → WhoMadeWho/Black Coffee `Silence & Secrets` | PROVISIONAL; Afro House catalog label supports a candidate, not a validated rhythmic example |
| S04 Afro House → Pop | `Silence & Secrets` → Peggy Gou `Nanana` (dance-pop) | PARTIAL; conventional Pop candidate needed for non-dance pop breadth |
| S05 House → Hip-Hop | `Cola` → Dizzy DROS `SLS DIARY` | PROVISIONAL; rap/hip-hop version and tempo gap must be checked |
| S06 R&B → vocal Pop | No clear R&B recording and no distinct vocal-pop reference | **MISSING** |
| S07 Hip-Hop/Rap → Pop | Dizzy DROS `SLS DIARY` → Peggy Gou `Nanana` | PROVISIONAL; single rap example / dance-pop narrow |
| S08 Amapiano/Dancehall/Reggaeton | `Lmoutchou — Toc Toc 4` is unclassified | **MISSING VERIFIED COVERAGE** |
| S09 Jazz/Rock/live drums | No reliable candidates in filenames | **MISSING** |
| S10 Ambient/orchestral/sparse intro | Some electronic tracks may contain sparse intros, but no independently verified full sparse/ambient scenario | **MISSING VERIFIED COVERAGE** |
| S11 Odd meter/meter shift | No verified unusual-meter examples | **MISSING** |
| S12 Multi-genre 20–30 min | Enough raw recordings for an electronic-centered set; some rap/Afrobeats/pop candidates | PARTIAL; must approve coherent multi-genre target and track rights |

## Explicit M0 collection blockers

1. Confirm the specific version/genre and scope of existing candidates using owner review; do not fill `genre` based solely on musical artist or filename.
2. Add/locate rights-permitted independent reference tracks to cover R&B, vocal Pop, syncopated non-house styles, Jazz/Rock/live instruments, Ambient/sparse passages and odd meter. Those are **collection gaps**, not demonstrated software failures.
3. The manifest still has `rights_status=unreviewed` and `split=unassigned` for all 34 tracks; personal possession does not establish external training, public distribution or benchmark release rights.
4. Decide recording-group and artist-group development vs test separation **before** threshold tuning or learned-model work.
5. Preserve unavailable cases as unavailable. Do not turn the 34-file crate into fictitious 12/12 genre coverage.

**M0 status:** collection inventory complete; genre candidate mapping drafted; scenario freeze, private rights review and held-out split **pending**.
