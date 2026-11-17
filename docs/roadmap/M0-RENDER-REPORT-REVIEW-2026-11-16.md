# M0 — final render report and set-plan audit (2026-11-16)

**Evidence:** User-uploaded `full-set.json` and `set-plan(2).json` from the pinned synthetic five-track baseline. Files inspected as structured JSON. This audit refers to those inputs; no audio listening or local execution was performed here. The render report embeds a set plan; the standalone plan matches the same `plan_id=4f03b0743775dca8`, version 8.0.0 and track order, although the entire nested objects are not byte-identical. Preserve both as distinct evidence.

## Verified summary

| Property | Observed |
| --- | --- |
| Rendered audio | Stereo 44,100 Hz, 206.926803 s (3m26.927s); peak 0.985 |
| Offline execution | `offline_full_set`, `immutable_plan_execution=true` |
| Executive | `success=true`, `transition_count=4`, `verified_transitions=4` |
| Final transition verification | `enabled=true`, `passed=true`, all four `passed=true` and `exact_transition_samples=true` |
| Technical transition overall scores | 0.9007, 0.8981, 0.8996, 0.9076 |
| Hard transition failures | 0 reported in each verification |
| Clipping | `clipping_fraction=0` on all four transitions |
| Phase-correlation warnings | All four; -0.4572, -0.5225, -0.5036, -0.3838 |
| Phase safety subscore | 0.1905, 0.1541, 0.1646, 0.2312 (low) |
| Musical verification submetrics | `musical_overall=0.0` and `rhythm_coherence=0.0` on all four (not proof of good musical results) |
| Plan track order | [0,1,2,3,4]; `improvement_percent=0.0` |
| Planned transitions | 4: two selected `safe/good`, two selected `emergency/risky` |
| Fallback count | `fallback_transition_count=2` (50%); `emergency_rate=0.5` |
| Full-set benchmark | `overall=0.644495`, `transition_quality=0.9015`, `story_fit=0.282586`, `transition_diversity=0.5` |
| Production-safe | **`production_safe=false`** in plan metadata |
| AI Headphones | `ai_headphones_authoritative_decisions=0`, `ai_headphones_rendered_candidates=0` for this run |
| Genre on synthetic tracks | Each track's classifier returned `unknown`; not a genre benchmark |
| Runtime | Offline executor reports `neural_control=false`, `deterministic=true`, safety governor action `allow` |

## Individual transition summary

| Transition | Planned style / quality | Special warnings | Exact render verification |
| --- | --- | --- | --- |
| Opening → Groove | safe / good | Bass overlap; low stereo phase correlation | Passed |
| Groove → Drive | **emergency / risky** | Final deterministic escape after no evidence matrix; bass overlap; low phase correlation | Passed technical checks |
| Drive → Peak | safe / good | Bass overlap; low stereo phase correlation | Passed |
| Peak → Close | **emergency / risky** | Final deterministic escape after no evidence matrix; bass overlap; low phase correlation | Passed technical checks |

## Decision and implications

**M0 synthetic pipeline execution: PASS. M0 full milestone: NOT YET PASSED.** The set rendered, verified exact samples and executed all four committed transitions. However, the planner's `production_safe=false` must be preserved and never overridden by the render-verification `passed=true`.

- **M1:** investigate low downbeat confidence and the neural grid benchmark separately from M0, without enabling live adoption. All 5 synthetic tracks had low reported downbeat confidence (~0.0011–0.0105).
- **M2:** target the two `emergency/risky` transitions and the repeated strong-bass overlap warnings. Do not erase/rename risky outcomes simply to satisfy a gate.
- **M3:** technical transition scores near 0.9 are not a measured blind preference or genuine musical quality. The run recorded zero AI Headphones render decisions and zero musical-overall scores.
- **M4:** story fit 0.282586 and unchanged track order require a separate set-level evaluation; synthetic preset order may constrain improvement.
- **M0 reproducibility:** model weights were unexpectedly downloaded during analysis, and the registry resided in the main workspace. Freeze cache/network/model config before claiming an offline/repeatable baseline.

Open M0 requirements: inspect audio WAV metadata if needed; capture runtime/environment and model cache state; freeze rights-cleared multi-genre private assets and holdout split; owner signoff. No live engine, neural control, playback or cue code changed.
