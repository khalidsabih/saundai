# DJAI 5.1.0 architecture

> **Historical status:** This is the DJAI 5.1 architecture snapshot. It is retained as implementation history, not as the current SAUND.AI runtime contract. See [CANONICAL_ARCHITECTURE.md](CANONICAL_ARCHITECTURE.md) for the current architecture, authority hierarchy and version/schema source of truth.


## Core invariant

`perception -> evidence -> policy -> private rehearsal -> professional judge -> content-bound execution plan -> deterministic renderer -> final verification`

The renderer never makes DJ decisions. Neural models may rank/narrow feasible actions, but deterministic validation keeps veto power. Set rehearsal and final execution share an explicit render context (sample rate and normalization gains), and final transition samples are technically verified before publication.

## Runtime

```text
FULL TRACKS
  -> automatic performance layers / source quality
  -> deterministic analysis + Neural Ears
  -> TrackAnalysis + phrase-synchronous TrackWorldModel
  -> SetState / receding-horizon Set Brain proposes futures
  -> TransitionRouter + bounded learned family residual
  -> cue search + phrase-derived duration candidates + learned duration ranking
  -> stem ownership choreography + learned profile ranking
  -> private deterministic finalist renders
  -> hard measurement + Professional DJ Judge + bounded learned rendered preference
       accepted -> immutable TransitionProgram -> renderer
       rejected -> alternate family -> blocked edge -> Set Brain replan
```

## Neural evidence is not authority

Foundation encoders are sensory inputs. DJAI does not equate generic embedding similarity with DJ compatibility or professional taste. Neural beat evidence, music representation and DJ-specific policy predictions retain provenance and confidence in analysis/plan metadata.

## Model registry

`config/model-registry.json` is default-deny. Models are `disabled`, `shadow`, `advisory` or `production`. Shadow predictions are observable but have zero influence. Production authority requires `production_approved=true` plus the training/promotion gates. Authority is bounded even after promotion.

Normal startup never downloads a checkpoint. `models-bootstrap` is explicit and model caches are persistent in Docker.

## DJ world model

`TrackWorldModel` is the stable representation seam between perception and learned DJ policy. It summarizes the track globally and as phrase states containing structural, energy, rhythm, vocal, bass, stem-role and embedding evidence.

## Structured policy heads

The 5.0 learned action space is deliberately interpretable:

- transition-family ranking;
- duration ranking over structurally valid values;
- stem-choreography profile ranking;
- rendered-transition preference score;
- next-track value residual.

Models do not output final waveform samples. They produce structured decisions that become ordinary `TransitionPlan`/`TransitionProgram` data.

## Source preparation

The original master is timing authority. Supplied official sources outrank cached AI separation. AI-derived vocals/drums/bass/other/instrumental are cached and carry quality confidence. Failed separation does not make a valid stereo master unusable.

## Unified transition solver

`TransitionResolution` remains the only normal automatic commitment boundary. `review_needed` is diagnostic/training state, never a final prepared-set performance state. The solver searches musical families hierarchically and shares one bounded renderer budget across them.

## Duration and stem composition

Duration is searched: family-standard values plus phrase-derived custom spans. With four-stem material, `stem_choreography` can encode independent gain curves for drums, bass, vocals and other material. Learned ranking may reorder profiles, but the renderer executes the exact frozen curves.

## Professional judge

The judge combines hard feasibility, rendered technical/musical evidence, set context, novelty, rejection memory, existing preference/imitation priors and the optional learned rendered-preference head. Hard failure cannot be overridden by a high neural score.

## Set feedback and lookahead

A rejected directed edge is blocked and set ordering runs again. The learned next-track head is only a bounded residual inside the existing receding-horizon graph policy; private transition rehearsal still decides whether the edge survives.

## Telemetry semantics

Every private rehearsal can remain training evidence. A locally accepted transition can still be superseded if a later edge forces global set replanning. At the end of set construction DJAI reconciles the current run's measurement records so `accepted=true` means that exact execution signature survived into the final `SetPlan`.

## Simple and Pro

Simple and Pro share one listener identity, model registry and transition brain. Simple may auto-deliver; Pro keeps explicit confirmation before final commitment and exposes plans/overrides.
