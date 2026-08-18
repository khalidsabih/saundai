# SAUND.AI 1.3.0 product surface

**SAUND.AI — Hear ahead.**

The product should stay simple at the point of creation, then reveal the musical intelligence when the user wants to understand or supervise the performance.

## Product architecture

```text
CREATE
SAUND thinks

STUDIO
SAUND explains

LIVE
SAUND performs

LAB
SAUND learns
```

These are not four versions of the same dashboard. Each surface has a different job.

## Create — default

```text
add music
-> choose direction
-> Create set
-> human-readable real progress
-> listen
-> export / live / Studio / feedback
```

Create never exposes compatibility matrices, model feature vectors, critic ensembles or internal search scores. It translates the engine into decisions and actions.

After the final render, Create may summarize real performance facts such as how many transitions were beatmatched, how many used stem choreography and how many transferred tempo ownership. Those values come from the prepared plan; they are not decorative AI telemetry.

## Studio — opt in

Studio is the explanation layer for the autonomous DJ. It remains read-only-first rather than becoming another DAW.

Studio exposes:

- selected music and set direction;
- a horizontal/stepped performance timeline for the prepared order;
- transition technique, length and purpose;
- matched versus native BPM and post-handoff tempo release;
- bounded beat-phase correction when actually present;
- Deck A -> Deck B tempo ownership;
- real stem choreography profile and stem-confidence state when trustworthy stems exist;
- learned cue/duration contribution when the chosen plan contains that metadata;
- preview, warnings and deterministic rendered-safety status.

If the selected transition is master-only, Studio says so rather than pretending isolated vocal/bass/drum control exists.

## Live — performance state

Live is not a generic audio-player card. It follows the exact prepared playback manifest and shows only real playback state:

- now / next;
- active track, transition or tempo-release segment;
- current BPM where the manifest provides it;
- Deck A / Deck B ownership;
- transition countdown;
- selected stem-choreography profile;
- pitch-preserving return to the incoming track's native BPM;
- whole-set timeline position.

The progressive Live stream must execute the same transition and post-transition tempo-release sequence as the finished WAV.

## Lab — learning state

Training Lab answers three questions before exposing research detail:

1. **What SAUND knows** — trained bootstrap heads.
2. **What SAUND is learning** — runtime candidates and decisive human preferences.
3. **What can affect Live** — advisory, shadow and hard-authoritative heads.

Detailed corpus counts, held-out metrics and governance remain available below that product-level summary.

## Audio UI

Audio previews continue to use native playback controls. There is no decorative visualizer and no waveform animation used as proof that the system is "thinking".

Performance diagrams are allowed only when they visualize actual plan/manifest state: track order, BPM ownership, stem profile, timing and transition segments.

## Product language

Customer-facing terms favor musical actions:

- Create / Studio / Live / Lab
- Direction
- Performance timeline
- Deck A / Deck B
- Tempo handoff
- Musical ownership
- Prepared transition
- Rehearsed / Review
- Listen / Finish / Export

Internal feature vectors, graph objectives and raw critic/model internals remain engineering concepts. Learned priors may be named when they materially affected a decision and the plan carries that evidence.

## Design direction

- near-black neutral base;
- warm light typography;
- restrained amber accent;
- generous space and strong hierarchy;
- minimal motion and no glow-heavy cyberpunk treatment;
- controls behave like a calm instrument, not a dashboard;
- all essential actions remain keyboard accessible;
- professional information density in Studio/Live without sacrificing Create simplicity.

## Cinematic opening

The existing cinematic entrance remains the first-visit brand moment using the optimized supplied performance footage and soundtrack assets.

It is no longer a mandatory ten-second gate on every visit. After a user has entered once, a local browser preference skips directly to Create on later visits. The opening can remain available as a brand asset/demo experience without slowing normal repeated use.

The master footage is not shipped to browsers. Derived desktop/mobile video and poster assets remain the browser delivery format. Audio-rights requirements in `docs/SAUND_HERO_AUDIO.md` remain unchanged.

## Motion

Motion has three jobs only: establish identity, show spatial continuity, and confirm interaction. It is never used as evidence that the AI is thinking.

Create progress continues to reflect real backend job state. Studio and Live state must be driven by real plan/manifest fields rather than fabricated progress, confidence or activity animations.
