# SAUND.AI Hero Audio

The Create entrance uses two separate audio assets so playback, muting, crossfading, and future replacement remain independent from the hero video.

## `frontend/public/media/saund-hero-intro.mp3`

Derived from the project-owner supplied file:

`DJ DROP - OPENER SHOW - English (UK) --- (DJ INTRO).mp3`

Release processing:

- approximately 10 seconds retained as the opening identity sting, ending at a natural phrase boundary;
- 44.1 kHz stereo;
- web loudness normalization to a restrained opening level;
- short entrance fade;
- no third-party material was added by the SAUND.AI build process.

**Rights note:** this is user-supplied source material. SAUND.AI does not assert a license for the underlying recording. The project owner must confirm that the recording/voice/music may be distributed publicly before the final 6.0.0 release.

## `frontend/public/media/saund-hero-bed.mp3`

Original SAUND.AI background bed synthesized specifically for this project from mathematical oscillators and deterministic noise. It uses no external samples or third-party recordings.

Characteristics:

- 44.1 kHz stereo;
- approximately 108 BPM;
- warm A-flat-major / F-minor harmonic palette;
- 64-bar evolving electronic bed;
- approximately 2 minutes 22 seconds before the browser loop repeats;
- deliberately lower loudness than the opener so it can remain behind the product experience.

The project owner may distribute this generated bed with SAUND.AI without third-party music attribution.

## Playback contract

1. The cinematic video can autoplay muted.
2. SAUND.AI attempts the opener with sound when the browser permits it.
3. If audible autoplay is blocked, the hero exposes one restrained **Sound on** control rather than presenting a broken player.
4. Enabling sound starts the opener from its beginning and restarts the hero video for a deliberate entrance.
5. If the customer stays on the intro, the final 3.4 seconds naturally hand off into the background bed.
6. **Enter SAUND.AI** is also an audio transition trigger: when clicked before the natural handoff, the opener crossfades immediately into the quieter work bed while the intro screen dissolves.
7. The first screen is viewport-locked and Create remains inaccessible until that Enter action.
8. The same control becomes **Mute** / **Sound on** and fades volume instead of hard-cutting it.
9. Starting a set or entering Studio fades/stops the soundtrack.

There is no waveform, spectrum, equalizer, or visualizer associated with this soundtrack. Audio is part of the opening atmosphere, not a technical display.
