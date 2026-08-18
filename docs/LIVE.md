# DJAI 4.0.0 live architecture

The live audio path executes a fully prepared immutable route. Strategic planning and private rehearsal happen before the audio deadline.

## Prepared alternatives

The live manifest exposes, per transition:

- `primary`: selected immutable action
- `backup`: alternate non-emergency executable action when available
- `emergency`: prepared emergency action when available

The progressive MP3 stream in 4.0 still follows the prepared **primary** route. Backup/emergency alternatives are exposed for future safe switching; the release does not claim crowd-reactive real-time switching yet.

## Future live brain

The intended evolution is rolling replanning several tracks ahead, with primary/backup/emergency prepared before each transition. Model latency must never be placed in the audio callback.
