# DJAI 5.1.0 quality system

Quality is evaluated after rendering finalist transition audio, not only from formulas or neural predictions over source tracks.

## Hard feasibility

Invalid source ranges, impossible duration/tempo policy, unavailable required sources, execution-signature errors and other deterministic violations are rejected before artistic preference can override them.

## Technical and musical render measurement

The technical gate includes loudness/spectral/bass/stem continuity, phase safety, clipping/artifact safety, silence safety and boundary continuity. Musical execution evidence includes rhythm coherence, transient clarity, groove continuity, harmonic continuity, energy-intent fit and drop payoff.

## Neural policy cannot bypass the gate

5.0 can use learned heads to rank transition family, duration, stem choreography and next-track value before rendering. Those predictions do not count as proof of quality.

The rendered DJ Judge may also receive a learned preference residual. Its authority is capped and a hard failure always rejects the candidate regardless of learned confidence.

## Acceptance and retry

`review_needed` is not a final automatic quality class. In prepared automatic construction only an accepted `TransitionResolution` may be committed.

A rejected family causes the solver to try another bounded musical family. If no family clears the rendered-audio gate, the directed edge is blocked and the Set Brain replans track order. A separately rendered/judged emergency escape is permitted only after normal edge/order search is exhausted.

## Model authority

DJ-specific learned heads start in shadow unless a validated checkpoint is installed. Shadow predictions are retained for comparison/Arena work but have zero effect. Advisory/production influence requires confidence and registry gates; production additionally requires rights review, held-out validation, blind Arena readiness, governance approval, and checkpoint pinning.

## Final-set truth

A transition accepted during an early partial route can be discarded when a later junction causes global replanning. Final telemetry therefore reconciles the complete run and marks `accepted=true` only for measured execution signatures that survive into the final set.
