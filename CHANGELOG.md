## 2.0.0-rc.5 — Direct local upload hotfix

- Local non-hosted uploads now use the backend-reported `hosted_mode: false` to force browser → FastAPI transport on the same host/API port.
- The API port has a browser fallback of 8000 even if the public build variable was not inlined.
- Large audio no longer depends on hostname allowlists to bypass the Next proxy.
- Local-only CORS now accepts localhost, private-LAN origins, and `.local` hostnames for direct browser→API uploads; hosted-mode CORS remains explicit/strict.

# Changelog

## 2.0.0-rc.4 — Frontend upload reliability

- Bypass the Next.js rewrite for local Docker API traffic so large multipart music uploads go directly from the browser to FastAPI.
- Bound same-origin fallback uploads to one file per request and raise the Next proxy fallback body ceiling.
- Expose the API upload limit to the frontend and improve opaque HTTP 500 diagnostics.
- Expand local CORS defaults to common localhost aliases.

## 2.0.0-rc.3 — Frontend Stabilization

- Preserve the exact completed automatic-set plan and render when opening Studio from the finished-set screen instead of mounting an empty Studio.
- Move Create's AI/status/action controls into one responsive header region; remove the RC2 top-right collision between AI presence and Studio/Lab/Account controls.
- Lock Create step/Studio/Lab navigation while an automatic set job is active, and lock Studio exits to Create/Lab while plan/preview/render work is active, so client-side job/result state cannot be stranded by navigation.
- Reconcile stale selected-track paths after successful library refreshes in Create and Studio.
- Treat brain/release diagnostics as optional in Studio: a diagnostics failure no longer marks the track API itself offline.
- Make visible AI evidence execution-signature safe; never attach Headphones/Crate Memory evidence to a final transition whose executable signature differs.
- Distinguish Headphones render attempts from previews that were actually heard; show failed private renders and non-authoritative rehearsal honestly.
- Map final set rendering to a distinct Performing state instead of calling render-time work Deciding.
- Add accessible track-analysis progress, reduce duplicate live-region announcements, trap/restore account-dialog focus and lock background scroll.
- Improve AI text readability, mobile/short-screen behavior, five-field Studio facts, disabled destination states and focus-visible controls.
- Fix RC build identity so the reported release-candidate number is derived from the actual version instead of remaining stuck at RC1.

## 2.0.0-rc.2 — Visible AI Experience

- Add a persistent SAUND AI presence to Create and Studio with evidence-backed Listening / Understanding / Planning / Auditioning / Deciding / Learning states.
- Surface track intelligence from real 1.6.1 readiness/features instead of a generic Ready badge.
- Add a live Create pipeline driven by actual job stages; the client only says AI Headphones is auditioning when measurement/transition rehearsal is reached.
- Add a Studio AI activity feed from `complete_move_planner.latest_route`, including lookahead, lazy-search, real Headphones previews, Crate Memory influence and hard-safety authority.
- Add real per-transition AI Headphones finalist cards and show when listening changed the analytical winner.
- Add human musical transition explanations derived from payoff, phrase, harmony, vocal/bass and Headphones evidence, explicitly not private model reasoning.
- Add Crate Memory explanations only when directional memory evidence actually exists.
- Add an energy-journey visualization and compact finished-set AI recap.
- Keep all 1.8/1.9/RC1 musical and safety authority unchanged; RC2 is a client-evidence release.

## 2.0.0-rc.1 — Release Candidate 1

- Promote the Complete DJMove planner to the RC Docker default and, critically, pass planner/search, AI Headphones, structural-operation and Crate Memory environment settings through Compose so `.env` matches the running engine.
- Change the API container health check from liveness-only `/api/health` to operational `/api/ready`.
- Add `/api/release/readiness` with separate operational and strict certification modes; expose the same RC status in `/api/runtime/status` and Studio.
- Add persisted-plan compatibility preflight: old signed plans are preserved but require replanning; future execution schemas are blocked rather than silently downgraded.
- Add non-mutating Crate Memory schema preflight so unknown SQLite schemas cannot be silently accepted.
- Add `scripts/release-candidate.py` for status and deterministic certification-manifest generation.
- Make the release benchmark champion gate mandatory automatically for RC versions, even when the development benchmark flag is off.
- Require certified RC packaging to embed `RELEASE_CERTIFICATION.json`; skipped release gates remain explicitly uncertified.
- Add RC readiness to `scripts/doctor.sh` and source/build fingerprints.
- Keep 1.8/1.9 musical authority unchanged: release certification can block promotion but never override rendered-audio or deterministic safety.

## 1.9.0 — Benchmarking, Reliability + UX Polish

- Add canonical small/standard/large/stress release benchmark suite contracts plus named specialist crates without bundling any music.
- Add a metrics-only SQLite benchmark history and named champion registry; track paths/titles/raw audio are never copied into the benchmark database.
- Add deterministic champion/challenger gates for rendered safety, protected payoffs, fallbacks, full-set quality, latency and peak-memory regressions, with optional human blind-preference enforcement.
- Persist final analysis/planning/render/total timing, process peak RSS and a release benchmark snapshot into completed automatic set plans; explicit Studio renders also refresh render telemetry.
- Add `/api/benchmarks/status`, benchmark capability flags and benchmark status to the professional brain endpoint.
- Add an advisory benchmark step to the release gate that can be made mandatory for RC/release CI and requires a current-version challenger.
- Add benchmark diagnostics to `scripts/doctor.sh` and a direct `scripts/release-benchmark.py` workflow for record/compare/promote/status.
- Add a Studio Performance proof strip showing real full-set quality, AI Headphones preview count, sparse graph exploration, Crate Memory influence and delivery timing.
- Keep 1.8.2 DJ decisions unchanged: benchmarking can block promotion, but it cannot override runtime audio safety.

## 1.8.2 — Crate Memory

- Add a local SQLite Crate Memory index for persistent record-specific and directional pair experience without storing raw audio.
- Learn preferred exit/entry bars, transition lengths, outgoing loop positions, incoming tempo comfort, technique families and stem strategies from explicit Taste Lab preferences.
- Track pair-specific failure reasons such as vocal clash, forced tempo, wrong next track and energy mismatch; keep A → B memory distinct from B → A.
- Persist deterministic record facts such as vocal-end and plausible structural loop bars separately from preference authority.
- Feed Crate Memory into Complete DJMove ranking only as a bounded residual; protected payoffs, playhead feasibility, AI Headphones, rendered measurement gates and deterministic DSP safety remain authoritative.
- Surface Crate Memory evidence on Complete DJMove candidates, AI Headphones evaluations, brain status and Training Lab.
- Update account/listener deletion so Crate Memory events are removed and aggregate preference influence is rebuilt from remaining evidence.
- Keep State × Move Taste as a separate general shadow head; Crate Memory is local, specific and immediately interpretable.

## 1.8.1 — DJState × DJMove Taste Learning

- Add a dedicated state-conditioned taste head whose input is the sanitized current `DJState` plus the complete `DJMove` chosen by the 1.7 planner.
- Capture current energy/tension/vocal/bass state, pending commitments, WAIT/transition timing, tempo/phase behavior, stem/structural operations, payoff state, projected musical deltas and future optionality in a stable learning feature schema.
- Attach the same state/move payload to new Taste Lab candidates and Complete DJMove runtime evidence without storing local track paths in the learning representation.
- Add human reason tags aligned with complete DJ decisions: wrong next track, early/late exit, transition length, forced tempo, vocal clash, energy error and payoff quality.
- Train the new head from decisive blind state-move comparisons with grouped validation across independent musical states.
- Expose State × Move Taste as the third governed Taste head beside Global Taste and Rendered DJ Judge in Training Lab.
- Surface state-conditioned taste shadow predictions inside Complete DJMove and AI Headphones evidence for audit/benchmarking.
- Keep the new model shadow-only with zero live ranking authority; feasibility, rendered-audio safety, AI Headphones and the final deterministic compiler remain authoritative.

## 1.8.0 — AI Headphones

- Add a private real-audio decision layer at the Complete DJMove commit boundary: cheap search stays broad, then only 2–4 finalists are rendered and listened to before the first move is committed.
- Reuse the production deterministic renderer, measurement gate, multi-critic judge, Rendered DJ Judge, rejection memory and governed taste evidence rather than inventing a parallel scoring stack.
- Add a cheap 12-move serious pool with next-track/technique diversity and an uncertainty-aware render budget that expands when the rendered winner is uncertain, rejected or too close to another finalist.
- Capture rendered transition embeddings during Headphones rehearsal so the Rendered DJ Judge can evaluate the actual private preview.
- Keep lookahead branches analytical; opener routes are scouted without audio and Headphones runs only on the single selected route, avoiding multiplicative render cost.
- Treat decoder/render failures as hard failed private rehearsals. Headphones receives no authority unless at least one finalist is both feasible and accepted.
- Record the private finalist population, critic domains, rendered/taste evidence, cache hits, winner changes and authority state in set-plan metadata.
- Preserve 1.7.2 signed structural operations, 1.7.1 lazy 2–3 move lookahead and the final deterministic rehearsal/safety compiler as the last veto.

## 1.7.2 — Executable WAIT + Structural Deck Operations

- Add a versioned deterministic deck-operation IR for WAIT, loops, jumps, intro extension, stem holds and breakdown skips.
- Compile deliberate WAIT decisions into the signed Complete DJMove instead of keeping wait only as rationale/metadata.
- Execute outgoing loops and forward jump/skip operations in full-set body rendering using explicit source coordinates.
- Execute incoming intro/loop holds inside the immutable transition program so rehearsal and final rendering share the same DSP path.
- Compile `HOLD_STEM` into explicit stem automation rather than letting planner code touch samples.
- Bind structural operations into the performance-program fingerprint and final execution signature; bump performance program to 3.1 and render engine to 6.2.0.
- Keep advanced structural edits opt-in by default while WAIT is enabled automatically.
- Preserve 1.7.1 sparse retrieval, 2–3 move lookahead, legacy/shadow modes and deterministic safety vetoes.

## 1.7.1 — Lazy Crate Search + Receding-Horizon DJMove Search

- Add cheap successor retrieval using tempo, harmony, energy/bridge usefulness, embeddings, genre and current set-journey context before expensive transition planning.
- In `move_authoritative`, materialize transition-graph edges only for candidate neighbourhoods reached by search; expand the neighbourhood when trapped and retain a full-graph legacy fallback only for recovery.
- Add bounded 2–3 complete-move lookahead with projected `DJState`/`SetState`, discounted continuation value and first-move-only commitment.
- Bound opener roots before expensive search instead of evaluating every possible opener on large crates.
- Keep `move_shadow` apples-to-apples by preserving the legacy full graph while the new planner itself still uses retrieved candidates.
- Record retrieval evidence, projected winning paths, realized lookahead depth, sparse/full graph authority and materialized-vs-possible edge counts in set-plan metadata.
- Keep rehearsal, rendered critics and deterministic audio execution authoritative.

## 1.7.0 — Complete DJMove Generator

- Add a complete-move generator that chooses next track, exit/entry phrase, wait, transition length, tempo/phase/stem strategy and technique as one `DJMove`.
- Score complete actions with current `DJState`, set-journey context, Set Director intent, executable transition quality and future optionality instead of treating song order as an isolated decision.
- Add explicit `DJAI_PLANNER_MODE=legacy|move_shadow|move_authoritative`; 1.7.0 defaults to shadow comparison while preserving the existing planners.
- In authoritative mode, pass the generated move into the path-dependent compiler as a concrete preference; preserve it when executable while keeping rehearsal and deterministic audio safety able to override it.
- Reject static graph actions that point behind the current playhead or violate protected musical payoffs.
- Record proposed-vs-delivered route evidence, top move candidates, decision components and generated-vs-compiled transition signature matches in set-plan metadata.
- Keep 1.7.0 at one-step decision depth; lazy candidate retrieval and 2–3 move lookahead remain the explicit 1.7.1 milestone.

## 1.6.1 — Background Track Readiness + Modular Feature Store

- Add a content-addressed, independently versioned feature store for descriptors, rhythm, structure, embeddings, cues, canonical timeline, world model and stems.
- Calculate/upload-register source SHA-256 fingerprints without an extra post-upload file scan and cache unchanged source identities in a stat-validated source index.
- Automatically migrate warm legacy TrackAnalysis caches into modular feature artifacts and reconstruct TrackAnalysis from complete current modular projections when the compatibility blob is missing.
- Schedule upload/library track preparation on the existing background/stem worker pool so foreground Create/Studio jobs are not queued behind readiness work.
- Make configured Demucs preparation part of track readiness while allowing DJ-ready core intelligence when stems are disabled or unavailable.
- Add per-track readiness/progress/job metadata to `/api/tracks`, an explicit readiness retry endpoint, feature-store diagnostics in `/api/runtime/status`, and 1.6.1 health/build identity flags.
- Show `DJ ready` or the active preparation stage in Create/Studio and poll the library only while preparation is active.
- Keep the 1.6.0 canonical timeline, current set planner, transition quality loop and deterministic renderer authoritative.

## 1.6.0 — Canonical Musical Timeline

- Add one versioned `src/djai/music/timeline.py` authority for beats, bars, adaptive phrases, sections and protected musical commitments.
- Move the 1.4 Move Brain onto the canonical timeline while preserving its public imports and fallback behavior.
- Convert `TrackWorldModel` into a compatibility projection of canonical phrases instead of maintaining a second fixed eight-bar phrase grid.
- Put canonical timeline version/fingerprints and exit/entry phrase IDs, roles and readiness onto transition-plan metadata.
- Resolve deck tempo-release exposure from the canonical incoming phrase and persist the same phrase identity into deck-performance metadata.
- Include a compact phrase/section/commitment timeline in set-plan track payloads and surface the structural story in Studio.
- Add an independently versioned timeline schema and content fingerprint so later caches/planners can invalidate musical structure without invalidating unrelated features.
- Fix the frontend diagnostic `setInterval` type as a browser `number`, removing the Node `Timeout` TypeScript build collision.
- Keep the existing order/set planner authoritative; 1.7.0 Complete DJMove planning remains a separate shadow-first milestone.

## 1.5.3 — Engineering Integrity

- Make `VERSION` the authoritative release identity and add `scripts/sync-version.py --check` so Python, frontend package/lock and frontend build metadata cannot silently drift.
- Replace stale semantic Docker defaults with `local`/explicit `SAUND_IMAGE_TAG` naming and add backend source fingerprint/build identity plus generated frontend `build-info.json`.
- Add persisted job start/finish/heartbeat/progress timestamps, elapsed/stage elapsed diagnostics, progress-stall detection and explicit cancellation state.
- Add cooperative job cancellation through the API and Create/Studio UI; cancellation is honored at safe progress/stage boundaries rather than terminating DSP/cache writes mid-operation.
- Resolve API and stem worker counts through one conservative single-host CPU budget and expose the effective budget in `/api/runtime/status`.
- Add runtime/cache/job diagnostics plus safe Docker build-cache cleanup tooling that never prunes volumes.
- Modernize the release/production audit to follow the current `VERSION`, reject patch debris and validate generated frontend build identity.
- Preserve all 1.5.2 DJ planning, Move Brain, large-crate, rendered-quality and deterministic-safety behavior.

## 1.5.2 — Honest Job Progress

- Replace stage-local Create Set percentages with monotonic job-wide progress across preparation, analysis, large-crate scouting/order search, rendered transition checking and final WAV rendering.
- Publish real per-stage counters (`current / total`) through persisted job state and SSE updates so the frontend can show work such as tracks analyzed, graph edges scouted and transitions rendered.
- Start long automatic jobs with an explicit non-zero preparation state and preserve real backend-driven progress only; no timer-based or fabricated activity animation is introduced.
- Keep 1.5.1 large-crate planning, final transition quality, hard safety and local caches unchanged.

## 1.5.1 — Large Crate Planning

- Replace full 12×12 cue evidence on every all-pairs ordering edge with a bounded 5×5 scout matrix for crates of eight or more tracks; final selected edges rebuild the full path-dependent cue matrix before rehearsal.
- Cache one phrase-level `MusicalTimeline` per immutable analysis state and attach one cached Move Brain opportunity to every graph edge so order search does not repeatedly rebuild TrackWorldModel state.
- Limit redundant receding-horizon opener searches on large crates while retaining greedy seeds, unified beam search and local improvement.
- Reduce large-crate local-improvement passes and change the default automatic replan cap from roughly `3 × track_count` to a bounded 3–8 repair rounds; explicit `DJAI_MAX_SET_REPLANS` still overrides it.
- Pool global MERT/MusicFM/statistical track representations from bounded windows distributed across the record instead of encoding the entire song, while keeping local section embeddings around transition regions.
- Add large-crate planning telemetry/status and explicit progress during initial route choice and route repair.
- Preserve the full final cue search, rendered candidate rehearsal, 44.1 kHz measurement contract, continuous-mix policy and hard-audio safety veto.

## 1.5.0 — Host-Ready Foundation

- Add a fail-closed hosted/private-beta runtime profile while preserving local-first defaults.
- Add managed PostgreSQL account/session support through `SAUND_DATABASE_URL`.
- Add S3-compatible presigned browser uploads, object verification/materialization, and signed result delivery.
- Isolate hosted source libraries per account and remove hosted audio objects/materializations during account deletion.
- Add secure hosted startup gates, auth rate limiting, invite/open/closed registration, Origin/Host hardening, and `/api/ready`.
- Add a hosted Docker target, durable-workspace entrypoint, cloud-safe build context, deployment templates, and hosted audit tooling.
- Keep DJ job execution deliberately single-node/persistent for the private beta; distributed worker scaling remains a separate infrastructure milestone.

## 1.4.0 — Human DJ Move Brain

- Introduce `DJState -> DJMove -> projected DJState` as the phrase-level planning abstraction instead of treating track choice, cue, duration and deck behavior as independent decisions.
- Build an adaptive bar-native musical timeline from section changes and phrase/boundary evidence while retaining `TrackWorldModel` role semantics.
- Detect build/drop promises and high-salience protected moments; ordinary exits before a pending payoff are removed when a post-payoff opportunity exists.
- Record explicit `wait_then_transition` moves when the stronger transition opportunity lies several bars ahead.
- Cache a phrase-level move-opportunity score on every A->B graph edge and feed it into receding-horizon and full-order scoring, including a weakest-edge penalty to avoid musical dead ends.
- Replace hard percentage-of-track minimums with bar-aligned musical-body minimums when the Move Brain is active; elapsed-time pacing becomes a weak bar-based prior.
- Re-rank executable transition candidates with complete-move quality including phrase readiness, tempo burden, vocal space, duration fit, payoff protection and future optionality before rendered rehearsal.
- Project the incoming deck's post-transition bar/BPM/rate state so the planner can reason about where a move leaves the set.
- Surface complete-move reasoning in Studio and Move Brain authority in Studio/Lab status without inventing telemetry.
- Preserve the legacy time-zero/mature-entry policy when the Move Brain is disabled, while allowing a genuine bar-zero intro to compete normally in human-DJ mode.
- Finalize measurement telemetry against the actually delivered set, materializing a measured continuity-rescue plan when it was created after the ordinary rehearsal logger; final accepted counts now match final transitions.
- Preserve all deterministic execution validation and rendered hard-safety vetoes.

## 1.3.0 — Autonomous DJ Frontend

- Keep Create intentionally simple while reframing the product around four surfaces: Create = thinks, Studio = explains, Live = performs, Lab = learns.
- Show a prepared performance timeline in Studio with track BPM/key/energy, transition technique, bar length, tempo handoff and actual stem-profile labels.
- Expand transition inspection to expose real deck behavior: matched BPM, bounded beat-phase nudge, tempo ownership transfer, pitch-preserving return to native BPM, stem-role handoff, learned cue/duration contribution and hard-safety status.
- Rebuild Live as a deck-performance surface with now/next state, current BPM, transition countdown, Deck A/B ownership, stem choreography and a timeline that includes `tempo_release`.
- Make the Live manifest and progressive stream execute the same post-transition tempo-release segment as the finished WAV instead of silently omitting it.
- Add a top-level Training Lab learning map for what SAUND knows, what it is learning and which heads can affect live decisions.
- Keep the cinematic opener for first visit, then skip the mandatory ten-second gate for returning users.
- Add a finished-set performance summary and direct path into Studio for inspecting the decisions behind the rendered set.

## 1.2.0 — DJ Deck Performance

- Add bounded beat-phase nudging for the incoming deck after cue/phrase selection, without allowing a nudge to move across musical phrases.
- Keep Deck B matched to Deck A through the overlap, then transfer tempo ownership and glide B back to native BPM with pitch-preserving variable time stretch.
- Plan the next set edge from the post-handoff deck state instead of permanently propagating the temporary match rate through the set.
- Add an explicit `tempo_release` segment to full-set rendering/timeline output and include the release in preview/final post-transition verification.
- Add the `deck_vocal_exchange` four-stem choreography: incoming drums first, outgoing vocal out before incoming vocal, single bass-owner swap, gradual instrumental handoff, incoming vocal late.
- Bind phase-nudge and tempo-release state into the immutable execution signature and bump decision/set-plan compatibility to 8.0.0 and renderer semantics to 6.1.0.
- Preserve all existing hard rendered-audio safety gates and fall back to the prior constant-rate/native behavior when grid confidence, source length, or stretch bounds do not justify a deck handoff.

## 1.1.2 — Responsive Taste Lab

- Reuse the canonical track-analysis cache when generating blind Taste Lab candidates instead of re-analyzing full songs from scratch.
- Expose blind-session generation timings and cache-hit counts from the API for diagnosable latency.
- Show immediate, continuously updated frontend progress while real transition candidates are rehearsed and rendered, so long renders no longer look like a dead click.
- Preserve the same rendered measurement, hard-safety gate, blind candidate semantics, and taste-learning evidence contract.

## 1.1.1 — Taste Learning Loop

- Capture structured, audio-free, path-pseudonymized evidence for every real rendered rehearsal candidate, including rejected and hard-failed alternatives.
- Turn blind A/B/Tie/Neither Training Lab votes into governed supervision for a separate Global Taste Ranker and Rendered DJ Judge.
- Generate DJ-specific synthetic rendered failures from real local blind-session transition audio in memory, re-measure them through the production measurement stack, and persist features/provenance rather than damaged WAVs.
- Train both taste heads through one snapshot/train operation, while allowing one head to report an evidence blocker without fabricating a checkpoint.
- Keep the Global Taste Ranker at zero live authority and install a Rendered DJ Judge checkpoint only when the registry remains shadow/disabled, unapproved, and unpinned.
- Hold out whole track-pair groups for Rendered DJ Judge validation whenever at least two groups exist, preventing pair leakage.
- Expose taste-learning/runtime-evidence status through the API, CLI, and Training Lab UI.
- Preserve deterministic hard-audio safety as authoritative and keep research/unreviewed training evidence non-promotable to production.

## 1.1.0 — Learned Brain Integration

- Load the latest frozen-corpus bootstrap artifacts into the live decision loop through a bounded runtime adapter instead of leaving trained priors disconnected from Create Set.
- Use the held-out cue-point prior as a bounded cue-pair ranking residual; infeasible cues remain infeasible and rendered hard-safety gates remain authoritative.
- Use the transition-duration prior to reorder only already-valid bar-length candidates, then let the existing rendered judge choose among executable plans.
- Feed ListenBrainz adjacency/source-artist priors into the hybrid next-track brain when track/artist identity can be resolved; unresolved tracks fall back to audio/structure compatibility.
- Export a train-only `sequence-runtime.sqlite` on bootstrap retraining so runtime scoring can use the full learned sequence counts rather than only the top-5,000 JSON summaries.
- Add local-development `advisory` mode and production default-to-`shadow` governance for research bootstrap models.
- Expose bootstrap runtime state through `/api/brain/status` and refresh the loaded bundle after Training Lab retraining.

## 1.0.3 — Natural transition intelligence

- Add a deterministic transition-naturalness objective over phrase/boundary alignment, section flow, duration fit, tempo strain, harmony, vocal/bass cleanliness, rendered musical continuity, and technique simplicity.
- Use naturalness as a bounded pre-render and post-render ranking signal; hard rendered-audio safety gates remain authoritative.
- Add naturalness to the professional DJ judge and adaptive early-accept policy so an merely safe but awkward pilot expands to better candidates instead of being committed early.
- Preserve learned-taste governance by blending learned ranker authority on top of the natural deterministic baseline.
- Make automatic rescue duration adaptive: difficult tempo/key/vocal pairs use a compact 2-bar handoff with a 4-second minimum, while compatible pairs retain a balanced 4-bar handoff.
- Jointly choose rescue exit/entry cues using phrase strength, structural section flow, tempo/harmony and source-conflict evidence.
- Compile a compact filter-build choreography that clears the outgoing master earlier and establishes incoming bass/vocals later on difficult pairs.
- Persist naturalness telemetry with rendered transition measurements for future Taste Lab/model evaluation.

## 1.0.1 — Create Set latency hardening

- Make `DJAI_AUTO_STEMS=background` genuinely non-blocking during automatic set creation; cached stems are consumed immediately and unfinished separation falls back to the stereo master.
- Add adaptive rendered rehearsal: evaluate two diverse finalists first and expand to the configured finalist pool only when the pilot is rejected, weak, or uncertain. Taste Lab remains full-evidence.
- Add an exact two-track order path that skips receding-horizon, unified beam, and local-search machinery when only two permutations exist.
- Reuse decoded pair audio across transition-family retries and keep MERT/MusicFM runtimes process-resident.
- Skip rendered-transition embeddings on the live path unless an authoritative deployed model actually consumes them; Taste Lab still captures them.
- Move automatic scoring-model refits out of the set-delivery critical path.
- Emit per-job latency telemetry for stems, analysis, planning, render, and total delivery.
- Preserve the 44.1 kHz rendered measurement contract and all hard safety vetoes.

## 1.0.0 — Production roadmap consolidation

- Complete RC11.6D with a confidence-aware hybrid next-track scorer and hard rendered-failure veto.
- Wire the unified AI planner into canonical set optimization and add full-set safety/taste benchmarking.
- Treat the existing TransitionRanker and governed transition policy heads as RC11.8/11.9 production components instead of duplicating parallel systems.
- Enforce account-scoped pseudonymous personalization, separate personalization/training consent, export/delete and production HTTP security defaults.
- Add default-deny model governance requiring rights, held-out, Arena, approval and checkpoint integrity for production neural authority.
- Add 1.0 production audit/release-gate integration and restore lightweight dataset governance files without bundling heavy data.

## 6.0.0 RC11 — Training Intelligence Patch

- Reframed cue bootstrap learning as grouped pairwise ranking.
- Added grid-aligned hard cue negatives with same-track expert-cue collision protection.
- Replaced extreme duration corruptions with harder neighboring alternatives.
- Added balanced classification metrics plus Top-1/Top-3/MRR ranking metrics.
- Persisted full latest bootstrap benchmark through Dataset Builder status and `/lab`.
- Kept all external bootstrap models research-only and non-promotable.


## 6.0.0 RC10 - Dataset Builder

- Add a provenance-aware external dataset source catalog for EDM-CUE, UnmixDB, FMA, ListenBrainz, and Medley2K; all external sources default to the research lane.
- Add checksum-verified configured asset fetching, explicit opt-in for large audio downloads, and fail-closed handling for manual-acquisition sources.
- Add source-specific normalizers into one SAUND.AI record schema with stable grouped train/validation/test splits, label source/confidence, and rights state.
- Hash ListenBrainz user identifiers and interpret adjacent listens only as a research sequence prior, never as technical mixability.
- Add traceable weak-label generation from expert/ground-truth positives plus synthetic cue/duration corruptions with lower confidence weights.
- Add immutable content-addressed bootstrap corpora with per-source/task manifests and production filtering that rejects uncleared external data.
- Add research-only bootstrap training for cue, transition-duration structure, and next-track adjacency priors. Bootstrap artifacts are non-promotable and cannot modify live SAUND.AI models.
- Keep label-derived quantities out of bootstrap model inputs and report held-out accuracy against an explicit majority-class baseline so synthetic weak-label experiments cannot look good by reading the answer from a feature.
- Extend the internal Training Lab with source/model readiness, corpus freezing, bootstrap shadow training, and held-out metrics. Public Create/Studio remains unchanged.
- Exclude downloaded datasets, normalized external records, frozen corpora, and bootstrap model artifacts from release packages.
- Add Dataset Builder regression coverage for rights preservation, user pseudonymization, weak-label provenance, immutable corpus policy, and shadow-only model isolation.

## 6.0.0 RC9 - Training Lab

- Add an internal `/lab` Training Lab for blind A/B transition taste collection without exposing candidate technique or score metadata during voting.
- Add first-train progress, optional musical reason tags, random track-pair sampling, local uploads, dataset status, immutable snapshot creation, and shadow-model training.
- Freeze model-ready datasets under `datasets/training-lab/` with content-addressed track IDs, deduplicated candidate features, rendered measurements, blind labels, session provenance, feature-schema hashes, and compressed pairwise matrices.
- Keep source audio outside dataset snapshots and record explicit rights/provenance readiness separately from research use.
- Train first pairwise rankers into `config/training-lab-models/` as shadow-only artifacts; Lab training never replaces the live transition ranker.
- Preserve hard rendered-audio safety as non-overridable by learned taste.
- Add regression coverage for snapshot immutability, audio exclusion, model-ready matrices, shadow-only training, and first-train readiness.

## 6.0.0 - Autonomous Performance Brain

- Harden the cinematic SAUND.AI entry gate so one authoritative progress clock drives both the visible bottom bar and the disabled Enter state; forced/programmatic clicks cannot bypass incomplete progress.
- Upgrade `TransitionProgram` to executable IR v3 with explicit source buses, automation curves and deterministic processors.
- Route production pair/set transitions through one generic DSP execution engine instead of renderer-side technique strategy branches.
- Add first-class `TransitionProposal` as the musical idea separate from the immutable executable program.
- Upgrade `TrackWorldModel` to 6.0 with phrase roles, tension/energy trajectories, transition readiness and phrase-graph relations.
- Add hierarchical Set Director intent and current-set `DJMemory`, including bounded surprise/risk budget and repetition penalties.
- Add multi-critic rendered judgement for technical, rhythm, harmony, vocals, bass, phrasing, energy and set context.
- Feed critic remedies back into bounded transition retries for surgical duration/family changes instead of blind regeneration.
- Add content-addressed rehearsal-measurement caching keyed by signed execution identity, exact render context and measurement policy, with source revalidation before cache use.
- Add a release-readiness gate that treats execution-integrity regressions as hard blockers and requires real benchmark evidence for broader quality claims.
- Preserve all 5.1 execution-hardening invariants and keep neural authority registry-governed; no fabricated expert-DJ weights are shipped.
- Bump app/plan/decision/planner/set-plan/renderer contracts to 6.0.0 and TransitionProgram to 3.0 while retaining the 5.0 TrackAnalysis/cache schema.
- Verify all 122 tests across 27 files, including real-audio and exact 44.1 kHz final-render integration.

## 5.1.0 - Execution Hardening

- Rehearse set transitions with the exact final sample rate and normalization gains.
- Render-validate deterministic last-resort fallbacks before automatic commitment.
- Bind masters, stems, supplied layers and renderer version into execution signatures.
- Verify exact final transition samples before writing them into the full-set WAV and publish completed WAVs atomically.
- Compare baseline/optimized sets with one canonical objective and report compiled edge totals separately.
- Make explicit neural weight zero mean zero authority.
- Resolve core runtime model/config files through the DJAI workspace environment.
- Budget automatic inner workers against concurrent top-level jobs.
- Restore Pro-mode manual confirmation semantics.
- Avoid expensive rendered-transition embeddings when no active learned ranker can consume them; retain explicit telemetry opt-in.
- Add execution-hardening regression tests and bump app/plan/decision/planner/set-plan versions to 5.1.0 while retaining the 5.0 analysis cache.

## 5.0.0 - Neural DJ Brain

- Add governed neural ears with explicit Beat This! bootstrap, MusicFM/MERT representation adapters, deterministic fallback and no hidden model downloads.
- Add phrase-synchronous `TrackWorldModel` as a stable DJ-specific neural/set-policy state contract.
- Add a versioned model registry with disabled/shadow/advisory/production authority, confidence gates, checkpoint fingerprints and production approval.
- Add trainable structured heads for transition family, duration, stem choreography, rendered DJ preference and next-track value.
- Integrate learned heads as bounded residuals into the existing self-rehearsing transition/set pipeline; hard feasibility and rendered-audio failure remain authoritative.
- Add explicit model status/bootstrap/training/promotion CLI commands and `/api/models`.
- Add Docker `brain` runtime target with source separation + neural dependencies and persistent model caches.
- Add mtime-aware model-registry/head caching so absent/shadow checkpoints add negligible graph-search overhead.
- Reconcile automatic measurement telemetry after global set replanning so superseded locally accepted edges become rejected training evidence.
- Bump analysis/cache/plan/decision/planner/set schemas to 5.0.0 and immutable performance programs to 2.0.
- Do not ship fabricated professional-DJ weights: DJ-specific heads start in shadow until rights-cleared data and held-out/Arena validation earn authority.

## 4.4.0 - Stem Choreography Engine

- Promote separated vocals/drums/bass/other from optional effects to first-class transition decision variables.
- Add `stem_choreography` with progressive-handoff, outgoing-vocal-bridge, instrumental-reset, drum-first-build and bass-ownership-swap profiles.
- Select stem ownership from vocal/bass collision, energy, structure and transition-family context instead of random effect choice.
- Execute immutable per-stem gain curves directly in the deterministic renderer.
- Require complete four-stem sources for a compiled stem-choreography plan and preserve stereo fallback when stems are unavailable.
- Add new technique support to route policy, intervention scoring, learned ranker features, set metadata and API capability reporting.
- Bump plan/decision/planner/set schemas to 4.4.0 and PerformanceProgram to 1.3 while retaining the 4.1 analysis-cache schema.

## 4.2.0 - Automatic Performance Layers + Transition Router

- Make one-full-master upload the normal source contract: automatically cache vocals, drums, bass, other and a derived instrumental.
- Default Docker API/engine images to the Demucs-enabled target and persist model caches in a shared Docker volume.
- Add per-track separation locking so upload background jobs and automatic set jobs cannot duplicate the same expensive separation.
- Make automatic set creation synchronize cached performance material before analysis/planning.
- Add v2 stem manifests with conservative reconstruction/timeline/peak integrity proxies and `quality_confidence`.
- Keep clean supplied originals as the highest-priority optional sources and fill partial bundles from AI separation when appropriate.
- Keep valid stereo masters playable if source separation is unavailable or fails.
- Add an inspectable Transition Router with direct-blend, layer-handoff, breakdown/drop, rhythmic-switch, hard-reset and bridge-preferred families.
- Penalize bridge-preferred directional edges during set ordering so an intermediate track can win instead of forcing a bad direct long blend.
- Carry router fit and layer quality through private rehearsal, professional judgement and immutable performance-program metadata.
- Bump plan/decision/set schemas to 4.2.0 and PerformanceProgram to 1.2 while intentionally retaining the 4.1 analysis cache schema.
- Add regression coverage for one-master AI layers, router classification, source priority and automatic API synchronization.

## 4.1.0 — Supplied Performance Layers

- Add first-class `PerformanceLayers` bundles for full mix + supplied instrumental + supplied vocals/acapella.
- Group conventionally named alternate masters into one library track across web upload, folder discovery and manual set planning.
- Validate duration compatibility and estimate small timeline offsets before using supplied layers; mismatched edits remain separate tracks.
- Give supplied original layers priority over automatic source separation and skip unnecessary Demucs work for complete bundles.
- Add `layer_vocal_bridge` and `layer_instrumental_handoff` transition families with immutable performance-program fingerprints and real DSP rendering.
- Privately rehearse/measure layer-aware candidates through the same professional judge path as ordinary transitions.
- Use supplied acapella activity as authoritative local vocal evidence for cue/transition analysis.
- Add explicit `djai layers-register` for non-standard filenames.
- Extend API/frontend track records with verified performance-layer capability.
- Bump cache/plan/decision/set schemas to 4.1.0 so stale plans cannot ignore the new source contract.

## 4.0.1 — Phrase-Role Mixing Policy Correction

- Removed the universal late/outro exit and early/intro entry bias from cue analysis.
- Added a preferred set mixing window with a soft upper exit boundary instead of sorting every eligible outro first.
- Added mature incoming-cue selection so automatic sets do not restart every next track at time zero when stronger structural cues exist.
- Rebalanced section compatibility and transition-intent priors around groove/build/breakdown/drop relationships.
- Emergency fallback now chooses a structural incoming cue instead of `entry_candidates[0]`.
- Added a conservative untrained-taste style prior: safe foundational blends receive a small prior, creative moves must win clearly, emergency moves remain fallback-biased until validated expert taste earns authority.
- Bumped analysis/cache/planner/decision/set-plan schemas so stale 4.0.0 cue decisions are not reused.

## 4.0.0 — Professional DJ Brain Core

- Rebuild automatic decision making around perception -> state -> imagination -> private rehearsal -> professional judgement -> retry -> immutable execution -> learning.
- Add bar-synchronous TrackMap and multi-scale musical spans.
- Add multidimensional SetState and receding-horizon SetPolicy.
- Add diversity-aware candidate generation and explicit TransitionPolicy.
- Add immutable TransitionProgram fingerprints to execution validation.
- Judge actual rehearsed transition audio with technical, rhythm, groove, harmonic, energy/drop, stem and rendered-embedding evidence.
- Add Professional DJ Judge, uncertainty retries and known-failure memory.
- Add pairwise TransitionRanker v2 and separate expert-imitation prior.
- Add A/B/Tie/Neither active-learning Taste Lab with reason tags.
- Add expert data schemas, synthetic technical-failure generation, staged training curriculum and production/research provenance lanes.
- Add permanent champion-vs-challenger DJAI Arena.
- Add prepared live primary/backup/emergency alternatives.
- Harden Python setup and isolate optional CUE-DETR reference dependencies.

## 3.1.2

- Correct CUE-DETR inference to follow the public ETH-DISCO spectrogram/window/DETR post-processing pipeline.
- Load the Facebook DETR image processor separately from the fine-tuned CUE-DETR checkpoint.
- Resolve CUE-DETR weights to a concrete local Hugging Face snapshot before loading.
- Add missing cue runtime dependencies (`timm`, `matplotlib`, `safetensors`).
- Add Apple MPS → CPU inference fallback.
- Add an optional Docker `cue` target for server-side CUE-DETR inference.
- Align Docker image labels and bootstrap version output with 3.1.2.

## 3.1.1 — EDM-CUE expert cue integration

- Added EDM-CUE annotation import/stats/source tooling.
- Added optional published CUE-DETR inference on authorized local audio.
- Added expert-cue reranking that preserves DJAI beat/downbeat and source-duration safety.
- Added dataset provenance registry and rights-safe external-data workflow.

## 3.1.0 — Elite Brain Foundation

- Added local whole-track and section-level music embeddings.
- Added optional local MERT integration with deterministic fallback.
- Added blind A/B Taste Lab and pairwise preference storage.
- Added grouped-CV TransitionRanker with confidence-scaled authority.
- Added benchmark manifest and brain benchmark tooling.
- Learned taste reranks only technically safe measured candidates.


## 3.0.0 — Automatic product, stems, learning, journeys, live, personalization

- Default listener experience is upload -> progress -> playback.
- Pro controls remain opt-in.
- Automatic fallback removes routine difficult-transition failures from the default path.
- Cached/background/wait Demucs stem workflow with dedicated stem jobs.
- Stem-derived vocal/bass evidence and stem-aware rendering.
- Grouped-CV learned scoring with confidence-scaled authority and automatic refit.
- Implicit feedback and bounded per-listener profiles.
- Target energy arcs for balanced, smooth, build, warm-up, peak, and sunset sets.
- Prepared-plan live manifest and progressive MP3 streaming.

## 2.2.0 — Phase 1 render-and-measure quality loop

- Top-K transition rendering and objective measurement.
- Measured candidate reranking and automatic measurement logs.

## 2.1.0 — Phase 0 performance foundation

- Parallel deterministic analysis and graph planning.
- Reusable pair-evidence matrices.

## 2.0.1 — Stabilized full project

- Unified decision engine and immutable execution plans.
- Clean packaging and updated tests.

## 6.0.0 RC11.1 — held-out ranking correction

- EDM-CUE labeled tracks are deterministically re-split by track into train/validation/test while preserving the upstream split as provenance metadata.
- Transition-duration bootstrap training is now grouped pairwise ranking rather than imbalanced binary classification.
- Duration negatives are closer plausible alternatives (0.75x, 0.875x, 1.125x, 1.25x) and remain explicitly synthetic weak labels.
- Bootstrap model/report schemas advance to 1.2; shadow-only and live-model isolation remain unchanged.
