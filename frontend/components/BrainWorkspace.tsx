"use client";

import { useEffect, useMemo, useState } from "react";
import { getBrainStatus } from "@/lib/api";
import type { AutomaticSetResult, BrainStatus, JourneyMode, MusicalTimelinePhrase } from "@/lib/types";
import { Icon } from "./Icon";

function human(value: unknown, fallback = "—") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function booleanState(value: unknown, on = "Active", off = "Off") {
  return value ? on : off;
}

function count(value: unknown) {
  const number = Number(value ?? 0);
  return Number.isFinite(number) ? new Intl.NumberFormat("en-US").format(number) : "0";
}

function percentage(value: unknown, fallback = "—") {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 100)}%` : fallback;
}

function directionLabel(journey: JourneyMode) {
  return ({ balanced: "Flow", warmup: "Warm", peak: "Drive", sunset: "Release", smooth: "Smooth", build: "Build" } as const)[journey] ?? journey;
}

function songRoleLabel(phrase: MusicalTimelinePhrase) {
  const role = phrase.song_role || phrase.role || phrase.section || "phrase";
  const index = phrase.song_role_index || 0;
  const numbered = ["verse", "hook"].includes(role) && index > 0;
  return `${human(role)}${numbered ? ` ${index}` : ""}`;
}

export function BrainWorkspace({
  result,
  journey,
  onBack,
  onOpenStudio,
}: {
  result: AutomaticSetResult | null;
  journey: JourneyMode;
  onBack: () => void;
  onOpenStudio: () => void;
}) {
  const [brain, setBrain] = useState<BrainStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const status = await getBrainStatus();
        if (!active) return;
        setBrain(status);
        setError(null);
      } catch (caught) {
        if (!active) return;
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 10_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const executive = brain?.dj_executive;
  const evidence = brain?.taste_learning?.runtime_evidence;
  const memory = brain?.crate_memory;
  const timelineStatus = brain?.musical_timeline;
  const mixingStatus = brain?.professional_mixing;
  const mixContract = brain?.professional_mix_contract;
  const structuralStatus = brain?.structural_operations;
  const modelRows = useMemo(() => Object.entries(brain?.bootstrap_runtime?.models ?? {}), [brain]);
  const availableModels = modelRows.filter(([, model]) => model.available).length;
  const learnedAuthority = brain?.bootstrap_runtime?.mode ?? "off";
  const researchShadowModels = modelRows.filter(([, model]) => model.available && model.research_shadow);
  const systemReady = Boolean(executive?.enabled && executive?.single_commit_authority);
  const learnedTaste = Boolean(
    brain?.taste_learning?.global_taste_ranker?.available
    || brain?.taste_learning?.rendered_dj_judge?.available
    || brain?.taste_learning?.state_move_taste?.available,
  );
  const currentHorizon = useMemo(() => result?.plan.ordered_tracks.slice(0, 3) ?? [], [result]);
  const currentTimeline = result?.plan.ordered_tracks[0]?.musical_timeline;
  const currentSongForm = currentTimeline?.phrases.slice(0, 14) ?? [];

  return (
    <main className="brain-system-shell">
      <header className="autonomous-topbar brain-system-topbar">
        <button type="button" className="autonomous-brand autonomous-brand--button" onClick={onBack} aria-label="Back to Perform">
          <span className="saund-wordmark"><strong>SAUND</strong><i>.</i><strong>AI</strong></span>
          <small>Autonomous DJ</small>
        </button>
        <nav className="autonomous-nav" aria-label="Primary">
          <button type="button" onClick={onBack}>Perform</button>
          <span className="is-active">Brain</span>
          <a href="/readiness">Readiness</a>
          <a href="/lab">Lab</a>
        </nav>
        <div className="autonomous-topbar__actions">
          <span className={`autonomous-health ${systemReady ? "" : "is-muted"}`}><i /> {systemReady ? "System online" : brain ? "Limited status" : "Reading system"}</span>
        </div>
      </header>

      {error && (
        <div className="brain-system-alert" role="alert">
          <Icon name="warning" />
          <span><strong>Brain status is temporarily unavailable.</strong><small>{error}</small></span>
        </div>
      )}

      <section className="brain-system">
        <header className="brain-system__hero brain-system__hero--data">
          <div>
            <span className="autonomous-kicker">THE INTELLIGENCE BEHIND THE DJ</span>
            <h1>See what SAUND<br />actually knows.</h1>
          </div>
          <div className="brain-system__hero-status">
            <p>The performance surface stays calm. Brain shows the real planning, listening, memory, musical-structure and authority state underneath it.</p>
            <span><i className={brain ? "is-live" : ""} /> {brain ? `Brain ${brain.version}` : "Loading runtime evidence"}</span>
          </div>
        </header>

        <section className="brain-live-facts" aria-label="Live brain facts">
          <article>
            <span>RUNTIME EVIDENCE</span>
            <strong>{brain ? count(evidence?.events) : "—"}</strong>
            <p>decision events observed</p>
            <small>{count(evidence?.candidates)} candidates · {count(evidence?.selected)} selected</small>
          </article>
          <article>
            <span>AI HEADPHONES</span>
            <strong>{brain ? count(evidence?.modes?.ai_headphones) : "—"}</strong>
            <p>private rehearsal decisions</p>
            <small>{count(evidence?.hard_failed)} hard failures rejected</small>
          </article>
          <article>
            <span>CRATE MEMORY</span>
            <strong>{brain ? count(memory?.track_records) : "—"}</strong>
            <p>tracks with remembered facts</p>
            <small>{count(memory?.preference_events)} human preference events</small>
          </article>
          <article>
            <span>LEARNED MODELS</span>
            <strong>{brain ? `${availableModels}/${modelRows.length}` : "—"}</strong>
            <p>bootstrap intelligence available</p>
            <small>{human(brain?.bootstrap_runtime?.mode, "loading")} authority</small>
          </article>
        </section>

        <section className="shadow-authority-panel" aria-label="Learned model authority">
          <header>
            <div><span>LEARNED AI AUTHORITY</span><strong>{human(learnedAuthority)}</strong></div>
            <b>{learnedAuthority === "production" ? "LIVE" : learnedAuthority === "advisory" ? "BOUNDED" : "NO DECISION AUTHORITY"}</b>
          </header>
          <div className="shadow-authority-panel__flow">
            <span><small>1</small><strong>Candidate generation</strong><em>Deterministic + governed</em></span>
            <i>→</i>
            <span><small>2</small><strong>Learned ranking</strong><em>{researchShadowModels.length ? researchShadowModels.length + " shadow model" + (researchShadowModels.length === 1 ? "" : "s") : "No research model loaded"}</em></span>
            <i>→</i>
            <span><small>3</small><strong>Executive</strong><em>Single commit authority</em></span>
            <i>→</i>
            <span><small>4</small><strong>Safety</strong><em>Non-overridable</em></span>
          </div>
          <footer>
            <span>Research predictions may be observed without reviving rejected candidates or bypassing rendered-audio gates.</span>
            <strong>{brain?.taste_learning?.state_move_taste?.authority ? percentage(brain.taste_learning.state_move_taste.authority) : "0% taste authority"}</strong>
          </footer>
        </section>

        <section className="brain-information-grid" aria-label="Brain intelligence status">
          <article className="brain-information-card">
            <header><Icon name="route" /><span>PLANNING</span><b>{brain?.professional_brain_enabled ? "ONLINE" : "BASELINE"}</b></header>
            <h2>Professional decision stack</h2>
            <dl>
              <div><dt>Analytical pool</dt><dd>{count(brain?.brain_config?.analytical_pool)}</dd></div>
              <div><dt>Finalists listened</dt><dd>{count(brain?.brain_config?.finalists)}</dd></div>
              <div><dt>Retry rounds</dt><dd>{count(brain?.brain_config?.retry_rounds)}</dd></div>
              <div><dt>Complete DJMove</dt><dd>{booleanState(brain?.dj_move_brain?.enabled)}</dd></div>
              <div><dt>Lookahead / optionality</dt><dd>{booleanState(brain?.dj_move_brain?.future_optionality)}</dd></div>
              <div><dt>Minimum body</dt><dd>{brain?.dj_move_brain?.minimum_body_bars ? `${brain.dj_move_brain.minimum_body_bars} bars` : "—"}</dd></div>
              <div><dt>Large crate threshold</dt><dd>{brain?.large_crate_planning?.threshold_tracks ? `${brain.large_crate_planning.threshold_tracks} tracks` : "—"}</dd></div>
            </dl>
          </article>

          <article className="brain-information-card is-musical">
            <header><Icon name="waveform" /><span>MUSICAL UNDERSTANDING</span><b>{human(timelineStatus?.schema_version, "—")}</b></header>
            <h2>Phrase + song-form model</h2>
            <dl>
              <div><dt>Canonical timeline</dt><dd>{booleanState(timelineStatus?.canonical_phrase_authority)}</dd></div>
              <div><dt>Song-form roles</dt><dd>{timelineStatus?.song_form_roles?.length ? `${timelineStatus.song_form_roles.length} roles` : "Coarse only"}</dd></div>
              <div><dt>First body protected</dt><dd>{booleanState(timelineStatus?.first_body_protection, "Protected", "No")}</dd></div>
              <div><dt>First hook promise</dt><dd>{booleanState(timelineStatus?.first_hook_pre_entry_protection, "Protected through Hook 1", "No")}</dd></div>
              <div><dt>Hard song-form guard</dt><dd>{booleanState(timelineStatus?.song_form_hard_guard, "Authoritative", "No")}</dd></div>
              <div><dt>Hooks protected</dt><dd>{booleanState(timelineStatus?.hook_protection, "Protected", "No")}</dd></div>
              <div><dt>Timeline cache</dt><dd>{count(timelineStatus?.cache_entries)} / {count(timelineStatus?.cache_limit)}</dd></div>
              <div><dt>Payoff protection</dt><dd>{percentage(brain?.dj_move_brain?.payoff_protection)}</dd></div>
              <div><dt>Professional mix families</dt><dd>{mixingStatus?.families?.length ? `${mixingStatus.families.length} active` : "—"}</dd></div>
              <div><dt>Musical ownership</dt><dd>{mixingStatus?.outgoing_owner_until_incoming_established ? "A → B guarded" : "—"}</dd></div>
              <div><dt>Bass ownership</dt><dd>{booleanState(mixingStatus?.single_bass_owner, "Single owner", "Legacy")}</dd></div>
              <div><dt>Reset effects</dt><dd>{mixingStatus?.reset_is_last_resort ? "Last resort" : "Legacy"}</dd></div>
            </dl>
          </article>

          <article className="brain-information-card">
            <header><Icon name="headphones" /><span>LISTENING + LEARNING</span><b>{learnedTaste ? "LEARNED" : "SHADOW"}</b></header>
            <h2>Private rehearsal and taste</h2>
            <dl>
              <div><dt>Gate passed</dt><dd>{count(evidence?.gate_passed)}</dd></div>
              <div><dt>Preference ranker</dt><dd>{brain?.preference_ranker?.ranker_loaded ? `Live · ${percentage(brain.preference_ranker.authority)}` : "Not loaded"}</dd></div>
              <div><dt>Global taste</dt><dd>{brain?.taste_learning?.global_taste_ranker?.available ? "Available" : "Shadow"}</dd></div>
              <div><dt>State → move taste</dt><dd>{brain?.taste_learning?.state_move_taste?.available ? "Available" : "Shadow"}</dd></div>
              <div><dt>Rendered judge</dt><dd>{brain?.taste_learning?.rendered_dj_judge?.available ? "Available" : "Shadow"}</dd></div>
              <div><dt>Pair memories</dt><dd>{count(memory?.pair_records)}</dd></div>
            </dl>
          </article>

          <article className="brain-information-card">
            <header><Icon name="shield" /><span>AUTHORITY + RECOVERY</span><b>{systemReady ? "NOMINAL" : "CHECK"}</b></header>
            <h2>One writer. Hard safety.</h2>
            <dl>
              <div><dt>Executive</dt><dd>{executive?.single_commit_authority ? "Single writer" : "—"}</dd></div>
              <div><dt>FUTURE replanning</dt><dd>{booleanState(executive?.continuous_live_replanning)}</dd></div>
              <div><dt>Prepared recovery</dt><dd>{booleanState(executive?.prepared_recovery_capable, "Ready", "Unavailable")}</dd></div>
              <div><dt>Rejected patterns</dt><dd>{count(brain?.rejection_memory_count)}</dd></div>
              <div><dt>Audio safety</dt><dd>{brain?.deck_performance?.hard_audio_safety_authoritative ? "Authoritative" : "—"}</dd></div>
              <div><dt>Mix contract</dt><dd>{mixContract?.combined_audio_quality_alone_is_insufficient ? "Deck-aware" : "Legacy"}</dd></div>
              <div><dt>Intro runway</dt><dd>{structuralStatus?.automatic_intro_extension ? "Auto-extend" : "Manual"}</dd></div>
              <div><dt>Production data</dt><dd>{brain?.production_data_ready ? "Ready" : "Blocked"}</dd></div>
            </dl>
          </article>
        </section>

        {result && currentTimeline && (
          <section className="brain-song-form" aria-label="Current track song form">
            <header>
              <div><span>CURRENT TRACK · SONG FORM</span><strong>{result.plan.ordered_tracks[0]?.title ?? "Opening track"}</strong></div>
              <small>Derived from the canonical phrase timeline · confidence stays visible</small>
            </header>
            <div className="brain-song-form__rail">
              {currentSongForm.map((phrase) => (
                <div key={phrase.phrase_id} className={`brain-song-form__phrase is-${phrase.song_role || phrase.role}`}>
                  <span>{songRoleLabel(phrase)}</span>
                  <strong>{phrase.start_bar}–{phrase.end_bar}</strong>
                  <small>{percentage(phrase.song_form_confidence ?? phrase.transition_readiness)}</small>
                </div>
              ))}
            </div>
            <footer>
              <span>Protected commitments</span>
              <strong>{currentTimeline.commitments.length}</strong>
              <p>{currentTimeline.commitments.map((item) => human(item.kind)).join(" · ") || "No protected payoff in the visible opening horizon."}</p>
            </footer>
          </section>
        )}

        <section className="brain-architecture" aria-label="SAUND autonomous architecture">
          <article className="brain-architecture__node is-brain">
            <header><span>01</span><small>BRAIN</small></header>
            <h2>Imagines.</h2>
            <p>Understands music, proposes complete DJ moves, looks ahead and listens privately before anything is committed.</p>
            <div className="brain-architecture__signals">
              <span>Set Director</span><span>DJMove</span><span>Lookahead</span><span>AI Headphones</span><span>Professional Brain</span>
            </div>
            <footer><i /> {booleanState(brain?.dj_move_brain?.enabled, "Musical reasoning active", "Reasoning unavailable")}</footer>
          </article>
          <div className="brain-architecture__arrow" aria-hidden="true"><Icon name="arrow" /></div>
          <article className="brain-architecture__node is-executive">
            <header><span>02</span><small>EXECUTIVE</small></header>
            <h2>Chooses.</h2>
            <p>The single writer of authoritative DJ state. NOW and NEXT stay protected while FUTURE may be reconsidered.</p>
            <div className="brain-architecture__authority">
              <span><b>NOW</b><i />LOCKED</span>
              <span><b>NEXT</b><i />LOCKED</span>
              <span><b>FUTURE</b><i className="is-fluid" />FLUID</span>
            </div>
            <footer><i /> {executive?.single_commit_authority ? "Single commit authority" : "Authority not confirmed"}</footer>
          </article>
          <div className="brain-architecture__arrow" aria-hidden="true"><Icon name="arrow" /></div>
          <article className="brain-architecture__node is-safety">
            <header><span>03</span><small>SAFETY</small></header>
            <h2>Protects.</h2>
            <p>Checks execution contracts and can reject unsafe state. Musical intelligence cannot bypass hard runtime safety.</p>
            <div className="brain-architecture__signals"><span>Commit validation</span><span>State integrity</span><span>Recovery ready</span><span>Fail closed</span></div>
            <footer><i /> Governed execution</footer>
          </article>
          <div className="brain-architecture__arrow" aria-hidden="true"><Icon name="arrow" /></div>
          <article className="brain-architecture__node is-executor">
            <header><span>04</span><small>EXECUTOR</small></header>
            <h2>Performs.</h2>
            <p>Deterministic deck and DSP execution. It does not choose tracks, techniques, musical intent or set direction.</p>
            <div className="brain-architecture__signals"><span>Load</span><span>Body</span><span>Transition</span><span>Tempo release</span><span>Verify</span></div>
            <footer><i /> {human(executive?.executor_contract, "Validated commands only")}</footer>
          </article>
        </section>

        <section className="brain-loop" aria-label="Executive decision loop">
          <header>
            <div><span>DECISION LOOP</span><strong>Reason → listen → commit → perform</strong></div>
            <small>{executive?.continuous_live_replanning ? "Continuous FUTURE control enabled" : "Prepared control"}</small>
          </header>
          <div className="brain-loop__rail">
            {["Observe", "Direct", "Search", "Audition", "Validate", "Commit", "Execute"].map((step, index) => (
              <div key={step}><span>{String(index + 1).padStart(2, "0")}</span><strong>{step}</strong><i /></div>
            ))}
          </div>
        </section>

        <section className="brain-system__lower">
          <div className="brain-status-grid">
            <article>
              <span>EXECUTIVE</span><strong>{human(executive?.mode, "Loading")}</strong>
              <dl>
                <div><dt>Authority</dt><dd>{executive?.single_commit_authority ? "Single writer" : "—"}</dd></div>
                <div><dt>Live replanning</dt><dd>{booleanState(executive?.continuous_live_replanning)}</dd></div>
                <div><dt>Recovery</dt><dd>{booleanState(executive?.prepared_recovery_capable, "Prepared", "Unavailable")}</dd></div>
              </dl>
            </article>
            <article>
              <span>ADAPTATION</span><strong>{executive?.crowd_control_enabled ? "Bounded live" : "Observe"}</strong>
              <dl>
                <div><dt>Crowd</dt><dd>{human(executive?.crowd_state_authority, "—")}</dd></div>
                <div><dt>Transition policy</dt><dd>{executive?.transition_policy_enabled ? "Bounded advisory" : "Observe"}</dd></div>
                <div><dt>Learned taste</dt><dd>{learnedTaste ? "Available" : "Baseline"}</dd></div>
              </dl>
            </article>
            <article>
              <span>PERFORMANCE</span><strong>{brain?.professional_brain_enabled ? "Professional Brain active" : "Baseline"}</strong>
              <dl>
                <div><dt>Tempo release</dt><dd>{booleanState(brain?.deck_performance?.tempo_release_enabled)}</dd></div>
                <div><dt>Phase nudge</dt><dd>{booleanState(brain?.deck_performance?.phase_nudge_enabled)}</dd></div>
                <div><dt>Hard safety</dt><dd>{brain?.deck_performance?.hard_audio_safety_authoritative ? "Authoritative" : "—"}</dd></div>
              </dl>
            </article>
          </div>

          <aside className="brain-session-context">
            <header><span>{result ? "CURRENT SESSION" : "SESSION CONTEXT"}</span><b>{directionLabel(journey)}</b></header>
            {result ? (
              <>
                <div className="brain-session-context__route">
                  {currentHorizon.map((track, index) => (
                    <div key={track.path}>
                      <small>{index === 0 ? "NOW" : index === 1 ? "NEXT · LOCKED" : "FUTURE"}</small>
                      <strong>{track.title}</strong><span>{track.bpm.toFixed(1)} BPM</span>
                    </div>
                  ))}
                </div>
                <p>{result.plan.transitions.length} committed handoff{result.plan.transitions.length === 1 ? "" : "s"} · {result.plan.order.length} tracks in the prepared route.</p>
              </>
            ) : (
              <div className="brain-session-context__empty">
                <Icon name="waveform" /><strong>No session is armed.</strong>
                <p>Start from Perform. The Brain becomes session-specific once the crate and direction are known.</p>
              </div>
            )}
          </aside>
        </section>

        {brain && (
          <details className="brain-raw-status">
            <summary><span>TECHNICAL STATUS</span><strong>Inspect complete /api/brain/status payload</strong><Icon name="arrow" /></summary>
            <pre>{JSON.stringify(brain, null, 2)}</pre>
          </details>
        )}

        <section className="brain-advanced">
          <div>
            <span>ADVANCED WORKSPACE</span><h2>Studio is a tool, not the Brain.</h2>
            <p>Use Studio when you want to collaborate with the intelligence, audition alternatives, branch routes and make guided changes.</p>
          </div>
          <button type="button" onClick={onOpenStudio}>
            <span><small>Human × Executive</small><strong>Open Studio</strong></span><Icon name="arrow" />
          </button>
        </section>
      </section>
    </main>
  );
}
