"use client";

import type { CSSProperties } from "react";
import type { SetPlan } from "@/lib/types";
import { AudioPreview } from "./AudioPreview";
import { Icon } from "./Icon";

function label(value?: string | null) {
  return value ? value.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase()) : "Automatic";
}

function criticData(metadata: Record<string, unknown> | undefined) {
  const raw = metadata?.critic_ensemble;
  if (!raw || typeof raw !== "object") return { failures: [] as string[], remedies: [] as string[], intent: null as string | null };
  const value = raw as { failure_domains?: unknown; remedies?: unknown; set_intent?: unknown };
  const failures = Array.isArray(value.failure_domains) ? value.failure_domains.filter((x): x is string => typeof x === "string") : [];
  const remedies = Array.isArray(value.remedies) ? value.remedies.filter((x): x is string => typeof x === "string") : [];
  let intent: string | null = null;
  if (value.set_intent && typeof value.set_intent === "object") {
    const name = (value.set_intent as { name?: unknown }).name;
    if (typeof name === "string") intent = name;
  }
  return { failures, remedies, intent };
}

function objectValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? value as Record<string, unknown> : {};
}

function numeric(value: unknown, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function pct(value: unknown) {
  return `${Math.round(numeric(value) * 100)}%`;
}

function stemProfileStory(profile: string | null) {
  if (profile === "deck_vocal_exchange") return [
    ["Drums", "B establishes under A"],
    ["Vocals", "A clears before B lead arrives"],
    ["Bass", "single-owner handoff"],
    ["Music", "A bed yields to B"],
  ] as const;
  if (profile === "outgoing_vocal_bridge") return [
    ["Drums", "B establishes first"], ["Vocals", "A bridges, then B reveals"], ["Bass", "ownership swaps once"], ["Music", "B bed takes over"],
  ] as const;
  if (profile === "instrumental_reset") return [
    ["Drums", "groove exchange"], ["Vocals", "clear through the middle"], ["Bass", "single-owner swap"], ["Music", "instrumental reset"],
  ] as const;
  if (profile === "drum_first_build") return [
    ["Drums", "B enters early"], ["Vocals", "B waits for payoff"], ["Bass", "swap near pivot"], ["Music", "B builds into ownership"],
  ] as const;
  if (profile === "bass_ownership_swap") return [
    ["Drums", "groove continuity"], ["Vocals", "late ownership handoff"], ["Bass", "one owner at a time"], ["Music", "balanced exchange"],
  ] as const;
  return [
    ["Drums", "progressive handoff"], ["Vocals", "incoming lead waits"], ["Bass", "single-owner handoff"], ["Music", "progressive exchange"],
  ] as const;
}

function laneStyle(profile: string | null, role: string, side: "a" | "b") {
  const maps: Record<string, Record<string, [number, number]>> = {
    deck_vocal_exchange: {
      "Drums-a": [0, 72], "Drums-b": [6, 100],
      "Vocals-a": [0, 52], "Vocals-b": [60, 100],
      "Bass-a": [0, 56], "Bass-b": [54, 100],
      "Music-a": [0, 84], "Music-b": [16, 100],
    },
    outgoing_vocal_bridge: {
      "Drums-a": [0, 48], "Drums-b": [10, 100],
      "Vocals-a": [0, 90], "Vocals-b": [86, 100],
      "Bass-a": [0, 54], "Bass-b": [52, 100],
      "Music-a": [0, 64], "Music-b": [24, 100],
    },
  };
  const range = maps[profile ?? ""]?.[`${role}-${side}`] ?? (side === "a" ? [0, 70] : [20, 100]);
  return { "--lane-start": `${range[0]}%`, "--lane-end": `${range[1]}%` } as CSSProperties;
}

export function TransitionInspector({ plan, index, previewUrl }: { plan: SetPlan; index: number; previewUrl?: string; previewReportPath?: string }) {
  const edge = plan.transitions[index];
  if (!edge) return null;
  const from = plan.tracks[edge.from_index];
  const to = plan.tracks[edge.to_index];
  const critic = criticData(edge.plan.metadata);
  const warnings = Array.from(new Set([...edge.warnings, ...(edge.plan.warning ? [edge.plan.warning] : [])].filter(Boolean).map(String)));
  const rationale = Array.from(new Set((edge.plan.rationale?.length ? edge.plan.rationale : edge.reasons).filter(Boolean))).slice(0, 4);
  const needsReview = edge.quality !== "good" || critic.failures.length > 0 || warnings.length > 0;
  const deck = objectValue(edge.plan.metadata?.deck_performance);
  const stem = objectValue(edge.plan.metadata?.stem_choreography);
  const bootstrap = objectValue(edge.plan.metadata?.bootstrap_learning);
  const cuePrior = objectValue(bootstrap.cue_point_prior);
  const durationSearch = objectValue(edge.plan.metadata?.duration_search);
  const durationBootstrap = objectValue(durationSearch.bootstrap_policy);
  const djMove = objectValue(edge.plan.metadata?.dj_move);
  const projectedState = objectValue(djMove.projected_state);
  const moveComponents = objectValue(djMove.components);
  const stemProfile = typeof stem.profile === "string" ? stem.profile : null;
  const matchedBpm = edge.plan.effective_bpm_b ?? edge.plan.target_bpm;
  const nativeBpm = edge.plan.post_transition_bpm_b ?? to.bpm;
  const releaseSeconds = edge.plan.tempo_release_seconds_b ?? numeric(deck.tempo_release_seconds_b);
  const phaseNudgeMs = (edge.plan.phase_nudge_seconds_b ?? numeric(deck.phase_nudge_seconds_b)) * 1000;
  const learnedCueWeight = numeric(cuePrior.effective_weight);
  const learnedDurationAvailable = durationBootstrap.available === true;
  const moveWaitBars = numeric(djMove.wait_bars);
  const moveExitBar = numeric(djMove.exit_bar, -1);
  const moveEntryBar = numeric(djMove.entry_bar, -1);
  const moveFuture = numeric(djMove.future_optionality);
  const projectedBar = numeric(projectedState.source_bar, -1);
  const projectedBpm = numeric(projectedState.effective_bpm, nativeBpm);
  const moveAvailable = typeof djMove.version === "string";

  return (
    <section className="panel inspector-panel performance-inspector" id="transition-detail">
      <header className="panel-header">
        <div className="panel-title"><span className="panel-number">04</span><div><span className="eyebrow">Transition {index + 1} / deck performance</span><h2>{from.title} <span>→</span> {to.title}</h2></div></div>
        <span className={`status-chip ${needsReview ? "review" : "ready"}`}><i />{needsReview ? "Review" : "Rehearsed"}</span>
      </header>

      <div className="transition-decision transition-decision--performance">
        <div><span>Technique</span><strong>{label(edge.plan.technique)}</strong></div>
        <div><span>Length</span><strong>{edge.plan.transition_bars} bars · {edge.plan.transition_seconds.toFixed(1)}s</strong></div>
        <div><span>Tempo handoff</span><strong>{releaseSeconds > 0.05 ? `${matchedBpm.toFixed(1)} → ${nativeBpm.toFixed(1)} BPM` : edge.plan.beatmatched ? `${matchedBpm.toFixed(1)} BPM locked` : "Native"}</strong></div>
        <div><span>Purpose</span><strong>{label(critic.intent ?? edge.plan.intent ?? edge.plan.style)}</strong></div>
      </div>

      <div className="deck-performance-board">
        <div className="deck-performance-board__head">
          <div><span className="eyebrow">Deck behavior</span><h3>SAUND is performing the handoff, not just fading tracks.</h3></div>
          <span className="deck-owner-chip">A → B tempo ownership</span>
        </div>

        <div className="deck-pair">
          <article className="deck-card deck-card--a">
            <span>DECK A / OUT</span>
            <strong>{from.title}</strong>
            <div><b>{from.bpm.toFixed(1)}</b><small>BPM native</small></div>
            <em>{from.key.camelot}</em>
          </article>
          <div className="deck-lock">
            <Icon name="pulse" />
            <strong>{edge.plan.beatmatched ? "Beat locked" : "Structural handoff"}</strong>
            <small>{Math.abs(phaseNudgeMs) >= 4 ? `${phaseNudgeMs > 0 ? "+" : ""}${phaseNudgeMs.toFixed(0)} ms phase nudge` : "phase already aligned"}</small>
          </div>
          <article className="deck-card deck-card--b">
            <span>DECK B / IN</span>
            <strong>{to.title}</strong>
            <div><b>{matchedBpm.toFixed(1)}</b><small>BPM during blend</small></div>
            <em>{to.key.camelot}</em>
          </article>
        </div>

        <div className="tempo-ownership-rail" aria-label="Tempo ownership transfer">
          <div className="tempo-ownership-rail__labels"><span>A owns tempo</span><span>handoff</span><span>B owns tempo</span></div>
          <div className="tempo-ownership-rail__line"><i /><b /><i /></div>
          <div className="tempo-ownership-rail__values">
            <strong>{from.bpm.toFixed(1)}</strong>
            <span>{matchedBpm.toFixed(1)} BPM blend</span>
            <strong>{releaseSeconds > 0.05 ? `${nativeBpm.toFixed(1)} native` : `${matchedBpm.toFixed(1)} held`}</strong>
          </div>
          {releaseSeconds > 0.05 && <small>Pitch-preserving release over {releaseSeconds.toFixed(1)} seconds after Deck B takes control.</small>}
        </div>
      </div>

      <div className={`stem-performance ${stemProfile ? "has-stems" : "is-master"}`}>
        <div className="stem-performance__head">
          <div><span className="eyebrow">Musical ownership</span><h3>{stemProfile ? label(stemProfile) : "Master-deck handoff"}</h3></div>
          <span>{stemProfile ? `stem confidence ${pct(stem.quality_confidence)}` : "full-master fallback"}</span>
        </div>
        {stemProfile ? (
          <div className="stem-lanes">
            {stemProfileStory(stemProfile).map(([role, story]) => (
              <div className="stem-lane" key={role}>
                <span>{role}</span>
                <div className="stem-lane__rail">
                  <i className="stem-lane__a" style={laneStyle(stemProfile, role, "a")} />
                  <i className="stem-lane__b" style={laneStyle(stemProfile, role, "b")} />
                </div>
                <small>{story}</small>
              </div>
            ))}
          </div>
        ) : (
          <p className="stem-performance__fallback">This transition is using the full masters. SAUND only exposes vocal/bass/drum ownership when trustworthy separated stems actually exist.</p>
        )}
      </div>

      {moveAvailable && (
        <div className="learned-decision-strip dj-move-strip">
          <span className="eyebrow">Complete DJ move</span>
          <div>
            <span><strong>{moveWaitBars >= 4 ? `Wait ${Math.round(moveWaitBars)} bars` : "Move on this opportunity"}</strong><small>{moveExitBar >= 0 && moveEntryBar >= 0 ? `A bar ${Math.round(moveExitBar)} → B bar ${Math.round(moveEntryBar)}` : "bar-native transition plan"}</small></span>
            <span><strong>{djMove.payoff_satisfied === true ? "Payoff respected" : djMove.protected_payoff_violation === true ? (["breakdown_to_drop", "dramatic_switch"].includes(typeof djMove.intent === "string" ? djMove.intent : "") ? "Payoff replacement" : "Protected moment warning") : "Phrase-safe handoff"}</strong><small>{label(typeof djMove.action === "string" ? djMove.action : "transition")} · phrase fit {pct(moveComponents.phrase_fit)}</small></span>
            <span><strong>{Math.round(moveFuture * 100)}% future optionality</strong><small>{projectedBar >= 0 ? `lands around B bar ${Math.round(projectedBar)} at ${projectedBpm.toFixed(1)} BPM` : "post-move state projected"}</small></span>
          </div>
        </div>
      )}

      <div className="decision-checks decision-checks--intelligence" aria-label="Transition checks">
        <span><Icon name="check" /><strong>Phrase + timing checked</strong></span>
        <span><Icon name="check" /><strong>Low end checked</strong></span>
        <span><Icon name="check" /><strong>Vocals checked</strong></span>
        <span><Icon name="shield" /><strong>Rendered before selection · hard safety authoritative</strong></span>
      </div>

      {(learnedCueWeight > 0 || learnedDurationAvailable) && (
        <div className="learned-decision-strip">
          <span className="eyebrow">Learned signals used</span>
          <div>
            {learnedCueWeight > 0 && <span><strong>Cue prior</strong><small>{Math.round(learnedCueWeight * 100)}% bounded influence on this cue ranking</small></span>}
            {learnedDurationAvailable && <span><strong>Duration prior</strong><small>reordered executable duration candidates before rehearsal</small></span>}
          </div>
        </div>
      )}

      {previewUrl ? (
        <div className="preview-card">
          <div className="section-heading"><span><Icon name="headphones" /> Hear this performance</span><small>Prepared transition + deck release context</small></div>
          <AudioPreview url={previewUrl} transitionStart={8} transitionEnd={8 + edge.plan.transition_seconds + Math.min(releaseSeconds, 8)} label={`Transition ${index + 1}`} />
        </div>
      ) : (
        <div className="preview-callout"><Icon name="headphones" /><span><strong>Hear what SAUND rehearsed.</strong><small>Create previews in Finish below.</small></span></div>
      )}

      <div className="decision-explanation">
        <div>
          <span className="eyebrow">Why SAUND chose it</span>
          {rationale.length ? rationale.map((reason, idx) => <p key={`${reason}-${idx}`}>{reason}</p>) : <p>Chosen as the strongest rehearsed path between these tracks.</p>}
        </div>

        {(warnings.length > 0 || critic.remedies.length > 0) && (
          <div className="decision-attention">
            <span className="eyebrow">What to know</span>
            {warnings.slice(0, 3).map((warning) => <p key={warning}><Icon name="warning" /> {warning}</p>)}
            {critic.remedies.slice(0, 2).map((remedy) => <p key={remedy}>{label(remedy)}</p>)}
          </div>
        )}
      </div>
    </section>
  );
}
