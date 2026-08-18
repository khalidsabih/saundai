"use client";

import type { SetPlan, SetTransition } from "@/lib/types";
import { Icon } from "./Icon";

function formatDuration(value: number) {
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function label(value?: string | null) {
  return value ? value.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase()) : "Automatic";
}

function directorIntents(plan: SetPlan): string[] {
  const value = plan.metadata?.set_director_story;
  if (!Array.isArray(value)) return [];
  const names = value.map((step) => {
    if (!step || typeof step !== "object") return null;
    const intent = (step as { intent?: unknown }).intent;
    if (!intent || typeof intent !== "object") return null;
    const name = (intent as { name?: unknown }).name;
    return typeof name === "string" ? label(name) : null;
  }).filter((name): name is string => Boolean(name));
  return names.filter((name, index) => index === 0 || name !== names[index - 1]);
}


function timelineStory(track: SetPlan["tracks"][number]) {
  const roles = (track.musical_timeline?.phrases ?? []).map((phrase) => label(phrase.role));
  const distinct = roles.filter((role, index) => index === 0 || role !== roles[index - 1]);
  const visible = distinct.slice(0, 4);
  const suffix = distinct.length > visible.length ? " → …" : "";
  const commitments = track.musical_timeline?.commitments.length ?? 0;
  if (!visible.length) return null;
  return `${visible.join(" → ")}${suffix}${commitments ? ` · ${commitments} protected payoff${commitments === 1 ? "" : "s"}` : ""}`;
}


function objectValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function performanceProof(plan: SetPlan) {
  const metadata = objectValue(plan.metadata);
  const release = objectValue(metadata.release_benchmark);
  const fullSet = objectValue(metadata.full_set_benchmark);
  const complete = objectValue(metadata.complete_move_planner);
  const performance = objectValue(metadata.performance);
  const latency = objectValue(performance.latency);
  const overall = numberValue(release.full_set_overall) ?? numberValue(fullSet.overall);
  const rendered = numberValue(complete.ai_headphones_rendered_candidates);
  const materialized = numberValue(complete.materialized_graph_pairs);
  const possible = numberValue(complete.possible_graph_pairs);
  const memory = numberValue(complete.crate_memory_informed_decisions);
  const totalSeconds = numberValue(release.total_seconds) ?? numberValue(latency.total_delivery_seconds);
  return { overall, rendered, materialized, possible, memory, totalSeconds };
}

function transitionDeckSummary(edge: SetTransition, nativeBpm: number) {
  const plan = edge.plan;
  const matched = plan.effective_bpm_b ?? plan.target_bpm;
  const release = plan.tempo_release_seconds_b ?? 0;
  if (release > 0.05 && Math.abs(nativeBpm - matched) > 0.05) {
    return `${matched.toFixed(1)} → ${nativeBpm.toFixed(1)} BPM`;
  }
  return plan.beatmatched ? `${matched.toFixed(1)} BPM locked` : `${nativeBpm.toFixed(1)} BPM native`;
}

function TransitionBridge({ edge, index, active, toBpm, onClick, playing, preparing }: {
  edge: SetTransition;
  index: number;
  active: boolean;
  toBpm: number;
  onClick: () => void;
  playing: boolean;
  preparing: boolean;
}) {
  const stem = edge.plan.metadata?.stem_choreography;
  const stemProfile = stem && typeof stem === "object" ? (stem as { profile?: unknown }).profile : null;
  return (
    <button
      className={`performance-bridge ${active ? "is-active" : ""}`}
      type="button"
      aria-pressed={active}
      onClick={onClick}
      title={`Inspect transition ${index + 1}`}
    >
      <span className="performance-bridge__line"><i /></span>
      <span className="performance-bridge__copy">
        <small>{String(index + 1).padStart(2, "0")} / {edge.plan.transition_bars} bars</small>
        <strong>{label(edge.plan.technique)}</strong>
        <em>{transitionDeckSummary(edge, toBpm)}</em>
        {typeof stemProfile === "string" && <b>{label(stemProfile)}</b>}
      </span>
      <span className="performance-bridge__listen" aria-hidden="true">
        <Icon name={playing ? "pause" : "headphones"} />
        <small>{preparing ? "Preparing" : playing ? "Playing" : "Listen"}</small>
      </span>
    </button>
  );
}

export function PlanView({
  plan,
  activeTransition,
  onActiveTransition,
  playingTransition = null,
  preparingTransition = null,
}: {
  plan: SetPlan;
  activeTransition: number;
  onActiveTransition: (index: number) => void;
  playingTransition?: number | null;
  preparingTransition?: number | null;
}) {
  const flagged = plan.transitions.filter((edge) => edge.quality !== "good").length;
  const intents = directorIntents(plan);
  const journeyValue = plan.metadata?.journey;
  const journey = typeof journeyValue === "string" ? label(journeyValue) : "Automatic";
  const proof = performanceProof(plan);

  return (
    <section className="panel plan-panel performance-plan" id="route">
      <header className="panel-header performance-plan__header">
        <div className="panel-title"><span className="panel-number">03</span><div><span className="eyebrow">Performance timeline</span><h2>See what SAUND intends to perform</h2></div></div>
        <span className={`status-chip ${flagged ? "review" : "ready"}`}><i />{flagged ? `${flagged} to review` : "Rehearsed"}</span>
      </header>

      <div className="plan-summary-strip performance-summary-strip">
        <span><strong>{plan.order.length}</strong><small>tracks</small></span>
        <span><strong>{formatDuration(plan.estimated_duration_seconds)}</strong><small>estimated</small></span>
        <span><strong>{journey}</strong><small>set direction</small></span>
        <span><strong>{plan.transitions.length}</strong><small>rendered decisions</small></span>
      </div>

      <details className="performance-evidence">
        <summary>
          <span><Icon name="spark" /> Performance evidence</span>
          <small>Quality, AI Headphones, search and memory</small>
          <Icon name="arrow" />
        </summary>
        <div className="performance-proof" aria-label="Performance proof">
          <div>
            <small>set quality</small>
            <strong>{proof.overall !== null ? `${Math.round(proof.overall * 100)}%` : "Pending"}</strong>
          </div>
          <div>
            <small>AI Headphones</small>
            <strong>{proof.rendered !== null ? `${proof.rendered} real previews` : "Not measured"}</strong>
          </div>
          <div>
            <small>search explored</small>
            <strong>{proof.materialized !== null && proof.possible !== null ? `${proof.materialized}/${proof.possible} edges` : "Not measured"}</strong>
          </div>
          <div>
            <small>Crate Memory</small>
            <strong>{proof.memory !== null ? `${proof.memory} informed moves` : "No memory evidence"}</strong>
          </div>
          {proof.totalSeconds !== null && (
            <div>
              <small>delivery time</small>
              <strong>{formatDuration(proof.totalSeconds)}</strong>
            </div>
          )}
        </div>

        {intents.length > 0 && (
          <div className="intent-story" aria-label="Set direction">
            <strong>Set movement</strong>
            <div>{intents.map((intent, index) => <span key={`${intent}-${index}`}>{intent}</span>)}</div>
          </div>
        )}
      </details>

      <div className="performance-timeline" aria-label="Prepared set performance timeline">
        {plan.ordered_tracks.map((track, orderedIndex) => {
          const transition = plan.transitions[orderedIndex];
          const structure = timelineStory(track);
          return (
            <div className="performance-timeline__step" key={track.path}>
              <article className="performance-track-card">
                <span className="performance-track-card__index">{String(orderedIndex + 1).padStart(2, "0")}</span>
                <div>
                  <strong>{track.title}</strong>
                  <small>{track.bpm.toFixed(1)} BPM · {track.key.camelot} · energy {Math.round(track.energy * 100)}</small>
                  {structure && <small>Structure: {structure}</small>}
                </div>
                <span className="performance-track-card__state">{orderedIndex === 0 ? "Deck A" : "Deck B / owner"}</span>
              </article>
              {transition && (
                <TransitionBridge
                  edge={transition}
                  index={orderedIndex}
                  active={activeTransition === orderedIndex}
                  toBpm={plan.tracks[transition.to_index].bpm}
                  onClick={() => onActiveTransition(orderedIndex)}
                  playing={playingTransition === orderedIndex}
                  preparing={preparingTransition === orderedIndex}
                />
              )}
            </div>
          );
        })}
      </div>

      <div className="performance-plan__legend">
        <span><Icon name="pulse" /> BPM + phase</span>
        <span><Icon name="waveform" /> stem ownership</span>
        <span><Icon name="shield" /> rendered safety</span>
        <small>Click a transition to hear it. Open “Why this transition?” only when you want the deeper deck evidence.</small>
      </div>
    </section>
  );
}
