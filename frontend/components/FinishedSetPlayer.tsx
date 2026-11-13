"use client";

import { useState } from "react";
import { mediaUrl, sendFeedback } from "@/lib/api";
import type { AutomaticSetResult, JourneyMode } from "@/lib/types";
import { FeedbackButtons } from "./FeedbackButtons";
import { Icon } from "./Icon";

function formatDuration(value: number) {
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function humanQuality(value: string) {
  return value ? value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()) : "Prepared";
}

function directionLabel(journey: JourneyMode) {
  return ({ balanced: "Flow", warmup: "Warm", peak: "Drive", sunset: "Release", smooth: "Smooth", build: "Build" } as const)[journey] ?? journey;
}

export function FinishedSetPlayer({
  result,
  journey,
  onReset,
  onStartLive,
  onOpenBrain,
  listener,
}: {
  result: AutomaticSetResult;
  journey: JourneyMode;
  onReset: () => void;
  onStartLive: () => void;
  onOpenBrain: () => void;
  listener: string;
}) {
  const reviewCount = result.plan.transitions.filter((edge) => edge.quality !== "good").length;
  const beatmatchedCount = result.plan.transitions.filter((edge) => edge.plan.beatmatched).length;
  const executiveMeta = result.plan.metadata?.dj_executive as Record<string, unknown> | undefined;
  const backupCount = Number(executiveMeta?.prepared_backup_moves ?? 0);
  const fallbackCount = result.plan.transitions.filter((edge) => Number(edge.plan.fallback_level ?? 0) > 0).length;
  const hardFailureCount = result.plan.transitions.reduce((sum, edge) => sum + (edge.plan.measurement?.hard_failures?.length ?? 0), 0);
  const preferredCount = Math.max(0, result.plan.transitions.length - fallbackCount);
  const energyPoints = result.plan.ordered_tracks.map((track, index, values) => {
    const x = values.length <= 1 ? 0 : (index / (values.length - 1)) * 100;
    const y = 100 - Math.max(0, Math.min(1, track.energy)) * 100;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [transitionFeedback, setTransitionFeedback] = useState<Record<number, "like" | "dislike">>({});

  return (
    <main className="session-ready-shell">
      <header className="autonomous-topbar autonomous-topbar--ready">
        <a className="autonomous-brand" href="#top" aria-label="SAUND.AI">
          <span className="saund-wordmark"><strong>SAUND</strong><i>.</i><strong>AI</strong></span>
          <small>Autonomous DJ</small>
        </a>
        <div className="session-ready__state"><i /> SET READY</div>
      </header>

      <section className="session-ready">
        <div className="session-ready__hero">
          <span className="autonomous-kicker">EXECUTIVE READY</span>
          <h1>The room is ready.</h1>
          <p>SAUND has understood the crate, planned the route, privately auditioned the handoffs and committed a protected opening horizon.</p>
        </div>

        <div className="session-ready__core">
          <section className="session-ready__opening">
            <span>PERFORMANCE RIBBON</span>
            <div className="session-ready__trackflow">
              {result.plan.ordered_tracks.slice(0, 3).map((track, index) => (
                <div key={track.path} className={index === 0 ? "is-now" : index === 1 ? "is-next" : "is-future"}>
                  <small>{index === 0 ? "NOW" : index === 1 ? "NEXT · LOCKED" : "FUTURE"}</small>
                  <strong>{track.title}</strong>
                  <span>{track.bpm.toFixed(1)} BPM</span>
                </div>
              ))}
            </div>
          </section>

          <aside className="session-ready__proof">
            <div><span>Direction</span><strong>{directionLabel(journey)}</strong></div>
            <div><span>Tracks</span><strong>{result.plan.order.length}</strong></div>
            <div><span>Duration</span><strong>{formatDuration(result.plan.estimated_duration_seconds)}</strong></div>
            <div><span>Transitions</span><strong>{beatmatchedCount}/{result.plan.transitions.length} rehearsed</strong></div>
            <div><span>Recovery</span><strong>{backupCount > 0 ? `${backupCount} backups ready` : "Primary path ready"}</strong></div>
            <div><span>Safety</span><strong className={reviewCount === 0 ? "is-clear" : ""}>{reviewCount === 0 ? "Clear" : `${reviewCount} review`}</strong></div>
          </aside>
        </div>

        <section className="set-review" aria-label="Prepared set review">
          <header className="set-review__header">
            <div><span>PERFORMANCE ROUTE</span><strong>{result.plan.order.length} tracks · {result.plan.transitions.length} transitions</strong></div>
            <div className="set-review__summary">
              <span><b>{preferredCount}</b> preferred</span>
              <span><b>{fallbackCount}</b> recovered</span>
              <span><b>{hardFailureCount}</b> unsafe</span>
            </div>
          </header>

          <div className="set-review__energy">
            <div className="set-review__energy-copy">
              <span>ENERGY ARC</span>
              <small>{directionLabel(journey)} · planned performance contour</small>
            </div>
            <svg viewBox="0 0 100 100" role="img" aria-label="Planned energy arc">
              <line x1="0" y1="75" x2="100" y2="75" />
              <line x1="0" y1="50" x2="100" y2="50" />
              <line x1="0" y1="25" x2="100" y2="25" />
              <polyline points={energyPoints} />
            </svg>
          </div>

          <div className="set-review__route">
            {result.plan.ordered_tracks.map((track, index) => {
              const edge = index < result.plan.transitions.length ? result.plan.transitions[index] : null;
              const fallback = Boolean(edge && Number(edge.plan.fallback_level ?? 0) > 0);
              const measured = edge?.plan.measured_score ?? edge?.plan.measurement?.overall ?? null;
              const neural = edge?.plan.metadata?.neural_next_track as Record<string, unknown> | undefined;
              const neuralAvailable = Boolean(neural?.available);
              const neuralScore = Number(neural?.score ?? 0.5);
              const neuralAuthority = Number(neural?.authority ?? 0);
              return (
                <article key={track.path} className={fallback ? "has-fallback" : ""}>
                  <div className="set-review__track">
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <div><strong>{track.title}</strong><small>{track.bpm.toFixed(1)} BPM · {track.key.camelot}</small></div>
                  </div>
                  {edge && (
                    <div className="set-review__handoff">
                      <span>{fallback ? "RECOVERY" : "REHEARSED"}</span>
                      <strong>{edge.plan.technique.replaceAll("_", " ")}</strong>
                      <small>{measured !== null ? `${Math.round(Number(measured) * 100)}% quality` : humanQuality(edge.quality)}{fallback ? " · fallback" : ""}</small>
                      {neuralAvailable && (
                        <small className="set-review__shadow">AI shadow {Math.round(neuralScore * 100)}% · authority {Math.round(neuralAuthority * 100)}%</small>
                      )}
                      <div className="set-review__feedback" aria-label={"Feedback for transition " + (index + 1)}>
                        <button
                          type="button"
                          className={transitionFeedback[index] === "like" ? "is-selected" : ""}
                          onClick={() => {
                            setTransitionFeedback((current) => ({ ...current, [index]: "like" }));
                            void sendFeedback({
                              event: "transition_like",
                              listener,
                              planId: result.plan.plan_id,
                              planPath: result.plan_path,
                              transitionIndex: index,
                              journey,
                            }).catch(() => undefined);
                          }}
                        >Good</button>
                        <button
                          type="button"
                          className={transitionFeedback[index] === "dislike" ? "is-selected" : ""}
                          onClick={() => {
                            setTransitionFeedback((current) => ({ ...current, [index]: "dislike" }));
                            void sendFeedback({
                              event: "transition_dislike",
                              listener,
                              planId: result.plan.plan_id,
                              planPath: result.plan_path,
                              transitionIndex: index,
                              journey,
                            }).catch(() => undefined);
                          }}
                        >Review</button>
                      </div>
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        </section>

        <div className="session-ready__actions">
          <button type="button" className="session-ready__go-live" onClick={onStartLive}>
            <span><small>Enter autonomous performance</small><strong>Perform Live</strong></span>
            <Icon name="arrow" />
          </button>
          <button type="button" className="session-ready__brain" onClick={onOpenBrain}>
            <Icon name="spark" />
            <span><small>Understand the whole night</small><strong>Review decisions</strong></span>
          </button>
        </div>

        <details className="session-ready__render">
          <summary>
            <span><Icon name="headphones" /><strong>{previewPlaying ? "Listening to prepared render" : "Listen to prepared render"}</strong></span>
            <small>The committed offline performance remains available for review and export.</small>
            <Icon name="arrow" />
          </summary>
          <div className="session-ready__render-body">
            <audio
              controls
              preload="metadata"
              src={mediaUrl(result.audio_url)}
              aria-label="Prepared SAUND.AI set"
              onPlay={() => setPreviewPlaying(true)}
              onPause={() => setPreviewPlaying(false)}
              onEnded={() => setPreviewPlaying(false)}
            />
            <a
              href={mediaUrl(result.audio_url)}
              download
              className="session-ready__download"
              onClick={() => {
                void sendFeedback({
                  event: "downloaded",
                  listener,
                  planId: result.plan.plan_id,
                  planPath: result.plan_path,
                  journey,
                }).catch(() => undefined);
              }}
            >
              <Icon name="download" /> Export WAV
            </a>
          </div>
        </details>

        <div className="session-ready__feedback">
          <FeedbackButtons
            planId={result.plan.plan_id}
            planPath={result.plan_path}
            journey={journey}
            listener={listener}
          />
        </div>

        <footer className="session-ready__footer">
          <button type="button" onClick={onReset}>New session</button>
          <span>Brain imagines · Executive commits · Executor performs</span>
        </footer>
      </section>
    </main>
  );
}
