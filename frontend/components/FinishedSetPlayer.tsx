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
  const [previewPlaying, setPreviewPlaying] = useState(false);

  return (
    <main className="session-ready-shell">
      <header className="autonomous-topbar autonomous-topbar--ready">
        <a className="autonomous-brand" href="#top" aria-label="SAUND.AI">
          <span className="saund-wordmark"><strong>SAUND</strong><i>.</i><strong>AI</strong></span>
          <small>Autonomous DJ</small>
        </a>
        <div className="session-ready__state"><i /> SESSION PREPARED</div>
      </header>

      <section className="session-ready">
        <div className="session-ready__hero">
          <span className="autonomous-kicker">EXECUTIVE READY</span>
          <h1>The room is ready.</h1>
          <p>SAUND has understood the crate, planned the route, privately auditioned the handoffs and committed a protected opening horizon.</p>
        </div>

        <div className="session-ready__core">
          <section className="session-ready__opening">
            <span>OPENING HORIZON</span>
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
            <div><span>Handoffs</span><strong>{beatmatchedCount}/{result.plan.transitions.length} prepared</strong></div>
            <div><span>Recovery</span><strong>{backupCount > 0 ? `${backupCount} backups ready` : "Primary path ready"}</strong></div>
            <div><span>Safety</span><strong className={reviewCount === 0 ? "is-clear" : ""}>{reviewCount === 0 ? "Clear" : `${reviewCount} review`}</strong></div>
          </aside>
        </div>

        <div className="session-ready__actions">
          <button type="button" className="session-ready__go-live" onClick={onStartLive}>
            <span><small>Enter autonomous performance</small><strong>Go live</strong></span>
            <Icon name="arrow" />
          </button>
          <button type="button" className="session-ready__brain" onClick={onOpenBrain}>
            <Icon name="spark" />
            <span><small>Inspect the decisions</small><strong>Open Brain</strong></span>
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
