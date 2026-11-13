"use client";

import { useState } from "react";
import { mediaUrl, sendFeedback } from "@/lib/api";
import type { AutomaticSetResult, JourneyMode } from "@/lib/types";
import { PerformanceRibbon, type TransitionFeedback } from "./set/PerformanceRibbon";
import { SetSummary } from "./set/SetSummary";
import { FeedbackButtons } from "./FeedbackButtons";
import { Icon } from "./Icon";
import { AppShell } from "./product/AppShell";

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
  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [transitionFeedback, setTransitionFeedback] = useState<Record<number, TransitionFeedback>>({});

  return (
    <AppShell active="create" onOpenBrain={onOpenBrain}>
      <main className="session-ready-shell">
        <section className="session-ready">
        <div className="session-ready__hero">
          <span className="autonomous-kicker">READY</span>
          <h1>Your set is ready.</h1>
          <p>SAUND understood the crate, explored the route, rehearsed the available evidence and prepared the performance plan.</p>
        </div>

        <div className="session-ready__core">
          <SetSummary plan={result.plan} journey={journey} />
        </div>

        <PerformanceRibbon
          plan={result.plan}
          journey={journey}
          feedback={transitionFeedback}
          onFeedback={(transitionIndex, value) => {
            setTransitionFeedback((current) => ({ ...current, [transitionIndex]: value }));
            void sendFeedback({
              event: value === "like" ? "transition_like" : "transition_dislike",
              listener,
              planId: result.plan.plan_id,
              planPath: result.plan_path,
              transitionIndex,
              journey,
            }).catch(() => undefined);
          }}
        />

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
    </AppShell>
  );
}
