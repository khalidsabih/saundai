"use client";

import type { JourneyMode, SetPlan } from "@/lib/types";

function directionLabel(journey: JourneyMode) {
  return ({
    balanced: "Flow",
    warmup: "Warm",
    peak: "Drive",
    sunset: "Release",
    smooth: "Smooth",
    build: "Build",
  } as const)[journey] ?? journey;
}

function humanQuality(value: string) {
  return value
    ? value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase())
    : "Prepared";
}

export type TransitionFeedback = "like" | "dislike";

export function PerformanceRibbon({
  plan,
  journey,
  feedback = {},
  onFeedback,
}: {
  plan: SetPlan;
  journey: JourneyMode;
  feedback?: Record<number, TransitionFeedback>;
  onFeedback?: (transitionIndex: number, value: TransitionFeedback) => void;
}) {
  const fallbackCount = plan.transitions.filter(
    (edge) => Number(edge.plan.fallback_level ?? 0) > 0,
  ).length;
  const hardFailureCount = plan.transitions.reduce(
    (sum, edge) => sum + (edge.plan.measurement?.hard_failures?.length ?? 0),
    0,
  );
  const rehearsedCount = plan.transitions.filter(
    (edge) => edge.plan.quality_checked || edge.plan.measurement !== null,
  ).length;
  const preferredCount = Math.max(0, plan.transitions.length - fallbackCount);

  const energyPoints = plan.ordered_tracks.map((track, index, values) => {
    const x = values.length <= 1 ? 0 : (index / (values.length - 1)) * 100;
    const y = 100 - Math.max(0, Math.min(1, track.energy)) * 100;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");

  return (
    <section className="performance-ribbon" aria-label="Performance Ribbon">
      <header className="performance-ribbon__header">
        <div>
          <span>PERFORMANCE RIBBON</span>
          <strong>{plan.order.length} tracks · {plan.transitions.length} transitions</strong>
        </div>
        <div className="performance-ribbon__summary" aria-label="Route summary">
          <span><b>{preferredCount}</b> primary</span>
          <span><b>{fallbackCount}</b> recovered</span>
          <span><b>{rehearsedCount}</b> measured</span>
          <span><b>{hardFailureCount}</b> hard failures</span>
        </div>
      </header>

      <div className="performance-ribbon__energy">
        <div>
          <span>ENERGY ARC</span>
          <small>{directionLabel(journey)} · planned performance contour</small>
        </div>
        <svg viewBox="0 0 100 100" role="img" aria-label="Planned set energy arc">
          <line x1="0" y1="75" x2="100" y2="75" />
          <line x1="0" y1="50" x2="100" y2="50" />
          <line x1="0" y1="25" x2="100" y2="25" />
          <polyline points={energyPoints} />
        </svg>
      </div>

      <div className="performance-ribbon__route" role="list" aria-label="Planned route">
        {plan.ordered_tracks.map((track, index) => {
          const edge = index < plan.transitions.length ? plan.transitions[index] : null;
          const fallback = Boolean(edge && Number(edge.plan.fallback_level ?? 0) > 0);
          const measured = edge?.plan.measured_score ?? edge?.plan.measurement?.overall ?? null;
          const neural = edge?.plan.metadata?.neural_next_track as Record<string, unknown> | undefined;
          const neuralAvailable = Boolean(neural?.available);
          const neuralScore = Number(neural?.score ?? 0.5);
          const neuralAuthority = Number(neural?.authority ?? 0);

          return (
            <article key={track.path} className={fallback ? "has-fallback" : ""} role="listitem">
              <div className="performance-ribbon__track">
                <span>{String(index + 1).padStart(2, "0")}</span>
                <div>
                  <strong>{track.title}</strong>
                  <small>{track.bpm.toFixed(1)} BPM · {track.key.camelot}</small>
                </div>
              </div>

              {edge && (
                <div className="performance-ribbon__transition">
                  <span>{fallback ? "RECOVERY" : edge.plan.quality_checked ? "REHEARSED" : "PLANNED"}</span>
                  <strong>{edge.plan.technique.replaceAll("_", " ")}</strong>
                  <small>
                    {measured !== null
                      ? `${Math.round(Number(measured) * 100)}% measured quality`
                      : humanQuality(edge.quality)}
                    {fallback ? " · fallback" : ""}
                  </small>

                  {neuralAvailable && (
                    <small className="performance-ribbon__shadow">
                      AI {neuralAuthority > 0 ? "advisory" : "shadow"} {Math.round(neuralScore * 100)}%
                      {" · "}authority {Math.round(neuralAuthority * 100)}%
                    </small>
                  )}

                  {onFeedback && (
                    <div
                      className="performance-ribbon__feedback"
                      aria-label={"Feedback for transition " + (index + 1)}
                    >
                      <button
                        type="button"
                        className={feedback[index] === "like" ? "is-selected" : ""}
                        aria-pressed={feedback[index] === "like"}
                        onClick={() => onFeedback(index, "like")}
                      >
                        Good
                      </button>
                      <button
                        type="button"
                        className={feedback[index] === "dislike" ? "is-selected" : ""}
                        aria-pressed={feedback[index] === "dislike"}
                        onClick={() => onFeedback(index, "dislike")}
                      >
                        Review
                      </button>
                    </div>
                  )}
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
