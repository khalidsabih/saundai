"use client";

import type { TrackRecord } from "@/lib/types";
import { Icon } from "./Icon";

const journeys = [
  { value: "balanced" as const, label: "Flow", detail: "Natural movement" },
  { value: "smooth" as const, label: "Smooth", detail: "Steady and controlled" },
  { value: "build" as const, label: "Build", detail: "Rise over time" },
];

export function SetBuilder({ tracks, onReorder, onRemove, mode, onModeChange, journey, onJourneyChange }: {
  tracks: TrackRecord[];
  onReorder: (tracks: TrackRecord[]) => void;
  onRemove: (path: string) => void;
  mode: "auto" | "manual";
  onModeChange: (value: "auto" | "manual") => void;
  journey: "balanced" | "smooth" | "build";
  onJourneyChange: (value: "balanced" | "smooth" | "build") => void;
}) {
  const ready = tracks.length >= 2;

  function move(index: number, direction: -1 | 1) {
    const target = index + direction;
    if (target < 0 || target >= tracks.length) return;
    const next = [...tracks];
    [next[index], next[target]] = [next[target], next[index]];
    onReorder(next);
  }

  return (
    <section className="panel builder-panel" id="set-builder">
      <header className="panel-header">
        <div className="panel-title"><span className="panel-number">02</span><div><span className="eyebrow">Direction</span><h2>Shape the set</h2></div></div>
        <div className="segmented-control" role="group" aria-label="Set ordering mode">
          <button className={mode === "auto" ? "active" : ""} aria-pressed={mode === "auto"} type="button" onClick={() => onModeChange("auto")}>SAUND.AI</button>
          <button className={mode === "manual" ? "active" : ""} aria-pressed={mode === "manual"} type="button" onClick={() => onModeChange("manual")}>My order</button>
        </div>
      </header>

      <div className="journey-card">
        <div><strong>How should it move?</strong><small>Choose the overall direction.</small></div>
        <div className="journey-options">
          {journeys.map((option) => (
            <button key={option.value} type="button" className={journey === option.value ? "active" : ""} aria-pressed={journey === option.value} onClick={() => onJourneyChange(option.value)}>
              <span><strong>{option.label}</strong><small>{option.detail}</small></span>
            </button>
          ))}
        </div>
      </div>

      {!ready ? (
        <div className="empty-state empty-state--tall"><strong>Select at least two tracks</strong><small>Your chosen music will appear here.</small></div>
      ) : (
        <>
          <div className="set-list-head"><span>#</span><span>Track</span><span>{mode === "manual" ? "Order" : "SAUND.AI can reorder"}</span></div>
          <div className="set-list scroll-area">
            {tracks.map((track, index) => (
              <div key={track.path} className="set-row">
                <span className="set-row__index">{String(index + 1).padStart(2, "0")}</span>
                <span className="set-row__copy"><strong>{track.name}</strong><small>{mode === "manual" ? "Position fixed" : "Position flexible"}</small></span>
                {mode === "manual" ? (
                  <span className="set-row__move">
                    <button type="button" onClick={() => move(index, -1)} disabled={index === 0} aria-label={`Move ${track.name} up`}>↑</button>
                    <button type="button" onClick={() => move(index, 1)} disabled={index === tracks.length - 1} aria-label={`Move ${track.name} down`}>↓</button>
                  </span>
                ) : <span className="set-row__state auto">Flexible</span>}
                <button className="icon-button" type="button" onClick={() => onRemove(track.path)} aria-label={`Remove ${track.name}`}><Icon name="close" /></button>
              </div>
            ))}
          </div>
        </>
      )}

      <div className="panel-footer"><span><i className={ready ? "is-ready" : ""} />{ready ? `${tracks.length} tracks ready` : "Waiting for music"}</span></div>
    </section>
  );
}
