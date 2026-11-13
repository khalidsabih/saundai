"use client";

import { useState } from "react";
import type { JourneyMode } from "@/lib/types";
import { Icon } from "../Icon";

const journeys: Array<{ value: JourneyMode; label: string; detail: string; cue: string }> = [
  { value: "balanced", label: "Flow", detail: "Natural movement with room to breathe.", cue: "Balanced" },
  { value: "warmup", label: "Warm", detail: "Patient opening, gradual energy and space.", cue: "Patient" },
  { value: "peak", label: "Drive", detail: "Push toward the strongest moments in the crate.", cue: "Forward" },
  { value: "sunset", label: "Release", detail: "Warm movement with a softer landing.", cue: "Open" },
];

export function SetIntentPanel({
  journey,
  targetMinutes,
  peakFraction,
  closingMinutes,
  endingEnergy,
  selectedCount,
  readyCount,
  preparingCount,
  failedCount,
  canBuild,
  authenticated,
  onJourney,
  onTargetMinutes,
  onPeakFraction,
  onClosingMinutes,
  onEndingEnergy,
  onBuild,
}: {
  journey: JourneyMode;
  targetMinutes: number | null;
  peakFraction: number;
  closingMinutes: number;
  endingEnergy: number;
  selectedCount: number;
  readyCount: number;
  preparingCount: number;
  failedCount: number;
  canBuild: boolean;
  authenticated: boolean;
  onJourney: (journey: JourneyMode) => void;
  onTargetMinutes: (minutes: number | null) => void;
  onPeakFraction: (value: number) => void;
  onClosingMinutes: (minutes: number) => void;
  onEndingEnergy: (value: number) => void;
  onBuild: () => void;
}) {
  const [showDetails, setShowDetails] = useState(false);
  const selectedReady = selectedCount >= 2 && readyCount === selectedCount && failedCount === 0;
  const currentJourney = journeys.find((item) => item.value === journey) ?? journeys[0];

  return (
    <aside className="session-command" aria-label="Set intent">
      <header className="session-section-head">
        <div><span>02</span><div><small>INTENT</small><strong>Shape the night</strong></div></div>
      </header>

      <div className="session-directions" role="radiogroup" aria-label="Set direction">
        {journeys.map((option) => (
          <button
            type="button"
            role="radio"
            aria-checked={journey === option.value}
            key={option.value}
            className={journey === option.value ? "is-selected" : ""}
            onClick={() => onJourney(option.value)}
          >
            <span><small>{option.cue}</small><strong>{option.label}</strong></span>
            <p>{option.detail}</p>
            <i>{journey === option.value ? <Icon name="check" /> : ""}</i>
          </button>
        ))}
      </div>

      <section className="set-brief" aria-label="Set brief">
        <div className="set-brief__row">
          <span><small>DURATION</small><strong>{targetMinutes ? targetMinutes + " min" : "Use crate"}</strong></span>
          <div className="set-brief__choices">
            {[60, 90, 120].map((minutes) => (
              <button
                type="button"
                key={minutes}
                className={targetMinutes === minutes ? "is-selected" : ""}
                onClick={() => onTargetMinutes(minutes)}
              >
                {minutes}
              </button>
            ))}
            <button
              type="button"
              className={targetMinutes === null ? "is-selected" : ""}
              onClick={() => onTargetMinutes(null)}
            >
              Auto
            </button>
          </div>
        </div>

        <div className="set-brief__row">
          <span>
            <small>ENDING ENERGY</small>
            <strong>{endingEnergy < 0.35 ? "Soft" : endingEnergy > 0.65 ? "Strong" : "Balanced"}</strong>
          </span>
          <input
            aria-label="Ending energy"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={endingEnergy}
            onChange={(event) => onEndingEnergy(Number(event.target.value))}
          />
        </div>

        <button
          type="button"
          className="set-brief__details-toggle"
          onClick={() => setShowDetails((value) => !value)}
          aria-expanded={showDetails}
        >
          <span>Advanced timing</span>
          <strong>{showDetails ? "Hide" : "Show"}</strong>
        </button>

        {showDetails && (
          <div className="set-brief__details">
            <label>
              <span>Peak timing <b>{Math.round(peakFraction * 100)}%</b></span>
              <input
                type="range"
                min="0.15"
                max="0.95"
                step="0.05"
                value={peakFraction}
                onChange={(event) => onPeakFraction(Number(event.target.value))}
              />
            </label>
            <label>
              <span>Closing window <b>{closingMinutes} min</b></span>
              <input
                type="range"
                min="1"
                max="30"
                step="1"
                value={closingMinutes}
                onChange={(event) => onClosingMinutes(Number(event.target.value))}
              />
            </label>
          </div>
        )}
      </section>

      <section
        className={"crate-preflight " + (failedCount ? "is-blocked" : selectedReady ? "is-ready" : "")}
        aria-label="Crate preflight"
      >
        <header>
          <span>CRATE PREFLIGHT</span>
          <strong>{selectedReady ? "Ready" : failedCount ? "Attention" : "Preparing"}</strong>
        </header>
        <div className="crate-preflight__stats">
          <div><b>{selectedCount}</b><small>selected</small></div>
          <div><b>{readyCount}</b><small>ready</small></div>
          <div><b>{preparingCount}</b><small>preparing</small></div>
          <div><b>{failedCount}</b><small>blocked</small></div>
        </div>
        <p>
          {failedCount > 0
            ? "Remove or repair tracks that failed readiness before building."
            : selectedReady
              ? "The selected crate is understood and ready for planning."
              : "SAUND can build once at least two selected tracks are usable."}
        </p>
      </section>

      <div className="session-command__summary">
        <small>SET INTENT</small>
        <strong>{currentJourney.label}</strong>
        <p>
          {selectedCount >= 2
            ? `${selectedCount} tracks selected for autonomous planning.`
            : "Choose the music SAUND may use."}
        </p>
      </div>

      <button className="session-start" type="button" disabled={!canBuild} onClick={onBuild}>
        <span>
          <small>{authenticated ? "SAUND will prepare the performance" : "Sign in when you build"}</small>
          <strong>Build my set</strong>
        </span>
        <Icon name="arrow" />
      </button>
    </aside>
  );
}
