"use client";

import type { ChangeEvent } from "react";
import { Icon } from "./Icon";

export function ControlDeck({ selectedCount, totalCount, tempoPolicy, onTempoPolicy, style, onStyle, coverage, onCoverage, mode, busy, onBuild }: {
  selectedCount: number;
  totalCount: number;
  tempoPolicy: "safe" | "stretch" | "cut";
  onTempoPolicy: (value: "safe" | "stretch" | "cut") => void;
  style: "auto" | "safe" | "creative" | "emergency";
  onStyle: (value: "auto" | "safe" | "creative" | "emergency") => void;
  coverage: "fast" | "balanced" | "long";
  onCoverage: (value: "fast" | "balanced" | "long") => void;
  mode: "auto" | "manual";
  busy: boolean;
  onBuild: () => void;
}) {
  const ready = selectedCount >= 2;
  return (
    <div className="control-card">
      <div className="control-card__header"><div><span className="eyebrow">Fine tune</span><strong>Set preferences</strong></div><span className="selection-count"><b>{selectedCount}</b> / {totalCount}</span></div>
      <div className="control-fields">
        <label className="control-field"><span><strong>Tempo freedom</strong><small>How far the set may reshape BPM</small></span><select value={tempoPolicy} onChange={(e: ChangeEvent<HTMLSelectElement>) => onTempoPolicy(e.target.value as typeof tempoPolicy)}><option value="safe">Natural</option><option value="stretch">More flexible</option><option value="cut">Keep original</option></select></label>
        <label className="control-field"><span><strong>Mix character</strong><small>How bold transitions may become</small></span><select value={style} onChange={(e: ChangeEvent<HTMLSelectElement>) => onStyle(e.target.value as typeof style)}><option value="auto">Choose for me</option><option value="safe">Smooth</option><option value="creative">Bolder</option><option value="emergency">Short and direct</option></select></label>
        <label className="control-field"><span><strong>Track time</strong><small>How much of each track to preserve</small></span><select value={coverage} onChange={(e: ChangeEvent<HTMLSelectElement>) => onCoverage(e.target.value as typeof coverage)}><option value="fast">Shorter</option><option value="balanced">Balanced</option><option value="long">Longer</option></select></label>
      </div>
      <button className="primary-action" type="button" disabled={busy || !ready} onClick={onBuild}>
        <span className="primary-action__icon"><Icon name="spark" /></span>
        <span><small>{mode === "auto" ? "SAUND.AI chooses the order" : "Your order stays fixed"}</small><strong>{busy ? "Listening…" : "Prepare set"}</strong></span><Icon name="arrow" />
      </button>
    </div>
  );
}
