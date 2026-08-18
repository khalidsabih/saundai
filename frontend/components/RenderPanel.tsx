"use client";

import type { RenderResult, SetPlan } from "@/lib/types";
import { mediaUrl } from "@/lib/api";
import { Icon } from "./Icon";

export function RenderPanel({ plan, onPreview, onRender, previewBusy, renderBusy, rendered }: {
  plan: SetPlan;
  onPreview: () => void;
  onRender: (confirmed: boolean) => void;
  previewBusy: boolean;
  renderBusy: boolean;
  rendered: RenderResult | null;
}) {
  const flagged = plan.transitions.filter((edge) => edge.quality !== "good").length;

  return (
    <section className="panel render-panel" id="render">
      <header className="panel-header">
        <div className="panel-title"><span className="panel-number">05</span><div><span className="eyebrow">Finish</span><h2>Listen, approve, export</h2></div></div>
        <span className={`status-chip ${plan.requires_confirmation ? "review" : "ready"}`}><i />{plan.requires_confirmation ? "Your approval needed" : "Ready"}</span>
      </header>

      {plan.requires_confirmation ? (
        <div className="confirmation-card"><Icon name="warning" /><div><strong>One last listen is recommended.</strong><p>{flagged} transition{flagged === 1 ? "" : "s"} use a fallback or need review. SAUND.AI will not render them without your confirmation.</p></div></div>
      ) : (
        <div className="ready-callout"><Icon name="check" /><span><strong>Everything passed the listening checks.</strong><small>You can preview transitions or render the full set.</small></span></div>
      )}

      <div className="render-actions">
        <button className="secondary-action" type="button" disabled={previewBusy || renderBusy} onClick={onPreview}>
          <span><Icon name="headphones" /></span><span><small>Before the final render</small><strong>{previewBusy ? "Creating previews…" : "Preview transitions"}</strong></span><Icon name="arrow" />
        </button>
        <button className="primary-action render-action" type="button" disabled={renderBusy || previewBusy} onClick={() => onRender(plan.requires_confirmation)}>
          <span className="primary-action__icon"><Icon name="render" /></span><span><small>{plan.requires_confirmation ? "Confirms reviewed transitions" : "Final output"}</small><strong>{renderBusy ? "Rendering…" : "Render full set"}</strong></span><Icon name="arrow" />
        </button>
      </div>

      {rendered && (
        <div className="render-result">
          <div className="render-result__header"><div><span className="render-complete"><Icon name="check" /></span><span><small>Ready</small><strong>{rendered.audio_path.split("/").at(-1)}</strong></span></div><a className="download-button" href={mediaUrl(rendered.audio_url)} download><Icon name="download" /><span><small>WAV</small><strong>Download</strong></span></a></div>
          <audio controls preload="metadata" src={mediaUrl(rendered.audio_url)} aria-label="Rendered SAUND.AI set" />
        </div>
      )}
    </section>
  );
}
