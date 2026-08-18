"use client";

import type { JobRecord } from "@/lib/types";

function elapsedLabel(seconds?: number) {
  if (!seconds || seconds < 1) return null;
  const total = Math.floor(seconds);
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return minutes > 0 ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

export function JobProgress({ job, onCancel }: { job: JobRecord | null; onCancel?: () => void }) {
  if (!job) return null;

  const percent = Math.round(Math.max(0, Math.min(1, job.progress)) * 100);
  const hasStageCount = (job.stage_total ?? 0) > 1;
  const stageCount = hasStageCount ? `${Math.min(job.stage_current ?? 0, job.stage_total ?? 0)} / ${job.stage_total}` : null;
  const elapsed = elapsedLabel(job.elapsed_seconds);
  const stageKey = job.stage.toLowerCase();
  const stage =
    job.cancel_requested ? "Stopping safely" :
    stageKey.includes("stem") ? "Preparing audio layers" :
    stageKey.includes("analy") || stageKey.includes("cached") ? "Understanding your tracks" :
    stageKey.includes("graph") || stageKey.includes("order") ? "Choosing the best order" :
    stageKey.includes("measure") || stageKey.includes("transition") ? "Mixing and checking transitions" :
    stageKey.includes("render") || stageKey.includes("final") ? "Finalizing your set" :
    stageKey === "completed" ? "Your set is ready" :
    stageKey === "cancelled" ? "Job cancelled" :
    job.stage.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());

  const detail = [job.message || "Processing your set", stageCount, elapsed].filter(Boolean).join(" · ");

  return (
    <div className={`job-progress${job.stalled ? " is-stalled" : ""}`} role="status" aria-live="polite">
      <div className="job-progress__header">
        <div>
          <span className="job-progress__dot" />
          <div>
            <strong>{stage}</strong>
            <small>{detail}</small>
          </div>
        </div>
        <div className="job-progress__actions">
          {job.stalled && <span className="job-progress__warning">No stage progress</span>}
          <span>{percent}%</span>
          {onCancel && job.cancellable && !job.cancel_requested && (
            <button type="button" onClick={onCancel}>Cancel</button>
          )}
        </div>
      </div>
      <div
        className="job-progress__track"
        role="progressbar"
        aria-label={stage}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.max(0, Math.min(100, percent))}
      >
        <span style={{ width: `${Math.max(2, Math.min(100, percent))}%` }} />
      </div>
    </div>
  );
}
