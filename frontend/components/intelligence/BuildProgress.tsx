"use client";

import type { JobRecord } from "@/lib/types";

type BuildStage = {
  index: number;
  eyebrow: string;
  headline: string;
  detail: string;
};

const stageLabels = ["Understand", "Explore", "Rehearse", "Decide"] as const;

function buildStage(job: JobRecord): BuildStage {
  const key = job.stage.toLowerCase();

  if (key.includes("render") || key.includes("final") || key.includes("deliver") || key.includes("encode")) {
    return {
      index: 3,
      eyebrow: "Decision committed",
      headline: "Preparing the performance.",
      detail: job.message || "Rendering the validated plan and checking the execution contract.",
    };
  }

  if (key.includes("headphone") || key.includes("rehears") || key.includes("measure") || key.includes("transition")) {
    return {
      index: 2,
      eyebrow: "Private rehearsal",
      headline: "Listening before committing.",
      detail: job.message || "SAUND is testing the critical handoffs against rendered audio.",
    };
  }

  if (key.includes("analy") || key.includes("stem") || key.includes("feature") || key.includes("cached")) {
    return {
      index: 0,
      eyebrow: "Understanding the crate",
      headline: "Reading the music.",
      detail: job.message || "Mapping rhythm, structure, energy, vocals and usable performance layers.",
    };
  }

  return {
    index: 1,
    eyebrow: "Exploring the route",
    headline: "Thinking ahead.",
    detail: job.message || "Comparing routes, future options and protected transitions.",
  };
}

export function BuildProgress({
  job,
  onCancel,
}: {
  job: JobRecord;
  onCancel?: () => void;
}) {
  const stage = buildStage(job);
  const progress = Math.round(Math.max(0, Math.min(1, job.progress)) * 100);

  return (
    <section className="session-preparing" aria-live="polite" aria-label="SAUND build progress">
      <div className="session-preparing__copy">
        <span className="autonomous-kicker">{stage.eyebrow}</span>
        <h2>{stage.headline}</h2>
        <p>{stage.detail}</p>
      </div>

      <div className="session-preparing__progress">
        <div
          className="session-preparing__rail"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={progress}
          aria-label="Set build progress"
        >
          <i style={{ width: progress + "%" }} />
        </div>

        <div className="session-preparing__stages" aria-label="Build stages">
          {stageLabels.map((label, index) => (
            <span
              key={label}
              className={index < stage.index ? "is-complete" : index === stage.index ? "is-active" : ""}
              aria-current={index === stage.index ? "step" : undefined}
            >
              <i />
              {label}
            </span>
          ))}
        </div>
      </div>

      <div className="session-preparing__status">
        <strong>{progress}%</strong>
        {job.cancellable && onCancel && (
          <button type="button" onClick={onCancel}>Cancel</button>
        )}
      </div>
    </section>
  );
}
