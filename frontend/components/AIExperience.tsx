import type { CSSProperties } from "react";
import { Icon, type IconName } from "./Icon";

export type AIExperienceStage = "listen" | "analyse" | "plan" | "play" | "live";

const stages: Array<{ id: AIExperienceStage; label: string; icon: IconName }> = [
  { id: "listen", label: "Listening", icon: "headphones" },
  { id: "analyse", label: "Analysing", icon: "waveform" },
  { id: "plan", label: "Planning", icon: "route" },
  { id: "play", label: "Playing", icon: "play" },
  { id: "live", label: "Live", icon: "live" },
];

export type AIExperienceMetric = {
  label: string;
  value: string;
};

export function AIExperience({
  stage,
  headline,
  detail,
  progress,
  metrics = [],
  compact = false,
}: {
  stage: AIExperienceStage;
  headline: string;
  detail: string;
  progress?: number | null;
  metrics?: AIExperienceMetric[];
  compact?: boolean;
}) {
  const activeIndex = stages.findIndex((item) => item.id === stage);
  const safeProgress = progress === undefined || progress === null
    ? null
    : Math.max(0, Math.min(1, progress));
  const progressStyle = safeProgress === null
    ? undefined
    : ({ "--ai-progress": `${Math.round(safeProgress * 100)}%` } as CSSProperties);

  return (
    <section
      className={`ai-experience ai-experience--${stage}${compact ? " ai-experience--compact" : ""}`}
      aria-label={`SAUND AI: ${stages[activeIndex]?.label ?? stage}`}
    >
      <div className="ai-experience__signal" aria-hidden="true">
        <span className="ai-experience__core"><Icon name={stages[activeIndex]?.icon ?? "spark"} /></span>
        <span className="ai-experience__ring ai-experience__ring--one" />
        <span className="ai-experience__ring ai-experience__ring--two" />
        <span className="ai-experience__bars">
          {Array.from({ length: 9 }, (_, index) => <i key={index} style={{ "--bar-index": index } as CSSProperties} />)}
        </span>
      </div>

      <div className="ai-experience__copy">
        <span className="ai-experience__eyebrow"><i /> SAUND AI · {stages[activeIndex]?.label}</span>
        <strong>{headline}</strong>
        <p>{detail}</p>
        {safeProgress !== null && (
          <div className="ai-experience__progress" style={progressStyle} aria-hidden="true"><i /></div>
        )}
      </div>

      {metrics.length > 0 && (
        <div className="ai-experience__metrics" aria-label="AI activity summary">
          {metrics.slice(0, 3).map((metric) => (
            <span key={`${metric.label}-${metric.value}`}>
              <strong>{metric.value}</strong>
              <small>{metric.label}</small>
            </span>
          ))}
        </div>
      )}

      <div className="ai-experience__rail" aria-label="SAUND AI stages">
        {stages.map((item, index) => {
          const state = index < activeIndex ? "complete" : index === activeIndex ? "active" : "upcoming";
          return (
            <div className={`ai-experience__stage is-${state}`} key={item.id}>
              <span><Icon name={item.icon} /></span>
              <small>{item.label}</small>
            </div>
          );
        })}
      </div>
    </section>
  );
}
