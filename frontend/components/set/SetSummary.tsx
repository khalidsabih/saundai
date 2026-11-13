import type { JourneyMode, SetPlan } from "@/lib/types";

function formatDuration(value: number) {
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

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

export function SetSummary({
  plan,
  journey,
}: {
  plan: SetPlan;
  journey: JourneyMode;
}) {
  const measuredCount = plan.transitions.filter(
    (edge) => edge.plan.quality_checked === true || edge.plan.measurement != null,
  ).length;
  const fallbackCount = plan.transitions.filter(
    (edge) => Number(edge.plan.fallback_level ?? 0) > 0,
  ).length;
  const hardFailureCount = plan.transitions.reduce(
    (sum, edge) => sum + (edge.plan.measurement?.hard_failures?.length ?? 0),
    0,
  );

  return (
    <aside className="session-ready__proof" aria-label="Set summary">
      <div><span>Direction</span><strong>{directionLabel(journey)}</strong></div>
      <div><span>Tracks</span><strong>{plan.order.length}</strong></div>
      <div><span>Duration</span><strong>{formatDuration(plan.estimated_duration_seconds)}</strong></div>
      <div><span>Measured</span><strong>{measuredCount}/{plan.transitions.length}</strong></div>
      <div><span>Recovery</span><strong>{fallbackCount ? `${fallbackCount} used` : "None used"}</strong></div>
      <div>
        <span>Hard failures</span>
        <strong className={hardFailureCount === 0 ? "is-clear" : ""}>{hardFailureCount}</strong>
      </div>
    </aside>
  );
}
