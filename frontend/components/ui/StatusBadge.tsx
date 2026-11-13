import type { ReactNode } from "react";

export type StatusTone = "neutral" | "analysis" | "decision" | "success" | "danger";

export function StatusBadge({
  tone = "neutral",
  children,
}: {
  tone?: StatusTone;
  children: ReactNode;
}) {
  return <span className={"ui-status ui-status--" + tone}>{children}</span>;
}
