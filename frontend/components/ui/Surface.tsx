import type { ElementType, ReactNode } from "react";

export function Surface({
  as: Tag = "section",
  className = "",
  children,
}: {
  as?: ElementType;
  className?: string;
  children: ReactNode;
}) {
  return <Tag className={["ui-surface", className].filter(Boolean).join(" ")}>{children}</Tag>;
}
