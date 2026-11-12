import type { Metadata } from "next";
import { ReadinessDashboard } from "@/components/ReadinessDashboard";

export const metadata: Metadata = {
  title: "Readiness — SAUND.AI",
  description: "Operational, library, model-authority and real-music validation readiness for SAUND.AI.",
  robots: { index: false, follow: false },
};

export default function ReadinessPage() {
  return <ReadinessDashboard />;
}
