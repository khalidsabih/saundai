import type { Metadata } from "next";
import { TrainingLab } from "@/components/TrainingLab";

export const metadata: Metadata = {
  title: "Training Lab — SAUND.AI",
  description: "Internal SAUND.AI pairwise listening, dataset versioning and shadow-model training surface.",
  robots: { index: false, follow: false },
};

export default function LabPage() {
  return <TrainingLab />;
}
