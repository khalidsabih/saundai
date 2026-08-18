"use client";

import { useState } from "react";
import { sendFeedback } from "@/lib/api";
import type { JourneyMode } from "@/lib/types";

export function FeedbackButtons({
  planId,
  planPath,
  journey,
  listener = "web",
}: {
  planId: string;
  planPath: string;
  journey: JourneyMode;
  listener?: string;
}) {
  const [choice, setChoice] = useState<"like" | "dislike" | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function choose(value: "like" | "dislike") {
    const previous = choice;
    setSaving(true);
    setError(null);
    try {
      await sendFeedback({
        event: value === "like" ? "set_like" : "set_dislike",
        listener,
        planId,
        planPath,
        journey,
        value: value === "like" ? 1 : -1,
      });
      setChoice(value);
    } catch {
      setChoice(previous);
      setError("Couldn’t save your feedback. Try again.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="simple-feedback" aria-label="Set feedback">
      <span aria-live="polite">
        {error ?? (choice ? "Thanks — saved." : "Did this set feel right?")}
      </span>
      <div>
        <button
          type="button"
          className={choice === "like" ? "is-selected" : ""}
          aria-pressed={choice === "like"}
          disabled={saving}
          onClick={() => void choose("like")}
        >
          Yes
        </button>
        <button
          type="button"
          className={choice === "dislike" ? "is-selected" : ""}
          aria-pressed={choice === "dislike"}
          disabled={saving}
          onClick={() => void choose("dislike")}
        >
          Not quite
        </button>
      </div>
    </div>
  );
}
