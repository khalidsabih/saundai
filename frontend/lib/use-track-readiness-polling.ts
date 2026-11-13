"use client";

import { useEffect, type Dispatch, type SetStateAction } from "react";
import { listTracks } from "./api";
import type { TrackRecord } from "./types";

/**
 * Shared background track-readiness refresh for Create and Studio.
 *
 * Matches the previous per-screen effect: only poll while any track is
 * preparing, refresh the audio library every two seconds, ignore transient
 * refresh errors, and always clear the interval on dependency changes/unmount.
 * Initial loading and its error semantics remain owned by each screen.
 */
export function useTrackReadinessPolling(
  tracks: TrackRecord[],
  setTracks: Dispatch<SetStateAction<TrackRecord[]>>,
) {
  useEffect(() => {
    const preparing = tracks.some((track) => track.readiness && !["ready", "failed", "cancelled"].includes(track.readiness.state));
    if (!preparing) return;
    const timer = window.setInterval(() => {
      void listTracks("audio").then(setTracks).catch(() => undefined);
    }, 2_000);
    return () => window.clearInterval(timer);
  }, [tracks, setTracks]);
}
