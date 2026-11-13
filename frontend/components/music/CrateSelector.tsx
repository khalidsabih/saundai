"use client";

import { useRef, useState } from "react";
import type { DragEvent } from "react";
import type { TrackRecord } from "@/lib/types";
import { Icon } from "../Icon";

function readinessState(track: TrackRecord) {
  const readiness = track.readiness;
  if (!readiness) return { label: "Waiting", tone: "waiting" };
  const state = readiness.state.toLowerCase();
  if (readiness.dj_ready || state === "ready") return { label: "Ready", tone: "ready" };
  if (state === "failed" || state === "cancelled") return { label: "Needs attention", tone: "failed" };
  if (state === "pending" || state === "queued") return { label: "Listening", tone: "active" };
  const percent = Math.round(Math.max(0, Math.min(1, readiness.progress ?? 0)) * 100);
  return { label: percent > 0 ? `Analysing ${percent}%` : "Analysing", tone: "active" };
}

export function CrateSelector({
  tracks,
  selectedPaths,
  uploading,
  disabled,
  readyCount,
  preparingCount,
  failedCount,
  readinessProgress,
  onFiles,
  onToggleTrack,
}: {
  tracks: TrackRecord[];
  selectedPaths: string[];
  uploading: boolean;
  disabled: boolean;
  readyCount: number;
  preparingCount: number;
  failedCount: number;
  readinessProgress: number;
  onFiles: (files: FileList) => void;
  onToggleTrack: (path: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);

  return (
    <section className="session-crate" aria-label="Session crate">
      <header className="session-section-head">
        <div><span>01</span><div><small>CRATE</small><strong>Your music</strong></div></div>
        <p>{selectedPaths.length} selected · {readyCount} ready</p>
      </header>

      <button
        className={`session-dropzone ${dragging ? "is-dragging" : ""}`}
        type="button"
        disabled={uploading || disabled}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(event: DragEvent<HTMLButtonElement>) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragOver={(event: DragEvent<HTMLButtonElement>) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={(event: DragEvent<HTMLButtonElement>) => {
          event.preventDefault();
          setDragging(false);
          if (event.dataTransfer.files.length) onFiles(event.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="audio/*,.mp3,.wav,.flac,.m4a,.aiff"
          hidden
          onChange={(event) => {
            if (event.target.files?.length) onFiles(event.target.files);
            event.currentTarget.value = "";
          }}
        />
        <span className="session-dropzone__icon"><Icon name="upload" /></span>
        <span>
          <strong>{uploading ? "Adding music…" : "Add music"}</strong>
          <small>Drop audio or choose files</small>
        </span>
        <b>{tracks.length ? `${tracks.length} in crate` : "Empty crate"}</b>
      </button>

      <div className="session-tracklist scroll-area" role="group" aria-label="Available tracks">
        {tracks.length === 0 ? (
          <div className="session-tracklist__empty">
            <Icon name="waveform" />
            <strong>Start with your music.</strong>
            <small>Add at least two tracks to build a set.</small>
          </div>
        ) : tracks.map((track, index) => {
          const selected = selectedPaths.includes(track.path);
          const state = readinessState(track);
          return (
            <button
              type="button"
              key={track.path}
              className={`session-track ${selected ? "is-selected" : ""}`}
              aria-pressed={selected}
              onClick={() => onToggleTrack(track.path)}
            >
              <span className="session-track__index">{String(index + 1).padStart(2, "0")}</span>
              <span className="session-track__main">
                <strong>{track.name}</strong>
                <small>
                  {(track.size / 1_048_576).toFixed(1)} MB
                  {track.performance_layers?.complete ? " · performance layers" : ""}
                </small>
              </span>
              <span className={`session-track__state is-${state.tone}`}><i />{state.label}</span>
              <span className="session-track__select">{selected ? <Icon name="check" /> : "+"}</span>
            </button>
          );
        })}
      </div>

      <footer className="session-crate__footer">
        <span>
          {failedCount > 0
            ? `${failedCount} selected track${failedCount === 1 ? "" : "s"} need attention`
            : preparingCount > 0
              ? `${preparingCount} still being understood`
              : selectedPaths.length >= 2
                ? "Selected music is ready"
                : "Select at least two tracks"}
        </span>
        {selectedPaths.length > 0 && <b>{Math.round(readinessProgress * 100)}% understood</b>}
      </footer>
    </section>
  );
}
