"use client";

import { useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import type { TrackRecord } from "@/lib/types";
import { Icon } from "./Icon";

export function TrackLibrary({ tracks, selected, onToggle, onUpload, busy }: {
  tracks: TrackRecord[];
  selected: string[];
  onToggle: (track: TrackRecord) => void;
  onUpload: (files: File[]) => Promise<void>;
  busy: boolean;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);

  async function ingest(files: FileList | null) {
    if (!files?.length) return;
    await onUpload(Array.from(files));
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <section className="panel library-panel" id="tracks">
      <header className="panel-header">
        <div className="panel-title">
          <span className="panel-number">01</span>
          <div><span className="eyebrow">Music</span><h2>Choose your tracks</h2></div>
        </div>
        <div className="panel-meta"><span><b>{selected.length}</b> selected</span></div>
      </header>

      <button
        className={`upload-zone ${dragging ? "is-dragging" : ""}`}
        type="button"
        disabled={busy}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(event: DragEvent<HTMLButtonElement>) => { event.preventDefault(); setDragging(true); }}
        onDragOver={(event: DragEvent<HTMLButtonElement>) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={(event: DragEvent<HTMLButtonElement>) => { event.preventDefault(); setDragging(false); void ingest(event.dataTransfer.files); }}
      >
        <input ref={inputRef} type="file" multiple accept="audio/*,.mp3,.wav,.flac,.m4a,.aiff" hidden onChange={(event: ChangeEvent<HTMLInputElement>) => void ingest(event.target.files)} />
        <span className="upload-zone__icon"><Icon name="upload" /></span>
        <span><strong>Add music</strong><small>Drop files here or choose from your computer</small></span>
        <span className="upload-zone__action">Browse</span>
      </button>

      <div className="track-list scroll-area">
        {tracks.length === 0 ? (
          <div className="empty-state">
            <strong>No music yet</strong><small>Add at least two tracks to begin.</small>
          </div>
        ) : tracks.map((track, index) => {
          const active = selected.includes(track.path);
          return (
            <button
              type="button"
              key={track.path}
              className={`track-row ${active ? "is-selected" : ""}`}
              aria-pressed={active}
              onClick={() => onToggle(track)}
            >
              <span className="track-row__index">{String(index + 1).padStart(2, "0")}</span>
              <span className="track-row__copy">
                <strong>{track.name}</strong>
                <small>
                  {(track.size / 1_048_576).toFixed(1)} MB
                  {track.performance_layers?.complete ? " · mix-ready" : ""}
                  {track.readiness ? ` · ${track.readiness.dj_ready ? "DJ ready" : track.readiness.message}` : ""}
                </small>
              </span>
              <span className="track-row__toggle" aria-hidden="true">{active ? <Icon name="check" /> : "+"}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
