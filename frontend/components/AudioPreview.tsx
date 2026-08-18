"use client";

function formatTime(value: number) {
  if (!Number.isFinite(value)) return "0:00";
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

export function AudioPreview({
  url,
  transitionStart,
  transitionEnd,
  label = "Audio preview",
}: {
  url: string;
  height?: number;
  transitionStart?: number;
  transitionEnd?: number;
  label?: string;
}) {
  return (
    <div className="audio-preview">
      <div className="audio-preview__header">
        <strong>{label}</strong>
        {transitionStart !== undefined && (
          <small>
            Transition {formatTime(transitionStart)} – {formatTime(transitionEnd ?? transitionStart)}
          </small>
        )}
      </div>
      <audio controls preload="metadata" src={url} aria-label={label}>
        Your browser does not support audio playback.
      </audio>
    </div>
  );
}
