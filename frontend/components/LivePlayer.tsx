"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { ChangeEvent, CSSProperties, SyntheticEvent } from "react";
import { getLiveManifest, liveStreamUrl, sendFeedback, subscribeLiveTelemetry } from "@/lib/api";
import type { AutomaticSetResult, JourneyMode, LiveManifest, LiveManifestItem, LiveTelemetrySnapshot } from "@/lib/types";
import { Icon } from "./Icon";

function formatTime(value: number) {
  const safe = Number.isFinite(value) ? Math.max(0, value) : 0;
  const minutes = Math.floor(safe / 60);
  const seconds = Math.floor(safe % 60);
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function label(value?: string | null) {
  return value ? value.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase()) : "Automatic";
}

function itemProgress(item: LiveManifestItem | null, currentTime: number) {
  if (!item) return 0;
  return Math.max(0, Math.min(1, (currentTime - item.set_start) / Math.max(0.001, item.set_end - item.set_start)));
}

function currentBpm(item: LiveManifestItem | null, progress: number, manifest: LiveManifest | null) {
  if (!item) return null;
  if (item.type === "tempo_release" && item.start_bpm !== undefined && item.end_bpm !== undefined) {
    return item.start_bpm + (item.end_bpm - item.start_bpm) * progress;
  }
  if (item.type === "transition" && item.target_bpm !== undefined) return item.target_bpm;
  if (item.track_index !== undefined) return manifest?.ordered_tracks.find((track) => track.index === item.track_index)?.bpm ?? null;
  return null;
}

function currentTrackIndex(item: LiveManifestItem | null) {
  if (!item) return null;
  if (item.type === "track") return item.track_index ?? null;
  if (item.type === "transition") return item.from_index ?? null;
  return item.track_index ?? item.to_index ?? null;
}

type LiveView = "performance" | "brain";
type LiveTrack = LiveManifest["ordered_tracks"][number];

export function LivePlayer({
  result,
  journey,
  listener,
  onClose,
}: {
  result: AutomaticSetResult;
  journey: JourneyMode;
  listener: string;
  onClose: () => void;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [manifest, setManifest] = useState<LiveManifest | null>(null);
  const [manifestError, setManifestError] = useState<string | null>(null);
  const [audioError, setAudioError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [view, setView] = useState<LiveView>("performance");
  const [telemetry, setTelemetry] = useState<LiveTelemetrySnapshot | null>(null);
  const [telemetryConnected, setTelemetryConnected] = useState<boolean | null>(null);
  const reactInstanceId = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const liveSessionId = `${result.plan.plan_id.slice(0, 12)}-${reactInstanceId || "player"}`;

  useEffect(() => {
    let mounted = true;
    setManifestError(null);
    void getLiveManifest(result.plan_path).then(
      (value) => { if (mounted) setManifest(value); },
      () => { if (mounted) setManifestError("The live timeline could not be loaded."); },
    );
    return () => { mounted = false; };
  }, [result.plan_path]);

  useEffect(() => {
    setTelemetry(null);
    setTelemetryConnected(null);
    return subscribeLiveTelemetry(result.plan_path, liveSessionId, setTelemetry, setTelemetryConnected);
  }, [liveSessionId, result.plan_path]);

  const displayManifest = useMemo<LiveManifest | null>(() => {
    if (!manifest) return null;
    if (!telemetry || telemetry.plan_id !== manifest.plan_id || telemetry.timeline.length === 0) return manifest;
    return {
      ...manifest,
      duration_seconds: telemetry.duration_seconds,
      timeline: telemetry.timeline,
      ordered_tracks: telemetry.ordered_tracks,
      runtime_control: telemetry.runtime_control,
    };
  }, [manifest, telemetry]);

  const active = useMemo(() => {
    if (!displayManifest) return null;
    return displayManifest.timeline.find((item) => currentTime >= item.set_start && currentTime < item.set_end)
      ?? displayManifest.timeline.at(-1)
      ?? null;
  }, [currentTime, displayManifest]);

  const activeIndex = displayManifest && active ? displayManifest.timeline.indexOf(active) : -1;
  const nextTrack = displayManifest && activeIndex >= 0
    ? displayManifest.timeline.slice(activeIndex + 1).find((item) => item.type === "track")
    : null;
  const nextTransition = displayManifest && activeIndex >= 0
    ? displayManifest.timeline.slice(Math.max(0, activeIndex)).find((item) => item.type === "transition" && item.set_start >= currentTime)
    : null;
  const previousTransition = displayManifest && activeIndex > 0
    ? [...displayManifest.timeline.slice(0, activeIndex)].reverse().find((item) => item.type === "transition")
    : null;
  const progress = itemProgress(active, currentTime);
  const bpm = currentBpm(active, progress, displayManifest);
  const activeTransition = active?.type === "transition" ? active : active?.type === "tempo_release" ? previousTransition : nextTransition;
  const countdown = nextTransition && nextTransition.set_start > currentTime ? Math.max(0, nextTransition.set_start - currentTime) : 0;
  const scheduleGeneration = telemetry?.schedule_generation ?? displayManifest?.runtime_control?.schedule_generation ?? 0;
  const crowdMode = String(telemetry?.crowd.directive.mode ?? "neutral");
  const crowdConfidence = Number(telemetry?.crowd.directive.confidence ?? telemetry?.crowd.state.confidence ?? 0);
  const safetyAction = telemetry?.safety.action ?? "allow";
  const executiveState = telemetry?.executive.replanning ? "Reconsidering future" : telemetry ? "Holding horizon" : "Synchronizing";
  const liveSafetyState = telemetryConnected === false
    ? { label: "DEGRADED · PREPARED PLAN CONTINUES", tone: "degraded" }
    : telemetry?.executive.replanning
      ? { label: "REPLANNING FUTURE", tone: "thinking" }
      : safetyAction.toLowerCase() !== "allow"
        ? { label: "RECOVERY / SAFETY ACTIVE", tone: "warning" }
        : { label: "PERFORMANCE SAFE", tone: "safe" };
  const dynamicLive = Boolean(displayManifest?.runtime_control?.continuous_replanning);
  const audibleTrackIndex = currentTrackIndex(active);
  const audibleTrack = audibleTrackIndex === null ? null : displayManifest?.ordered_tracks.find((track) => track.index === audibleTrackIndex) ?? null;
  const nextTrackInfo = nextTrack?.track_index === undefined ? null : displayManifest?.ordered_tracks.find((track) => track.index === nextTrack.track_index) ?? null;
  const duration = Math.max(0, displayManifest?.duration_seconds ?? result.plan.estimated_duration_seconds ?? 0);
  const setProgress = duration > 0 ? Math.max(0, Math.min(1, currentTime / duration)) : 0;

  const futureTracks = useMemo(() => {
    const source = displayManifest?.ordered_tracks ?? [];
    if (telemetry?.horizons.future?.length) {
      return telemetry.horizons.future
        .map((slot) => source.find((track) => track.index === slot.track_index))
        .filter((track): track is LiveTrack => Boolean(track))
        .slice(0, 4);
    }
    if (!source.length) return [];
    const currentPosition = audibleTrackIndex === null ? -1 : source.findIndex((track) => track.index === audibleTrackIndex);
    return source.slice(Math.max(0, currentPosition + 2), Math.max(0, currentPosition + 6));
  }, [audibleTrackIndex, displayManifest?.ordered_tracks, telemetry?.horizons.future]);

  const moveTitle = active?.type === "transition"
    ? label(active.technique)
    : active?.type === "tempo_release"
      ? "Tempo ownership"
      : nextTransition
        ? label(nextTransition.technique)
        : "Holding track";
  const moveIntent = activeTransition?.intent ? label(activeTransition.intent) : dynamicLive ? "Prepared move" : "Prepared performance";
  const fromTitle = activeTransition?.from_title ?? audibleTrack?.title ?? "Outgoing";
  const toTitle = activeTransition?.to_title ?? nextTrack?.title ?? "Incoming";
  const executiveDetail = telemetry?.executive.replanning
    ? telemetry?.executive.last_event?.detail || "NOW and NEXT are protected while the Brain evaluates a new FUTURE."
    : telemetry?.detail || (dynamicLive ? "The committed horizon is stable. FUTURE remains eligible for a safe replan." : "Following the prepared performance.");

  async function togglePlayback() {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      try {
        await audio.play();
        setAudioError(null);
      } catch {
        setAudioError("Playback could not start. Check browser audio permissions and that the SAUND.AI engine is running.");
      }
    } else {
      audio.pause();
    }
  }

  function seek(event: ChangeEvent<HTMLInputElement>) {
    const value = Number(event.target.value);
    if (!Number.isFinite(value) || !audioRef.current) return;
    audioRef.current.currentTime = value;
    setCurrentTime(value);
  }

  const errors = manifestError || audioError;

  return (
    <main className={`performance-shell ${view === "brain" ? "is-brain" : "is-performance"}`}>
      <header className="performance-topbar">
        <div className="performance-brand">
          <span className="saund-wordmark"><strong>SAUND</strong><i>.</i><strong>AI</strong></span>
          <small>Autonomous DJ</small>
        </div>

        <nav className="performance-view-switch" aria-label="Performance view">
          <button type="button" className={view === "performance" ? "is-active" : ""} onClick={() => setView("performance")}>Perform</button>
          <button type="button" className={view === "brain" ? "is-active" : ""} onClick={() => setView("brain")}>Brain</button>
        </nav>

        <div className="performance-topbar__status">
          <span className={`performance-live-state ${playing ? "is-live" : ""}`}><i />{playing ? "LIVE" : "READY"}</span>
          <span>{formatTime(currentTime)}</span>
          <button type="button" onClick={onClose} aria-label="Close live session"><Icon name="close" /></button>
        </div>
      </header>

      {errors && (
        <div className="performance-alert" role="alert">
          <Icon name="warning" />
          <span><strong>Performance needs attention.</strong><small>{errors}</small></span>
        </div>
      )}

      {view === "performance" ? (
        <>
        <div className={"performance-safety-banner is-" + liveSafetyState.tone} role="status" aria-live="polite">
          <span><Icon name="shield" />{liveSafetyState.label}</span>
          <small>{telemetryConnected === false
            ? "Telemetry is reconnecting; committed audio execution remains independent."
            : telemetry?.safety.reason || "NOW and NEXT are protected. Hard safety remains authoritative."}</small>
        </div>
        <div className="performance-cockpit">
          <section className="performance-now" aria-label="Now playing">
            <div className="performance-now__status">
              <span><i className={playing ? "is-live" : ""} /> NOW · {playing ? "EXECUTING" : "ARMED"}</span>
              <small>{active?.type === "transition" ? "Transition in progress" : active?.type === "tempo_release" ? "Incoming deck owns tempo" : "Current musical owner"}</small>
            </div>

            <div className="performance-now__identity">
              <h1>{audibleTrack?.title ?? active?.title ?? active?.from_title ?? "Preparing playback"}</h1>
              {active?.type === "transition" && <p>{active.from_title} <span>→</span> {active.to_title}</p>}
              <div className="performance-now__facts">
                <span><strong>{bpm === null ? "—" : bpm.toFixed(1)}</strong><small>BPM</small></span>
                <span><strong>{displayManifest?.ordered_tracks.find((track) => track.index === audibleTrackIndex)?.key ?? "—"}</strong><small>KEY</small></span>
                <span><strong>{Math.round(progress * 100)}%</strong><small>{active?.type === "transition" ? "MOVE" : "SECTION"}</small></span>
              </div>
            </div>

            <div className="performance-transport">
              <button type="button" className="performance-play" onClick={() => void togglePlayback()} aria-label={playing ? "Pause live performance" : "Play live performance"}>
                <Icon name={playing ? "pause" : "play"} />
              </button>
              <div className="performance-transport__rail">
                <input
                  type="range"
                  min={0}
                  max={Math.max(1, duration)}
                  step="0.1"
                  value={Math.min(currentTime, Math.max(1, duration))}
                  onChange={seek}
                  aria-label="Set position"
                  style={{ "--set-progress": `${Math.round(setProgress * 10000) / 100}%` } as CSSProperties}
                />
                <div><span>{formatTime(currentTime)}</span><span>{formatTime(duration)}</span></div>
              </div>
            </div>

            <article className="performance-move" aria-label="Current DJ move">
              <header>
                <div><span>CURRENT MOVE</span><strong>{moveTitle}</strong></div>
                <small>{moveIntent}</small>
              </header>
              <div className="performance-move__signal" style={{ "--move-progress": `${Math.round(progress * 100)}%` } as CSSProperties}>
                <div><span>A</span><i><b /></i><small>{fromTitle}</small></div>
                <div><span>B</span><i><b /></i><small>{toTitle}</small></div>
              </div>
              <footer>
                <span>{activeTransition?.stem_profile ? `Stem choreography · ${label(activeTransition.stem_profile)}` : "Master transition"}</span>
                <span>{countdown > 0 && active?.type !== "transition" ? `Next move in ${formatTime(countdown)}` : active?.type === "transition" ? "Handoff executing" : "Holding phrase"}</span>
              </footer>
            </article>
          </section>

          <aside className="performance-intelligence" aria-label="Executive state">
            <section className="executive-presence">
              <header><span>EXECUTIVE</span><b>GEN {scheduleGeneration}</b></header>
              <div className={`executive-orbit ${telemetry?.executive.replanning ? "is-thinking" : ""}`} aria-hidden="true"><i /><i /><i /></div>
              <strong>{executiveState}</strong>
              <p>{executiveDetail}</p>
              <div className="executive-horizon">
                <span><b>NOW</b><i className="is-locked" />LOCKED</span>
                <span><b>NEXT</b><i className="is-locked" />LOCKED</span>
                <span><b>FUTURE</b><i className="is-fluid" />FLUID</span>
              </div>
            </section>

            <div className="performance-signals">
              <article>
                <span>CROWD</span>
                <strong>{label(crowdMode)}</strong>
                <small>{crowdConfidence > 0 ? `${Math.round(crowdConfidence * 100)}% confidence` : "Learning the room"}</small>
              </article>
              <article className={`is-safety is-${safetyAction.toLowerCase()}`}>
                <span>SAFETY</span>
                <strong>{label(safetyAction)}</strong>
                <small>{telemetry?.safety.reason || "Execution nominal"}</small>
              </article>
            </div>
          </aside>

          <section className="performance-horizon" aria-label="Next and future horizon">
            <article className="performance-next">
              <header><span>NEXT</span><b>LOCKED</b></header>
              <div>
                <small>{nextTrackInfo?.bpm?.toFixed(1) ?? nextTransition?.target_bpm?.toFixed(1) ?? "—"} BPM</small>
                <h2>{nextTrack?.title ?? "Set ending"}</h2>
                <p>{nextTransition ? label(nextTransition.technique) : "Prepared ending"}</p>
              </div>
              <footer>
                <span><Icon name="headphones" /> Prepared</span>
                <span><Icon name="shield" /> Protected</span>
                {nextTransition?.backup && <span><Icon name="refresh" /> Backup ready</span>}
              </footer>
            </article>

            <section className="performance-future">
              <header>
                <div><span>FUTURE</span><small>Executive horizon</small></div>
                <p>{telemetry?.executive.replanning ? "Thinking…" : dynamicLive ? "Eligible to change at a safe boundary" : "Prepared order"}</p>
              </header>
              <div className="performance-future__tracks" key={`future-${scheduleGeneration}`}>
                {futureTracks.length ? futureTracks.map((track, index) => (
                  <article key={`${track.index}-${index}`}>
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <div><strong>{track.title}</strong><small>{track.bpm.toFixed(1)} BPM · {track.key}</small></div>
                    <i />
                  </article>
                )) : (
                  <div className="performance-future__empty">Closing the session.</div>
                )}
              </div>
              <footer>
                <span>{telemetryConnected === false ? "Telemetry reconnecting · audio unaffected" : `Schedule generation ${scheduleGeneration}`}</span>
                {telemetry?.executive.last_event?.action && <small>{label(telemetry.executive.last_event.action)}</small>}
              </footer>
            </section>
          </section>
        </div>
        </>
      ) : (
        <div className="brain-console">
          <header className="brain-console__hero">
            <div><span className="autonomous-kicker">DECISION SYSTEM</span><h1>What the DJ is doing underneath.</h1></div>
            <p>Performance remains authoritative. This view exposes the bounded reasoning and execution state without changing it.</p>
          </header>

          <div className="brain-console__grid">
            <section className="brain-card brain-card--executive">
              <header><span>EXECUTIVE</span><b>{executiveState}</b></header>
              <dl>
                <div><dt>Generation</dt><dd>{scheduleGeneration}</dd></div>
                <div><dt>Cycle</dt><dd>{telemetry?.executive.cycle ?? 0}</dd></div>
                <div><dt>Policy</dt><dd>{label(telemetry?.executive.transition_policy_authority ?? "bounded advisory")}</dd></div>
                <div><dt>Crowd authority</dt><dd>{label(telemetry?.executive.crowd_state_authority ?? "bounded advisory")}</dd></div>
              </dl>
              <p>{executiveDetail}</p>
            </section>

            <section className="brain-card">
              <header><span>PLAYBACK STATE</span><b>{label(telemetry?.playback_state.executor_phase ?? "ready")}</b></header>
              <dl>
                <div><dt>Track</dt><dd>{telemetry?.playback_state.current_track ?? audibleTrackIndex ?? "—"}</dd></div>
                <div><dt>Source cursor</dt><dd>{telemetry ? `${telemetry.playback_state.source_cursor.toFixed(2)}s` : "—"}</dd></div>
                <div><dt>Edge</dt><dd>{telemetry?.playback_state.edge_position ?? "—"}</dd></div>
                <div><dt>Expected</dt><dd>{label(telemetry?.playback_state.expected_command ?? "none")}</dd></div>
              </dl>
            </section>

            <section className="brain-card">
              <header><span>CROWD DIRECTIVE</span><b>{label(crowdMode)}</b></header>
              <dl>
                <div><dt>Confidence</dt><dd>{Math.round(crowdConfidence * 100)}%</dd></div>
                <div><dt>Engagement</dt><dd>{typeof telemetry?.crowd.state.engagement === "number" ? `${Math.round(Number(telemetry.crowd.state.engagement) * 100)}%` : "—"}</dd></div>
                <div><dt>Momentum</dt><dd>{typeof telemetry?.crowd.state.momentum === "number" ? Number(telemetry.crowd.state.momentum).toFixed(2) : "—"}</dd></div>
                <div><dt>Authority</dt><dd>Bounded</dd></div>
              </dl>
            </section>

            <section className={`brain-card brain-card--safety is-${safetyAction.toLowerCase()}`}>
              <header><span>SAFETY GOVERNOR</span><b>{label(safetyAction)}</b></header>
              <p>{telemetry?.safety.reason || "Execution contract healthy."}</p>
              <dl>
                <div><dt>NOW</dt><dd>Immutable</dd></div>
                <div><dt>NEXT</dt><dd>Protected</dd></div>
                <div><dt>FUTURE</dt><dd>{dynamicLive ? "Replannable" : "Prepared"}</dd></div>
              </dl>
            </section>

            <section className="brain-card brain-card--move">
              <header><span>DJ MOVE</span><b>{moveTitle}</b></header>
              <div className="brain-move-route">
                <span><small>OUT</small><strong>{fromTitle}</strong></span>
                <i><Icon name="arrow" /></i>
                <span><small>IN</small><strong>{toTitle}</strong></span>
              </div>
              <dl>
                <div><dt>Intent</dt><dd>{moveIntent}</dd></div>
                <div><dt>Target BPM</dt><dd>{activeTransition?.target_bpm?.toFixed(1) ?? "—"}</dd></div>
                <div><dt>Stem profile</dt><dd>{label(activeTransition?.stem_profile ?? "none")}</dd></div>
                <div><dt>Phase nudge</dt><dd>{activeTransition?.phase_nudge_seconds_b ? `${(activeTransition.phase_nudge_seconds_b * 1000).toFixed(0)} ms` : "None"}</dd></div>
              </dl>
            </section>

            <section className="brain-card brain-card--recovery">
              <header><span>RECOVERY</span><b>{activeTransition?.backup ? "Prepared" : "Primary only"}</b></header>
              {activeTransition?.backup ? (
                <>
                  <div className="brain-recovery-program"><small>PRIMARY</small><strong>{label(activeTransition.primary?.technique ?? activeTransition.technique)}</strong></div>
                  <div className="brain-recovery-program"><small>BACKUP</small><strong>{label(activeTransition.backup.technique)}</strong></div>
                </>
              ) : <p>No alternate execution is exposed for this edge. The committed primary remains authoritative.</p>}
            </section>
          </div>
        </div>
      )}

      <audio
        ref={audioRef}
        className="performance-audio"
        src={liveStreamUrl(result.plan_path, "192k", liveSessionId)}
        autoPlay
        onPlay={() => { setPlaying(true); setAudioError(null); }}
        onPause={() => setPlaying(false)}
        onError={() => { setPlaying(false); setAudioError("The audio stream could not be opened. Check that the SAUND.AI engine is still running."); }}
        onTimeUpdate={(event: SyntheticEvent<HTMLAudioElement>) => setCurrentTime(event.currentTarget.currentTime)}
        onEnded={() => {
          setPlaying(false);
          void sendFeedback({
            event: "playback_completed",
            listener,
            planId: result.plan.plan_id,
            planPath: result.plan_path,
            journey,
          }).catch(() => undefined);
        }}
      />

      <footer className="performance-footer">
        <span><Icon name="shield" /> Brain imagines · Executive commits · Executor performs</span>
        <span>{result.plan.order.length} tracks · {label(journey)} · GEN {scheduleGeneration}</span>
      </footer>
    </main>
  );
}
