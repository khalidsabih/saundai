"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerEvent } from "react";
import {
  cancelJob,
  createAutomaticSet,
  listTracks,
  readAutomaticSetFromJob,
  uploadTracks,
  waitForJob,
  JobCancelledError,
} from "@/lib/api";
import { useSaundSession } from "@/lib/session";
import type { AutomaticSetResult, JobRecord, JourneyMode, TrackRecord } from "@/lib/types";
import { AccountDialog } from "./AccountDialog";
import { BrainWorkspace } from "./BrainWorkspace";
import { FinishedSetPlayer } from "./FinishedSetPlayer";
import { HeroSoundtrack, type HeroSoundtrackHandle } from "./HeroSoundtrack";
import { Icon } from "./Icon";
import { LivePlayer } from "./LivePlayer";
import { AppShell } from "./product/AppShell";
import { BuildProgress } from "./intelligence/BuildProgress";
import { CrateSelector } from "./music/CrateSelector";
import { SetIntentPanel } from "./set/SetIntentPanel";

const INTRO_PROGRESS_DELAY_MS = 180;
const INTRO_PROGRESS_DURATION_MS = 6_800;
const INTRO_SEEN_KEY = "saund.ai.intro.seen.v1";

function outputName(journey: JourneyMode) {
  const stamp = new Date().toISOString().replaceAll(":", "-").replace("T", "_").slice(0, 19);
  return `SAUND_${journey}_${stamp}.wav`;
}

function friendlyError(message: string) {
  const lower = message.toLowerCase();
  if (lower.includes("short-term loudness") || lower.includes("loudness jump")) {
    return {
      title: "SAUND stopped an unsafe handoff.",
      detail: "A transition changed level too abruptly, so the Safety Governor refused to commit it.",
      technical: message,
    };
  }
  if (lower.includes("last-resort transition failed rendered technical validation")) {
    return {
      title: "SAUND could not finish this route safely.",
      detail: "The prepared recovery transition also failed rendered-audio validation. No unsafe set was committed.",
      technical: message,
    };
  }
  if (lower.includes("process pool") || lower.includes("terminated abruptly")) {
    return {
      title: "The analysis worker stopped unexpectedly.",
      detail: "The music is still safe. Retry with the crate unchanged; completed analysis can be reused.",
      technical: message,
    };
  }
  return { title: "SAUND could not continue.", detail: message, technical: message };
}


export function SimpleExperience({ onOpenStudio }: { onOpenStudio: () => void }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const soundtrackRef = useRef<HeroSoundtrackHandle | null>(null);
  const introProgressBarRef = useRef<HTMLSpanElement | null>(null);
  const introProgressRef = useRef(0);
  const {
    selectedPaths,
    setSelectedPaths,
    listener,
    createJourney: journey,
    setCreateJourney: setJourney,
    account,
    authReady,
  } = useSaundSession();

  const [tracks, setTracks] = useState<TrackRecord[]>([]);
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [result, setResult] = useState<AutomaticSetResult | null>(null);
  const [live, setLive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [entered, setEntered] = useState(false);
  const [introResolved, setIntroResolved] = useState(false);
  const [introExiting, setIntroExiting] = useState(false);
  const [introReady, setIntroReady] = useState(false);
  const [soundtrackActive, setSoundtrackActive] = useState(true);
  const [accountDialogOpen, setAccountDialogOpen] = useState(false);
  const [accountDialogReason, setAccountDialogReason] = useState<"account" | "build">("account");
  const [buildAfterAuth, setBuildAfterAuth] = useState(false);
  const [brainOpen, setBrainOpen] = useState(false);
  const [targetMinutes, setTargetMinutes] = useState<number | null>(90);
  const [peakFraction, setPeakFraction] = useState(0.65);
  const [closingMinutes, setClosingMinutes] = useState(10);
  const [endingEnergy, setEndingEnergy] = useState(0.45);
  const [showTechnicalError, setShowTechnicalError] = useState(false);

  const selectedTracks = tracks.filter((track) => selectedPaths.includes(track.path));
  const readySelectedCount = selectedTracks.filter((track) => track.readiness?.dj_ready || track.readiness?.state === "ready").length;
  const failedSelectedCount = selectedTracks.filter((track) => ["failed", "cancelled"].includes(track.readiness?.state ?? "")).length;
  const preparingSelectedCount = selectedTracks.filter((track) => track.readiness && !["ready", "failed", "cancelled"].includes(track.readiness.state)).length;
  const selectedReadinessProgress = selectedTracks.length === 0
    ? 0
    : selectedTracks.reduce((sum, track) => {
      if (track.readiness?.dj_ready || track.readiness?.state === "ready") return sum + 1;
      return sum + Math.max(0, Math.min(1, track.readiness?.progress ?? 0));
    }, 0) / selectedTracks.length;
  const selectedReady = selectedPaths.length >= 2 && readySelectedCount === selectedPaths.length && failedSelectedCount === 0;
  const canStartSession = selectedPaths.length >= 2 && failedSelectedCount === 0 && !job && !uploading;

  const refresh = useCallback(async () => {
    try {
      setTracks(await listTracks("audio"));
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  useEffect(() => {
    const preparing = tracks.some((track) => track.readiness && !["ready", "failed", "cancelled"].includes(track.readiness.state));
    if (!preparing) return;
    const timer = window.setInterval(() => {
      void listTracks("audio").then(setTracks).catch(() => undefined);
    }, 2_000);
    return () => window.clearInterval(timer);
  }, [tracks]);

  useEffect(() => {
    if (window.location.hash === "#brain") {
      setEntered(true);
      setSoundtrackActive(false);
      setIntroReady(true);
      setBrainOpen(true);
      introProgressRef.current = 1;
    }
  }, []);

  useEffect(() => {
    try {
      if (window.localStorage.getItem(INTRO_SEEN_KEY) === "1") {
        setEntered(true);
        setSoundtrackActive(false);
        setIntroReady(true);
        introProgressRef.current = 1;
      }
    } catch {
      // The entrance remains usable when storage is unavailable.
    } finally {
      setIntroResolved(true);
    }
  }, []);

  useEffect(() => {
    if (entered) return;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const bar = introProgressBarRef.current;
    introProgressRef.current = reduceMotion ? 1 : 0;
    setIntroReady(reduceMotion);
    if (bar) bar.style.transform = `scaleX(${introProgressRef.current})`;
    if (reduceMotion) return;

    let frame = 0;
    const startsAt = performance.now() + INTRO_PROGRESS_DELAY_MS;
    const advance = (now: number) => {
      const progress = now <= startsAt ? 0 : Math.min(1, (now - startsAt) / INTRO_PROGRESS_DURATION_MS);
      introProgressRef.current = progress;
      if (bar) bar.style.transform = `scaleX(${progress})`;
      if (progress >= 1) {
        setIntroReady(true);
        return;
      }
      frame = window.requestAnimationFrame(advance);
    };
    frame = window.requestAnimationFrame(advance);
    return () => window.cancelAnimationFrame(frame);
  }, [entered]);

  useEffect(() => {
    const locked = !entered;
    document.documentElement.classList.toggle("saund-intro-locked", locked);
    document.body.classList.toggle("saund-intro-locked", locked);
    return () => {
      document.documentElement.classList.remove("saund-intro-locked");
      document.body.classList.remove("saund-intro-locked");
    };
  }, [entered]);

  function handleIntroPointer(event: PointerEvent<HTMLElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
    const y = ((event.clientY - rect.top) / rect.height - 0.5) * 2;
    event.currentTarget.style.setProperty("--intro-x", `${(x * 8).toFixed(2)}px`);
    event.currentTarget.style.setProperty("--intro-y", `${(y * 6).toFixed(2)}px`);
  }

  function resetIntroPointer(event: PointerEvent<HTMLElement>) {
    event.currentTarget.style.setProperty("--intro-x", "0px");
    event.currentTarget.style.setProperty("--intro-y", "0px");
  }

  function enterCreate() {
    if (!introReady || introProgressRef.current < 1 || introExiting) return;
    setSoundtrackActive(false);
    try { window.localStorage.setItem(INTRO_SEEN_KEY, "1"); } catch { /* optional */ }
    setIntroExiting(true);
    window.setTimeout(() => {
      setEntered(true);
      setIntroExiting(false);
    }, 900);
  }

  async function ingest(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    setError(null);
    try {
      const uploaded = await uploadTracks(Array.from(files));
      const paths = uploaded.files.map((file) => file.path);
      setSelectedPaths((current) => Array.from(new Set([...current, ...paths])));
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setUploading(false);
    }
  }

  async function executeCreateSet() {
    if (selectedPaths.length < 2) return;
    setSoundtrackActive(false);
    setError(null);
    setResult(null);
    setLive(false);
    try {
      const initial = await createAutomaticSet({
        trackPaths: selectedPaths,
        journey,
        listener,
        outputName: outputName(journey),
        targetDurationSeconds: targetMinutes ? targetMinutes * 60 : null,
        peakFraction,
        closingWindowSeconds: closingMinutes * 60,
        endingEnergy,
      });
      const completed = await waitForJob(initial, setJob);
      setResult(readAutomaticSetFromJob(completed));
      setJob(null);
    } catch (caught) {
      setJob(null);
      if (caught instanceof JobCancelledError) {
        setError(null);
        return;
      }
      if (typeof caught === "object" && caught !== null && "status" in caught && (caught as { status?: number }).status === 401) {
        setBuildAfterAuth(true);
        setAccountDialogReason("build");
        setAccountDialogOpen(true);
        return;
      }
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function cancelActiveJob() {
    if (!job?.cancellable) return;
    try {
      setJob(await cancelJob(job.job_id));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  function requestCreateSet() {
    if (!canStartSession) return;
    if (!account) {
      setBuildAfterAuth(true);
      setAccountDialogReason("build");
      setAccountDialogOpen(true);
      return;
    }
    void executeCreateSet();
  }

  function openAccount() {
    setBuildAfterAuth(false);
    setAccountDialogReason("account");
    setAccountDialogOpen(true);
  }

  function handleAuthenticated() {
    const shouldBuild = buildAfterAuth;
    setAccountDialogOpen(false);
    setBuildAfterAuth(false);
    if (shouldBuild) window.setTimeout(() => void executeCreateSet(), 0);
  }

  function reset() {
    setResult(null);
    setLive(false);
    setJob(null);
    setError(null);
  }

  if (!introResolved) {
    return <main className="autonomous-shell autonomous-shell--resolving" aria-label="Opening SAUND.AI" />;
  }

  if (result && live) {
    return <LivePlayer result={result} journey={journey} listener={listener} onClose={() => setLive(false)} />;
  }

  if (brainOpen) {
    return (
      <BrainWorkspace
        result={result}
        journey={journey}
        onBack={() => setBrainOpen(false)}
        onOpenStudio={onOpenStudio}
      />
    );
  }

  if (result) {
    return (
      <FinishedSetPlayer
        result={result}
        journey={journey}
        onReset={reset}
        onStartLive={() => setLive(true)}
        onOpenBrain={() => setBrainOpen(true)}
        listener={listener}
      />
    );
  }

  return (
    <main className={`autonomous-shell ${entered ? "is-entered" : "is-intro"}`} id="top">
      <div className={`saund-global-controls ${entered ? "is-workspace" : "is-intro"}`}>
        <HeroSoundtrack
          ref={soundtrackRef}
          phase={!soundtrackActive ? "off" : (entered || introExiting ? "workspace" : "intro")}
          onSoundStart={() => {
            const video = videoRef.current;
            if (!video || entered) return;
            video.currentTime = 0;
            void video.play().catch(() => undefined);
          }}
        />
      </div>

      {!entered && (
        <section
          className={`saund-intro autonomous-intro ${introExiting ? "is-exiting" : ""}`}
          aria-labelledby="saund-intro-title"
          onPointerMove={handleIntroPointer}
          onPointerLeave={resetIntroPointer}
        >
          <video
            ref={videoRef}
            className="saund-intro__video"
            autoPlay
            muted
            loop
            playsInline
            preload="metadata"
            poster="/media/saund-hero-poster.jpg"
            aria-hidden="true"
          >
            <source media="(max-width: 700px)" src="/media/saund-hero-mobile.mp4" type="video/mp4" />
            <source src="/media/saund-hero-desktop.mp4" type="video/mp4" />
          </video>
          <div className="saund-intro__shade" aria-hidden="true" />
          <div className="saund-intro__grain" aria-hidden="true" />

          <header className="autonomous-intro__top">
            <span className="saund-wordmark"><strong>SAUND</strong><i>.</i><strong>AI</strong></span>
            <span>Autonomous DJ system</span>
          </header>

          <div className="autonomous-intro__statement">
            <p className="autonomous-kicker">LISTEN · THINK · PERFORM · ADAPT</p>
            <h1 id="saund-intro-title">An AI that DJs.</h1>
            <p>Not a playlist. Not a recommendation engine. A musical intelligence that prepares, commits and performs the next move.</p>
          </div>

          <div className="autonomous-intro__system" aria-hidden="true">
            <span><i /> Brain online</span>
            <span><i /> Executive armed</span>
            <span><i /> Safety governed</span>
          </div>

          <button
            type="button"
            className="autonomous-intro__enter"
            onClick={enterCreate}
            disabled={!introReady || introExiting}
            aria-label="Start a session"
            aria-describedby="saund-intro-entry-status"
          >
            <span>
              <small id="saund-intro-entry-status">{introReady ? "Enter the system" : "Opening"}</small>
              <strong>Start a session</strong>
            </span>
            <Icon name="arrow" />
          </button>

          <div className="saund-intro__progress" aria-hidden="true"><span ref={introProgressBarRef} /></div>
        </section>
      )}

      {entered && (
        <AppShell
          active="create"
          accountLabel={account ? account.email.split("@")[0] : authReady ? "Log in" : "Account"}
          onAccount={openAccount}
          onOpenBrain={() => setBrainOpen(true)}
        >
          <section className="autonomous-launch" aria-label="Create an autonomous DJ set">
            <div className="session-launch">
            <header className="session-launch__hero">
              <div>
                <span className="autonomous-kicker">CREATE</span>
                <h1>Tell SAUND how the night should feel.</h1>
              </div>
              <p>Choose the music and the direction. SAUND understands the crate, explores the route, rehearses the hard moments and commits only when the performance is safe.</p>
            </header>

            <div className="session-launch__grid">
              <CrateSelector
                tracks={tracks}
                selectedPaths={selectedPaths}
                uploading={uploading}
                disabled={Boolean(job)}
                readyCount={readySelectedCount}
                preparingCount={preparingSelectedCount}
                failedCount={failedSelectedCount}
                readinessProgress={selectedReadinessProgress}
                onFiles={(files) => void ingest(files)}
                onToggleTrack={(path) => setSelectedPaths((current) =>
                  current.includes(path)
                    ? current.filter((item) => item !== path)
                    : [...current, path]
                )}
              />

              <SetIntentPanel
                journey={journey}
                targetMinutes={targetMinutes}
                peakFraction={peakFraction}
                closingMinutes={closingMinutes}
                endingEnergy={endingEnergy}
                selectedCount={selectedPaths.length}
                readyCount={readySelectedCount}
                preparingCount={preparingSelectedCount}
                failedCount={failedSelectedCount}
                canBuild={canStartSession}
                authenticated={Boolean(account)}
                onJourney={setJourney}
                onTargetMinutes={setTargetMinutes}
                onPeakFraction={setPeakFraction}
                onClosingMinutes={setClosingMinutes}
                onEndingEnergy={setEndingEnergy}
                onBuild={requestCreateSet}
              />
            </div>

            {job && (
              <BuildProgress job={job} onCancel={() => void cancelActiveJob()} />
            )}

            {error && (() => {
              const failure = friendlyError(error);
              return (
                <div className="autonomous-error autonomous-error--safe" role="alert">
                  <Icon name="shield" />
                  <span>
                    <strong>{failure.title}</strong>
                    <small>{failure.detail}</small>
                    <button type="button" className="autonomous-error__details" onClick={() => setShowTechnicalError((value) => !value)}>
                      {showTechnicalError ? "Hide technical details" : "Show technical details"}
                    </button>
                    {showTechnicalError && <code>{failure.technical}</code>}
                  </span>
                  <button type="button" onClick={() => { setError(null); setShowTechnicalError(false); }} aria-label="Dismiss error"><Icon name="close" /></button>
                </div>
              );
            })()}
            </div>
          </section>
        </AppShell>
      )}

      <AccountDialog
        open={accountDialogOpen}
        reason={accountDialogReason}
        onClose={() => { setAccountDialogOpen(false); setBuildAfterAuth(false); }}
        onAuthenticated={handleAuthenticated}
      />
    </main>
  );
}
