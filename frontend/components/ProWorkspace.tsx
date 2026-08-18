"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  cancelJob,
  createPlan,
  createPreviews,
  createStudioTransitionPreview,
  getBrainStatus,
  getReleaseReadiness,
  listTracks,
  mediaUrl,
  readPlanFromJob,
  readStudioTransitionPreview,
  renderSet,
  sendFeedback,
  uploadTracks,
  waitForJob,
  JobCancelledError,
} from "@/lib/api";
import { useSaundSession } from "@/lib/session";
import type { BrainStatus, JobRecord, PreviewResult, ReleaseReadiness, RenderResult, SetPlan, StudioTransitionPreviewResult, TrackRecord } from "@/lib/types";
import { AccountDialog } from "./AccountDialog";
import { ControlDeck } from "./ControlDeck";
import { Icon } from "./Icon";
import { JobProgress } from "./JobProgress";
import { PlanView } from "./PlanView";
import { RenderPanel } from "./RenderPanel";
import { SetBuilder } from "./SetBuilder";
import { TrackLibrary } from "./TrackLibrary";
import { TransitionInspector } from "./TransitionInspector";
import { StudioIntelligenceWorkbench, type StudioBranch } from "./StudioIntelligenceWorkbench";
import { orderedPathForCandidate, type StudioMoveCandidate } from "@/lib/studio-intelligence";

type PendingStudioAction =
  | { kind: "plan" }
  | { kind: "previews" }
  | { kind: "render"; confirmed: boolean }
  | null;

function isLoginRequired(error: unknown) {
  return typeof error === "object" && error !== null && "status" in error && (error as { status?: number }).status === 401;
}

function labelRuntime(value: string) {
  if (value === "advisory") return "Learned advisory";
  if (value === "shadow") return "Shadow learning";
  return value.replaceAll("_", " ");
}

const navItems = [
  { href: "#studio-setup", label: "Crate", icon: "library" as const },
  { href: "#studio-intelligence", label: "Intelligence", icon: "spark" as const },
  { href: "#transition", label: "Audition", icon: "headphones" as const },
  { href: "#render", label: "Finish", icon: "render" as const },
];

type CollaborationMode = "autonomous" | "guided" | "manual";

export function ProWorkspace({ onExit }: { onExit: () => void }) {
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const alternativeAudioRef = useRef<HTMLAudioElement | null>(null);
  const {
    selectedPaths,
    setSelectedPaths,
    listener,
    studioJourney: journey,
    setStudioJourney: setJourney,
    account,
    authReady,
    refreshAccount,
  } = useSaundSession();
  const [tracks, setTracks] = useState<TrackRecord[]>([]);
  const [mode, setMode] = useState<"auto" | "manual">("auto");
  const [collaborationMode, setCollaborationMode] = useState<CollaborationMode>("guided");
  const [tempoPolicy, setTempoPolicy] = useState<"safe" | "stretch" | "cut">("safe");
  const [style, setStyle] = useState<"auto" | "safe" | "creative" | "emergency">("auto");
  const [coverage, setCoverage] = useState<"fast" | "balanced" | "long">("balanced");
  const [plan, setPlan] = useState<SetPlan | null>(null);
  const [planPath, setPlanPath] = useState<string | null>(null);
  const [activeTransition, setActiveTransition] = useState(0);
  const [previews, setPreviews] = useState<PreviewResult | null>(null);
  const [rendered, setRendered] = useState<RenderResult | null>(null);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<"connecting" | "online" | "offline">("connecting");
  const [brain, setBrain] = useState<BrainStatus | null>(null);
  const [release, setRelease] = useState<ReleaseReadiness | null>(null);
  const [accountDialogOpen, setAccountDialogOpen] = useState(false);
  const [accountDialogReason, setAccountDialogReason] = useState<"account" | "build">("account");
  const [pendingAction, setPendingAction] = useState<PendingStudioAction>(null);
  const [setupExpanded, setSetupExpanded] = useState(true);
  const [queuedPreviewIndex, setQueuedPreviewIndex] = useState<number | null>(null);
  const [previewPlaybackIndex, setPreviewPlaybackIndex] = useState<number | null>(null);
  const [previewState, setPreviewState] = useState<"idle" | "preparing" | "ready" | "playing" | "paused">("idle");
  const [branches, setBranches] = useState<StudioBranch[]>([]);
  const [activeBranchId, setActiveBranchId] = useState<string | null>(null);
  const [alternativePreview, setAlternativePreview] = useState<StudioTransitionPreviewResult | null>(null);
  const [alternativePreviewPlaying, setAlternativePreviewPlaying] = useState(false);
  const [previewingVariant, setPreviewingVariant] = useState<string | null>(null);
  const [studioNotice, setStudioNotice] = useState<string | null>(null);

  const selectedTracks = useMemo(
    () => selectedPaths.map((path) => tracks.find((track) => track.path === path)).filter(Boolean) as TrackRecord[],
    [selectedPaths, tracks],
  );

  const resetOutput = useCallback(() => {
    previewAudioRef.current?.pause();
    alternativeAudioRef.current?.pause();
    setPlan(null);
    setPlanPath(null);
    setPreviews(null);
    setRendered(null);
    setActiveTransition(0);
    setQueuedPreviewIndex(null);
    setPreviewPlaybackIndex(null);
    setPreviewState("idle");
    setAlternativePreview(null);
    setAlternativePreviewPlaying(false);
    setPreviewingVariant(null);
    setBranches([]);
    setActiveBranchId(null);
    setStudioNotice(null);
    setSetupExpanded(true);
  }, []);

  const resetAuditions = useCallback(() => {
    previewAudioRef.current?.pause();
    alternativeAudioRef.current?.pause();
    setPreviews(null);
    setRendered(null);
    setQueuedPreviewIndex(null);
    setPreviewPlaybackIndex(null);
    setPreviewState("idle");
    setAlternativePreview(null);
    setAlternativePreviewPlaying(false);
    setPreviewingVariant(null);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [values, brainStatus, releaseStatus] = await Promise.all([
        listTracks("audio"),
        getBrainStatus(),
        getReleaseReadiness().catch(() => null),
      ]);
      setTracks(values);
      setBrain(brainStatus);
      setRelease(releaseStatus);
      setStatus("online");
    } catch (caught) {
      setStatus("offline");
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
    if (authReady && !account) resetOutput();
  }, [account, authReady, resetOutput]);

  async function handleUpload(files: File[]) {
    setBusy(true);
    setError(null);
    try {
      const result = await uploadTracks(files);
      await refresh();
      setSelectedPaths((current) => [...current, ...result.files.map((file) => file.path).filter((path) => !current.includes(path))]);
      resetOutput();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }

  function toggleTrack(track: TrackRecord) {
    setSelectedPaths((current) => current.includes(track.path) ? current.filter((path) => path !== track.path) : [...current, track.path]);
    resetOutput();
  }

  function coveragePolicy() {
    return coverage === "fast"
      ? { minimum_body_seconds: 15, minimum_body_fraction: 0.2 }
      : coverage === "long"
        ? { minimum_body_seconds: 60, minimum_body_fraction: 0.6 }
        : { minimum_body_seconds: 30, minimum_body_fraction: 0.35 };
  }

  async function runStudioPlanning(input: {
    paths: string[];
    manual: boolean;
    preferredStyle: "auto" | "safe" | "creative" | "emergency";
    label: string;
    detail: string;
    kind: StudioBranch["kind"];
    replacePrimary?: boolean;
    authPrompt?: boolean;
  }) {
    if (input.paths.length < 2) return;
    setBusy(true);
    setError(null);
    setStudioNotice(null);
    if (input.replacePrimary) resetOutput();
    else resetAuditions();
    try {
      const initial = await createPlan({
        track_paths: input.paths,
        manual: input.manual,
        experience: "pro",
        // Studio proposals still use the normal planner / Executive path.
        automatic_delivery: false,
        listener,
        tempo_policy: tempoPolicy,
        preferred_style: input.preferredStyle,
        journey,
        ...coveragePolicy(),
      });
      const completed = await waitForJob(initial, setJob);
      const result = await readPlanFromJob(completed);
      const branch: StudioBranch = {
        id: `${input.kind}-${result.plan.plan_id}`,
        label: input.label,
        detail: input.detail,
        plan: result.plan,
        planPath: result.planPath,
        kind: input.kind,
      };
      setBranches((current) => input.replacePrimary
        ? [branch]
        : [...current.filter((item) => item.id !== branch.id), branch]);
      setActiveBranchId(branch.id);
      setPlan(result.plan);
      setPlanPath(result.planPath);
      setActiveTransition(0);
      setSetupExpanded(false);
      setStudioNotice(input.replacePrimary ? "The Executive committed the first Studio route." : `${input.label} is ready to compare.`);
      window.setTimeout(() => document.querySelector("#studio-intelligence")?.scrollIntoView({ behavior: "smooth", block: "start" }), 120);
    } catch (caught) {
      if (caught instanceof JobCancelledError) {
        setError(null);
      } else if (isLoginRequired(caught) && input.authPrompt) {
        await refreshAccount();
        setPendingAction({ kind: "plan" });
        setAccountDialogReason("build");
        setAccountDialogOpen(true);
      } else {
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    } finally {
      setBusy(false);
      setJob(null);
    }
  }

  async function buildPlan() {
    await runStudioPlanning({
      paths: selectedPaths,
      manual: mode === "manual",
      preferredStyle: style,
      label: "Main direction",
      detail: `${mode === "manual" ? "Your route" : "Executive route"} · ${journey}`,
      kind: "primary",
      replacePrimary: true,
      authPrompt: true,
    });
  }

  async function exploreBranch(preferredStyle: "safe" | "creative" | "auto") {
    if (!plan) return;
    const kind: StudioBranch["kind"] = preferredStyle === "safe" ? "safe" : preferredStyle === "creative" ? "creative" : "reroute";
    const label = preferredStyle === "safe" ? "Safer route" : preferredStyle === "creative" ? "Bolder route" : "Fresh route";
    const detail = preferredStyle === "safe"
      ? "Lower-risk search with the same crate"
      : preferredStyle === "creative"
        ? "More expressive transition search"
        : "New autonomous route search";
    await runStudioPlanning({
      paths: selectedPaths,
      manual: false,
      preferredStyle,
      label,
      detail,
      kind,
    });
  }

  async function repairCurrentRoute() {
    if (!plan) return;
    const orderedPaths = plan.ordered_tracks.map((track) => track.path);
    await runStudioPlanning({
      paths: orderedPaths,
      manual: true,
      preferredStyle: "safe",
      label: "Repaired route",
      detail: "Order held · safer transition preparation",
      kind: "repair",
    });
  }

  async function tryCandidateNext(candidate: StudioMoveCandidate) {
    if (!plan) return;
    const candidatePaths = orderedPathForCandidate(plan, activeTransition, candidate.nextIndex);
    if (!candidatePaths) {
      setStudioNotice(`${candidate.title} is already the committed next track.`);
      return;
    }
    await runStudioPlanning({
      paths: candidatePaths,
      manual: true,
      preferredStyle: style,
      label: `Try ${candidate.title}`,
      detail: `Human-guided next track · ${candidate.technique.replaceAll("_", " ")}`,
      kind: "candidate",
    });
  }

  function selectBranch(branch: StudioBranch) {
    resetAuditions();
    setPlan(branch.plan);
    setPlanPath(branch.planPath);
    setActiveBranchId(branch.id);
    setActiveTransition(0);
    setStudioNotice(`Viewing ${branch.label}. Nothing else was discarded.`);
  }


  async function generatePreviews() {
    if (!planPath) return;
    setBusy(true);
    setError(null);
    try {
      const completed = await waitForJob(await createPreviews(planPath), setJob);
      setPreviews(completed.result as unknown as PreviewResult);
    } catch (caught) {
      if (caught instanceof JobCancelledError) {
        setError(null);
      } else if (isLoginRequired(caught)) {
        await refreshAccount();
        setPendingAction({ kind: "previews" });
        setAccountDialogReason("build");
        setAccountDialogOpen(true);
      } else {
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    } finally {
      setBusy(false);
      setJob(null);
    }
  }

  useEffect(() => {
    if (!previews || queuedPreviewIndex === null) return;
    const url = previews.preview_urls[queuedPreviewIndex];
    if (!url) {
      setQueuedPreviewIndex(null);
      setPreviewState("idle");
      return;
    }
    setPreviewPlaybackIndex(queuedPreviewIndex);
    setQueuedPreviewIndex(null);
  }, [previews, queuedPreviewIndex]);

  const activePreviewUrl = previewPlaybackIndex === null
    ? null
    : previews?.preview_urls[previewPlaybackIndex]
      ? mediaUrl(previews.preview_urls[previewPlaybackIndex])
      : null;

  useEffect(() => {
    if (!activePreviewUrl) return;
    const audio = previewAudioRef.current;
    if (!audio) return;
    audio.currentTime = 0;
    void audio.play().then(
      () => setPreviewState("playing"),
      () => setPreviewState("ready"),
    );
  }, [activePreviewUrl, previewPlaybackIndex]);

  function activateTransition(index: number) {
    alternativeAudioRef.current?.pause();
    setAlternativePreview(null);
    setAlternativePreviewPlaying(false);
    setPreviewingVariant(null);
    setActiveTransition(index);
    const url = previews?.preview_urls[index];
    const audio = previewAudioRef.current;

    if (!url) {
      if (previewState === "preparing") {
        // Preview rendering is plan-wide. While it is already running, let a
        // later click simply become the target that auto-plays on completion.
        setQueuedPreviewIndex(index);
        return;
      }
      if (busy) return;
      setQueuedPreviewIndex(index);
      setPreviewState("preparing");
      requestProtectedAction({ kind: "previews" });
      return;
    }

    if (previewPlaybackIndex === index && audio) {
      if (!audio.paused) {
        audio.pause();
        return;
      }
      audio.currentTime = 0;
      void audio.play().catch(() => setPreviewState("ready"));
      return;
    }

    setPreviewPlaybackIndex(index);
  }

  function inspectTransition(index: number) {
    alternativeAudioRef.current?.pause();
    setAlternativePreview(null);
    setAlternativePreviewPlaying(false);
    setPreviewingVariant(null);
    setActiveTransition(index);
  }

  async function auditionVariant(variantIndex: number) {
    if (!planPath || !plan?.transitions[activeTransition]) return;
    const previewKey = `${activeTransition}:${variantIndex}`;
    setBusy(true);
    setError(null);
    setStudioNotice(null);
    setPreviewingVariant(previewKey);
    previewAudioRef.current?.pause();
    alternativeAudioRef.current?.pause();
    setAlternativePreviewPlaying(false);
    try {
      const initial = await createStudioTransitionPreview({
        planPath,
        transitionIndex: activeTransition,
        alternativeIndex: variantIndex === 0 ? null : variantIndex - 1,
      });
      const completed = await waitForJob(initial, setJob);
      const result = readStudioTransitionPreview(completed);
      setAlternativePreview(result);
      window.setTimeout(() => {
        const audio = alternativeAudioRef.current;
        if (!audio) return;
        audio.currentTime = 0;
        void audio.play().catch(() => setAlternativePreviewPlaying(false));
      }, 0);
    } catch (caught) {
      if (caught instanceof JobCancelledError) setError(null);
      else setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
      setJob(null);
      setPreviewingVariant(null);
    }
  }

  function toggleAlternativePreview() {
    const audio = alternativeAudioRef.current;
    if (!audio) return;
    if (audio.paused) void audio.play().catch(() => setAlternativePreviewPlaying(false));
    else audio.pause();
  }

  async function preferVariant(variantIndex: number) {
    if (!planPath || !plan || variantIndex <= 0) return;
    const edge = plan.transitions[activeTransition];
    const alternative = edge?.alternatives[variantIndex - 1];
    if (!edge || !alternative) return;
    setStudioNotice("Recording your transition preference…");
    try {
      const result = await sendFeedback({
        event: "alternative_selected",
        listener,
        planId: plan.plan_id,
        planPath,
        transitionIndex: activeTransition,
        value: Math.max(-1, Math.min(1, alternative.score)),
        journey,
        metadata: {
          source: "studio_intelligence_v1",
          selected_execution_signature: alternative.execution_signature,
          selected_technique: alternative.technique,
          selected_intent: alternative.intent,
          primary_execution_signature: edge.plan.execution_signature,
          primary_technique: edge.plan.technique,
          rehearsal_only: true,
        },
      });
      const applied = result.personalization_applied === true;
      setStudioNotice(applied
        ? "Preference learned for future decisions. The current committed plan was not changed."
        : "Preference recorded. Enable personalization if you want it to influence future decisions.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setStudioNotice(null);
    }
  }

  async function generateSet(confirmed: boolean) {
    if (!planPath || !plan) return;
    setBusy(true);
    setError(null);
    try {
      const initial = await renderSet({
        planPath,
        outputName: `SAUND_${plan.plan_id.slice(0, 8)}.wav`,
        confirm: confirmed,
        confirmationToken: plan.confirmation_token,
      });
      const completed = await waitForJob(initial, setJob);
      setRendered(completed.result as unknown as RenderResult);
    } catch (caught) {
      if (caught instanceof JobCancelledError) {
        setError(null);
      } else if (isLoginRequired(caught)) {
        await refreshAccount();
        setPendingAction({ kind: "render", confirmed });
        setAccountDialogReason("build");
        setAccountDialogOpen(true);
      } else {
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    } finally {
      setBusy(false);
      setJob(null);
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

  function requestProtectedAction(action: Exclude<PendingStudioAction, null>) {
    if (!account) {
      setPendingAction(action);
      setAccountDialogReason("build");
      setAccountDialogOpen(true);
      return;
    }
    if (action.kind === "plan") void buildPlan();
    if (action.kind === "previews") void generatePreviews();
    if (action.kind === "render") void generateSet(action.confirmed);
  }

  function openAccount() {
    setPendingAction(null);
    setAccountDialogReason("account");
    setAccountDialogOpen(true);
  }

  function handleAuthenticated() {
    const action = pendingAction;
    setPendingAction(null);
    setAccountDialogOpen(false);
    if (!action) return;
    window.setTimeout(() => {
      if (action.kind === "plan") void buildPlan();
      if (action.kind === "previews") void generatePreviews();
      if (action.kind === "render") void generateSet(action.confirmed);
    }, 0);
  }

  return (
    <div className="studio-shell" id="top">
      <header className="studio-header">
        <button className="studio-brand" type="button" onClick={onExit} aria-label="Return to Create">
          <span className="saund-wordmark"><strong>SAUND</strong><i>.</i><strong>AI</strong></span>
          <small>Studio</small>
        </button>

        <nav className="studio-nav" aria-label="Studio navigation">
          {navItems.map((item) => (
            <a href={item.href} key={item.href}>
              <Icon name={item.icon} />
              <span>{item.label}</span>
            </a>
          ))}
        </nav>

        <div className="studio-header__actions">
          <button className="studio-create-link" type="button" onClick={onExit}>Create</button>
          <a className="studio-create-link" href="/lab">Lab</a>
          <button
            className="studio-create-link"
            type="button"
            onClick={openAccount}
            aria-label={account ? `Account: ${account.email}` : "Log in to SAUND.AI"}
          >
            {account ? "Account" : authReady ? "Log in" : "Account"}
          </button>
          <span className={`studio-status studio-status--${status}`}>
            <i />{status === "online" ? "Ready" : status === "offline" ? "Offline" : "Connecting"}
          </span>
        </div>
      </header>

      <main className="studio-main">
        <section className="studio-intro studio-intro--editorial studio-intro--optimized studio-intro--intelligent">
          <div className="studio-intro__copy">
            <span className="studio-index">Studio / Human × Executive</span>
            <h1>Direct the DJ.<br /><em>Explore before you commit.</em></h1>
            <p>Studio is the collaborative layer: shape intent, compare real branches, hear executable alternatives, and keep SAUND's full intelligence in every decision.</p>

            <div className="studio-collaboration" role="group" aria-label="Studio collaboration mode">
              {([
                ["autonomous", "Autonomous", "SAUND decides. You inspect and teach."],
                ["guided", "Guided", "You direct intent. SAUND solves execution."],
                ["manual", "Manual", "You fix the route. SAUND still rehearses and protects it."],
              ] as const).map(([value, title, detail]) => (
                <button
                  key={value}
                  type="button"
                  className={collaborationMode === value ? "is-active" : ""}
                  aria-pressed={collaborationMode === value}
                  onClick={() => {
                    setCollaborationMode(value);
                    setMode(value === "manual" ? "manual" : "auto");
                    resetOutput();
                  }}
                >
                  <strong>{title}</strong>
                  <small>{detail}</small>
                </button>
              ))}
            </div>

            <div className="studio-intro__facts" aria-label="Current Studio state">
              <span><small>Crate</small><strong>{selectedPaths.length} selected</strong></span>
              <span><small>Move brain</small><strong>{brain?.dj_move_brain?.enabled ? "Complete DJMove" : brain?.bootstrap_runtime?.mode ? labelRuntime(brain.bootstrap_runtime.mode) : "Loading"}</strong></span>
              <span><small>Authority</small><strong>{release ? (release.authority.planner_mode === "move_authoritative" ? "Executive route" : "Shadow route") : "Checking"}</strong></span>
            </div>
          </div>
          <ControlDeck selectedCount={selectedPaths.length} totalCount={tracks.length} tempoPolicy={tempoPolicy} onTempoPolicy={(value) => { setTempoPolicy(value); resetOutput(); }} style={style} onStyle={(value) => { setStyle(value); resetOutput(); }} coverage={coverage} onCoverage={(value) => { setCoverage(value); resetOutput(); }} mode={mode} busy={busy} onBuild={() => requestProtectedAction({ kind: "plan" })} />
        </section>

        {error && (
          <div className="error-banner" role="alert">
            <Icon name="warning" /><div><strong>We could not complete that.</strong><span>{error}</span></div><button type="button" onClick={() => setError(null)} aria-label="Dismiss error"><Icon name="close" /></button>
          </div>
        )}

        {studioNotice && (
          <div className="studio-notice" role="status">
            <Icon name="spark" />
            <span>{studioNotice}</span>
            <button type="button" onClick={() => setStudioNotice(null)} aria-label="Dismiss Studio notice"><Icon name="close" /></button>
          </div>
        )}

        <JobProgress job={busy ? job : null} onCancel={() => void cancelActiveJob()} />

        <section className="studio-setup-shell" id="studio-setup">
          {plan && (
            <button className="studio-setup-toggle" type="button" onClick={() => setSetupExpanded((value) => !value)} aria-expanded={setupExpanded}>
              <span>{setupExpanded ? "Hide setup" : "Edit music & direction"}</span>
              <small>{selectedPaths.length} tracks · {journey}</small>
              <Icon name="sliders" />
            </button>
          )}
          {setupExpanded && (
            <section className="workspace-grid workspace-grid--editorial">
              <TrackLibrary tracks={tracks} selected={selectedPaths} onToggle={toggleTrack} onUpload={handleUpload} busy={busy} />
              <SetBuilder tracks={selectedTracks} onReorder={(values) => { setSelectedPaths(values.map((track) => track.path)); resetOutput(); }} onRemove={(path) => { setSelectedPaths((current) => current.filter((item) => item !== path)); resetOutput(); }} mode={mode} onModeChange={(value) => { setMode(value); setCollaborationMode(value === "manual" ? "manual" : "guided"); resetOutput(); }} journey={journey} onJourneyChange={(value) => { setJourney(value); resetOutput(); }} />
            </section>
          )}
        </section>

        {plan && (
          <div className="results-stack results-stack--editorial">
            <StudioIntelligenceWorkbench
              plan={plan}
              activeTransition={activeTransition}
              branches={branches}
              activeBranchId={activeBranchId}
              busy={busy}
              previewingVariant={previewingVariant}
              alternativePreview={alternativePreview}
              alternativePreviewPlaying={alternativePreviewPlaying}
              onSelectTransition={inspectTransition}
              onSelectBranch={selectBranch}
              onExplore={(value) => void exploreBranch(value)}
              onRepairRoute={() => void repairCurrentRoute()}
              onTryCandidate={(candidate) => void tryCandidateNext(candidate)}
              onPreviewVariant={(variantIndex) => void auditionVariant(variantIndex)}
              onToggleAlternativePreview={toggleAlternativePreview}
              onPreferVariant={(variantIndex) => void preferVariant(variantIndex)}
            />

            <audio
              ref={alternativeAudioRef}
              src={alternativePreview?.preview_url ? mediaUrl(alternativePreview.preview_url) : undefined}
              preload="metadata"
              onPlay={() => setAlternativePreviewPlaying(true)}
              onPause={() => setAlternativePreviewPlaying(false)}
              onEnded={() => setAlternativePreviewPlaying(false)}
              onError={() => setAlternativePreviewPlaying(false)}
            />

            <details className="studio-route-detail">
              <summary>
                <span><strong>Full performance timeline</strong><small>Track structure, deck ownership and the committed handoffs</small></span>
                <Icon name="arrow" />
              </summary>
              <PlanView
                plan={plan}
                activeTransition={activeTransition}
                onActiveTransition={activateTransition}
                playingTransition={previewState === "playing" ? previewPlaybackIndex : null}
                preparingTransition={previewState === "preparing" ? queuedPreviewIndex : null}
              />
            </details>

            <section className="studio-preview-dock" id="transition" aria-live="polite">
              <div>
                <span className="eyebrow">Transition audition</span>
                <strong>{plan.transitions[activeTransition]
                  ? `${plan.tracks[plan.transitions[activeTransition].from_index].title} → ${plan.tracks[plan.transitions[activeTransition].to_index].title}`
                  : "Select a transition"}</strong>
                <small>{previewState === "preparing"
                  ? "SAUND is preparing real transition previews."
                  : previewState === "playing"
                    ? "Playing the rendered handoff you selected."
                    : previews
                      ? "Click any transition in the timeline to hear it immediately."
                      : "Click a transition once to prepare and hear the real rendered handoff."}</small>
              </div>
              {activePreviewUrl && (
                <button
                  type="button"
                  className="studio-preview-dock__toggle"
                  onClick={() => {
                    const audio = previewAudioRef.current;
                    if (!audio) return;
                    if (audio.paused) void audio.play().catch(() => setPreviewState("ready"));
                    else audio.pause();
                  }}
                >
                  <Icon name={previewState === "playing" ? "pause" : "play"} />
                  {previewState === "playing" ? "Pause" : "Play again"}
                </button>
              )}
              <audio
                ref={previewAudioRef}
                src={activePreviewUrl ?? undefined}
                preload="metadata"
                onPlay={() => setPreviewState("playing")}
                onPause={() => setPreviewState((state) => state === "playing" ? "paused" : state)}
                onEnded={() => setPreviewState("ready")}
                onError={() => setPreviewState("ready")}
              />
            </section>

            <details className="studio-transition-detail">
              <summary><span>Why this transition?</span><small>Decks, stems, timing and decision evidence</small><Icon name="arrow" /></summary>
              <TransitionInspector plan={plan} index={activeTransition} previewUrl={previews?.preview_urls[activeTransition] ? mediaUrl(previews.preview_urls[activeTransition]) : undefined} previewReportPath={previews?.report_path} />
            </details>
            <RenderPanel plan={plan} onPreview={() => requestProtectedAction({ kind: "previews" })} onRender={(confirmed) => requestProtectedAction({ kind: "render", confirmed })} previewBusy={busy} renderBusy={busy} rendered={rendered} />
          </div>
        )}

        <footer className="studio-footer">
          <span className="saund-wordmark"><strong>SAUND</strong><i>.</i><strong>AI</strong></span>
          <span>Hear ahead.</span>
          <span>Local / rehearsed / deterministic execution</span>
        </footer>
      </main>

      <AccountDialog
        open={accountDialogOpen}
        reason={accountDialogReason}
        onClose={() => { setAccountDialogOpen(false); setPendingAction(null); }}
        onAuthenticated={handleAuthenticated}
      />
    </div>
  );
}
