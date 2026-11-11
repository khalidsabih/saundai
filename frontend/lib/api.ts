import type { AccountPreferences, AccountProfile, AccountProject, AccountRegistrationInput, AuthSession, AutomaticSetResult, BrainStatus, HostingStatus, JobRecord, JourneyMode, LiveManifest, LiveTelemetrySnapshot, Phase16ValidationStatus, ReleaseReadiness, SetPlan, StudioTransitionPreviewResult, TrackRecord } from "./types";

export const API_BASE = (process.env.NEXT_PUBLIC_SAUND_API_URL ?? "").replace(/\/$/, "");

export class JobCancelledError extends Error {
  constructor(message = "Job cancelled.") {
    super(message);
    this.name = "JobCancelledError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store", credentials: "include", ...init });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail ?? payload ?? `Request failed (${response.status})`;
    const error = new Error(typeof detail === "string" ? detail : detail.message ?? JSON.stringify(detail));
    Object.assign(error, { status: response.status, detail });
    throw error;
  }
  return payload as T;
}


export async function getAuthSession() {
  return request<AuthSession>("/api/auth/me");
}

export async function registerAccount(input: AccountRegistrationInput) {
  return request<AuthSession>("/api/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

export async function loginAccount(input: { email: string; password: string }) {
  return request<AuthSession>("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

export async function logoutAccount() {
  return request<AuthSession>("/api/auth/logout", { method: "POST" });
}

export async function updateAccountPreferences(input: Partial<AccountPreferences>) {
  return request<AuthSession>("/api/account/preferences", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

export async function updateAccountProfile(input: Partial<AccountProfile>) {
  return request<AuthSession>("/api/account/profile", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

export async function getAccountProjects(limit = 30) {
  return request<{ projects: AccountProject[]; count: number }>(`/api/account/projects?limit=${limit}`);
}

export async function health() {
  return request<Record<string, unknown>>("/api/health");
}

export async function getHostingStatus() {
  return request<HostingStatus>("/api/hosting/status");
}

export async function getBrainStatus() {
  return request<BrainStatus>("/api/brain/status");
}

export async function getReleaseReadiness() {
  return request<ReleaseReadiness>("/api/release/status");
}

export async function getPhase16ValidationStatus() {
  return request<Phase16ValidationStatus>("/api/validation/phase16");
}

export async function listTracks(folder = "audio") {
  const payload = await request<{ tracks: TrackRecord[] }>(`/api/tracks?folder=${encodeURIComponent(folder)}&recursive=true`);
  return payload.tracks;
}

export async function uploadTracks(files: File[]) {
  const hosting = await getHostingStatus().catch(() => null);
  if (hosting?.direct_uploads) {
    const intents = await request<{ uploads: Array<{ filename: string; object_key: string; upload_url: string; method: string; headers: Record<string, string>; expected_size: number }> }>("/api/tracks/upload-intents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        files: files.map((file) => ({ filename: file.name, size: file.size, content_type: file.type || null })),
      }),
    });
    if (intents.uploads.length !== files.length) throw new Error("SAUND.AI did not return an upload target for every track.");
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      const intent = intents.uploads[index];
      const response = await fetch(intent.upload_url, { method: intent.method || "PUT", headers: intent.headers, body: file });
      if (!response.ok) throw new Error(`Direct upload failed for ${file.name} (${response.status}).`);
    }
    return request<{ files: TrackRecord[] }>("/api/tracks/upload-complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        files: intents.uploads.map((intent, index) => ({
          filename: files[index].name,
          object_key: intent.object_key,
          expected_size: files[index].size,
        })),
      }),
    });
  }
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  return request<{ files: TrackRecord[] }>("/api/tracks/upload", { method: "POST", body });
}

export async function createPlan(input: {
  track_paths: string[];
  manual: boolean;
  tempo_policy: "safe" | "stretch" | "cut";
  preferred_style: "auto" | "safe" | "creative" | "emergency";
  journey: "balanced" | "smooth" | "build";
  experience?: "simple" | "pro";
  automatic_delivery?: boolean;
  listener?: string;
  minimum_body_seconds: number;
  minimum_body_fraction: number;
}) {
  const payload = await request<{ job: JobRecord }>("/api/set/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...input, folder: null, use_cache: true }),
  });
  return payload.job;
}

export async function createPreviews(planPath: string) {
  const payload = await request<{ job: JobRecord }>("/api/set/previews", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan_path: planPath, pre_roll_seconds: 8, post_roll_seconds: 8 }),
  });
  return payload.job;
}

export async function createStudioTransitionPreview(input: {
  planPath: string;
  transitionIndex: number;
  alternativeIndex?: number | null;
}) {
  const payload = await request<{ job: JobRecord }>("/api/studio/transition-preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      plan_path: input.planPath,
      transition_index: input.transitionIndex,
      alternative_index: input.alternativeIndex ?? null,
      pre_roll_seconds: 8,
      post_roll_seconds: 8,
    }),
  });
  return payload.job;
}

export function readStudioTransitionPreview(job: JobRecord): StudioTransitionPreviewResult {
  if (job.status !== "completed" || !job.result) {
    throw new Error(job.error || job.message || "Studio transition rehearsal did not complete.");
  }
  const result = job.result as unknown as Partial<StudioTransitionPreviewResult>;
  if (!result.preview_url || typeof result.transition_index !== "number") {
    throw new Error("Studio transition rehearsal completed without a playable preview.");
  }
  return result as StudioTransitionPreviewResult;
}

export async function renderSet(input: {
  planPath: string;
  outputName: string;
  confirm: boolean;
  confirmationToken?: string | null;
}) {
  const payload = await request<{ job: JobRecord }>("/api/set/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      plan_path: input.planPath,
      output_name: input.outputName,
      confirm: input.confirm,
      confirmation_token: input.confirmationToken,
    }),
  });
  return payload.job;
}

export async function getJob(jobId: string) {
  return request<JobRecord>(`/api/jobs/${jobId}`);
}

export async function cancelJob(jobId: string) {
  return request<JobRecord>(`/api/jobs/${jobId}/cancel`, { method: "POST" });
}

async function pollForJob(
  initial: JobRecord,
  onUpdate?: (job: JobRecord) => void,
  intervalMs = 900,
): Promise<JobRecord> {
  let job = initial;
  onUpdate?.(job);
  while (job.status === "queued" || job.status === "running") {
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
    job = await getJob(job.job_id);
    onUpdate?.(job);
  }
  if (job.status === "failed") {
    throw new Error(job.message || job.error || "SAUND.AI job failed");
  }
  if (job.status === "cancelled") {
    throw new JobCancelledError(job.message || "Job cancelled.");
  }
  return job;
}

export async function waitForJob(
  initial: JobRecord,
  onUpdate?: (job: JobRecord) => void,
  intervalMs = 900,
): Promise<JobRecord> {
  if (initial.status === "completed") {
    onUpdate?.(initial);
    return initial;
  }
  if (initial.status === "failed") {
    onUpdate?.(initial);
    throw new Error(initial.message || initial.error || "SAUND.AI job failed");
  }
  if (initial.status === "cancelled") {
    onUpdate?.(initial);
    throw new JobCancelledError(initial.message || "Job cancelled.");
  }
  if (typeof EventSource === "undefined") {
    return pollForJob(initial, onUpdate, intervalMs);
  }

  onUpdate?.(initial);
  return new Promise<JobRecord>((resolve, reject) => {
    let current = initial;
    let settled = false;
    let fallbackStarted = false;
    const source = new EventSource(`${API_BASE}/api/jobs/${initial.job_id}/events`, { withCredentials: true });
    let diagnosticTimer: number | null = null;

    const clearDiagnosticTimer = () => {
      if (diagnosticTimer !== null) {
        window.clearInterval(diagnosticTimer);
        diagnosticTimer = null;
      }
    };

    const finishFromServer = async () => {
      if (settled) return;
      try {
        source.close();
        clearDiagnosticTimer();
        const final = await pollForJob(await getJob(initial.job_id), onUpdate, intervalMs);
        current = final;
        settled = true;
        if (final.status === "failed") {
          reject(new Error(final.message || final.error || "SAUND.AI job failed"));
        } else {
          resolve(final);
        }
      } catch (caught) {
        settled = true;
        source.close();
        clearDiagnosticTimer();
        reject(caught);
      }
    };

    diagnosticTimer = window.setInterval(async () => {
      if (settled || fallbackStarted) return;
      try {
        current = await getJob(initial.job_id);
        onUpdate?.(current);
        if (current.status === "completed" || current.status === "failed" || current.status === "cancelled") {
          void finishFromServer();
        }
      } catch {
        // SSE remains the primary path; an occasional diagnostic poll may fail transiently.
      }
    }, 10_000);

    source.onmessage = (event) => {
      try {
        const update = JSON.parse(event.data) as Partial<JobRecord>;
        current = {
          ...current,
          status: update.status ?? current.status,
          progress: update.progress ?? current.progress,
          stage: update.stage ?? current.stage,
          message: update.message ?? current.message,
          stage_current: update.stage_current ?? current.stage_current,
          stage_total: update.stage_total ?? current.stage_total,
          cancel_requested: update.cancel_requested ?? current.cancel_requested,
          cancellable: update.cancellable ?? current.cancellable,
          stalled: update.stalled ?? current.stalled,
          elapsed_seconds: update.elapsed_seconds ?? current.elapsed_seconds,
          stage_elapsed_seconds: update.stage_elapsed_seconds ?? current.stage_elapsed_seconds,
          progress_age_seconds: update.progress_age_seconds ?? current.progress_age_seconds,
        };
        onUpdate?.(current);
        if (current.status === "completed" || current.status === "failed" || current.status === "cancelled") {
          void finishFromServer();
        }
      } catch {
        // Ignore malformed progress records; the final job fetch remains authoritative.
      }
    };

    source.onerror = () => {
      if (settled || fallbackStarted) return;
      fallbackStarted = true;
      source.close();
      clearDiagnosticTimer();
      void pollForJob(current, onUpdate, intervalMs).then(
        (final) => { settled = true; resolve(final); },
        (error) => { settled = true; reject(error); },
      );
    };
  });
}

export async function readPlanFromJob(job: JobRecord): Promise<{ plan: SetPlan; planPath: string }> {
  if (job.status !== "completed") {
    throw new Error(`Planning job ended in unexpected state: ${job.status}. ${job.message || ""}`.trim());
  }
  const result = job.result as { plan?: SetPlan; plan_path?: string } | null;
  if (!result?.plan || !result.plan_path) {
    const detail = job.error || job.message || "The completed job did not contain a plan payload.";
    throw new Error(`Planning job completed without a usable plan. ${detail}`);
  }
  return { plan: result.plan, planPath: result.plan_path };
}

export function mediaUrl(path: string) {
  return path.startsWith("http") ? path : `${API_BASE}${path.startsWith("/") ? "" : "/api/media/"}${path}`;
}

export async function saveRating(input: {
  reportPath: string;
  alternative?: "safe" | "creative" | "emergency";
  transitionIndex?: number;
  beat: number;
  phrase: number;
  bass: number;
  energy: number;
  cue: number;
  overall: number;
  notes: string;
}) {
  return request<{ ratings_path: string }>("/api/ratings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      report_path: input.reportPath,
      alternative: input.alternative ?? null,
      transition_index: input.transitionIndex ?? null,
      beat: input.beat,
      phrase: input.phrase,
      bass: input.bass,
      energy: input.energy,
      cue: input.cue,
      overall: input.overall,
      notes: input.notes,
      listener: "web",
    }),
  });
}


export async function createAutomaticSet(input: {
  trackPaths: string[];
  journey: JourneyMode;
  listener?: string;
  outputName?: string;
  targetDurationSeconds?: number | null;
  peakFraction?: number;
  closingWindowSeconds?: number;
  endingEnergy?: number;
  measurementTopK?: number;
}) {
  const payload = await request<{ job: JobRecord }>("/api/sets/automatic", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      track_paths: input.trackPaths,
      journey: input.journey,
      listener: input.listener ?? "web",
      output_name: input.outputName ?? "SAUND_set.wav",
      target_duration_seconds: input.targetDurationSeconds ?? null,
      peak_fraction: input.peakFraction ?? 0.65,
      closing_window_seconds: input.closingWindowSeconds ?? 600,
      ending_energy: input.endingEnergy ?? 0.45,
      measurement_top_k: input.measurementTopK ?? 4,
      use_cache: true,
    }),
  });
  return payload.job;
}

export function readAutomaticSetFromJob(job: JobRecord): AutomaticSetResult {
  if (job.status !== "completed" || !job.result) {
    throw new Error(job.error || job.message || "Automatic set job did not complete.");
  }
  const result = job.result as unknown as Partial<AutomaticSetResult>;
  if (!result.plan || !result.plan_path || !result.audio_url) {
    throw new Error("Automatic set job completed without a playable result.");
  }
  return result as AutomaticSetResult;
}

export async function sendFeedback(input: {
  event: "set_like" | "set_dislike" | "transition_like" | "transition_dislike" | "alternative_selected" | "replanned_after_preview" | "render_completed" | "downloaded" | "skip" | "replay" | "playback_completed";
  listener?: string;
  planId?: string;
  planPath?: string;
  transitionIndex?: number;
  value?: number;
  journey?: JourneyMode;
  metadata?: Record<string, unknown>;
}) {
  return request<{
    feedback_path: string;
    listener_profile: Record<string, unknown> | null;
    personalization_applied: boolean;
    training_eligible: boolean;
    privacy_version: string;
  }>("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      event: input.event,
      listener: input.listener ?? "web",
      plan_id: input.planId ?? null,
      plan_path: input.planPath ?? null,
      transition_index: input.transitionIndex ?? null,
      value: input.value ?? null,
      journey: input.journey ?? null,
      metadata: input.metadata ?? {},
    }),
  });
}

export async function getLiveManifest(planPath: string) {
  return request<LiveManifest>(`/api/live/manifest?plan_path=${encodeURIComponent(planPath)}`);
}

export function liveStreamUrl(planPath: string, bitrate = "192k", sessionId?: string) {
  const params = new URLSearchParams({ plan_path: planPath, bitrate });
  if (sessionId) params.set("session_id", sessionId);
  return `${API_BASE}/api/live/stream?${params.toString()}`;
}

export function liveEventsUrl(planPath: string, sessionId: string) {
  const params = new URLSearchParams({ plan_path: planPath, session_id: sessionId });
  return `${API_BASE}/api/live/events?${params.toString()}`;
}

export function subscribeLiveTelemetry(
  planPath: string,
  sessionId: string,
  onUpdate: (snapshot: LiveTelemetrySnapshot) => void,
  onConnectionChange?: (connected: boolean) => void,
) {
  if (typeof EventSource === "undefined") return () => undefined;
  const source = new EventSource(liveEventsUrl(planPath, sessionId), { withCredentials: true });
  source.onopen = () => onConnectionChange?.(true);
  source.onmessage = (event) => {
    try {
      const snapshot = JSON.parse(event.data) as LiveTelemetrySnapshot;
      onUpdate(snapshot);
      onConnectionChange?.(true);
      if (["complete", "failed", "closed"].includes(snapshot.status)) source.close();
    } catch {
      // A malformed telemetry event must never affect the audio stream.
    }
  };
  source.onerror = () => onConnectionChange?.(false);
  return () => source.close();
}

export async function createListeningSession(input: {
  trackA: string;
  trackB: string;
  candidates?: number;
}) {
  return request<import("./types").ListeningLabSession>("/api/intelligence/listening-session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      track_a: input.trackA,
      track_b: input.trackB,
      candidates: input.candidates ?? 4,
    }),
  });
}

export async function saveListeningPreference(input: {
  sessionId: string;
  candidateA: string;
  candidateB: string;
  choice: "a" | "b" | "tie" | "neither";
  listener?: string;
  reasons?: string[];
}) {
  return request<{ preferences_path: string }>("/api/intelligence/preference", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: input.sessionId,
      candidate_a: input.candidateA,
      candidate_b: input.candidateB,
      choice: input.choice,
      listener: input.listener ?? "web-pro",
      reasons: input.reasons ?? [],
    }),
  });
}

export async function getRankerStatus() {
  return request<import("./types").RankerStatus>("/api/intelligence/ranker/status");
}

export async function trainTransitionRanker() {
  return request<Record<string, unknown>>("/api/intelligence/ranker/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
}

export async function getHumanReviewStatus() {
  return request<import("./types").HumanReviewStatus>("/api/training-lab/human-review/status");
}

export async function createHumanReviewSession(input: {
  evaluator: string;
  cohort: "general_listener" | "experienced_dj" | "professional_dj";
  scenarioId?: string;
}) {
  return request<import("./types").HumanReviewSession>("/api/training-lab/human-review/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      evaluator: input.evaluator,
      cohort: input.cohort,
      scenario_id: input.scenarioId ?? null,
    }),
  });
}

export async function saveHumanReviewVote(input: {
  sessionId: string;
  choice: "a" | "b" | "tie" | "neither";
  reasons?: string[];
  confidence?: number;
  experienceYears?: number;
}) {
  return request<{ preferences_path: string; status: import("./types").HumanReviewStatus }>("/api/training-lab/human-review/vote", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: input.sessionId,
      choice: input.choice,
      reasons: input.reasons ?? [],
      confidence: input.confidence ?? 0.75,
      experience_years: input.experienceYears ?? null,
    }),
  });
}

export async function getTrainingLabStatus() {
  return request<import("./types").TrainingLabStatus>("/api/training-lab/status");
}

export async function getTrainingLabSessions(limit = 30) {
  return request<{ sessions: Array<Record<string, unknown>> }>(`/api/training-lab/sessions?limit=${encodeURIComponent(String(limit))}`);
}

export async function createTrainingSnapshot(name = "first-train") {
  return request<{ dataset_id: string; dataset_path: string; manifest: Record<string, unknown> }>("/api/training-lab/snapshot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function trainTrainingLabShadowModel(snapshotId?: string) {
  return request<import("./types").TrainingLabTrainResult>("/api/training-lab/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ snapshot_id: snapshotId ?? null, ridge: 0.18 }),
  });
}

export async function getDatasetBuilderStatus() {
  return request<import("./types").DatasetBuilderStatus>("/api/dataset-builder/status");
}

export async function buildBootstrapCorpus(name = "bootstrap-v1", production = false) {
  const params = new URLSearchParams({ name, production: production ? "true" : "false" });
  return request<{ corpus_id: string; corpus_path: string; manifest: Record<string, unknown> }>(`/api/dataset-builder/build?${params.toString()}`, {
    method: "POST",
  });
}


export async function trainBootstrapShadowModels(corpusId?: string) {
  const params = new URLSearchParams();
  if (corpusId) params.set("corpus_id", corpusId);
  const query = params.toString();
  return request<import("./types").BootstrapTrainingResult>(`/api/dataset-builder/train${query ? `?${query}` : ""}`, {
    method: "POST",
  });
}
