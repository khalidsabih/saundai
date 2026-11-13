/**
 * Shared Create/Studio job lifecycle.
 *
 * Deliberately preserves the previously verified SSE, diagnostic polling,
 * disconnect fallback, final server status and cancellation behaviors.
 * Live telemetry is separately owned and is not changed here.
 */
import type { JobRecord } from "./types";
import { API_BASE, request } from "./api-transport";

export class JobCancelledError extends Error {
  constructor(message = "Job cancelled.") {
    super(message);
    this.name = "JobCancelledError";
  }
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

