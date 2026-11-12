"use client";

import { useEffect, useMemo, useState } from "react";
import { getBrainStatus, getPhase16ValidationStatus, getReleaseReadiness, listTracks } from "@/lib/api";
import type { BrainStatus, Phase16ValidationStatus, ReleaseReadiness, TrackRecord } from "@/lib/types";
import { Icon } from "./Icon";

function human(value: unknown, fallback = "—") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function pct(value: number | null | undefined) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "—";
}

export function ReadinessDashboard() {
  const [tracks, setTracks] = useState<TrackRecord[]>([]);
  const [brain, setBrain] = useState<BrainStatus | null>(null);
  const [release, setRelease] = useState<ReleaseReadiness | null>(null);
  const [phase16, setPhase16] = useState<Phase16ValidationStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const results = await Promise.allSettled([
        listTracks("audio"),
        getBrainStatus(),
        getReleaseReadiness(),
        getPhase16ValidationStatus(),
      ]);
      if (!active) return;
      if (results[0].status === "fulfilled") setTracks(results[0].value);
      if (results[1].status === "fulfilled") setBrain(results[1].value);
      if (results[2].status === "fulfilled") setRelease(results[2].value);
      if (results[3].status === "fulfilled") setPhase16(results[3].value);
      const failed = results.find((item) => item.status === "rejected");
      setError(failed && failed.status === "rejected" ? String(failed.reason) : null);
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 15_000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  const readyTracks = tracks.filter((track) => track.readiness?.dj_ready || track.readiness?.state === "ready").length;
  const failedTracks = tracks.filter((track) => ["failed", "cancelled"].includes(track.readiness?.state ?? "")).length;
  const preparingTracks = tracks.filter((track) => track.readiness && !["ready", "failed", "cancelled"].includes(track.readiness.state)).length;
  const phasePassed = Boolean(phase16?.summary?.phase16_passed);
  const executive = brain?.dj_executive;
  const researchMode = brain?.bootstrap_runtime?.mode ?? "off";
  const shadowModels = useMemo(
    () => Object.entries(brain?.bootstrap_runtime?.models ?? {}).filter(([, model]) => model.available),
    [brain],
  );

  return (
    <main className="readiness-shell">
      <header className="autonomous-topbar">
        <a className="autonomous-brand" href="/" aria-label="SAUND.AI">
          <span className="saund-wordmark"><strong>SAUND</strong><i>.</i><strong>AI</strong></span>
          <small>Autonomous DJ</small>
        </a>
        <nav className="autonomous-nav" aria-label="Primary">
          <a href="/">Perform</a>
          <a href="/#brain">Brain</a>
          <span className="is-active">Readiness</span>
          <a href="/lab">Lab</a>
        </nav>
        <div className="autonomous-topbar__actions">
          <span className={"autonomous-health " + (release?.operational_ready ? "" : "is-muted")}><i /> {release?.operational_ready ? "Operational" : "Qualification in progress"}</span>
        </div>
      </header>

      <section className="readiness-page">
        <header className="readiness-hero">
          <div><span className="autonomous-kicker">SYSTEM QUALIFICATION</span><h1>Know what SAUND is ready to do.</h1></div>
          <p>Operational health, crate readiness, learned-model authority and real-music validation stay separate so a healthy engine is never confused with a production-qualified DJ.</p>
        </header>

        {error && <div className="readiness-alert"><Icon name="warning" /><span><strong>Some readiness data is unavailable.</strong><small>{error}</small></span></div>}

        <section className="readiness-overview">
          <article className={release?.operational_ready ? "is-good" : "is-warn"}>
            <span>ENGINE</span><strong>{release?.operational_ready ? "Operational" : "Check"}</strong>
            <p>{release ? `${release.blocker_count} blockers · ${release.warning_count} warnings` : "Reading release state"}</p>
          </article>
          <article className={phasePassed ? "is-good" : "is-warn"}>
            <span>REAL-MUSIC VALIDATION</span><strong>{phase16?.available ? (phasePassed ? "Passed" : "Failed") : "Not run"}</strong>
            <p>{phase16?.summary ? `${phase16.summary.rehearsal_ready_runs}/${phase16.summary.run_count} journeys rehearsal-ready` : "Run Phase 16 against a representative crate."}</p>
          </article>
          <article className={failedTracks === 0 && tracks.length > 0 ? "is-good" : "is-warn"}>
            <span>LIBRARY</span><strong>{readyTracks}/{tracks.length || 0} ready</strong>
            <p>{preparingTracks} preparing · {failedTracks} blocked</p>
          </article>
          <article>
            <span>LEARNED AUTHORITY</span><strong>{human(researchMode)}</strong>
            <p>{shadowModels.length} research model{shadowModels.length === 1 ? "" : "s"} available · hard safety authoritative</p>
          </article>
        </section>

        <section className="readiness-grid">
          <article className="readiness-card">
            <header><span>ENGINE + EXECUTIVE</span><b>{release?.operational_ready ? "READY" : "CHECK"}</b></header>
            <dl>
              <div><dt>Single commit authority</dt><dd>{executive?.single_commit_authority ? "Confirmed" : "—"}</dd></div>
              <div><dt>NOW / NEXT protection</dt><dd>{executive?.authoritative ? "Authoritative" : "Prepared"}</dd></div>
              <div><dt>FUTURE replanning</dt><dd>{executive?.continuous_live_replanning ? "Enabled" : "Prepared only"}</dd></div>
              <div><dt>Prepared recovery</dt><dd>{executive?.prepared_recovery_capable ? "Available" : "Unavailable"}</dd></div>
              <div><dt>Hard audio safety</dt><dd>{release?.authority.hard_audio_safety_authoritative ? "Authoritative" : "Unknown"}</dd></div>
            </dl>
          </article>

          <article className="readiness-card">
            <header><span>PHASE 16</span><b>{phase16?.available ? (phasePassed ? "PASS" : "FAIL") : "NO RUN"}</b></header>
            {phase16?.summary ? (
              <dl>
                <div><dt>Journeys</dt><dd>{phase16.summary.run_count}</dd></div>
                <div><dt>Planning ready</dt><dd>{phase16.summary.planning_ready_runs}/{phase16.summary.run_count}</dd></div>
                <div><dt>Rehearsal ready</dt><dd>{phase16.summary.rehearsal_ready_runs}/{phase16.summary.run_count}</dd></div>
                <div><dt>Hard failures</dt><dd>{phase16.summary.hard_failure_count}</dd></div>
                <div><dt>Fallback transitions</dt><dd>{phase16.summary.fallback_transition_count}</dd></div>
              </dl>
            ) : <p>No Phase 16 report is available yet.</p>}
          </article>

          <article className="readiness-card readiness-card--library">
            <header><span>CRATE</span><b>{failedTracks ? "ATTENTION" : tracks.length ? "READY" : "EMPTY"}</b></header>
            <div className="readiness-library-stats">
              <div><strong>{tracks.length}</strong><span>Total</span></div>
              <div><strong>{readyTracks}</strong><span>Ready</span></div>
              <div><strong>{preparingTracks}</strong><span>Preparing</span></div>
              <div><strong>{failedTracks}</strong><span>Blocked</span></div>
            </div>
          </article>

          <article className="readiness-card">
            <header><span>LEARNED MODELS</span><b>{human(researchMode)}</b></header>
            <dl>
              {shadowModels.length ? shadowModels.map(([name, model]) => (
                <div key={name}><dt>{human(name)}</dt><dd>{model.research_shadow ? "Shadow" : human(researchMode)} · {pct(model.heldout_confidence)}</dd></div>
              )) : <div><dt>Research models</dt><dd>None active</dd></div>}
              <div><dt>Production authority</dt><dd>{researchMode === "production" ? "Enabled" : "Not granted"}</dd></div>
              <div><dt>Safety override</dt><dd>Never</dd></div>
            </dl>
          </article>
        </section>

        {phase16?.runs?.length ? (
          <section className="readiness-validation-runs">
            <header><span>REAL-MUSIC RUNS</span><strong>Phase 16</strong></header>
            {phase16.runs.map((run) => (
              <article key={run.journey} className={run.rehearsal_ready ? "is-good" : "is-failed"}>
                <div><span>{human(run.journey)}</span><strong>{run.rehearsal_ready ? "Rehearsal ready" : "Blocked"}</strong></div>
                <small>{run.error ?? `${run.hard_failure_count ?? 0} hard failures · quality ${pct(run.quality_pass_rate)}`}</small>
              </article>
            ))}
          </section>
        ) : null}
      </section>
    </main>
  );
}
