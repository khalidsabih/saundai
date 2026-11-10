"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  buildBootstrapCorpus,
  createHumanReviewSession,
  createListeningSession,
  createTrainingSnapshot,
  getBrainStatus,
  getDatasetBuilderStatus,
  getTrainingLabStatus,
  listTracks,
  saveHumanReviewVote,
  saveListeningPreference,
  trainTrainingLabShadowModel,
  trainBootstrapShadowModels,
  uploadTracks,
} from "@/lib/api";
import type {
  DatasetBuilderStatus,
  ListeningLabCandidate,
  ListeningLabSession,
  HumanReviewSession,
  TrackRecord,
  TrainingLabStatus,
  TrainingLabTrainResult,
  BootstrapTrainingResult,
  BootstrapShadowModelResult,
  BootstrapMetricRow,
  BrainStatus,
} from "@/lib/types";

const REASONS = [
  "Wrong next track",
  "Left too early",
  "Left too late",
  "Transition too long",
  "Tempo felt forced",
  "Vocal clash",
  "Energy wrong",
  "Great payoff",
];

function pct(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${Math.round(value * 100)}%`;
}

function number(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return String(Math.round(value));
}

function metricPct(value: number | undefined) {
  if (value === undefined || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function benchmarkPrimary(model: BootstrapShadowModelResult, row?: BootstrapMetricRow) {
  if (!row) return "—";
  if (model.primary_metric === "top1_accuracy") return metricPct(row.top1_accuracy);
  if (model.primary_metric === "balanced_accuracy") return metricPct(row.balanced_accuracy);
  return metricPct(row.accuracy);
}

function benchmarkBaseline(model: BootstrapShadowModelResult, row?: BootstrapMetricRow) {
  if (!row) return "—";
  if (model.primary_metric === "top1_accuracy") return metricPct(row.top1_accuracy);
  if (model.primary_metric === "balanced_accuracy") return metricPct(row.balanced_accuracy);
  return metricPct(row.accuracy);
}

function compactName(track?: TrackRecord) {
  if (!track) return "Choose a track";
  const withoutExt = track.name.replace(/\.[^.]+$/, "");
  return withoutExt.length > 42 ? `${withoutExt.slice(0, 39)}…` : withoutExt;
}

export function TrainingLab() {
  const [tracks, setTracks] = useState<TrackRecord[]>([]);
  const [trackA, setTrackA] = useState("");
  const [trackB, setTrackB] = useState("");
  const [session, setSession] = useState<ListeningLabSession | null>(null);
  const [pairIndex, setPairIndex] = useState(0);
  const [reasons, setReasons] = useState<string[]>([]);
  const [reviewSession, setReviewSession] = useState<HumanReviewSession | null>(null);
  const [reviewCohort, setReviewCohort] = useState<"general_listener" | "experienced_dj" | "professional_dj">("experienced_dj");
  const [reviewReasons, setReviewReasons] = useState<string[]>([]);
  const [reviewConfidence, setReviewConfidence] = useState(0.75);
  const [reviewerId, setReviewerId] = useState("training-reviewer");
  const [status, setStatus] = useState<TrainingLabStatus | null>(null);
  const [builder, setBuilder] = useState<DatasetBuilderStatus | null>(null);
  const [brain, setBrain] = useState<BrainStatus | null>(null);
  const [trainResult, setTrainResult] = useState<TrainingLabTrainResult | null>(null);
  const [bootstrapTrainResult, setBootstrapTrainResult] = useState<BootstrapTrainingResult | null>(null);
  const [busy, setBusy] = useState<"load" | "generate" | "vote" | "review-load" | "review-vote" | "snapshot" | "train" | "upload" | "corpus" | "bootstrap-train" | null>("load");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [generateElapsed, setGenerateElapsed] = useState(0);
  const audioA = useRef<HTMLAudioElement | null>(null);
  const audioB = useRef<HTMLAudioElement | null>(null);
  const reviewAudioA = useRef<HTMLAudioElement | null>(null);
  const reviewAudioB = useRef<HTMLAudioElement | null>(null);

  const refresh = useCallback(async () => {
    const [trackRows, labStatus, builderStatus, brainStatus] = await Promise.all([listTracks("audio"), getTrainingLabStatus(), getDatasetBuilderStatus(), getBrainStatus()]);
    setTracks(trackRows);
    setStatus(labStatus);
    setBuilder(builderStatus);
    setBrain(brainStatus);
    setTrackA((current) => current || trackRows[0]?.path || "");
    setTrackB((current) => current || trackRows.find((row) => row.path !== trackRows[0]?.path)?.path || "");
  }, []);

  useEffect(() => {
    void refresh()
      .catch((caught) => setError(caught instanceof Error ? caught.message : String(caught)))
      .finally(() => setBusy(null));
  }, [refresh]);

  useEffect(() => {
    if (busy !== "generate") {
      setGenerateElapsed(0);
      return;
    }
    const started = Date.now();
    setGenerateElapsed(0);
    const timer = window.setInterval(() => {
      setGenerateElapsed(Math.floor((Date.now() - started) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [busy]);

  const pairCount = session ? Math.floor(session.candidates.length / 2) : 0;
  const pair = useMemo(() => {
    if (!session) return [];
    return session.candidates.slice(pairIndex * 2, pairIndex * 2 + 2);
  }, [session, pairIndex]);
  const selectedA = tracks.find((track) => track.path === trackA);
  const selectedB = tracks.find((track) => track.path === trackB);
  const trainProgress = status
    ? Math.min(1, status.preferences.decisive / Math.max(1, status.targets.train_minimum))
    : 0;
  const benchmarkProgress = status
    ? Math.min(1, status.preferences.decisive / Math.max(1, status.targets.benchmark_minimum))
    : 0;
  const bootstrapBenchmark = bootstrapTrainResult ?? builder?.latest_benchmark ?? null;
  const globalTasteShadow = status?.shadow_models.find((model) => model.kind === "global_taste_ranker");
  const stateMoveTasteShadow = status?.shadow_models.find((model) => model.kind === "state_move_taste");
  const renderedJudgeShadow = status?.shadow_models.find((model) => model.kind === "rendered_dj_judge");
  const trainedTasteHeads = Number(Boolean(globalTasteShadow)) + Number(Boolean(stateMoveTasteShadow)) + Number(Boolean(renderedJudgeShadow));

  function randomPair() {
    if (tracks.length < 2) return;
    const first = Math.floor(Math.random() * tracks.length);
    let second = Math.floor(Math.random() * (tracks.length - 1));
    if (second >= first) second += 1;
    setTrackA(tracks[first].path);
    setTrackB(tracks[second].path);
    setSession(null);
    setPairIndex(0);
    setNotice(null);
  }

  async function generate() {
    if (!trackA || !trackB || trackA === trackB) return;
    setBusy("generate");
    setError(null);
    setNotice("Preparing blind candidates: reusing cached track analysis when available, then rehearsing and rendering real transition audio.");
    setTrainResult(null);
    try {
      const created = await createListeningSession({ trackA, trackB, candidates: 4 });
      setSession(created);
      setPairIndex(0);
      setReasons([]);
      const total = created.generation?.total_seconds;
      const hits = created.generation?.analysis_cache_hits;
      const cacheTotal = created.generation?.analysis_cache_total;
      const timing = typeof total === "number" ? ` in ${total.toFixed(1)}s` : "";
      const cache = typeof hits === "number" && typeof cacheTotal === "number" ? ` · analysis cache ${hits}/${cacheTotal}` : "";
      setNotice(`Blind candidates are ready${timing}${cache}. Listen without trying to guess how they were made.`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function vote(choice: "a" | "b" | "tie" | "neither") {
    if (!session || pair.length !== 2) return;
    setBusy("vote");
    setError(null);
    setNotice(null);
    try {
      await saveListeningPreference({
        sessionId: session.session_id,
        candidateA: pair[0].label,
        candidateB: pair[1].label,
        choice,
        listener: "training-expert",
        reasons,
      });
      setReasons([]);
      if (pairIndex + 1 < pairCount) {
        setPairIndex((value) => value + 1);
        setNotice("Preference recorded. Next blind pair.");
      } else {
        setSession(null);
        setPairIndex(0);
        setNotice("Session complete. Generate another track pair when you are ready.");
      }
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function loadHumanReview() {
    setBusy("review-load");
    setError(null);
    setNotice(null);
    try {
      const created = await createHumanReviewSession({
        evaluator: reviewerId.trim() || "training-reviewer",
        cohort: reviewCohort,
      });
      setReviewSession(created);
      setReviewReasons([]);
      setReviewConfidence(0.75);
      setNotice(`Active-learning review ready · priority ${Math.round(created.queue_priority * 100)}%. Teacher metadata stays hidden while you listen.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function voteHumanReview(choice: "a" | "b" | "tie" | "neither") {
    if (!reviewSession || reviewSession.candidates.length !== 2) return;
    setBusy("review-vote");
    setError(null);
    try {
      const result = await saveHumanReviewVote({
        sessionId: reviewSession.session_id,
        choice,
        reasons: reviewReasons,
        confidence: reviewConfidence,
      });
      setReviewSession(null);
      setReviewReasons([]);
      setNotice(`Human preference recorded. ${result.status.queue_outstanding} reviewable active-learning cases remain.`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function snapshot() {
    setBusy("snapshot");
    setError(null);
    setNotice(null);
    try {
      const created = await createTrainingSnapshot("first-train");
      setNotice(`Immutable dataset created: ${created.dataset_id}`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function train() {
    setBusy("train");
    setError(null);
    setNotice(null);
    try {
      const trained = await trainTrainingLabShadowModel();
      setTrainResult(trained);
      const trainedHeads = [trained.global_taste_ranker, trained.state_move_taste, trained.rendered_dj_judge].filter((head) => head.status === "trained").length;
      const blockedHeads = 3 - trainedHeads;
      setNotice(`${trainedHeads} taste head${trainedHeads === 1 ? "" : "s"} trained in shadow mode${blockedHeads ? `; ${blockedHeads} still needs more evidence` : ""}. Hard audio safety is unchanged.`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function freezeBootstrapCorpus() {
    setBusy("corpus");
    setError(null);
    setNotice(null);
    try {
      const created = await buildBootstrapCorpus("bootstrap-v1", false);
      setNotice(`Bootstrap corpus frozen: ${created.corpus_id}`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function trainBootstrap() {
    setBusy("bootstrap-train");
    setError(null);
    setNotice(null);
    setBootstrapTrainResult(null);
    try {
      const trained = await trainBootstrapShadowModels(builder?.corpora[0]?.corpus_id);
      setBootstrapTrainResult(trained);
      const trainedCount = trained.models.filter((item) => item.status === "trained").length;
      setNotice(`${trainedCount} bootstrap model${trainedCount === 1 ? "" : "s"} trained. Runtime priors refreshed in ${trained.runtime?.mode ?? "governed"} mode; hard safety remains authoritative.`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function addTracks(files: File[]) {
    if (!files.length) return;
    setBusy("upload");
    setError(null);
    try {
      const result = await uploadTracks(files);
      await refresh();
      if (result.files[0]) setTrackA(result.files[0].path);
      if (result.files[1]) setTrackB(result.files[1].path);
      setNotice(`${result.files.length} track${result.files.length === 1 ? "" : "s"} added to the local training library.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  if (busy === "load" && !status) {
    return (
      <main className="lab-shell lab-shell--loading">
        <span className="lab-kicker">SAUND.AI / INTERNAL</span>
        <h1>Opening Training Lab.</h1>
      </main>
    );
  }

  return (
    <div className="lab-shell">
      <header className="lab-header">
        <div className="lab-brandline">
          <a href="/" className="lab-wordmark" aria-label="Return to SAUND.AI">
            <strong>SAUND</strong><i>.</i><strong>AI</strong>
          </a>
          <span>Training Lab</span>
        </div>
        <div className="lab-header__meta">
          <span>INTERNAL / LOCAL</span>
          <span className="lab-live-dot"><i /> Data stays here</span>
          <a href="/">Open product ↗</a>
        </div>
      </header>

      <main className="lab-main">
        <section className="lab-hero">
          <div>
            <span className="lab-kicker">SAUND LEARNS / PAIRWISE TASTE</span>
            <h1>Teach the ear.<br /><em>Keep safety deterministic.</em></h1>
          </div>
          <p>
            Listen blind, choose what feels better, then freeze the evidence into an immutable dataset. The learned model can rerank safe transitions; it cannot override rendered-audio failures.
          </p>
        </section>

        {error && <div className="lab-alert lab-alert--error" role="alert"><strong>Training Lab stopped.</strong><span>{error}</span><button type="button" onClick={() => setError(null)}>Dismiss</button></div>}
        {notice && <div className="lab-alert" role="status"><span>{notice}</span><button type="button" onClick={() => setNotice(null)}>Dismiss</button></div>}

        <section className="lab-metrics" aria-label="Training data status">
          <article><span>Tracks</span><strong>{status?.track_count ?? 0}</strong><small>local library</small></article>
          <article><span>Blind choices</span><strong>{status?.preferences.decisive ?? 0}</strong><small>{status?.preferences.unique_track_pairs ?? 0} track pairs</small></article>
          <article><span>Rendered</span><strong>{status?.rendered_candidates ?? 0}</strong><small>candidate transitions</small></article>
          <article><span>Taste heads</span><strong>{trainedTasteHeads} / 3</strong><small>{status?.runtime_evidence.candidates ?? 0} runtime candidates captured</small></article>
        </section>

        <section className="lab-panel lab-listen-panel" aria-label="Active-learning human review">
          <div className="lab-panel__number">H</div>
          <div className="lab-panel__title lab-panel__title--row">
            <div>
              <span>HUMAN GROUND TRUTH / ACTIVE LEARNING</span>
              <h2>Review what the teachers disagree on.</h2>
              <p>Exact queued transitions are replayed blind. Your cohort, reasons and confidence are stored as human preference evidence; teacher scores remain hidden.</p>
            </div>
            <small>{status?.human_preference_program?.queue_outstanding ?? 0} outstanding / {status?.human_preference_program?.active_learning_votes ?? 0} reviewed</small>
          </div>

          <div className="lab-track-pair">
            <label>
              <span>REVIEWER</span>
              <input value={reviewerId} onChange={(event) => setReviewerId(event.target.value)} disabled={busy !== null || Boolean(reviewSession)} />
              <strong>Pseudonymized before storage</strong>
            </label>
            <label>
              <span>COHORT</span>
              <select value={reviewCohort} onChange={(event) => setReviewCohort(event.target.value as typeof reviewCohort)} disabled={busy !== null || Boolean(reviewSession)}>
                <option value="general_listener">General listener</option>
                <option value="experienced_dj">Experienced DJ</option>
                <option value="professional_dj">Professional DJ</option>
              </select>
              <strong>{reviewCohort.replaceAll("_", " ")}</strong>
            </label>
          </div>

          {reviewSession ? (
            <>
              <div className="lab-listen-grid">
                <article className="lab-listen-card">
                  <div className="lab-listen-card__head"><span>Candidate</span><strong>A</strong></div>
                  <div className="lab-listen-card__center" aria-hidden="true"><span>A</span></div>
                  <audio ref={reviewAudioA} className="lab-audio" controls preload="metadata" src={reviewSession.candidates[0]?.audio_url} onPlay={() => { if (reviewAudioB.current && !reviewAudioB.current.paused) reviewAudioB.current.pause(); }} />
                </article>
                <article className="lab-listen-card">
                  <div className="lab-listen-card__head"><span>Candidate</span><strong>B</strong></div>
                  <div className="lab-listen-card__center" aria-hidden="true"><span>B</span></div>
                  <audio ref={reviewAudioB} className="lab-audio" controls preload="metadata" src={reviewSession.candidates[1]?.audio_url} onPlay={() => { if (reviewAudioA.current && !reviewAudioA.current.paused) reviewAudioA.current.pause(); }} />
                </article>
              </div>
              <div className="lab-reasons">
                <span>Why? <em>optional</em></span>
                <div>
                  {REASONS.map((reason) => {
                    const tag = reason.toLowerCase().replace(/\s+/g, "_");
                    const active = reviewReasons.includes(tag);
                    return <button type="button" key={`review-${reason}`} className={active ? "is-active" : ""} aria-pressed={active} onClick={() => setReviewReasons((current) => active ? current.filter((item) => item !== tag) : [...current, tag])}>{reason}</button>;
                  })}
                </div>
              </div>
              <label>
                <span>Confidence {Math.round(reviewConfidence * 100)}%</span>
                <input type="range" min="0" max="1" step="0.05" value={reviewConfidence} onChange={(event) => setReviewConfidence(Number(event.target.value))} />
              </label>
              <div className="lab-vote-row">
                <button type="button" onClick={() => void voteHumanReview("a")} disabled={busy !== null}><span>A</span><strong>Feels better</strong></button>
                <button type="button" onClick={() => void voteHumanReview("b")} disabled={busy !== null}><span>B</span><strong>Feels better</strong></button>
                <button type="button" className="lab-vote-row__quiet" onClick={() => void voteHumanReview("tie")} disabled={busy !== null}>Tie</button>
                <button type="button" className="lab-vote-row__quiet" onClick={() => void voteHumanReview("neither")} disabled={busy !== null}>Neither</button>
              </div>
            </>
          ) : (
            <div className="lab-actions">
              <button type="button" className="lab-primary" onClick={() => void loadHumanReview()} disabled={busy !== null || !(status?.human_preference_program?.queue_outstanding ?? 0)}>
                {busy === "review-load" ? "Rendering queued review…" : "Open next active review"}<span>→</span>
              </button>
            </div>
          )}
        </section>

        <section className="lab-learning-map" aria-label="SAUND learning lifecycle">
          <article>
            <span>WHAT SAUND KNOWS</span>
            <strong>{builder?.latest_benchmark?.models.filter((model) => model.status === "trained").length ?? 0} bootstrap heads</strong>
            <small>cue · duration · adjacency · sequence</small>
          </article>
          <article>
            <span>WHAT SAUND IS LEARNING</span>
            <strong>{status?.runtime_evidence.candidates ?? 0} runtime candidates</strong>
            <small>{status?.preferences.decisive ?? 0} decisive human preferences</small>
          </article>
          <article>
            <span>WHAT CAN AFFECT LIVE</span>
            <div className="lab-authority-list">
              <b><i className="is-advisory" />Cue + duration priors <em>{brain?.bootstrap_runtime?.mode ?? "—"}</em></b>
              <b><i className="is-advisory" />DJ Move Brain <em>{brain?.dj_move_brain?.enabled ? "phrase-state" : "fallback"}</em></b>
              <b><i />Global Taste <em>{globalTasteShadow ? "shadow" : "collecting"}</em></b>
              <b><i />State × Move Taste <em>{stateMoveTasteShadow ? "shadow" : "collecting"}</em></b>
              <b><i className="is-advisory" />Crate Memory <em>{status?.crate_memory.pair_records ?? 0} pairs</em></b>
              <b><i />Rendered Judge <em>{renderedJudgeShadow ? "shadow" : "collecting"}</em></b>
              <b><i className="is-hard" />Hard safety <em>authoritative</em></b>
            </div>
          </article>
        </section>

        <section className="lab-bootstrap" aria-label="External bootstrap data">
          <div className="lab-bootstrap__head">
            <div>
              <span className="lab-kicker">BOOTSTRAP DATA / EXTERNAL SOURCES</span>
              <h2>Start informed. <em>Then learn taste.</em></h2>
            </div>
            <p>Public and expert datasets are normalized into one provenance-aware schema. External sources remain research-only until a source-specific production rights review.</p>
          </div>

          <div className="lab-bootstrap__metrics">
            <div><span>Normalized records</span><strong>{builder?.counts.normalized_records ?? 0}</strong></div>
            <div><span>Weak examples</span><strong>{builder?.counts.weak_examples ?? 0}</strong></div>
            <div><span>Imported sources</span><strong>{builder?.imports.length ?? 0} / {builder?.catalog.sources.length ?? 5}</strong></div>
            <div><span>Frozen corpora</span><strong>{builder?.corpora.length ?? 0}</strong><small>{builder?.bootstrap_models.length ?? 0} shadow runs</small></div>
          </div>

          <div className="lab-bootstrap__sources">
            {(builder?.catalog.sources ?? []).map((source) => {
              const imported = builder?.imports.find((item) => item.source_id === source.id);
              return (
                <article key={source.id}>
                  <div><span>{source.id.toUpperCase()}</span><i className={imported ? "is-ready" : ""} /></div>
                  <strong>{source.name}</strong>
                  <p>{source.purpose.slice(0, 2).join(" · ")}</p>
                  <small>{imported ? `${imported.records.toLocaleString()} normalized` : "Ready to acquire / import"}</small>
                </article>
              );
            })}
          </div>

          <div className="lab-bootstrap__readiness">
            {Object.entries(builder?.model_readiness ?? {}).map(([model, item]) => (
              <div key={model}><span>{model.replaceAll("_", " ")}</span><strong>{item.examples.toLocaleString()}</strong><small>{item.source}</small></div>
            ))}
          </div>

          <div className="lab-bootstrap__actions">
            <code>./scripts/saund.sh dataset-sources</code>
            <code>./scripts/saund.sh dataset-import &lt;source&gt; &lt;path&gt;</code>
            <button type="button" className="lab-primary" disabled={!builder?.counts.normalized_records || busy !== null} onClick={() => void freezeBootstrapCorpus()}>
              {busy === "corpus" ? "Freezing corpus…" : "Freeze bootstrap corpus"}<span>→</span>
            </button>
            <button type="button" className="lab-secondary" disabled={!builder?.corpora.length || busy !== null} onClick={() => void trainBootstrap()}>
              {busy === "bootstrap-train" ? "Training shadows…" : "Train bootstrap shadows"}<span>→</span>
            </button>
          </div>

          {bootstrapBenchmark && (
            <section className="lab-benchmark" aria-label="Persistent bootstrap benchmark">
              <div className="lab-benchmark__head">
                <div>
                  <span>BENCHMARK / {bootstrapBenchmark.corpus_id}</span>
                  <strong>Research bootstrap priors</strong>
                </div>
                <small>{bootstrapTrainResult?.runtime?.mode ? `Runtime: ${bootstrapTrainResult.runtime.mode}` : "Runtime use is bounded by governance mode"}</small>
              </div>

              <div className="lab-benchmark__models">
                {bootstrapBenchmark.models.map((model) => {
                  const retrieval = model.evaluation;
                  if (model.status === "trained" && retrieval?.evaluation_kind === "held_out_next_track_retrieval") {
                    const validation = retrieval.metrics.validation;
                    const test = retrieval.metrics.test;
                    const delta = retrieval.improvement_vs_global_popularity.test.hit_at_10;
                    const testSubgroups = retrieval.subgroups?.test;
                    const generalizationRows = testSubgroups ? ([
                      ["seen edge", testSubgroups.seen_edge],
                      ["novel pair", testSubgroups.novel_pair_candidate],
                      ["cold source", testSubgroups.cold_source],
                      ["cold target", testSubgroups.cold_target],
                    ] as const) : [];
                    return (
                      <article className="lab-benchmark__model" key={model.model_id}>
                        <header>
                          <div><span>{model.model_id.replaceAll("_", " ")}</span><small>held-out next-track retrieval</small></div>
                          <strong>{`${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pts @10`}</strong>
                        </header>
                        <div className="lab-benchmark__splits">
                          {([
                            ["validation", validation],
                            ["test", test],
                          ] as const).map(([split, metric]) => (
                            <div key={`${model.model_id}-${split}`}>
                              <span>{split}</span>
                              <strong>Hit@10 {metricPct(metric.model.hit_at_10)}</strong>
                              <small>global {metricPct(metric.global_popularity_baseline.hit_at_10)} · random {metricPct(metric.random_candidate_baseline.hit_at_10)}</small>
                              <small>Hit@1 {metricPct(metric.model.hit_at_1)} · Hit@5 {metricPct(metric.model.hit_at_5)} · MRR {metricPct(metric.model.mrr)}</small>
                              <small>{metric.examples.toLocaleString()} edges · source coverage {metricPct(metric.source_coverage)} · seen-edge coverage {metricPct(metric.edge_coverage)}</small>
                            </div>
                          ))}
                          {generalizationRows.map(([label, metric]) => (
                            <div key={`${model.model_id}-generalization-${label}`}>
                              <span>test / {label}</span>
                              <strong>Hit@10 {metricPct(metric.model.hit_at_10)}</strong>
                              <small>global {metricPct(metric.global_popularity_baseline.hit_at_10)} · MRR {metricPct(metric.model.mrr)}</small>
                              <small>{metric.examples.toLocaleString()} edges · target coverage {metricPct(metric.target_coverage)}</small>
                            </div>
                          ))}
                          <div>
                            <span>candidate universe</span>
                            <strong>{retrieval.candidate_count.toLocaleString()}</strong>
                            <small>unique train destination tracks</small>
                            <small>behavioral adjacency only · not technical mixability</small>
                          </div>
                        </div>
                      </article>
                    );
                  }
                  if (model.status !== "trained" || !model.metrics || !model.baseline) {
                    return (
                      <article className="lab-benchmark__model" key={model.model_id}>
                        <header><span>{model.model_id.replaceAll("_", " ")}</span><strong>{model.status}</strong></header>
                        <p>{model.reason ?? `${model.examples ?? 0} examples`}</p>
                      </article>
                    );
                  }
                  const delta = model.primary_improvement_vs_baseline?.test;
                  return (
                    <article className="lab-benchmark__model" key={model.model_id}>
                      <header>
                        <div><span>{model.model_id.replaceAll("_", " ")}</span><small>{model.evaluation_kind?.replaceAll("_", " ")}</small></div>
                        <strong>{delta === undefined ? "—" : `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pts`}</strong>
                      </header>
                      <div className="lab-benchmark__splits">
                        {(["train", "validation", "test"] as const).map((split) => {
                          const metric = model.metrics?.[split];
                          const baseline = model.baseline?.metrics[split];
                          return (
                            <div key={`${model.model_id}-${split}`}>
                              <span>{split}</span>
                              <strong>{benchmarkPrimary(model, metric)}</strong>
                              <small>baseline {benchmarkBaseline(model, baseline)}</small>
                              <small>{metric?.examples.toLocaleString() ?? 0} candidates{metric?.groups ? ` · ${metric.groups.toLocaleString()} groups` : ""}</small>
                              {metric?.top3_recall !== undefined && <small>top-3 {metricPct(metric.top3_recall)} · MRR {metricPct(metric.mrr)}</small>}
                              {metric?.balanced_accuracy !== undefined && <small>F1 {metricPct(metric.f1)} · ROC {metricPct(metric.roc_auc)} · PR {metricPct(metric.pr_auc)}</small>}
                            </div>
                          );
                        })}
                      </div>
                    </article>
                  );
                })}
              </div>
            </section>
          )}
        </section>

        <section className="lab-grid">
          <article className="lab-panel lab-source-panel">
            <div className="lab-panel__number">01</div>
            <div className="lab-panel__title">
              <span>SOURCE PAIR</span>
              <h2>Choose two tracks.</h2>
              <p>Use varied genres, tempos, structures and energy relationships. Diversity matters more than sheer volume.</p>
            </div>

            <div className="lab-track-pair">
              <label>
                <span>FROM</span>
                <select value={trackA} onChange={(event) => { setTrackA(event.target.value); setSession(null); }} disabled={busy !== null}>
                  <option value="">Choose a track</option>
                  {tracks.map((track) => <option value={track.path} key={`a-${track.path}`}>{track.name}</option>)}
                </select>
                <strong>{compactName(selectedA)}</strong>
              </label>
              <span className="lab-pair-arrow" aria-hidden="true">→</span>
              <label>
                <span>INTO</span>
                <select value={trackB} onChange={(event) => { setTrackB(event.target.value); setSession(null); }} disabled={busy !== null}>
                  <option value="">Choose a track</option>
                  {tracks.map((track) => <option value={track.path} key={`b-${track.path}`}>{track.name}</option>)}
                </select>
                <strong>{compactName(selectedB)}</strong>
              </label>
            </div>

            <div className="lab-actions lab-actions--source">
              <button type="button" className="lab-secondary" onClick={randomPair} disabled={tracks.length < 2 || busy !== null}>Random pair</button>
              <label className="lab-secondary lab-file-button">
                {busy === "upload" ? "Adding…" : "Add tracks"}
                <input type="file" accept="audio/*" multiple onChange={(event) => { const files = Array.from(event.target.files ?? []); event.target.value = ""; void addTracks(files); }} disabled={busy !== null} />
              </label>
              <button type="button" className="lab-primary" onClick={() => void generate()} disabled={!trackA || !trackB || trackA === trackB || busy !== null}>
                {busy === "generate" ? `Rendering candidates… ${generateElapsed}s` : "Generate blind pair"}<span>↗</span>
              </button>
            </div>
            {busy === "generate" && (
              <div className="lab-generation-progress" role="status" aria-live="polite">
                <strong>Blind render in progress · {generateElapsed}s</strong>
                <span>Cached analysis → candidate rehearsal → rendered A/B previews. Keep this page open; the pair appears here as soon as the backend finishes.</span>
              </div>
            )}
          </article>

          <article className="lab-panel lab-listen-panel">
            <div className="lab-panel__number">02</div>
            <div className="lab-panel__title lab-panel__title--row">
              <div>
                <span>BLIND LISTEN</span>
                <h2>{session ? `Pair ${pairIndex + 1} / ${pairCount}` : "Listen without metadata."}</h2>
              </div>
              {session && <small>{session.session_id}</small>}
            </div>

            {session && pair.length === 2 ? (
              <>
                <div className="lab-listen-grid">
                  <article className="lab-listen-card">
                    <div className="lab-listen-card__head"><span>Candidate</span><strong>A</strong></div>
                    <div className="lab-listen-card__center" aria-hidden="true"><span>A</span></div>
                    <audio ref={audioA} className="lab-audio" controls preload="metadata" src={pair[0].audio_url} aria-label="Blind transition candidate A" onPlay={() => { if (audioB.current && !audioB.current.paused) audioB.current.pause(); }} />
                  </article>
                  <article className="lab-listen-card">
                    <div className="lab-listen-card__head"><span>Candidate</span><strong>B</strong></div>
                    <div className="lab-listen-card__center" aria-hidden="true"><span>B</span></div>
                    <audio ref={audioB} className="lab-audio" controls preload="metadata" src={pair[1].audio_url} aria-label="Blind transition candidate B" onPlay={() => { if (audioA.current && !audioA.current.paused) audioA.current.pause(); }} />
                  </article>
                </div>

                <div className="lab-reasons">
                  <span>What drove the decision? <em>optional</em></span>
                  <div>
                    {REASONS.map((reason) => {
                      const tag = reason.toLowerCase().replace(/\s+/g, "_");
                      const active = reasons.includes(tag);
                      return <button type="button" key={reason} className={active ? "is-active" : ""} aria-pressed={active} onClick={() => setReasons((current) => active ? current.filter((item) => item !== tag) : [...current, tag])}>{reason}</button>;
                    })}
                  </div>
                </div>

                <div className="lab-vote-row">
                  <button type="button" onClick={() => void vote("a")} disabled={busy !== null}><span>A</span><strong>Feels better</strong></button>
                  <button type="button" onClick={() => void vote("b")} disabled={busy !== null}><span>B</span><strong>Feels better</strong></button>
                  <button type="button" className="lab-vote-row__quiet" onClick={() => void vote("tie")} disabled={busy !== null}>No preference</button>
                  <button type="button" className="lab-vote-row__quiet" onClick={() => void vote("neither")} disabled={busy !== null}>Neither works</button>
                </div>
              </>
            ) : (
              <div className="lab-empty-listen">
                <span>A / B</span>
                <h3>No scores. No technique names. No hints.</h3>
                <p>Generate a source pair. SAUND.AI will rehearse safe finalists and hide everything except the sound.</p>
              </div>
            )}
          </article>

          <article className="lab-panel lab-data-panel">
            <div className="lab-panel__number">03</div>
            <div className="lab-panel__title">
              <span>DATA + MODEL</span>
              <h2>Freeze. Train. Measure.</h2>
              <p>Dataset snapshots combine blind human preference with measured synthetic failures. All taste heads train in shadow mode first; hard audio safety stays authoritative.</p>
            </div>

            <div className="lab-target">
              <div><span>First train</span><strong>{status?.preferences.decisive ?? 0} / {status?.targets.train_minimum ?? 24}</strong></div>
              <div className="lab-target__bar"><i style={{ transform: `scaleX(${trainProgress})` }} /></div>
              <small>{status?.train_ready ? "Enough labels to train a first model." : "Keep collecting decisive blind preferences."}</small>
            </div>

            <div className="lab-target lab-target--benchmark">
              <div><span>Meaningful benchmark</span><strong>{status?.preferences.decisive ?? 0} / {status?.targets.benchmark_minimum ?? 100}</strong></div>
              <div className="lab-target__bar"><i style={{ transform: `scaleX(${benchmarkProgress})` }} /></div>
              <small>100 is the first deployment-quality checkpoint, not a guarantee.</small>
            </div>

            <dl className="lab-model-stats">
              <div><dt>Runtime evidence</dt><dd>{status?.runtime_evidence.candidates ?? 0}</dd></div>
              <div><dt>Crate Memory</dt><dd>{status?.crate_memory.pair_records ?? 0} pairs</dd></div>
              <div><dt>Global Taste shadow</dt><dd>{globalTasteShadow ? pct(globalTasteShadow.cv_accuracy) : "Not trained"}</dd></div>
              <div><dt>Rendered Judge shadow</dt><dd>{renderedJudgeShadow ? pct(renderedJudgeShadow.cv_accuracy) : "Not trained"}</dd></div>
              <div><dt>Hard safety</dt><dd>Authoritative</dd></div>
            </dl>

            {trainResult && (
              <div className="lab-train-result">
                <span>TASTE LEARNING / SHADOW</span>
                <strong>{trainResult.dataset_id}</strong>
                <div>
                  <em>Global Taste</em><b>{trainResult.global_taste_ranker.status === "trained" ? pct(trainResult.global_taste_ranker.cv_accuracy) : "Blocked"}</b>
                  <em>Rendered Judge</em><b>{trainResult.rendered_dj_judge.status === "trained" ? pct(trainResult.rendered_dj_judge.validation_accuracy) : "Blocked"}</b>
                </div>
              </div>
            )}

            <div className="lab-actions lab-actions--data">
              <button type="button" className="lab-secondary" onClick={() => void snapshot()} disabled={!status?.preferences.total || busy !== null}>{busy === "snapshot" ? "Freezing…" : "Freeze dataset"}</button>
              <button type="button" className="lab-primary" onClick={() => void train()} disabled={!status?.train_ready || busy !== null}>{busy === "train" ? "Training taste heads…" : "Train taste heads"}<span>→</span></button>
            </div>

            <div className="lab-policy">
              <span><i className={status?.policy.raw_audio_bundled_in_snapshots ? "is-warn" : ""} /> Raw audio is not copied into datasets</span>
              <span><i className={status?.policy.shadow_training_changes_live_model ? "is-warn" : ""} /> New taste heads start at zero authority</span>
              <span><i className={status?.policy.hard_audio_safety_can_be_overridden_by_ranker ? "is-warn" : ""} /> Hard audio failures remain non-overridable</span>
            </div>
          </article>
        </section>

        <section className="lab-ledger">
          <div className="lab-ledger__head">
            <div><span className="lab-kicker">DATA LEDGER</span><h2>Every train has a source.</h2></div>
            <p>{status?.rights.production_training_ready ? "Source rights and the production registry are marked ready." : "Current snapshots are safe for local research, but production training stays flagged until source rights are explicitly confirmed."}</p>
          </div>
          <div className="lab-ledger__grid">
            <div><span>Latest dataset</span><strong>{status?.snapshots[0]?.dataset_id ?? "Not frozen yet"}</strong><small>{status?.snapshots[0] ? `${status.snapshots[0].comparison_count} decisive labels / ${status.snapshots[0].track_count} tracks` : "Create labels, then freeze a snapshot."}</small></div>
            <div><span>Preference balance</span><strong>{number(status?.preferences.choice_a)} A / {number(status?.preferences.choice_b)} B</strong><small>Blind position is randomized inside each pair.</small></div>
            <div><span>Rights state</span><strong>{status?.rights.production_training_ready ? "Production-cleared" : "Research only"}</strong><small>Set SAUND_TRAINING_SOURCE_RIGHTS_CONFIRMED only for cleared audio.</small></div>
          </div>
        </section>
      </main>

      <footer className="lab-footer">
        <span>SAUND.AI / TRAINING LAB</span>
        <span>Listen → choose → snapshot → train → benchmark</span>
        <span>Hear ahead.</span>
      </footer>
    </div>
  );
}
