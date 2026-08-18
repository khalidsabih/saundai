"use client";

import { useMemo } from "react";
import type { SetPlan, StudioTransitionPreviewResult } from "@/lib/types";
import {
  allTransitionVariants,
  branchScore,
  humanize,
  routeLabel,
  studioDecisionForTransition,
  studioRecoveryForTransition,
  studioSetIntelligence,
  transitionConfidence,
  transitionVariantLabel,
  type StudioMoveCandidate,
} from "@/lib/studio-intelligence";
import { Icon } from "./Icon";

export type StudioBranch = {
  id: string;
  label: string;
  detail: string;
  plan: SetPlan;
  planPath: string;
  kind: "primary" | "safe" | "creative" | "repair" | "candidate" | "reroute";
};

type Props = {
  plan: SetPlan;
  activeTransition: number;
  branches: StudioBranch[];
  activeBranchId: string | null;
  busy: boolean;
  previewingVariant: string | null;
  alternativePreview: StudioTransitionPreviewResult | null;
  alternativePreviewPlaying: boolean;
  onSelectTransition: (index: number) => void;
  onSelectBranch: (branch: StudioBranch) => void;
  onExplore: (style: "safe" | "creative" | "auto") => void;
  onRepairRoute: () => void;
  onTryCandidate: (candidate: StudioMoveCandidate) => void;
  onPreviewVariant: (variantIndex: number) => void;
  onToggleAlternativePreview: () => void;
  onPreferVariant: (variantIndex: number) => void;
};

function pct(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function signedPct(value: number) {
  if (!Number.isFinite(value) || Math.abs(value) < 0.0005) return null;
  const rounded = Math.round(value * 100);
  return `${rounded > 0 ? "+" : ""}${rounded}%`;
}

function scoreTone(score: number) {
  if (score >= 0.82) return "strong";
  if (score >= 0.68) return "steady";
  return "watch";
}

function uniqueCandidateTracks(candidates: StudioMoveCandidate[]) {
  const byTrack = new Map<number, StudioMoveCandidate>();
  for (const candidate of candidates) {
    const current = byTrack.get(candidate.nextIndex);
    if (!current || candidate.selectionScore > current.selectionScore) byTrack.set(candidate.nextIndex, candidate);
  }
  return [...byTrack.values()].sort((a, b) => b.selectionScore - a.selectionScore);
}

function RouteCanvas({ plan, activeTransition, onSelectTransition }: {
  plan: SetPlan;
  activeTransition: number;
  onSelectTransition: (index: number) => void;
}) {
  return (
    <div className="studio-ai-route" aria-label="Studio performance route">
      {plan.ordered_tracks.map((track, position) => {
        const edge = plan.transitions[position];
        return (
          <div className="studio-ai-route__step" key={`${track.path}-${position}`}>
            <article className="studio-ai-track">
              <span>{String(position + 1).padStart(2, "0")}</span>
              <div>
                <strong>{track.title}</strong>
                <small>{track.bpm.toFixed(1)} BPM · {track.key.camelot} · E{Math.round(track.energy * 100)}</small>
              </div>
            </article>
            {edge && (
              <button
                type="button"
                className={`studio-ai-edge ${activeTransition === position ? "is-active" : ""}`}
                onClick={() => onSelectTransition(position)}
                aria-pressed={activeTransition === position}
              >
                <i />
                <span>
                  <strong>{humanize(edge.plan.technique)}</strong>
                  <small>{edge.plan.transition_bars} bars · {pct(transitionConfidence(edge.plan))}</small>
                </span>
                <Icon name="arrow" />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function StudioIntelligenceWorkbench({
  plan,
  activeTransition,
  branches,
  activeBranchId,
  busy,
  previewingVariant,
  alternativePreview,
  alternativePreviewPlaying,
  onSelectTransition,
  onSelectBranch,
  onExplore,
  onRepairRoute,
  onTryCandidate,
  onPreviewVariant,
  onToggleAlternativePreview,
  onPreferVariant,
}: Props) {
  const setIntel = useMemo(() => studioSetIntelligence(plan), [plan]);
  const decision = useMemo(() => studioDecisionForTransition(plan, activeTransition), [plan, activeTransition]);
  const recovery = useMemo(() => studioRecoveryForTransition(plan, activeTransition), [plan, activeTransition]);
  const edge = plan.transitions[activeTransition];
  const variants = edge ? allTransitionVariants(edge) : [];
  const candidateTracks = uniqueCandidateTracks(decision?.candidates ?? []);
  const selectedFrom = edge ? plan.tracks[edge.from_index] : null;
  const selectedTo = edge ? plan.tracks[edge.to_index] : null;

  return (
    <section className="studio-ai-workbench" id="studio-intelligence">
      <header className="studio-ai-workbench__head">
        <div>
          <span className="eyebrow">Studio intelligence</span>
          <h2>Direct the DJ. Keep the intelligence in the loop.</h2>
          <p>Explore branches, inspect lookahead, rehearse alternatives, and hand every accepted route back through the Executive.</p>
        </div>
        <div className="studio-ai-authority">
          <span><i className={setIntel.executiveAuthoritative ? "is-live" : ""} /> Executive</span>
          <strong>{setIntel.executiveAuthoritative ? "Commit authority" : "Observe mode"}</strong>
          <small>{humanize(setIntel.commitAuthority)}</small>
        </div>
      </header>

      <div className="studio-ai-proof" aria-label="Intelligence used in this set">
        <span><small>Complete moves</small><strong>{setIntel.generatedMoves || "—"}</strong></span>
        <span><small>Lookahead</small><strong>{setIntel.maxLookahead ? `${setIntel.maxLookahead} moves` : "—"}</strong></span>
        <span><small>AI Headphones</small><strong>{setIntel.headphonesRendered ? `${setIntel.headphonesRendered} heard` : "—"}</strong></span>
        <span><small>Crate Memory</small><strong>{setIntel.crateMemoryMoves ? `${setIntel.crateMemoryMoves} moves` : "Neutral"}</strong></span>
        <span><small>Recovery</small><strong>{setIntel.preparedBackups ? `${setIntel.preparedBackups} prepared` : "No backup"}</strong></span>
        <span><small>Route quality</small><strong>{pct(setIntel.routeQuality)}</strong></span>
      </div>

      {branches.length > 0 && (
        <div className="studio-branches">
          <div className="studio-branches__label">
            <span className="eyebrow">Branches</span>
            <small>Nothing is destructive until you choose a branch.</small>
          </div>
          <div className="studio-branches__rail">
            {branches.map((branch) => {
              const active = branch.id === activeBranchId;
              return (
                <button key={branch.id} className={`studio-branch ${active ? "is-active" : ""}`} type="button" onClick={() => onSelectBranch(branch)}>
                  <span>{branch.kind === "primary" ? "MAIN" : branch.kind.toUpperCase()}</span>
                  <strong>{branch.label}</strong>
                  <small>{branch.detail}</small>
                  <em>{pct(branchScore(branch.plan))} · {branch.plan.transitions.filter((item) => item.quality !== "good").length} review</em>
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className="studio-ai-grid">
        <div className="studio-ai-canvas">
          <div className="studio-ai-canvas__toolbar">
            <div>
              <span className="eyebrow">Performance route</span>
              <strong>{routeLabel(plan, 5)}</strong>
            </div>
            <div className="studio-ai-explore-actions">
              <button type="button" disabled={busy} onClick={() => onExplore("safe")}><Icon name="shield" /> Safer branch</button>
              <button type="button" disabled={busy} onClick={() => onExplore("creative")}><Icon name="spark" /> Bolder branch</button>
              <button type="button" disabled={busy} onClick={() => onExplore("auto")}><Icon name="refresh" /> Rethink route</button>
            </div>
          </div>

          <RouteCanvas plan={plan} activeTransition={activeTransition} onSelectTransition={onSelectTransition} />

          {setIntel.setIntent.length > 0 && (
            <div className="studio-ai-journey">
              <span>Set Director</span>
              <div>{setIntel.setIntent.map((intent, index) => <strong key={`${intent}-${index}`}>{intent}</strong>)}</div>
            </div>
          )}
        </div>

        <aside className="studio-ai-inspector" aria-live="polite">
          {edge && selectedFrom && selectedTo ? (
            <>
              <div className="studio-ai-inspector__head">
                <div>
                  <span className="eyebrow">Selected move</span>
                  <h3>{selectedFrom.title} <i>→</i> {selectedTo.title}</h3>
                </div>
                <span className={`studio-ai-score studio-ai-score--${scoreTone(edge.plan.score)}`}>{pct(edge.plan.score)}</span>
              </div>

              <div className="studio-ai-decision">
                <span><small>Executive chose</small><strong>{humanize(edge.plan.technique)}</strong></span>
                <span><small>Intent</small><strong>{humanize(edge.plan.intent ?? edge.plan.style)}</strong></span>
                <span><small>Commit</small><strong>{decision ? pct(decision.commitScore) : pct(edge.objective_score)}</strong></span>
                <span><small>Risk</small><strong>{edge.quality === "good" ? "Low" : humanize(edge.quality)}</strong></span>
              </div>

              {decision?.winnerRationale.length ? (
                <div className="studio-ai-why">
                  <span className="eyebrow">Why this move</span>
                  <div>
                    {decision.winnerRationale.map((reason, index) => (
                      <span key={`${reason}-${index}`}>{humanize(reason)}</span>
                    ))}
                  </div>
                </div>
              ) : null}

              {decision && (
                <div className="studio-ai-reasoning">
                  <div className="studio-ai-reasoning__row">
                    <span><Icon name="route" /> Lookahead</span>
                    <strong>{decision.lookaheadDepth} moves</strong>
                    <small>{decision.projectedPath.length
                      ? decision.projectedPath.map((item) => item.title).join(" → ")
                      : "Receding-horizon route search"}</small>
                  </div>
                  <div className="studio-ai-reasoning__row">
                    <span><Icon name="headphones" /> AI Headphones</span>
                    <strong>{decision.headphonesWinnerScore !== null ? pct(decision.headphonesWinnerScore) : decision.headphonesAuthoritative ? "Authoritative" : "Advisory"}</strong>
                    <small>{decision.headphonesChangedChoice ? "Listening changed the preferred move." : decision.headphonesReason || `${decision.rehearsalBudget} candidate rehearsal budget`}</small>
                  </div>
                  <div className="studio-ai-reasoning__row">
                    <span><Icon name="pulse" /> Thinking budget</span>
                    <strong>{humanize(decision.brittleness)}</strong>
                    <small>{decision.intentObjective ? humanize(decision.intentObjective) : "Adaptive search depth"}</small>
                  </div>
                  <div className="studio-ai-reasoning__row">
                    <span><Icon name="shield" /> Recovery</span>
                    <strong>{recovery.ready ? `${recovery.backups.length} prepared` : "No same-target backup"}</strong>
                    <small>{recovery.backups[0] ? `${humanize(recovery.backups[0].technique)} · ${humanize(recovery.backups[0].readiness)}` : "Primary execution remains protected by hard safety."}</small>
                  </div>
                </div>
              )}

              <div className="studio-ai-command-row">
                <button type="button" disabled={busy} onClick={onRepairRoute}><Icon name="shield" /><span><strong>Repair this route</strong><small>Keep order, ask for safer transitions</small></span></button>
              </div>

              {candidateTracks.length > 0 && (
                <div className="studio-ai-candidates">
                  <div className="studio-ai-section-head">
                    <div><span className="eyebrow">Next-track search</span><strong>What else the Brain considered</strong></div>
                    <small>{decision?.retrievalCount ? `${decision.retrievalCount} retrieved` : `${candidateTracks.length} finalists`}</small>
                  </div>
                  {candidateTracks.slice(0, 5).map((candidate, index) => {
                    const selected = candidate.nextIndex === edge.to_index;
                    return (
                      <article className={`studio-ai-candidate ${selected ? "is-winner" : ""}`} key={`${candidate.nextIndex}-${candidate.executionSignature}`}>
                        <span className="studio-ai-candidate__rank">{String(index + 1).padStart(2, "0")}</span>
                        <div>
                          <strong>{candidate.title}</strong>
                          <small>{humanize(candidate.technique)} · optionality {pct(candidate.futureOptionality)}</small>
                          {(signedPct(candidate.crateAdjustment) || signedPct(candidate.crowdAdjustment) || candidate.tasteAvailable) && (
                            <span className="studio-ai-candidate__signals">
                              {signedPct(candidate.crateAdjustment) && <em>Memory {signedPct(candidate.crateAdjustment)}</em>}
                              {candidate.tasteAvailable && <em>Taste ready</em>}
                              {signedPct(candidate.crowdAdjustment) && <em>Crowd {signedPct(candidate.crowdAdjustment)}</em>}
                            </span>
                          )}
                        </div>
                        <span className="studio-ai-candidate__score">{pct(candidate.selectionScore)}</span>
                        {selected ? <em>Chosen</em> : (
                          <button type="button" disabled={busy} onClick={() => onTryCandidate(candidate)}>Try next</button>
                        )}
                      </article>
                    );
                  })}
                </div>
              )}

              {variants.length > 0 && (
                <div className="studio-ai-alternatives">
                  <div className="studio-ai-section-head">
                    <div><span className="eyebrow">Transition options</span><strong>Rehearse the executable alternatives</strong></div>
                    <small>Current plan is never mutated by audition.</small>
                  </div>
                  <div className="studio-ai-alternative-list">
                    {variants.slice(0, 5).map((variant, variantIndex) => {
                      const previewKey = `${activeTransition}:${variantIndex}`;
                      const isPreparing = previewingVariant === previewKey;
                      const isLoaded = alternativePreview?.transition_index === activeTransition
                        && (alternativePreview.alternative_index ?? -1) === (variantIndex === 0 ? -1 : variantIndex - 1);
                      return (
                        <article className={`studio-ai-alternative ${variantIndex === 0 ? "is-selected" : ""}`} key={variant.execution_signature || `${variant.technique}-${variantIndex}`}>
                          <div>
                            <span>{transitionVariantLabel(variant, variantIndex)}</span>
                            <strong>{humanize(variant.technique)}</strong>
                            <small>{variant.transition_bars} bars · {humanize(variant.intent ?? variant.style)} · {pct(transitionConfidence(variant))}</small>
                          </div>
                          <div className="studio-ai-alternative__actions">
                            <button type="button" disabled={busy || isPreparing} onClick={() => isLoaded ? onToggleAlternativePreview() : onPreviewVariant(variantIndex)}>
                              <Icon name={isLoaded && alternativePreviewPlaying ? "pause" : "headphones"} />
                              {isPreparing ? "Rendering" : isLoaded && alternativePreviewPlaying ? "Pause" : isLoaded ? "Play" : "Audition"}
                            </button>
                            {variantIndex > 0 && <button type="button" className="quiet" onClick={() => onPreferVariant(variantIndex)}>Teach preference</button>}
                          </div>
                        </article>
                      );
                    })}
                  </div>
                </div>
              )}

              {recovery.alternateTargets.length > 0 && (
                <div className="studio-ai-recovery-targets">
                  <span className="eyebrow">Route repair evidence</span>
                  <strong>Different-target fallbacks were kept advisory.</strong>
                  <div>{recovery.alternateTargets.slice(0, 3).map((item) => <span key={`${item.nextIndex}-${item.technique}`}>{item.title} · {humanize(item.technique)} · {pct(item.selectionScore)}</span>)}</div>
                </div>
              )}
            </>
          ) : (
            <div className="studio-ai-empty"><Icon name="route" /><strong>Select a transition</strong><small>Studio will show the move, alternatives, lookahead and recovery evidence.</small></div>
          )}
        </aside>
      </div>
    </section>
  );
}
