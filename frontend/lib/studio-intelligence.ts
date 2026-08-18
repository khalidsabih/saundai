import type { SetPlan, SetTransition, TransitionPlan } from "@/lib/types";

export type StudioMoveCandidate = {
  nextIndex: number;
  title: string;
  selectionScore: number;
  moveScore: number;
  technique: string;
  intent: string;
  transitionBars: number | null;
  futureOptionality: number;
  waitBars: number;
  executionSignature: string;
  crateAdjustment: number;
  crowdAdjustment: number;
  tasteAvailable: boolean;
  rationale: string[];
};

export type StudioHeadphonesEvaluation = {
  executionSignature: string;
  headphoneScore: number;
  accepted: boolean;
  feasible: boolean;
  measurementPassed: boolean;
  judgeAccepted: boolean;
  failureDomains: string[];
};

export type StudioDecisionInsight = {
  searchScore: number;
  commitScore: number;
  lookaheadDepth: number;
  projectedPath: Array<{ index: number; title: string }>;
  currentIndex: number;
  winningNextIndex: number;
  candidates: StudioMoveCandidate[];
  headphonesAuthoritative: boolean;
  headphonesReason: string;
  headphonesWinnerScore: number | null;
  headphonesChangedChoice: boolean;
  headphonesEvaluations: StudioHeadphonesEvaluation[];
  brittleness: "easy" | "normal" | "brittle" | string;
  rehearsalBudget: string;
  intentObjective: string;
  allowedTechniques: string[];
  hardConstraints: string[];
  retrievalCount: number;
  winnerRationale: string[];
};

export type StudioRecoveryInsight = {
  ready: boolean;
  backups: Array<{
    technique: string;
    readiness: string;
    selectionScore: number;
    fallbackLevel: number;
    executionSignature: string;
  }>;
  alternateTargets: Array<{
    nextIndex: number;
    title: string;
    technique: string;
    selectionScore: number;
  }>;
};

export type StudioSetIntelligence = {
  executiveAuthoritative: boolean;
  commitAuthority: string;
  decisionCount: number;
  preparedBackups: number;
  hotBackups: number;
  headphonesDecisions: number;
  headphonesRendered: number;
  crateMemoryMoves: number;
  generatedMoves: number;
  maxLookahead: number;
  routeQuality: number;
  riskyEdges: number;
  improvementPercent: number;
  setIntent: string[];
};

function objectValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function objects(value: unknown): Record<string, unknown>[] {
  return arrayValue(value).map(objectValue).filter((item) => Object.keys(item).length > 0);
}

function numberValue(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function stringValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function boolValue(value: unknown): boolean {
  return value === true;
}

function stringList(value: unknown): string[] {
  return arrayValue(value).filter((item): item is string => typeof item === "string");
}

export function humanize(value?: string | null): string {
  if (!value) return "Automatic";
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function planTitle(plan: SetPlan, index: number): string {
  return plan.tracks[index]?.title ?? `Track ${index + 1}`;
}

function latestRoute(plan: SetPlan): Record<string, unknown> {
  const complete = objectValue(objectValue(plan.metadata).complete_move_planner);
  return objectValue(complete.latest_route);
}

function executiveReport(plan: SetPlan): Record<string, unknown> {
  return objectValue(objectValue(plan.metadata).dj_executive);
}

export function studioDecisionForTransition(plan: SetPlan, transitionIndex: number): StudioDecisionInsight | null {
  const edge = plan.transitions[transitionIndex];
  if (!edge) return null;
  const decisions = objects(latestRoute(plan).decisions);
  const decision = decisions.find((item) => {
    const winner = objectValue(item.winner);
    return numberValue(item.current_index, -1) === edge.from_index
      && numberValue(winner.next_index, -1) === edge.to_index;
  }) ?? decisions.find((item) => numberValue(item.current_index, -1) === edge.from_index);
  if (!decision) return null;

  const winner = objectValue(decision.winner);
  const candidates = objects(decision.candidates).map((candidate): StudioMoveCandidate => {
    const move = objectValue(candidate.move);
    const components = objectValue(candidate.components);
    const crate = objectValue(candidate.crate_memory);
    const taste = objectValue(candidate.state_move_taste_shadow);
    const nextIndex = numberValue(candidate.next_index, -1);
    return {
      nextIndex,
      title: planTitle(plan, nextIndex),
      selectionScore: numberValue(candidate.selection_score),
      moveScore: numberValue(move.score),
      technique: stringValue(move.technique, "transition"),
      intent: stringValue(move.intent, "continue"),
      transitionBars: typeof move.transition_bars === "number" ? move.transition_bars : null,
      futureOptionality: numberValue(move.future_optionality),
      waitBars: numberValue(move.wait_bars),
      executionSignature: stringValue(candidate.execution_signature),
      crateAdjustment: numberValue(components.crate_memory_adjustment ?? crate.adjustment),
      crowdAdjustment: numberValue(components.crowd_adjustment),
      tasteAvailable: boolValue(taste.available),
      rationale: stringList(move.rationale).slice(0, 4),
    };
  });

  const headphones = objectValue(decision.ai_headphones);
  const headphonesWinner = objectValue(headphones.winner);
  const evaluations = objects(headphones.evaluations).map((evaluation): StudioHeadphonesEvaluation => ({
    executionSignature: stringValue(evaluation.execution_signature),
    headphoneScore: numberValue(evaluation.headphone_score),
    accepted: boolValue(evaluation.accepted),
    feasible: boolValue(evaluation.feasible),
    measurementPassed: boolValue(evaluation.measurement_passed),
    judgeAccepted: boolValue(evaluation.judge_accepted),
    failureDomains: stringList(evaluation.failure_domains),
  }));
  const budget = objectValue(decision.thinking_budget);
  const brittle = objectValue(budget.brittleness);
  const moveIntent = objectValue(decision.move_intent);
  const retrieval = objectValue(decision.retrieval);
  const projectedPath = arrayValue(decision.projected_path)
    .map((value) => numberValue(value, -1))
    .filter((index) => index >= 0)
    .map((index) => ({ index, title: planTitle(plan, index) }));

  return {
    searchScore: numberValue(decision.search_score, numberValue(winner.selection_score)),
    commitScore: numberValue(decision.commit_score, numberValue(decision.search_score)),
    lookaheadDepth: Math.max(1, Math.round(numberValue(decision.lookahead_depth, 1))),
    projectedPath,
    currentIndex: edge.from_index,
    winningNextIndex: numberValue(winner.next_index, edge.to_index),
    candidates,
    headphonesAuthoritative: boolValue(headphones.authoritative),
    headphonesReason: stringValue(headphones.reason),
    headphonesWinnerScore: typeof headphonesWinner.headphone_score === "number" ? headphonesWinner.headphone_score : null,
    headphonesChangedChoice: boolValue(headphonesWinner.changed_after_listening),
    headphonesEvaluations: evaluations,
    brittleness: stringValue(brittle.level, "normal"),
    rehearsalBudget: `${Math.round(numberValue(budget.headphones_minimum_renders))}–${Math.round(numberValue(budget.headphones_maximum_renders))}`,
    intentObjective: stringValue(moveIntent.musical_objective, stringValue(objectValue(winner.move).intent, "continue the set")),
    allowedTechniques: stringList(moveIntent.allowed_techniques),
    hardConstraints: stringList(moveIntent.hard_constraints),
    retrievalCount: Math.round(numberValue(retrieval.returned_count, numberValue(retrieval.candidate_count))),
    winnerRationale: stringList(objectValue(winner.move).rationale).slice(0, 4),
  };
}

export function studioRecoveryForTransition(plan: SetPlan, transitionIndex: number): StudioRecoveryInsight {
  const edge = plan.transitions[transitionIndex];
  if (!edge) return { ready: false, backups: [], alternateTargets: [] };
  const executive = executiveReport(plan);
  const transactions = objects(executive.transactions);
  const transaction = transactions.find((item) => numberValue(item.current_index, -1) === edge.from_index
    && numberValue(item.next_index, -1) === edge.to_index);
  const recovery = objectValue(transaction?.prepared_recovery);
  const backups = objects(recovery.prepared_backups).map((item) => ({
    technique: stringValue(item.technique, "fallback"),
    readiness: stringValue(item.readiness, "prepared"),
    selectionScore: numberValue(item.selection_score),
    fallbackLevel: Math.round(numberValue(item.fallback_level)),
    executionSignature: stringValue(item.execution_signature),
  }));
  const alternateTargets = objects(recovery.alternate_targets).map((item) => {
    const nextIndex = numberValue(item.next_index, -1);
    return {
      nextIndex,
      title: planTitle(plan, nextIndex),
      technique: stringValue(item.technique, "transition"),
      selectionScore: numberValue(item.selection_score),
    };
  });
  return {
    ready: boolValue(recovery.recovery_ready) || backups.length > 0,
    backups,
    alternateTargets,
  };
}

export function studioSetIntelligence(plan: SetPlan): StudioSetIntelligence {
  const metadata = objectValue(plan.metadata);
  const executive = executiveReport(plan);
  const utilization = objectValue(executive.utilization);
  const complete = objectValue(metadata.complete_move_planner);
  const story = objects(metadata.set_director_story)
    .map((step) => stringValue(objectValue(step.intent).name))
    .filter(Boolean)
    .filter((value, index, values) => index === 0 || value !== values[index - 1])
    .map(humanize);
  const routeQuality = plan.transitions.length
    ? plan.transitions.reduce((sum, edge) => sum + edge.plan.score, 0) / plan.transitions.length
    : 0;
  return {
    executiveAuthoritative: boolValue(executive.authoritative),
    commitAuthority: stringValue(executive.commit_authority, "Executive"),
    decisionCount: Math.round(numberValue(executive.decision_count, plan.transitions.length)),
    preparedBackups: Math.round(numberValue(utilization.prepared_backup_moves)),
    hotBackups: Math.round(numberValue(utilization.hot_backup_moves)),
    headphonesDecisions: Math.round(numberValue(utilization.ai_headphones_decisions, numberValue(complete.ai_headphones_authoritative_decisions))),
    headphonesRendered: Math.round(numberValue(utilization.ai_headphones_rendered_candidates, numberValue(complete.ai_headphones_rendered_candidates))),
    crateMemoryMoves: Math.round(numberValue(utilization.crate_memory_informed_decisions, numberValue(complete.crate_memory_informed_decisions))),
    generatedMoves: Math.round(numberValue(utilization.complete_moves_generated, numberValue(complete.complete_move_generated_candidates))),
    maxLookahead: Math.round(numberValue(utilization.maximum_lookahead_depth, numberValue(complete.lookahead_moves))),
    routeQuality,
    riskyEdges: plan.transitions.filter((edge) => edge.quality !== "good").length,
    improvementPercent: plan.improvement_percent,
    setIntent: story,
  };
}

export function transitionVariantLabel(plan: TransitionPlan, index: number): string {
  if (index === 0) return "SAUND pick";
  if (plan.fallback_level && plan.fallback_level > 0) return `Fallback ${plan.fallback_level}`;
  if (plan.transition_mode === "emergency_cut") return "Emergency";
  return `Alternative ${index}`;
}

export function transitionConfidence(plan: TransitionPlan): number {
  if (typeof plan.measured_score === "number") return plan.measured_score;
  if (typeof plan.analytical_score === "number") return plan.analytical_score;
  return plan.score;
}

export function orderedPathForCandidate(plan: SetPlan, transitionIndex: number, candidateNextIndex: number): string[] | null {
  const position = transitionIndex + 1;
  const order = [...plan.order];
  const currentPosition = order.indexOf(candidateNextIndex);
  if (currentPosition < 0 || currentPosition === position) return null;
  order.splice(currentPosition, 1);
  order.splice(position, 0, candidateNextIndex);
  return order.map((index) => plan.tracks[index]?.path).filter((path): path is string => Boolean(path));
}

export function routeLabel(plan: SetPlan, max = 4): string {
  const names = plan.ordered_tracks.map((track) => track.title);
  if (names.length <= max) return names.join(" → ");
  return `${names.slice(0, max - 1).join(" → ")} → … → ${names[names.length - 1]}`;
}

export function branchScore(plan: SetPlan): number {
  const quality = studioSetIntelligence(plan);
  const riskPenalty = quality.riskyEdges * 0.025;
  return Math.max(0, Math.min(1, 0.62 * quality.routeQuality + 0.38 * Math.max(0, Math.min(1, plan.optimized_score)) - riskPenalty));
}

export function allTransitionVariants(edge: SetTransition): TransitionPlan[] {
  const seen = new Set<string>();
  return [edge.plan, ...edge.alternatives].filter((plan) => {
    const key = plan.execution_signature || `${plan.technique}:${plan.exit_time_a}:${plan.entry_time_b}:${plan.transition_bars}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
