export type AccountUser = {
  id: string;
  email: string;
  created_at: string;
};

export type AccountPrimaryUse = "personal" | "dj" | "producer" | "creator" | "venue_event" | "other";
export type AccountExperienceLevel = "starting" | "intermediate" | "experienced" | "professional";
export type AccountGenre = "House" | "Techno" | "Hip-Hop" | "Drum & Bass" | "Pop" | "Afro" | "Trance" | "R&B" | "Other";
export type AccountMusicalPriority =
  | "smooth_transitions"
  | "energy_progression"
  | "creative_transitions"
  | "harmonic_mixing"
  | "club_ready_sound"
  | "discovery";

export type AccountProfile = {
  first_name: string;
  last_name: string;
  artist_name: string | null;
  country: string;
  primary_use: AccountPrimaryUse | null;
  experience_level: AccountExperienceLevel | null;
  preferred_genres: AccountGenre[];
  musical_priorities: AccountMusicalPriority[];
};

export type AccountPreferences = {
  personalization_enabled: boolean;
  improve_saund_ai: boolean;
  marketing_consent: boolean;
};

export type AccountRegistrationInput = AccountProfile & {
  email: string;
  password: string;
  personalization_enabled: boolean;
  improve_saund_ai: boolean;
  marketing_consent: boolean;
  invite_code?: string | null;
};

export type HostingStatus = {
  schema_version: string;
  hosted_mode: boolean;
  environment: string;
  deployment_profile: string;
  database_backend: string;
  object_store: string;
  direct_uploads: boolean;
  upload_auth_required: boolean;
  signup_mode: "open" | "invite" | "closed";
  persistent_workspace_required: boolean;
  job_execution: string;
  horizontal_worker_scaling: boolean;
  public_origin?: string | null;
  api_origin?: string | null;
};

export type AuthSession = {
  authenticated: boolean;
  user: AccountUser | null;
  profile: AccountProfile | null;
  preferences: AccountPreferences | null;
  session_expires_at?: number;
};

export type AccountProject = {
  project_id: string;
  job_id: string;
  kind: "automatic-set" | "set-plan" | string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  title: string;
  created_at_utc: string;
  updated_at_utc: string;
  journey?: string | null;
  track_count: number;
  track_paths: string[];
  plan_path?: string | null;
  audio_path?: string | null;
  audio_url?: string | null;
};

export type TrackReadinessFeature = {
  status: "pending" | "queued" | "running" | "ready" | "stale" | "skipped" | string;
  version: string;
  available: boolean;
  message?: string;
};

export type TrackReadiness = {
  schema_version: string;
  fingerprint: string;
  state: "pending" | "queued" | "analyzing" | "partial" | "ready" | "failed" | "cancelled" | string;
  dj_ready: boolean;
  progress: number;
  message: string;
  job_id?: string | null;
  updated_at_utc?: string | null;
  features: Record<string, TrackReadinessFeature>;
};

export type TrackRecord = {
  name: string;
  path: string;
  size: number;
  media_url: string;
  performance_layers?: {
    bundle_id: string;
    complete: boolean;
    verified: boolean;
    instrumental: boolean;
    vocals: boolean;
    source_quality: string;
    quality_confidence?: number;
    warnings: string[];
  } | null;
  stem_status?: string;
  stem_job_id?: string | null;
  readiness_job_id?: string | null;
  readiness?: TrackReadiness | null;
};

export type JobRecord = {
  job_id: string;
  kind: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  stage: string;
  message: string;
  stage_current?: number;
  stage_total?: number;
  started_at_utc?: string | null;
  finished_at_utc?: string | null;
  heartbeat_at_utc?: string | null;
  last_progress_at_utc?: string | null;
  elapsed_seconds?: number;
  stage_elapsed_seconds?: number;
  heartbeat_age_seconds?: number;
  progress_age_seconds?: number;
  stalled?: boolean;
  cancel_requested?: boolean;
  cancellable?: boolean;
  result: Record<string, unknown> | null;
  error: string | null;
};

export type TransitionPlan = {
  style: string;
  intent?: string;
  intent_score?: number;
  technique: string;
  technique_score?: number;
  score: number;
  quality?: string;
  exit_time_a: number;
  entry_time_b: number;
  exit_section?: string;
  entry_section?: string;
  cue_confidence?: number;
  transition_bars: number;
  transition_seconds: number;
  target_bpm: number;
  transition_mode: string;
  beatmatched: boolean;
  tempo_strategy: string;
  effective_bpm_a?: number;
  effective_bpm_b?: number;
  render_rate_a?: number;
  render_rate_b?: number;
  phase_nudge_seconds_b?: number;
  tempo_release_seconds_b?: number;
  tempo_release_source_duration_b?: number;
  tempo_release_start_rate_b?: number;
  tempo_release_end_rate_b?: number;
  post_transition_rate_b?: number;
  post_transition_bpm_b?: number;
  execution_signature?: string;
  analytical_score?: number;
  measured_score?: number;
  quality_checked?: boolean;
  measurement_passed?: boolean;
  measurement_rank?: number;
  fallback_level?: number;
  fallback_reason?: string | null;
  auto_approved?: boolean;
  listener_adjustment?: number;
  measurement?: {
    version: string;
    sample_rate: number;
    duration_seconds: number;
    loudness_continuity: number;
    spectral_continuity: number;
    bass_cleanliness: number;
    vocal_cleanliness?: number;
    stem_bass_cleanliness?: number;
    stem_evidence?: boolean;
    phase_safety: number;
    artifact_safety: number;
    silence_safety: number;
    rhythm_coherence?: number;
    transient_clarity?: number;
    groove_continuity?: number;
    harmonic_continuity?: number;
    energy_intent_fit?: number;
    drop_payoff?: number;
    musical_overall?: number;
    overall: number;
    peak: number;
    clipping_fraction: number;
    phase_correlation: number;
    loudness_jump_db: number;
    low_frequency_buildup_db: number;
    silence_fraction: number;
    boundary_jump: number;
    hard_failures: string[];
    warnings: string[];
  } | null;
  warning?: string | null;
  rationale?: string[];
  metadata?: Record<string, unknown>;
  compatibility: {
    overall: number;
    tempo: number;
    harmonic: number;
    local_harmonic?: number;
    local_key_a?: string | null;
    local_key_b?: string | null;
    local_key_confidence_a?: number;
    local_key_confidence_b?: number;
    energy: number;
    texture: number;
    beat_grid: number;
    phrase: number;
    sections: number;
    genre?: number;
    vocal_overlap?: number;
    bass_conflict?: number;
    boundary?: number;
    section_fit?: number;
    tempo_difference_percent: number;
  };
};

export type SetTransition = {
  from_index: number;
  to_index: number;
  selected_style: string;
  objective_score: number;
  quality: "good" | "risky" | "poor";
  plan: TransitionPlan;
  alternatives: TransitionPlan[];
  warnings: string[];
  reasons: string[];
};

export type MusicalTimelinePhrase = {
  phrase_index: number;
  phrase_id: string;
  start_bar: number;
  end_bar: number;
  role: string;
  section: string;
  song_role?: string;
  song_role_index?: number;
  song_form_confidence?: number;
  importance: number;
  transition_readiness: number;
  vocal_probability?: number;
  energy?: number;
};

export type MusicalTimelineSummary = {
  version: string;
  track: string;
  fingerprint: string;
  phrase_count: number;
  section_count: number;
  phrases: MusicalTimelinePhrase[];
  sections: Array<{ section_id: string; label: string; start_bar: number; end_bar: number; phrase_ids: string[] }>;
  commitments: Array<{ kind: string; created_at_bar: number; payoff_bar: number; protect_until_bar: number; importance: number }>;
};

export type PlanTrack = {
  index: number;
  path: string;
  title: string;
  duration_seconds: number;
  bpm: number;
  key: { camelot: string; tonic: string; mode: string; confidence: number };
  genre: {
    label: string;
    confidence: number;
    probabilities: Record<string, number>;
  };
  energy: number;
  variable_tempo: boolean;
  time_signature: number;
  waveform_peaks: number[];
  musical_timeline?: MusicalTimelineSummary;
};

export type SetPlan = {
  plan_id: string;
  source: string;
  manual: boolean;
  automatic_delivery?: boolean;
  listener?: string;
  requires_confirmation: boolean;
  confirmation_token: string | null;
  baseline_order: number[];
  order: number[];
  optimized_score: number;
  baseline_score: number;
  improvement_percent: number;
  estimated_duration_seconds: number;
  tracks: PlanTrack[];
  ordered_tracks: PlanTrack[];
  transitions: SetTransition[];
  warnings: string[];
  metadata?: Record<string, unknown>;
};

export type PreviewResult = {
  preview_paths: string[];
  preview_urls: string[];
  report_path: string;
};

export type StudioTransitionPreviewResult = {
  preview_path: string;
  preview_url: string;
  transition_index: number;
  alternative_index: number | null;
  variant: string;
  execution_signature: string;
  technique: string;
  intent?: string | null;
  score: number;
};

export type RenderResult = {
  audio_path: string;
  report_path: string;
  audio_url: string;
};


export type JourneyMode = "balanced" | "smooth" | "build" | "warmup" | "peak" | "sunset";

export type AutomaticSetResult = {
  plan_path: string;
  plan: SetPlan;
  audio_path: string;
  report_path: string;
  audio_url: string;
  live_stream_url: string;
  live_manifest_url: string;
};

export type LiveManifestItem = {
  type: "track" | "transition" | "tempo_release";
  set_start: number;
  set_end: number;
  track_index?: number;
  title?: string;
  from_index?: number;
  to_index?: number;
  from_title?: string;
  to_title?: string;
  technique?: string;
  intent?: string;
  fallback_level?: number;
  target_bpm?: number;
  matched_rate_b?: number;
  post_transition_rate_b?: number;
  phase_nudge_seconds_b?: number;
  stem_profile?: string | null;
  stem_confidence?: number | null;
  start_rate?: number;
  end_rate?: number;
  start_bpm?: number;
  end_bpm?: number;
  pitch_preserving?: boolean;
  tempo_owner?: string;
  primary?: { technique: string; intent?: string; execution_signature: string };
  backup?: { technique: string; intent?: string; execution_signature: string } | null;
  emergency?: { technique: string; intent?: string; execution_signature: string } | null;
  backup_techniques?: string[];
};

export type LiveRuntimeControl = {
  continuous_replanning: boolean;
  now_mutable: boolean;
  next_mutable: boolean;
  future_replannable: boolean;
  manifest_is_initial_schedule: boolean;
  telemetry_supported?: boolean;
  schedule_generation?: number;
};

export type LiveManifest = {
  plan_id: string;
  duration_seconds: number;
  timeline: LiveManifestItem[];
  ordered_tracks: Array<{ index: number; title: string; bpm: number; key: string }>;
  runtime_control?: LiveRuntimeControl;
};

export type LivePlaybackState = {
  state_version: number;
  schedule_generation: number;
  executor_phase: string;
  current_track: number;
  source_cursor: number;
  set_time: number;
  edge_position: number;
  expected_command: string | null;
  commit_digest: string;
  safety_action: string;
};

export type LiveHorizonSlot = {
  horizon: "now" | "next" | "future" | string;
  status: string;
  track_index: number | null;
  edge_position: number | null;
  from_index: number | null;
  to_index: number | null;
  execution_signature: string | null;
  replannable: boolean;
};

export type LiveTelemetrySnapshot = {
  version: string;
  sequence: number;
  session_id: string;
  plan_id: string;
  status: "streaming" | "complete" | "failed" | "closed" | string;
  updated_at_utc: string;
  detail?: string;
  schedule_generation: number;
  active_order: number[];
  duration_seconds: number;
  timeline: LiveManifestItem[];
  ordered_tracks: Array<{ index: number; title: string; bpm: number; key: string }>;
  runtime_control: LiveRuntimeControl;
  playback_state: LivePlaybackState;
  horizons: {
    now: LiveHorizonSlot;
    next: LiveHorizonSlot;
    future: LiveHorizonSlot[];
  };
  safety: {
    action: string;
    reason?: string;
    decision?: Record<string, unknown> | null;
  };
  executive: {
    enabled: boolean;
    cycle: number;
    replanning: boolean;
    last_committed_generation: number;
    crowd_state_authority: string;
    transition_policy_authority: string;
    last_event?: {
      action?: string;
      detail?: string;
      schedule_generation?: number;
      payload?: Record<string, unknown>;
    } | null;
  };
  crowd: {
    state: Record<string, unknown>;
    directive: Record<string, unknown>;
  };
};

export type ListeningLabCandidate = {
  label: string;
  audio_url: string;
};

export type HumanReviewStatus = {
  schema_version: string;
  queue_total: number;
  queue_reviewable: number;
  queue_outstanding: number;
  reviewed_scenarios: number;
  active_learning_votes: number;
  cohorts: Record<string, number>;
  top_reasons: { reason: string; count: number }[];
  mean_human_confidence: number | null;
  policy: {
    blind_ab_required: boolean;
    exact_queued_transition_replay: boolean;
    teacher_scores_hidden_from_reviewers: boolean;
    local_review_cache_exported_to_training_data: boolean;
    human_labels_are_ground_truth_anchor: boolean;
    human_labels_bypass_hard_safety: boolean;
  };
};

export type HumanReviewSession = {
  session_id: string;
  teacher_scenario_id: string;
  queue_priority: number;
  queue_reasons: string[];
  evaluator_cohort: "general_listener" | "experienced_dj" | "professional_dj";
  candidates: ListeningLabCandidate[];
};

export type ListeningLabSession = {
  session_id: string;
  candidates: ListeningLabCandidate[];
  generation?: {
    analysis_seconds?: number;
    candidate_generation_seconds?: number;
    rehearsal_seconds?: number;
    preview_render_seconds?: number;
    total_seconds?: number;
    analysis_cache_hits?: number;
    analysis_cache_total?: number;
  };
};

export type RankerStatus = {
  decisive_preferences: number;
  unique_track_pairs: number;
  ranker_loaded: boolean;
  cv_accuracy: number | null;
  baseline_accuracy: number | null;
  win_margin: number | null;
  authority: number;
  deployment_ready: boolean;
};

export type TrainingLabSnapshot = {
  dataset_id: string;
  created_at_utc?: string | null;
  comparison_count: number;
  candidate_count: number;
  track_count: number;
  production_training_ready: boolean;
  path: string;
};

export type TrainingLabShadowModel = {
  model_path: string;
  model_id: string;
  kind?: string | null;
  dataset_id?: string | null;
  comparison_count: number;
  cv_accuracy?: number | null;
  baseline_accuracy?: number | null;
  authority?: number | null;
  shadow_only: boolean;
  trained_at_utc?: string | null;
};

export type TrainingLabStatus = {
  schema_version: string;
  track_count: number;
  listening_sessions: number;
  rendered_candidates: number;
  preferences: {
    total: number;
    decisive: number;
    tie: number;
    neither: number;
    choice_a: number;
    choice_b: number;
    unique_track_pairs: number;
    listeners: Record<string, number>;
    top_reasons: { reason: string; count: number }[];
  };
  targets: {
    train_minimum: number;
    benchmark_minimum: number;
  };
  train_ready: boolean;
  train_blockers: string[];
  live_ranker: RankerStatus;
  human_preference_program?: HumanReviewStatus;
  runtime_evidence: {
    schema_version: string;
    path: string;
    events: number;
    candidates: number;
    selected: number;
    gate_passed: number;
    hard_failed: number;
    modes: Record<string, number>;
    raw_audio_stored: boolean;
    track_paths_stored: boolean;
  };
  crate_memory: {
    version: string;
    schema_version: number;
    available: boolean;
    enabled?: boolean;
    track_records: number;
    pair_records: number;
    preference_events: number;
    positive_pair_memories?: number;
    failure_pair_memories?: number;
    maximum_live_adjustment: number;
    hard_audio_safety_authoritative: boolean;
  };
  learning_heads: {
    global_taste_ranker: { human_decisive_examples: number; minimum_to_train: number; ready: boolean; authority: string };
    state_move_taste: {
      human_decisive_examples: number;
      state_groups: number;
      minimum_to_train: number;
      ready: boolean;
      authority: string;
      feature_count: number;
      status: { available: boolean; model_id?: string; comparison_count?: number; cv_accuracy?: number | null; baseline_accuracy?: number | null };
    };
    rendered_dj_judge: { human_labeled_candidates: number; synthetic_failures_enabled: boolean; authority: string };
  };
  snapshots: TrainingLabSnapshot[];
  shadow_models: TrainingLabShadowModel[];
  rights: {
    source_rights_confirmed: boolean;
    registry_ready: boolean;
    production_training_ready: boolean;
    blockers: Record<string, unknown>[];
  };
  policy: {
    raw_audio_bundled_in_snapshots: boolean;
    shadow_training_changes_live_model: boolean;
    hard_audio_safety_can_be_overridden_by_ranker: boolean;
  };
};

export type TrainingLabTrainResult = {
  schema_version: string;
  dataset_id: string;
  snapshot_path: string;
  global_taste_ranker: {
    status: "trained" | "blocked";
    learning_head?: string;
    model_id?: string;
    model_path?: string;
    shadow_only: boolean;
    comparison_count?: number;
    cv_accuracy?: number | null;
    baseline_accuracy?: number | null;
    win_margin?: number | null;
    reason?: string;
  };
  state_move_taste: {
    status: "trained" | "blocked";
    learning_head: string;
    model_id?: string;
    model_path?: string;
    shadow_only: boolean;
    authority?: number;
    comparison_count?: number;
    state_group_count?: number;
    train_accuracy?: number | null;
    cv_accuracy?: number | null;
    baseline_accuracy?: number | null;
    reason?: string;
    reason_tags?: Record<string, number>;
  };
  rendered_dj_judge: {
    status: "trained" | "blocked";
    learning_head: string;
    model_id?: string;
    model_path?: string;
    shadow_only: boolean;
    validation_accuracy?: number | null;
    validation_mae?: number | null;
    baseline_accuracy?: number | null;
    validation_gain?: number | null;
    reason?: string;
    dataset?: { examples?: number; groups?: number; embedding_examples?: number; label_sources?: Record<string, number> };
    runtime_shadow_checkpoint?: { installed: boolean; path?: string; reason?: string; effective_authority?: number };
  };
  runtime_evidence: TrainingLabStatus["runtime_evidence"];
  policy: {
    new_heads_shadow_only: boolean;
    hard_audio_safety_authoritative: boolean;
    training_changes_transition_acceptance: boolean;
    research_data_can_promote_to_production: boolean;
  };
};

export type DatasetBuilderSource = {
  id: string;
  name: string;
  lane: "research" | "production";
  purpose: string[];
  homepage: string;
  importer: string;
  contains_audio: boolean;
  rights_status: string;
  default_asset?: string | null;
  notes: string;
};

export type BootstrapMetricRow = {
  examples: number;
  groups?: number;
  pairwise_comparisons?: number;
  accuracy: number;
  mae?: number;
  balanced_accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  roc_auc?: number;
  pr_auc?: number;
  positive_rate?: number;
  pairwise_accuracy?: number;
  top1_accuracy?: number;
  top3_recall?: number;
  mrr?: number;
  random_top1?: number;
  random_top3?: number;
  random_mrr?: number;
};

export type NextTrackRetrievalScores = {
  hit_at_1: number;
  hit_at_5: number;
  hit_at_10: number;
  mrr: number;
};

export type NextTrackRetrievalMetric = {
  examples: number;
  known_source_examples: number;
  cold_source_examples: number;
  seen_edge_examples: number;
  unseen_edge_examples: number;
  target_seen_examples: number;
  cold_target_examples?: number;
  source_coverage: number;
  edge_coverage: number;
  target_coverage: number;
  model: NextTrackRetrievalScores;
  global_popularity_baseline: NextTrackRetrievalScores;
  random_candidate_baseline: NextTrackRetrievalScores;
};

export type NextTrackRetrievalSubgroups = {
  seen_edge: NextTrackRetrievalMetric;
  unseen_edge_known_source: NextTrackRetrievalMetric;
  cold_source: NextTrackRetrievalMetric;
  cold_target: NextTrackRetrievalMetric;
  known_source_seen_target: NextTrackRetrievalMetric;
  novel_pair_candidate: NextTrackRetrievalMetric;
};

export type NextTrackRetrievalEvaluation = {
  evaluation_kind: "held_out_next_track_retrieval" | string;
  primary_metric: "hit_at_10" | string;
  model_kind: string;
  split_policy: string;
  candidate_universe: string;
  candidate_count: number;
  metrics: {
    validation: NextTrackRetrievalMetric;
    test: NextTrackRetrievalMetric;
  };
  subgroups?: {
    validation: NextTrackRetrievalSubgroups;
    test: NextTrackRetrievalSubgroups;
  };
  improvement_vs_global_popularity: {
    validation: NextTrackRetrievalScores;
    test: NextTrackRetrievalScores;
  };
  generalization_improvement_vs_global_popularity?: {
    validation: Record<keyof NextTrackRetrievalSubgroups, NextTrackRetrievalScores>;
    test: Record<keyof NextTrackRetrievalSubgroups, NextTrackRetrievalScores>;
  };
  generalization_note?: string;
  interpretation: string;
};

export type BootstrapShadowModelResult = {
  model_id: string;
  status: "trained" | "skipped";
  path?: string;
  reason?: string;
  examples?: number;
  unique_edges?: number;
  unique_sources?: number;
  unique_targets?: number;
  evaluation_kind?: "grouped_pairwise_ranking" | "binary_classification" | "held_out_next_track_retrieval" | string;
  evaluation?: NextTrackRetrievalEvaluation;
  primary_metric?: string;
  metrics?: {
    train: BootstrapMetricRow;
    validation: BootstrapMetricRow;
    test: BootstrapMetricRow;
  };
  baseline?: {
    kind: string;
    class?: number;
    metrics: {
      train: BootstrapMetricRow;
      validation: BootstrapMetricRow;
      test: BootstrapMetricRow;
    };
  };
  accuracy_improvement_vs_baseline?: { train: number; validation: number; test: number };
  primary_improvement_vs_baseline?: { train: number; validation: number; test: number };
};

export type BootstrapTrainingResult = {
  schema_version: string;
  corpus_id: string;
  trained_at_utc: string;
  shadow_only: boolean;
  live_models_changed: boolean;
  runtime?: {
    mode: "off" | "shadow" | "advisory" | string;
    requested_mode?: string;
    corpus_id?: string | null;
    production_guarded?: boolean;
    hard_safety_authoritative?: boolean;
  };
  models: BootstrapShadowModelResult[];
};

export type DatasetBuilderStatus = {
  schema_version: string;
  catalog: {
    schema_version: string;
    policy: Record<string, unknown>;
    sources: DatasetBuilderSource[];
  };
  imports: Array<{
    source_id: string;
    records: number;
    path: string;
    rights_cleared: boolean;
    counts: Record<string, number>;
  }>;
  counts: {
    normalized_records: number;
    weak_examples: number;
    kinds: Record<string, number>;
    sources: Record<string, number>;
    weak_tasks: Record<string, number>;
  };
  corpora: Array<{
    corpus_id: string;
    created_at_utc?: string | null;
    records: number;
    weak_examples: number;
    production_ready: boolean;
    path: string;
  }>;
  bootstrap_models: Array<{
    corpus_id: string;
    trained_at_utc?: string | null;
    trained_models: number;
    model_count: number;
    live_models_changed: boolean;
    path: string;
    models?: BootstrapShadowModelResult[];
  }>;
  latest_benchmark: BootstrapTrainingResult | null;
  model_readiness: Record<string, { examples: number; source: string }>;
  policy: {
    external_sources_default_research: boolean;
    raw_audio_bundled_in_normalized_data: boolean;
    synthetic_labels_equal_human_preferences: boolean;
    production_default_deny: boolean;
  };
};

export type BrainStatus = {
  version: string;
  professional_brain_enabled?: boolean;
  brain_config?: {
    analytical_pool?: number;
    finalists?: number;
    retry_rounds?: number;
    diversity_weight?: number;
    minimum_judge_score?: number;
  };
  preference_ranker?: RankerStatus;
  professional_mixing?: {
    version?: string;
    compatibility_driven?: boolean;
    families?: string[];
    ownership_roles?: string[];
    single_bass_owner?: boolean;
    avoid_dual_lead_vocals?: boolean;
    outgoing_owner_until_incoming_established?: boolean;
    reset_is_last_resort?: boolean;
    pair_may_be_rejected?: boolean;
  };
  professional_mix_contract?: {
    version?: string;
    continuous_mix_requires?: string[];
    structural_resets_require_explicit_justification?: boolean;
    combined_audio_quality_alone_is_insufficient?: boolean;
  };
  structural_operations?: {
    version?: string;
    enabled?: boolean;
    true_wait?: boolean;
    automatic_outgoing_loop?: boolean;
    automatic_intro_extension?: boolean;
    automatic_stem_hold?: boolean;
    automatic_jumps?: boolean;
    supported?: string[];
  };
  bootstrap_runtime?: {
    schema_version?: string;
    mode?: string;
    requested_mode?: string;
    production_guarded?: boolean;
    production_approved?: boolean;
    corpus_id?: string | null;
    models?: Record<string, {
      available?: boolean;
      heldout_confidence?: number | null;
      research_shadow?: boolean;
      promotable_to_live?: boolean;
      model_id?: string | null;
    }>;
  };
  taste_learning?: {
    runtime_evidence?: TrainingLabStatus["runtime_evidence"];
    latest_snapshot_id?: string | null;
    global_taste_ranker?: { available?: boolean; shadow_only?: boolean; comparison_count?: number; cv_accuracy?: number | null; human_decisive_examples?: number };
    rendered_dj_judge?: { available?: boolean; shadow_only?: boolean; human_examples?: number; synthetic_examples?: number; validation_accuracy?: number | null };
    state_move_taste?: { available?: boolean; shadow_only?: boolean; authority?: number; minimum_comparisons?: number; comparison_count?: number };
  };
  crate_memory?: TrainingLabStatus["crate_memory"] & {
    directional_pairs?: boolean;
    structural_track_facts?: boolean;
    human_preference_authority?: boolean;
    self_training_from_planner_choices?: boolean;
  };
  dj_executive?: {
    version?: string;
    mode?: string;
    enabled?: boolean;
    authoritative?: boolean;
    single_commit_authority?: boolean;
    commit_authority?: string;
    supervisory_orchestrator?: string;
    orchestration_model?: string;
    executor_contract?: string;
    crowd_state_connected?: boolean;
    crowd_state_authority?: string;
    crowd_control_enabled?: boolean;
    crowd_minimum_confidence?: number | null;
    crowd_max_track_adjustment?: number | null;
    transition_policy_connected?: boolean;
    transition_policy_authority?: string;
    transition_policy_enabled?: boolean;
    continuous_live_replanning?: boolean;
    prepared_recovery_capable?: boolean;
    commit_validation_fail_closed?: boolean;
    adaptive_thinking?: { enabled?: boolean; minimum_budget?: number; maximum_budget?: number };
  };
  dj_move_brain?: {
    schema_version?: string;
    enabled?: boolean;
    planning_unit?: string;
    coordinates?: string;
    order_weight?: number;
    plan_weight?: number;
    payoff_protection?: number;
    protected_importance?: number;
    minimum_body_bars?: number;
    soft_target_bars?: number;
    soft_latest_bars?: number;
    explicit_wait_actions?: boolean;
    musical_commitments?: boolean;
    song_form_roles?: boolean;
    first_body_protection?: boolean;
    hook_protection?: boolean;
    first_hook_pre_entry_protection?: boolean;
    song_form_hard_guard?: boolean;
    future_optionality?: boolean;
    hard_audio_safety_authoritative?: boolean;
    complete_move_generator?: Record<string, unknown>;
    structural_deck_operations?: Record<string, unknown>;
  };
  musical_timeline?: {
    schema_version?: string;
    authority?: string;
    canonical_phrase_authority?: boolean;
    song_form_roles?: string[];
    first_body_protection?: boolean;
    hook_protection?: boolean;
    first_hook_pre_entry_protection?: boolean;
    song_form_hard_guard?: boolean;
    cache_entries?: number;
    cache_limit?: number;
  };
  large_crate_planning?: {
    schema_version?: string;
    threshold_tracks?: number;
    graph_strategy?: string;
    graph_scout_cues_per_side?: number;
    receding_horizon_opener_limit?: number;
    local_improve_passes?: number;
  };
  deck_performance?: {
    schema_version?: string;
    tempo_release_enabled?: boolean;
    phase_nudge_enabled?: boolean;
    phase_nudge_max_ms?: number;
    tempo_release_max_delta?: number;
    pitch_preserving?: boolean;
    tempo_ownership_transfer?: boolean;
    stem_vocal_exchange_profile?: boolean;
    hard_audio_safety_authoritative?: boolean;
  };
  expert_imitation_examples?: number;
  rejection_memory_count?: number;
  release_benchmarks?: { available?: boolean; run_count?: number; champion_count?: number; champions?: unknown[] };
  production_data_ready?: boolean;
  production_data_blockers?: unknown[];
  arena?: { decisive?: number; deployment_ready?: boolean; champion?: string | null; challenger?: string | null };
  curriculum?: unknown[];
};

export type ReleaseReadiness = {
  schema_version: string;
  version: string;
  release_channel: string;
  candidate_number: number;
  operational_ready: boolean;
  release_certifiable: boolean | null;
  blocker_count: number;
  warning_count: number;
  authority: {
    planner_mode: string;
    ai_headphones: boolean;
    crate_memory: boolean;
    structural_deck_operations: boolean;
    hard_audio_safety_authoritative: boolean;
  };
};
