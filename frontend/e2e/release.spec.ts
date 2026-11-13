import { expect, test, type Page } from "@playwright/test";

const tracks = [
  { name: "Aurora.wav", path: "audio/Aurora.wav", size: 8_000_000, media_url: "/api/media/audio/Aurora.wav" },
  { name: "Afterglow.wav", path: "audio/Afterglow.wav", size: 9_000_000, media_url: "/api/media/audio/Afterglow.wav" },
  { name: "Closing.wav", path: "audio/Closing.wav", size: 7_000_000, media_url: "/api/media/audio/Closing.wav" },
];

const setPlan = {
  plan_id: "release-smoke-plan",
  source: "e2e",
  manual: false,
  automatic_delivery: true,
  requires_confirmation: false,
  confirmation_token: null,
  baseline_order: [0, 1],
  order: [0, 1],
  optimized_score: 0.9,
  baseline_score: 0.8,
  improvement_percent: 12.5,
  estimated_duration_seconds: 360,
  tracks: [
    { index: 0, path: tracks[0].path, title: "Aurora", duration_seconds: 180, bpm: 124, key: { camelot: "8A", tonic: "A", mode: "minor", confidence: 0.9 }, genre: { label: "electronic", confidence: 0.8, probabilities: {} }, energy: 0.6, variable_tempo: false, time_signature: 4, waveform_peaks: [] },
    { index: 1, path: tracks[1].path, title: "Afterglow", duration_seconds: 180, bpm: 124, key: { camelot: "9A", tonic: "E", mode: "minor", confidence: 0.9 }, genre: { label: "electronic", confidence: 0.8, probabilities: {} }, energy: 0.8, variable_tempo: false, time_signature: 4, waveform_peaks: [] },
  ],
  ordered_tracks: [
    { index: 0, path: tracks[0].path, title: "Aurora", duration_seconds: 180, bpm: 124, key: { camelot: "8A", tonic: "A", mode: "minor", confidence: 0.9 }, genre: { label: "electronic", confidence: 0.8, probabilities: {} }, energy: 0.6, variable_tempo: false, time_signature: 4, waveform_peaks: [] },
    { index: 1, path: tracks[1].path, title: "Afterglow", duration_seconds: 180, bpm: 124, key: { camelot: "9A", tonic: "E", mode: "minor", confidence: 0.9 }, genre: { label: "electronic", confidence: 0.8, probabilities: {} }, energy: 0.8, variable_tempo: false, time_signature: 4, waveform_peaks: [] },
  ],
  transitions: [
    { from_index: 0, to_index: 1, selected_style: "safe", objective_score: 0.9, quality: "good", warnings: [], reasons: [], alternatives: [], plan: { style: "safe", technique: "equal_power_blend", score: 0.9, exit_time_a: 160, entry_time_b: 8, transition_bars: 16, transition_seconds: 30, target_bpm: 124, transition_mode: "beatmatched", beatmatched: true, tempo_strategy: "native", compatibility: { overall: 0.9, tempo: 1, harmonic: 0.9, energy: 0.9, texture: 0.9, beat_grid: 1, phrase: 1, sections: 1, tempo_difference_percent: 0 }, quality_checked: true, measurement_passed: true } },
  ],
  warnings: [],
  metadata: { journey: "peak" },
};


const authSession = {
  authenticated: true,
  user: { id: "user-e2e", email: "listener@example.com", created_at: "2026-08-13T20:00:00+00:00" },
  preferences: { personalization_enabled: false, improve_saund_ai: false },
};

async function mockAuth(page: Page, authenticated = true) {
  await page.route("**/api/auth/me", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(authenticated
        ? authSession
        : { authenticated: false, user: null, preferences: null }),
    });
  });
}

async function mockLibrary(page: Page) {
  await page.route("**/api/tracks?**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ tracks }) });
  });
}

async function completeIntro(page: Page) {
  const enter = page.getByRole("button", { name: "Start a session" });
  await expect(enter).toBeDisabled();
  // A forced synthetic click must not bypass the authoritative progress gate.
  await enter.dispatchEvent("click");
  await expect(page.getByRole("heading", { name: "Tell SAUND how the night should feel." })).not.toBeVisible();

  // Reduced-motion makes the timeline complete immediately by design, which
  // keeps the rest of the cross-browser smoke suite fast.
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.reload();
  const readyEnter = page.getByRole("button", { name: "Start a session" });
  await expect(readyEnter).toBeEnabled();
  await readyEnter.click();
}

test("SAUND.AI opens as a locked cinematic intro before Create", async ({ page }) => {
  await mockLibrary(page);
  await page.goto("/");

  const intro = page.locator(".saund-intro");
  const video = page.locator("video.saund-intro__video");
  await expect(intro).toBeVisible();
  await expect(page.getByRole("heading", { name: "An AI that DJs." })).toBeVisible();
  await expect(page.getByText("Autonomous DJ system")).toHaveCount(1);
  await expect(video).toHaveAttribute("poster", "/media/saund-hero-poster.jpg");
  await expect(video.locator('source[src="/media/saund-hero-desktop.mp4"]')).toHaveCount(1);
  await expect(video.locator('source[src="/media/saund-hero-mobile.mp4"]')).toHaveCount(1);
  await expect(video).not.toHaveAttribute("controls", /.*/);
  await expect(page.getByRole("button", { name: /Sound on|Play sound|Mute/ })).toBeVisible();
  const enter = page.getByRole("button", { name: "Start a session" });
  await expect(enter).toBeVisible();
  await expect(enter).toBeDisabled();
  await expect(page.getByRole("heading", { name: "Tell SAUND how the night should feel." })).not.toBeVisible();
  expect(await page.evaluate(() => getComputedStyle(document.body).overflow)).toBe("hidden");

  // Even a synthetic click before completion must not enter the workspace.
  await enter.dispatchEvent("click");
  await expect(page.getByRole("heading", { name: "Tell SAUND how the night should feel." })).not.toBeVisible();

  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.reload();
  const readyEnter = page.getByRole("button", { name: "Start a session" });
  await expect(readyEnter).toBeEnabled();
  await readyEnter.click();
  await expect(page.getByRole("heading", { name: "Tell SAUND how the night should feel." })).toBeVisible();
  await expect(page.getByRole("button", { name: /Sound on|Play sound|Mute/ })).toHaveCount(0);
  expect(await page.evaluate(() => getComputedStyle(document.body).overflow)).not.toBe("hidden");
});

test("guest can shape a set and is asked to log in only when building", async ({ page }) => {
  await mockAuth(page, false);
  await mockLibrary(page);
  let buildCalled = false;
  await page.route("**/api/sets/automatic", async (route) => {
    buildCalled = true;
    await route.fulfill({ status: 401, contentType: "application/json", body: JSON.stringify({ detail: { code: "login_required", message: "Log in to build your SAUND.AI set." } }) });
  });

  await page.goto("/");
  await completeIntro(page);
  await page.getByRole("button", { name: /Aurora\.wav/ }).click();
  await page.getByRole("button", { name: /Afterglow\.wav/ }).click();
  await page.getByRole("radio", { name: /Drive/ }).click();
  await page.getByRole("button", { name: /Build my set/ }).click();

  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Log in to start." })).toBeVisible();
  await expect(page.getByText("Your tracks and direction stay exactly as they are.")).toBeVisible();
  expect(buildCalled).toBe(false);
});

test("Create turns selected music into a finished set", async ({ page }) => {
  await mockAuth(page, true);
  await mockLibrary(page);
  await page.route("**/api/sets/automatic", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        job: {
          job_id: "create-smoke",
          kind: "automatic_set",
          status: "completed",
          progress: 1,
          stage: "done",
          message: "Ready",
          error: null,
          result: {
            plan: setPlan,
            plan_path: "output/set-plan.json",
            audio_path: "output/set.wav",
            report_path: "output/set.json",
            audio_url: "/api/media/output/set.wav",
            live_stream_url: "/api/live/stream?plan_path=output/set-plan.json",
            live_manifest_url: "/api/live/manifest?plan_path=output/set-plan.json",
          },
        },
      }),
    });
  });
  await page.route("**/api/media/**", async (route) => route.fulfill({ status: 204 }));

  await page.goto("/");
  await expect(page.getByText("Autonomous DJ system")).toHaveCount(1);
  await completeIntro(page);
  await expect(page.getByRole("heading", { name: "Tell SAUND how the night should feel." })).toBeVisible();
  await page.getByRole("button", { name: /Aurora\.wav/ }).click();
  await page.getByRole("button", { name: /Afterglow\.wav/ }).click();
  await page.getByRole("radio", { name: /Drive/ }).click();
  await page.getByRole("button", { name: /Build my set/ }).click();

  await expect(page.getByRole("heading", { name: "Your set is ready." })).toBeVisible();
  await expect(page.getByText("PERFORMANCE RIBBON", { exact: true })).toBeVisible();
  await expect(page.getByText(/NEXT · LOCKED/)).toHaveCount(0);
  await expect(page.getByRole("button", { name: /Perform Live/ })).toBeVisible();
  await expect(page.getByText("2", { exact: true })).toBeVisible();
});


test("live performance makes NOW, NEXT, FUTURE and the Executive the product surface", async ({ page }) => {
  await mockAuth(page, true);
  await mockLibrary(page);
  await page.route("**/api/sets/automatic", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        job: {
          job_id: "live-ui-smoke",
          kind: "automatic_set",
          status: "completed",
          progress: 1,
          stage: "done",
          message: "Ready",
          error: null,
          result: {
            plan: setPlan,
            plan_path: "output/set-plan.json",
            audio_path: "output/set.wav",
            report_path: "output/set.json",
            audio_url: "/api/media/output/set.wav",
            live_stream_url: "/api/live/stream?plan_path=output/set-plan.json",
            live_manifest_url: "/api/live/manifest?plan_path=output/set-plan.json",
          },
        },
      }),
    });
  });

  const timeline = [
    { type: "track", set_start: 0, set_end: 150, track_index: 0, title: "Aurora" },
    { type: "transition", set_start: 150, set_end: 180, from_index: 0, to_index: 1, from_title: "Aurora", to_title: "Afterglow", technique: "equal_power_blend", intent: "energy_continuity", target_bpm: 124 },
    { type: "track", set_start: 180, set_end: 360, track_index: 1, title: "Afterglow" },
  ];
  const orderedTracks = [
    { index: 0, title: "Aurora", bpm: 124, key: "8A" },
    { index: 1, title: "Afterglow", bpm: 124, key: "9A" },
  ];

  await page.route("**/api/live/manifest?**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        plan_id: setPlan.plan_id,
        duration_seconds: 360,
        timeline,
        ordered_tracks: orderedTracks,
        runtime_control: { continuous_replanning: true, now_mutable: false, next_mutable: false, future_replannable: true, manifest_is_initial_schedule: true, telemetry_supported: true, schedule_generation: 0 },
      }),
    });
  });
  await page.route("**/api/live/events?**", async (route) => {
    const snapshot = {
      version: "1",
      sequence: 1,
      session_id: "e2e",
      plan_id: setPlan.plan_id,
      status: "streaming",
      updated_at_utc: "2026-08-16T20:00:00Z",
      detail: "Holding committed horizon",
      schedule_generation: 0,
      active_order: [0, 1],
      duration_seconds: 360,
      timeline,
      ordered_tracks: orderedTracks,
      runtime_control: { continuous_replanning: true, now_mutable: false, next_mutable: false, future_replannable: true, manifest_is_initial_schedule: false, telemetry_supported: true, schedule_generation: 0 },
      playback_state: { state_version: 1, schedule_generation: 0, executor_phase: "armed", current_track: 0, source_cursor: 0, set_time: 0, edge_position: 0, expected_command: "body", commit_digest: "e2e", safety_action: "allow" },
      horizons: {
        now: { horizon: "now", status: "executing", track_index: 0, edge_position: 0, from_index: null, to_index: null, execution_signature: null, replannable: false },
        next: { horizon: "next", status: "prepared", track_index: 1, edge_position: 0, from_index: 0, to_index: 1, execution_signature: "edge-0", replannable: false },
        future: [],
      },
      safety: { action: "allow", reason: "Execution contract healthy" },
      executive: { enabled: true, cycle: 1, replanning: false, last_committed_generation: 0, crowd_state_authority: "bounded_advisory", transition_policy_authority: "bounded_advisory", last_event: null },
      crowd: { state: { confidence: 0.4, engagement: 0.7, momentum: 0.2 }, directive: { mode: "hold", confidence: 0.4 } },
    };
    await route.fulfill({ status: 200, contentType: "text/event-stream", body: `data: ${JSON.stringify(snapshot)}\n\n` });
  });
  await page.route("**/api/live/stream?**", async (route) => route.fulfill({ status: 200, contentType: "audio/wav", body: "" }));
  await page.route("**/api/media/**", async (route) => route.fulfill({ status: 204 }));

  await page.goto("/");
  await completeIntro(page);
  await page.getByRole("button", { name: /Aurora\.wav/ }).click();
  await page.getByRole("button", { name: /Afterglow\.wav/ }).click();
  await page.getByRole("button", { name: /Build my set/ }).click();
  await page.getByRole("button", { name: /Perform Live/ }).click();

  await expect(page.getByText(/NOW ·/).first()).toBeVisible();
  await expect(page.getByText("NEXT", { exact: true })).toBeVisible();
  await expect(page.getByText("FUTURE", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("EXECUTIVE", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Afterglow", { exact: true }).first()).toBeVisible();

  await page.getByRole("button", { name: "Brain", exact: true }).click();
  await expect(page.getByRole("heading", { name: "What the DJ is doing underneath." })).toBeVisible();
  await expect(page.getByText("SAFETY GOVERNOR", { exact: true })).toBeVisible();
});

test("Brain is a first-class autonomous system view and Studio is separate", async ({ page }) => {
  await mockLibrary(page);
  await page.route("**/api/brain/status", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        version: "6.0.0",
        professional_brain_enabled: true,
        dj_move_brain: { enabled: true },
        deck_performance: { tempo_release_enabled: true, phase_nudge_enabled: true, hard_audio_safety_authoritative: true },
        dj_executive: {
          mode: "authoritative",
          enabled: true,
          single_commit_authority: true,
          executor_contract: "validated_immutable_command_only",
          crowd_state_authority: "live_bounded_advisory",
          crowd_control_enabled: true,
          transition_policy_enabled: true,
          continuous_live_replanning: true,
          prepared_recovery_capable: true,
        },
      }),
    });
  });
  await page.goto("/");
  await completeIntro(page);
  await expect(page.getByRole("heading", { name: "Tell SAUND how the night should feel." })).toBeVisible();
  await page.getByRole("button", { name: /Aurora\.wav/ }).click();
  await page.getByRole("button", { name: /Afterglow\.wav/ }).click();
  await page.getByRole("button", { name: "Brain", exact: true }).click();

  await expect(page.getByRole("heading", { name: /See what SAUND actually knows\./ })).toBeVisible();
  await expect(page.getByText("BRAIN", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("EXECUTIVE", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("SAFETY", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("EXECUTOR", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: /Open Studio/ })).toBeVisible();
  await expect(page.getByRole("heading", { name: /Shape the set/ })).toHaveCount(0);

  await page.getByRole("button", { name: "Perform", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Tell SAUND how the night should feel." })).toBeVisible();
  await expect(page.getByRole("button", { name: /Aurora\.wav/ })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: /Afterglow\.wav/ })).toHaveAttribute("aria-pressed", "true");
});

test("Studio is the Human × Executive intelligence workbench", async ({ page }) => {
  await mockLibrary(page);
  await page.route("**/api/brain/status", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ version: "6.0.0", dj_move_brain: { enabled: true }, dj_executive: { enabled: true, single_commit_authority: true } }) });
  });
  await page.goto("/");
  await completeIntro(page);
  await page.getByRole("button", { name: "Brain", exact: true }).click();
  await page.getByRole("button", { name: /Open Studio/ }).click();

  await expect(page.getByRole("heading", { name: /Direct the DJ/ })).toBeVisible();
  await expect(page.getByText("Studio / Human × Executive")).toBeVisible();
  await expect(page.getByRole("button", { name: /Autonomous/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /Guided/ })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: /Manual/ })).toBeVisible();
  await expect(page.getByText(/Explore before you commit/)).toBeVisible();
});

test("API failure becomes an actionable visible error", async ({ page }) => {
  await page.route("**/api/tracks?**", async (route) => {
    await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Engine unavailable" }) });
  });
  await page.goto("/");
  await completeIntro(page);
  await expect(page.getByRole("heading", { name: "Tell SAUND how the night should feel." })).toBeVisible();
  await expect(page.getByRole("region", { name: "Create an autonomous DJ set" }).getByRole("alert")).toContainText("Engine unavailable");
  await expect(page.getByRole("button", { name: "Dismiss error" })).toBeVisible();
});

test("Training Lab collects blind preferences without exposing candidate metadata", async ({ page }) => {
  await mockLibrary(page);
  // TrainingLab refreshes library, training status, dataset builder and brain
  // together. Mock all four to keep the browser contract test independent of
  // a running API or slow network proxy.
  await page.route("**/api/dataset-builder/status", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({
      schema_version: "1.0",
      catalog: { schema_version: "1.0", policy: {}, sources: [] },
      imports: [],
      counts: { normalized_records: 0, weak_examples: 0, kinds: {}, sources: {}, weak_tasks: {} },
      corpora: [],
      bootstrap_models: [],
      latest_benchmark: null,
      model_readiness: {},
      policy: {
        external_sources_default_research: true,
        raw_audio_bundled_in_normalized_data: false,
        synthetic_labels_equal_human_preferences: false,
        production_default_deny: true,
      },
    }) });
  });
  await page.route("**/api/brain/status", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({}) });
  });
  const labStatus = {
    schema_version: "1.0",
    track_count: 3,
    listening_sessions: 0,
    rendered_candidates: 0,
    preferences: { total: 0, decisive: 0, tie: 0, neither: 0, choice_a: 0, choice_b: 0, unique_track_pairs: 0, listeners: {}, top_reasons: [] },
    targets: { train_minimum: 24, benchmark_minimum: 100 },
    train_ready: false,
    train_blockers: ["need_24_more_decisive_preferences"],
    live_ranker: { decisive_preferences: 0, unique_track_pairs: 0, ranker_loaded: false, cv_accuracy: null, baseline_accuracy: null, win_margin: null, authority: 0, deployment_ready: false },
    runtime_evidence: {
      schema_version: "1.0", path: "", events: 0, candidates: 0, selected: 0,
      gate_passed: 0, hard_failed: 0, modes: {},
      raw_audio_stored: false, track_paths_stored: false,
    },
    crate_memory: {
      version: "1", schema_version: 1, available: false,
      track_records: 0, pair_records: 0, preference_events: 0,
      maximum_live_adjustment: 0, hard_audio_safety_authoritative: true,
    },
    learning_heads: {
      global_taste_ranker: { human_decisive_examples: 0, minimum_to_train: 24, ready: false, authority: "shadow" },
      state_move_taste: { human_decisive_examples: 0, state_groups: 0, minimum_to_train: 24, ready: false, authority: "shadow", feature_count: 0, status: { available: false } },
      rendered_dj_judge: { human_labeled_candidates: 0, synthetic_failures_enabled: true, authority: "shadow" },
    },
    snapshots: [],
    shadow_models: [],
    rights: { source_rights_confirmed: false, registry_ready: true, production_training_ready: false, blockers: [] },
    policy: { raw_audio_bundled_in_snapshots: false, shadow_training_changes_live_model: false, hard_audio_safety_can_be_overridden_by_ranker: false },
  };
  await page.route("**/api/training-lab/status", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(labStatus) });
  });
  await page.route("**/api/intelligence/listening-session", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        session_id: "lab-e2e",
        candidates: ["A", "B", "C", "D"].map((label) => ({ label, audio_url: `/api/media/output/listening-lab/${label}.wav` })),
      }),
    });
  });
  await page.route("**/api/intelligence/preference", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ preferences_path: "ratings/transition-preferences.jsonl" }) });
  });
  await page.route("**/api/media/**", async (route) => route.fulfill({ status: 204 }));

  await page.goto("/lab");
  await expect(page.getByRole("heading", { name: /Teach the ear/ })).toBeVisible();
  await expect(page.getByText("INTERNAL / LOCAL")).toHaveCount(1);
  await page.getByRole("button", { name: /Generate blind pair/ }).click();
  await expect(page.getByRole("heading", { name: "Pair 1 / 2" })).toBeVisible();
  await expect(page.getByLabel("Blind transition candidate A")).toBeVisible();
  await expect(page.getByLabel("Blind transition candidate B")).toBeVisible();
  await page.getByRole("button", { name: "A Feels better" }).click();
  await expect(page.getByRole("heading", { name: "Pair 2 / 2" })).toBeVisible();
  await expect(page.getByText(/filter_build|eq_bass_swap|professional_judge_score/)).toHaveCount(0);
});
