import { expect, test } from "@playwright/test";

test("readiness dashboard separates engine, crate, model authority and Phase 16", async ({ page }) => {
  await page.route("**/api/tracks?**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        tracks: [
          { name: "A.wav", path: "audio/uploads/A.wav", size: 1000, media_url: "/api/media/A.wav", readiness: { state: "ready", dj_ready: true, progress: 1, message: "ready", schema_version: "1", fingerprint: "a", features: {} } },
          { name: "B.wav", path: "audio/uploads/B.wav", size: 1000, media_url: "/api/media/B.wav", readiness: { state: "ready", dj_ready: true, progress: 1, message: "ready", schema_version: "1", fingerprint: "b", features: {} } },
        ],
      }),
    });
  });

  await page.route("**/api/brain/status", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        version: "test",
        dj_executive: {
          enabled: true,
          authoritative: true,
          single_commit_authority: true,
          continuous_live_replanning: true,
          prepared_recovery_capable: true,
        },
        bootstrap_runtime: {
          mode: "shadow",
          models: {
            next_track_policy: { available: true, research_shadow: true, heldout_confidence: 0.84 },
          },
        },
      }),
    });
  });

  await page.route("**/api/release/status", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        schema_version: "2",
        version: "test",
        release_channel: "development",
        candidate_number: 1,
        operational_ready: true,
        release_certifiable: null,
        blocker_count: 0,
        warning_count: 0,
        authority: {
          planner_mode: "move_authoritative",
          ai_headphones: true,
          crate_memory: true,
          structural_deck_operations: true,
          hard_audio_safety_authoritative: true,
        },
      }),
    });
  });

  await page.route("**/api/validation/phase16", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        available: true,
        phase: 16,
        name: "Real-Music Validation",
        summary: {
          run_count: 3,
          planning_ready_runs: 3,
          rehearsal_ready_runs: 3,
          hard_failure_count: 0,
          fallback_transition_count: 2,
          all_planning_ready: true,
          all_rehearsal_ready: true,
          phase16_passed: true,
        },
        runs: [
          { journey: "balanced", rehearsal_ready: true, hard_failure_count: 0, quality_pass_rate: 1 },
          { journey: "build", rehearsal_ready: true, hard_failure_count: 0, quality_pass_rate: 0.98 },
          { journey: "peak", rehearsal_ready: true, hard_failure_count: 0, quality_pass_rate: 1 },
        ],
      }),
    });
  });

  await page.goto("/readiness");

  await expect(page.getByRole("heading", { name: "Know what SAUND is ready to do." })).toBeVisible();
  await expect(page.getByText("Operational", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Passed", { exact: true })).toBeVisible();
  await expect(page.getByText("2/2 ready", { exact: true })).toBeVisible();
  await expect(page.getByText("Shadow", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("3/3 journeys rehearsal-ready", { exact: true })).toBeVisible();
});
