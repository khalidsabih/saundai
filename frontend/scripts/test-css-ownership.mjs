#!/usr/bin/env node
/**
 * Phase 5 CSS ownership regression gate.
 *
 * Asserts that the feature split preserves the exact bytes of the old
 * globals.css and that layout still imports the split in the old order
 * before the pre-existing later cascade layers.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const frontend = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const read = (name) => readFileSync(resolve(frontend, "app", name), "utf8");
const original = execFileSync(
  "git",
  ["show", "a4469d3:frontend/app/globals.css"],
  { cwd: frontend, encoding: "utf8" },
);
const extracted = ["globals.css", "training-lab.css", "account-foundation.css", "legacy-experiences.css"];
const reconstructed = extracted.map(read).join("");
assert.equal(reconstructed, original, "Extracted CSS must reconstruct the original globals.css byte-for-byte");

const layout = readFileSync(resolve(frontend, "app/layout.tsx"), "utf8");
const expected = [
  ...extracted,
  "design-tokens.css",
  "autonomous.css",
  "studio-intelligence.css",
  "product-system.css",
];
const imports = [...layout.matchAll(/^import "\.\/(.+\.css)";$/gm)].map((match) => match[1]);
assert.deepEqual(imports, expected, "Global CSS cascade imports must retain the verified order");
console.log("Phase 5 CSS ownership: original stylesheet contents and cascade order match.");
