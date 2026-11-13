/**
 * Next.js standalone output omits public/ and .next/static/ on purpose.
 * The Docker image copies them separately; mirror that layout for local E2E.
 * This script does not mutate source files or regenerate the production build.
 */
import { cpSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const root = process.cwd();
const standalone = join(root, ".next", "standalone");
const server = join(standalone, "server.js");

if (!existsSync(server)) {
  throw new Error("No standalone build found. Run npm run build before npm run e2e.");
}

const copies = [
  [join(root, "public"), join(standalone, "public")],
  [join(root, ".next", "static"), join(standalone, ".next", "static")],
];

for (const [source, destination] of copies) {
  if (!existsSync(source)) continue;
  mkdirSync(destination, { recursive: true });
  cpSync(source, destination, { recursive: true, force: true });
}
