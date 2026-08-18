import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const pkg = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
const info = {
  product: "SAUND.AI",
  version: pkg.version,
  build_commit: process.env.SAUND_BUILD_COMMIT || null,
  build_time_utc: process.env.SAUND_BUILD_TIME_UTC || new Date().toISOString(),
  source: "next-production-build",
};
fs.mkdirSync(path.join(root, "public"), { recursive: true });
fs.writeFileSync(path.join(root, "public", "build-info.json"), `${JSON.stringify(info, null, 2)}\n`);
console.log(`SAUND.AI frontend build identity: ${info.version} ${info.build_commit ?? "no-commit"}`);
