// Copy the canonical (or latest) NeoServe run summary into public/ so the dashboard
// serves it statically. Run: npm run refresh
import { cpSync, existsSync, readdirSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const resultsRoot = join(here, "..", "..", "results");
const canonical = join(resultsRoot, "canonical", "summary.json");

function latestSummary() {
  const dirs = readdirSync(resultsRoot)
    .map((d) => join(resultsRoot, d, "summary.json"))
    .filter((p) => existsSync(p));
  dirs.sort((a, b) => statSync(a).mtimeMs - statSync(b).mtimeMs);
  return dirs[dirs.length - 1];
}

const src = existsSync(canonical) ? canonical : latestSummary();
if (!src) {
  console.error("No NeoServe summary.json found. Run the harness first.");
  process.exit(1);
}
const dest = join(here, "..", "public", "summary.json");
cpSync(src, dest);
console.log(`refreshed dashboard data from ${src}`);
