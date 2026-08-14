/**
 * Does the "what is this?" marker actually explain anything?
 *
 * The markers attach to figures that only exist once a match is on screen, and
 * the harness has no match to open — so this exercises the mechanism directly:
 * put an element that names a figure into the page, let the observer find it,
 * click the marker it adds, and read what comes back.
 *
 * What it is really checking is that the wording in the popover came from
 * src/domain/metrics/catalogue.py over the API, rather than from a copy of the
 * text living in the frontend. If those two ever diverge this is what says so.
 *
 * Usage:  node tools/domtest/explain.mjs
 */

import { loadPage } from "./harness.mjs";

const PAGE = "match-breakdown.html";
const FIGURE = "aim_stats.preaim";

const { window, document, errors, restore } = await loadPage(PAGE);
const result = { page: PAGE, figure: FIGURE, errors };

const host = document.createElement("span");
host.dataset.explain = FIGURE;
host.textContent = "Crosshair Placement";
document.body.appendChild(host);

// The MutationObserver runs on a microtask; give it one.
await new Promise((done) => setTimeout(done, 50));

const marker = host.querySelector(".explain-marker");
result.marker_added = Boolean(marker);
result.marker_action = marker?.dataset.action ?? null;

if (marker) {
  marker.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await new Promise((done) => setTimeout(done, 200));
  const popover = document.querySelector(".explain-popover");
  result.popover_shown = Boolean(popover);
  result.text = popover ? popover.textContent.replace(/\s+/g, " ").trim() : "";
}

restore();
window.close();
process.stdout.write(JSON.stringify(result, null, 2) + "\n");
