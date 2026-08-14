/**
 * Records what each page is wired to do, by doing it.
 *
 * For every element carrying an event handler, this dispatches the event and
 * writes down which of the page's own functions ran. The functions are
 * replaced with recorders first, so nothing actually happens — no modal opens,
 * no request is made, no navigation. What comes out is a map of
 *
 *     element  +  event  ->  the functions it calls
 *
 * which is the property that has to survive moving code between files, or
 * moving a handler out of markup and into a listener. If the snapshot is
 * unchanged, every button still does what it did.
 *
 * Usage:  node tools/domtest/wiring.mjs [page.html ...]
 */

import { spawnSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, FRONTEND, loadPage } from "./harness.mjs";

const EVENTS = ["click", "change", "input", "submit", "keyup", "keydown", "blur", "focus"];

// Names a page can use without defining them. Anything jsdom already puts on
// window is not the page's, so the diff below finds only what the page added.
// Same options as the harness, or its own additions look like the page's.
const { JSDOM } = await import("jsdom");
const BASELINE = new Set([
  ...Object.keys(new JSDOM("", { pretendToBeVisual: true }).window),
  "fetch", // stubbed by the harness
  "tailwind",
]);

function pageGlobals(window) {
  return Object.keys(window)
    .filter((key) => !BASELINE.has(key) && typeof window[key] === "function")
    .sort();
}

/** Replace each page function with a recorder. Returns the log they write to. */
function instrument(window, names) {
  const fired = [];

  // Actions are held by reference in the registry, so overwriting the global
  // of the same name would not reach them.
  const registry = window.ACTIONS;
  if (registry) {
    for (const name of Object.keys(registry)) {
      registry[name] = () => {
        fired.push(name);
      };
    }
  }

  for (const name of names) {
    try {
      Object.defineProperty(window, name, {
        configurable: true,
        writable: true,
        value: (...args) => {
          fired.push(name);
          return undefined;
        },
      });
    } catch {
      // Non-writable globals are not ours to instrument; leaving them alone
      // means they run for real, which the error log will show if it matters.
    }
  }
  return fired;
}

/** Elements the markup wires directly, named while they are still in place. */
function handlerElements(document) {
  const found = [];
  for (const el of document.querySelectorAll("*")) {
    for (const event of EVENTS) {
      if (el.hasAttribute(`on${event}`)) found.push({ el, event, element: describe(el) });
    }
    // Where this is heading: one attribute naming an action, dispatched by a
    // single delegated listener. Both forms are read so the snapshot survives
    // the change from one to the other.
    if (el.hasAttribute("data-action")) {
      found.push({
        el,
        event: el.getAttribute("data-event") || "click",
        element: describe(el),
      });
    }
  }
  return found;
}

async function wiringFor(name) {
  // Collected rather than replaced, and the harness calls this twice: once when
  // parsing finishes, which is the only moment the markup is all still there,
  // and once after the module scripts have run, which is when anything they
  // add exists. Neither on its own covers the page.
  const elements = [];
  const seen = new Set();
  const collect = (doc) => {
    for (const found of handlerElements(doc)) {
      if (seen.has(found.el)) continue;
      seen.add(found.el);
      elements.push(found);
    }
  };

  const { window, document, errors, restore } = await loadPage(name, {
    onDomReady: collect,
  });
  const globals = pageGlobals(window);
  const fired = instrument(window, globals);
  // Everything after this point was provoked by this file, not by loading the
  // page, so the two are kept apart.
  const loadErrors = errors.splice(0, errors.length);

  const handlers = [];
  for (const { el, event, element } of elements) {
    fired.length = 0;
    const errorMark = errors.length;
    // Pages replace sections with rendered HTML, so by now some of these are
    // detached. Put them back for the duration of the click: a listener
    // delegated from the document can only see an element that is in it.
    const replaced = !el.isConnected;
    if (replaced) document.body.appendChild(el);
    el.dispatchEvent(
      new window.Event(event, { bubbles: true, cancelable: true, composed: true })
    );
    if (replaced) el.remove();

    const entry = { element, event, fires: [...new Set(fired)].sort() };
    // Worth recording: this element is not what the user actually clicks by
    // the time the page has loaded, so its handler is the template's, not the
    // rendered one.
    if (replaced) entry.rerendered = true;
    // Listeners the page attached itself are not instrumented — they are
    // anonymous — so they run for real and can fail on data that is not there.
    const raised = errors.slice(errorMark);
    if (raised.length) entry.raises = raised;
    handlers.push(entry);
  }

  handlers.sort((a, b) =>
    `${a.element} ${a.event}`.localeCompare(`${b.element} ${b.event}`)
  );
  restore();
  window.close();
  return { errors: loadErrors, globals, handlers };
}

const pages =
  process.argv.slice(2).length > 0
    ? process.argv.slice(2)
    : readdirSync(FRONTEND).filter((name) => name.endsWith(".html")).sort();

const result = {};
if (pages.length === 1) {
  result[pages[0]] = await wiringFor(pages[0]);
} else {
  // One process per page. Node caches modules by URL, and a relative import
  // inside a module does not carry the parent's query string, so a second page
  // in the same process would be handed the first page's already-evaluated
  // copy of js/actions.js — registry, listeners, closed document and all.
  for (const page of pages) {
    const child = spawnSync(process.execPath, [fileURLToPath(import.meta.url), page], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "inherit"],
      maxBuffer: 64 * 1024 * 1024,
    });
    if (child.status !== 0) {
      process.stderr.write(`${page}: the harness exited ${child.status}\n`);
      process.exit(child.status ?? 1);
    }
    Object.assign(result, JSON.parse(child.stdout));
  }
}
process.stdout.write(JSON.stringify(result, null, 2) + "\n");
