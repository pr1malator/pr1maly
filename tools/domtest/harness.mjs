/**
 * Loads a frontend page in jsdom, with just enough of a browser around it.
 *
 * There are no browser tests, so every check on the frontend has so far been
 * static: does this name exist, does this file parse. That cannot answer the
 * question that actually matters — when someone clicks this button, does
 * anything happen. This can: the page runs its own scripts, and a click can be
 * dispatched at an element and observed.
 *
 * Three things are faked, and only three:
 *   - the network, because the app is not running
 *   - <canvas>, because jsdom has no 2D context and the charts want one
 *   - Tailwind, because it is 400 KB of styling that no wiring depends on
 */

import { parse } from "acorn";
import { JSDOM, ResourceLoader, VirtualConsole } from "jsdom";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
/** Everything a bare jsdom window has, so the page's own additions stand out. */
const BROWSER_BASELINE = new Set(Object.keys(new JSDOM("", { pretendToBeVisual: true }).window));
export const FRONTEND = resolve(HERE, "..", "..", "frontend");
export const ORIGIN = "http://localhost:8000";

/** Serve http://localhost:8000/frontend/... from disk instead of the network. */
class LocalFiles extends ResourceLoader {
  fetch(url, options) {
    const path = new URL(url).pathname.replace(/^\/frontend\//, "");
    // Tailwind is styling. Running it costs seconds per page and its own CDN
    // build prints a production warning at us for the privilege.
    if (path.startsWith("vendor/")) return Promise.resolve(Buffer.from(""));
    try {
      return Promise.resolve(readFileSync(join(FRONTEND, path.split("?")[0])));
    } catch {
      return super.fetch(url, options);
    }
  }
}

/**
 * What the API would answer. Shapes only — the values are deliberately empty,
 * because a page that renders nothing still has to wire up its buttons.
 */
// Real responses, when a test needs the page to render what the backend would
// actually send. DOMTEST_API_FIXTURES names a JSON file of pathname -> body,
// which is how the catalogue check proves the wording on screen came from
// src/domain/metrics/catalogue.py rather than from a copy in the frontend.
const OVERRIDES = process.env.DOMTEST_API_FIXTURES
  ? JSON.parse(readFileSync(process.env.DOMTEST_API_FIXTURES, "utf8"))
  : {};

function apiResponse(pathname) {
  if (pathname in OVERRIDES) return OVERRIDES[pathname];
  const table = {
    "/api/accounts": [],
    "/api/friends": [],
    "/api/matches": [],
    "/api/config": { steam_id: "", api_base: "" },
    "/api/tags": { tags: [] },
    "/api/onboarding": { completed: true },
    "/api/ai/config": { active_provider: "openai", providers: {} },
    "/api/analyzer/version": { version: 20, stale: 0, metrics: [] },
    "/api/metrics": { metrics: [] },
    "/api/steam/status": { running: false, logged_in: false },
    "/api/steam/auto-sync": { enabled: false },
    "/api/storage/config": { retention: "keep" },
    "/api/sync/config": { folder: "" },
  };
  if (pathname in table) return table[pathname];
  // Anything unlisted: an object is the safer default, since most handlers
  // read a property off the result and only a few iterate it.
  return {};
}

/** Top-level const/let/class names in the classic scripts the page loads. */
function lexicalGlobals(dom) {
  const names = new Set();
  for (const script of dom.window.document.querySelectorAll("script[src]")) {
    if (script.type === "module") continue;
    const src = script.getAttribute("src").split("?")[0];
    if (src.startsWith("vendor/")) continue;
    let source;
    try {
      source = readFileSync(join(FRONTEND, src), "utf8");
    } catch {
      continue;
    }
    let ast;
    try {
      ast = parse(source, { ecmaVersion: 2023, sourceType: "script" });
    } catch {
      continue;
    }
    for (const node of ast.body) {
      if (node.type === "VariableDeclaration" && node.kind !== "var") {
        for (const declarator of node.declarations) {
          if (declarator.id.type === "Identifier") names.add(declarator.id.name);
        }
      }
      if (node.type === "ClassDeclaration" && node.id) names.add(node.id.name);
    }
  }
  return names;
}

/**
 * Run the page's <script type="module"> files.
 *
 * jsdom does not implement module scripts at all — it parses the tag and skips
 * it. So Node runs them instead: its own loader resolves the import graph, and
 * the modules see jsdom's window and document because they are installed as
 * globals first. What that does not reproduce is module *scheduling*; by this
 * point the document is parsed and load has fired, which is later than a
 * browser would run them, so anything a module does at import time happens
 * against a finished document either way.
 *
 * The query string matters: Node caches modules by URL, and without it the
 * second page would get the first page's already-evaluated instance, still
 * holding a reference to a document that has been closed.
 */
async function runModuleScripts(dom, page, record) {
  const modules = [...dom.window.document.querySelectorAll('script[type="module"][src]')];
  if (!modules.length) return () => {};

  // Node defines some of these itself, occasionally as getter-only, so each is
  // installed defensively and put back afterwards.
  const saved = new Map();
  const install = (key, value) => {
    if (value === undefined) return;
    const existing = Object.getOwnPropertyDescriptor(globalThis, key);
    saved.set(key, existing);
    try {
      Object.defineProperty(globalThis, key, {
        value, configurable: true, writable: true, enumerable: false,
      });
    } catch {
      saved.delete(key);
    }
  };

  // Deliberately not the timers: jsdom's window.setTimeout delegates to the
  // global one, so installing it over the global makes it call itself.
  for (const key of ["window", "document", "navigator", "location", "customElements",
                     "Element", "HTMLElement", "Node", "Event", "CustomEvent",
                     "MutationObserver", "Image", "FormData", "URLSearchParams",
                     "fetch", "getComputedStyle", "localStorage", "history",
                     "requestAnimationFrame", "cancelAnimationFrame"]) {
    install(key, dom.window[key]);
  }

  // And whatever the page's classic scripts put on window — theme.js and
  // charts.js are loaded that way on purpose, and a module referring to
  // `drawTrendChart` resolves it through the global scope chain in a browser.
  // Here the module runs in Node's realm, which has no such chain, so the
  // names have to be copied across.
  for (const key of Object.keys(dom.window)) {
    if (!BROWSER_BASELINE.has(key)) install(key, dom.window[key]);
  }

  // `const` and `let` at the top of a classic script are not window
  // properties: they live in the global lexical scope, which a browser shares
  // with modules and which nothing here can enumerate. charts.js keeps
  // TREND_METRICS and the trend state there, and breakdown.js both reads and
  // writes them, so each one gets a bridge into jsdom's realm.
  for (const name of lexicalGlobals(dom)) {
    if (saved.has(name)) continue;
    saved.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, {
      configurable: true,
      get: () => dom.window.eval(name),
      set: (value) => {
        dom.window.__bridge = value;
        dom.window.eval(`${name} = window.__bridge;`);
      },
    });
  }

  for (const script of modules) {
    const relative = script.getAttribute("src").split("?")[0];
    const url = pathToFileURL(join(FRONTEND, relative)).href + `?page=${encodeURIComponent(page)}`;
    try {
      await import(url);
    } catch (err) {
      record(`module ${relative}: ${err.message}`);
    }
  }
  // Module scripts run before DOMContentLoaded in a browser. jsdom already
  // fired it, so anything listening for it has to be told by hand.
  dom.window.document.dispatchEvent(new dom.window.Event("DOMContentLoaded", { bubbles: true }));
  await new Promise((done) => setTimeout(done, 150));

  // The globals stay installed until the caller is finished with the page:
  // the code under test keeps running after this returns, and a handler
  // dispatched later still expects `document` to mean this document.
  return () => {
    for (const [key, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  };
}

/**
 * @param onDomReady  called with the document the moment parsing finishes,
 *   before the page's own load handlers run. That is the only point at which
 *   the markup is all present: pages replace whole sections with rendered
 *   HTML, and an element that has been replaced can no longer be found.
 */
export async function loadPage(name, { onError, onDomReady } = {}) {
  // A page can be asked for with a query string — match-breakdown.html shows a
  // list without ?id= and a match with it, which is two different pages.
  const [pageFile] = name.split("?");
  const file = join(FRONTEND, pageFile);
  const errors = [];
  const record = (message) => {
    errors.push(message);
    if (onError) onError(message);
  };

  // The page's own console output goes to stderr, so stdout stays parseable
  // for whatever is consuming the snapshot.
  const virtualConsole = new VirtualConsole();
  for (const level of ["log", "info", "warn", "error", "debug"]) {
    virtualConsole.on(level, (...args) => process.stderr.write(`[page] ${args.join(" ")}\n`));
  }
  virtualConsole.on("jsdomError", (err) => {
    // Script errors already arrive through the window 'error' listener below.
    if (!/^Uncaught/.test(err.message)) record(`jsdom: ${err.message}`);
  });

  const dom = new JSDOM(readFileSync(file, "utf8"), {
    runScripts: "dangerously",
    resources: new LocalFiles(),
    virtualConsole,
    url: `${ORIGIN}/frontend/${name}`,  // query included on purpose
    pretendToBeVisual: true,
    beforeParse(window) {
      window.fetch = async (input, init = {}) => {
        const url = new URL(typeof input === "string" ? input : input.url, ORIGIN);
        const body = apiResponse(url.pathname);
        return {
          ok: true,
          status: 200,
          headers: { get: () => "application/json" },
          json: async () => body,
          text: async () => JSON.stringify(body),
          blob: async () => ({}),
        };
      };
      // The charts ask for a context and draw into it. Returning null makes
      // them skip; returning a stub makes them run and fail on the first
      // method, which is noisier and tells us nothing extra.
      window.HTMLCanvasElement.prototype.getContext = () => null;
      window.scrollTo = () => {};
      // Not implemented by jsdom, and pages call it after rendering.
      window.Element.prototype.scrollIntoView = () => {};
      // Tailwind is skipped above, but the pages configure it on load.
      window.tailwind = { config: {} };
      // Registered before the page has had a chance to register its own, so
      // this runs first and sees the markup untouched.
      if (onDomReady) {
        window.document.addEventListener("DOMContentLoaded", () =>
          onDomReady(window.document, window)
        );
      }
      window.addEventListener("error", (event) => record(`error: ${event.message}`));
      window.addEventListener("unhandledrejection", (event) =>
        record(`rejection: ${event.reason}`)
      );
    },
  });

  await new Promise((done) => dom.window.addEventListener("load", done, { once: true }));
  const restore = await runModuleScripts(dom, name, record);
  // Let the page's own load handlers settle: they await fetches, and the
  // listeners they register afterwards are the point of this whole exercise.
  await new Promise((done) => setTimeout(done, 250));

  return { dom, window: dom.window, document: dom.window.document, errors, restore };
}

/**
 * A stable name for an element, so the same button can be recognised across a
 * refactor. Structure survives an attribute change; an id survives more.
 */
export function describe(el) {
  if (el.id) return `#${el.id}`;
  const parts = [];
  for (let node = el; node && node.tagName && node.tagName !== "BODY"; node = node.parentElement) {
    if (node.id) {
      parts.unshift(`#${node.id}`);
      break;
    }
    const siblings = [...(node.parentElement?.children ?? [])].filter(
      (sibling) => sibling.tagName === node.tagName
    );
    const index = siblings.indexOf(node) + 1;
    parts.unshift(
      siblings.length > 1
        ? `${node.tagName.toLowerCase()}:nth-of-type(${index})`
        : node.tagName.toLowerCase()
    );
  }
  return parts.join(" > ");
}
