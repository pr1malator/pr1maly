/* Markup names an action; JavaScript supplies it.
 *
 *   <button data-action="openUploadModal">
 *   <button data-action="setTrendTimescale" data-args="[20]">
 *   <select data-action="loadMap" data-event="change">
 *   <button data-action="closeSyncModal openSettingsModal" data-args='[[], ["storage"]]'>
 *
 * This replaces inline onclick="". The difference that matters is not style:
 * an inline handler is a piece of JavaScript evaluated against global scope, so
 * every function a page's markup calls has to be a global, and nothing but a
 * browser can tell you whether it still is. An action is a string. Markup can
 * be checked against the set of registered names without running anything, and
 * the functions themselves can live in modules and be imported like code.
 *
 * The rules:
 *
 *   data-action   one name, or several separated by spaces, run in order.
 *   data-args     arguments. For one action, its argument list. For several,
 *                 a list of argument lists, positionally matched.
 *   data-event    the event to listen for. Defaults to "click".
 *
 * Every action is called as fn(...args, event, element).
 *
 * Listeners are attached to the elements themselves, not delegated from the
 * document. That is deliberate: delegation would run the handler after the
 * event had already passed every ancestor, so stopPropagation could no longer
 * do what it did as an inline handler. Elements rendered later are picked up
 * by the observer below, so nothing has to remember to call wireActions.
 */

const ACTIONS = Object.create(null);
const WIRED = new WeakSet();

/** Two things markup is genuinely entitled to say about an event itself. */
ACTIONS.preventDefault = (event) => event.preventDefault();
ACTIONS.stopPropagation = (event) => event.stopPropagation();

export function registerActions(map) {
  for (const [name, fn] of Object.entries(map)) {
    if (typeof fn !== "function") {
      console.error(`registerActions: ${name} is not a function`);
      continue;
    }
    ACTIONS[name] = fn;
  }
}

function actionNames(el) {
  return (el.dataset.action || "").trim().split(/\s+/).filter(Boolean);
}

/**
 * data-args for one action is that action's arguments; for several it is a
 * list of argument lists. Returns one list per action either way.
 */
function actionArguments(el, count) {
  const raw = el.dataset.args;
  if (!raw) return Array.from({ length: count }, () => []);

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    console.error(`data-args is not JSON: ${raw}`, err);
    return Array.from({ length: count }, () => []);
  }
  if (!Array.isArray(parsed)) parsed = [parsed];
  if (count === 1) return [parsed];

  return Array.from({ length: count }, (_, i) => {
    const set = parsed[i];
    return Array.isArray(set) ? set : set === undefined ? [] : [set];
  });
}

function runActions(el, event) {
  const names = actionNames(el);
  const args = actionArguments(el, names.length);
  names.forEach((name, i) => {
    const fn = ACTIONS[name];
    if (!fn) {
      console.error(`no action registered as "${name}"`, el);
      return;
    }
    fn(...args[i], event, el);
  });
}

/** Attach listeners to every [data-action] under (and including) root. */
export function wireActions(root = document) {
  const elements = [...root.querySelectorAll("[data-action]")];
  if (root.matches?.("[data-action]")) elements.push(root);
  for (const el of elements) {
    if (WIRED.has(el)) continue;
    WIRED.add(el);
    el.addEventListener(el.dataset.event || "click", (event) => runActions(el, event));
  }
}

/**
 * Escapes a set of arguments for a data-args attribute, for the code that
 * builds HTML as strings:
 *
 *   `<button data-action="removeAccount" data-args="${actionArgs(id)}">`
 */
export function actionArgs(...values) {
  return JSON.stringify(values)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
    .replaceAll("<", "&lt;");
}

// Module scripts are deferred, so this normally runs after parsing but before
// DOMContentLoaded. The readyState check is for the case where something
// imports this later than that.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => wireActions());
} else {
  wireActions();
}

// Most of this UI is rendered from JavaScript into innerHTML, and a button
// that was never wired is indistinguishable from one that does nothing.
// Watching for them costs less than remembering at ~30 render sites.
new MutationObserver((records) => {
  for (const record of records) {
    for (const node of record.addedNodes) {
      if (node.nodeType === 1) wireActions(node);
    }
  }
}).observe(document.documentElement, { childList: true, subtree: true });

// The one global this file keeps, and it is a test seam: tests/test_frontend_dom.py
// swaps each entry for a recorder to find out what a page's buttons actually
// call. A module's exports cannot be reached from outside the module graph.
window.ACTIONS = ACTIONS;
