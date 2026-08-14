/* "What is this number?" — answered from the catalogue, not from the markup.
 *
 * Any element can carry data-explain="aim_stats.preaim". This puts a small
 * marker beside it, and clicking the marker shows what the figure measures,
 * how it is derived, and where any tier shown against it came from.
 *
 * The wording lives in src/domain/metrics/catalogue.py and arrives over
 * /api/metrics, so the interface cannot drift from the backend or from
 * METRICS.md — all three read the same table.
 *
 * The tier line is the reason this exists. The interface grades people ELITE
 * or NEEDS WORK against thresholds that were, for the most part, chosen by
 * hand rather than measured from a population. Saying so is not a disclaimer;
 * it is the difference between a number someone can act on and one they have
 * to take on faith.
 */

import { registerActions } from './actions.js';
import { API } from './api.js';

let _catalogue = null;
let _popover = null;

async function _load() {
  if (_catalogue) return _catalogue;
  const res = await fetch(API + '/metrics');
  const body = await res.json();
  _catalogue = {
    fields: Object.fromEntries((body.fields || []).map(f => [f.key, f])),
    provenance: body.provenance || {},
    statistics: body.statistics || {},
  };
  return _catalogue;
}

function _escape(text) {
  const holder = document.createElement('div');
  holder.textContent = text == null ? '' : text;
  return holder.innerHTML;
}

const TIER_LABEL = {
  measured: 'Measured',
  published: 'Published elsewhere',
  heuristic: 'Hand-set',
};

function _card(field, catalogue) {
  const tier = field.tiers
    ? `<div class="mt-2 pt-2 border-t border-white/10">
         <div class="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant mb-1">
           Grading &middot; ${_escape(TIER_LABEL[field.tiers] || field.tiers)}
         </div>
         <div class="text-[10px] text-on-surface-variant leading-relaxed">${_escape(field.tiers_meaning)}</div>
       </div>`
    : '';

  const note = field.note
    ? `<div class="mt-2 text-[10px] text-on-surface-variant/80 leading-relaxed italic">${_escape(field.note)}</div>`
    : '';

  return `
    <div class="font-headline text-[11px] font-bold uppercase tracking-widest text-on-surface mb-2">
      ${_escape(field.label)}${field.unit ? ' <span class="text-on-surface-variant font-normal">(' + _escape(field.unit) + ')</span>' : ''}
    </div>
    <div class="text-[11px] text-on-surface leading-relaxed">${_escape(field.measures)}</div>
    <div class="mt-2">
      <div class="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant mb-1">How it is worked out</div>
      <div class="text-[10px] text-on-surface-variant leading-relaxed">${_escape(field.derived)}</div>
    </div>
    ${note}
    ${tier}
    <div class="mt-2 pt-2 border-t border-white/10 text-[9px] text-on-surface-variant/70 leading-relaxed">
      ${_escape(catalogue.statistics.median || '')}
      ${_escape(catalogue.statistics.confidence || '')}
    </div>
  `;
}

function _close() {
  if (_popover) { _popover.remove(); _popover = null; }
}

async function explain(key, event, marker) {
  event?.stopPropagation();
  const catalogue = await _load();
  const field = catalogue.fields[key];
  _close();
  if (!field) {
    console.error(`nothing in the catalogue describes ${key}`);
    return;
  }

  _popover = document.createElement('div');
  _popover.className = 'explain-popover';
  _popover.style.cssText =
    'position:fixed;z-index:400;max-width:320px;padding:12px;border-radius:12px;' +
    'background:rgb(var(--c-surface-container-high));' +
    'border:1px solid rgb(var(--c-outline-variant));' +
    'box-shadow:0 8px 32px rgba(0,0,0,0.35);';
  _popover.innerHTML = _card(field, catalogue);
  document.body.appendChild(_popover);

  const anchor = (marker || event?.target)?.getBoundingClientRect?.();
  if (anchor) {
    const width = _popover.offsetWidth;
    const height = _popover.offsetHeight;
    _popover.style.left = Math.max(8, Math.min(anchor.left, window.innerWidth - width - 8)) + 'px';
    _popover.style.top = (anchor.bottom + height + 8 < window.innerHeight
      ? anchor.bottom + 6
      : Math.max(8, anchor.top - height - 6)) + 'px';
  }

  setTimeout(() => document.addEventListener('click', _close, { once: true }), 0);
}

/* Put a marker on everything that names a figure. Runs on load and again
   whenever the page renders more of itself, because most of this interface is
   built from JavaScript after the data arrives. */
export function markExplainable(root = document) {
  const targets = [...root.querySelectorAll('[data-explain]')];
  // The root itself counts. Sections here are rendered by assigning innerHTML,
  // so the element naming a figure is often the added node rather than one of
  // its descendants — and a marker that never appears is indistinguishable
  // from a figure nobody thought to explain.
  if (root.matches?.('[data-explain]')) targets.push(root);

  for (const el of targets) {
    if (el.dataset.explainMarked) continue;
    el.dataset.explainMarked = '1';

    const marker = document.createElement('button');
    marker.type = 'button';
    marker.className = 'explain-marker material-symbols-outlined';
    marker.textContent = 'help';
    marker.title = 'What is this?';
    marker.style.cssText =
      'font-size:13px;line-height:1;vertical-align:middle;margin-left:4px;' +
      'opacity:0.45;cursor:pointer;background:none;border:0;padding:0;' +
      'color:inherit;';
    marker.addEventListener('mouseenter', () => { marker.style.opacity = '1'; });
    marker.addEventListener('mouseleave', () => { marker.style.opacity = '0.45'; });
    marker.dataset.action = 'explain';
    marker.dataset.args = JSON.stringify([el.dataset.explain]);
    el.appendChild(marker);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => markExplainable());
} else {
  markExplainable();
}

// Same reasoning as js/actions.js: this interface renders itself from data, so
// waiting to be told would mean remembering at every render site.
new MutationObserver((records) => {
  for (const record of records) {
    for (const node of record.addedNodes) {
      if (node.nodeType === 1) markExplainable(node);
    }
  }
}).observe(document.documentElement, { childList: true, subtree: true });


/* What this file offers the markup. See js/actions.js. */
registerActions({
  explain,
});
