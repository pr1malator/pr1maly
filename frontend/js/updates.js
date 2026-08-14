/* Update panel — shared by every page that hosts the settings modal.
 *
 * Same shape as reanalyze.js: this file owns both the markup and the
 * behaviour, and a page opts in with a tab button, an empty
 * <div id="settings-updates-panel">, and 'updates' in its tab list.
 *
 * Opening the tab does not check anything. It shows the version you are
 * running and whatever the last check found; asking is a button, or a setting
 * you turn on. The wording says what is contacted and what is sent, because a
 * self-hosted app that quietly phones home is exactly the thing this one is
 * meant not to be.
 */

import { registerActions } from './actions.js';
import { API } from './api.js';

const UPDATES_PANEL_HTML = `
  <h3 class="font-headline text-[11px] font-bold uppercase tracking-widest text-on-surface-variant mb-1 flex items-center gap-2">
    <span class="material-symbols-outlined text-sm">update</span> Updates
  </h3>

  <div class="rounded-xl bg-surface-container-highest/60 p-3 mb-3">
    <div class="flex items-baseline justify-between gap-3">
      <span class="text-[9px] uppercase tracking-widest text-on-surface-variant">You are running</span>
      <span id="update-current" class="font-mono text-sm text-on-surface">&mdash;</span>
    </div>
    <div id="update-result" class="text-[10px] text-on-surface-variant mt-2 leading-relaxed"></div>
    <div id="update-command" class="hidden mt-2"></div>
  </div>

  <label class="flex items-start gap-2 cursor-pointer mb-3">
    <input type="checkbox" id="update-enabled" class="mt-0.5 w-4 h-4 rounded bg-surface-container-highest border-white/20 accent-primary"
           data-action="saveUpdateConfig" data-event="change">
    <span class="text-[10px] text-on-surface-variant leading-relaxed">
      Check once a day for a new version.
      <span class="block text-on-surface-variant/70">
        Requests a small file from the project's site. Nothing about this install is sent &mdash;
        no version, no identifier, no usage &mdash; and the server keeps no log of the request,
        so nobody counts installs. Nothing is ever installed for you either.
        Off by default; the button below works either way.
      </span>
    </span>
  </label>

  <button id="update-check-btn" data-action="checkForUpdates"
          class="px-3 py-1.5 rounded-lg bg-primary/20 text-primary text-[10px] font-bold uppercase tracking-widest hover:bg-primary/30 transition-colors">
    Check now
  </button>
`;

function _panel() {
  const panel = document.getElementById('settings-updates-panel');
  if (!panel) return null;
  if (!panel.dataset.built) {
    panel.innerHTML = UPDATES_PANEL_HTML;
    panel.dataset.built = '1';
  }
  return panel;
}

function _inDocker() {
  // Best effort, and it only decides which command to print.
  return window.location.port === '8000' && window.location.hostname === 'localhost';
}

function _render(status) {
  const current = document.getElementById('update-current');
  const result = document.getElementById('update-result');
  const command = document.getElementById('update-command');
  const toggle = document.getElementById('update-enabled');
  if (!current) return;

  current.textContent = 'v' + (status.current || '?');
  if (toggle) toggle.checked = !!status.enabled;
  command.classList.add('hidden');
  command.innerHTML = '';

  if (status.error) {
    result.innerHTML = '<span class="text-on-surface-variant/70">Could not reach the project site. '
      + 'That is what being offline looks like, and nothing else is affected.</span>';
    return;
  }

  if (!status.latest) {
    result.textContent = status.enabled
      ? 'No check has run yet.'
      : 'Automatic checks are off. Press Check now to look once.';
    return;
  }

  if (status.update_available) {
    result.innerHTML =
      '<span class="text-primary font-bold">v' + status.latest + ' is available</span>'
      + (status.released ? ' <span class="text-on-surface-variant/70">(' + status.released + ')</span>' : '')
      + (status.notes ? ' &middot; <a class="underline hover:text-primary" target="_blank" rel="noreferrer noopener" href="' + status.notes + '">release notes</a>' : '');
    command.classList.remove('hidden');
    command.innerHTML =
      '<div class="text-[9px] uppercase tracking-widest text-on-surface-variant mb-1">To update</div>'
      + '<code class="block text-[10px] font-mono bg-surface-container p-2 rounded-lg text-on-surface">'
      + (_inDocker() ? 'docker compose pull &amp;&amp; docker compose up -d' : 'git pull &amp;&amp; pip install -r requirements.txt')
      + '</code>';
  } else {
    result.innerHTML = '<span class="text-secondary">This is the current version.</span>'
      + (status.checked_at ? ' <span class="text-on-surface-variant/70">Checked '
        + new Date(status.checked_at).toLocaleDateString() + '.</span>' : '');
  }
}

export async function loadUpdateStatus() {
  if (!_panel()) return;
  try {
    _render(await (await fetch(API + '/updates')).json());
  } catch (err) {
    console.error('Could not read update status:', err);
  }
}

async function checkForUpdates() {
  const btn = document.getElementById('update-check-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Checking...'; }
  try {
    _render(await (await fetch(API + '/updates/check', { method: 'POST' })).json());
  } catch (err) {
    console.error('Update check failed:', err);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Check now'; }
  }
}

async function saveUpdateConfig(event, checkbox) {
  try {
    const res = await fetch(API + '/updates/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: checkbox.checked }),
    });
    _render(await res.json());
  } catch (err) {
    console.error('Could not save the update setting:', err);
  }
}


/* What this file offers the markup. See js/actions.js. */
registerActions({
  checkForUpdates,
  saveUpdateConfig,
});
