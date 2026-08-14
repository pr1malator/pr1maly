/* pr1maly — Steam fetcher panel.
 *
 * Manual Fetch, Sync Folder, Storage retention and Auto-Sync: the settings
 * surface shared by the career breakdown and the single-match page.
 *
 * This was 885 identical lines inlined in both of those pages.
 */

import { registerActions, actionArgs } from './actions.js';
import { API } from './api.js';
import { hooks } from './hooks.js';

// ═══════════════════════════════════════════════════════════════════
// Fetch from Steam — drives /api/steam/*
// Kept separate from the Sync Folder logic on purpose: this downloads
// demos from Valve, Sync Folder imports what is already on disk.
// ═══════════════════════════════════════════════════════════════════
const STEAM_API = API;
let _steamPoll = null;

function steamEsc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function openSteamModal() {
  const m = document.getElementById('steam-modal');
  m.classList.remove('hidden'); m.classList.add('flex');
  loadSteamStatus();
  startAutoSyncPolling();
}

function closeSteamModal() {
  const m = document.getElementById('steam-modal');
  m.classList.add('hidden'); m.classList.remove('flex');
  if (_steamPoll) { clearInterval(_steamPoll); _steamPoll = null; }
  if (!isAutoSyncModalOpen()) stopAutoSyncPolling();
}

document.getElementById('steam-modal').addEventListener('click', e => {
  if (e.target === e.currentTarget) closeSteamModal();
});

export async function loadSteamStatus() {
  try {
    const res = await fetch(STEAM_API + '/steam/status');
    if (!res.ok) throw new Error('Could not read fetcher status');
    renderSteamStatus(await res.json());
  } catch (err) {
    document.getElementById('steam-unavailable').classList.remove('hidden');
    document.getElementById('steam-unavailable-text').textContent = err.message;
  }
}

function renderSteamStatus(s) {
  // Availability banner — Node and its packages live outside the Python app.
  const banner = document.getElementById('steam-unavailable');
  const bannerText = document.getElementById('steam-unavailable-text');
  if (!s.fetcher_present) {
    banner.classList.remove('hidden');
    bannerText.textContent = 'The fetcher/ folder is missing. This feature is not part of the public release.';
  } else if (!s.node_installed) {
    banner.classList.remove('hidden');
    bannerText.textContent = 'Node.js was not found. The Steam fetcher needs Node 18 or newer.';
  } else if (!s.deps_installed) {
    banner.classList.remove('hidden');
    bannerText.textContent = 'Fetcher dependencies are not installed. Run: cd fetcher && npm install';
  } else {
    banner.classList.add('hidden');
  }

  // Per-account setup state.
  const list = document.getElementById('steam-accounts');
  list.innerHTML = s.accounts.map(a => {
    const badge = (ok, yes, no) => ok
      ? '<span class="text-secondary">' + yes + '</span>'
      : '<span class="text-on-surface-variant/60">' + no + '</span>';
    const counts = a.total
      ? '<span class="text-[9px] font-mono text-on-surface-variant">' +
        a.downloaded + ' got / ' + (a.pending + a.failed) + ' due / ' + a.expired + ' gone</span>'
      : '<span class="text-[9px] text-on-surface-variant/60">no matches known</span>';
    // Tracking is setup: it decides whether an account's history is recorded at
    // all. Whether its demos get downloaded is a property of a fetch run, so
    // that toggle lives in the Download step rather than here.
    const toggle = (stage, on, label) =>
      '<label class="flex items-center gap-1 cursor-pointer normal-case tracking-normal">' +
      '<input type="checkbox" class="accent-secondary"' + (on ? ' checked' : '') +
      ' data-event="change" data-action="setSteamToggle" data-args="' + actionArgs(a.name, stage) + '"/>' +
      label + '</label>';

    return '<div class="px-3 py-2 rounded-lg bg-surface-container-highest/60">' +
      '<div class="flex items-center justify-between gap-2 mb-1">' +
      '<span class="text-xs font-bold">' + steamEsc(a.name) + '</span>' +
      '<div class="flex items-center gap-3 text-[9px] font-bold uppercase tracking-widest">' +
      (a.authenticated
        ? '<span class="text-secondary">signed in</span>'
        : '<button data-action="startSteamAuth" data-args="' + actionArgs(a.name) + '" class="px-2 py-1 rounded-md bg-primary/20 text-primary hover:bg-primary/30 transition-colors uppercase tracking-widest">sign in</button>') +
      badge(a.configured, 'codes set', 'no codes') + '</div></div>' +
      '<div class="flex items-center justify-between gap-3 text-[9px] text-on-surface-variant">' +
      '<div class="flex items-center gap-3">' +
      toggle('walk', a.walk_enabled !== false, 'track') +
      '</div>' + counts + '</div></div>';
  }).join('');

  renderSteamApiKeyStatus(s);

  const sel = document.getElementById('steam-code-account');
  const previous = sel.value;
  sel.innerHTML = s.accounts.map(a =>
    '<option value="' + steamEsc(a.name) + '">' + steamEsc(a.name) +
    (a.configured ? ' — configured' : '') + '</option>'
  ).join('');
  if (previous) sel.value = previous;

  // Download step.
  const pendingText = document.getElementById('steam-pending-text');
  const downloadBtn = document.getElementById('steam-download-btn');
  const running = s.job && s.job.running;

  if (s.pending_total > 0) {
    pendingText.innerHTML = '<span class="text-primary font-bold">' + s.pending_total +
      '</span> match(es) waiting to be downloaded.';
  } else {
    pendingText.textContent = 'Nothing outstanding. Run step 1 to look for new matches.';
  }

  renderSteamDownloadAccounts(s);
  downloadBtn.disabled = running || !s.available || s.pending_total === 0;
  downloadBtn.dataset.pending = s.pending_total;
  updateDownloadButton();
  document.getElementById('steam-check-btn').disabled = running || !s.available;

  if (s.job && (s.job.running || (s.job.lines && s.job.lines.length))) renderSteamJob(s.job);
  if (running && !_steamPoll) startSteamPolling();
  renderAutoSync(s.auto_sync);
  renderSteamSetupGate(s);
}

// "API key stored." on its own is not much use with several accounts: the key
// works for all of them, but it is issued by one, and when it stops working you
// need to know whose profile to regenerate it from.
function renderSteamApiKeyStatus(s) {
  const status = document.getElementById('steam-api-key-status');
  const sel = document.getElementById('steam-api-key-account');

  if (sel) {
    const previous = sel.value;
    sel.innerHTML = '<option value="">(not recorded)</option>' + (s.accounts || []).map(a =>
      '<option value="' + steamEsc(a.name) + '">' + steamEsc(a.name) + '</option>'
    ).join('');
    sel.value = previous || s.api_key_account || '';
    // The environment overrides the stored key, so the field would be lying.
    sel.disabled = s.api_key_source === 'environment';
  }

  if (!s.api_key_set) {
    status.textContent = 'No API key yet.';
    status.className = 'text-[9px] text-on-surface-variant mb-4';
    return;
  }

  const tail = s.api_key_tail ? ' ····' + s.api_key_tail : '';
  if (s.api_key_source === 'environment') {
    status.innerHTML = '<span class="text-secondary">API key from STEAM_API_KEY' + steamEsc(tail)
      + '.</span> The environment overrides anything stored here.';
  } else if (s.api_key_account) {
    status.innerHTML = '<span class="text-secondary">API key stored' + steamEsc(tail)
      + '</span>, issued by <span class="text-primary font-bold">' + steamEsc(s.api_key_account) + '</span>.';
  } else {
    status.innerHTML = '<span class="text-secondary">API key stored' + steamEsc(tail)
      + '.</span> Pick the account that issued it so you know where to renew it.';
  }
  status.className = 'text-[9px] text-on-surface-variant mb-4';
}

// Which accounts the next download covers. This is deliberately here and not in
// Settings: it changes what the button below is about to do, and the pending
// total reacts to it immediately.
function renderSteamDownloadAccounts(s) {
  const wrap = document.getElementById('steam-download-accounts-wrap');
  const list = document.getElementById('steam-download-accounts');
  const note = document.getElementById('steam-download-excluded');
  if (!wrap || !list) return;

  // Only accounts the ledger actually knows about can be downloaded.
  const known = (s.accounts || []).filter(a => a.configured || a.total);
  wrap.classList.toggle('hidden', known.length === 0);
  if (!known.length) return;

  const busy = !!(s.job && s.job.running);
  list.innerHTML = known.map(a => {
    const on = a.download_enabled !== false;
    const due = a.outstanding;
    const detail = due
      ? '<span class="text-primary font-bold">' + due + ' due</span>'
      : '<span class="text-on-surface-variant/50">nothing due</span>';
    return '<label class="flex items-center justify-between gap-3 px-3 py-1.5 rounded-lg bg-surface-container-highest/60 cursor-pointer' +
      (on ? '' : ' opacity-50') + '">' +
      '<span class="flex items-center gap-2">' +
      '<input type="checkbox" class="accent-primary"' + (on ? ' checked' : '') + (busy ? ' disabled' : '') +
      ' data-event="change" data-action="setSteamToggle" data-args="' + actionArgs(a.name, 'download') + '"/>' +
      '<span class="text-[10px] font-bold">' + steamEsc(a.name) + '</span></span>' +
      '<span class="text-[9px] font-mono">' + detail + '</span></label>';
  }).join('');

  // Say what is being left out, so a zero total is never a mystery.
  const off = known.filter(a => a.download_enabled === false);
  const skipped = off.reduce((n, a) => n + a.outstanding, 0);
  note.textContent = off.length
    ? off.length + ' account(s) excluded' + (skipped ? ', holding ' + skipped + ' match(es) back' : '') + '.'
    : '';
}

/* Called from markup as an action, so it reads the tickbox it was fired from
   rather than being handed its value. */
async function setSteamToggle(name, stage, event, checkbox) {
  const enabled = checkbox.checked;
  const body = stage === 'walk' ? { walk_enabled: enabled } : { download_enabled: enabled };
  try {
    const res = await fetch(STEAM_API + '/steam/accounts/' + encodeURIComponent(name) + '/toggles', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!res.ok) throw new Error((await res.json()).detail);
  } catch (err) {
    document.getElementById('steam-codes-status').textContent = err.message;
    document.getElementById('steam-codes-status').className = 'text-[9px] text-error';
  }
  loadSteamStatus();
}

async function saveSteamApiKey() {
  const input = document.getElementById('steam-api-key');
  const status = document.getElementById('steam-api-key-status');
  try {
    const res = await fetch(STEAM_API + '/steam/api-key', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: input.value.trim(),
        account: (document.getElementById('steam-api-key-account') || {}).value || null
      })
    });
    if (!res.ok) throw new Error((await res.json()).detail);
    input.value = '';
    // loadSteamStatus repaints the status line with the issuing account, so
    // there is nothing useful to say here that it would not immediately replace.
    loadSteamStatus();
  } catch (err) {
    status.textContent = err.message;
    status.className = 'text-[9px] text-error mb-4';
  }
}

async function saveSteamCodes() {
  const name = document.getElementById('steam-code-account').value;
  const status = document.getElementById('steam-codes-status');
  try {
    const res = await fetch(STEAM_API + '/steam/accounts/' + encodeURIComponent(name), {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        auth_code: document.getElementById('steam-auth-code').value.trim(),
        share_code: document.getElementById('steam-share-code').value.trim()
      })
    });
    if (!res.ok) throw new Error((await res.json()).detail);
    document.getElementById('steam-auth-code').value = '';
    document.getElementById('steam-share-code').value = '';
    status.textContent = 'Saved for ' + name + '. Now run a fetch.';
    status.className = 'text-[9px] text-secondary';
    loadSteamStatus();
  } catch (err) {
    status.textContent = err.message;
    status.className = 'text-[9px] text-error';
  }
}

// Keeps the button honest about what it will actually do.
function updateDownloadButton() {
  const btn = document.getElementById('steam-download-btn');
  const input = document.getElementById('steam-download-limit');
  if (!btn || !input) return;

  const pending = parseInt(btn.dataset.pending || '0', 10);
  if (!pending) { btn.textContent = 'Download Demos'; return; }

  const limit = parseInt(input.value, 10);
  const count = Number.isFinite(limit) && limit > 0 ? Math.min(limit, pending) : pending;
  btn.textContent = count < pending
    ? 'Download newest ' + count + ' of ' + pending
    : 'Download ' + pending + ' Demo' + (pending > 1 ? 's' : '');
}

async function runSteamJob(kind) {
  const endpoint = kind === 'check' ? '/steam/check' : '/steam/download';
  document.getElementById('steam-check-btn').disabled = true;
  document.getElementById('steam-download-btn').disabled = true;
  document.getElementById('steam-done-hint').classList.add('hidden');

  try {
    const init = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' };
    if (kind === 'download') {
      const limit = parseInt(document.getElementById('steam-download-limit').value, 10);
      if (Number.isFinite(limit) && limit > 0) init.body = JSON.stringify({ limit });
    }

    const res = await fetch(STEAM_API + endpoint, init);
    if (!res.ok) throw new Error((await res.json()).detail);
    renderSteamJob(await res.json());
    startSteamPolling();
  } catch (err) {
    document.getElementById('steam-output-wrap').classList.remove('hidden');
    document.getElementById('steam-log').textContent = err.message;
    document.getElementById('steam-job-state').textContent = 'failed';
    loadSteamStatus();
  }
}

function startSteamPolling() {
  if (_steamPoll) clearInterval(_steamPoll);
  _steamPoll = setInterval(async () => {
    try {
      const job = await (await fetch(STEAM_API + '/steam/job')).json();
      renderSteamJob(job);
      if (!job.running) {
        clearInterval(_steamPoll); _steamPoll = null;
        // Downloads land in the demo folder — remind the user to import them.
        if (job.type === 'download' && job.exit_code === 0 && !job.auto) {
          document.getElementById('steam-done-hint').classList.remove('hidden');
        }
        loadSteamStatus();
      }
    } catch (e) { clearInterval(_steamPoll); _steamPoll = null; }
  }, 1200);
}

let _steamQrDismissed = false;

async function startSteamAuth(name) {
  const panel = document.getElementById('steam-qr-panel');
  _steamQrDismissed = false;
  panel.classList.remove('hidden');
  document.getElementById('steam-qr-title').textContent = 'Signing in — ' + name;
  document.getElementById('steam-qr-image').innerHTML = '';
  document.getElementById('steam-qr-state').textContent = 'Asking Steam for a QR code...';
  try {
    const res = await fetch(STEAM_API + '/steam/auth/' + encodeURIComponent(name), { method: 'POST' });
    if (!res.ok) throw new Error((await res.json()).detail);
    renderSteamJob(await res.json());
    startSteamPolling();
  } catch (err) {
    document.getElementById('steam-qr-state').textContent = err.message;
  }
}

// Dismissing the code cancels the attempt. The challenge event stays in the
// job's history, so hiding alone would let the next poll re-open the panel —
// and the sign-in would sit pending until Steam's own timeout.
async function hideSteamQr() {
  _steamQrDismissed = true;
  document.getElementById('steam-qr-panel').classList.add('hidden');
  try {
    await fetch(STEAM_API + '/steam/job/cancel', { method: 'POST' });
  } catch (e) { /* nothing running */ }
  loadSteamStatus();
}

// Drives the QR panel from the structured events auth-qr.js emits.
function applySteamAuthEvents(job) {
  if (_steamQrDismissed) return;
  const panel = document.getElementById('steam-qr-panel');
  const image = document.getElementById('steam-qr-image');
  const state = document.getElementById('steam-qr-state');

  for (const ev of (job.events || [])) {
    if (ev.event === 'challenge') {
      panel.classList.remove('hidden');
      if (!image.innerHTML) image.innerHTML = ev.svg;
      state.textContent = 'Scan this with the Steam mobile app, then approve it.';
    } else if (ev.event === 'scanned') {
      state.textContent = 'Scanned. Now approve it in the app...';
    } else if (ev.event === 'authenticated') {
      image.innerHTML = '';
      state.innerHTML = '<span class="text-secondary font-bold">Signed in'
        + (ev.steamId ? ' as ' + steamEsc(ev.steamId) : '') + '.</span>'
        + (ev.warning ? '<br>' + steamEsc(ev.warning) : '');
    } else if (ev.event === 'error') {
      image.innerHTML = '';
      state.innerHTML = '<span class="text-error">' + steamEsc(ev.message) + '</span>';
    }
  }
}

async function cancelSteamJob() {
  const btn = document.getElementById('steam-job-cancel');
  btn.disabled = true;
  try {
    await fetch(STEAM_API + '/steam/job/cancel', { method: 'POST' });
  } catch (e) { /* already finished */ }
}

function formatElapsed(seconds) {
  return seconds < 60 ? seconds + 's' : Math.floor(seconds / 60) + 'm ' + (seconds % 60) + 's';
}

function renderSteamJob(job) {
  document.getElementById('steam-output-wrap').classList.remove('hidden');

  // Cancel is offered for any running job, not just QR sign-in — a stuck
  // download should not need the container restarting to stop.
  const cancelBtn = document.getElementById('steam-job-cancel');
  cancelBtn.classList.toggle('hidden', !job.running);
  if (job.running) cancelBtn.disabled = false;

  // Ticks with the poll, then freezes on the final value when the job ends.
  const elapsed = document.getElementById('steam-job-elapsed');
  if (job.started_at) {
    const end = job.finished_at ? new Date(job.finished_at) : new Date();
    const secs = Math.max(0, Math.round((end - new Date(job.started_at)) / 1000));
    elapsed.textContent = formatElapsed(secs);
  } else {
    elapsed.textContent = '';
  }
  document.getElementById('steam-job-label').textContent =
    job.type === 'check' ? 'Checking for new matches'
      : job.type === 'auth' ? 'Signing in to Steam'
      : 'Downloading demos';
  if (job.type === 'auth') applySteamAuthEvents(job);

  const state = document.getElementById('steam-job-state');
  if (job.running) {
    state.textContent = 'running...';
    state.className = 'text-[10px] font-mono font-bold text-primary';
  } else if (job.exit_code === 0) {
    state.textContent = 'done';
    state.className = 'text-[10px] font-mono font-bold text-secondary';
  } else if (job.cancelled) {
    state.textContent = 'cancelled';
    state.className = 'text-[10px] font-mono font-bold text-on-surface-variant';
  } else if (job.exit_code != null) {
    state.textContent = 'exit ' + job.exit_code;
    state.className = 'text-[10px] font-mono font-bold text-error';
  }

  const log = document.getElementById('steam-log');
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  log.textContent = (job.lines || []).join('\n');
  if (atBottom) log.scrollTop = log.scrollHeight;
}


// ─── Auto-Sync modal ────────────────────────────────────────────────────
// Auto-Sync has its own entry point because it is a state, not an action:
// you switch it on once and it keeps working. The sidebar dot reports that
// state without the modal being open at all.
function isAutoSyncModalOpen() {
  const m = document.getElementById('auto-sync-modal');
  return m && !m.classList.contains('hidden');
}

function openAutoSyncModal() {
  const m = document.getElementById('auto-sync-modal');
  m.classList.remove('hidden'); m.classList.add('flex');
  loadSteamStatus();
  startAutoSyncPolling();
  refreshPresence();
}

function closeAutoSyncModal() {
  const m = document.getElementById('auto-sync-modal');
  m.classList.add('hidden'); m.classList.remove('flex');
  const steam = document.getElementById('steam-modal');
  if (!steam || steam.classList.contains('hidden')) stopAutoSyncPolling();
}

document.getElementById('auto-sync-modal').addEventListener('click', e => {
  if (e.target === e.currentTarget) closeAutoSyncModal();
});

// Setup moved to Settings, so both action modals have to say when it is
// missing — otherwise their buttons just sit there doing nothing.
function renderSteamSetupGate(s) {
  const ready = s.available && s.api_key_set && (s.accounts || []).some(a => a.configured && a.authenticated);
  let why = '';
  if (!s.fetcher_present) why = 'The fetcher is not installed, so demos cannot be downloaded from Valve.';
  else if (!s.available) why = 'The Steam fetcher is not ready yet.';
  else if (!s.api_key_set) why = 'No Steam Web API key yet. Add one in Settings to look up your matches.';
  else if (!(s.accounts || []).some(a => a.authenticated)) why = 'No account is signed in to Steam yet.';
  else why = 'No account has its match-sharing codes set yet.';

  [['steam-needs-setup', 'steam-needs-setup-text'],
   ['auto-sync-needs-setup', 'auto-sync-needs-setup-text']].forEach(([wrapId, textId]) => {
    const wrap = document.getElementById(wrapId);
    if (!wrap) return;
    wrap.classList.toggle('hidden', ready);
    if (!ready) document.getElementById(textId).textContent = why + ' Setup lives in Settings → Steam.';
  });
  return ready;
}

// The dot is the whole point of giving Auto-Sync a sidebar entry: whether it
// is running should be readable without opening anything.
function renderAutoSyncDot(state) {
  const dot = document.getElementById('auto-sync-dot');
  if (!dot) return;
  const on = state && state.enabled;
  dot.classList.toggle('hidden', !on);
  if (!on) return;
  // Amber while it is held back, so "paused" does not read as "working".
  const held = state.phase === 'paused' || state.phase === 'error' || state.phase === 'blocked';
  dot.classList.toggle('bg-tertiary', !held);
  dot.classList.toggle('bg-error', held);
  dot.title = 'Auto-Sync: ' + (state.detail || state.phase);
}

// Runs regardless of any modal being open, so the dot is live from page load.
let _autoSyncDotPoll = null;
function startAutoSyncDotPoll() {
  if (_autoSyncDotPoll) return;
  const tick = async () => {
    try {
      const s = await (await fetch(STEAM_API + '/steam/auto-sync')).json();
      renderAutoSyncDot(s);
    } catch (e) { /* backend down — leave the dot as it was */ }
  };
  tick();
  _autoSyncDotPoll = setInterval(tick, 20000);
}

// ─── Auto-Sync ──────────────────────────────────────────────────────────
// The manual buttons above run one burst and stop. Auto-Sync is a switch: the
// backend loops on its own, so this side only reflects state and edits
// settings. It keeps its own slow poll because the panel has to stay live
// between jobs, when the 1.2s job poll is not running.
let _autoSyncPoll = null;
let _autoSyncState = null;
let _autoSyncSaving = false;

const AUTO_PHASES = {
  off:         { label: 'off',          cls: 'text-on-surface-variant' },
  waiting:     { label: 'waiting',      cls: 'text-secondary' },
  checking:    { label: 'checking',     cls: 'text-secondary' },
  downloading: { label: 'downloading',  cls: 'text-primary' },
  importing:   { label: 'analysing',    cls: 'text-primary' },
  paused:      { label: 'paused',       cls: 'text-tertiary' },
  blocked:     { label: 'waiting',      cls: 'text-on-surface-variant' },
  error:       { label: 'problem',      cls: 'text-error' }
};

function renderAutoSync(state) {
  if (!state) return;
  _autoSyncState = state;
  const cfg = state.config || {};
  const on = !!state.enabled;

  // Toggle.
  const toggle = document.getElementById('auto-sync-toggle');
  const knob = document.getElementById('auto-sync-knob');
  toggle.setAttribute('aria-checked', on ? 'true' : 'false');
  toggle.classList.toggle('bg-primary', on);
  toggle.classList.toggle('bg-surface-container-highest', !on);
  knob.classList.toggle('translate-x-5', on);
  knob.classList.toggle('bg-on-primary-container', on);
  knob.classList.toggle('bg-on-surface-variant', !on);
  const hint = document.getElementById('auto-sync-toggle-hint');
  if (hint) {
    hint.textContent = on
      ? 'On — ' + ((AUTO_PHASES[state.phase] || AUTO_PHASES.off).label) + (state.detail ? ' · ' + state.detail : '')
      : 'Off';
    hint.className = 'text-[9px] ' + (on ? 'text-tertiary' : 'text-on-surface-variant');
  }

  // Don't fight the user mid-edit.
  if (!_autoSyncSaving && document.activeElement !== document.getElementById('auto-sync-interval')) {
    document.getElementById('auto-sync-interval').value = cfg.interval_minutes;
  }
  document.getElementById('auto-sync-pause-playing').checked = cfg.pause_when_playing !== false;

  // Live state — only meaningful while it is on, or just after it stopped.
  const wrap = document.getElementById('auto-sync-state');
  wrap.classList.toggle('hidden', !on && state.phase === 'off');
  const phase = AUTO_PHASES[state.phase] || AUTO_PHASES.off;
  const phaseEl = document.getElementById('auto-sync-phase');
  phaseEl.textContent = phase.label;
  phaseEl.className = 'text-[10px] font-bold uppercase tracking-widest ' + phase.cls;
  document.getElementById('auto-sync-detail').textContent = state.detail || '';
  document.getElementById('auto-sync-countdown').textContent = autoCountdown(state.next_action_at);

  const t = state.totals || {};
  const totals = document.getElementById('auto-sync-totals');
  totals.textContent = (t.downloaded || t.imported || t.failed)
    ? t.downloaded + ' downloaded  ·  ' + t.imported + ' analysed' + (t.failed ? '  ·  ' + t.failed + ' failed' : '')
    : '';

  // Demos that would not parse. They are set aside rather than retried, so say
  // so — otherwise they just quietly never appear.
  const skipped = Object.keys(state.skipped || {}).filter(k => state.skipped[k] >= 3);
  const skipWrap = document.getElementById('auto-sync-skipped');
  skipWrap.classList.toggle('hidden', skipped.length === 0);
  if (skipped.length) {
    skipWrap.querySelector('p').textContent =
      skipped.length + ' demo(s) could not be analysed and were set aside so the rest could carry on: '
      + skipped.join(', ');
  }

  // CS2 detection.
  const presence = state.presence;
  const pEl = document.getElementById('auto-sync-presence');
  if (cfg.pause_when_playing === false) {
    pEl.textContent = '';
  } else if (!presence) {
    pEl.textContent = 'Not checked yet.';
  } else if (presence.playing === true) {
    pEl.innerHTML = '<span class="text-tertiary">In CS2 now: ' + steamEsc(presence.in_game.join(', ')) + '</span>';
  } else if (presence.playing === false) {
    pEl.innerHTML = '<span class="text-secondary">Detected — nobody is in CS2.</span>';
  } else {
    pEl.innerHTML = '<span class="text-error/80">Cannot detect: ' + steamEsc(presence.detail) + '</span>';
  }

  renderAutoSyncDot(state);

  const actWrap = document.getElementById('auto-sync-activity-wrap');
  const act = state.activity || [];
  actWrap.classList.toggle('hidden', act.length === 0);
  document.getElementById('auto-sync-activity').innerHTML = act.slice(-8).reverse().map(a =>
    '<p class="text-[9px] font-mono text-on-surface-variant/70">'
    + '<span class="text-on-surface-variant/40">' + steamEsc(autoClock(a.at)) + '</span> '
    + steamEsc(a.text) + '</p>'
  ).join('');
}

// The job slot is shared with Manual Fetch, so its output is mirrored here
// rather than making the user open the other modal to see what is happening.
function renderAutoSyncJob(job) {
  const wrap = document.getElementById('auto-sync-output-wrap');
  if (!wrap) return;
  const useful = job && job.auto && (job.running || (job.lines || []).length);
  wrap.classList.toggle('hidden', !useful);
  if (!useful) return;

  const state = document.getElementById('auto-sync-job-state');
  state.textContent = job.running ? (job.type || '') + '...' : (job.exit_code === 0 ? 'done' : 'stopped');
  state.className = 'text-[10px] font-mono font-bold ' + (job.running ? 'text-primary' : 'text-on-surface-variant');

  const log = document.getElementById('auto-sync-log');
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  log.textContent = (job.lines || []).join('\n');
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function autoClock(iso) {
  const d = new Date(iso);
  return isNaN(d) ? '' : d.toTimeString().slice(0, 5);
}

function autoCountdown(iso) {
  if (!iso) return '';
  const secs = Math.round((new Date(iso) - new Date()) / 1000);
  if (!isFinite(secs) || secs <= 0) return '';
  return 'next in ' + (secs < 60 ? secs + 's' : Math.round(secs / 60) + 'm');
}

async function putAutoSync(body) {
  const res = await fetch(STEAM_API + '/steam/auto-sync', {
    method: 'PUT', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();
}

async function toggleAutoSync() {
  const btn = document.getElementById('auto-sync-toggle');
  const wantOn = btn.getAttribute('aria-checked') !== 'true';
  btn.disabled = true;
  try {
    renderAutoSync(await putAutoSync({ enabled: wantOn }));
    if (wantOn) startAutoSyncPolling();
  } catch (err) {
    const detail = document.getElementById('auto-sync-detail');
    document.getElementById('auto-sync-state').classList.remove('hidden');
    detail.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}

async function saveAutoSyncSettings() {
  const interval = parseInt(document.getElementById('auto-sync-interval').value, 10);
  _autoSyncSaving = true;
  try {
    renderAutoSync(await putAutoSync({
      interval_minutes: Number.isFinite(interval) && interval >= 0 ? interval : 5,
      pause_when_playing: document.getElementById('auto-sync-pause-playing').checked
    }));
  } catch (err) {
    document.getElementById('auto-sync-detail').textContent = err.message;
  } finally {
    _autoSyncSaving = false;
  }
}

// Slow poll so the panel stays live between jobs. Every few seconds is plenty
// for something whose default cadence is one match every five minutes; the
// countdown is recomputed locally in between.
function startAutoSyncPolling() {
  if (_autoSyncPoll) return;
  _autoSyncPoll = setInterval(async () => {
    try {
      const s = await (await fetch(STEAM_API + '/steam/status')).json();
      renderAutoSync(s.auto_sync);
      renderAutoSyncJob(s.job);
      // Auto-sync drives the same job slot, so keep the Manual Fetch log in step.
      if (s.job && (s.job.running || (s.job.lines || []).length)) renderSteamJob(s.job);
      document.getElementById('steam-check-btn').disabled = (s.job && s.job.running) || !s.available;
    } catch (e) { /* backend restarting — the next tick retries */ }
  }, 3000);
}

function stopAutoSyncPolling() {
  if (_autoSyncPoll) { clearInterval(_autoSyncPoll); _autoSyncPoll = null; }
}

// Presence is a Steam Web API call, so it is refreshed on open rather than on
// every poll. The backend caches it and the loop refreshes it as it runs.
async function refreshPresence() {
  try {
    const p = await (await fetch(STEAM_API + '/steam/presence')).json();
    if (_autoSyncState) renderAutoSync({ ..._autoSyncState, presence: p });
  } catch (e) { /* leave the last known answer */ }
}

// ─── Storage ────────────────────────────────────────────────────────
// Demos are disposable once analysed; the retention window exists only so
// recent matches can be re-parsed after a metrics change.

function storageBytes(b) {
  return b >= 1073741824 ? (b / 1073741824).toFixed(2) + ' GB' : (b / 1048576).toFixed(0) + ' MB';
}

async function loadStorageStatus(preview) {
  const summary = document.getElementById('storage-summary');
  if (!summary) return;

  // With `preview`, the on-screen settings are sent as query parameters and the
  // server evaluates them without saving, so a retention number can be tuned
  // against real figures before committing to it.
  let query = '';
  if (preview) {
    const params = new URLSearchParams();
    const keep = parseInt(document.getElementById('storage-keep').value, 10);
    if (!isNaN(keep) && keep >= 0) params.set('keep_recent', keep);
    params.set('per_account', document.getElementById('storage-per-account').checked);
    params.set('fetched_only', document.getElementById('storage-fetched-only').checked);
    query = '?' + params.toString();
  }

  try {
    const res = await fetch(STEAM_API + '/storage/status' + query);
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || 'Could not read storage status');
    renderStorage(body);
  } catch (err) {
    summary.innerHTML = '<span class="text-[10px] text-error col-span-2">' + steamEsc(err.message) + '</span>';
  }
}

let _storagePreviewTimer = null;

// Retune as the user types rather than making them press Apply to find out.
function previewStorage() {
  clearTimeout(_storagePreviewTimer);
  _storagePreviewTimer = setTimeout(() => loadStorageStatus(true), 250);
}

function renderStorage(s) {
  const cell = (label, value, cls) =>
    '<div class="px-3 py-2 rounded-lg bg-surface-container-highest/60">' +
    '<div class="text-[9px] uppercase tracking-widest text-on-surface-variant">' + label + '</div>' +
    '<div class="text-[11px] font-mono font-bold ' + (cls || '') + '">' + value + '</div></div>';

  document.getElementById('storage-summary').innerHTML =
    cell('On disk', s.total_files + ' demos, ' + storageBytes(s.total_bytes)) +
    cell('Imported', s.imported_files + ' of ' + s.total_files) +
    cell('Kept', s.protected_files + ' demos, ' + storageBytes(s.protected_bytes)) +
    cell('Safe to delete', s.deletable_files + ', ' + storageBytes(s.deletable_bytes),
         s.deletable_files ? 'text-error' : '');

  // Never fight the user's typing: only reset the controls from saved settings.
  if (!s.preview) {
    document.getElementById('storage-keep').value = s.config.keep_recent;
    document.getElementById('storage-per-account').checked = s.config.per_account !== false;
    document.getElementById('storage-fetched-only').checked = !!s.config.fetched_only;
    document.getElementById('storage-auto').checked = !!s.config.auto_cleanup;
  } else {
    const note = document.getElementById('storage-result');
    note.textContent = 'Preview of unsaved settings — press Apply to keep them.';
    note.className = 'text-[9px] text-on-surface-variant mt-2 leading-relaxed';
  }

  // Deleting always uses the saved settings, so keep it disabled while the
  // figures on screen come from unsaved ones.
  const btn = document.getElementById('storage-delete-btn');
  btn.disabled = s.deletable_files === 0 || s.preview;
  btn.textContent = s.deletable_files === 0
    ? 'Nothing to delete'
    : (s.preview
        ? 'Apply these settings to delete ' + s.deletable_files + ' demos'
        : 'Delete ' + s.deletable_files + ' demos, free ' + storageBytes(s.deletable_bytes));
}

async function saveStorageConfig() {
  const result = document.getElementById('storage-result');
  try {
    const res = await fetch(STEAM_API + '/storage/config', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        keep_recent: parseInt(document.getElementById('storage-keep').value, 10),
        per_account: document.getElementById('storage-per-account').checked,
        fetched_only: document.getElementById('storage-fetched-only').checked,
        auto_cleanup: document.getElementById('storage-auto').checked
      })
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail);
    result.textContent = 'Settings saved.';
    result.className = 'text-[9px] text-secondary mt-2 leading-relaxed';
    loadStorageStatus();
  } catch (err) {
    result.textContent = err.message;
    result.className = 'text-[9px] text-error mt-2 leading-relaxed';
  }
}

async function runStorageCleanup() {
  const result = document.getElementById('storage-result');
  const btn = document.getElementById('storage-delete-btn');

  // Deleting is irreversible past Valve's ~30 day retention, so confirm against
  // the exact figures rather than showing a generic prompt.
  let preview;
  try {
    const res = await fetch(STEAM_API + '/storage/cleanup', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dry_run: true })
    });
    preview = await res.json();
    if (!res.ok) throw new Error(preview.detail);
  } catch (err) {
    result.textContent = err.message;
    result.className = 'text-[9px] text-error mt-2 leading-relaxed';
    return;
  }

  if (!preview.deleted_count) { loadStorageStatus(); return; }
  const ok = confirm(
    'Delete ' + preview.deleted_count + ' demo file(s), freeing ' +
    storageBytes(preview.freed_bytes) + '?\n\n' +
    'Their analysis stays in the database. The demo files themselves cannot be ' +
    're-downloaded once Valve drops them (about 30 days).'
  );
  if (!ok) return;

  btn.disabled = true;
  try {
    const res = await fetch(STEAM_API + '/storage/cleanup', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dry_run: false })
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail);
    result.innerHTML = 'Deleted ' + body.deleted_count + ' demo(s), freed ' +
      storageBytes(body.freed_bytes) + '.' +
      (body.errors.length ? '<br><span class="text-error">' + steamEsc(body.errors.join('; ')) + '</span>' : '');
    result.className = 'text-[9px] text-secondary mt-2 leading-relaxed';
  } catch (err) {
    result.textContent = err.message;
    result.className = 'text-[9px] text-error mt-2 leading-relaxed';
  }
  loadStorageStatus();
}

/* Storage is a Settings tab, so refresh its figures when that tab is shown.
   The page calls this from its own switchSettingsTab.

   It used to reach out and replace that function with a wrapper, which worked
   only because both were globals — and quietly did nothing at all if this file
   happened to load first, as it does on performance.html. */
export function onSettingsTabShown(tab) {
  if (tab === 'storage') loadStorageStatus();
}

// The fetcher ships with the app but needs Node and its own npm install, so it
// can be present as source and still not runnable. Hide the entry point
// entirely in that case rather than offering a button that leads to an error.
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const s = await (await fetch(STEAM_API + '/steam/status')).json();
    if (!s.fetcher_present) {
      document.querySelectorAll('[data-action~="openSteamModal"], [data-action~="openAutoSyncModal"], #settings-tab-steam')
        .forEach(b => b.classList.add('hidden'));
    } else {
      startAutoSyncDotPoll();
    }
  } catch (e) { /* backend unreachable — leave the buttons visible */ }
});


/* What this file offers the markup. See js/actions.js. */
registerActions({
  cancelSteamJob,
  closeAutoSyncModal,
  closeSteamModal,
  hideSteamQr,
  openAutoSyncModal,
  openSteamModal,
  previewStorage,
  runSteamJob,
  runStorageCleanup,
  saveAutoSyncSettings,
  saveSteamApiKey,
  saveSteamCodes,
  saveStorageConfig,
  setSteamToggle,
  startSteamAuth,
  toggleAutoSync,
  updateDownloadButton,
});
