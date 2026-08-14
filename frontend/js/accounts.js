/* pr1maly — accounts, friends, and the upload/settings modals.
 *
 * Sixteen functions that were byte-for-byte identical in breakdown.html,
 * match-breakdown.html and performance.html.
 *
 * Deliberately NOT moved here, because the three pages genuinely disagree
 * about what they do: openSettingsModal, switchSettingsTab,
 * populateBulkAccountSelector, startBulkUpload, loadAIConfig, renderAISettings
 * and saveAISettings. Merging those means choosing which page's behaviour is
 * right, which is a decision rather than a refactor.
 */

import { actionArgs, registerActions } from './actions.js';
import { API } from './api.js';
import { esc } from './escape.js';
import { hooks } from './hooks.js';

/* The two states of an upload-mode button. Identical in all three pages that
   had a copy, so they live with the code that reads them. */
const UPLOAD_MODE_ON = 'bg-primary text-on-primary-fixed';
const UPLOAD_MODE_OFF = 'bg-surface-container-highest text-on-surface-variant hover:bg-white/10';

export async function loadAccounts() {
  try {
    const res = await fetch(API + '/accounts');
    const accounts = await res.json();
    const sel = document.getElementById('upload-steam-id');
    sel.innerHTML = '';
    accounts.forEach(a => {
      const opt = document.createElement('option');
      opt.value = a.steam_id;
      opt.textContent = a.name;
      sel.appendChild(opt);
    });
    renderAccountsList(accounts);
  } catch (err) { console.error('Failed to load accounts:', err); }
}

function renderAccountsList(accounts) {
  const list = document.getElementById('accounts-list');
  if (!list) return;
  if (!accounts.length) { list.innerHTML = '<p class="text-on-surface-variant text-xs">No accounts configured.</p>'; return; }
  list.innerHTML = accounts.map(a => `
    <div class="flex items-center gap-3 p-3 rounded-lg bg-surface-container-highest border border-transparent group">
      <div class="flex-1 min-w-0">
        <div class="font-bold text-sm text-on-surface">${esc(a.name)}</div>
        <div class="text-[10px] text-on-surface-variant">${esc(a.display_name || '')} &middot; ${esc(a.rank || 'Unranked')}</div>
        <div class="text-[9px] text-on-surface-variant/50 font-mono">${esc(a.steam_id)}</div>
      </div>
      <div class="flex gap-1">
        <button data-action="removeAccount" data-args="${actionArgs(a.steam_id)}" class="text-[10px] px-2 py-1 rounded bg-error/20 text-error hover:bg-error/30 transition-colors" title="Remove">Remove</button>
      </div>
    </div>`).join('');
}

async function addAccount() {
  const name = document.getElementById('new-acct-name').value.trim();
  const steamId = document.getElementById('new-acct-steamid').value.trim();
  const displayName = document.getElementById('new-acct-display').value.trim();
  const rank = document.getElementById('new-acct-rank').value.trim();
  if (!name || !steamId) return;
  const statusEl = document.getElementById('acct-status');
  try {
    const res = await fetch(API + '/accounts', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ name, steam_id: steamId, display_name: displayName, rank })
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail); }
    statusEl.textContent = 'Account added!'; statusEl.className = 'text-xs text-secondary mt-2';
    document.getElementById('new-acct-name').value = '';
    document.getElementById('new-acct-steamid').value = '';
    document.getElementById('new-acct-display').value = '';
    document.getElementById('new-acct-rank').value = '';
    loadAccounts();
  } catch (err) { statusEl.textContent = err.message; statusEl.className = 'text-xs text-error mt-2'; }
}

async function removeAccount(steamId) {
  if (!confirm('Remove this account?')) return;
  await fetch(API + '/accounts/' + steamId, { method: 'DELETE' });
  loadAccounts();
}

export async function loadFriends() {
  try {
    const res = await fetch(API + '/friends');
    const friends = await res.json();
    renderFriendsList(friends);
  } catch (err) { console.error('Failed to load friends:', err); }
}

function renderFriendsList(friends) {
  const list = document.getElementById('friends-list');
  if (!list) return;
  if (!friends.length) { list.innerHTML = '<p class="text-on-surface-variant text-xs">No friends added yet.</p>'; return; }
  list.innerHTML = friends.map(f => `
    <div class="flex items-center gap-3 p-3 rounded-lg bg-surface-container-highest border border-transparent group">
      <div class="flex-1 min-w-0">
        <div class="font-bold text-sm text-accent">${esc(f.name || 'Unnamed')}</div>
        <div class="text-[9px] text-on-surface-variant/50 font-mono">${esc(f.steam_id)}</div>
      </div>
      <div class="flex gap-1">
        <button data-action="removeFriend" data-args="${actionArgs(f.steam_id)}" class="text-[10px] px-2 py-1 rounded bg-error/20 text-error hover:bg-error/30 transition-colors" title="Remove">Remove</button>
      </div>
    </div>`).join('');
}

async function addFriend() {
  const steamId = document.getElementById('new-friend-steamid').value.trim();
  const name = document.getElementById('new-friend-name').value.trim();
  if (!steamId) return;
  const statusEl = document.getElementById('friend-status');
  try {
    const res = await fetch(API + '/friends', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ steam_id: steamId, name })
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail); }
    statusEl.textContent = 'Friend added!'; statusEl.className = 'text-xs text-accent mt-2';
    document.getElementById('new-friend-steamid').value = '';
    document.getElementById('new-friend-name').value = '';
    loadFriends();
  } catch (err) { statusEl.textContent = err.message; statusEl.className = 'text-xs text-error mt-2'; }
}

async function removeFriend(steamId) {
  if (!confirm('Remove this friend?')) return;
  await fetch(API + '/friends/' + steamId, { method: 'DELETE' });
  loadFriends();
}

export function closeSettingsModal() {
  document.getElementById('settings-modal').classList.add('hidden');
  document.getElementById('settings-modal').classList.remove('flex');
}

function openUploadModal(mode) {
  document.getElementById('upload-modal').classList.remove('hidden');
  document.getElementById('upload-modal').classList.add('flex');
  switchUploadMode(mode || 'single');
}

export function closeUploadModal() { document.getElementById('upload-modal').classList.add('hidden'); document.getElementById('upload-modal').classList.remove('flex'); }

function switchUploadMode(mode) {
  const bulk = mode === 'bulk';
  document.getElementById('upload-pane-single').classList.toggle('hidden', bulk);
  document.getElementById('upload-pane-bulk').classList.toggle('hidden', !bulk);
  ['single', 'bulk'].forEach(m => {
    const btn = document.getElementById('upload-mode-' + m);
    const base = 'px-4 py-2 rounded-full text-[10px] font-bold uppercase tracking-widest transition-all ';
    btn.className = base + (m === mode ? UPLOAD_MODE_ON : UPLOAD_MODE_OFF);
  });
  // The bulk pane keeps state between openings (queued files, progress), so
  // reset it as it comes into view rather than on modal open.
  if (bulk) resetBulkUpload();
}

function openBulkUploadModal() { openUploadModal('bulk'); }

function closeBulkUploadModal() { closeUploadModal(); }

function resetBulkUpload() {
  hooks.populateBulkAccountSelector();
  document.getElementById('bulk-dem-files').value = '';
  document.getElementById('bulk-info-files').value = '';
  document.getElementById('bulk-file-preview').classList.add('hidden');
  document.getElementById('bulk-progress').classList.add('hidden');
  document.getElementById('bulk-upload-btn').disabled = false;
  document.getElementById('bulk-upload-btn').textContent = 'Process All Demos';
}

/* The confirmation tickbox beside the factory reset button. This was an
   expression inside an onchange attribute on three pages — the only piece of
   logic standing between a stray click and every match being deleted, written
   somewhere nothing could test it. */
function syncFactoryResetButton(event, checkbox) {
  document.getElementById('reset-btn').disabled = !checkbox.checked;
}

async function doFactoryReset() {
  const status = document.getElementById('reset-status');
  const btn = document.getElementById('reset-btn');
  btn.disabled = true; btn.textContent = 'RESETTING...';
  status.classList.add('hidden');
  try {
    const res = await fetch(API + '/factory-reset', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'ok') {
      status.textContent = 'Reset complete. Reloading...';
      status.className = 'text-xs text-center text-secondary';
      status.classList.remove('hidden');
      setTimeout(() => window.location.reload(), 1000);
    } else {
      status.textContent = 'Partial reset: ' + (data.errors || []).join(', ');
      status.className = 'text-xs text-center text-error';
      status.classList.remove('hidden');
      btn.disabled = false; btn.textContent = 'Reset Everything';
    }
  } catch (err) {
    status.textContent = err.message;
    status.className = 'text-xs text-center text-error';
    status.classList.remove('hidden');
    btn.disabled = false; btn.textContent = 'Reset Everything';
  }
}


/* What this file offers the markup. See js/actions.js. */
registerActions({
  addAccount,
  addFriend,
  closeSettingsModal,
  closeUploadModal,
  doFactoryReset,
  openUploadModal,
  removeAccount,
  removeFriend,
  switchUploadMode,
  syncFactoryResetButton,
});
