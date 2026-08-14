import './theme-actions.js';
import { loadUpdateStatus } from './updates.js';
import { registerActions, actionArgs } from './actions.js';
import { API } from './api.js';
import { esc } from './escape.js';
import { hooks } from './hooks.js';
import { closeSettingsModal, closeUploadModal, loadAccounts, loadFriends } from './accounts.js';
import { loadSteamStatus, onSettingsTabShown } from './steam-panel.js';
import { loadReanalyzeList } from '../reanalyze.js';

// --- Utility ---

// --- Account Management ---

// --- Settings Modal ---
let _aiConfig = null;
let _aiProviders = null;
let _selectedProvider = null;

function openSettingsModal(tab) {
  document.getElementById('settings-modal').classList.remove('hidden');
  document.getElementById('settings-modal').classList.add('flex');
  switchSettingsTab(tab || 'accounts');
  loadAccounts();
  loadAIConfig();
}
document.getElementById('settings-modal').addEventListener('click', e => { if (e.target === e.currentTarget) closeSettingsModal(); });

function switchSettingsTab(tab) {
  const tabs = ['accounts', 'friends', 'ai', 'steam', 'storage', 'reanalyze', 'updates', 'reset'];
  tabs.forEach(t => {
    const btn = document.getElementById('settings-tab-' + t);
    const panel = document.getElementById('settings-' + t + '-panel');
    if (!btn || !panel) return;
    if (t === tab) {
      btn.className = btn.className.replace('bg-surface-container-highest text-on-surface-variant hover:bg-white/10', '').replace('bg-primary text-on-primary-fixed', '') + ' bg-primary text-on-primary-fixed';
      panel.classList.remove('hidden');
    } else {
      btn.className = btn.className.replace('bg-primary text-on-primary-fixed', '').replace('bg-surface-container-highest text-on-surface-variant hover:bg-white/10', '') + ' bg-surface-container-highest text-on-surface-variant hover:bg-white/10';
      panel.classList.add('hidden');
    }
  });
  if (tab === 'friends') loadFriends();
  if (tab === 'steam') loadSteamStatus();
  if (tab === 'reanalyze') loadReanalyzeList();
  if (tab === 'updates') loadUpdateStatus();
  onSettingsTabShown(tab);
}

// Re-analyze panel lives in reanalyze.js, shared with the other pages that
// host this modal.




// --- Friends Management ---




// --- AI Config ---
async function loadAIConfig() {
  try {
    const [configRes, providersRes] = await Promise.all([
      fetch(API + '/ai/config'),
      fetch(API + '/ai/providers')
    ]);
    _aiConfig = await configRes.json();
    _aiProviders = await providersRes.json();
    _selectedProvider = _aiConfig.active_provider || Object.keys(_aiProviders)[0];
    renderAISettings();
    renderAISelector();
  } catch (err) { console.error('Failed to load AI config:', err); }
}

// /api/ai/providers returns {key: {label, models, default_model}} and
// /api/ai/config masks keys as api_key_set / api_key_masked. This read the
// provider entry as a bare array of model names and the config as if it still
// carried the raw key: the model dropdown threw on .map of an object, which
// aborted the rest of the function, so the model list and the system
// instructions never rendered and a configured key always read "No key set".
function renderAISettings() {
  if (!_aiProviders || !_aiConfig) return;
  const tabsEl = document.getElementById('ai-provider-tabs');
  tabsEl.innerHTML = Object.entries(_aiProviders).map(([key, p]) => {
    const isActive = key === _selectedProvider;
    const hasKey = _aiConfig.providers?.[key]?.api_key_set;
    return `<button data-action="selectAIProvider" data-args="${actionArgs(key)}" class="px-3 py-1.5 rounded-full text-[9px] font-bold uppercase tracking-widest transition-all ${isActive ? 'bg-primary text-on-primary-fixed' : 'bg-surface-container-highest text-on-surface-variant hover:bg-white/10'}">${esc(p.label || key)}${hasKey ? ' ✓' : ''}</button>`;
  }).join('');

  const prov = _aiConfig.providers?.[_selectedProvider] || {};
  const status = document.getElementById('ai-key-status');
  // The key never comes back from the server, so the input starts empty and a
  // blank save leaves the stored key alone.
  document.getElementById('ai-key-input').value = '';
  if (prov.api_key_set) {
    status.textContent = 'Key set: ' + (prov.api_key_masked || '••••');
    status.className = 'text-[9px] text-secondary mt-1';
  } else {
    status.textContent = 'No key configured';
    status.className = 'text-[9px] text-on-surface-variant mt-1';
  }

  const modelSel = document.getElementById('ai-settings-model');
  const models = _aiProviders[_selectedProvider]?.models || [];
  const active = _aiConfig.active_model || prov.default_model
    || _aiProviders[_selectedProvider]?.default_model || models[0];
  modelSel.innerHTML = models.map(m =>
    `<option value="${esc(m)}" ${m === active ? 'selected' : ''}>${esc(m)}</option>`
  ).join('');
  document.getElementById('ai-system-instructions').value = _aiConfig.system_instructions || '';
}

// The provider/model pickers beside each AI Analyze button, listing only
// providers that actually have a key — picking one without a key would just
// buy a 400 from the endpoint. The map card and the career card each carry
// their own pair; only one of the two is on screen at a time.
const AI_SELECTORS = [
  { prov: 'ai-provider-select', model: 'ai-model-select' },
  { prov: 'ai-overall-provider-select', model: 'ai-overall-model-select' },
];

function renderAISelector() {
  for (const sel of AI_SELECTORS) renderAISelectorPair(sel.prov, sel.model);
}

function renderAISelectorPair(provId, modelId) {
  const provSel = document.getElementById(provId);
  const modelSel = document.getElementById(modelId);
  if (!provSel || !modelSel || !_aiProviders || !_aiConfig) return;

  const usable = Object.entries(_aiProviders)
    .filter(([key]) => _aiConfig.providers?.[key]?.api_key_set);
  if (!usable.length) {
    provSel.innerHTML = '<option value="">No API keys</option>';
    modelSel.innerHTML = '<option value="">—</option>';
    return;
  }
  provSel.innerHTML = usable.map(([key, p]) =>
    `<option value="${esc(key)}" ${key === _aiConfig.active_provider ? 'selected' : ''}>${esc(p.label || key)}</option>`
  ).join('');
  renderAIModelOptions(provId, modelId);
}

function renderAIModelOptions(provId, modelId) {
  const provSel = document.getElementById(provId || 'ai-provider-select');
  const modelSel = document.getElementById(modelId || 'ai-model-select');
  if (!provSel || !modelSel || !_aiProviders) return;
  const prov = provSel.value;
  const models = _aiProviders[prov]?.models || [];
  const active = (prov === _aiConfig?.active_provider && _aiConfig?.active_model)
    || _aiConfig?.providers?.[prov]?.default_model
    || _aiProviders[prov]?.default_model
    || models[0];
  modelSel.innerHTML = models.map(m =>
    `<option value="${esc(m)}" ${m === active ? 'selected' : ''}>${esc(m)}</option>`
  ).join('');
}

function openAISettings() { openSettingsModal('ai'); }

function selectAIProvider(provider) {
  _selectedProvider = provider;
  renderAISettings();
}

async function saveAISettings() {
  const statusEl = document.getElementById('ai-settings-status');
  try {
    const model = document.getElementById('ai-settings-model').value;
    const key = document.getElementById('ai-key-input').value.trim();
    // Send only what this panel edits. Echoing the whole config back would
    // return the masked provider entries the GET handed out, and the PUT
    // replies {status: ok} rather than the config — reading that back into
    // _aiConfig left the panel unable to render until a page reload.
    const body = {
      active_provider: _selectedProvider,
      active_model: model,
      system_instructions: document.getElementById('ai-system-instructions').value,
      providers: { [_selectedProvider]: { default_model: model } },
    };
    if (key) body.providers[_selectedProvider].api_key = key;

    const res = await fetch(API + '/ai/config', {
      method: 'PUT', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    if (!res.ok) throw new Error('Failed to save');
    await loadAIConfig();
    statusEl.textContent = 'Saved!'; statusEl.className = 'text-xs text-center mt-3 text-secondary';
    statusEl.classList.remove('hidden');
    setTimeout(() => statusEl.classList.add('hidden'), 2000);
  } catch (err) { statusEl.textContent = err.message; statusEl.className = 'text-xs text-center mt-3 text-error'; statusEl.classList.remove('hidden'); }
}

// --- Upload Modal (single + bulk in one) ---
document.getElementById('upload-modal').addEventListener('click', e => { if (e.target === e.currentTarget) closeUploadModal(); });



// Kept as its own function: the bulk pane is reset both when switching into it
// and by the "Bulk Upload" entry points elsewhere in the app.

// --- Auto-detect player from .info file ---
document.getElementById('upload-info-file').addEventListener('change', async function() {
  const detectStatus = document.getElementById('player-detect-status');
  const sel = document.getElementById('upload-steam-id');
  const dateInput = document.getElementById('upload-date');
  detectStatus.classList.add('hidden');
  if (!this.files.length) return;
  const fd = new FormData();
  fd.append('info_file', this.files[0]);
  try {
    const res = await fetch(API + '/matches/detect-player', { method: 'POST', body: fd });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Detection failed'); }
    const data = await res.json();
    if (data.match_date && !dateInput.value) dateInput.value = data.match_date;
    if (data.matched && data.matched.length > 0) {
      const matchedId = data.matched[0].steam_id;
      if (sel.querySelector('option[value="' + matchedId + '"]')) {
        sel.value = matchedId;
      }
      const names = data.matched.map(a => a.name).join(', ');
      detectStatus.textContent = 'Detected player: ' + names;
      detectStatus.className = 'text-xs mt-1 text-secondary';
      detectStatus.classList.remove('hidden');
    } else {
      detectStatus.innerHTML = 'No configured account found in this match. <button type="button" data-action="openSettingsModal" data-args=\'["accounts"]\' class="underline text-primary hover:text-primary-dim">Add account</button>';
      detectStatus.className = 'text-xs mt-1 text-error';
      detectStatus.classList.remove('hidden');
    }
  } catch (err) {
    detectStatus.textContent = err.message;
    detectStatus.className = 'text-xs mt-1 text-error';
    detectStatus.classList.remove('hidden');
  }
});

document.getElementById('upload-form').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = document.getElementById('upload-btn');
  const status = document.getElementById('upload-status');
  const fileInput = document.getElementById('upload-file');
  if (!fileInput.files.length) return;
  btn.disabled = true; btn.textContent = 'PROCESSING...';
  status.classList.remove('hidden','text-error','text-secondary'); status.classList.add('text-on-surface-variant'); status.textContent = 'Uploading and parsing demo...';
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  const infoInput = document.getElementById('upload-info-file');
  if (infoInput.files.length) fd.append('info_file', infoInput.files[0]);
  fd.append('steam_id', document.getElementById('upload-steam-id').value);
  fd.append('context_notes', document.getElementById('upload-notes').value);
  fd.append('tags', document.getElementById('upload-tags').value);
  const dateVal = document.getElementById('upload-date').value;
  if (dateVal) fd.append('match_date', dateVal);
  try {
    const res = await fetch(API + '/matches/upload', { method: 'POST', body: fd });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Upload failed'); }
    status.classList.remove('text-on-surface-variant'); status.classList.add('text-secondary'); status.textContent = 'Demo processed! Refreshing...';
    setTimeout(() => { closeUploadModal(); _overallTrends = null; _overallPerf = null; loadPerformance(); }, 800);
  } catch (err) {
    status.classList.remove('text-on-surface-variant'); status.classList.add('text-error'); status.textContent = err.message;
  } finally { btn.disabled = false; btn.textContent = 'Process Demo'; }
});

// --- Bulk upload — now the "bulk" mode of the upload modal above.
// These two survive as aliases so existing call sites keep working.

async function populateBulkAccountSelector() {
  const sel = document.getElementById('bulk-steam-id');
  try {
    const res = await fetch(API + '/accounts');
    const data = await res.json();
    sel.innerHTML = '';
    const accounts = Array.isArray(data) ? data : (data.accounts || []);
    accounts.forEach(a => {
      const opt = document.createElement('option');
      opt.value = a.steam_id; opt.textContent = a.name + ' (' + a.steam_id + ')';
      if (a.active) opt.selected = true;
      sel.appendChild(opt);
    });
  } catch(e) { sel.innerHTML = '<option value="">No accounts</option>'; }
}

document.getElementById('bulk-dem-files').addEventListener('change', function() {
  const preview = document.getElementById('bulk-file-preview');
  const list = document.getElementById('bulk-file-list');
  if (!this.files.length) { preview.classList.add('hidden'); return; }
  preview.classList.remove('hidden');
  list.innerHTML = '';
  for (const f of this.files) {
    const row = document.createElement('div');
    row.className = 'text-xs text-on-surface-variant flex items-center gap-2';
    row.innerHTML = '<span class="material-symbols-outlined text-[14px]">description</span>' + f.name + ' <span class="text-on-surface-variant/50">(' + (f.size / 1024 / 1024).toFixed(1) + ' MB)</span>';
    list.appendChild(row);
  }
});

async function startBulkUpload() {
  const demInput = document.getElementById('bulk-dem-files');
  const infoInput = document.getElementById('bulk-info-files');
  if (!demInput.files.length) return;

  const btn = document.getElementById('bulk-upload-btn');
  const progressSection = document.getElementById('bulk-progress');
  const progressText = document.getElementById('bulk-progress-text');
  const progressCount = document.getElementById('bulk-progress-count');
  const progressBar = document.getElementById('bulk-progress-bar');
  const resultsDiv = document.getElementById('bulk-results');

  btn.disabled = true; btn.textContent = 'PROCESSING...';
  progressSection.classList.remove('hidden');
  resultsDiv.innerHTML = '';
  progressBar.style.width = '0%';
  const total = demInput.files.length;
  progressCount.textContent = '0 / ' + total;
  progressText.textContent = 'Uploading ' + total + ' demo' + (total > 1 ? 's' : '') + '...';

  const fd = new FormData();
  for (const f of demInput.files) fd.append('files', f);
  for (const f of infoInput.files) fd.append('info_files', f);
  fd.append('steam_id', document.getElementById('bulk-steam-id').value);

  try {
    const res = await fetch(API + '/matches/upload-bulk', { method: 'POST', body: fd });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Bulk upload failed'); }
    const data = await res.json();
    progressBar.style.width = '100%';
    progressCount.textContent = data.processed + ' / ' + data.total;
    progressText.textContent = data.processed === data.total ? 'All demos processed!' : data.processed + ' of ' + data.total + ' succeeded';
    progressText.className = 'text-[10px] font-bold uppercase tracking-widest ' + (data.processed === data.total ? 'text-secondary' : 'text-warning');

    data.results.forEach(r => {
      const row = document.createElement('div');
      const ok = r.status === 'ok';
      row.className = 'flex items-center gap-2 text-xs p-2 rounded-lg ' + (ok ? 'bg-secondary/10 text-secondary' : 'bg-error/10 text-error');
      const icon = ok ? 'check_circle' : 'error';
      const detail = ok ? (r.map_name || '') + (r.player_name ? ' — ' + r.player_name : '') : (r.error || 'Failed');
      row.innerHTML = '<span class="material-symbols-outlined text-[16px]">' + icon + '</span><span class="font-medium truncate flex-1">' + r.filename + '</span><span class="text-on-surface-variant/70 text-[10px]">' + detail + '</span>';
      resultsDiv.appendChild(row);
    });

    if (data.processed > 0) setTimeout(() => { _overallTrends = null; _overallPerf = null; loadPerformance(); }, 500);
  } catch(err) {
    progressText.textContent = err.message;
    progressText.className = 'text-[10px] font-bold uppercase tracking-widest text-error';
  } finally {
    btn.disabled = false; btn.textContent = 'Process All Demos';
  }
}

// --- Sync Folder ---
let _syncDemos = [];

function openSyncModal() {
  const m = document.getElementById('sync-modal');
  m.classList.remove('hidden'); m.classList.add('flex');
  document.getElementById('sync-results').classList.add('hidden');
  document.getElementById('sync-progress').classList.add('hidden');
  document.getElementById('sync-scan-btn').disabled = false;
  document.getElementById('sync-scan-btn').textContent = 'Scan for New Demos';
  _syncDemos = [];
  loadSyncConfig();
  populateSyncAccountSelector();
}
function closeSyncModal() { const m = document.getElementById('sync-modal'); m.classList.add('hidden'); m.classList.remove('flex'); }
document.getElementById('sync-modal').addEventListener('click', e => { if (e.target === e.currentTarget) closeSyncModal(); });

async function loadSyncConfig() {
  try {
    const res = await fetch(API + '/sync/config');
    const cfg = await res.json();
    document.getElementById('sync-folder-input').value = cfg.folder || '';
    document.getElementById('sync-folder-status').textContent = cfg.folder ? 'Folder configured' : '';
  } catch(e) {}
}

async function saveSyncFolder() {
  const folder = document.getElementById('sync-folder-input').value.trim();
  const st = document.getElementById('sync-folder-status');
  if (!folder) { st.textContent = 'Enter a folder path'; st.className = 'text-[9px] text-error mt-1'; return; }
  try {
    const res = await fetch(API + '/sync/config', {
      method: 'PUT', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ folder })
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail); }
    st.textContent = 'Folder saved!'; st.className = 'text-[9px] text-secondary mt-1';
  } catch(err) { st.textContent = err.message; st.className = 'text-[9px] text-error mt-1'; }
}

async function populateSyncAccountSelector() {
  const sel = document.getElementById('sync-steam-id');
  try {
    const res = await fetch(API + '/accounts');
    const data = await res.json();
    sel.innerHTML = '';
    const accounts = Array.isArray(data) ? data : (data.accounts || []);
    accounts.forEach(a => {
      const opt = document.createElement('option');
      opt.value = a.steam_id; opt.textContent = a.name + ' (' + a.steam_id + ')';
      if (a.active) opt.selected = true;
      sel.appendChild(opt);
    });
  } catch(e) { sel.innerHTML = '<option value="">No accounts</option>'; }
}

async function syncScan() {
  const btn = document.getElementById('sync-scan-btn');
  btn.disabled = true; btn.textContent = 'SCANNING...';
  document.getElementById('sync-results').classList.add('hidden');
  document.getElementById('sync-progress').classList.add('hidden');
  try {
    const sid = document.getElementById('sync-steam-id').value;
    const res = await fetch(API + '/sync/scan' + (sid ? '?steam_id=' + encodeURIComponent(sid) : ''));
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail); }
    const data = await res.json();
    _syncDemos = data.new;
    const resultsDiv = document.getElementById('sync-results');
    const foundText = document.getElementById('sync-found-text');
    const fileList = document.getElementById('sync-file-list');

    if (!_syncDemos.length) {
      foundText.textContent = 'No new demos found (' + data.total_found + ' total in folder, all already imported).';
      fileList.innerHTML = '';
      resultsDiv.classList.remove('hidden');
      document.getElementById('sync-process-btn').classList.add('hidden');
      return;
    }

    foundText.textContent = _syncDemos.length + ' new demo' + (_syncDemos.length > 1 ? 's' : '') + ' found (' + data.total_found + ' total in folder)';
    document.getElementById('sync-process-btn').classList.remove('hidden');
    fileList.innerHTML = _syncDemos.map((d, i) => `
      <label class="flex items-start gap-3 p-2.5 rounded-lg bg-surface-container-highest hover:bg-white/5 cursor-pointer transition-colors">
        <input type="checkbox" class="sync-file-cb accent-secondary mt-1" data-idx="${i}" checked />
        <span class="material-symbols-outlined text-[14px] text-secondary mt-0.5">${d.has_info ? 'verified' : 'description'}</span>
        <div class="flex-1 min-w-0">
          <span class="text-xs text-on-surface font-medium block truncate">${esc(d.filename)}</span>
          <div class="flex gap-3 mt-0.5">
            ${d.match_date ? `<span class="text-[10px] text-on-surface-variant">${esc(d.match_date)}</span>` : ''}
            ${d.map_name ? `<span class="text-[10px] text-primary font-semibold">${esc(d.map_name)}</span>` : ''}
          </div>
        </div>
        <span class="text-[10px] text-on-surface-variant/50 mt-0.5 shrink-0">${d.size_mb} MB</span>
      </label>`).join('');
    resultsDiv.classList.remove('hidden');
  } catch(err) {
    document.getElementById('sync-found-text').textContent = err.message;
    document.getElementById('sync-found-text').className = 'text-xs text-error';
    document.getElementById('sync-results').classList.remove('hidden');
    document.getElementById('sync-file-list').innerHTML = '';
    document.getElementById('sync-process-btn').classList.add('hidden');
  } finally {
    btn.disabled = false; btn.textContent = 'Scan for New Demos';
  }
}

function syncSelectAll(checked) {
  document.querySelectorAll('.sync-file-cb').forEach(cb => cb.checked = checked);
}

async function syncProcess() {
  const selected = [];
  document.querySelectorAll('.sync-file-cb:checked').forEach(cb => {
    const idx = parseInt(cb.dataset.idx);
    if (_syncDemos[idx]) selected.push(_syncDemos[idx].filename);
  });
  if (!selected.length) return;

  const btn = document.getElementById('sync-process-btn');
  const progressSection = document.getElementById('sync-progress');
  const progressText = document.getElementById('sync-progress-text');
  const progressCount = document.getElementById('sync-progress-count');
  const progressBar = document.getElementById('sync-progress-bar');
  const resultsDiv = document.getElementById('sync-process-results');

  btn.disabled = true; btn.textContent = 'PROCESSING...';
  progressSection.classList.remove('hidden');
  resultsDiv.innerHTML = '';
  progressBar.style.width = '0%';
  progressCount.textContent = '0 / ' + selected.length;
  progressText.textContent = 'Processing ' + selected.length + ' demo' + (selected.length > 1 ? 's' : '') + '...';

  try {
    const res = await fetch(API + '/sync/process', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ filenames: selected, steam_id: document.getElementById('sync-steam-id').value })
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Sync failed'); }
    const data = await res.json();
    progressBar.style.width = '100%';
    progressCount.textContent = data.processed + ' / ' + data.total;
    progressText.textContent = data.processed === data.total ? 'All demos processed!' : data.processed + ' of ' + data.total + ' succeeded';
    progressText.className = 'text-[10px] font-bold uppercase tracking-widest ' + (data.processed === data.total ? 'text-secondary' : 'text-warning');

    data.results.forEach(r => {
      const row = document.createElement('div');
      const ok = r.status === 'ok';
      row.className = 'flex items-center gap-2 text-xs p-2 rounded-lg ' + (ok ? 'bg-secondary/10 text-secondary' : 'bg-error/10 text-error');
      const icon = ok ? 'check_circle' : 'error';
      const detail = ok ? (r.map_name || '') + (r.player_name ? ' — ' + r.player_name : '') : (r.detail || 'Failed');
      row.innerHTML = '<span class="material-symbols-outlined text-[16px]">' + icon + '</span><span class="font-medium truncate flex-1">' + esc(r.filename) + '</span><span class="text-on-surface-variant/70 text-[10px]">' + detail + '</span>';
      resultsDiv.appendChild(row);
    });

    if (data.processed > 0) setTimeout(() => { _overallTrends = null; _overallPerf = null; loadPerformance(); }, 500);
  } catch(err) {
    progressText.textContent = err.message;
    progressText.className = 'text-[10px] font-bold uppercase tracking-widest text-error';
  } finally {
    btn.disabled = false; btn.textContent = 'Process Selected';
  }
}

// --- Map filter state ---
let activeMap = null;
let _overallTrends = null;
let _overallPerf = null;

// --- Player filter state ---
let _selectedPlayerIds = new Set(); // empty = show all

// --- Trend chart state ---
// The metric list and view state moved to charts.js with the chart that
// reads them. This one is the page's own loaded data.
let _trendDataPoints = [];

// --- Build API query string helper ---
function _buildParams(mapFilter) {
  const p = new URLSearchParams();
  if (mapFilter) p.set('maps', mapFilter);
  if (_selectedPlayerIds.size > 0) p.set('steam_ids', [..._selectedPlayerIds].join(','));
  const s = p.toString();
  return s ? '?' + s : '';
}

// --- Player filter ---
function togglePlayerFilter(steamId) {
  if (steamId === null) {
    _selectedPlayerIds.clear();
  } else {
    if (_selectedPlayerIds.has(steamId)) _selectedPlayerIds.delete(steamId);
    else _selectedPlayerIds.add(steamId);
  }
  _renderPlayerFilterBar();
  _overallTrends = null; _overallPerf = null;
  loadPerformance(activeMap);
}

function _renderPlayerFilterBar() {
  const allBtn = document.getElementById('pf-all');
  const allActive = _selectedPlayerIds.size === 0;
  allBtn.className = allActive
    ? 'px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest bg-primary/20 text-primary border border-primary/30 transition-all'
    : 'px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest text-on-surface-variant border border-white/10 hover:border-primary/30 hover:text-primary transition-all';
  document.querySelectorAll('.pf-chip').forEach(chip => {
    const isActive = _selectedPlayerIds.has(chip.dataset.sid);
    chip.className = (isActive
      ? 'pf-chip px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest bg-secondary/20 text-secondary border border-secondary/30 cursor-pointer transition-all'
      : 'pf-chip px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest text-on-surface-variant border border-white/10 hover:border-secondary/30 hover:text-secondary cursor-pointer transition-all');
  });
}

async function loadPlayerFilter() {
  try {
    const res = await fetch(API + '/accounts');
    const data = await res.json();
    const accounts = Array.isArray(data) ? data : (data.accounts || []);
    if (accounts.length < 2) return;
    const bar = document.getElementById('player-filter-bar');
    const chips = document.getElementById('pf-chips');
    chips.innerHTML = accounts.map(a =>
      `<button class="pf-chip px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest text-on-surface-variant border border-white/10 hover:border-secondary/30 hover:text-secondary cursor-pointer transition-all" data-sid="${esc(a.steam_id)}" data-action="togglePlayerFilter" data-args="${actionArgs(a.steam_id)}">${esc(a.name)}</button>`
    ).join('');
    bar.classList.remove('hidden');
  } catch(e) {}
}

// --- Trend chart ---
function toggleTrendMetric(key) {
  if (_trendMetrics.has(key) && _trendMetrics.size > 1) _trendMetrics.delete(key);
  else _trendMetrics.add(key);
  _renderTrendMetricToggles();
  drawTrendChart(_trendDataPoints);
}

function setTrendTimescale(scale) {
  _trendTimescale = scale;
  _renderTrendTimescaleButtons();
  drawTrendChart(_trendDataPoints);
}

function _renderTrendMetricToggles() {
  TREND_METRICS.forEach(m => {
    const btn = document.getElementById('tm-' + m.key);
    if (!btn) return;
    if (_trendMetrics.has(m.key)) {
      const c = m.color();
      btn.style.borderColor = c;
      btn.style.color = c;
      btn.style.background = c + '22';
      btn.className = 'px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest border transition-all';
    } else {
      btn.style.borderColor = ''; btn.style.color = ''; btn.style.background = '';
      btn.className = 'px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest border border-white/10 text-on-surface-variant hover:border-white/30 transition-all';
    }
  });
}

function _renderTrendTimescaleButtons() {
  const scales = ['all', 5, 10, 20];
  scales.forEach(s => {
    const btn = document.getElementById('ts-' + s);
    if (!btn) return;
    btn.className = _trendTimescale === s
      ? 'px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest bg-primary/20 text-primary border border-primary/30 transition-all'
      : 'px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest text-on-surface-variant border border-white/10 hover:border-primary/30 hover:text-primary transition-all';
  });
}



const ACTIVE_DUTY_MAPS = [
  { id: 'de_dust2',   name: 'Dust II',   icon: 'wb_sunny' },
  { id: 'de_inferno', name: 'Inferno',   icon: 'local_fire_department' },
  { id: 'de_mirage',  name: 'Mirage',    icon: 'landscape' },
  { id: 'de_ancient', name: 'Ancient',   icon: 'temple_hindu' },
  { id: 'de_anubis',  name: 'Anubis',    icon: 'pyramid' },
  { id: 'de_overpass',name: 'Overpass',  icon: 'bridge' },
  { id: 'de_nuke',    name: 'Nuke',      icon: 'nuclear' },
  { id: 'de_cache',   name: 'Cache',     icon: 'database' },
];

function renderMapGrid(availableMaps) {
  const grid = document.getElementById('map-grid');
  grid.innerHTML = '';
  const clearBtn = document.getElementById('map-clear-btn');
  clearBtn.classList.toggle('hidden', !activeMap);

  for (const m of ACTIVE_DUTY_MAPS) {
    const hasData = availableMaps.includes(m.id);
    const isActive = activeMap === m.id;
    const card = document.createElement('button');
    card.className = isActive
      ? 'relative flex flex-col items-center justify-center p-4 rounded-xl border-2 border-primary bg-primary/10 shadow-[0_0_20px_rgba(204,151,255,0.2)] transition-all duration-300 group'
      : hasData
        ? 'relative flex flex-col items-center justify-center p-4 rounded-xl border border-white/10 bg-surface-container-highest hover:border-secondary/40 hover:bg-surface-container-high transition-all duration-300 cursor-pointer group'
        : 'relative flex flex-col items-center justify-center p-4 rounded-xl border border-white/5 bg-surface-container-highest/50 opacity-40 cursor-not-allowed group';
    // Maps with no data are dimmed by the card class already; desaturating the
    // icon as well stops a bright badge from pulling the eye to a dead card.
    card.innerHTML = `
      <img src="${mapIconUrl(m.id)}" alt="" loading="lazy"
           class="w-14 h-14 object-contain mb-2 ${hasData ? '' : 'grayscale'}"
           onerror="this.remove()"/>
      <span class="font-headline text-xs font-bold uppercase tracking-widest ${isActive ? 'text-primary' : 'text-on-surface'}">${esc(m.name)}</span>
      <span class="text-[9px] text-on-surface-variant/60 uppercase mt-0.5">${esc(m.id)}</span>
      ${hasData && !isActive ? '<span class="absolute top-2 right-2 w-1.5 h-1.5 rounded-full bg-secondary"></span>' : ''}
      ${isActive ? '<span class="absolute top-2 right-2 material-symbols-outlined text-primary text-sm">check_circle</span>' : ''}
    `;
    if (hasData) {
      card.addEventListener('click', () => {
        if (isActive) { clearMapSelection(); }
        else { activeMap = m.id; loadPerformance(m.id); }
      });
    }
    grid.appendChild(card);
  }
}

function clearMapSelection() {
  activeMap = null;
  loadPerformance();
}

// --- Distribution strip chart ---


// --- Render performance panel (shared logic for overall & map-specific) ---
function _renderPerfPanel(prefix, trends, perf) {
  const el = id => document.getElementById(id);
  const pts = trends.data_points || [];
  const n = pts.length;
  if (!n) return;

  // Rating
  const ratings = pts.map(d => d.hltv_rating || 0);
  const avgRating = trends.averages?.avg_rating || 0;
  el(prefix + '-rating').textContent = avgRating.toFixed(2);
  el(prefix + '-rating-range').textContent = Math.min(...ratings).toFixed(2) + ' — ' + Math.max(...ratings).toFixed(2);
  drawDistStrip(prefix + '-rating-strip', ratings, avgRating, TC.purple || '#cc97ff');

  // ADR
  const adrs = pts.map(d => d.adr || 0);
  const avgAdr = trends.averages?.avg_adr || 0;
  el(prefix + '-adr').textContent = avgAdr.toFixed(0);
  el(prefix + '-adr-range').textContent = Math.min(...adrs).toFixed(0) + ' — ' + Math.max(...adrs).toFixed(0);
  drawDistStrip(prefix + '-adr-strip', adrs, avgAdr, TC.cyan || '#53ddfc');

  // K/D
  const kds = pts.map(d => (d.deaths || 1) > 0 ? (d.kills || 0) / (d.deaths || 1) : (d.kills || 0));
  const totalK = pts.reduce((s, d) => s + (d.kills || 0), 0);
  const totalD = pts.reduce((s, d) => s + (d.deaths || 0), 0);
  const avgKD = totalD ? totalK / totalD : 0;
  el(prefix + '-kd').textContent = avgKD.toFixed(2);
  el(prefix + '-kd-range').textContent = Math.min(...kds).toFixed(2) + ' — ' + Math.max(...kds).toFixed(2);
  drawDistStrip(prefix + '-kd-strip', kds, avgKD, TC.onText || '#dee5ff');

  // KAST
  const kasts = pts.map(d => d.kast || 0);
  const avgKast = trends.averages?.avg_kast || 0;
  el(prefix + '-kast').textContent = avgKast.toFixed(1) + '%';
  el(prefix + '-kast-range').textContent = Math.min(...kasts).toFixed(0) + '% — ' + Math.max(...kasts).toFixed(0) + '%';
  drawDistStrip(prefix + '-kast-strip', kasts, avgKast, TC.cyan || '#53ddfc');

  // HS% (aggregate only, no per-match strip)
  const hs = perf.hs_pct || 0;
  el(prefix + '-hs').textContent = hs.toFixed(1) + '%';
  el(prefix + '-hs-range').textContent = (perf.total_rounds || 0).toLocaleString() + ' rounds';
  // No strip for HS — draw empty
  const hsCanvas = document.getElementById(prefix + '-hs-strip');
  if (hsCanvas) {
    const ctx = hsCanvas.getContext('2d');
    const w = hsCanvas.clientWidth; const h = hsCanvas.clientHeight;
    hsCanvas.width = w * 2; hsCanvas.height = h * 2;
    ctx.scale(2, 2); ctx.clearRect(0, 0, w, h);
    // Draw single bar for HS%
    ctx.fillStyle = TC.track || 'rgba(255,255,255,0.04)';
    ctx.beginPath(); ctx.roundRect(0, h / 2 - 2, w, 4, 2); ctx.fill();
    ctx.fillStyle = TC.pink || '#ff86c3';
    ctx.beginPath(); ctx.roundRect(0, h / 2 - 2, w * hs / 100, 4, 2); ctx.fill();
  }

  // Win Rate
  const wins = pts.filter(d => (d.match_result || '').toLowerCase() === 'win').length;
  const wr = n ? (wins / n * 100) : 0;
  el(prefix + '-winrate').textContent = wr.toFixed(0) + '%';
  el(prefix + '-winrate-detail').textContent = wins + 'W / ' + (n - wins) + 'L';
  const wrBar = el(prefix + '-winrate-bar');
  if (wrBar) wrBar.style.width = wr.toFixed(1) + '%';

  // AIM Rating (per-match from aim_stats.aim_rating)
  const aimVals = pts.map(d => d.aim_rating).filter(v => v != null);
  if (aimVals.length) {
    const avgAim = trends.averages?.avg_aim_rating || (aimVals.reduce((s,v) => s+v, 0) / aimVals.length);
    el(prefix + '-aim').textContent = avgAim.toFixed(1);
    el(prefix + '-aim-range').textContent = Math.min(...aimVals).toFixed(1) + ' — ' + Math.max(...aimVals).toFixed(1);
    drawDistStrip(prefix + '-aim-strip', aimVals, avgAim, TC.amber || '#fbbf24');
  } else {
    el(prefix + '-aim').textContent = '—';
    el(prefix + '-aim-range').textContent = 'No data';
  }

  // Utility Rating (per-match from utility_data.utility_rating)
  const utilVals = pts.map(d => d.utility_rating).filter(v => v != null);
  if (utilVals.length) {
    const avgUtil = trends.averages?.avg_utility_rating || (utilVals.reduce((s,v) => s+v, 0) / utilVals.length);
    el(prefix + '-util').textContent = avgUtil.toFixed(1);
    el(prefix + '-util-range').textContent = Math.min(...utilVals).toFixed(1) + ' — ' + Math.max(...utilVals).toFixed(1);
    drawDistStrip(prefix + '-util-strip', utilVals, avgUtil, TC.success || '#34d399');
  } else {
    el(prefix + '-util').textContent = '—';
    el(prefix + '-util-range').textContent = 'No data';
  }
}

function renderOverallPerformance(trends, perf) {
  const section = document.getElementById('overall-perf-section');
  const pts = trends.data_points || [];
  if (!pts.length) { section.classList.add('hidden'); return; }
  section.classList.remove('hidden');
  document.getElementById('overall-match-count').textContent = pts.length;
  _renderPerfPanel('overall', trends, perf);
}

function renderMapPerformance(trends, perf, mapFilter) {
  const section = document.getElementById('map-perf-section');
  if (!mapFilter) { section.classList.add('hidden'); return; }
  const pts = trends.data_points || [];
  if (!pts.length) { section.classList.add('hidden'); return; }
  section.classList.remove('hidden');
  const mapName = mapFilter.replace('de_', '').charAt(0).toUpperCase() + mapFilter.replace('de_', '').slice(1);
  document.getElementById('map-perf-title').textContent = mapName + ' — Performance';
  document.getElementById('map-perf-match-count').textContent = pts.length;
  _renderPerfPanel('map-perf', trends, perf);
}

async function loadPerformance(mapFilter) {
  try {
    // Overall data — player-filtered but not map-filtered
    if (!_overallTrends || !_overallPerf) {
      const overallParam = _buildParams(null);
      const [oTrendsRes, oPerfRes] = await Promise.all([
        fetch(API + '/trends' + overallParam),
        fetch(API + '/performance' + overallParam),
      ]);
      _overallTrends = await oTrendsRes.json();
      _overallPerf = await oPerfRes.json();
    }
    renderOverallPerformance(_overallTrends, _overallPerf);

    // Fetch map-filtered data or reuse overall
    let trends, perf;
    if (mapFilter) {
      const mapParam = _buildParams(mapFilter);
      const [trendsRes, perfRes] = await Promise.all([
        fetch(API + '/trends' + mapParam),
        fetch(API + '/performance' + mapParam),
      ]);
      trends = await trendsRes.json();
      perf = await perfRes.json();
    } else {
      trends = _overallTrends;
      perf = _overallPerf;
    }

    // Main trend chart — always overall (player-filtered, never map-filtered)
    _trendDataPoints = _overallTrends.data_points || [];
    const trendSection = document.getElementById('trend-graph-section');
    if (_trendDataPoints.length >= 2) {
      trendSection.classList.remove('hidden');
      _renderTrendMetricToggles();
      drawTrendChart(_trendDataPoints);
    } else {
      trendSection.classList.add('hidden');
    }

    // Map-specific trend chart — only shown when a map is selected
    const mapTrendSection = document.getElementById('map-trend-section');
    const mapTrendPts = mapFilter ? (trends.data_points || []) : [];
    if (mapFilter && mapTrendPts.length >= 2) {
      mapTrendSection.classList.remove('hidden');
      const mapLabel = document.getElementById('map-trend-label');
      if (mapLabel) mapLabel.textContent = mapFilter.replace('de_', '').replace('cs_', '').replace(/^./, c => c.toUpperCase()) + ' — Trend';
      drawTrendChart(mapTrendPts, 'map-trend-chart');
    } else {
      mapTrendSection.classList.add('hidden');
    }

    renderMapGrid(trends.available_maps || _overallTrends.available_maps || []);
    renderMechanics(perf, trends);
    renderMapPerformance(trends, perf, mapFilter);
    renderSideStats(perf, mapFilter);
    renderSideRoles(perf, mapFilter);
    renderOverallAssessment(mapFilter);
  } catch (err) { console.error('Failed to load performance:', err); }
}

function renderMechanics(perf, trends) {
  const el = id => document.getElementById(id);
  const hs = perf.hs_pct || 0;
  el('stat-hs-pct').textContent = hs.toFixed(1);
  el('hs-ring').setAttribute('stroke-dashoffset', (440 * (1 - hs / 100)).toFixed(0));
  el('stat-open-duel').textContent = perf.opening_kill_pct ? perf.opening_kill_pct.toFixed(0) + '%' : '—';
  el('stat-top-weapon').textContent = perf.top_weapon || '—';
  el('stat-total-rounds').textContent = (perf.total_rounds || 0).toLocaleString();
  el('stat-hltv-rating').textContent = (trends.averages?.avg_rating || 0).toFixed(2);
  el('stat-total-matches').textContent = (perf.total_matches || 0).toLocaleString();
}

function renderSideStats(perf, mapFilter) {
  const section = document.getElementById('side-stats-section');
  if (!mapFilter) { section.classList.add('hidden'); return; }
  section.classList.remove('hidden');

  const el = id => document.getElementById(id);
  const mapName = mapFilter.replace('de_', '').charAt(0).toUpperCase() + mapFilter.replace('de_', '').slice(1);
  el('side-stats-title').textContent = mapName + ' — Side Effectiveness';

  // CT stats
  el('side-ct-winrate').textContent = perf.ct_win_pct + '%';
  el('side-ct-bar').style.width = perf.ct_win_pct + '%';
  const ct = perf.ct_role || {};
  el('side-ct-adr').textContent = ct.adr || '—';
  el('side-ct-survival').textContent = ct.survival_pct ? ct.survival_pct.toFixed(0) + '%' : '—';
  el('side-ct-rounds').textContent = (ct.rounds || 0) + ' rounds';
  el('side-ct-kd').textContent = (ct.kills || 0) + 'K / ' + (ct.deaths || 0) + 'D';

  // T stats
  el('side-t-winrate').textContent = perf.t_win_pct + '%';
  el('side-t-bar').style.width = perf.t_win_pct + '%';
  const t = perf.t_role || {};
  el('side-t-adr').textContent = t.adr || '—';
  el('side-t-survival').textContent = t.survival_pct ? t.survival_pct.toFixed(0) + '%' : '—';
  el('side-t-rounds').textContent = (t.rounds || 0) + ' rounds';
  el('side-t-kd').textContent = (t.kills || 0) + 'K / ' + (t.deaths || 0) + 'D';
}

function renderSideRoles(perf, mapFilter) {
  const section = document.getElementById('side-roles-section');
  if (!mapFilter) { section.classList.add('hidden'); return; }
  section.classList.remove('hidden');

  const el = id => document.getElementById(id);
  const mapName = mapFilter.replace('de_', '').charAt(0).toUpperCase() + mapFilter.replace('de_', '').slice(1);
  el('side-roles-title').textContent = mapName + ' — Roles & Patterns';

  // Heuristic roles first; the AI assessment refines them if one exists.
  _heuristicRoles = { ct: perf.ct_role || {}, t: perf.t_role || {} };
  applyRoleData(_heuristicRoles.ct, _heuristicRoles.t, false);
  applyPatternData(null);

  fetch(API + '/performance/ai-assessment?maps=' + encodeURIComponent(mapFilter))
    .then(r => r.ok ? r.json() : null)
    .then(data => { if (data) applyAssessment(data); })
    .catch(() => {});
}

let _heuristicRoles = { ct: {}, t: {} };

// One assessment covers both halves, so one function applies it.
function applyAssessment(data) {
  if (data.ct_role && data.ct_role.name) {
    // The model supplies name, icon and description. Opening-duel counts and
    // the radar axes stay from the heuristic pass, which measured them — the
    // model was never given them to return, so overwriting wiped both.
    applyRoleData(
      { ..._heuristicRoles.ct, ...data.ct_role },
      { ..._heuristicRoles.t, ...(data.t_role || {}) },
      true,
    );
  }
  if (data.headline || (data.aim && data.aim.name)) {
    applyPatternData(data);
  }
}

function applyRoleData(ctRole, tRole, isAI) {
  const el = id => document.getElementById(id);
  const ct = ctRole || {};
  el('ct-role-icon').textContent = ct.icon || 'help';
  el('ct-role-name').textContent = ct.name || 'Unknown';
  el('ct-role-desc').textContent = ct.description || 'No data available.';
  el('ct-role-fk').textContent = ct.opening_kills ?? '—';
  el('ct-role-fd').textContent = ct.opening_deaths ?? '—';

  const t = tRole || {};
  el('t-role-icon').textContent = t.icon || 'help';
  el('t-role-name').textContent = t.name || 'Unknown';
  el('t-role-desc').textContent = t.description || 'No data available.';
  el('t-role-fk').textContent = t.opening_kills ?? '—';
  el('t-role-fd').textContent = t.opening_deaths ?? '—';

  // Draw radar charts
  if (ct.axes) drawRoleRadar('ct-role-radar', ct.axes, TC.cyan || '#53ddfc');
  if (t.axes) drawRoleRadar('t-role-radar', t.axes, TC.pink || '#ff86c3');

  // Show/hide AI badge
  const badge = document.getElementById('ai-role-badge');
  if (badge) badge.classList.toggle('hidden', !isAI);
}

// --- Radar chart for 5-axis role assessment ---


// --- AI Patterns ---
// Roles answer "where does this player play"; patterns answer "how". Same map
// filter, same persisted-then-refresh flow as the role card above.
//
// The career card renders the same way over the same shape, with a fourth
// section: across maps there are no roles to name — callouts belong to the map
// they came from — so the comparison of the maps themselves takes their place.
const PATTERN_SECTIONS = [
  { key: 'aim',       label: 'Aim',       accent: 'secondary' },
  { key: 'utility',   label: 'Utility',   accent: 'tertiary' },
  { key: 'behaviour', label: 'Behaviour', accent: 'primary' },
];
const OVERALL_SECTIONS = PATTERN_SECTIONS.concat(
  [{ key: 'maps', label: 'Map Pool', accent: 'secondary' }]
);

const MAP_PATTERN_IDS = {
  empty: 'ai-patterns-empty', cards: 'ai-patterns-cards',
  meta: 'ai-patterns-meta', headline: 'ai-assessment-headline', badge: null,
};
const OVERALL_PATTERN_IDS = {
  empty: 'ai-overall-empty', cards: 'ai-overall-cards',
  meta: 'ai-overall-meta', headline: 'ai-overall-headline', badge: 'ai-overall-badge',
};

function applyPatternData(data, ids, sections) {
  ids = ids || MAP_PATTERN_IDS;
  sections = sections || PATTERN_SECTIONS;
  const empty = document.getElementById(ids.empty);
  const cardsEl = document.getElementById(ids.cards);
  const headline = document.getElementById(ids.headline);
  const badge = ids.badge ? document.getElementById(ids.badge) : null;
  if (badge) badge.classList.toggle('hidden', !data);
  if (!empty || !cardsEl) return;

  if (!data) {
    empty.classList.remove('hidden');
    cardsEl.classList.add('hidden');
    headline.classList.add('hidden');
    document.getElementById(ids.meta).textContent = '';
    return;
  }

  empty.classList.add('hidden');
  cardsEl.classList.remove('hidden');
  headline.textContent = data.headline || '';
  headline.classList.toggle('hidden', !data.headline);

  const cards = sections.map(sec => {
    const p = data[sec.key] || {};
    if (!p.name && !p.description) return '';
    const bullets = (p.tendencies || []).length
      ? '<ul class="mt-3 space-y-1">' + p.tendencies.map(t =>
          `<li class="text-[11px] text-on-surface-variant flex gap-2"><span class="text-${sec.accent}">›</span><span>${esc(t)}</span></li>`
        ).join('') + '</ul>'
      : '';
    return `
      <div class="p-4 bg-surface-container-highest rounded-xl border border-${sec.accent}/20">
        <div class="flex items-center gap-3 mb-2">
          <div class="h-9 w-9 rounded-lg bg-surface-container flex items-center justify-center border border-${sec.accent}/30">
            <span class="material-symbols-outlined text-${sec.accent} text-lg">${esc(p.icon || 'insights')}</span>
          </div>
          <div class="flex-1 min-w-0">
            <span class="text-[9px] font-bold text-${sec.accent} uppercase tracking-widest">${sec.label}</span>
            <p class="text-sm font-bold truncate">${esc(p.name || '—')}</p>
          </div>
        </div>
        <p class="text-xs text-on-surface-variant leading-relaxed">${esc(p.description || '')}</p>
        ${bullets}
      </div>`;
  }).join('');
  cardsEl.innerHTML = cards;

  const meta = [];
  if (data.matches) meta.push(data.matches + ' matches, ' + (data.rounds || 0) + ' rounds');
  if (data.model) meta.push(data.model);
  document.getElementById(ids.meta).textContent = meta.join(' · ');
}

// One button, one call. Roles and patterns were two endpoints and two
// requests, which split what is really one question — where the player plays
// and how — across two prompts that could not see each other's data.
//
// The same runner drives the career card: same endpoint, no map filter, and
// the response rendered into the other set of elements.
async function runAssessment({ btnId, btnTextId, provId, modelId, maps, onResult }) {
  const btn = document.getElementById(btnId);
  const btnText = document.getElementById(btnTextId);
  btn.disabled = true;
  btn.classList.add('opacity-60');
  btnText.textContent = 'Analyzing...';
  try {
    const params = new URLSearchParams();
    if (maps) params.set('maps', maps);
    // Honour the selector next to the button; blank falls back to the
    // configured default server-side.
    const prov = document.getElementById(provId)?.value;
    const model = document.getElementById(modelId)?.value;
    if (prov) params.set('provider', prov);
    if (model) params.set('model', model);

    const res = await fetch(API + '/performance/ai-assessment?' + params, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'AI analysis failed');
    }
    onResult(await res.json());
    btnText.textContent = 'AI ✓';
    btn.classList.remove('opacity-60');
    btn.classList.add('border-secondary/50', 'text-secondary');
    setTimeout(() => {
      btnText.textContent = 'AI Analyze';
      btn.classList.remove('border-secondary/50', 'text-secondary');
      btn.disabled = false;
    }, 3000);
  } catch (err) {
    btnText.textContent = err.message.length > 30 ? 'Failed' : err.message;
    btn.title = err.message;
    btn.classList.remove('opacity-60');
    btn.classList.add('border-error/50', 'text-error');
    setTimeout(() => {
      btnText.textContent = 'AI Analyze';
      btn.classList.remove('border-error/50', 'text-error');
      btn.disabled = false;
    }, 4000);
  }
}

function analyzeWithAI() {
  if (!activeMap) return;
  return runAssessment({
    btnId: 'ai-role-btn', btnTextId: 'ai-role-btn-text',
    provId: 'ai-provider-select', modelId: 'ai-model-select',
    maps: activeMap, onResult: applyAssessment,
  });
}

function analyzeOverallWithAI() {
  return runAssessment({
    btnId: 'ai-overall-btn', btnTextId: 'ai-overall-btn-text',
    provId: 'ai-overall-provider-select', modelId: 'ai-overall-model-select',
    maps: null,
    onResult: data => applyPatternData(data, OVERALL_PATTERN_IDS, OVERALL_SECTIONS),
  });
}

// The career card shows whenever no map is selected — the same condition that
// hides the per-map one.
function renderOverallAssessment(mapFilter) {
  const section = document.getElementById('ai-overall-section');
  if (!section) return;
  if (mapFilter) { section.classList.add('hidden'); return; }
  section.classList.remove('hidden');

  applyPatternData(null, OVERALL_PATTERN_IDS, OVERALL_SECTIONS);
  fetch(API + '/performance/ai-assessment')
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (data && (data.headline || (data.aim && data.aim.name))) {
        applyPatternData(data, OVERALL_PATTERN_IDS, OVERALL_SECTIONS);
      }
    })
    .catch(() => {});
}

// --- Sidebar collapse ---
function toggleSidebar() {
  const sb = document.getElementById('sidebar');
  const main = document.getElementById('main-area');
  sb.classList.toggle('collapsed');
  const collapsed = sb.classList.contains('collapsed');
  main.style.paddingLeft = collapsed ? '6rem' : '';
  try { localStorage.setItem('sidebar-collapsed', collapsed ? '1' : '0'); } catch(e) {}
}
(function applySidebarState() {
  try {
    if (localStorage.getItem('sidebar-collapsed') === '1') {
      const sb = document.getElementById('sidebar');
      const main = document.getElementById('main-area');
      if (sb) { sb.classList.add('collapsed'); }
      if (main) { main.style.paddingLeft = '6rem'; }
    }
  } catch(e) {}
})();

// ─── Factory Reset ───

document.addEventListener('DOMContentLoaded', () => {
  loadAccounts();
  loadPlayerFilter();
  _renderTrendMetricToggles();
  _renderTrendTimescaleButtons();
  loadPerformance();
  checkOnboarding();
});

// ─── Onboarding ───
let _obStep = 0;
const OB_TOTAL = 5;
let _obSelectedAccount = null;
let _obFriends = [];
let _obAllPlayers = [];

async function checkOnboarding() {
  try {
    const res = await fetch(API + '/onboarding');
    if (res.ok) {
      const data = await res.json();
      if (data.completed) return;
    }
  } catch (_) {}
  openOnboarding();
}

function openOnboarding() {
  _obStep = 0; _obSelectedAccount = null; _obFriends = []; _obAllPlayers = [];
  updateObStep();
  const m = document.getElementById('onboarding-modal');
  m.classList.remove('hidden'); m.classList.add('flex');
}

function closeOnboarding() {
  const m = document.getElementById('onboarding-modal');
  m.classList.add('hidden'); m.classList.remove('flex');
  if (document.getElementById('ob-dismiss-check').checked) {
    fetch(API + '/onboarding', { method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({completed: true}) }).catch(() => {});
  }
}

function updateObStep() {
  for (let i = 0; i < OB_TOTAL; i++) {
    const step = document.getElementById('ob-step-' + i);
    const dot = document.getElementById('ob-dot-' + i);
    if (step) step.classList.toggle('hidden', i !== _obStep);
    if (dot) {
      dot.classList.toggle('bg-primary', i === _obStep);
      dot.classList.toggle('bg-white/20', i !== _obStep);
    }
  }
  const prev = document.getElementById('ob-prev-btn');
  const next = document.getElementById('ob-next-btn');
  prev.classList.toggle('hidden', _obStep === 0);
  if (_obStep === OB_TOTAL - 1) {
    next.textContent = 'Get Started';
  } else {
    next.textContent = (_obStep === 2 || _obStep === 3) ? 'Skip / Next' : 'Next';
  }
  // Init AI provider tabs when entering step 3
  if (_obStep === 3) obInitAIStep();
}

function obNext() {
  if (_obStep >= OB_TOTAL - 1) { closeOnboarding(); return; }
  // Save AI config when leaving step 3
  if (_obStep === 3) obSaveAIConfig();
  _obStep++;
  updateObStep();
}

function obPrev() {
  if (_obStep <= 0) return;
  _obStep--;
  updateObStep();
}

// ─── Onboarding: AI config step ───
let _obAIProviders = null;
let _obSelectedProvider = '';

async function obInitAIStep() {
  if (_obAIProviders) { obRenderAIProviders(); return; }
  try {
    const res = await fetch(API + '/ai/providers');
    _obAIProviders = await res.json();
    _obSelectedProvider = _obSelectedProvider || Object.keys(_obAIProviders)[0];
    obRenderAIProviders();
  } catch (_) {}
}

function obRenderAIProviders() {
  if (!_obAIProviders) return;
  const tabsEl = document.getElementById('ob-ai-provider-tabs');
  tabsEl.innerHTML = Object.keys(_obAIProviders).map(p => {
    const isActive = p === _obSelectedProvider;
    return `<button type="button" data-action="obSelectAIProvider" data-args="${actionArgs(p)}" class="px-3 py-1.5 rounded-full text-[9px] font-bold uppercase tracking-widest transition-all ${isActive ? 'bg-primary text-on-primary-fixed' : 'bg-surface-container-highest text-on-surface-variant hover:bg-white/10'}">${p}</button>`;
  }).join('');
  const modelSel = document.getElementById('ob-ai-model');
  const models = _obAIProviders[_obSelectedProvider] || [];
  const defaultModel = models.length > 0 ? models[0] : '';
  modelSel.innerHTML = models.map(m => `<option value="${m}" ${m === defaultModel ? 'selected' : ''}>${m}</option>`).join('');
}

function obSelectAIProvider(provider) {
  _obSelectedProvider = provider;
  document.getElementById('ob-ai-key').value = '';
  obRenderAIProviders();
}

async function obSaveAIConfig() {
  const key = document.getElementById('ob-ai-key').value.trim();
  if (!key) return; // Skip if no key entered
  const model = document.getElementById('ob-ai-model').value;
  try {
    const body = {
      active_provider: _obSelectedProvider,
      active_model: model,
      providers: {}
    };
    body.providers[_obSelectedProvider] = { api_key: key };
    await fetch(API + '/ai/config', {
      method: 'PUT', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
  } catch (_) {}
}

// ─── Onboarding: .dem.info player detection ───
document.getElementById('ob-info-file').addEventListener('change', async function() {
  const status = document.getElementById('ob-detect-status');
  const section = document.getElementById('ob-players-section');
  status.classList.add('hidden');
  section.classList.add('hidden');
  if (!this.files.length) return;
  status.textContent = 'Detecting players...';
  status.className = 'text-xs mt-2 text-on-surface-variant';
  status.classList.remove('hidden');
  const fd = new FormData();
  fd.append('info_file', this.files[0]);
  const demInput = document.getElementById('ob-dem-file');
  if (demInput && demInput.files.length) fd.append('demo_file', demInput.files[0]);
  try {
    const res = await fetch(API + '/matches/detect-player', { method: 'POST', body: fd });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Detection failed'); }
    const data = await res.json();
    _obAllPlayers = [];
    // Build combined player list from matched + unmatched
    if (data.matched) data.matched.forEach(a => _obAllPlayers.push({ steam_id: a.steam_id, name: a.name, known: true }));
    if (data.unmatched) data.unmatched.forEach(u => _obAllPlayers.push({ steam_id: u.steam_id, name: u.name || '', known: false }));
    if (_obAllPlayers.length === 0) {
      status.textContent = 'No players found in this file.';
      status.className = 'text-xs mt-2 text-error';
      return;
    }
    status.textContent = _obAllPlayers.length + ' player(s) detected.';
    status.className = 'text-xs mt-2 text-secondary';
    renderObPlayers();
    section.classList.remove('hidden');
  } catch (err) {
    status.textContent = err.message;
    status.className = 'text-xs mt-2 text-error';
  }
});

function renderObPlayers() {
  const list = document.getElementById('ob-players-list');
  list.innerHTML = _obAllPlayers.map((p, i) => {
    const isMe = _obSelectedAccount && _obSelectedAccount.steam_id === p.steam_id;
    const isFriend = _obFriends.some(f => f.steam_id === p.steam_id);
    const label = p.name ? esc(p.name) + ' <span class="text-on-surface-variant/50 font-mono text-[9px]">' + esc(p.steam_id) + '</span>' : '<span class="font-mono text-[10px]">' + esc(p.steam_id) + '</span>';
    const meBtnClass = isMe
      ? 'bg-primary text-on-primary-container'
      : 'bg-surface-container-highest text-primary border border-primary/30 hover:bg-primary/10';
    const friendBtnClass = isFriend
      ? 'bg-accent/20 text-accent'
      : 'bg-surface-container-highest text-accent border border-accent/30 hover:bg-accent/10';
    return `<div class="flex items-center gap-3 p-3 rounded-lg bg-surface-container-highest/50 border border-white/5">
      <div class="flex-1 min-w-0 text-xs">${label}</div>
      <button type="button" data-action="obSetMe" data-args="[${i}]" class="text-[9px] px-3 py-1.5 rounded-full font-bold uppercase tracking-wider transition-all ${meBtnClass}">
        ${isMe ? '✓ Me' : 'Set as Me'}
      </button>
      <button type="button" data-action="obToggleFriend" data-args="[${i}]" class="text-[9px] px-3 py-1.5 rounded-full font-bold uppercase tracking-wider transition-all ${friendBtnClass}" ${isMe ? 'disabled style="opacity:0.3;pointer-events:none"' : ''}>
        ${isFriend ? '✓ Friend' : 'Add Friend'}
      </button>
    </div>`;
  }).join('');
  updateObSummary();
}

async function obSetMe(idx) {
  const p = _obAllPlayers[idx];
  // Remove from friends if tagged
  _obFriends = _obFriends.filter(f => f.steam_id !== p.steam_id);
  // Prompt for a name if unknown
  let name = p.name;
  if (!name) {
    name = prompt('Enter a name for this account:', '') || 'Main Account';
  }
  _obSelectedAccount = { steam_id: p.steam_id, name: name };
  // Save to backend
  try {
    const res = await fetch(API + '/accounts', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name, steam_id: p.steam_id, display_name: name, rank: '' })
    });
    if (!res.ok) {
      const err = await res.json();
      // Ignore "already exists" errors
      if (!err.detail?.includes?.('exists')) throw new Error(err.detail);
    }
    // Activate this account
    await fetch(API + '/accounts/' + p.steam_id + '/activate', { method: 'PUT' });
  } catch (err) { console.error('Failed to save account:', err); }
  renderObPlayers();
}

async function obToggleFriend(idx) {
  const p = _obAllPlayers[idx];
  if (_obSelectedAccount && _obSelectedAccount.steam_id === p.steam_id) return;
  const existing = _obFriends.findIndex(f => f.steam_id === p.steam_id);
  if (existing >= 0) {
    // Remove friend
    _obFriends.splice(existing, 1);
    try { await fetch(API + '/friends/' + p.steam_id, { method: 'DELETE' }); } catch(e) {}
  } else {
    // Add friend
    let name = p.name;
    if (!name) {
      name = prompt('Enter a name for this friend:', '') || 'Friend';
    }
    _obFriends.push({ steam_id: p.steam_id, name: name });
    try {
      await fetch(API + '/friends', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steam_id: p.steam_id, name: name })
      });
    } catch(e) {}
  }
  renderObPlayers();
}

function updateObSummary() {
  const summary = document.getElementById('ob-selections-summary');
  const acctEl = document.getElementById('ob-my-account');
  const friendsEl = document.getElementById('ob-my-friends');
  if (!_obSelectedAccount && _obFriends.length === 0) {
    summary.classList.add('hidden');
    return;
  }
  summary.classList.remove('hidden');
  acctEl.innerHTML = _obSelectedAccount
    ? '<span class="text-primary font-bold">You:</span> ' + esc(_obSelectedAccount.name) + ' <span class="text-on-surface-variant/50 font-mono">(' + esc(_obSelectedAccount.steam_id) + ')</span>'
    : '<span class="text-on-surface-variant">No account selected yet</span>';
  friendsEl.innerHTML = _obFriends.length
    ? '<span class="font-bold">Friends:</span> ' + _obFriends.map(f => esc(f.name || f.steam_id)).join(', ')
    : '';
}


document.addEventListener('DOMContentLoaded', () => {
  // The selectors beside the AI Analyze buttons need the provider list, which
  // used to load only when the settings modal was opened.
  loadAIConfig();
  for (const sel of AI_SELECTORS) {
    document.getElementById(sel.prov)
      ?.addEventListener('change', () => renderAIModelOptions(sel.prov, sel.model));
  }
});


/* What this file offers the markup. See js/actions.js. */
registerActions({
  analyzeOverallWithAI,
  analyzeWithAI,
  clearMapSelection,
  closeOnboarding,
  closeSyncModal,
  obNext,
  obPrev,
  obSelectAIProvider,
  obSetMe,
  obToggleFriend,
  openAISettings,
  openSettingsModal,
  openSyncModal,
  saveAISettings,
  saveSyncFolder,
  selectAIProvider,
  setTrendTimescale,
  startBulkUpload,
  switchSettingsTab,
  syncProcess,
  syncScan,
  syncSelectAll,
  togglePlayerFilter,
  toggleSidebar,
  toggleTrendMetric,
});


/* What the shared panels need back from this page. See js/hooks.js. */
Object.assign(hooks, { switchSettingsTab, populateBulkAccountSelector });
