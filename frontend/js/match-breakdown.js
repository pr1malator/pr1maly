import './theme-actions.js';
import { loadUpdateStatus } from './updates.js';
import './explain.js';
import { registerActions, actionArgs } from './actions.js';
import { API } from './api.js';
import { esc } from './escape.js';
import { hooks } from './hooks.js';
import { closeSettingsModal, closeUploadModal, loadAccounts, loadFriends } from './accounts.js';
import { loadSteamStatus, onSettingsTabShown } from './steam-panel.js';
import { loadReanalyzeList } from '../reanalyze.js';

// --- Account Management ---

// --- Settings Modal ---
function openSettingsModal(tab) {
  document.getElementById('settings-modal').classList.remove('hidden');
  document.getElementById('settings-modal').classList.add('flex');
  switchSettingsTab(tab || 'accounts');
  loadAccounts();
  loadAIConfig().then(() => renderAISettings());
}
document.getElementById('settings-modal').addEventListener('click', e => { if (e.target === e.currentTarget) closeSettingsModal(); });

function openAISettings() { openSettingsModal('ai'); }

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
  if (tab === 'reanalyze') loadReanalyzeList();
  if (tab === 'updates') loadUpdateStatus();
  onSettingsTabShown(tab);
  if (tab === 'steam') loadSteamStatus();
}




// --- Friends Management ---




let currentMatchId = null;
let currentMatchData = null;
let currentMapName = null;
let minimapData = null;
let aiConfig = null;
let aiProviders = null;
let settingsActiveProvider = 'openai';

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
    const data = await res.json();
    status.classList.remove('text-on-surface-variant'); status.classList.add('text-secondary'); status.textContent = 'Demo processed!';
    setTimeout(() => { window.location.href = 'match-breakdown.html?id=' + data.match_id; }, 800);
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
    (data.accounts || []).forEach(a => {
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
  } catch(err) {
    progressText.textContent = err.message;
    progressText.className = 'text-[10px] font-bold uppercase tracking-widest text-error';
  } finally {
    btn.disabled = false; btn.textContent = 'Process Selected';
  }
}

async function loadAIConfig() {
  try {
    const [configRes, provRes] = await Promise.all([
      fetch(API + '/ai/config'),
      fetch(API + '/ai/providers'),
    ]);
    aiConfig = await configRes.json();
    aiProviders = await provRes.json();
  } catch (e) { console.error('Failed to load AI config:', e); }
}

function renderAISettings() {
  if (!aiProviders || !aiConfig) return;
  // Provider tabs
  const tabs = document.getElementById('ai-provider-tabs');
  tabs.innerHTML = '';
  for (const [key, p] of Object.entries(aiProviders)) {
    const isActive = key === settingsActiveProvider;
    const hasKey = aiConfig.providers?.[key]?.api_key_set;
    const btn = document.createElement('button');
    btn.className = `px-4 py-2 rounded-full text-[10px] font-bold uppercase tracking-widest transition-all ${isActive ? 'bg-primary text-on-primary-fixed' : 'bg-surface-container-highest text-on-surface-variant hover:bg-white/10'}`;
    btn.textContent = p.label + (hasKey ? ' ✓' : '');
    btn.onclick = () => { settingsActiveProvider = key; renderAISettings(); };
    tabs.appendChild(btn);
  }
  // Key status
  const provConfig = aiConfig.providers?.[settingsActiveProvider];
  const keyStatus = document.getElementById('ai-key-status');
  document.getElementById('ai-key-input').value = '';
  if (provConfig?.api_key_set) {
    keyStatus.textContent = 'Key set: ' + provConfig.api_key_masked;
    keyStatus.className = 'text-[9px] text-secondary mt-1';
  } else {
    keyStatus.textContent = 'No key configured';
    keyStatus.className = 'text-[9px] text-on-surface-variant mt-1';
  }
  // Model select
  const modelSel = document.getElementById('ai-settings-model');
  const models = aiProviders[settingsActiveProvider]?.models || [];
  const currentDefault = provConfig?.default_model || aiProviders[settingsActiveProvider]?.default_model || '';
  modelSel.innerHTML = models.map(m => `<option value="${m}" ${m === currentDefault ? 'selected' : ''}>${m}</option>`).join('');
  // System instructions
  document.getElementById('ai-system-instructions').value = aiConfig.system_instructions || '';
  // Prompts
  renderPromptTemplates();
}

function renderPromptTemplates() {
  const list = document.getElementById('ai-prompts-list');
  const prompts = aiConfig?.prompts || [];
  list.innerHTML = '';
  prompts.forEach((p, i) => {
    const nameEl = document.createElement('input');
    nameEl.type = 'text'; nameEl.value = p.name; nameEl.dataset.promptName = i;
    nameEl.className = 'w-full bg-surface-container-highest text-on-surface text-[10px] font-bold uppercase tracking-widest px-3 py-1.5 rounded-lg border border-transparent focus:border-secondary/30 focus:ring-0';
    const promptEl = document.createElement('input');
    promptEl.type = 'text'; promptEl.value = p.prompt; promptEl.dataset.promptText = i;
    promptEl.className = 'w-full bg-surface-container-highest text-on-surface-variant text-[10px] px-3 py-1.5 rounded-lg border border-transparent focus:border-secondary/30 focus:ring-0';
    const row = document.createElement('div'); row.className = 'flex gap-2 items-start';
    const inputs = document.createElement('div'); inputs.className = 'flex-1 space-y-1';
    inputs.appendChild(nameEl); inputs.appendChild(promptEl);
    const btn = document.createElement('button'); btn.className = 'material-symbols-outlined text-on-surface-variant hover:text-error text-sm mt-1';
    btn.textContent = 'close'; btn.onclick = () => removePromptTemplate(i);
    row.appendChild(inputs); row.appendChild(btn); list.appendChild(row);
  });
}

function addPromptTemplate() {
  if (!aiConfig) return;
  if (!aiConfig.prompts) aiConfig.prompts = [];
  aiConfig.prompts.push({ name: 'New Prompt', prompt: '' });
  renderPromptTemplates();
}

function removePromptTemplate(i) {
  if (!aiConfig?.prompts) return;
  aiConfig.prompts.splice(i, 1);
  renderPromptTemplates();
}

async function saveAISettings() {
  const statusEl = document.getElementById('ai-settings-status');
  // Collect prompt edits
  const prompts = (aiConfig?.prompts || []).map((p, i) => ({
    name: document.querySelector(`[data-prompt-name="${i}"]`)?.value || p.name,
    prompt: document.querySelector(`[data-prompt-text="${i}"]`)?.value || p.prompt,
  }));
  const apiKey = document.getElementById('ai-key-input').value.trim();
  const model = document.getElementById('ai-settings-model').value;
  const body = {
    active_provider: settingsActiveProvider,
    active_model: model,
    system_instructions: document.getElementById('ai-system-instructions').value,
    prompts: prompts,
    providers: {},
  };
  body.providers[settingsActiveProvider] = { default_model: model };
  if (apiKey) body.providers[settingsActiveProvider].api_key = apiKey;
  try {
    const res = await fetch(API + '/ai/config', { method: 'PUT', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    if (!res.ok) throw new Error('Save failed');
    statusEl.classList.remove('hidden','text-error'); statusEl.classList.add('text-secondary'); statusEl.textContent = 'Saved!';
    await loadAIConfig();
    updateChatUI();
    setTimeout(() => statusEl.classList.add('hidden'), 2000);
  } catch (e) {
    statusEl.classList.remove('hidden','text-secondary'); statusEl.classList.add('text-error'); statusEl.textContent = e.message;
  }
}

// --- Chat UI ---
function updateChatUI() {
  if (!aiConfig || !aiProviders) return;
  const provSel = document.getElementById('ai-provider-select');
  const modelSel = document.getElementById('ai-model-select');
  const subtitle = document.getElementById('ai-subtitle');
  const dot = document.getElementById('ai-status-dot');
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');

  // Populate provider dropdown
  provSel.innerHTML = '';
  let hasAnyKey = false;
  for (const [key, p] of Object.entries(aiProviders)) {
    if (aiConfig.providers?.[key]?.api_key_set) {
      provSel.innerHTML += `<option value="${key}" ${key === aiConfig.active_provider ? 'selected' : ''}>${p.label}</option>`;
      hasAnyKey = true;
    }
  }
  if (!hasAnyKey) {
    provSel.innerHTML = '<option value="">No API keys</option>';
    subtitle.textContent = 'Click settings to add an API key';
    dot.classList.remove('bg-secondary'); dot.classList.add('bg-outline');
    input.disabled = true; sendBtn.disabled = true;
    return;
  }

  // Populate model dropdown
  const selProv = provSel.value;
  const models = aiProviders[selProv]?.models || [];
  const activeModel = aiConfig.active_model || aiProviders[selProv]?.default_model || models[0];
  modelSel.innerHTML = models.map(m => `<option value="${m}" ${m === activeModel ? 'selected' : ''}>${m}</option>`).join('');

  subtitle.textContent = currentMatchId ? 'Ready to analyze match' : 'Select a match first';
  dot.classList.remove('bg-outline'); dot.classList.add('bg-secondary');
  input.disabled = !currentMatchId; sendBtn.disabled = !currentMatchId;

  // Show prompt suggestions
  renderSuggestions();
}

/* Was an assignment with no declaration, which quietly created a global.
   A module is strict mode, where that is a ReferenceError. */
const providerChanged = () => {
  const prov = document.getElementById('ai-provider-select').value;
  if (!prov || !aiProviders) return;
  const models = aiProviders[prov]?.models || [];
  const def = aiConfig?.providers?.[prov]?.default_model || aiProviders[prov]?.default_model || models[0];
  const sel = document.getElementById('ai-model-select');
  sel.innerHTML = models.map(m => `<option value="${m}" ${m === def ? 'selected' : ''}>${m}</option>`).join('');
};
document.getElementById('ai-provider-select').addEventListener('change', providerChanged);

function renderSuggestions() {
  const cont = document.getElementById('chat-suggestions');
  const msgs = document.getElementById('chat-messages');
  const prompts = aiConfig?.prompts || [];
  if (!prompts.length || !currentMatchId || msgs.children.length > 0) { cont.classList.add('hidden'); return; }
  const colors = ['primary', 'secondary', 'tertiary'];
  cont.innerHTML = '';
  prompts.forEach((p, i) => {
    const btn = document.createElement('button');
    btn.className = `px-3 py-1.5 rounded-full border border-${colors[i % 3]}/20 text-[10px] font-bold uppercase tracking-widest text-${colors[i % 3]} hover:bg-${colors[i % 3]}/10 transition-colors`;
    btn.textContent = p.name;
    btn.onclick = () => sendChat(p.prompt);
    cont.appendChild(btn);
  });
  cont.classList.remove('hidden');
}

function appendMessage(role, content) {
  const cont = document.getElementById('chat-messages');
  const div = document.createElement('div');
  if (role === 'user') {
    div.className = 'flex gap-3 justify-end';
    div.innerHTML = `<div class="bg-primary/10 p-3 rounded-tl-xl rounded-bl-xl rounded-br-xl text-xs text-on-surface leading-relaxed max-w-[80%]">${escapeHtml(content)}</div>`;
  } else if (role === 'assistant') {
    div.className = 'flex gap-3';
    div.innerHTML = `
      <div class="w-8 h-8 rounded bg-surface-container-highest flex items-center justify-center shrink-0">
        <span class="material-symbols-outlined text-primary text-sm">smart_toy</span>
      </div>
      <div class="bg-surface-container-highest/50 p-3 rounded-tr-xl rounded-br-xl rounded-bl-xl text-xs text-on-surface-variant leading-relaxed max-w-[80%] whitespace-pre-wrap">${formatAIResponse(content)}</div>`;
    // Auto-highlight callouts on minimap
    const callouts = extractCalloutsFromText(content);
    if (callouts.length && currentMapName) {
      highlightCalloutsOnMinimap(callouts);
    }
  } else {
    div.className = 'flex gap-3';
    div.innerHTML = `<div class="bg-error/10 p-3 rounded-xl text-xs text-error leading-relaxed">${escapeHtml(content)}</div>`;
  }
  cont.appendChild(div);
  cont.scrollTop = cont.scrollHeight;
  document.getElementById('chat-suggestions').classList.add('hidden');
}

function showTyping() {
  const cont = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.id = 'typing-indicator';
  div.className = 'flex gap-3';
  div.innerHTML = `
    <div class="w-8 h-8 rounded bg-surface-container-highest flex items-center justify-center shrink-0">
      <span class="material-symbols-outlined text-primary text-sm">smart_toy</span>
    </div>
    <div class="bg-surface-container-highest/50 p-3 rounded-tr-xl rounded-br-xl rounded-bl-xl text-xs text-on-surface-variant flex gap-1 items-center">
      <div class="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style="animation-delay:0ms"></div>
      <div class="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style="animation-delay:150ms"></div>
      <div class="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style="animation-delay:300ms"></div>
    </div>`;
  cont.appendChild(div);
  cont.scrollTop = cont.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

function formatAIResponse(text) {
  // Basic markdown-like formatting
  let html = escapeHtml(text);
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong class="text-on-surface">$1</strong>');
  html = html.replace(/`(.+?)`/g, '<code class="bg-surface-container-highest px-1 py-0.5 rounded text-secondary text-[10px]">$1</code>');
  return html;
}

async function sendChat(text) {
  if (!currentMatchId) return;
  const input = document.getElementById('chat-input');
  const msg = text || input.value.trim();
  if (!msg) return;
  input.value = '';

  const provider = document.getElementById('ai-provider-select').value;
  const model = document.getElementById('ai-model-select').value;

  appendMessage('user', msg);
  showTyping();
  document.getElementById('chat-input').disabled = true;
  document.getElementById('chat-send').disabled = true;

  try {
    const res = await fetch(API + '/matches/' + encodeURIComponent(currentMatchId) + '/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ message: msg, provider, model }),
    });
    hideTyping();
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Request failed'); }
    const data = await res.json();
    appendMessage('assistant', data.content);
  } catch (err) {
    hideTyping();
    appendMessage('error', err.message);
  } finally {
    document.getElementById('chat-input').disabled = false;
    document.getElementById('chat-send').disabled = false;
    document.getElementById('chat-input').focus();
  }
}

async function loadChatHistory() {
  if (!currentMatchId) return;
  try {
    const res = await fetch(API + '/matches/' + encodeURIComponent(currentMatchId) + '/chat');
    if (!res.ok) return;
    const data = await res.json();
    const cont = document.getElementById('chat-messages');
    cont.innerHTML = '';
    for (const msg of data.messages) {
      appendMessage(msg.role, msg.content);
    }
  } catch (e) { console.error('Failed to load chat history:', e); }
}

async function clearChat() {
  if (!currentMatchId) return;
  try {
    await fetch(API + '/matches/' + encodeURIComponent(currentMatchId) + '/chat', { method: 'DELETE' });
    document.getElementById('chat-messages').innerHTML = '';
    renderSuggestions();
  } catch (e) { console.error('Failed to clear chat:', e); }
}

// Enter to send
document.getElementById('chat-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

// --- Match Detail Loading ---
async function loadMatchDetail() {
  const params = new URLSearchParams(window.location.search);
  currentMatchId = params.get('id');

  // Load AI config in parallel
  await loadAIConfig();
  updateChatUI();

  if (!currentMatchId) {
    await loadMatchList();
    return;
  }

  try {
    const [matchRes, avgRes] = await Promise.all([
      fetch(API + '/matches/' + encodeURIComponent(currentMatchId)),
      fetch(API + '/matches/career-averages'),
    ]);
    if (!matchRes.ok) throw new Error('Match not found');
    const m = await matchRes.json();
    const careerAvg = avgRes.ok ? (await avgRes.json()).averages || {} : {};
    renderMatchDetail(m);
    applyTrendIndicators(m, careerAvg);
    // Set up 2D replay link & load replay
    const replayLink = document.getElementById('replay-link');
    if (replayLink) replayLink.classList.remove('hidden');
    initReplayPlayer();
    // Load chat history after match renders
    await loadChatHistory();
    updateChatUI();
  } catch (err) {
    console.error('Failed to load match:', err);
  }
}

function renderMatchDetail(m) {
  currentMatchData = m;
  // Store match data globally
  currentMapName = m.map_name || '';
  const matchResult = (m.match_result || '').toLowerCase();
  const isMatchWin = matchResult === 'win' || matchResult === 'victory';
  document.documentElement.dataset.matchResult = isMatchWin ? 'win' : 'loss';

  // Hero section
  const heroSection = document.querySelector('section.border-l-8');
  if (heroSection) {
    heroSection.classList.remove('border-secondary', 'border-good', 'border-error');
    heroSection.classList.add(isMatchWin ? 'border-good' : 'border-error');
  }

  const mapBadge = document.querySelector('.bg-secondary\\/10.text-secondary');
  if (mapBadge) mapBadge.textContent = 'MAP: ' + (m.map_name || '').toUpperCase();

  // Map panel on the left of the hero banner. Hidden rather than left empty
  // when there is no icon for the map, and the banner takes its own left
  // padding back so the score does not end up flush against the edge.
  const mapPanel = document.getElementById('match-map-panel');
  const mapIcon = document.getElementById('match-map-icon');
  if (mapPanel && mapIcon) {
    const url = mapIconUrl(m.map_name);
    const show = !!url;
    mapPanel.classList.toggle('hidden', !show);
    if (heroSection) heroSection.classList.toggle('pl-8', !show);
    if (show) {
      mapIcon.alt = mapLabel(m.map_name);
      mapIcon.onerror = () => {
        mapPanel.classList.add('hidden');
        if (heroSection) heroSection.classList.add('pl-8');
      };
      mapIcon.src = url;
    }
  }

  // Date badge
  const dateBadge = document.getElementById('match-date-badge');
  if (dateBadge && m.date) {
    dateBadge.textContent = m.date;
    dateBadge.classList.remove('hidden');
  }

  const matchIdBadge = document.querySelector('.text-on-surface-variant.font-label');
  if (matchIdBadge) matchIdBadge.textContent = 'MATCH_ID: #' + (m.match_id || '').substring(0, 8);

  const partialBadge = document.getElementById('partial-import-badge');
  const reimportBtn = document.getElementById('reimport-match-btn');
  if (partialBadge) {
    if (m.partial_import) {
      const warningText = m.parse_warning ? ` title="${esc(m.parse_warning)}"` : '';
      partialBadge.classList.remove('hidden');
      partialBadge.innerHTML = `<span class="inline-flex items-center gap-1"${warningText}><span class="material-symbols-outlined text-[11px]">warning</span>PARTIAL IMPORT</span>`;
    } else {
      partialBadge.classList.add('hidden');
    }
  }
  if (reimportBtn) {
    if (m.partial_import) reimportBtn.classList.remove('hidden');
    else reimportBtn.classList.add('hidden');
  }

  // Score and result
  const scoreEl = document.querySelector('h1.text-5xl');
  if (scoreEl) {
    const result = m.match_result || '';
    const resultColor = isMatchWin ? 'text-good' : 'text-error';
    scoreEl.innerHTML = `${m.team_score || 0} <span class="text-primary">:</span> ${m.enemy_score || 0} <span class="${resultColor} ml-4 text-2xl font-light tracking-widest">${result.toUpperCase()}</span>`;
  }

  // Hero stats
  const heroStats = document.querySelectorAll('.flex.gap-12 > div');
  if (heroStats[0]) heroStats[0].querySelector('.text-4xl').textContent = (m.hltv_rating || 0).toFixed(2);
  if (heroStats[1]) heroStats[1].querySelector('.text-4xl').textContent = (m.adr || 0).toFixed(1);
  if (heroStats[2]) heroStats[2].querySelector('.text-4xl').textContent = (m.impact || 0).toFixed(2);

  // Parse enriched data from round_stats
  const enrichedRounds = (m.round_stats || []).map(r => {
    let enriched = {};
    if (r.enriched_json) {
      try { enriched = JSON.parse(r.enriched_json); } catch(e) {}
    }
    return { ...r, enriched };
  });

  renderMechanics(m, enrichedRounds);
  renderUtility(enrichedRounds);
  renderPatterns(m, enrichedRounds);
  renderSideAnalysis(m, enrichedRounds);
  renderAimRoles(m);
  renderRoles(m);
  renderBehavioralAxes(m);
  renderUtilityEconomics(m);
  renderBenchmarks(m);
  _drawFlashScatterChart(enrichedRounds);
  renderScoreboard(m.my_team || [], m.enemy_team || []);
  renderEconomyTimeline(enrichedRounds);
  renderRoundTimeline(enrichedRounds, m);

  // Store enriched rounds globally for replay round list
  window._enrichedRounds = enrichedRounds;

  // Auto-open minimap on round 1
  openMinimap(1);
}

// --- Stat bar helper ---
function renderBar(container, value, max, colorClass) {
  const segments = 5;
  const filled = Math.round((value / max) * segments);
  container.innerHTML = '';
  for (let i = 0; i < segments; i++) {
    const seg = document.createElement('div');
    seg.className = `flex-1 rounded-sm ${i < filled ? colorClass : 'bg-surface-container-highest'}`;
    container.appendChild(seg);
  }
}

// --- Trend indicator helper ---
function _trendBadge(matchVal, avgVal, higherIsBetter = true) {
  if (avgVal == null || avgVal === 0 || matchVal == null) return '';
  const diff = matchVal - avgVal;
  const pct = Math.abs(diff / avgVal * 100);
  if (pct < 2) return '<span class="ml-1.5 text-[9px] text-on-surface-variant font-bold" title="avg: ' + avgVal.toFixed(1) + '">≈</span>';
  const positive = higherIsBetter ? diff > 0 : diff < 0;
  const arrow = diff > 0 ? '▲' : '▼';
  const color = positive ? 'text-good' : 'text-error';
  return '<span class="ml-1.5 text-[9px] font-bold ' + color + '" title="avg: ' + avgVal.toFixed(1) + '">' + arrow + ' ' + pct.toFixed(0) + '%</span>';
}

function applyTrendIndicators(m, avg) {
  if (!avg || Object.keys(avg).length === 0) return;

  // Parse enriched rounds (same as renderMatchDetail)
  const rounds = (m.round_stats || []).map(r => {
    let enriched = {};
    if (r.enriched_json) { try { enriched = JSON.parse(r.enriched_json); } catch(e) {} }
    return { ...r, enriched };
  });

  // --- Mechanics ---
  // HS%
  let totalKills = 0, hsKills = 0;
  for (const r of rounds) { for (const k of (r.enriched?.kills_detail || [])) { totalKills++; if (k.headshot) hsKills++; } }
  const hsPct = totalKills > 0 ? (hsKills / totalKills * 100) : 0;
  _injectTrend('stat-hs-pct', hsPct, avg.hs_pct, true);

  // K/D
  _injectTrend('stat-kd', m.kd_ratio || 0, avg.kd_ratio, true);

  // KAST
  _injectTrend('stat-kast-val', m.kast || 0, avg.kast, true);

  // --- Utility (quick card) ---
  let mFlashed = 0, mBlindDur = 0, mBlindHits = 0, mHeDmg = 0, mMollyDmg = 0;
  for (const r of rounds) {
    const u = r.enriched?.utility || {};
    mFlashed += u.enemies_flashed || 0;
    if ((u.enemies_flashed || 0) > 0 && (u.avg_blind_duration || 0) > 0) {
      mBlindDur += u.avg_blind_duration * u.enemies_flashed;
      mBlindHits += u.enemies_flashed;
    }
    mHeDmg += u.he_damage || 0;
    for (const md of (u.molotov_damage || [])) mMollyDmg += md.damage || 0;
  }
  _injectTrend('stat-flash-count', mFlashed, avg.enemies_flashed, true);
  _injectTrend('stat-blind-dur', mBlindHits > 0 ? mBlindDur / mBlindHits : 0, avg.avg_blind_duration, true);
  _injectTrend('stat-he-dmg', mHeDmg, avg.he_damage, true);
  _injectTrend('stat-molly-dmg', mMollyDmg, avg.molotov_damage, true);

  // --- Pattern Recognition ---
  let clutchWon = 0, clutchTotal = 0;
  for (const r of rounds) { const c = r.enriched?.clutch; if (c) { clutchTotal++; if (c.won) clutchWon++; } }
  if (clutchTotal > 0) _injectTrend('stat-clutch', clutchWon / clutchTotal * 100, avg.clutch_win_pct, true);

  let deaths = 0, traded = 0;
  for (const r of rounds) { if (r.deaths > 0) { deaths++; if (r.traded) traded++; } }
  _injectTrend('stat-trade', deaths > 0 ? traded / deaths * 100 : 0, avg.trade_pct, true);

  let openK = 0, openD = 0;
  for (const r of rounds) { const od = r.enriched?.opening_duel; if (od) { if (od.role === 'opening_kill') openK++; else if (od.role === 'opening_death') openD++; } }
  const odTotal = openK + openD;
  if (odTotal > 0) _injectTrend('stat-opening', openK / odTotal * 100, avg.opening_kill_rate, true);

  const mk = (m.rounds_2k||0)+(m.rounds_3k||0)+(m.rounds_4k||0)+(m.rounds_5k||0);
  _injectTrend('stat-multikill', mk, avg.multikill_rounds, true);

  // --- Aim ---
  const aim = m.aim_stats ? (typeof m.aim_stats === 'string' ? JSON.parse(m.aim_stats) : m.aim_stats) : null;
  // Compare like with like: the cards show medians, so the trend arrows have
  // to read medians too. Older matches only stored a mean.
  const headline = (b) => (b == null ? null : (b.median != null ? b.median : b.avg));
  if (aim?.aim_rating != null) _injectTrend('aim-score', aim.aim_rating, avg.aim_rating, true);
  if (headline(aim?.movement) != null) _injectTrend('aim-mov-avg', headline(aim.movement), avg.movement_avg, false);
  if (headline(aim?.ttk) != null) _injectTrend('aim-ttk-avg', headline(aim.ttk), avg.ttk_avg, false);
  if (headline(aim?.preaim) != null) _injectTrend('aim-preaim-avg', headline(aim.preaim), avg.preaim_avg, false);
  if (headline(aim?.reaction) != null) _injectTrend('aim-rxn-avg', headline(aim.reaction), avg.reaction_avg, false);

  // --- Utility: Use Rate ---
  const ud = m.utility_data ? (typeof m.utility_data === 'string' ? JSON.parse(m.utility_data) : m.utility_data) : null;
  if (ud?.economics?.use_rate != null) _injectTrend('util-use-rate', ud.economics.use_rate, avg.use_rate, true);
}

/* Both injectors prefer a dedicated slot when the card declares one.
   Appending straight into the row was what threw the Utility card's alignment
   out: the rows are flex, so every row ended up with a different number of
   children depending on which badges happened to apply, and justify-between
   then spread the values to different positions on every line. A slot is a
   cell that is always present, empty or not, so the columns stay put. Cards
   without slots keep the original inline behaviour. */
function _statSlot(kind, elementId) {
  return document.querySelector('[data-' + kind + '-slot="' + elementId + '"]');
}

function _injectTrend(elementId, matchVal, avgVal, higherIsBetter) {
  const el = document.getElementById(elementId);
  if (!el) return;
  const badge = _trendBadge(matchVal, avgVal, higherIsBetter);
  const slot = _statSlot('trend', elementId);
  if (slot) { slot.innerHTML = badge; return; }
  if (badge) el.insertAdjacentHTML('afterend', badge);
}

// --- Mechanics ---
function renderMechanics(m, rounds) {
  // HS%
  let totalKills = 0, hsKills = 0;
  for (const r of rounds) {
    for (const k of (r.enriched?.kills_detail || [])) {
      totalKills++;
      if (k.headshot) hsKills++;
    }
  }
  const hsPct = totalKills > 0 ? (hsKills / totalKills * 100) : 0;
  document.getElementById('stat-hs-pct').textContent = hsPct.toFixed(1) + '%';
  renderBar(document.getElementById('stat-hs-bar'), hsPct, 100, 'bg-secondary');

  // K/D
  const kd = m.kd_ratio || 0;
  document.getElementById('stat-kd').textContent = kd.toFixed(2);
  renderBar(document.getElementById('stat-kd-bar'), Math.min(kd, 2.5), 2.5, 'bg-primary');

  // KAST
  const kast = m.kast || 0;
  document.getElementById('stat-kast-val').textContent = kast.toFixed(1) + '%';
  renderBar(document.getElementById('stat-kast-bar'), kast, 100, 'bg-tertiary');
}

// --- Utility ---
function renderUtility(rounds) {
  let totalFlashed = 0, totalBlindDur = 0, blindHits = 0, heDmg = 0, mollyDmg = 0;
  for (const r of rounds) {
    const u = r.enriched?.utility || {};
    totalFlashed += u.enemies_flashed || 0;
    if (u.enemies_flashed > 0 && u.avg_blind_duration > 0) {
      totalBlindDur += u.avg_blind_duration * u.enemies_flashed;
      blindHits += u.enemies_flashed;
    }
    heDmg += u.he_damage || 0;
    for (const md of (u.molotov_damage || [])) mollyDmg += md.damage || 0;
  }
  const avgBlind = blindHits > 0 ? (totalBlindDur / blindHits) : 0;
  // Numbers only — the unit is a static cell of the grid so the figures can
  // line up on their own column.
  document.getElementById('stat-flash-count').textContent = totalFlashed;
  document.getElementById('stat-blind-dur').textContent = avgBlind.toFixed(1);
  document.getElementById('stat-he-dmg').textContent = heDmg;
  document.getElementById('stat-molly-dmg').textContent = mollyDmg;
}

// --- Patterns ---
function renderPatterns(m, rounds) {
  // Clutch
  let clutchWon = 0, clutchTotal = 0;
  for (const r of rounds) {
    const c = r.enriched?.clutch;
    if (c) { clutchTotal++; if (c.won) clutchWon++; }
  }
  if (clutchTotal > 0) {
    document.getElementById('stat-clutch').textContent = Math.round(clutchWon / clutchTotal * 100) + '%';
    document.getElementById('stat-clutch-detail').textContent = `${clutchWon}/${clutchTotal} clutches won`;
  } else {
    document.getElementById('stat-clutch').textContent = '—';
    document.getElementById('stat-clutch-detail').textContent = 'No clutch situations';
  }

  // Trade %
  let deaths = 0, traded = 0;
  for (const r of rounds) {
    if (r.deaths > 0) { deaths++; if (r.traded) traded++; }
  }
  const tradePct = deaths > 0 ? Math.round(traded / deaths * 100) : 0;
  document.getElementById('stat-trade').textContent = tradePct + '%';
  document.getElementById('stat-trade-detail').textContent = `${traded}/${deaths} deaths traded`;

  // Opening duels
  let openKills = 0, openDeaths = 0;
  for (const r of rounds) {
    const od = r.enriched?.opening_duel;
    if (od) {
      if (od.role === 'opening_kill') openKills++;
      else if (od.role === 'opening_death') openDeaths++;
    }
  }
  const openTotal = openKills + openDeaths;
  if (openTotal > 0) {
    document.getElementById('stat-opening').textContent = openKills + '/' + openTotal;
    document.getElementById('stat-opening-detail').textContent = `${Math.round(openKills / openTotal * 100)}% opening kill rate`;
  } else {
    document.getElementById('stat-opening').textContent = '—';
    document.getElementById('stat-opening-detail').textContent = 'No first kills';
  }

  // Multi-kills
  const mk = (m.rounds_2k||0) + (m.rounds_3k||0) + (m.rounds_4k||0) + (m.rounds_5k||0);
  const mkParts = [];
  if (m.rounds_5k) mkParts.push(m.rounds_5k + '×5K');
  if (m.rounds_4k) mkParts.push(m.rounds_4k + '×4K');
  if (m.rounds_3k) mkParts.push(m.rounds_3k + '×3K');
  if (m.rounds_2k) mkParts.push(m.rounds_2k + '×2K');
  document.getElementById('stat-multikill').textContent = mk;
  document.getElementById('stat-multikill-detail').textContent = mkParts.join(', ') || 'None';
}

// --- Side Analysis ---
function renderSideAnalysis(m, rounds) {
  let ctWins = 0, ctLosses = 0, ctKills = 0, ctDeaths = 0;
  let tWins = 0, tLosses = 0, tKills = 0, tDeaths = 0;
  let pistolWins = 0, pistolTotal = 0;

  for (const r of rounds) {
    const side = r.enriched?.side;
    const winner = r.enriched?.round_winner;
    if (!side) continue;
    const playerTeam = side === 'CT' ? 'CT' : 'T';
    const won = winner === playerTeam;

    const isPistol = r.enriched?.economy?.buy_type === 'PISTOL';
    if (isPistol) { pistolTotal++; if (won) pistolWins++; }

    if (side === 'CT') {
      if (won) ctWins++; else ctLosses++;
      ctKills += r.kills || 0;
      ctDeaths += r.deaths || 0;
    } else {
      if (won) tWins++; else tLosses++;
      tKills += r.kills || 0;
      tDeaths += r.deaths || 0;
    }
  }

  const ctTotal = ctWins + ctLosses;
  const tTotal = tWins + tLosses;

  document.getElementById('side-ct-score').textContent = `${ctWins} : ${ctLosses}`;
  document.getElementById('side-ct-bar').style.width = ctTotal > 0 ? Math.round(ctWins / ctTotal * 100) + '%' : '0%';
  document.getElementById('side-ct-stats').innerHTML =
    `<div>${ctKills}K / ${ctDeaths}D across ${ctTotal} rounds</div>`;

  document.getElementById('side-t-score').textContent = `${tWins} : ${tLosses}`;
  document.getElementById('side-t-bar').style.width = tTotal > 0 ? Math.round(tWins / tTotal * 100) + '%' : '0%';
  document.getElementById('side-t-stats').innerHTML =
    `<div>${tKills}K / ${tDeaths}D across ${tTotal} rounds</div>`;

  document.getElementById('pistol-text').textContent = `${pistolWins}/${pistolTotal} pistol rounds won`;
}

// --- Aim & Roles ---
function renderAimRoles(m) {
  const aim = m.aim_stats;
  if (!aim || (!aim.movement && !aim.preaim && !aim.ttk && !aim.reaction)) return;
  buildAimKpiMeta(aim.thresholds);
  document.getElementById('aim-roles-section').classList.remove('hidden');

  // Overall aim rating. Null means nothing measurable was found in this match
  // — distinct from a genuine zero, so it must not render as one.
  const rating = aim.aim_rating;
  const ratingEl = document.getElementById('aim-score');
  const barEl = document.getElementById('aim-score-bar');
  if (rating == null) {
    ratingEl.textContent = '—';
    ratingEl.className = ratingEl.className.replace(/text-\S+/g, '') + ' text-on-surface-variant';
    document.getElementById('aim-score-label').textContent = 'Not measurable';
    barEl.style.width = '0%';
    document.getElementById('aim-rating-badge').textContent = 'AIM: n/a';
  } else {
    ratingEl.textContent = rating.toFixed(0);
    const ratingLabel = rating >= 70 ? 'Excellent' : rating >= 50 ? 'Good' : rating >= 30 ? 'Average' : 'Needs Work';
    const ratingColor = rating >= 70 ? 'text-good' : rating >= 50 ? 'text-accent' : rating >= 30 ? 'text-warn' : 'text-error';
    ratingEl.className = ratingEl.className.replace(/text-\S+/g, '') + ' ' + ratingColor;
    document.getElementById('aim-score-label').textContent = ratingLabel;
    barEl.style.width = rating + '%';
    const barColor = rating >= 70 ? 'bg-good' : rating >= 50 ? 'bg-accent' : rating >= 30 ? 'bg-warn' : 'bg-error';
    barEl.className = 'h-full rounded-full transition-all ' + barColor;
    document.getElementById('aim-rating-badge').textContent = 'AIM: ' + rating.toFixed(0) + '/100';
  }

  // Sample size drives how much each metric was trusted, so show it.
  const sampleNote = (m) => {
    if (!m || m.n == null) return '';
    const tone = m.confidence === 'high' ? 'text-good'
               : m.confidence === 'medium' ? 'text-on-surface-variant'
               : 'text-caution';
    return ` <span class="${tone}">n=${m.n}</span>`;
  };

  // Movement
  const mv = aim.movement || {};
  if (mv.median != null) {
    document.getElementById('aim-mov-avg').textContent = mv.median.toFixed(0);
    document.getElementById('aim-mov-range').innerHTML = 'Median of ' + mv.min.toFixed(0) + ' – ' + mv.max.toFixed(0) + ' u/s' + sampleNote(mv);
    document.getElementById('aim-mov-cs').textContent = (mv.counterstrafed_pct || 0).toFixed(0) + '% C-Strafed';
    document.getElementById('aim-mov-standing').textContent = (mv.standing_pct || 0).toFixed(0) + '% Static';
    document.getElementById('aim-mov-stopped').textContent = (mv.stopped_pct || 0).toFixed(0) + '% Coasted';
    document.getElementById('aim-mov-running').textContent = (mv.running_pct || 0).toFixed(0) + '% Running';
    drawStripChart('aim-mov-chart', mv.speeds || [], mv.median, {
      thresholds: aimBounds('movement'),
      range: aimRange('movement'),
      colors: [TC.amber||'#fbbf24', TC.success||'#34d399', TC.ct||'#60a5fa', TC.error||'#ef4444'],
      bgColor: TC.bg||'#0f1930',
      avgColor: TC.cyan||'#53ddfc',
      lowPenalty: mv.low_penalty || [],
      weapons: mv.weapons || [],
      outcomes: mv.outcomes || [],
    });
    // Gun context note for running kills
    const noteEl = document.getElementById('aim-mov-gun-note');
    if (noteEl && mv.running_total > 0) {
      const lpn = mv.running_low_penalty || 0;
      if (lpn > 0) {
        noteEl.textContent = lpn + ' of ' + mv.running_total + ' running kill' + (mv.running_total > 1 ? 's' : '') + ' with SMG/shotgun/pistol';
        noteEl.classList.remove('hidden');
      }
    }
  }

  // Peek speed — how much speed was carried into the duel. Drawn without tier
  // bands: on its own the number is not good or bad, it is the context the
  // counter-strafe below it has to be read against.
  const peek = aim.peek || {};
  if (peek.median != null) {
    document.getElementById('aim-peek-block').classList.remove('hidden');
    document.getElementById('aim-peek-avg').textContent = peek.median.toFixed(0);
    document.getElementById('aim-peek-range').innerHTML =
      'Peak of ' + peek.min.toFixed(0) + ' – ' + peek.max.toFixed(0)
      + ' u/s in the 0.5s before firing' + sampleNote(peek);
    drawStripChart('aim-peek-chart', peek.values || [], peek.median, {
      thresholds: aimBounds('peek'),
      zones: aimZones('peek'),
      range: aimRange('peek'),
      colors: [TC.amber||'#fbbf24', TC.success||'#34d399', TC.ct||'#60a5fa', TC.error||'#ef4444'],
      bgColor: TC.bg||'#0f1930',
      avgColor: TC.cyan||'#53ddfc',
      outcomes: peek.outcomes || [],
    });
    // Legend for the chart above: one chip per shaded region, carrying that
    // region's shade, so a band and the number describing it are visibly the
    // same thing rather than two colour schemes for one split.
    const byZone = peek.by_zone || [];
    const zoneSpan = (i) => (byZone[i + 1]
      ? byZone[i].at + '–' + byZone[i + 1].at + ' u/s'
      : byZone[i].at + '+ u/s');
    document.getElementById('aim-peek-split').innerHTML = byZone.map((z, i) =>
      '<span class="px-1.5 py-0.5 rounded-full text-[8px] font-bold" style="'
      + _zoneChipStyle(i) + '" title="' + z.n + ' of ' + peek.n
      + ' engagements entered at ' + zoneSpan(i) + '">'
      + esc(z.label) + ' ' + z.pct.toFixed(0) + '%</span>'
    ).join('');

    // Counter-strafe rate split by how fast the peek was. This is the whole
    // point of measuring peek speed: a stop that holds off a walk but not off
    // a full-speed peek is a specific thing to practise, and the pooled rate
    // hides it completely.
    const buckets = (mv.counterstrafe_by_peek || []).filter(b => b.attempts > 0);
    const csEl = document.getElementById('aim-peek-cs');
    const bucketEl = document.getElementById('aim-peek-cs-buckets');
    if (buckets.length) {
      csEl.textContent = 'Rifle stops counter-strafed, by peek speed';
      // Chip body carries the peek zone it belongs to; only the rate inside is
      // graded, on the same bands and the same palette as the benchmark badge
      // that grades the pooled version of this number — one rate, one colour.
      const bands = aimBounds('counterstrafe');
      const tierKeys = ['pro', 'high_amateur', 'average', 'below_average'];
      const rateColor = (rate) => {
        const i = bands.findIndex(b => rate >= b);
        const cfg = _TIER_CONFIG[tierKeys[i < 0 ? tierKeys.length - 1 : i]];
        return TC[cfg.tcText] || cfg.text;
      };
      bucketEl.innerHTML = buckets.map(b => {
        const zi = _peekZoneIndexAt(byZone, b.min);
        const span = b.max == null ? b.min + '+' : b.min + '–' + b.max;
        return '<span class="px-1.5 py-0.5 rounded-full text-[8px] font-bold" style="'
          + _zoneChipStyle(zi) + '" title="' + esc(b.label) + ' peeks (' + span + ' u/s): '
          + b.good + ' of ' + b.attempts + ' stopped properly">'
          + esc(b.label) + ' <span style="color:' + rateColor(b.rate) + '">'
          + b.rate.toFixed(0) + '%</span> <span class="opacity-60">'
          + b.good + '/' + b.attempts + '</span></span>';
      }).join('');
    } else {
      csEl.textContent = '';
      bucketEl.innerHTML = '';
    }
  }

  // Engagement Time
  const ttk = aim.ttk || {};
  if (ttk.median != null) {
    document.getElementById('aim-ttk-avg').textContent = ttk.median.toFixed(2);
    document.getElementById('aim-ttk-range').innerHTML = 'Median of ' + ttk.min.toFixed(2) + 's – ' + ttk.max.toFixed(2) + 's' + sampleNote(ttk);
    document.getElementById('aim-ttk-note').textContent =
      (ttk.values || []).length + ' engagements (first shot → kill)'
      + (ttk.excluded_outliers ? ', ' + ttk.excluded_outliers + ' over 1s excluded' : '');
    drawStripChart('aim-ttk-chart', ttk.values || [], ttk.median, {
      thresholds: aimBounds('ttk'),
      range: aimRange('ttk'),
      colors: [TC.amber||'#fbbf24', TC.success||'#34d399', TC.ct||'#60a5fa', TC.error||'#ef4444'],
      bgColor: TC.bg||'#0f1930',
      avgColor: TC.cyan||'#53ddfc',
      outcomes: ttk.outcomes || [],
    });
    if (ttk.total_shots && ttk.total_hits) {
      document.getElementById('aim-ttk-accuracy').textContent =
        ttk.total_hits + '/' + ttk.total_shots + ' shots hit (' + (ttk.accuracy_pct || 0).toFixed(0) + '% accuracy in engagements)';
    }
  }

  // Pre-aim
  const pa = aim.preaim || {};
  if (pa.median != null) {
    document.getElementById('aim-preaim-avg').textContent = pa.median.toFixed(1);
    document.getElementById('aim-preaim-range').innerHTML = 'Median of ' + pa.min.toFixed(1) + '° – ' + pa.max.toFixed(1) + '°' + sampleNote(pa);
    document.getElementById('aim-preaim-exc').textContent = (pa.excellent_pct || 0).toFixed(0) + '% <5°';
    document.getElementById('aim-preaim-good').textContent = (pa.good_pct || 0).toFixed(0) + '% <10°';
    document.getElementById('aim-preaim-mod').textContent = (pa.moderate_pct || 0).toFixed(0) + '% <20°';
    document.getElementById('aim-preaim-poor').textContent = (pa.poor_pct || 0).toFixed(0) + '% 20°+';
    drawStripChart('aim-preaim-chart', pa.errors || [], pa.median, {
      thresholds: aimBounds('preaim'),
      range: aimRange('preaim'),
      colors: [TC.amber||'#fbbf24', TC.success||'#34d399', TC.ct||'#60a5fa', TC.error||'#ef4444'],
      bgColor: TC.bg||'#0f1930',
      avgColor: TC.cyan||'#53ddfc',
      outcomes: pa.outcomes || [],
    });
  }

  // Reaction Time
  const rxn = aim.reaction || {};
  if (rxn.median != null && (rxn.values || []).length > 0) {
    document.getElementById('aim-reaction-card').classList.remove('hidden');
    document.getElementById('aim-rxn-avg').textContent = rxn.median.toFixed(0);
    document.getElementById('aim-rxn-range').innerHTML =
      'Median of ' + rxn.min.toFixed(0) + 'ms – ' + rxn.max.toFixed(0) + 'ms' + sampleNote(rxn)
      + ' <span class="text-on-surface-variant/60">· diagnostic, not in rating</span>';
    document.getElementById('aim-rxn-lightning').textContent = (rxn.lightning_pct || 0).toFixed(0) + '% 🏆<150ms';
    document.getElementById('aim-rxn-fast').textContent = (rxn.fast_pct || 0).toFixed(0) + '% <200ms';
    document.getElementById('aim-rxn-average').textContent = (rxn.average_pct || 0).toFixed(0) + '% <300ms';
    document.getElementById('aim-rxn-slow').textContent = (rxn.slow_pct || 0).toFixed(0) + '% 300ms+';
    drawStripChart('aim-rxn-chart', rxn.values || [], rxn.median, {
      thresholds: aimBounds('reaction'),
      range: aimRange('reaction'),
      colors: [TC.amber||'#fbbf24', TC.success||'#34d399', TC.ct||'#60a5fa', TC.error||'#ef4444'],
      bgColor: TC.bg||'#0f1930',
      avgColor: TC.cyan||'#53ddfc',
      outcomes: rxn.outcomes || [],
    });
  }

  // Accuracy
  const acc = aim.accuracy || {};
  if (acc.median != null && (acc.values || []).length > 0) {
    document.getElementById('aim-accuracy-card').classList.remove('hidden');
    // Pooled over every bullet — see the note in _calculate_aim_stats.
    const accHeadline = acc.pooled_pct != null ? acc.pooled_pct : acc.median;
    document.getElementById('aim-acc-avg').textContent = accHeadline.toFixed(0);
    document.getElementById('aim-acc-range').innerHTML =
      (acc.total_shots ? acc.total_hits + '/' + acc.total_shots + ' bullets' : 'Median')
      + sampleNote(acc);
    document.getElementById('aim-acc-fb').textContent =
      'First bullet landed in ' + (acc.first_bullet_pct || 0).toFixed(0) + '% of engagements';
    document.getElementById('aim-acc-head').textContent = (acc.head_pct || 0).toFixed(0) + '% Head';
    document.getElementById('aim-acc-upper').textContent = (acc.upper_pct || 0).toFixed(0) + '% Chest';
    document.getElementById('aim-acc-lower').textContent = (acc.lower_pct || 0).toFixed(0) + '% Limbs';
    // Reference line matches the headline figure rather than a second statistic.
    drawStripChart('aim-acc-chart', acc.values || [], accHeadline, {
      thresholds: aimBounds('accuracy'),
      range: aimRange('accuracy'),
      colors: [TC.amber||'#fbbf24', TC.success||'#34d399', TC.ct||'#60a5fa', TC.error||'#ef4444'],
      bgColor: TC.bg||'#0f1930',
      avgColor: TC.cyan||'#53ddfc',
      outcomes: acc.outcomes || [],
      invert: true,
    });
  }

  // 2D Scatter Plot
  const encounters = aim.encounters || [];
  if (encounters.length > 1) {
    buildAimKpiMeta(aim.thresholds);
    document.getElementById('aim-scatter-section').classList.remove('hidden');
    window._aimEncounters = encounters;
    const xSel = document.getElementById('aim-scatter-x');
    const ySel = document.getElementById('aim-scatter-y');

    // Metrics are measured over different engagements: movement and crosshair
    // placement come from every duel, while engagement time, reaction and
    // accuracy only exist where the player got the kill. Showing each option's
    // count makes that visible instead of leaving the point counts a mystery.
    const counts = {};
    for (const key of Object.keys(AIM_KPI_META)) {
      counts[key] = encounters.filter(e => e[key] !== undefined).length;
    }
    const options = Object.entries(AIM_KPI_META)
      .filter(([key]) => counts[key] > 0)
      .map(([key, meta]) => `<option value="${key}">${meta.label} — ${counts[key]}</option>`)
      .join('');
    const keep = (sel, fallback) => (counts[sel.value] > 0 ? sel.value : fallback);
    const prevX = keep(xSel, 'movement');
    const prevY = keep(ySel, 'preaim');
    xSel.innerHTML = options;
    ySel.innerHTML = options;
    xSel.value = counts[prevX] ? prevX : Object.keys(counts).find(k => counts[k] > 0);
    ySel.value = counts[prevY] ? prevY : xSel.value;

    const redraw = () => drawAimScatter('aim-scatter-canvas', encounters, xSel.value, ySel.value);
    xSel.onchange = redraw;
    ySel.onchange = redraw;
    redraw();
  }
}

// --- Aim KPI bands ---
// Built from aim_stats.thresholds so the scatter grades on exactly the same
// numbers as the cards and the benchmark badges. The literals below are only a
// fallback for matches analysed before the backend started shipping them.




// Bounds for a metric, from the shared table the backend ships.


// Fixed axis span for a metric, when it declares one.


// Named regions of an ungraded axis, when it declares them.


// Shading for those regions. Deliberately one neutral hue at rising alpha
// rather than the tier palette: more shade means more speed, not better or
// worse. Anything drawn in the tier colours is a grade, and peek speed is not.



// The same ramp at chip strength. A 4%-alpha wash reads fine as a band across
// a chart and is invisible on a pill, so the legend keeps the hue and the
// ordering but not the literal alpha — a chip and its band still read as the
// same thing, which is the point of matching them at all.


// Which shaded region a peek-speed boundary belongs to, so a counter-strafe
// chip can take the same shade as the legend chip covering the same speeds.






// --- Roles ---
const _ROLE_PALETTE = [
  '#60a5fa', '#34d399', '#facc15', '#c084fc', '#f472b6',
  '#fb923c', '#a78bfa', '#22d3ee', '#f87171', '#4ade80',
];

function _roleColor(role) {
  // Simple hash → stable index into palette
  let h = 0;
  for (let i = 0; i < role.length; i++) h = ((h << 5) - h + role.charCodeAt(i)) | 0;
  return _ROLE_PALETTE[Math.abs(h) % _ROLE_PALETTE.length];
}

function renderRoles(m) {
  const rd = m.role_data;
  if (!rd || !rd.rounds || !rd.rounds.length) return;
  document.getElementById('roles-section').classList.remove('hidden');

  const shortMap = (rd.map || '').replace('de_', '');
  document.getElementById('roles-map-badge').textContent = shortMap || 'unknown';

  document.getElementById('roles-ct-primary').textContent = rd.ct_primary ? '★ ' + rd.ct_primary : '';
  document.getElementById('roles-t-primary').textContent = rd.t_primary ? '★ ' + rd.t_primary : '';

  _renderRoleSummary('roles-ct-summary', rd.ct_summary || {});
  _renderRoleSummary('roles-t-summary', rd.t_summary || {});

  const ctRounds = rd.rounds.filter(r => r.side === 'CT');
  const tRounds = rd.rounds.filter(r => r.side === 'T');

  _renderRoleStrip('roles-ct-rounds', ctRounds);
  _renderRoleStrip('roles-t-rounds', tRounds);

  _drawSpiderChart('roles-ct-spider', rd.roles_ct || [], ctRounds, TC.ct||'#60a5fa');
  _drawSpiderChart('roles-t-spider', rd.roles_t || [], tRounds, TC.orange||'#fb923c');
}

function _renderRoleSummary(containerId, summary) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const entries = Object.entries(summary).sort((a, b) => b[1] - a[1]);
  el.innerHTML = entries.map(([role, count]) => {
    const color = _roleColor(role);
    return '<span class="text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full" ' +
      'style="background:' + color + '20; color:' + color + '">' +
      role + ' ×' + count + '</span>';
  }).join('');
}

function _renderRoleStrip(containerId, rounds) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = rounds.map(r => {
    const role = r.role || '?';
    const color = _roleColor(role);
    return '<div title="R' + r.round + ': ' + role + '" class="w-5 h-5 rounded-sm flex items-center justify-center text-[7px] font-bold cursor-default" ' +
      'style="background:' + color + '25; color:' + color + '; border: 1px solid ' + color + '40">' +
      r.round + '</div>';
  }).join('');
}

function _drawSpiderChart(canvasId, roles, rounds, tintColor) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !roles.length) return;
  const ctx = canvas.getContext('2d');
  const cw = 400, ch = 400;
  canvas.width = cw * 2; canvas.height = ch * 2;
  ctx.scale(2, 2);

  const cx = cw / 2, cy = ch / 2;
  const maxR = Math.min(cx, cy) - 40;
  const n = roles.length;
  const angleStep = (Math.PI * 2) / n;
  const startAngle = -Math.PI / 2; // top

  // Background
  ctx.fillStyle = TC.bg||'#0f1930';
  ctx.fillRect(0, 0, cw, ch);

  // Grid rings (20%, 40%, 60%, 80%, 100%)
  for (let ring = 1; ring <= 5; ring++) {
    const r = maxR * ring / 5;
    ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.07)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const a = startAngle + angleStep * (i % n);
      const px = cx + Math.cos(a) * r;
      const py = cy + Math.sin(a) * r;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.stroke();
  }

  // Axis lines + labels
  ctx.font = 'bold 9px Space Grotesk, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < n; i++) {
    const a = startAngle + angleStep * i;
    const ex = cx + Math.cos(a) * maxR;
    const ey = cy + Math.sin(a) * maxR;
    ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
    // Label
    const lx = cx + Math.cos(a) * (maxR + 22);
    const ly = cy + Math.sin(a) * (maxR + 22);
    ctx.fillStyle = _roleColor(roles[i]);
    ctx.fillText(roles[i], lx, ly);
  }

  // Per-round polygons (faint)
  const scoredRounds = rounds.filter(r => r.scores && Object.keys(r.scores).length > 0);
  for (const r of scoredRounds) {
    ctx.strokeStyle = hexToRgba(tintColor, 0.12);
    ctx.fillStyle = hexToRgba(tintColor, 0.03);
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const a = startAngle + angleStep * (i % n);
      const v = r.scores[roles[i % n]] || 0;
      const pr = maxR * v;
      const px = cx + Math.cos(a) * pr;
      const py = cy + Math.sin(a) * pr;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  // Average polygon (bold)
  if (scoredRounds.length > 0) {
    const avg = {};
    for (const role of roles) avg[role] = 0;
    for (const r of scoredRounds) {
      for (const role of roles) avg[role] += (r.scores[role] || 0);
    }
    for (const role of roles) avg[role] /= scoredRounds.length;

    ctx.fillStyle = hexToRgba(tintColor, 0.18);
    ctx.strokeStyle = tintColor;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const a = startAngle + angleStep * (i % n);
      const v = avg[roles[i % n]] || 0;
      const pr = maxR * v;
      const px = cx + Math.cos(a) * pr;
      const py = cy + Math.sin(a) * pr;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Dots on average vertices
    for (let i = 0; i < n; i++) {
      const a = startAngle + angleStep * i;
      const v = avg[roles[i]] || 0;
      const pr = maxR * v;
      const px = cx + Math.cos(a) * pr;
      const py = cy + Math.sin(a) * pr;
      ctx.fillStyle = tintColor;
      ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
    }
  }
}



// --- Benchmark Tier Badges ---
// Tier labels describe the band a value falls in, not a population it was
// compared against. The thresholds behind them are hand-set: nothing here has
// been calibrated against real pro or amateur data, so calling a band "PRO"
// asserted something we cannot support. Renamed until percentile calibration
// against a real population replaces the constants.
const _TIER_CONFIG = {
  pro:            { label: 'EXCELLENT',  bg: 'rgba(251,191,36,0.15)', text: '#fbbf24', icon: '★', tcText: 'amber' },
  high_amateur:   { label: 'STRONG',     bg: 'rgba(52,211,153,0.15)', text: '#34d399', icon: '⬆', tcText: 'success' },
  average:        { label: 'FAIR',       bg: 'rgba(96,165,250,0.15)', text: '#60a5fa', icon: '—', tcText: 'ct' },
  below_average:  { label: 'NEEDS WORK', bg: 'rgba(239,68,68,0.15)',  text: '#ef4444', icon: '⬇', tcText: 'error' },
};

function _benchmarkBadgeHTML(tier) {
  const cfg = _TIER_CONFIG[tier];
  if (!cfg) return '';
  const textColor = (TC[cfg.tcText] || cfg.text);
  // No margin here: inline callers get it from .bench-badge, slotted ones take
  // their spacing from the grid gap instead.
  return '<span style="display:inline-flex;align-items:center;gap:3px;padding:1px 7px;border-radius:9999px;' +
    'font-size:8px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;white-space:nowrap;' +
    'background:' + hexToRgba(textColor, 0.15) + ';color:' + textColor + ';vertical-align:middle">' +
    cfg.icon + ' ' + cfg.label + '</span>';
}

function _injectBadge(elementId, tier) {
  const el = document.getElementById(elementId);
  if (!el || !tier) return;

  const slot = _statSlot('bench', elementId);
  if (slot) { slot.innerHTML = _benchmarkBadgeHTML(tier); return; }

  // Remove any existing badge
  const existing = el.parentElement.querySelector('.bench-badge');
  if (existing) existing.remove();
  const badge = document.createElement('span');
  badge.className = 'bench-badge';
  badge.style.marginLeft = '6px';
  badge.innerHTML = _benchmarkBadgeHTML(tier);
  el.parentElement.appendChild(badge);
}

function renderBenchmarks(m) {
  const b = m.benchmarks;
  if (!b) return;

  // Utility quick-stats card
  if (b.enemies_flashed) _injectBadge('stat-flash-count', b.enemies_flashed.tier);
  if (b.utility_damage) {
    // Inject badge next to HE + Molotov label area (use HE as anchor)
    _injectBadge('stat-he-dmg', b.utility_damage.tier);
  }

  // Utility & Economics section
  if (b.utility_waste_pct) _injectBadge('util-use-rate', b.utility_waste_pct.tier);

  // Aim section
  if (b.counterstrafe) _injectBadge('aim-mov-avg', b.counterstrafe.tier);
  if (b.preaim_offset) _injectBadge('aim-preaim-avg', b.preaim_offset.tier);
  if (b.reaction_time) _injectBadge('aim-rxn-avg', b.reaction_time.tier);
  if (b.engagement_ttk) _injectBadge('aim-ttk-avg', b.engagement_ttk.tier);
}

// --- Behavioral Assessment (5-axis radar per side) ---
document.getElementById('beh-info-toggle')?.addEventListener('click', () => {
  document.getElementById('beh-info-panel')?.classList.toggle('hidden');
});

function renderBehavioralAxes(m) {
  const ba = m.behavioral_axes;
  if (!ba) return;
  const section = document.getElementById('behavioral-section');

  const ctAxes = ba.ct?.axes;
  const tAxes = ba.t?.axes;
  if ((!ctAxes || Object.values(ctAxes).every(v => v === 0)) &&
      (!tAxes || Object.values(tAxes).every(v => v === 0))) return;

  section.classList.remove('hidden');

  // CT
  if (ctAxes) {
    _drawBehavioralRadar('beh-ct-radar', ctAxes, TC.ct||'#60a5fa');
    const dominant = Object.entries(ctAxes).sort((a, b) => b[1] - a[1])[0];
    document.getElementById('beh-ct-dominant').textContent = dominant ? '★ ' + dominant[0].charAt(0).toUpperCase() + dominant[0].slice(1) + ' (' + dominant[1] + ')' : '';
    _renderSuccessMetrics('beh-ct-success', ba.ct.success || {}, TC.ct||'#60a5fa');
  }

  // T
  if (tAxes) {
    _drawBehavioralRadar('beh-t-radar', tAxes, TC.orange||'#fb923c');
    const dominant = Object.entries(tAxes).sort((a, b) => b[1] - a[1])[0];
    document.getElementById('beh-t-dominant').textContent = dominant ? '★ ' + dominant[0].charAt(0).toUpperCase() + dominant[0].slice(1) + ' (' + dominant[1] + ')' : '';
    _renderSuccessMetrics('beh-t-success', ba.t.success || {}, TC.orange||'#fb923c');
  }
}

function _drawBehavioralRadar(canvasId, axes, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const labels = ['Aggression', 'Trading', 'Isolation', 'Survival', 'Sniper'];
  const keys = ['aggression', 'trading', 'isolation', 'survival', 'sniper'];
  const values = keys.map(k => (axes[k] || 0) / 100);
  const n = labels.length;
  const cx = w / 2;
  const cy = h / 2;
  const R = Math.min(cx, cy) - 32;
  const angleStep = (Math.PI * 2) / n;
  const startAngle = -Math.PI / 2;

  // Grid rings (5 levels)
  for (let ring = 1; ring <= 5; ring++) {
    const r = R * ring / 5;
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const a = startAngle + i * angleStep;
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = TC.grid || 'rgba(255,255,255,' + (ring === 5 ? 0.12 : 0.05) + ')';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Axis lines + labels
  const icons = ['⚔', '🤝', '👁', '🛡', '🎯'];
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < n; i++) {
    const a = startAngle + i * angleStep;
    const xEnd = cx + Math.cos(a) * R;
    const yEnd = cy + Math.sin(a) * R;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(xEnd, yEnd);
    ctx.strokeStyle = TC.grid || 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.stroke();

    const labelR = R + 20;
    const lx = cx + Math.cos(a) * labelR;
    const ly = cy + Math.sin(a) * labelR;
    ctx.font = '600 9px system-ui, sans-serif';
    ctx.fillStyle = TC.gridText || 'rgba(255,255,255,0.5)';
    ctx.fillText(icons[i] + ' ' + labels[i], lx, ly);
  }

  // Data polygon
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const a = startAngle + i * angleStep;
    const v = Math.max(values[i], 0.04);
    const x = cx + Math.cos(a) * R * v;
    const y = cy + Math.sin(a) * R * v;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = hexToRgba(color, 0.15);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Vertices + value labels
  for (let i = 0; i < n; i++) {
    const a = startAngle + i * angleStep;
    const v = Math.max(values[i], 0.04);
    const x = cx + Math.cos(a) * R * v;
    const y = cy + Math.sin(a) * R * v;
    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    ctx.strokeStyle = TC.dotStroke || '#0a1628'; ctx.lineWidth = 1.5; ctx.stroke();

    // Value
    const vR = Math.max(v * R + 12, 16);
    const vx = cx + Math.cos(a) * vR;
    const vy = cy + Math.sin(a) * vR;
    ctx.fillStyle = color;
    ctx.font = 'bold 9px system-ui, sans-serif';
    ctx.fillText(Math.round(axes[keys[i]] || 0), vx, vy);
  }
}

function _renderSuccessMetrics(containerId, success, tintColor) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const allAxes = ['aggression', 'trading', 'isolation', 'survival', 'sniper'];
  const labels = {aggression: 'Aggression', trading: 'Trading', isolation: 'Isolation', survival: 'Survival', sniper: 'Sniper'};
  const icons = {aggression: '⚔', trading: '🤝', isolation: '👁', survival: '🛡', sniper: '🎯'};

  let html = '';
  for (const axis of allAxes) {
    const s = success[axis];
    if (!s || !s.rounds) continue;
    const pct = s.win_pct || 0;
    const barColor = pct >= 60 ? (TC.success||'#34d399') : pct >= 45 ? (TC.amber||'#fbbf24') : (TC.fail||'#f87171');
    html += '<div class="flex items-center gap-3">' +
      '<span class="text-sm w-5 text-center">' + icons[axis] + '</span>' +
      '<span class="text-[9px] font-bold uppercase tracking-widest w-20 text-on-surface-variant">' + labels[axis] + '</span>' +
      '<div class="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">' +
        '<div class="h-full rounded-full transition-all duration-700" style="width:' + pct + '%;background:' + barColor + '"></div>' +
      '</div>' +
      '<span class="text-[10px] font-bold w-12 text-right" style="color:' + barColor + '">' + pct.toFixed(0) + '%</span>' +
      '<span class="text-[8px] text-on-surface-variant w-14 text-right">' + s.wins + 'W/' + (s.rounds - s.wins) + 'L</span>' +
    '</div>';
  }
  if (!html) html = '<p class="text-[10px] text-on-surface-variant">No round data</p>';
  el.innerHTML = html;
}

// --- Utility & Economics ---
function renderUtilityEconomics(m) {
  const ud = m.utility_data;
  if (!ud) return;
  document.getElementById('utility-section').classList.remove('hidden');

  // Rating badge. Null means no utility was bought or thrown — distinct from
  // a zero, which would read as "used utility, used it badly".
  const rating = ud.utility_rating;
  const badge = document.getElementById('utility-rating-badge');
  badge.textContent = rating == null ? 'n/a' : rating.toFixed(0) + ' / 100';

  // Top KPIs
  const eco = ud.economics || {};
  document.getElementById('util-total-spent').textContent = '$' + (eco.total_spent || 0).toLocaleString();
  document.getElementById('util-total-wasted').textContent = '$' + (eco.total_wasted || 0).toLocaleString();
  document.getElementById('util-use-rate').textContent = (eco.use_rate || 0).toFixed(0) + '%';
  document.getElementById('util-flash-assists').textContent = (ud.flash || {}).flash_assists || 0;

  // Flash card
  const fl = ud.flash || {};
  document.getElementById('util-flash-stats').innerHTML = _utilStatRows([
    ['Thrown', fl.thrown || 0],
    ['Enemies Flashed', fl.enemies_flashed || 0],
    ['Effective (1s+)', (fl.effective_flashes != null
        ? fl.effective_flashes + ' <span class="text-on-surface-variant">(' + (fl.effective_flash_pct || 0).toFixed(0) + '%)</span>'
        : '—')],
    ['Blind Sec / Flash', (fl.blind_seconds_per_flash || 0).toFixed(2) + 's'],
    ['Team Flashed', '<span class="text-error">' + (fl.team_flashed || 0) + '</span>'],
    ['Avg Blind Duration', (fl.avg_enemy_blind_duration || 0).toFixed(1) + 's'],
    ['Enemies / Flash', (fl.enemies_per_flash || 0).toFixed(2)],
  ]);

  // HE card
  const he = ud.he || {};
  document.getElementById('util-he-stats').innerHTML = _utilStatRows([
    ['Thrown', he.thrown || 0],
    ['Total Damage', he.total_damage || 0],
    ['Avg Dmg / Throw', (he.avg_damage_per_throw || 0).toFixed(1)],
    ['Hits', he.hits || 0],
  ]);

  // Molly card
  const mo = ud.molotov || {};
  document.getElementById('util-molly-stats').innerHTML = _utilStatRows([
    ['Thrown', mo.thrown || 0],
    ['Total Damage', mo.total_damage || 0],
    ['Avg Dmg / Throw', (mo.avg_damage_per_throw || 0).toFixed(1)],
    ['Hits', mo.hits || 0],
  ]);

  // Smoke card
  const sm = ud.smoke || {};
  const topZones = (sm.top_zones || []).map(z => z.zone + ' (' + z.count + ')').join(', ') || '—';
  document.getElementById('util-smoke-stats').innerHTML = _utilStatRows([
    ['Thrown', sm.thrown || 0],
    ['Molly Extinguishes', sm.molotov_extinguishes || 0],
    ['Top Zones', '<span class="text-[10px]">' + topZones + '</span>'],
  ]);

  // Teamplayer expanded section
  _renderTeamplayer(ud.teamplayer || {});

  // Per-round strip chart
  _drawUtilityRoundChart(ud.per_round || []);
}

function _utilStatRows(rows) {
  return rows.map(([label, value]) =>
    '<div class="flex justify-between"><span class="text-on-surface-variant/70">' + label + '</span><span class="text-on-surface font-medium">' + value + '</span></div>'
  ).join('');
}

function _renderTeamplayer(tp) {
  const section = document.getElementById('teamplayer-section');
  if (!section) return;
  const rounds = tp.per_round || [];
  const hasData = (tp.team_attacks || 0) + (tp.team_flashes || 0) + (tp.drops_for_teammates || 0) > 0;
  if (!hasData) return;
  section.classList.remove('hidden');

  const canvas = document.getElementById('teamplayer-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = 280;
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const pad = { left: 50, right: 20, top: 25, bottom: 90 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  // Background
  ctx.fillStyle = TC.bg||'#0f1930';
  ctx.fillRect(0, 0, w, h);

  // Build data for ALL rounds (not just ones with events)
  const totalRounds = Math.max(1, ...rounds.map(r => r.round || 0));
  const roundMap = {};
  for (const r of rounds) roundMap[r.round] = r;

  const allRounds = [];
  for (let i = 1; i <= totalRounds; i++) {
    const r = roundMap[i] || { round: i, attacks: [], team_flashes: [], drops: [] };
    const dmg = (r.attacks || []).reduce((s, a) => s + (a.damage || 0), 0);
    const flashCount = (r.team_flashes || []).length;
    allRounds.push({ ...r, total_damage: dmg, flash_count: flashCount });
  }

  const n = allRounds.length;
  const colW = Math.floor(cw / n);
  const barW = Math.max(4, colW - 3);

  // Max value for Y scale (damage + flash durations as equivalent)
  const maxDmg = Math.max(30, ...allRounds.map(r => r.total_damage));

  // Y axis
  ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  const yStep = Math.max(10, Math.ceil(maxDmg / 4 / 10) * 10);
  for (let d = 0; d <= maxDmg; d += yStep) {
    const y = pad.top + ch - (d / maxDmg) * ch;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + cw, y);
    ctx.stroke();
    ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.45)';
    ctx.font = '10px Space Grotesk, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(d + 'hp', pad.left - 6, y + 3);
  }

  // Axes
  ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.15)';
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + ch);
  ctx.lineTo(pad.left + cw, pad.top + ch);
  ctx.stroke();

  // Draw bars per round
  for (let i = 0; i < n; i++) {
    const r = allRounds[i];
    const x = pad.left + i * colW;

    // Team damage bar (red)
    if (r.total_damage > 0) {
      const barH = (r.total_damage / maxDmg) * ch;
      ctx.fillStyle = 'rgba(248,113,113,0.15)';
      ctx.fillRect(x - 1, pad.top + ch - barH - 1, barW + 2, barH + 2);
      ctx.fillStyle = TC.fail||'#f87171';
      ctx.fillRect(x, pad.top + ch - barH, barW, barH);

      // Damage label on bar
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.font = 'bold 8px Space Grotesk, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(r.total_damage, x + barW / 2, pad.top + ch - barH - 3);
    }

    // Team flash indicators (yellow dots above bar, one per flash)
    const flashes = r.team_flashes || [];
    if (flashes.length > 0) {
      const dotY = pad.top + ch - Math.max((r.total_damage / maxDmg) * ch, 0) - 14;
      for (let f = 0; f < flashes.length; f++) {
        const dotX = x + barW / 2 + (f - (flashes.length - 1) / 2) * 6;
        ctx.fillStyle = TC.flash||'#facc15';
        ctx.beginPath();
        ctx.arc(dotX, dotY, 3, 0, Math.PI * 2);
        ctx.fill();
      }
      // Flash count label if > 1
      if (flashes.length > 1) {
        ctx.fillStyle = TC.flash||'#facc15';
        ctx.font = 'bold 7px Space Grotesk, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(flashes.length + '\u00D7', x + barW / 2, dotY - 6);
      }
    }

    // Drop labels below axis
    const drops = r.drops || [];
    const itemStartY = pad.top + ch + 10;
    const lineH = 9;
    ctx.font = '7px Space Grotesk, sans-serif';
    ctx.textAlign = 'center';
    for (let j = 0; j < drops.length; j++) {
      ctx.fillStyle = '#34d399'; // emerald
      const shortName = drops[j].replace(/^(weapon_)?/, '').substring(0, 6);
      ctx.fillText(shortName, x + barW / 2, itemStartY + j * lineH);
    }

    // Victim names below axis (attacks + flashes)
    const nameY = itemStartY + drops.length * lineH;
    const attacks = r.attacks || [];
    let nameIdx = 0;
    for (const a of attacks) {
      ctx.fillStyle = 'rgba(248,113,113,0.7)';
      ctx.font = '7px Space Grotesk, sans-serif';
      ctx.textAlign = 'center';
      const label = (a.victim || '?').substring(0, 6);
      ctx.fillText(label, x + barW / 2, nameY + nameIdx * lineH);
      nameIdx++;
    }
    for (const f of flashes) {
      ctx.fillStyle = 'rgba(250,204,21,0.7)';
      ctx.font = '7px Space Grotesk, sans-serif';
      ctx.textAlign = 'center';
      const label = (f.victim || '?').substring(0, 6);
      ctx.fillText(label, x + barW / 2, nameY + nameIdx * lineH);
      nameIdx++;
    }

    // Round number
    ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.45)';
    ctx.font = '9px Space Grotesk, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('R' + r.round, x + barW / 2, h - 3);
  }

  // Top-left summary
  ctx.font = 'bold 11px Space Grotesk, sans-serif';
  ctx.textAlign = 'left';
  let lx = pad.left + 4;
  ctx.fillStyle = TC.fail||'#f87171';
  ctx.fillRect(lx, pad.top + 5, 8, 8);
  lx += 12;
  ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.7)';
  ctx.fillText((tp.team_attacks || 0) + ' attacks (' + (tp.team_attack_damage || 0) + ' dmg)', lx, pad.top + 13);
  lx += ctx.measureText((tp.team_attacks || 0) + ' attacks (' + (tp.team_attack_damage || 0) + ' dmg)').width + 16;
  ctx.fillStyle = TC.flash||'#facc15';
  ctx.fillRect(lx, pad.top + 5, 8, 8);
  lx += 12;
  ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.7)';
  ctx.fillText((tp.team_flashes || 0) + ' team flashes', lx, pad.top + 13);
  lx += ctx.measureText((tp.team_flashes || 0) + ' team flashes').width + 16;
  ctx.fillStyle = '#34d399';
  ctx.fillRect(lx, pad.top + 5, 8, 8);
  lx += 12;
  ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.7)';
  ctx.fillText((tp.drops_for_teammates || 0) + ' drops', lx, pad.top + 13);
}

function _escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s || '?';
  return d.innerHTML;
}

function _drawUtilityRoundChart(perRound) {
  const canvas = document.getElementById('util-round-chart');
  if (!canvas || !perRound.length) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = 230;
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const n = perRound.length;
  // Damage is HP and flash is seconds, so they cannot share a y-axis. They
  // used to: the flash legend implied a bar, but blind time was printed as a
  // caption under the axis because there was nowhere to draw it. Two stacked
  // panels over the same round columns, each with its own scale.
  const pad = { left: 50, right: 20, top: 18, bottom: 46 };
  const cw = w - pad.left - pad.right;
  const gap = 16;
  const chDmg = Math.round((h - pad.top - pad.bottom - gap) * 0.62);
  const chFl = (h - pad.top - pad.bottom - gap) - chDmg;
  const dmgBase = pad.top + chDmg;
  const flTop = dmgBase + gap;
  const flBase = flTop + chFl;
  const colW = Math.floor(cw / n);
  const barW = Math.max(4, colW - 3);

  const maxDmg = Math.max(30, ...perRound.map(r => (r.he_damage || 0) + (r.molotov_damage || 0)));
  const maxBlind = Math.max(4, ...perRound.map(r => r.enemy_blind_duration || 0));

  ctx.fillStyle = TC.bg||'#0f1930';
  ctx.fillRect(0, 0, w, h);

  const gridText = TC.gridText||'rgba(255,255,255,0.4)';
  ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  const yStep = Math.max(10, Math.ceil(maxDmg / 4 / 10) * 10);
  for (let d = 0; d <= maxDmg; d += yStep) {
    const y = dmgBase - (d / maxDmg) * chDmg;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
    ctx.fillStyle = gridText;
    ctx.font = '9px Space Grotesk, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(d, pad.left - 6, y + 3);
  }

  // Panel captions, so each scale says what it is.
  ctx.textAlign = 'left';
  ctx.font = '8px Space Grotesk, sans-serif';
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.fillText('DAMAGE (HP)', pad.left + 2, pad.top - 6);
  ctx.fillStyle = 'rgba(250,204,21,0.55)';
  ctx.fillText('ENEMY BLIND (S)', pad.left + 2, flTop - 4);

  ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.15)';
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, dmgBase); ctx.lineTo(pad.left + cw, dmgBase);
  ctx.moveTo(pad.left, flTop); ctx.lineTo(pad.left, flBase); ctx.lineTo(pad.left + cw, flBase);
  ctx.stroke();
  ctx.fillStyle = gridText;
  ctx.font = '9px Space Grotesk, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText(maxBlind.toFixed(0) + 's', pad.left - 6, flTop + 4);

  for (let i = 0; i < n; i++) {
    const r = perRound[i];
    const x = pad.left + i * colW;
    const heDmg = r.he_damage || 0;
    const mollyDmg = r.molotov_damage || 0;
    const totalDmg = heDmg + mollyDmg;
    const blindDur = r.enemy_blind_duration || 0;
    const flashCount = r.enemies_flashed || 0;
    let y = dmgBase;

    if (mollyDmg > 0) {
      const barH = (mollyDmg / maxDmg) * chDmg;
      ctx.fillStyle = TC.orange||'#fb923c';
      ctx.fillRect(x, y - barH, barW, barH);
      y -= barH;
    }
    if (heDmg > 0) {
      const barH = (heDmg / maxDmg) * chDmg;
      ctx.fillStyle = TC.he||'#f87171';
      ctx.fillRect(x, y - barH, barW, barH);
      y -= barH;
    }
    if (totalDmg > 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.font = 'bold 8px Space Grotesk, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(totalDmg, x + barW / 2, y - 2);
    }

    // Blind time as an actual bar, on its own scale.
    if (blindDur > 0) {
      const barH = Math.max(1, (blindDur / maxBlind) * chFl);
      ctx.fillStyle = TC.flash||'#facc15';
      ctx.globalAlpha = 0.8;
      ctx.fillRect(x, flBase - barH, barW, barH);
      ctx.globalAlpha = 1;
      if (barW >= 12) {
        ctx.fillStyle = 'rgba(250,204,21,0.85)';
        ctx.font = '7px Space Grotesk, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(blindDur.toFixed(1), x + barW / 2, flBase - barH - 2);
      }
    }
    if (flashCount > 0 && barW >= 12) {
      ctx.fillStyle = 'rgba(250,204,21,0.45)';
      ctx.font = '7px Space Grotesk, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(flashCount + 'x', x + barW / 2, flBase + 9);
    }

    ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.3)';
    ctx.font = '8px Space Grotesk, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(r.round, x + barW / 2, h - 3);

    ctx.fillStyle = r.side === 'CT' ? 'rgba(96,165,250,0.5)' : 'rgba(251,146,60,0.5)';
    ctx.fillRect(x, pad.top, barW, 3);
  }
}

// --- Per-instance Flash Scatter Chart ---
function _drawFlashScatterChart(rounds) {
  // Collect all flash instances across rounds
  const instances = [];
  for (const r of rounds) {
    const u = r.enriched?.utility || {};
    const fi = u.flash_instances || [];
    for (const f of fi) {
      // Flashing yourself is neither an enemy flash nor a team flash, and it
      // put the player onto their own friendly-fire chart.
      if (f.is_self) continue;
      instances.push({
        round: r.enriched?.round || r.round_number || 0,
        name: f.name,
        duration: f.duration,
        is_friendly: f.is_friendly,
      });
    }
  }
  if (!instances.length) return;

  document.getElementById('util-flash-scatter-wrap').classList.remove('hidden');
  const canvas = document.getElementById('util-flash-scatter');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = 260;
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  // Chart area
  const pad = { left: 55, right: 20, top: 30, bottom: 35 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  // Background
  ctx.fillStyle = TC.bg||'#0f1930';
  ctx.fillRect(0, 0, w, h);

  // Max duration for axes
  const maxDur = Math.max(4, ...instances.map(i => i.duration)) * 1.15;

  // Separate enemy and friendly instances
  const enemies = instances.filter(i => !i.is_friendly);
  const friendlies = instances.filter(i => i.is_friendly);

  // Draw axes
  ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + ch);
  ctx.lineTo(pad.left + cw, pad.top + ch);
  ctx.stroke();

  // Y-axis label (count/index) — each instance is a dot at its duration
  ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.5)';
  ctx.font = 'bold 12px Space Grotesk, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Blind Duration (seconds)', pad.left + cw / 2, h - 3);

  // X-axis: duration scale
  ctx.textAlign = 'center';
  const tickSteps = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  for (const ts of tickSteps) {
    if (ts > maxDur) break;
    const tx = pad.left + (ts / maxDur) * cw;
    ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.45)';
    ctx.font = '12px Space Grotesk, sans-serif';
    ctx.fillText(ts + 's', tx, pad.top + ch + 16);
    // Grid line
    ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.05)';
    ctx.beginPath();
    ctx.moveTo(tx, pad.top);
    ctx.lineTo(tx, pad.top + ch);
    ctx.stroke();
  }

  // Draw two horizontal lanes: Enemies (top) and Friendlies (bottom)
  const laneH = ch / 2;
  const enemyY = pad.top + laneH * 0.5;
  const friendlyY = pad.top + laneH * 1.5;

  // Lane labels
  ctx.font = 'bold 12px Space Grotesk, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillStyle = TC.success||'#34d399';
  ctx.fillText('Enemy', pad.left - 8, enemyY + 4);
  ctx.fillStyle = TC.fail||'#f87171';
  ctx.fillText('Friendly', pad.left - 8, friendlyY + 4);

  // Lane divider
  ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.08)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top + laneH);
  ctx.lineTo(pad.left + cw, pad.top + laneH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw enemy dots (spread vertically with jitter to avoid overlap)
  const drawDots = (arr, baseY, color, glowColor) => {
    const spread = laneH * 0.35;
    for (let i = 0; i < arr.length; i++) {
      const inst = arr[i];
      const dx = pad.left + (inst.duration / maxDur) * cw;
      // Jitter Y so overlapping dots are visible
      const jitter = (i % 2 === 0 ? 1 : -1) * ((i % 5) * 4);
      const dy = baseY + jitter;

      // Glow
      ctx.fillStyle = glowColor;
      ctx.beginPath();
      ctx.arc(dx, dy, 10, 0, Math.PI * 2);
      ctx.fill();

      // Dot
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(dx, dy, 5, 0, Math.PI * 2);
      ctx.fill();

      // Name tooltip label (only show for first few to reduce clutter)
      if (arr.length <= 15 || i % 3 === 0) {
        ctx.font = '8px Space Grotesk, sans-serif';
        ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.5)';
        ctx.textAlign = 'left';
        ctx.fillText(`${inst.name} (R${inst.round})`, dx + 8, dy + 3);
      }
    }
  };

  drawDots(enemies, enemyY, TC.success||'#34d399', hexToRgba(TC.success||'#34d399',0.2));
  drawDots(friendlies, friendlyY, TC.fail||'#f87171', hexToRgba(TC.fail||'#f87171',0.2));

  // Summary stats
  ctx.font = 'bold 12px Space Grotesk, sans-serif';
  ctx.textAlign = 'left';
  const avgEnemy = enemies.length ? (enemies.reduce((s, e) => s + e.duration, 0) / enemies.length).toFixed(2) : '0';
  const avgFriendly = friendlies.length ? (friendlies.reduce((s, e) => s + e.duration, 0) / friendlies.length).toFixed(2) : '0';
  ctx.fillStyle = TC.success||'#34d399';
  ctx.fillText(`${enemies.length} enemy — avg ${avgEnemy}s`, pad.left + 4, pad.top + 16);
  ctx.fillStyle = TC.fail||'#f87171';
  ctx.fillText(`${friendlies.length} friendly — avg ${avgFriendly}s`, pad.left + cw * 0.5, pad.top + 16);
}

// --- Scoreboard ---
const RANK_NAMES = ['Unranked','Silver 1','Silver 2','Silver 3','Silver 4','Silver Elite','Silver Elite Master','Gold Nova 1','Gold Nova 2','Gold Nova 3','Gold Nova Master','MG1','MG2','MGE','DMG','LE','LEM','SMFC','Global Elite'];

function formatRank(p) {
  const rankNum = p.rank || 0;
  const rankType = p.rank_type_id || 0;
  if (rankNum <= 0) return { display: '—', title: 'Unranked', color: 'text-on-surface-variant/50' };
  // Premier (rank_type_id 11): CS Rating integer
  if (rankType === 11) {
    const rating = rankNum;
    let color = 'text-on-surface-variant';           // Gray: 0–4,999
    if (rating >= 30000) color = 'text-caution';       // Gold
    else if (rating >= 25000) color = 'text-bad';     // Red
    else if (rating >= 20000) color = 'text-tertiary';    // Pink
    else if (rating >= 15000) color = 'text-primary';  // Purple
    else if (rating >= 10000) color = 'text-info';    // Blue
    else if (rating >= 5000) color = 'text-info';      // Light Blue
    const display = rating.toLocaleString();
    return { display, title: 'CS Rating ' + display, color };
  }
  // Competitive (rank_type_id 12 or fallback): skill group 1-18
  const rankName = RANK_NAMES[rankNum] || 'Unranked';
  const color = rankNum >= 15 ? 'text-caution' : rankNum >= 11 ? 'text-accent' : rankNum >= 7 ? 'text-caution' : rankNum >= 1 ? 'text-on-surface-variant' : 'text-on-surface-variant/50';
  return { display: rankName, title: rankName, color };
}

let scoreboardData = { my: [], enemy: [] };
let scoreboardSort = { my: { col: 'hltv_rating', asc: false }, enemy: { col: 'hltv_rating', asc: false } };

function renderScoreboard(myTeam, enemyTeam) {
  scoreboardData.my = myTeam;
  scoreboardData.enemy = enemyTeam;
  renderSortedTeam('scoreboard-my-team', myTeam, true, 'my');
  renderSortedTeam('scoreboard-enemy-team', enemyTeam, false, 'enemy');
  initSortHeaders();
}

function renderSortedTeam(containerId, players, isMyTeam, teamKey) {
  const s = scoreboardSort[teamKey];
  const sorted = [...players].sort((a, b) => {
    let va = a[s.col], vb = b[s.col];
    if (s.col === 'name') { va = (va||'').toLowerCase(); vb = (vb||'').toLowerCase(); }
    if (va < vb) return s.asc ? -1 : 1;
    if (va > vb) return s.asc ? 1 : -1;
    return 0;
  });
  renderTeamRows(containerId, sorted, isMyTeam);
}

function initSortHeaders() {
  document.querySelectorAll('[data-team] [data-sort]').forEach(el => {
    if (el.dataset.bound) return;
    el.dataset.bound = '1';
    el.style.cursor = 'pointer';
    el.addEventListener('click', () => {
      const teamKey = el.closest('[data-team]').dataset.team;
      const col = el.dataset.sort;
      if (scoreboardSort[teamKey].col === col) {
        scoreboardSort[teamKey].asc = !scoreboardSort[teamKey].asc;
      } else {
        scoreboardSort[teamKey] = { col, asc: col === 'name' };
      }
      const containerId = teamKey === 'my' ? 'scoreboard-my-team' : 'scoreboard-enemy-team';
      renderSortedTeam(containerId, scoreboardData[teamKey], teamKey === 'my', teamKey);
      // Update arrow indicators
      el.closest('[data-team]').querySelectorAll('.sort-arrow').forEach(a => a.textContent = '');
      el.querySelector('.sort-arrow').textContent = scoreboardSort[teamKey].asc ? '▲' : '▼';
    });
  });
}

function renderTeamRows(containerId, players, isMyTeam) {
  const container = document.getElementById(containerId);
  const rows = container.querySelectorAll('.player-row');
  rows.forEach(r => r.remove());
  for (const p of players) {
    const isUser = p.is_user;
    const ratingColor = (p.hltv_rating || 0) >= 1.2 ? 'text-good' : (p.hltv_rating || 0) >= 1.0 ? 'text-on-surface' : 'text-error';
    const highlight = isUser ? 'bg-primary/5 border-l-2 border-primary' : 'hover:bg-white/5';
    const r = formatRank(p);
    const row = document.createElement('div');
    row.className = `player-row grid items-center text-xs px-3 py-2 rounded-lg ${highlight} transition-colors`;
    row.style.gridTemplateColumns = '2.5fr 1.3fr 0.7fr 0.7fr 0.7fr 1fr 1fr 1fr';
    row.innerHTML = `
      <div class=\"font-bold truncate ${isUser ? 'text-primary' : 'text-on-surface'}\">${p.steam_id ? `<a href="https://steamcommunity.com/profiles/${esc(p.steam_id)}" target="_blank" rel="noopener noreferrer" class="hover:underline">${esc(p.name || '?')}</a>` : esc(p.name || '?')}${isUser ? ' <span class=\"text-[9px] text-primary-dim\">(you)</span>' : ''}${p.is_friend ? ' <span class=\"text-[9px] text-accent\">(friend)</span>' : ''}</div>
      <div class=\"text-center text-[10px] ${r.color}\" title=\"${r.title}\">${r.display}</div>
      <div class=\"text-center font-semibold text-on-surface\">${p.kills||0}</div>
      <div class=\"text-center text-on-surface-variant\">${p.deaths||0}</div>
      <div class=\"text-center text-on-surface-variant\">${p.assists||0}</div>
      <div class=\"text-center text-on-surface-variant\">${(p.adr||0).toFixed(1)}</div>
      <div class=\"text-center text-on-surface-variant\">${(p.kast||0).toFixed(0)}%</div>
      <div class=\"text-center font-bold ${ratingColor}\">${(p.hltv_rating||0).toFixed(2)}</div>`;
    container.appendChild(row);
  }
}

// --- Economy Timeline (canvas-based) ---


// --- Round Timeline ---
function renderRoundTimeline(rounds, matchData) {
  const cont = document.getElementById('round-timeline');
  for (const r of rounds) {
    const e = r.enriched || {};
    const side = e.side || '?';
    const winner = e.round_winner;
    const won = side === winner;
    const eco = e.economy || {};
    const kills = r.kills || 0;
    const deaths = r.deaths || 0;
    const damage = r.damage || 0;

    const borderColor = won ? 'border-good/30' : 'border-error/30';
    const bgColor = won ? 'bg-good/5' : 'bg-error/5';
    const resultBadge = won
      ? '<span class="bg-good/20 text-good px-2 py-0.5 rounded-full text-[9px] font-bold">WIN</span>'
      : '<span class="bg-error/20 text-error px-2 py-0.5 rounded-full text-[9px] font-bold">LOSS</span>';

    // Economy badge
    const buyColors = { 'FULL BUY':'text-secondary','HALF BUY':'text-primary','FORCE BUY':'text-tertiary','PISTOL':'text-warn','ECO':'text-on-surface-variant' };
    const buyBg = { 'FULL BUY':'bg-secondary/10','HALF BUY':'bg-primary/10','FORCE BUY':'bg-tertiary/10','PISTOL':'bg-warn/10','ECO':'bg-white/5' };
    const buyType = eco.buy_type || '?';
    const ecoBadge = `<span class="${buyBg[buyType] || ''} ${buyColors[buyType] || ''} px-2 py-0.5 rounded-full text-[9px] font-bold">${buyType} $${eco.player_spend||0}</span>`;

    // Side badge
    const sideBadge = `<span class="px-2 py-0.5 rounded-full text-[9px] font-bold ${side === 'CT' ? 'bg-accent/10 text-accent' : 'bg-warn/10 text-warn'}">${side}</span>`;

    // Kill details
    let killLines = '';
    for (const k of (e.kills_detail || [])) {
      const hs = k.headshot ? ' <span class="text-error">HS</span>' : '';
      const specials = k.specials ? ` <span class="text-primary-dim">[${k.specials.join(', ')}]</span>` : '';
      const pos = [];
      if (k.victim_position && k.victim_position !== 'unknown') pos.push(`at ${esc(k.victim_position)}`);
      if (k.attacker_position && k.attacker_position !== 'unknown') pos.push(`from ${esc(k.attacker_position)}`);
      const posStr = pos.length ? ` <span class="text-on-surface-variant">(${pos.join(' ')})</span>` : '';
      // Movement quality badge
      let movBadge = '';
      if (k.movement) {
        const mq = k.movement.movement_quality;
        const spd = k.movement.shot_speed;
        const pre = k.movement.pre_speed;
        const st = k.movement.stop_ticks;
        if (mq === 'running') {
          movBadge = ` <span class="bg-error/20 text-error px-1.5 py-0.5 rounded-full text-[9px] font-bold" title="Firing while moving at ${spd} u/s">${spd} u/s</span>`;
        } else if (mq === 'counter-strafed') {
          movBadge = ` <span class="bg-good/20 text-good px-1.5 py-0.5 rounded-full text-[9px] font-bold" title="Cancelled ${pre} u/s in ${st} ticks, shot at ${spd} u/s">c-strafe</span>`;
        } else if (mq === 'stopped') {
          movBadge = ` <span class="bg-warn/20 text-warn px-1.5 py-0.5 rounded-full text-[9px] font-bold" title="Coasted down from ${pre} u/s over ${st} ticks instead of counter-strafing">coasted</span>`;
        } else {
          movBadge = ` <span class="bg-accent/20 text-accent px-1.5 py-0.5 rounded-full text-[9px] font-bold" title="Already static — never moved before the shot">still</span>`;
        }
      }
      // Pre-aim quality badge
      let preaimBadge = '';
      if (k.preaim) {
        const pq = k.preaim.preaim_quality;
        const err = k.preaim.crosshair_error;
        const colors = { excellent: 'bg-good/20 text-good', good: 'bg-accent/20 text-accent', moderate: 'bg-warn/20 text-warn', poor: 'bg-error/20 text-error' };
        preaimBadge = ` <span class="${colors[pq] || ''} px-1.5 py-0.5 rounded-full text-[9px] font-bold" title="Crosshair ${err}° off target">${err}°</span>`;
      }
      // Time to damage badge
      let ttdBadge = '';
      if (k.ttd && k.ttd.ttk_seconds > 0) {
        const sec = k.ttd.ttk_seconds;
        const hits = k.ttd.hits;
        const col = sec < 0.2 ? 'bg-good/20 text-good' : sec < 0.5 ? 'bg-warn/20 text-warn' : 'bg-error/20 text-error';
        ttdBadge = ` <span class="${col} px-1.5 py-0.5 rounded-full text-[9px] font-bold" title="${hits} hits over ${sec}s">TTK ${sec}s</span>`;
      }
      killLines += `<div class="text-[10px] text-good ml-4">→ ${esc(k.victim)}${posStr} — ${esc(k.weapon || '?')}${hs}${specials}${movBadge}${preaimBadge}${ttdBadge}</div>`;
    }

    // Death detail
    let deathLine = '';
    if (e.death_detail) {
      const d = e.death_detail;
      const hs = d.headshot ? ' <span class="text-error">HS</span>' : '';
      const pos = [];
      if (d.victim_position && d.victim_position !== 'unknown') pos.push(`at ${esc(d.victim_position)}`);
      if (d.killer_position && d.killer_position !== 'unknown') pos.push(`from ${esc(d.killer_position)}`);
      const posStr = pos.length ? ` <span class="text-on-surface-variant">(${pos.join(' ')})</span>` : '';
      deathLine = `<div class="text-[10px] text-error ml-4">✗ by ${esc(d.killer)}${posStr} — ${esc(d.weapon || '?')}${hs}</div>`;
    }

    // Opening duel
    let openingLine = '';
    if (e.opening_duel) {
      const od = e.opening_duel;
      if (od.role === 'opening_kill') openingLine = `<span class="bg-good/20 text-good px-2 py-0.5 rounded-full text-[9px] font-bold">⚡ ENTRY</span>`;
      else openingLine = `<span class="bg-error/20 text-error px-2 py-0.5 rounded-full text-[9px] font-bold">💀 FIRST DEATH</span>`;
    }

    // Clutch
    let clutchLine = '';
    if (e.clutch) {
      const cw = e.clutch.won ? 'text-good' : 'text-error';
      clutchLine = `<span class="bg-primary/20 ${cw} px-2 py-0.5 rounded-full text-[9px] font-bold">🏆 1v${e.clutch.vs} ${e.clutch.won ? 'WON' : 'LOST'}</span>`;
    }

    // Bomb
    let bombLine = '';
    if (e.bomb) {
      if (e.bomb.planted) bombLine += `<span class="text-[10px] text-warn ml-4">💣 Planted</span> `;
      if (e.bomb.defused) bombLine += `<span class="text-[10px] text-accent ml-4">🔧 Defused</span> `;
    }

    // Utility
    let utilLine = '';
    const u = e.utility || {};
    const utilParts = [];
    const flashVictims = u.flash_victims || [];
    if (flashVictims.length) {
      const fv = flashVictims.map(v => `${esc(v.name)} ${v.duration}s`).join(', ');
      utilParts.push(`⚡ Flashed: ${fv}`);
    }
    if (u.flash_assists) utilParts.push(`Flash assist ×${u.flash_assists}`);
    if (u.he_damage) utilParts.push(`HE: ${u.he_damage}hp`);
    const molly = u.molotov_damage || [];
    if (molly.length) {
      const mv = molly.map(v => `${esc(v.victim)} ${v.damage}hp`).join(', ');
      utilParts.push(`🔥 Molotov: ${mv}`);
    }
    if (utilParts.length) utilLine = `<div class="text-[10px] text-on-surface-variant ml-4">${utilParts.join(' | ')}</div>`;

    // Grenade throw → land trajectories
    let nadeLine = '';
    const grenades = u.grenades || [];
    if (grenades.length) {
      const nadeIcons = { flash: '⚡', he: '💥', molotov: '🔥', smoke: '💨' };
      const nadeColors = { flash: 'text-caution', he: 'text-bad', molotov: 'text-warn', smoke: 'text-on-surface-variant' };
      const bits = grenades.map(g => {
        const icon = nadeIcons[g.type] || '🧨';
        const cls = nadeColors[g.type] || 'text-on-surface-variant';
        const from = g.throw_callout || '?';
        const to = g.land_callout || '?';
        return `<span class="${cls}">${icon} ${esc(from)} → ${esc(to)}</span>`;
      });
      nadeLine = `<div class="text-[10px] text-on-surface-variant ml-4">${bits.join(' &nbsp;·&nbsp; ')}</div>`;
    }

    const row = document.createElement('div');
    const roundNum = r.round_number || e.round || 0;
    row.className = `${bgColor} border-l-2 ${borderColor} rounded-r-lg p-3 cursor-pointer hover:brightness-125 transition-all`;
    row.title = 'Click to view on minimap';
    row.onclick = () => openMinimap(roundNum);
    row.innerHTML = `
      <div class="flex items-center gap-2 flex-wrap">
        <span class="text-[10px] font-bold text-on-surface w-8">R${roundNum}</span>
        ${sideBadge} ${ecoBadge} ${resultBadge} ${openingLine} ${clutchLine}
        <button class="material-symbols-outlined text-on-surface-variant hover:text-primary text-sm" title="View on minimap" data-action="stopPropagation openMinimap" data-args="[[], [${roundNum}]]">map</button>
        <span class="ml-auto text-xs font-bold text-on-surface">${kills}K ${deaths}D <span class="text-on-surface-variant font-normal">${damage}dmg</span></span>
      </div>
      ${killLines}${deathLine}${bombLine}${utilLine}${nadeLine}`;
    cont.appendChild(row);
  }
}

/* Opening a match from the list. Was window.location.href assembled inside an
   onclick attribute, on a row whose id came from the database. */
function openMatch(matchId) {
  window.location.href = 'match-breakdown.html?id=' + encodeURIComponent(matchId);
}

async function deleteMatch(matchId, event) {
  event.stopPropagation();
  if (!confirm('Delete this match and all its data?')) return;
  try {
    const res = await fetch(API + '/matches/' + encodeURIComponent(matchId), { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete failed');
    await loadMatchList();
  } catch (err) { console.error('Delete error:', err); alert('Failed to delete match.'); }
}

async function deleteCurrentMatch() {
  if (!currentMatchId) return;
  if (!confirm('Delete this match and all its data? You will be returned to the match list.')) return;
  try {
    const res = await fetch(API + '/matches/' + encodeURIComponent(currentMatchId), { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete failed');
    window.location.href = 'match-breakdown.html';
  } catch (err) { console.error('Delete error:', err); alert('Failed to delete match.'); }
}

async function reimportCurrentMatch() {
  if (!currentMatchId) return;

  const picker = document.createElement('input');
  picker.type = 'file';
  picker.accept = '.dem';
  picker.onchange = async () => {
    const file = picker.files && picker.files[0];
    if (!file) return;

    const btn = document.getElementById('reimport-match-btn');
    const oldText = btn ? btn.innerHTML : '';
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span class="material-symbols-outlined text-xs">hourglass_top</span> Re-importing...';
    }

    try {
      const fd = new FormData();
      fd.append('file', file);
      if (currentMatchData && currentMatchData.player_steam_id) {
        fd.append('steam_id', currentMatchData.player_steam_id);
      }

      const res = await fetch(API + '/matches/' + encodeURIComponent(currentMatchId) + '/reimport', {
        method: 'POST',
        body: fd,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Re-import failed');
      }
      const data = await res.json();
      window.location.href = 'match-breakdown.html?id=' + encodeURIComponent(data.match_id);
    } catch (err) {
      alert(err.message || 'Re-import failed');
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = oldText;
      }
    }
  };
  picker.click();
}

function exportMatch() {
  window.print();
}

async function loadMatchList() {
  try {
    // Get filter value if it exists
    const filterEl = document.getElementById('match-user-filter');
    const filterSteamId = filterEl ? filterEl.value : '';
    const url = filterSteamId ? API + '/matches?player_steam_id=' + encodeURIComponent(filterSteamId) : API + '/matches';
    const res = await fetch(url);
    const matches = await res.json();

    // Load accounts for the filter dropdown
    let accounts = [];
    try { const acctRes = await fetch(API + '/accounts'); accounts = await acctRes.json(); } catch(e) {}

    const mainContent = document.getElementById('main-content');
    if (!mainContent) return;

    let listHtml = `
      <section class="mb-8">
        <div class="flex items-center justify-between mb-6">
          <h1 class="text-3xl font-headline font-bold uppercase tracking-widest">MATCH_HISTORY</h1>
          <select id="match-user-filter" data-action="loadMatchList" data-event="change" class="bg-surface-container-highest text-on-surface text-xs px-4 py-2 rounded-full border border-white/10 focus:border-secondary/30 focus:ring-0 font-bold uppercase tracking-widest">
            <option value="">All Accounts</option>
            ${accounts.map(a => `<option value="${a.steam_id}" ${a.steam_id === filterSteamId ? 'selected' : ''}>${esc(a.name)}</option>`).join('')}
          </select>
        </div>
        <div class="space-y-2">`;

    if (!matches.length) {
      listHtml += '<p class="text-on-surface-variant text-sm text-center py-8">No matches found.</p>';
    }

    for (const m of matches) {
      const isWin = m.match_result === 'Victory' || m.match_result === 'win';
      const borderColor = isWin ? 'border-good' : 'border-error';
      const scoreColor = isWin ? 'text-good' : 'text-error';
      const dateStr = m.date || '';
      const partialBadge = m.partial_import
        ? '<div class="mt-1 inline-flex items-center gap-1 rounded-full bg-caution/15 text-caution px-2 py-0.5 text-[9px] font-bold uppercase tracking-widest"><span class="material-symbols-outlined text-[10px]">warning</span>Partial</div>'
        : '';
      // Find player name from accounts  
      const acct = accounts.find(a => a.steam_id === m.player_steam_id);
      const playerTag = acct ? acct.name : '';

      listHtml += `
        <div class="grid grid-cols-4 md:grid-cols-12 items-center bg-surface-container hover:bg-surface-container-high transition-all p-4 rounded-xl border-l-4 ${borderColor} group cursor-pointer" data-action="openMatch" data-args="${actionArgs(m.match_id)}">
          <div class="md:col-span-2 flex items-center gap-3">
            ${mapIconHtml(m.map_name)}
            <div>
              <div class="font-bold text-xs uppercase tracking-widest">${mapLabel(m.map_name)}</div>
              ${dateStr ? '<div class="text-[9px] text-on-surface-variant">'+dateStr+'</div>' : ''}
              ${partialBadge}
              ${playerTag && !filterSteamId ? '<div class="text-[9px] text-secondary font-bold">'+esc(playerTag)+'</div>' : ''}
            </div>
          </div>
          <div class="md:col-span-2 font-headline text-xl font-bold ${scoreColor}">${m.team_score||0} - ${m.enemy_score||0}</div>
          <div class="hidden md:block md:col-span-2">
            <div class="text-[10px] font-bold text-on-surface-variant uppercase mb-1">K / D / A</div>
            <div class="text-sm font-bold text-on-surface">${m.kills||0} / ${m.deaths||0} / ${m.assists||0}</div>
          </div>
          <div class="hidden md:block md:col-span-2">
            <div class="text-[10px] font-bold text-on-surface-variant uppercase mb-1">RATING</div>
            <div class="text-sm font-bold ${(m.hltv_rating||0) >= 1.0 ? 'text-good' : 'text-on-surface'}">${(m.hltv_rating||0).toFixed(2)}</div>
          </div>
          <div class="md:col-span-2 text-right md:text-left">
            <div class="text-[10px] font-bold text-on-surface-variant uppercase mb-1">STATUS</div>
            <div class="text-xs font-bold ${isWin ? 'text-good' : 'text-error'} uppercase tracking-widest">${isWin ? 'VICTORY' : 'DEFEAT'}</div>
          </div>
          <div class="hidden md:flex md:col-span-2 justify-end gap-2">
            <span class="material-symbols-outlined text-on-surface-variant group-hover:text-primary transition-colors">analytics</span>
            <button data-action="deleteMatch" data-args="${actionArgs(m.match_id)}" class="material-symbols-outlined text-on-surface-variant hover:text-error transition-colors text-lg" title="Delete match">delete</button>
          </div>
        </div>`;
    }
    listHtml += '</div></section>';
    mainContent.innerHTML = listHtml;
  } catch (err) { console.error('Failed to load matches:', err); }
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


document.addEventListener('DOMContentLoaded', () => { loadAccounts(); loadMatchDetail(); });

// ---------------------------------------------------------------------------
// Minimap Renderer
// ---------------------------------------------------------------------------
let radarImage = null;
let radarLoaded = false;
let schematicZones = null; // cached zone rectangles for schematic radar
let _debugPositions = null; // cached debug positions
let _debugMode = false;    // toggle for showing all positions

// Zoom & pan state
let mapZoom = 1.0;
let mapPanX = 0;   // offset in canvas-pixel space
let mapPanY = 0;
let _mapDragging = false;
let _mapDragStartX = 0;
let _mapDragStartY = 0;
let _mapPanStartX = 0;
let _mapPanStartY = 0;
let _currentRoundNum = 1;
const MAP_ZOOM_MIN = 1.0;
const MAP_ZOOM_MAX = 6.0;
const MAP_ZOOM_STEP = 0.3;

async function openMinimap(roundNum) {
  if (!currentMatchId) return;

  document.getElementById('minimap-title').textContent = 'Round ' + roundNum;

  // Fetch minimap data if not cached
  if (!minimapData) {
    try {
      const res = await fetch(API + '/matches/' + encodeURIComponent(currentMatchId) + '/minimap');
      if (!res.ok) throw new Error('Minimap data unavailable');
      minimapData = await res.json();
    } catch (err) {
      console.error('Minimap fetch error:', err);
      document.getElementById('minimap-no-radar').classList.remove('hidden');
      return;
    }
  }

  // Fetch schematic zone data if not cached (used as fallback or overlay)
  if (!schematicZones && currentMapName) {
    try {
      const zr = await fetch(API + '/minimap/' + encodeURIComponent(currentMapName) + '/schematic');
      if (zr.ok) schematicZones = (await zr.json()).zones;
    } catch (_) { /* non-fatal */ }
  }

  _currentRoundNum = roundNum;

  // Try loading radar PNG; fall back to schematic
  if (radarLoaded && radarImage) {
    drawMinimap(roundNum);
  } else if (!radarImage && minimapData.radar_image) {
    radarImage = new Image();
    radarImage.onload = () => {
      radarLoaded = true;
      drawMinimap(roundNum);
    };
    radarImage.onerror = () => {
      radarLoaded = false;
      console.log('Radar PNG not found, using schematic');
      drawMinimap(roundNum);
    };
    radarImage.src = minimapData.radar_image;
  } else {
    drawMinimap(roundNum);
  }

  // Attach zoom/pan listeners (idempotent)
  _attachMinimapListeners();

  // Scroll to minimap on mobile only
  if (window.innerWidth < 1024) {
    document.getElementById('minimap-section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function closeMinimap() {
  // No-op — minimap is always visible in the merged layout
}

function drawMinimap(roundNum) {
  const canvas = document.getElementById('minimap-canvas');
  const ctx = canvas.getContext('2d');

  // Reset transform, clear, then apply zoom/pan
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, 1024, 1024);
  ctx.setTransform(mapZoom, 0, 0, mapZoom, mapPanX, mapPanY);

  // Draw radar PNG if available, otherwise schematic
  if (radarLoaded && radarImage) {
    ctx.drawImage(radarImage, 0, 0, 1024, 1024);
  } else {
    drawSchematicBackground(ctx);
  }
  document.getElementById('minimap-no-radar').classList.add('hidden');

  drawEvents(ctx, roundNum);
  drawDebugPositions(ctx);

  // Reset transform so UI overlays are unaffected
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  // Update zoom label
  const lbl = document.getElementById('minimap-zoom-label');
  if (lbl) lbl.textContent = mapZoom.toFixed(1) + 'x';
}

// Colour palette for zone categories
const _ZONE_COLORS = {
  site:  { fill: 'rgba(59,130,246,0.18)', stroke: 'rgba(59,130,246,0.55)', text: 'rgba(147,197,253,0.85)' },
  spawn: { fill: 'rgba(34,197,94,0.12)',  stroke: 'rgba(34,197,94,0.45)',  text: 'rgba(134,239,172,0.8)' },
  mid:   { fill: 'rgba(168,85,247,0.14)', stroke: 'rgba(168,85,247,0.45)', text: 'rgba(196,181,253,0.8)' },
  t_area:{ fill: 'rgba(251,146,60,0.12)', stroke: 'rgba(251,146,60,0.4)',  text: 'rgba(253,186,116,0.8)' },
  zone:  { fill: 'rgba(148,163,184,0.10)',stroke: 'rgba(148,163,184,0.3)', text: 'rgba(203,213,225,0.7)' },
};

function drawSchematicBackground(ctx) {
  // Dark background
  ctx.fillStyle = TC.canvasBg||'#0a1628';
  ctx.fillRect(0, 0, 1024, 1024);

  // Subtle grid
  ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.03)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 1024; i += 64) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 1024); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(1024, i); ctx.stroke();
  }

  if (!schematicZones || !schematicZones.length) return;

  // Draw zone rectangles
  for (const z of schematicZones) {
    const c = _ZONE_COLORS[z.cat] || _ZONE_COLORS.zone;
    const w = z.px2 - z.px1;
    const h = z.py2 - z.py1;

    // Filled rect
    ctx.fillStyle = c.fill;
    ctx.fillRect(z.px1, z.py1, w, h);

    // Border
    ctx.strokeStyle = c.stroke;
    ctx.lineWidth = 1.2;
    ctx.strokeRect(z.px1, z.py1, w, h);

    // Label (skip for very small zones)
    if (w > 30 && h > 16) {
      ctx.font = w < 60 ? '8px Space Grotesk, sans-serif' : 'bold 10px Space Grotesk, sans-serif';
      ctx.fillStyle = c.text;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(z.label, z.cx, z.cy);
      ctx.textAlign = 'start';
      ctx.textBaseline = 'alphabetic';
    }
  }

  // Map name watermark
  if (currentMapName) {
    ctx.font = 'bold 14px Space Grotesk, sans-serif';
    ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.12)';
    ctx.textAlign = 'right';
    ctx.fillText(currentMapName.replace('de_', '').toUpperCase(), 1010, 1014);
    ctx.textAlign = 'start';
  }
}

function drawMinimapFallback(roundNum) {
  drawMinimap(roundNum);
}

// ---------------------------------------------------------------------------
// Zoom & Pan helpers
// ---------------------------------------------------------------------------
let _minimapListenersAttached = false;
function _attachMinimapListeners() {
  if (_minimapListenersAttached) return;
  _minimapListenersAttached = true;
  const canvas = document.getElementById('minimap-canvas');

  // Wheel zoom (pinch-to-zoom on trackpad sends wheel events)
  canvas.addEventListener('wheel', function (e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    // Mouse position in canvas-pixel space
    const mx = (e.clientX - rect.left) * (1024 / rect.width);
    const my = (e.clientY - rect.top)  * (1024 / rect.height);

    const oldZoom = mapZoom;
    const delta = e.deltaY < 0 ? MAP_ZOOM_STEP : -MAP_ZOOM_STEP;
    mapZoom = Math.min(MAP_ZOOM_MAX, Math.max(MAP_ZOOM_MIN, mapZoom + delta));

    // Zoom toward mouse pointer
    const factor = mapZoom / oldZoom;
    mapPanX = mx - factor * (mx - mapPanX);
    mapPanY = my - factor * (my - mapPanY);
    _clampPan();
    drawMinimap(_currentRoundNum);
  }, { passive: false });

  // Drag to pan
  canvas.addEventListener('mousedown', function (e) {
    if (e.button !== 0) return;
    _mapDragging = true;
    const rect = canvas.getBoundingClientRect();
    _mapDragStartX = e.clientX;
    _mapDragStartY = e.clientY;
    _mapPanStartX = mapPanX;
    _mapPanStartY = mapPanY;
    canvas.style.cursor = 'grabbing';
  });
  window.addEventListener('mousemove', function (e) {
    if (!_mapDragging) return;
    const canvas = document.getElementById('minimap-canvas');
    const rect = canvas.getBoundingClientRect();
    const scale = 1024 / rect.width;
    mapPanX = _mapPanStartX + (e.clientX - _mapDragStartX) * scale;
    mapPanY = _mapPanStartY + (e.clientY - _mapDragStartY) * scale;
    _clampPan();
    drawMinimap(_currentRoundNum);
  });
  window.addEventListener('mouseup', function () {
    if (_mapDragging) {
      _mapDragging = false;
      const canvas = document.getElementById('minimap-canvas');
      if (canvas) canvas.style.cursor = mapZoom > 1 ? 'grab' : 'grab';
    }
  });

  // Alt+click to show radar pixel coordinate (calibration helper)
  canvas.addEventListener('click', function (e) {
    if (!e.altKey) return;
    const rect = canvas.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * (1024 / rect.width);
    const sy = (e.clientY - rect.top) * (1024 / rect.height);
    const px = Math.round((sx - mapPanX) / mapZoom);
    const py = Math.round((sy - mapPanY) / mapZoom);
    const lbl = document.getElementById('minimap-coords');
    lbl.textContent = `(${px}, ${py})`;
    lbl.classList.remove('hidden');
    console.log(`Radar pixel: (${px}, ${py})`);
  });

  // Touch support for mobile
  let _touchStartDist = 0;
  let _touchStartZoom = 1;
  canvas.addEventListener('touchstart', function (e) {
    if (e.touches.length === 1) {
      _mapDragging = true;
      _mapDragStartX = e.touches[0].clientX;
      _mapDragStartY = e.touches[0].clientY;
      _mapPanStartX = mapPanX;
      _mapPanStartY = mapPanY;
    } else if (e.touches.length === 2) {
      _mapDragging = false;
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      _touchStartDist = Math.hypot(dx, dy);
      _touchStartZoom = mapZoom;
    }
  }, { passive: true });
  canvas.addEventListener('touchmove', function (e) {
    e.preventDefault();
    if (e.touches.length === 1 && _mapDragging) {
      const rect = canvas.getBoundingClientRect();
      const scale = 1024 / rect.width;
      mapPanX = _mapPanStartX + (e.touches[0].clientX - _mapDragStartX) * scale;
      mapPanY = _mapPanStartY + (e.touches[0].clientY - _mapDragStartY) * scale;
      _clampPan();
      drawMinimap(_currentRoundNum);
    } else if (e.touches.length === 2) {
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      const dist = Math.hypot(dx, dy);
      mapZoom = Math.min(MAP_ZOOM_MAX, Math.max(MAP_ZOOM_MIN, _touchStartZoom * (dist / _touchStartDist)));
      _clampPan();
      drawMinimap(_currentRoundNum);
    }
  }, { passive: false });
  canvas.addEventListener('touchend', function () { _mapDragging = false; }, { passive: true });
}

function _clampPan() {
  // Prevent panning beyond the map edges
  const maxPan = 1024 * (mapZoom - 1);
  mapPanX = Math.min(0, Math.max(-maxPan, mapPanX));
  mapPanY = Math.min(0, Math.max(-maxPan, mapPanY));
}

function minimapZoomIn() {
  const oldZoom = mapZoom;
  mapZoom = Math.min(MAP_ZOOM_MAX, mapZoom + MAP_ZOOM_STEP);
  // Zoom toward center
  const factor = mapZoom / oldZoom;
  mapPanX = 512 - factor * (512 - mapPanX);
  mapPanY = 512 - factor * (512 - mapPanY);
  _clampPan();
  drawMinimap(_currentRoundNum);
}

function minimapZoomOut() {
  const oldZoom = mapZoom;
  mapZoom = Math.max(MAP_ZOOM_MIN, mapZoom - MAP_ZOOM_STEP);
  const factor = mapZoom / oldZoom;
  mapPanX = 512 - factor * (512 - mapPanX);
  mapPanY = 512 - factor * (512 - mapPanY);
  _clampPan();
  drawMinimap(_currentRoundNum);
}

function minimapResetView() {
  mapZoom = 1.0;
  mapPanX = 0;
  mapPanY = 0;
  drawMinimap(_currentRoundNum);
}

function drawEvents(ctx, roundNum) {
  if (!minimapData) return;

  const rd = minimapData.rounds.find(r => r.round === roundNum);
  if (!rd || !rd.events.length) {
    renderEventList([]);
    return;
  }

  const events = rd.events;
  renderEventList(events);

  for (const ev of events) {
    if (ev.type === 'grenade') {
      ctx.save();
      drawGrenadeEvent(ctx, ev);
      ctx.restore();
      continue;
    }
    const x = ev.px, y = ev.py;
    if (x < 0 || x > 1024 || y < 0 || y > 1024) continue;

    ctx.save();
    if (ev.type === 'kill') {
      // Green crosshair for kills made by user
      ctx.strokeStyle = TC.success||'#34d399';
      ctx.fillStyle = TC.success||'#34d399';
      ctx.lineWidth = 2;
      drawCrosshair(ctx, x, y, 10);
      ctx.font = 'bold 11px Space Grotesk, sans-serif';
      ctx.fillStyle = 'rgba(52,211,153,0.9)';
      ctx.fillText(ev.victim || '', x + 14, y + 4);
    } else if (ev.type === 'death' || ev.type === 'player_death') {
      // Red skull X for deaths
      ctx.strokeStyle = ev.type === 'player_death' ? (TC.death||'#ff4466') : (TC.error||'#ef4444');
      ctx.lineWidth = ev.type === 'player_death' ? 3 : 2;
      drawSkullX(ctx, x, y, ev.type === 'player_death' ? 12 : 8);
      if (ev.type === 'player_death') {
        ctx.font = 'bold 11px Space Grotesk, sans-serif';
        ctx.fillStyle = 'rgba(255,68,102,0.9)';
        ctx.fillText('YOU', x + 14, y + 4);
      }
    } else if (ev.type === 'killer_pos') {
      // Orange triangle for enemy who killed user
      ctx.fillStyle = 'rgba(251,146,60,0.8)';
      drawTriangle(ctx, x, y, 10);
      ctx.font = 'bold 10px Space Grotesk, sans-serif';
      ctx.fillStyle = 'rgba(251,146,60,0.9)';
      ctx.fillText(ev.name || '', x + 14, y + 4);
    } else if (ev.type === 'flash_victim') {
      // Yellow flash icon for enemies blinded by player
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(253,224,71,0.25)';
      ctx.fill();
      ctx.strokeStyle = TC.flash||'#fde047';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // Lightning bolt
      ctx.font = 'bold 13px sans-serif';
      ctx.fillStyle = TC.flash||'#fde047';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('\u26A1', x, y);
      ctx.textAlign = 'start';
      ctx.textBaseline = 'alphabetic';
      // Label with duration
      ctx.font = 'bold 10px Space Grotesk, sans-serif';
      ctx.fillStyle = 'rgba(253,224,71,0.9)';
      const dur = ev.duration ? ev.duration.toFixed(1) + 's' : '';
      ctx.fillText((ev.name || '') + (dur ? ' ' + dur : ''), x + 14, y + 4);
    } else if (ev.type === 'he_victim') {
      // Red-orange circle for HE damage victims
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(248,113,113,0.25)';
      ctx.fill();
      ctx.strokeStyle = TC.he||'#f87171';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // Explosion icon
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = TC.he||'#f87171';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('\uD83D\uDCA5', x, y);
      ctx.textAlign = 'start';
      ctx.textBaseline = 'alphabetic';
      // Label with damage
      ctx.font = 'bold 10px Space Grotesk, sans-serif';
      ctx.fillStyle = 'rgba(248,113,113,0.9)';
      const heDmg = ev.damage ? ev.damage + 'dmg' : '';
      ctx.fillText((ev.name || '') + (heDmg ? ' ' + heDmg : ''), x + 14, y + 4);
    } else if (ev.type === 'molotov_victim') {
      // Orange circle for molotov/incendiary damage victims
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(249,115,22,0.25)';
      ctx.fill();
      ctx.strokeStyle = TC.molotov||'#f97316';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // Fire icon
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = TC.molotov||'#f97316';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('\uD83D\uDD25', x, y);
      ctx.textAlign = 'start';
      ctx.textBaseline = 'alphabetic';
      // Label with damage
      ctx.font = 'bold 10px Space Grotesk, sans-serif';
      ctx.fillStyle = 'rgba(249,115,22,0.9)';
      const molDmg = ev.damage ? ev.damage + 'dmg' : '';
      ctx.fillText((ev.name || '') + (molDmg ? ' ' + molDmg : ''), x + 14, y + 4);
    } else if (ev.type === 'grenade') {
      drawGrenadeEvent(ctx, ev);
    }
    ctx.restore();
  }
}

// Grenade colours by type
function _getNadeColors() {
  return {
    flash:  { main: TC.flash||'#fde047', glow: `rgba(253,224,71,0.3)`, label: 'Flash' },
    he:     { main: TC.he||'#f87171', glow: `rgba(248,113,113,0.3)`, label: 'HE' },
    molotov:{ main: TC.molotov||'#f97316', glow: `rgba(249,115,22,0.3)`, label: 'Molotov' },
    smoke:  { main: TC.smoke||'#cbd5e1', glow: `rgba(203,213,225,0.25)`, label: 'Smoke' },
  };
}

function drawGrenadeEvent(ctx, ev) {
  const nc = _getNadeColors()[ev.nade_type] || _getNadeColors().he;
  const hasThrow = ev.throw_px != null && ev.throw_py != null;
  const hasLand = ev.land_px != null && ev.land_py != null;

  if (hasThrow && hasLand) {
    // Dashed arc line from throw to land
    ctx.save();
    ctx.strokeStyle = nc.main;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.6;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(ev.throw_px, ev.throw_py);
    ctx.lineTo(ev.land_px, ev.land_py);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1.0;
    ctx.restore();
  }

  // Throw position: small filled circle with ring
  if (hasThrow) {
    ctx.save();
    ctx.fillStyle = nc.glow;
    ctx.beginPath();
    ctx.arc(ev.throw_px, ev.throw_py, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = nc.main;
    ctx.beginPath();
    ctx.arc(ev.throw_px, ev.throw_py, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // Land position: nade-type icon
  if (hasLand) {
    ctx.save();
    const lx = ev.land_px, ly = ev.land_py;

    if (ev.nade_type === 'flash') {
      // Starburst
      ctx.fillStyle = nc.glow;
      ctx.beginPath();
      ctx.arc(lx, ly, 14, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = nc.main;
      ctx.lineWidth = 2;
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI * 2 / 6) * i;
        ctx.beginPath();
        ctx.moveTo(lx + Math.cos(a) * 6, ly + Math.sin(a) * 6);
        ctx.lineTo(lx + Math.cos(a) * 12, ly + Math.sin(a) * 12);
        ctx.stroke();
      }
    } else if (ev.nade_type === 'smoke') {
      // Cloud circle
      ctx.fillStyle = 'rgba(203,213,225,0.2)';
      ctx.beginPath();
      ctx.arc(lx, ly, 18, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = nc.main;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(lx, ly, 18, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    } else if (ev.nade_type === 'molotov') {
      // Flame circle
      ctx.fillStyle = 'rgba(249,115,22,0.25)';
      ctx.beginPath();
      ctx.arc(lx, ly, 16, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = nc.main;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(lx, ly, 16, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      // HE: explosion burst
      ctx.fillStyle = nc.glow;
      ctx.beginPath();
      ctx.arc(lx, ly, 14, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = nc.main;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(lx, ly, 10, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Label
    const label = ev.land_callout || nc.label;
    ctx.font = 'bold 9px Space Grotesk, sans-serif';
    ctx.fillStyle = nc.main;
    ctx.fillText(label, lx + 16, ly + 3);
    ctx.restore();
  }
}

function drawCrosshair(ctx, x, y, size) {
  ctx.beginPath();
  ctx.arc(x, y, size * 0.6, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - size, y); ctx.lineTo(x + size, y);
  ctx.moveTo(x, y - size); ctx.lineTo(x, y + size);
  ctx.stroke();
}

function drawSkullX(ctx, x, y, size) {
  ctx.beginPath();
  ctx.moveTo(x - size, y - size); ctx.lineTo(x + size, y + size);
  ctx.moveTo(x + size, y - size); ctx.lineTo(x - size, y + size);
  ctx.stroke();
}

function drawTriangle(ctx, x, y, size) {
  ctx.beginPath();
  ctx.moveTo(x, y - size);
  ctx.lineTo(x - size * 0.866, y + size * 0.5);
  ctx.lineTo(x + size * 0.866, y + size * 0.5);
  ctx.closePath();
  ctx.fill();
}

function renderEventList(events) {
  const cont = document.getElementById('minimap-event-list');
  cont.innerHTML = '';
  if (!events.length) {
    cont.innerHTML = '<div class="text-on-surface-variant col-span-2 text-center py-2">No position data for this round</div>';
    return;
  }
  for (const ev of events) {
    const div = document.createElement('div');
    div.className = 'flex items-center gap-2 px-2 py-1 rounded bg-surface-container-highest/50';
    let icon = '', color = '', label = '';
    if (ev.type === 'kill') {
      icon = 'gps_fixed'; color = 'text-good';
      label = `Killed ${esc(ev.victim)} — ${esc(ev.weapon||'?')}${ev.headshot ? ' HS' : ''}`;
    } else if (ev.type === 'player_death') {
      icon = 'dangerous'; color = 'text-error';
      label = `Died to ${esc(ev.killer)} — ${esc(ev.weapon||'?')}`;
    } else if (ev.type === 'death') {
      icon = 'skull'; color = 'text-bad/60';
      label = `${esc(ev.name)} killed (enemy pos)`;
    } else if (ev.type === 'killer_pos') {
      icon = 'warning'; color = 'text-warn';
      label = `${esc(ev.name)} (killed you from here)`;
    } else if (ev.type === 'flash_victim') {
      icon = 'flash_on'; color = 'text-caution';
      const dur = ev.duration ? ev.duration.toFixed(1) + 's' : '';
      label = `⚡ ${esc(ev.name)} flashed${dur ? ' ' + dur : ''}`;
    } else if (ev.type === 'he_victim') {
      icon = 'explosion'; color = 'text-bad';
      const dmg = ev.damage ? ev.damage + 'dmg' : '';
      label = `💥 ${esc(ev.name)} HE${dmg ? ' ' + dmg : ''}`;
    } else if (ev.type === 'molotov_victim') {
      icon = 'local_fire_department'; color = 'text-warn';
      const dmg = ev.damage ? ev.damage + 'dmg' : '';
      label = `🔥 ${esc(ev.name)} molotov${dmg ? ' ' + dmg : ''}`;
    } else if (ev.type === 'grenade') {
      const nadeIcons = { flash: 'flash_on', he: 'explosion', molotov: 'local_fire_department', smoke: 'cloud' };
      const nadeColors = { flash: 'text-caution', he: 'text-bad', molotov: 'text-warn', smoke: 'text-on-surface-variant' };
      const nadeLabels = { flash: 'Flash', he: 'HE', molotov: 'Molotov', smoke: 'Smoke' };
      icon = nadeIcons[ev.nade_type] || 'bomb';
      color = nadeColors[ev.nade_type] || 'text-on-surface-variant';
      const from = ev.throw_callout ? ` from ${esc(ev.throw_callout)}` : '';
      const to = ev.land_callout ? ` → ${esc(ev.land_callout)}` : '';
      label = `${nadeLabels[ev.nade_type] || 'Nade'}${from}${to}`;
    }
    div.innerHTML = `<span class="material-symbols-outlined ${color}" style="font-size:14px">${icon}</span><span class="${color}">${label}</span>`;
    cont.appendChild(div);
  }
}

// ---------------------------------------------------------------------------
// AI Coach Zone Highlighting
// ---------------------------------------------------------------------------
function highlightCalloutsOnMinimap(calloutNames) {
  if (!currentMapName || !minimapData) return;

  fetch(API + '/minimap/zones', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ map_name: currentMapName, callouts: calloutNames }),
  })
  .then(r => r.json())
  .then(data => {
    if (!data.zones || !data.zones.length) return;
    const canvas = document.getElementById('minimap-canvas');
    const ctx = canvas.getContext('2d');

    // Apply current zoom/pan transform
    ctx.setTransform(mapZoom, 0, 0, mapZoom, mapPanX, mapPanY);

    for (const z of data.zones) {
      ctx.save();
      // Pulsing ring for zone center
      ctx.strokeStyle = 'rgba(251,191,36,0.7)';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.arc(z.px, z.py, 30, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      // Zone label
      ctx.font = 'bold 12px Space Grotesk, sans-serif';
      ctx.fillStyle = 'rgba(251,191,36,0.9)';
      const w = ctx.measureText(z.callout).width;
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(z.px - w/2 - 4, z.py - 46, w + 8, 16);
      ctx.fillStyle = 'rgba(251,191,36,0.95)';
      ctx.fillText(z.callout, z.px - w/2, z.py - 34);
      ctx.restore();
    }

    // Reset transform
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Show minimap section if hidden
    const section = document.getElementById('minimap-section');
    if (section.classList.contains('hidden')) {
      section.classList.remove('hidden');
      document.getElementById('minimap-title').textContent = 'AI Callouts';
    }
  })
  .catch(err => console.error('Zone highlight error:', err));
}

// ---------------------------------------------------------------------------
// Debug: show ALL event positions from all rounds on the radar
// ---------------------------------------------------------------------------
const _ZONE_DOT_COLORS = {};
let _zoneColorIdx = 0;
const _DOT_PALETTE = [
  '#ef4444','#f97316','#eab308','#22c55e','#14b8a6','#3b82f6','#8b5cf6',
  '#ec4899','#f43f5e','#84cc16','#06b6d4','#6366f1','#d946ef','#fb923c',
  '#a3e635','#2dd4bf','#818cf8','#f472b6','#fbbf24','#34d399',
];
function _zoneColor(zone) {
  if (!_ZONE_DOT_COLORS[zone]) {
    _ZONE_DOT_COLORS[zone] = _DOT_PALETTE[_zoneColorIdx % _DOT_PALETTE.length];
    _zoneColorIdx++;
  }
  return _ZONE_DOT_COLORS[zone];
}

async function toggleDebugPositions() {
  _debugMode = !_debugMode;
  const btn = document.getElementById('debug-positions-btn');
  btn.style.color = _debugMode ? '#fbbf24' : '';

  if (_debugMode && !_debugPositions && currentMapName) {
    try {
      const res = await fetch(API + '/minimap/' + encodeURIComponent(currentMapName) + '/debug-positions');
      if (res.ok) _debugPositions = await res.json();
    } catch (e) { console.error('Debug positions fetch error:', e); }
  }
  drawMinimap(_currentRoundNum);
}

function drawDebugPositions(ctx) {
  if (!_debugMode || !_debugPositions || !_debugPositions.positions) return;

  const pts = _debugPositions.positions;

  // Draw dots
  for (const p of pts) {
    const col = _zoneColor(p.zone);
    ctx.fillStyle = col;
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    ctx.arc(p.px, p.py, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1.0;
  }

  // Compute per-zone centroids and draw labels
  const zoneSums = {};
  for (const p of pts) {
    if (!zoneSums[p.zone]) zoneSums[p.zone] = { sx: 0, sy: 0, n: 0 };
    zoneSums[p.zone].sx += p.px;
    zoneSums[p.zone].sy += p.py;
    zoneSums[p.zone].n++;
  }
  for (const [zone, s] of Object.entries(zoneSums)) {
    const cx = s.sx / s.n;
    const cy = s.sy / s.n;
    const col = _zoneColor(zone);

    // Background pill
    ctx.font = 'bold 10px Space Grotesk, sans-serif';
    const w = ctx.measureText(zone + ' (' + s.n + ')').width;
    ctx.fillStyle = 'rgba(0,0,0,0.75)';
    ctx.fillRect(cx - w/2 - 3, cy - 18, w + 6, 14);

    // Label
    ctx.fillStyle = col;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(zone + ' (' + s.n + ')', cx, cy - 11);
    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';

    // Centroid crosshair
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
    ctx.stroke();
  }

  // Summary in corner
  const zoneCount = Object.keys(zoneSums).length;
  ctx.font = 'bold 11px Space Grotesk, sans-serif';
  ctx.fillStyle = 'rgba(251,191,36,0.9)';
  ctx.fillText('DEBUG: ' + pts.length + ' positions / ' + zoneCount + ' zones', 10, 20);

  // Draw a pixel coordinate grid for calibration
  ctx.font = '9px monospace';
  ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.3)';
  ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.06)';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 1024; i += 100) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 1024); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(1024, i); ctx.stroke();
    ctx.fillText(String(i), i + 2, 10);
    ctx.fillText(String(i), 2, i + 10);
  }
}

function extractCalloutsFromText(text) {
  // Known callout names to search for in AI responses
  const knownCallouts = [
    'A Site','A Default','A Ramp','Stairs','Tetris','Firebox','Jungle','Ticket',
    'CT Spawn','Snipers Nest','Connector','Chair','Top Mid','Mid Window','Catwalk',
    'Window','Short','Mid','Underpass','Ladder Room','B Site','B Van','Bench',
    'Market','Market Door','B Apartments','B Short','Kitchen','T Spawn','T Ramp',
    'A Palace','A Main','B Apartments Entrance','A Side','B Side','Mid Area','T Area',
    'Pit','Balcony','Library','Arch','Graveyard','Truck','Alt Mid','Top Mid',
    'Banana','Oranges','Car','Dark','CT','Construction','New Box','Coffins',
    'T Apartments','Apartments','Second Mid','Boiler',
    'A Long','A Long Doors','A Pit','A Car','Goose','Mid Doors','Xbox',
    'Lower Tunnels','Upper Tunnels','B Tunnels','B Doors','B Window','B Back Site','B Car',
    'T Mid','Outside Long','Hut','Heaven','Hell','Squeaky','Main','Lobby','Ramp',
    'Outside','Secret','A Bridge','A Connector','B Main','B Pillar','B Connector',
  ];
  const found = [];
  for (const c of knownCallouts) {
    if (text.includes(c) && !found.includes(c)) found.push(c);
  }
  return found;
}

// ===================================================================
// 2D Replay Player (embedded)
// ===================================================================
let replayMatchMeta = null;
let replayRoundData = null;
let replayRadarImage = null;
let replayRadarLoaded = false;
let replayPlaying = false;
let replaySpeed = 1;
let replayFrameIdx = 0;
let replayLastTs = 0;
let replayAccum = 0;
let replayCanvas = null;
let replayCtx = null;

const REPLAY_TICK_RATE = 64;
const REPLAY_SAMPLE_INTERVAL = 32;
const REPLAY_MS_PER_FRAME = (REPLAY_SAMPLE_INTERVAL / REPLAY_TICK_RATE) * 1000;

async function initReplayPlayer() {
  replayCanvas = document.getElementById('replay-canvas');
  if (!replayCanvas) return;
  replayCtx = replayCanvas.getContext('2d');

  document.getElementById('replay-loading').classList.remove('hidden');
  document.getElementById('replay-player').classList.add('hidden');
  document.getElementById('replay-no-data').classList.add('hidden');

  try {
    const res = await fetch(API + '/matches/' + encodeURIComponent(currentMatchId) + '/replay');
    if (!res.ok) throw new Error('No replay');
    replayMatchMeta = await res.json();

    document.getElementById('replay-loading').classList.add('hidden');

    if (!replayMatchMeta.has_replay) {
      document.getElementById('replay-no-data').classList.remove('hidden');
      return;
    }

    document.getElementById('replay-player').classList.remove('hidden');
    document.getElementById('replay-round-controls').classList.remove('hidden');

    // Populate round list
    const rl = document.getElementById('replay-round-list');
    rl.innerHTML = '';
    const enriched = window._enrichedRounds || [];
    for (const r of replayMatchMeta.rounds) {
      const rn = r.round;
      const er = enriched.find(e => (e.round_number || e.enriched?.round) === rn);
      const side = er?.enriched?.side || '';
      const winner = er?.enriched?.round_winner || '';
      const playerSide = side === 'CT' ? 'CT' : (side === 'T' ? 'T' : '');
      const won = winner && playerSide ? winner === playerSide : null;

      const btn = document.createElement('button');
      btn.className = 'replay-round-btn';
      btn.dataset.round = rn;
      btn.onclick = () => replayLoadRound(rn);

      let html = '<span class="rr-num">' + rn + '</span>';
      if (playerSide) {
        html += '<span class="rr-side ' + playerSide.toLowerCase() + '">' + playerSide + '</span>';
      }
      if (won !== null) {
        html += '<span class="rr-result ' + (won ? 'win' : 'loss') + '">' + (won ? 'W' : 'L') + '</span>';
      }
      btn.innerHTML = html;
      rl.appendChild(btn);
    }
    document.getElementById('replay-round-list-wrap').classList.remove('hidden');

    // Load radar
    replayRadarImage = new Image();
    replayRadarImage.onload = () => { replayRadarLoaded = true; replayDrawFrame(); };
    replayRadarImage.onerror = () => { replayRadarLoaded = false; replayDrawFrame(); };
    replayRadarImage.src = replayMatchMeta.radar_image;

    // Load first round
    const first = replayMatchMeta.rounds[0]?.round || 1;
    await replayLoadRound(first);

    // Set up timeline scrubber
    replaySetupTimeline();
  } catch (err) {
    document.getElementById('replay-loading').classList.add('hidden');
    document.getElementById('replay-no-data').classList.remove('hidden');
    console.error('Replay init error:', err);
  }
}

async function replayLoadRound(roundNum) {
  replayStop();
  replayFrameIdx = 0;
  replayAccum = 0;
  document.getElementById('replay-kill-feed').innerHTML = '';

  try {
    const res = await fetch(API + '/matches/' + encodeURIComponent(currentMatchId) + '/replay?round_number=' + roundNum);
    if (!res.ok) throw new Error('Failed to fetch round');
    replayRoundData = await res.json();

    // Highlight active round in list
    document.querySelectorAll('.replay-round-btn').forEach(b => b.classList.toggle('active', parseInt(b.dataset.round) === roundNum));
    document.getElementById('replay-ri-round').textContent = roundNum;
    document.getElementById('replay-ri-frames').textContent = replayRoundData.frames?.length || 0;
    const totalTicks = replayRoundData.frames?.length ? replayRoundData.frames[replayRoundData.frames.length - 1][0] : 0;
    document.getElementById('replay-ri-duration').textContent = replayFormatTime(totalTicks / REPLAY_TICK_RATE);

    replayBuildPlayerList(replayRoundData.players || {});
    replayUpdateTime();
    replayDrawFrame();
  } catch (err) {
    console.error('Replay round error:', err);
  }
}

function replayBuildPlayerList(players) {
  const ctEl = document.getElementById('replay-ct-players');
  const tEl = document.getElementById('replay-t-players');
  ctEl.innerHTML = '<div class="text-[10px] font-bold text-info uppercase tracking-widest mb-1">Counter-Terrorists</div>';
  tEl.innerHTML = '<div class="text-[10px] font-bold text-caution uppercase tracking-widest mb-1">Terrorists</div>';

  for (const [sid, info] of Object.entries(players)) {
    const div = document.createElement('div');
    div.className = 'flex items-center gap-2 px-2 py-1 rounded';
    div.id = 'rp-player-' + sid;
    const color = info.team === 3 ? 'bg-info' : 'bg-caution';
    div.innerHTML = '<div class="w-2.5 h-2.5 rounded-full ' + color + ' flex-shrink-0"></div>' +
      '<span class="text-xs text-on-surface truncate">' + replayEscape(info.name) + '</span>' +
      '<span class="ml-auto text-[10px] text-on-surface-variant rp-hp" data-sid="' + sid + '">100</span>';
    if (info.team === 3) ctEl.appendChild(div);
    else tEl.appendChild(div);
  }
}

function replayDrawFrame() {
  if (!replayCtx) return;
  const ctx = replayCtx;
  ctx.clearRect(0, 0, 1024, 1024);

  if (replayRadarLoaded && replayRadarImage) {
    ctx.drawImage(replayRadarImage, 0, 0, 1024, 1024);
  } else {
    ctx.fillStyle = TC.canvasBg||'#0a1628';
    ctx.fillRect(0, 0, 1024, 1024);
  }

  if (!replayRoundData || !replayRoundData.frames || replayRoundData.frames.length === 0) return;

  const players = replayRoundData.players || {};
  const frames = replayRoundData.frames;
  const events = replayRoundData.events || [];
  const idx = Math.min(replayFrameIdx, frames.length - 1);
  const currentTick = frames[idx][0];
  const positions = frames[idx][1];

  // Interpolation
  let ip = positions;
  if (replayPlaying && idx < frames.length - 1) {
    const np = frames[idx + 1][1];
    const td = frames[idx + 1][0] - frames[idx][0];
    if (td > 0) {
      const t = Math.min(replayAccum / (REPLAY_MS_PER_FRAME / replaySpeed), 1);
      ip = {};
      for (const sid of Object.keys(positions)) {
        if (np[sid]) {
          ip[sid] = [
            positions[sid][0] + (np[sid][0] - positions[sid][0]) * t,
            positions[sid][1] + (np[sid][1] - positions[sid][1]) * t,
            positions[sid][2],
          ];
        } else {
          ip[sid] = positions[sid];
        }
      }
      for (const sid of Object.keys(np)) {
        if (!ip[sid]) ip[sid] = np[sid];
      }
    }
  }

  // Movement trails
  const trailLen = Math.min(10, idx);
  if (trailLen > 0) {
    ctx.globalAlpha = 0.15;
    ctx.lineWidth = 2;
    for (const sid of Object.keys(players)) {
      ctx.strokeStyle = players[sid].team === 3 ? (TC.ct||'#60a5fa') : (TC.t||'#facc15');
      ctx.beginPath();
      let started = false;
      for (let ti = Math.max(0, idx - trailLen); ti <= idx; ti++) {
        const pos = frames[ti][1][sid];
        if (!pos || pos[2] <= 0) continue;
        if (!started) { ctx.moveTo(pos[0], pos[1]); started = true; }
        else ctx.lineTo(pos[0], pos[1]);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // Kill lines
  for (const ev of events) {
    if (ev.type !== 'kill') continue;
    const diff = currentTick - ev.t;
    if (diff < 0 || diff > REPLAY_SAMPLE_INTERVAL * 4) continue;
    const ap = positions[ev.attacker], vp = positions[ev.victim];
    if (!ap || !vp) continue;
    const alpha = Math.max(0, 1 - diff / (REPLAY_SAMPLE_INTERVAL * 4));
    ctx.globalAlpha = alpha * 0.6;
    ctx.strokeStyle = TC.kill||'#ff6e84';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(ap[0], ap[1]);
    ctx.lineTo(vp[0], vp[1]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    // Death X
    if (diff < REPLAY_SAMPLE_INTERVAL * 6) {
      const xa = Math.max(0, 1 - diff / (REPLAY_SAMPLE_INTERVAL * 6));
      ctx.globalAlpha = xa * 0.8;
      ctx.strokeStyle = TC.kill||'#ff6e84';
      ctx.lineWidth = 3;
      const sz = 8;
      ctx.beginPath();
      ctx.moveTo(vp[0] - sz, vp[1] - sz); ctx.lineTo(vp[0] + sz, vp[1] + sz);
      ctx.moveTo(vp[0] + sz, vp[1] - sz); ctx.lineTo(vp[0] - sz, vp[1] + sz);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  // Grenade events
  const NADE_DURATIONS = { flash: 128, he: 128, smoke: 1152, molotov: 448 };
  const NADE_COLORS = { flash: '#fffbe6', he: '#ff6e84', smoke: '#a3e635', molotov: '#fb923c' };
  const NADE_ICONS = { flash: '⚡', he: '💥', smoke: '💨', molotov: '🔥' };
  const NADE_RADIUS = { flash: 14, he: 14, smoke: 24, molotov: 22 };
  for (const ev of events) {
    if (ev.type !== 'grenade' || ev.px == null || ev.py == null) continue;
    const diff = currentTick - ev.t;
    const duration = NADE_DURATIONS[ev.grenade] || 128;
    if (diff < 0 || diff > duration) continue;
    const progress = diff / duration;
    const gx = ev.px, gy = ev.py;
    const color = NADE_COLORS[ev.grenade] || '#ffffff';
    const baseRadius = NADE_RADIUS[ev.grenade] || 14;

    // Flight path from thrower to detonation
    if (ev.thrower) {
      let throwerPos = null;
      for (let fi = Math.max(0, idx - 4); fi <= Math.min(frames.length - 1, idx + 2); fi++) {
        if (frames[fi][1][ev.thrower]) {
          const fp = frames[fi][1][ev.thrower];
          if (fp[2] > 0) { throwerPos = fp; break; }
        }
      }
      if (throwerPos) {
        ctx.globalAlpha = Math.max(0.1, 0.5 * (1 - progress));
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(throwerPos[0], throwerPos[1]);
        ctx.lineTo(gx, gy);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = Math.max(0.1, 0.4 * (1 - progress));
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(throwerPos[0], throwerPos[1], 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Filled circle
    if (ev.grenade === 'smoke') {
      const fadeStart = 1 - (3 * REPLAY_TICK_RATE / duration);
      const smokeAlpha = progress < fadeStart ? 0.45 : 0.45 * (1 - (progress - fadeStart) / (1 - fadeStart));
      ctx.globalAlpha = Math.max(0.08, smokeAlpha);
      ctx.fillStyle = '#a3e635';
      ctx.beginPath();
      ctx.arc(gx, gy, baseRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = Math.max(0.15, smokeAlpha * 0.7);
      ctx.strokeStyle = '#a3e635';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(gx, gy, baseRadius + 3, 0, Math.PI * 2);
      ctx.stroke();
    } else if (ev.grenade === 'molotov') {
      const fadeStart = 1 - (2 * REPLAY_TICK_RATE / duration);
      const fireAlpha = progress < fadeStart ? 0.5 : 0.5 * (1 - (progress - fadeStart) / (1 - fadeStart));
      ctx.globalAlpha = Math.max(0.08, fireAlpha);
      ctx.fillStyle = '#fb923c';
      ctx.beginPath();
      ctx.arc(gx, gy, baseRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = Math.max(0.15, fireAlpha * 0.7);
      ctx.strokeStyle = '#ff6347';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(gx, gy, baseRadius + 3, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      const burstRadius = baseRadius + progress * 12;
      const burstAlpha = Math.max(0.1, 0.7 * (1 - progress));
      ctx.globalAlpha = burstAlpha;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(gx, gy, burstRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = burstAlpha * 0.8;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(gx, gy, burstRadius + 3, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Activation timer for smoke / molotov
    if (ev.grenade === 'smoke' || ev.grenade === 'molotov') {
      const totalSec = (ev.grenade === 'smoke') ? 18 : 7;
      const elapsedSec = diff / REPLAY_TICK_RATE;
      const remainSec = Math.max(0, totalSec - elapsedSec);
      const fadeAlpha = progress < 0.8 ? 0.9 : 0.9 * (1 - (progress - 0.8) / 0.2);
      ctx.globalAlpha = Math.max(0.15, fadeAlpha);
      ctx.font = 'bold 11px "Space Grotesk", sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#ffffff';
      ctx.fillText(remainSec.toFixed(1) + 's', gx, gy + baseRadius + 12);
    }

    // Center icon
    const iconAlpha = (ev.grenade === 'smoke' || ev.grenade === 'molotov')
      ? Math.max(0.15, progress < 0.85 ? 0.9 : 0.9 * (1 - (progress - 0.85) / 0.15))
      : Math.max(0.2, 1 - progress);
    ctx.globalAlpha = iconAlpha;
    ctx.font = 'bold 13px "Space Grotesk", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(NADE_ICONS[ev.grenade] || '◆', gx, gy);

    // Thrower name label
    if (ev.thrower && players[ev.thrower]) {
      const throwerInfo = players[ev.thrower];
      ctx.globalAlpha = Math.max(0.1, 0.7 * (1 - progress * 0.8));
      ctx.font = 'bold 9px "Space Grotesk", sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = throwerInfo.team === 3 ? (TC.ctName||'#93c5fd') : (TC.tName||'#fde047');
      ctx.fillText(throwerInfo.name || '', gx, gy - baseRadius - 4);
    }

    ctx.globalAlpha = 1;
    ctx.textBaseline = 'alphabetic';
  }

  // Player dots
  for (const [sid, pos] of Object.entries(ip)) {
    const px = pos[0], py = pos[1], hp = pos[2];
    const info = players[sid];
    if (!info) continue;
    const isCT = info.team === 3;
    if (hp <= 0) {
      ctx.globalAlpha = 0.3;
      ctx.strokeStyle = isCT ? (TC.ct||'#60a5fa') : (TC.t||'#facc15');
      ctx.lineWidth = 2;
      const sz = 5;
      ctx.beginPath();
      ctx.moveTo(px - sz, py - sz); ctx.lineTo(px + sz, py + sz);
      ctx.moveTo(px + sz, py - sz); ctx.lineTo(px - sz, py + sz);
      ctx.stroke();
      ctx.globalAlpha = 1;
      continue;
    }
    const radius = 10;
    ctx.shadowColor = isCT ? 'rgba(96,165,250,0.5)' : 'rgba(250,204,21,0.5)';
    ctx.shadowBlur = 8;
    ctx.fillStyle = isCT ? (TC.ct||'#3b82f6') : (TC.t||'#eab308');
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    if (hp < 100) {
      ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.3)';
      const hpAngle = (hp / 100) * Math.PI * 2;
      ctx.strokeStyle = hp > 50 ? (TC.hpGood||'#4ade80') : hp > 25 ? (TC.hpWarn||'#facc15') : (TC.kill||'#ff6e84');
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(px, py, radius + 2, -Math.PI / 2, -Math.PI / 2 + hpAngle, false);
      ctx.stroke();
    }
    ctx.font = 'bold 10px "Space Grotesk", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = isCT ? (TC.ctName||'#93c5fd') : (TC.tName||'#fde047');
    ctx.fillText(info.name || sid.slice(0, 6), px, py - radius - 5);
  }

  // HP in sidebar
  document.querySelectorAll('.rp-hp').forEach(el => {
    const sid = el.dataset.sid;
    const pos = positions[sid];
    if (pos) {
      el.textContent = pos[2] > 0 ? pos[2] : '\u2620';
      el.className = 'ml-auto text-[10px] rp-hp ' + (pos[2] > 0 ? 'text-on-surface-variant' : 'text-error/50');
    }
  });

  // Kill feed
  for (const ev of events) {
    if (ev.t <= currentTick && !ev._shown) {
      ev._shown = true;
      if (ev.type === 'kill') replayAddKill(ev, players);
      else if (ev.type === 'grenade') replayAddGrenade(ev, players);
    }
  }
}

function replayAddKill(ev, players) {
  const feed = document.getElementById('replay-kill-feed');
  const aName = players[ev.attacker]?.name || '?';
  const vName = players[ev.victim]?.name || '?';
  const aC = players[ev.attacker]?.team === 3 ? 'text-info' : 'text-caution';
  const vC = players[ev.victim]?.team === 3 ? 'text-info' : 'text-caution';
  const hs = ev.headshot ? ' <span class="text-error">HS</span>' : '';
  const div = document.createElement('div');
  div.className = 'kill-feed-item flex items-center gap-1 py-0.5';
  div.innerHTML = '<span class="' + aC + ' font-bold truncate max-w-[70px]">' + replayEscape(aName) + '</span>' +
    '<span class="text-on-surface-variant/50 text-[10px]">[' + replayEscape(ev.weapon || '?') + hs + ']</span>' +
    '<span class="' + vC + ' truncate max-w-[70px]">' + replayEscape(vName) + '</span>';
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
  setTimeout(() => { div.classList.add('fade-out'); setTimeout(() => div.remove(), 500); }, 8000);
}

function replayAddGrenade(ev, players) {
  const feed = document.getElementById('replay-kill-feed');
  const throwerName = players[ev.thrower]?.name || '';
  const throwerTeam = players[ev.thrower]?.team;
  const tC = throwerTeam === 3 ? 'text-info' : throwerTeam === 2 ? 'text-caution' : 'text-on-surface-variant';
  const labels = { flash: '⚡ Flash', he: '💥 HE', smoke: '💨 Smoke', molotov: '🔥 Molotov' };
  const colors = { flash: 'text-caution', he: 'text-bad', smoke: 'text-good', molotov: 'text-warn' };
  const label = labels[ev.grenade] || ev.grenade;
  const nC = colors[ev.grenade] || 'text-on-surface-variant';
  const div = document.createElement('div');
  div.className = 'kill-feed-item flex items-center gap-1 py-0.5';
  if (throwerName) {
    div.innerHTML = '<span class="' + tC + ' font-bold truncate max-w-[70px]">' + replayEscape(throwerName) + '</span>' +
      '<span class="' + nC + ' text-[10px]">' + label + '</span>';
  } else {
    div.innerHTML = '<span class="' + nC + ' text-[10px]">' + label + '</span>';
  }
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
  setTimeout(() => { div.classList.add('fade-out'); setTimeout(() => div.remove(), 500); }, 5000);
}

function replayGameLoop(ts) {
  if (!replayPlaying) return;
  if (replayLastTs === 0) replayLastTs = ts;
  const dt = ts - replayLastTs;
  replayLastTs = ts;
  replayAccum += dt * replaySpeed;

  let stepped = false;
  while (replayAccum >= REPLAY_MS_PER_FRAME) {
    replayAccum -= REPLAY_MS_PER_FRAME;
    replayFrameIdx++;
    stepped = true;
  }

  if (replayRoundData && replayRoundData.frames && replayFrameIdx >= replayRoundData.frames.length) {
    replayFrameIdx = replayRoundData.frames.length - 1;
    replayStop();
    return;
  }

  if (stepped) { replayDrawFrame(); replayUpdateTime(); }
  else if (replayPlaying) replayDrawFrame();
  requestAnimationFrame(replayGameLoop);
}

function replayTogglePlay() { if (replayPlaying) replayStop(); else replayPlay(); }

function replayPlay() {
  if (!replayRoundData?.frames?.length) return;
  if (replayFrameIdx >= replayRoundData.frames.length - 1) {
    replayFrameIdx = 0;
    replayAccum = 0;
    if (replayRoundData.events) replayRoundData.events.forEach(e => e._shown = false);
    document.getElementById('replay-kill-feed').innerHTML = '';
  }
  replayPlaying = true;
  replayLastTs = 0;
  document.getElementById('replay-play-icon').textContent = 'pause';
  requestAnimationFrame(replayGameLoop);
}

function replayStop() {
  replayPlaying = false;
  document.getElementById('replay-play-icon').textContent = 'play_arrow';
}

function replaySetSpeed(s) {
  replaySpeed = s;
  document.querySelectorAll('#replay-section .speed-btn').forEach(btn => {
    btn.classList.toggle('active', parseFloat(btn.textContent) === s);
  });
}

function replayPrevRound() {
  if (!replayMatchMeta?.rounds) return;
  const activeBtn = document.querySelector('.replay-round-btn.active');
  const cur = activeBtn ? parseInt(activeBtn.dataset.round) : 0;
  const rounds = replayMatchMeta.rounds.map(r => r.round);
  const i = rounds.indexOf(cur);
  if (i > 0) replayLoadRound(rounds[i - 1]);
}

function replayNextRound() {
  if (!replayMatchMeta?.rounds) return;
  const activeBtn = document.querySelector('.replay-round-btn.active');
  const cur = activeBtn ? parseInt(activeBtn.dataset.round) : 0;
  const rounds = replayMatchMeta.rounds.map(r => r.round);
  const i = rounds.indexOf(cur);
  if (i < rounds.length - 1) replayLoadRound(rounds[i + 1]);
}

function replaySetupTimeline() {
  const bar = document.getElementById('replay-timeline-bar');
  if (!bar) return;
  let dragging = false;
  function seekTo(e) {
    if (!replayRoundData?.frames?.length) return;
    const rect = bar.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    replayFrameIdx = Math.floor(pct * (replayRoundData.frames.length - 1));
    replayAccum = 0;
    const tick = replayRoundData.frames[replayFrameIdx][0];
    if (replayRoundData.events) replayRoundData.events.forEach(ev => { ev._shown = ev.t <= tick; });
    document.getElementById('replay-kill-feed').innerHTML = '';
    if (replayRoundData.events) {
      for (const ev of replayRoundData.events) {
        if (ev.t <= tick && ev.type === 'kill') replayAddKill(ev, replayRoundData.players || {});
      }
    }
    replayDrawFrame();
    replayUpdateTime();
  }
  bar.addEventListener('mousedown', (e) => { dragging = true; seekTo(e); });
  window.addEventListener('mousemove', (e) => { if (dragging) seekTo(e); });
  window.addEventListener('mouseup', () => { dragging = false; });
}

function replayUpdateTime() {
  if (!replayRoundData?.frames?.length) return;
  const total = replayRoundData.frames.length - 1;
  const idx = Math.min(replayFrameIdx, total);
  const pct = total > 0 ? (idx / total) * 100 : 0;
  document.getElementById('replay-timeline-progress').style.width = pct + '%';
  document.getElementById('replay-timeline-thumb').style.left = 'calc(' + pct + '% - 8px)';
  const curTick = replayRoundData.frames[idx]?.[0] || 0;
  const totTick = replayRoundData.frames[total]?.[0] || 0;
  document.getElementById('replay-time-current').textContent = replayFormatTime(curTick / REPLAY_TICK_RATE);
  document.getElementById('replay-time-total').textContent = replayFormatTime(totTick / REPLAY_TICK_RATE);
}

function replayFormatTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
}

function replayEscape(str) {
  const d = document.createElement('div');
  d.textContent = str || '';
  return d.innerHTML;
}

/* The "watch the replay" link in the header. Was an expression in an onclick
   attribute, alongside a preventDefault the link needs to stay on the page. */
function scrollToReplaySection() {
  document.getElementById('replay-section').scrollIntoView({ behavior: 'smooth' });
}

/* Fullscreen, for the two panels that are a square map plus the controls that
   drive it: the 2D replay, and round-by-round with its minimap.

   The section covers the viewport and stops scrolling, so the map can no longer
   be sized against the viewport — at 90vh tall it pushed the replay's playback
   controls below the fold, in the one view whose point is to have everything
   visible at once. The map gets what its column has left over instead, measured
   here and handed to the stylesheet as --fs-canvas.

   Measured rather than assumed because it varies: the replay's round list is
   there or it isn't, the event list under the minimap grows with the round, and
   the controls wrap on a narrow window. */
const FULLSCREEN_PANELS = {
  replay: {
    section: 'replay-section',
    icon: 'replay-fs-icon',
    square: 'replay-canvas-wrap',   // gets the height left over
    below: ['replay-controls'],     // and what it is left over from
  },
  rounds: {
    section: 'rounds-section',
    icon: 'rounds-fs-icon',
    square: 'minimap-container',
    below: ['minimap-event-list'],
  },
};

const FULLSCREEN_MIN_CANVAS = 200;

function _outerHeight(el) {
  const style = getComputedStyle(el);
  return el.offsetHeight + parseFloat(style.marginTop) + parseFloat(style.marginBottom);
}

function fitFullscreenPanels() {
  for (const panel of Object.values(FULLSCREEN_PANELS)) {
    const section = document.getElementById(panel.section);
    const square = document.getElementById(panel.square);
    if (!section || !square) continue;
    if (!section.classList.contains('panel-fullscreen')) {
      section.style.removeProperty('--fs-canvas');
      continue;
    }

    // Whatever is above the square already has its height and does not move
    // when the square shrinks, so where the square starts is settled. Below
    // it, add up what has to stay visible.
    const bottom = section.getBoundingClientRect().bottom
      - parseFloat(getComputedStyle(section).paddingBottom);
    let taken = 0;
    for (const id of panel.below) {
      const el = document.getElementById(id);
      if (el && el.offsetParent !== null) taken += _outerHeight(el);
    }
    const available = bottom - square.getBoundingClientRect().top - taken;
    section.style.setProperty('--fs-canvas', Math.max(FULLSCREEN_MIN_CANVAS, available) + 'px');
  }
}

function togglePanelFullscreen(panel) {
  const section = document.getElementById(panel.section);
  const isFs = section.classList.toggle('panel-fullscreen');
  document.getElementById(panel.icon).textContent = isFs ? 'fullscreen_exit' : 'fullscreen';
  document.body.style.overflow = isFs ? 'hidden' : '';
  fitFullscreenPanels();
  // Leaving puts the section back in the flow, below wherever the page has
  // been sitting all this time.
  if (!isFs) section.scrollIntoView({ block: 'nearest' });
}

function replayToggleFullscreen() {
  togglePanelFullscreen(FULLSCREEN_PANELS.replay);
}

function roundsToggleFullscreen() {
  togglePanelFullscreen(FULLSCREEN_PANELS.rounds);
}

// ESC leaves whichever panel is fullscreen.
window.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  for (const panel of Object.values(FULLSCREEN_PANELS)) {
    if (document.getElementById(panel.section)?.classList.contains('panel-fullscreen')) {
      togglePanelFullscreen(panel);
    }
  }
});

window.addEventListener('resize', fitFullscreenPanels);


/* What this file offers the markup. See js/actions.js. */
registerActions({
  addPromptTemplate,
  clearChat,
  closeSyncModal,
  deleteCurrentMatch,
  deleteMatch,
  exportMatch,
  loadMatchList,
  minimapResetView,
  minimapZoomIn,
  minimapZoomOut,
  openAISettings,
  openMatch,
  openMinimap,
  openSettingsModal,
  openSyncModal,
  reimportCurrentMatch,
  replayNextRound,
  replayPrevRound,
  replaySetSpeed,
  replayToggleFullscreen,
  replayTogglePlay,
  roundsToggleFullscreen,
  saveAISettings,
  saveSyncFolder,
  scrollToReplaySection,
  sendChat,
  startBulkUpload,
  switchSettingsTab,
  syncProcess,
  syncScan,
  syncSelectAll,
  toggleDebugPositions,
  toggleSidebar,
});


/* What the shared panels need back from this page. See js/hooks.js. */
Object.assign(hooks, { switchSettingsTab, populateBulkAccountSelector });
