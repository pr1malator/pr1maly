import './theme-actions.js';
import { loadUpdateStatus } from './updates.js';
import { registerActions, actionArgs } from './actions.js';
import { API } from './api.js';
import { esc } from './escape.js';
import { hooks } from './hooks.js';
import { closeSettingsModal, closeUploadModal, loadAccounts, loadFriends } from './accounts.js';
import { loadReanalyzeList, renderStaleBanner } from '../reanalyze.js';

// --- Utility ---

// --- Account Management ---

// --- Settings Modal ---
let _aiConfig = null;
let _aiProviders = null;
let _selectedProvider = null;

/* Opening a match from the list. Was window.location.href assembled inside an
   onclick attribute, on a row whose id came from the database. */
function openMatch(matchId) {
  window.location.href = 'match-breakdown.html?id=' + encodeURIComponent(matchId);
}

function openSettingsModal(tab) {
  document.getElementById('settings-modal').classList.remove('hidden');
  document.getElementById('settings-modal').classList.add('flex');
  switchSettingsTab(tab || 'accounts');
  loadAccounts();
  loadAIConfig();
}
document.getElementById('settings-modal').addEventListener('click', e => { if (e.target === e.currentTarget) closeSettingsModal(); });

function switchSettingsTab(tab) {
  const tabs = ['accounts', 'friends', 'ai', 'reanalyze', 'updates', 'reset'];
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
}




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
  } catch (err) { console.error('Failed to load AI config:', err); }
}

function renderAISettings() {
  if (!_aiProviders || !_aiConfig) return;
  const tabsEl = document.getElementById('ai-provider-tabs');
  tabsEl.innerHTML = Object.keys(_aiProviders).map(p => {
    const isActive = p === _selectedProvider;
    return `<button data-action="selectAIProvider" data-args="${actionArgs(p)}" class="px-3 py-1.5 rounded-full text-[9px] font-bold uppercase tracking-widest transition-all ${isActive ? 'bg-primary text-on-primary-fixed' : 'bg-surface-container-highest text-on-surface-variant hover:bg-white/10'}">${p}</button>`;
  }).join('');
  const prov = _aiConfig.providers?.[_selectedProvider] || {};
  document.getElementById('ai-key-input').value = prov.api_key || '';
  document.getElementById('ai-key-status').textContent = prov.api_key ? 'Key configured' : 'No key set';
  const modelSel = document.getElementById('ai-settings-model');
  const models = _aiProviders[_selectedProvider] || [];
  modelSel.innerHTML = models.map(m => `<option value="${m}" ${m === _aiConfig.active_model ? 'selected' : ''}>${m}</option>`).join('');
  document.getElementById('ai-system-instructions').value = _aiConfig.system_instructions || '';
}

function selectAIProvider(provider) {
  _selectedProvider = provider;
  renderAISettings();
}

async function saveAISettings() {
  const statusEl = document.getElementById('ai-settings-status');
  try {
    const cfg = { ..._aiConfig };
    cfg.active_provider = _selectedProvider;
    cfg.active_model = document.getElementById('ai-settings-model').value;
    cfg.system_instructions = document.getElementById('ai-system-instructions').value;
    if (!cfg.providers) cfg.providers = {};
    if (!cfg.providers[_selectedProvider]) cfg.providers[_selectedProvider] = {};
    cfg.providers[_selectedProvider].api_key = document.getElementById('ai-key-input').value;
    const res = await fetch(API + '/ai/config', {
      method: 'PUT', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(cfg)
    });
    if (!res.ok) throw new Error('Failed to save');
    _aiConfig = await res.json();
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
    // Auto-fill match date if found and not already set
    if (data.match_date && !dateInput.value) dateInput.value = data.match_date;
    if (data.matched && data.matched.length > 0) {
      // Auto-select the first matched player
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
    status.classList.remove('text-on-surface-variant'); status.classList.add('text-secondary'); status.textContent = 'Demo processed! Refreshing...';
    setTimeout(() => { closeUploadModal(); loadDashboard(); }, 800);
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
      const partial = ok && (r.partial_import || (r.stats && r.stats.partial_import));
      const partialLabel = partial ? ' — PARTIAL IMPORT' : '';
      const detail = ok
        ? (r.map_name || '') + (r.player_name ? ' — ' + r.player_name : '') + partialLabel
        : (r.error || 'Failed');
      row.innerHTML = '<span class="material-symbols-outlined text-[16px]">' + icon + '</span><span class="font-medium truncate flex-1">' + r.filename + '</span><span class="text-on-surface-variant/70 text-[10px]">' + detail + '</span>';
      resultsDiv.appendChild(row);
    });
    if (data.processed > 0) setTimeout(() => loadDashboard(), 500);
  } catch(err) {
    progressText.textContent = err.message;
    progressText.className = 'text-[10px] font-bold uppercase tracking-widest text-error';
  } finally {
    btn.disabled = false; btn.textContent = 'Process All Demos';
  }
}

// --- Data Loading ---
async function loadDashboard() {
  try {
    const [trendsRes, matchesRes] = await Promise.all([
      fetch(API + '/trends'), fetch(API + '/matches')
    ]);
    const trends = await trendsRes.json();
    const matches = await matchesRes.json();
    renderKPIs(trends, matches);
    renderChart(trends);
    renderRecentMatches(matches);
    renderStaleBanner('stale-banner');
  } catch (err) { console.error('Failed to load dashboard:', err); }
}

function renderKPIs(trends, matches) {
  const avg = trends.averages || {};
  const pts = trends.data_points || [];
  const n = pts.length;
  // K/D
  const totalK = pts.reduce((s,d) => s + (d.kills||0), 0);
  const totalD = pts.reduce((s,d) => s + (d.deaths||0), 0);
  const kd = totalD ? (totalK / totalD).toFixed(2) : '0.00';
  // Win rate
  const wins = pts.filter(d => d.match_result === 'Victory').length;
  const winRate = n ? ((wins / n) * 100).toFixed(1) : '0.0';
  // KPIs
  const kpiCards = document.querySelectorAll('.grid.grid-cols-2.md\\:grid-cols-5 > div');
  if (kpiCards[0]) { kpiCards[0].querySelector('.text-3xl').textContent = kd; }
  if (kpiCards[1]) { kpiCards[1].querySelector('.text-3xl').textContent = (avg.avg_kast || 0).toFixed(1) + '%'; }
  if (kpiCards[2]) { kpiCards[2].querySelector('.text-3xl').textContent = (avg.avg_rating || 0).toFixed(2); }
  if (kpiCards[3]) {
    kpiCards[3].querySelector('.text-3xl').textContent = winRate + '%';
    const bar = kpiCards[3].querySelector('.bg-secondary');
    if (bar) bar.style.width = winRate + '%';
  }
  if (kpiCards[4]) { kpiCards[4].querySelector('.text-3xl').textContent = (avg.avg_adr || 0).toFixed(0); }
  // Update labels for aim → ADR
  const aimLabel = kpiCards[4]?.querySelector('.text-\\[10px\\].font-bold.text-on-surface-variant');
  if (aimLabel) aimLabel.textContent = 'ADR';
  const aimSubtext = kpiCards[4]?.querySelector('.text-\\[10px\\].text-primary-dim');
  if (aimSubtext) aimSubtext.innerHTML = '<span class="material-symbols-outlined text-xs">show_chart</span> Avg Damage/Round';
}

function renderChart(trends) {
  const pts = trends.data_points || [];
  const container = document.querySelector('.h-64.flex.items-end');
  if (!container || !pts.length) return;
  const labels = document.querySelector('.mt-4.flex.justify-between');
  // Find max for scaling
  const values = pts.map(d => d.hltv_rating || 0);
  const maxVal = Math.max(...values, 1.5);
  // Build bars
  let barsHtml = '';
  for (const v of values) {
    const pct = Math.round((v / maxVal) * 100);
    const isHigh = v >= 1.3;
    const barColor = isHigh
      ? 'bg-secondary/30 border-t-2 border-secondary hover:bg-secondary/50'
      : 'bg-surface-container-highest hover:bg-primary/40';
    barsHtml += `<div class="flex-1 ${barColor} rounded-t-sm transition-all relative group" style="height:${pct}%">
      <div class="absolute -top-8 left-1/2 -translate-x-1/2 ${isHigh ? 'bg-secondary text-on-secondary' : 'bg-primary text-on-primary'} text-[10px] font-bold px-2 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity">${v.toFixed(2)}</div>
    </div>`;
  }
  // Keep grid lines
  const gridLines = container.querySelector('.absolute.inset-0');
  container.innerHTML = (gridLines ? gridLines.outerHTML : '') + barsHtml;
  // Labels
  if (labels && pts.length > 1) {
    const step = Math.max(1, Math.floor(pts.length / 5));
    let labelHtml = '';
    for (let i = 0; i < pts.length; i += step) {
      const mapName = (pts[i].map_name || '').replace('de_','').toUpperCase();
      labelHtml += `<span>${mapName || 'Match ' + (i+1)}</span>`;
    }
    if (pts.length - 1 > (Math.floor((pts.length-1)/step) * step)) {
      const last = pts[pts.length-1];
      labelHtml += `<span>Latest</span>`;
    }
    labels.innerHTML = labelHtml;
  }
}

function renderRecentMatches(matches) {
  const container = document.querySelector('.space-y-2');
  if (!container || !matches.length) return;
  let html = '';
  const recent = matches.slice(0, 10);
  for (const m of recent) {
    const isWin = m.match_result === 'Victory';
    const borderColor = isWin ? 'border-secondary' : 'border-error';
    const scoreColor = isWin ? 'text-secondary' : 'text-error';
    const statusColor = isWin ? 'text-secondary' : 'text-error';
    const score = `${m.team_score || 0} - ${m.enemy_score || 0}`;
    const kda = `${m.kills || 0} / ${m.deaths || 0} / ${m.assists || 0}`;
    const rating = (m.hltv_rating || 0).toFixed(2);
    const ratingColor = (m.hltv_rating || 0) >= 1.0 ? 'text-secondary' : 'text-on-surface';
    const result = m.match_result === 'Victory' ? 'VICTORY' : m.match_result === 'Defeat' ? 'DEFEAT' : (m.match_result || '').toUpperCase();
    const isPartial = !!m.partial_import;
    const partialBadge = isPartial
      ? '<div class="mt-1 inline-flex items-center gap-1 rounded-full bg-caution/15 text-caution px-2 py-0.5 text-[9px] font-bold uppercase tracking-widest"><span class="material-symbols-outlined text-[10px]">warning</span>Partial Import</div>'
      : '';
    // Say so when a row's numbers came from an older analyzer, otherwise two
    // matches side by side can be measured differently with nothing to show it.
    const staleBadge = m.analysis_stale
      ? '<div class="mt-1 inline-flex items-center gap-1 rounded-full bg-white/5 text-on-surface-variant px-2 py-0.5 text-[9px] font-bold uppercase tracking-widest" title="Analysed by an older version — re-analyze to update"><span class="material-symbols-outlined text-[10px]">update</span>v' + (m.analyzer_version || 0) + '</div>'
      : '';
    html += `<div class="grid grid-cols-4 md:grid-cols-12 items-center bg-surface-container hover:bg-surface-container-high transition-all p-4 rounded-xl border-l-4 ${borderColor} group cursor-pointer" data-action="openMatch" data-args="${actionArgs(m.match_id)}">
      <div class="md:col-span-2 flex items-center gap-3">
        ${mapIconHtml(m.map_name)}
        <div>
          <div class="font-bold text-xs uppercase tracking-widest">${mapLabel(m.map_name)}</div>
          ${partialBadge}${staleBadge}
        </div>
      </div>
      <div class="md:col-span-2 font-headline text-xl font-bold ${scoreColor}">${score}</div>
      <div class="hidden md:block md:col-span-2">
        <div class="text-[10px] font-bold text-on-surface-variant uppercase mb-1">K / D / A</div>
        <div class="text-sm font-bold text-on-surface">${kda}</div>
      </div>
      <div class="hidden md:block md:col-span-2">
        <div class="text-[10px] font-bold text-on-surface-variant uppercase mb-1">RATING</div>
        <div class="text-sm font-bold ${ratingColor}">${rating}</div>
      </div>
      <div class="md:col-span-2 text-right md:text-left">
        <div class="text-[10px] font-bold text-on-surface-variant uppercase mb-1">STATUS</div>
        <div class="text-xs font-bold ${statusColor} uppercase tracking-widest">${result}</div>
      </div>
      <div class="hidden md:flex md:col-span-2 justify-end">
        <span class="material-symbols-outlined text-on-surface-variant group-hover:text-primary transition-colors">analytics</span>
      </div>
    </div>`;
  }
  container.innerHTML = html;
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

// --- Init ---

document.addEventListener('DOMContentLoaded', () => { loadAccounts(); loadDashboard(); });


/* What this file offers the markup. See js/actions.js. */
registerActions({
  openMatch,
  openSettingsModal,
  saveAISettings,
  selectAIProvider,
  startBulkUpload,
  switchSettingsTab,
  toggleSidebar,
});


/* What the shared panels need back from this page. See js/hooks.js. */
Object.assign(hooks, { switchSettingsTab, populateBulkAccountSelector });
