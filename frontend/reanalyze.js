/* Re-analyze panel — shared by every page that hosts the settings modal.
 *
 * Each page carries three separate copies of that modal, so this file owns
 * both the markup and the behaviour: a page opts in with a tab button, an
 * empty <div id="settings-reanalyze-panel">, and 'reanalyze' in its tab list.
 *
 * Re-parsing is minutes of work per demo, so the run is driven one request at
 * a time from the browser rather than as a single server-side batch: that
 * keeps the progress bar honest and lets one bad demo fail without taking the
 * rest of the queue with it.
 */

// How many matches "Newest N outdated" picks. Deliberately small: while the
// metrics are being tuned the point is a readable sample, not a full re-parse
// of the library on every iteration.
const REANALYZE_RECENT_N = 5;

const REANALYZE_PANEL_HTML = `
  <h3 class="font-headline text-[11px] font-bold uppercase tracking-widest text-on-surface-variant mb-1 flex items-center gap-2">
    <span class="material-symbols-outlined text-sm">restart_alt</span> Re-analyze Matches
  </h3>
  <p class="text-[9px] text-on-surface-variant mb-3 leading-relaxed">
    Every match records the analyzer version that produced it. When the metrics change, older
    matches keep their old numbers until they are re-analyzed against the demo on disk. Uploaded
    demos are not kept, so those matches show as <span class="text-on-surface-variant/70">no demo</span>
    and can only be refreshed by importing the file again.
  </p>

  <div id="reanalyze-summary" class="grid grid-cols-3 gap-2 mb-3"></div>

  <div class="flex flex-wrap items-center gap-2 mb-1">
    <button onclick="selectReanalyzable('recent')" class="px-3 py-1.5 rounded-lg bg-primary/20 text-primary text-[10px] font-bold uppercase tracking-widest hover:bg-primary/30 transition-colors">Newest 5 outdated</button>
    <button onclick="selectReanalyzable('stale')" class="px-3 py-1.5 rounded-lg bg-surface-container-highest text-on-surface-variant text-[10px] font-bold uppercase tracking-widest hover:bg-white/10 transition-colors">All outdated</button>
    <button onclick="selectReanalyzable('all')" class="px-3 py-1.5 rounded-lg bg-surface-container-highest text-on-surface-variant text-[10px] font-bold uppercase tracking-widest hover:bg-white/10 transition-colors">Everything</button>
    <button onclick="selectReanalyzable('none')" class="px-3 py-1.5 rounded-lg bg-surface-container-highest text-on-surface-variant text-[10px] font-bold uppercase tracking-widest hover:bg-white/10 transition-colors">Clear</button>
  </div>
  <p class="text-[9px] text-on-surface-variant mb-3 leading-relaxed">
    A handful of recent matches is usually enough to see whether a metric change did what it
    should &mdash; re-analyzing the whole library every time is a lot of parsing for a sample you
    may invalidate again.
  </p>

  <div id="reanalyze-list" class="space-y-1 mb-3 max-h-80 overflow-y-auto pr-1">
    <p class="text-[10px] text-on-surface-variant">Loading matches...</p>
  </div>

  <div id="reanalyze-progress" class="hidden mb-3">
    <div class="flex justify-between items-center mb-1">
      <span id="reanalyze-progress-text" class="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Re-analyzing...</span>
      <span id="reanalyze-progress-count" class="text-[10px] font-bold text-primary">0 / 0</span>
    </div>
    <div class="h-1.5 w-full bg-surface-container-highest rounded-full overflow-hidden">
      <div id="reanalyze-progress-bar" class="h-full bg-primary rounded-full transition-all duration-300" style="width: 0%"></div>
    </div>
    <p class="text-[9px] text-on-surface-variant mt-1 leading-relaxed">
      Roughly 5&ndash;10 seconds per demo. Keep this tab open until it finishes.
    </p>
  </div>

  <button id="reanalyze-btn" onclick="runReanalyze()" disabled class="w-full bg-primary/20 text-primary py-2.5 rounded-full font-headline text-[10px] font-bold uppercase tracking-widest hover:bg-primary/30 transition-colors disabled:opacity-40 disabled:pointer-events-none">
    Select matches to re-analyze
  </button>
  <div id="reanalyze-result" class="text-[9px] text-on-surface-variant mt-2 leading-relaxed space-y-0.5"></div>
`;

let _reanalyzeMatches = [];

function _reanalyzeApi() {
  return (typeof API !== 'undefined' && API) ? API : (window.location.origin + '/api');
}

async function loadReanalyzeList() {
  const panel = document.getElementById('settings-reanalyze-panel');
  if (!panel) return;
  if (!panel.dataset.built) {
    panel.innerHTML = REANALYZE_PANEL_HTML;
    panel.dataset.built = '1';
  }

  const list = document.getElementById('reanalyze-list');
  const summary = document.getElementById('reanalyze-summary');
  const api = _reanalyzeApi();
  try {
    const [mRes, vRes] = await Promise.all([
      fetch(api + '/matches'),
      fetch(api + '/analyzer/version'),
    ]);
    _reanalyzeMatches = await mRes.json();
    const version = await vRes.json();

    const stale = _reanalyzeMatches.filter(m => m.analysis_stale).length;
    const missing = _reanalyzeMatches.filter(m => !m.demo_available).length;
    summary.innerHTML =
      _reanalyzeStat('Analyzer', 'v' + version.analyzer_version, 'text-primary') +
      _reanalyzeStat('Outdated', stale, stale ? 'text-amber-300' : 'text-secondary') +
      _reanalyzeStat('No demo', missing, missing ? 'text-on-surface-variant' : 'text-secondary');

    if (!_reanalyzeMatches.length) {
      list.innerHTML = '<p class="text-[10px] text-on-surface-variant">No matches stored yet.</p>';
      return;
    }
    list.innerHTML = _reanalyzeMatches.map(m => _reanalyzeRow(m)).join('');
    updateReanalyzeButton();
  } catch (err) {
    list.innerHTML = `<p class="text-[10px] text-error">Could not load matches: ${err.message}</p>`;
  }
}

function _reanalyzeStat(label, value, colorClass) {
  return `<div class="bg-surface-container-highest rounded-lg p-2">
    <div class="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant">${label}</div>
    <div class="text-sm font-bold font-headline ${colorClass}">${value}</div>
  </div>`;
}

function _reanalyzeRow(m) {
  const canDo = !!m.demo_available;
  const badge = !canDo
    ? '<span class="px-1.5 py-0.5 rounded-full text-[8px] font-bold bg-surface-container-highest text-on-surface-variant">no demo</span>'
    : m.analysis_stale
      ? `<span class="px-1.5 py-0.5 rounded-full text-[8px] font-bold bg-amber-400/20 text-amber-300">v${m.analyzer_version} outdated</span>`
      : `<span class="px-1.5 py-0.5 rounded-full text-[8px] font-bold bg-emerald-400/20 text-emerald-400">v${m.analyzer_version} current</span>`;
  const map = (m.map_name || 'unknown').replace('de_', '').toUpperCase();
  const score = `${m.team_score || 0}-${m.enemy_score || 0}`;
  return `<label class="flex items-center gap-2 p-2 rounded-lg bg-surface-container-highest/50 ${canDo ? 'cursor-pointer hover:bg-surface-container-highest' : 'opacity-50 cursor-not-allowed'} transition-colors">
    <input type="checkbox" class="reanalyze-check accent-primary" value="${m.match_id}"
           data-stale="${m.analysis_stale ? '1' : '0'}" ${canDo ? '' : 'disabled'}
           onchange="updateReanalyzeButton()"/>
    <span class="text-[10px] font-bold w-16 shrink-0">${map}</span>
    <span class="text-[10px] text-on-surface-variant w-12 shrink-0">${score}</span>
    <span class="text-[10px] text-on-surface-variant w-20 shrink-0">${m.date || ''}</span>
    <span class="ml-auto shrink-0">${badge}</span>
  </label>`;
}

function selectReanalyzable(mode) {
  // Rows render in the order /api/matches returns them, which is date DESC,
  // so "newest" is simply the first N eligible checkboxes.
  let remaining = REANALYZE_RECENT_N;
  document.querySelectorAll('.reanalyze-check').forEach(cb => {
    if (cb.disabled) return;
    if (mode === 'all') cb.checked = true;
    else if (mode === 'none') cb.checked = false;
    else if (mode === 'stale') cb.checked = cb.dataset.stale === '1';
    else if (mode === 'recent') {
      const take = cb.dataset.stale === '1' && remaining > 0;
      cb.checked = take;
      if (take) remaining--;
    }
  });
  updateReanalyzeButton();
}

function selectedReanalyzeIds() {
  return Array.from(document.querySelectorAll('.reanalyze-check:checked')).map(cb => cb.value);
}

function updateReanalyzeButton() {
  const btn = document.getElementById('reanalyze-btn');
  if (!btn) return;
  const n = selectedReanalyzeIds().length;
  btn.disabled = n === 0;
  btn.textContent = n === 0
    ? 'Select matches to re-analyze'
    : `Re-analyze ${n} match${n > 1 ? 'es' : ''}`;
}

async function runReanalyze() {
  const ids = selectedReanalyzeIds();
  if (!ids.length) return;

  const btn = document.getElementById('reanalyze-btn');
  const section = document.getElementById('reanalyze-progress');
  const text = document.getElementById('reanalyze-progress-text');
  const count = document.getElementById('reanalyze-progress-count');
  const bar = document.getElementById('reanalyze-progress-bar');
  const result = document.getElementById('reanalyze-result');
  const api = _reanalyzeApi();

  btn.disabled = true;
  section.classList.remove('hidden');
  result.innerHTML = '';
  bar.style.width = '0%';
  count.textContent = '0 / ' + ids.length;

  let done = 0, ok = 0;
  const failures = [];

  for (const id of ids) {
    text.textContent = `Re-analyzing ${done + 1} of ${ids.length}...`;
    try {
      const res = await fetch(api + '/matches/' + encodeURIComponent(id) + '/reanalyze', { method: 'POST' });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      ok++;
    } catch (err) {
      failures.push(err.message);
    }
    done++;
    count.textContent = done + ' / ' + ids.length;
    bar.style.width = (done / ids.length * 100) + '%';
  }

  text.textContent = failures.length ? `${ok} of ${ids.length} succeeded` : 'All matches re-analyzed';
  text.className = 'text-[10px] font-bold uppercase tracking-widest ' + (failures.length ? 'text-warning' : 'text-secondary');
  if (failures.length) {
    result.innerHTML = failures.map(f => `<div class="text-error">${f}</div>`).join('');
  }

  await loadReanalyzeList();
}

/* Banner for pages that list matches: says how many are outdated and opens
 * the panel. Without this the state is only visible to someone who already
 * knows to look inside Settings. */
async function renderStaleBanner(containerId) {
  const el = document.getElementById(containerId);
  if (!el) return;
  try {
    const res = await fetch(_reanalyzeApi() + '/analyzer/version');
    const v = await res.json();
    if (!v.stale_matches) { el.classList.add('hidden'); return; }
    el.classList.remove('hidden');
    el.innerHTML = `<div class="flex items-center gap-3 p-3 rounded-xl bg-amber-400/10 border border-amber-400/20">
      <span class="material-symbols-outlined text-amber-300 text-lg">update</span>
      <div class="flex-1 min-w-0">
        <div class="text-[11px] font-bold uppercase tracking-widest text-amber-300">
          ${v.stale_matches} of ${v.total_matches} matches use older metrics
        </div>
        <div class="text-[10px] text-on-surface-variant">
          Analyzer is at v${v.analyzer_version}. Re-analyze to bring them up to date.
        </div>
      </div>
      <button onclick="openSettingsModal('reanalyze')" class="shrink-0 px-3 py-1.5 rounded-lg bg-amber-400/20 text-amber-300 text-[10px] font-bold uppercase tracking-widest hover:bg-amber-400/30 transition-colors">
        Re-analyze
      </button>
    </div>`;
  } catch (err) {
    el.classList.add('hidden');
  }
}
