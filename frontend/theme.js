/* pr1maly — Themes
   Persists to localStorage, respects system preference on first visit.

   Each theme declares whether it sits on a light background. Nothing infers it
   from the name: theme.css keys its structural rules — ghost borders, ambient
   shadows, white-overlay substitutes — off data-scheme, so a new light theme
   inherits all of that by saying so here rather than by being called "light". */
var THEMES = [
  { id: 'midnight', label: 'Midnight', light: false, swatch: '#cc97ff' },
  { id: 'gray',     label: 'Gray',     light: true,  swatch: '#6d4ea3' },
  { id: 'intense',  label: 'Intense',  light: false, swatch: '#00e5ff' },
  { id: 'pro',      label: 'Pro',      light: true,  swatch: '#ff2e63' },
];
var DEFAULT_THEME = 'midnight';

/* The toggle only ever wrote "dark" or "light". Map those onto the themes they
   became so an existing preference survives the upgrade. */
var _LEGACY = { dark: 'midnight', light: 'gray' };

function _themeById(id) {
  for (var i = 0; i < THEMES.length; i++) {
    if (THEMES[i].id === id) return THEMES[i];
  }
  return null;
}

function currentTheme() {
  return document.documentElement.getAttribute('data-theme') || DEFAULT_THEME;
}

function _applyTheme(id) {
  var theme = _themeById(id) || _themeById(DEFAULT_THEME);
  document.documentElement.setAttribute('data-theme', theme.id);
  document.documentElement.setAttribute('data-scheme', theme.light ? 'light' : 'dark');
  return theme;
}

(function() {
  var saved = localStorage.getItem('pr1maly-theme');
  if (saved && _LEGACY[saved]) saved = _LEGACY[saved];
  if (saved && _themeById(saved)) {
    _applyTheme(saved);
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
    _applyTheme('gray');
  } else {
    _applyTheme(DEFAULT_THEME);
  }
})();

/* Changing the theme reloads the page, which is heavier than it looks like it
   should be and is nonetheless the honest fix.

   Everything driven by CSS re-colours the instant the variables change. The
   charts do not: they are painted onto canvases with colours read out of those
   variables at draw time, so a canvas keeps whatever pixels it was given until
   something redraws it. Switching theme without a reload left every strip
   chart, scatter, radar, minimap and economy timeline sitting in the previous
   theme's palette, on a page that had otherwise changed around them.

   The alternative is a redraw hook on every page that owns a canvas, which is
   a contract five pages have to remember to honour. A reload cannot forget.
   The theme is applied before reloading so the new colours are already in
   place while the page comes back, rather than flashing the old ones. */
function setTheme(id, opts) {
  var theme = _applyTheme(id);
  localStorage.setItem('pr1maly-theme', theme.id);
  updateThemeIcon();
  _refreshTC();
  document.dispatchEvent(new CustomEvent('themechange', { detail: theme }));
  if (!(opts && opts.noReload)) {
    if ('scrollRestoration' in history) history.scrollRestoration = 'auto';
    location.reload();
  }
}

/* Kept so the existing button keeps working: it now steps through the list
   rather than flipping between two. */
function toggleTheme() {
  var ids = THEMES.map(function(t) { return t.id; });
  var next = ids[(ids.indexOf(currentTheme()) + 1) % ids.length];
  setTheme(next);
}

function updateThemeIcon() {
  var theme = _themeById(currentTheme()) || _themeById(DEFAULT_THEME);
  document.querySelectorAll('.theme-toggle-icon').forEach(function(el) {
    el.textContent = theme.light ? 'dark_mode' : 'light_mode';
  });
  document.querySelectorAll('.theme-name').forEach(function(el) {
    el.textContent = theme.label;
  });
  document.querySelectorAll('.theme-option').forEach(function(el) {
    var on = el.dataset.theme === theme.id;
    el.setAttribute('aria-current', on ? 'true' : 'false');
    var tick = el.querySelector('.theme-option-tick');
    if (tick) tick.style.visibility = on ? 'visible' : 'hidden';
  });
}

/* --- TC: Theme Colors for canvas / Chart.js ---
   Read CSS custom properties so JS drawing code gets theme-aware values. */
function _refreshTC() {
  var s = getComputedStyle(document.documentElement);
  var g = function(n) { return s.getPropertyValue('--' + n).trim(); };
  window.TC = {
    bg:        g('chart-bg'),
    canvasBg:  g('chart-canvas-bg'),
    ct:        g('chart-ct'),
    ctDot:     g('chart-ct-dot'),
    ctName:    g('chart-ct-name'),
    t:         g('chart-t'),
    tDot:      g('chart-t-dot'),
    tName:     g('chart-t-name'),
    hpGood:    g('chart-hp-good'),
    hpWarn:    g('chart-hp-warn'),
    kill:      g('chart-kill'),
    death:     g('chart-death'),
    success:   g('chart-success'),
    fail:      g('chart-fail'),
    error:     g('chart-error'),
    amber:     g('chart-amber'),
    orange:    g('chart-orange'),
    molotov:   g('chart-molotov'),
    cyan:      g('chart-cyan'),
    sky:       g('chart-sky'),
    purple:    g('chart-purple'),
    purpleAlt: g('chart-purple-alt'),
    pink:      g('chart-pink'),
    smoke:     g('chart-smoke'),
    onText:    g('chart-on-text'),
    dotStroke: g('chart-dot-stroke'),
    buyFull:   g('chart-buy-full'),
    buyHalf:   g('chart-buy-half'),
    buyForce:  g('chart-buy-force'),
    buyPistol: g('chart-buy-pistol'),
    flash:     g('chart-flash'),
    he:        g('chart-he'),
    preAim:    g('chart-pre-aim'),
    grid:      g('chart-grid'),
    gridText:  g('chart-grid-text'),
    track:     g('chart-track'),
    avgStroke: g('chart-avg-stroke'),
    /* Read from the scheme, not the theme name — there is more than one light
       theme now, and drawing code that asks this is asking about the
       background it is painting on, not which palette is selected. */
    isLight:   document.documentElement.getAttribute('data-scheme') === 'light'
  };
}
window.TC = {};

/* --- Theme menu ----------------------------------------------------------
   Built here rather than in each page's markup: five pages carry the theme
   button, and a list of themes that has to be edited in five places is a list
   that will disagree with itself by the third one. */
var _themeMenu = null;

function _buildThemeMenu() {
  var menu = document.createElement('div');
  menu.id = 'theme-menu';
  menu.setAttribute('role', 'menu');
  menu.style.cssText =
    'position:fixed;z-index:300;display:none;min-width:150px;padding:6px;' +
    'border-radius:12px;font-family:inherit;' +
    'background:rgb(var(--c-surface-container-high));' +
    'border:1px solid rgb(var(--c-outline-variant));' +
    'box-shadow:0 8px 32px rgba(0,0,0,0.35);';

  THEMES.forEach(function(t) {
    var item = document.createElement('button');
    item.className = 'theme-option';
    item.dataset.theme = t.id;
    item.setAttribute('role', 'menuitem');
    item.style.cssText =
      'display:flex;align-items:center;gap:10px;width:100%;padding:7px 10px;' +
      'border:0;border-radius:8px;background:transparent;cursor:pointer;' +
      'font-size:11px;font-weight:700;letter-spacing:0.08em;' +
      'text-transform:uppercase;color:rgb(var(--c-on-surface));';
    item.onmouseenter = function() { item.style.background = 'rgb(var(--c-surface-container-highest))'; };
    item.onmouseleave = function() { item.style.background = 'transparent'; };
    item.onclick = function() { setTheme(t.id); closeThemeMenu(); };

    var dot = document.createElement('span');
    dot.style.cssText =
      'width:12px;height:12px;border-radius:50%;flex:none;background:' + t.swatch +
      ';box-shadow:0 0 0 1px rgb(var(--c-outline-variant));';
    var label = document.createElement('span');
    label.textContent = t.label;
    var tick = document.createElement('span');
    tick.className = 'material-symbols-outlined theme-option-tick';
    tick.textContent = 'check';
    tick.style.cssText = 'margin-left:auto;font-size:14px;color:rgb(var(--c-primary));';

    item.appendChild(dot);
    item.appendChild(label);
    item.appendChild(tick);
    menu.appendChild(item);
  });

  document.body.appendChild(menu);
  return menu;
}

function closeThemeMenu() {
  if (_themeMenu) _themeMenu.style.display = 'none';
}

/* Called from markup as an action, so the event comes first and the button
   the user clicked comes after it. */
function toggleThemeMenu(event, btn) {
  if (!_themeMenu) _themeMenu = _buildThemeMenu();
  if (_themeMenu.style.display === 'block') { closeThemeMenu(); return; }

  var r = (btn || document.querySelector('[data-theme-button]')).getBoundingClientRect();
  _themeMenu.style.display = 'block';
  // Measure after showing, then keep it on screen — the sidebar collapses and
  // the button sits near the bottom on short viewports.
  var h = _themeMenu.offsetHeight;
  var w = _themeMenu.offsetWidth;
  _themeMenu.style.left = Math.min(r.left, window.innerWidth - w - 8) + 'px';
  _themeMenu.style.top = Math.max(8, Math.min(r.top, window.innerHeight - h - 8)) + 'px';
  updateThemeIcon();
}

document.addEventListener('click', function(e) {
  if (!_themeMenu || _themeMenu.style.display !== 'block') return;
  if (_themeMenu.contains(e.target) || e.target.closest('[data-theme-button]')) return;
  closeThemeMenu();
});
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeThemeMenu();
});

/* Update icon + chart colors on DOMContentLoaded */
document.addEventListener('DOMContentLoaded', function() {
  updateThemeIcon();
  _refreshTC();
});

/* ─── Map icons ───────────────────────────────────────────────────────────
   Valve's map badges, in frontend/img/maps/, named exactly as the map is in
   the database (de_mirage.png). Shared here rather than copied into each page
   so there is one naming rule and one fallback.

   Coverage is not guaranteed: a Premier pool rotates and a demo can be from
   any map, so every caller gets a text abbreviation underneath that shows
   through when the icon 404s. */

var MAP_ABBREV = {
  de_dust2: 'D2', de_mirage: 'MIR', de_inferno: 'INF', de_ancient: 'ANC',
  de_anubis: 'ANB', de_nuke: 'NUK', de_vertigo: 'VRT', de_overpass: 'OVP',
  de_cache: 'CCH', de_train: 'TRN', cs_office: 'OFF', de_cbble: 'CBL',
  de_dust: 'DST', cs_italy: 'ITA'
};

/* Normalise whatever a caller has: "de_mirage", "mirage", "Mirage".

   The map name reaches us from a demo header, so it is attacker-controlled by
   whoever produced the file — and people trade demos. Everything here ends up
   in innerHTML or an img src, so the key is reduced to the characters a real
   map name uses and anything else is rejected outright. */
function mapKey(mapName) {
  var n = String(mapName || '').trim().toLowerCase();
  if (!n || !/^[a-z0-9_]+$/.test(n)) return '';
  if (/^(de|cs|ar|gg|dm)_/.test(n)) return n;
  // Bare names come from the older UI paths that strip the prefix.
  return (n === 'office' || n === 'italy' || n === 'agency' ? 'cs_' : 'de_') + n;
}

/* Text shown to the user, from the same untrusted source. Stripped rather than
   HTML-escaped: these results go into innerHTML in some places and into an alt
   attribute in others, and escaping would render entities literally in the
   second. Stripping leaves a plain string that is safe in both. A workshop map
   with unusual punctuation loses the punctuation and still reads. */
function mapText(mapName) {
  return String(mapName == null ? '' : mapName).trim().replace(/[^\w \-]/g, '');
}

function mapIconUrl(mapName) {
  var key = mapKey(mapName);
  return key ? 'img/maps/' + key + '.png' : '';
}

function mapAbbrev(mapName) {
  var key = mapKey(mapName);
  if (MAP_ABBREV[key]) return MAP_ABBREV[key];
  return mapText(mapName).replace(/^(de|cs|ar|gg|dm)_/i, '').substring(0, 3).toUpperCase();
}

function mapLabel(mapName) {
  var bare = mapText(mapName).replace(/^(de|cs|ar|gg|dm)_/i, '');
  if (!bare) return '';
  if (bare.toLowerCase() === 'dust2') return 'Dust II';
  return bare.charAt(0).toUpperCase() + bare.slice(1);
}

/* An <img> over the abbreviation: the text is the placeholder, and the icon
   replaces it once it loads. onerror removes the image so a missing file falls
   back to text rather than a broken-image glyph.

   The badges have transparent corners, so the text is hidden on load rather
   than merely covered — otherwise it shows through around the artwork. */
function mapIconHtml(mapName, sizeClass, extraClass) {
  var url = mapIconUrl(mapName);
  var abbr = mapAbbrev(mapName);
  var box = sizeClass || 'w-10 h-10';
  return '<div class="' + box + ' relative rounded bg-surface-container-highest overflow-hidden shrink-0 ' +
    (extraClass || '') + '">' +
    '<span class="absolute inset-0 flex items-center justify-center font-bold text-[10px] text-on-surface-variant">' +
    abbr + '</span>' +
    (url
      ? '<img src="' + url + '" alt="" loading="lazy" class="relative w-full h-full object-contain" ' +
        'onload="this.previousElementSibling.classList.add(\'hidden\')" ' +
        'onerror="this.remove()"/>'
      : '') +
    '</div>';
}
