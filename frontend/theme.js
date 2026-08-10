/* pr1maly — Theme toggle
   Persists to localStorage, respects system preference on first visit. */
(function() {
  var saved = localStorage.getItem('pr1maly-theme');
  if (saved) {
    document.documentElement.setAttribute('data-theme', saved);
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
    document.documentElement.setAttribute('data-theme', 'light');
  }
})();

function toggleTheme() {
  var current = document.documentElement.getAttribute('data-theme');
  var next = current === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('pr1maly-theme', next);
  updateThemeIcon();
  _refreshTC();
}

function updateThemeIcon() {
  var isLight = document.documentElement.getAttribute('data-theme') === 'light';
  var icons = document.querySelectorAll('.theme-toggle-icon');
  icons.forEach(function(el) { el.textContent = isLight ? 'dark_mode' : 'light_mode'; });
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
    isLight:   document.documentElement.getAttribute('data-theme') === 'light'
  };
}
window.TC = {};

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
