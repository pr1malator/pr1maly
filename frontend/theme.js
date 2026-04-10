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
