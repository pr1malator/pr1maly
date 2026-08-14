/* Degraded map-icon helpers, in case theme.js is older than the page.
 *
 * Was duplicated verbatim in breakdown.html, match-breakdown.html and
 * performance.html. */

/* Map-icon helpers come from theme.js. This page is served no-store but a
   browser can still be holding an older theme.js than the HTML it is running
   — that is exactly what happened when the helpers were first added, and it
   blanked the map grid and the match list with a ReferenceError.

   Whole sections should not hinge on one shared file being in step, so the
   degraded forms are defined here when theme.js did not supply them. They
   render the plain text tile the UI used before icons existed. */
if (typeof mapIconUrl !== 'function') {
  window.mapIconUrl = function () { return ''; };
}
if (typeof mapAbbrev !== 'function') {
  window.mapAbbrev = function (n) {
    return String(n || '').replace(/^(de|cs|ar|gg|dm)_/, '').substring(0, 3).toUpperCase();
  };
}
if (typeof mapLabel !== 'function') {
  window.mapLabel = function (n) {
    var b = String(n || '').replace(/^(de|cs|ar|gg|dm)_/, '');
    return b ? b.charAt(0).toUpperCase() + b.slice(1) : '';
  };
}
if (typeof mapIconHtml !== 'function') {
  window.mapIconHtml = function (n, sizeClass, extraClass) {
    return '<div class="' + (sizeClass || 'w-10 h-10') + ' rounded bg-surface-container-highest ' +
      'flex items-center justify-center font-bold text-[10px] text-on-surface-variant shrink-0 ' +
      (extraClass || '') + '">' + mapAbbrev(n) + '</div>';
  };
}
