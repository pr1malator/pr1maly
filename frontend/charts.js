/* pr1maly — chart drawing, shared by the app and the landing page.
   ==========================================================================
   These functions used to live inside match-breakdown.html. The landing page
   had its own reimplementation of the same charts drawn from hardcoded
   numbers, which was accurate the day it was written and has been drifting
   ever since — the app grew peek speed, a reworked economy timeline and new
   bands, and none of it reached the copy people see first.

   One file, loaded by both, means the marketing page cannot show a chart the
   product does not draw. It also means changing a chart here changes it in
   both places, which is the point.

   Depends only on `TC` from theme.js (colours read out of the CSS variables,
   so every chart follows the active theme) and on the threshold table the
   backend ships in aim_stats.thresholds. No DOM beyond the canvas it is
   handed, and no knowledge of where the data came from.
   ========================================================================== */

const AIM_TIER_LABELS = ['Excellent', 'Strong', 'Fair', 'Needs Work'];
let AIM_KPI_META = {};

const AIM_KPI_FALLBACK = {
  movement:   { label: 'Shot Speed',          unit: 'u/s',   bounds: [15, 40, 100],   lower_better: true, range: [0, 250]  },
  // Ungraded on purpose — see the note on the backend table. The zones name
  // regions of the axis (which kind of peek), they are not tiers.
  peek:       { label: 'Peek Speed',          unit: 'u/s',   bounds: [],              lower_better: true, range: [0, 250],
                zones: [{ at: 0, label: 'Held' }, { at: 85, label: 'Walk' },
                        { at: 130, label: 'Half speed' }, { at: 180, label: 'Full speed' }] },
  stop_ticks: { label: 'Counter-strafe',      unit: 'ticks', bounds: [3, 7, 15],      lower_better: true, range: [0, 32]  },
  counterstrafe: { label: 'Counter-strafe Rate', unit: '%',  bounds: [80, 60, 35],    lower_better: false, range: [0, 100] },
  preaim:     { label: 'Crosshair Placement', unit: '°', bounds: [5, 10, 20],   lower_better: true, range: [0, 45]  },
  ttk:        { label: 'Engagement Time',     unit: 's',     bounds: [0.4, 0.65, 1.1],lower_better: true, range: [0, 1.0]  },
  reaction:   { label: 'Reaction Time',       unit: 'ms',    bounds: [150, 200, 300], lower_better: true, range: [0, 800]  },
  accuracy:   { label: 'Accuracy',            unit: '%',     bounds: [75, 50, 30],    lower_better: false, range: [0, 100] },
};

function buildAimKpiMeta(thresholds) {
  const src = thresholds && Object.keys(thresholds).length ? thresholds : AIM_KPI_FALLBACK;
  AIM_KPI_META = {};
  for (const [key, t] of Object.entries(src)) {
    AIM_KPI_META[key] = {
      label: t.label + (t.unit ? ' (' + t.unit + ')' : ''),
      unit: t.unit,
      tiers: t.bounds,
      tierLabels: AIM_TIER_LABELS,
      lowerBetter: t.lower_better,
      range: t.range || null,
      // Named regions of an ungraded axis (peek speed). Categories, not tiers.
      zones: t.zones || null,
    };
  }
}

function aimBounds(key) {
  const meta = AIM_KPI_META[key] || AIM_KPI_FALLBACK[key];
  return meta ? meta.tiers || meta.bounds : [];
}

function aimRange(key) {
  const meta = AIM_KPI_META[key] || AIM_KPI_FALLBACK[key];
  return meta && meta.range ? meta.range : null;
}

function aimZones(key) {
  const meta = AIM_KPI_META[key] || AIM_KPI_FALLBACK[key];
  return meta && meta.zones ? meta.zones : [];
}

function _zoneHue() { return TC.onSurfaceVariant || '#8899aa'; }

function _zoneAlpha(i) { return Math.round((0.04 + i * 0.05) * 100) / 100; }

function _zoneChipStyle(i) {
  const hue = _zoneHue();
  const round2 = v => Math.round(v * 100) / 100;
  return 'background:' + hexToRgba(hue, round2(0.1 + i * 0.09))
       + ';color:' + hexToRgba(hue, round2(Math.min(1, 0.7 + i * 0.1)));
}

function _peekZoneIndexAt(byZone, at) {
  const i = (byZone || []).findIndex(z => z.at === at);
  return i < 0 ? 0 : i;
}

function drawAimScatter(canvasId, encounters, xKey, yKey) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Filter to encounters that have both KPIs
  const pts = encounters.filter(e => e[xKey] !== undefined && e[yKey] !== undefined);
  if (!pts.length) {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * 2; canvas.height = h * 2;
    ctx.scale(2, 2);
    ctx.fillStyle = TC.bg || '#0f1930';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = TC.onSurfaceVariant || '#8899aa';
    ctx.font = '11px Space Grotesk, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No encounters with both KPIs available', w / 2, h / 2);
    return;
  }

  const xMeta = AIM_KPI_META[xKey] || AIM_KPI_FALLBACK[xKey];
  const yMeta = AIM_KPI_META[yKey] || AIM_KPI_FALLBACK[yKey];
  if (!xMeta || !yMeta) return;

  const xVals = pts.map(p => p[xKey]);
  const yVals = pts.map(p => p[yKey]);

  // Axis range: the span the metric declares, so a value sits in the same
  // place in every match and two scatters can be compared by shape. Scaling to
  // each match's own spread made an identical distribution draw differently
  // from game to game — the same reason the strip charts stopped doing it.
  // Metrics without a declared span still fall back to tier boundaries plus
  // data extents with padding.
  const xAllBounds = [...xVals, ...xMeta.tiers];
  const yAllBounds = [...yVals, ...yMeta.tiers];
  const xMin = xMeta.range ? xMeta.range[0] : Math.min(...xAllBounds) * 0.85;
  const xMax = xMeta.range ? xMeta.range[1] : (Math.max(...xAllBounds) * 1.15 || 1);
  const yMin = yMeta.range ? yMeta.range[0] : Math.min(...yAllBounds) * 0.85;
  const yMax = yMeta.range ? yMeta.range[1] : (Math.max(...yAllBounds) * 1.15 || 1);

  const pad = { top: 18, right: 18, bottom: 32, left: 48 };
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * 2; canvas.height = h * 2;
  ctx.scale(2, 2);

  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  // For lowerBetter KPIs: left/bottom = low = best (natural).
  // For higherBetter KPIs: flip so left/bottom = high = best (matches strip chart inversion).
  // A fixed axis can be overshot, so values are pinned to the edge rather than
  // drawn outside the plot — the same rule the strip charts follow.
  const clampV = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const toX = v => { let n = (clampV(v, xMin, xMax) - xMin) / (xMax - xMin); if (!xMeta.lowerBetter) n = 1 - n; return pad.left + n * plotW; };
  const toY = v => { let n = (clampV(v, yMin, yMax) - yMin) / (yMax - yMin); if (yMeta.lowerBetter) n = 1 - n; return pad.top + n * plotH; };

  // Background
  ctx.fillStyle = TC.bg || '#0f1930';
  ctx.fillRect(0, 0, w, h);

  // Draw tier zone rectangles (intersection of X and Y tier bands)
  const tierColors = ['#fbbf24', '#34d399', '#60a5fa', '#ef4444'];
  const xSorted = [...xMeta.tiers].sort((a,b) => a - b);
  const ySorted = [...yMeta.tiers].sort((a,b) => a - b);
  const xBounds = [xMin, ...xSorted.filter(t => t > xMin && t < xMax), xMax];
  const yBounds = [yMin, ...ySorted.filter(t => t > yMin && t < yMax), yMax];

  // Compute tier index: for lowerBetter the more thresholds exceeded the worse;
  // for higherBetter (accuracy) the more thresholds exceeded the better.
  function tierIndex(val, sorted, meta) {
    const n = sorted.filter(t => val >= t).length;
    return meta.lowerBetter ? n : sorted.length - n;
  }

  for (let xi = 0; xi < xBounds.length - 1; xi++) {
    for (let yi = 0; yi < yBounds.length - 1; yi++) {
      const xTier = tierIndex(xBounds[xi], xSorted, xMeta);
      const yTier = tierIndex(yBounds[yi], ySorted, yMeta);
      // Combined tier: worst of the two (higher index = worse)
      const combinedTier = Math.max(xTier, yTier);
      const color = tierColors[Math.min(combinedTier, tierColors.length - 1)];

      const rx1 = toX(xBounds[xi]);
      const rx2 = toX(xBounds[xi + 1]);
      const ry1 = toY(yBounds[yi + 1]); // note: Y is inverted
      const ry2 = toY(yBounds[yi]);
      ctx.fillStyle = hexToRgba(color, 0.06);
      ctx.fillRect(rx1, ry1, rx2 - rx1, ry2 - ry1);
    }
  }

  // Zone stripes for an ungraded axis (peek speed). The graded axis keeps
  // supplying the tier colour underneath, so the two read as a grid: which
  // kind of peek it was, against how well it was stopped.
  function drawZones(meta, axis) {
    const zones = meta.zones || [];
    if (!zones.length) return;
    const lo = axis === 'x' ? xMin : yMin;
    const hi = axis === 'x' ? xMax : yMax;
    const to = axis === 'x' ? toX : toY;
    for (let i = 0; i < zones.length; i++) {
      const from = Math.max(zones[i].at, lo);
      const until = Math.min(i + 1 < zones.length ? zones[i + 1].at : hi, hi);
      if (until <= from) continue;
      const a = to(from), b = to(until);
      const near = Math.min(a, b), span = Math.abs(b - a);

      ctx.fillStyle = hexToRgba(_zoneHue(), _zoneAlpha(i));
      if (axis === 'x') ctx.fillRect(near, pad.top, span, plotH);
      else ctx.fillRect(pad.left, near, plotW, span);

      // Boundary between this zone and the one below it.
      if (i > 0) {
        ctx.strokeStyle = hexToRgba(_zoneHue(), 0.3);
        ctx.setLineDash([2, 3]);
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        if (axis === 'x') { ctx.moveTo(a, pad.top); ctx.lineTo(a, pad.top + plotH); }
        else { ctx.moveTo(pad.left, a); ctx.lineTo(pad.left + plotW, a); }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Zone name, in the slot the tier labels use on a graded axis.
      if (span < 24) continue;
      ctx.fillStyle = hexToRgba(_zoneHue(), 0.75);
      ctx.font = '7px Space Grotesk, sans-serif';
      ctx.textAlign = 'center';
      const mid = near + span / 2;
      if (axis === 'x') {
        ctx.fillText(zones[i].label.toUpperCase(), mid, pad.top + plotH + 22);
      } else {
        ctx.save();
        ctx.translate(10, mid);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(zones[i].label.toUpperCase(), 0, 0);
        ctx.restore();
      }
    }
  }
  drawZones(xMeta, 'x');
  drawZones(yMeta, 'y');

  // Draw tier threshold lines
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 0.5;
  xSorted.forEach(t => {
    if (t > xMin && t < xMax) {
      const x = toX(t);
      ctx.strokeStyle = hexToRgba('#ffffff', 0.15);
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
    }
  });
  ySorted.forEach(t => {
    if (t > yMin && t < yMax) {
      const y = toY(t);
      ctx.strokeStyle = hexToRgba('#ffffff', 0.15);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
    }
  });
  ctx.setLineDash([]);

  // Tier labels on edges. An axis with no bounds is ungraded (peek speed) and
  // gets none: a single label spanning the whole axis would read as a verdict
  // on every point, which is exactly what leaving the metric ungraded means to
  // avoid. The grading still shows through from the other axis.
  ctx.font = '7px Space Grotesk, sans-serif';
  ctx.textAlign = 'center';
  const xLabelBounds = xSorted.length
    ? [xMin, ...xSorted.filter(t => t > xMin && t < xMax), xMax]
    : [];
  for (let i = 0; i < xLabelBounds.length - 1; i++) {
    const mid = toX((xLabelBounds[i] + xLabelBounds[i + 1]) / 2);
    const tier = tierIndex(xLabelBounds[i], xSorted, xMeta);
    const label = xMeta.tierLabels[Math.min(tier, xMeta.tierLabels.length - 1)];
    ctx.fillStyle = hexToRgba(tierColors[Math.min(tier, tierColors.length - 1)], 0.5);
    ctx.fillText(label, mid, pad.top + plotH + 22);
  }
  ctx.textAlign = 'right';
  const yLabelBounds = ySorted.length
    ? [yMin, ...ySorted.filter(t => t > yMin && t < yMax), yMax]
    : [];
  for (let i = 0; i < yLabelBounds.length - 1; i++) {
    const mid = toY((yLabelBounds[i] + yLabelBounds[i + 1]) / 2);
    const tier = tierIndex(yLabelBounds[i], ySorted, yMeta);
    const label = yMeta.tierLabels[Math.min(tier, yMeta.tierLabels.length - 1)];
    ctx.fillStyle = hexToRgba(tierColors[Math.min(tier, tierColors.length - 1)], 0.5);
    ctx.save();
    ctx.translate(10, mid);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }

  // Axis tick values
  ctx.fillStyle = TC.onSurfaceVariant || '#8899aa';
  ctx.font = '8px Space Grotesk, sans-serif';
  ctx.textAlign = 'center';
  const xTicks = [xMin, ...xSorted.filter(t => t > xMin && t < xMax), xMax];
  xTicks.forEach(t => {
    ctx.fillText(t % 1 === 0 ? t.toFixed(0) : t.toFixed(1), toX(t), pad.top + plotH + 12);
  });
  ctx.textAlign = 'right';
  const yTicks = [yMin, ...ySorted.filter(t => t > yMin && t < yMax), yMax];
  yTicks.forEach(t => {
    ctx.fillText(t % 1 === 0 ? t.toFixed(0) : t.toFixed(1), pad.left - 4, toY(t) + 3);
  });

  // Axis labels
  ctx.fillStyle = TC.onSurfaceVariant || '#8899aa';
  ctx.font = 'bold 9px Space Grotesk, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(xMeta.label, pad.left + plotW / 2, h - 2);
  ctx.save();
  ctx.translate(12, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yMeta.label, 0, 0);
  ctx.restore();

  // Plot border
  ctx.strokeStyle = hexToRgba('#ffffff', 0.1);
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.left, pad.top, plotW, plotH);

  // Plot points with outcome indicators
  const dotR = Math.max(3, Math.min(5, plotW / pts.length));
  for (let i = 0; i < pts.length; i++) {
    const px = toX(pts[i][xKey]);
    const py = toY(pts[i][yKey]);
    const outcome = pts[i].outcome || '';

    if (outcome === 'death') {
      const s = dotR * 1.2;
      ctx.globalAlpha = 0.85;
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = Math.max(1.5, dotR * 0.5);
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(px - s, py - s); ctx.lineTo(px + s, py + s);
      ctx.moveTo(px + s, py - s); ctx.lineTo(px - s, py + s);
      ctx.stroke();
    } else if (outcome === 'damage') {
      ctx.globalAlpha = 0.45;
      ctx.fillStyle = '#9ca3af';
      ctx.beginPath();
      ctx.arc(px, py, dotR * 0.7, 0, Math.PI * 2);
      ctx.fill();
    } else {
      // kill
      ctx.globalAlpha = 0.8;
      ctx.fillStyle = '#34d399';
      ctx.beginPath();
      ctx.arc(px, py, dotR, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
}

function drawStripChart(canvasId, values, avg, opts) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !values.length) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * 2; canvas.height = h * 2;
  ctx.scale(2, 2); // retina

  const thresholds = opts.thresholds || [];
  const colors = opts.colors || [TC.success || '#34d399'];
  // Fixed axis when the metric declares one, so the same value sits in the
  // same place every match and the shape of the distribution is comparable.
  // Falls back to scaling on the data for anything without a declared span.
  let minV, maxV;
  if (opts.range && opts.range.length === 2) {
    [minV, maxV] = opts.range;
  } else {
    const allVals = [...values, avg];
    minV = Math.min(...allVals) * 0.9;
    maxV = Math.max(...allVals) * 1.1 || 1;
  }
  const range = maxV - minV || 1;
  // Anything past the end of a fixed axis is pinned to the edge rather than
  // drawn outside the chart.
  const clampV = v => Math.max(minV, Math.min(maxV, v));

  // Background
  ctx.fillStyle = opts.bgColor || TC.bg || '#0f1930';
  ctx.fillRect(0, 0, w, h);

  // Draw threshold zones
  const invert = opts.invert || false;
  const getX = invert
    ? v => ((maxV - clampV(v)) / range) * w
    : v => ((clampV(v) - minV) / range) * w;
  // A metric that ships no bounds is ungraded (peek speed). Falling through to
  // the loop below would paint the whole strip in the top tier's colour and
  // label it EXCELLENT, which is a verdict the metric does not carry. It gets
  // its named regions instead, in the neutral ramp, or a flat wash if it
  // declares none.
  const zones = opts.zones || [];
  if (!thresholds.length) {
    if (!zones.length) {
      ctx.fillStyle = hexToRgba(_zoneHue(), 0.06);
      ctx.fillRect(0, 0, w, h);
    }
    for (let i = 0; i < zones.length; i++) {
      const from = Math.max(zones[i].at, minV);
      const until = Math.min(i + 1 < zones.length ? zones[i + 1].at : maxV, maxV);
      if (until <= from) continue;
      const a = getX(from), b = getX(until);
      const bandL = Math.min(a, b), bandW = Math.abs(b - a);
      ctx.fillStyle = hexToRgba(_zoneHue(), _zoneAlpha(i));
      ctx.fillRect(bandL, 0, bandW, h);
      if (bandW > 34) {
        ctx.fillStyle = hexToRgba(_zoneHue(), 0.7);
        ctx.font = '7px Space Grotesk, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(zones[i].label.toUpperCase(), bandL + 3, h - 3);
      }
    }
  }
  let prevX = 0;
  const bandCount = thresholds.length ? thresholds.length + 1 : 0;
  for (let i = 0; i < bandCount; i++) {
    const nextX = i < thresholds.length ? getX(thresholds[i]) : w;
    ctx.fillStyle = (colors[i] || colors[colors.length - 1]).replace(')', ',0.08)').replace('rgb', 'rgba').replace('#', '');
    // Use hex to rgba
    const c = colors[i] || colors[colors.length - 1];
    ctx.fillStyle = hexToRgba(c, 0.08);
    const bandL = Math.max(0, prevX);
    const bandW = Math.min(w, nextX) - bandL;
    ctx.fillRect(bandL, 0, bandW, h);
    // Name the band. Unlabelled shading invites being read as a category of
    // whatever the badges underneath happen to be showing.
    if (bandW > 34) {
      ctx.fillStyle = hexToRgba(c, 0.55);
      ctx.font = '7px Space Grotesk, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText((AIM_TIER_LABELS[i] || '').toUpperCase(), bandL + 3, h - 3);
    }
    prevX = nextX;
  }

  // Plot each value as a dot with encounter outcome indicator
  const dotR = Math.max(2, Math.min(4, w / values.length / 3));
  const lowPenalty = opts.lowPenalty || [];
  const weapons = opts.weapons || [];
  const outcomes = opts.outcomes || [];

  // Encounters that scored identically land on the same x and hid each other:
  // nine engagements at 100% drew as one dot, so the strip never matched the
  // stated sample size. Stack collisions vertically instead.
  const columnCount = {};
  const columnSeen = {};
  for (let i = 0; i < values.length; i++) {
    const k = Math.round(getX(values[i]));
    columnCount[k] = (columnCount[k] || 0) + 1;
  }

  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    const x = getX(v);
    const cx = Math.max(dotR, Math.min(w - dotR, x));
    const col = Math.round(x);
    const total = columnCount[col] || 1;
    const seen = (columnSeen[col] = (columnSeen[col] || 0) + 1) - 1;
    // Spread the column around the centre line, capped so dots stay in frame.
    const spread = Math.min(dotR * 2.2, (h - dotR * 2) / Math.max(1, total));
    const cy = h / 2 + (seen - (total - 1) / 2) * spread;
    const outcome = outcomes[i] || '';

    if (outcome === 'death') {
      // Red cross (×) — player died in this encounter
      const s = dotR * 1.3;
      ctx.globalAlpha = 0.85;
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = Math.max(1.5, dotR * 0.6);
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(cx - s, cy - s);
      ctx.lineTo(cx + s, cy + s);
      ctx.moveTo(cx + s, cy - s);
      ctx.lineTo(cx - s, cy + s);
      ctx.stroke();
    } else if (outcome === 'damage') {
      // Grey dot — damage only, no kill
      ctx.globalAlpha = 0.45;
      ctx.fillStyle = '#9ca3af';
      ctx.beginPath();
      ctx.arc(cx, cy, dotR * 0.75, 0, Math.PI * 2);
      ctx.fill();
    } else if (outcome === 'kill') {
      // Green circle — got the kill
      ctx.globalAlpha = 0.8;
      ctx.fillStyle = '#34d399';
      ctx.beginPath();
      ctx.arc(cx, cy, dotR, 0, Math.PI * 2);
      ctx.fill();
    } else {
      // Fallback: original rendering (threshold-colored dot / diamond)
      let color = colors[0];
      for (let t = 0; t < thresholds.length; t++) {
        if (v >= thresholds[t]) color = colors[t + 1] || color;
      }
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.7;
      if (lowPenalty[i]) {
        const d = dotR * 1.4;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.moveTo(cx, cy - d);
        ctx.lineTo(cx + d, cy);
        ctx.lineTo(cx, cy + d);
        ctx.lineTo(cx - d, cy);
        ctx.closePath();
        ctx.fill();
      } else {
        ctx.beginPath();
        ctx.arc(cx, cy, dotR, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // Average line
  ctx.globalAlpha = 1;
  const avgX = getX(avg);
  ctx.strokeStyle = opts.avgColor || TC.cyan || '#53ddfc';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(avgX, 2);
  ctx.lineTo(avgX, h - 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Average label
  ctx.font = 'bold 9px Space Grotesk, sans-serif';
  ctx.fillStyle = opts.avgColor || TC.cyan || '#53ddfc';
  ctx.textAlign = avgX > w / 2 ? 'right' : 'left';
  const labelX = avgX > w / 2 ? avgX - 4 : avgX + 4;
  ctx.fillText('AVG', labelX, 10);
}

function hexToRgba(hex, alpha) {
  hex = hex.replace('#', '');
  if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}

// canvasId is a parameter so the landing page can draw this into its own
// element. Defaulted rather than required: the app has one economy timeline
// and naming it at every call site would be noise.
function renderEconomyTimeline(rounds, canvasId) {
  const canvas = document.getElementById(canvasId || 'economy-timeline-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = 440;
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const buyColorMap = {
    'FULL BUY': TC.buyFull||'#c497ff', 'HALF BUY': TC.buyHalf||'#bb86fc', 'FORCE BUY': TC.buyForce||'#03dac6',
    'PISTOL': TC.buyPistol||'#fb923c', 'ECO': TC.gridText||'rgba(255,255,255,0.35)'
  };

  // Purchases arrive as the demo's own display names — "High Explosive
  // Grenade", "Kevlar & Helmet", "AK-47" — so the short forms are keyed on
  // those. The previous map was keyed on engine names ('hegrenade', 'kevlar')
  // and never matched, which is why every long item fell through to a 5-letter
  // truncation and rendered as "High", "Incen", "Deser".
  const ITEM_SHORT = {
    'high explosive grenade': 'HE', 'incendiary grenade': 'Molly',
    'molotov': 'Molly', 'smoke grenade': 'Smoke', 'flashbang': 'Flash',
    'decoy grenade': 'Decoy', 'kevlar & helmet': 'Armor+H',
    'kevlar vest': 'Armor', 'defuse kit': 'Kit', 'zeus x27': 'Zeus',
    'desert eagle': 'Deagle',
  };

  // Three lanes, always in the same order and at the same height, so a glance
  // down a column separates what was shot with, what was worn, and what was
  // thrown. Row counts come from the busiest real rounds: two guns, four
  // grenades, and three pieces of gear — buying a vest and then upgrading to
  // armour + helmet in the same round records both purchases alongside a kit.
  const ITEM_LANES = [
    { key: 'gun',  label: 'GUNS', rows: 2 },
    { key: 'gear', label: 'GEAR', rows: 3 },
    { key: 'util', label: 'UTIL', rows: 4 },
  ];
  const _UTIL_ITEM = /grenade|flashbang|molotov|decoy/i;
  const _GEAR_ITEM = /kevlar|helmet|defus|zeus|taser|armor/i;
  const itemLane = (name) =>
    _UTIL_ITEM.test(name) ? 'util' : _GEAR_ITEM.test(name) ? 'gear' : 'gun';

  const pad = { left: 55, right: 20, top: 25, bottom: 145 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  // Background
  ctx.fillStyle = TC.bg||'#0f1930';
  ctx.fillRect(0, 0, w, h);

  const n = rounds.length;
  if (!n) return;

  const colW = Math.floor(cw / n);

  // Find max money for Y scale
  let maxMoney = 16000;
  for (const r of rounds) {
    const eco = r.enriched?.economy || {};
    if ((eco.start_money || 0) > maxMoney) maxMoney = eco.start_money;
    if ((eco.end_money || 0) > maxMoney) maxMoney = eco.end_money;
  }
  maxMoney = Math.ceil(maxMoney / 4000) * 4000;

  const moneyToY = (val) => pad.top + ch - (val / maxMoney) * ch;

  // --- Draw grid ---
  ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let m = 0; m <= maxMoney; m += 4000) {
    const y = moneyToY(m);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + cw, y);
    ctx.stroke();
    ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.45)';
    ctx.font = '11px Space Grotesk, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('$' + (m / 1000) + 'k', pad.left - 6, y + 4);
  }

  // Axes
  ctx.strokeStyle = TC.grid||'rgba(255,255,255,0.15)';

  // --- Item lane guides ---
  // Named on the left and separated by a hairline, so the three groups read as
  // lanes running across the whole match rather than three lists per column.
  {
    let laneY = pad.top + ch + 12;
    const lineH = 10, laneGap = 5;
    for (const lane of ITEM_LANES) {
      ctx.fillStyle = TC.gridText || 'rgba(255,255,255,0.3)';
      ctx.globalAlpha = 0.5;
      ctx.font = '7px Space Grotesk, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(lane.label, pad.left - 6, laneY);
      ctx.globalAlpha = 1;

      const sepY = laneY + lane.rows * lineH + laneGap - 7;
      if (lane !== ITEM_LANES[ITEM_LANES.length - 1]) {
        ctx.strokeStyle = TC.grid || 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad.left, sepY);
        ctx.lineTo(pad.left + cw, sepY);
        ctx.stroke();
      }
      laneY += lane.rows * lineH + laneGap;
    }
  }

  // --- Draw per-round columns ---
  const endMoneyPts = [];

  for (let i = 0; i < n; i++) {
    const r = rounds[i];
    const eco = r.enriched?.economy || {};
    const buyType = eco.buy_type || 'ECO';
    const spend = eco.player_spend || 0;
    const items = eco.items || [];
    const startMoney = eco.start_money;
    const endMoney = eco.end_money;
    const side = r.enriched?.side || '?';
    const winner = r.enriched?.round_winner;
    const won = side === winner;

    const x = pad.left + i * colW;
    const barW = Math.max(4, colW - 3);

    // Column background tint
    ctx.fillStyle = won ? 'rgba(52,211,153,0.06)' : 'rgba(248,113,113,0.06)';
    ctx.fillRect(x, pad.top, barW, ch);

    // Buy type color bar at bottom of chart area
    ctx.fillStyle = buyColorMap[buyType] || TC.gridText||'rgba(255,255,255,0.2)';
    ctx.fillRect(x, pad.top + ch, barW, 4);

    // Spend bar, with an arrow marking the level it reaches on the money axis.
    // The bar carries the magnitude, the arrow reads off the same scale as the
    // money line, and the colour of both says which kind of buy it was.
    if (spend > 0) {
      const buyColor = buyColorMap[buyType] || TC.gridText || 'rgba(255,255,255,0.2)';
      const sy = moneyToY(spend);
      const cx = x + barW / 2;
      ctx.fillStyle = buyColor;
      ctx.globalAlpha = 0.35;
      ctx.fillRect(x, sy, barW, pad.top + ch - sy);
      ctx.globalAlpha = 1.0;

      // Downward arrow sitting on top of the bar.
      const aw = 5, ah = 6, gap = 2;
      ctx.beginPath();
      ctx.moveTo(cx, sy - gap);
      ctx.lineTo(cx - aw, sy - gap - ah);
      ctx.lineTo(cx + aw, sy - gap - ah);
      ctx.closePath();
      ctx.fill();

      // What it cost, above the arrow.
      ctx.font = 'bold 8px Space Grotesk, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('$' + (spend >= 1000 ? (spend / 1000).toFixed(1) + 'k' : spend),
                   cx, sy - gap - ah - 3);
    }

    // Side indicator (top bar)
    ctx.fillStyle = side === 'CT' ? 'rgba(96,165,250,0.5)' : 'rgba(251,146,60,0.5)';
    ctx.fillRect(x, pad.top, barW, 3);

    // --- Items, grouped into fixed lanes below the axis ---
    const util = r.enriched?.utility || {};
    const taken = r.enriched?.damage_taken || {};
    // Grenades bought but never thrown are the clearest waste on the chart, so
    // count throws by type rather than inferring use from damage alone — a
    // flash that blinded nobody was still thrown.
    const thrown = { flash: 0, smoke: 0, he: 0, molotov: 0 };
    for (const g of (util.grenades || [])) {
      if (thrown[g.type] !== undefined) thrown[g.type]++;
    }
    const usedThisType = {};
    // Weapons that actually did something: a kill or damage with that gun,
    // rather than the old rule of "any kill this round", which lit up a rifle
    // for a kill taken with the pistol.
    const firedWeapons = new Set();
    for (const k of (r.enriched?.kills_detail || [])) {
      if (k.weapon) firedWeapons.add(String(k.weapon).toLowerCase());
    }
    for (const d of (r.enriched?.damage_encounters || [])) {
      if (d.weapon) firedWeapons.add(String(d.weapon).toLowerCase());
    }

    // Effect states: green = it did something measurable, neutral = deployed
    // but nothing measurable came of it, red = bought and never used at all.
    // Anything the demo cannot answer stays neutral rather than being guessed.
    const EFFECT = TC.success || '#34d399';
    const NEUTRAL = TC.gridText || 'rgba(255,255,255,0.45)';
    const UNUSED = TC.fail || '#f87171';

    function itemColor(name, lane) {
      const key = name.toLowerCase();
      if (lane === 'util') {
        if (/flash/.test(key)) {
          if (!thrown.flash) return UNUSED;
          return (util.enemies_flashed || 0) > 0 ? EFFECT : NEUTRAL;
        }
        if (/high explosive/.test(key)) {
          if (!thrown.he) return UNUSED;
          return (util.he_damage || 0) > 0 ? EFFECT : NEUTRAL;
        }
        if (/molotov|incendiary/.test(key)) {
          if (!thrown.molotov) return UNUSED;
          const md = (util.molotov_damage || []).reduce((s, m) => s + (m.damage || 0), 0);
          return md > 0 ? EFFECT : NEUTRAL;
        }
        if (/smoke/.test(key)) {
          // Nothing measures whether a smoke did its job, so a thrown one is
          // only ever neutral — claiming impact for every smoke was the old
          // behaviour and it made the colour meaningless.
          return thrown.smoke ? NEUTRAL : UNUSED;
        }
        // Decoys are not among the throw types the backend tracks, so whether
        // one was thrown is simply unknown — not a reason to call it wasted.
        return NEUTRAL;
      }
      if (lane === 'gear') {
        if (/kevlar|armor|helmet/.test(key)) {
          // Armour counts as used when it actually stopped damage. Demos
          // without dmg_armor report null, and then taking any damage at all
          // is the closest honest answer.
          const armor = taken.armor;
          if (armor != null) return armor > 0 ? EFFECT : NEUTRAL;
          return (taken.health || 0) > 0 ? EFFECT : NEUTRAL;
        }
        if (/defus/.test(key)) {
          return r.enriched?.bomb?.defused ? EFFECT : NEUTRAL;
        }
        return firedWeapons.has(key) ? EFFECT : NEUTRAL;
      }
      // A gun with no kill and no damage may still have been fired, so the
      // absence of a hit is "no effect", never "never used".
      return firedWeapons.has(key) ? EFFECT : NEUTRAL;
    }

    const laneItems = { gun: [], gear: [], util: [] };
    for (const item of items) laneItems[itemLane(item)].push(item);

    const itemStartY = pad.top + ch + 12;
    const lineH = 10;
    const laneGap = 5;
    ctx.font = '8px Space Grotesk, sans-serif';
    ctx.textAlign = 'center';

    let laneY = itemStartY;
    for (const lane of ITEM_LANES) {
      const list = laneItems[lane.key];
      for (let j = 0; j < Math.min(list.length, lane.rows); j++) {
        const item = list[j];
        const key = item.toLowerCase();
        // The last visible row absorbs any overflow rather than dropping it
        // silently, so the column never claims fewer purchases than were made.
        const extra = list.length - lane.rows;
        const isLastVisible = j === lane.rows - 1 && extra > 0;
        const label = isLastVisible
          ? '+' + (extra + 1) + ' more'
          : (ITEM_SHORT[key] || item);
        ctx.fillStyle = isLastVisible ? NEUTRAL : itemColor(item, lane.key);
        ctx.fillText(label, x + barW / 2, laneY + j * lineH);
      }
      laneY += lane.rows * lineH + laneGap;
    }

    // Money line points. Only the end balance is drawn: the start of a round
    // is the end of the one before it, so the second line traced the same
    // shape one column over and doubled the ink for nothing.
    if (endMoney != null) endMoneyPts.push({ x: x + barW / 2, y: moneyToY(endMoney), val: endMoney });

    // Round number label
    ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.45)';
    ctx.font = '10px Space Grotesk, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('R' + (r.round_number || r.enriched?.round || '?'), x + barW / 2, h - 5);

    // Win/Loss indicator
    ctx.fillStyle = won ? (TC.success||'#34d399') : (TC.fail||'#f87171');
    ctx.font = 'bold 10px Space Grotesk, sans-serif';
    ctx.fillText(won ? 'W' : 'L', x + barW / 2, h - 16);
  }

  // --- Draw money lines ---
  const drawLine = (pts, color, label) => {
    if (pts.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i].x, pts[i].y);
    }
    ctx.stroke();
    // Dots
    for (const p of pts) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  drawLine(endMoneyPts, TC.sky||'#38bdf8', 'End $');

  // --- Legend ---
  const legendY = pad.top + 14;
  let lx = pad.left + 4;
  const legendText = (text, color, font) => {
    ctx.font = font || '10px Space Grotesk, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = color;
    ctx.fillText(text, lx, legendY);
    lx += ctx.measureText(text).width + 16;
  };

  ctx.strokeStyle = TC.sky||'#38bdf8'; ctx.lineWidth = 1.5; ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(lx, legendY - 4); ctx.lineTo(lx + 16, legendY - 4); ctx.stroke();
  lx += 20;
  legendText('End $', TC.gridText||'rgba(255,255,255,0.7)', 'bold 11px Space Grotesk, sans-serif');

  ctx.fillStyle = TC.gridText||'rgba(255,255,255,0.7)';
  ctx.beginPath();
  ctx.moveTo(lx + 5, legendY - 7); ctx.lineTo(lx, legendY - 13); ctx.lineTo(lx + 10, legendY - 13);
  ctx.closePath(); ctx.fill();
  lx += 14;
  legendText('spent', TC.gridText||'rgba(255,255,255,0.7)', 'bold 11px Space Grotesk, sans-serif');

  legendText('\u25CF used, had effect', TC.success||'#34d399');
  legendText('\u25CF no effect measured', TC.gridText||'rgba(255,255,255,0.45)');
  // Not "never used": grenades carry into later rounds, so all this column can
  // honestly say is that the buy went unthrown in the round it was made.
  legendText('\u25CF not used this round', TC.fail||'#f87171');
}

buildAimKpiMeta(null);

/* ── Trend charts ─────────────────────────────────────────────────────────
   The career view's three: the match-by-match progression line, the
   distribution strips beside each headline stat, and the five-axis role
   radar. They came out of breakdown.html for the same reason the match
   charts came out of match-breakdown.html — the landing page had nothing to
   say about trends, and hand-drawing a second version is what put the rest
   of that page out of date.

   All three were already parameterised by canvas id, which is what made them
   cheap to move. Like the rest of this file they reach for nothing but TC.
   ────────────────────────────────────────────────────────────────────────── */

/* Which series the trend line shows, and over how many matches. State rather
   than arguments because the chart is redrawn on every toggle and the page
   owns the buttons; top-level let bindings are shared across classic scripts,
   so the page's setters still reach these. Colours are functions so a theme
   change is picked up on the next draw rather than frozen at load. */
let _trendTimescale = 'all'; // 'all' | 5 | 10 | 20
let _trendMetrics = new Set(['rating', 'kd']); // enabled metric keys

const TREND_METRICS = [
  { key: 'rating', label: 'Rating', color: () => TC.purple  || '#cc97ff', getValue: d => d.hltv_rating },
  { key: 'adr',    label: 'ADR',    color: () => TC.cyan    || '#53ddfc', getValue: d => d.adr },
  { key: 'kd',     label: 'K/D',    color: () => TC.onText  || '#dee5ff', getValue: d => (d.deaths || 0) > 0 ? (d.kills || 0) / d.deaths : (d.kills || 0) },
  { key: 'kast',   label: 'KAST%',  color: () => TC.sky     || '#7dd3fc', getValue: d => d.kast },
  { key: 'aim',    label: 'Aim',    color: () => TC.amber   || '#fbbf24', getValue: d => d.aim_rating },
  { key: 'util',   label: 'Utility',color: () => TC.success || '#34d399', getValue: d => d.utility_rating },
];

function drawTrendChart(dataPoints, canvasId = 'trend-chart') {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  let pts = [...(dataPoints || [])];
  if (_trendTimescale !== 'all' && typeof _trendTimescale === 'number') pts = pts.slice(-_trendTimescale);

  const n = pts.length;
  const dpr = window.devicePixelRatio || 2;
  const w = canvas.offsetWidth || 600;
  const h = 200;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  if (n < 2) {
    ctx.fillStyle = TC.gridText || 'rgba(255,255,255,0.2)';
    ctx.font = '11px Manrope, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Not enough data to show trend', w / 2, h / 2);
    return;
  }

  const pad = { top: 18, right: 14, bottom: 26, left: 14 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;
  const getX = i => pad.left + (n > 1 ? (i / (n - 1)) * cw : cw / 2);

  // Grid lines
  ctx.strokeStyle = TC.grid || 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const gy = pad.top + (ch / 4) * g;
    ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(pad.left + cw, gy); ctx.stroke();
  }

  // Win / loss markers above chart
  pts.forEach((d, i) => {
    const won = (d.match_result || '').toLowerCase() === 'win';
    ctx.fillStyle = won ? '#34d39999' : '#f8717199';
    ctx.beginPath(); ctx.arc(getX(i), pad.top - 6, 3, 0, Math.PI * 2); ctx.fill();
  });

  // Draw each enabled metric
  for (const metric of TREND_METRICS) {
    if (!_trendMetrics.has(metric.key)) continue;
    const vals = pts.map(d => metric.getValue(d));
    const valid = vals.filter(v => v != null && !isNaN(v));
    if (!valid.length) continue;

    const minV = Math.min(...valid);
    const maxV = Math.max(...valid);
    const range = maxV - minV || 1;
    const color = metric.color();
    const getY = v => v == null || isNaN(v) ? null : pad.top + ch * (1 - (v - minV) / range);

    // Raw thin line
    ctx.beginPath(); ctx.strokeStyle = color + '55'; ctx.lineWidth = 1;
    let first = true;
    for (let i = 0; i < n; i++) {
      const y = getY(vals[i]);
      if (y == null) { first = true; continue; }
      first ? ctx.moveTo(getX(i), y) : ctx.lineTo(getX(i), y);
      first = false;
    }
    ctx.stroke();

    // 5-match rolling average — thick solid line
    const win5 = vals.map((_, i) => {
      const sl = vals.slice(Math.max(0, i - 4), i + 1).filter(v => v != null && !isNaN(v));
      return sl.length ? sl.reduce((a, b) => a + b, 0) / sl.length : null;
    });
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
    first = true;
    for (let i = 0; i < n; i++) {
      const y = getY(win5[i]);
      if (y == null) { first = true; continue; }
      first ? ctx.moveTo(getX(i), y) : ctx.lineTo(getX(i), y);
      first = false;
    }
    ctx.stroke();

    // Dots at raw values
    for (let i = 0; i < n; i++) {
      const y = getY(vals[i]);
      if (y == null) continue;
      ctx.beginPath(); ctx.arc(getX(i), y, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = color; ctx.fill();
    }
  }

  // X-axis date labels
  ctx.fillStyle = TC.gridText || 'rgba(255,255,255,0.3)';
  ctx.font = '8px Manrope, sans-serif';
  const fmt = d => d ? d.slice(0, 10) : '';
  ctx.textAlign = 'left';  ctx.fillText(fmt(pts[0].date), pad.left, h - 4);
  ctx.textAlign = 'right'; ctx.fillText(fmt(pts[n - 1].date), pad.left + cw, h - 4);
  if (n > 5) {
    const mi = Math.floor(n / 2);
    ctx.textAlign = 'center';
    ctx.fillText(fmt(pts[mi].date), getX(mi), h - 4);
  }
}

function drawDistStrip(canvasId, values, avg, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !values.length) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * 2; canvas.height = h * 2;
  ctx.scale(2, 2);
  ctx.clearRect(0, 0, w, h);

  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const pad = 4;
  const usable = w - pad * 2;
  const range = maxV - minV || 1;

  // Background track
  ctx.fillStyle = TC.track || 'rgba(255,255,255,0.04)';
  ctx.beginPath();
  ctx.roundRect(0, h / 2 - 2, w, 4, 2);
  ctx.fill();

  // Dots for each match value
  const r = Math.min(4, Math.max(2.5, 60 / values.length));
  for (const v of values) {
    const x = pad + ((v - minV) / range) * usable;
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, h / 2, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // Average marker
  const avgX = pad + ((avg - minV) / range) * usable;
  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(avgX, h / 2, r + 1.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = TC.avgStroke || '#fff';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawRoleRadar(canvasId, axes, color) {
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
  const R = Math.min(cx, cy) - 28;
  const angleStep = (Math.PI * 2) / n;
  const startAngle = -Math.PI / 2; // top

  // Grid rings
  for (let ring = 1; ring <= 4; ring++) {
    const r = R * ring / 4;
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const a = startAngle + i * angleStep;
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = TC.grid || 'rgba(255,255,255,' + (ring === 4 ? 0.12 : 0.06) + ')';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Axis lines + labels
  ctx.font = '600 8px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < n; i++) {
    const a = startAngle + i * angleStep;
    const xEnd = cx + Math.cos(a) * R;
    const yEnd = cy + Math.sin(a) * R;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(xEnd, yEnd);
    ctx.strokeStyle = TC.grid || 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Label
    const labelR = R + 16;
    const lx = cx + Math.cos(a) * labelR;
    const ly = cy + Math.sin(a) * labelR;
    ctx.fillStyle = TC.gridText || 'rgba(255,255,255,0.45)';
    ctx.fillText(labels[i], lx, ly);
  }

  // Data polygon (filled)
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const a = startAngle + i * angleStep;
    const v = Math.max(values[i], 0.04); // minimum visibility
    const x = cx + Math.cos(a) * R * v;
    const y = cy + Math.sin(a) * R * v;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = color + '18'; // ~10% opacity
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Data points
  for (let i = 0; i < n; i++) {
    const a = startAngle + i * angleStep;
    const v = Math.max(values[i], 0.04);
    const x = cx + Math.cos(a) * R * v;
    const y = cy + Math.sin(a) * R * v;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = TC.dotStroke || '#0a1628';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Value label near point
    const valR = Math.max(v * R + 10, 14);
    const vx = cx + Math.cos(a) * valR;
    const vy = cy + Math.sin(a) * valR;
    ctx.fillStyle = color;
    ctx.font = 'bold 8px system-ui, sans-serif';
    ctx.fillText(Math.round(axes[keys[i]] || 0), vx, vy);
  }
}
