import './theme-actions.js';
import { registerActions } from './actions.js';

// This page makes no requests — it is a coordinate-plotting tool that works
// entirely from the radar image and the callout list below. It used to declare
// an API base without the /api suffix every other page adds, which looked like
// a bug for as long as anyone assumed it was used.
const CALLOUTS = {
  de_mirage: [
    "B Site","Bench","B Van","Market","Market Door","Kitchen",
    "B Apartments","B Short",
    "A Site","A Default","Firebox","Tetris","Stairs","A Ramp",
    "Jungle","Ticket",
    "CT Spawn","Snipers Nest",
    "Connector","Chair",
    "Top Mid","Mid Window","Window","Catwalk","Short","Mid",
    "Underpass","Ladder Room",
    "T Spawn","T Ramp","A Palace","A Main","B Apartments Entrance",
    "A Side","B Side","Mid Area","T Area"
  ],
  de_inferno: [
    "A Site","Pit","Balcony","Library","Arch","Graveyard","Truck",
    "Mid","Alt Mid","Underpass","Top Mid",
    "B Site","Banana","Oranges","Car","Dark","CT","Construction","New Box","Coffins",
    "T Spawn","T Apartments","Apartments","Second Mid","Boiler",
    "A Side","B Side","T Area","T Approach"
  ],
  de_dust2: [
    "A Site","A Short (Cat)","A Long","A Long Doors","A Pit","A Car",
    "Goose","A Ramp","CT Spawn",
    "Mid Doors","Mid","Catwalk","Lower Tunnels","Upper Tunnels","Xbox",
    "B Site","B Tunnels","B Doors","B Window","B Back Site","B Car",
    "T Spawn","T Mid","Outside Long",
    "A Side","B Side","Mid Area"
  ],
  de_anubis: [
    "A Site","A Main","A Bridge","A Connector",
    "Mid","Top Mid",
    "B Site","B Main","B Pillar","B Connector",
    "CT Spawn","T Spawn",
    "A Side","B Side"
  ],
  de_nuke: [
    "A Site","Hut","Heaven","Hell","Squeaky","Main","Lobby",
    "Ramp","Outside","Secret","B Site","T Spawn","CT Spawn"
  ],
  de_ancient: [
    "A Site","A Main","A Short","Cave","Donut",
    "Mid","Top Mid",
    "B Site","B Main","B Ramp","B Pillar",
    "CT Spawn","T Spawn"
  ],
  de_overpass: [
    "A Site","A Long","A Short","Truck","Bathrooms",
    "B Site","B Short","Monster","Fountain","Connector",
    "CT Spawn","T Spawn"
  ],
  de_cache: [
    "A Site","CT Short","A Main","Highway",
    "Catwalk","Squeaky","Mid","Boiler",
    "B Site","B Ramp","B Halls",
    "CT Spawn","Garage","T Spawn",
    "A Side","B Side","Mid Area","T Area"
  ],
  de_vertigo: [
    "A Site","A Ramp","A Short",
    "B Site","B Stairs","B Ramp",
    "Mid","CT Spawn","T Spawn"
  ],
  de_train: [
    "A Site","Ivy","Connector","Pop Dog",
    "B Site","B Upper","B Lower",
    "Mid","CT Spawn","T Spawn"
  ]
};

let currentMap = '';
let currentCallout = null;
let placements = {};  // { calloutName: {px, py} }
let placementOrder = [];  // for undo

function loadMap() {
  const sel = document.getElementById('map-select');
  currentMap = sel.value;
  if (!currentMap) return;

  // Load radar image
  const img = document.getElementById('radar-img');
  const shortName = currentMap.replace('de_','').replace('cs_','');
  img.src = `/frontend/img/radar/${shortName}.png`;

  // Reset
  placements = {};
  placementOrder = [];
  currentCallout = null;
  document.getElementById('current-callout').textContent = 'Click a callout below';

  // Build callout list
  const list = document.getElementById('callout-list');
  list.innerHTML = '';
  const callouts = CALLOUTS[currentMap] || [];
  for (const c of callouts) {
    const div = document.createElement('div');
    div.className = 'callout-item';
    div.innerHTML = `<span class="dot pending" id="dot-${c}"></span><span>${c}</span><span class="coords" id="coords-${c}"></span>`;
    div.onclick = () => selectCallout(c, div);
    list.appendChild(div);
  }
  updateProgress();
  redrawCanvas();
}

function selectCallout(name, el) {
  document.querySelectorAll('.callout-item').forEach(e => e.classList.remove('active'));
  el.classList.add('active');
  currentCallout = name;
  document.getElementById('current-callout').textContent = name;
}

function updateProgress() {
  const total = (CALLOUTS[currentMap] || []).length;
  const done = Object.keys(placements).length;
  document.getElementById('progress').textContent = `${done} / ${total} placed`;
}

// Canvas click — place callout
document.getElementById('radar-canvas').addEventListener('click', function(e) {
  if (!currentCallout || !currentMap) return;

  const rect = this.getBoundingClientRect();
  const px = Math.round((e.clientX - rect.left) * (1024 / rect.width));
  const py = Math.round((e.clientY - rect.top) * (1024 / rect.height));

  placements[currentCallout] = { px, py };
  placementOrder.push(currentCallout);

  // Update UI
  const dot = document.getElementById('dot-' + currentCallout);
  if (dot) { dot.classList.remove('pending'); dot.classList.add('placed'); }
  const coords = document.getElementById('coords-' + currentCallout);
  if (coords) coords.textContent = `(${px}, ${py})`;

  // Auto-advance to next unplaced callout
  const callouts = CALLOUTS[currentMap] || [];
  const currentIdx = callouts.indexOf(currentCallout);
  let nextIdx = -1;
  for (let i = currentIdx + 1; i < callouts.length; i++) {
    if (!placements[callouts[i]]) { nextIdx = i; break; }
  }
  if (nextIdx === -1) {
    for (let i = 0; i < currentIdx; i++) {
      if (!placements[callouts[i]]) { nextIdx = i; break; }
    }
  }

  if (nextIdx >= 0) {
    const items = document.querySelectorAll('.callout-item');
    selectCallout(callouts[nextIdx], items[nextIdx]);
    items[nextIdx].scrollIntoView({ block: 'nearest' });
  } else {
    document.getElementById('current-callout').textContent = 'All placed!';
    currentCallout = null;
  }

  updateProgress();
  redrawCanvas();
});

// Mouse move: show crosshair coordinates
document.getElementById('radar-canvas').addEventListener('mousemove', function(e) {
  const rect = this.getBoundingClientRect();
  const px = Math.round((e.clientX - rect.left) * (1024 / rect.width));
  const py = Math.round((e.clientY - rect.top) * (1024 / rect.height));
  this.title = `(${px}, ${py})`;
});

function redrawCanvas() {
  const canvas = document.getElementById('radar-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 1024, 1024);
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const accent = isLight ? '#6d4ea3' : '#fbbf24';
  const labelBg = isLight ? 'rgba(255,255,255,0.85)' : 'rgba(0,0,0,0.7)';

  for (const [name, pos] of Object.entries(placements)) {
    // Crosshair
    ctx.strokeStyle = isLight ? 'rgba(109,78,163,0.6)' : 'rgba(251,191,36,0.6)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pos.px - 12, pos.py); ctx.lineTo(pos.px + 12, pos.py);
    ctx.moveTo(pos.px, pos.py - 12); ctx.lineTo(pos.px, pos.py + 12);
    ctx.stroke();

    // Circle
    ctx.strokeStyle = isLight ? 'rgba(109,78,163,0.8)' : 'rgba(251,191,36,0.8)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(pos.px, pos.py, 8, 0, Math.PI * 2);
    ctx.stroke();

    // Label
    ctx.font = 'bold 10px sans-serif';
    const w = ctx.measureText(name).width;
    ctx.fillStyle = labelBg;
    ctx.fillRect(pos.px - w/2 - 3, pos.py - 22, w + 6, 14);
    ctx.fillStyle = accent;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(name, pos.px, pos.py - 15);
    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
  }
}

function undoLast() {
  if (placementOrder.length === 0) return;
  const last = placementOrder.pop();
  delete placements[last];
  const dot = document.getElementById('dot-' + last);
  if (dot) { dot.classList.remove('placed'); dot.classList.add('pending'); }
  const coords = document.getElementById('coords-' + last);
  if (coords) coords.textContent = '';
  updateProgress();
  redrawCanvas();
}

function resetAll() {
  if (!confirm('Reset all placements?')) return;
  placements = {};
  placementOrder = [];
  document.querySelectorAll('.dot').forEach(d => { d.classList.remove('placed'); d.classList.add('pending'); });
  document.querySelectorAll('.coords').forEach(c => c.textContent = '');
  currentCallout = null;
  document.getElementById('current-callout').textContent = 'Click a callout below';
  updateProgress();
  redrawCanvas();
}

function exportJSON() {
  const result = {};
  for (const [name, pos] of Object.entries(placements)) {
    result[name] = [pos.px, pos.py];
  }
  const json = JSON.stringify({ map: currentMap, positions: result }, null, 2);
  document.getElementById('output-area').value = json;

  // Copy to clipboard
  navigator.clipboard.writeText(json).then(() => {
    document.getElementById('output-btn').textContent = 'Copied!';
    setTimeout(() => { document.getElementById('output-btn').textContent = 'Export JSON'; }, 2000);
  });
}


/* What this file offers the markup. See js/actions.js. */
registerActions({
  exportJSON,
  loadMap,
  resetAll,
  undoLast,
});
