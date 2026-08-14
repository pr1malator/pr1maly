import './theme-actions.js';
import { registerActions } from './actions.js';

// ===================================================================
// Sidebar toggle
// ===================================================================
function toggleSidebar() {
  const sb = document.getElementById('sidebar');
  sb.classList.toggle('collapsed');
  const main = document.getElementById('main-area');
  main.style.paddingLeft = sb.classList.contains('collapsed') ? '5.5rem' : '';
}

// ===================================================================
// State
// ===================================================================
let matchId = null;
let matchMeta = null;      // overview response (round list)
let roundData = null;      // current round's frame data
let radarImage = null;
let radarLoaded = false;

let playing = false;
let speed = 1;
let currentFrameIdx = 0;
let lastTimestamp = 0;
let accumulator = 0;       // ms accumulated for frame stepping

const TICK_RATE = 64;
const SAMPLE_INTERVAL = 32;
const MS_PER_FRAME = (SAMPLE_INTERVAL / TICK_RATE) * 1000;  // ~500ms real time per frame

// Kill feed items with expiry
let killFeedItems = [];

// Canvas
let canvas, ctx;

// ===================================================================
// Init
// ===================================================================
window.addEventListener('DOMContentLoaded', () => {
  canvas = document.getElementById('replay-canvas');
  ctx = canvas.getContext('2d');

  // Get match_id from URL params
  const params = new URLSearchParams(window.location.search);
  matchId = params.get('id');
  const startRound = parseInt(params.get('round')) || 1;

  if (!matchId) {
    document.getElementById('match-info').textContent = 'No match ID provided. Open a match from the Matches page.';
    return;
  }

  // Update back link
  const backLink = document.getElementById('back-to-match');
  backLink.href = `match-breakdown.html?id=${encodeURIComponent(matchId)}`;
  backLink.classList.remove('hidden');

  // Update match link in sidebar
  const navMatch = document.getElementById('nav-match');
  if (navMatch) navMatch.href = `match-breakdown.html?id=${encodeURIComponent(matchId)}`;

  loadMatchReplay(startRound);

  // Timeline scrubber click/drag
  setupTimeline();
});

// ===================================================================
// Data loading
// ===================================================================
async function loadMatchReplay(startRound) {
  try {
    const res = await fetch(`/api/matches/${encodeURIComponent(matchId)}/replay`);
    if (!res.ok) throw new Error('Failed to fetch replay metadata');
    matchMeta = await res.json();

    if (!matchMeta.has_replay) {
      document.getElementById('no-replay-msg').classList.remove('hidden');
      document.getElementById('match-info').textContent = `${matchMeta.map_name || 'Unknown map'} — No replay data`;
      return;
    }

    document.getElementById('replay-container').classList.remove('hidden');
    document.getElementById('match-info').textContent = `${matchMeta.map_name} — ${matchMeta.total_rounds} rounds`;

    // Populate round selector
    const sel = document.getElementById('round-select');
    sel.innerHTML = '';
    for (const r of matchMeta.rounds) {
      const opt = document.createElement('option');
      opt.value = r.round;
      opt.textContent = r.round;
      sel.appendChild(opt);
    }

    // Load radar image
    radarImage = new Image();
    radarImage.onload = () => { radarLoaded = true; drawFrame(); };
    radarImage.onerror = () => { radarLoaded = false; drawFrame(); };
    radarImage.src = matchMeta.radar_image;

    // Load first round
    const targetRound = matchMeta.rounds.find(r => r.round === startRound) ? startRound : matchMeta.rounds[0]?.round || 1;
    await loadRound(targetRound);
  } catch (err) {
    document.getElementById('match-info').textContent = 'Error loading replay: ' + err.message;
    console.error(err);
  }
}

/* The round dropdown. loadRound itself takes a number and is called from the
   next/previous controls too, so the markup gets its own way in. */
function loadRoundFromSelect(event, select) {
  loadRound(parseInt(select.value));
}

async function loadRound(roundNum) {
  stop();
  currentFrameIdx = 0;
  accumulator = 0;
  killFeedItems = [];
  document.getElementById('kill-feed').innerHTML = '';

  try {
    const res = await fetch(`/api/matches/${encodeURIComponent(matchId)}/replay?round_number=${roundNum}`);
    if (!res.ok) throw new Error('Failed to fetch round data');
    roundData = await res.json();

    // Update round selector
    document.getElementById('round-select').value = roundNum;

    // Update round info
    document.getElementById('ri-round').textContent = roundNum;
    document.getElementById('ri-frames').textContent = roundData.frames?.length || 0;
    const totalTicks = roundData.frames?.length ? roundData.frames[roundData.frames.length - 1][0] : 0;
    const durationSec = totalTicks / TICK_RATE;
    document.getElementById('ri-duration').textContent = formatTime(durationSec);

    // Build player list
    buildPlayerList(roundData.players || {});

    // Update timeline
    updateTimeDisplay();
    drawFrame();
  } catch (err) {
    console.error('Error loading round:', err);
  }
}

// ===================================================================
// Player list
// ===================================================================
function buildPlayerList(players) {
  const ctEl = document.getElementById('ct-players');
  const tEl = document.getElementById('t-players');
  ctEl.innerHTML = '<div class="text-[10px] font-bold text-info uppercase tracking-widest mb-1">Counter-Terrorists</div>';
  tEl.innerHTML = '<div class="text-[10px] font-bold text-caution uppercase tracking-widest mb-1">Terrorists</div>';

  for (const [sid, info] of Object.entries(players)) {
    const div = document.createElement('div');
    div.className = 'flex items-center gap-2 px-2 py-1 rounded';
    div.id = `player-${sid}`;
    const color = info.team === 3 ? 'bg-info' : 'bg-caution';
    div.innerHTML = `
      <div class="w-2.5 h-2.5 rounded-full ${color} flex-shrink-0"></div>
      <span class="text-xs text-on-surface truncate">${escapeHtml(info.name)}</span>
      <span class="ml-auto text-[10px] text-on-surface-variant player-hp" data-sid="${sid}">100</span>
    `;
    if (info.team === 3) ctEl.appendChild(div);
    else tEl.appendChild(div);
  }
}

// ===================================================================
// Canvas rendering
// ===================================================================
function drawFrame() {
  if (!ctx) return;
  ctx.clearRect(0, 0, 1024, 1024);

  // Draw radar background
  if (radarLoaded && radarImage) {
    ctx.drawImage(radarImage, 0, 0, 1024, 1024);
  } else {
    ctx.fillStyle = TC.canvasBg || '#0a1628';
    ctx.fillRect(0, 0, 1024, 1024);
  }

  if (!roundData || !roundData.frames || roundData.frames.length === 0) return;

  const players = roundData.players || {};
  const frames = roundData.frames;
  const events = roundData.events || [];
  const idx = Math.min(currentFrameIdx, frames.length - 1);
  const currentTick = frames[idx][0];
  const positions = frames[idx][1];

  // Interpolate between current and next frame for smooth movement
  let interpPositions = positions;
  if (playing && idx < frames.length - 1) {
    const nextPositions = frames[idx + 1][1];
    const tickDelta = frames[idx + 1][0] - frames[idx][0];
    if (tickDelta > 0) {
      const t = Math.min(accumulator / (MS_PER_FRAME / speed), 1);
      interpPositions = {};
      for (const sid of Object.keys(positions)) {
        if (nextPositions[sid]) {
          interpPositions[sid] = [
            positions[sid][0] + (nextPositions[sid][0] - positions[sid][0]) * t,
            positions[sid][1] + (nextPositions[sid][1] - positions[sid][1]) * t,
            positions[sid][2],  // HP doesn't interpolate
          ];
        } else {
          interpPositions[sid] = positions[sid];
        }
      }
      // Add players only in next frame
      for (const sid of Object.keys(nextPositions)) {
        if (!interpPositions[sid]) interpPositions[sid] = nextPositions[sid];
      }
    }
  }

  // Draw movement trails (last 10 frames)
  const trailLen = Math.min(10, idx);
  if (trailLen > 0) {
    ctx.globalAlpha = 0.15;
    ctx.lineWidth = 2;
    for (const sid of Object.keys(players)) {
      const team = players[sid].team;
      ctx.strokeStyle = team === 3 ? (TC.ct || '#60a5fa') : (TC.t || '#facc15');
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

  // Draw kill lines for recent kills
  for (const ev of events) {
    if (ev.type !== 'kill') continue;
    const evTick = ev.t;
    const tickDiff = currentTick - evTick;
    if (tickDiff < 0 || tickDiff > SAMPLE_INTERVAL * 4) continue; // Show for ~2s

    const attackerPos = positions[ev.attacker];
    const victimPos = positions[ev.victim];
    if (!attackerPos || !victimPos) continue;

    const alpha = Math.max(0, 1 - tickDiff / (SAMPLE_INTERVAL * 4));
    ctx.globalAlpha = alpha * 0.6;
    ctx.strokeStyle = TC.kill || '#ff6e84';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(attackerPos[0], attackerPos[1]);
    ctx.lineTo(victimPos[0], victimPos[1]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // Death X marker
    if (tickDiff < SAMPLE_INTERVAL * 6) {
      const xAlpha = Math.max(0, 1 - tickDiff / (SAMPLE_INTERVAL * 6));
      ctx.globalAlpha = xAlpha * 0.8;
      ctx.strokeStyle = TC.kill || '#ff6e84';
      ctx.lineWidth = 3;
      const sz = 8;
      ctx.beginPath();
      ctx.moveTo(victimPos[0] - sz, victimPos[1] - sz);
      ctx.lineTo(victimPos[0] + sz, victimPos[1] + sz);
      ctx.moveTo(victimPos[0] + sz, victimPos[1] - sz);
      ctx.lineTo(victimPos[0] - sz, victimPos[1] + sz);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  // Draw grenade events
  // Per-type display durations in ticks (TICK_RATE = 64)
  const NADE_DURATIONS = { flash: 128, he: 128, smoke: 1152, molotov: 448 }; // flash/HE 2s, smoke 18s, molotov 7s
  const NADE_COLORS = { flash: '#fffbe6', he: '#ff6e84', smoke: '#a3e635', molotov: '#fb923c' };
  const NADE_ICONS = { flash: '⚡', he: '💥', smoke: '💨', molotov: '🔥' };
  const NADE_RADIUS = { flash: 14, he: 14, smoke: 24, molotov: 22 };
  for (const ev of events) {
    if (ev.type !== 'grenade' || ev.px == null || ev.py == null) continue;
    const diff = currentTick - ev.t;
    const duration = NADE_DURATIONS[ev.grenade] || 128;
    if (diff < 0 || diff > duration) continue;

    const progress = diff / duration; // 0→1
    const gx = ev.px, gy = ev.py;
    const color = NADE_COLORS[ev.grenade] || '#ffffff';
    const baseRadius = NADE_RADIUS[ev.grenade] || 14;

    // ------- Flight path from thrower to detonation -------
    if (ev.thrower) {
      // Find thrower position at the grenade tick from frames
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
        // Small dot at thrower origin
        ctx.globalAlpha = Math.max(0.1, 0.4 * (1 - progress));
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(throwerPos[0], throwerPos[1], 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // ------- Filled circle (smoke/molotov persistent, HE/flash burst) -------
    if (ev.grenade === 'smoke') {
      // Smoke cloud — solid green disc, fades in last 3s
      const fadeStart = 1 - (3 * TICK_RATE / duration); // start fading at 15s
      const smokeAlpha = progress < fadeStart ? 0.45 : 0.45 * (1 - (progress - fadeStart) / (1 - fadeStart));
      ctx.globalAlpha = Math.max(0.08, smokeAlpha);
      ctx.fillStyle = '#a3e635';
      ctx.beginPath();
      ctx.arc(gx, gy, baseRadius, 0, Math.PI * 2);
      ctx.fill();
      // Outer ring
      ctx.globalAlpha = Math.max(0.15, smokeAlpha * 0.7);
      ctx.strokeStyle = '#a3e635';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(gx, gy, baseRadius + 3, 0, Math.PI * 2);
      ctx.stroke();
    } else if (ev.grenade === 'molotov') {
      // Molotov fire — solid orange disc, fades in last 2s
      const fadeStart = 1 - (2 * TICK_RATE / duration);
      const fireAlpha = progress < fadeStart ? 0.5 : 0.5 * (1 - (progress - fadeStart) / (1 - fadeStart));
      ctx.globalAlpha = Math.max(0.08, fireAlpha);
      ctx.fillStyle = '#fb923c';
      ctx.beginPath();
      ctx.arc(gx, gy, baseRadius, 0, Math.PI * 2);
      ctx.fill();
      // Flickering outer ring
      ctx.globalAlpha = Math.max(0.15, fireAlpha * 0.7);
      ctx.strokeStyle = '#ff6347';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(gx, gy, baseRadius + 3, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      // HE / Flash — burst circle that expands then fades
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

    // ------- Activation timer for smoke / molotov -------
    if (ev.grenade === 'smoke' || ev.grenade === 'molotov') {
      const totalSec = (ev.grenade === 'smoke') ? 18 : 7;
      const elapsedSec = diff / TICK_RATE;
      const remainSec = Math.max(0, totalSec - elapsedSec);
      const fadeAlpha = progress < 0.8 ? 0.9 : 0.9 * (1 - (progress - 0.8) / 0.2);
      ctx.globalAlpha = Math.max(0.15, fadeAlpha);
      ctx.font = 'bold 11px "Space Grotesk", sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#ffffff';
      ctx.fillText(remainSec.toFixed(1) + 's', gx, gy + baseRadius + 12);
    }

    // ------- Center icon -------
    const iconAlpha = (ev.grenade === 'smoke' || ev.grenade === 'molotov')
      ? Math.max(0.15, progress < 0.85 ? 0.9 : 0.9 * (1 - (progress - 0.85) / 0.15))
      : Math.max(0.2, 1 - progress);
    ctx.globalAlpha = iconAlpha;
    ctx.font = 'bold 13px "Space Grotesk", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(NADE_ICONS[ev.grenade] || '◆', gx, gy);

    // ------- Thrower name label -------
    if (ev.thrower && players[ev.thrower]) {
      const throwerInfo = players[ev.thrower];
      ctx.globalAlpha = Math.max(0.1, 0.7 * (1 - progress * 0.8));
      ctx.font = 'bold 9px "Space Grotesk", sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = throwerInfo.team === 3 ? (TC.ctName || '#93c5fd') : (TC.tName || '#fde047');
      ctx.fillText(throwerInfo.name || '', gx, gy - baseRadius - 4);
    }

    ctx.globalAlpha = 1;
    ctx.textBaseline = 'alphabetic';
  }

  // Draw player dots
  for (const [sid, pos] of Object.entries(interpPositions)) {
    const px = pos[0], py = pos[1], hp = pos[2];
    const info = players[sid];
    if (!info) continue;

    const isCT = info.team === 3;
    const alive = hp > 0;

    if (!alive) {
      // Dead: small X
      ctx.globalAlpha = 0.3;
      ctx.strokeStyle = isCT ? (TC.ct || '#60a5fa') : (TC.t || '#facc15');
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

    // Outer glow
    ctx.shadowColor = isCT ? 'rgba(96, 165, 250, 0.5)' : 'rgba(250, 204, 21, 0.5)';
    ctx.shadowBlur = 8;

    // Player circle
    ctx.fillStyle = isCT ? (TC.ctDot || '#3b82f6') : (TC.tDot || '#eab308');
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.shadowBlur = 0;

    // HP ring (if damaged)
    if (hp < 100) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(px, py, radius + 2, -Math.PI / 2, -Math.PI / 2 + (Math.PI * 2), false);
      ctx.stroke();

      const hpAngle = (hp / 100) * Math.PI * 2;
      const hpColor = hp > 50 ? (TC.hpGood || '#4ade80') : hp > 25 ? (TC.hpWarn || '#facc15') : (TC.kill || '#ff6e84');
      ctx.strokeStyle = hpColor;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(px, py, radius + 2, -Math.PI / 2, -Math.PI / 2 + hpAngle, false);
      ctx.stroke();
    }

    // Player name
    ctx.font = 'bold 10px "Space Grotesk", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = isCT ? (TC.ctName || '#93c5fd') : (TC.tName || '#fde047');
    ctx.fillText(info.name || sid.slice(0, 6), px, py - radius - 5);
  }

  // Update player HP in sidebar
  document.querySelectorAll('.player-hp').forEach(el => {
    const sid = el.dataset.sid;
    const pos = positions[sid];
    if (pos) {
      el.textContent = pos[2] > 0 ? pos[2] : '☠';
      el.className = `ml-auto text-[10px] player-hp ${pos[2] > 0 ? 'text-on-surface-variant' : 'text-error/50'}`;
    }
  });

  // Process kill events for kill feed
  for (const ev of events) {
    if (ev.t <= currentTick && !ev._shown) {
      ev._shown = true;
      if (ev.type === 'kill') addKillFeedItem(ev, players);
      else if (ev.type === 'grenade') addGrenadeFeedItem(ev, players);
    }
  }
}

// ===================================================================
// Kill feed
// ===================================================================
function addKillFeedItem(ev, players) {
  const feed = document.getElementById('kill-feed');
  const attackerName = players[ev.attacker]?.name || '?';
  const victimName = players[ev.victim]?.name || '?';
  const attackerTeam = players[ev.attacker]?.team;
  const victimTeam = players[ev.victim]?.team;
  const aColor = attackerTeam === 3 ? 'text-info' : 'text-caution';
  const vColor = victimTeam === 3 ? 'text-info' : 'text-caution';
  const hs = ev.headshot ? ' <span class="text-error">HS</span>' : '';
  const weapon = ev.weapon || '?';

  const div = document.createElement('div');
  div.className = 'kill-feed-item flex items-center gap-1 py-0.5';
  div.innerHTML = `<span class="${aColor} font-bold truncate max-w-[70px]">${escapeHtml(attackerName)}</span>
    <span class="text-on-surface-variant/50 text-[10px]">[${escapeHtml(weapon)}${hs}]</span>
    <span class="${vColor} truncate max-w-[70px]">${escapeHtml(victimName)}</span>`;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;

  // Auto-remove after 8 seconds
  setTimeout(() => {
    div.classList.add('fade-out');
    setTimeout(() => div.remove(), 500);
  }, 8000);
}

function addGrenadeFeedItem(ev, players) {
  const feed = document.getElementById('kill-feed');
  const throwerName = players[ev.thrower]?.name || '';
  const throwerTeam = players[ev.thrower]?.team;
  const tColor = throwerTeam === 3 ? 'text-info' : throwerTeam === 2 ? 'text-caution' : 'text-on-surface-variant';
  const labels = { flash: '⚡ Flash', he: '💥 HE', smoke: '💨 Smoke', molotov: '🔥 Molotov' };
  const colors = { flash: 'text-caution', he: 'text-bad', smoke: 'text-good', molotov: 'text-warn' };
  const label = labels[ev.grenade] || ev.grenade;
  const nColor = colors[ev.grenade] || 'text-on-surface-variant';

  const div = document.createElement('div');
  div.className = 'kill-feed-item flex items-center gap-1 py-0.5';
  if (throwerName) {
    div.innerHTML = `<span class="${tColor} font-bold truncate max-w-[70px]">${escapeHtml(throwerName)}</span>
      <span class="${nColor} text-[10px]">${label}</span>`;
  } else {
    div.innerHTML = `<span class="${nColor} text-[10px]">${label}</span>`;
  }
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;

  setTimeout(() => {
    div.classList.add('fade-out');
    setTimeout(() => div.remove(), 500);
  }, 5000);
}

// ===================================================================
// Playback loop
// ===================================================================
function gameLoop(timestamp) {
  if (!playing) return;

  if (lastTimestamp === 0) lastTimestamp = timestamp;
  const dt = timestamp - lastTimestamp;
  lastTimestamp = timestamp;

  accumulator += dt * speed;
  const msPerFrame = MS_PER_FRAME;

  let stepped = false;
  while (accumulator >= msPerFrame) {
    accumulator -= msPerFrame;
    currentFrameIdx++;
    stepped = true;
  }

  if (roundData && roundData.frames && currentFrameIdx >= roundData.frames.length) {
    currentFrameIdx = roundData.frames.length - 1;
    stop();
    return;
  }

  if (stepped) {
    drawFrame();
    updateTimeDisplay();
  } else if (playing) {
    // Still interpolate between frames for smooth motion
    drawFrame();
  }

  requestAnimationFrame(gameLoop);
}

function togglePlay() {
  if (playing) stop();
  else play();
}

function play() {
  if (!roundData || !roundData.frames || roundData.frames.length === 0) return;
  // If at end, restart
  if (currentFrameIdx >= roundData.frames.length - 1) {
    currentFrameIdx = 0;
    accumulator = 0;
    // Reset kill feed shown flags
    if (roundData.events) roundData.events.forEach(e => e._shown = false);
    killFeedItems = [];
    document.getElementById('kill-feed').innerHTML = '';
  }
  playing = true;
  lastTimestamp = 0;
  document.getElementById('play-icon').textContent = 'pause';
  requestAnimationFrame(gameLoop);
}

function stop() {
  playing = false;
  document.getElementById('play-icon').textContent = 'play_arrow';
}

function setSpeed(s) {
  speed = s;
  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.classList.toggle('active', parseFloat(btn.textContent) === s);
  });
}

function prevRound() {
  if (!matchMeta?.rounds) return;
  const sel = document.getElementById('round-select');
  const cur = parseInt(sel.value);
  const rounds = matchMeta.rounds.map(r => r.round);
  const idx = rounds.indexOf(cur);
  if (idx > 0) loadRound(rounds[idx - 1]);
}

function nextRound() {
  if (!matchMeta?.rounds) return;
  const sel = document.getElementById('round-select');
  const cur = parseInt(sel.value);
  const rounds = matchMeta.rounds.map(r => r.round);
  const idx = rounds.indexOf(cur);
  if (idx < rounds.length - 1) loadRound(rounds[idx + 1]);
}

// ===================================================================
// Timeline scrubber
// ===================================================================
function setupTimeline() {
  const bar = document.getElementById('timeline-bar');
  let dragging = false;

  function seekTo(e) {
    if (!roundData?.frames?.length) return;
    const rect = bar.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    currentFrameIdx = Math.floor(pct * (roundData.frames.length - 1));
    accumulator = 0;

    // Reset kill events shown state up to current tick
    const currentTick = roundData.frames[currentFrameIdx][0];
    if (roundData.events) {
      roundData.events.forEach(ev => { ev._shown = ev.t <= currentTick; });
    }
    // Rebuild kill feed
    document.getElementById('kill-feed').innerHTML = '';
    if (roundData.events) {
      for (const ev of roundData.events) {
        if (ev.t <= currentTick && ev.type === 'kill') {
          addKillFeedItem(ev, roundData.players || {});
        }
      }
    }

    drawFrame();
    updateTimeDisplay();
  }

  bar.addEventListener('mousedown', (e) => { dragging = true; seekTo(e); });
  window.addEventListener('mousemove', (e) => { if (dragging) seekTo(e); });
  window.addEventListener('mouseup', () => { dragging = false; });
}

function updateTimeDisplay() {
  if (!roundData?.frames?.length) return;
  const total = roundData.frames.length - 1;
  const idx = Math.min(currentFrameIdx, total);
  const pct = total > 0 ? (idx / total) * 100 : 0;

  document.getElementById('timeline-progress').style.width = pct + '%';
  document.getElementById('timeline-thumb').style.left = `calc(${pct}% - 8px)`;

  const currentTick = roundData.frames[idx]?.[0] || 0;
  const totalTick = roundData.frames[total]?.[0] || 0;
  document.getElementById('time-current').textContent = formatTime(currentTick / TICK_RATE);
  document.getElementById('time-total').textContent = formatTime(totalTick / TICK_RATE);
}

// ===================================================================
// Utilities
// ===================================================================
function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}

// ===================================================================
// Keyboard shortcuts
// ===================================================================
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

  switch (e.key) {
    case ' ':
      e.preventDefault();
      togglePlay();
      break;
    case 'ArrowRight':
      e.preventDefault();
      if (e.shiftKey) nextRound();
      else if (roundData?.frames) {
        currentFrameIdx = Math.min(currentFrameIdx + 5, roundData.frames.length - 1);
        accumulator = 0;
        drawFrame();
        updateTimeDisplay();
      }
      break;
    case 'ArrowLeft':
      e.preventDefault();
      if (e.shiftKey) prevRound();
      else if (roundData?.frames) {
        currentFrameIdx = Math.max(currentFrameIdx - 5, 0);
        accumulator = 0;
        drawFrame();
        updateTimeDisplay();
      }
      break;
    case '+':
    case '=':
      { const speeds = [0.25, 0.5, 1, 2, 4]; const i = speeds.indexOf(speed); if (i < speeds.length - 1) setSpeed(speeds[i + 1]); }
      break;
    case '-':
      { const speeds = [0.25, 0.5, 1, 2, 4]; const i = speeds.indexOf(speed); if (i > 0) setSpeed(speeds[i - 1]); }
      break;
  }
});


/* What this file offers the markup. See js/actions.js. */
registerActions({
  loadRoundFromSelect,
  nextRound,
  prevRound,
  setSpeed,
  togglePlay,
  toggleSidebar,
});
