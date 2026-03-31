/* === ResearchClawBench Frontend === */
/* Set STATIC_MODE = true for GitHub Pages (no backend). Defined in index.html before this script. */
if (typeof STATIC_MODE === 'undefined') var STATIC_MODE = false;
const API = '';

/* ── Background wave-mosaic ─────────────────────────────────────────────── */
(function () {
  var canvas, ctx, t = 0, rafId = null, cachedRgb = '10,10,10';
  var lastDraw = 0, TILE = 26, GAP = 1, FRAME_MS = 1000 / 24;

  function rgb() {
    var th = document.documentElement.getAttribute('data-theme') || 'white';
    return th === 'dark' ? '220,210,175' : th === 'yellow' ? '120,85,20' : th === 'blue' ? '38,88,155' : '10,10,10';
  }

  function frame(ts) {
    rafId = requestAnimationFrame(frame);
    if (ts - lastDraw < FRAME_MS) return;
    lastDraw = ts;
    var w = canvas.width, h = canvas.height;
    var cols = Math.ceil(w / TILE) + 1, rows = Math.ceil(h / TILE) + 1;
    var pre = 'rgba(' + cachedRgb + ',';
    ctx.clearRect(0, 0, w, h);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var wave = 0.6 * Math.sin(c * 0.21 + t * 0.36) * Math.sin(r * 0.17 + t * 0.28)
                 + 0.4 * Math.sin(c * 0.11 - r * 0.13 + t * 0.19);
        var v = ((wave + 1) * 0.5); v = v * v * v;
        var a = Math.round((0.004 + v * 0.186) * 100) / 100;
        if (a < 0.02) continue;
        ctx.fillStyle = pre + a + ')';
        ctx.fillRect(c * TILE + GAP, r * TILE + GAP, TILE - GAP, TILE - GAP);
      }
    }
    t += 0.007;
  }

  var resizeTimer;
  function resize() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      var nw = window.innerWidth, nh = window.innerHeight;
      if (nw === canvas.width && Math.abs(nh - canvas.height) <= 90) return;
      canvas.width = nw; canvas.height = nh;
    }, 120);
  }

  document.addEventListener('DOMContentLoaded', function () {
    canvas = document.createElement('canvas');
    canvas.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;will-change:transform;';
    document.body.insertBefore(canvas, document.body.firstChild);
    ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth; canvas.height = window.innerHeight;
    window.addEventListener('resize', resize);
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) { if (rafId) { cancelAnimationFrame(rafId); rafId = null; } }
      else { if (!rafId) rafId = requestAnimationFrame(frame); }
    });
    rafId = requestAnimationFrame(frame);
  });

  new MutationObserver(function () { cachedRgb = rgb(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();

/* ── Theme Switcher ──────────────────────────────────────────────────── */
(function () {
  var THEMES = ['white', 'yellow', 'blue', 'dark'];
  var LABELS = { white: 'Pure White', yellow: 'Warm Yellow', blue: 'Cool Blue', dark: 'Dark' };

  function apply(theme) {
    if (theme === 'white') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('site-theme', theme); } catch (e) {}
    document.querySelectorAll('.theme-dot').forEach(d => d.classList.toggle('active', d.dataset.theme === theme));
  }

  var saved = 'white';
  try { saved = localStorage.getItem('site-theme') || 'white'; } catch (e) {}
  apply(saved);

  document.addEventListener('DOMContentLoaded', function () {
    var sw = document.createElement('div'); sw.id = 'theme-switcher';
    THEMES.forEach(function (t) {
      var b = document.createElement('button');
      b.className = 'theme-dot'; b.dataset.theme = t; b.title = LABELS[t];
      b.onclick = function () { apply(t); };
      sw.appendChild(b);
    });
    document.body.appendChild(sw);
    apply(saved);
  });
})();

/* ── State ───────────────────────────────────────────────────────────── */
let state = { currentTaskId: null, currentRunId: null, eventSource: null, tasks: {}, selectedAgent: null, userSelectedFile: false, autoTrackTimer: null, agentLogos: {}, autoFollow: true, lastTab: 'research', _cachedInputFiles: [], _selectEpoch: 0 };

/* ── Init ────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
  await loadConfig();
  await loadTasks();
  await loadDashboard();
  setupTabs();
  setupButtons();
});

/* ── Config / Agent Presets ──────────────────────────────────────────── */
const ICON_COLORS = { C: '#3f3f46', X: '#166534', O: '#92400e' };

async function loadConfig() {
  if (STATIC_MODE) {
    // Static mode: hardcoded preset info, no agent selection UI
    state.agentLogos = { 'Claude Code': 'static/logos/anthropic.svg', 'Codex CLI': 'static/logos/openai.svg', 'OpenClaw': 'static/logos/openclaw.svg', 'Nanobot': 'static/logos/nanobot.svg', 'EvoScientist': 'static/logos/evo.svg' };
    const container = document.getElementById('agent-options');
    if (container) container.style.display = 'none';
    return;
  }
  try {
    const res = await fetch(`${API}/api/config`);
    const cfg = await res.json();
    state.agentLogos = cfg.agent_logos || {};
    const container = document.getElementById('agent-options');
    container.innerHTML = '';
    for (const [key, preset] of Object.entries(cfg.presets || {})) {
      const btn = document.createElement('div');
      btn.className = 'agent-option'; btn.dataset.agent = key;
      if (preset.logo) {
        btn.innerHTML = `<img class="agent-logo" src="${preset.logo}" alt="${esc(preset.icon)}">${esc(preset.label)}`;
      } else {
        const color = ICON_COLORS[preset.icon] || '#3f3f46';
        btn.innerHTML = `<span class="agent-option-icon" style="background:${color}">${esc(preset.icon)}</span>${esc(preset.label)}`;
      }
      btn.onclick = () => selectAgent(key);
      container.appendChild(btn);
    }
    const first = Object.keys(cfg.presets || {})[0];
    if (first) selectAgent(first);
  } catch (e) { console.error(e); }
}

function selectAgent(key) {
  state.selectedAgent = key;
  document.querySelectorAll('.agent-option').forEach(el => el.classList.toggle('active', el.dataset.agent === key));
}

/* ── Dashboard: Frontier Chart + Leaderboard ────────────────────────── */
let frontierChart = null;
const AGENT_COLORS = ['#3b82f6','#ef4444','#22c55e','#f59e0b','#8b5cf6','#ec4899','#14b8a6','#f97316'];

async function loadDashboard() {
  try {
    const allTasks = STATIC_MODE
      ? await fetchStaticJSON('data/tasks.json')
      : await (await fetch(`${API}/api/tasks`)).json();
    const taskList = [];
    for (const tasks of Object.values(allTasks)) taskList.push(...tasks);
    taskList.sort();

    const presetAgents = STATIC_MODE
      ? Object.keys(state.agentLogos)
      : Object.values((await (await fetch(`${API}/api/config`)).json()).presets || {}).map(p => p.label);

    const data = STATIC_MODE
      ? (await fetchStaticJSON('data/leaderboard.json')) || { tasks: [], agents: [], scores: {}, frontier: {} }
      : await (await fetch(`${API}/api/leaderboard`)).json();
    // Ensure all tasks on x-axis
    data.tasks = taskList;
    for (const t of taskList) {
      if (!(t in data.frontier)) data.frontier[t] = null;
    }
    // Ensure all preset agents appear in preset order
    const orderedAgents = [];
    for (const name of presetAgents) {
      orderedAgents.push(name);
      if (!data.scores[name]) data.scores[name] = {};
    }
    // Add any agents from data that aren't presets (at the end)
    for (const name of data.agents) {
      if (!orderedAgents.includes(name)) orderedAgents.push(name);
    }
    data.agents = orderedAgents;

    renderFrontierChart(data);
    renderLeaderboard(data);
  } catch (e) { console.error('Dashboard load failed:', e); }
}

function renderFrontierChart(data) {
  const ctx = document.getElementById('frontier-chart');
  if (!ctx) return;

  const labels = data.tasks;
  // X-axis: show only domain name (strip _NNN), but keep all 40 ticks
  const domainLabels = labels.map(t => t.replace(/_\d+$/, ''));
  const datasets = [];

  // Each agent as a line
  data.agents.forEach((agent, i) => {
    const color = AGENT_COLORS[i % AGENT_COLORS.length];
    const scores = labels.map(t => data.scores[agent]?.[t]?.score ?? null);
    datasets.push({
      label: agent,
      data: scores,
      borderColor: color,
      backgroundColor: color + '20',
      borderWidth: 1.5,
      pointRadius: 2.5,
      pointHoverRadius: 5,
      tension: 0,
      spanGaps: true,
    });
  });

  // Frontier line
  const frontierScores = labels.map(t => data.frontier[t] ?? null);
  datasets.push({
    label: 'Frontier',
    data: frontierScores,
    borderColor: 'rgba(128,128,128,0.6)',
    backgroundColor: 'rgba(128,128,128,0.06)',
    borderWidth: 2.5,
    borderDash: [6, 3],
    pointRadius: 0,
    fill: true,
    tension: 0.2,
    spanGaps: true,
  });

  // Human Level baseline at 50
  datasets.push({
    label: 'Human Level (50)',
    data: labels.map(() => 50),
    borderColor: 'rgba(239,68,68,0.45)',
    borderWidth: 1.5,
    borderDash: [4, 4],
    pointRadius: 0,
    fill: false,
    tension: 0,
    order: 100,
  });

  const style = getComputedStyle(document.documentElement);
  const textColor = style.getPropertyValue('--text-tertiary').trim() || '#a1a1aa';
  const gridColor = style.getPropertyValue('--border').trim() || 'rgba(0,0,0,0.08)';

  if (frontierChart) frontierChart.destroy();

  // Custom plugin: zone labels on Y axis left side
  const zonePlugin = {
    id: 'zoneLabels',
    afterDraw(chart) {
      const yScale = chart.scales.y;
      const area = chart.chartArea;
      const c = chart.ctx;

      // New-Discovery label (above 50, centered at y=75)
      c.save();
      c.font = '700 22px DM Sans, sans-serif';
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillStyle = 'rgba(34,197,94,0.55)';
      const newY = yScale.getPixelForValue(75);
      c.translate(area.left - 42, newY);
      c.rotate(-Math.PI / 2);
      c.fillText('New-Discovery', 0, 0);
      c.restore();

      // Re-Discovery label (below 50, centered at y=25)
      c.save();
      c.font = '700 22px DM Sans, sans-serif';
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillStyle = 'rgba(59,130,246,0.55)';
      const reY = yScale.getPixelForValue(25);
      c.translate(area.left - 42, reY);
      c.rotate(-Math.PI / 2);
      c.fillText('Re-Discovery', 0, 0);
      c.restore();
    }
  };

  // Track domains for labeling
  frontierChart = new Chart(ctx, {
    type: 'line',
    data: { labels: domainLabels, datasets },
    plugins: [zonePlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { left: 36 } },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(0,0,0,0.85)',
          titleFont: { family: "'DM Sans', sans-serif", size: 17 },
          bodyFont: { family: "'Fira Code', monospace", size: 16 },
          callbacks: {
            title: (items) => labels[items[0].dataIndex] || '',
            label: c2 => `${c2.dataset.label}: ${c2.parsed.y !== null ? c2.parsed.y.toFixed(1) : '-'}`,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: textColor,
            font: { size: 14, weight: '600' },
            maxRotation: 0,
            minRotation: 0,
            autoSkip: false,
            callback: function(value, index) {
              // Show domain name at midpoint of each 4-task group
              if (index % 4 === 1) return domainLabels[index];
              return '';
            },
          },
          grid: {
            color: gridColor,
            lineWidth: 1,
          },
        },
        y: {
          min: 0, max: 100,
          ticks: { color: textColor, font: { size: 14 }, stepSize: 10 },
          grid: { color: gridColor },
        },
      },
    },
  });

  // Custom HTML legend with logos
  const legendEl = document.getElementById('chart-legend');
  legendEl.innerHTML = frontierChart.data.datasets.map((ds, i) => {
    const logo = state.agentLogos[ds.label];
    const logoHtml = logo ? `<img src="${logo}" alt="">` : '';
    if (ds.label === 'Frontier') {
      return `<div class="chart-legend-item"><span class="chart-legend-swatch dashed" style="border-color:${ds.borderColor}"></span>${ds.label}</div>`;
    }
    if (ds.label.startsWith('Human')) {
      return `<div class="chart-legend-item"><span class="chart-legend-swatch dashed" style="border-color:${ds.borderColor}"></span>${ds.label}</div>`;
    }
    return `<div class="chart-legend-item">${logoHtml}<span class="chart-legend-swatch" style="background:${ds.borderColor}"></span>${esc(ds.label)}</div>`;
  }).join('');
}

function renderLeaderboard(data) {
  const wrap = document.getElementById('leaderboard-wrap');

  // Live clock (static mode only)
  if (STATIC_MODE) {
    function updateClock() {
      const el = document.getElementById('live-clock');
      if (el) el.textContent = new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
    }
    updateClock();
    if (!window._clockInterval) window._clockInterval = setInterval(updateClock, 1000);
  }

  function scoreColor(v) {
    if (v <= 0) return 'transparent';
    const t = Math.max(0, Math.min(100, v)) / 100;
    const hue = t < 0.5 ? t * 2 * 55 : 55 + (t - 0.5) * 2 * 65;
    return `hsl(${hue}, 75%, ${42 + t * 12}%)`;
  }
  function cellStyle(v) { return v > 0 ? `background:${scoreColor(v)};color:#fff;font-weight:600;` : ''; }

  let html = '<table class="leaderboard"><thead><tr><th>Task</th>';
  data.agents.forEach(a => html += `<th>${agentLogoHtml(a, 20)} ${esc(a)}</th>`);
  html += '<th>Frontier</th></tr></thead><tbody>';

  data.tasks.forEach(task => {
    html += `<tr><td>${esc(task)}</td>`;
    data.agents.forEach(agent => {
      const entry = data.scores[agent]?.[task];
      if (entry) {
        const s = entry.score;
        html += `<td onclick="goToRun('${entry.run_id}')" style="cursor:pointer"><span class="score-cell" style="${cellStyle(s)}">${s.toFixed(1)}</span></td>`;
      } else {
        html += STATIC_MODE
          ? '<td><span class="running-cell">running</span></td>'
          : '<td class="no-score">-</td>';
      }
    });
    const f = data.frontier[task];
    if (f && f > 0) {
      html += `<td><span class="score-cell" style="${cellStyle(f)}">${f.toFixed(1)}</span></td>`;
    } else {
      html += '<td class="no-score">-</td>';
    }
    html += '</tr>';
  });

  // Average row — only count tasks that have scores
  html += '<tr class="frontier-row"><td>Average</td>';
  data.agents.forEach(agent => {
    const scores = data.tasks.map(t => data.scores[agent]?.[t]?.score).filter(v => v != null);
    const avg = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
    html += `<td><span class="score-cell" style="${cellStyle(avg)}">${scores.length ? avg.toFixed(1) : '-'}</span></td>`;
  });
  const fScores = data.tasks.map(t => data.frontier[t]).filter(v => v != null && v > 0);
  const fAvg = fScores.length ? fScores.reduce((a, b) => a + b, 0) / fScores.length : 0;
  html += `<td><span class="score-cell" style="${cellStyle(fAvg)}">${fScores.length ? fAvg.toFixed(1) : '-'}</span></td>`;
  html += '</tr></tbody></table>';

  wrap.innerHTML = html;
}

function goToRun(runId) {
  // Extract task_id from run_id (format: TaskId_timestamp)
  const parts = runId.split('_');
  const taskId = parts.slice(0, -2).join('_'); // e.g. "Energy_000" from "Energy_000_20260318_..."
  selectTask(taskId);
  setTimeout(() => selectRun(runId), 300);
}

/* ── Tasks ───────────────────────────────────────────────────────────── */
async function loadTasks() {
  const grouped = STATIC_MODE
    ? await fetchStaticJSON('data/tasks.json')
    : await (await fetch(`${API}/api/tasks`)).json();
  state.tasks = grouped;
  const browser = document.getElementById('task-browser');
  browser.innerHTML = '';
  let totalTasks = 0, totalDomains = 0;
  for (const [domain, tasks] of Object.entries(grouped)) {
    totalDomains++; totalTasks += tasks.length;
    const group = document.createElement('div'); group.className = 'domain-group';
    const toggle = document.createElement('button'); toggle.className = 'domain-toggle';
    toggle.innerHTML = `<span class="arrow">&#9654;</span>${domain}<span class="domain-count">${tasks.length}</span>`;
    const taskList = document.createElement('div'); taskList.className = 'domain-tasks';
    toggle.onclick = () => { toggle.classList.toggle('open'); taskList.classList.toggle('open'); };
    for (const id of tasks) {
      const btn = document.createElement('button');
      btn.className = 'task-item'; btn.textContent = id; btn.dataset.taskId = id;
      btn.onclick = () => selectTask(id);
      taskList.appendChild(btn);
    }
    group.appendChild(toggle); group.appendChild(taskList); browser.appendChild(group);
  }
  document.getElementById('welcome-stats').innerHTML =
    `<div class="stat"><div class="stat-value">${totalTasks}</div><div class="stat-label">Tasks</div></div>` +
    `<div class="stat"><div class="stat-value">${totalDomains}</div><div class="stat-label">Domains</div></div>`;
}

async function selectTask(taskId) {
  // Stop any previous run's streaming/tracking
  if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
  stopAutoTrack();
  const stopBtn0 = document.getElementById('btn-stop-run');
  if (stopBtn0) stopBtn0.style.display = 'none';

  const epoch = ++state._selectEpoch;
  const stale = () => state._selectEpoch !== epoch;

  state.currentTaskId = taskId; state.currentRunId = null;
  document.querySelectorAll('.task-item').forEach(el => el.classList.toggle('active', el.dataset.taskId === taskId));
  document.getElementById('welcome-screen').style.display = 'none';
  document.getElementById('task-view').style.display = 'flex';

  // Immediately clear stale content from previous task
  state._cachedInputFiles = [];
  document.getElementById('file-tree').innerHTML = '<div class="placeholder" style="padding:8px;opacity:.6">Loading...</div>';
  document.getElementById('file-content-header').textContent = '';
  document.getElementById('file-content-body').innerHTML = '<div class="placeholder">Loading...</div>';
  document.getElementById('terminal-body').innerHTML = '<div class="placeholder">Loading...</div>';
  document.getElementById('report-content').innerHTML = '<div class="placeholder">Loading...</div>';
  document.getElementById('task-checklist-preview').innerHTML = '<div class="placeholder">Loading...</div>';
  document.getElementById('score-total-area').innerHTML = '';
  // Reset paper iframe
  const _paperIframe = document.getElementById('paper-iframe');
  if (_paperIframe) {
    _paperIframe.style.display = 'block';
    _paperIframe.src = 'about:blank';
    const _ph = _paperIframe.parentElement.querySelector('.placeholder');
    if (_ph) _ph.remove();
  }
  document.getElementById('task-domain').textContent = taskId.replace(/_\d+$/, '');
  document.getElementById('task-title').textContent = taskId;

  const info = STATIC_MODE
    ? await fetchStaticJSON(`data/tasks/${taskId}/info.json`)
    : await (await fetch(`${API}/api/tasks/${taskId}/info`)).json();
  if (stale()) return;
  document.getElementById('task-description').textContent = info?.task || '';

  try {
    const checklist = STATIC_MODE
      ? await fetchStaticJSON(`data/tasks/${taskId}/checklist.json`)
      : await (await fetch(`${API}/api/tasks/${taskId}/checklist`)).json();
    document.getElementById('score-total-area').innerHTML = '';
    const imgBase = STATIC_MODE ? `data/tasks/${taskId}/` : `${API}/api/tasks/${taskId}/target_image?path=`;
    document.getElementById('task-checklist-preview').innerHTML = checklist.map((item, i) => {
      const imgHtml = item.type === 'image' && item.path
        ? `<img class="checklist-img" src="${STATIC_MODE ? imgBase + item.path : imgBase + encodeURIComponent(item.path)}" alt="target" onclick="openImageOverlay(this.src)">`
        : '';
      return `<div class="checklist-item" data-checklist-idx="${i}">
        <div class="checklist-item-header">
          <div><span class="score-item-type ${item.type}">${item.type}</span><span class="score-item-weight">Weight: ${item.weight}</span></div>
          <div class="checklist-score-slot" id="checklist-score-${i}"></div>
        </div>
        <p class="checklist-text${item.content&&item.content.length>200?' truncated':''}">${esc(item.content||'')}</p>${item.content&&item.content.length>200?'<button class="checklist-toggle" onclick="const p=this.previousElementSibling;p.classList.toggle(\'truncated\');this.textContent=p.classList.contains(\'truncated\')?\'show more\':\'show less\'">show more</button>':''}
        ${imgHtml}
      </div>`;
    }).join('');
  } catch (e) { document.getElementById('task-checklist-preview').innerHTML = '<p class="placeholder">No checklist</p>'; }
  if (stale()) return;

  const paperIframe = document.getElementById('paper-iframe');
  if (paperIframe) {
    if (STATIC_MODE) {
      // Check if paper was exported by trying to load it
      const paperUrl = `data/tasks/${taskId}/paper.pdf`;
      const paperContainer = paperIframe.parentElement;
      try {
        const resp = await fetch(paperUrl);
        if (stale()) return;
        const ct = resp.headers.get('content-type') || '';
        if (resp.ok && ct.includes('pdf')) {
          paperIframe.style.display = 'block';
          paperIframe.src = paperUrl;
          // Remove any fallback message
          const old = paperContainer.querySelector('.placeholder');
          if (old) old.remove();
        } else {
          paperIframe.style.display = 'none';
          paperIframe.src = 'about:blank';
          const old = paperContainer.querySelector('.placeholder');
          if (old) old.remove();
          paperContainer.insertAdjacentHTML('beforeend', '<div class="placeholder" style="padding:20px">Paper PDF too large for GitHub Pages.<br><br>View on <a href="https://github.com/InternScience/ResearchClawBench" target="_blank" style="color:var(--accent)">GitHub</a></div>');
        }
      } catch (_) {
        paperIframe.style.display = 'none';
        paperIframe.src = 'about:blank';
        const old2 = paperContainer.querySelector('.placeholder');
        if (old2) old2.remove();
        paperContainer.insertAdjacentHTML('beforeend', '<div class="placeholder" style="padding:20px">Paper PDF not available.<br><br>View on <a href="https://github.com/InternScience/ResearchClawBench" target="_blank" style="color:var(--accent)">GitHub</a></div>');
      }
    } else {
      paperIframe.src = `${API}/api/tasks/${taskId}/paper`;
    }
  }
  await loadRuns(taskId);
  if (stale()) return;
  document.getElementById('terminal-body').innerHTML = '<div class="placeholder">Select a run to see agent output</div>';
  document.getElementById('terminal-status').textContent = 'Agent Output';
  document.getElementById('terminal-status').className = 'terminal-status';

  // Auto-select latest run, or show task file structure
  const runs = STATIC_MODE
    ? (state._runsIndex || []).filter(r => r.task_id === taskId).sort((a,b) => (b.timestamp||'').localeCompare(a.timestamp||''))
    : await (await fetch(`${API}/api/runs?task_id=${taskId}`)).json();
  if (stale()) return;
  if (runs.length > 0) {
    await selectRun(runs[0].run_id);
  } else {
    state.currentRunId = null;
    // Load task file tree (both modes)
    if (STATIC_MODE) {
      await loadStaticTaskFiles(taskId);
      if (stale()) return;
      const instrUrl = `data/tasks/${taskId}/workspace/INSTRUCTIONS.md`;
      renderFileContent('INSTRUCTIONS.md', 'INSTRUCTIONS.md', instrUrl, null, `data/tasks/${taskId}/workspace/`, 'INSTRUCTIONS.md');
    } else {
      await loadTaskFiles(taskId);
      if (stale()) return;
      const baseUrl = `${API}/api/tasks/${taskId}/file?path=`;
      renderFileContent('INSTRUCTIONS.md', 'INSTRUCTIONS.md', baseUrl + encodeURIComponent('INSTRUCTIONS.md'), null, baseUrl, 'INSTRUCTIONS.md');
    }
    // Clear non-file panels
    document.getElementById('terminal-body').innerHTML = '<div class="placeholder">No runs yet</div>';
    showDuration(null);
    document.getElementById('report-content').innerHTML = '<div class="placeholder">Report appears after run completes</div>';
    document.getElementById('score-total-area').innerHTML = '';
    document.querySelectorAll('.checklist-score-slot').forEach(el => el.innerHTML = '');
    document.querySelectorAll('.score-item-reasoning').forEach(el => el.remove());
    const scoreBtn = document.getElementById('btn-score');
    if (scoreBtn) scoreBtn.textContent = 'Score';
  }
  switchTab(state.lastTab);
}

/* ── Runs ────────────────────────────────────────────────────────────── */
async function loadRuns(taskId) {
  let runs;
  if (STATIC_MODE) {
    if (!state._runsIndex) state._runsIndex = await fetchStaticJSON('data/runs_index.json') || [];
    runs = state._runsIndex.filter(r => !taskId || r.task_id === taskId).sort((a,b) => (b.timestamp||'').localeCompare(a.timestamp||''));
  } else {
    runs = await (await fetch(`${API}/api/runs?task_id=${taskId||''}`)).json();
  }
  const div = document.getElementById('run-history');
  if (!runs.length) { div.innerHTML = '<p class="placeholder" style="padding:4px 8px">No runs yet</p>'; return; }
  div.innerHTML = runs.map(r => {
    const ts = r.timestamp || '';
    const fmt = ts.length >= 15 ? `${ts.slice(0,4)}-${ts.slice(4,6)}-${ts.slice(6,8)} ${ts.slice(9,11)}:${ts.slice(11,13)}` : ts;
    const modelStr = r.model ? `<span class="run-item-model">${esc(r.model)}</span>` : '';
    return `
    <div class="run-item ${r.run_id===state.currentRunId?'active':''}" data-run-id="${r.run_id}">
      <span class="status-dot ${r.status}"></span>
      <div class="run-item-info" onclick="selectRun('${r.run_id}')">
        <div class="run-item-task">${agentLogoHtml(r.agent_name, 14)} ${r.agent_name||'Agent'} ${modelStr}</div>
        <div class="run-item-time">${fmt}</div>
      </div>
      <button class="run-item-del" onclick="event.stopPropagation();deleteRun('${r.run_id}')" title="Delete run">&times;</button>
    </div>`;
  }).join('');
}

async function deleteRun(runId) {
  if (!confirm('Delete this run?')) return;
  await fetch(`${API}/api/runs/${runId}`, { method: 'DELETE' });
  if (state.currentRunId === runId) {
    state.currentRunId = null;
    document.getElementById('file-tree').innerHTML = '';
    document.getElementById('file-content-body').innerHTML = '<div class="placeholder">Select a file</div>';
    document.getElementById('terminal-body').innerHTML = '<div class="placeholder">Start a run to see AI steps...</div>';
  }
  await loadRuns(state.currentTaskId);
  loadDashboard();
}

async function selectRun(runId) {
  // Stop any ongoing streaming/tracking from previous run
  if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
  stopAutoTrack();

  state.currentRunId = runId;
  const fileTrack = document.getElementById('toggle-file-track');
  if (fileTrack) fileTrack.classList.toggle('on', !state.userSelectedFile);
  const stopBtn = document.getElementById('btn-stop-run');
  if (stopBtn) stopBtn.style.display = 'none';
  document.querySelectorAll('.run-item').forEach(el => el.classList.toggle('active', el.dataset.runId === runId));
  document.getElementById('file-content-header').textContent = 'No file selected';
  document.getElementById('file-content-body').innerHTML = '<div class="placeholder">Loading...</div>';
  document.getElementById('terminal-status').textContent = 'Agent Output';
  document.getElementById('terminal-status').className = 'terminal-status';
  document.getElementById('terminal-body').innerHTML = '<div class="placeholder">Loading...</div>';
  document.getElementById('file-tree').innerHTML = '<div class="placeholder" style="padding:8px;opacity:.6">Loading...</div>';
  // Immediately clear eval tab to avoid stale content during load
  document.getElementById('report-content').innerHTML = '<div class="placeholder">Loading...</div>';
  document.getElementById('score-total-area').innerHTML = '';
  document.querySelectorAll('.checklist-score-slot').forEach(el => el.innerHTML = '');
  document.querySelectorAll('.score-item-reasoning').forEach(el => el.remove());
  { const _s = document.getElementById('btn-score'); if (_s) _s.textContent = 'Score'; }

  const isStale = () => state.currentRunId !== runId;

  if (STATIC_MODE) {
    const runData = await fetchStaticJSON(`data/runs/${runId}/data.json`);
    if (isStale()) return;
    showDuration(runData?.duration_seconds);
    // File tree
    const files = await fetchStaticJSON(`data/runs/${runId}/files.json`);
    if (isStale()) return;
    if (files && files.length) {
      renderFileTree(files, runId, null);
      if (state.userSelectedFile) {
        // Follow off — show INSTRUCTIONS.md as default
        const instrUrl = `data/runs/${runId}/workspace/INSTRUCTIONS.md`;
        renderFileContent('INSTRUCTIONS.md', 'INSTRUCTIONS.md', instrUrl, null, `data/runs/${runId}/workspace/`, 'INSTRUCTIONS.md');
      } else {
        let best = null;
        for (const f of files) { if (f.type !== 'file' || !isViewableFile(f.name) || f.exported === false || !isAgentOutput(f.path)) continue; if (!best || (f.mtime && f.mtime > (best.mtime || 0))) best = f; }
        if (best) {
          const url = `data/runs/${runId}/workspace/${best.path}`;
          renderFileContent(best.path, best.name, url, null, `data/runs/${runId}/workspace/`, best.path);
        }
      }
    }
    // Agent output (limit to last 500 lines to prevent freeze)
    const outputLines = await fetchStaticJSON(`data/runs/${runId}/output.json`);
    if (isStale()) return;
    const termBody = document.getElementById('terminal-body');
    termBody.innerHTML = '';
    if (outputLines && outputLines.length) {
      const MAX_RENDER = 500;
      const toRender = outputLines.length > MAX_RENDER ? outputLines.slice(-MAX_RENDER) : outputLines;
      if (outputLines.length > MAX_RENDER) {
        const note = document.createElement('div');
        note.className = 'chat-bubble chat-bubble-system';
        note.textContent = `${outputLines.length - MAX_RENDER} earlier messages hidden (${outputLines.length} total)`;
        termBody.appendChild(note);
      }
      const prevFollow = state.autoFollow;
      state.autoFollow = false;
      for (const line of toRender) {
        try { appendMsg(JSON.parse(line)); }
        catch (_) { if (line.trim()) appendLine(line.trim(), ''); }
      }
      state.autoFollow = prevFollow;
      if (prevFollow) termBody.scrollTop = termBody.scrollHeight;
    } else { termBody.innerHTML = '<div class="placeholder">No agent output</div>'; }
    // Report
    if (runData && runData.report) {
      let html = marked.parse(runData.report);
      html = html.replace(/(<img[^>]*src=")([^"]+)(")/g, (m, pre, src, post) => {
        if (src.startsWith('http') || src.startsWith('/')) return m;
        let resolved = 'report/' + src;
        while (resolved.includes('../')) resolved = resolved.replace(/[^/]+\/\.\.\//g, '');
        return pre + `data/runs/${runId}/workspace/${resolved}` + post;
      });
      document.getElementById('report-content').innerHTML = html;
    } else { document.getElementById('report-content').innerHTML = '<div class="placeholder">No report</div>'; }
    // Score
    document.getElementById('score-total-area').innerHTML = '';
    document.querySelectorAll('.checklist-score-slot').forEach(el => el.innerHTML = '');
    document.querySelectorAll('.score-item-reasoning').forEach(el => el.remove());
    if (runData && runData.score && runData.score.items) renderScore(runData.score);
    switchTab(state.lastTab);
  } else {
    const meta = await (await fetch(`${API}/api/runs/${runId}/meta`)).json();
    if (isStale()) return;
    showDuration(meta.duration_seconds);
    if (meta.status === 'running') {
      // Calculate elapsed time from run start timestamp (YYYYMMDD_HHMMSS)
      let elapsed = 0;
      if (meta.timestamp) {
        const ts = meta.timestamp;
        const startDate = new Date(
          parseInt(ts.slice(0,4)), parseInt(ts.slice(4,6))-1, parseInt(ts.slice(6,8)),
          parseInt(ts.slice(9,11)), parseInt(ts.slice(11,13)), parseInt(ts.slice(13,15))
        );
        elapsed = Math.max(0, Math.floor((Date.now() - startDate.getTime()) / 1000));
      }
      startDurationTimer(elapsed);
      startStreaming(runId); switchTab(state.lastTab); loadWorkspace(runId);
    } else {
      // Load workspace + file first, then the rest in parallel
      const wsFiles = await loadWorkspace(runId);
      if (isStale()) return;
      if (state.userSelectedFile) {
        // Follow off — show INSTRUCTIONS.md as default
        const baseUrl = `${API}/api/runs/${runId}/file?path=`;
        renderFileContent('INSTRUCTIONS.md', 'INSTRUCTIONS.md', baseUrl + encodeURIComponent('INSTRUCTIONS.md'), null, baseUrl, 'INSTRUCTIONS.md');
      } else {
        autoOpenLatestFile(runId, wsFiles);
      }
      await Promise.all([loadSavedOutput(runId), loadReport(runId), loadScore(runId)]);
      if (isStale()) return;
      switchTab(state.lastTab);
    }
  }
}

async function autoOpenLatestFile(runId, files) {
  if (state.userSelectedFile || state.currentRunId !== runId) return;
  try {
    if (!files) files = await (await fetch(`${API}/api/runs/${runId}/output-files`)).json();
    let latest = null;
    for (const f of files) {
      if (f.type !== 'file' || !isViewableFile(f.name) || !isAgentOutput(f.path)) continue;
      if (!latest || (f.mtime && f.mtime > (latest.mtime || 0))) latest = f;
    }
    if (latest) loadFile(runId, latest.path, latest.name, null, true);
  } catch (_) {}
}

async function loadSavedOutput(runId) {
  const body = document.getElementById('terminal-body');
  body.innerHTML = '';
  try {
    const lines = await (await fetch(`${API}/api/runs/${runId}/output?tail=500`)).json();
    if (state.currentRunId !== runId) return;
    if (!lines.length) { body.innerHTML = '<div class="placeholder">No agent output recorded</div>'; return; }
    // Limit rendering to last 500 lines to prevent browser freeze on large outputs
    const MAX_RENDER = 500;
    const skipped = lines.length > MAX_RENDER ? lines.length - MAX_RENDER : 0;
    const toRender = skipped ? lines.slice(-MAX_RENDER) : lines;
    if (skipped) {
      const note = document.createElement('div');
      note.className = 'chat-bubble chat-bubble-system';
      note.textContent = `${skipped} earlier messages hidden (${lines.length} total)`;
      body.appendChild(note);
    }
    // Batch DOM: disable auto-scroll during bulk insert
    const prevFollow = state.autoFollow;
    state.autoFollow = false;
    for (const line of toRender) {
      try { appendMsg(JSON.parse(line)); } catch (_) { appendLine(line, ''); }
    }
    state.autoFollow = prevFollow;
    if (prevFollow) body.scrollTop = body.scrollHeight;
  } catch (_) { body.innerHTML = '<div class="placeholder">Failed to load output</div>'; }
}

/* ── Start Run ───────────────────────────────────────────────────────── */
async function startRun() {
  if (!state.currentTaskId || !state.selectedAgent) return;
  const body = { task_id: state.currentTaskId, agent: state.selectedAgent };
  const btn = document.getElementById('btn-start-run');
  btn.disabled = true; btn.innerHTML = '<span class="btn-icon">&#9654;</span> Starting...';
  try {
    const data = await (await fetch(`${API}/api/runs`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) })).json();
    state.currentRunId = data.run_id;
    startDurationTimer();
    switchTab('research'); startStreaming(data.run_id); await loadRuns(state.currentTaskId);
  } catch (e) { alert('Failed: ' + e.message); }
  finally { btn.disabled = false; btn.innerHTML = '<span class="btn-icon">&#9654;</span> Start Run'; }
}

/* ── SSE Streaming ───────────────────────────────────────────────────── */
function startStreaming(runId) {
  if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
  state.userSelectedFile = false;
  const body = document.getElementById('terminal-body'); body.innerHTML = '';
  const st = document.getElementById('terminal-status'); st.textContent = 'Agent Output'; st.className = 'terminal-status running';
  { const _el = document.getElementById('btn-stop-run'); if (_el) _el.style.display = 'inline-flex'; }

  // Start auto-tracking latest code file
  startAutoTrack(runId);

  const es = new EventSource(`${API}/api/runs/${runId}/stream`);
  state.eventSource = es;
  es.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data); appendMsg(d);
      if (d.type === 'system' && d.subtype === 'done') {
        es.close(); state.eventSource = null;
        stopAutoTrack(); onStreamEnd(d.status, runId);
      }
    } catch (_) { appendLine(e.data, 'msg-text'); }
  };
  es.onerror = () => { es.close(); state.eventSource = null; stopAutoTrack(); onStreamEnd('disconnected', runId); };
}

function onStreamEnd(status, runId) {
  const st = document.getElementById('terminal-status');
  st.textContent = 'Agent Output';
  st.className = 'terminal-status completed';
  stopDurationTimer();
  // Keep the timer's final value displayed; it will be overwritten by meta.duration_seconds when available
  { const _el = document.getElementById('btn-stop-run'); if (_el) _el.style.display = 'none'; }
  // Wait for _meta.json to be written, then refresh
  setTimeout(async () => {
    if (state.currentRunId !== runId) return;
    loadRuns(state.currentTaskId);
    loadWorkspace(runId);
    loadReport(runId);
    // Update duration from server (accurate)
    try {
      const meta = await (await fetch(`${API}/api/runs/${runId}/meta`)).json();
      if (meta.duration_seconds != null) showDuration(meta.duration_seconds);
    } catch (_) {}
  }, 1500);
}

async function stopRun() {
  if (!state.currentRunId) return;
  { const _el = document.getElementById('btn-stop-run'); if (_el) _el.textContent = 'Stopping...'; }
  try {
    await fetch(`${API}/api/runs/${state.currentRunId}/stop`, { method: 'POST' });
  } catch (_) {}
}

function startAutoTrack(runId) {
  stopAutoTrack();
  let lastFileCount = 0;
  state.autoTrackTimer = setInterval(async () => {
    if (state.currentRunId !== runId) { stopAutoTrack(); return; }
    try {
      // Only fetch output files (lightweight), merge with cached input files for tree
      const outFiles = await (await fetch(`${API}/api/runs/${runId}/output-files`)).json();
      const allFiles = _sortFlatTree([...(state._cachedInputFiles || []), ...outFiles]);

      // Refresh file tree if file count changed
      if (allFiles.length !== lastFileCount) {
        lastFileCount = allFiles.length;
        renderFileTree(allFiles, runId, null);
      }

      // Auto-show most recently modified viewable file (from output files only)
      if (!state.userSelectedFile) {
        let latest = null;
        for (const f of outFiles) {
          if (f.type !== 'file' || !isViewableFile(f.name)) continue;
          if (!latest || (f.mtime && f.mtime > (latest.mtime || 0))) latest = f;
        }
        if (latest) {
          loadFile(runId, latest.path, latest.name, null, true);
        }
      }
    } catch (_) {}
  }, 3000);
}

function stopAutoTrack() {
  if (state.autoTrackTimer) { clearInterval(state.autoTrackTimer); state.autoTrackTimer = null; }
}

function appendMsg(d) {
  const body = document.getElementById('terminal-body');

  function addBubble(cls, label, content) {
    const el = document.createElement('div');
    el.className = `chat-bubble ${cls}`;
    el.innerHTML = (label ? `<span class="chat-label">${label}</span>` : '') + content;
    body.appendChild(el);
    if (state.autoFollow) body.scrollTop = body.scrollHeight;
  }

  // Helper: extract readable text from tool_use input
  function formatToolInput(name, input) {
    if (!input) return '';
    // Common tool patterns
    if (input.command) return esc(input.command);
    if (input.content) return esc(typeof input.content === 'string' ? input.content.substring(0, 300) : JSON.stringify(input.content).substring(0, 300));
    if (input.file_path) return esc(input.file_path) + (input.old_string ? '\n...' : '');
    if (input.pattern) return esc(input.pattern);
    if (input.query) return esc(input.query);
    if (input.url) return esc(input.url);
    // Fallback: show key=value pairs
    const pairs = Object.entries(input).map(([k, v]) => {
      const val = typeof v === 'string' ? v : JSON.stringify(v);
      return `${k}: ${val.substring(0, 100)}`;
    });
    return esc(pairs.join('\n').substring(0, 300));
  }

  // Helper: clean text — strip line number prefixes (e.g. "     1→"), JSON wrappers, etc.
  function cleanText(s) {
    if (!s) return '';
    // Strip line number prefixes like "     1→" or "   12→"
    s = s.replace(/^ *\d+→/gm, '');
    // Trim leading/trailing whitespace
    return s.trim();
  }

  // Helper: extract readable text from tool_result content
  function formatToolResult(content) {
    if (typeof content === 'string') return esc(cleanText(content).substring(0, 400));
    if (Array.isArray(content)) {
      return content.map(c => {
        if (c.type === 'tool_result' && c.content) return esc(cleanText(String(c.content)).substring(0, 300));
        if (c.type === 'text') return esc(cleanText(c.text || '').substring(0, 300));
        return esc(cleanText(JSON.stringify(c)).substring(0, 200));
      }).join('\n');
    }
    return esc(JSON.stringify(content).substring(0, 400));
  }

  const t = d.type || 'text';

  // -- Claude Code stream-json wraps messages: {"type":"assistant","message":{"role":"assistant","content":[...]}}
  // Unwrap to the inner message if present
  const msg = d.message || d;
  const role = msg.role || '';

  // -- Role-based messages --
  if (role === 'assistant' && msg.content) {
    const parts = Array.isArray(msg.content) ? msg.content : [msg.content];
    for (const part of parts) {
      if (typeof part === 'string') {
        if (part.trim()) addBubble('chat-bubble-ai', '', esc(part));
      } else if (part.type === 'text' && part.text?.trim()) {
        addBubble('chat-bubble-ai', '', esc(part.text));
      } else if (part.type === 'tool_use') {
        addBubble('chat-bubble-tool', esc(part.name || 'tool'), formatToolInput(part.name, part.input));
      }
    }
    return;
  }
  if (role === 'user' && msg.content) {
    // Tool results — only show errors
    const parts = Array.isArray(msg.content) ? msg.content : [{ content: msg.content }];
    for (const part of parts) {
      if (part.type === 'tool_result' && part.is_error) {
        const raw = cleanText(String(part.content || ''));
        if (raw.length > 5) addBubble('chat-bubble-error', 'error', esc(raw.substring(0, 400)));
      }
    }
    return;
  }

  // -- Type-based messages --
  if (t === 'assistant' && d.message) {
    const c = typeof d.message === 'string' ? d.message : (d.message.content || JSON.stringify(d.message));
    if (typeof c === 'string') {
      if (c.trim()) addBubble('chat-bubble-ai', '', esc(c));
    } else if (Array.isArray(c)) {
      const txt = c.filter(x => x.type === 'text').map(x => x.text).join('\n');
      if (txt.trim()) addBubble('chat-bubble-ai', '', esc(txt));
      c.filter(x => x.type === 'tool_use').forEach(tool => {
        addBubble('chat-bubble-tool', esc(tool.name || 'tool'), formatToolInput(tool.name, tool.input));
      });
    }
  } else if (t === 'tool_use') {
    addBubble('chat-bubble-tool', esc(d.name || d.tool || 'tool'), formatToolInput(d.name, d.input || d.args));
  } else if (t === 'tool_result') {
    const rt = formatToolResult(d.content || d.output || '');
    if (rt.trim()) addBubble('chat-bubble-result', 'result', rt);
  } else if (t === 'result') {
    const rt = typeof d.result === 'string' ? d.result : JSON.stringify(d.result || '').substring(0, 1000);
    addBubble('chat-bubble-ai', 'output', esc(rt));
  } else if (t === 'error') {
    addBubble('chat-bubble-error', 'error', esc(d.error || d.message || JSON.stringify(d)));
  } else if (t === 'system') {
    if (d.subtype === 'done') addBubble('chat-bubble-system', '', `Run ${d.status}`);
    // Skip all other system messages (init, etc.)
  } else if (t === 'user') {
    // Already handled above via role check; skip any remaining
  } else {
    // Fallback for unknown types — only show if there's plain text content
    if (typeof d.content === 'string' && d.content.trim()) {
      addBubble('chat-bubble-ai', '', esc(d.content.substring(0, 500)));
    }
  }

  if (state.autoFollow) body.scrollTop = body.scrollHeight;
}

function appendLine(text, cls) {
  if (!text || !text.trim()) return;
  // Skip raw JSON objects that failed to parse meaningfully
  const t = text.trim();
  if (t.startsWith('{') && t.endsWith('}')) return;
  if (t.startsWith('[') && t.endsWith(']')) return;
  const body = document.getElementById('terminal-body');
  const el = document.createElement('div');
  el.className = 'chat-bubble chat-bubble-ai';
  el.textContent = text;
  body.appendChild(el);
  if (state.autoFollow) body.scrollTop = body.scrollHeight;
}

/* ── Workspace ───────────────────────────────────────────────────────── */
async function loadTaskFiles(taskId) {
  try {
    const files = await (await fetch(`${API}/api/tasks/${taskId}/files`)).json();
    if (state.currentTaskId !== taskId) return;
    renderFileTree(files, null, taskId);
  } catch (e) { console.error(e); }
}

async function loadStaticTaskFiles(taskId) {
  try {
    const files = await fetchStaticJSON(`data/tasks/${taskId}/files.json`);
    if (state.currentTaskId !== taskId) return;
    if (files && files.length) {
      renderStaticTaskFileTree(files, taskId);
    } else {
      document.getElementById('file-tree').innerHTML = '<div class="placeholder" style="padding:8px">No files</div>';
    }
  } catch (e) { console.error(e); }
}

function renderStaticTaskFileTree(files, taskId) {
  const tree = document.getElementById('file-tree'); tree.innerHTML = '';
  const dirMap = {};
  for (const f of files) {
    if (f.type === 'truncated') {
      const depth = (f.path.match(/\//g) || []).length;
      const item = document.createElement('div');
      item.className = 'file-tree-item tree-truncated';
      item.style.paddingLeft = `${6 + depth * 14}px`;
      item.innerHTML = `<span class="file-tree-icon" style="visibility:hidden">.</span><span class="tree-truncated-text">${esc(f.name)}</span>`;
      const pp = f.path.includes('/') ? f.path.substring(0, f.path.lastIndexOf('/')) : null;
      (pp && dirMap[pp] || tree).appendChild(item);
      continue;
    }
    const depth = (f.path.match(/\//g) || []).length;
    const item = document.createElement('div');
    item.className = `file-tree-item ${f.type==='directory'?'dir':''}`;
    item.style.paddingLeft = `${6 + depth * 14}px`;
    if (f.type === 'directory') {
      const truncLabel = f.truncated ? ' <span class="tree-truncated-badge">…</span>' : '';
      item.innerHTML = `<span class="file-tree-icon folder-arrow">&#9660;</span><span class="file-tree-icon">&#128193;</span>${esc(f.name)}${truncLabel}`;
      item.dataset.path = f.path; item.dataset.open = 'true';
      const child = document.createElement('div'); child.className = 'file-tree-children';
      item.onclick = (e) => { e.stopPropagation(); const open = item.dataset.open === 'true'; item.dataset.open = open ? 'false' : 'true'; child.style.display = open ? 'none' : 'block'; item.querySelector('.folder-arrow').innerHTML = open ? '&#9654;' : '&#9660;'; };
      dirMap[f.path] = child;
      const pp = f.path.includes('/') ? f.path.substring(0, f.path.lastIndexOf('/')) : null;
      (pp && dirMap[pp] || tree).appendChild(item); (pp && dirMap[pp] || tree).appendChild(child);
    } else {
      const icon = fileIcon(f.name);
      item.innerHTML = `<span class="file-tree-icon" style="visibility:hidden">.</span><span class="file-tree-icon">${icon}</span>${esc(f.name)}`;
      if (isViewableFile(f.name)) {
        item.onclick = (e) => {
          e.stopPropagation();
          document.querySelectorAll('.file-tree-item').forEach(el => el.classList.remove('active'));
          item.classList.add('active');
          if (f.exported === false) {
            document.getElementById('file-content-header').textContent = f.path;
            document.getElementById('file-content-body').innerHTML = '<div class="placeholder">File too large to preview.<br><br>View source on <a href="https://github.com/InternScience/ResearchClawBench" target="_blank" style="color:var(--accent)">GitHub</a></div>';
          } else {
            const url = `data/tasks/${taskId}/workspace/${f.path}`;
            renderFileContent(f.path, f.name, url, null, `data/tasks/${taskId}/workspace/`, f.path);
          }
        };
      } else {
        item.onclick = (e) => {
          e.stopPropagation();
          document.querySelectorAll('.file-tree-item').forEach(el => el.classList.remove('active'));
          item.classList.add('active');
          document.getElementById('file-content-header').textContent = f.path;
          document.getElementById('file-content-body').innerHTML = '<div class="placeholder">Binary file — cannot preview.<br><br>View source on <a href="https://github.com/InternScience/ResearchClawBench" target="_blank" style="color:var(--accent)">GitHub</a></div>';
        };
      }
      const pp = f.path.includes('/') ? f.path.substring(0, f.path.lastIndexOf('/')) : null;
      (pp && dirMap[pp] || tree).appendChild(item);
    }
  }
  // Apply current expand/collapse state
  if (!treeExpanded) {
    tree.querySelectorAll('.file-tree-item.dir').forEach(item => {
      item.dataset.open = 'false';
      item.querySelector('.folder-arrow').innerHTML = '&#9654;';
    });
    tree.querySelectorAll('.file-tree-children').forEach(c => { c.style.display = 'none'; });
  }
}

async function loadWorkspace(runId) {
  try {
    const [inputFiles, outputFiles] = await Promise.all([
      (await fetch(`${API}/api/runs/${runId}/input-files`)).json(),
      (await fetch(`${API}/api/runs/${runId}/output-files`)).json(),
    ]);
    if (state.currentRunId !== runId) return [];
    state._cachedInputFiles = inputFiles;
    const allFiles = _sortFlatTree([...inputFiles, ...outputFiles]);
    renderFileTree(allFiles, runId, null);
    return outputFiles;
  } catch (e) { console.error(e); return []; }
}

function _sortFlatTree(items) {
  // Group by top-level directory, sort groups alphabetically, root files last
  const groups = {};
  const rootFiles = [];
  for (const item of items) {
    const top = item.path.split('/')[0];
    if (!item.path.includes('/') && item.type === 'file') { rootFiles.push(item); continue; }
    if (!groups[top]) groups[top] = [];
    groups[top].push(item);
  }
  const sorted = [];
  for (const key of Object.keys(groups).sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()))) {
    sorted.push(...groups[key]);
  }
  sorted.push(...rootFiles);
  return sorted;
}

function renderFileTree(files, runId, taskId) {
  // runId = for workspace files, taskId = for task files (no run)
  const tree = document.getElementById('file-tree'); tree.innerHTML = '';
  const dirMap = {};
  for (const f of files) {
    if (f.type === 'truncated') {
      const depth = (f.path.match(/\//g) || []).length;
      const item = document.createElement('div');
      item.className = 'file-tree-item tree-truncated';
      item.style.paddingLeft = `${6 + depth * 14}px`;
      item.innerHTML = `<span class="file-tree-icon" style="visibility:hidden">.</span><span class="tree-truncated-text">${esc(f.name)}</span>`;
      const pp = f.path.includes('/') ? f.path.substring(0, f.path.lastIndexOf('/')) : null;
      (pp && dirMap[pp] || tree).appendChild(item);
      continue;
    }
    const depth = (f.path.match(/\//g) || []).length;
    const item = document.createElement('div');
    item.className = `file-tree-item ${f.type==='directory'?'dir':''}`;
    item.style.paddingLeft = `${6 + depth * 14}px`;
    if (f.type === 'directory') {
      const truncLabel = f.truncated ? ' <span class="tree-truncated-badge">…</span>' : '';
      item.innerHTML = `<span class="file-tree-icon folder-arrow">&#9660;</span><span class="file-tree-icon">&#128193;</span>${esc(f.name)}${truncLabel}`;
      item.dataset.path = f.path; item.dataset.open = 'true';
      const child = document.createElement('div'); child.className = 'file-tree-children';
      item.onclick = (e) => {
        e.stopPropagation();
        const open = item.dataset.open === 'true'; item.dataset.open = open ? 'false' : 'true';
        child.style.display = open ? 'none' : 'block';
        item.querySelector('.folder-arrow').innerHTML = open ? '&#9654;' : '&#9660;';
      };
      dirMap[f.path] = child;
      const pp = f.path.includes('/') ? f.path.substring(0, f.path.lastIndexOf('/')) : null;
      (pp && dirMap[pp] || tree).appendChild(item);
      (pp && dirMap[pp] || tree).appendChild(child);
    } else {
      item.innerHTML = `<span class="file-tree-icon" style="visibility:hidden">.</span><span class="file-tree-icon">${fileIcon(f.name)}</span>${esc(f.name)}`;
      item.onclick = (e) => {
        e.stopPropagation();
        if (STATIC_MODE && runId) {
          document.querySelectorAll('.file-tree-item').forEach(el => el.classList.remove('active'));
          item.classList.add('active');
          if (f.exported === false) {
            document.getElementById('file-content-header').textContent = f.path;
            const reason = isViewableFile(f.name) ? 'File too large to preview.' : 'Binary file — cannot preview.';
            document.getElementById('file-content-body').innerHTML = `<div class="placeholder">${reason}<br><br>View source on <a href="https://github.com/InternScience/ResearchClawBench" target="_blank" style="color:var(--accent)">GitHub</a></div>`;
          } else if (f.shared) {
            // Input file shared across all runs — load from task workspace
            const url = `data/tasks/${state.currentTaskId}/workspace/${f.path}`;
            renderFileContent(f.path, f.name, url, null, `data/tasks/${state.currentTaskId}/workspace/`, f.path);
          } else {
            const url = `data/runs/${runId}/workspace/${f.path}`;
            renderFileContent(f.path, f.name, url, null, `data/runs/${runId}/workspace/`, f.path);
          }
        } else if (runId) {
          loadFile(runId, f.path, f.name, e);
        } else if (taskId) {
          loadTaskFile(taskId, f.path, f.name, e);
        }
      };
      const pp = f.path.includes('/') ? f.path.substring(0, f.path.lastIndexOf('/')) : null;
      (pp && dirMap[pp] || tree).appendChild(item);
    }
  }
  // Apply current expand/collapse state
  if (!treeExpanded) {
    tree.querySelectorAll('.file-tree-item.dir').forEach(item => {
      item.dataset.open = 'false';
      item.querySelector('.folder-arrow').innerHTML = '&#9654;';
    });
    tree.querySelectorAll('.file-tree-children').forEach(c => { c.style.display = 'none'; });
  }
}

async function loadTaskFile(taskId, path, name, evt) {
  state.userSelectedFile = true;
  document.getElementById('toggle-file-track').classList.remove('on');
  const baseUrl = `${API}/api/tasks/${taskId}/file?path=`;
  const url = baseUrl + encodeURIComponent(path);
  renderFileContent(path, name, url, evt, baseUrl, path);
}

async function loadFile(runId, path, name, evt, isAutoTrack) {
  if (!isAutoTrack) {
    state.userSelectedFile = true;
    document.getElementById('toggle-file-track').classList.remove('on');
  }
  const baseUrl = `${API}/api/runs/${runId}/file?path=`;
  const url = baseUrl + encodeURIComponent(path);
  renderFileContent(path, name, url, evt, baseUrl, path);
}

let _renderFileEpoch = 0;
async function renderFileContent(path, name, url, evt, baseUrl, filePath) {
  const myEpoch = ++_renderFileEpoch;
  const staleFile = () => _renderFileEpoch !== myEpoch;
  document.querySelectorAll('.file-tree-item').forEach(el => el.classList.remove('active'));
  if (evt && evt.currentTarget) evt.currentTarget.classList.add('active');
  document.getElementById('file-content-header').textContent = path;
  const div = document.getElementById('file-content-body');
  const ext = name.split('.').pop().toLowerCase();

  const GITHUB_REPO = 'https://github.com/InternScience/ResearchClawBench';

  if (!isViewableFile(name)) {
    if (STATIC_MODE) {
      div.innerHTML = `<div class="placeholder">This file cannot be previewed due to GitHub Pages limitations.<br><br>View source file on <a href="${GITHUB_REPO}" target="_blank" style="color:var(--accent)">GitHub</a></div>`;
    } else {
      div.innerHTML = `<div class="placeholder">Binary file — cannot preview: ${esc(name)}</div>`;
    }
    return;
  }

  const GITHUB_FALLBACK = '<div class="placeholder">This file is too large for GitHub Pages.<br><br>View source on <a href="https://github.com/InternScience/ResearchClawBench" target="_blank" style="color:var(--accent)">GitHub</a></div>';

  if (VIEWABLE_IMG_EXTS.has(ext)) {
    if (STATIC_MODE) {
      const img = new Image();
      img.src = url;
      img.style.maxWidth = '100%';
      img.style.borderRadius = '6px';
      img.onerror = () => { div.innerHTML = GITHUB_FALLBACK; };
      div.innerHTML = '';
      div.appendChild(img);
    } else {
      div.innerHTML = `<img src="${url}">`;
    }
  } else if (VIEWABLE_EMBED_EXTS.has(ext)) {
    div.innerHTML = `<iframe src="${url}" style="width:100%;height:100%;border:none"></iframe>`;
  } else if (VIEWABLE_TABLE_EXTS.has(ext)) {
    if (STATIC_MODE) { div.innerHTML = '<div class="placeholder">Excel preview not available in static mode</div>'; return; }
    try {
      const res = await fetch(url.replace('/file?', '/xlsx_preview?'));
      if (staleFile()) return;
      const data = await res.json();
      if (data.error) { div.innerHTML = `<div class="placeholder">${esc(data.error)}</div>`; return; }
      let html = '<div style="overflow:auto"><table class="xlsx-table"><thead><tr>';
      if (data.rows && data.rows.length) {
        data.rows[0].forEach((_, i) => html += `<th>${i}</th>`);
        html += '</tr></thead><tbody>';
        data.rows.forEach(row => {
          html += '<tr>' + row.map(c => `<td>${esc(String(c ?? ''))}</td>`).join('') + '</tr>';
        });
        html += '</tbody></table></div>';
        html += `<div style="margin-top:8px;font-size:11px;color:var(--text-tertiary)">${data.rows.length} rows × ${data.rows[0].length} cols</div>`;
      } else {
        html = '<div class="placeholder">Empty spreadsheet</div>';
      }
      div.innerHTML = html;
    } catch (_) { div.innerHTML = STATIC_MODE ? '<div class="placeholder">File not available due to GitHub Pages size limits.<br><br>View source on <a href="https://github.com/InternScience/ResearchClawBench" target="_blank" style="color:var(--accent)">GitHub</a></div>' : '<div class="placeholder">Failed to load file</div>'; }
  } else if (VIEWABLE_CSV_EXTS.has(ext)) {
    try {
      const raw = await (await fetch(url)).text();
      if (staleFile()) return;
      const text = raw.length > 500000 ? raw.substring(0, 500000) : raw;
      // Detect delimiter
      const firstLine = text.split('\n')[0] || '';
      const delim = ext === 'tsv' ? '\t' : firstLine.includes('\t') ? '\t' : firstLine.includes(',') ? ',' : /\s{2,}/.test(firstLine) ? /\s{2,}/ : ',';
      const lines = text.split('\n').filter(l => l.trim()).slice(0, 200);
      const rows = lines.map(l => l.split(delim));
      let html = '<div style="overflow:auto"><table class="xlsx-table"><thead><tr>';
      rows[0].forEach((c, i) => html += `<th>${esc(c.trim())}</th>`);
      html += '</tr></thead><tbody>';
      rows.slice(1).forEach(row => {
        html += '<tr>' + row.map(c => `<td>${esc(c.trim())}</td>`).join('') + '</tr>';
      });
      html += '</tbody></table></div>';
      html += `<div style="margin-top:8px;font-size: 13.5px;color:var(--text-tertiary)">${lines.length} rows × ${rows[0].length} cols${raw.length > 500000 ? ' (truncated)' : ''}</div>`;
      div.innerHTML = html;
    } catch (_) { div.innerHTML = STATIC_MODE ? '<div class="placeholder">File not available due to GitHub Pages size limits.<br><br>View source on <a href="https://github.com/InternScience/ResearchClawBench" target="_blank" style="color:var(--accent)">GitHub</a></div>' : '<div class="placeholder">Failed to load file</div>'; }
  } else {
    try {
      let text = await (await fetch(url)).text();
      if (staleFile()) return;
      const MAX_PREVIEW_SIZE = 500000; // 500KB
      let truncated = false;
      if (text.length > MAX_PREVIEW_SIZE) {
        const lines = text.substring(0, MAX_PREVIEW_SIZE).split('\n');
        text = lines.join('\n');
        truncated = true;
      }
      const truncNote = truncated ? `<div style="padding:8px;font-size:12px;color:var(--text-tertiary);border-top:1px solid var(--border)">File truncated (too large to preview in full)</div>` : '';
      if (ext === 'md') {
        // Render markdown with image URL rewriting
        let html = marked.parse(text);
        // Rewrite relative image src to API URLs
        // filePath e.g. "report/report.md" -> dir = "report/"
        const fileDir = filePath.includes('/') ? filePath.substring(0, filePath.lastIndexOf('/') + 1) : '';
        html = html.replace(/(<img\s[^>]*src=")([^"]+)(")/g, (match, pre, src, post) => {
          if (src.startsWith('http') || src.startsWith('/')) return match;
          // Resolve relative path: ../outputs/fig.png from report/ -> outputs/fig.png
          let resolved = fileDir + src;
          // Simplify ../ references
          while (resolved.includes('../')) {
            resolved = resolved.replace(/[^/]+\/\.\.\//g, '');
          }
          return pre + baseUrl + encodeURIComponent(resolved) + post;
        });
        div.innerHTML = `<div class="file-md-render">${html}</div>${truncNote}`;
      } else {
        const langMap = {py:'python',js:'javascript',json:'json',sh:'bash',yml:'yaml',yaml:'yaml',txt:null,csv:null,mat:null};
        const lang = langMap[ext];
        if (lang && typeof hljs !== 'undefined') {
          div.innerHTML = `<pre class="file-code-block"><code>${hljs.highlight(text,{language:lang,ignoreIllegals:true}).value}</code></pre>${truncNote}`;
        } else {
          div.innerHTML = `<pre class="file-code-block"><code>${esc(text)}</code></pre>${truncNote}`;
        }
      }
    } catch (_) { div.innerHTML = STATIC_MODE ? '<div class="placeholder">File not available due to GitHub Pages size limits.<br><br>View source on <a href="https://github.com/InternScience/ResearchClawBench" target="_blank" style="color:var(--accent)">GitHub</a></div>' : '<div class="placeholder">Failed to load file</div>'; }
  }
}

// File types that can be displayed in the viewer
const VIEWABLE_TEXT_EXTS = new Set(['txt','md','py','js','json','jsonl','yml','yaml','sh','bash','r','R','html','css','xml','ini','cfg','conf','toml','log','tex','bib','sql','c','cpp','h','java','go','rs','jl','m','ipynb']);
const VIEWABLE_IMG_EXTS = new Set(['png','jpg','jpeg','gif','bmp','webp','svg']);
const VIEWABLE_EMBED_EXTS = new Set(['pdf']);
const VIEWABLE_TABLE_EXTS = new Set(['xlsx','xls']);
const VIEWABLE_CSV_EXTS = new Set(['csv','tsv','dat']);

function isAgentOutput(path) {
  // Only consider files that could be agent-generated for auto-open/follow
  // Skip data/ and related_work/ (input files the agent doesn't modify)
  return !path.startsWith('data/') && !path.startsWith('related_work/');
}

function isViewableFile(name) {
  const ext = name.split('.').pop().toLowerCase();
  return VIEWABLE_TEXT_EXTS.has(ext) || VIEWABLE_IMG_EXTS.has(ext) || VIEWABLE_EMBED_EXTS.has(ext) || VIEWABLE_TABLE_EXTS.has(ext) || VIEWABLE_CSV_EXTS.has(ext);
}

function fileIcon(n) {
  const ext = n.split('.').pop().toLowerCase();
  const m = {py:'&#128013;',js:'&#9881;',json:'{}',md:'&#9998;',png:'&#128444;',jpg:'&#128444;',jpeg:'&#128444;',gif:'&#128444;',csv:'&#128202;',xlsx:'&#128202;',pdf:'&#128214;',mat:'&#128202;',npy:'&#128202;',txt:'&#128196;'};
  return m[ext] || '&#128196;';
}

/* ── Report ──────────────────────────────────────────────────────────── */
async function loadReport(runId) {
  try {
    const res = await fetch(`${API}/api/runs/${runId}/report`);
    if (state.currentRunId !== runId) return;
    if (!res.ok) { document.getElementById('report-content').innerHTML = '<div class="placeholder">No report generated yet</div>'; return; }
    const data = await res.json();
    marked.setOptions({
      highlight: (code, lang) => lang && hljs.getLanguage(lang) ? hljs.highlight(code,{language:lang}).value : hljs.highlightAuto(code).value,
      breaks: true,
    });
    document.getElementById('report-content').innerHTML = marked.parse(data.markdown || '');
  } catch (_) { document.getElementById('report-content').innerHTML = '<div class="placeholder">No report</div>'; }
}

/* ── Scoring ─────────────────────────────────────────────────────────── */
async function triggerScoring() {
  if (!state.currentRunId) return;
  const btn = document.getElementById('btn-score'); btn.disabled = true; btn.textContent = 'Scoring...';
  document.getElementById('score-total-area').innerHTML = '<p class="placeholder">Scoring in progress...</p>';
  try {
    await fetch(`${API}/api/runs/${state.currentRunId}/score`, { method: 'POST' });
    const runId = state.currentRunId;
    let pollCount = 0;
    const poll = setInterval(async () => {
      pollCount++;
      if (pollCount > 100) {
        clearInterval(poll);
        document.getElementById('score-total-area').innerHTML = `<p style="color:var(--err)">Scoring timed out</p>`;
        btn.disabled = false; btn.textContent = 'Score';
        return;
      }
      try {
        const res = await fetch(`${API}/api/runs/${runId}/score`);
        if (res.ok) {
          const s = await res.json();
          if (s.items) {
            clearInterval(poll);
            renderScore(s);
            btn.disabled = false; btn.textContent = 'Re-Score';
            loadDashboard(); // Refresh leaderboard
          } else if (s.error) {
            clearInterval(poll);
            document.getElementById('score-total-area').innerHTML = `<p style="color:var(--err)">Scoring failed: ${esc(s.error)}</p>`;
            btn.disabled = false; btn.textContent = 'Score';
          }
        }
      } catch (_) {}
    }, 3000);
  } catch (e) {
    document.getElementById('score-total-area').innerHTML = `<p style="color:var(--err)">Failed: ${esc(e.message)}</p>`;
    btn.disabled = false; btn.textContent = 'Score';
  }
}

async function loadScore(runId) {
  // Clear previous scores
  document.getElementById('score-total-area').innerHTML = '';
  document.querySelectorAll('.checklist-score-slot').forEach(el => el.innerHTML = '');
  document.querySelectorAll('.score-item-reasoning').forEach(el => el.remove());
  { const _s = document.getElementById('btn-score'); if (_s) _s.textContent = 'Score'; }
  try {
    const res = await fetch(`${API}/api/runs/${runId}/score`);
    if (state.currentRunId !== runId) return;
    if (res.ok) { const s = await res.json(); if (s.items) { renderScore(s); const _s2 = document.getElementById('btn-score'); if (_s2) _s2.textContent = 'Re-Score'; } }
  } catch (_) {}
}

function renderScore(s) {
  function ringSvg(score, size = 32) {
    const r = (size - 5) / 2, circ = 2 * Math.PI * r;
    const pct = Math.max(0, Math.min(100, score)) / 100;
    const offset = circ * (1 - pct);
    const color = score >= 50 ? '#22c55e' : score >= 25 ? '#eab308' : '#ef4444';
    return `<svg class="score-ring" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
      <circle class="score-ring-bg" cx="${size/2}" cy="${size/2}" r="${r}"/>
      <circle class="score-ring-fill" cx="${size/2}" cy="${size/2}" r="${r}" stroke="${color}" stroke-dasharray="${circ}" stroke-dashoffset="${offset}"/>
    </svg>`;
  }

  // Show total score at the top
  document.getElementById('score-total-area').innerHTML = `
    <div class="score-total-bar">
      <div class="score-total-value">${s.total_score}</div>
      <div class="score-total-label">Total Score &middot; 50 = matches paper</div>
    </div>`;

  // Inject per-item scores into existing checklist items
  for (const item of s.items) {
    const slot = document.getElementById(`checklist-score-${item.index}`);
    if (slot) {
      slot.innerHTML = `<div class="score-ring-wrap">${ringSvg(item.score)}<span class="score-ring-value">${item.score}</span></div>`;
    }
    // Add reasoning below the checklist item content
    const el = document.querySelector(`.checklist-item[data-checklist-idx="${item.index}"]`);
    if (el && item.reasoning) {
      // Remove old reasoning if re-scoring
      const old = el.querySelector('.score-item-reasoning');
      if (old) old.remove();
      const reasonEl = document.createElement('div');
      reasonEl.className = 'score-item-reasoning';
      reasonEl.textContent = item.reasoning;
      el.appendChild(reasonEl);
    }
  }
}

/* ── Tabs / Buttons ──────────────────────────────────────────────────── */
function setupTabs() { document.querySelectorAll('.tab').forEach(tab => tab.onclick = () => switchTab(tab.dataset.tab)); }

function switchTab(name) {
  state.lastTab = name;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `panel-${name}`));
  if (!STATIC_MODE) {
    if (name === 'research' && state.currentRunId) loadWorkspace(state.currentRunId);
    if (name === 'eval' && state.currentRunId) { loadReport(state.currentRunId); loadScore(state.currentRunId); }
  }
}

function setupButtons() {
  // Safe bind — skip if element doesn't exist (static mode)
  function bind(id, event, fn) { const el = document.getElementById(id); if (el) el[event] = fn; }
  bind('btn-start-run', 'onclick', startRun);
  bind('btn-score', 'onclick', triggerScoring);
  bind('btn-stop-run', 'onclick', stopRun);
  bind('btn-back', 'onclick', backToDashboard);
  bind('sidebar-title', 'onclick', backToDashboard);
  bind('btn-auto-follow', 'onclick', toggleAutoFollow);
  bind('btn-file-follow', 'onclick', toggleFileFollow);
  bind('btn-toggle-tree', 'onclick', toggleTreeCollapse);
  bind('btn-toggle-domains', 'onclick', toggleDomainCollapse);

  // Auto-disable follow when user scrolls UP (not when already at bottom)
  const termBody = document.getElementById('terminal-body');
  if (termBody) termBody.addEventListener('wheel', (e) => {
    if (!state.autoFollow) return;
    // Only cancel if scrolling up, or not at the bottom
    const atBottom = termBody.scrollHeight - termBody.scrollTop - termBody.clientHeight < 30;
    if (e.deltaY < 0 || !atBottom) {
      state.autoFollow = false;
      const _tt = document.getElementById('toggle-track'); if (_tt) _tt.classList.remove('on');
    }
  });
}

function toggleAutoFollow() {
  state.autoFollow = !state.autoFollow;
  document.getElementById('toggle-track').classList.toggle('on', state.autoFollow);
  if (state.autoFollow) {
    const body = document.getElementById('terminal-body');
    body.scrollTop = body.scrollHeight;
  }
}

function toggleFileFollow() {
  state.userSelectedFile = !state.userSelectedFile;
  const isFollow = !state.userSelectedFile;
  document.getElementById('toggle-file-track').classList.toggle('on', isFollow);
  if (!isFollow) return;
  // If turned on, immediately show latest file
  if (state.currentRunId) {
    (async () => {
      try {
        const files = await (await fetch(`${API}/api/runs/${state.currentRunId}/output-files`)).json();
        let latest = null;
        for (const f of files) {
          if (f.type !== 'file' || !isViewableFile(f.name)) continue;
          if (!latest || (f.mtime && f.mtime > (latest.mtime || 0))) latest = f;
        }
        if (latest) loadFile(state.currentRunId, latest.path, latest.name, null, true);
      } catch (_) {}
    })();
  } else if (state.currentTaskId) {
    // No run — fallback to INSTRUCTIONS.md
    const baseUrl = `${API}/api/tasks/${state.currentTaskId}/file?path=`;
    renderFileContent('INSTRUCTIONS.md', 'INSTRUCTIONS.md', baseUrl + encodeURIComponent('INSTRUCTIONS.md'), null, baseUrl, 'INSTRUCTIONS.md');
  }
}

let treeExpanded = true;
function toggleTreeCollapse() {
  treeExpanded = !treeExpanded;
  document.getElementById('toggle-tree-track').classList.toggle('on', treeExpanded);
  document.querySelectorAll('#file-tree .file-tree-item.dir').forEach(item => {
    item.dataset.open = treeExpanded ? 'true' : 'false';
    item.querySelector('.folder-arrow').innerHTML = treeExpanded ? '&#9660;' : '&#9654;';
  });
  document.querySelectorAll('#file-tree .file-tree-children').forEach(c => {
    c.style.display = treeExpanded ? 'block' : 'none';
  });
}

let domainsExpanded = false;
function toggleDomainCollapse() {
  domainsExpanded = !domainsExpanded;
  const btn = document.getElementById('btn-toggle-domains');
  if (btn) btn.innerHTML = domainsExpanded ? '&#9660;' : '&#9654;';
  document.querySelectorAll('.domain-toggle').forEach(t => t.classList.toggle('open', domainsExpanded));
  document.querySelectorAll('.domain-tasks').forEach(t => t.classList.toggle('open', domainsExpanded));
}

function openImageOverlay(src) {
  const overlay = document.createElement('div');
  overlay.className = 'image-overlay';
  overlay.innerHTML = `<div class="image-overlay-close">&times;</div><img src="${src}">`;
  overlay.onclick = () => overlay.remove();
  document.body.appendChild(overlay);
}

function backToDashboard() {
  document.getElementById('task-view').style.display = 'none';
  document.getElementById('welcome-screen').style.display = 'block';
  state.currentTaskId = null; state.currentRunId = null;
  document.querySelectorAll('.task-item').forEach(el => el.classList.remove('active'));
  document.getElementById('run-history').innerHTML = '';
  loadDashboard();
}

/* ── Util ────────────────────────────────────────────────────────────── */
function esc(s) { if (!s) return ''; const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function agentLogoHtml(name, size = 16) {
  const logo = state.agentLogos[name];
  if (logo) return `<img class="agent-logo" src="${logo}" alt="" style="width:${size}px;height:${size}px;vertical-align:middle;">`;
  return '';
}

async function fetchStaticJSON(path) {
  try { return await (await fetch(path)).json(); } catch (_) { return null; }
}

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return '00h 00m 00s';
  seconds = Math.floor(seconds);
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return `${String(h).padStart(2,'0')}h ${String(m).padStart(2,'0')}m ${String(s).padStart(2,'0')}s`;
}

let _durationTimer = null;
let _durationStart = null;

function showDuration(seconds) {
  stopDurationTimer();
  const el = document.getElementById('duration-display');
  if (el) el.textContent = formatDuration(seconds);
}

function startDurationTimer(offsetSeconds) {
  stopDurationTimer();
  const offset = offsetSeconds || 0;
  _durationStart = Date.now() - offset * 1000;
  const el = document.getElementById('duration-display');
  if (el) el.textContent = formatDuration(offset);
  _durationTimer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - _durationStart) / 1000);
    if (el) el.textContent = formatDuration(elapsed);
  }, 1000);
}

function stopDurationTimer() {
  if (_durationTimer) { clearInterval(_durationTimer); _durationTimer = null; }
}
