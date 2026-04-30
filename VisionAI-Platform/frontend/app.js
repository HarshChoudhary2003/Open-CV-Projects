/**
 * VisionAI Platform — Frontend Application Logic
 * Handles: Auth, WebSocket streaming, telemetry, charts, alerts, faces
 */

const API = "http://localhost:8000/api/v1";
const WS_BASE = "ws://localhost:8000";

let authToken = localStorage.getItem("vai_token") || null;
let currentCamera = "cam0";
let wsStream = null;
let wsTelemetry = null;
let timelineChart = null;
let classChart = null;
let emotionChart = null;
let lastTelemetry = null;

// ── Clock ─────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  document.getElementById("clock").textContent = now.toLocaleTimeString("en-US", {
    hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit"
  });
}
setInterval(updateClock, 1000);
updateClock();

// ── Auth ──────────────────────────────────────────────────────
async function login() {
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value.trim();
  const errEl = document.getElementById("login-error");
  errEl.textContent = "";

  try {
    const res = await fetch(`${API}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    if (!res.ok) {
      const err = await res.json();
      errEl.textContent = err.detail || "Login failed";
      return;
    }

    const data = await res.json();
    authToken = data.access_token;
    localStorage.setItem("vai_token", authToken);
    showDashboard();
  } catch (e) {
    errEl.textContent = "Cannot reach backend. Is the server running?";
  }
}

function logout() {
  authToken = null;
  localStorage.removeItem("vai_token");
  disconnectStreams();
  document.getElementById("login-screen").classList.remove("hidden");
  document.getElementById("dashboard").classList.add("hidden");
}

function showDashboard() {
  document.getElementById("login-screen").classList.add("hidden");
  document.getElementById("dashboard").classList.remove("hidden");
  showSection("feeds");
  loadAlerts();
  loadFaces();
}

// Auto-login if token exists
window.addEventListener("load", () => {
  if (authToken) {
    showDashboard();
  }

  // Listen for enter key on login
  document.getElementById("password").addEventListener("keydown", (e) => {
    if (e.key === "Enter") login();
  });
});

// ── Section Navigation ────────────────────────────────────────
function showSection(name) {
  document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
  document.getElementById(`section-${name}`).classList.add("active");
  if (name === "analytics") loadAnalytics();
  if (name === "alerts") loadAlerts();
  if (name === "faces") loadFaces();
}

// ── Camera Controls ───────────────────────────────────────────
async function startCamera() {
  const source = parseInt(document.getElementById("camera-select").value);
  currentCamera = `cam${source}`;

  try {
    const res = await fetch(`${API}/cameras/start`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        camera_id: currentCamera,
        source: source,
        config: {
          enable_detection: true,
          enable_face: true,
          enable_emotion: true,
          enable_pose: true,
          enable_ocr: false,
          enable_anomaly: true,
          enable_agent: true,
        },
      }),
    });

    if (!res.ok) {
      const e = await res.json();
      alert(e.detail || "Failed to start camera");
      return;
    }

    document.getElementById("no-feed").classList.add("hidden");
    document.getElementById("video-frame").classList.remove("hidden");
    document.getElementById("sys-status").textContent = `CAM ${currentCamera.toUpperCase()} ACTIVE`;

    connectStreams();
  } catch (e) {
    alert("Cannot reach backend: " + e.message);
  }
}

async function stopCamera() {
  disconnectStreams();
  await fetch(`${API}/cameras/${currentCamera}/stop`, {
    method: "POST", headers: authHeaders(),
  }).catch(() => {});

  document.getElementById("video-frame").classList.add("hidden");
  document.getElementById("no-feed").classList.remove("hidden");
  document.getElementById("sys-status").textContent = "SYSTEM ONLINE";
}

// ── WebSocket Streams ─────────────────────────────────────────
function connectStreams() {
  disconnectStreams();

  // Binary video stream
  wsStream = new WebSocket(`${WS_BASE}/ws/stream/${currentCamera}`);
  wsStream.binaryType = "arraybuffer";

  wsStream.onmessage = (evt) => {
    if (typeof evt.data === "string") {
      // JSON telemetry mixed in
      try {
        const data = JSON.parse(evt.data);
        if (!data.ping) updateTelemetry(data);
      } catch {}
      return;
    }
    // Binary JPEG
    const blob = new Blob([evt.data], { type: "image/jpeg" });
    const url = URL.createObjectURL(blob);
    const img = document.getElementById("video-frame");
    const old = img.src;
    img.src = url;
    if (old) URL.revokeObjectURL(old);
  };

  wsStream.onclose = () => {
    document.getElementById("fps-badge").textContent = "-- FPS";
  };

  // Telemetry-only stream
  wsTelemetry = new WebSocket(`${WS_BASE}/ws/telemetry/${currentCamera}`);
  wsTelemetry.onmessage = (evt) => {
    try {
      const data = JSON.parse(evt.data);
      if (!data.ping) updateTelemetry(data);
    } catch {}
  };
}

function disconnectStreams() {
  if (wsStream) { wsStream.close(); wsStream = null; }
  if (wsTelemetry) { wsTelemetry.close(); wsTelemetry = null; }
}

// ── Telemetry UI Update ───────────────────────────────────────
function updateTelemetry(data) {
  lastTelemetry = data;

  // FPS
  if (data.fps) {
    document.getElementById("fps-badge").textContent = `${data.fps.toFixed(1)} FPS`;
  }

  // Stat chips
  const dets = data.detections || [];
  const faces = data.faces || [];
  const emotions = data.emotions || [];

  document.getElementById("stat-objects").innerHTML =
    `Objects: <b>${dets.length}</b>`;
  document.getElementById("stat-faces").innerHTML =
    `Faces: <b>${faces.length}</b>`;

  const dominantEmotion = emotions.length > 0 ? emotions[0].emotion : "--";
  document.getElementById("stat-emotion").innerHTML =
    `Emotion: <b>${dominantEmotion}</b>`;

  const threat = data.anomaly_score || 0;
  document.getElementById("stat-threat").innerHTML =
    `Threat: <b>${(threat * 100).toFixed(0)}%</b>`;

  // Threat meter
  drawThreatMeter(threat);
  document.getElementById("threat-value").textContent = `${(threat * 100).toFixed(0)}%`;
  const labels = ["SECURE", "ELEVATED", "HIGH RISK", "CRITICAL"];
  const labelColors = ["#00e676", "#ffd600", "#ff6d00", "#ff1744"];
  const li = Math.min(Math.floor(threat * 4), 3);
  const tl = document.getElementById("threat-label");
  tl.textContent = labels[li];
  tl.style.color = labelColors[li];

  // Detection list
  const counts = {};
  dets.forEach(d => { counts[d.class_name] = (counts[d.class_name] || 0) + 1; });
  const detList = document.getElementById("detection-list");
  detList.innerHTML = Object.entries(counts).length
    ? Object.entries(counts).map(([cls, n]) =>
        `<div class="det-item"><span>${cls}</span><span class="det-count">${n}</span></div>`
      ).join("")
    : '<span style="color:var(--text-dim)">No objects</span>';

  // Agent alerts
  const actions = data.agent_actions || [];
  if (actions.length > 0) {
    const liveAlerts = document.getElementById("live-alerts");
    actions.forEach(a => {
      const sev = a.payload?.severity || "LOW";
      const div = document.createElement("div");
      div.className = `alert-item ${sev}`;
      div.textContent = `[${sev}] ${a.payload?.rule || a.action_type}`;
      liveAlerts.insertBefore(div, liveAlerts.firstChild);
      // Limit to 6 items
      while (liveAlerts.children.length > 6) {
        liveAlerts.removeChild(liveAlerts.lastChild);
      }
    });
  }

  // Pose / gesture
  if (data.pose) {
    document.getElementById("gesture-val").textContent = data.pose.gesture_label || data.pose.gesture || "--";
    document.getElementById("pose-val").textContent = data.pose.pose_label || "--";
  }

  // OCR
  const ocrTexts = data.ocr_texts || [];
  document.getElementById("ocr-results").textContent =
    ocrTexts.length > 0 ? ocrTexts.map(o => o.text).join(" | ") : "No text detected.";
}

// ── Threat Meter (Canvas) ─────────────────────────────────────
function drawThreatMeter(score) {
  const canvas = document.getElementById("threat-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const cx = 80, cy = 80, r = 65;

  ctx.clearRect(0, 0, 160, 160);

  // Background arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI * 0.75, Math.PI * 2.25);
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 12;
  ctx.lineCap = "round";
  ctx.stroke();

  // Filled arc
  const gradient = ctx.createLinearGradient(0, 0, 160, 0);
  gradient.addColorStop(0, "#00e676");
  gradient.addColorStop(0.5, "#ffd600");
  gradient.addColorStop(1, "#ff1744");

  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI * 0.75, Math.PI * 0.75 + Math.PI * 1.5 * score);
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 12;
  ctx.lineCap = "round";
  ctx.stroke();

  // Tick marks
  for (let i = 0; i <= 10; i++) {
    const angle = Math.PI * 0.75 + (Math.PI * 1.5 * i) / 10;
    const x1 = cx + (r - 16) * Math.cos(angle);
    const y1 = cy + (r - 16) * Math.sin(angle);
    const x2 = cx + (r - 8) * Math.cos(angle);
    const y2 = cy + (r - 8) * Math.sin(angle);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

// ── Analytics ─────────────────────────────────────────────────
async function loadAnalytics() {
  const hours = document.getElementById("hours-select")?.value || 24;

  try {
    const [summaryRes, timelineRes] = await Promise.all([
      fetch(`${API}/analytics/summary?hours=${hours}`, { headers: authHeaders() }),
      fetch(`${API}/analytics/timeline?hours=${hours}`, { headers: authHeaders() }),
    ]);

    const summary = await summaryRes.json();
    const timeline = await timelineRes.json();

    // Summary card
    const summaryEl = document.getElementById("summary-stats");
    summaryEl.innerHTML = `
      <div class="summary-row"><span>Total Detections</span><b>${summary.total_detections}</b></div>
      <div class="summary-row"><span>Time Period</span><b>${hours}h</b></div>
      ${Object.entries(summary.by_class || {}).slice(0, 6).map(([k, v]) =>
        `<div class="summary-row"><span>${k}</span><b>${v}</b></div>`
      ).join("")}
    `;

    // Timeline chart
    const tlData = timeline.timeline || [];
    const tlLabels = tlData.map(t => new Date(t.timestamp).toLocaleTimeString());
    const tlValues = tlData.map(t => t.count);

    if (timelineChart) timelineChart.destroy();
    const tlCtx = document.getElementById("timeline-chart").getContext("2d");
    timelineChart = new Chart(tlCtx, {
      type: "line",
      data: {
        labels: tlLabels,
        datasets: [{
          label: "Detections",
          data: tlValues,
          borderColor: "#00e5ff",
          backgroundColor: "rgba(0,229,255,0.08)",
          fill: true,
          tension: 0.4,
          pointRadius: 2,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#546e7a", maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.04)" } },
          y: { ticks: { color: "#546e7a" }, grid: { color: "rgba(255,255,255,0.04)" } },
        },
      },
    });

    // Class breakdown chart
    const classes = Object.keys(summary.by_class || {}).slice(0, 10);
    const classCounts = classes.map(k => summary.by_class[k]);
    const colors = classes.map((_, i) =>
      `hsl(${(i * 36) % 360}, 80%, 60%)`
    );

    if (classChart) classChart.destroy();
    const ccCtx = document.getElementById("class-chart").getContext("2d");
    classChart = new Chart(ccCtx, {
      type: "doughnut",
      data: {
        labels: classes,
        datasets: [{
          data: classCounts,
          backgroundColor: colors,
          borderColor: "rgba(0,0,0,0.3)",
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: "right", labels: { color: "#b0bec5", font: { size: 11 } } },
        },
      },
    });

    // Emotion chart (demo data if no real data)
    const emotionLabels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"];
    const emotionColors = ["#ff1744","#00c853","#9c27b0","#ffd600","#78909c","#2196f3","#ff6d00"];

    if (emotionChart) emotionChart.destroy();
    const ecCtx = document.getElementById("emotion-chart").getContext("2d");
    emotionChart = new Chart(ecCtx, {
      type: "bar",
      data: {
        labels: emotionLabels,
        datasets: [{
          label: "Emotion Score",
          data: emotionLabels.map(() => Math.random() * 0.5),
          backgroundColor: emotionColors,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#546e7a" }, grid: { color: "rgba(255,255,255,0.04)" } },
          y: { ticks: { color: "#546e7a" }, max: 1, grid: { color: "rgba(255,255,255,0.04)" } },
        },
      },
    });

  } catch (e) {
    console.warn("Analytics load failed:", e);
  }
}

// ── Alerts ────────────────────────────────────────────────────
async function loadAlerts() {
  const severity = document.getElementById("severity-filter")?.value || "";
  let url = `${API}/alerts?limit=100`;
  if (severity) url += `&severity=${severity}`;

  try {
    const res = await fetch(url, { headers: authHeaders() });
    if (!res.ok) return;
    const alerts = await res.json();
    const tbody = document.getElementById("alerts-tbody");
    tbody.innerHTML = alerts.map(a => `
      <tr>
        <td>${new Date(a.timestamp).toLocaleString()}</td>
        <td>${a.camera_id}</td>
        <td><code>${a.alert_type}</code></td>
        <td><span class="severity-badge ${a.severity}">${a.severity}</span></td>
        <td>${a.description}</td>
        <td>
          ${a.acknowledged
            ? '<span style="color:var(--accent-green)">✓ ACK</span>'
            : `<button class="ack-btn" onclick="ackAlert(${a.id})">ACK</button>`
          }
        </td>
      </tr>
    `).join("") || "<tr><td colspan='6' style='text-align:center;color:var(--text-dim)'>No alerts found</td></tr>";
  } catch (e) {
    console.warn("Alerts load failed:", e);
  }
}

async function ackAlert(id) {
  await fetch(`${API}/alerts/${id}/acknowledge`, {
    method: "PATCH", headers: authHeaders(),
  });
  loadAlerts();
}

// ── Faces ─────────────────────────────────────────────────────
async function loadFaces() {
  try {
    const res = await fetch(`${API}/faces`, { headers: authHeaders() });
    if (!res.ok) return;
    const faces = await res.json();
    const grid = document.getElementById("faces-list");
    grid.innerHTML = faces.length
      ? faces.map(f => `
          <div class="face-card">
            <div class="face-avatar">👤</div>
            <div class="face-name">${f.name}</div>
            <div class="face-access">${f.access_level.toUpperCase()}</div>
            <button onclick="deleteFace('${f.person_id}')"
              style="margin-top:8px;font-size:0.65rem;background:transparent;border:1px solid var(--accent-red);
                     color:var(--accent-red);padding:2px 8px;border-radius:4px;cursor:pointer;">
              DELETE
            </button>
          </div>
        `).join("")
      : "<p style='color:var(--text-dim);font-family:var(--font-mono);font-size:0.8rem;'>No faces registered.</p>";
  } catch (e) {
    console.warn("Faces load failed:", e);
  }
}

async function registerFace() {
  const name = document.getElementById("face-name").value.trim();
  const access = document.getElementById("face-access").value;
  const file = document.getElementById("face-file").files[0];
  const msgEl = document.getElementById("face-msg");

  if (!name || !file) {
    msgEl.textContent = "Please provide a name and photo.";
    msgEl.style.color = "var(--accent-red)";
    return;
  }

  const fd = new FormData();
  fd.append("name", name);
  fd.append("access_level", access);
  fd.append("file", file);

  try {
    const res = await fetch(`${API}/faces/register`, {
      method: "POST",
      headers: { Authorization: `Bearer ${authToken}` },
      body: fd,
    });

    if (!res.ok) {
      const e = await res.json();
      msgEl.textContent = e.detail || "Registration failed";
      msgEl.style.color = "var(--accent-red)";
      return;
    }

    const data = await res.json();
    msgEl.textContent = `✓ ${data.name} registered (ID: ${data.person_id.slice(0, 8)}...)`;
    msgEl.style.color = "var(--accent-green)";
    loadFaces();
  } catch (e) {
    msgEl.textContent = "Server error: " + e.message;
    msgEl.style.color = "var(--accent-red)";
  }
}

async function deleteFace(personId) {
  if (!confirm("Delete this person from the registry?")) return;
  await fetch(`${API}/faces/${personId}`, {
    method: "DELETE", headers: authHeaders(),
  });
  loadFaces();
}

// ── Reports ───────────────────────────────────────────────────
function exportCSV() {
  const hours = document.getElementById("hours-select")?.value || 24;
  const url = `${API}/reports/csv/detections?hours=${hours}`;
  downloadWithAuth(url, `detections_${hours}h.csv`);
}

function exportPDF() {
  const hours = document.getElementById("hours-select")?.value || 24;
  const url = `${API}/reports/pdf/summary?hours=${hours}`;
  downloadWithAuth(url, `visionai_report_${hours}h.pdf`);
}

async function downloadWithAuth(url, filename) {
  try {
    const res = await fetch(url, { headers: authHeaders() });
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
  } catch (e) {
    alert("Download failed: " + e.message);
  }
}

// ── Helpers ───────────────────────────────────────────────────
function authHeaders() {
  return {
    Authorization: `Bearer ${authToken}`,
    "Content-Type": "application/json",
  };
}

// Draw initial threat meter
drawThreatMeter(0);
