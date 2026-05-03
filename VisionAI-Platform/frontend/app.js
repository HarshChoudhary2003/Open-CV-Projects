/* ═══════════════════════════════════════════════
   VisionAI v2.0 — Frontend Application
   ═══════════════════════════════════════════════ */

const API  = 'http://localhost:8000/api/v1';
const WS   = 'ws://localhost:8000';

let token        = null;
let streamWS     = null;
let telemetryWS  = null;
let copilotWS    = null;
let charts       = {};
let anomalyData  = [];
let currentCam   = '0';
let heatmapMode  = false;

/* ══ Particles background on login ══ */
(function initParticles(){
  const c = document.getElementById('login-particles');
  if(!c) return;
  for(let i=0;i<60;i++){
    const d = document.createElement('div');
    d.style.cssText=`position:absolute;width:${1+Math.random()*2}px;height:${1+Math.random()*2}px;
      background:rgba(0,255,200,${0.1+Math.random()*0.3});border-radius:50%;
      left:${Math.random()*100}%;top:${Math.random()*100}%;
      animation:float ${4+Math.random()*6}s linear ${Math.random()*4}s infinite;`;
    c.appendChild(d);
  }
  const style = document.createElement('style');
  style.textContent=`@keyframes float{0%{transform:translateY(0) scale(1);opacity:.5}
    50%{transform:translateY(-30px) scale(1.2);opacity:1}
    100%{transform:translateY(-60px) scale(0.8);opacity:0}}`;
  document.head.appendChild(style);
})();

/* ══ Clock ══ */
setInterval(()=>{
  const el = document.getElementById('clock');
  if(el) el.textContent = new Date().toLocaleTimeString('en-GB',{hour12:false});
},1000);

/* ══ Auth ══ */
async function login(){
  const u = document.getElementById('username').value;
  const p = document.getElementById('password').value;
  const btn = document.getElementById('login-btn');
  btn.disabled = true;
  btn.querySelector('.btn-text').textContent = 'AUTHENTICATING...';
  try {
    const fd = new FormData();
    fd.append('username', u); fd.append('password', p);
    const r = await fetch(`${API}/auth/token`, {method:'POST', body:fd});
    if(!r.ok) throw new Error('Invalid credentials');
    const d = await r.json();
    token = d.access_token;
    document.getElementById('login-screen').classList.add('hidden');
    document.getElementById('dashboard').classList.remove('hidden');
    document.getElementById('status-dot').classList.add('online');
    document.getElementById('sys-status').textContent = 'SYSTEM ONLINE';
    initDashboard();
  } catch(e){
    document.getElementById('login-error').textContent = e.message;
    btn.disabled = false;
    btn.querySelector('.btn-text').textContent = 'INITIALIZE SYSTEM';
  }
}

function logout(){
  token = null;
  stopCamera();
  if(copilotWS){ copilotWS.close(); copilotWS=null; }
  document.getElementById('dashboard').classList.add('hidden');
  document.getElementById('login-screen').classList.remove('hidden');
}

function h(){ return { Authorization:`Bearer ${token}` }; }

/* ══ Dashboard Init ══ */
function initDashboard(){
  initCharts();
  loadAlerts();
  loadFaces();
  loadZones();
  initCopilotWS();
  loadAnalytics();
  setInterval(loadAlerts, 15000);
  setInterval(()=>{ if(document.getElementById('section-analytics').classList.contains('active')) loadAnalytics(); }, 30000);
}

/* ══ Section Navigation ══ */
function showSection(name){
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById(`section-${name}`).classList.add('active');
  document.getElementById(`nav-${name}`)?.classList.add('active');
  if(name==='analytics') loadAnalytics();
  if(name==='zones') loadZones();
  if(name==='heatmap') refreshHeatmap();
}

/* ══ Camera Control ══ */
async function startCamera(){
  const src = document.getElementById('camera-select').value;
  currentCam = `cam${src}`;
  try {
    await fetch(`${API}/cameras/${currentCam}/start`, {
      method:'POST', headers:{...h(),'Content-Type':'application/json'},
      body: JSON.stringify({source: parseInt(src), config:{enable_heatmap:true, enable_zones:true, enable_predictive:true}})
    });
  } catch(e){ console.warn('Start camera:', e); }

  // Video stream WS
  if(streamWS) streamWS.close();
  streamWS = new WebSocket(`${WS}/ws/stream/${currentCam}`);
  streamWS.binaryType = 'blob';
  streamWS.onopen = ()=>{
    document.getElementById('no-feed').classList.add('hidden');
    document.getElementById('video-frame').classList.remove('hidden');
    document.getElementById('status-dot').classList.add('online');
  };
  streamWS.onmessage = e=>{
    if(e.data instanceof Blob){
      const url = URL.createObjectURL(e.data);
      const img = document.getElementById('video-frame');
      if(img.src.startsWith('blob:')) URL.revokeObjectURL(img.src);
      img.src = url;
    } else {
      try { updateTelemetry(JSON.parse(e.data)); } catch{}
    }
  };
  streamWS.onclose = ()=>{
    document.getElementById('no-feed').classList.remove('hidden');
    document.getElementById('video-frame').classList.add('hidden');
  };

  // Telemetry WS
  if(telemetryWS) telemetryWS.close();
  telemetryWS = new WebSocket(`${WS}/ws/telemetry/${currentCam}`);
  telemetryWS.onmessage = e=>{ try{ updateTelemetry(JSON.parse(e.data)); }catch{} };
}

async function stopCamera(){
  if(streamWS){ streamWS.close(); streamWS=null; }
  if(telemetryWS){ telemetryWS.close(); telemetryWS=null; }
  try { await fetch(`${API}/cameras/${currentCam}/stop`, {method:'POST', headers:h()}); } catch{}
}

async function takeSnapshot(){
  try {
    const r = await fetch(`${API}/cameras/${currentCam}/snapshot`, {method:'POST', headers:h()});
    const d = await r.json();
    showToast('SNAPSHOT', `Saved: ${d.path||'snapshot'}`, 'LOW');
  } catch(e){ showToast('ERROR', 'Snapshot failed', 'HIGH'); }
}

function toggleHeatmapOverlay(){
  heatmapMode = document.getElementById('heatmap-toggle').checked;
}

/* ══ Telemetry Updates ══ */
function updateTelemetry(data){
  if(!data || data.ping) return;

  // FPS
  const fps = data.fps ? data.fps.toFixed(1) : '--';
  const fpsEl = document.getElementById('fps-badge');
  if(fpsEl) fpsEl.textContent = `${fps} FPS`;

  // Stats chips
  setText('stat-objects', `Objects: <b>${(data.detections||[]).length}</b>`);
  setText('stat-faces',   `Faces: <b>${(data.faces||[]).length}</b>`);
  const em = data.emotions?.[0]?.emotion || '--';
  setText('stat-emotion', `Emotion: <b>${em}</b>`);
  const pct = Math.round((data.anomaly_score||0)*100);
  setText('stat-threat',  `Threat: <b>${pct}%</b>`);
  const zi = (data.zone_intrusions||[]).length;
  setText('stat-zones',   `Zones: <b>${zi}</b>`);

  // Threat meter
  drawThreatMeter(data.anomaly_score||0);
  const tq = document.getElementById('threat-quick');
  if(tq){
    const lvl = pct>75?'CRITICAL':pct>40?'HIGH':'SECURE';
    tq.textContent = lvl;
    tq.className = 'threat-quick' + (pct>40?' high':'');
  }

  // Detection list
  const detList = document.getElementById('detection-list');
  if(detList){
    const counts = {};
    (data.detections||[]).forEach(d=>counts[d.class_name]=(counts[d.class_name]||0)+1);
    detList.innerHTML = Object.entries(counts).map(([k,v])=>
      `<div class="det-item"><span>${k}</span><b>${v}</b></div>`).join('') || '<div class="no-pred">No objects</div>';
  }

  // Live alerts from agent actions
  const liveAlerts = document.getElementById('live-alerts');
  if(liveAlerts && (data.agent_actions||[]).length){
    (data.agent_actions).forEach(a=>{
      if(a.action_type==='ALERT'){
        const sev = a.payload?.severity||'LOW';
        const div = document.createElement('div');
        div.className = `alert-item ${sev}`;
        div.textContent = `${sev}: ${a.payload?.rule?.replace(/_/g,' ')||'alert'}`;
        liveAlerts.prepend(div);
        if(liveAlerts.children.length > 6) liveAlerts.lastChild.remove();
        showToast(a.payload?.rule||'ALERT', `${sev}: ${a.payload?.context_summary||''}`, sev);
      }
    });
  }

  // Predictive events
  const predList = document.getElementById('pred-events');
  if(predList){
    const preds = data.predicted_events||[];
    predList.innerHTML = preds.length
      ? preds.map(p=>`<div class="pred-item">⚡ ${p.description||p.reasons?.join(', ')||'Suspicious'}</div>`).join('')
      : '<div class="no-pred">No predictions active</div>';
  }

  // Pose
  if(data.pose){
    setText2('gesture-val', data.pose.gesture||'--');
    setText2('pose-val',    data.pose.label||'--');
  }

  // Scene narrative
  if(data.scene_narrative){
    const el = document.getElementById('scene-narrative');
    if(el) el.textContent = data.scene_narrative;
    const el2 = document.getElementById('live-narrative');
    if(el2) el2.textContent = data.scene_narrative;
  }

  // OCR
  if(data.ocr_texts?.length){
    const ocr = document.getElementById('ocr-results');
    if(ocr) ocr.textContent = data.ocr_texts.map(o=>o.text).join(' | ');
  }

  // Anomaly chart data
  anomalyData.push({x: new Date().toLocaleTimeString(), y: Math.round((data.anomaly_score||0)*100)});
  if(anomalyData.length > 30) anomalyData.shift();
  if(charts.anomaly){
    charts.anomaly.data.labels = anomalyData.map(d=>d.x);
    charts.anomaly.data.datasets[0].data = anomalyData.map(d=>d.y);
    charts.anomaly.update('none');
  }

  // Heatmap frame
  if(heatmapMode && data.heatmap_jpeg){
    const img = document.getElementById('video-frame');
    if(img) img.src = 'data:image/jpeg;base64,' + data.heatmap_jpeg;
  }
}

function setText(id, html){ const e=document.getElementById(id); if(e) e.innerHTML=html; }
function setText2(id, txt){ const e=document.getElementById(id); if(e) e.textContent=txt; }

/* ══ Threat Meter (Canvas) ══ */
function drawThreatMeter(score){
  const canvas = document.getElementById('threat-canvas');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const cx=75, cy=75, r=60;
  ctx.clearRect(0,0,150,150);
  ctx.beginPath(); ctx.arc(cx,cy,r,0.75*Math.PI,2.25*Math.PI);
  ctx.strokeStyle='rgba(0,200,160,0.12)'; ctx.lineWidth=10; ctx.lineCap='round'; ctx.stroke();
  const angle = 0.75*Math.PI + score*1.5*Math.PI;
  const g = ctx.createLinearGradient(cx-r,cy,cx+r,cy);
  g.addColorStop(0,'#00ffc8'); g.addColorStop(0.5,'#ff8c00'); g.addColorStop(1,'#ff3a3a');
  ctx.beginPath(); ctx.arc(cx,cy,r,0.75*Math.PI,angle);
  ctx.strokeStyle=g; ctx.lineWidth=10; ctx.lineCap='round'; ctx.stroke();
  // Value text
  const tv=document.getElementById('threat-value'); if(tv) tv.textContent=Math.round(score*100)+'%';
  const tl=document.getElementById('threat-label');
  if(tl) tl.textContent=score>0.75?'CRITICAL':score>0.4?'HIGH':'SECURE';
}

/* ══ Analytics ══ */
async function loadAnalytics(){
  const hrs = document.getElementById('hours-select')?.value || 24;
  try {
    const [sumR, tlR, clR, emR] = await Promise.all([
      fetch(`${API}/analytics/summary?hours=${hrs}`,    {headers:h()}),
      fetch(`${API}/analytics/timeline?hours=${hrs}`,   {headers:h()}),
      fetch(`${API}/analytics/classes?hours=${hrs}`,    {headers:h()}),
      fetch(`${API}/analytics/emotions?hours=${hrs}`,   {headers:h()}),
    ]);
    const sum = await sumR.json();
    const tl  = await tlR.json();
    const cl  = await clR.json();
    const em  = await emR.json();
    renderSummary(sum);
    updateChart(charts.timeline, tl.labels||[], tl.data||[], 'line');
    updateChart(charts.classes,  cl.labels||[], cl.data||[],  'doughnut');
    updateChart(charts.emotions, em.labels||[], em.data||[], 'bar');
  } catch(e){ console.warn('Analytics:', e); }
}

function renderSummary(s){
  const el = document.getElementById('summary-stats');
  if(!el) return;
  const items = [
    {v: s.total_detections||0, l:'DETECTIONS'},
    {v: s.total_alerts||0,     l:'ALERTS'},
    {v: s.unique_faces||0,     l:'FACES SEEN'},
    {v: s.avg_fps?.toFixed(1)||'--', l:'AVG FPS'},
  ];
  el.innerHTML = items.map(i=>`
    <div class="summary-item">
      <div class="s-val">${i.v}</div>
      <div class="s-label">${i.l}</div>
    </div>`).join('');
}

function updateChart(chart, labels, data, type){
  if(!chart) return;
  chart.data.labels = labels;
  chart.data.datasets[0].data = data;
  chart.update();
}

/* ══ Charts Init ══ */
function initCharts(){
  const defaults = { responsive:true, maintainAspectRatio:false,
    plugins:{legend:{labels:{color:'#6a8898',font:{family:'Share Tech Mono',size:11}}}},
    scales:{x:{ticks:{color:'#6a8898'},grid:{color:'rgba(0,200,160,0.06)'}},
            y:{ticks:{color:'#6a8898'},grid:{color:'rgba(0,200,160,0.06)'}}} };
  const noScales = { responsive:true, maintainAspectRatio:false,
    plugins:{legend:{labels:{color:'#6a8898',font:{family:'Share Tech Mono',size:10}}}} };

  charts.timeline = new Chart(document.getElementById('timeline-chart'), {
    type:'line', options:{...defaults, plugins:{...defaults.plugins}},
    data:{labels:[],datasets:[{label:'Detections',data:[],
      borderColor:'#00ffc8',backgroundColor:'rgba(0,255,200,0.08)',tension:.4,fill:true,pointRadius:3}]}
  });
  charts.classes = new Chart(document.getElementById('class-chart'), {
    type:'doughnut', options:noScales,
    data:{labels:[],datasets:[{data:[],
      backgroundColor:['#00ffc8','#00b3ff','#c864ff','#ff8c00','#ff3a3a','#ffd000'],
      borderColor:'rgba(0,0,0,0.3)',borderWidth:2}]}
  });
  charts.emotions = new Chart(document.getElementById('emotion-chart'), {
    type:'bar', options:{...defaults, plugins:{...defaults.plugins}},
    data:{labels:[],datasets:[{label:'Count',data:[],
      backgroundColor:'rgba(200,100,255,0.5)',borderColor:'#c864ff',borderWidth:1}]}
  });
  charts.anomaly = new Chart(document.getElementById('anomaly-chart'), {
    type:'line', options:{...defaults, plugins:{...defaults.plugins},
      scales:{...defaults.scales, y:{...defaults.scales.y, min:0, max:100}}},
    data:{labels:[],datasets:[{label:'Anomaly %',data:[],
      borderColor:'#ff3a3a',backgroundColor:'rgba(255,58,58,0.08)',tension:.4,fill:true,pointRadius:2}]}
  });
}

/* ══ Heatmap ══ */
async function refreshHeatmap(){
  const img = document.getElementById('heatmap-frame');
  if(!img) return;
  const t = Date.now();
  img.src = `${API.replace('/api/v1','')}/api/v1/cameras/${currentCam}/heatmap?t=${t}`;
  img.onload = ()=>{
    document.getElementById('no-heatmap').classList.add('hidden');
    img.classList.remove('hidden');
  };
  img.onerror = ()=>{ img.classList.add('hidden'); document.getElementById('no-heatmap').classList.remove('hidden'); };
}

async function resetHeatmap(){
  try {
    await fetch(`${API}/cameras/${currentCam}/heatmap/reset`, {method:'POST', headers:h()});
    showToast('HEATMAP','Heatmap data cleared','LOW');
    refreshHeatmap();
  } catch{}
}

/* ══ Zone Management ══ */
async function loadZones(){
  try {
    const r = await fetch(`${API}/zones`, {headers:h()});
    const zones = await r.json();
    const list = document.getElementById('zones-list');
    if(!list) return;
    if(!zones.length){
      list.innerHTML='<div class="no-zones">No zones configured. Create one to enable intrusion detection.</div>';
      return;
    }
    list.innerHTML = zones.map(z=>`
      <div class="zone-item ${z.zone_type}">
        <div class="zone-name">${z.name}</div>
        <div class="zone-meta">Type: ${z.zone_type} | Cam: ${z.camera_id}</div>
        <div class="zone-meta">Points: ${z.polygon.length}</div>
        <button class="ack-btn" onclick="deleteZone('${z.zone_id}')">🗑 Delete</button>
      </div>`).join('');
  } catch(e){ console.warn('Zones:', e); }
}

async function createZone(){
  const name = document.getElementById('zone-name').value.trim();
  const type = document.getElementById('zone-type').value;
  const cam  = document.getElementById('zone-camera').value;
  const raw  = document.getElementById('zone-polygon').value.trim();
  if(!name||!raw){ showMsg('zone-msg','Fill all fields','error'); return; }
  let polygon;
  try {
    polygon = raw.split('\n').map(line=>{
      const [x,y] = line.split(',').map(Number);
      return [x,y];
    });
  } catch{ showMsg('zone-msg','Invalid polygon format','error'); return; }
  try {
    await fetch(`${API}/zones`, {method:'POST', headers:{...h(),'Content-Type':'application/json'},
      body: JSON.stringify({name, polygon, zone_type:type, camera_id:cam, color:[0,0,255]})});
    showMsg('zone-msg','Zone created!','ok');
    loadZones();
  } catch(e){ showMsg('zone-msg','Error: '+e.message,'error'); }
}

async function deleteZone(id){
  try {
    await fetch(`${API}/zones/${id}`, {method:'DELETE', headers:h()});
    loadZones();
  } catch{}
}

/* ══ AI Copilot ══ */
function initCopilotWS(){
  copilotWS = new WebSocket(`${WS}/ws/copilot`);
  copilotWS.onmessage = e=>{
    try {
      const d = JSON.parse(e.data);
      if(d.answer) addChatMessage('assistant', d.answer);
      if(d.narrative){
        const el = document.getElementById('live-narrative');
        if(el) el.textContent = d.narrative;
      }
      if(d.history) renderQAHistory(d.history);
      removeChatThinking();
    } catch{}
  };
  copilotWS.onerror = ()=>{ copilotWS=null; };
}

function sendCopilotQuestion(){
  const inp = document.getElementById('copilot-input');
  const q = inp.value.trim();
  if(!q) return;
  inp.value = '';
  addChatMessage('user', q);
  addChatThinking();
  if(copilotWS && copilotWS.readyState===WebSocket.OPEN){
    copilotWS.send(JSON.stringify({question:q}));
  } else {
    // REST fallback
    fetch(`${API}/copilot/ask`, {method:'POST',
      headers:{...h(),'Content-Type':'application/json'},
      body: JSON.stringify({question:q})
    }).then(r=>r.json()).then(d=>{
      removeChatThinking();
      addChatMessage('assistant', d.answer);
      if(d.narrative){ const el=document.getElementById('live-narrative'); if(el) el.textContent=d.narrative; }
    }).catch(e=>{ removeChatThinking(); addChatMessage('assistant','Error: '+e.message); });
  }
}

function askQuick(q){ document.getElementById('copilot-input').value=q; sendCopilotQuestion(); }

function addChatMessage(role, text){
  const container = document.getElementById('chat-messages');
  if(!container) return;
  const div = document.createElement('div');
  div.className = `chat-msg ${role==='user'?'user':'system'}`;
  div.innerHTML = `<div class="msg-avatar">${role==='user'?'👤':'🧠'}</div>
    <div class="msg-body">
      <div class="msg-author">${role==='user'?'OPERATOR':'VisionAI Copilot'}</div>
      <div class="msg-text">${text}</div>
    </div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function addChatThinking(){
  const container = document.getElementById('chat-messages');
  if(!container) return;
  const div = document.createElement('div');
  div.className = 'chat-msg system'; div.id = 'thinking-msg';
  div.innerHTML = `<div class="msg-avatar">🧠</div>
    <div class="msg-body"><div class="msg-text thinking">Analyzing scene...</div></div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function removeChatThinking(){ document.getElementById('thinking-msg')?.remove(); }

function renderQAHistory(history){
  const el = document.getElementById('qa-history');
  if(!el) return;
  el.innerHTML = history.slice(-10).reverse().map(h=>`
    <div class="qa-item">
      <div class="qa-q">Q: ${h.question}</div>
      <div class="qa-a">A: ${h.answer}</div>
    </div>`).join('');
}

/* ══ Alerts ══ */
async function loadAlerts(){
  const sev = document.getElementById('severity-filter')?.value||'';
  try {
    const url = `${API}/alerts?limit=50${sev?'&severity='+sev:''}`;
    const r = await fetch(url, {headers:h()});
    const d = await r.json();
    const tbody = document.getElementById('alerts-tbody');
    if(!tbody) return;
    tbody.innerHTML = (d.items||d||[]).map(a=>`
      <tr>
        <td>${new Date(a.created_at*1000||a.timestamp*1000).toLocaleTimeString()}</td>
        <td>${a.camera_id||'--'}</td>
        <td>${(a.alert_type||a.rule||'--').replace(/_/g,' ')}</td>
        <td><span class="badge badge-${a.severity}">${a.severity}</span></td>
        <td>${a.description||a.context_summary||'--'}</td>
        <td>${a.acknowledged?'✓':`<button class="ack-btn" onclick="ackAlert('${a.id}')">ACK</button>`}</td>
      </tr>`).join('');
  } catch(e){ console.warn('Alerts:', e); }
}

async function ackAlert(id){
  try { await fetch(`${API}/alerts/${id}/acknowledge`, {method:'POST', headers:h()}); loadAlerts(); } catch{}
}

async function acknowledgeAll(){
  try { await fetch(`${API}/alerts/acknowledge-all`, {method:'POST', headers:h()}); loadAlerts(); } catch{}
}

/* ══ Faces ══ */
async function loadFaces(){
  try {
    const r = await fetch(`${API}/faces`, {headers:h()});
    const faces = await r.json();
    const el = document.getElementById('faces-list');
    if(!el) return;
    el.innerHTML = (faces.items||faces||[]).map(f=>`
      <div class="face-item">
        <div class="face-avatar">👤</div>
        <div class="face-name">${f.name}</div>
        <div class="face-access">${f.access_level||'visitor'}</div>
        <button class="face-del" onclick="deleteFace('${f.person_id}')">🗑 Remove</button>
      </div>`).join('');
  } catch{}
}

async function registerFace(){
  const name   = document.getElementById('face-name').value.trim();
  const access = document.getElementById('face-access').value;
  const file   = document.getElementById('face-file').files[0];
  if(!name||!file){ showMsg('face-msg','Name and photo required','error'); return; }
  const fd = new FormData();
  fd.append('name', name); fd.append('access_level', access); fd.append('file', file);
  try {
    const r = await fetch(`${API}/faces/register`, {method:'POST', headers:h(), body:fd});
    if(!r.ok) throw new Error('Registration failed');
    showMsg('face-msg','Registered successfully!','ok');
    loadFaces();
  } catch(e){ showMsg('face-msg',e.message,'error'); }
}

async function deleteFace(id){
  try { await fetch(`${API}/faces/${id}`, {method:'DELETE', headers:h()}); loadFaces(); } catch{}
}

/* ══ Reports ══ */
async function exportCSV(){
  const hrs = document.getElementById('hours-select')?.value||24;
  window.open(`${API}/reports/csv?hours=${hrs}&token=${token}`);
}
async function exportPDF(){
  const hrs = document.getElementById('hours-select')?.value||24;
  window.open(`${API}/reports/pdf?hours=${hrs}&token=${token}`);
}

/* ══ Toast Notifications ══ */
function showToast(title, msg, severity='LOW'){
  const container = document.getElementById('toast-container');
  if(!container) return;
  const t = document.createElement('div');
  t.className = `toast ${severity}`;
  t.innerHTML = `<div class="toast-title">${severity} | ${title.toUpperCase().replace(/_/g,' ')}</div>${msg}`;
  container.appendChild(t);
  setTimeout(()=>{ t.style.animation='fade-out .4s ease forwards'; setTimeout(()=>t.remove(),400); }, 4000);
}

/* ══ Helpers ══ */
function showMsg(id, txt, type){
  const el = document.getElementById(id);
  if(!el) return;
  el.textContent = txt;
  el.style.color = type==='error'?'var(--danger)':'var(--accent)';
}
