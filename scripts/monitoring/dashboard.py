"""Dashboard — reads from state.json, serves single page with WebSocket push."""
import asyncio
import json
import time
import webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import websockets

STATE_FILE = Path(__file__).parent / "state.json"
ws_clients = set()
ws_loop = None


def read_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except:
        return {"vms": [], "evals": [], "leaderboard": [], "submissions": [], "last_update": "no data"}


HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>NM i AI 2026</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>tailwind.config={theme:{extend:{colors:{bg:'#09090b',card:'#18181b',border:'#27272a',muted:'#71717a',accent:'#22c55e',accent2:'#3b82f6',warn:'#f59e0b',danger:'#ef4444'}}}}</script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
body{font-family:'Inter',sans-serif}.mono{font-family:'JetBrains Mono',monospace}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}.pulse{animation:pulse 1.5s ease-in-out infinite}
</style>
</head><body class="bg-bg text-white min-h-screen p-6">
<div class="max-w-7xl mx-auto">

<!-- Header -->
<div class="flex items-center justify-between mb-6">
  <div><h1 class="text-xl font-bold tracking-tight">NM i AI 2026 — Command Center</h1>
  <p class="text-muted text-xs mt-1">NorgesGruppen Object Detection</p></div>
  <div class="flex items-center gap-3">
    <div class="w-2 h-2 rounded-full" id="ws-dot"></div>
    <span class="text-muted text-xs mono" id="time">—</span>
  </div>
</div>

<!-- Score Banner -->
<div class="bg-card border border-border rounded-xl p-5 mb-5">
  <div class="grid grid-cols-6 gap-4">
    <div><p class="text-muted text-[10px] uppercase tracking-wider mb-1">Test Score</p><p class="text-2xl font-bold mono text-accent" id="test-score">—</p></div>
    <div><p class="text-muted text-[10px] uppercase tracking-wider mb-1">Val Score</p><p class="text-2xl font-bold mono text-accent2" id="val-score">—</p></div>
    <div><p class="text-muted text-[10px] uppercase tracking-wider mb-1">Leader</p><p class="text-2xl font-bold mono text-warn" id="leader">—</p></div>
    <div><p class="text-muted text-[10px] uppercase tracking-wider mb-1">Gap</p><p class="text-2xl font-bold mono text-danger" id="gap">—</p></div>
    <div><p class="text-muted text-[10px] uppercase tracking-wider mb-1">Val→Test Slope</p><p class="text-2xl font-bold mono text-danger" id="ratio">—</p></div>
    <div><p class="text-muted text-[10px] uppercase tracking-wider mb-1">UTC Reset</p><p class="text-2xl font-bold mono text-muted" id="reset">—</p></div>
  </div>
</div>

<!-- Main Grid -->
<div class="grid grid-cols-3 gap-4 mb-5" id="vms"></div>

<!-- Eval Results -->
<div class="bg-card border border-border rounded-xl p-5 mb-5">
  <h2 class="text-xs font-semibold text-muted uppercase tracking-wider mb-3">Local Eval Results (must beat 0.8695 to submit)</h2>
  <div class="overflow-x-auto">
    <table class="w-full text-sm">
      <thead><tr class="text-muted text-[10px] uppercase tracking-wider border-b border-border">
        <th class="text-left py-2 px-2">Variant</th>
        <th class="text-right py-2 px-2">Combined</th>
        <th class="text-right py-2 px-2">Det mAP</th>
        <th class="text-right py-2 px-2">Cls mAP</th>
        <th class="text-right py-2 px-2">Runtime</th>
        <th class="text-right py-2 px-2">Est. Test</th>
      </tr></thead>
      <tbody id="evals"></tbody>
    </table>
  </div>
</div>

<!-- Bottom Row -->
<div class="grid grid-cols-2 gap-4">
  <!-- Submissions -->
  <div class="bg-card border border-border rounded-xl p-5">
    <h2 class="text-xs font-semibold text-muted uppercase tracking-wider mb-3">Submission History</h2>
    <div id="subs" class="space-y-2"></div>
  </div>
  <!-- Leaderboard -->
  <div class="bg-card border border-border rounded-xl p-5">
    <h2 class="text-xs font-semibold text-muted uppercase tracking-wider mb-3">Leaderboard</h2>
    <div id="lb" class="space-y-1"></div>
  </div>
</div>
</div>

<script>
// Linear calibration from 2 data points: test = 0.2562 * val + 0.5186
const EST_A = 0.2562;
const EST_B = 0.5186;
let ws, reconnect;

function connect() {
  ws = new WebSocket('ws://localhost:8051');
  ws.onopen = () => { document.getElementById('ws-dot').className='w-2 h-2 rounded-full bg-accent pulse'; clearInterval(reconnect); reconnect=null; };
  ws.onclose = () => { document.getElementById('ws-dot').className='w-2 h-2 rounded-full bg-danger'; if(!reconnect) reconnect=setInterval(connect,3000); };
  ws.onmessage = e => render(JSON.parse(e.data));
}

function render(d) {
  document.getElementById('time').textContent = d.last_update||'—';

  // Scores
  const testScore = d.best_test_score||0;
  const valScore = d.best_val_score||0;
  const leader = d.leader_score||0.7694;
  document.getElementById('test-score').textContent = testScore.toFixed(4);
  document.getElementById('val-score').textContent = valScore.toFixed(4);
  document.getElementById('leader').textContent = leader.toFixed(4);
  document.getElementById('gap').textContent = (leader - testScore).toFixed(4);
  document.getElementById('ratio').textContent = '0.256'; // slope: only 25.6% of val gains transfer to test

  // Reset timer
  const now = new Date();
  const utc = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()+1));
  const diff = utc - now;
  document.getElementById('reset').textContent = `${Math.floor(diff/3600000)}h ${Math.floor((diff%3600000)/60000)}m`;

  // VMs
  document.getElementById('vms').innerHTML = (d.vms||[]).map(vm => {
    const gpuPct = vm.gpu_total > 0 ? (vm.gpu_used/vm.gpu_total*100) : 0;
    const mc = vm.map50 > 0.9 ? 'text-accent' : vm.map50 > 0.5 ? 'text-warn' : 'text-muted';
    const status = vm.running ? '<span class="text-accent text-[10px]"><span class="inline-block w-1.5 h-1.5 rounded-full bg-accent pulse mr-1"></span>Running</span>' : vm.error ? '<span class="text-danger text-[10px]">Error</span>' : '<span class="text-muted text-[10px]">Idle</span>';
    return `<div class="bg-card border border-border rounded-xl p-4">
      <div class="flex items-center justify-between mb-3"><span class="text-xs font-semibold">${vm.label}</span>${status}</div>
      <div class="grid grid-cols-3 gap-2 mb-2">
        <div><p class="text-muted text-[9px]">mAP@0.5</p><p class="text-lg font-bold mono ${mc}">${(vm.map50||0).toFixed(3)}</p></div>
        <div><p class="text-muted text-[9px]">Epoch</p><p class="text-lg font-bold mono">${vm.epoch||0}</p></div>
        <div><p class="text-muted text-[9px]">GPU</p><p class="text-lg font-bold mono">${gpuPct.toFixed(0)}%</p></div>
      </div>
      <div class="w-full h-1 bg-bg rounded-full overflow-hidden"><div class="h-full rounded-full ${gpuPct>90?'bg-danger':gpuPct>50?'bg-warn':'bg-accent'}" style="width:${gpuPct}%"></div></div>
    </div>`;
  }).join('');

  // Evals
  const baseline = 0.8695;
  document.getElementById('evals').innerHTML = (d.evals||[]).map((e,i) => {
    const est = (EST_A * e.combined + EST_B).toFixed(4);
    const cls = e.combined > baseline ? 'text-accent' : 'text-danger';
    const best = i === 0 ? 'bg-accent/5' : '';
    return `<tr class="${best} border-b border-border/50 hover:bg-bg/50">
      <td class="py-2 px-2 font-medium ${i===0?'text-accent':''}">${e.name||'—'}${i===0?' ★':''}</td>
      <td class="py-2 px-2 text-right mono font-bold ${cls}">${e.combined.toFixed(4)}</td>
      <td class="py-2 px-2 text-right mono">${(e.det_mAP||0).toFixed(4)}</td>
      <td class="py-2 px-2 text-right mono">${(e.cls_mAP||0).toFixed(4)}</td>
      <td class="py-2 px-2 text-right mono text-muted">${(e.runtime||0).toFixed(0)}s</td>
      <td class="py-2 px-2 text-right mono ${parseFloat(est)>0.77?'text-accent':'text-warn'}">${est}</td>
    </tr>`;
  }).join('');

  // Submissions
  document.getElementById('subs').innerHTML = (d.submissions||[]).map(s => {
    const c = (s.test_score||0) > 0.7 ? 'text-accent' : (s.test_score||0) > 0.5 ? 'text-warn' : 'text-danger';
    return `<div class="flex items-center justify-between p-2 bg-bg rounded-lg">
      <div><span class="text-xs font-medium">${s.version}</span><span class="text-muted text-[10px] ml-2">${s.model}</span></div>
      <span class="mono text-sm font-bold ${c}">${s.test_score??'—'}</span></div>`;
  }).join('');

  // Leaderboard
  document.getElementById('lb').innerHTML = (d.leaderboard||[]).slice(0,15).map(e => {
    const us = e.team && e.team.includes('Paralov');
    return `<div class="flex items-center justify-between py-1 px-2 rounded ${us?'bg-accent/10':'hover:bg-bg/50'}">
      <div class="flex items-center gap-2"><span class="text-muted text-[10px] mono w-5">#${e.rank}</span><span class="text-xs ${us?'font-bold text-accent':''}">${e.team}</span></div>
      <span class="mono text-xs font-medium ${us?'text-accent':''}">${e.score.toFixed(4)}</span></div>`;
  }).join('');
}

connect();
// Also poll state.json as fallback
setInterval(async () => {
  try { const r = await fetch('/api/state'); render(await r.json()); } catch(e) {}
}, 20000);
</script>
</body></html>"""


async def ws_handler(websocket):
    ws_clients.add(websocket)
    try:
        await websocket.send(json.dumps(read_state()))
        async for _ in websocket:
            pass
    finally:
        ws_clients.discard(websocket)


async def ws_server():
    async with websockets.serve(ws_handler, "localhost", 8051):
        await asyncio.Future()


def push_loop():
    global ws_clients
    while ws_loop is None:
        time.sleep(1)
    while True:
        msg = json.dumps(read_state())
        dead = set()
        for c in ws_clients.copy():
            try:
                asyncio.run_coroutine_threadsafe(c.send(msg), ws_loop)
            except:
                dead.add(c)
        ws_clients -= dead
        time.sleep(15)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(read_state()).encode())
        else:
            self.send_error(404)
    def log_message(self, *a): pass


def run_ws():
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    ws_loop.run_until_complete(ws_server())


if __name__ == "__main__":
    threading.Thread(target=lambda: HTTPServer(("localhost", 8050), Handler).serve_forever(), daemon=True).start()
    threading.Thread(target=run_ws, daemon=True).start()
    threading.Thread(target=push_loop, daemon=True).start()
    print("Dashboard: http://localhost:8050")
    webbrowser.open("http://localhost:8050")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
