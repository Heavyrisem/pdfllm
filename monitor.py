"""
pdfllm MCP 모니터링 대시보드

stdio MCP 프로토콜을 유지하면서 별도 HTTP 서버(:7337)로
도구 호출 현황을 실시간으로 확인할 수 있는 대시보드를 제공합니다.

환경변수:
    PDFLLM_MONITOR_PORT: 대시보드 포트 (기본 7337)
"""

from __future__ import annotations

import base64
import functools
import json
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional


# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------

IMAGE_TOOLS = {"get_overview", "get_tile"}
MAX_CALLS = 100          # 유지할 최대 호출 이력
MAX_IMAGE_CALLS = 20     # 이미지 bytes를 유지할 최대 건수


@dataclass
class CallRecord:
    id: str
    tool: str
    params: str          # JSON 요약 (최대 200자)
    status: str          # "running" | "success" | "error"
    start_time: float
    end_time: Optional[float] = None
    error: Optional[str] = None
    image_bytes: Optional[bytes] = None

    def duration_ms(self) -> Optional[int]:
        if self.end_time is None:
            return None
        return int((self.end_time - self.start_time) * 1000)

    def elapsed_s(self) -> float:
        """현재까지 경과 시간 (초)."""
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def to_dict(self, include_image: bool = False) -> dict:
        d = {
            "id": self.id,
            "tool": self.tool,
            "params": self.params,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms(),
            "error": self.error,
            "has_image": self.image_bytes is not None,
        }
        if include_image and self.image_bytes:
            d["image_b64"] = base64.b64encode(self.image_bytes).decode()
        return d


# ---------------------------------------------------------------------------
# 싱글톤 상태
# ---------------------------------------------------------------------------

class MonitorState:
    def __init__(self) -> None:
        self.server_start: float = time.time()
        self._lock = threading.Lock()
        self.calls: deque[CallRecord] = deque(maxlen=MAX_CALLS)
        self.active_calls: dict[str, CallRecord] = {}
        self.tool_stats: dict[str, dict] = {}
        self._image_call_ids: deque[str] = deque(maxlen=MAX_IMAGE_CALLS)

    def begin_call(self, tool_name: str, kwargs: dict) -> str:
        call_id = uuid.uuid4().hex[:8]
        params = json.dumps(
            {k: v for k, v in kwargs.items() if k != "pdf_path" or len(kwargs) == 1},
            ensure_ascii=False,
            default=str,
        )[:200]
        record = CallRecord(
            id=call_id,
            tool=tool_name,
            params=params,
            status="running",
            start_time=time.time(),
        )
        with self._lock:
            self.active_calls[call_id] = record
            if tool_name not in self.tool_stats:
                self.tool_stats[tool_name] = {
                    "total": 0,
                    "success": 0,
                    "error": 0,
                    "total_ms": 0,
                }
        return call_id

    def end_call(
        self,
        call_id: str,
        status: str,
        error: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
    ) -> None:
        with self._lock:
            record = self.active_calls.pop(call_id, None)
            if record is None:
                return
            record.status = status
            record.end_time = time.time()
            record.error = error

            if image_bytes is not None:
                # 오래된 이미지 bytes 해제
                if len(self._image_call_ids) >= MAX_IMAGE_CALLS:
                    old_id = self._image_call_ids[0]
                    for c in self.calls:
                        if c.id == old_id:
                            c.image_bytes = None
                            break
                record.image_bytes = image_bytes
                self._image_call_ids.append(call_id)

            self.calls.append(record)

            stats = self.tool_stats.setdefault(record.tool, {
                "total": 0, "success": 0, "error": 0, "total_ms": 0,
            })
            stats["total"] += 1
            ms = record.duration_ms() or 0
            stats["total_ms"] += ms
            if status == "success":
                stats["success"] += 1
            else:
                stats["error"] += 1

    def get_status(self) -> dict:
        with self._lock:
            recent = list(reversed(list(self.calls)))[:50]
            stats_out = {}
            for tool, s in self.tool_stats.items():
                avg = int(s["total_ms"] / s["total"]) if s["total"] else 0
                stats_out[tool] = {
                    "total": s["total"],
                    "success": s["success"],
                    "error": s["error"],
                    "avg_ms": avg,
                }
            return {
                "uptime_s": int(time.time() - self.server_start),
                "active_calls": [r.to_dict() for r in self.active_calls.values()],
                "recent_calls": [r.to_dict(include_image=False) for r in recent],
                "tool_stats": stats_out,
                "total_calls": sum(s["total"] for s in self.tool_stats.values()),
                "tools": list(self.tool_stats.keys()),
            }

    def get_image(self, call_id: str) -> Optional[bytes]:
        with self._lock:
            for c in self.calls:
                if c.id == call_id:
                    return c.image_bytes
        return None


monitor_state = MonitorState()


# ---------------------------------------------------------------------------
# 데코레이터
# ---------------------------------------------------------------------------

def track_call(tool_name: str):
    """MCP 도구 함수를 감싸는 모니터링 데코레이터."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            call_id = monitor_state.begin_call(tool_name, kwargs)
            try:
                result = fn(*args, **kwargs)
                image_data = None
                if tool_name in IMAGE_TOOLS and hasattr(result, "data"):
                    image_data = result.data  # FastMCP Image.data = bytes
                monitor_state.end_call(call_id, status="success", image_bytes=image_data)
                return result
            except Exception as e:
                monitor_state.end_call(call_id, status="error", error=str(e))
                raise
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# 대시보드 HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pdfllm MCP Monitor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
  header {
    display: flex; align-items: center; justify-content: space-between;
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 10px 18px; position: sticky; top: 0; z-index: 10;
  }
  header h1 { font-size: 15px; font-weight: 600; color: #58a6ff; }
  .status-badge {
    display: flex; align-items: center; gap: 6px;
    background: #1c2a1e; border: 1px solid #2ea043; border-radius: 12px;
    padding: 3px 10px; color: #3fb950; font-size: 12px;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #3fb950; }
  .uptime { color: #8b949e; font-size: 12px; }
  .layout { display: flex; height: calc(100vh - 45px); overflow: hidden; }
  /* 좌측: 도구 통계 */
  .sidebar {
    width: 220px; min-width: 200px; background: #161b22;
    border-right: 1px solid #30363d; overflow-y: auto; padding: 10px 8px;
    flex-shrink: 0;
  }
  .sidebar h2 { font-size: 11px; color: #8b949e; text-transform: uppercase;
    letter-spacing: .05em; padding: 4px 6px 8px; }
  .tool-card {
    background: #0d1117; border: 1px solid #21262d; border-radius: 6px;
    padding: 8px 10px; margin-bottom: 6px; cursor: default;
  }
  .tool-card:hover { border-color: #388bfd; }
  .tool-name { font-weight: 600; color: #e6edf3; font-size: 12px; margin-bottom: 4px; }
  .tool-meta { color: #8b949e; font-size: 11px; line-height: 1.6; }
  .s-ok { color: #3fb950; }
  .s-err { color: #f85149; }
  /* 우측 */
  .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  /* 실행중 */
  .active-section {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 8px 14px; min-height: 42px;
  }
  .active-section h2 { font-size: 11px; color: #8b949e; text-transform: uppercase;
    letter-spacing: .05em; margin-bottom: 6px; }
  .active-row {
    display: flex; align-items: center; gap: 8px;
    background: #1c2a38; border: 1px solid #1f6feb; border-radius: 6px;
    padding: 5px 10px; margin-bottom: 4px;
  }
  .spinner { width: 12px; height: 12px; border: 2px solid #58a6ff33;
    border-top-color: #58a6ff; border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .active-tool { color: #58a6ff; font-weight: 600; font-size: 12px; }
  .active-params { color: #8b949e; font-size: 11px; flex: 1; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
  .active-elapsed { color: #d29922; font-size: 11px; white-space: nowrap; }
  /* 로그 */
  .log-section { flex: 1; overflow-y: auto; padding: 10px 14px; }
  .log-section h2 { font-size: 11px; color: #8b949e; text-transform: uppercase;
    letter-spacing: .05em; margin-bottom: 8px; }
  .call-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 8px 12px; margin-bottom: 6px;
  }
  .call-card.err { border-color: #6e2c2c; }
  .call-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
  .call-icon { font-size: 13px; }
  .call-tool { font-weight: 600; color: #e6edf3; font-size: 12px; }
  .call-dur { color: #8b949e; font-size: 11px; }
  .call-ago { color: #6e7681; font-size: 11px; margin-left: auto; }
  .call-params { color: #8b949e; font-size: 11px; margin-top: 2px; word-break: break-all; }
  .call-error { color: #f85149; font-size: 11px; margin-top: 4px; }
  .call-img { margin-top: 6px; max-width: 100%; border-radius: 4px;
    border: 1px solid #30363d; display: block; }
  .empty { color: #6e7681; font-size: 12px; padding: 8px 4px; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
</head>
<body>
<header>
  <h1>pdfllm MCP Monitor</h1>
  <span class="status-badge"><span class="dot"></span>Running <span class="uptime" id="uptime"></span></span>
</header>
<div class="layout">
  <aside class="sidebar">
    <h2>도구 통계</h2>
    <div id="tool-stats"></div>
  </aside>
  <main class="main">
    <div class="active-section">
      <h2>현재 실행 중</h2>
      <div id="active-calls"><span class="empty">실행 중인 도구 없음</span></div>
    </div>
    <div class="log-section">
      <h2>최근 호출</h2>
      <div id="recent-calls"></div>
    </div>
  </main>
</div>
<script>
const ALL_TOOLS = [
  "get_page_count","suggest_grid","get_overview","get_tile",
  "get_tile_as_pdf","get_tile_text","get_structure","find_cells"
];
const IMAGE_TOOLS = new Set(["get_overview","get_tile"]);
let cachedImages = {};  // call_id -> data URL

function fmtUptime(s) {
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60;
  if(h) return h+'h '+String(m).padStart(2,'0')+'m '+String(sec).padStart(2,'0')+'s';
  return String(m).padStart(2,'0')+'m '+String(sec).padStart(2,'0')+'s';
}
function ago(ts) {
  const d = Math.floor(Date.now()/1000 - ts);
  if(d<60) return d+'초 전';
  if(d<3600) return Math.floor(d/60)+'분 전';
  return Math.floor(d/3600)+'시간 전';
}
function escHtml(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function fetchImage(callId) {
  if(cachedImages[callId]) return cachedImages[callId];
  try {
    const r = await fetch('/api/image/'+callId);
    if(!r.ok) return null;
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    cachedImages[callId] = url;
    return url;
  } catch { return null; }
}

async function render(data) {
  // Uptime
  document.getElementById('uptime').textContent = fmtUptime(data.uptime_s);

  // Tool stats
  const stats = data.tool_stats || {};
  const statsHtml = ALL_TOOLS.map(t => {
    const s = stats[t];
    if(!s) return `<div class="tool-card"><div class="tool-name">${t}</div><div class="tool-meta" style="color:#6e7681">미사용</div></div>`;
    const avg = s.avg_ms ? s.avg_ms+'ms' : '-';
    return `<div class="tool-card">
      <div class="tool-name">${t}</div>
      <div class="tool-meta">
        호출: ${s.total} &nbsp;
        <span class="s-ok">✓${s.success}</span>
        ${s.error ? `<span class="s-err"> ✗${s.error}</span>` : ''}
        <br>평균: ${avg}
      </div>
    </div>`;
  }).join('');
  document.getElementById('tool-stats').innerHTML = statsHtml;

  // Active calls
  const active = data.active_calls || [];
  if(active.length === 0) {
    document.getElementById('active-calls').innerHTML = '<span class="empty">실행 중인 도구 없음</span>';
  } else {
    const now = Date.now()/1000;
    document.getElementById('active-calls').innerHTML = active.map(c => {
      const elapsed = (now - c.start_time).toFixed(1)+'s';
      return `<div class="active-row">
        <div class="spinner"></div>
        <span class="active-tool">${escHtml(c.tool)}</span>
        <span class="active-params">${escHtml(c.params)}</span>
        <span class="active-elapsed">${elapsed}</span>
      </div>`;
    }).join('');
  }

  // Recent calls
  const recent = data.recent_calls || [];
  if(recent.length === 0) {
    document.getElementById('recent-calls').innerHTML = '<span class="empty">호출 기록 없음</span>';
    return;
  }

  // 이미지 도구 결과 미리 fetch
  const imagePromises = recent
    .filter(c => c.has_image && !cachedImages[c.id])
    .map(c => fetchImage(c.id));
  await Promise.all(imagePromises);

  const cards = recent.map(c => {
    const icon = c.status === 'success' ? '✓' : c.status === 'error' ? '✗' : '⏳';
    const dur = c.duration_ms != null ? c.duration_ms+'ms' : '-';
    const errHtml = c.error ? `<div class="call-error">오류: ${escHtml(c.error)}</div>` : '';
    const imgUrl = cachedImages[c.id];
    const imgHtml = imgUrl ? `<img class="call-img" src="${imgUrl}" loading="lazy">` : '';
    return `<div class="call-card${c.status==='error'?' err':''}">
      <div class="call-header">
        <span class="call-icon ${c.status==='success'?'s-ok':c.status==='error'?'s-err':''}">${icon}</span>
        <span class="call-tool">${escHtml(c.tool)}</span>
        <span class="call-dur">${dur}</span>
        <span class="call-ago">${ago(c.start_time)}</span>
      </div>
      <div class="call-params">${escHtml(c.params)}</div>
      ${errHtml}
      ${imgHtml}
    </div>`;
  }).join('');
  document.getElementById('recent-calls').innerHTML = cards;
}

async function poll() {
  try {
    const r = await fetch('/api/status');
    if(r.ok) await render(await r.json());
  } catch {}
  setTimeout(poll, 1000);
}
poll();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP 서버
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 콘솔 로그 억제
        pass

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]

        if path == "/":
            body = DASHBOARD_HTML.encode()
            self._send(200, "text/html; charset=utf-8", body)

        elif path == "/api/status":
            data = monitor_state.get_status()
            body = json.dumps(data, ensure_ascii=False).encode()
            self._send(200, "application/json", body)

        elif path.startswith("/api/image/"):
            call_id = path[len("/api/image/"):]
            img = monitor_state.get_image(call_id)
            if img:
                self._send(200, "image/jpeg", img)
            else:
                self._send(404, "text/plain", b"not found")

        else:
            self._send(404, "text/plain", b"not found")


class MonitorServer:
    def __init__(self, port: Optional[int] = None) -> None:
        self.port = port or int(os.environ.get("PDFLLM_MONITOR_PORT", 7337))

    def start(self) -> None:
        try:
            httpd = HTTPServer(("", self.port), _Handler)
        except OSError as e:
            print(f"[monitor] 포트 {self.port} 바인드 실패: {e}", flush=True)
            return
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        print(f"[monitor] 대시보드: http://localhost:{self.port}", flush=True)
