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
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

IMAGE_TOOLS = {"get_region"}
ALL_TOOLS = ["get_page_count", "detect_layout", "get_region"]
MAX_CALLS = 100       # 유지할 최대 호출 이력
MAX_IMAGE_CALLS = 20  # 이미지 bytes를 유지할 최대 건수

_ASSETS_DIR = Path(__file__).parent / "assets"


def _read_asset(filename: str) -> bytes:
    return (_ASSETS_DIR / filename).read_bytes()


# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------


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
    bbox_pt: Optional[list] = None       # get_region bbox
    page_size_pt: Optional[list] = None  # 페이지 전체 크기

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
        if self.bbox_pt:
            d["bbox_pt"] = self.bbox_pt
            d["page_size_pt"] = self.page_size_pt
        return d


# ---------------------------------------------------------------------------
# 싱글톤 상태
# ---------------------------------------------------------------------------


class MonitorState:
    def __init__(self) -> None:
        self.server_start: float = time.time()
        self._lock = threading.Lock()
        self.calls: deque[CallRecord] = deque(maxlen=MAX_CALLS)
        self._call_by_id: dict[str, CallRecord] = {}  # O(1) 조회용
        self.active_calls: dict[str, CallRecord] = {}
        self.tool_stats: dict[str, dict] = {}
        self._image_call_ids: deque[str] = deque(maxlen=MAX_IMAGE_CALLS)
        self.instance_id = f"pid_{os.getpid()}"
        self.remote_instances: dict[str, dict] = {}  # instance_id → snapshot
        self.remote_images: dict[str, bytes] = {}    # call_id → jpeg bytes
        self._remote_lock = threading.Lock()

    def begin_call(self, tool_name: str, kwargs: dict) -> str:
        call_id = uuid.uuid4().hex[:8]
        params = json.dumps(
            {k: v for k, v in kwargs.items() if k != "pdf_path" or len(kwargs) == 1},
            ensure_ascii=False,
            default=str,
        )[:200]

        bbox_pt = None
        page_size_pt = None
        if tool_name == "get_region" and "bbox_pt" in kwargs:
            bbox_pt = kwargs["bbox_pt"]
            try:
                import fitz
                pdf_path = kwargs.get("pdf_path", "")
                page_idx = kwargs.get("page_idx", 0)
                if pdf_path:
                    doc = fitz.open(pdf_path)
                    page = doc[page_idx]
                    page_size_pt = [page.rect.width, page.rect.height]
                    doc.close()
            except Exception:
                pass

        record = CallRecord(
            id=call_id,
            tool=tool_name,
            params=params,
            status="running",
            start_time=time.time(),
            bbox_pt=bbox_pt,
            page_size_pt=page_size_pt,
        )
        with self._lock:
            self.active_calls[call_id] = record
            if tool_name not in self.tool_stats:
                self.tool_stats[tool_name] = {
                    "total": 0,
                    "success": 0,
                    "error": 0,
                    "total_ms": 0,
                    "max_ms": 0,
                    "min_ms": None,
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
                # 오래된 이미지 bytes 해제 (O(1) dict 조회)
                if len(self._image_call_ids) >= MAX_IMAGE_CALLS:
                    old_id = self._image_call_ids[0]
                    old_rec = self._call_by_id.get(old_id)
                    if old_rec:
                        old_rec.image_bytes = None
                record.image_bytes = image_bytes
                self._image_call_ids.append(call_id)

            # deque 만료 전 _call_by_id 정리
            if len(self.calls) >= MAX_CALLS:
                oldest = self.calls[0]
                self._call_by_id.pop(oldest.id, None)

            self.calls.append(record)
            self._call_by_id[call_id] = record

            stats = self.tool_stats.setdefault(
                record.tool,
                {
                    "total": 0,
                    "success": 0,
                    "error": 0,
                    "total_ms": 0,
                    "max_ms": 0,
                    "min_ms": None,
                },
            )
            stats["total"] += 1
            ms = record.duration_ms() or 0
            stats["total_ms"] += ms
            if stats["min_ms"] is None or ms < stats["min_ms"]:
                stats["min_ms"] = ms
            if ms > stats["max_ms"]:
                stats["max_ms"] = ms
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
                    "min_ms": s.get("min_ms"),
                    "max_ms": s.get("max_ms", 0),
                }
            return {
                "uptime_s": int(time.time() - self.server_start),
                "active_calls": [r.to_dict() for r in self.active_calls.values()],
                "recent_calls": [r.to_dict(include_image=False) for r in recent],
                "tool_stats": stats_out,
                "total_calls": sum(s["total"] for s in self.tool_stats.values()),
                "tools": list(self.tool_stats.keys()),
                "all_tools": ALL_TOOLS,
            }

    def update_remote(self, instance_id: str, snapshot: dict) -> None:
        """Reporter에서 받은 상태 스냅샷 저장."""
        with self._remote_lock:
            snapshot["last_seen"] = time.time()
            self.remote_instances[instance_id] = snapshot
            for call in snapshot.get("recent_calls", []):
                if call.get("image_b64"):
                    self.remote_images[call["id"]] = base64.b64decode(
                        call.pop("image_b64")
                    )

    def get_all_status(self) -> dict:
        """로컬 + 원격 인스턴스 통합 상태 반환."""
        local = self.get_status()
        local["instance_id"] = self.instance_id
        with self._remote_lock:
            alive = {
                iid: s
                for iid, s in self.remote_instances.items()
                if time.time() - s.get("last_seen", 0) < 5
            }
        return {
            "self_instance_id": self.instance_id,
            "instances": {self.instance_id: local, **alive},
        }

    def get_image(self, call_id: str) -> Optional[bytes]:
        with self._lock:
            rec = self._call_by_id.get(call_id)
            if rec:
                return rec.image_bytes
        with self._remote_lock:
            return self.remote_images.get(call_id)


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
                monitor_state.end_call(
                    call_id, status="success", image_bytes=image_data
                )
                return result
            except Exception as e:
                monitor_state.end_call(call_id, status="error", error=str(e))
                raise

        return wrapper

    return decorator


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

    def do_POST(self):  # noqa: N802
        if self.path == "/api/report":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            monitor_state.update_remote(body["instance_id"], body["state"])
            self._send(200, "application/json", b'{"ok":true}')
        else:
            self._send(404, "text/plain", b"not found")

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]

        if path == "/":
            self._send(200, "text/html; charset=utf-8", _read_asset("dashboard.html"))

        elif path == "/dashboard.css":
            self._send(200, "text/css; charset=utf-8", _read_asset("dashboard.css"))

        elif path == "/dashboard.js":
            self._send(200, "application/javascript; charset=utf-8", _read_asset("dashboard.js"))

        elif path == "/api/status":
            data = monitor_state.get_all_status()
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
        except OSError:
            self._start_reporter()
            return
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        print(f"[monitor] 대시보드: http://localhost:{self.port}", flush=True)

    def _start_reporter(self) -> None:
        master_url = f"http://localhost:{self.port}/api/report"
        instance_id = monitor_state.instance_id
        print(f"[monitor] Reporter 모드 → {master_url}", flush=True)

        def _loop():
            import urllib.request

            while True:
                try:
                    state = monitor_state.get_status()
                    for call in state["recent_calls"]:
                        if call.get("has_image"):
                            img = monitor_state.get_image(call["id"])
                            if img:
                                call["image_b64"] = base64.b64encode(img).decode()
                    payload = json.dumps(
                        {"instance_id": instance_id, "state": state},
                        ensure_ascii=False,
                    ).encode()
                    req = urllib.request.Request(
                        master_url,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    urllib.request.urlopen(req, timeout=1)
                except Exception:
                    pass
                time.sleep(1)

        threading.Thread(target=_loop, daemon=True).start()
