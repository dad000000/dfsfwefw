from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .time_ms import now_ms

log = logging.getLogger("utils.ndjson_recorder")


def _ts_name() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


@dataclass
class RecorderStats:
    enabled: bool
    path: str
    lines: int = 0
    bytes: int = 0
    dropped: int = 0
    last_write_ts_ms: int = 0
    started_ts_ms: int = 0
    rotated_parts: int = 0
    last_error: str = ""


class NDJSONRecorder:
    """Asynchronous NDJSON recorder.

    Designed for offline replay (Phase 2). Safe defaults:
      - bounded queue (drops on overflow; observable)
      - periodic flush (no per-line fsync)
      - optional size-based rotation

    NOTE: This is local disk I/O only; never touches Binance.
    """

    def __init__(
        self,
        path: str,
        *,
        enabled: bool = True,
        flush_ms: int = 2000,
        queue_max: int = 5000,
        rotate_max_mb: int = 256,
    ) -> None:
        self.enabled = bool(enabled)
        self._base_path = str(path)
        self._flush_ms = max(200, int(flush_ms))
        self._queue_max = max(100, int(queue_max))
        self._rotate_max_bytes = max(0, int(rotate_max_mb)) * 1024 * 1024

        self._q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=self._queue_max)
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

        self._part: int = 0
        self._path: str = self._base_path

        self._lines: int = 0
        self._bytes: int = 0
        self._dropped: int = 0
        self._last_write_ts_ms: int = 0
        self._started_ts_ms: int = 0
        self._last_error: str = ""

    @staticmethod
    def make_default_path(rec_dir: str, symbol: str) -> str:
        sym = str(symbol or "SYMBOL").upper()
        base = f"recorder_{sym}_{_ts_name()}.ndjson"
        return os.path.join(str(rec_dir), base)

    def path(self) -> str:
        return str(self._path)

    def stats(self) -> RecorderStats:
        return RecorderStats(
            enabled=bool(self.enabled),
            path=str(self._path),
            lines=int(self._lines),
            bytes=int(self._bytes),
            dropped=int(self._dropped),
            last_write_ts_ms=int(self._last_write_ts_ms),
            started_ts_ms=int(self._started_ts_ms),
            rotated_parts=int(self._part),
            last_error=str(self._last_error or ""),
        )

    def meta_dict(self) -> Dict[str, Any]:
        st = self.stats()
        # keep this flat for dashboard
        return {
            "recorder_enabled": bool(st.enabled),
            "recorder_path": str(st.path),
            "recorder_lines": int(st.lines),
            "recorder_bytes": int(st.bytes),
            "recorder_dropped": int(st.dropped),
            "recorder_queue_len": int(self._q.qsize()) if self.enabled else 0,
            "recorder_last_write_ts_ms": int(st.last_write_ts_ms),
            "recorder_started_ts_ms": int(st.started_ts_ms),
            "recorder_rotated_parts": int(st.rotated_parts),
            "recorder_last_error": str(st.last_error or ""),
        }

    async def start(self) -> None:
        if not self.enabled:
            return
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._started_ts_ms = int(now_ms())
        self._last_error = ""

        # ensure directory exists
        d = os.path.dirname(os.path.abspath(self._base_path))
        if d:
            os.makedirs(d, exist_ok=True)

        # initial path
        self._part = 0
        self._path = self._base_path
        self._task = asyncio.create_task(self._writer_loop(), name="ndjson_recorder")

    async def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except Exception:
                # best effort
                try:
                    self._task.cancel()
                except Exception:
                    pass

    def record(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if not isinstance(event, dict):
            return
        if "ts_ms" not in event:
            event["ts_ms"] = int(now_ms())
        try:
            self._q.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1

    def _rotate_path(self) -> str:
        self._part += 1
        root, ext = os.path.splitext(self._base_path)
        if not ext:
            ext = ".ndjson"
        return f"{root}_part{self._part}{ext}"

    async def _writer_loop(self) -> None:
        f = None
        last_flush = time.monotonic()
        try:
            f = open(self._path, "a", encoding="utf-8")
            while not self._stop.is_set():
                try:
                    ev = await asyncio.wait_for(self._q.get(), timeout=self._flush_ms / 1000.0)
                    line = json.dumps(ev, ensure_ascii=False, separators=(",", ":"))
                    f.write(line + "\n")
                    self._lines += 1
                    self._bytes += len(line) + 1
                    self._last_write_ts_ms = int(ev.get("ts_ms") or 0)
                except asyncio.TimeoutError:
                    # periodic flush
                    pass
                except Exception as e:
                    self._last_error = f"write_error:{e}"[:240]

                # flush timer
                now = time.monotonic()
                if (now - last_flush) >= (self._flush_ms / 1000.0):
                    try:
                        f.flush()
                    except Exception as e:
                        self._last_error = f"flush_error:{e}"[:240]
                    last_flush = now

                # rotation
                if self._rotate_max_bytes > 0 and self._bytes >= self._rotate_max_bytes:
                    try:
                        f.flush()
                    except Exception:
                        pass
                    try:
                        f.close()
                    except Exception:
                        pass
                    self._bytes = 0
                    self._path = self._rotate_path()
                    f = open(self._path, "a", encoding="utf-8")

            # drain remaining quickly
            drain_deadline = time.monotonic() + 0.5
            while time.monotonic() < drain_deadline and not self._q.empty():
                try:
                    ev = self._q.get_nowait()
                    line = json.dumps(ev, ensure_ascii=False, separators=(",", ":"))
                    f.write(line + "\n")
                    self._lines += 1
                    self._bytes += len(line) + 1
                    self._last_write_ts_ms = int(ev.get("ts_ms") or 0)
                except Exception:
                    break
            try:
                f.flush()
            except Exception:
                pass
        except Exception as e:
            self._last_error = f"recorder_fatal:{e}"[:240]
            log.warning("recorder_fatal err=%s", e)
        finally:
            try:
                if f:
                    f.close()
            except Exception:
                pass

