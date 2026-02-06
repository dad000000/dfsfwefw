from __future__ import annotations

import time
import threading

# Process-wide monotonic wall-clock milliseconds.
# Uses time.time() but clamps to a strictly non-decreasing sequence to protect against
# system clock adjustments (e.g., NTP stepping backwards).
_lock = threading.Lock()
_last_ms: int = 0


def now_ms() -> int:
    global _last_ms
    ms = int(time.time() * 1000)
    with _lock:
        if ms <= _last_ms:
            ms = _last_ms + 1
        _last_ms = ms
    return ms

