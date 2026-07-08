"""Inline OpenCV window showing rendered RGB + depth straight from this
node's own render output -- bypasses ROS transport/image_transport/RViz/
rqt entirely, so it isolates whether a smoothness problem is in this
node's own render/publish path or downstream of it (see CLAUDE.md's
phase-1 "publish exactly what a real camera driver would" framing --
RViz/rqt are consumers, not part of what this node is responsible for).
Debug-only: never touches what's actually published, only what's already
in hand right after rendering.

All GUI calls (imshow/waitKey/destroyWindow) happen on one dedicated
thread (_run), never whatever thread calls show(). This build's cv2
highgui backend is Qt-based (the "QSettings::value: Empty key passed"
warning it prints on first use is the tell), and Qt windows require every
call to stay on the thread that created them -- a MultiThreadedExecutor
can (and does) dispatch the render timer callback to a different worker
thread across frames, which crashed this the first time it called cv2
directly from show(). show() just hands the frame off through a
single-slot queue; the display thread owns the window for its entire
life.
"""
from __future__ import annotations

import queue
import threading

import cv2
import numpy as np

_WINDOW_NAME = "gs_sensors debug view (RGB | depth)"

_frame_queue: queue.Queue = queue.Queue(maxsize=1)
_thread: threading.Thread | None = None
_stop = threading.Event()


def _compose(rgb: np.ndarray, depth: np.ndarray | None, max_depth_m: float) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if depth is None:
        return bgr
    clipped = np.clip(depth, 0.0, max_depth_m)
    depth_u8 = (clipped / max_depth_m * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    return np.hstack([bgr, depth_color])


def _run() -> None:
    # 0.1s poll timeout so close() (which just sets _stop) is noticed
    # promptly without needing a sentinel value pushed through the queue.
    while not _stop.is_set():
        try:
            rgb, depth, max_depth_m = _frame_queue.get(timeout=0.1)
        except queue.Empty:
            cv2.waitKey(1)  # keep pumping the GUI event loop while idle
            continue
        cv2.imshow(_WINDOW_NAME, _compose(rgb, depth, max_depth_m))
        cv2.waitKey(1)
    cv2.destroyAllWindows()


def show(rgb: np.ndarray, depth: np.ndarray | None, max_depth_m: float) -> None:
    """Safe to call from any thread. Starts the display thread lazily on
    first use (fine here specifically because the render timer callback
    that calls this is single-flight per its own callback group -- this
    isn't general-purpose concurrent-safe lazy-init). Drops the frame
    instead of blocking if the display thread hasn't consumed the
    previous one yet -- a debug view always wants the newest frame, not a
    backlog."""
    global _thread
    if _thread is None:
        _stop.clear()
        _thread = threading.Thread(target=_run, name="gs_sensors_debug_view", daemon=True)
        _thread.start()
    if _frame_queue.full():
        try:
            _frame_queue.get_nowait()
        except queue.Empty:
            pass
    try:
        _frame_queue.put_nowait((rgb.copy(), None if depth is None else depth.copy(), max_depth_m))
    except queue.Full:
        pass


def close() -> None:
    """Safe to call even if show() was never called."""
    global _thread
    if _thread is None:
        return
    _stop.set()
    _thread.join(timeout=2.0)
    _thread = None
