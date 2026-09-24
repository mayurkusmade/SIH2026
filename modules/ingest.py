"""
Live Ingestion Layer for the AnantaNetra / IBVAP edge grid.

Why this exists
---------------
The first iteration decoded one local MP4 inside the Streamlit render loop. That
made three of the platform's core claims untrue:

  * "Retrofits existing RTSP/ONVIF CCTV" - there was no RTSP ingest at all.
  * "Multi-camera grid" - one feed at a time, with a shared tracker that reset
    on every channel switch.
  * "Zero-latency, operator-fatigue-free" - the UI thread ran YOLO inline, so one
    viewer blocked the frame and a live camera would drift further behind reality
    with every passing second.

This module fixes all three with a two-stage thread pipeline per camera:

    [CameraWorker]     grab thread  -->  LatestFrame (depth-1 buffer)
                                              |
    [AnalysisWorker]   inference thread -->  annotated frame + EventBus
                                              |
                                        [StreamManager]  --> UI / logger / uplink

Design rules that matter operationally:

  * **Depth-1 buffer, drop-old semantics.** Inference consumes the newest frame
    and discards what it missed. On a live border feed, processing a stale frame
    is worse than skipping it - latency must never accumulate.
  * **Never block on I/O.** Every camera gets its own grab thread, so one dead
    camera cannot stall the grid.
  * **Self-healing.** A camera that drops, stalls, or is unplugged reconnects
    with exponential backoff and is reported as a first-class health state.
  * **Bounded inference concurrency.** N cameras share a fixed permit pool, so a
    4-camera grid cannot oversubscribe edge CPU.

OpenCV is imported lazily inside the capture-backed source, so this module stays
importable (and testable) on a machine without the vision stack.
"""

import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------
LIVE_SCHEMES = ("rtsp://", "rtmp://", "rtsps://", "http://", "https://", "udp://", "tcp://")
# Deliberately narrow: generic path noise ("profile2", "stream1") is NOT an ONVIF
# signal, and mislabelling a plain RTSP camera as ONVIF would be a lie in the UI.
ONVIF_HINTS = ("onvif", "/onvif", "device_service", "media_service", "axis-media", "media.amp")


@dataclass(frozen=True)
class SourceSpec:
    """Everything needed to open one camera feed."""

    uri: str
    kind: str  # file | rtsp | onvif | webcam | synthetic
    is_live: bool
    label: str

    def describe(self) -> str:
        return f"{self.kind}:{self.label}"


def classify_source(uri: Any) -> SourceSpec:
    """
    Classifies a source string into an ingest spec.

    Accepts local files, RTSP/RTMP/UDP URLs (what IP cameras and ONVIF device
    streams actually expose), integer-ish webcam device indexes, and a
    'synthetic:' pseudo-URI used for demos and verification.
    """
    text = str(uri or "").strip()

    if not text:
        return SourceSpec("", "file", False, "unspecified")

    lower = text.lower()

    if lower.startswith("synthetic"):
        return SourceSpec(text, "synthetic", True, "synthetic test feed")

    if lower.startswith("rtsp://") or lower.startswith("rtsps://"):
        kind = "onvif" if any(hint in lower for hint in ONVIF_HINTS) else "rtsp"
        return SourceSpec(text, kind, True, _safe_label(text))

    if any(lower.startswith(scheme) for scheme in LIVE_SCHEMES):
        return SourceSpec(text, "onvif" if "onvif" in lower else "rtsp", True, _safe_label(text))

    if text.isdigit():
        return SourceSpec(text, "webcam", True, f"device {text}")

    # Anything else is treated as a local media file.
    return SourceSpec(text, "file", False, os.path.basename(text) or "file")


def _safe_label(uri: str) -> str:
    """Labels a camera URL for display without leaking embedded credentials."""
    if "@" in uri and "://" in uri:
        scheme, rest = uri.split("://", 1)
        host = rest.split("@", 1)[1]
        return f"{scheme}://***@{host}"
    return uri


def source_from_camera(camera: dict) -> SourceSpec:
    """
    Resolves a registry camera entry into an ingest spec.

    A deployment overrides `video_source` with the real RTSP URL; `rtsp_url`
    takes precedence when present so a field URL can be injected without editing
    the registry.
    """
    return classify_source(
        camera.get("rtsp_url") or camera.get("video_source") or ""
    )


# ---------------------------------------------------------------------------
# Reconnection policy
# ---------------------------------------------------------------------------
@dataclass
class ReconnectPolicy:
    """Exponential backoff with jitter for a camera that will not come back."""

    initial_s: float = 0.5
    factor: float = 2.0
    max_s: float = 15.0
    jitter: float = 0.2

    def delay_for(self, attempt: int, rng: Optional[random.Random] = None) -> float:
        base = min(self.max_s, self.initial_s * (self.factor ** max(0, attempt - 1)))
        if self.jitter <= 0:
            return base
        rng = rng or random
        spread = base * self.jitter
        return max(0.0, base + rng.uniform(-spread, spread))


# ---------------------------------------------------------------------------
# Depth-1 frame buffer
# ---------------------------------------------------------------------------
class LatestFrame:
    """
    Single-slot frame buffer: producers overwrite, consumers take the newest.

    This is the anti-lag primitive. A queue would preserve every frame and build
    unbounded latency; a depth-1 slot guarantees the analysis stage always works
    on the freshest available image.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._ts = 0.0
        self.dropped = 0
        self.published = 0

    def publish(self, frame, seq: int, ts: Optional[float] = None) -> None:
        with self._lock:
            if self._frame is not None:
                self.dropped += 1  # consumer had not taken the previous frame yet
            self._frame = frame
            self._seq = seq
            self._ts = ts if ts is not None else time.time()
            self.published += 1

    def take(self) -> Optional[Tuple[Any, int, float]]:
        """Returns (frame, seq, ts) or None. Clears the slot (drop-old)."""
        with self._lock:
            if self._frame is None:
                return None
            item = (self._frame, self._seq, self._ts)
            self._frame = None
            return item

    def peek(self) -> Optional[Tuple[Any, int, float]]:
        with self._lock:
            if self._frame is None:
                return None
            return (self._frame, self._seq, self._ts)

    def clear(self) -> None:
        with self._lock:
            self._frame = None

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    @property
    def age_s(self) -> float:
        with self._lock:
            if self._ts <= 0:
                return float("inf")
            return max(0.0, time.time() - self._ts)


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------
class FrameSource:
    """Base class for anything that can produce frames."""

    def __init__(self, spec: SourceSpec):
        self.spec = spec
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def read(self) -> Tuple[bool, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def release(self) -> None:  # pragma: no cover - overridden
        self._open = False

    def describe(self) -> str:
        return self.spec.describe()


class OpenCVFrameSource(FrameSource):
    """
    cv2.VideoCapture-backed source covering files, RTSP/ONVIF streams and USB
    capture devices.

    Live-stream specifics that matter in the field:
      * RTSP over TCP (UDP silently shreds packets on flaky links).
      * Open/read timeouts so a vanished camera cannot block a worker forever.
      * Buffer size 1 so ffmpeg does not accumulate seconds of stale video.
    """

    def __init__(
        self,
        spec: SourceSpec,
        capture_factory: Optional[Callable[[Any], Any]] = None,
        cv2_module: Any = None,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
        prefer_tcp: bool = True,
        loop_files: bool = True,
    ):
        super().__init__(spec)
        # Both seams are injectable so the RTSP/EOF/timeout logic can be verified
        # on a machine with neither OpenCV nor a camera attached.
        self._capture_factory = capture_factory
        self._cv2 = cv2_module
        self._capture = None
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.prefer_tcp = prefer_tcp
        self.loop_files = loop_files
        self.reopen_count = 0
        self.last_error = ""

    # -- lifecycle ----------------------------------------------------------
    def _make_capture(self, cv2):
        if self._capture_factory is not None:
            return self._capture_factory(self.spec.uri)
        target = int(self.spec.uri) if self.spec.kind == "webcam" else self.spec.uri
        return cv2.VideoCapture(target)

    def _set(self, cv2, capture, prop_name: str, value) -> None:
        prop = getattr(cv2, prop_name, None)
        if prop is None:
            return
        try:
            capture.set(prop, value)
        except Exception:
            pass

    def open(self) -> bool:
        cv2 = self._cv2
        if cv2 is None:
            try:
                import cv2  # noqa: F401  (deliberately late import)
            except Exception as exc:
                self.last_error = f"OpenCV unavailable: {exc}"
                return False

        if self.spec.is_live and self.prefer_tcp and self.spec.kind in ("rtsp", "onvif"):
            # Only applied for stream sources; never mutates file decoding.
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                "rtsp_transport;tcp|timeout;5000000",
            )

        try:
            capture = self._make_capture(cv2)
        except Exception as exc:
            self.last_error = str(exc)
            return False

        if capture is None:
            self.last_error = "capture factory returned None"
            return False

        # Live sources: refuse to buffer latency. Files: leave defaults alone.
        if self.spec.is_live:
            self._set(cv2, capture, "CAP_PROP_BUFFERSIZE", 1)
            self._set(cv2, capture, "CAP_PROP_OPEN_TIMEOUT_MSEC", self.open_timeout_ms)
            self._set(cv2, capture, "CAP_PROP_READ_TIMEOUT_MSEC", self.read_timeout_ms)

        try:
            opened = bool(capture.isOpened())
        except Exception:
            opened = False

        if not opened:
            try:
                capture.release()
            except Exception:
                pass
            self.last_error = "stream/file did not open"
            self._open = False
            return False

        self._capture = capture
        self._open = True
        self.reopen_count += 1
        self.last_error = ""
        return True

    def read(self) -> Tuple[bool, Any]:
        if not self._open or self._capture is None:
            return False, None
        try:
            ok, frame = self._capture.read()
        except Exception as exc:
            self.last_error = str(exc)
            return False, None

        if ok and frame is not None:
            return True, frame

        # File sources loop forever (demo behaviour); live sources must reconnect.
        if not self.spec.is_live and self.loop_files:
            try:
                cv2 = self._cv2
                if cv2 is None:
                    import cv2  # noqa: F401

                prop = getattr(cv2, "CAP_PROP_POS_FRAMES", None)
                if prop is not None:
                    self._capture.set(prop, 0)
                    ok, frame = self._capture.read()
                    if ok and frame is not None:
                        return True, frame
            except Exception:
                pass

        self.last_error = "end of stream"
        return False, None

    def release(self) -> None:
        capture = self._capture
        self._capture = None
        self._open = False
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass


class SyntheticFrameSource(FrameSource):
    """
    numpy-only frame generator: moving rectangles on a gradient.

    It exists so the ingest grid can be demonstrated and verified anywhere - no
    camera, no OpenCV, no model weights - and so reconnection/stall behaviour can
    be provoked deterministically in tests.
    """

    def __init__(
        self,
        spec: Optional[SourceSpec] = None,
        width: int = 640,
        height: int = 360,
        fps: float = 25.0,
        fail_first_n_opens: int = 0,
        fail_reads_after: Optional[int] = None,
        stall_after: Optional[int] = None,
        stall_s: float = 0.0,
        seed: int = 0,
    ):
        super().__init__(spec or classify_source("synthetic:0"))
        self.width = width
        self.height = height
        self.frame_interval = 1.0 / max(1.0, fps)
        # NOTE: counted per source INSTANCE, i.e. per open attempt. A worker that
        # rebuilds its source each retry therefore needs a shared counter (see the
        # flaky-camera case in test_phase5.py) rather than this knob alone.
        self.fail_first_n_opens = fail_first_n_opens
        self.fail_reads_after = fail_reads_after
        self.stall_after = stall_after
        self.stall_s = stall_s
        self._rng = random.Random(seed)
        self._open_attempts = 0
        self._frames_read = 0
        self._next_frame_at = 0.0
        self.last_error = ""

    def open(self) -> bool:
        self._open_attempts += 1
        if self._open_attempts <= self.fail_first_n_opens:
            self.last_error = "simulated open failure"
            self._open = False
            return False
        self._open = True
        self._next_frame_at = 0.0
        return True

    def read(self) -> Tuple[bool, Any]:
        if not self._open:
            return False, None

        if self.stall_after is not None and self._frames_read == self.stall_after:
            if self.stall_s:
                time.sleep(self.stall_s)
            # A stalled live feed: no data, no error - exactly like a camera that
            # lost radio link but never closed the socket.
            return False, None

        if self.fail_reads_after is not None and self._frames_read >= self.fail_reads_after:
            self.last_error = "simulated stream drop"
            return False, None

        # Model the camera's own cadence.
        now = time.time()
        if now < self._next_frame_at:
            time.sleep(min(0.05, self._next_frame_at - now))
        self._next_frame_at = time.time() + self.frame_interval

        try:
            import numpy as np
        except Exception as exc:
            self.last_error = f"numpy unavailable: {exc}"
            return False, None

        idx = self._frames_read
        self._frames_read += 1
        h, w = self.height, self.width
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(30, 90, w, dtype=np.uint8)
        frame[:, :, 1] = np.linspace(30, 70, w, dtype=np.uint8)
        frame[:, :, 2] = 40

        # Two moving "people"/"vehicle" blobs so detection has something to see.
        for i, (speed, size) in enumerate(((3.0, 24), (5.5, 40))):
            cx = int((10 + idx * speed + i * 120) % (w - size))
            cy = int(h * (0.45 + 0.2 * i))
            frame[cy:cy + size, cx:cx + size] = (200, 200, 200)
        return True, frame

    def release(self) -> None:
        self._open = False

    @property
    def frames_read(self) -> int:
        return self._frames_read


def build_frame_source(
    spec: SourceSpec,
    capture_factory: Optional[Callable[[Any], Any]] = None,
    cv2_module: Any = None,
    **kwargs
) -> FrameSource:
    """Factory: synthetic specs get the numpy generator, everything else OpenCV."""
    if spec.kind == "synthetic":
        return SyntheticFrameSource(spec, **{
            k: v for k, v in kwargs.items()
            if k in ("width", "height", "fps", "fail_first_n_opens",
                     "fail_reads_after", "stall_after", "stall_s", "seed")
        })
    return OpenCVFrameSource(
        spec, capture_factory=capture_factory, cv2_module=cv2_module
    )


# ---------------------------------------------------------------------------
# Ingestion worker
# ---------------------------------------------------------------------------
class WorkerState:
    STARTING = "STARTING"
    LIVE = "LIVE"
    STALLED = "STALLED"
    RECONNECTING = "RECONNECTING"
    STOPPED = "STOPPED"


class CameraWorker(threading.Thread):
    """
    One grab thread per camera.

    Owns its source, publishes into a depth-1 buffer, and self-heals on failure.
    It performs NO inference - that is the AnalysisWorker's job - so a slow model
    can never stop us from draining the camera's socket.
    """

    def __init__(
        self,
        camera_id: str,
        source_factory: Callable[[], FrameSource],
        policy: Optional[ReconnectPolicy] = None,
        stall_timeout_s: float = 3.0,
        max_fps: Optional[float] = None,
        seed: int = 0,
        node_id: str = "",
    ):
        super().__init__(name=f"ingest-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.node_id = node_id
        self._source_factory = source_factory
        self.policy = policy or ReconnectPolicy()
        self.stall_timeout_s = stall_timeout_s
        self.max_fps = max_fps
        self.buffer = LatestFrame()

        self._stop_evt = threading.Event()
        self._force_reconnect = threading.Event()
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

        self._state = WorkerState.STARTING
        self._source: Optional[FrameSource] = None
        self._frames = 0
        self._failures = 0
        self._reconnects = 0
        self._ever_live = False
        self._last_frame_at = 0.0
        self._started_at = time.time()
        self._fps_in = 0.0
        self._fps_window: Deque[float] = deque(maxlen=30)
        self.last_error = ""

    # -- introspection ------------------------------------------------------
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _set_state(self, state: str) -> None:
        with self._lock:
            self._state = state

    @property
    def last_frame_at(self) -> float:
        with self._lock:
            return self._last_frame_at

    @property
    def frames_read(self) -> int:
        with self._lock:
            return self._frames

    @property
    def reconnects(self) -> int:
        """Successful re-opens AFTER this camera had already been live.

        A camera that never dialled in successfully increments `failures`, not
        `reconnects` - the two mean different things to an operator: 'never came
        up' versus 'went down and came back'.
        """
        with self._lock:
            return self._reconnects

    def health(self) -> dict:
        with self._lock:
            state = self._state
            frames = self._frames
            reconnects = self._reconnects
            failures = self._failures
            last_frame_at = self._last_frame_at
            fps = self._fps_in
            ever_live = self._ever_live
            source_desc = self._source.describe() if self._source else ""
        age = (time.time() - last_frame_at) if last_frame_at else float("inf")
        return {
            "camera_id": self.camera_id,
            "node_id": self.node_id,
            "state": state,
            "source": source_desc,
            "frames_read": frames,
            "input_fps": round(fps, 1),
            "last_frame_age_s": round(age, 2) if age != float("inf") else None,
            "reconnects": reconnects,
            "failures": failures,
            "ever_live": ever_live,
            "buffer_dropped": self.buffer.dropped,
            "stall_timeout_s": self.stall_timeout_s,
            "last_error": self.last_error,
            "alive": self.is_alive(),
        }

    # -- control ------------------------------------------------------------
    def stop(self, timeout: float = 3.0) -> bool:
        self._stop_evt.set()
        return self.join(timeout=timeout) is None and not self.is_alive()

    def mark_stalled(self, reason: str = "no frames within stall timeout") -> None:
        """Flags the camera as stalled; used by the watchdog before recovery."""
        with self._lock:
            if self._state == WorkerState.LIVE:
                self._state = WorkerState.STALLED
                self.last_error = reason

    def force_reconnect(self) -> None:
        """
        Called by the watchdog when a live camera has gone quiet without erroring.

        Releasing the capture unblocks a read() wedged inside ffmpeg, which is the
        only way to recover a socket that is open but carrying no data.
        """
        self._force_reconnect.set()
        source = self._source
        if source is not None:
            try:
                source.release()
            except Exception:
                pass

    # -- thread body --------------------------------------------------------
    def _sleep_backoff(self, attempt: int) -> bool:
        """Sleeps before the next reconnect attempt. Returns False if stopped."""
        delay = self.policy.delay_for(attempt, self._rng)
        return not self._stop_evt.wait(delay)

    def run(self) -> None:
        attempt = 0
        last_fps_t = time.time()

        while not self._stop_evt.is_set():
            # (Re)establish the source when needed.
            if self._source is None or not self._source.is_open:
                if self._force_reconnect.is_set():
                    self._force_reconnect.clear()
                source = self._source_factory()
                self._source = source
                if not source.open():
                    attempt += 1
                    with self._lock:
                        self._failures += 1
                        self.last_error = getattr(source, "last_error", "open failed")
                        self._state = WorkerState.RECONNECTING
                    if not self._sleep_backoff(attempt):
                        break
                    continue
                if getattr(self, "_opened_once", False):
                    # Every successful (re)open after the first is a recovery.
                    with self._lock:
                        self._reconnects += 1
                self._opened_once = True
                attempt = 0
                with self._lock:
                    self._state = WorkerState.LIVE

            ok, frame = self._source.read()

            if not ok or frame is None:
                reason = getattr(self._source, "last_error", "no frame")
                try:
                    self._source.release()
                except Exception:
                    pass
                attempt += 1
                with self._lock:
                    self._failures += 1
                    self.last_error = reason
                    self._state = WorkerState.RECONNECTING
                self._source = None
                if not self._sleep_backoff(attempt):
                    break
                continue

            attempt = 0
            now = time.time()
            with self._lock:
                self._frames += 1
                self._last_frame_at = now
                self._ever_live = True
                self._state = WorkerState.LIVE
            self.buffer.publish(frame, self._frames, now)

            # Rolling input FPS so the UI can show a real camera rate.
            elapsed = now - last_fps_t
            if elapsed >= 0.5:
                with self._lock:
                    self._fps_window.append(self._frames / elapsed)
                    self._fps_in = sum(self._fps_window) / len(self._fps_window)
                last_fps_t = now
            elif not self._fps_window:
                # Provisional rate from uptime, so a freshly started camera never
                # displays a misleading 0.0 fps in the grid.
                uptime = max(1e-3, now - self._started_at)
                with self._lock:
                    self._fps_in = self._frames / uptime

            if self.max_fps:
                target = 1.0 / self.max_fps
                slack = target - (time.time() - now)
                if slack > 0:
                    self._stop_evt.wait(slack)

        self._set_state(WorkerState.STOPPED)
        if self._source is not None:
            try:
                self._source.release()
            except Exception:
                pass
            self._source = None


# ---------------------------------------------------------------------------
# Analysis worker
# ---------------------------------------------------------------------------
def encode_jpeg(frame, quality: int = 80) -> Optional[bytes]:
    """
    Encodes an annotated frame as JPEG *where the frame already is* - off the UI
    thread.

    Why this exists: Streamlit's `st.image(numpy_array)` re-encodes the array to
    PNG on every repaint and ships it as base64. Measured on this project's own
    feeds: 51 ms + ~2 MB per 1080p frame for PNG, versus 7 ms + ~200 KB for JPEG.
    Doing it here means the UI thread only forwards bytes it was handed, the
    operator's link carries a tenth of the traffic, and the analysis thread
    (which is idle waiting on the next frame anyway) pays the cost.

    Returns None when cv2 is unavailable or the encode fails, so the caller can
    fall back to handing the UI the raw array.
    """
    if frame is None or not getattr(frame, "shape", None):
        return None
    try:
        import cv2
    except Exception:
        return None
    try:
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:
            return None
        return buf.tobytes()
    except Exception:
        return None


class AnalysisWorker(threading.Thread):
    """
    One inference thread per camera.

    Pulls the newest frame from the buffer, runs the caller-supplied vision
    pipeline, and publishes the annotated result. Because it lives off the UI
    thread, the dashboard always renders - even when YOLO takes 120 ms.
    """

    def __init__(
        self,
        camera: dict,
        frame_buffer: LatestFrame,
        process_fn: Callable[[Any, dict], Tuple[Any, List[dict]]],
        bus: Optional["EventBus"] = None,
        budget: Optional[threading.Semaphore] = None,
        min_interval_s: float = 0.0,
        idle_timeout_s: float = 0.5,
        jpeg_quality: int = 80,
    ):
        camera_id = camera.get("camera_id", "CAM-UNKNOWN")
        super().__init__(name=f"analysis-{camera_id}", daemon=True)
        self.camera = dict(camera)
        self.camera_id = camera_id
        self.frame_buffer = frame_buffer
        self.process_fn = process_fn
        self.bus = bus
        self.budget = budget
        self.min_interval_s = min_interval_s
        self.idle_timeout_s = idle_timeout_s
        self.jpeg_quality = int(jpeg_quality)

        self.out_buffer = LatestFrame()
        # Encoded twin of out_buffer: the UI paints these bytes directly instead
        # of asking Streamlit to re-encode a full-resolution array on every tick.
        self.out_jpeg = LatestFrame()
        self._jpeg_bytes = 0
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._fusion_started_at = 0.0
        self._frames = 0
        self._drops = 0
        self._latency_ms = 0.0
        self._fps = 0.0
        self._fps_window: Deque[float] = deque(maxlen=30)
        self._last_processed_at = 0.0
        self.last_error = ""
        self.last_events = 0

    # -- introspection ------------------------------------------------------
    def health(self) -> dict:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "node_id": self.camera.get("node_id", ""),
                "frames_processed": self._frames,
                "analysis_fps": round(self._fps, 1),
                "latency_ms": round(self._latency_ms, 1),
                "skipped_frames": self._drops,
                "output_dropped": self.out_buffer.dropped,
                "frame_kb": round(self._jpeg_bytes / 1024.0, 1),
                "last_events": self.last_events,
                "last_error": self.last_error,
                "alive": self.is_alive(),
            }

    def stop(self, timeout: float = 3.0) -> bool:
        self._stop_evt.set()
        return self.join(timeout=timeout) is None and not self.is_alive()

    @property
    def frames_processed(self) -> int:
        with self._lock:
            return self._frames

    # -- thread body --------------------------------------------------------
    def run(self) -> None:
        last_fps_t = time.time()
        self._fusion_started_at = last_fps_t
        while not self._stop_evt.is_set():
            item = self.frame_buffer.take()
            if item is None:
                self._stop_evt.wait(0.01)
                continue

            frame, seq, ts = item
            acquired = True
            if self.budget is not None:
                # Bounded inference concurrency: keeps N cameras from thrashing
                # an edge CPU with N simultaneous YOLO passes.
                acquired = self.budget.acquire(timeout=self.idle_timeout_s)
                if not acquired:
                    with self._lock:
                        self._drops += 1
                    continue

            started = time.time()
            try:
                context = {
                    "camera_id": self.camera_id,
                    "node_id": self.camera.get("node_id", ""),
                    "frame_seq": seq,
                    "frame_ts": ts,
                    "latency_budget_ms": self.min_interval_s * 1000.0,
                }
                annotated, events = self.process_fn(frame, context)
            except Exception as exc:  # one bad frame must not kill the camera
                with self._lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                events, annotated = [], None
            finally:
                if self.budget is not None and acquired:
                    self.budget.release()

            elapsed = time.time() - started
            with self._lock:
                self._frames += 1
                self._last_processed_at = time.time()
                # Exponential moving average keeps the UI number readable.
                self._latency_ms = (
                    elapsed * 1000.0 if not self._latency_ms
                    else 0.8 * self._latency_ms + 0.2 * elapsed * 1000.0
                )
                self.last_events = len(events or [])

            if annotated is not None:
                self.out_buffer.publish(annotated, seq, ts)
                jpeg = encode_jpeg(annotated, quality=self.jpeg_quality)
                if jpeg:
                    self.out_jpeg.publish(jpeg, seq, ts)
                    self._jpeg_bytes = len(jpeg)

            for event in events or []:
                event.setdefault("camera_id", self.camera_id)
                event.setdefault("node_id", self.camera.get("node_id", ""))
                event.setdefault("frame_seq", seq)
                if self.bus is not None:
                    self.bus.publish(event)

            now = time.time()
            elapsed_window = now - last_fps_t
            if elapsed_window >= 0.5:
                with self._lock:
                    self._fps_window.append(self._frames / elapsed_window)
                    self._fps = sum(self._fps_window) / len(self._fps_window)
                last_fps_t = now
            elif not self._fps_window and self._fusion_started_at:
                uptime = max(1e-3, now - self._fusion_started_at)
                with self._lock:
                    self._fps = self._frames / uptime


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------
class EventBus:
    """
    Bounded, thread-safe alert queue feeding the logger and the telemetry uplink.

    Overflow is *counted*, never silent: a C2 operator must be able to see that
    the outpost shed events under load.
    """

    def __init__(self, capacity: int = 2000):
        self.capacity = int(capacity)
        self._lock = threading.Lock()
        self._items: Deque[dict] = deque()
        self.dropped = 0
        self.total = 0

    def publish(self, event: dict) -> bool:
        with self._lock:
            self.total += 1
            if len(self._items) >= self.capacity:
                self.dropped += 1
                return False
            self._items.append(event)
            return True

    def drain(self, max_items: int = 200) -> List[dict]:
        out: List[dict] = []
        with self._lock:
            while self._items and len(out) < max_items:
                out.append(self._items.popleft())
        return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# ---------------------------------------------------------------------------
# Stream manager
# ---------------------------------------------------------------------------
class StreamManager:
    """
    Owns the camera grid: start/stop, health, watchdog, and event fan-in.

    The watchdog is the piece that makes an unattended border deployment viable -
    it notices a silently stalled camera and forces a reconnect instead of
    waiting for a human to spot a frozen tile.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        policy: Optional[ReconnectPolicy] = None,
        stall_timeout_s: float = 3.0,
        max_concurrent_analysis: int = 2,
        capture_factory: Optional[Callable[[Any], Any]] = None,
        cv2_module: Any = None,
        source_factory: Optional[Callable[[dict], FrameSource]] = None,
        watchdog_interval_s: float = 0.5,
        jpeg_quality: int = 80,
    ):
        self.bus = event_bus or EventBus()
        self.policy = policy or ReconnectPolicy()
        self.stall_timeout_s = stall_timeout_s
        self.max_concurrent_analysis = max(1, int(max_concurrent_analysis))
        self.jpeg_quality = max(30, min(95, int(jpeg_quality)))
        self.budget = threading.Semaphore(self.max_concurrent_analysis)
        self._capture_factory = capture_factory
        self._cv2 = cv2_module
        self._source_factory_override = source_factory
        self.watchdog_interval_s = watchdog_interval_s

        self._lock = threading.Lock()
        self._cameras: Dict[str, dict] = {}
        self._ingest: Dict[str, CameraWorker] = {}
        self._analysis: Dict[str, AnalysisWorker] = {}
        self._watchdog: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()
        self.watchdog_actions = 0
        self.started_at = 0.0

    # -- grid management ----------------------------------------------------
    def add_camera(
        self,
        camera: dict,
        process_fn: Optional[Callable[[Any, dict], Tuple[Any, List[dict]]]] = None,
        source: Optional[FrameSource] = None,
        max_fps: Optional[float] = None,
    ) -> str:
        """
        Registers a camera and starts its threads.

        `process_fn(frame, context) -> (annotated_frame, events)` is optional: a
        camera with no analyser still streams (useful for pure monitoring tiles).
        """
        camera_id = camera.get("camera_id") or camera.get("label") or "CAM-UNKNOWN"
        with self._lock:
            self._cameras[camera_id] = dict(camera)

        ingest = CameraWorker(
            camera_id=camera_id,
            source_factory=lambda: self._make_source(camera, source),
            policy=self.policy,
            stall_timeout_s=self.stall_timeout_s,
            max_fps=max_fps,
            node_id=camera.get("node_id", ""),
        )
        with self._lock:
            self._ingest[camera_id] = ingest
        ingest.start()

        if process_fn is not None:
            analyzer = AnalysisWorker(
                camera=camera,
                frame_buffer=ingest.buffer,
                process_fn=process_fn,
                bus=self.bus,
                budget=self.budget,
                jpeg_quality=self.jpeg_quality,
            )
            with self._lock:
                self._analysis[camera_id] = analyzer
            analyzer.start()

        self._ensure_watchdog()
        return camera_id

    def _make_source(self, camera: dict, override: Optional[FrameSource]) -> FrameSource:
        if self._source_factory_override is not None:
            return self._source_factory_override(camera)
        if override is not None:
            return override
        return build_frame_source(
            source_from_camera(camera),
            capture_factory=self._capture_factory,
            cv2_module=self._cv2,
        )

    def start_all(self) -> int:
        """Starts the watchdog; workers were already started by add_camera."""
        self._ensure_watchdog()
        self.started_at = time.time()
        with self._lock:
            return len(self._ingest)

    def stop_camera(self, camera_id: str, timeout: float = 3.0) -> bool:
        with self._lock:
            ingest = self._ingest.pop(camera_id, None)
            analyzer = self._analysis.pop(camera_id, None)
        ok = True
        if analyzer is not None:
            ok = analyzer.stop(timeout=timeout) and ok
        if ingest is not None:
            ok = ingest.stop(timeout=timeout) and ok
        return ok

    def stop_all(self, timeout: float = 5.0) -> bool:
        self._watchdog_stop.set()
        watchdog = self._watchdog
        self._watchdog = None
        if watchdog is not None:
            watchdog.join(timeout=timeout)

        with self._lock:
            ids = list(self._ingest.keys())
        ok = True
        for camera_id in ids:
            ok = self.stop_camera(camera_id, timeout=timeout) and ok
        with self._lock:
            self._cameras.clear()
        return ok

    # -- watchdog -----------------------------------------------------------
    def _ensure_watchdog(self) -> None:
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="ingest-watchdog", daemon=True
        )
        self._watchdog.start()

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(self.watchdog_interval_s):
            self.check_health()

    def check_health(self) -> int:
        """
        Marks stalled cameras and forces reconnects. Returns the number of actions.
        """
        actions = 0
        with self._lock:
            workers = list(self._ingest.items())

        for camera_id, worker in workers:
            if not worker.is_alive():
                continue
            age = time.time() - worker.last_frame_at if worker.last_frame_at else None
            if age is None:
                continue
            # Act only on a camera that believes it is healthy. Once it is
            # STALLED or already RECONNECTING, the worker owns its own recovery -
            # repeatedly releasing the capture from here would just thrash it.
            if age > self.stall_timeout_s and worker.state == WorkerState.LIVE:
                worker.mark_stalled()
                worker.force_reconnect()
                actions += 1
                self.watchdog_actions += 1
        return actions

    # -- data access --------------------------------------------------------
    def latest_frame(self, camera_id: str):
        """Newest raw frame (not consumed) for a camera, or None."""
        with self._lock:
            worker = self._ingest.get(camera_id)
        if worker is None:
            return None
        peek = worker.buffer.peek()
        return peek[0] if peek else None

    def latest_annotated(self, camera_id: str):
        """Newest analysed frame, consuming it (drop-old semantics)."""
        with self._lock:
            analyzer = self._analysis.get(camera_id)
        if analyzer is None:
            return None
        item = analyzer.out_buffer.take()
        return item[0] if item else None

    def peek_annotated(self, camera_id: str):
        with self._lock:
            analyzer = self._analysis.get(camera_id)
        if analyzer is None:
            return None
        item = analyzer.out_buffer.peek()
        return item[0] if item else None

    def latest_annotated_jpeg(self, camera_id: str) -> Optional[bytes]:
        """Newest analysed frame as JPEG bytes, consuming it (drop-old)."""
        with self._lock:
            analyzer = self._analysis.get(camera_id)
        if analyzer is None:
            return None
        item = analyzer.out_jpeg.take()
        return item[0] if item else None

    def peek_annotated_jpeg(self, camera_id: str) -> Optional[bytes]:
        """Newest JPEG bytes without consuming them - for repaints that did not
        advance the frame counter (a slow link must not eat the next frame)."""
        with self._lock:
            analyzer = self._analysis.get(camera_id)
        if analyzer is None:
            return None
        item = analyzer.out_jpeg.peek()
        return item[0] if item else None

    def drain_events(self, max_items: int = 200) -> List[dict]:
        return self.bus.drain(max_items=max_items)

    def health(self) -> List[dict]:
        """Per-camera grid health: ingest state + analysis stats, side by side."""
        with self._lock:
            cameras = list(self._cameras.keys())
            ingest = dict(self._ingest)
            analysis = dict(self._analysis)

        rows = []
        for camera_id in cameras:
            row: dict = {"camera_id": camera_id}
            worker = ingest.get(camera_id)
            if worker is not None:
                row.update(worker.health())
            analyzer = analysis.get(camera_id)
            if analyzer is not None:
                ah = analyzer.health()
                row.update({
                    "analysis_fps": ah["analysis_fps"],
                    "latency_ms": ah["latency_ms"],
                    "frames_processed": ah["frames_processed"],
                    "skipped_frames": ah["skipped_frames"],
                    "frame_kb": ah["frame_kb"],
                    "last_events": ah["last_events"],
                })
                if ah["last_error"]:
                    row["last_error"] = ah["last_error"]
            else:
                row.update({"analysis_fps": None, "latency_ms": None,
                            "frames_processed": None, "skipped_frames": None,
                            "last_events": None})
            rows.append(row)
        return rows

    def summary(self) -> dict:
        """Grid-level roll-up for the dashboard header."""
        rows = self.health()
        live = sum(1 for r in rows if r.get("state") == WorkerState.LIVE)
        stalled = sum(1 for r in rows if r.get("state") == WorkerState.STALLED)
        reconnecting = sum(1 for r in rows if r.get("state") == WorkerState.RECONNECTING)
        return {
            "cameras": len(rows),
            "live": live,
            "stalled": stalled,
            "reconnecting": reconnecting,
            "frames_in": sum(int(r.get("frames_read") or 0) for r in rows),
            "frames_analysed": sum(int(r.get("frames_processed") or 0) for r in rows),
            "events_pending": len(self.bus),
            "events_total": self.bus.total,
            "events_dropped": self.bus.dropped,
            "watchdog_actions": self.watchdog_actions,
            "analysis_permits": self.max_concurrent_analysis,
            "jpeg_quality": self.jpeg_quality,
            "uplink_kb": round(
                sum(float(r.get("frame_kb") or 0.0) for r in rows), 1
            ),
            "latency_ms": round(
                sum(float(r.get("latency_ms") or 0.0) for r in rows) / max(1, len(rows)), 1
            ),
        }
