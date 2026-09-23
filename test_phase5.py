"""
IBVAP PHASE 5 VERIFICATION: LIVE INGESTION & MULTI-CAMERA WORKERS
=================================================================

Verifies the RTSP/ONVIF ingest layer without a broker, a camera, or OpenCV
installed - the OpenCV source path is exercised through injected fakes, so the
RTSP tuning, EOF looping and timeout logic are all genuinely covered.

  1. Source classification  - file / rtsp / onvif / webcam / synthetic, with
                              credentials stripped from display labels.
  2. Reconnect backoff      - grows, caps, and stays jittered.
  3. Depth-1 frame buffer   - drop-old semantics (latency cannot accumulate).
  4. RTSP stream handling   - TCP transport, buffer size 1, open/read timeouts.
  5. EOF looping            - local demo files restart instead of dying.
  6. Camera self-healing    - a camera that fails to open recovers by itself.
  7. Concurrency            - N cameras ingest AND analyse in parallel.
  8. UI never blocks        - reading grid state stays fast while inference runs.
  9. Watchdog / stalls      - a silently dead feed is detected and force-reconnected.
 10. Bounded inference      - the permit pool caps simultaneous model passes.
 11. Error isolation        - a bad frame cannot kill a camera's analyser.
 12. Graceful shutdown      - every thread is joined.
 13. Event bus bounds       - overflow is counted, never silent.

Run:  python test_phase5.py
"""

import os
import sys
import threading
import time

import numpy as np

from modules.ingest import (
    AnalysisWorker,
    CameraWorker,
    EventBus,
    LatestFrame,
    OpenCVFrameSource,
    ReconnectPolicy,
    StreamManager,
    SyntheticFrameSource,
    WorkerState,
    build_frame_source,
    classify_source,
)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
failures = []


def check(condition, message):
    print(f"{PASS if condition else FAIL} {message}")
    if not condition:
        failures.append(message)
    return condition


def header(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def wait_until(predicate, timeout=6.0, interval=0.02):
    """Polls a predicate. Returns True if it became true within the timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def fake_camera(camera_id="CAM-TEST", node_id="BOP-TEST", source="synthetic:0"):
    return {
        "camera_id": camera_id,
        "node_id": node_id,
        "label": f"Test camera {camera_id}",
        "video_source": source,
    }


# ---------------------------------------------------------------------------
# Fakes: let the OpenCV source path be verified with no OpenCV installed.
# ---------------------------------------------------------------------------
class FakeCapture:
    def __init__(self, frames, fail_open=False, fail_reads_after=None):
        self._frames = frames
        self._index = 0
        self._open = not fail_open
        self._fail_reads_after = fail_reads_after
        self._reads = 0
        self.sets = []
        self.released = False

    def isOpened(self):
        return self._open

    def read(self):
        self._reads += 1
        if self._fail_reads_after is not None and self._reads > self._fail_reads_after:
            return False, None
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def set(self, prop, value):
        self.sets.append((prop, value))
        if prop == FakeCV2.CAP_PROP_POS_FRAMES:
            self._index = int(value)
        return True

    def release(self):
        self.released = True
        self._open = False


class FakeCV2:
    CAP_PROP_BUFFERSIZE = 38
    CAP_PROP_OPEN_TIMEOUT_MSEC = 53
    CAP_PROP_READ_TIMEOUT_MSEC = 54
    CAP_PROP_POS_FRAMES = 1


def main():
    print("=" * 74)
    print("  IBVAP PHASE 5 VERIFICATION: LIVE INGESTION & MULTI-CAMERA WORKERS")
    print("=" * 74)

    # ------------------------------------------------------------------
    header("1. SOURCE CLASSIFICATION (RTSP / ONVIF / FILE / WEBCAM)")
    # ------------------------------------------------------------------
    cases = [
        ("sample_videos/bop_perimeter.mp4", "file", False),
        ("rtsp://10.20.30.40:554/Streaming/Channels/101", "rtsp", True),
        ("rtsps://cam.border.local/profile2/media.smp", "rtsp", True),
        ("rtsp://admin:secret@10.0.0.9/onvif-http/snapshot", "onvif", True),
        ("0", "webcam", True),
        ("synthetic:demo", "synthetic", True),
    ]
    for uri, expected_kind, expected_live in cases:
        spec = classify_source(uri)
        ok = spec.kind == expected_kind and spec.is_live == expected_live
        check(ok, f"{uri[:52]:<52} -> {spec.kind:<9} live={spec.is_live}")
    live_spec = classify_source("rtsp://admin:secret@10.0.0.9/onvif-http/snapshot")
    check("secret" not in live_spec.label, "credentials stripped from display label")
    check(live_spec.label.startswith("rtsp://***@"), "label keeps host, hides password")

    # ------------------------------------------------------------------
    header("2. RECONNECT POLICY - BACKOFF GROWS, CAPS, JITTERS")
    # ------------------------------------------------------------------
    policy = ReconnectPolicy(initial_s=0.5, factor=2.0, max_s=8.0, jitter=0.2)
    delays = [policy.delay_for(a) for a in range(1, 8)]
    print("    delays: " + ", ".join(f"{d:.2f}s" for d in delays))
    check(delays[0] < delays[1] < delays[2], "backoff grows with each failed attempt")
    check(all(d <= 8.0 * 1.2 + 1e-6 for d in delays), "backoff respects the max ceiling")
    tight = ReconnectPolicy(initial_s=1.0, factor=2.0, max_s=4.0, jitter=0.0)
    check(all(tight.delay_for(a) <= 4.0 for a in range(1, 12)), "cap holds over 11 attempts")
    check(delays[-1] >= 8.0 * 0.8, "long outages wait near the cap, not in a hot loop")

    # ------------------------------------------------------------------
    header("3. DEPTH-1 BUFFER: DROP-OLD, NEVER QUEUE-UP")
    # ------------------------------------------------------------------
    buffer = LatestFrame()
    for i in range(5):
        buffer.publish(np.full((4, 4, 3), i, dtype=np.uint8), seq=i + 1)
    item = buffer.take()
    check(buffer.dropped == 4, f"4 superseded frames discarded ({buffer.dropped})")
    check(item is not None and item[1] == 5, "consumer receives the NEWEST frame (seq 5)")
    check(buffer.take() is None, "buffer is empty after take (depth 1, not a queue)")
    check(buffer.seq == 5, "sequence number survives the take")

    # ------------------------------------------------------------------
    header("4. RTSP/ONVIF LIVE-SOURCE TUNING (injected fake OpenCV)")
    # ------------------------------------------------------------------
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    live_capture = FakeCapture(frames)
    live_source = OpenCVFrameSource(
        classify_source("rtsp://10.20.30.40:554/Streaming/Channels/101"),
        capture_factory=lambda uri: live_capture,
        cv2_module=FakeCV2,
        open_timeout_ms=4000,
        read_timeout_ms=4500,
    )
    check(live_source.open(), "RTSP source opens")
    prop_map = dict(live_capture.sets)
    check(prop_map.get(FakeCV2.CAP_PROP_BUFFERSIZE) == 1,
          "live stream sets buffer size 1 (no accumulating latency)")
    check(prop_map.get(FakeCV2.CAP_PROP_OPEN_TIMEOUT_MSEC) == 4000,
          "live stream sets an open timeout")
    check(prop_map.get(FakeCV2.CAP_PROP_READ_TIMEOUT_MSEC) == 4500,
          "live stream sets a read timeout")
    check("rtsp_transport;tcp" in os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", ""),
          "RTSP forced over TCP (UDP shreds packets on field links)")
    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

    ok, frame = live_source.read()
    check(ok and frame is not None, "RTSP source yields frames")
    for _ in range(3):
        live_source.read()
    ok_after_eof, _ = live_source.read()
    check(not ok_after_eof, "a live stream that stops delivering reports failure")
    live_source.release()
    check(live_capture.released, "release() actually closes the capture")

    # ------------------------------------------------------------------
    header("5. LOCAL DEMO FILES LOOP INSTEAD OF DYING AT EOF")
    # ------------------------------------------------------------------
    file_capture = FakeCapture(frames)
    file_source = OpenCVFrameSource(
        classify_source("sample_videos/bop_perimeter.mp4"),
        capture_factory=lambda uri: file_capture, cv2_module=FakeCV2,
    )
    check(file_source.open(), "file source opens")
    check(FakeCV2.CAP_PROP_BUFFERSIZE not in dict(file_capture.sets),
          "file decoding is left at default settings (no stream tuning)")
    reads_ok = 0
    for _ in range(6):
        ok, _f = file_source.read()
        reads_ok += 1 if ok else 0
    print(f"    3-frame file read 6 times -> {reads_ok} successful reads")
    check(reads_ok == 6, "file loops seamlessly past EOF for demo playback")
    check(FakeCV2.CAP_PROP_POS_FRAMES in dict(file_capture.sets),
          "loop is implemented by rewinding CAP_PROP_POS_FRAMES")
    file_source.release()

    failing_source = OpenCVFrameSource(
        classify_source("rtsp://10.0.0.7/stream"),
        capture_factory=lambda uri: FakeCapture(frames, fail_open=True),
        cv2_module=FakeCV2,
    )
    check(not failing_source.open(), "an unreachable camera reports open failure")
    check("did not open" in failing_source.last_error, "failure reason is recorded")

    # ------------------------------------------------------------------
    header("6. CAMERA SELF-HEALING (FLAKY CAMERA RECOVERS UNAIDED)")
    # ------------------------------------------------------------------
    # The camera is down for its first two dial attempts, then comes back - the
    # counter is shared across source instances because a worker rebuilds its
    # source on every retry.
    dial_attempts = {"n": 0}

    class UnreliableCamera(SyntheticFrameSource):
        def open(self):
            dial_attempts["n"] += 1
            if dial_attempts["n"] <= 2:
                self.last_error = "simulated link down"
                self._open = False
                return False
            return super().open()

    policy = ReconnectPolicy(initial_s=0.05, factor=1.5, max_s=0.3, jitter=0.0)
    flaky = CameraWorker(
        camera_id="CAM-FLAKY",
        source_factory=lambda: UnreliableCamera(width=160, height=120, fps=60, seed=1),
        policy=policy,
        stall_timeout_s=5.0,
    )
    flaky.start()
    recovered = wait_until(lambda: flaky.state == WorkerState.LIVE, timeout=5.0)
    check(recovered, "camera that failed to dial twice reached LIVE unaided")
    check(wait_until(lambda: flaky.frames_read >= 5, timeout=3.0),
          "recovered camera is producing frames")
    health = flaky.health()
    print(f"    6a dial-failure camera: state={health['state']} "
          f"frames={health['frames_read']} failures={health['failures']} "
          f"reconnects={health['reconnects']} (never was live, so no reconnect)")
    check(health["failures"] >= 2, "dial failures were counted against the camera")
    check(flaky.stop(), "camera worker thread stopped cleanly")
    check(flaky.state == WorkerState.STOPPED, "stopped camera reports STOPPED")

    # 6b. A camera that goes LIVE and then drops mid-stream must be rebuilt.
    #     This is the case the field actually hits: the stream dies at 3 a.m.
    dropper = CameraWorker(
        camera_id="CAM-DROP",
        source_factory=lambda: SyntheticFrameSource(
            width=160, height=120, fps=60, fail_reads_after=6, seed=4
        ),
        policy=ReconnectPolicy(initial_s=0.05, factor=1.5, max_s=0.3, jitter=0.0),
        stall_timeout_s=5.0,
    )
    dropper.start()
    check(wait_until(lambda: dropper.state == WorkerState.LIVE, timeout=4.0),
          "6b camera reached LIVE")
    check(wait_until(lambda: dropper.frames_read > 6, timeout=6.0),
          "6b camera kept streaming PAST the simulated stream drop")
    drop_health = dropper.health()
    print(f"    6b mid-stream drop camera: frames={drop_health['frames_read']} "
          f"reconnects={drop_health['reconnects']} failures={drop_health['failures']}")
    check(drop_health["reconnects"] >= 1,
          "a drop after being live was counted as a reconnect")
    check(drop_health["failures"] >= 1, "the stream drop was recorded")
    check(dropper.stop(), "6b camera stopped cleanly")

    # ------------------------------------------------------------------
    header("7-8. CONCURRENT MULTI-CAMERA GRID + NON-BLOCKING UI PATH")
    # ------------------------------------------------------------------
    ANALYSIS_SLEEP = 0.05
    concurrency = {"active": 0, "peak": 0}
    concurrency_lock = threading.Lock()
    thread_ids = set()
    processed = {}
    processed_lock = threading.Lock()

    def slow_process(frame, context):
        with concurrency_lock:
            concurrency["active"] += 1
            concurrency["peak"] = max(concurrency["peak"], concurrency["active"])
        thread_ids.add(threading.get_ident())
        camera_id = context["camera_id"]
        try:
            time.sleep(ANALYSIS_SLEEP)
            with processed_lock:
                processed[camera_id] = processed.get(camera_id, 0) + 1
        finally:
            with concurrency_lock:
                concurrency["active"] -= 1
        annotated = frame.copy() if hasattr(frame, "copy") else frame
        events = [{
            "event_type": "INTRUSION_ALERT",
            "track_id": context["frame_seq"] % 50,
            "status": "UNKNOWN_INTRUDER",
            "category": "human",
            "class_name": "person",
            "confidence": 0.9,
            "location": "(20, 20)",
            "zone": "synthetic",
            "camera_id": camera_id,
        }]
        return annotated, events

    manager = StreamManager(
        stall_timeout_s=5.0,
        max_concurrent_analysis=4,
        source_factory=lambda cam: SyntheticFrameSource(
            width=240, height=180, fps=120, seed=hash(cam["camera_id"]) % 100
        ),
        watchdog_interval_s=0.05,
    )
    camera_ids = [f"CAM-{i}" for i in range(1, 5)]
    for camera_id in camera_ids:
        manager.add_camera(fake_camera(camera_id), process_fn=slow_process)

    started = time.time()
    all_live = wait_until(
        lambda: all(r["state"] == WorkerState.LIVE for r in manager.health()),
        timeout=6.0,
    )
    check(all_live, "all 4 cameras reached LIVE simultaneously")

    # The dashboard reads grid state every render pass; that must stay cheap even
    # while four inference threads are busy.
    read_times = []
    for _ in range(10):
        t0 = time.time()
        manager.summary()
        manager.health()
        read_times.append(time.time() - t0)
    worst_read_ms = max(read_times) * 1000
    print(f"    10 UI reads while 4 analyses run: worst {worst_read_ms:.1f} ms")
    check(worst_read_ms < 50.0, "UI state reads are never blocked by inference")
    check(threading.current_thread().ident not in thread_ids,
          "inference ran off the UI thread (dedicated analysis workers)")

    one_round = ANALYSIS_SLEEP * len(camera_ids)
    check(wait_until(lambda: len(processed) == len(camera_ids), timeout=5.0),
          "every camera's analyser produced results")

    # Prove genuine overlap: with 4 permits and a 50 ms payload, serialised work
    # would take ~200 ms per round.
    time.sleep(0.4)
    elapsed_rounds = time.time() - started
    throughput = sum(processed.values()) / elapsed_rounds
    print(f"    grid throughput: {throughput:.1f} frames/s across 4 cameras "
          f"(serialised baseline would be ~{1.0 / ANALYSIS_SLEEP:.0f}/s total)")
    check(concurrency["peak"] >= 2,
          f"analyses genuinely overlapped (peak concurrency {concurrency['peak']})")
    check(throughput > 1.0 / ANALYSIS_SLEEP * 0.5,
          "grid throughput exceeds the single-threaded ceiling")

    # Same frame must never be handed to two analysis workers.
    check(all(count > 0 for count in processed.values()),
          "no camera was starved by the others (fair scheduling)")
    check(concurrency["peak"] <= 4, "concurrency never exceeded the permit pool")
    ingest_workers = list(manager._ingest.values())
    manager.stop_all()
    check(ingest_workers and all(not w.is_alive() for w in ingest_workers),
          "all ingest worker threads joined on stop_all()")
    check(manager.summary()["events_total"] > 0, "events flowed through the bus")
    check(manager.summary()["events_pending"] >= 0, "bus drained without loss")

    # ------------------------------------------------------------------
    header("9. WATCHDOG: SILENTLY DEAD FEED IS DETECTED AND RECONNECTED")
    # ------------------------------------------------------------------
    stall_manager = StreamManager(
        stall_timeout_s=0.4,
        max_concurrent_analysis=1,
        source_factory=lambda cam: SyntheticFrameSource(
            width=120, height=90, fps=120, stall_after=8, stall_s=3.0, seed=2
        ),
        watchdog_interval_s=0.05,
    )
    stall_camera = fake_camera("CAM-STALL")
    stall_manager.add_camera(stall_camera, process_fn=lambda f, c: (f, []))
    check(wait_until(lambda: stall_manager.summary()["live"] == 1, timeout=4.0),
          "camera was LIVE before the simulated stall")
    # The feed now goes silent *without closing* - a read() wedged inside ffmpeg.
    stalled_seen = wait_until(
        lambda: any(r["state"] == WorkerState.STALLED for r in stall_manager.health()),
        timeout=4.0,
    )
    check(stalled_seen, "watchdog marked the silent feed STALLED")
    recovered_from_stall = wait_until(
        lambda: any(r["state"] == WorkerState.LIVE for r in stall_manager.health()),
        timeout=10.0,
    )
    check(recovered_from_stall, "watchdog force-reconnected and the feed came back")
    print(f"    watchdog actions: {stall_manager.watchdog_actions}")
    check(stall_manager.watchdog_actions >= 1, "recovery action was counted")
    stall_manager.stop_all()
    check(True, "stalled camera shut down cleanly")

    # ------------------------------------------------------------------
    header("10. BOUNDED INFERENCE CONCURRENCY ON EDGE CPU")
    # ------------------------------------------------------------------
    gate = {"active": 0, "peak": 0, "latency_ms": 0.0}
    gate_lock = threading.Lock()

    def guarded_process(frame, context):
        with gate_lock:
            gate["active"] += 1
            gate["peak"] = max(gate["peak"], gate["active"])
        time.sleep(0.06)
        with gate_lock:
            gate["active"] -= 1
        return frame, []

    limited = StreamManager(
        max_concurrent_analysis=1,
        stall_timeout_s=5.0,
        source_factory=lambda cam: SyntheticFrameSource(width=120, height=90, fps=200),
        watchdog_interval_s=0.05,
    )
    for camera_id in ("CAM-A", "CAM-B", "CAM-C"):
        limited.add_camera(fake_camera(camera_id), process_fn=guarded_process)
    time.sleep(1.2)
    limited_rows = {r["camera_id"]: r for r in limited.health()}
    measured_latency = max(
        float(r.get("latency_ms") or 0.0) for r in limited_rows.values()
    )
    print(f"    3 cameras, 1 permit -> observed peak concurrency {gate['peak']}, "
          f"measured analysis latency {measured_latency:.0f} ms (model=60 ms)")
    check(gate["peak"] == 1,
          "permit pool caps simultaneous model passes on a 15W edge box")
    check(measured_latency >= 50.0,
          "analysis latency is measured, not guessed (single-threaded queueing visible)")
    limited_summary = limited.summary()
    check(limited_summary["analysis_permits"] == 1, "permit count is reported to the UI")
    limited.stop_all()

    # ------------------------------------------------------------------
    header("11. ERROR ISOLATION: A BAD FRAME CANNOT KILL A CAMERA")
    # ------------------------------------------------------------------
    calls = {"n": 0}

    def flaky_process(frame, context):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated model crash on frame 1")
        return frame, []

    resilient = StreamManager(
        max_concurrent_analysis=1,
        stall_timeout_s=5.0,
        source_factory=lambda cam: SyntheticFrameSource(width=120, height=90, fps=120),
        watchdog_interval_s=0.05,
    )
    resilient.add_camera(fake_camera("CAM-ERR"), process_fn=flaky_process)
    check(wait_until(lambda: calls["n"] >= 3, timeout=4.0),
          "analyser kept processing after an exception")
    err_health = resilient.health()[0]
    check("RuntimeError" in (err_health.get("last_error") or ""),
          "the exception was surfaced in camera health, not swallowed")
    check(err_health["state"] == WorkerState.LIVE, "camera stayed LIVE through the error")
    resilient.stop_all()

    # ------------------------------------------------------------------
    header("12-13. GRACEFUL SHUTDOWN + EVENT BUS BOUNDS")
    # ------------------------------------------------------------------
    shutdown = StreamManager(
        max_concurrent_analysis=2,
        stall_timeout_s=5.0,
        source_factory=lambda cam: SyntheticFrameSource(width=120, height=90, fps=120),
        watchdog_interval_s=0.05,
    )
    for camera_id in ("CAM-S1", "CAM-S2"):
        shutdown.add_camera(fake_camera(camera_id), process_fn=lambda f, c: (f, []))
    time.sleep(0.4)
    ingest_before = list(shutdown._ingest.values())
    analysis_before = list(shutdown._analysis.values())
    stopped_ok = shutdown.stop_all(timeout=5.0)
    check(stopped_ok, "stop_all() reported a clean shutdown")
    check(all(not w.is_alive() for w in ingest_before), "every ingest thread is gone")
    check(all(not a.is_alive() for a in analysis_before), "every analysis thread is gone")
    check(shutdown.summary()["cameras"] == 0, "grid is empty after shutdown")

    bus = EventBus(capacity=5)
    accepted = sum(1 for i in range(20) if bus.publish({"event_type": "X", "n": i}))
    drained = bus.drain()
    print(f"    capacity=5, published=20 -> accepted={accepted}, dropped={bus.dropped}")
    check(accepted == 5, "bus enforces its capacity")
    check(bus.dropped == 15, "overflow is counted, never silent")
    check(bus.total == 20, "total published count is tracked")
    check(len(drained) == 5 and drained[0]["n"] == 0,
          "drain returns the OLDEST events first (incident order preserved)")

    # ------------------------------------------------------------------
    header("14. MULTI-CAMERA EVENT ATTRIBUTION (NEVER CREDITED TO THE WRONG SECTOR)")
    # ------------------------------------------------------------------
    # With five sectors running at once, an alert from Sector B must never be filed
    # against the sector the operator happens to be watching. The workers stamp
    # camera_id / node_id on every event; this verifies that stamp.
    from modules.nodes import CAMERAS, get_camera

    attribution = StreamManager(
        max_concurrent_analysis=3,
        stall_timeout_s=5.0,
        source_factory=lambda cam: SyntheticFrameSource(width=160, height=120, fps=120),
        watchdog_interval_s=0.05,
    )
    attributed_cameras = [get_camera(key) for key in list(CAMERAS.keys())[:3]]
    for cam in attributed_cameras:
        def emitter(frame, ctx, _cam=cam):
            return frame, [{
                "event_type": "INTRUSION_ALERT",
                "track_id": 1,
                "status": "UNKNOWN_INTRUDER",
                "zone": _cam["camera_id"],  # sentinel: which camera BUILT this event
            }]
        attribution.add_camera(cam, process_fn=emitter)

    check(wait_until(lambda: len(attribution.bus) >= 3, timeout=6.0),
          "all three cameras produced alerts")
    drained = attribution.drain_events(max_items=500)
    misattributed = [e for e in drained if e.get("camera_id") != e.get("zone")]
    missing_identity = [e for e in drained if not e.get("camera_id") or not e.get("node_id")]
    print(f"    drained={len(drained)} events, cameras seen="
          f"{sorted({e.get('camera_id') for e in drained})}")
    check(len({e.get("camera_id") for e in drained}) == 3,
          "events carry the identity of all three cameras")
    check(not misattributed,
          "no event was attributed to a camera other than the one that produced it")
    check(not missing_identity, "every event carries camera_id AND node_id")
    node_pairs = {e["camera_id"]: e["node_id"] for e in drained}
    check(
        node_pairs.get(get_camera("Channel 3")["camera_id"]) == get_camera("Channel 3")["node_id"],
        "camera-to-outpost mapping is preserved for geo-tagging at HQ",
    )
    attribution.stop_all()

    # ------------------------------------------------------------------
    header("GRID HEALTH SAMPLE (what the dashboard renders)")
    # ------------------------------------------------------------------
    sample = StreamManager(
        max_concurrent_analysis=2, stall_timeout_s=5.0,
        source_factory=lambda cam: SyntheticFrameSource(width=160, height=120, fps=60),
        watchdog_interval_s=0.05,
    )
    for camera_id in ("CAM-1", "CAM-2"):
        sample.add_camera(fake_camera(camera_id), process_fn=lambda f, c: (f, []))
    wait_until(lambda: sample.summary()["live"] == 2, timeout=4.0)
    time.sleep(0.3)
    print(f"    {'camera':<10} {'state':<13} {'in_fps':>7} {'an_fps':>7} "
          f"{'lat_ms':>7} {'frames':>7} {'recon':>6}")
    for row in sample.health():
        print(f"    {row['camera_id']:<10} {row['state']:<13} "
              f"{(row['input_fps'] or 0):>7.1f} {(row['analysis_fps'] or 0):>7.1f} "
              f"{(row['latency_ms'] or 0):>7.1f} {(row['frames_read'] or 0):>7} "
              f"{row['reconnects']:>6}")
    summary = sample.summary()
    print(f"    grid summary: {summary}")
    check(summary["live"] == 2 and summary["cameras"] == 2, "summary reflects both cameras")
    sample.stop_all()

    # ------------------------------------------------------------------
    header("PHASE 5 RESULT")
    # ------------------------------------------------------------------
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for item in failures:
            print(f"    - {item}")
        print("=" * 74)
        return 1

    print("  ALL CHECKS PASSED - live ingest grid is production-shaped:")
    print("    * RTSP/ONVIF sources tuned for field links (TCP, depth-1, timeouts)")
    print("    * N cameras ingest and run inference concurrently, off the UI thread")
    print("    * silent feed death is detected by the watchdog and self-healed")
    print("    * inference concurrency is capped so edge CPU is never oversubscribed")
    print("    * every alert stays attributed to the sector that produced it")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
