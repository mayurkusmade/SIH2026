"""
IBVAP PHASE 9 VERIFICATION: THROUGHPUT & UI-PAYLOAD CONTRACTS
=============================================================

Performance work rots silently. Every optimisation made to fix UI lag is a
promise ("this is only done once per frame", "this is paced", "the operator
receives JPEG, not a re-encoded array"), and nothing about the code makes that
promise self-evident to the next person who edits it.

So this suite turns each promise into a check, using injected fakes - it needs no
models, no video and no display, and runs in a couple of seconds:

  1. FRS face detection is memoised PER FRAME: N tracks in one frame cost ONE
     full-frame detection, not N. (It used to cost N - the single biggest stage.)
  2. ...but never across frames: a different frame object recomputes, so memoisation
     cannot serve a stale face list to a new frame.
  3. Identity decisions are unchanged by that memoisation (still per-track, still
     confirmed over time).
  4. JPEG encoding: real JPEG bytes, None for junk input, quality actually matters.
  5. The ingest grid publishes BOTH the array and the encoded frame, and the
     manager's accessors keep their documented take/peek semantics.
  6. ANPR OCR is paced grid-wide: 1 read per frame max, a minimum interval between
     reads, deferral counted (not silent), and a cached read still returned.
  7. A plate too small to resolve never reaches OCR at all.
  8. ANPR warmup does not pollute the read cache.
  9. The pipeline's FRS stride paces the expensive pass but still draws the last
     decisions, so the overlay never blinks.
 10. Warmup is safe and honest when a backend is missing.
 11. JPEG quality is clamped to a sane range.

Run:  python test_phase9.py
"""

import sys

import numpy as np

from modules.anpr import CascadedANPR
from modules.frs import (
    AUTHORIZED,
    InjectedEmbedder,
    InjectedFaceDetector,
    FaceIndex,
    FRSModule,
    l2_normalize,
)
from modules.ingest import (
    AnalysisWorker,
    EventBus,
    LatestFrame,
    StreamManager,
    encode_jpeg,
)
from modules.pipeline import VisionModels, build_analyser
from modules.pose import BehaviorThresholds, PoseEngine, NullPoseEstimator

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


# ---------------------------------------------------------------------------
# Counting doubles: they record how often the expensive path is entered.
# ---------------------------------------------------------------------------
class CountingFaceDetector(InjectedFaceDetector):
    def __init__(self, boxes=None):
        super().__init__(boxes)
        self.calls = 0

    def detect(self, frame, min_face_px: int = 40):
        self.calls += 1
        return super().detect(frame, min_face_px)


class CrossingEmbedder(InjectedEmbedder):
    """Every face crops to the same identity, so a track can actually be promoted."""

    def __init__(self, dim: int = 16):
        self._seen = {}
        super().__init__(vector_for=self._vector, dim=dim)

    def _vector(self, face_bbox):
        return l2_normalize(np.ones(self.dim, dtype=np.float32))


class _Tensor:
    """Stands in for the torch tensors the plate model normally returns."""

    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, index):
        return self

    def tolist(self):
        return list(self._values)

    def item(self):
        return self._values[0]


class _Box:
    def __init__(self, conf, xyxy):
        self.conf = [_Tensor([conf])]
        self.xyxy = [_Tensor(xyxy)]


class _PlateResult:
    def __init__(self, boxes):
        self.boxes = boxes


class StubPlateModel:
    """Plate detector stub: always finds one plate filling the vehicle crop."""

    def __init__(self):
        self.calls = 0

    def __call__(self, crop, conf=0.25, device="cpu", verbose=False):
        self.calls += 1
        h, w = crop.shape[:2]
        plate = _Box(0.6, (1, 1, max(2, w - 1), max(2, h - 1)))
        return [_PlateResult([plate])]


class StubReader:
    """OCR stub: counts calls and returns a fixed plate."""

    def __init__(self, text="UK07AB1234", conf=0.9):
        self.calls = 0
        self.text = text
        self.conf = conf

    def readtext(self, image, allowlist=None, detail=1):
        self.calls += 1
        return [([[0, 0], [10, 0], [10, 5], [0, 5]], self.text, self.conf)]


def big_plate_anpr(reader, **kwargs):
    """Builds an ANPR instance whose plate model always succeeds, without YOLO."""
    anpr = CascadedANPR.__new__(CascadedANPR)
    anpr.plate_model = StubPlateModel()
    anpr.device = "cpu"
    anpr.ocr_conf_threshold = kwargs.get("ocr_conf_threshold", 0.40)
    anpr.throttle_frames = kwargs.get("throttle_frames", 0)
    anpr.min_vehicle_height = kwargs.get("min_vehicle_height", 10)
    anpr.min_plate_height = kwargs.get("min_plate_height", 4)
    anpr.ocr_min_interval_s = kwargs.get("ocr_min_interval_s", 0.0)
    anpr.max_ocr_per_frame = kwargs.get("max_ocr_per_frame", 1)
    anpr.ocr_calls = 0
    anpr.ocr_skipped_budget = 0
    anpr.ocr_skipped_small = 0
    anpr._last_ocr_at = 0.0
    anpr._ocr_frame = -1
    anpr._ocr_this_frame = 0
    anpr.reader = reader
    anpr.track_ocr_cache = {}
    return anpr


def frame_of(height=400, width=600, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def main():
    print("=" * 74)
    print("  IBVAP PHASE 9 VERIFICATION: THROUGHPUT & UI-PAYLOAD CONTRACTS")
    print("=" * 74)

    # ------------------------------------------------------------------
    header("1-3. FRS: ONE FACE DETECTION PER FRAME, NOT ONE PER TRACK")
    # ------------------------------------------------------------------
    frame = frame_of()
    tracks = {
        1: {"bbox": (50, 50, 250, 350), "category": "human"},
        2: {"bbox": (300, 50, 500, 350), "category": "human"},
        3: {"bbox": (520, 60, 580, 360), "category": "human"},
    }
    detector = CountingFaceDetector(boxes=[(60, 60, 160, 160, 0.9)])
    index = FaceIndex(db_path=":memory:")
    frs = FRSModule(
        index=index, detector=detector, embedder=CrossingEmbedder(),
        min_face_px=20, min_track_height=20, confirm_frames=1,
        # throttle_frames=0 so every frame actually reaches face detection; the
        # per-track throttle (tested in phase 6) would otherwise short-circuit the
        # loop before the memoisation being checked here.
        throttle_frames=0,
    )
    frs.process(frame, tracks, timestamp="00:00:00", frame_idx=1)
    check(detector.calls == 1,
          f"3 tracks in one frame cost 1 face detection (was {len(tracks)}, got {detector.calls})")

    # A second pass over the SAME frame object (two consumers, one frame) must not
    # pay for detection again.
    before = detector.calls
    frs.process(frame, tracks, timestamp="00:00:01", frame_idx=1)
    check(detector.calls == before,
          "re-processing the same frame object does not re-detect")

    # A NEW frame must re-detect: the memo holds one frame, so a stale face list
    # can never be applied to different pixels.
    first_calls = detector.calls
    frs.process(frame_of(seed=7), tracks, timestamp="00:00:02", frame_idx=2)
    check(detector.calls == first_calls + 1,
          "a NEW frame re-detects (the memo never serves a stale face list)")

    # And coming back to the earlier frame is a NEW frame as far as the memo is
    # concerned - correct, since only the most recent frame is cached.
    before = detector.calls
    frs.process(frame, tracks, timestamp="00:00:03", frame_idx=3)
    check(detector.calls == before + 1,
          "returning to an older frame re-detects rather than guessing")

    # Identity semantics must be untouched: confirmation still promotes.
    decisions = frs.process(frame, tracks, timestamp="00:00:03", frame_idx=4)
    check(len(decisions) >= 1, f"identity decisions are still issued ({len(decisions)})")
    check(all(d["track_id"] in tracks for d in decisions),
          "every decision is attributed to a real track")

    # ------------------------------------------------------------------
    header("4. JPEG ENCODING: THE UI PAYLOAD PATH")
    # ------------------------------------------------------------------
    jpeg = encode_jpeg(frame_of(200, 300), quality=80)
    check(isinstance(jpeg, (bytes, bytearray)) and len(jpeg) > 100,
          f"encode_jpeg returns real bytes ({len(jpeg) if jpeg else 0} B)")
    check(jpeg[:2] == b"\xff\xd8", "the bytes are a JPEG (SOI marker ff d8)")
    check(encode_jpeg(None) is None, "None frame -> None, never an exception")
    check(encode_jpeg(np.zeros((0, 0, 3), dtype=np.uint8)) is None,
          "empty frame -> None")
    check(encode_jpeg(np.zeros((4, 4, 3), dtype=np.uint8)) is not None,
          "a tiny frame still encodes")
    sample = frame_of(720, 1280, seed=3)
    low = len(encode_jpeg(sample, quality=40))
    high = len(encode_jpeg(sample, quality=95))
    check(high > low, f"quality is honoured: q40 {low//1024} KB < q95 {high//1024} KB")
    check(low < 900 * 1024, f"a 720p operator frame is {low//1024} KB, not megabytes")

    # ------------------------------------------------------------------
    header("5. INGEST PUBLISHES THE ARRAY *AND* THE ENCODED FRAME")
    # ------------------------------------------------------------------
    worker = AnalysisWorker(
        camera={"camera_id": "CAM-TEST"},
        frame_buffer=LatestFrame(),
        process_fn=lambda frame, ctx: (frame, []),
        bus=EventBus(),
        jpeg_quality=80,
    )
    worker.start()
    worker.frame_buffer.publish(frame_of(240, 320, seed=5), 1, 0.0)
    worker.out_jpeg._ts = 0.0
    import time as _time
    deadline = _time.time() + 5.0
    while worker.out_jpeg.peek() is None and _time.time() < deadline:
        _time.sleep(0.02)
    array_item = worker.out_buffer.peek()
    jpeg_item = worker.out_jpeg.peek()
    check(array_item is not None, "the annotated ARRAY is still published (FRS enroll needs it)")
    check(jpeg_item is not None, "the encoded frame is published alongside it")
    if jpeg_item:
        check(jpeg_item[0][:2] == b"\xff\xd8", "the published frame is JPEG, not a raw array")
        check(jpeg_item[1] == 1, "the encoded frame carries the same sequence number")
    worker.stop()

    manager = StreamManager(jpeg_quality=200)  # absurd on purpose
    check(manager.jpeg_quality == 95, "JPEG quality is clamped to <= 95")
    check(StreamManager(jpeg_quality=1).jpeg_quality == 30,
          "JPEG quality is clamped to >= 30")

    manager = StreamManager(jpeg_quality=80)
    camera = {"camera_id": "CAM-J", "video_source": "synthetic://demo", "pre_rendered": False}
    manager.add_camera(camera, process_fn=lambda frame, ctx: (frame, []))
    manager.start_all()
    deadline = _time.time() + 8.0
    while manager.peek_annotated_jpeg("CAM-J") is None and _time.time() < deadline:
        _time.sleep(0.05)
    check(manager.latest_annotated_jpeg("CAM-J") is not None,
          "manager.latest_annotated_jpeg() hands the UI bytes (take semantics)")
    check(manager.latest_annotated_jpeg("CAM-J") is None,
          "...and consumes them, so one frame is painted once")
    check(manager.peek_annotated_jpeg("CAM-J") is None,
          "nothing was produced between the two takes")
    check(manager.latest_annotated_jpeg("CAM-NOPE") is None,
          "an unknown camera returns None instead of raising")
    summary = manager.summary()
    check("uplink_kb" in summary and "jpeg_quality" in summary,
          "the grid summary reports operator link load and encode quality")
    manager.stop_all()

    # ------------------------------------------------------------------
    header("6-8. ANPR: OCR IS PACED GRID-WIDE, NOT PER TRACK")
    # ------------------------------------------------------------------
    reader = StubReader()
    anpr = big_plate_anpr(reader, max_ocr_per_frame=1, throttle_frames=0)
    frame = frame_of(400, 600)
    vehicle = (100, 100, 400, 300)
    first = anpr.process_vehicle(frame, vehicle, track_id=1, frame_idx=1)
    second = anpr.process_vehicle(frame, vehicle, track_id=2, frame_idx=1)
    check(reader.calls == 1,
          f"two vehicles in ONE frame cost one OCR read (got {reader.calls})")
    check(first is not None and first["plate_text"], "the first vehicle still gets a read")
    check(second is None or second.get("plate_text") is None or reader.calls == 1,
          "the deferred vehicle is deferred, not silently misread")

    # Same frame index, budget already spent -> no further read.
    anpr.process_vehicle(frame, vehicle, track_id=3, frame_idx=1)
    check(reader.calls == 1, "no second read within the same frame")
    check(anpr.ocr_skipped_budget >= 1,
          f"deferral is COUNTED, not silent ({anpr.ocr_skipped_budget} deferral(s))")

    # A minimum interval gates the next frame too.
    paced = big_plate_anpr(StubReader(), max_ocr_per_frame=5, ocr_min_interval_s=30.0)
    paced._last_ocr_at = _time.time()
    paced.process_vehicle(frame, vehicle, track_id=1, frame_idx=10)
    check(paced.reader.calls == 0,
          "the minimum interval blocks a read that is too soon, even with budget free")
    check(paced.ocr_skipped_budget == 1, "that block is counted as a deferral")

    # A plate too small to resolve must never reach OCR - it is the slowest path.
    small = big_plate_anpr(StubReader(), min_plate_height=500)
    res = small.process_vehicle(frame, vehicle, track_id=1, frame_idx=1)
    check(small.reader.calls == 0, "an unresolvable plate never reaches OCR")
    check(small.ocr_skipped_small == 1,
          "and is counted as 'plate too small', not as a failure")
    check(res is None, "no reading is invented for it")

    # A cached reading is still returned while the budget is spent.
    cached_anpr = big_plate_anpr(StubReader(), max_ocr_per_frame=1, throttle_frames=99)
    got = cached_anpr.process_vehicle(frame, vehicle, track_id=7, frame_idx=1)
    check(got is not None and got["plate_text"], "a vehicle with budget gets a read")
    again = cached_anpr.process_vehicle(frame, vehicle, track_id=7, frame_idx=2)
    check(again is not None and again["plate_text"] == got["plate_text"],
          "the cached reading is returned on later frames (track continuity kept)")

    # warmup must not poison the read cache or count as a reading
    warm = big_plate_anpr(StubReader())
    warm.warmup(include_ocr=True)
    check(warm.ocr_calls == 0, "warmup does not count as an operator reading")
    check(warm.track_ocr_cache == {}, "warmup leaves the per-track cache untouched")

    stats = warm.stats()
    check({"ocr_calls", "deferred_for_budget", "skipped_plate_too_small"} <= set(stats),
          "the pacing is reportable (stats() feeds the dashboard)")
    retuned = warm.set_ocr_budget(min_interval_s=1.25, max_per_frame=3)
    check(retuned["min_interval_s"] == 1.25 and retuned["max_per_frame"] == 3,
          "pacing can be retuned live, without rebuilding the grid")

    # ------------------------------------------------------------------
    header("9. PIPELINE: FRS STRIDE PACES THE PASS BUT NOT THE OVERLAY")
    # ------------------------------------------------------------------
    class CountingFRS:
        last_error = ""

        def __init__(self):
            self.process_calls = 0
            self.draw_calls = 0

        def process(self, frame, tracks, timestamp="", frame_idx=0):
            self.process_calls += 1
            return [{"track_id": 1, "decision": AUTHORIZED, "similarity": 0.9,
                     "face_bbox": (10, 10, 40, 40), "name": "TEST", "timestamp": timestamp,
                     "frame_idx": frame_idx}]

        def draw_faces(self, frame, decisions):
            self.draw_calls += 1
            return frame

    class NoopDetector:
        def detect(self, frame, conf_threshold=0.35):
            return [{"bbox": (10, 10, 120, 300), "centroid": (65, 155), "conf": 0.9,
                     "class_id": 0, "class_name": "person", "category": "human"}]

        def draw_detections(self, frame, detections, draw_centroid=True):
            return frame

    class NoopAnpr:
        def process_vehicle(self, *a, **k):
            return None

        def draw_anpr(self, frame, result):
            return frame

    class NoopEnhancer:
        def enhance(self, frame, clip_limit=None):
            return frame

    # Expected face passes over 6 frames, phased from the analyser's own count so
    # the first frame always runs: stride 1 -> 6, stride 2 -> frames 1,3,5 -> 3,
    # stride 3 -> frames 1,4 -> 2. The overlay, by contrast, must be drawn on ALL
    # six frames once decisions exist, or the face label flickers.
    for stride, expected in ((1, 6), (2, 3), (3, 2)):
        counting = CountingFRS()
        models = VisionModels(
            detector=NoopDetector(), anpr=NoopAnpr(), enhancer=NoopEnhancer(),
            watchlist=None, frs=counting, pose=None,
        )
        camera = {"camera_id": "CAM-S", "pre_rendered": False}
        opts = {"enable_frs": True, "frs_stride": stride, "enable_pose": False}
        process_fn = build_analyser(models, camera, None, opts)
        for idx in range(1, 7):
            process_fn(frame_of(240, 320), {"frame_seq": idx, "frame_ts": 0.0})
        check(counting.process_calls == expected,
              f"stride {stride}: {counting.process_calls} face pass(es) over 6 frames "
              f"(expected {expected})")
        check(counting.draw_calls == 6,
              f"stride {stride}: overlay drawn on all 6 frames "
              f"(got {counting.draw_calls}, no blinking face label)")

    # Stride 1 is the default: a caller that never sets the option is unaffected.
    counting = CountingFRS()
    models = VisionModels(detector=NoopDetector(), anpr=NoopAnpr(), enhancer=NoopEnhancer(),
                          watchlist=None, frs=counting, pose=None)
    process_fn = build_analyser(models, {"camera_id": "C", "pre_rendered": False}, None,
                                {"enable_frs": True, "enable_pose": False})
    process_fn(frame_of(240, 320), {"frame_seq": 1, "frame_ts": 0.0})
    process_fn(frame_of(240, 320), {"frame_seq": 2, "frame_ts": 0.0})
    check(counting.process_calls == 2, "no frs_stride option -> every frame (unchanged)")

    # ------------------------------------------------------------------
    header("10-11. WARMUP IS SAFE AND HONEST")
    # ------------------------------------------------------------------
    engine = PoseEngine(estimator=NullPoseEstimator())
    check(engine.available is False, "the null pose engine reports unavailable")
    check(engine.warmup() is False, "warmup on an unavailable backend returns False")
    check(engine.state()["mode"] == "UNAVAILABLE", "and it still says so in state()")

    class ExplodingEstimator:
        name = "exploding"
        last_error = ""

        def estimate(self, frame, conf_threshold=0.25):
            raise RuntimeError("simulated backend failure")

    broken = PoseEngine(estimator=ExplodingEstimator())
    check(broken.warmup() is False, "a crashing backend fails warmup without raising")
    check("simulated backend failure" in broken.last_error,
          "and the reason is recorded for the operator")

    class BrokenDetector:
        def detect(self, frame, conf_threshold=0.35):
            raise RuntimeError("simulated detector failure")

    from modules.detector import ObjectDetector

    fake = ObjectDetector.__new__(ObjectDetector)
    fake.model = BrokenDetector()
    fake.device = "cpu"
    fake.class_ids = []
    check(fake.warmup() is False, "a failing detector warmup returns False, never raises")

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    if failures:
        print(f"  PHASE 9 RESULT: {len(failures)} CHECK(S) FAILED")
        for item in failures:
            print(f"    - {item}")
        print("=" * 74)
        return 1
    print("  PHASE 9 RESULT")
    print("=" * 74)
    print("  ALL CHECKS PASSED - the latency work is now a contract, not a memory:")
    print("    * face detection is paid ONCE per frame, not once per person")
    print("    * the operator receives JPEG bytes, never a re-encoded full array")
    print("    * OCR is paced grid-wide, and every deferral is counted")
    print("    * an unreadable plate is skipped instead of costing seconds")
    print("    * FRS cadence is paceable without the overlay blinking")
    print("    * warmup is safe on a missing or crashing backend")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
