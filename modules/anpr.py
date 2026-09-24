import os
import re
import time
import cv2
import numpy as np
from ultralytics import YOLO

# NOTE: KMP_DUPLICATE_LIB_OK=TRUE used to be set here. It was removed on purpose.
# It is Intel's "continue anyway" escape hatch for a duplicated OpenMP runtime,
# and Intel's own message warns it "may cause crashes or silently produce
# incorrect results". This is a system deciding whether someone crossed a border:
# silently wrong tensors are worse than a loud crash. The real fix is an isolated
# virtualenv (no --system-site-packages); see README "Environment".
os.environ["PYTHONIOENCODING"] = "utf-8"


class CascadedANPR:
    """
    Cascaded Automatic Number Plate Recognition (ANPR) Module.
    - Triggered ONLY on detected vehicle classes (car, truck, bus, motorcycle).
    - Detects license plate ROI using YOLOv8 plate detector.
    - Preprocesses in-memory plate crop and extracts text via EasyOCR.
    - Applies temporal caching to retain highest-confidence reading per vehicle track ID.
    - Throttles per-track OCR calls to maintain real-time performance.
    - Flags ambiguous or low-confidence reads for manual operator review.
    """
    def __init__(
        self,
        plate_model_path: str = "models/licensePlateDetector.pt",
        device: str = "cpu",
        ocr_conf_threshold: float = 0.40,
        throttle_frames: int = 15,
        min_vehicle_height: int = 50,
        min_plate_height: int = 16,
        ocr_min_interval_s: float = 0.35,
        max_ocr_per_frame: int = 1
    ):
        self.plate_model = YOLO(plate_model_path)
        self.device = device
        self.ocr_conf_threshold = ocr_conf_threshold
        self.throttle_frames = throttle_frames
        self.min_vehicle_height = min_vehicle_height
        self.min_plate_height = min_plate_height
        # Global OCR pacing. The per-track throttle alone still multiplies the
        # OCR cost by the number of vehicles in view, so the budget has to be
        # grid-wide, not per-track.
        self.ocr_min_interval_s = max(0.0, float(ocr_min_interval_s))
        self.max_ocr_per_frame = max(1, int(max_ocr_per_frame))

        # Budget bookkeeping, surfaced through stats() so the throttling is
        # visible to an operator instead of looking like a dead ANPR tab.
        self.ocr_calls = 0
        self.ocr_skipped_budget = 0
        self.ocr_skipped_small = 0
        self._last_ocr_at = 0.0
        self._ocr_frame = -1
        self._ocr_this_frame = 0

        # Initialize EasyOCR reader (CPU mode, English)
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        # Cache: track_id -> dict(plate_text, ocr_conf, status, plate_bbox, last_frame)
        self.track_ocr_cache = {}

    def warmup(self, include_ocr: bool = True) -> bool:
        """
        Pays the first-call cost of BOTH stages at load time.

        The plate detector initialises its graph on first use (~1-2 s) and
        EasyOCR's first readtext is worse (~3-5 s on CPU). On live video that lands
        on the first vehicle to appear, which looks like the ANPR tab hanging.
        `include_ocr=False` warms only the detector, for callers that want to keep
        startup fast and accept a one-off cost on the first real plate.
        """
        ok = True
        try:
            blank = np.zeros((320, 640, 3), dtype="uint8")
            self.plate_model(blank, conf=0.25, device=self.device, verbose=False)
        except Exception:
            ok = False
        if include_ocr:
            try:
                # A blank crop is enough to initialise the recogniser; the result
                # is discarded, and this never touches the read cache.
                self.reader.readtext(
                    np.full((48, 160), 255, dtype="uint8"),
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    detail=1,
                )
            except Exception:
                ok = False
        return ok

    def clean_text(self, text: str) -> str:
        """Strips noise and keeps alphanumeric plate characters."""
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        return cleaned

    def preprocess_plate(self, crop: np.ndarray) -> np.ndarray:
        """Upscales and normalizes contrast for optimal OCR parsing."""
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop

        scale = max(2.0, 80.0 / max(h, 1))
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return norm

    def process_vehicle(self, frame: np.ndarray, vehicle_bbox: tuple, track_id: int = None, frame_idx: int = 0):
        """
        Runs cascaded plate detection on a vehicle ROI when in inspection range.
        """
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vh = vy2 - vy1

        # Only inspect vehicles close enough for plate resolution
        if vh < self.min_vehicle_height:
            if track_id in self.track_ocr_cache:
                return self.track_ocr_cache[track_id]
            return None

        # Check cache and throttling
        if track_id is not None and track_id in self.track_ocr_cache:
            cached = self.track_ocr_cache[track_id]

            # If already verified with high confidence (>65%), return cached read permanently
            if cached["status"] == "VERIFIED" and cached["ocr_conf"] > 0.65:
                return cached

            # Throttling between attempts: wait at least throttle_frames before re-attempting OCR
            if (frame_idx - cached.get("last_frame", 0)) < self.throttle_frames:
                return cached

        # ---- Global OCR budget gate --------------------------------------
        # Measured on this project's own checkpost feed: one EasyOCR call costs
        # ~1.6 s on an edge CPU (4.9 s worst case). A per-track throttle still
        # multiplies that by the number of vehicles in view - three cars means
        # ~5 s of OCR for a single frame, which the operator experiences as a
        # frozen feed. So attempts are paced grid-wide: at most
        # `max_ocr_per_frame` per analysed frame and no more often than
        # `ocr_min_interval_s`. Nothing is lost: a deferred track keeps its
        # cached best read and simply takes the next free slot, and the plate
        # detector pass below is skipped too, so deferral is cheap.
        now = time.time()
        if self._ocr_frame != frame_idx:
            self._ocr_frame = frame_idx
            self._ocr_this_frame = 0
        if not (
            self._ocr_this_frame < self.max_ocr_per_frame
            and (now - self._last_ocr_at) >= self.ocr_min_interval_s
        ):
            self.ocr_skipped_budget += 1
            if track_id is not None and track_id in self.track_ocr_cache:
                return self.track_ocr_cache[track_id]
            return None

        fh, fw = frame.shape[:2]
        vx1, vy1 = max(0, vx1), max(0, vy1)
        vx2, vy2 = min(fw, vx2), min(fh, vy2)

        veh_crop = frame[vy1:vy2, vx1:vx2]
        if veh_crop.size == 0 or veh_crop.shape[0] < 30 or veh_crop.shape[1] < 40:
            return None

        # 1. License Plate Detection inside vehicle ROI
        plate_results = self.plate_model(
            veh_crop,
            conf=0.25,
            device=self.device,
            verbose=False
        )

        if not plate_results or plate_results[0].boxes is None or len(plate_results[0].boxes) == 0:
            return None

        best_box = max(plate_results[0].boxes, key=lambda b: float(b.conf[0].item()))
        bx1, by1, bx2, by2 = map(int, best_box.xyxy[0].tolist())
        plate_conf = float(best_box.conf[0].item())

        # Map back to full frame coordinates
        px1, py1 = vx1 + bx1, vy1 + by1
        px2, py2 = vx1 + bx2, vy1 + by2
        px1, py1 = max(0, px1 - 2), max(0, py1 - 2)
        px2, py2 = min(fw, px2 + 2), min(fh, py2 + 2)

        plate_crop = frame[py1:py2, px1:px2]
        if plate_crop.size == 0 or plate_crop.shape[0] < 12 or plate_crop.shape[1] < 24:
            return None

        # 2a. Resolution gate BEFORE paying for OCR. A plate box only a few
        # pixels tall cannot be read by any model: EasyOCR is both slowest and
        # least accurate on hopeless crops, so the honest answer is "not yet
        # resolvable" - the vehicle is tracked and retried as it approaches and
        # its plate grows, which is the real-world case anyway.
        if (py2 - py1) < self.min_plate_height:
            self.ocr_skipped_small += 1
            if track_id is not None and track_id in self.track_ocr_cache:
                return self.track_ocr_cache[track_id]
            return None

        # 2. OCR on preprocessed plate crop. Plates are a closed alphabet, so
        # constraining recognition to it removes whole classes of misreads
        # (punctuation, symbols), and plates often OCR as several fragments -
        # left cluster + right cluster - so fragment order matters when joining.
        prep = self.preprocess_plate(plate_crop)
        # Budget consumed here, at the point the expensive call actually happens.
        self._last_ocr_at = time.time()
        self._ocr_this_frame += 1
        self.ocr_calls += 1
        ocr_res = self.reader.readtext(
            prep,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
        )

        parsed_text = ""
        ocr_conf = 0.0

        if ocr_res:
            # Horizontal reading order (x-center), then keep the longest text:
            # a joined plate beats any single fragment for recall, and the
            # confidence recorded is the best fragment's, not a fiction.
            ordered = sorted(ocr_res, key=lambda r: (r[0][0][0], r[0][0][1]))
            parsed_text = self.clean_text("".join(r[1] for r in ordered))
            ocr_conf = float(max(r[2] for r in ordered))

        # 3. Validation & Status assignment
        if len(parsed_text) >= 4 and ocr_conf >= self.ocr_conf_threshold:
            status = "VERIFIED"
        else:
            status = "FLAGGED_FOR_MANUAL_REVIEW"
            if not parsed_text:
                parsed_text = "PLATE_UNREADABLE"

        result = {
            "track_id": track_id,
            "plate_bbox": (px1, py1, px2, py2),
            "plate_conf": round(plate_conf, 3),
            "plate_text": parsed_text,
            "ocr_conf": round(ocr_conf, 3),
            "status": status,
            "last_frame": frame_idx
        }

        # 4. Temporal Best-per-Track Smoothing
        if track_id is not None:
            if track_id in self.track_ocr_cache:
                old = self.track_ocr_cache[track_id]
                # If new result is verified or has better confidence, update cache
                if ocr_conf > old["ocr_conf"] or (old["status"] != "VERIFIED" and status == "VERIFIED"):
                    self.track_ocr_cache[track_id] = result
                else:
                    # Update timestamp only, retain superior reading
                    old["last_frame"] = frame_idx
                    result = old
            else:
                self.track_ocr_cache[track_id] = result

        return result

    def set_ocr_budget(self, min_interval_s: float = None, max_per_frame: int = None) -> dict:
        """
        Live retune of the OCR pacing - the operator's throughput knob.

        Deliberately a setter on the RUNNING instance rather than a constructor
        argument: re-keying the camera grid to change pacing would tear down every
        ingest and analysis thread mid-demo. This changes one number.
        """
        if min_interval_s is not None:
            self.ocr_min_interval_s = max(0.0, float(min_interval_s))
        if max_per_frame is not None:
            self.max_ocr_per_frame = max(1, int(max_per_frame))
        return self.stats()

    def stats(self) -> dict:
        """
        OCR pacing telemetry - so a throttled ANPR reads as "paced", never as
        "broken". Equivalent to grid health for the ingestion layer.
        """
        return {
            "ocr_calls": self.ocr_calls,
            "deferred_for_budget": self.ocr_skipped_budget,
            "skipped_plate_too_small": self.ocr_skipped_small,
            "tracks_cached": len(self.track_ocr_cache),
            "min_interval_s": self.ocr_min_interval_s,
            "max_per_frame": self.max_ocr_per_frame,
        }

    def draw_anpr(self, frame: np.ndarray, anpr_result: dict):
        """Draws plate box and status label."""
        if not anpr_result or "plate_bbox" not in anpr_result:
            return frame

        px1, py1, px2, py2 = anpr_result["plate_bbox"]
        status = anpr_result["status"]
        text = anpr_result["plate_text"]
        conf = anpr_result["ocr_conf"]

        color = (0, 255, 255) if status == "VERIFIED" else (0, 140, 255)

        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        label = f"PLATE: {text} ({conf*100:.0f}%)" if status == "VERIFIED" else f"! {text} (MANUAL REVIEW) !"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        (tw, th), baseline = cv2.getTextSize(label, font, scale, 1)
        tag_y = max(0, py1 - 4)
        cv2.rectangle(frame, (px1, tag_y - th - 4), (px1 + tw + 6, tag_y), color, -1)
        cv2.putText(frame, label, (px1 + 2, tag_y - 2), font, scale, (0, 0, 0), 1, cv2.LINE_AA)

        return frame
