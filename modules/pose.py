"""
Suspicious-behaviour analytics from human pose keypoints (AnantaNetra / IBVAP).

What this adds
--------------
The platform could already answer *did someone cross the line?* and *who are
they?*. It could not answer the question that precedes both: **is this person
behaving in a way that warrants attention?** A man lying still for four minutes,
a figure crawling low under the tripwire, someone stopping dead at the fence for
thirty seconds: none of those are line crossings, and all of them matter at a
border.

How it works
------------
A pose estimator (YOLOv8-Pose via ultralytics, or any injected backend) returns
17 COCO keypoints per person. From those keypoints this module derives a small set
of physically-meaningful body metrics - torso inclination, limb-normalised speed,
foot-point dwell spread, hip rise/drop - and evaluates them over a time window.

Two deliberate design decisions
-------------------------------
1. **Scale independence.** Every threshold is expressed in units of the person's
   own body height (heights/second, fraction-of-height) rather than in pixels, so
   the same calibration holds for a figure 4 m from the camera and one 40 m away.
   Thresholds in pixels are the reason naive behaviour analytics fall apart on a
   long-range perimeter camera.

2. **Honesty guard.** These are *heuristics on geometry*, not trained behaviour
   classifiers. With no pose model present the engine reports UNAVAILABLE and
   emits nothing - it never fabricates a behaviour. Every event it does emit
   carries `provenance: POSE_HEURISTIC` so an audit trail can never present a
   geometric inference as a classified fact.

Behaviours detected
-------------------
  LOITERING          stopped and lingering in a small area
  RUNNING            sustained limb-normalised speed
  CROUCH_INTRUSION   low, wide, near-horizontal posture (crawling under a fence)
  FALL_ALERT         sharp drop to a sustained horizontal, motionless posture
  FENCE_CLIMB        hands above the tripwire with hips rising
  ARM_RAISED_SIGNAL  a sustained raised arm while otherwise stationary
  GROUP_CONVERGENCE  several people closing to within a body-length of each other

No third-party dependency is required at import time; numpy and OpenCV are both
optional. That keeps this module verifiable (and loadable on a bare edge box)
with no model weights installed.
"""

import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# COCO-17 keypoint layout (what YOLOv8-Pose emits)
# ---------------------------------------------------------------------------
COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
KEYPOINT_INDEX = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

# Skeleton edges used by the on-frame overlay.
SKELETON_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
)

# ---------------------------------------------------------------------------
# Behaviour catalogue
# ---------------------------------------------------------------------------
# severity here is the wire priority class; the telemetry module owns the final
# mapping so the map, the log and the packet can never disagree.
BEHAVIOR_CATALOGUE: Dict[str, dict] = {
    "FENCE_CLIMB": {
        "label": "Fence Climb Attempt",
        "severity": "CRITICAL",
        "description": "Hands above the tripwire with the body rising - a climb, not a crossing.",
    },
    "FALL_ALERT": {
        "label": "Person Down / Fall Detected",
        "severity": "CRITICAL",
        "description": "Sharp drop to a sustained horizontal, motionless posture.",
    },
    "CROUCH_INTRUSION": {
        "label": "Low Crawl / Crouch",
        "severity": "WARNING",
        "description": "Sustained low, wide, near-horizontal posture - evasion of the tripwire.",
    },
    "LOITERING": {
        "label": "Loitering",
        "severity": "WARNING",
        "description": "Stationary in a small area beyond the dwell threshold.",
    },
    "RUNNING": {
        "label": "Suspicious Running",
        "severity": "WARNING",
        "description": "Sustained speed inconsistent with patrolling or grazing traffic.",
    },
    "GROUP_CONVERGENCE": {
        "label": "Group Convergence",
        "severity": "NOTICE",
        "description": "Several people closing to within a body-length of each other.",
    },
    "ARM_RAISED_SIGNAL": {
        "label": "Sustained Raised Arm",
        "severity": "NOTICE",
        "description": "A held raised arm while otherwise stationary - possible signalling.",
    },
}

# Order the dashboard lists them in (most operationally urgent first).
BEHAVIOR_ORDER = (
    "FENCE_CLIMB", "FALL_ALERT", "CROUCH_INTRUSION",
    "LOITERING", "RUNNING", "GROUP_CONVERGENCE", "ARM_RAISED_SIGNAL",
)


@dataclass
class BehaviorThresholds:
    """
    Tunables, all in body-relative units so one calibration covers the whole
    perimeter. Defaults are deliberately conservative: a border post that cries
    wolf gets switched off by its own operators, which is the real failure mode.
    """

    # Signal quality gates
    min_keypoint_conf: float = 0.30
    min_track_height_px: float = 40.0
    window_s: float = 8.0
    max_samples: int = 150
    cooldown_s: float = 20.0

    # LOITERING
    loiter_min_dwell_s: float = 8.0
    loiter_radius_ratio: float = 0.35      # spread of foot points / body height
    loiter_max_speed_ratio: float = 0.25   # heights per second

    # RUNNING
    run_speed_ratio: float = 1.60          # heights per second
    run_sustain_s: float = 1.0

    # CROUCH / CRAWL
    crouch_torso_angle_deg: float = 55.0
    crouch_aspect_ratio: float = 0.85      # width / height of the person box
    crouch_flat_aspect_ratio: float = 1.05  # box wider than tall: flat on the ground
    crouch_sustain_s: float = 0.7

    # FALL
    fall_torso_angle_deg: float = 65.0
    fall_aspect_ratio: float = 1.15
    fall_drop_ratio: float = 0.22          # hip drop / body height
    fall_sustain_s: float = 1.2
    fall_max_speed_ratio: float = 0.35     # motionless AFTER the drop

    # FENCE CLIMB
    climb_hand_clearance_ratio: float = 0.10   # wrists above shoulders / height
    climb_rise_ratio: float = 0.12             # hip rise / height
    climb_sustain_s: float = 0.5
    climb_fence_tolerance_ratio: float = 0.30  # hands this close to the fence line

    # RAISED ARM
    signal_arm_sustain_s: float = 2.5
    signal_max_speed_ratio: float = 0.30

    # GROUP CONVERGENCE
    group_min_persons: int = 3
    group_radius_ratio: float = 1.60       # pairwise foot distance / mean height
    group_sustain_s: float = 3.0


# ---------------------------------------------------------------------------
# Pose sample
# ---------------------------------------------------------------------------
class PoseSample:
    """
    One person's keypoints at one instant, with the derived body metrics that the
    behaviour rules actually use.

    Metrics are computed once on construction and cached, because the same
    sample is re-read on every window evaluation.
    """

    __slots__ = (
        "track_id", "keypoints", "bbox", "detection_conf", "ts", "frame_idx",
        "height", "width", "aspect", "foot_point", "shoulder_mid", "hip_mid",
        "torso_angle_deg", "arms_over_head", "min_wrist_y", "max_wrist_y",
        "shoulder_y", "hip_y", "nose_y", "valid", "valid_ratio",
        "_threshold",
    )

    def __init__(
        self,
        track_id: int,
        keypoints: Sequence[Sequence[float]],
        bbox: Optional[Sequence[float]] = None,
        detection_conf: float = 1.0,
        ts: float = 0.0,
        frame_idx: int = 0,
        min_keypoint_conf: float = 0.30,
    ):
        self.track_id = track_id
        # Normalise to a plain list of (x, y, conf) triples.
        self.keypoints = [
            (float(k[0]), float(k[1]), float(k[2]) if len(k) > 2 else 1.0)
            for k in (keypoints or [])
        ]
        self.ts = float(ts)
        self.frame_idx = int(frame_idx)
        self.detection_conf = float(detection_conf)
        self.height = 0.0
        self.width = 0.0
        self.aspect = 0.0
        self.foot_point = (0.0, 0.0)
        self.shoulder_mid = (0.0, 0.0)
        self.hip_mid = (0.0, 0.0)
        self.torso_angle_deg = 0.0
        self.arms_over_head = False
        self.min_wrist_y = None
        self.max_wrist_y = None
        self.shoulder_y = None
        self.hip_y = None
        self.nose_y = None
        self.valid = False
        self.valid_ratio = 0.0

        self._threshold = float(min_keypoint_conf)
        self._resolve_bbox(bbox)
        self._derive()

    # -- helpers ------------------------------------------------------------
    def point(self, name: str) -> Optional[Tuple[float, float, float]]:
        index = KEYPOINT_INDEX.get(name)
        if index is None or index >= len(self.keypoints):
            return None
        return self.keypoints[index]

    def _confident(self, name: str) -> Optional[Tuple[float, float]]:
        point = self.point(name)
        if point is None or point[2] < self._threshold:
            return None
        return (point[0], point[1])

    @staticmethod
    def _mid(a, b) -> Optional[Tuple[float, float]]:
        if a is None or b is None:
            return None
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    def _resolve_bbox(self, bbox) -> None:
        """
        Prefers a detected box, but falls back to the keypoint extent so a pose
        with no detection box still yields usable scale.
        """
        if bbox is not None:
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox)
                self.width = max(0.0, x2 - x1)
                self.height = max(0.0, y2 - y1)
                self.foot_point = ((x1 + x2) / 2.0, y2)
            except Exception:
                pass
        if self.height <= 0.0:
            confident = [k for k in self.keypoints if k[2] >= self._threshold]
            if confident:
                xs = [k[0] for k in confident]
                ys = [k[1] for k in confident]
                self.width = max(xs) - min(xs)
                self.height = max(ys) - min(ys)
                self.foot_point = (sum(xs) / len(xs), max(ys))
        self.aspect = (self.width / self.height) if self.height > 0 else 0.0

    def _derive(self) -> None:
        confident_count = sum(1 for k in self.keypoints if k[2] >= self._threshold)
        self.valid_ratio = confident_count / max(1, len(self.keypoints))

        shoulders = (
            self._confident("left_shoulder"), self._confident("right_shoulder")
        )
        hips = (self._confident("left_hip"), self._confident("right_hip"))
        wrists = (self._confident("left_wrist"), self._confident("right_wrist"))

        self.shoulder_mid = self._mid(*shoulders) or (0.0, 0.0)
        self.hip_mid = self._mid(*hips) or (0.0, 0.0)
        nose = self._confident("nose")

        confident_shoulders = [s for s in shoulders if s is not None]
        confident_hips = [h for h in hips if h is not None]
        confident_wrists = [w for w in wrists if w is not None]

        # A torso needs at least one shoulder and one hip to be meaningful.
        self.valid = bool(confident_shoulders and confident_hips and self.height > 0)
        if not self.valid:
            return

        self.shoulder_y = sum(s[1] for s in confident_shoulders) / len(confident_shoulders)
        self.hip_y = sum(h[1] for h in confident_hips) / len(confident_hips)
        if confident_wrists:
            self.min_wrist_y = min(w[1] for w in confident_wrists)
            self.max_wrist_y = max(w[1] for w in confident_wrists)
        self.nose_y = nose[1] if nose else None

        # Torso inclination from vertical. 0 deg = upright, 90 deg = horizontal.
        dx = self.hip_mid[0] - self.shoulder_mid[0]
        dy = self.hip_mid[1] - self.shoulder_mid[1]
        self.torso_angle_deg = math.degrees(math.atan2(abs(dx), max(1e-6, abs(dy))))

        # "Hands over head": either wrist above the nose, or clearly above the
        # shoulder line. The second test is what catches a climb, where the head
        # may be looking down at the wall.
        above = False
        if confident_wrists:
            if self.nose_y is not None and self.min_wrist_y < self.nose_y:
                above = True
            if self.min_wrist_y < self.shoulder_y - 0.02 * self.height:
                above = True
        self.arms_over_head = above

    # -- convenience --------------------------------------------------------
    def describe(self) -> str:
        return (
            f"torso {self.torso_angle_deg:.0f}deg, aspect {self.aspect:.2f}, "
            f"h {self.height:.0f}px, kv {self.valid_ratio * 100:.0f}%"
        )


def make_sample(
    track_id: int,
    keypoints: Sequence[Sequence[float]],
    bbox: Optional[Sequence[float]] = None,
    detection_conf: float = 1.0,
    ts: float = 0.0,
    frame_idx: int = 0,
    min_keypoint_conf: float = 0.30,
) -> PoseSample:
    """Explicit constructor - keypoints come from many different backends."""
    return PoseSample(
        track_id, keypoints, bbox=bbox, detection_conf=detection_conf,
        ts=ts, frame_idx=frame_idx, min_keypoint_conf=min_keypoint_conf,
    )


# ---------------------------------------------------------------------------
# Per-track rolling state
# ---------------------------------------------------------------------------
class TrackPoseState:
    """Rolling keypoint history for one tracked person."""

    __slots__ = ("track_id", "samples", "last_seen", "first_seen", "missing")

    def __init__(self, track_id: int, max_samples: int = 150):
        self.track_id = track_id
        self.samples: deque = deque(maxlen=int(max_samples))
        self.last_seen = 0.0
        self.first_seen = 0.0
        self.missing = 0

    def append(self, sample: PoseSample) -> None:
        if not self.samples:
            self.first_seen = sample.ts
        self.samples.append(sample)
        self.last_seen = sample.ts
        self.missing = 0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _sustained(
    flags: Sequence[bool], times: Sequence[float], sustain_s: float,
    min_fraction: float = 0.8,
) -> Tuple[bool, float]:
    """
    True when `flags` are mostly True across a trailing time span of `sustain_s`.

    Trailing-window evaluation is what stops a single noisy frame from emitting an
    alert - a real behavioural signal persists, a keypoint twitch does not.
    """
    if not flags:
        return (False, 0.0)
    end = times[-1]
    start = end - float(sustain_s)
    window = [(f, t) for f, t in zip(flags, times) if t >= start]
    if not window:
        return (False, 0.0)
    true_count = sum(1 for f, _ in window if f)
    fraction = true_count / len(window)
    span_covered = end - window[0][1]
    ok = fraction >= min_fraction and span_covered >= sustain_s * 0.75
    return (ok, max(0.0, end - window[0][1]) if ok else 0.0)


# ---------------------------------------------------------------------------
# Behaviour analyser
# ---------------------------------------------------------------------------
class BehaviorAnalyser:
    """
    Turns a stream of pose samples into behaviour alerts.

    Windows and thresholds are per-analyser, and per (track, behaviour) cooldowns
    prevent a man loitering for five minutes from producing three hundred alerts.
    """

    def __init__(
        self,
        thresholds: Optional[BehaviorThresholds] = None,
        enabled: Optional[Iterable[str]] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.thresholds = thresholds or BehaviorThresholds()
        self.enabled: Set[str] = set(enabled) if enabled else set(BEHAVIOR_ORDER)
        self._clock = clock
        self.tracks: "OrderedDict[int, TrackPoseState]" = OrderedDict()
        self._last_fired: Dict[Tuple[int, str], float] = {}
        # Sentinel must be -inf, not 0.0: on a synthetic/edge clock that starts
        # near zero, a 0.0 sentinel reads as "fired moments ago" and silently
        # swallows the first group alert of the session.
        self._group_last_fired = -float("inf")

        self.samples_ingested = 0
        self.events_emitted = 0
        self.suppressed_by_cooldown = 0
        self.behaviors_disabled = 0
        self.last_error = ""
        self.counts: Dict[str, int] = {name: 0 for name in BEHAVIOR_ORDER}

    # -- window maths -------------------------------------------------------
    def _recently_fired(self, track_id: int, behaviors: Iterable[str], now: float) -> bool:
        """
        True when a related, higher-priority behaviour already fired for this
        track inside the cooldown window.

        This is what keeps one physical event from producing a chain of
        overlapping alerts across successive frames (e.g. a climb immediately
        followed by 'raised arm', or a fall followed by 'crawling').
        """
        for behavior in behaviors:
            last = self._last_fired.get((int(track_id), behavior))
            if last is not None and (now - last) < self.thresholds.cooldown_s:
                return True
        return False

    def _metrics(self, state: TrackPoseState) -> dict:
        """
        Computes the body-relative metrics for a track from its rolling window.

        All speeds are in body-heights per second and all distances in fractions
        of body height, which is what makes one calibration work across range.
        """
        samples = [s for s in state.samples if s.valid]
        if not samples:
            return {}

        height = _median([s.height for s in samples]) or samples[-1].height
        if height <= 0:
            return {}

        times = [s.ts for s in samples]
        speeds: List[float] = []
        speed_times: List[float] = []
        drops: List[float] = []
        for previous, current in zip(samples, samples[1:]):
            dt = current.ts - previous.ts
            if dt <= 1e-3:
                continue
            distance = math.hypot(
                current.foot_point[0] - previous.foot_point[0],
                current.foot_point[1] - previous.foot_point[1],
            )
            speeds.append(distance / dt / height)
            speed_times.append(current.ts)
            # Hip drop measured against the track's own height: a fall is a fast
            # downward hip excursion, not merely a low posture.
            drops.append((current.hip_y - previous.hip_y) / height)

        foot_points = [s.foot_point for s in samples]
        spread = 0.0
        for i, first in enumerate(foot_points):
            for second in foot_points[i + 1 :]:
                spread = max(spread, math.hypot(first[0] - second[0], first[1] - second[1]))

        torso_angles = [s.torso_angle_deg for s in samples]
        aspects = [s.aspect for s in samples]

        return {
            "height_px": height,
            "dwell_s": times[-1] - times[0] if len(times) > 1 else 0.0,
            "speed_hps": speeds[-1] if speeds else 0.0,
            "mean_speed_hps": _median(speeds) if speeds else 0.0,
            "speeds": speeds,
            "speed_times": speed_times,
            "spread_ratio": (spread / height) if height else 0.0,
            "torso_angle_deg": _median(torso_angles),
            "torso_angles": torso_angles,
            "aspect": _median(aspects),
            "aspects": aspects,
            "hip_y": samples[-1].hip_y,
            "hip_y_min": min(s.hip_y for s in samples),
            # Drop measured from the window's highest hip position (a fall), and
            # rise measured start-to-now (a climb). These are genuinely different
            # quantities: a drop-from-minimum can never be negative, so it can
            # never express "rising" - conflating the two silently disables the
            # climb rule.
            "hip_drop_ratio": (samples[-1].hip_y - min(s.hip_y for s in samples)) / height,
            "hip_rise_ratio": (samples[-1].hip_y - samples[0].hip_y) / height,
            "max_drop": max(drops) if drops else 0.0,
            "shoulder_y": samples[-1].shoulder_y,
            "min_wrist_y": min(
                (s.min_wrist_y for s in samples if s.min_wrist_y is not None),
                default=None,
            ),
            "latest_wrist_y": samples[-1].min_wrist_y,
            "arms_flags": [s.arms_over_head for s in samples],
            # Per-sample shoulder-to-wrist clearance, in body heights. The climb
            # rule needs this rather than the coarse arms-over-head flag: a climber
            # whose head rises past the wire still has their arms extended above
            # the shoulders, and the climb must not stop being detected there.
            "clearances": [
                (
                    (s.shoulder_y - s.min_wrist_y) / s.height
                    if (
                        s.height > 0
                        and s.shoulder_y is not None
                        and s.min_wrist_y is not None
                    )
                    else -1.0
                )
                for s in samples
            ],
            "times": times,
            "latest": samples[-1],
        }

    def metrics_for(self, track_id: int) -> dict:
        """Metrics for the dashboard table (empty dict when nothing is tracked)."""
        state = self.tracks.get(track_id)
        return self._metrics(state) if state else {}

    # -- alert construction -------------------------------------------------
    def _emit(
        self,
        behavior: str,
        track_id: int,
        confidence: float,
        detail: str,
        timestamp: Optional[str],
        ts: float,
        bbox,
        duration_s: float = 0.0,
    ) -> Optional[dict]:
        if behavior not in self.enabled:
            self.behaviors_disabled += 1
            return None

        key = (int(track_id), behavior)
        last = self._last_fired.get(key)
        if last is not None and (ts - last) < self.thresholds.cooldown_s:
            self.suppressed_by_cooldown += 1
            return None
        self._last_fired[key] = ts
        self.counts[behavior] = self.counts.get(behavior, 0) + 1
        self.events_emitted += 1

        meta = BEHAVIOR_CATALOGUE.get(behavior, {})
        return {
            # The event_type is the behaviour itself: the operator should see
            # "FENCE_CLIMB", not a generic BEHAVIOR_ALERT to go and look up.
            "event_type": behavior,
            "behavior": behavior,
            "behavior_label": meta.get("label", behavior),
            "severity": meta.get("severity", "NOTICE"),
            "track_id": int(track_id),
            "category": "human",
            "class_name": "person",
            "confidence": round(float(confidence), 3),
            "status": meta.get("severity", "NOTICE"),
            "details": detail,
            "zone": "Behaviour Analytics",
            "direction": "",
            "identity": "",
            "location": "",
            "timestamp": timestamp,
            "duration_s": round(float(duration_s), 2),
            # Provenance: a geometric inference must never be presented as a
            # classified fact in the audit trail.
            "provenance": "POSE_HEURISTIC",
            "bbox": bbox,
        }

    # -- main entry ---------------------------------------------------------
    def update(
        self,
        samples: Iterable[PoseSample],
        timestamp: Optional[str] = None,
        fence_y: Optional[float] = None,
    ) -> List[dict]:
        """
        Ingests this frame's pose samples and returns any behaviour alerts.

        `fence_y` is the calibrated tripwire row in frame coordinates; when given,
        the climb rule requires the hands to be near it, which is what separates
        "climbing the fence" from "stretching at the checkpost".
        """
        alerts: List[dict] = []
        thresholds = self.thresholds
        try:
            accepted = []
            for sample in samples or []:
                if not sample.valid:
                    continue
                if sample.height < thresholds.min_track_height_px:
                    # Too small to read a torso from: geometry would be noise.
                    continue
                accepted.append(sample)
                state = self.tracks.get(sample.track_id)
                if state is None:
                    state = TrackPoseState(sample.track_id, thresholds.max_samples)
                    self.tracks[sample.track_id] = state
                state.append(sample)
                self.samples_ingested += 1

            now = accepted[-1].ts if accepted else self._clock()
            self._expire(now)

            for sample in accepted:
                state = self.tracks.get(sample.track_id)
                if state is None:
                    continue
                metrics = self._metrics(state)
                if not metrics:
                    continue
                alerts.extend(
                    self._evaluate(sample, metrics, timestamp, fence_y)
                )

            # Group geometry is evaluated on the CURRENT frame only: one sample
            # per track. Passing the whole batch of accepted samples would let a
            # single person walking across the frame chain into a "group" of one
            # track with itself, which is how this rule would otherwise false-fire
            # on every moving subject.
            seen = {sample.track_id for sample in accepted}
            latest_per_track = [
                state.samples[-1]
                for track_id, state in self.tracks.items()
                if track_id in seen and state.samples
            ]
            alerts.extend(self._evaluate_group(latest_per_track, timestamp))
        except Exception as exc:  # analytics must never break the video loop
            self.last_error = f"{type(exc).__name__}: {exc}"
        return alerts

    def _expire(self, now: float) -> None:
        """Drops tracks that have left the frame, and their cooldown history."""
        horizon = self.thresholds.window_s * 2.5
        stale = [
            tid for tid, state in self.tracks.items()
            if now - state.last_seen > horizon
        ]
        for tid in stale:
            self.tracks.pop(tid, None)
        for key in [k for k in self._last_fired if k[0] in stale]:
            self._last_fired.pop(key, None)

    # -- individual rules ---------------------------------------------------
    def _evaluate(
        self, sample: PoseSample, metrics: dict,
        timestamp: Optional[str], fence_y: Optional[float],
    ) -> List[dict]:
        thresholds = self.thresholds
        out: List[dict] = []
        fired: Set[str] = set()
        times = metrics["times"]
        # A proper x1y1x2y2 person box, so the evidence crop in the telemetry
        # packet frames the person instead of the whole frame.
        bbox = (
            sample.foot_point[0] - sample.width / 2.0,
            sample.foot_point[1] - sample.height,
            sample.foot_point[0] + sample.width / 2.0,
            sample.foot_point[1],
        )

        def add(behavior, confidence, detail, duration=0.0):
            alert = self._emit(
                behavior, sample.track_id, confidence, detail,
                timestamp, sample.ts, bbox, duration,
            )
            if alert:
                out.append(alert)
                fired.add(behavior)

        # --- FENCE_CLIMB ---------------------------------------------------
        if fence_y is not None and metrics["min_wrist_y"] is not None:
            clearance = (metrics["shoulder_y"] - metrics["min_wrist_y"]) / metrics["height_px"]
            near_fence = abs(metrics["min_wrist_y"] - float(fence_y)) <= (
                thresholds.climb_fence_tolerance_ratio * metrics["height_px"]
            )
            rising = metrics["hip_rise_ratio"] <= -thresholds.climb_rise_ratio
            hands_high = [
                value >= thresholds.climb_hand_clearance_ratio
                for value in metrics["clearances"]
            ]
            if near_fence and clearance >= thresholds.climb_hand_clearance_ratio:
                ok, span = _sustained(
                    hands_high, times, thresholds.climb_sustain_s
                )
                if ok and rising:
                    add(
                        "FENCE_CLIMB", 0.85,
                        f"Hands {clearance:.2f}x body height above shoulder, hips rising "
                        f"onto tripwire (dwell {span:.1f}s)",
                        duration=span,
                    )

        # --- FALL_ALERT ----------------------------------------------------
        # The fall signature on the CURRENT frame, before the confirmation delay
        # is applied. It is used to arbitrate against the crawl rule below.
        wide = metrics["aspect"] >= thresholds.fall_aspect_ratio
        dropped = (
            metrics["hip_drop_ratio"] >= thresholds.fall_drop_ratio
            or metrics["max_drop"] >= thresholds.fall_drop_ratio
        )
        still = metrics["speed_hps"] <= thresholds.fall_max_speed_ratio
        fall_pending = (
            metrics["torso_angle_deg"] >= thresholds.fall_torso_angle_deg
            and wide and dropped and still
        )
        if metrics["torso_angle_deg"] >= thresholds.fall_torso_angle_deg:
            horizontal = [
                angle >= thresholds.fall_torso_angle_deg for angle in metrics["torso_angles"]
            ]
            ok, span = _sustained(horizontal, times, thresholds.fall_sustain_s)
            if ok and fall_pending:
                add(
                    "FALL_ALERT", 0.80,
                    f"Person down: torso {metrics['torso_angle_deg']:.0f} deg from "
                    f"vertical, hip drop {metrics['hip_drop_ratio']:.2f}x height, "
                    f"motionless for {span:.1f}s",
                    duration=span,
                )

        # --- CROUCH_INTRUSION ---------------------------------------------
        # A crawl is horizontal AND low (a person bent at the waist is steep but
        # still tall in frame, and must not alarm); a body already flat on the
        # ground is caught by the box being wider than it is tall. Suppressed
        # when FALL already fired: one physical event, one alert.
        low_flags = [
            (
                angle >= thresholds.crouch_torso_angle_deg
                and aspect >= thresholds.crouch_aspect_ratio
            )
            or aspect >= thresholds.crouch_flat_aspect_ratio
            for angle, aspect in zip(metrics["torso_angles"], metrics["aspects"])
        ]
        ok, span = _sustained(low_flags, times, thresholds.crouch_sustain_s)
        # `fall_pending` (not merely "a fall already fired") is the arbitration:
        # the crawl rule confirms in 0.7 s while a fall needs 1.2 s, so without
        # this the same collapse would always be reported as a crawl first.
        if ok and not fall_pending and "FALL_ALERT" not in fired:
            add(
                "CROUCH_INTRUSION", 0.70,
                f"Low crawl posture: torso {metrics['torso_angle_deg']:.0f} deg, "
                f"box aspect {metrics['aspect']:.2f} over {span:.1f}s",
                duration=span,
            )

        # --- LOITERING -----------------------------------------------------
        if (
            metrics["dwell_s"] >= thresholds.loiter_min_dwell_s
            and metrics["spread_ratio"] <= thresholds.loiter_radius_ratio
            and metrics["mean_speed_hps"] <= thresholds.loiter_max_speed_ratio
        ):
            add(
                "LOITERING", 0.65,
                f"Stationary {metrics['dwell_s']:.0f}s within "
                f"{metrics['spread_ratio']:.2f}x height (movement "
                f"{metrics['mean_speed_hps']:.2f} heights/s)",
                duration=metrics["dwell_s"],
            )

        # --- RUNNING -------------------------------------------------------
        speed_flags = [
            speed >= thresholds.run_speed_ratio for speed in metrics["speeds"]
        ]
        ok, span = _sustained(speed_flags, metrics["speed_times"], thresholds.run_sustain_s)
        if ok:
            add(
                "RUNNING", 0.70,
                f"Running at {metrics['speed_hps']:.2f} body-heights/s "
                f"(threshold {thresholds.run_speed_ratio:.2f}) for {span:.1f}s",
                duration=span,
            )

        # --- ARM_RAISED_SIGNAL --------------------------------------------
        ok, span = _sustained(
            metrics["arms_flags"], times, thresholds.signal_arm_sustain_s
        )
        if (
            ok
            and metrics["mean_speed_hps"] <= thresholds.signal_max_speed_ratio
            and "FENCE_CLIMB" not in fired
            and not self._recently_fired(sample.track_id, ("FENCE_CLIMB",), sample.ts)
        ):
            add(
                "ARM_RAISED_SIGNAL", 0.55,
                f"Arm held above head for {span:.1f}s while stationary",
                duration=span,
            )

        return out

    # -- group rule ---------------------------------------------------------
    def _evaluate_group(
        self, samples: Sequence[PoseSample], timestamp: Optional[str]
    ) -> List[dict]:
        """
        Flags a converging cluster of people.

        Group geometry cannot be derived from a single track, so it is evaluated
        across tracks for the current frame and attributed to the lowest track ID
        in the group (a deterministic choice, which keeps the alert stable while
        the individuals move around inside the cluster).
        """
        if "GROUP_CONVERGENCE" not in self.enabled:
            self.behaviors_disabled += 1
            return []
        thresholds = self.thresholds
        if len(samples) < thresholds.group_min_persons:
            return []

        heights = [s.height for s in samples]
        mean_height = _median(heights)
        if mean_height <= 0:
            return []
        limit = thresholds.group_radius_ratio * mean_height

        # Union-find over "within limit" pairs: a chain of people walking in
        # single file is one group, which is what a smuggler escort looks like.
        parent = list(range(len(samples)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, first in enumerate(samples):
            for j in range(i + 1, len(samples)):
                second = samples[j]
                distance = math.hypot(
                    first.foot_point[0] - second.foot_point[0],
                    first.foot_point[1] - second.foot_point[1],
                )
                if distance <= limit:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[b] = a

        groups: Dict[int, List[int]] = {}
        for index in range(len(samples)):
            groups.setdefault(find(index), []).append(samples[index].track_id)

        now = samples[-1].ts
        alerts = []
        for members in groups.values():
            if len(members) < thresholds.group_min_persons:
                continue
            if (now - self._group_last_fired) < thresholds.cooldown_s:
                self.suppressed_by_cooldown += 1
                continue
            self._group_last_fired = now
            leader = min(members)
            self.counts["GROUP_CONVERGENCE"] = self.counts.get("GROUP_CONVERGENCE", 0) + 1
            self.events_emitted += 1
            meta = BEHAVIOR_CATALOGUE["GROUP_CONVERGENCE"]
            alerts.append({
                "event_type": "GROUP_CONVERGENCE",
                "behavior": "GROUP_CONVERGENCE",
                "behavior_label": meta["label"],
                "severity": meta["severity"],
                "track_id": leader,
                "category": "human",
                "class_name": "person",
                "confidence": 0.6,
                "status": meta["severity"],
                "details": (
                    f"{len(members)} persons converged within "
                    f"{thresholds.group_radius_ratio:.1f} body-lengths "
                    f"(tracks {sorted(members)})"
                ),
                "zone": "Behaviour Analytics",
                "direction": "",
                "identity": "",
                "location": "",
                "timestamp": timestamp,
                "duration_s": 0.0,
                "provenance": "POSE_HEURISTIC",
                "bbox": None,
            })
        return alerts

    # -- reporting ----------------------------------------------------------
    def reset(self) -> None:
        self.tracks.clear()
        self._last_fired.clear()
        self._group_last_fired = 0.0

    def track_metrics(self, limit: int = 12) -> List[dict]:
        """Per-track metric snapshot for the dashboard table."""
        rows = []
        for track_id, state in list(self.tracks.items()):
            metrics = self._metrics(state)
            if not metrics:
                continue
            rows.append({
                "track_id": track_id,
                "dwell_s": round(metrics["dwell_s"], 1),
                "speed_hps": round(metrics["speed_hps"], 2),
                "spread_ratio": round(metrics["spread_ratio"], 2),
                "torso_angle_deg": round(metrics["torso_angle_deg"], 1),
                "aspect": round(metrics["aspect"], 2),
                "height_px": round(metrics["height_px"], 0),
                "last_seen_s": round(max(0.0, self._clock() - state.last_seen), 1),
            })
        rows.sort(key=lambda r: r["track_id"])
        return rows[:limit]

    def stats(self) -> dict:
        return {
            "samples_ingested": self.samples_ingested,
            "events_emitted": self.events_emitted,
            "active_tracks": len(self.tracks),
            "suppressed_by_cooldown": self.suppressed_by_cooldown,
            "behaviors_disabled": self.behaviors_disabled,
            "counts": dict(self.counts),
            "enabled": sorted(self.enabled),
            "last_error": self.last_error,
        }


# ---------------------------------------------------------------------------
# Pose estimators (pluggable backends)
# ---------------------------------------------------------------------------
class NullPoseEstimator:
    """Used when no pose model is installed. Reports why, and detects nothing."""

    name = "unavailable"

    def __init__(self, reason: str = "no pose model installed"):
        self.reason = reason

    def estimate(self, frame, conf_threshold: float = 0.25) -> List[Tuple[List, List]]:
        return []


class UltralyticsPoseEstimator:
    """
    YOLOv8-Pose / YOLO11-Pose through ultralytics.

    The model is loaded lazily so importing this module never costs a model load
    and never fails on a box without the weights.
    """

    name = "ultralytics-pose"

    def __init__(self, model_path: str = "yolov8n-pose.pt", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.load_error = ""

    def load(self) -> bool:
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            return True
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    @property
    def available(self) -> bool:
        return self.model is not None

    def estimate(self, frame, conf_threshold: float = 0.25) -> List[Tuple[List, List]]:
        if self.model is None:
            return []
        results = self.model.predict(
            frame, conf=float(conf_threshold), device=self.device, verbose=False
        )
        people: List[Tuple[List, List]] = []
        for result in results or []:
            keypoints = getattr(result, "keypoints", None)
            boxes = getattr(result, "boxes", None)
            if keypoints is None or boxes is None:
                continue
            try:
                data = keypoints.data.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
            except Exception:
                continue
            for index in range(len(data)):
                points = [[float(v) for v in row] for row in data[index]]
                box = [float(v) for v in xyxy[index]] if index < len(xyxy) else None
                conf = float(confs[index]) if index < len(confs) else 1.0
                people.append((points, [box, conf]))
        return people


class InjectedPoseEstimator:
    """
    Test/demo backend: a callable returns the people for a frame.

    This is what lets the behaviour rules be verified for real without a pose
    model or a GPU on the machine running the tests.
    """

    name = "injected"

    def __init__(self, people_fn: Callable):
        self._people_fn = people_fn

    def estimate(self, frame, conf_threshold: float = 0.25) -> List[Tuple[List, List]]:
        return list(self._people_fn(frame))


def build_pose_estimator(model_path: str = "yolov8n-pose.pt", device: str = "cpu"):
    """Returns (estimator, note) with an honest explanation of what loaded."""
    estimator = UltralyticsPoseEstimator(model_path, device)
    if estimator.load():
        return estimator, f"Pose model loaded: {model_path}"
    return (
        NullPoseEstimator(estimator.load_error or f"could not load {model_path}"),
        f"Pose backend unavailable ({estimator.load_error}) - behaviour analytics OFF.",
    )


# ---------------------------------------------------------------------------
# Track association
# ---------------------------------------------------------------------------
def match_samples_to_tracks(
    people: Sequence[Tuple[List, List]],
    tracks: Dict[int, dict],
    ts: float,
    frame_idx: int = 0,
    min_keypoint_conf: float = 0.30,
    max_distance_ratio: float = 0.75,
) -> Tuple[List[PoseSample], int]:
    """
    Binds pose detections to tracker IDs.

    Without this, a behaviour alert could not be correlated with the fence or
    identity event for the same person - the operator would see "someone is
    crawling" with no idea *who*. Matching is nearest-centroid, normalised by body
    height so it still works at long range.

    Returns (samples, unmatched) - unmatched people are counted, not hidden.
    """
    if not people:
        return ([], 0)

    person_tracks = [
        (tid, data) for tid, data in (tracks or {}).items()
        if str(data.get("category", "human")) != "vehicle"
    ]

    samples: List[PoseSample] = []
    unmatched = 0
    for points, extra in people:
        box = None
        conf = 1.0
        if isinstance(extra, (list, tuple)) and extra:
            box = extra[0]
            if len(extra) > 1:
                conf = float(extra[1])

        probe = PoseSample(
            0, points, bbox=box, detection_conf=conf, ts=ts,
            frame_idx=frame_idx, min_keypoint_conf=min_keypoint_conf,
        )
        if not probe.valid:
            unmatched += 1
            continue

        # Compare like with like: the tracker reports a box CENTROID, so the pose
        # sample is reduced to its box centre too. (Matching a centroid against a
        # foot point would skew vertical distance by half a body height and
        # mis-associate anyone at a different range.)
        box_centre = (probe.foot_point[0], probe.foot_point[1] - probe.height / 2.0)
        best_id, best_distance = None, None
        for track_id, data in person_tracks:
            centroid = data.get("centroid") or (0.0, 0.0)
            distance = math.hypot(
                centroid[0] - box_centre[0], centroid[1] - box_centre[1]
            )
            if best_distance is None or distance < best_distance:
                best_id, best_distance = track_id, distance

        if best_id is None or best_distance is None or best_distance > (
            max_distance_ratio * probe.height
        ):
            unmatched += 1
            continue

        probe.track_id = int(best_id)
        samples.append(probe)

    return (samples, unmatched)


# ---------------------------------------------------------------------------
# Engine (what the pipeline and the dashboard talk to)
# ---------------------------------------------------------------------------
class PoseEngine:
    """
    Pose estimation plus behaviour analytics, with an honest availability state.

    `mode` is one of:
      ACTIVE       - a real pose backend loaded; heuristics are running
      UNAVAILABLE  - no backend; NOTHING is inferred (never fakes a behaviour)
    """

    def __init__(
        self,
        estimator=None,
        thresholds: Optional[BehaviorThresholds] = None,
        enabled_behaviors: Optional[Iterable[str]] = None,
        note: str = "",
        clock: Callable[[], float] = time.time,
    ):
        self.estimator = estimator if estimator is not None else NullPoseEstimator()
        self.note = note
        self.analyser = BehaviorAnalyser(thresholds=thresholds, enabled=enabled_behaviors,
                                         clock=clock)
        self._clock = clock
        self.people_detected = 0
        self.unmatched_people = 0
        self.frames = 0
        self.last_error = ""
        self.last_latency_ms = 0.0

    # -- availability -------------------------------------------------------
    @property
    def available(self) -> bool:
        return not isinstance(self.estimator, NullPoseEstimator)

    @property
    def mode(self) -> str:
        return "ACTIVE" if self.available else "UNAVAILABLE"

    def state(self) -> dict:
        return {
            "mode": self.mode,
            "available": self.available,
            "backend": getattr(self.estimator, "name", "unknown"),
            "estimator": type(self.estimator).__name__,
            "note": self.note or (
                "Behaviour analytics running on geometric pose heuristics."
                if self.available
                else "No pose backend: behaviour analytics offline."
            ),
            "frames_analysed": self.frames,
            "people_detected": self.people_detected,
            "unmatched_people": self.unmatched_people,
            "last_latency_ms": round(self.last_latency_ms, 1),
            "last_error": self.last_error,
            "behaviors": [
                {
                    "behavior": name,
                    "label": BEHAVIOR_CATALOGUE[name]["label"],
                    "severity": BEHAVIOR_CATALOGUE[name]["severity"],
                    "description": BEHAVIOR_CATALOGUE[name]["description"],
                    "enabled": name in self.analyser.enabled,
                    "count": self.analyser.counts.get(name, 0),
                }
                for name in BEHAVIOR_ORDER
            ],
            "analyser": self.analyser.stats(),
        }

    # -- runtime controls ---------------------------------------------------
    def configure(self, enabled_behaviors=None, **threshold_overrides) -> None:
        """Applies operator changes live, without rebuilding the engine."""
        if enabled_behaviors is not None:
            self.analyser.enabled = set(enabled_behaviors)
        for key, value in threshold_overrides.items():
            if value is not None and hasattr(self.analyser.thresholds, key):
                setattr(self.analyser.thresholds, key, value)

    # -- processing ---------------------------------------------------------
    def process(
        self,
        frame,
        tracks: Dict[int, dict],
        timestamp: Optional[str] = None,
        frame_idx: int = 0,
        fence_y: Optional[float] = None,
        conf_threshold: float = 0.25,
    ) -> Tuple[List[dict], List[PoseSample]]:
        """
        Returns (behaviour_alerts, matched_samples).

        Samples are handed back so the caller can draw the skeleton overlay; a
        behaviour alert the operator cannot see the evidence for is not usable.
        """
        if not self.available:
            return ([], [])

        started = self._clock()
        try:
            people = self.estimator.estimate(frame, conf_threshold=conf_threshold)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return ([], [])

        self.frames += 1
        self.people_detected += len(people or [])

        samples, unmatched = match_samples_to_tracks(
            people, tracks, ts=self._clock(), frame_idx=frame_idx,
            min_keypoint_conf=self.analyser.thresholds.min_keypoint_conf,
        )
        self.unmatched_people += unmatched
        alerts = self.analyser.update(samples, timestamp=timestamp, fence_y=fence_y)
        self.last_latency_ms = (self._clock() - started) * 1000.0
        return (alerts, samples)

    # -- overlay ------------------------------------------------------------
    def draw_overlay(self, frame, samples: Sequence[PoseSample], alerts: Sequence[dict] = ()):
        """
        Draws skeleton + behaviour tag. Degrades safely when OpenCV is absent.
        """
        if frame is None or not getattr(frame, "shape", None):
            return frame
        try:
            import cv2
        except Exception:
            return frame
        try:
            for sample in samples or []:
                for start, end in SKELETON_EDGES:
                    if start >= len(sample.keypoints) or end >= len(sample.keypoints):
                        continue
                    ax, ay, ac = sample.keypoints[start]
                    bx, by, bc = sample.keypoints[end]
                    if ac < self.analyser.thresholds.min_keypoint_conf:
                        continue
                    if bc < self.analyser.thresholds.min_keypoint_conf:
                        continue
                    cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)),
                             (0, 214, 255), 2)
            for alert in alerts or []:
                track_id = alert.get("track_id")
                sample = next(
                    (s for s in (samples or []) if s.track_id == track_id), None
                )
                if sample is None:
                    continue
                colour = (0, 0, 255) if alert.get("severity") == "CRITICAL" else (0, 165, 255)
                x, y = int(sample.foot_point[0]), int(sample.foot_point[1])
                cv2.putText(
                    frame, str(alert.get("behavior", "")), (max(0, x - 90), max(20, y + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2,
                )
        except Exception as exc:
            self.last_error = f"overlay: {type(exc).__name__}: {exc}"
        return frame


def build_pose_engine(
    model_path: str = "models/yolov8n-pose.pt",
    device: str = "cpu",
    thresholds: Optional[BehaviorThresholds] = None,
    enabled_behaviors: Optional[Iterable[str]] = None,
) -> Tuple[PoseEngine, str]:
    """Factory mirroring build_frs: returns (engine, note) with what really loaded."""
    estimator, note = build_pose_estimator(model_path, device)
    engine = PoseEngine(
        estimator=estimator, thresholds=thresholds,
        enabled_behaviors=enabled_behaviors, note=note,
    )
    return (engine, note)


# ---------------------------------------------------------------------------
# Operator sensitivity presets
# ---------------------------------------------------------------------------
# A field commander cannot be asked to reason about body-heights per second, but
# they can absolutely say "stop crying wolf" or "I want everything". These are the
# same thresholds, expressed as the only three choices an operator should need.
SENSITIVITY_PRESETS: Dict[str, BehaviorThresholds] = {
    "CONSERVATIVE": BehaviorThresholds(
        loiter_min_dwell_s=12.0,
        loiter_radius_ratio=0.30,
        loiter_max_speed_ratio=0.20,
        run_speed_ratio=2.00,
        run_sustain_s=1.40,
        crouch_sustain_s=1.00,
        fall_sustain_s=1.60,
        climb_sustain_s=0.80,
        signal_arm_sustain_s=3.50,
        group_sustain_s=4.00,
    ),
    "BALANCED": BehaviorThresholds(),
    "AGGRESSIVE": BehaviorThresholds(
        loiter_min_dwell_s=5.0,
        loiter_radius_ratio=0.45,
        loiter_max_speed_ratio=0.35,
        run_speed_ratio=1.30,
        run_sustain_s=0.70,
        crouch_sustain_s=0.50,
        fall_sustain_s=0.90,
        climb_sustain_s=0.40,
        signal_arm_sustain_s=1.80,
        group_sustain_s=2.00,
    ),
}


def preset_thresholds(name: str) -> BehaviorThresholds:
    """
    Returns a FRESH thresholds object for a named preset.

    A fresh copy matters: the engine mutates thresholds in place when an operator
    retunes live, so handing out the shared preset object would silently rewrite
    the preset for every subsequent session.
    """
    import copy

    key = str(name).split()[0].strip().upper()
    return copy.deepcopy(SENSITIVITY_PRESETS.get(key, SENSITIVITY_PRESETS["BALANCED"]))


def behaviour_severity() -> Dict[str, str]:
    """Wire priority class per behaviour, for the telemetry severity table."""
    return {name: meta["severity"] for name, meta in BEHAVIOR_CATALOGUE.items()}
