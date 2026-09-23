"""
IBVAP PHASE 8 VERIFICATION: POSE / SUSPICIOUS-BEHAVIOUR ANALYTICS
=================================================================

Verifies the behaviour layer by synthesising anatomically-proportioned COCO-17
skeletons for each scenario and feeding them through the real analyser. No pose
model is required, and none of the rules are mocked away: the same geometry the
field system uses is what is being measured here.

Thresholds are all expressed in body-relative units, so the synthesised skeletons
are built at a fixed pixel height and every time series runs at 10 fps.

  1. Torso geometry - inclinations, scale independence, validity gates.
  2. FALSE-POSITIVE GUARD - ordinary walking must raise nothing at all.
  3. LOITERING - dwell detection, and the cooldown that stops alert spam.
  4. RUNNING - sustained limb-normalised speed.
  5. CROUCH_INTRUSION - crawl detection, and a person merely bending over is NOT
     reported (the single most important precision check in this file).
  6. FALL_ALERT - drop + horizontal + stillness, and mutual exclusion with crawl.
  7. FENCE_CLIMB - hands on the wire with hips rising, and the fence-proximity
     requirement proven to be load-bearing.
  8. ARM_RAISED_SIGNAL - sustained raised arm while stationary.
  9. GROUP_CONVERGENCE - union-find over body-length proximity.
 10. Signal quality gates - tiny figures and unreadable keypoints are ignored.
 11. Track association - pose bound to tracker IDs, vehicles excluded.
 12. HONESTY GUARD - with no pose backend nothing is inferred, ever.
 13. Engine controls, overlay safety, and wire-severity consistency.

Run:  python test_phase8.py
"""

import math
import os
import shutil
import sys
import tempfile

import numpy as np

from modules import pose
from modules.pose import (
    BEHAVIOR_CATALOGUE,
    BEHAVIOR_ORDER,
    COCO_KEYPOINTS,
    BehaviorThresholds,
    BehaviorAnalyser,
    InjectedPoseEstimator,
    NullPoseEstimator,
    PoseEngine,
    SKELETON_EDGES,
    build_pose_engine,
    make_sample,
    match_samples_to_tracks,
    preset_thresholds,
)
from modules.logger import BEHAVIOR_EVENT_TYPES, EventLogger
from modules.nodes import get_camera
from modules.pipeline import VisionModels, build_analyser, heartbeat_event, route_event
from modules.telemetry import LoopbackTransport, TelemetryPublisher, severity_for

PASS = "  [PASS]"
FAIL = "  [FAIL]"
failures = []

FPS = 10.0
DT = 1.0 / FPS
BODY_H = 200.0          # standing height in pixels
FRAME_H = 720


def check(condition, message):
    print(f"{PASS if condition else FAIL} {message}")
    if not condition:
        failures.append(message)
    return bool(condition)


def header(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


# ---------------------------------------------------------------------------
# Synthetic skeleton construction
# ---------------------------------------------------------------------------
# Anatomically-proportioned layout, in fractions of standing body height, with
# y measured downward from the crown. Enough to exercise real torso geometry.
STAND_LAYOUT = {
    "nose": (0.00, 0.07), "left_eye": (-0.025, 0.05), "right_eye": (0.025, 0.05),
    "left_ear": (-0.050, 0.06), "right_ear": (0.050, 0.06),
    "left_shoulder": (-0.110, 0.18), "right_shoulder": (0.110, 0.18),
    "left_elbow": (-0.130, 0.32), "right_elbow": (0.130, 0.32),
    "left_wrist": (-0.130, 0.46), "right_wrist": (0.130, 0.46),
    "left_hip": (-0.070, 0.50), "right_hip": (0.070, 0.50),
    "left_knee": (-0.060, 0.72), "right_knee": (0.060, 0.72),
    "left_ankle": (-0.060, 0.95), "right_ankle": (0.060, 0.95),
}
UPPER_BODY = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
)


def _rotate(points, pivot, degrees):
    theta = math.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    px, py = pivot
    rotated = {}
    for name, (x, y) in points.items():
        rx, ry = x - px, y - py
        rotated[name] = (
            px + rx * cos_t + ry * sin_t,
            py - rx * sin_t + ry * cos_t,
        )
    return rotated


def skeleton(
    center_x=300.0,
    ground_y=600.0,
    height=BODY_H,
    tilt_deg=0.0,
    arm="down",
    bend_deg=0.0,
    conf=0.92,
):
    """
    Builds one COCO-17 person.

    tilt_deg  rotates the whole body about the ground contact (a fall / a crawl).
    bend_deg  flexes only the upper body at the hips (someone stooping), which is
              anatomically a completely different signal from a crawl.
    arm       'down' | 'up' (both wrists overhead) | 'one_up' (signalling).
    """
    points = {name: (dx * height, dy * height) for name, (dx, dy) in STAND_LAYOUT.items()}

    if arm in ("up", "one_up"):
        # Arms extended overhead put the wrist ABOVE the crown (negative y in this
        # layout), which is the real geometry of a person gripping a fence top or
        # signalling. Wrists merely level with the head would understate the
        # shoulder clearance the climb rule measures.
        sides = ("left", "right") if arm == "up" else ("left",)
        for side in sides:
            sign = -1.0 if side == "left" else 1.0
            points[f"{side}_elbow"] = (sign * 0.085 * height, -0.02 * height)
            points[f"{side}_wrist"] = (sign * 0.075 * height, -0.16 * height)

    if bend_deg:
        pivot = (0.0, 0.50 * height)
        upper = {name: points[name] for name in UPPER_BODY}
        points.update(_rotate(upper, pivot, bend_deg))

    if tilt_deg:
        points = _rotate(points, (0.0, 0.95 * height), tilt_deg)

    lowest = max(y for _, y in points.values())
    mean_x = sum(x for x, _ in points.values()) / len(points)
    shift_y = ground_y - lowest
    shift_x = center_x - mean_x

    keypoints = []
    for name in COCO_KEYPOINTS:
        x, y = points[name]
        keypoints.append([x + shift_x, y + shift_y, conf])

    xs = [k[0] for k in keypoints]
    ys = [k[1] for k in keypoints]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return keypoints, bbox


def sample_at(track_id, t, **kwargs):
    keypoints, bbox = skeleton(**kwargs)
    return make_sample(track_id, keypoints, bbox=bbox, ts=t, frame_idx=int(t * FPS))


def sequence(track_id, duration_s, start_t=0.0, x_at=None, **pose_kwargs):
    """
    10 fps series of one person. x_at(t) gives the horizontal path in PIXELS PER
    SECOND, so paths stay realistic as body size changes: a person walking at
    1.4 m/s is ~0.8 body-heights/second, i.e. 140 px/s on a 180 px figure.
    """
    out = []
    steps = int(duration_s * FPS)
    for index in range(steps):
        t = start_t + index * DT
        kwargs = dict(pose_kwargs)
        if callable(x_at):
            kwargs["center_x"] = x_at(t - start_t)
        out.append(sample_at(track_id, t, **kwargs))
    return out


def analyse(samples, enabled=None, fence_y=None, thresholds=None, timestamp="04:12:00"):
    analyser = BehaviorAnalyser(thresholds=thresholds, enabled=enabled)
    alerts = analyser.update(samples, timestamp=timestamp, fence_y=fence_y)
    return analyser, alerts


def behaviors_of(alerts):
    return sorted({a["behavior"] for a in alerts})


def main():
    # ------------------------------------------------------------------
    header("1. TORSO GEOMETRY AND SCALE INDEPENDENCE")
    # ------------------------------------------------------------------
    upright = sample_at(1, 0.0, tilt_deg=0.0)
    check(upright.valid, "an upright, fully-visible person is a valid sample")
    check(abs(upright.torso_angle_deg) < 3.0,
          f"an upright torso reads ~0 deg from vertical ({upright.torso_angle_deg:.1f})")

    for angle in (45.0, 70.0, 85.0):
        tilted = sample_at(1, 0.0, tilt_deg=angle)
        check(abs(tilted.torso_angle_deg - angle) < 6.0,
              f"a body tilted {angle:.0f} deg reads back as "
              f"{tilted.torso_angle_deg:.1f} deg")

    short = sample_at(1, 0.0, height=60.0)
    check(abs(short.torso_angle_deg) < 3.0,
          "torso geometry is unchanged at 60 px body height (scale independent)")
    check(abs(upright.aspect - short.aspect) < 0.05,
          "aspect ratio is identical at 3x the body height")

    # A sample whose shoulders are unusable must be rejected, not guessed at.
    blind = list(skeleton()[0])
    for name in ("left_shoulder", "right_shoulder"):
        blind[COCO_KEYPOINTS.index(name)][2] = 0.05
    unreadable = make_sample(2, blind, ts=0.0)
    check(not unreadable.valid,
          "a person whose shoulders are invisible is INVALID - geometry is not invented")

    raised = sample_at(3, 0.0, arm="up")
    check(raised.arms_over_head, "both wrists overhead registers as arms-over-head")
    check(not upright.arms_over_head, "a relaxed standing pose does not")
    stooped = sample_at(4, 0.0, bend_deg=70.0)
    check(stooped.torso_angle_deg > 60.0,
          "stooping at the waist still yields a large torso angle (hence needs the "
          "aspect test to be told apart from crawling)")

    # ------------------------------------------------------------------
    header("2. FALSE-POSITIVE GUARD: ORDINARY WALKING RAISES NOTHING")
    # ------------------------------------------------------------------
    walking = sequence(11, 12.0, x_at=lambda t: 260.0 + 140.0 * t)
    analyser, alerts = analyse(walking)
    check(alerts == [], f"12 s of normal walking produces no behaviour alerts ({behaviors_of(alerts)})")
    check(analyser.samples_ingested == len(walking), "every walking sample was ingested")
    check(analyser.counts["LOITERING"] == 0, "walking is correctly not loitering")
    check(analyser.counts["RUNNING"] == 0, "walking is correctly not running")

    # A genuine stroll (0.33 body-heights/s) is slower than walking but still
    # travelling: the dwell rule must not mistake it for loitering.
    strolling = sequence(12, 12.0, x_at=lambda t: 260.0 + 60.0 * t)
    _, stroll_alerts = analyse(strolling)
    check("LOITERING" not in behaviors_of(stroll_alerts),
          f"a slow but continuous stroll still does not trip the dwell rule "
          f"({behaviors_of(stroll_alerts)})")

    shuffling = sequence(13, 12.0, x_at=lambda t: 300.0 + 0.4 * t)
    _, shuffle_alerts = analyse(shuffling, enabled={"LOITERING"})
    check(behaviors_of(shuffle_alerts) == ["LOITERING"],
          "but someone who has genuinely stopped for 8 s within a body-length IS "
          "flagged - the rule is strict, not toothless")

    # ------------------------------------------------------------------
    header("3. LOITERING")
    # ------------------------------------------------------------------
    loitering = sequence(21, 10.0, x_at=lambda t: 300.0 + 1.5 * math.sin(t * 3.0))
    analyser, alerts = analyse(loitering, enabled={"LOITERING"})
    check(behaviors_of(alerts) == ["LOITERING"],
          f"a person lingering 10 s in one spot is flagged ({behaviors_of(alerts)})")
    check(len(alerts) == 1, "exactly one alert for one continuous dwell")
    check(alerts[0]["severity"] == "WARNING", "loitering is a WARNING, not a CRITICAL")
    check(alerts[0]["provenance"] == "POSE_HEURISTIC",
          "the alert is labelled as a geometric heuristic, not a classified behaviour")
    check(alerts[0]["duration_s"] >= 8.0,
          f"the alert reports the dwell duration ({alerts[0]['duration_s']}s)")
    check(alerts[0]["track_id"] == 21, "the alert is attributed to the right track")
    check(len(alerts[0]["bbox"]) == 4 and alerts[0]["bbox"][2] > alerts[0]["bbox"][0],
          "the alert carries a usable x1y1x2y2 evidence box, not a placeholder")

    # The same person keeps lingering: cooldown must absorb the repeat.
    extended = sequence(21, 10.0, start_t=10.0,
                        x_at=lambda t: 300.0 + 1.5 * math.sin((t + 10.0) * 3.0))
    _, repeats = analyse(extended, enabled={"LOITERING"}, thresholds=BehaviorThresholds())
    analyser.update(extended)
    check(analyser.suppressed_by_cooldown > 0,
          f"repeat detections inside the cooldown are suppressed "
          f"({analyser.suppressed_by_cooldown} suppressed)")
    check(analyser.counts["LOITERING"] < 10,
          "five minutes of loitering does not produce an alert per frame")

    # ------------------------------------------------------------------
    header("4. RUNNING")
    # ------------------------------------------------------------------
    running = sequence(31, 2.5, x_at=lambda t: 200.0 + 300.0 * t)
    analyser, alerts = analyse(running, enabled={"RUNNING"})
    check(behaviors_of(alerts) == ["RUNNING"],
          f"300 px/s (1.67 body-heights/s) is flagged as running ({behaviors_of(alerts)})")
    check("1.6" in alerts[0]["details"] or "body-heights/s" in alerts[0]["details"],
          "the alert explains the measured speed against the threshold")

    fast_brief = sequence(32, 0.5, x_at=lambda t: 200.0 + 400.0 * t)
    _, brief = analyse(fast_brief, enabled={"RUNNING"})
    check("RUNNING" not in behaviors_of(brief),
          "a half-second sprint does not fire - one noisy frame is not a behaviour")

    # ------------------------------------------------------------------
    header("5. CROUCH / CRAWL")
    # ------------------------------------------------------------------
    crawl_path = lambda t: 300.0 + 60.0 * t
    crawling = []
    crawling += sequence(41, 1.0, tilt_deg=0.0, x_at=lambda t: 300.0)
    crawling += sequence(41, 3.0, start_t=1.0, tilt_deg=72.0, x_at=crawl_path)
    analyser, alerts = analyse(crawling, enabled={"CROUCH_INTRUSION"})
    check(behaviors_of(alerts) == ["CROUCH_INTRUSION"],
          f"a low crawl is detected ({behaviors_of(alerts)})")
    check(alerts[0]["severity"] == "WARNING", "a crawl is a WARNING")

    # Precision: the same person merely stooping must NOT be reported.
    stooping = sequence(42, 3.0, bend_deg=70.0, x_at=lambda t: 300.0)
    _, stoop_alerts = analyse(stooping, enabled={"CROUCH_INTRUSION"})
    check(stoop_alerts == [],
          "a person bending over at the waist is NOT reported as a crawl "
          "(precision guard)")

    # Precision: someone standing next to the fence must not be a crawl either.
    _, stand_alerts = analyse(sequence(43, 3.0), enabled={"CROUCH_INTRUSION"})
    check(stand_alerts == [], "a standing person is not a crawl")

    # ------------------------------------------------------------------
    header("6. FALL_ALERT")
    # ------------------------------------------------------------------
    falling = []
    falling += sequence(51, 2.0, tilt_deg=0.0, x_at=lambda t: 300.0)
    falling += sequence(51, 2.5, start_t=2.0, tilt_deg=82.0, x_at=lambda t: 300.0)
    analyser, alerts = analyse(falling, enabled={"FALL_ALERT", "CROUCH_INTRUSION"})
    check("FALL_ALERT" in behaviors_of(alerts),
          f"a standing person collapsing to the ground raises FALL_ALERT ({behaviors_of(alerts)})")
    fall_alert = next(a for a in alerts if a["behavior"] == "FALL_ALERT")
    check(fall_alert["severity"] == "CRITICAL",
          "a person down is CRITICAL - a fallen jawan needs help now")
    check("Person down" in fall_alert["details"], "the alert explains why (torso + hip drop)")
    check("CROUCH_INTRUSION" not in behaviors_of(alerts),
          "one fall produces ONE alert - the crawl rule stays silent (no alert chain)")

    # Precision: a person who is simply lying down from the start, motionless, is
    # not a *fall* - there is no drop transition to observe.
    already_down = sequence(52, 4.0, tilt_deg=85.0, x_at=lambda t: 300.0)
    _, down_alerts = analyse(already_down, enabled={"FALL_ALERT"})
    check(not [a for a in down_alerts if a["behavior"] == "FALL_ALERT"],
          "a body already prone with no drop transition is not reported as a fall")

    # Precision: a crawl keeps moving, so it must not be mistaken for a fall.
    _, crawl_for_fall = analyse(crawling, enabled={"FALL_ALERT"})
    check(not crawl_for_fall, "a moving crawl is not reported as a motionless fall")

    # ------------------------------------------------------------------
    header("7. FENCE_CLIMB")
    # ------------------------------------------------------------------
    # A climber GRIPS the wire: the hands stay on the fence line while the body
    # comes up to them. Raising hands and hips together would be a jump, and would
    # move the hands off the wire - a different physical event.
    climbing = []
    _unrisen, _bbox = skeleton(center_x=300.0, ground_y=600.0, arm="up")
    climb_fence_y = min(k[1] for k in _unrisen)
    for index in range(int(1.2 * FPS)):
        t = index * DT
        rise = 40.0 * t
        keypoints, bbox = skeleton(center_x=300.0, ground_y=600.0 - rise, arm="up")
        for side in ("left", "right"):
            keypoints[COCO_KEYPOINTS.index(f"{side}_wrist")][1] = climb_fence_y
        climbing.append(make_sample(61, keypoints, bbox=bbox, ts=t, frame_idx=index))

    analyser, alerts = analyse(
        climbing, enabled={"FENCE_CLIMB", "ARM_RAISED_SIGNAL"}, fence_y=climb_fence_y
    )
    check("FENCE_CLIMB" in behaviors_of(alerts),
          f"hands on the wire with hips rising is a climb ({behaviors_of(alerts)})")
    climb_alert = next((a for a in alerts if a["behavior"] == "FENCE_CLIMB"), None)
    check(climb_alert and climb_alert["severity"] == "CRITICAL",
          "a climb attempt is CRITICAL")
    check("ARM_RAISED_SIGNAL" not in behaviors_of(alerts),
          "the climb supersedes the weaker raised-arm rule (no overlapping alerts)")

    # The fence-proximity test is load-bearing: same motion, no fence reference.
    _, no_fence = analyse(
        climbing, enabled={"FENCE_CLIMB"}, fence_y=None
    )
    check(no_fence == [],
          "with no fence calibration the climb rule stays silent - it does not guess")

    # Same motion, but the hands are 2 body-heights below the wire: arms up in a
    # field is not a climb.
    _, far_fence = analyse(
        climbing, enabled={"FENCE_CLIMB"}, fence_y=climb_fence_y + 2.0 * BODY_H
    )
    check(far_fence == [], "arms raised far from the wire are not a climb")

    # Arms overhead but the body is not rising: a signal, not a climb.
    _, static_arms = analyse(
        sequence(62, 1.5, tilt_deg=0.0, arm="up"),
        enabled={"FENCE_CLIMB"}, fence_y=climb_fence_y,
    )
    check(static_arms == [], "hands overhead without the body rising is not a climb")

    # ------------------------------------------------------------------
    header("8. ARM_RAISED_SIGNAL")
    # ------------------------------------------------------------------
    _, signal_alerts = analyse(
        sequence(71, 5.0, arm="one_up", x_at=lambda t: 300.0),
        enabled={"ARM_RAISED_SIGNAL"},
    )
    check(behaviors_of(signal_alerts) == ["ARM_RAISED_SIGNAL"],
          f"a held raised arm is flagged as possible signalling ({behaviors_of(signal_alerts)})")
    check(signal_alerts[0]["severity"] == "NOTICE",
          "possible signalling is only a NOTICE - it must not compete with a breach")

    _, relaxed = analyse(sequence(72, 5.0, arm="down", x_at=lambda t: 300.0),
                         enabled={"ARM_RAISED_SIGNAL"})
    check(relaxed == [], "arms down never raises the signalling rule")

    # ------------------------------------------------------------------
    header("9. GROUP_CONVERGENCE")
    # ------------------------------------------------------------------
    def group_frame(t, count, spacing):
        return [
            sample_at(80 + i, t, center_x=300.0 + i * spacing, ground_y=600.0)
            for i in range(count)
        ]

    tight = []
    for index in range(int(4.0 * FPS)):
        tight += group_frame(index * DT, 3, 90.0)
    _, tight_alerts = analyse(tight, enabled={"GROUP_CONVERGENCE"})
    check(behaviors_of(tight_alerts) == ["GROUP_CONVERGENCE"],
          f"three people within a body-length are flagged ({behaviors_of(tight_alerts)})")
    check(tight_alerts[0]["track_id"] == 80,
          "the group alert is attributed to the lowest track ID (stable attribution)")
    check("tracks" in tight_alerts[0]["details"],
          "the alert names the members so the operator knows who to look at")

    spread = []
    for index in range(int(4.0 * FPS)):
        spread += group_frame(index * DT, 3, 500.0)
    _, spread_alerts = analyse(spread, enabled={"GROUP_CONVERGENCE"})
    check(not spread_alerts,
          "three people 2.8 body-lengths apart are not a convergence")

    _, pair_alerts = analyse(
        [s for index in range(int(4.0 * FPS))
         for s in group_frame(index * DT, 2, 90.0)],
        enabled={"GROUP_CONVERGENCE"},
    )
    check(not pair_alerts, "two people passing each other is not a group")

    # ------------------------------------------------------------------
    header("10. SIGNAL-QUALITY GATES")
    # ------------------------------------------------------------------
    tiny = sequence(91, 3.0, height=26.0, x_at=lambda t: 300.0)
    analyser, alerts = analyse(tiny)
    check(alerts == [], "a figure too small to read a torso from raises nothing")
    check(analyser.tracks == {},
          "no track state is even created for an unreadable figure (no memory leak)")

    noisy = []
    for index in range(30):
        keypoints, bbox = skeleton(center_x=300.0, ground_y=600.0)
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            keypoints[COCO_KEYPOINTS.index(name)][2] = 0.02
        noisy.append(make_sample(92, keypoints, bbox=bbox, ts=index * DT))
    analyser, alerts = analyse(noisy)
    check(alerts == [], "keypoints below the confidence floor are discarded silently")

    check(analyse([])[1] == [], "an empty frame is a no-op")
    analyser_empty, _ = analyse([])
    check(analyser_empty.track_metrics() == [], "track metrics are empty with no tracks")

    # ------------------------------------------------------------------
    header("11. TRACK ASSOCIATION")
    # ------------------------------------------------------------------
    tracks = {
        5: {"centroid": (300, 500), "category": "human", "bbox": (270, 400, 330, 600)},
        6: {"centroid": (900, 500), "category": "human", "bbox": (870, 400, 930, 600)},
        7: {"centroid": (300, 520), "category": "vehicle", "bbox": (200, 400, 400, 600)},
    }
    keypoints, bbox = skeleton(center_x=305.0, ground_y=600.0)
    people = [(keypoints, [list(bbox), 0.94])]
    samples, unmatched = match_samples_to_tracks(people, tracks, ts=1.0)
    check(len(samples) == 1 and unmatched == 0, "a person is bound to the nearest track")
    check(samples[0].track_id == 5,
          "association picks the true nearest person track, not the vehicle")
    check(samples[0].detection_conf == 0.94, "the detection confidence is carried through")

    far_people = [(skeleton(center_x=3000.0, ground_y=600.0)[0],
                   [list(skeleton(center_x=3000.0, ground_y=600.0)[1]), 0.9])]
    samples_far, unmatched_far = match_samples_to_tracks(far_people, tracks, ts=1.0)
    check(samples_far == [] and unmatched_far == 1,
          "a pose with no plausible track is COUNTED as unmatched, never mismatched")

    unreadable_people = [(list(skeleton()[0]), [list(skeleton()[1]), 0.9])]
    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        unreadable_people[0][0][COCO_KEYPOINTS.index(name)][2] = 0.01
    _, unmatched_bad = match_samples_to_tracks(unreadable_people, tracks, ts=1.0)
    check(unmatched_bad == 1, "an unreadable pose is unmatched rather than guessed")

    check(match_samples_to_tracks([], tracks, ts=1.0) == ([], 0),
          "associating nothing is a no-op")

    # ------------------------------------------------------------------
    header("12. HONESTY GUARD: NO BACKEND, NO BEHAVIOURS")
    # ------------------------------------------------------------------
    engine, note = build_pose_engine(model_path="models/definitely-absent-pose.pt")
    check(engine.mode == "UNAVAILABLE",
          "with no pose weights the engine reports UNAVAILABLE (ultralytics absent or "
          "weights missing)")
    check(not engine.available, "available is False, so nothing downstream will trust it")
    check("OFF" in note or "unavailable" in note.lower(),
          "the operator is told behaviour analytics is off, and why")

    frame = None
    events, samples = engine.process(frame, tracks={5: {"centroid": (300, 500)}})
    check(events == [] and samples == [],
          "an unavailable engine emits NOTHING - it never fabricates a behaviour")
    check(engine.state()["mode"] == "UNAVAILABLE", "state() reports the honest mode")
    check(engine.state()["analyser"]["events_emitted"] == 0,
          "no behaviour is ever counted while the backend is unavailable")

    null_events, _ = PoseEngine(estimator=NullPoseEstimator("no model")).process(
        None, tracks={5: {"centroid": (300, 500)}}
    )
    check(null_events == [], "a null estimator yields no events")

    # ------------------------------------------------------------------
    header("13. ENGINE, CONTROLS AND WIRE CONSISTENCY")
    # ------------------------------------------------------------------
    keypoints, bbox = skeleton(center_x=300.0, ground_y=600.0)
    people_fn = lambda frame: [(keypoints, [list(bbox), 0.95])]
    clock_holder = {"t": 0.0}
    live = PoseEngine(
        estimator=InjectedPoseEstimator(people_fn),
        enabled_behaviors={"LOITERING"},
        clock=lambda: clock_holder["t"],
    )
    check(live.mode == "ACTIVE", "an injected backend reports ACTIVE")

    emitted = []
    for index in range(int(11.0 * FPS)):
        clock_holder["t"] = index * DT
        events, samples = live.process(
            frame=None, tracks={5: {"centroid": (300, 500), "category": "human"}},
            timestamp="04:20:00", frame_idx=index,
        )
        emitted += events
    check(live.mode == "ACTIVE", "the engine stays available while running")
    check(live.people_detected >= 100, "every posed person is counted")
    check(live.analyser.counts["LOITERING"] >= 1,
          "behaviour events flow through the engine, not just the analyser")
    check(all(e["provenance"] == "POSE_HEURISTIC" for e in emitted),
          "every engine event carries its heuristic provenance")
    check(all("bbox" in e for e in emitted), "engine events carry an evidence box")
    check(live.state()["last_latency_ms"] >= 0.0, "the engine reports its own latency")

    live.configure(enabled_behaviors={"FALL_ALERT"}, loiter_min_dwell_s=3.0)
    check(live.analyser.enabled == {"FALL_ALERT"}, "enabled behaviours are reconfigurable live")
    check(live.analyser.thresholds.loiter_min_dwell_s == 3.0,
          "thresholds are reconfigurable live for field calibration")
    live.configure(bogus_threshold=99)
    check(not hasattr(live.analyser.thresholds, "bogus_threshold"),
          "an unknown threshold override is ignored, not injected")

    safe_overlay = live.draw_overlay(None, [], [])
    check(safe_overlay is None, "overlay drawing degrades safely with no frame/OpenCV")

    # The wire severity table must agree with the catalogue, or the map and the
    # packet would classify the same behaviour differently.
    mismatches = [
        name for name, meta in BEHAVIOR_CATALOGUE.items()
        if severity_for(name) != meta["severity"]
    ]
    check(not mismatches,
          f"every behaviour's severity matches the telemetry mapping (mismatches: {mismatches})")
    check(set(BEHAVIOR_EVENT_TYPES) == set(BEHAVIOR_CATALOGUE.keys()),
          "the logger's behaviour event types match the catalogue exactly")
    check(pose.behaviour_severity() == {
        name: meta["severity"] for name, meta in BEHAVIOR_CATALOGUE.items()
    }, "behaviour_severity() mirrors the catalogue")
    check(all(name in BEHAVIOR_CATALOGUE for name in BEHAVIOR_ORDER),
          "the dashboard ordering covers the whole catalogue")
    check(len(BEHAVIOR_ORDER) == len(set(BEHAVIOR_ORDER)),
          "the dashboard ordering contains no duplicates")
    check(all(edge[0] < len(COCO_KEYPOINTS) and edge[1] < len(COCO_KEYPOINTS)
              for edge in SKELETON_EDGES),
          "every skeleton edge indexes a real keypoint")

    # ------------------------------------------------------------------
    header("14. STATE RESET AND REPORTING CONTRACT")
    # ------------------------------------------------------------------
    live.analyser.reset()
    check(live.analyser.tracks == {}, "reset() clears track history")
    check(live.analyser.track_metrics() == [], "track metrics are empty after reset")
    check(live.analyser.stats()["samples_ingested"] > 0,
          "lifetime statistics survive a reset (an audit trail must not lose history)")

    state = live.state()
    for key in ("mode", "available", "backend", "note", "frames_analysed",
                "people_detected", "unmatched_people", "behaviors", "analyser"):
        check(key in state, f"state() exposes '{key}' for the dashboard")
    check(len(state["behaviors"]) == len(BEHAVIOR_CATALOGUE),
          "state() describes every behaviour for the UI")
    check(all({"behavior", "label", "severity", "description", "enabled", "count"} <= set(b)
              for b in state["behaviors"]),
          "each behaviour entry is render-ready")
    print(f"    engine: {state['mode']} | behaviours enabled: "
          f"{sum(1 for b in state['behaviors'] if b['enabled'])}/{len(state['behaviors'])}")

    # ------------------------------------------------------------------
    header("15. SENSITIVITY PRESETS")
    # ------------------------------------------------------------------
    conservative = preset_thresholds("Conservative (fewest false alarms)")
    balanced = preset_thresholds("Balanced")
    aggressive = preset_thresholds("Aggressive (catch everything)")
    check(conservative.loiter_min_dwell_s > aggressive.loiter_min_dwell_s,
          "conservative demands more dwell than aggressive before calling it loitering")
    check(conservative.run_speed_ratio > aggressive.run_speed_ratio,
          "conservative needs a faster subject before calling it running")
    check(conservative.fall_sustain_s > aggressive.fall_sustain_s,
          "conservative needs a fall confirmed for longer")
    check(conservative.climb_sustain_s > aggressive.climb_sustain_s,
          "conservative needs a climb held for longer")
    check(balanced.loiter_min_dwell_s == BehaviorThresholds().loiter_min_dwell_s,
          "the balanced preset is the documented default calibration")

    # A preset must be handed out as a fresh object: the engine mutates thresholds
    # in place, so returning the shared preset would silently rewrite it forever.
    poisoned = preset_thresholds("Balanced")
    poisoned.loiter_min_dwell_s = 999.0
    check(preset_thresholds("Balanced").loiter_min_dwell_s != 999.0,
          "preset_thresholds returns a fresh copy, so live retuning cannot corrupt it")
    check(preset_thresholds("nonsense preset").loiter_min_dwell_s
          == BehaviorThresholds().loiter_min_dwell_s,
          "an unknown preset name falls back to balanced calibration")

    # ------------------------------------------------------------------
    header("16. PIPELINE INTEGRATION (shared with the dashboard and the agent)")
    # ------------------------------------------------------------------
    # The analyser is verified for real above. These stubs deliberately replace the
    # MODEL LAYER only, so the shared pipeline's wiring can be exercised with no
    # OpenCV, no model weights and no GPU.
    class FakeDetector:
        def __init__(self, path):
            self.path = list(path)
            self.index = 0

        def detect(self, frame, conf_threshold=0.35):
            x, y = self.path[min(self.index, len(self.path) - 1)]
            self.index += 1
            return [{
                "centroid": (x, y), "bbox": (x - 30, y - 90, x + 30, y + 10),
                "conf": 0.91, "class_name": "person", "category": "human",
            }]

        def draw_detections(self, frame, detections):
            return frame

    class FakeEnhancer:
        def enhance(self, frame, clip_limit=3.0):
            return frame

    class FakeWatchlist:
        def verify_vehicle(self, plate):
            return (False, None)

    class StubPose:
        """Stands in for the pose ENGINE so the pipeline wiring can be tested."""

        available = True

        def __init__(self, alerts, blow_up=False):
            self.alerts = list(alerts)
            self.blow_up = blow_up
            self.calls = 0
            self.last_error = ""

        def process(self, frame, tracks, timestamp=None, frame_idx=0,
                    fence_y=None, conf_threshold=0.25):
            self.calls += 1
            if self.blow_up:
                raise RuntimeError("stub pose backend exploded")
            return list(self.alerts), []

        def draw_overlay(self, frame, samples, alerts=()):
            return frame

    workdir = tempfile.mkdtemp(prefix="ibvap_pipe_")
    frame = np.zeros((720, 1280, 3), dtype="uint8")
    camera = get_camera("Channel 2")          # a fence-enabled sector
    fence_cfg = camera["fence"]
    crawl_alert = {
        "event_type": "CROUCH_INTRUSION", "behavior": "CROUCH_INTRUSION",
        "severity": "WARNING", "track_id": 1, "category": "human",
        "class_name": "person", "confidence": 0.7, "status": "WARNING",
        "details": "Low crawl posture", "zone": "Behaviour Analytics",
        "direction": "", "identity": "", "location": "",
        "timestamp": "04:31:00", "duration_s": 1.1,
        "provenance": "POSE_HEURISTIC", "bbox": (270, 410, 330, 600),
    }

    stub = StubPose([crawl_alert])
    models = VisionModels(
        detector=FakeDetector([(400, 200), (400, 240), (400, 270),
                               (400, 300), (400, 330), (400, 360)]),
        anpr=None, enhancer=FakeEnhancer(), watchlist=FakeWatchlist(),
        frs=None, pose=stub,
    )
    process = build_analyser(
        models, camera, fence_cfg,
        {"enable_pose": True, "tripwire_y": 280, "enable_watchlist": False,
         "enable_frs": False},
    )
    check(callable(process), "the shared pipeline returns a callable process_fn")
    check(hasattr(process, "fence") and hasattr(process, "tracker"),
          "the pipeline exposes its live fence and tracker for retuning")
    check(process.fence is not None and process.fence.p1[1] == 280,
          "the per-camera fence is built at the calibrated tripwire row")

    logger = EventLogger(
        csv_path=os.path.join(workdir, "alerts.csv"), node_id=camera["node_id"]
    )
    publisher = TelemetryPublisher(
        transport=LoopbackTransport(), max_payload_bytes=10 * 1024
    )

    events = []
    for index in range(6):
        _, frame_events = process(frame, {
            "frame_seq": index,
            "frame_ts": 1_700_000_000.0 + index * 0.1,
            "timestamp": f"04:30:0{index}",
        })
        events += frame_events

    check(stub.calls == 6, "behaviour analytics runs once per processed frame")
    kinds = {e["event_type"] for e in events}
    check("CROUCH_INTRUSION" in kinds,
          f"the shared pipeline surfaces behaviour alerts ({sorted(kinds)})")
    check(all("provenance" in e for e in events if e["event_type"] == "CROUCH_INTRUSION"),
          "behaviour events keep their heuristic provenance through the pipeline")

    for event in events:
        route_event(event, camera, frame, logger, publisher=publisher)
    check(len(logger.events) == len(events),
          f"route_event logs every event it is handed ({len(logger.events)}/{len(events)})")
    check(all(record["node_id"] == camera["node_id"] for record in logger.events),
          "routed events carry the outpost identity")
    check(all(record["utc_timestamp"].endswith("Z") for record in logger.events),
          "routed events carry a UTC timestamp for cross-outpost reconciliation")

    stats = publisher.stats()
    check(stats["events"] >= len(events),
          "route_event transmits every event it is handed")
    check(stats["over_budget"] == 0, "no routed behaviour packet exceeded the byte budget")
    check(stats["avg_payload_bytes"] <= 10 * 1024,
          f"behaviour telemetry stays inside the 10 KB budget "
          f"(avg {stats['avg_payload_bytes']} B)")
    routed_severity = {
        record["event_type"]: record["severity"]
        for record in publisher.recent_records(80)
    }
    check(routed_severity.get("CROUCH_INTRUSION") == "WARNING",
          "a crawl is routed as a WARNING packet, matching the catalogue")

    # Containment: the analytical layer must never be able to stop the video.
    exploding = StubPose([], blow_up=True)
    brittle = build_analyser(
        VisionModels(detector=FakeDetector([(400, 200)]), anpr=None,
                     enhancer=FakeEnhancer(), watchlist=FakeWatchlist(),
                     frs=None, pose=exploding),
        camera, fence_cfg, {"enable_pose": True, "tripwire_y": 280},
    )
    annotated, brittle_events = brittle(
        frame, {"frame_seq": 1, "frame_ts": 1.7e9, "timestamp": "04:32:00"}
    )
    check(annotated is not None,
          "a crashing behaviour backend cannot break the video loop")
    check(isinstance(brittle_events, list),
          "the frame still returns a valid event list when analytics fail")
    check("exploded" in exploding.last_error,
          "the failure is captured on the engine for the operator to see, not raised")

    # Off must mean off - no model pass is even attempted.
    unused = StubPose([crawl_alert])
    disabled = build_analyser(
        VisionModels(detector=FakeDetector([(400, 200)]), anpr=None,
                     enhancer=FakeEnhancer(), watchlist=FakeWatchlist(),
                     frs=None, pose=unused),
        camera, fence_cfg, {"enable_pose": False, "tripwire_y": 280},
    )
    disabled(frame, {"frame_seq": 1, "frame_ts": 1.7e9, "timestamp": "04:33:00"})
    check(unused.calls == 0,
          "with analytics switched off no inference is performed at all (no wasted CPU)")

    # The outpost liveness beacon.
    heartbeat = heartbeat_event("BOP-SECTOR-A", "3 channel(s) monitored")
    check(heartbeat["event_type"] == "SYSTEM" and heartbeat["status"] == "HEARTBEAT",
          "the heartbeat is a SYSTEM/HEARTBEAT event")
    check(severity_for("SYSTEM") == "INFO",
          "a heartbeat is INFO, so a quiet outpost never outranks a real alert")
    check(heartbeat["node_id"] == "BOP-SECTOR-A",
          "the heartbeat names the outpost, so HQ can spot a silent sector")

    shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------
    header("PHASE 8 RESULT")
    # ------------------------------------------------------------------
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for item in failures:
            print(f"    - {item}")
        print("=" * 74)
        return 1

    print("  ALL CHECKS PASSED - behaviour analytics that is geometric, calibrated")
    print("  in body-relative units, and honest about what it is:")
    print("    * 6 behaviours + group convergence, all threshold-calibrated")
    print("    * ordinary walking raises NOTHING (the false-alarm guard)")
    print("    * stooping at the waist is not a crawl; a moving crawl is not a fall")
    print("    * the climb rule requires real fence proximity, and never guesses")
    print("    * with no pose weights installed, nothing is ever inferred")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
