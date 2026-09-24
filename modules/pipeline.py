"""
The single-camera vision pipeline (AnantaNetra / IBVAP).

Why this is its own module
--------------------------
The pipeline used to live inside `app.py`, which means it could only ever run
inside Streamlit - i.e. only with a browser attached. A Border Out Post needs the
opposite: the same detection/fence/ANPR/behaviour logic running headless on an
edge box with no display at all, streaming telemetry to Sector HQ.

So the pipeline lives here, free of any UI dependency, and is called by:
  * `app.py`     - the operator dashboard (threaded grid or inline loop).
  * `run_edge_daemon.py` - the containerized headless agent.

There is exactly ONE implementation of the vision logic and ONE implementation of
the event sink (`route_event`), which is what makes the guarantee "an alert can
never be logged without also being transmitted" true rather than aspirational.

Heavy dependencies (cv2 / ultralytics / easyocr) are never imported here: the
model objects are injected, so this module imports (and the pipeline can be
exercised) on a bare Python.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from modules.fence import VirtualFence
from modules.tracker import CentroidTracker


@dataclass
class VisionModels:
    """
    The loaded model stack, injected rather than imported.

    Keeping this explicit means a deployment can substitute a different detector,
    a stub in a test, or an ONNX runtime build without touching the pipeline.
    """

    detector: Any
    anpr: Any
    enhancer: Any
    watchlist: Any
    frs: Any = None
    pose: Any = None


def build_analyser(
    models: VisionModels,
    camera: dict,
    fence_cfg,
    opts: dict,
    tracker=None,
) -> Callable:
    """
    Builds the per-camera vision pipeline.

    Returns process_fn(frame, ctx) -> (annotated_frame, events), used by BOTH the
    legacy inline loop and the threaded ingest grid, so there is exactly one
    implementation of the detection/fence/ANPR/behaviour logic.

    `opts` is a long-lived dict shared with the session, which means operator
    changes (confidence, CLAHE, watchlist, pose thresholds) take effect live
    WITHOUT rebuilding the camera grid and restarting threads.

    Each camera gets its OWN tracker: a tracker shared across cameras corrupts
    track IDs the moment two sectors are visible at once.
    """
    fence = None
    fence_y = None
    if fence_cfg:
        zone_pts = fence_cfg.get("polygon") if isinstance(fence_cfg, dict) else None
        if zone_pts and len(zone_pts) >= 3:
            # Operator-drawn polygon zone (any shape/size): the restricted area
            # is the zone itself, and entering it is the breach event.
            fence = VirtualFence(polygon=zone_pts, zone_name=fence_cfg["zone_name"])
        else:
            tripwire_y = opts.get("tripwire_y") or fence_cfg["default_tripwire_y"]
            x_start, x_end = fence_cfg["tripwire_x_range"]
            fence = VirtualFence(
                line_coords=((x_start, tripwire_y), (x_end, tripwire_y)),
                zone_name=fence_cfg["zone_name"],
            )
        # The pose layer needs the wire's row to tell a climb from a stretch.
        fence_y = fence.fence_y

    camera_tracker = tracker if tracker is not None else CentroidTracker(
        max_disappeared=20, max_distance=90.0
    )

    # Last identity decisions, kept so a skipped FRS frame can still DRAW them.
    # Identity does not change between two frames ~300 ms apart, and the module
    # already confirms over several passes, so the overlay stays truthful while
    # the expensive full-frame face pass is paced.
    frs_state = {"decisions": [], "map": {}, "seen": 0}

    # Newest ANPR read per vehicle track, so the plate overlay stays visible on
    # EVERY frame. The draw used to happen only on the frame an OCR attempt
    # succeeded (~1 read/second budgeted), so the box+text blinked off between
    # reads and the operator lost sight of which car carried which plate.
    last_anpr = {}

    def process(frame, ctx):
        events: List[dict] = []
        if camera.get("pre_rendered"):
            return frame, events

        # Timestamp/reference frame the alert to when the frame was CAPTURED, not
        # when inference happened to finish - matters on a lagging edge box.
        frame_ts = ctx.get("frame_ts") or time.time()
        timestamp = ctx.get("timestamp") or datetime.fromtimestamp(frame_ts).strftime("%H:%M:%S")
        frame_idx = int(ctx.get("frame_seq") or 0)

        if opts.get("enable_clahe"):
            frame = models.enhancer.enhance(frame, clip_limit=opts.get("clahe_limit", 3.0))

        detections = models.detector.detect(
            frame, conf_threshold=opts.get("conf_thresh", 0.35)
        )
        active_tracks = camera_tracker.update(detections)
        annotated_frame = models.detector.draw_detections(frame, detections)
        use_watchlist = models.watchlist if opts.get("enable_watchlist") else None

        # Identity FIRST: the fence needs FRS decisions to classify the crossing,
        # and the face overlay should sit under the fence annotation.
        identity_map = {}
        if opts.get("enable_frs") and models.frs is not None:
            # Face detection on a full 1080p frame is the most expensive stage in
            # the pipeline (~200 ms/frame measured), so it is paceable: at a
            # stride of 2 the operator gets the same identity decisions across a
            # video that runs twice as smoothly, because a person's face does not
            # change between consecutive frames.
            frs_stride = max(1, int(opts.get("frs_stride", 1) or 1))
            # Phased from the analyser's own frame count, not from the absolute
            # frame index: the FIRST analysed frame always runs, so face labels
            # appear immediately instead of waiting for the first stride boundary.
            run_frs = frs_stride == 1 or (frs_state["seen"] % frs_stride) == 0
            frs_state["seen"] += 1
            if run_frs:
                try:
                    decisions = models.frs.process(
                        frame, active_tracks, timestamp=timestamp, frame_idx=frame_idx
                    )
                except Exception as exc:  # a broken FRS must never stop the video
                    decisions = []
                    models.frs.last_error = f"{type(exc).__name__}: {exc}"
                frs_state["decisions"] = decisions or []
                frs_state["map"] = (
                    {d["track_id"]: d for d in decisions} if decisions else {}
                )
            if frs_state["decisions"]:
                identity_map = dict(frs_state["map"])
                annotated_frame = models.frs.draw_faces(
                    annotated_frame, frs_state["decisions"]
                )

        if fence is not None:
            new_intrusions = fence.check_intrusions(
                active_tracks, timestamp=timestamp, watchlist=use_watchlist,
                identity_map=identity_map,
            )
            for alert in new_intrusions:
                events.append({
                    **alert,
                    "details": f"{alert.get('identity', 'UNKNOWN')} | "
                               f"{alert.get('direction', 'CROSSING')} at {alert['location']} "
                               f"| ID:{alert.get('identity_source', 'NONE')}",
                    "bbox": active_tracks.get(alert["track_id"], {}).get("bbox"),
                })
            annotated_frame = fence.draw_fence(
                annotated_frame, active_tracks, watchlist=use_watchlist
            )
        else:
            for track_id, data in active_tracks.items():
                if data.get("category") != "vehicle":
                    continue
                anpr_res = models.anpr.process_vehicle(
                    frame, data["bbox"], track_id=track_id, frame_idx=frame_idx
                )
                # Fresh-read bookkeeping: a NEW dict from process_vehicle is a
                # genuine new (or improved) read - log it and fire its event.
                # A repeated return of the SAME dict is the module's cache being
                # replayed (throttle window, budget deferral, verified read) -
                # that still gets DRAWN every frame, but must not re-log the
                # same plate as a fresh detection over and over.
                anpr_fresh = anpr_res is not None and anpr_res is not last_anpr.get(track_id)
                if anpr_res is not None:
                    last_anpr[track_id] = anpr_res
                    if len(last_anpr) > 200:  # bounded: drop the oldest track
                        last_anpr.pop(next(iter(last_anpr)))
                else:
                    anpr_res = last_anpr.get(track_id)
                if anpr_res is None:
                    continue
                is_auth_veh, veh_rec = (
                    models.watchlist.verify_vehicle(anpr_res["plate_text"])
                    if anpr_fresh and opts.get("enable_watchlist") else (False, None)
                )
                if anpr_fresh:
                    veh_status = "AUTHORIZED_PATROL_VEHICLE" if is_auth_veh else anpr_res["status"]
                    veh_details = (
                        f"{veh_rec['unit']} ({veh_rec['vehicle_type']})" if is_auth_veh
                        else f"Plate Conf: {anpr_res['plate_conf']*100:.0f}%"
                    )
                    events.append({
                        "event_type": "AUTHORIZED_VEHICLE" if is_auth_veh else "VEHICLE_ANPR",
                        "track_id": track_id,
                        "category": "vehicle",
                        "class_name": data.get("class_name", "car"),
                        "confidence": data.get("conf", 0.0),
                        "plate_text": anpr_res["plate_text"],
                        "ocr_confidence": anpr_res["ocr_conf"],
                        "status": veh_status,
                        "details": veh_details,
                        "zone": "Checkpost Charlie Ingress",
                        "identity": veh_details,
                        "location": "",
                        "timestamp": timestamp,
                        "bbox": data.get("bbox"),
                    })
                # The plate box + read text are drawn EVERY frame from the
                # tracked state, so the overlay never blinks off between the
                # budgeted OCR passes.
                annotated_frame = models.anpr.draw_anpr(annotated_frame, anpr_res)

        # Behaviour analytics LAST: it consumes the final track set (so a pose is
        # bound to the same track ID the fence just used) and its overlay sits on
        # top of everything else.
        if opts.get("enable_pose") and models.pose is not None:
            try:
                behavior_alerts, samples = models.pose.process(
                    frame, active_tracks, timestamp=timestamp, frame_idx=frame_idx,
                    fence_y=fence_y,
                    conf_threshold=opts.get("pose_conf", 0.25),
                )
            except Exception as exc:  # behaviour analytics must never stop the video
                behavior_alerts, samples = [], []
                models.pose.last_error = f"{type(exc).__name__}: {exc}"
            if samples:
                annotated_frame = models.pose.draw_overlay(
                    annotated_frame, samples, behavior_alerts
                )
            events.extend(behavior_alerts)

        return annotated_frame, events

    process.fence = fence  # exposed so the UI can retune the tripwire live
    process.tracker = camera_tracker
    return process


def route_event(event: dict, camera: dict, frame, logger, publisher=None) -> Optional[dict]:
    """
    The single event sink: local audit log, then Sector HQ uplink.

    Both the dashboard and the headless daemon funnel through here, so an alert
    can never be written to the log without also being transmitted (or vice
    versa), and the camera/node attribution rules live in exactly one place.
    """
    event_camera = camera or {}
    logger.log_event(
        event_type=event["event_type"],
        track_id=event.get("track_id"),
        category=event.get("category", "human"),
        class_name=event.get("class_name", "person"),
        confidence=event.get("confidence", 0.0),
        plate_text=event.get("plate_text", "-"),
        ocr_confidence=event.get("ocr_confidence", 0.0),
        status=event.get("status", "VERIFIED"),
        details=event.get("details", ""),
        timestamp=event.get("timestamp"),
        node_id=event.get("node_id") or event_camera.get("node_id"),
        camera_id=event.get("camera_id") or event_camera.get("camera_id"),
    )
    if publisher is None:
        return None
    return publisher.publish_event(
        event, event_camera, frame=frame, bbox=event.get("bbox")
    )


def heartbeat_event(node_id: str, note: str = "", **extra) -> dict:
    """
    A liveness beacon for a sector with no detections.

    A border post going silent is itself an incident: without a heartbeat, "no
    alerts" and "the outpost is dead" look identical at Sector HQ.
    """
    return {
        "event_type": "SYSTEM",
        "track_id": None,
        "category": "system",
        "class_name": "heartbeat",
        "confidence": 1.0,
        "status": "HEARTBEAT",
        "details": note or f"Edge node {node_id} alive",
        "zone": "Uplink",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "node_id": node_id,
        **extra,
    }
