"""
IBVAP — Intelligent Border Video Analytics Platform (simplified operator view).

The dashboard was deliberately reduced to the five capabilities an operator at a
border post actually needs on screen:

  1. Live CCTV footage with tracking overlays (one channel at a time).
  2. VEHICLE TRACKING records  — every plate read (ANPR) with date, time and
     whether that vehicle is on the authorized roster.
  3. PERSON TRACKING & IDENTIFICATION records — every person event with the
     intrusion timestamp and whether the face was identified as authorized
     (biometric, via the local FAISS-style vector index) or unknown.
  4. BEHAVIOUR ANALYSIS — pose-derived alerts (loitering, crawling, falls,
     fence climbs, group convergence).
  5. The FAISS/SQLite identity index and the authorized-vehicle roster, shown
     directly beneath the tables they support.

The sidebar holds exactly the operator controls that map onto those needs:
camera selection, virtual-fence editing, image enhancement, and alert settings.
Everything else (GIS map, telemetry uplink, roadmap, pipeline plumbing) stays
available in the modules and in git history, but is no longer on the page.

The vision engine itself is UNCHANGED: the same shared pipeline
(modules/pipeline.py) that the headless edge daemon runs is used here, so
detection, fence, ANPR, FRS and behaviour logic exist in exactly one place.
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from modules.detector import ObjectDetector
from modules.tracker import CentroidTracker
from modules.anpr import CascadedANPR
from modules.logger import EventLogger, BEHAVIOR_EVENT_TYPES
from modules.enhancer import LowLightEnhancer
from modules.watchlist import WatchlistDatabase
from modules.nodes import (
    CAMERAS,
    channel_key_from_selection,
    deployment_config,
    get_camera,
    resolve_video_source,
)
from modules.ingest import EventBus, StreamManager
from modules.frs import FaceIndex, build_frs
from modules.pose import (
    BEHAVIOR_ORDER,
    build_pose_engine,
    preset_thresholds,
)
from modules import pipeline as vision_pipeline
from modules.pipeline import VisionModels, route_event

# Runtime deployment overrides (IBVAP_* environment variables). This is how a
# container or an edge appliance is provisioned without editing any code.
DEPLOY = deployment_config()
os.makedirs(DEPLOY["data_dir"], exist_ok=True)

# Local biometric roster store (SQLite + numpy cosine search, fully offline).
FRS_DB_PATH = os.path.join(DEPLOY["data_dir"], "frs_index.sqlite")

# The loaded model stack, published by load_models() and consumed by the shared
# pipeline in modules/pipeline.py - the same object the headless edge daemon uses.
MODELS = None

st.set_page_config(
    page_title="IBVAP — Border CCTV Video Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fixed operating points for the simplified view. These were operator knobs in
# the previous UI; they are now tuned once, here, to the values the optimization
# phase measured as best on the edge CPU, so the sidebar stays focused.
CONF_THRESHOLD = 0.35
FRAME_STRIDE = 1
FRS_STRIDE = 2          # face recognition every 2nd frame (identity unchanged between frames)
OCR_MIN_INTERVAL = 0.35  # seconds between grid-wide EasyOCR reads
MAX_INFERENCE_PERMITS = 2
JPEG_QUALITY = 80
UI_REFRESH_S = 0.5

# Person event types that reach the person record table, with readable labels.
PERSON_LABELS = {
    "INTRUSION_ALERT": "Perimeter intrusion",
    "AUTHORIZED_PATROL": "Authorized patrol",
    "WATCHLIST_HIT": "Watchlist subject",
}
for _behavior in BEHAVIOR_EVENT_TYPES:
    PERSON_LABELS[_behavior] = _behavior.replace("_", " ").title()
PERSON_EVENT_TYPES = tuple(PERSON_LABELS.keys())

VEHICLE_EVENT_TYPES = ("VEHICLE_ANPR", "AUTHORIZED_VEHICLE")
VEHICLE_COLUMNS = ["Date", "Time", "Plate Number", "Authorized", "Camera", "Track", "Read Conf", "Details"]
PERSON_COLUMNS = ["Date", "Time", "Event", "Authorized", "Identity", "Camera", "Track", "Details"]


@st.cache_resource
def load_models():
    """Initializes models once to eliminate reload lag."""
    detector = ObjectDetector(model_path="models/yolov8n.pt", device="cpu")
    anpr = CascadedANPR(
        plate_model_path="models/licensePlateDetector.pt",
        device="cpu",
        ocr_conf_threshold=0.40,
        throttle_frames=15,
        min_vehicle_height=50
    )
    enhancer = LowLightEnhancer(clip_limit=3.0)

    # Biometric identity: one local face index shared by the FRS module and the
    # roster, so an enrollment made from the UI is immediately matchable.
    face_index = FaceIndex(db_path=FRS_DB_PATH)
    watchlist = WatchlistDatabase(face_index=face_index)
    frs, frs_note = build_frs(
        index=face_index,
        allow_non_biometric=False,
        confirm_frames=3,
        min_face_px=40,
    )

    # Pose / behaviour analytics. When no pose weights are installed this reports
    # UNAVAILABLE with the reason and then infers nothing whatsoever.
    pose_engine, pose_note = build_pose_engine(
        model_path="models/yolov8n-pose.pt",
        device="cpu",
        enabled_behaviors=BEHAVIOR_ORDER,
    )

    # Warm every inference path ONCE, here, while the UI is still showing its
    # loading spinner (YOLO ~4.2 s, pose ~2 s, EasyOCR 3-5 s, FRS ~700 ms first
    # call on this project's own feeds; steady state is a fraction of that).
    warmup_notes = []
    if not detector.warmup():
        warmup_notes.append(f"detector: {detector.last_error}")
    if pose_engine.available and not pose_engine.warmup():
        warmup_notes.append(f"pose: {pose_engine.last_error}")
    if not anpr.warmup():
        warmup_notes.append("anpr: first-plate read will be slow")
    if not frs.warmup():
        warmup_notes.append("frs: first-face pass will be slow")
    if warmup_notes:
        print("[IBVAP] warmup incomplete: " + "; ".join(warmup_notes))

    return (detector, anpr, enhancer, watchlist, frs, frs_note, pose_engine, pose_note)


def build_analyser(camera: dict, fence_cfg, opts: dict, tracker=None):
    """
    Per-camera vision pipeline.

    The implementation lives in `modules/pipeline.py` so the dashboard and the
    headless edge daemon (`run_edge_daemon.py`) run EXACTLY the same vision code.
    """
    if MODELS is None:  # pragma: no cover - load_models() runs at import time
        raise RuntimeError("load_models() must run before build_analyser()")
    return vision_pipeline.build_analyser(MODELS, camera, fence_cfg, opts, tracker)


def effective_fence_cfg(cam: dict):
    """
    The fence configuration for a camera: the registry default, overridden by
    the operator's drawn polygon zone when one exists for this camera.
    """
    cfg = cam.get("fence")
    if cfg is None:
        return None
    drawn = st.session_state.get("fence_overrides", {}).get(cam["camera_id"])
    if drawn and len(drawn) >= 3:
        return {**cfg, "polygon": drawn}
    return cfg


def ensure_stream_manager(cameras, opts: dict, permits: int, jpeg_quality: int = JPEG_QUALITY):
    """
    Starts (or reuses) the threaded ingest grid for this camera set.

    The manager lives in session state - NOT in st.cache_resource - because a
    cached resource has no teardown hook: re-keying it would orphan a whole set
    of running camera threads, invisibly double-processing every sector. Here a
    re-key always stops the previous grid first.
    """
    signature = (
        tuple(cam["camera_id"] for cam in cameras),
        max(1, int(permits)),
        int(jpeg_quality),
        # Fence SHAPE is baked into each analyser at build time, so a newly
        # drawn zone must rebuild the grid (the Y-slider alone does not - it is
        # retuned live on the running analyser).
        tuple(
            tuple(map(tuple, (effective_fence_cfg(cam) or {}).get("polygon") or []))
            for cam in cameras
        ),
    )
    manager = st.session_state.get("manager")

    if manager is None or st.session_state.get("manager_signature") != signature:
        if manager is not None:
            manager.stop_all()
        manager = StreamManager(
            event_bus=EventBus(capacity=4000),
            stall_timeout_s=3.0,
            max_concurrent_analysis=max(1, int(permits)),
            watchdog_interval_s=0.5,
            jpeg_quality=int(jpeg_quality),
        )
        analysers = {}
        for cam in cameras:
            process_fn = build_analyser(cam, effective_fence_cfg(cam), opts)
            analysers[cam["camera_id"]] = process_fn
            manager.add_camera(cam, process_fn=process_fn)
        st.session_state.manager = manager
        st.session_state.analysers = analysers
        st.session_state.manager_signature = signature
    return manager


# ---------------------------------------------------------------------------
# Record-table builders (vehicle / person), straight from the event log.
# ---------------------------------------------------------------------------

def split_ts(ts) -> tuple:
    """Splits 'YYYY-MM-DD HH:MM:SS' into a (date, time) pair for the tables."""
    text = str(ts or "").strip()
    if " " in text:
        date_part, time_part = text.split(" ", 1)
        return date_part, time_part
    return "—", text or "—"


def identity_from_details(details: str) -> str:
    """Event details lead with the identity string: 'UNKNOWN PERSON | ...'."""
    return str(details or "").split(" | ")[0].strip() or "—"


def build_vehicle_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vehicle tracking records: one row per plate the ANPR engine read, newest
    first — plate number plus the date/time it was detected and whether that
    plate is on the authorized vehicle roster.
    """
    empty = pd.DataFrame(columns=VEHICLE_COLUMNS)
    if df.empty:
        return empty
    veh = df[
        df["event_type"].isin(VEHICLE_EVENT_TYPES)
        & df["plate_text"].notna()
    ]
    # A plate cell must hold an actual plate number: drop the sentinel the ANPR
    # engine writes when OCR failed, and 1-3 character fragments read off plates
    # still too far away to be legible. The raw log keeps everything.
    plate = veh["plate_text"].astype(str).str.strip()
    veh = veh[(plate.str.len() >= 4) & (plate != "PLATE_UNREADABLE")]
    if veh.empty:
        return empty
    rows = []
    for _, ev in veh.iterrows():
        date_part, time_part = split_ts(ev["timestamp"])
        rows.append({
            "Date": date_part,
            "Time": time_part,
            "Plate Number": ev["plate_text"],
            "Authorized": "YES" if ev["event_type"] == "AUTHORIZED_VEHICLE" else "NO",
            "Camera": ev["camera_id"],
            "Track": ev["track_id"],
            "Read Conf": f"{float(ev['ocr_confidence'] or 0) * 100:.0f}%",
            "Details": ev["details"],
        })
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


def build_person_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Person tracking records: one row per person event, newest first — the
    timestamp the event occurred and whether the person was identified as
    authorized (biometric match) or remains unknown.
    """
    empty = pd.DataFrame(columns=PERSON_COLUMNS)
    if df.empty:
        return empty
    per = df[df["event_type"].isin(PERSON_EVENT_TYPES)]
    if per.empty:
        return empty
    rows = []
    for _, ev in per.iterrows():
        date_part, time_part = split_ts(ev["timestamp"])
        authorized = str(ev["event_type"]).startswith("AUTHORIZED")
        rows.append({
            "Date": date_part,
            "Time": time_part,
            "Event": PERSON_LABELS.get(ev["event_type"], ev["event_type"]),
            "Authorized": "YES" if authorized else "NO",
            "Identity": identity_from_details(ev["details"]),
            "Camera": ev["camera_id"],
            "Track": ev["track_id"],
            "Details": ev["details"],
        })
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


def paint_banner(banner_slot, df: pd.DataFrame, paused: bool) -> None:
    """One-line alert banner above the video, driven by the newest event."""
    if paused:
        banner_slot.caption("🔔 Alert pop-ups paused — recording continues below.")
        return
    if df.empty:
        banner_slot.caption("🟢 Monitoring — no incidents recorded yet.")
        return
    latest = df.iloc[-1]
    etype = latest["event_type"]
    when = latest["timestamp"]
    if etype in ("WATCHLIST_HIT", "INTRUSION_ALERT"):
        banner_slot.error(
            f"🚨 {identity_from_details(latest['details'])} — {latest['camera_id']} at {when}"
        )
    elif etype in ("AUTHORIZED_PATROL", "AUTHORIZED_VEHICLE"):
        banner_slot.success(f"✅ Authorized: {latest['details']} — {latest['camera_id']} at {when}")
    elif etype in BEHAVIOR_EVENT_TYPES:
        line = (
            f"🧍 Behaviour: {PERSON_LABELS.get(etype, etype)} — Track #{latest['track_id']} • "
            f"{latest['details']}"
        )
        if latest["status"] == "CRITICAL":
            banner_slot.error(line)
        else:
            banner_slot.warning(line)
    else:
        banner_slot.caption(f"🟢 Monitoring — last event: {etype} at {when}")


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

(
    detector, anpr, enhancer, watchlist, frs, frs_note, pose_engine, pose_note
) = load_models()

# Rebuild the shared model bundle on EVERY script run from the cached objects
# above. Cheap (it only wraps references), and it is what keeps MODELS defined
# on Streamlit's second run - see the note in load_models().
MODELS = VisionModels(
    detector=detector,
    anpr=anpr,
    enhancer=enhancer,
    watchlist=watchlist,
    frs=frs,
    pose=pose_engine,
)

# Persistent session state.
if "logger" not in st.session_state:
    st.session_state.logger = EventLogger(
        csv_path=os.path.join(DEPLOY["data_dir"], "alerts.csv"),
        node_id=DEPLOY["node_id"],
    )
if "tracker" not in st.session_state:
    st.session_state.tracker = CentroidTracker(max_disappeared=20, max_distance=90.0)
if "active_channel" not in st.session_state:
    st.session_state.active_channel = "Channel 2"

logger = st.session_state.logger

# Apply the fixed operating points to the (cached) analytics engines so a
# restart can never leave them at stale values from an older session.
pose_engine.configure(
    enabled_behaviors=list(BEHAVIOR_ORDER),
    **vars(preset_thresholds("Balanced")),
)
frs.allow_non_biometric = False
frs.min_face_px = 40
frs.confirm_frames = 3
anpr_pacer = getattr(anpr, "set_ocr_budget", None)
if callable(anpr_pacer):
    anpr_pacer(min_interval_s=OCR_MIN_INTERVAL)

# ===========================================================================
# SIDEBAR — camera select · virtual fence editing · image enhancement · alerts
# ===========================================================================
st.sidebar.markdown("### 🛡️ IBVAP Command Station")
st.sidebar.caption("Ministry of Home Affairs — Sashastra Seema Bal")
st.sidebar.markdown("---")

CHANNEL_LABELS = {
    "Channel 1": "Channel 1 — Sector A · BOP Perimeter (persons & fence)",
    "Channel 2": "Channel 2 — Sector B · Pedestrian Crossing (persons & fence)",
    "Channel 3": "Channel 3 — Checkpost Charlie (vehicles & plates)",
    "Channel 4": "Channel 4 — Backup pre-recorded demo feed",
}
channel_key = st.sidebar.selectbox(
    "📹 CCTV Camera",
    list(CHANNEL_LABELS.keys()),
    index=1,
    format_func=lambda key: CHANNEL_LABELS[key],
)

# Detect a channel switch and reset the tracker so track IDs restart cleanly.
if st.session_state.active_channel != channel_key:
    st.session_state.active_channel = channel_key
    st.session_state.tracker = CentroidTracker(max_disappeared=20, max_distance=90.0)
    # Drop the held analysed frame: it belongs to the PREVIOUS channel, and
    # repainting it over the new feed would show the wrong scene (with the
    # wrong fence) until the new channel's first analysis lands.
    st.session_state.last_annotated_jpeg = None
    st.session_state.last_video_camera = None
    # An editor snapshot from another channel would put the drawing canvas over
    # the wrong scene, so any open editing session dies with the switch.
    st.session_state.editor_points = None
    st.session_state.editor_snapshot = None

active_camera = get_camera(channel_key)
logger.set_context(node_id=active_camera["node_id"], camera_id=active_camera["camera_id"])

# --- Virtual fence editing -------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Virtual Fence")
fence_cfg = active_camera.get("fence")
fence_overrides = st.session_state.setdefault("fence_overrides", {})
drawn_zone = fence_overrides.get(active_camera["camera_id"])
tripwire_y = None

if fence_cfg is None:
    st.sidebar.caption(
        "No fence on this channel — the checkpost lane tracks vehicles and reads plates instead."
    )
else:
    if drawn_zone:
        st.sidebar.success(f"Custom restricted zone active — {len(drawn_zone)}-sided area")
        if st.sidebar.button("↩️ Reset to default straight line"):
            fence_overrides.pop(active_camera["camera_id"], None)
            st.session_state.editor_points = None
            st.rerun()
    else:
        tripwire_y = st.sidebar.slider(
            "Fence line position (Y)",
            min_value=fence_cfg["slider_min"],
            max_value=fence_cfg["slider_max"],
            value=fence_cfg["default_tripwire_y"],
            step=10,
            help="Drag to move the straight tripwire — or draw a custom-shaped zone below.",
        )
    st.sidebar.caption(f"Zone: {fence_cfg['zone_name']}")
    if st.sidebar.button("✏️ Edit fence shape (draw a polygon)", width="stretch"):
        # A fresh component key guarantees the editor starts with NO stale click
        # from a previous editing session.
        st.session_state.editor_points = []
        st.session_state.editor_key = f"{active_camera['camera_id']}_{time.time_ns()}"

# --- Image enhancement -----------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🌟 Image Enhancement")
enable_clahe = st.sidebar.checkbox(
    "🌙 Low-light / fog enhancement (CLAHE)",
    value=True,
    help="Restores visibility in pitch-black or foggy border feeds BEFORE detection runs.",
)
clahe_limit = 3.0
if enable_clahe:
    clahe_limit = st.sidebar.slider(
        "Contrast strength (clip limit)", 1.0, 6.0, 3.0, 0.5,
        help="Higher = stronger local contrast boost. Too high can amplify sensor noise.",
    )
st.sidebar.caption("Enhancement is applied to every frame before detection and identification.")

# --- Alerts ----------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔔 Alerts")
alert_paused = st.sidebar.checkbox(
    "Pause alert pop-ups (keep recording)",
    value=False,
    help="Hides the flashing banner. Every event is STILL logged to the record tables and CSV.",
)
run_stream = st.sidebar.checkbox("▶️ Run CCTV feed", value=True)
if st.sidebar.button("🔄 Clear all records"):
    logger.clear()
    st.session_state.tracker = CentroidTracker(max_disappeared=20, max_distance=90.0)
    st.sidebar.success("All records and active tracks cleared.")
    st.rerun()

st.sidebar.caption(f"🧬 Identity index: {FRS_DB_PATH}")

# Live operator options handed to every analyser (long-lived dict, so changes
# reach running analysers without tearing down the camera grid).
analyser_opts = st.session_state.setdefault("analyser_opts", {})
analyser_opts.update({
    "conf_thresh": CONF_THRESHOLD,
    "enable_clahe": enable_clahe,
    "clahe_limit": clahe_limit,
    "enable_watchlist": True,
    "enable_frs": True,
    "enable_pose": bool(pose_engine.available),
    "frs_stride": FRS_STRIDE,
    "tripwire_y": tripwire_y,
})

# ===========================================================================
# MAIN PAGE — one CCTV view + the three record blocks underneath
# ===========================================================================
st.markdown("## 🛡️ IBVAP — Border CCTV Video Analytics")
st.caption(
    "Vehicle tracking · person tracking & identification · behaviour analysis — "
    "all processing on-premise, nothing leaves this machine."
)

# KPI strip (driven by the same record tables shown below).
k1, k2, k3, k4, k5 = st.columns(5)
ph_k_plates = k1.empty()
ph_k_auth_veh = k2.empty()
ph_k_persons = k3.empty()
ph_k_unauth = k4.empty()
ph_k_behavior = k5.empty()

video_col, side_col = st.columns([3, 1])
with video_col:
    st.markdown(f"**📹 Live CCTV — {active_camera['label']}**")
    ph_banner = st.empty()
    ph_video = st.empty()
    # Claim the slots up front: a fragment may only paint into containers that
    # were written to during the full script run.
    ph_banner.caption("Monitoring feed…")
    ph_video.info("Initialising CCTV pipeline…")

with side_col:
    frs_state = frs.state()
    st.markdown("**🧬 Identification Engine**")
    if frs_state["mode"] == "BIOMETRIC":
        st.success("Biometric mode — offline face matching active")
    elif frs_state["mode"] == "DEGRADED_NON_BIOMETRIC":
        st.warning("Degraded mode — appearance matching only")
    else:
        st.error("FRS unavailable — simulated roster fallback")
    st.caption(
        f"{frs_state['index']['identities']} identity(ies) • "
        f"{frs_state['index']['faces']} face vector(s) • "
        f"{frs_state['index']['dim'] or '—'}-d embeddings"
    )
    pose_state = pose_engine.state()
    st.markdown("**🧍 Behaviour Engine**")
    if pose_state["available"]:
        st.success(f"Active — {pose_state['backend']}")
        st.caption(f"{len(pose_state['behaviors'])} behaviours monitored")
    else:
        st.error("Offline — pose model not loaded")
    anpr_stats_fn = getattr(anpr, "stats", None)
    if callable(anpr_stats_fn):
        st.markdown("**🔤 Plate Reader (ANPR)**")
        paced = anpr_stats_fn()
        st.caption(
            f"{paced['ocr_calls']} read(s) • {paced['tracks_cached']} track(s) cached • "
            f"paced {paced['min_interval_s']}s apart"
        )

if not os.path.exists(resolve_video_source(active_camera) or ""):
    st.error(f"Target video feed not located: {resolve_video_source(active_camera)}")

# ===========================================================================
# VIRTUAL FENCE EDITOR — draw a restricted zone of ANY shape on the feed
# ===========================================================================
# The editor pauses the live grid (freeing the CPU for the editor), shows a
# full-resolution frame, and lets the operator click vertices directly on the
# image: click to add points, button to close the shape. Every click is
# immediately re-drawn, so shaping an L-shaped yard or a wedge between two
# paths is a live, visual operation - not a numbers exercise.
if fence_cfg is not None and st.session_state.get("editor_points") is not None:
    video_path_editor = resolve_video_source(active_camera)
    editor_frame = st.session_state.get("editor_snapshot")
    if editor_frame is None and video_path_editor and os.path.exists(video_path_editor):
        cap = cv2.VideoCapture(video_path_editor)
        best = None
        # Sample a few frames and keep the busiest one: frame 0 of a demo loop
        # is often a fade-in, which would be a terrible canvas to draw on.
        for _ in range(12):
            ok, f = cap.read()
            if not ok:
                break
            if best is None or f.mean() > best.mean():
                best = f
        cap.release()
        editor_frame = best
        st.session_state.editor_snapshot = best

    if editor_frame is None:
        st.error("Could not grab a frame to draw on — check the camera's video source.")
        st.session_state.editor_points = None
    else:
        eh, ew = editor_frame.shape[:2]
        pts = st.session_state.editor_points
        editor_col, guide_col = st.columns([3, 1])
        with editor_col:
            st.markdown("#### ✏️ Fence Editor — click to place zone corners")
            clicked = streamlit_image_coordinates(
                cv2.cvtColor(editor_frame, cv2.COLOR_BGR2RGB),
                key=st.session_state.get("editor_key", "fence_editor"),
            )
            if clicked is not None and len(pts) < 24:
                px = int(clicked["x"] * ew / clicked["width"])
                py = int(clicked["y"] * eh / clicked["height"])
                new_pt = (px, py)
                if not pts or pts[-1] != new_pt:  # ignore double-fires of one click
                    pts.append(new_pt)
                    st.session_state.editor_points = pts
                    st.rerun()
        with guide_col:
            st.markdown("**How to draw**")
            st.markdown(
                "1. Click on the image to drop each corner of the restricted "
                "area — any shape, any size.\n"
                "2. 3 or more corners make a zone.\n"
                "3. Press **Apply** when the outline looks right."
            )
            st.caption(f"{len(pts)} corner(s) placed • frame {ew}x{eh}")
            if st.button("↩️ Undo last corner"):
                pts.pop()
                st.session_state.editor_points = pts
                # Bump the component key: the click-coordinate component replays
                # its last click under an unchanged key, which would re-add the
                # corner that was just removed.
                st.session_state.editor_key = f"{active_camera['camera_id']}_{time.time_ns()}"
                st.rerun()
            if st.button("🗑️ Clear all corners"):
                st.session_state.editor_points = []
                st.session_state.editor_key = f"{active_camera['camera_id']}_{time.time_ns()}"
                st.rerun()
            if len(pts) >= 3 and st.button("✅ Apply zone", width="stretch"):
                fence_overrides[active_camera["camera_id"]] = list(pts)
                st.session_state.editor_points = None
                st.session_state.editor_snapshot = None
                st.rerun()
            if st.button("✖️ Cancel", width="stretch"):
                st.session_state.editor_points = None
                st.session_state.editor_snapshot = None
                st.rerun()
        # Preview canvas: frame + the shape drawn so far, freshly computed
        # each rerun so every new corner appears instantly.
        preview = editor_frame.copy()
        if pts:
            for qx, qy in pts:
                cv2.circle(preview, (qx, qy), 7, (0, 255, 255), -1)
            if len(pts) >= 2:
                qpts = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(preview, [qpts], False, (0, 0, 255), 2)
            if len(pts) >= 3:
                overlay = preview.copy()
                cv2.fillPoly(overlay, [np.array(pts, np.int32)], (0, 0, 180))
                cv2.addWeighted(overlay, 0.25, preview, 0.75, 0, dst=preview)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), width="stretch")

# --- Record tables (below the CCTV footage) --------------------------------
st.markdown("---")
st.markdown("### 🚗 Vehicle Tracking Records")
st.caption(
    "Every number plate the ANPR engine reads — with the date, the time it was detected, "
    "and whether that vehicle is on the authorized roster. Newest first."
)
ph_vehicle_table = st.empty()
ph_vehicle_table.dataframe(build_vehicle_table(logger.get_dataframe()), width="stretch", hide_index=True)

st.markdown("### 🧍 Person Tracking & Identification Records")
st.caption(
    "Every person event at the fence — the exact time the intrusion occurred and whether "
    "the person was identified as authorized or remains unknown. Newest first."
)
ph_person_table = st.empty()
ph_person_table.dataframe(build_person_table(logger.get_dataframe()), width="stretch", hide_index=True)

st.markdown("#### 🗂️ FAISS Identity Database (authorized persons)")
ph_fais = st.empty()
biometric_roster = watchlist.get_biometric_dataframe()
if biometric_roster.empty:
    ph_fais.info(
        "No biometric identities enrolled yet — the DEMO roster below is simulated, not biometric."
    )
else:
    ph_fais.dataframe(biometric_roster, width="stretch", hide_index=True)

st.markdown("### 🧍 Behaviour Analysis")
ph_behaviour = st.empty()

# Authorized vehicle roster: the reference the vehicle table is checked against.
with st.expander("🚙 Authorized Vehicle Roster (plates the system treats as friendly)", expanded=False):
    st.dataframe(watchlist.get_vehicles_dataframe(), width="stretch", hide_index=True)


def paint_kpis() -> None:
    """KPI strip — mirrors exactly what the two record tables show."""
    df = logger.get_dataframe()
    veh_df = build_vehicle_table(df)
    per_df = build_person_table(df)
    behavior_count = int(df["event_type"].isin(BEHAVIOR_EVENT_TYPES).sum()) if not df.empty else 0
    ph_k_plates.metric("🚗 Plates Read", len(veh_df))
    ph_k_auth_veh.metric(
        "✅ Authorized Vehicles",
        int((veh_df["Authorized"] == "YES").sum()) if not veh_df.empty else 0,
    )
    ph_k_persons.metric("🧍 Person Events", len(per_df))
    ph_k_unauth.metric(
        "🚨 Unauthorized Persons",
        int((per_df["Authorized"] == "NO").sum()) if not per_df.empty else 0,
    )
    ph_k_behavior.metric("🧍 Behaviour Alerts", behavior_count)


def paint_behaviour() -> None:
    """Behaviour analysis block: live per-track metrics + alert history."""
    pose_state = pose_engine.state()
    if not pose_state["available"]:
        ph_behaviour.warning(
            "Behaviour analytics is OFFLINE — no pose backend loaded, so NOTHING is being "
            f"inferred. Reason: {pose_state['note']}"
        )
        return

    metrics_rows = pose_engine.analyser.track_metrics()
    df = logger.get_dataframe()
    if not df.empty:
        beh = df[df["event_type"].isin(BEHAVIOR_EVENT_TYPES)]
        recent = beh.tail(8).iloc[::-1][
            ["timestamp", "event_type", "track_id", "status", "details"]
        ].rename(columns={
            "timestamp": "Time", "event_type": "Behaviour", "track_id": "Track",
            "status": "Severity", "details": "Evidence",
        })
    else:
        recent = pd.DataFrame(columns=["Time", "Behaviour", "Track", "Severity", "Evidence"])

    if metrics_rows:
        ph_behaviour.dataframe(
            pd.DataFrame(recent), width="stretch", hide_index=True
        )
        st.caption(
            f"Live tracks: {len(metrics_rows)} • "
            f"{pose_state['analyser']['events_emitted']} behaviour alert(s) raised • "
            f"{pose_state['analyser']['suppressed_by_cooldown']} suppressed by cooldown • "
            f"{pose_state['last_latency_ms']} ms/frame"
        )
    elif not recent.empty:
        ph_behaviour.dataframe(recent, width="stretch", hide_index=True)
    else:
        ph_behaviour.caption("No posed tracks or behaviour alerts yet in this session.")


def paint_records() -> None:
    """Refresh both record tables + the identity roster from the event log."""
    df = logger.get_dataframe()
    ph_vehicle_table.dataframe(build_vehicle_table(df), width="stretch", hide_index=True)
    ph_person_table.dataframe(build_person_table(df), width="stretch", hide_index=True)
    roster = watchlist.get_biometric_dataframe()
    if roster.empty:
        ph_fais.info(
            "No biometric identities enrolled yet — the DEMO roster is simulated, not biometric."
        )
    else:
        ph_fais.dataframe(roster, width="stretch", hide_index=True)
    paint_behaviour()


# ===========================================================================
# LIVE ENGINE — threaded ingest + inference, identical to the edge daemon
# ===========================================================================
cameras_by_id = {channel_key: active_camera}


def consume(events) -> None:
    """Single event sink: local audit log (CSV + in-memory tables)."""
    for event in events:
        event_camera = cameras_by_id.get(event.get("camera_id"), active_camera)
        route_event(event, event_camera, None, logger, publisher=None)


def render_view() -> None:
    """One UI tick: drain events, paint video, banner, KPIs and the tables."""
    manager = st.session_state.get("manager")
    if manager is not None:
        consume(manager.drain_events(max_items=500))

        # Paint the ALREADY-ENCODED JPEG from the analysis thread (~7 ms / ~200 KB
        # per tick versus 51 ms / ~2 MB PNG for a raw array on the UI thread).
        #
        # Overlays are baked into the ANALYSED frame only, so the fence and the
        # plate box exist nowhere else. Inference on the edge CPU (~0.5-7 s per
        # pass) runs slower than this UI tick (0.5 s), so on most ticks nothing
        # new has arrived. Holding the LAST ANALYSED frame keeps the fence and
        # plate overlay continuously visible - the fallback to the RAW grab-
        # thread frame used here before is what made the fence vanish from the
        # recording whenever the UI outran the model. Raw is shown ONLY before
        # the first analysed frame of a channel exists.
        camera_id = active_camera["camera_id"]
        jpeg = manager.latest_annotated_jpeg(camera_id)
        live_frame = manager.latest_annotated(camera_id)
        if jpeg is not None:
            st.session_state.last_annotated_jpeg = jpeg
            st.session_state.last_video_camera = camera_id
            st.session_state.last_frame = live_frame
            ph_video.image(jpeg, width="stretch")
        elif live_frame is not None and st.session_state.get("last_annotated_jpeg") is None:
            # No JPEG encoder available AND nothing held yet: rare cv2-missing
            # fallback. Keep the array in the hold slot too, converted per tick.
            st.session_state.last_frame = live_frame
            st.session_state.last_video_camera = camera_id
            ph_video.image(cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB), width="stretch")
        elif (
            st.session_state.get("last_annotated_jpeg") is not None
            and st.session_state.get("last_video_camera") == camera_id
        ):
            # Inference is still working on the next frame: HOLD the last
            # analysed picture on the wall display rather than flashing the
            # un-annotated raw feed. The fence/plate overlay therefore never
            # disappears mid-session.
            ph_video.image(st.session_state.last_annotated_jpeg, width="stretch")
        else:
            # First ticks on a channel before ANY analysed frame exists.
            raw = manager.latest_frame(camera_id)
            if raw is not None:
                ph_video.image(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), width="stretch")
            else:
                ph_video.info("Connecting to camera feed…")

    paint_banner(ph_banner, logger.get_dataframe(), alert_paused)
    paint_kpis()
    paint_records()


if run_stream:
    manager = ensure_stream_manager(
        [active_camera], analyser_opts, MAX_INFERENCE_PERMITS, JPEG_QUALITY
    )

    # Live tripwire retune: move the line on the RUNNING analyser rather than
    # rebuilding the grid, so fence editing feels instant from the sidebar.
    if fence_cfg and tripwire_y is not None:
        live_analyser = st.session_state.get("analysers", {}).get(active_camera["camera_id"])
        live_fence = getattr(live_analyser, "fence", None)
        if live_fence is not None:
            x_start, x_end = fence_cfg["tripwire_x_range"]
            live_fence.update_line((x_start, tripwire_y), (x_end, tripwire_y))

    if hasattr(st, "fragment"):
        @st.fragment(run_every=f"{UI_REFRESH_S}s")
        def live_view_fragment():
            render_view()

        live_view_fragment()
    else:
        # Streamlit too old for fragments: bounded paint loop, then re-run.
        cycles = max(1, int(10 / UI_REFRESH_S))
        for _ in range(cycles):
            render_view()
            time.sleep(UI_REFRESH_S)
        st.rerun()
else:
    st.info("CCTV feed paused. Tick '▶️ Run CCTV feed' in the sidebar to resume.")
    paint_banner(ph_banner, logger.get_dataframe(), alert_paused)
    paint_kpis()
    paint_records()
