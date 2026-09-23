import base64
import io
import json
import os
import time
from datetime import datetime
import cv2
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from modules.detector import ObjectDetector
from modules.tracker import CentroidTracker
from modules.fence import VirtualFence
from modules.anpr import CascadedANPR
from modules.logger import EventLogger
from modules.enhancer import LowLightEnhancer
from modules.watchlist import WatchlistDatabase
from modules.nodes import (
    CAMERAS,
    OUTPOSTS,
    channel_key_from_selection,
    deployment_config,
    get_camera,
    resolve_video_source,
)
from modules.ingest import EventBus, StreamManager, source_from_camera
from modules.frs import FaceIndex, build_frs
from modules.pose import (
    BEHAVIOR_CATALOGUE,
    BEHAVIOR_ORDER,
    BehaviorThresholds,
    build_pose_engine,
    preset_thresholds,
)
from modules.nodes import RESPONSE_SPEED_KMH
from modules import gis
from modules import pipeline as vision_pipeline
from modules.logger import BEHAVIOR_EVENT_TYPES
from modules.pipeline import VisionModels, route_event
from modules.telemetry import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    LINK_PROFILES,
    SnapshotEncoder,
    TelemetryPublisher,
    build_transport,
    severity_for,
)

# Runtime deployment overrides (IBVAP_* environment variables). This is how a
# container or an edge appliance is provisioned without editing any code.
DEPLOY = deployment_config()
os.makedirs(DEPLOY["data_dir"], exist_ok=True)

# Local biometric roster store (SQLite + numpy cosine search, fully offline).
FRS_DB_PATH = os.path.join(DEPLOY["data_dir"], "frs_index.sqlite")

# The loaded model stack, published by load_models() and consumed by the shared
# pipeline in modules/pipeline.py - the same object the headless edge daemon uses.
MODELS = None

# Human-readable labels for the behaviour toggles, so the sidebar shows
# "Fence Climb Attempt (CRITICAL)" rather than an internal constant.
BEHAVIOR_ORDER_BY_NAME = {
    name: f"{meta['label']} ({meta['severity']})"
    for name, meta in BEHAVIOR_CATALOGUE.items()
}

# Page Configuration
st.set_page_config(
    page_title="IBVAP — Border CCTV Video Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Military / Surveillance CSS Styling
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Top Header Bar */
    .header-box {
        background: linear-gradient(90deg, #161b22 0%, #1f2937 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-title {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #58a6ff;
        margin: 0;
    }
    .header-subtitle {
        font-size: 13px;
        color: #8b949e;
        margin: 4px 0 0 0;
    }
    .badge-live {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        letter-spacing: 1px;
        animation: pulse 2s infinite;
    }
    .badge-status {
        background-color: #238636;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Metric Containers */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="metric-container"]:hover {
        border-color: #58a6ff;
    }

    /* Video Player Container */
    .video-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }

    /* Roadmap Card */
    .roadmap-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .roadmap-title {
        font-size: 16px;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 6px;
    }
    .roadmap-desc {
        font-size: 13px;
        color: #8b949e;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


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

    # NOTE: the shared model bundle is assembled by the CALLER, at module level.
    # This function is wrapped in @st.cache_resource, so after the first run its
    # body does not execute again - and a module global assigned inside a cached
    # function is reset to None on every Streamlit rerun while the cache keeps
    # returning the old objects, leaving that global permanently None.
    return (
        detector, anpr, enhancer, watchlist, frs, frs_note, pose_engine, pose_note
    )


@st.cache_resource(show_spinner=False)
def load_telemetry_publisher(mode: str, host: str, port: int):
    """Builds the telemetry uplink once per TRANSPORT identity.

    Link profile and byte budget are deliberately NOT part of the cache key:
    they are runtime conditions applied via set_link_profile()/set_budget(), so
    an operator can change them mid-demo without losing the store-and-forward
    backlog or the delivery statistics.

    sleep=False is deliberate: simulated satellite latency must never stall the
    live video loop. Latency is still accounted for in the link statistics.
    """
    transport, note = build_transport(
        mode=mode,
        profile="SATELLITE",
        sleep=False,
        mqtt_kwargs={"host": host, "port": int(port)},
    )
    publisher = TelemetryPublisher(transport=transport)
    return publisher, note


def build_analyser(camera: dict, fence_cfg, opts: dict, tracker=None):
    """
    Per-camera vision pipeline.

    The implementation lives in `modules/pipeline.py` so the dashboard and the
    headless edge daemon (`run_edge_daemon.py`) run EXACTLY the same vision code.
    The only thing that differs between the two deployments is the UI - never the
    analytics, and never the alerting contract.
    """
    if MODELS is None:  # pragma: no cover - load_models() runs at import time
        raise RuntimeError("load_models() must run before build_analyser()")
    return vision_pipeline.build_analyser(MODELS, camera, fence_cfg, opts, tracker)


def consume_events(events, publish_telemetry: bool, resolve_target) -> None:
    """
    Single sink for pipeline events: local audit log + Sector HQ uplink.

    Both the inline loop and the threaded grid funnel through here, so an alert
    can never be logged without also being transmitted (or vice versa).

    `resolve_target(event) -> (camera, frame)` attributes each event to the
    camera that actually produced it. In grid mode that matters: five sectors are
    running at once, and an alert from Sector B must never be filed against the
    sector the operator happens to be watching.
    """
    for event in events:
        event_camera, frame = resolve_target(event)
        result = route_event(
            event,
            event_camera,
            frame,
            logger,
            publisher=publisher if publish_telemetry else None,
        )
        if publish_telemetry:
            st.session_state.last_telemetry = result


def paint_metrics(metric_slots):
    """Writes the four KPI cards into the given placeholders. Returns the log."""
    stats = logger.get_stats()
    df = logger.get_dataframe()
    auth_count = (
        len(df[df["event_type"].str.contains("AUTHORIZED", na=False)])
        if not df.empty else 0
    )
    metric_slots[0].metric("🚨 Intrusion Alerts", stats["intrusions"])
    metric_slots[1].metric("✅ Authorized Patrols", auth_count)
    metric_slots[2].metric("🚗 Vehicles Tracked", stats["vehicles_detected"])
    metric_slots[3].metric("🪪 Verified Plates Read", stats["plates_read"])
    if len(metric_slots) > 4:
        metric_slots[4].metric("🧍 Behaviour Alerts", stats["behavior_alerts"])
    return df


def refresh_dashboard(metric_slots, banner_slot, alerts_slot) -> None:
    """Repaints KPIs, the alert banner and the incident feed from the logger.

    Every slot is written on EVERY paint - including the "no incidents yet"
    case. A fragment may only fill a container that was written to during the
    full script run (Streamlit reserves the position then), so an early return
    here leaves the incident column unclaimed and the FIRST real alert crashes
    the live view. That is a bug an empty log hides perfectly.
    """
    df = paint_metrics(metric_slots)
    feed_columns = ["timestamp", "event_type", "track_id", "status", "details"]

    if df.empty:
        banner_slot.caption("🟢 Monitoring sector feed — no incidents logged yet.")
        alerts_slot.dataframe(
            pd.DataFrame(columns=feed_columns), width="stretch", hide_index=True
        )
        return

    last_event = df.iloc[-1]
    if last_event["event_type"] == "WATCHLIST_HIT":
        banner_slot.error(
            f"🚨 BIOMETRIC WATCHLIST MATCH: {last_event['details']} • Immediate intercept dispatched"
        )
    elif last_event["event_type"] in ("AUTHORIZED_PATROL", "AUTHORIZED_VEHICLE"):
        banner_slot.success(
            f"✅ MATCH FOUND (Authorized Patrol): {last_event['details']} • False Alarm Suppressed"
        )
    elif last_event["event_type"] == "INTRUSION_ALERT":
        banner_slot.error(
            f"🚨 NO MATCH (Unknown Intruder): Track #{last_event['track_id']} Breached Perimeter! • Security Dispatched"
        )
    elif last_event["event_type"] in BEHAVIOR_EVENT_TYPES:
        # Behaviour alerts describe conduct, not a breach: a CRITICAL one (a climb,
        # a man down) still warrants the red banner, a WARNING the amber one.
        line = (
            f"🧍 BEHAVIOUR: {last_event['event_type']} — Track #{last_event['track_id']} • "
            f"{last_event['details']}"
        )
        if last_event["status"] == "CRITICAL":
            banner_slot.error(line)
        else:
            banner_slot.warning(line)
    alerts_slot.dataframe(
        df.tail(7)[feed_columns],
        width="stretch",
        hide_index=True,
    )


def ensure_stream_manager(cameras, opts: dict, permits: int):
    """
    Starts (or reuses) the threaded ingest grid for this camera set.

    The manager lives in session state - NOT in st.cache_resource - because a
    cached resource has no teardown hook: re-keying it (say, changing the permit
    count) would orphan a whole set of running camera threads, invisibly double-
    processing every sector. Here a re-key always stops the previous grid first.
    """
    signature = (tuple(cam["camera_id"] for cam in cameras), max(1, int(permits)))
    manager = st.session_state.get("manager")

    if manager is None or st.session_state.get("manager_signature") != signature:
        if manager is not None:
            manager.stop_all()
        manager = StreamManager(
            event_bus=EventBus(capacity=4000),
            stall_timeout_s=3.0,
            max_concurrent_analysis=max(1, int(permits)),
            watchdog_interval_s=0.5,
        )
        analysers = {}
        for cam in cameras:
            process_fn = build_analyser(cam, cam.get("fence"), opts)
            analysers[cam["camera_id"]] = process_fn
            manager.add_camera(cam, process_fn=process_fn)
        st.session_state.manager = manager
        st.session_state.analysers = analysers
        st.session_state.manager_signature = signature
    return manager


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

# Persistent Session State Setup
if "logger" not in st.session_state:
    st.session_state.logger = EventLogger(
        csv_path=os.path.join(DEPLOY["data_dir"], "alerts.csv"),
        node_id=DEPLOY["node_id"],
    )

if "tracker" not in st.session_state:
    st.session_state.tracker = CentroidTracker(max_disappeared=20, max_distance=90.0)

if "active_channel" not in st.session_state:
    st.session_state.active_channel = "Channel 1"

if "publisher" not in st.session_state:
    st.session_state.publisher = None

if "last_telemetry" not in st.session_state:
    st.session_state.last_telemetry = None

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

logger = st.session_state.logger
tracker = st.session_state.tracker

# Sidebar Controls
st.sidebar.markdown("### 🛡️ IBVAP Command Station")
st.sidebar.caption("Ministry of Home Affairs — Sashastra Seema Bal")
st.sidebar.markdown("---")

channel_selection = st.sidebar.radio(
    "Surveillance Sector Feed",
    [
        "Channel 1: Sector A - BOP Perimeter (1080p Long-Range)",
        "Channel 2: Sector B - Pedestrian Crossing (vedio-sih26.mp4)",
        "Channel 3: Sector C - Checkpost Charlie (Vehicles & ANPR)",
        "Channel 4: Backup Pre-recorded Demo (Insurance Run)"
    ],
    index=1  # Default to Sector B (vedio-sih26.mp4) to showcase newly added human crossing feed!
)

# Detect Channel Switch and Reset Tracker
current_ch_key = channel_key_from_selection(channel_selection)
if st.session_state.active_channel != current_ch_key:
    st.session_state.active_channel = current_ch_key
    st.session_state.tracker = CentroidTracker(max_disappeared=20, max_distance=90.0)
    tracker = st.session_state.tracker

# Resolve the camera record FIRST: the logger context, the fence calibration and
# the video source routing all read from it. Reading it before it existed raised
# a NameError that py_compile could never catch - it only appeared when the app
# was actually run.
active_camera = get_camera(current_ch_key)

# Attribute every logged event and telemetry packet to the active camera.
logger.set_context(
    node_id=active_camera["node_id"], camera_id=active_camera["camera_id"]
)

# Virtual Fence / Tripwire interactive calibration (bounds come from the
# camera registry so a new outpost only needs a registry entry, not new code).
fence_cfg = active_camera.get("fence")
is_perimeter_mode = fence_cfg is not None
tripwire_y = None

if fence_cfg:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Virtual Fence Calibration")
    tripwire_y = st.sidebar.slider(
        "Tripwire Boundary Line (Y-position)",
        min_value=fence_cfg["slider_min"],
        max_value=fence_cfg["slider_max"],
        value=fence_cfg["default_tripwire_y"],
        step=10,
        help="Calibrate the intrusion tripwire line height for this sector"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🌟 Image Enhancement (Step 2)")
enable_clahe = st.sidebar.checkbox(
    "🌙 Low-Light / Fog CLAHE",
    value=False,
    help="Restores visibility in pitch-black or foggy border feeds using OpenCV CLAHE"
)
clahe_limit = 3.0
if enable_clahe:
    clahe_limit = st.sidebar.slider("Contrast Clip Limit", 1.0, 6.0, 3.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Watchlist Verification (Steps 7 & 8)")
enable_watchlist = st.sidebar.checkbox(
    "🔍 Local Watchlist Search",
    value=True,
    help="Cross-references breach events against the Authorized SSB Patrol & Vehicle Roster"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Stream Telemetry")
conf_thresh = st.sidebar.slider("Detection Sensitivity (Confidence)", 0.25, 0.70, 0.35, 0.05)
frame_stride = st.sidebar.slider("Frame Processing Stride", 1, 3, 1, 1, help="Higher stride yields smoother playback on CPU")

st.sidebar.markdown("---")
st.sidebar.subheader("🧬 Facial Recognition (FRS)")
enable_frs = st.sidebar.checkbox(
    "Enable biometric identity",
    value=True,
    help="Local 1:N face matching against the roster and wanted list. Runs fully on-premise."
)
frs_min_face_px = st.sidebar.slider(
    "Minimum face size (px)", 20, 140, 40, 5,
    help="Faces smaller than this are ignored rather than matched on blur"
)
frs_confirm_frames = st.sidebar.slider(
    "Frames to confirm an identity", 1, 10, 3, 1,
    help="Cross-frame agreement required before an identity is acted on (anti-flicker)"
)
allow_non_biometric = st.sidebar.checkbox(
    "Allow NON-biometric fallback to authorize",
    value=False,
    help="Off by default. If enabled, an appearance descriptor (not face recognition) may produce identity decisions - reported as DEGRADED."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🧍 Behaviour Analytics (Pose)")
enable_pose = st.sidebar.checkbox(
    "Enable behaviour analytics",
    value=bool(pose_engine.available),
    disabled=not pose_engine.available,
    help=("Detects loitering, crawling, falls, fence climbs and group convergence from "
          "body pose. Geometric heuristics, not a trained behaviour classifier.")
)
if not pose_engine.available:
    st.sidebar.caption(f"🔴 {pose_note}")
    st.sidebar.caption("Drop `yolov8n-pose.pt` into `models/` to switch this on.")
else:
    st.sidebar.caption(
        f"🟢 {pose_engine.state()['backend']} • "
        f"{len(pose_engine.state()['behaviors'])} behaviours available"
    )

pose_behaviors = st.sidebar.multiselect(
    "Enabled behaviours",
    BEHAVIOR_ORDER,
    default=list(BEHAVIOR_ORDER),
    format_func=lambda name: BEHAVIOR_ORDER_BY_NAME.get(name, name),
    help="Turn rules off when a sector has recurring, known-good causes of false alarms."
)
pose_sensitivity = st.sidebar.selectbox(
    "Behaviour sensitivity",
    ["Conservative (fewest false alarms)", "Balanced", "Aggressive (catch everything)"],
    index=1,
    help="Scales the behavioural thresholds. Conservative needs longer, cleaner evidence."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ GIS Command View")
gis_cluster_radius_m = st.sidebar.slider(
    "Incident cluster radius (m)", 50, 2000, 300, 50,
    help="Geo-tagged incidents closer than this collapse into one map marker"
)
gis_response_speed = st.sidebar.slider(
    "Response party speed (km/h)", 10, 80, int(RESPONSE_SPEED_KMH), 5,
    help="Ground speed used for the straight-line interception ETA. A LOWER BOUND."
)
gis_show_coverage = st.sidebar.checkbox(
    "Show camera coverage cones", value=True,
    help="Overlapping fans show border actually watched; gaps show where it is not."
)
gis_show_ao = st.sidebar.checkbox(
    "Show area of operations", value=True,
    help="Convex hull of the outposts - the stretch of border this sector owns."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🧵 Pipeline Architecture")
pipeline_mode = st.sidebar.radio(
    "Execution mode",
    ["Threaded grid (multi-camera)", "Inline (legacy single channel)"],
    index=0,
    help="Threaded mode gives every camera a grab thread plus an inference thread, so the dashboard never blocks on the model and several sectors stream at once."
)
show_grid = st.sidebar.checkbox(
    "▦ Show ALL sectors as a 2x2 grid",
    value=False,
    help="Runs every registered camera concurrently - CPU heavy, but it is the full border grid."
)
threaded_refresh_s = st.sidebar.slider(
    "UI refresh interval (s)", 0.2, 3.0, 0.5, 0.1,
    help="How often the threaded view repaints. Independent of camera frame rate."
)
max_permits = st.sidebar.slider(
    "Max concurrent inference workers", 1, 4, 2, 1,
    help="Caps simultaneous model passes so a 15W edge box is never oversubscribed"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Low-Bandwidth Uplink")
enable_telemetry = st.sidebar.checkbox(
    "Enable telemetry transmission",
    value=True,
    help="Transmits structured alerts + compressed evidence crops to Sector HQ. Raw video never leaves the outpost."
)
telemetry_mode = st.sidebar.selectbox(
    "Uplink transport",
    ["simulated", "mqtt"],
    index=0,
    help="'simulated' runs broker-free with a modelled field link; 'mqtt' publishes to a real broker via paho-mqtt."
)
link_profile_name = st.sidebar.selectbox(
    "Field link profile",
    list(LINK_PROFILES.keys()),
    index=2,
    help="Latency and packet-loss characteristics imposed on the uplink"
)
mqtt_host, mqtt_port = "localhost", 1883
if telemetry_mode == "mqtt":
    mqtt_host = st.sidebar.text_input("MQTT broker host", value="localhost")
    mqtt_port = st.sidebar.number_input(
        "MQTT broker port", min_value=1, max_value=65535, value=1883, step=1
    )
payload_budget_kb = st.sidebar.slider(
    "Packet byte budget (KB)",
    min_value=1,
    max_value=20,
    value=int(DEFAULT_MAX_PAYLOAD_BYTES / 1024),
    step=1,
    help="Hard ceiling for one telemetry packet. Evidence crops are auto-shrunk to fit."
)
force_outage = st.sidebar.checkbox(
    "🔌 Simulate total link outage",
    value=False,
    help="Drops the uplink entirely so store-and-forward buffering can be demonstrated"
)

run_stream = st.sidebar.checkbox("▶️ Run Surveillance Feed", value=True)

if st.sidebar.button("🔄 Clear Event Logs & Reset"):
    logger.clear()
    st.session_state.tracker = CentroidTracker(max_disappeared=20, max_distance=90.0)
    st.sidebar.success("Logs and active tracks cleared.")
    st.rerun()

# Build (or re-use) the uplink, then apply the live operator controls.
publisher, telemetry_note = load_telemetry_publisher(
    telemetry_mode,
    mqtt_host,
    int(mqtt_port),
)
st.session_state.publisher = publisher
publisher.set_link_profile(link_profile_name)
publisher.set_budget(int(payload_budget_kb) * 1024)
publisher.set_outage(bool(force_outage))
publisher.flush()

# Apply pose runtime controls to the (cached) behaviour engine. Thresholds are
# applied as a whole named preset, so an operator changes sensitivity rather than
# tuning body-relative constants they have no way to reason about.
pose_engine.configure(
    enabled_behaviors=list(pose_behaviors),
    **vars(preset_thresholds(pose_sensitivity)),
)
pose_state = pose_engine.state()

# Apply FRS runtime controls to the (cached) biometric stack.
frs.allow_non_biometric = bool(allow_non_biometric)
frs.min_face_px = int(frs_min_face_px)
frs.confirm_frames = int(frs_confirm_frames)
frs_state = frs.state()
if frs_state["mode"] == "BIOMETRIC":
    st.sidebar.caption(
        f"🟢 BIOMETRIC • {frs_state['embedder']} ({frs_state['embedding_dim']}-d) • "
        f"{frs_state['index']['faces']} enrolled face(s)"
    )
elif frs_state["mode"] == "DEGRADED_NON_BIOMETRIC":
    st.sidebar.caption("🟠 DEGRADED • non-biometric matching (opt-in)")
else:
    st.sidebar.caption(
        "🔴 FRS UNAVAILABLE • identity falls back to the DEMO roster "
        "(simulated track IDs, not biometrics)"
    )

# Top Header Layout
st.markdown("""
<div class="header-box">
    <div>
        <h1 class="header-title">🛡️ INTELLIGENT BORDER VIDEO ANALYTICS PLATFORM (IBVAP)</h1>
        <p class="header-subtitle">SSB Police II Division • Edge CCTV Computer Vision Network • Problem Statement: SIH26187</p>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
        <span class="badge-status">ONLINE • CPU MODE</span>
        <span class="badge-live">● LIVE FEED</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Metric Cards (Using dynamic placeholders to eliminate visual jitter)
c1, c2, c3, c4, c5 = st.columns(5)
ph_metric1 = c1.empty()
ph_metric2 = c2.empty()
ph_metric3 = c3.empty()
ph_metric4 = c4.empty()
ph_metric5 = c5.empty()

# Initial KPI values
initial_stats = logger.get_stats()
init_df = logger.get_dataframe()
auth_count = len(init_df[init_df["event_type"].str.contains("AUTHORIZED", na=False)]) if not init_df.empty else 0

ph_metric1.metric("🚨 Intrusion Alerts", initial_stats["intrusions"])
ph_metric2.metric("✅ Authorized Patrols", auth_count)
ph_metric3.metric("🚗 Vehicles Tracked", initial_stats["vehicles_detected"])
ph_metric4.metric("🪪 Verified Plates Read", initial_stats["plates_read"])
ph_metric5.metric("🧍 Behaviour Alerts", initial_stats["behavior_alerts"])

# Tab Layout: Live / Logs / Uplink Telemetry / Biometrics / Roadmap
tab_live, tab_gis, tab_pose, tab_logs, tab_link, tab_frs, tab_roadmap = st.tabs([
    "📺 Live Surveillance Post",
    "🗺️ GIS Command Map",
    "🧍 Behaviour Analytics",
    "📊 Event Log & Analytics",
    "📡 Low-Bandwidth Uplink",
    "🧬 FRS & Identity",
    "🚀 SSB Operational Roadmap (Future Work)"
])

# -------------------------------------------------------------
# TAB 1: LIVE SURVEILLANCE POST
# -------------------------------------------------------------
with tab_live:
    col_player, col_feed = st.columns([3, 2])

    with col_player:
        st.markdown(f"**📹 Sector Video Stream:** `{channel_selection.split(':')[1].strip()}`")
        # Every video surface is created ONCE, here, and repainted by whatever
        # engine is running - the threaded grid, the inline loop, or a fragment.
        # A fragment that creates its own placeholders paints a SECOND copy of
        # the feed further down the page and leaves these frozen, so the layout
        # must not depend on which engine is selected.
        camera = active_camera
        grid_cameras = [get_camera(key) for key in CAMERAS] if show_grid else [camera]
        ph_banner = st.empty()
        ph_video = st.empty()
        # A fragment may only paint into a container that was ALREADY written to
        # during the full script run - Streamlit needs a reserved position for it
        # (otherwise: StreamlitInvalidLayoutContextError). So claim every slot
        # here, with the honest "nothing yet" state, before the fragment runs.
        ph_banner.caption("Monitoring sector feed…")
        ph_video.info("Initialising sector video pipeline…")
        tile_slots = {}
        if show_grid:
            tile_columns = st.columns(2)
            for index, grid_camera in enumerate(grid_cameras):
                with tile_columns[index % 2]:
                    st.markdown(f"**{grid_camera['label']}**")
                    tile_slots[grid_camera["camera_id"]] = st.empty()
                    tile_slots[grid_camera["camera_id"]].info("Awaiting first frame…")

    with col_feed:
        st.markdown("**🚨 Real-Time Security Incident Stream**")
        ph_alerts = st.empty()
        ph_alerts.caption("No incidents logged yet.")

    # Route video source and fence parameters from the camera registry.
    video_path = resolve_video_source(camera)
    metric_slots = [ph_metric1, ph_metric2, ph_metric3, ph_metric4, ph_metric5]
    # Claim the KPI slots before the fragment runs (they are painted again below).
    paint_metrics(metric_slots)

    # Operator controls live in a LONG-LIVED dict, so live changes reach running
    # analysers without tearing down the camera grid or restarting threads.
    analyser_opts = st.session_state.setdefault("analyser_opts", {})
    analyser_opts.update({
        "conf_thresh": conf_thresh,
        "enable_clahe": enable_clahe,
        "clahe_limit": clahe_limit,
        "enable_watchlist": enable_watchlist,
        "enable_frs": enable_frs,
        "enable_pose": bool(enable_pose and pose_engine.available),
        "tripwire_y": tripwire_y,
    })

    if pipeline_mode.startswith("Threaded"):
        # ==================================================================
        # THREADED MULTI-CAMERA PATH: ingest + inference happen in background
        # threads; this block only paints whatever is newest.
        # ==================================================================
        manager = ensure_stream_manager(grid_cameras, analyser_opts, max_permits)

        # Live tripwire retune: move the line on the RUNNING analyser rather than
        # rebuilding the grid, so calibration feels instant during a demo.
        if fence_cfg and tripwire_y is not None:
            live_analyser = st.session_state.get("analysers", {}).get(camera["camera_id"])
            live_fence = getattr(live_analyser, "fence", None)
            if live_fence is not None:
                x_start, x_end = fence_cfg["tripwire_x_range"]
                live_fence.update_line((x_start, tripwire_y), (x_end, tripwire_y))

        cameras_by_id = {cam["camera_id"]: cam for cam in grid_cameras}

        def resolve_event_target(event):
            """Attributes an alert to its OWN camera, with that camera's frame."""
            event_camera = cameras_by_id.get(event.get("camera_id"), camera)
            return event_camera, manager.latest_frame(event_camera["camera_id"])

        def render_threaded_view():
            # 1. Drain everything the analysis threads produced since the last paint.
            new_events = manager.drain_events(max_items=500)
            consume_events(new_events, enable_telemetry, resolve_event_target)

            # 2. Paint the video surface(s) INTO the placeholders the main script
            #    created, so the feed appears where the operator expects it.
            if show_grid:
                for grid_camera in grid_cameras:
                    slot = tile_slots.get(grid_camera["camera_id"])
                    if slot is None:
                        continue
                    tile_frame = (
                        manager.peek_annotated(grid_camera["camera_id"])
                        or manager.latest_frame(grid_camera["camera_id"])
                    )
                    if tile_frame is not None:
                        slot.image(
                            cv2.cvtColor(tile_frame, cv2.COLOR_BGR2RGB), width="stretch"
                        )
                    else:
                        slot.info("Awaiting first frame…")
            else:
                camera_id = camera["camera_id"]
                live_frame = manager.latest_annotated(camera_id)
                if live_frame is None:
                    live_frame = manager.peek_annotated(camera_id)
                if live_frame is None:
                    # Inference on a CPU edge box is slower than this repaint, so an
                    # empty buffer is NORMAL, not a disconnected camera. Holding the
                    # last analysed frame keeps the wall display continuous - the
                    # alternative blanked a healthy feed on every other cycle.
                    if st.session_state.get("last_video_camera") == camera_id:
                        live_frame = st.session_state.get("last_video_frame")

                if live_frame is not None:
                    st.session_state.last_frame = live_frame
                    st.session_state.last_video_frame = live_frame
                    st.session_state.last_video_camera = camera_id
                    ph_video.image(
                        cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB), width="stretch"
                    )
                else:
                    # Status lives in the video slot it describes, and painting an
                    # element here also claims the slot for later fragment reruns.
                    ph_video.info("Connecting to camera feed…")

            # 3. Refresh KPIs, banner and incident feed - in place, so the header
            #    cards and the incident column track the live video instead of
            #    lagging at whatever they held when the script last fully ran.
            refresh_dashboard(metric_slots, ph_banner, ph_alerts)

            # 4. Grid health: a stalled camera must be visible, not silent.
            with st.expander("📷 Ingest Grid Health — threads, streams, analysis"):
                st.dataframe(
                    pd.DataFrame(manager.health()),
                    width="stretch",
                    hide_index=True,
                )
                grid_stats = manager.summary()
                st.caption(
                    f"{grid_stats['cameras']} camera(s) • {grid_stats['live']} live • "
                    f"{grid_stats['stalled']} stalled • "
                    f"{grid_stats['reconnecting']} reconnecting • "
                    f"{grid_stats['frames_in']} frames grabbed / "
                    f"{grid_stats['frames_analysed']} analysed • "
                    f"{grid_stats['events_dropped']} events shed • "
                    f"{grid_stats['watchdog_actions']} watchdog recovery action(s) • "
                    f"{grid_stats['analysis_permits']} inference permit(s)"
                )

        if run_stream:
            if hasattr(st, "fragment"):
                @st.fragment(run_every=f"{threaded_refresh_s}s")
                def live_surveillance_fragment():
                    render_threaded_view()

                live_surveillance_fragment()
            else:
                # Streamlit too old for fragments: bounded paint loop, then re-run.
                cycles = max(1, int(10 / max(0.1, threaded_refresh_s)))
                for _ in range(cycles):
                    render_threaded_view()
                    time.sleep(threaded_refresh_s)
                st.rerun()
            st.caption(
                "Threaded mode: each camera owns a grab thread (I/O) and an analysis thread "
                "(inference). The view repaints on a timer, so a slow model can never stall "
                "the feed - and skipped frames stay skipped instead of piling up as lag."
            )
        else:
            st.info("Surveillance feed paused. Check 'Run Surveillance Feed' in the sidebar to resume.")

        # Keep the top-of-page KPI cards in step on every full script run.
        paint_metrics(metric_slots)

    use_inline_pipeline = not pipeline_mode.startswith("Threaded")

    if use_inline_pipeline:
        # ==================================================================
        # LEGACY INLINE PATH: single channel, inference inside the render loop.
        # Kept as a fallback and for the pre-rendered insurance demo.
        # ==================================================================
        fence = None
        if fence_cfg:
            y_pos = tripwire_y if tripwire_y is not None else fence_cfg["default_tripwire_y"]
            x_start, x_end = fence_cfg["tripwire_x_range"]
            fence = VirtualFence(
                line_coords=((x_start, y_pos), (x_end, y_pos)),
                zone_name=fence_cfg["zone_name"]
            )
        is_perimeter_mode = fence is not None

    # Execute Live Stream Loop (inline pipeline only)
    if use_inline_pipeline and run_stream and os.path.exists(video_path):
        # Built once per loop run so track IDs persist for the whole playback.
        analyser = build_analyser(
            camera, fence_cfg, analyser_opts, tracker=st.session_state.tracker
        )
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while cap.isOpened() and run_stream:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_idx += 1
            if frame_idx % frame_stride != 0:
                continue

            now_str = datetime.now().strftime("%H:%M:%S")

            # One shared vision pipeline (identical to the threaded grid's), so
            # detection / fence / ANPR logic exists in exactly one place.
            annotated_frame, events = analyser(
                frame,
                {
                    "frame_seq": frame_idx,
                    "frame_ts": time.time(),
                    "timestamp": now_str,
                    "camera_id": camera["camera_id"],
                },
            )
            consume_events(
                events,
                enable_telemetry,
                lambda event, _cam=camera, _frame=frame: (_cam, _frame),
            )
            if camera.get("pre_rendered"):
                time.sleep(0.02)  # pace pre-rendered playback for the demo

            # Keep the latest processed frame so the uplink can attach real
            # evidence crops on demand (reference only, no copy per frame).
            st.session_state.last_frame = annotated_frame

            # Render to Streamlit Display
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            ph_video.image(rgb_frame, width="stretch")

            # Update Metrics cleanly in placeholders
            stats = logger.get_stats()
            df = logger.get_dataframe()
            auth_count = len(df[df["event_type"].str.contains("AUTHORIZED", na=False)]) if not df.empty else 0

            ph_metric1.metric("🚨 Intrusion Alerts", stats["intrusions"])
            ph_metric2.metric("✅ Authorized Patrols", auth_count)
            ph_metric3.metric("🚗 Vehicles Tracked", stats["vehicles_detected"])
            ph_metric4.metric("🪪 Verified Plates Read", stats["plates_read"])

            # Update Dynamic Alert Banner & Recent Incident Feed
            if not df.empty:
                last_event = df.iloc[-1]
                if last_event["event_type"] == "AUTHORIZED_PATROL" or last_event["event_type"] == "AUTHORIZED_VEHICLE":
                    ph_banner.success(f"✅ MATCH FOUND (Authorized Patrol): {last_event['details']} • False Alarm Suppressed")
                elif last_event["event_type"] == "INTRUSION_ALERT":
                    ph_banner.error(f"🚨 NO MATCH (Unknown Intruder): Track #{last_event['track_id']} Breached Perimeter! • Security Dispatched")

                recent_df = df.tail(7)[["timestamp", "event_type", "track_id", "status", "details"]]
                ph_alerts.dataframe(
                    recent_df,
                    width="stretch",
                    hide_index=True
                )

        cap.release()

    elif use_inline_pipeline and not os.path.exists(video_path):
        st.error(f"Target video feed not located: {video_path}")
    elif use_inline_pipeline:
        st.info("Surveillance feed paused. Check 'Run Surveillance Feed' in the sidebar to resume.")

# -------------------------------------------------------------
# TAB 2: GIS COMMAND MAP
# -------------------------------------------------------------
with tab_gis:
    st.subheader("🗺️ Sector Geographic Command Picture")
    st.caption(
        "Rendered completely offline as inline SVG - no basemap tiles, no CDN, no API key - "
        "so it still works at an air-gapped outpost with the satellite link down."
    )

    # Live link health comes from the ingest grid, when it is running. Without it
    # the map shows UNKNOWN rather than inventing a healthy sector.
    live_manager = st.session_state.get("manager")
    state_by_camera = {}
    if live_manager is not None:
        try:
            state_by_camera = {
                row["camera_id"]: row.get("state", "UNKNOWN")
                for row in live_manager.health()
            }
        except Exception:
            state_by_camera = {}

    log_df = logger.get_dataframe()
    log_records = log_df.to_dict("records") if not log_df.empty else []
    snapshot = gis.gis_snapshot(
        log_records,
        CAMERAS,
        outposts=OUTPOSTS,
        state_by_camera=state_by_camera,
        cluster_radius_m=float(gis_cluster_radius_m),
        response_speed_kmh=float(gis_response_speed),
        severity_for=severity_for,
        alert_limit=400,
    )
    summary = snapshot["summary"]

    g1, g2, g3, g4 = st.columns(4)
    online = sum(1 for n in snapshot["nodes"] if n["state"] == "ONLINE")
    g1.metric("🛡️ Outposts on the map", f"{online}/{snapshot['outpost_count']}")
    g2.metric("📍 Incident clusters", summary["clusters"])
    g3.metric("🔴 Critical clusters", summary["critical_clusters"])
    g4.metric("⚠️ Coverage gaps", snapshot["coverage_gaps"])

    components.html(
        gis.render_tactical_map(
            snapshot["nodes"],
            snapshot["alerts"],
            snapshot["clusters"],
            width=1080,
            height=620,
            title="Sector Tactical Picture",
            cluster_radius_m=float(gis_cluster_radius_m),
            show_coverage=bool(gis_show_coverage),
            show_ao=bool(gis_show_ao),
        ),
        # Tall enough for the map plus the per-outpost coverage insets. The iframe
        # does not scroll, so an off-by-a-little height would hide the feature.
        height=1000,
        scrolling=False,
    )

    if snapshot["unresolved_records"]:
        st.warning(
            f"{snapshot['unresolved_records']} of {snapshot['records_considered']} log records "
            "could not be geo-located (no camera of that ID in the registry). They are excluded "
            "from the picture rather than silently plotted at the wrong place."
        )
    if snapshot["coverage_gaps"]:
        st.warning(
            f"⚠️ {snapshot['coverage_gaps']} incident(s) fall OUTSIDE every camera coverage cone. "
            "Those are surveillance gaps in the perimeter, not just alerts."
        )

    # ---- Interception planning ----------------------------------------------
    st.markdown("#### 🚔 Interception Recommendation")
    intercept = snapshot["intercept"]
    if not intercept:
        st.info("No geo-located incident yet. Run a sector feed to populate the response picture.")
    else:
        st.success(intercept["sentence"])
        st.caption(
            f"Excludes road network: {float(gis_response_speed):.0f} km/h ground speed gives an "
            "optimistic ETA. Treat it as a floor for dispatch planning, never as a promise."
        )
        rank_df = pd.DataFrame(intercept["ranking"])
        if not rank_df.empty:
            st.dataframe(
                rank_df[[
                    "name", "distance_km", "compass", "bearing_deg", "eta_text", "strength"
                ]].rename(columns={
                    "name": "Outpost", "distance_km": "Distance (km)",
                    "compass": "Bearing", "bearing_deg": "Bearing (deg)",
                    "eta_text": "ETA (lower bound)", "strength": "Held strength",
                }),
                width="stretch",
                hide_index=True,
            )

    # ---- Outpost table, coverage and relay distances -------------------------
    gis_left, gis_right = st.columns(2)
    with gis_left:
        st.markdown("#### 🛡️ Outpost Status")
        outpost_rows = []
        for node in snapshot["nodes"]:
            outpost_rows.append({
                "Outpost": node["node_id"],
                "Name": node["name"],
                "Sector": node["sector"],
                "Link": node["state"],
                "Cameras": len(node["cameras"]),
                "Coverage (km)": node["coverage_range_km"],
                "Held strength": node.get("strength", "-"),
            })
        st.dataframe(pd.DataFrame(outpost_rows), width="stretch", hide_index=True)

    with gis_right:
        st.markdown("#### 📡 Outpost Relay Distances")
        matrix = snapshot["distance_matrix"]
        if matrix:
            st.dataframe(
                pd.DataFrame(matrix)[["from", "to", "distance_km", "bearing_deg"]].rename(
                    columns={"from": "From", "to": "To", "distance_km": "Distance (km)",
                             "bearing_deg": "Bearing (deg)"}
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("A single outpost has no relay distances to report.")

    # ---- Standards-compliant export ----------------------------------------
    st.markdown("#### 📤 Export to Sector HQ (GeoJSON)")
    st.caption(
        "The same picture, in the format a staff map already understands. Axis order is "
        "[longitude, latitude] per RFC 7946, so it opens correctly in QGIS or any GIS tool."
    )
    ex1, ex2, ex3 = st.columns(3)
    ex1.download_button(
        "⬇️ Outposts",
        data=gis.dumps_geojson(gis.nodes_geojson(snapshot["nodes"])),
        file_name="ibvap_outposts.geojson",
        mime="application/geo+json",
        width="stretch",
    )
    ex2.download_button(
        "⬇️ Coverage cones",
        data=gis.dumps_geojson(gis.coverage_geojson(snapshot["coverage_cones"])),
        file_name="ibvap_coverage.geojson",
        mime="application/geo+json",
        width="stretch",
    )
    ex3.download_button(
        "⬇️ Incidents & clusters",
        data=gis.dumps_geojson(gis.alerts_geojson(snapshot["alerts"], snapshot["clusters"])),
        file_name="ibvap_incidents.geojson",
        mime="application/geo+json",
        width="stretch",
    )

# -------------------------------------------------------------
# TAB 3: BEHAVIOUR ANALYTICS (POSE)
# -------------------------------------------------------------
with tab_pose:
    st.subheader("🧍 Suspicious-Behaviour Analytics")
    st.caption(
        "Derived from body pose, not from a line crossing: a person lying still, crawling "
        "low under the tripwire or stopping dead at the fence are all invisible to a "
        "tripwire and all worth knowing about."
    )

    pose_live = pose_engine.state()
    if pose_live["mode"] != "ACTIVE":
        st.error(
            "🔴 Behaviour analytics is OFFLINE - no pose backend loaded, so NOTHING is being "
            f"inferred. Reason: {pose_live['note']}"
        )
        st.info(
            "To enable: install `ultralytics` and place a pose model at "
            "`models/yolov8n-pose.pt` (or set `BEHAVIORS` to use another keypoint model). "
            "Nothing is ever reported while this is offline - the platform does not guess."
        )
    else:
        st.success(
            f"🟢 ACTIVE via {pose_live['backend']} • {pose_live['frames_analysed']} frames "
            f"analysed • {pose_live['people_detected']} posed figures • "
            f"{pose_live['last_latency_ms']} ms/frame"
        )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("🧍 Behaviour alerts", pose_live["analyser"]["events_emitted"])
    p2.metric("🎯 Tracks analysed", pose_live["analyser"]["active_tracks"])
    p3.metric("🔁 Suppressed (cooldown)", pose_live["analyser"]["suppressed_by_cooldown"])
    p4.metric("❓ Unmatched poses", pose_live["unmatched_people"])

    st.markdown("#### 📖 Behaviour Catalogue")
    catalogue_rows = []
    for entry in pose_live["behaviors"]:
        catalogue_rows.append({
            "Behaviour": entry["label"],
            "Key": entry["behavior"],
            "Severity": entry["severity"],
            "Enabled": "✅" if entry["enabled"] else "⏸️",
            "Raised": entry["count"],
            "What it means": entry["description"],
        })
    st.dataframe(pd.DataFrame(catalogue_rows), width="stretch", hide_index=True)

    st.markdown("#### 📈 Live Body Metrics (per track)")
    metrics_rows = pose_engine.analyser.track_metrics()
    if metrics_rows:
        st.dataframe(
            pd.DataFrame(metrics_rows).rename(columns={
                "track_id": "Track",
                "dwell_s": "In window (s)",
                "speed_hps": "Speed (body-heights/s)",
                "spread_ratio": "Dwell spread (x height)",
                "torso_angle_deg": "Torso from vertical (deg)",
                "aspect": "Box aspect (w/h)",
                "height_px": "Body height (px)",
                "last_seen_s": "Last seen (s ago)",
            }),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Every threshold is expressed in units of the person's own body height, so one "
            "calibration holds for a figure 4 m from the camera and one 40 m away."
        )
    else:
        st.caption("No posed tracks yet in this session.")

    st.markdown("#### 🚨 Behaviour Alert History")
    behaviour_df = logger.get_dataframe()
    if not behaviour_df.empty:
        behaviour_only = behaviour_df[
            behaviour_df["event_type"].isin(BEHAVIOR_EVENT_TYPES)
        ]
        if not behaviour_only.empty:
            st.dataframe(
                behaviour_only.tail(15)[[
                    "timestamp", "event_type", "track_id", "status", "details"
                ]].rename(columns={
                    "timestamp": "Time", "event_type": "Behaviour", "track_id": "Track",
                    "status": "Severity", "details": "Evidence",
                }),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No behaviour alerts recorded yet.")
    else:
        st.caption("No events logged yet.")

    if pose_live.get("last_error"):
        st.warning(f"Last analytics error (contained, the video kept running): {pose_live['last_error']}")

    st.markdown("#### ⚖️ What this is, and what it is not")
    st.markdown(
        """
- **It is** geometry on keypoints: torso inclination, limb-normalised speed, foot-point
  dwell spread and hip rise/drop, each confirmed over a time window so one noisy frame
  cannot raise an alert.
- **It is not** a trained behaviour classifier, and every alert says so: the audit record
  carries `POSE_HEURISTIC` provenance and the event is never written as a fact.
- **No liveness / anti-spoofing.** A printed photograph held up to the camera is not yet
  detected. That is a known limitation, listed in the roadmap rather than implied away.
- **Thresholds are intended to be tuned per sector.** Use the sensitivity control in the
  sidebar; a border post that cries wolf gets switched off by its own operators, which is
  the real failure mode of behaviour analytics.
        """
    )

# -------------------------------------------------------------
# TAB 4: HISTORICAL LOGS & AUDIT TRAIL
# -------------------------------------------------------------
with tab_logs:
    st.subheader("📋 Comprehensive Surveillance Incident Register")
    full_df = logger.get_dataframe()

    if not full_df.empty:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            event_filter = st.multiselect(
                "Filter by Event Type",
                options=full_df["event_type"].unique().tolist(),
                default=full_df["event_type"].unique().tolist()
            )
        with col_f2:
            status_filter = st.multiselect(
                "Filter by Verification Status",
                options=full_df["status"].unique().tolist(),
                default=full_df["status"].unique().tolist()
            )

        filtered_df = full_df[
            (full_df["event_type"].isin(event_filter)) &
            (full_df["status"].isin(status_filter))
        ]

        st.dataframe(filtered_df, width="stretch", hide_index=True)

        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Exported Incident CSV",
            data=csv_bytes,
            file_name=f"ssb_ibvap_incident_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No incident records currently logged in this session.")

    # Step 7: Watchlist Database Viewer
    st.markdown("---")
    with st.expander("🗂️ Step 7: Local Offline Watchlist Database (FAISS / SQLite Architecture)", expanded=True):
        st.caption("Sub-millisecond local offline matching for authorized border personnel and official patrol vehicles.")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown("##### 👮 Authorized Border Patrol Roster (ArcFace / Face Vector)")
            st.dataframe(watchlist.get_personnel_dataframe(), width="stretch", hide_index=True)
        with col_w2:
            st.markdown("##### 🚙 Authorized Patrol & Logistics Vehicles (ANPR Plates)")
            st.dataframe(watchlist.get_vehicles_dataframe(), width="stretch", hide_index=True)

# -------------------------------------------------------------
# TAB 5: SSB OPERATIONAL ROADMAP (PRESENTATION SLIDES)
# -------------------------------------------------------------
with tab_roadmap:
    st.subheader("🚀 Operational Architecture & Future Scaling Roadmap")
    st.caption("Strategic expansion roadmap for border deployment as specified in Problem Statement SIH26187.")

    st.markdown("""
    <div class="roadmap-card">
        <div class="roadmap-title">1. Thermal & Low-Light / Night-Vision Integration (NIR / LWIR)</div>
        <div class="roadmap-desc">
            <b>SSB Requirement:</b> Surveillance across unlit riverine borders and dense foliage at night.<br>
            <b>Architecture:</b> Integration of FLIR / Long-Wave Infrared thermal video feeds. YOLOv8 fine-tuned on multispectral (FLIR/KAIST) datasets allows zero-light human heat-signature detection without requiring active illuminators.
        </div>
    </div>
    
    <div class="roadmap-card">
        <div class="roadmap-title">2. Edge Computing Architecture (NVIDIA Jetson AGX / Orin Nano)</div>
        <div class="roadmap-desc">
            <b>SSB Requirement:</b> High reliability in remote Border Out Posts (BOPs) with limited or intermittent satellite backhaul.<br>
            <b>Architecture:</b> TensorRT-compiled INT8 models running locally on low-power ruggedized edge boxes (15W). Video processing occurs 100% on-premise; only lightweight encrypted telemetry alerts (< 2 KB) are transmitted to Sector HQ.
        </div>
    </div>
    
    <div class="roadmap-card">
        <div class="roadmap-title">3. Facial Recognition System (FRS) & Watchlist Matching</div>
        <div class="roadmap-desc">
            <b>SSB Requirement:</b> Intercepting known suspects, cross-border smugglers, and persons of interest.<br>
            <b>Architecture:</b> Cascaded face detector (RetinaFace/YOLOv8-Face) triggered upon pedestrian boundary approach. Feature embeddings extracted using lightweight ArcFace (MobileFaceNet) matched against local encrypted SQLite/Milvus vector index in < 15ms.
        </div>
    </div>
    
    <div class="roadmap-card">
        <div class="roadmap-title">4. Multi-Camera Spatial Fusion & Cross-Camera Re-Identification (Re-ID)</div>
        <div class="roadmap-desc">
            <b>SSB Requirement:</b> Tracking a suspect traversing between multiple perimeter CCTV towers and road checkpoints.<br>
            <b>Architecture:</b> Deep visual appearance embeddings (OSNet) associated across overlapping and non-overlapping camera fields of view to construct a unified 3D spatial track trajectory on an interactive map.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 6: FACIAL RECOGNITION & IDENTITY
# -------------------------------------------------------------
with tab_frs:
    st.subheader("🧬 Facial Recognition & Identity Verification")
    st.caption(
        "Local 1:N biometric matching against the SSB roster and the wanted list. "
        "Inference is entirely on-premise: no face image or embedding ever leaves the outpost."
    )

    frs_state = frs.state()
    index_stats = frs_state["index"]

    # State the truth FIRST - a viewer must know what is actually running before
    # reading any identity result. This is the line between a demo and a system.
    if frs_state["mode"] == "BIOMETRIC":
        st.success(
            f"✅ BIOMETRIC MODE — detector `{frs_state['detector']}` + embedder "
            f"`{frs_state['embedder']}` ({frs_state['embedding_dim']}-d), "
            f"fully offline index at `{index_stats['db_path']}`"
        )
    elif frs_state["mode"] == "DEGRADED_NON_BIOMETRIC":
        st.warning(
            "⚠️ DEGRADED MODE — identity is matched with a NON-BIOMETRIC appearance "
            "descriptor (no face-recognition model installed). Authorizations must "
            "not be treated as identity confirmation."
        )
    else:
        st.error(
            f"⛔ FRS UNAVAILABLE — {frs_state['reason']}. Identity currently falls back "
            "to the DEMO roster (simulated track IDs), which is NOT biometrics."
        )
        st.caption(frs_note)

    f1, f2, f3, f4, f5, f6 = st.columns(6)
    f1.metric("Enrolled Identities", index_stats["identities"])
    f2.metric("Enrolled Faces", index_stats["faces"])
    f3.metric("Embedding Dim", index_stats["dim"] or "—")
    f4.metric("Match Threshold", f"{index_stats['match_threshold']:.2f}")
    f5.metric("Avg Search", f"{index_stats['avg_search_ms']:.2f} ms")
    f6.metric("Decisions Made", frs_state["decisions_made"])

    st.markdown("---")
    st.markdown("#### 🎚️ Roster Separability & Threshold Calibration")
    capacity = frs.index.capacity_report()
    headroom = capacity["headroom"]
    percentile = capacity["stats"]["percentile"]
    if capacity["verdict"] == "TIGHT":
        st.warning(
            f"⚠️ TIGHT — headroom {headroom} (threshold {capacity['match_threshold']:.2f} "
            f"vs 99.9th-pct impostor similarity {percentile}). {capacity['advice']}"
        )
    else:
        st.info(
            f"✅ OK — headroom {headroom} between the match threshold "
            f"({capacity['match_threshold']:.2f}) and the 99.9th-pct impostor "
            f"similarity ({percentile}) across {capacity['stats']['pairs']} sampled pairs."
        )

    if st.button("🎚️ Calibrate threshold from roster statistics"):
        result = frs.index.calibrate()
        st.success(
            f"Threshold {result['before']:.2f} → {result['after']:.2f} "
            f"(review band {result['review_threshold']:.2f}). Calibration only ever "
            f"raises the gate, never loosens it below {result['floor']:.2f}."
        )
        st.caption(
            "Raise it when the roster has look-alikes or the embedding space is tight; "
            "the false-accept rate falls, at the cost of recall."
        )

    st.markdown("---")
    st.markdown("#### 👤 Enrolled Biometric Identities")
    biometric_df = watchlist.get_biometric_dataframe()
    if biometric_df.empty:
        st.info(
            "No biometric identities enrolled yet — enroll from the live feed below, "
            "or rely on the DEMO roster (clearly labelled as simulated)."
        )
    else:
        st.dataframe(biometric_df, width="stretch", hide_index=True)

    st.markdown("#### ➕ Enroll a Face From the Current Feed")
    enrollment_frame = st.session_state.last_frame
    if enrollment_frame is None:
        st.info(
            "Run a surveillance feed first — enrollment captures a face from the "
            "most recent processed frame."
        )
    else:
        col_face, col_form = st.columns([1, 2])
        with col_face:
            try:
                st.image(
                    cv2.cvtColor(enrollment_frame, cv2.COLOR_BGR2RGB),
                    caption="Frame used for enrollment",
                    width="stretch",
                )
            except Exception:
                st.caption("Frame preview unavailable.")
        with col_form:
            enroll_id = st.text_input("Service ID / Person ID", value="SSB-")
            enroll_name = st.text_input("Name", value="")
            enroll_role = st.selectbox(
                "Roster role",
                ["AUTHORIZED_PERSONNEL", "WATCHLIST"],
                help="WATCHLIST subjects raise a CRITICAL WATCHLIST_HIT on perimeter breach",
            )
            enroll_unit = st.text_input("Unit / Notes", value="")
            if st.button("📸 Capture & enroll largest face"):
                capture = frs.enroll_face(enrollment_frame)
                if capture is None:
                    st.error(
                        "No face found in the current frame (or the face is below the "
                        "minimum size). Move closer to the camera and retry."
                    )
                else:
                    if not capture["biometric"]:
                        st.warning(
                            "Storing a NON-BIOMETRIC descriptor — usable for demo "
                            "matching only."
                        )
                    stored = frs.index.add(
                        capture["embedding"],
                        person_id=enroll_id or "UNSPECIFIED",
                        name=enroll_name,
                        role=enroll_role,
                        unit=enroll_unit,
                        notes=capture["source"],
                    )
                    st.success(
                        f"Enrolled `{enroll_id or 'UNSPECIFIED'}` as face `{stored}` "
                        f"({capture['source']}, box {capture['face_bbox']})."
                    )
                    st.rerun()

    st.markdown("---")
    st.markdown("#### 🕵️ Live Identity Cache — Per Tracked Person")
    cached = list(frs.track_cache.items())[-25:]
    if cached:
        cache_rows = [
            {
                "Track": track_id,
                "Decision": entry.get("decision"),
                "Best cos": round(float(entry.get("best_similarity", 0.0)), 3),
                "Frames": entry.get("frames_seen"),
                "Confirmed": bool(entry.get("promoted")),
                "Identity": (entry.get("best_record") or {}).get("name", ""),
                "Person ID": (entry.get("best_record") or {}).get("person_id", ""),
            }
            for track_id, entry in cached
        ]
        st.dataframe(pd.DataFrame(cache_rows), width="stretch", hide_index=True)
    else:
        st.info("No identities tracked yet in this session.")

    reset_col, note_col = st.columns([1, 3])
    with reset_col:
        if st.button("🧹 Reset identity cache"):
            cleared = frs.purge_tracks()
            st.success(f"{cleared} cached track identit(ies) cleared.")
    with note_col:
        st.caption(
            "An identity is only acted on after cross-frame agreement (or a single "
            "high-confidence read). Unconfirmed matches are surfaced as REVIEW for a "
            "human decision, never auto-authorized."
        )

    with st.expander("⚙️ Loaded Backend Detail & Diagnostics", expanded=False):
        st.json(frs_state)

# -------------------------------------------------------------
# TAB 7: LOW-BANDWIDTH TELEMETRY UPLINK (SECTOR HQ VIEW)
# -------------------------------------------------------------
with tab_link:
    st.subheader("📡 Sector HQ Uplink — Low-Bandwidth Telemetry")
    st.caption(
        "Every detection leaves the outpost as ONE structured packet: alert metadata plus a compressed "
        "evidence crop. Raw video never crosses the link."
    )

    tstats = publisher.stats()
    link_state = "🟢 LINK ONLINE" if tstats["online"] else "🔴 LINK OUTAGE — STORE-AND-FORWARD ACTIVE"
    st.markdown(
        f"**Uplink:** {link_state} &nbsp;|&nbsp; **Transport:** `{tstats['transport']}` "
        f"&nbsp;|&nbsp; **Link:** {tstats['link_label']} "
        f"&nbsp;|&nbsp; **Budget:** {tstats['max_payload_bytes']} B per packet "
        f"(metadata floor {tstats['metadata_floor_bytes']} B)"
    )
    st.info(telemetry_note)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Packets Delivered", tstats["sent"])
    k2.metric("Avg Packet Size", f"{tstats['avg_payload_bytes']:.0f} B")
    k3.metric("Budget Used", f"{tstats['budget_utilisation']}%")
    k4.metric("Buffered", tstats["buffered"])
    k5.metric("Retransmitted", tstats["retransmissions"])
    k6.metric("Dropped", tstats["dropped"])

    ctl1, ctl2 = st.columns([1, 1])
    with ctl1:
        if st.button("🚨 Transmit Test Alert", width="stretch"):
            test_frame = st.session_state.last_frame
            test_bbox = None
            if test_frame is not None:
                fh, fw = test_frame.shape[:2]
                test_bbox = (int(fw * 0.42), int(fh * 0.32), int(fw * 0.58), int(fh * 0.86))
            st.session_state.last_telemetry = publisher.publish_event(
                {
                    "event_type": "INTRUSION_ALERT",
                    "track_id": 999,
                    "category": "human",
                    "class_name": "person",
                    "confidence": 0.91,
                    "status": "UNKNOWN_INTRUDER",
                    "zone": "Manual Uplink Test",
                    "direction": "INBOUND (Southbound)",
                    "identity": "UNKNOWN PERSON",
                    "location": "(640, 360)",
                },
                camera,
                frame=test_frame,
                bbox=test_bbox,
            )
            st.rerun()
    with ctl2:
        if st.button("📤 Retransmit Backlog Now", width="stretch"):
            drained = publisher.drain_all()
            st.success(f"{drained} buffered packet(s) retransmitted to Sector HQ.")

    last_result = st.session_state.last_telemetry
    if last_result:
        st.markdown(
            f"**Last packet:** `{last_result['event_id']}` → {last_result['status']} "
            f"| {last_result['bytes']} B (snapshot {last_result['snapshot_bytes']} B) "
            f"| topic `{last_result['topic']}` | backlog {last_result['queue_depth']}"
        )
        st.caption(
            "Tip: enable 'Simulate total link outage' in the sidebar, fire several alerts, then clear the "
            "toggle to watch the store-and-forward backlog drain in order."
        )

    st.markdown("---")
    st.markdown("#### 🛰️ What Sector HQ Actually Receives")
    transport_inner = getattr(publisher.transport, "inner", None)
    sent_packets = list(getattr(transport_inner, "sent", []))
    if sent_packets:
        newest = sent_packets[-1]
        try:
            decoded_packet = json.loads(newest["payload"].decode("utf-8"))
        except Exception:
            decoded_packet = None

        if decoded_packet:
            snap = decoded_packet.get("snapshot") or {}
            col_img, col_json = st.columns([1, 2])
            with col_img:
                if snap.get("data"):
                    try:
                        snapshot_image = Image.open(io.BytesIO(base64.b64decode(snap["data"])))
                        st.image(
                            snapshot_image,
                            caption=(
                                f"Evidence crop as received — {snap['bytes']} B JPEG, "
                                f"{snap['width']}x{snap['height']} @q{snap['quality']} | "
                                f"whole packet {newest['bytes']} B on topic {newest['topic']}"
                            ),
                            width="stretch",
                        )
                    except Exception as exc:
                        st.warning(f"Could not decode snapshot: {exc}")
                else:
                    st.info("Newest packet carried no snapshot (metadata-only alert).")
            with col_json:
                st.code(
                    json.dumps(
                        {k: v for k, v in decoded_packet.items() if k != "snapshot"},
                        indent=2,
                    ),
                    language="json",
                )
    else:
        st.info("No packets transmitted yet — run a surveillance feed or transmit a test alert.")

    st.markdown("---")
    st.markdown("#### 📦 Delivered Packet Stream")
    recent_rows = publisher.recent_records(limit=25)
    if recent_rows:
        packet_df = pd.DataFrame(recent_rows)[
            ["utc", "seq", "event_type", "severity", "topic_kind", "bytes",
             "snapshot_bytes", "delivery", "attempts", "retransmit", "queue_depth"]
        ]
        st.dataframe(packet_df, width="stretch", hide_index=True)
    else:
        st.info("Uplink idle — no packets have been transmitted in this session.")

    st.markdown("#### 🗃️ Store-and-Forward Backlog")
    backlog = publisher.queue_snapshot()
    if backlog:
        backlog_df = pd.DataFrame(backlog)[
            ["event_id", "event_type", "severity", "utc", "bytes", "attempts", "age_s"]
        ]
        st.warning(f"{len(backlog)} packet(s) held on the outpost awaiting link recovery.")
        st.dataframe(backlog_df, width="stretch", hide_index=True)
    else:
        st.success("Backlog empty — every telemetry packet has been delivered.")

    dead = list(publisher.dead_letters)
    if dead:
        st.markdown("#### ⚠️ Dead-Letter Store (undeliverable telemetry)")
        dead_rows = [
            {
                "reason": entry.get("reason"),
                "attempts": entry.get("attempts", "-"),
                "event_type": entry.get("meta", {}).get("event_type"),
                "event_id": entry.get("meta", {}).get("event_id"),
            }
            for entry in dead
        ]
        st.dataframe(pd.DataFrame(dead_rows), width="stretch", hide_index=True)
        st.caption(
            "Packets land here only when retries are exhausted on a live-but-hostile link, or when "
            "buffered telemetry has gone too stale to be operationally useful."
        )
