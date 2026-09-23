"""
Edge Node & Camera Registry for the AnantaNetra / IBVAP surveillance grid.

Single source of truth for:
  - Which physical Border Out Post (BOP) each camera belongs to.
  - Geographic placement (used by the GIS Command Map and by telemetry packets
    so that Sector HQ can plot an alert without receiving any video).
  - The video source for that camera (local file for the demo, RTSP/ONVIF URL
    for a real deployment) and its default virtual-fence calibration.

Keeping this here removes the hardcoded video paths and camera identities that
previously lived inline in app.py, and gives every telemetry packet a stable
camera_id / node_id so events from different outposts can be attributed.
"""

import os


# ---------------------------------------------------------------------------
# Deployment-wide identity
# ---------------------------------------------------------------------------
# Every packet leaving an outpost is stamped with the node that produced it.
DEFAULT_NODE_ID = "BOP-SECTOR-A"
DEFAULT_NODE_NAME = "BOP Sector A - Border Out Post"
DEFAULT_NODE_LAT = 31.6340
DEFAULT_NODE_LON = 74.8723

# Assumed ground speed for a quick-response party. Used ONLY to produce a
# straight-line ETA lower bound on the GIS map, and labelled as such: without
# road topology no honest system can do better than that.
RESPONSE_SPEED_KMH = 40.0


# ---------------------------------------------------------------------------
# Physical outposts
# ---------------------------------------------------------------------------
# One marker per outpost on the command map. Cameras belong to an outpost; a
# Sector Commander thinks in outposts, not in video channels.
OUTPOSTS = {
    "BOP-SECTOR-A": {
        "name": "BOP Sector A (Forward Perimiter)",
        "lat": 31.6340,
        "lon": 74.8723,
        "strength": "1 section (14 personnel)",
        "kind": "BOP",
    },
    "BOP-SECTOR-B": {
        "name": "BOP Sector B (Crossing Approach)",
        "lat": 32.0419,
        "lon": 74.5560,
        "strength": "1 section (12 personnel)",
        "kind": "BOP",
    },
    "CHECKPOST-CHARLIE": {
        "name": "Checkpost Charlie (Axis Road)",
        "lat": 32.2733,
        "lon": 74.6300,
        "strength": "1 section + QRT (10 personnel)",
        "kind": "CHECKPOST",
    },
}


CAMERAS = {
    "Channel 1": {
        "camera_id": "CAM-A-PERIMETER",
        "node_id": "BOP-SECTOR-A",
        "label": "Sector A - BOP Perimeter (1080p Long-Range)",
        "sector": "SECTOR_A",
        "lat": 31.6340,
        "lon": 74.8723,
        # Video source: RTSP/ONVIF URL in the field, local sample for the demo.
        "video_source": "sample_videos/bop_perimeter.mp4",
        "fallback_sources": [],
        "fence": {
            "required": True,
            "default_tripwire_y": 600,
            "tripwire_x_range": (50, 1870),
            "zone_name": "Sector A Perimeter",
            "slider_min": 100,
            "slider_max": 1000,
        },
        # Ground coverage cone: the fan this camera actually watches. Drawn on
        # the GIS map and used to flag an alert that falls in a coverage gap.
        "coverage": {"bearing": 322.0, "half_angle": 21.0, "range_m": 1200.0},
    },
    "Channel 2": {
        "camera_id": "CAM-B-TRIPWIRE",
        "node_id": "BOP-SECTOR-B",
        "label": "Sector B - Pedestrian Crossing (vedio-sih26.mp4)",
        "sector": "SECTOR_B",
        "lat": 32.0419,
        "lon": 74.5560,
        "video_source": "vedio-sih26.mp4",
        "fallback_sources": [
            "sample_videos/pedestrian_crossing.mp4",
            "sample_videos/vedio-sih26.mp4",
        ],
        "fence": {
            "required": True,
            "default_tripwire_y": 280,
            "tripwire_x_range": (20, 828),
            "zone_name": "Sector B Tripwire",
            "slider_min": 50,
            "slider_max": 450,
        },
        "coverage": {"bearing": 344.0, "half_angle": 32.0, "range_m": 400.0},
    },
    "Channel 3": {
        "camera_id": "CAM-C-CHECKPOST",
        "node_id": "CHECKPOST-CHARLIE",
        "label": "Sector C - Checkpost Charlie (Vehicles & ANPR)",
        "sector": "SECTOR_C",
        "lat": 32.2733,
        "lon": 74.6300,
        "video_source": "sample_videos/checkpost_traffic.mp4",
        "fallback_sources": [],
        "fence": None,  # Vehicle checkpoint: ANPR path, no tripwire.
        "coverage": {"bearing": 24.0, "half_angle": 42.0, "range_m": 260.0},
    },
    "Channel 4": {
        "camera_id": "CAM-D-BACKUP",
        "node_id": "BOP-SECTOR-A",
        "label": "Backup Pre-recorded Demo (Insurance Run)",
        "sector": "SECTOR_A",
        "lat": 31.6340,
        "lon": 74.8723,
        "video_source": "sample_videos/backup_annotated_run.mp4",
        "fallback_sources": ["sample_videos/checkpost_traffic.mp4"],
        "fence": None,
        "pre_rendered": True,
    },
}


def get_camera(channel_key: str) -> dict:
    """
    Returns the camera record for a channel key such as 'Channel 2'.

    Falls back to Channel 1 so an unknown channel can never crash the
    surveillance loop mid-stream.
    """
    return CAMERAS.get(channel_key, CAMERAS["Channel 1"])


def channel_key_from_selection(channel_selection: str) -> str:
    """
    Normalizes a UI radio label ('Channel 2: Sector B - ...') to 'Channel 2'.
    """
    return channel_selection.split(":")[0].strip()


def resolve_video_source(camera: dict) -> str:
    """
    Picks the first existing video source for a camera.

    Preserves the demo behaviour where the pedestrian crossing feed may live at
    the repo root or inside sample_videos/, and where the pre-rendered backup
    run silently falls back to the checkpost feed if it was never rendered.
    """
    primary = camera.get("video_source")
    if primary and os.path.exists(primary):
        return primary

    for candidate in camera.get("fallback_sources", []):
        if os.path.exists(candidate):
            return candidate

    # Return the primary even if missing so callers can report the intended path.
    return primary


# ---------------------------------------------------------------------------
# Deployment configuration (environment-driven, container-friendly)
# ---------------------------------------------------------------------------
# An edge box is provisioned by whoever installed it - a jawan or a field
# engineer - and shipped images should be configurable without editing code,
# exactly the way a real appliance is. Every override is optional and the demo
# defaults are unchanged when no environment is set.
ENV_PREFIX = "IBVAP_"

# The complete set of recognised suffixes. Exposed so the packaging can be
# verified against the code (test_phase7.py), which is what catches a compose
# file passing an environment variable the application never reads.
DEPLOYMENT_ENV_KEYS = (
    "ROLE",
    "NODE_ID",
    "NODE_NAME",
    "CHANNELS",
    "TELEMETRY",
    "MQTT_HOST",
    "MQTT_PORT",
    "LINK_PROFILE",
    "PACKET_BUDGET_KB",
    "GRID",
    "CONF_THRESHOLD",
    "DATA_DIR",
)


def _env_get(env: dict, name: str, default):
    value = env.get(ENV_PREFIX + name)
    return default if value is None or str(value).strip() == "" else value


def deployment_config(environ=None) -> dict:
    """
    Reads the IBVAP_* runtime overrides.

    Recognised (all optional):
      IBVAP_ROLE            edge | c2            (what this node is for)
      IBVAP_NODE_ID         outpost identity stamped on every packet
      IBVAP_NODE_NAME       human label for the outpost
      IBVAP_CHANNELS        comma list of registry keys, e.g. 'Channel 1,Channel 2'
      IBVAP_TELEMETRY       simulated | mqtt
      IBVAP_MQTT_HOST       broker host
      IBVAP_MQTT_PORT       broker port
      IBVAP_LINK_PROFILE    FIBER | 4G | SATELLITE | DEGRADED_SATCOM
      IBVAP_PACKET_BUDGET_KB hard ceiling for one telemetry packet
      IBVAP_GRID            true -> start every registered camera
      IBVAP_CONF_THRESHOLD  detector confidence floor
      IBVAP_DATA_DIR        where alerts.csv / frs_index.sqlite live (a volume)
    """
    env = os.environ if environ is None else dict(environ)

    def as_int(name, default):
        try:
            return int(float(_env_get(env, name, default)))
        except (TypeError, ValueError):
            return default

    def as_float(name, default):
        try:
            return float(_env_get(env, name, default))
        except (TypeError, ValueError):
            return default

    raw_channels = str(_env_get(env, "CHANNELS", "")).strip()
    channels = [c.strip() for c in raw_channels.split(",") if c.strip()] or None

    role = str(_env_get(env, "ROLE", "edge")).strip().lower()
    if role not in ("edge", "c2"):
        role = "edge"

    return {
        "role": role,
        "node_id": str(_env_get(env, "NODE_ID", DEFAULT_NODE_ID)),
        "node_name": str(_env_get(env, "NODE_NAME", DEFAULT_NODE_NAME)),
        "channels": channels,
        "telemetry_mode": str(_env_get(env, "TELEMETRY", "simulated")).strip().lower(),
        "mqtt_host": str(_env_get(env, "MQTT_HOST", "localhost")),
        "mqtt_port": as_int("MQTT_PORT", 1883),
        "link_profile": str(_env_get(env, "LINK_PROFILE", "SATELLITE")).strip().upper(),
        "packet_budget_kb": max(1, as_int("PACKET_BUDGET_KB", 10)),
        "grid": str(_env_get(env, "GRID", "false")).strip().lower()
        in ("1", "true", "yes", "on"),
        "conf_threshold": min(0.95, max(0.05, as_float("CONF_THRESHOLD", 0.35))),
        "data_dir": str(_env_get(env, "DATA_DIR", ".")),
    }


def channel_keys() -> list:
    """Registry order, so CLI/env channel lists match the UI ordering."""
    return list(CAMERAS.keys())



def outposts() -> dict:
    """Outpost table keyed by node_id, used by the GIS layer and the uplink."""
    return OUTPOSTS


def coverage_cone(camera: dict) -> dict:
    """
    Ground coverage cone for a camera, or {} when it has none.

    A pre-rendered demo feed covers nothing, and say so rather than inheriting a
    neighbour's cone and overstating how much border is watched.
    """
    if camera.get("pre_rendered"):
        return {}
    cone = camera.get("coverage")
    if not cone:
        return {}
    return {
        "camera_id": camera.get("camera_id"),
        "bearing": float(cone.get("bearing", 0.0)),
        "half_angle": float(cone.get("half_angle", 0.0)),
        "range_m": float(cone.get("range_m", 0.0)),
    }


def node_geo(camera: dict) -> dict:
    """Geo-location block carried inside every telemetry packet."""
    return {
        "node_id": camera.get("node_id", DEFAULT_NODE_ID),
        "camera_id": camera.get("camera_id", "CAM-UNKNOWN"),
        "sector": camera.get("sector", "UNKNOWN"),
        "lat": camera.get("lat", DEFAULT_NODE_LAT),
        "lon": camera.get("lon", DEFAULT_NODE_LON),
    }
