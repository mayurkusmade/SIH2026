"""
GIS / Tactical Map Layer for the AnantaNetra (IBVAP) border grid.

Why this exists
---------------
The pipeline could already detect an intrusion, but a single-camera view cannot
answer the question a Sector Commander actually asks:

    *Where is the breach, which outpost is closest, and how long to intercept?*

That question is geographic, so the platform needs a geo layer: outpost
placement, camera coverage cones, geo-tagged incident clusters, distance/bearing
between nodes, and an interception recommendation.

Design constraint: 100% offline
-------------------------------
The mission statement promises a 100% air-gapped outpost, which makes a
tile-backed web map (Carto / Mapbox / OSM / Leaflet) the wrong choice: in the
field, no tiles can be fetched and no API key can be validated. So this module
renders the tactical picture as a **self-contained inline SVG** - no CDN, no
basemap tiles, no network access of any kind - while still exporting
standards-compliant **GeoJSON** so a connected Sector HQ can drop the identical
picture into QGIS or any staff map. `test_phase7.py` asserts that the rendered
map contains no external URL, which is what keeps that promise honest.

Projection note: a local tangent plane (equirectangular about the area-of-
interest centroid) is used rather than Web-Mercator. At sector scale - tens of
kilometres - the two agree to well under a pixel, and the tangent plane keeps the
scale bar exact and the code readable.

No third-party dependency is required: only the standard library. That matters
because this layer must run on the same bare edge box as the telemetry agent.
"""

import html
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# One degree of latitude, in metres (mean). Used for the tangent-plane fit.
EARTH_M_PER_DEG_LAT = 111_320.0

# Severity ordering drives both the map legend and cluster escalation.
SEVERITY_RANK = {"CRITICAL": 3, "WARNING": 2, "NOTICE": 1, "INFO": 0, "": 0}
SEVERITY_COLOR = {
    "CRITICAL": "#ff3b30",
    "WARNING": "#ff9f0a",
    "NOTICE": "#3d9bff",
    "INFO": "#32d74b",
}
NODE_STATE_COLOR = {
    "ONLINE": "#32d74b",
    "LIVE": "#32d74b",
    "STALLED": "#ff9f0a",
    "RECONNECTING": "#ff9f0a",
    "OFFLINE": "#ff3b30",
    "UNKNOWN": "#8b949e",
}

_COMPASS_16 = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)


# ---------------------------------------------------------------------------
# Geodesy helpers
# ---------------------------------------------------------------------------
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    lat1, lon1 = float(a[0]), float(a[1])
    lat2, lon2 = float(b[0]), float(b[1])
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    h = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * 6_371_008.8 * math.asin(min(1.0, math.sqrt(h)))


def bearing_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Initial bearing in degrees (0 = true north, clockwise) from a to b."""
    lat1, lat2 = math.radians(float(a[0])), math.radians(float(b[0]))
    d_lon = math.radians(float(b[1]) - float(a[1]))
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def destination_point(
    lat: float, lon: float, bearing: float, distance_m: float
) -> Tuple[float, float]:
    """Point reached by travelling `distance_m` along `bearing` from (lat, lon)."""
    radius = 6_371_008.8
    delta = distance_m / radius
    phi1, lambda1 = math.radians(lat), math.radians(lon)
    theta = math.radians(bearing)
    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta)
        + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    return (math.degrees(phi2), (math.degrees(lambda2) + 540.0) % 360.0 - 180.0)


def angle_difference_deg(a: float, b: float) -> float:
    """Smallest absolute angular separation between two bearings, in degrees."""
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def compass(bearing: float) -> str:
    """16-point compass label for a bearing."""
    return _COMPASS_16[int((float(bearing) % 360.0) / 22.5 + 0.5) % 16]


def sector_polygon(
    lat: float,
    lon: float,
    bearing: float,
    half_angle: float,
    range_m: float,
    steps: int = 24,
) -> List[Tuple[float, float]]:
    """
    The fan a camera actually covers: apex at the camera, arc at its range.

    Cone geometry is what makes the map command-useful - overlapping fans show
    where the border is genuinely watched, and gaps show where it is not.
    """
    if range_m <= 0:
        return [(lat, lon)]
    steps = max(2, int(steps))
    start = float(bearing) - float(half_angle)
    span = 2.0 * float(half_angle)
    arc = [
        destination_point(lat, lon, start + span * i / steps, range_m)
        for i in range(steps + 1)
    ]
    return [(lat, lon)] + arc


def point_in_sector(
    point: Tuple[float, float],
    lat: float,
    lon: float,
    bearing: float,
    half_angle: float,
    range_m: float,
) -> bool:
    """True if `point` lies inside a camera's coverage cone."""
    distance = haversine_m((lat, lon), point)
    if distance > float(range_m):
        return False
    if distance <= 1e-6:
        return True
    return angle_difference_deg(bearing_deg((lat, lon), point), bearing) <= float(half_angle)


def coverage_state(point: Tuple[float, float], coverage_cones: Sequence[dict]) -> dict:
    """
    How well a location is watched.

    Returns the number of covering cones plus the identity of the nearest one, so
    an alert arriving outside every cone can be flagged as a coverage gap rather
    than quietly plotted as if the sector were under observation.
    """
    covering = []
    nearest = None
    nearest_d = None
    for cone in coverage_cones or []:
        cam = (cone.get("lat"), cone.get("lon"))
        d = haversine_m(cam, point)
        if nearest_d is None or d < nearest_d:
            nearest_d, nearest = d, cone
        if point_in_sector(
            point,
            cam[0],
            cam[1],
            cone.get("bearing", 0.0),
            cone.get("half_angle", 0.0),
            cone.get("range_m", 0.0),
        ):
            covering.append(cone.get("camera_id"))
    return {
        "covered": bool(covering),
        "covering_cameras": covering,
        "nearest_camera": (nearest or {}).get("camera_id"),
        "nearest_camera_distance_m": round(nearest_d, 1) if nearest_d is not None else None,
    }


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
class TacticalProjection:
    """
    Fits a set of (lat, lon) points into a pixel viewport, north-up.

    Metres-per-pixel is uniform on both axes, so a circle on the map is a circle
    on the ground and the scale bar is truthful.
    """

    def __init__(
        self,
        points: Iterable[Tuple[float, float]],
        width: int = 980,
        height: int = 560,
        padding: int = 64,
    ):
        pts = [(float(p[0]), float(p[1])) for p in points if p and p[0] is not None]
        if not pts:
            pts = [(0.0, 0.0)]

        self.width = int(width)
        self.height = int(height)
        self.padding = int(padding)
        self.center_lat = sum(p[0] for p in pts) / len(pts)
        self.center_lon = sum(p[1] for p in pts) / len(pts)
        self.m_per_deg_lon = EARTH_M_PER_DEG_LAT * max(
            1e-6, math.cos(math.radians(self.center_lat))
        )

        offsets = [self._offset_m(*p) for p in pts]
        min_e = min(o[0] for o in offsets)
        max_e = max(o[0] for o in offsets)
        min_n = min(o[1] for o in offsets)
        max_n = max(o[1] for o in offsets)

        # A floor on the envelope stops a single node (or a tight cluster) from
        # producing a divide-by-zero and an infinitely zoomed map.
        self.span_e = max(max_e - min_e, 250.0)
        self.span_n = max(max_n - min_n, 250.0)
        self.bounds_m = (min_e, max_e, min_n, max_n)

        usable_w = max(1.0, self.width - 2.0 * self.padding)
        usable_h = max(1.0, self.height - 2.0 * self.padding)
        self.px_per_m = min(usable_w / self.span_e, usable_h / self.span_n)
        self.m_per_px = 1.0 / self.px_per_m

        self.offset_x = self.padding + (usable_w - self.span_e * self.px_per_m) / 2.0
        self.offset_y = self.padding + (usable_h - self.span_n * self.px_per_m) / 2.0

    # -- transforms ---------------------------------------------------------
    def _offset_m(self, lat: float, lon: float) -> Tuple[float, float]:
        """East/north offset in metres from the projection centre."""
        east = (float(lon) - self.center_lon) * self.m_per_deg_lon
        north = (float(lat) - self.center_lat) * EARTH_M_PER_DEG_LAT
        return (east, north)

    def project(self, lat: float, lon: float) -> Tuple[float, float]:
        """(lat, lon) -> (x, y) pixels, north up (so north decreases y)."""
        east, north = self._offset_m(lat, lon)
        min_e, _, min_n, _ = self.bounds_m
        x = self.offset_x + (east - min_e) * self.px_per_m
        y = self.offset_y + (self.span_n - (north - min_n)) * self.px_per_m
        return (x, y)

    def unproject(self, x: float, y: float) -> Tuple[float, float]:
        """(x, y) pixels -> (lat, lon). Used by the tests to prove invertibility."""
        min_e, _, min_n, _ = self.bounds_m
        east = (float(x) - self.offset_x) / self.px_per_m + min_e
        north = min_n + (self.span_n - (float(y) - self.offset_y) / self.px_per_m)
        lat = self.center_lat + north / EARTH_M_PER_DEG_LAT
        lon = self.center_lon + east / self.m_per_deg_lon
        return (lat, lon)

    def path_from_latlon(self, points: Sequence[Tuple[float, float]]) -> str:
        """SVG polyline 'points' attribute for a list of (lat, lon)."""
        return " ".join(
            f"{self.project(lat, lon)[0]:.1f},{self.project(lat, lon)[1]:.1f}"
            for lat, lon in points
        )

    @property
    def span_km(self) -> Tuple[float, float]:
        min_e, max_e, min_n, max_n = self.bounds_m
        return ((max_e - min_e) / 1000.0, (max_n - min_n) / 1000.0)


def convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Monotone-chain convex hull of (x, y) pixel points.

    Drawn as the area-of-operations outline: the stretch of border the sector is
    actually responsible for.
    """
    pts = sorted({(round(float(x), 4), round(float(y), 4)) for x, y in points})
    if len(pts) <= 2:
        return list(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


# ---------------------------------------------------------------------------
# Alert clustering
# ---------------------------------------------------------------------------
@dataclass
class AlertCluster:
    """A group of geographically-close incidents, as one map marker."""

    lat: float
    lon: float
    count: int = 0
    severity: str = "INFO"
    event_types: List[str] = field(default_factory=list)
    cameras: List[str] = field(default_factory=list)
    first_utc: str = ""
    last_utc: str = ""
    latest_detail: str = ""

    def escalate(self, severity: str) -> None:
        if SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(self.severity, 0):
            self.severity = severity

    @property
    def rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 0)


def cluster_alerts(
    alerts: Iterable[dict], radius_m: float = 300.0
) -> List[AlertCluster]:
    """
    Greedy distance clustering of geo-tagged alerts, worst-first on output.

    Greedy first-fit is deliberate: with a few hundred alerts at sector scale it
    is as good as DBSCAN here and is O(n * clusters) with deterministic output,
    which matters because the same incident list must always render identically
    for the audit trail.
    """
    entries = [a for a in alerts if a.get("lat") is not None and a.get("lon") is not None]
    entries.sort(key=lambda a: (str(a.get("utc") or ""), str(a.get("event_id") or "")))

    clusters: List[AlertCluster] = []
    for alert in entries:
        point = (float(alert["lat"]), float(alert["lon"]))
        target = None
        best = float(radius_m)
        for existing in clusters:
            d = haversine_m((existing.lat, existing.lon), point)
            if d <= best:
                best, target = d, existing
        if target is None:
            target = AlertCluster(lat=point[0], lon=point[1])
            clusters.append(target)
        else:
            # Keep the marker centroid honest as members accumulate.
            n = target.count
            target.lat = (target.lat * n + point[0]) / (n + 1)
            target.lon = (target.lon * n + point[1]) / (n + 1)

        target.count += 1
        target.escalate(str(alert.get("severity") or "INFO"))
        event_type = str(alert.get("event_type") or "")
        if event_type and event_type not in target.event_types:
            target.event_types.append(event_type)
        camera = str(alert.get("camera_id") or "")
        if camera and camera not in target.cameras:
            target.cameras.append(camera)
        utc = str(alert.get("utc") or "")
        if utc:
            target.first_utc = target.first_utc or utc
            target.last_utc = utc
        target.latest_detail = str(alert.get("details") or target.latest_detail)

    clusters.sort(key=lambda c: (-c.rank, -c.count, str(c.last_utc)))
    return clusters


def cluster_summary(clusters: Sequence[AlertCluster]) -> dict:
    """Headline counts for the map panel and the KPI strip."""
    by_severity: Dict[str, int] = {}
    for cluster in clusters:
        by_severity[cluster.severity] = by_severity.get(cluster.severity, 0) + cluster.count
    return {
        "clusters": len(clusters),
        "incidents": sum(c.count for c in clusters),
        "critical_clusters": sum(1 for c in clusters if c.severity == "CRITICAL"),
        "by_severity": by_severity,
        "first_utc": min((c.first_utc for c in clusters if c.first_utc), default=""),
        "last_utc": max((c.last_utc for c in clusters if c.last_utc), default=""),
    }


# ---------------------------------------------------------------------------
# Interception planning
# ---------------------------------------------------------------------------
def format_eta(seconds: float) -> str:
    """Human ETA: '38 s', '4 min 12 s', '1 h 05 min'."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        minutes, secs = divmod(int(round(seconds)), 60)
        return f"{minutes} min {secs:02d} s"
    hours, remainder = divmod(int(round(seconds)), 3600)
    return f"{hours} h {remainder // 60:02d} min"


def rank_responders(
    lat: float,
    lon: float,
    outposts: Dict[str, dict],
    speed_kmh: float = 40.0,
    limit: Optional[int] = None,
) -> List[dict]:
    """
    Orders outposts by real travel time to a location - the actual C2 decision.

    Straight-line distance is used and labelled as such: without road topology an
    ETA can only ever be a lower bound, and pretending otherwise would be worse
    than saying so.
    """
    target = (float(lat), float(lon))
    speed_ms = max(0.1, float(speed_kmh) * 1000.0 / 3600.0)
    ranked = []
    for node_id, outpost in (outposts or {}).items():
        position = (outpost.get("lat"), outpost.get("lon"))
        if position[0] is None or position[1] is None:
            continue
        distance = haversine_m(position, target)
        bearing = bearing_deg(position, target)
        ranked.append(
            {
                "node_id": node_id,
                "name": outpost.get("name", node_id),
                "distance_m": round(distance, 1),
                "distance_km": round(distance / 1000.0, 2),
                "bearing_deg": round(bearing, 1),
                "compass": compass(bearing),
                "eta_s": distance / speed_ms,
                "eta_text": format_eta(distance / speed_ms),
                "strength": outpost.get("strength"),
                "via": "straight-line (road network not modelled)",
            }
        )
    ranked.sort(key=lambda r: r["distance_m"])
    return ranked[:limit] if limit else ranked


def intercept_advice(
    lat: float,
    lon: float,
    outposts: Dict[str, dict],
    speed_kmh: float = 40.0,
    coverage_cones: Optional[Sequence[dict]] = None,
    alternates: int = 2,
) -> dict:
    """One call returning the full response picture for a breach location."""
    ranked = rank_responders(lat, lon, outposts, speed_kmh=speed_kmh)
    coverage = coverage_state((lat, lon), coverage_cones or [])
    primary = ranked[0] if ranked else None
    if primary:
        sentence = (
            f"Dispatch {primary['name']}: {primary['distance_km']} km "
            f"{primary['compass']} of the breach, ETA {primary['eta_text']} "
            f"at {speed_kmh:.0f} km/h."
        )
    else:
        sentence = "No outpost registered for this area - reinforce sector coverage."
    if not coverage["covered"]:
        sentence += " NOTE: breach location is outside every camera cone (coverage gap)."
    return {
        "primary": primary,
        "alternates": ranked[1 : 1 + alternates],
        "ranking": ranked,
        "coverage": coverage,
        "sentence": sentence,
    }


# ---------------------------------------------------------------------------
# Layer builders (registry + live records -> map layer)
# ---------------------------------------------------------------------------
def build_node_layer(
    cameras: Dict[str, dict],
    state_by_camera: Optional[Dict[str, str]] = None,
    outposts: Optional[Dict[str, dict]] = None,
    camera_fov: Optional[Dict[str, dict]] = None,
) -> List[dict]:
    """
    Collapses the camera registry into one marker per physical outpost.

    Several cameras normally share a BOP, and an operator thinks in outposts, not
    in channels - so markers are grouped by node_id and their cones are attached
    to the marker.
    """
    state_by_camera = state_by_camera or {}
    camera_fov = camera_fov or {}
    outposts = outposts or {}
    nodes: Dict[str, dict] = {}

    for key, camera in (cameras or {}).items():
        node_id = camera.get("node_id") or "UNKNOWN-NODE"
        node = nodes.setdefault(
            node_id,
            {
                "node_id": node_id,
                "name": (outposts.get(node_id) or {}).get("name", node_id),
                "lat": camera.get("lat"),
                "lon": camera.get("lon"),
                "sector": camera.get("sector", "UNKNOWN"),
                "cameras": [],
                "states": [],
                "sensor_states": [],
                "coverage": [],
            },
        )
        state = str(state_by_camera.get(camera.get("camera_id"), "UNKNOWN")).upper()
        node["cameras"].append(
            {
                "channel": key,
                "camera_id": camera.get("camera_id"),
                "label": camera.get("label", key),
                "state": state,
                "pre_rendered": bool(camera.get("pre_rendered")),
                "source": camera.get("video_source", ""),
            }
        )
        # A pre-rendered demo feed is not a sensor. It must not be allowed to
        # degrade the outpost's health picture on the command map (which would
        # paint a fully operational BOP red during a presentation).
        if camera.get("pre_rendered"):
            node["states"].append("DEMO")
        else:
            node["states"].append(state)
            node["sensor_states"].append(state)

        # A pre-rendered demo feed covers no ground: it must not inherit a cone
        # and overstate how much border is actually under observation.
        fov = None if camera.get("pre_rendered") else (
            camera_fov.get(camera.get("camera_id")) or camera.get("coverage")
        )
        if fov and fov.get("range_m"):
            node["coverage"].append(
                {
                    "camera_id": camera.get("camera_id"),
                    "bearing": float(fov.get("bearing", 0.0)),
                    "half_angle": float(fov.get("half_angle", 0.0)),
                    "range_m": float(fov.get("range_m", 0.0)),
                    "lat": camera.get("lat"),
                    "lon": camera.get("lon"),
                }
            )

    for node in nodes.values():
        # Node health = the worst sensor camera on it. A dead camera must never
        # be hidden behind a healthy sibling on the command map, and a camera we
        # have no telemetry for must not be reported as healthy either.
        sensors = node["sensor_states"]
        if any(s == "OFFLINE" for s in sensors):
            node["state"] = "OFFLINE"
        elif any(s in ("STALLED", "RECONNECTING") for s in sensors):
            node["state"] = "STALLED"
        elif sensors and all(s in ("LIVE", "ONLINE") for s in sensors):
            node["state"] = "ONLINE"
        else:
            node["state"] = "UNKNOWN"
        outpost = outposts.get(node["node_id"]) or {}
        node["strength"] = outpost.get("strength")
        node["coverage_range_km"] = round(
            max([c["range_m"] for c in node["coverage"]] or [0.0]) / 1000.0, 3
        )

    ordered = sorted(nodes.values(), key=lambda n: str(n["node_id"]))
    return ordered


def enrich_with_geo(
    records: Iterable[dict],
    cameras: Dict[str, dict],
    default_lat: Optional[float] = None,
    default_lon: Optional[float] = None,
) -> Tuple[List[dict], int]:
    """
    Adds lat/lon to local log records using the camera registry.

    On the wire this lookup is unnecessary - every telemetry packet already
    carries its node geo stamp - but the dashboard reads a local CSV, which holds
    only camera_id. Records whose camera is unknown are counted and returned
    rather than dropped, so the map can state how many incidents it could not
    place instead of silently under-reporting.
    """
    index = {
        cam.get("camera_id"): cam for cam in (cameras or {}).values() if cam.get("camera_id")
    }
    enriched: List[dict] = []
    unresolved = 0
    for record in records:
        camera = index.get(record.get("camera_id"))
        lat = lon = None
        if camera:
            lat, lon = camera.get("lat"), camera.get("lon")
        elif default_lat is not None:
            lat, lon = default_lat, default_lon
        if lat is None or lon is None:
            unresolved += 1
            continue
        enriched.append({**record, "lat": float(lat), "lon": float(lon)})
    return enriched, unresolved


def alert_layer_from_records(
    records: Iterable[dict], severity_for=None
) -> List[dict]:
    """
    Normalizes log records into map alerts, reusing the telemetry severity
    mapping so the map and the wire can never disagree about what is CRITICAL.
    """
    alerts = []
    for record in records:
        event_type = str(record.get("event_type") or "SYSTEM")
        severity = (
            severity_for(event_type, str(record.get("status") or ""))
            if severity_for
            else "INFO"
        )
        alerts.append(
            {
                "event_id": record.get("event_id"),
                "event_type": event_type,
                "severity": severity,
                "status": record.get("status"),
                "camera_id": record.get("camera_id"),
                "node_id": record.get("node_id"),
                "utc": record.get("utc_timestamp") or record.get("utc"),
                "details": record.get("details"),
                "track_id": record.get("track_id"),
                "lat": record.get("lat"),
                "lon": record.get("lon"),
                "ts": record.get("ts"),
            }
        )
    return alerts


def gis_snapshot(
    records: Iterable[dict],
    cameras: Dict[str, dict],
    outposts: Optional[Dict[str, dict]] = None,
    state_by_camera: Optional[Dict[str, str]] = None,
    cluster_radius_m: float = 300.0,
    response_speed_kmh: float = 40.0,
    severity_for=None,
    alert_limit: int = 400,
) -> dict:
    """
    Everything the GIS tab needs, in one call: nodes, alerts, clusters, coverage
    gaps, distance matrix and the interception picture for the newest incident.
    """
    outposts = outposts or {}
    nodes = build_node_layer(cameras, state_by_camera, outposts)
    records = list(records)
    placed, unresolved = enrich_with_geo(records, cameras)
    alerts = alert_layer_from_records(placed[-alert_limit:], severity_for=severity_for)
    clusters = cluster_alerts(alerts, radius_m=cluster_radius_m)

    cones = [cone for node in nodes for cone in node["coverage"]]
    gaps = [
        alert for alert in alerts
        if not coverage_state((alert["lat"], alert["lon"]), cones)["covered"]
    ]

    latest = alerts[-1] if alerts else None
    advice = (
        intercept_advice(
            latest["lat"], latest["lon"], outposts,
            speed_kmh=response_speed_kmh, coverage_cones=cones,
        )
        if latest and outposts
        else None
    )

    return {
        "nodes": nodes,
        "alerts": alerts,
        "clusters": clusters,
        "summary": cluster_summary(clusters),
        "coverage_cones": cones,
        "coverage_gaps": len(gaps),
        "unresolved_records": unresolved,
        "records_considered": len(records),
        "intercept": advice,
        "distance_matrix": distance_matrix(nodes),
        "outpost_count": len(nodes),
        "response_speed_kmh": response_speed_kmh,
        "cluster_radius_m": cluster_radius_m,
    }


def distance_matrix(nodes: Sequence[dict]) -> List[dict]:
    """
    Inter-outpost distances, nearest links first.

    Used to draw the command relay between outposts on the map and to answer
    'how far is backup?' without leaving the dashboard.
    """
    rows = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if None in (a.get("lat"), a.get("lon"), b.get("lat"), b.get("lon")):
                continue
            distance = haversine_m((a["lat"], a["lon"]), (b["lat"], b["lon"]))
            rows.append(
                {
                    "from": a["node_id"],
                    "to": b["node_id"],
                    "distance_m": round(distance, 1),
                    "distance_km": round(distance / 1000.0, 2),
                    "bearing_deg": round(bearing_deg((a["lat"], a["lon"]), (b["lat"], b["lon"])), 1),
                }
            )
    rows.sort(key=lambda r: r["distance_m"])
    return rows


# ---------------------------------------------------------------------------
# Standards-compliant export (for QGIS / Sector HQ staff map)
# ---------------------------------------------------------------------------
def nodes_geojson(nodes: Sequence[dict]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [node["lon"], node["lat"]]},
                "properties": {
                    "node_id": node["node_id"],
                    "name": node["name"],
                    "sector": node["sector"],
                    "state": node["state"],
                    "cameras": len(node["cameras"]),
                    "coverage_range_km": node.get("coverage_range_km"),
                },
            }
            for node in nodes
            if node.get("lat") is not None and node.get("lon") is not None
        ],
    }


def closed_ring(points: Sequence[Tuple[float, float]]) -> List[List[float]]:
    """
    Converts (lat, lon) vertices to a [lon, lat] ring closed for GeoJSON.

    RFC 7946 requires the first and last position to be identical; a ring that is
    not closed is silently invalid in most GIS tools, which is exactly the kind of
    export bug that only surfaces when someone else tries to open the file.
    """
    ring = [[float(lon), float(lat)] for lat, lon in points]
    if ring and ring[0] != ring[-1]:
        ring.append(list(ring[0]))
    return ring


def coverage_geojson(cones: Sequence[dict], steps: int = 24) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        closed_ring(
                            sector_polygon(
                                cone["lat"], cone["lon"], cone["bearing"],
                                cone["half_angle"], cone["range_m"], steps=steps,
                            )
                        )
                    ],
                },
                "properties": {
                    "camera_id": cone.get("camera_id"),
                    "bearing": cone.get("bearing"),
                    "half_angle": cone.get("half_angle"),
                    "range_m": cone.get("range_m"),
                },
            }
            for cone in cones or []
            if cone.get("lat") is not None
        ],
    }


def alerts_geojson(alerts: Iterable[dict], clusters: Sequence[AlertCluster] = ()) -> dict:
    features = []
    for alert in alerts:
        if alert.get("lat") is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [alert["lon"], alert["lat"]]},
                "properties": {
                    "event_type": alert.get("event_type"),
                    "severity": alert.get("severity"),
                    "status": alert.get("status"),
                    "camera_id": alert.get("camera_id"),
                    "node_id": alert.get("node_id"),
                    "utc": alert.get("utc"),
                    "track_id": alert.get("track_id"),
                    "details": alert.get("details"),
                },
            }
        )
    for cluster in clusters:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [cluster.lon, cluster.lat]},
                "properties": {
                    "kind": "cluster",
                    "count": cluster.count,
                    "severity": cluster.severity,
                    "event_types": cluster.event_types,
                    "cameras": cluster.cameras,
                    "first_utc": cluster.first_utc,
                    "last_utc": cluster.last_utc,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def dumps_geojson(feature_collection: dict) -> str:
    return json.dumps(feature_collection, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Offline SVG renderer
# ---------------------------------------------------------------------------
def _nice_scale_bar(m_per_px: float, target_px: float = 130.0) -> Tuple[float, float, str]:
    """Largest round ground length that fits the target pixel width."""
    target_m = max(1.0, m_per_px * target_px)
    for step in (10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10_000, 25_000, 50_000, 100_000):
        if step >= target_m:
            metres = float(step)
            break
    else:
        metres = target_m
    label = f"{metres / 1000:.1f} km" if metres >= 1000 else f"{metres:.0f} m"
    return (metres / m_per_px, metres, label)


def _grid_step_m(span_km: Tuple[float, float]) -> float:
    span = max(span_km)
    for step in (250, 500, 1000, 2000, 5000, 10_000, 25_000, 50_000):
        if span * 1000.0 / step <= 9:
            return float(step)
    return 100_000.0


def _tt(utc: str) -> str:
    """Renders an RFC3339 UTC stamp as a compact local-agnostic time."""
    text = str(utc or "")
    return text[11:19] + "Z" if len(text) >= 19 else (text or "—")


def render_coverage_inset(
    nodes: Sequence[dict],
    width: int = 1080,
    height: int = 196,
    title: str = "Local Coverage Detail (each panel at its own true scale)",
) -> str:
    """
    One mini-map per outpost, each drawn at its OWN scale.

    On a real sector the outposts are tens of kilometres apart while a perimeter
    camera watches a few hundred metres of ground, so on the wide map every
    coverage cone collapses to a sub-pixel dot and the single most operationally
    useful question - *what does this camera actually watch?* - becomes
    unanswerable. Each panel here is projected independently and carries its own
    scale bar, so the answer is legible without lying about the geometry.
    """
    panels_nodes = [
        node for node in (nodes or [])
        if node.get("coverage") and node.get("lat") is not None and node.get("lon") is not None
    ]
    if not panels_nodes:
        return ""

    panel_w = max(230, int(width / len(panels_nodes)) - 14)
    panel_h = max(90, height - 62)
    panels: List[str] = []

    for node in panels_nodes:
        frame_points = [(node["lat"], node["lon"])]
        for cone in node["coverage"]:
            frame_points.extend(
                sector_polygon(cone["lat"], cone["lon"], cone["bearing"],
                               cone["half_angle"], cone["range_m"], steps=12)
            )
        projection = TacticalProjection(frame_points, width=panel_w, height=panel_h,
                                        padding=16)
        parts = [
            f"<svg viewBox='0 0 {panel_w} {panel_h}' "
            "style='width:100%;height:auto;display:block' "
            "xmlns='http://www.w3.org/2000/svg' "
            f"aria-label='Coverage for {html.escape(str(node['node_id']))}'>",
            f"<rect x='0' y='0' width='{panel_w}' height='{panel_h}' fill='#070b10' "
            "stroke='#30363d'/>",
        ]

        for index, cone in enumerate(node["coverage"]):
            polygon = sector_polygon(
                cone["lat"], cone["lon"], cone["bearing"],
                cone["half_angle"], cone["range_m"], steps=30,
            )
            points = projection.path_from_latlon(polygon)
            parts.append(
                f"<polygon points='{points}' fill='rgba(61,155,255,0.16)' "
                "stroke='rgba(61,155,255,0.75)' stroke-width='1.2'/>"
            )
            # Range arc, so the panel reads as a measured distance and not a shape.
            apex_x, apex_y = projection.project(cone["lat"], cone["lon"])
            arc_x, arc_y = projection.project(*destination_point(
                cone["lat"], cone["lon"], cone["bearing"], cone["range_m"]
            ))
            parts.append(
                f"<line x1='{apex_x:.1f}' y1='{apex_y:.1f}' x2='{arc_x:.1f}' "
                f"y2='{arc_y:.1f}' stroke='rgba(61,155,255,0.5)' stroke-width='1' "
                "stroke-dasharray='3 3'>"
                f"<title>{html.escape(str(cone.get('camera_id')))}: "
                f"{cone['range_m'] / 1000.0:.2f} km at bearing {cone['bearing']:.0f}&#176;</title>"
                "</line>"
            )
        x, y = projection.project(node["lat"], node["lon"])
        parts.append(
            f"<rect x='{x - 4:.1f}' y='{y - 4:.1f}' width='8' height='8' "
            f"transform='rotate(45 {x:.1f} {y:.1f})' fill='#0d1117' "
            "stroke='#c9d1d9' stroke-width='1.6'/>"
        )
        bar_px, _bar_m, bar_label = _nice_scale_bar(projection.m_per_px, target_px=70)
        bx, by = 10, panel_h - 12
        parts.append(
            f"<line x1='{bx}' y1='{by}' x2='{bx + bar_px:.1f}' y2='{by}' "
            "stroke='#c9d1d9' stroke-width='2'/>"
            f"<text class='sub' x='{bx}' y='{by - 5}'>{html.escape(bar_label)}</text>"
        )
        parts.append("</svg>")

        max_range = max(cone["range_m"] for cone in node["coverage"]) / 1000.0
        panels.append(
            "<div class='inset'>"
            f"<div class='inset-title'>{html.escape(str(node['node_id']))} "
            f"<span class='sub'>&#183; {len(node['coverage'])} cone(s) &#183; "
            f"max {max_range:.2f} km</span></div>"
            + "".join(parts)
            + "</div>"
        )

    return (
        "<div class='insets'>"
        f"<div class='inset-title' style='margin:10px 0 6px 14px'>{html.escape(title)}</div>"
        "<div class='inset-row'>" + "".join(panels) + "</div></div>"
    )


def render_tactical_map(
    nodes: Sequence[dict],
    alerts: Iterable[dict] = (),
    clusters: Sequence[AlertCluster] = (),
    width: int = 980,
    height: int = 560,
    title: str = "Sector Tactical Picture",
    cluster_radius_m: float = 300.0,
    show_coverage: bool = True,
    show_links: bool = True,
    show_ao: bool = True,
    footer_note: str = "",
) -> str:
    """
    Fully self-contained tactical map as an HTML document.

    No tiles, no CDN, no network: suitable for an air-gapped outpost and for an
    iframe that must render with the satellite link down.
    """
    nodes = [n for n in (nodes or []) if n.get("lat") is not None and n.get("lon") is not None]
    alerts = [a for a in (alerts or []) if a.get("lat") is not None and a.get("lon") is not None]
    clusters = list(clusters or [])

    # Frame the map on nodes, coverage extents and incidents together - a marker
    # outside the frame is a marker the commander cannot see.
    frame_points: List[Tuple[float, float]] = [(n["lat"], n["lon"]) for n in nodes]
    frame_points += [(a["lat"], a["lon"]) for a in alerts]
    for node in nodes:
        for cone in node.get("coverage", []) or []:
            frame_points.extend(
                sector_polygon(
                    cone["lat"], cone["lon"], cone["bearing"],
                    cone["half_angle"], cone["range_m"], steps=8,
                )
            )
    projection = TacticalProjection(frame_points, width=width, height=height)

    parts: List[str] = []
    add = parts.append

    add("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    add(f"<title>{html.escape(title)}</title>")
    add(
        "<style>"
        "html,body{margin:0;padding:0;background:#070b10;color:#e6edf3;"
        "font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;}"
        ".svgwrap{position:relative;}"
        ".maptitle{position:absolute;left:16px;top:10px;font-size:13px;letter-spacing:.14em;"
        "text-transform:uppercase;color:#8b949e;}"
        ".legend{font-size:11.5px;fill:#c9d1d9;}"
        ".lbl{font-size:11px;fill:#e6edf3;}"
        ".sub{font-size:10px;fill:#8b949e;}"
        ".crit{animation:pulse 1.15s ease-in-out infinite;transform-origin:center;}"
        "@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}"
        ".insets{border-top:1px solid #30363d;margin-top:4px;}"
        ".inset-row{display:flex;gap:14px;padding:0 14px 14px;}"
        ".inset{flex:1 1 0;min-width:0;}"
        ".inset-title{font-size:11px;color:#e6edf3;margin-bottom:4px;letter-spacing:.04em;}"
        "text{pointer-events:none;}"
        "</style></head><body><div class='svgwrap'>"
    )
    # width:100% + height:auto lets the browser derive the height from the
    # viewBox aspect ratio. A fixed height attribute letterboxes the map inside a
    # narrower panel, which on a 13" command laptop wastes half the screen.
    add(
        f"<svg viewBox='0 0 {width} {height}' "
        "style='width:100%;height:auto;display:block' "
        "preserveAspectRatio='xMidYMid meet' "
        "xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-label='{html.escape(title)}'>"
    )
    add(f"<rect x='0' y='0' width='{width}' height='{height}' fill='#070b10'/>")    # --- graticule ---------------------------------------------------------
    # Ground-truth grid: lines at round kilometre multiples, so an operator can
    # estimate a distance off the map by eye.
    step_m = _grid_step_m(projection.span_km)
    grid_stroke = "#131c26"
    centre_lat, centre_lon = projection.unproject(width / 2.0, height / 2.0)
    d_lon = step_m / max(1e-6, projection.m_per_deg_lon)
    d_lat = step_m / EARTH_M_PER_DEG_LAT
    lon_base = math.floor(centre_lon / d_lon)
    for k in range(lon_base - 8, lon_base + 9):
        x, _ = projection.project(centre_lat, k * d_lon)
        if -1.0 <= x <= width + 1.0:
            add(f"<line x1='{x:.1f}' y1='0' x2='{x:.1f}' y2='{height}' stroke='{grid_stroke}'/>")
    lat_base = math.floor(centre_lat / d_lat)
    for k in range(lat_base - 8, lat_base + 9):
        _, y = projection.project(k * d_lat, centre_lon)
        if -1.0 <= y <= height + 1.0:
            add(f"<line x1='0' y1='{y:.1f}' x2='{width}' y2='{y:.1f}' stroke='{grid_stroke}'/>")

    # --- area of operations (convex hull of the outposts) ------------------
    if show_ao and len(nodes) >= 3:
        hull = convex_hull([projection.project(n["lat"], n["lon"]) for n in nodes])
        if len(hull) >= 3:
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in hull)
            add(
                f"<polygon points='{points}' fill='rgba(61,155,255,0.05)' "
                "stroke='#1f6feb' stroke-width='1.2' stroke-dasharray='7 6'/>"
            )

    # --- coverage cones ----------------------------------------------------
    if show_coverage:
        for node in nodes:
            for cone in node.get("coverage", []) or []:
                polygon = sector_polygon(
                    cone["lat"], cone["lon"], cone["bearing"],
                    cone["half_angle"], cone["range_m"], steps=26,
                )
                points = projection.path_from_latlon(polygon)
                add(
                    f"<polygon points='{points}' fill='rgba(61,155,255,0.10)' "
                    "stroke='rgba(61,155,255,0.55)' stroke-width='1' stroke-dasharray='4 4'>"
                    f"<title>{html.escape(str(cone.get('camera_id')))}: "
                    f"{cone['range_m'] / 1000.0:.2f} km arc at {cone['bearing']:.0f}&#176;, "
                    f"{cone['half_angle'] * 2:.0f}&#176; field of view</title></polygon>"
                )

    # --- inter-outpost links ----------------------------------------------
    if show_links:
        for row in distance_matrix(nodes)[: max(0, len(nodes) - 1) * 2]:
            a = next((n for n in nodes if n["node_id"] == row["from"]), None)
            b = next((n for n in nodes if n["node_id"] == row["to"]), None)
            if not a or not b:
                continue
            ax, ay = projection.project(a["lat"], a["lon"])
            bx, by = projection.project(b["lat"], b["lon"])
            add(
                f"<line x1='{ax:.1f}' y1='{ay:.1f}' x2='{bx:.1f}' y2='{by:.1f}' "
                "stroke='#30363d' stroke-width='1.4' stroke-dasharray='2 5'/>"
            )
            mid_x, mid_y = (ax + bx) / 2.0, (ay + by) / 2.0
            add(
                f"<text class='sub' x='{mid_x:.1f}' y='{mid_y - 4:.1f}' "
                f"text-anchor='middle'>{row['distance_km']:.1f} km relay</text>"
            )

    # --- nodes -------------------------------------------------------------
    # Incidents that land on an outpost are counted onto that outpost's own
    # label rather than drawn as a second label on top of it: a real breach is
    # almost always reported by the camera AT the outpost, so the collision case
    # is the common case, not the edge case.
    node_pixels = [projection.project(n["lat"], n["lon"]) for n in nodes]
    node_incidents: Dict[int, int] = {}
    for cluster in clusters:
        point = (cluster.lat, cluster.lon)
        nearest_index, nearest_distance = None, None
        for index, node in enumerate(nodes):
            distance = haversine_m((node["lat"], node["lon"]), point)
            if nearest_distance is None or distance < nearest_distance:
                nearest_index, nearest_distance = index, distance
        if nearest_distance is not None and nearest_distance <= max(
            cluster_radius_m, 400.0
        ):
            node_incidents[nearest_index] = (
                node_incidents.get(nearest_index, 0) + cluster.count
            )

    for index, node in enumerate(nodes):
        x, y = projection.project(node["lat"], node["lon"])
        colour = NODE_STATE_COLOR.get(node.get("state", "UNKNOWN"), "#8b949e")
        incidents_here = node_incidents.get(index, 0)
        add(
            f"<g><rect x='{x - 7:.1f}' y='{y - 7:.1f}' width='14' height='14' "
            f"transform='rotate(45 {x:.1f} {y:.1f})' fill='#0d1117' "
            f"stroke='{colour}' stroke-width='2.2'>"
            f"<title>{html.escape(str(node['node_id']))} &#8212; {html.escape(str(node['name']))} "
            f"| {html.escape(str(node.get('state')))} | {len(node.get('cameras', []))} camera(s)</title></rect>"
        )
        add(
            f"<text class='lbl' x='{x + 14:.1f}' y='{y - 2:.1f}'>"
            f"{html.escape(str(node['node_id']))}</text>"
        )
        add(
            f"<text class='sub' x='{x + 14:.1f}' y='{y + 11:.1f}' fill='#c9d1d9'>"
            f"{html.escape(str(node.get('state', 'UNKNOWN')))} "
            + (
                f" &#183; {len(node.get('cameras', []))} cam"
                if node.get("cameras") else ""
            )
            + (
                f" &#183; {node['coverage_range_km']:.1f} km arc"
                if node.get("coverage_range_km") else ""
            )
            + (
                f" &#183; <tspan fill='{SEVERITY_COLOR['CRITICAL']}'>"
                f"{incidents_here} incident(s) here</tspan>"
                if incidents_here else ""
            )
            + "</text></g>"
        )

    # --- incident clusters -------------------------------------------------
    # Incident positions are absolute and never nudged: a marker moved for
    # legibility is a marker that lies. Overlapping LABELS are suppressed instead.
    used_labels: List[Tuple[float, float]] = []
    for cluster in clusters:
        x, y = projection.project(cluster.lat, cluster.lon)
        colour = SEVERITY_COLOR.get(cluster.severity, SEVERITY_COLOR["NOTICE"])
        radius = 6.0 + 2.6 * math.log2(cluster.count + 1)
        pulse = " class='crit'" if cluster.severity == "CRITICAL" else ""
        types = ", ".join(cluster.event_types[:3])
        add(
            f"<circle{pulse} cx='{x:.1f}' cy='{y:.1f}' r='{radius:.1f}' "
            f"fill='{colour}' fill-opacity='0.30' stroke='{colour}' stroke-width='1.8'>"
            f"<title>{html.escape(cluster.severity)} &#215; {cluster.count} "
            f"({html.escape(types)}) {html.escape(_tt(cluster.last_utc))} "
            f"&#8212; {html.escape(cluster.latest_detail[:90])}</title></circle>"
        )
        if cluster.count > 1:
            add(
                f"<text class='lbl' x='{x:.1f}' y='{y + 3.6:.1f}' text-anchor='middle' "
                f"font-size='10' fill='#070b10' font-weight='700'>{cluster.count}</text>"
            )
        # A label is only drawn for an incident that is NOT on an outpost (that
        # case is already counted on the outpost label above), and only when it
        # will not land on a label already on the canvas. The marker itself, its
        # count numeral and its tooltip are always drawn - nothing is hidden.
        dx, dy = radius + 5.0, radius + 2.0
        label_x, label_y, label_w = x + dx + 3.0, y + dy + 3.5, 74.0
        crowded = any(
            abs(label_x - lx) < label_w and abs(label_y - ly) < 13.0
            for lx, ly in used_labels
        )
        for node_x, node_y in node_pixels:
            if abs(label_x - (node_x + 14)) < 116 and abs(label_y - (node_y + 4)) < 22:
                crowded = True
                break
        add(
            f"<line x1='{x + radius:.1f}' y1='{y:.1f}' x2='{x + dx:.1f}' "
            f"y2='{y + dy:.1f}' stroke='{colour}' stroke-width='1'/>"
            + ("" if crowded else (
                f"<text class='sub' x='{label_x:.1f}' y='{label_y:.1f}' fill='{colour}'>"
                f"{html.escape(cluster.event_types[0] if cluster.event_types else '')} "
                f"{html.escape(_tt(cluster.last_utc))}</text>"
            ))
        )
        if not crowded:
            used_labels.append((label_x, label_y))

    # --- scale bar + north arrow ------------------------------------------
    bar_px, bar_m, bar_label = _nice_scale_bar(projection.m_per_px)
    bx, by = 22, height - 26
    add(
        f"<g><line x1='{bx}' y1='{by}' x2='{bx + bar_px:.1f}' y2='{by}' "
        "stroke='#c9d1d9' stroke-width='2.5'/>"
        f"<line x1='{bx}' y1='{by - 5}' x2='{bx}' y2='{by + 5}' stroke='#c9d1d9' stroke-width='2'/>"
        f"<line x1='{bx + bar_px:.1f}' y1='{by - 5}' x2='{bx + bar_px:.1f}' y2='{by + 5}' "
        "stroke='#c9d1d9' stroke-width='2'/>"
        f"<text class='legend' x='{bx}' y='{by - 10}'>{html.escape(bar_label)} "
        f"&nbsp;|&nbsp; {projection.m_per_px:.1f} m/px</text></g>"
    )
    nx, ny = width - 40, 42
    add(
        f"<g><polygon points='{nx},{ny - 16} {nx - 7},{ny + 6} {nx + 7},{ny + 6}' "
        "fill='#c9d1d9'/>"
        f"<text class='legend' x='{nx}' y='{ny + 20}' text-anchor='middle'>N</text></g>"
    )

    # --- legend ------------------------------------------------------------
    lx, ly = 20, 20
    add(f"<g><rect x='{lx}' y='{ly}' width='412' height='134' rx='6' "
        "fill='rgba(13,17,23,0.88)' stroke='#30363d'/>")
    add(f"<text class='legend' x='{lx + 12}' y='{ly + 20}' font-weight='700'>"
        f"{html.escape(title)}</text>")
    add(f"<text class='sub' x='{lx + 12}' y='{ly + 36}'>"
        f"{len(nodes)} outpost(s) &#183; {len(clusters)} cluster(s) &#183; "
        f"{sum(c.count for c in clusters)} incident(s)</text>")
    add(f"<text class='sub' x='{lx + 12}' y='{ly + 52}'>"
        f"span {projection.span_km[0]:.1f} &#215; {projection.span_km[1]:.1f} km "
        f"&#183; cluster radius {cluster_radius_m:.0f} m</text>")
    add(f"<text class='legend' x='{lx + 12}' y='{ly + 74}'>"
        "&#9670; outpost &#8212; colour = link health</text>")
    add(f"<text class='legend' x='{lx + 12}' y='{ly + 90}'>"
        "shaded fan = camera coverage arc</text>")
    add(f"<text class='legend' x='{lx + 12}' y='{ly + 106}'>"
        "numeral in dot = incidents in cluster</text>")
    add(f"<text class='legend' x='{lx + 12}' y='{ly + 122}'>"
        "dashed line = outpost relay distance</text>")
    for i, (label, colour) in enumerate(
        (("CRITICAL", "CRITICAL"), ("WARNING", "WARNING"), ("NOTICE", "NOTICE"), ("INFO", "INFO"))
    ):
        row_y = ly + 74 + i * 16
        add(
            f"<circle cx='{lx + 244}' cy='{row_y - 4}' r='4.5' "
            f"fill='{SEVERITY_COLOR[colour]}' fill-opacity='0.35' "
            f"stroke='{SEVERITY_COLOR[colour]}'/>"
            f"<text class='legend' x='{lx + 256}' y='{row_y}'>{label}</text>"
        )
    add("</g>")

    # --- empty state -------------------------------------------------------
    if not nodes and not clusters:
        add(
            f"<text class='legend' x='{width / 2:.0f}' y='{height / 2:.0f}' "
            "text-anchor='middle'>No geo-located outposts or incidents yet "
            "&#8212; run the surveillance feed.</text>"
        )

    add("</svg>")

    # --- local coverage detail ---------------------------------------------
    # On a real sector the outposts are tens of kilometres apart while a camera
    # watches a few hundred metres, so on the wide map every coverage cone
    # collapses to a sub-pixel dot and the flagship feature becomes invisible.
    # These insets redraw each outpost's cones at their OWN true scale, so the
    # question "what does this camera actually watch?" stays answerable.
    if show_coverage:
        add(render_coverage_inset(nodes, width=width, height=196))

    note = footer_note or (
        "Rendered offline: inline SVG, no basemap tiles, no CDN, no API key. "
        "GeoJSON export available for Sector HQ staff maps."
    )
    add(f"<div class='legend' style='padding:8px 16px 14px;'>{html.escape(note)}</div>")
    add("</div></body></html>")
    return "".join(parts)
