"""
IBVAP PHASE 7 VERIFICATION: GIS COMMAND MAP
===========================================

Verifies the geographic command layer against the mission's hardest constraint.
The platform promises a 100% air-gapped outpost, which means the tactical map
must render with no basemap tiles, no CDN and no API key - so section 9 asserts
that the rendered map fetches nothing from the network. That is the single check
that keeps the offline claim honest rather than aspirational.

  1. Geodesy primitives - distance, bearing, destination, compass.
  2. Coverage cones - what a camera actually watches, and containment tests.
  3. Projection - north-up, metric-preserving, invertible, frame-fitted.
  4. Alert clustering - proximity merge, severity escalation, stable ordering.
  5. Interception planning - nearest-responder ranking, honest ETA lower bounds,
     straight-line labelling, coverage-gap flagging.
  6. Registry snapshot - cameras collapse to outposts, worst-camera health wins.
  7. Log-record geo enrichment - placement and unplaceable-alert accounting.
  8. GeoJSON export for Sector HQ staff maps (QGIS), including axis order.
  9. OFFLINE RENDER - self-contained SVG, zero network fetches, HTML escaping.
 10. Edge cases - empty grid, single outpost, coincident points.

Run:  python test_phase7.py
"""

import html as html_module
import json
import os
import sys

from modules import gis
from modules.gis import (
    AlertCluster,
    TacticalProjection,
    alerts_geojson,
    bearing_deg,
    build_node_layer,
    cluster_alerts,
    cluster_summary,
    compass,
    convex_hull,
    coverage_geojson,
    coverage_state,
    destination_point,
    distance_matrix,
    dumps_geojson,
    enrich_with_geo,
    format_eta,
    gis_snapshot,
    haversine_m,
    intercept_advice,
    nodes_geojson,
    point_in_sector,
    rank_responders,
    render_tactical_map,
    sector_polygon,
)
from modules.nodes import CAMERAS, DEPLOYMENT_ENV_KEYS, OUTPOSTS, deployment_config
from modules.telemetry import severity_for

PASS = "  [PASS]"
FAIL = "  [FAIL]"
failures = []


def check(condition, message):
    print(f"{PASS if condition else FAIL} {message}")
    if not condition:
        failures.append(message)
    return bool(condition)


def header(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def close(actual, expected, tol, message):
    return check(abs(actual - expected) <= tol, f"{message} ({actual} ~= {expected})")


def alert(lat, lon, event_type="INTRUSION_ALERT", utc="2026-09-23T04:00:00Z",
          camera_id="CAM-A-PERIMETER", details="test", event_id=None):
    return {
        "event_id": event_id, "event_type": event_type, "utc": utc,
        "camera_id": camera_id, "node_id": "BOP-SECTOR-A", "details": details,
        "status": "VERIFIED", "track_id": 7, "lat": lat, "lon": lon,
        "severity": severity_for(event_type, "VERIFIED"),
    }


def main():
    # ------------------------------------------------------------------
    header("1. GEODESY PRIMITIVES")
    # ------------------------------------------------------------------
    # One degree of latitude on a sphere of R=6371.0088 km is 111.19 km.
    one_degree = haversine_m((0.0, 0.0), (1.0, 0.0))
    close(one_degree, 111_195.0, 200.0, "1 degree of latitude measures ~111.2 km")
    check(haversine_m((31.5, 74.5), (31.5, 74.5)) == 0.0, "distance to self is zero")
    check(
        abs(haversine_m((31.5, 74.5), (31.6, 74.6))
            - haversine_m((31.6, 74.6), (31.5, 74.5))) < 1e-6,
        "distance is symmetric",
    )

    # destination_point must invert haversine exactly enough for map work.
    target = destination_point(31.6340, 74.8723, 30.0, 1500.0)
    close(haversine_m((31.6340, 74.8723), target), 1500.0, 1.0,
          "destination_point then haversine round-trips 1500 m")
    close(bearing_deg((31.6340, 74.8723), target) % 360.0, 30.0, 0.2,
          "destination_point then bearing round-trips 30 deg")
    check(destination_point(0.0, 0.0, 0.0, 111_195.0)[0] > 0.99,
          "bearing 0 travels north")
    check(compass(0) == "N" and compass(90) == "E" and compass(180) == "S"
          and compass(270) == "W", "cardinal bearings map to N/E/S/W")
    check(compass(350) == "N" and compass(100) == "E",
          "bearings wrap correctly across the 360/0 seam")
    check(gis.angle_difference_deg(5.0, 355.0) == 10.0,
          "angular difference handles wrap-around (5 vs 355 = 10 deg)")

    # ------------------------------------------------------------------
    header("2. COVERAGE CONES")
    # ------------------------------------------------------------------
    poly = sector_polygon(31.6340, 74.8723, 322.0, 21.0, 1200.0, steps=24)
    check(len(poly) == 26, "sector polygon = apex + arc vertices")
    check(poly[0] == (31.6340, 74.8723), "sector polygon apex is the camera position")
    centro = (31.6340, 74.8723)
    check(point_in_sector(destination_point(*centro, 322.0, 600.0), *centro, 322.0, 21.0, 1200.0),
          "a target straight down the camera axis is covered")
    check(not point_in_sector(destination_point(*centro, 30.0, 600.0), *centro, 322.0, 21.0, 1200.0),
          "a target 292 deg off-axis is NOT covered")
    check(point_in_sector(destination_point(*centro, 343.0, 600.0), *centro, 322.0, 21.0, 1200.0),
          "a target exactly on the cone edge is covered (inclusive)")
    check(not point_in_sector(destination_point(*centro, 344.0, 600.0), *centro, 322.0, 21.0, 1200.0),
          "a target just outside the cone edge is NOT covered")
    check(not point_in_sector(destination_point(*centro, 322.0, 1300.0), *centro, 322.0, 21.0, 1200.0),
          "a target beyond the camera range is NOT covered")
    check(point_in_sector(centro, *centro, 322.0, 21.0, 1200.0),
          "a target at the camera itself is covered")
    check(sector_polygon(31.6, 74.8, 0.0, 30.0, 0.0, steps=4) == [(31.6, 74.8)],
          "a zero-range cone degenerates to a point instead of dividing by zero")

    cones = [
        {"camera_id": "CAM-A", "lat": 31.6340, "lon": 74.8723,
         "bearing": 322.0, "half_angle": 21.0, "range_m": 1200.0},
        {"camera_id": "CAM-B", "lat": 32.0419, "lon": 74.5560,
         "bearing": 344.0, "half_angle": 32.0, "range_m": 400.0},
    ]
    inside = coverage_state(destination_point(31.6340, 74.8723, 322.0, 500.0), cones)
    check(inside["covered"] and inside["covering_cameras"] == ["CAM-A"],
          "coverage_state reports which camera covers a point")
    outside = coverage_state((29.0, 77.0), cones)
    check(not outside["covered"], "a point under no cone is reported as a coverage gap")
    check(outside["nearest_camera"] in ("CAM-A", "CAM-B"),
          "coverage gap still names the nearest camera")

    # ------------------------------------------------------------------
    header("3. PROJECTION")
    # ------------------------------------------------------------------
    a = (31.7000, 74.9000)   # north-east
    b = (31.6000, 74.8000)   # south-west
    proj = TacticalProjection([a, b], width=800, height=400, padding=50)
    ax, ay = proj.project(*a)
    bx, by = proj.project(*b)
    check(ay < by, "north is up (a northern point has a smaller y)")
    check(ax > bx, "east is right (an eastern point has a larger x)")
    check(proj.padding - 1 <= min(ax, bx) and max(ax, bx) <= 800 - proj.padding + 1,
          "projected points stay inside the horizontal padding")
    check(proj.padding - 1 <= min(ay, by) and max(ay, by) <= 400 - proj.padding + 1,
          "projected points stay inside the vertical padding")

    ground = haversine_m(a, b)
    pixel = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    close(pixel * proj.m_per_px, ground, ground * 0.01,
          "pixel distance x m/px reproduces true ground distance (within 1%)")

    lat_back, lon_back = proj.unproject(ax, ay)
    close(lat_back, a[0], 1e-6, "unproject inverts project (latitude)")
    close(lon_back, a[1], 1e-6, "unproject inverts project (longitude)")

    single = TacticalProjection([(31.6340, 74.8723)])
    sx, sy = single.project(31.6340, 74.8723)
    check(0 <= sx <= single.width and 0 <= sy <= single.height,
          "a single-point grid still projects to a finite, on-canvas position")
    check(single.m_per_px > 0, "a single-point grid keeps a finite scale")

    empty = TacticalProjection([])
    check(empty.m_per_px > 0 and empty.span_km == (0.0, 0.0),
          "an empty grid keeps a finite scale and honestly reports a zero ground span")

    # ------------------------------------------------------------------
    header("4. ALERT CLUSTERING")
    # ------------------------------------------------------------------
    near = [
        alert(31.6340, 74.8723, utc="2026-09-23T04:00:00Z", event_id="a"),
        alert(31.6345, 74.8728, utc="2026-09-23T04:01:00Z", event_id="b"),
        alert(31.6342, 74.8720, utc="2026-09-23T04:02:00Z", event_id="c"),
        alert(32.0419, 74.5560, utc="2026-09-23T04:03:00Z", event_id="d",
              camera_id="CAM-B-TRIPWIRE"),
    ]
    clusters = cluster_alerts(near, radius_m=300.0)
    check(len(clusters) == 2, "three nearby alerts merge; the remote one does not")
    merged = clusters[0] if clusters[0].count == 3 else clusters[1]
    check(merged.count == 3, "cluster counts its members")
    check(merged.first_utc == "2026-09-23T04:00:00Z"
          and merged.last_utc == "2026-09-23T04:02:00Z",
          "cluster records its time span (first/last UTC)")

    mixed = cluster_alerts([
        alert(31.6340, 74.8723, "SYSTEM", utc="2026-09-23T04:00:00Z", event_id="x"),
        alert(31.6341, 74.8724, "WATCHLIST_HIT", utc="2026-09-23T04:01:00Z", event_id="y"),
        alert(31.6342, 74.8725, "INTRUSION_ALERT", utc="2026-09-23T04:02:00Z", event_id="z"),
    ], radius_m=300.0)
    check(len(mixed) == 1 and mixed[0].severity == "CRITICAL",
          "a cluster escalates to its worst member's severity")
    check(mixed[0].severity == severity_for("WATCHLIST_HIT"),
          "CRITICAL here equals the telemetry severity mapping (one source of truth)")

    check(cluster_alerts([], radius_m=300.0) == [], "clustering an empty list is a no-op")
    summary = cluster_summary(clusters)
    check(summary["clusters"] == 2 and summary["incidents"] == 4,
          "cluster summary totals clusters and incidents")
    check(summary["by_severity"].get("CRITICAL") == 4,
          "cluster summary counts incidents per severity")

    # ------------------------------------------------------------------
    header("5. INTERCEPTION PLANNING")
    # ------------------------------------------------------------------
    check(format_eta(38) == "38 s", "ETA formats seconds")
    check(format_eta(900) == "15 min 00 s", "ETA formats minutes and seconds")
    check(format_eta(3900) == "1 h 05 min", "ETA formats hours")
    check(format_eta(-5) == "0 s", "ETA never goes negative")

    ten_km = destination_point(31.6340, 74.8723, 90.0, 10_000.0)
    ranked = rank_responders(ten_km[0], ten_km[1], OUTPOSTS, speed_kmh=40.0)
    check(ranked[0]["node_id"] == "BOP-SECTOR-A",
          "the nearest outpost ranks first for an incident beside it")
    close(ranked[0]["distance_km"], 10.0, 0.05, "10 km east is measured as 10 km")
    close(ranked[0]["eta_s"], 900.0, 6.0, "10 km at 40 km/h gives a 15-minute ETA")
    check(ranked[0]["compass"] == "E", "responder bearing is reported as a compass point")
    check("straight-line" in ranked[0]["via"],
          "the ETA is labelled as a straight-line lower bound, not a road route")
    check([r["distance_m"] for r in ranked] == sorted(r["distance_m"] for r in ranked),
          "responders are returned in ascending distance order")
    check(all(a["distance_m"] <= b["distance_m"] for a, b in zip(ranked, ranked[1:])),
          "ranking is monotonic - no outpost is offered out of order")

    advice = intercept_advice(*ten_km, OUTPOSTS, speed_kmh=40.0, coverage_cones=cones)
    check(advice["primary"]["node_id"] == "BOP-SECTOR-A", "intercept advice names a primary")
    check(len(advice["alternates"]) == 2, "intercept advice offers alternates")
    check("Dispatch" in advice["sentence"] and "ETA" in advice["sentence"],
          "intercept advice renders a dispatch sentence")
    covered_point = destination_point(31.6340, 74.8723, 322.0, 500.0)
    covered_advice = intercept_advice(*covered_point, OUTPOSTS,
                                      coverage_cones=cones)
    check("coverage gap" not in covered_advice["sentence"],
          "no coverage-gap warning when a camera cone does cover the breach")
    far_advice = intercept_advice(35.0, 80.0, OUTPOSTS, coverage_cones=cones)
    check("coverage gap" in far_advice["sentence"],
          "an incident outside every cone is flagged as a coverage gap")
    check(intercept_advice(31.0, 74.0, {})["primary"] is None,
          "intercept advice without any outpost registered does not crash")

    matrix = distance_matrix(build_node_layer(CAMERAS, outposts=OUTPOSTS))
    check(len(matrix) == 3, "distance matrix covers every outpost pair (3 choose 2)")
    check(matrix[0]["distance_m"] <= matrix[-1]["distance_m"],
          "distance matrix is sorted nearest-pair first")

    # ------------------------------------------------------------------
    header("6. REGISTRY SNAPSHOT (real camera registry)")
    # ------------------------------------------------------------------
    states = {"CAM-A-PERIMETER": "LIVE", "CAM-B-TRIPWIRE": "ONLINE",
              "CAM-C-CHECKPOST": "LIVE", "CAM-D-BACKUP": "UNKNOWN"}
    nodes = build_node_layer(CAMERAS, states, OUTPOSTS)
    check(len(nodes) == 3, "4 camera channels collapse to 3 physical outposts")
    node_a = next(n for n in nodes if n["node_id"] == "BOP-SECTOR-A")
    check(len(node_a["cameras"]) == 2, "both Sector A channels are attached to one marker")
    check(node_a["state"] == "ONLINE", "outpost health aggregates its cameras")
    check(len(node_a["coverage"]) == 1,
          "the pre-rendered backup feed contributes no coverage cone")

    degraded = build_node_layer(CAMERAS, {**states, "CAM-A-PERIMETER": "STALLED"}, OUTPOSTS)
    node_a2 = next(n for n in degraded if n["node_id"] == "BOP-SECTOR-A")
    check(node_a2["state"] == "STALLED",
          "worst-camera health wins - a stalled feed is never hidden by a healthy sibling")
    dead = build_node_layer(CAMERAS, {**states, "CAM-A-PERIMETER": "OFFLINE"}, OUTPOSTS)
    node_a3 = next(n for n in dead if n["node_id"] == "BOP-SECTOR-A")
    check(node_a3["state"] == "OFFLINE", "an offline camera marks its outpost OFFLINE")
    demo = build_node_layer(CAMERAS, {**states, "CAM-D-BACKUP": "OFFLINE"}, OUTPOSTS)
    node_a4 = next(n for n in demo if n["node_id"] == "BOP-SECTOR-A")
    check(node_a4["state"] == "ONLINE",
          "the pre-rendered demo feed is not a sensor and cannot redden a healthy BOP")
    blind = build_node_layer(CAMERAS, {"CAM-A-PERIMETER": "LIVE"}, OUTPOSTS)
    node_b = next(n for n in blind if n["node_id"] == "BOP-SECTOR-B")
    check(node_b["state"] == "UNKNOWN",
          "an outpost with no camera telemetry is UNKNOWN - never reported as healthy")
    node_a5 = next(n for n in blind if n["node_id"] == "BOP-SECTOR-A")
    check(node_a5["state"] == "ONLINE",
          "an outpost whose only sensor camera is live reads ONLINE")

    snap = gis_snapshot(
        [dict(r) for r in [
            {"event_type": "INTRUSION_ALERT", "camera_id": "CAM-A-PERIMETER",
             "utc_timestamp": "2026-09-23T04:00:00Z", "status": "VERIFIED",
             "details": "UNKNOWN | CROSSING", "track_id": 3},
            {"event_type": "VEHICLE_ANPR", "camera_id": "CAM-C-CHECKPOST",
             "utc_timestamp": "2026-09-23T04:05:00Z", "status": "VERIFIED",
             "details": "Plate Conf: 91%", "track_id": 9},
        ]],
        CAMERAS, outposts=OUTPOSTS, state_by_camera=states, severity_for=severity_for,
    )
    check(snap["outpost_count"] == 3, "snapshot counts outposts, not channels")
    check(len(snap["coverage_cones"]) == 3, "snapshot exposes 3 camera coverage cones")
    check(len(snap["alerts"]) == 2, "snapshot geolocates both log records")
    check(snap["alerts"][0]["severity"] == "CRITICAL"
          and snap["alerts"][1]["severity"] == "NOTICE",
          "snapshot severity matches the uplink's own mapping")
    check(snap["summary"]["incidents"] == 2, "snapshot summarises 2 incidents")
    check(snap["intercept"]["primary"] is not None,
          "snapshot derives an intercept recommendation for the newest incident")
    check(snap["coverage_gaps"] == 0,
          "both incidents fall inside a camera cone and are not flagged as gaps")

    # ------------------------------------------------------------------
    header("7. LOG-RECORD GEO ENRICHMENT")
    # ------------------------------------------------------------------
    placed, unresolved = enrich_with_geo(
        [{"camera_id": "CAM-A-PERIMETER", "event_type": "INTRUSION_ALERT"},
         {"camera_id": "CAM-C-CHECKPOST", "event_type": "VEHICLE_ANPR"}],
        CAMERAS,
    )
    check(len(placed) == 2 and unresolved == 0,
          "records resolve to lat/lon through the camera registry")
    check(placed[0]["lat"] == CAMERAS["Channel 1"]["lat"], "resolved latitude is the camera's")
    check(isinstance(placed[0]["lon"], float), "resolved longitude is numeric")

    placed2, unresolved2 = enrich_with_geo(
        [{"camera_id": "CAM-A-PERIMETER"}, {"camera_id": "CAM-GONE"}], CAMERAS
    )
    check(len(placed2) == 1 and unresolved2 == 1,
          "an unknown camera is COUNTED as unplaceable, never silently dropped")
    check(unresolved2 == 1 and len(placed2) + unresolved2 == 2,
          "placed + unplaceable always reconciles to the input count")

    # ------------------------------------------------------------------
    header("8. GEOJSON EXPORT (Sector HQ staff maps / QGIS)")
    # ------------------------------------------------------------------
    nodes_geo = nodes_geojson(nodes)
    check(nodes_geo["type"] == "FeatureCollection", "nodes export as a FeatureCollection")
    check(len(nodes_geo["features"]) == 3, "every outpost becomes a Feature")
    node_a_feature = next(f for f in nodes_geo["features"]
                          if f["properties"]["node_id"] == "BOP-SECTOR-A")
    check(node_a_feature["geometry"]["type"] == "Point", "outposts export as Points")
    check(node_a_feature["geometry"]["coordinates"] == [
        CAMERAS["Channel 1"]["lon"], CAMERAS["Channel 1"]["lat"],
    ], "GeoJSON axis order is [longitude, latitude] - the classic export bug")
    check(node_a_feature["properties"]["cameras"] == 2,
          "outpost feature carries its camera count")

    coverage_geo = coverage_geojson(snap["coverage_cones"])
    check(coverage_geo["features"][0]["geometry"]["type"] == "Polygon",
          "coverage cones export as Polygons")
    ring = coverage_geo["features"][0]["geometry"]["coordinates"][0]
    check(len(ring) >= 3 and ring[0] == ring[-1],
          "the polygon ring is explicitly closed, as GeoJSON requires")

    alerts_geo = alerts_geojson(snap["alerts"], snap["clusters"])
    check(len(alerts_geo["features"]) == 2 + len(snap["clusters"]),
          "alerts export as points plus one feature per cluster")
    text = dumps_geojson(alerts_geo)
    check(json.loads(text)["type"] == "FeatureCollection",
          "exported GeoJSON round-trips through json.loads")

    # ------------------------------------------------------------------
    header("9. OFFLINE RENDER (the air-gap guarantee)")
    # ------------------------------------------------------------------
    svg = render_tactical_map(nodes, snap["alerts"], snap["clusters"], title="SIH Test Sector")
    check(isinstance(svg, str) and svg.startswith("<!DOCTYPE html>"),
          "map renders as a complete standalone HTML document")
    # One SVG for the wide map, plus one coverage inset per covered outpost.
    covered_nodes = [n for n in nodes if n["coverage"]]
    check(svg.count("<svg") == 1 + len(covered_nodes)
          and svg.count("</svg>") == svg.count("<svg"),
          f"one SVG root for the map plus one per coverage inset "
          f"({svg.count('<svg')} total, balanced)")
    check(f"aria-label='{html_module.escape('SIH Test Sector')}'" in svg,
          "the wide map is identifiable, so a caller can target it")
    check("<script" not in svg.lower(),
          "map ships no script tags - nothing executable in the operator's browser")
    check("SIH Test Sector" in svg, "the map title is rendered")
    check("BOP-SECTOR-A" in svg and "CHECKPOST-CHARLIE" in svg,
          "every outpost is labelled on the map")
    check("CRITICAL" in svg, "the map legend names the CRITICAL severity band")
    check("span" in svg and "km" in svg, "map reports the ground span it covers")
    check("m/px" in svg, "map reports its own scale, so on-screen distances are usable")
    check(">N<" in svg, "map carries a north arrow")

    # The air-gap assertion: nothing the browser must fetch.
    fetch_markers = ["src=", "<link", "@import", "url(", "<iframe", "fetch("]
    found = [marker for marker in fetch_markers if marker.lower() in svg.lower()]
    check(not found, f"map fetches NOTHING from the network (found: {found})")
    hosts = ["cdn.", "unpkg", "jsdelivr", "mapbox", "carto", "tile.openstreetmap",
             "openstreetmap", "googleapis", "leaflet", "plotly", "deck.gl"]
    hit_hosts = [host for host in hosts if host in svg.lower()]
    check(not hit_hosts, f"map references no tile provider or CDN (found: {hit_hosts})")
    check(svg.count("http") == svg.count("http://www.w3.org/2000/svg"),
          "the only URL in the map is the SVG namespace declaration, not a fetchable link")

    # Detail of the newest incident must reach the operator, not just a dot.
    check("UNKNOWN | CROSSING" in svg, "cluster tooltips carry the incident detail")

    # XSS / injection: labels come from the registry and could be operator-edited.
    hostile = [{
        "node_id": "<script>alert(1)</script>", "name": "Evil", "lat": 31.6, "lon": 74.8,
        "sector": "X", "state": "ONLINE", "cameras": [], "coverage": [],
        "coverage_range_km": 0.0, "strength": None,
    }]
    hostile_svg = render_tactical_map(hostile)
    check("<script>alert(1)</script>" not in hostile_svg,
          "a hostile registry label cannot inject markup into the map")
    check("&lt;script&gt;" in hostile_svg, "the hostile label is HTML-escaped instead")

    empty_svg = render_tactical_map([])
    check("No geo-located outposts" in empty_svg,
          "an empty grid renders an explanatory empty state, not a blank canvas")
    no_incidents = render_tactical_map(nodes)
    check(no_incidents.count("<svg") == 1 + len(covered_nodes),
          "the map renders with outposts but zero incidents")
    check("incident(s) here" not in no_incidents,
          "an outpost with no incidents on it claims none")

    # --- local coverage detail -------------------------------------------
    # Real outposts are tens of km apart while a camera watches a few hundred
    # metres, so on the wide map every cone is a sub-pixel dot. The insets redraw
    # each outpost's cones at their OWN scale so the flagship feature stays visible.
    from modules.gis import render_coverage_inset

    inset = render_coverage_inset(nodes, width=1080)
    check(inset.count("<svg") == len(covered_nodes),
          f"one coverage inset per covered outpost ({len(covered_nodes)})")
    check("each panel at its own true scale" in inset,
          "the inset panel states that each panel is independently scaled")
    for node in covered_nodes:
        check(node["node_id"] in inset, f"{node['node_id']} appears in the coverage inset")
    check(inset.count("1.2 km") >= 1 or "1.20 km" in inset,
          "the inset reports the outpost's maximum cone range")
    check(inset.count("km</text>") + inset.count(" m</text>") >= len(covered_nodes),
          "every inset panel carries its own scale bar, so on-screen sizes are usable")
    check(render_coverage_inset([]) == "",
          "an outpost with no coverage cone produces no inset rather than an empty panel")
    check(gis.render_coverage_inset([{"node_id": "X", "lat": None, "lon": None,
                                      "coverage": []}]) == "",
          "a node without geo data is skipped rather than crashing the inset")

    # An incident ON an outpost is counted on the outpost's own label instead of
    # drawn as a colliding second label.
    on_asset = render_tactical_map(nodes, snap["alerts"], snap["clusters"])
    check("incident(s) here" in on_asset,
          "incidents landing on an outpost are counted on that outpost's label")
    check(on_asset.count("INTRUSION_ALERT") >= 1,
          "the incident type is still available on the marker tooltip")

    # ------------------------------------------------------------------
    header("10. EDGE CASES AND MAP GEOMETRY")
    # ------------------------------------------------------------------
    hull = convex_hull([(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)])
    check(len(hull) == 4, "convex hull ignores interior points")
    check(convex_hull([(1, 1)]) == [(1, 1)], "hull of one point is that point")
    check(len(convex_hull([(1, 1), (1, 1), (2, 2)])) == 2,
          "hull de-duplicates coincident points")

    coords = [(31.9, 74.9), (31.4, 74.4), (31.65, 74.65)]   # ~55 km span
    wide = TacticalProjection(coords, width=900, height=300, padding=40)
    check(wide.span_km[0] > 40 and wide.span_km[1] > 40,
          "projection measures the real north and east span in km")
    xs = [wide.project(*c)[0] for c in coords]
    check(min(xs) >= 39 and max(xs) <= 861,
          "every point fits the viewport even in a wide, short canvas")

    same = TacticalProjection([(31.6, 74.8), (31.6, 74.8)])
    check(same.m_per_px > 0, "coincident points do not divide by zero")

    one_km = destination_point(31.6, 74.8, 90.0, 1000.0)
    proj2 = TacticalProjection([(31.6, 74.8), one_km], width=600, height=600, padding=60)
    px = abs(proj2.project(*one_km)[0] - proj2.project(31.6, 74.8)[0])
    close(px * proj2.m_per_px, 1000.0, 15.0,
          "a 1 km east-west vector stays 1 km on the canvas")

    # ------------------------------------------------------------------
    header("11. DEPLOYMENT AND PACKAGING CONTRACT")
    # ------------------------------------------------------------------
    defaults = deployment_config({})
    check(defaults["role"] == "edge", "an unconfigured node defaults to the edge role")
    check(defaults["channels"] is None, "no channels configured means no channel filter")
    check(defaults["telemetry_mode"] == "simulated",
          "an unconfigured node defaults to the broker-free simulated link")
    check(defaults["packet_budget_kb"] == 10, "the default packet budget is 10 KB")
    check(defaults["data_dir"] == ".", "data defaults to the working directory")

    custom = deployment_config({
        "IBVAP_ROLE": "c2",
        "IBVAP_NODE_ID": "SECTOR-HQ",
        "IBVAP_CHANNELS": "Channel 1, Channel 3 ,",
        "IBVAP_TELEMETRY": "MQTT",
        "IBVAP_MQTT_HOST": "broker",
        "IBVAP_MQTT_PORT": "8883",
        "IBVAP_LINK_PROFILE": "degraded_satcom",
        "IBVAP_PACKET_BUDGET_KB": "4",
        "IBVAP_GRID": "true",
        "IBVAP_CONF_THRESHOLD": "0.55",
        "IBVAP_DATA_DIR": "/app/data",
    })
    check(custom["role"] == "c2", "IBVAP_ROLE selects the command role")
    check(custom["channels"] == ["Channel 1", "Channel 3"],
          "a comma list of channels is parsed and trimmed")
    check(custom["telemetry_mode"] == "mqtt", "the transport name is case-insensitive")
    check(custom["mqtt_port"] == 8883 and isinstance(custom["mqtt_port"], int),
          "the MQTT port is parsed as an integer")
    check(custom["link_profile"] == "DEGRADED_SATCOM",
          "the link profile is upper-cased to match LINK_PROFILES")
    check(custom["packet_budget_kb"] == 4, "the packet budget is honoured")
    check(custom["grid"] is True, "IBVAP_GRID=true enables the full camera grid")
    check(custom["conf_threshold"] == 0.55, "the detector threshold is configurable")
    check(custom["data_dir"] == "/app/data", "the data directory is configurable")

    broken = deployment_config({
        "IBVAP_MQTT_PORT": "not-a-number",
        "IBVAP_PACKET_BUDGET_KB": "",
        "IBVAP_CONF_THRESHOLD": "4.0",
        "IBVAP_ROLE": "overlord",
        "IBVAP_GRID": "maybe",
    })
    check(broken["mqtt_port"] == 1883, "a malformed port falls back to the default")
    check(broken["packet_budget_kb"] == 10, "an empty value falls back to the default")
    check(broken["conf_threshold"] <= 0.95,
          "an out-of-range confidence is clamped, so a fat-fingered env var cannot "
          "silence or spam the detector")
    check(broken["role"] == "edge", "an unknown role falls back to edge rather than failing open")
    check(broken["grid"] is False, "an unparsable boolean is treated as false")

    # --- packaging files exist and agree with the code --------------------
    here = os.path.dirname(os.path.abspath(__file__))
    dockerfile = os.path.join(here, "Dockerfile")
    compose = os.path.join(here, "docker-compose.yml")
    ignore = os.path.join(here, ".dockerignore")
    broker_conf = os.path.join(here, "deploy", "mosquitto.conf")
    daemon = os.path.join(here, "run_edge_daemon.py")
    for label, path in (
        ("Dockerfile", dockerfile), ("docker-compose.yml", compose),
        (".dockerignore", ignore), ("deploy/mosquitto.conf", broker_conf),
        ("run_edge_daemon.py", daemon),
    ):
        check(os.path.exists(path), f"{label} is present for containerized deployment")

    docker_text = open(dockerfile, encoding="utf-8").read()
    check("USER " in docker_text, "the image runs as a non-root user, not as root")
    check("HEALTHCHECK" in docker_text,
          "the image declares a HEALTHCHECK, so an orchestrator can detect a dead outpost")
    check("libgl1" in docker_text,
          "the image installs OpenCV's shared libraries (libGL) - the classic "
          "containerized-OpenCV failure")
    check("run_edge_daemon.py" in docker_text,
          "the default container command is the headless agent, not a UI")
    check("VOLUME" in docker_text,
          "runtime state is a volume, so a container restart cannot erase the log")

    compose_text = open(compose, encoding="utf-8").read()
    check("mosquitto" in compose_text, "the stack provides an MQTT broker")
    check("condition: service_healthy" in compose_text,
          "edge nodes wait for a healthy broker instead of crash-looping on startup")
    check("./models:/app/models:ro" in compose_text,
          "model weights are mounted read-only rather than baked into the image")
    check(compose_text.count("IBVAP_NODE_ID") >= 4,
          "each outpost declares its own node identity")
    check("DEGRADED_SATCOM" in compose_text,
          "the stack exercises the worst link profile, not just the happy path")
    check("listener 1883" in open(broker_conf, encoding="utf-8").read(),
          "the broker listens on the port the edge nodes publish to")
    check("SECURITY" in open(broker_conf, encoding="utf-8").read(),
          "the broker config documents that anonymous access is demo-only")

    # The contract that actually breaks deployments: a compose file setting an
    # environment variable the application never reads.
    import re as _re

    recognized = {"IBVAP_" + key for key in DEPLOYMENT_ENV_KEYS}
    declared = set(_re.findall(r"IBVAP_[A-Z_]+", compose_text))
    unknown_env = sorted(declared - recognized)
    check(not unknown_env,
          f"every IBVAP_* variable in docker-compose.yml is read by the code "
          f"(unrecognised: {unknown_env})")
    undeclared = sorted(recognized - declared)
    check(len(undeclared) < len(recognized),
          f"the compose stack exercises the deployment config "
          f"(unused by compose: {undeclared})")
    check(len(declared) >= 8,
          f"the compose stack provisions the deployment via environment "
          f"({len(declared)} IBVAP_* variables declared)")

    daemon_text = open(daemon, encoding="utf-8").read()
    check("--dry-run" in daemon_text,
          "the edge agent can validate its wiring without a vision stack")
    check("SIGTERM" in daemon_text,
          "the edge agent shuts down cleanly on SIGTERM (container stop)")
    check("heartbeat" in daemon_text.lower(),
          "the edge agent beacons liveness, so a dead outpost is visible at HQ")
    check("modules/pipeline" in daemon_text or "from modules.pipeline import" in daemon_text,
          "the edge agent runs the SAME pipeline module as the dashboard")

    # ------------------------------------------------------------------
    header("GIS SNAPSHOT SAMPLE (what the dashboard renders)")
    # ------------------------------------------------------------------
    for key in ("nodes", "alerts", "clusters", "summary", "coverage_cones",
                "coverage_gaps", "unresolved_records", "intercept", "distance_matrix",
                "outpost_count", "response_speed_kmh"):
        check(key in snap, f"snapshot exposes '{key}' for the GIS tab")
    if snap["intercept"]:
        print(f"    intercept: {snap['intercept']['sentence']}")
    print(f"    clusters: {[(c.severity, c.count) for c in snap['clusters']]}")

    # ------------------------------------------------------------------
    header("PHASE 7 RESULT")
    # ------------------------------------------------------------------
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for item in failures:
            print(f"    - {item}")
        print("=" * 74)
        return 1

    print("  ALL CHECKS PASSED - the command map is geographic and genuinely offline:")
    print("    * outposts, camera coverage cones and incident clusters on one picture")
    print("    * nearest-responder ranking with an honest straight-line ETA lower bound")
    print("    * incidents outside every camera cone flagged as coverage gaps")
    print("    * the rendered map fetches nothing - no tiles, no CDN, no API key")
    print("    * GeoJSON export (correct [lon, lat] axis order) for Sector HQ staff maps")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
