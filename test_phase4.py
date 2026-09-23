"""
IBVAP PHASE 4 VERIFICATION: LOW-BANDWIDTH TELEMETRY LINK
=========================================================

Verifies the satellite/cellular telemetry contract without needing a broker,
OpenCV, or the detection models:

  1. Payload budget    - every packet stays under the 10 KB hard limit, and the
                         evidence snapshot is a real, decodable JPEG.
  2. Graceful degrade  - under a tiny budget the snapshot shrinks, then drops,
                         while the metadata alert still fits.
  3. Store-and-forward - during a total outage nothing is lost; the backlog
                         drains in FIFO order once the link returns.
  4. Lossy field link  - satellite packet loss produces retransmissions, not
                         silent data loss.
  5. Bounded buffer    - queue overflow discards the OLDEST telemetry only.
  6. Dead-lettering    - an unrecoverable outage eventually drops packets and
                         records why, rather than growing without bound.

Run:  python test_phase4.py
"""

import base64
import io
import os
import sys

import numpy as np

from modules.nodes import get_camera
from modules.telemetry import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    LINK_PROFILES,
    LoopbackTransport,
    SimulatedTransport,
    SnapshotEncoder,
    TelemetryPublisher,
)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
failures = []


def check(condition, message):
    print(f"{PASS if condition else FAIL} {message}")
    if not condition:
        failures.append(message)
    return condition


def synthetic_frame(width=1920, height=1080, noisy=True):
    """Builds a BGR frame resembling a CCTV view (noise compresses worst-case)."""
    if noisy:
        base = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        # A brighter rectangle where a plate/vehicle ROI would sit.
        base[500:620, 800:1000] = 200
        return base
    gradient = np.linspace(40, 210, width, dtype=np.uint8)
    row = np.tile(gradient, (height, 1))
    return np.dstack([row, row, row]).astype(np.uint8)


def make_event(event_type="INTRUSION_ALERT", **overrides):
    event = {
        "event_type": event_type,
        "track_id": 17,
        "category": "human",
        "class_name": "person",
        "confidence": 0.87,
        "status": "UNKNOWN_INTRUDER",
        "zone": "Sector B Tripwire",
        "direction": "INBOUND (Southbound)",
        "identity": "UNKNOWN PERSON",
        "location": "(412, 295)",
    }
    event.update(overrides)
    return event


def header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main():
    print("=" * 72)
    print("  IBVAP PHASE 4 VERIFICATION: LOW-BANDWIDTH TELEMETRY LINK")
    print("=" * 72)

    camera = get_camera("Channel 2")
    frame = synthetic_frame()
    bbox = (380, 250, 470, 520)

    # ------------------------------------------------------------------
    header("1. PAYLOAD BUDGET + REAL EVIDENCE SNAPSHOT")
    # ------------------------------------------------------------------
    backend = SnapshotEncoder.available_backend()
    check(backend is not None, f"JPEG encoder backend detected: {backend}")

    publisher = TelemetryPublisher(
        transport=LoopbackTransport(),
        max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES,
    )

    packet = publisher.build_packet(make_event(), camera, frame=frame, bbox=bbox)
    payload_bytes = packet["payload"]
    meta = packet["meta"]

    print(f"    budget={DEFAULT_MAX_PAYLOAD_BYTES} B | packet={meta['bytes']} B "
          f"| snapshot={meta.get('snapshot')}")
    check(len(payload_bytes) <= DEFAULT_MAX_PAYLOAD_BYTES,
          f"packet fits the 10 KB budget ({len(payload_bytes)} B)")
    check(not packet["over_budget"], "packet not flagged over budget")
    check(meta.get("snapshot") is not None, "evidence snapshot attached")

    # Decode the transmitted crop to prove it is a usable image, not garbage.
    import json
    from PIL import Image

    decoded = json.loads(payload_bytes.decode("utf-8"))
    snap = decoded.get("snapshot") or {}
    raw = base64.b64decode(snap["data"])
    image = Image.open(io.BytesIO(raw))
    image.load()
    check(raw[:2] == b"\xff\xd8", "snapshot bytes are a valid JPEG (SOI marker)")
    check(min(image.size) > 0, f"snapshot decodes to {image.size[0]}x{image.size[1]} px")
    check(decoded.get("utc", "").endswith("Z"), "packet carries a UTC timestamp")
    check(decoded.get("geo", {}).get("lat") is not None, "packet carries geo placement")
    check(decoded.get("camera_id") == camera["camera_id"],
          f"packet attributed to camera {camera['camera_id']}")
    check(decoded["severity"] == "CRITICAL", "intrusion labelled CRITICAL severity")

    # Noisy full-frame worst case must still fit.
    noisy_packet = publisher.build_packet(
        make_event("VEHICLE_ANPR", plate_text="K433ZR"), camera,
        frame=synthetic_frame(noisy=True), bbox=None
    )
    print(f"    worst-case full-frame incompressible packet: "
          f"{noisy_packet['meta']['bytes']} B")
    check(noisy_packet["meta"]["bytes"] <= DEFAULT_MAX_PAYLOAD_BYTES,
          "incompressible full-frame packet still fits the budget")

    # ------------------------------------------------------------------
    header("2. GRACEFUL DEGRADATION UNDER A STARVED BUDGET")
    # ------------------------------------------------------------------
    for budget in (2048, 1024, 512):
        tight = TelemetryPublisher(
            transport=LoopbackTransport(), max_payload_bytes=budget
        ).build_packet(make_event(), camera, frame=frame, bbox=bbox)
        tmeta = tight["meta"]
        snapshot_state = (
            f"{tmeta['snapshot']['width']}x{tmeta['snapshot']['height']}"
            f"@q{tmeta['snapshot']['quality']}"
            if tmeta["snapshot"] else "metadata-only"
        )
        print(f"    budget={budget} B -> packet={tmeta['bytes']} B | {snapshot_state}")
        check(tmeta["bytes"] <= budget, f"packet honours the {budget} B budget")

    # The metadata floor: the smallest alert this platform can produce. Below it
    # the packet is still transmitted (never drop a CRITICAL alert to save bytes)
    # but is explicitly flagged so HQ knows the budget was violated.
    floor = publisher.builder.metadata_floor_bytes(make_event())
    print(f"    metadata-only alert floor: {floor} B")
    check(floor <= 512, f"metadata-only alert is tiny ({floor} B)")

    below_floor = TelemetryPublisher(
        transport=LoopbackTransport(), max_payload_bytes=floor - 64
    ).build_packet(make_event(), camera, frame=frame, bbox=bbox)
    print(f"    budget={floor - 64} B (below floor) -> packet={below_floor['meta']['bytes']} B "
          f"| over_budget={below_floor['over_budget']} "
          f"| snapshot_dropped={below_floor['meta']['snapshot_dropped']}")
    check(below_floor["meta"]["snapshot_dropped"],
          "snapshot dropped rather than overflowing the budget")
    check(below_floor["over_budget"],
          "impossible budget is flagged instead of silently oversized")
    check(below_floor["meta"]["bytes"] <= floor + 8,
          "trimming strips optional fields down to the floor")

    # ------------------------------------------------------------------
    header("3. STORE-AND-FORWARD DURING A TOTAL LINK OUTAGE")
    # ------------------------------------------------------------------
    link = SimulatedTransport(
        inner=LoopbackTransport(), profile=LINK_PROFILES["SATELLITE"],
        sleep=False, seed=7,
    )
    store = TelemetryPublisher(transport=link, queue_capacity=200)

    link.forced_outage = True
    outage_events = 12
    statuses = [
        store.publish_event(make_event(track_id=i), camera, frame=frame, bbox=bbox)["status"]
        for i in range(outage_events)
    ]
    buffered = store.stats()["buffered"]
    print(f"    outage: {outage_events} events published -> {buffered} buffered, "
          f"0 lost (all statuses BUFFERED: {all(s == 'BUFFERED' for s in statuses)})")
    check(all(s == "BUFFERED" for s in statuses), "every outage event was buffered")
    check(buffered >= outage_events, f"backlog holds all {outage_events} events")
    check(store.stats()["sent"] == 0, "nothing claimed as sent while offline")

    link.forced_outage = False
    delivered = store.drain_all()
    stats_after = store.stats()
    print(f"    link restored: {delivered} packets retransmitted, "
          f"backlog={stats_after['buffered']}, dropped={stats_after['dropped']}")
    check(stats_after["buffered"] == 0, "backlog fully drained after link recovery")
    check(stats_after["dropped"] == 0, "no telemetry lost across the outage")

    delivered_seqs = [
        row["seq"] for row in reversed(store.recent_records(limit=200))
        if row["delivery"] == "SENT"
    ]
    check(delivered_seqs == sorted(delivered_seqs),
          "packets delivered in original FIFO order (no reordering)")

    # ------------------------------------------------------------------
    header("4. LOSSY FIELD LINK -> RETRANSMISSION, NOT SILENT LOSS")
    # ------------------------------------------------------------------
    lossy_link = SimulatedTransport(
        inner=LoopbackTransport(), profile=LINK_PROFILES["DEGRADED_SATCOM"],
        sleep=False, seed=11,
    )
    lossy = TelemetryPublisher(transport=lossy_link, max_retries=25, backoff_base_s=0.0)
    for i in range(25):
        lossy.publish_event(make_event(track_id=100 + i), camera, frame=frame, bbox=bbox)
        lossy.flush()
    lossy.drain_all()
    lstats = lossy.stats()
    print(f"    profile={lstats['link_label']} | attempts_failed="
          f"{lstats['send_failures']} | retransmissions={lstats['retransmissions']}")
    check(lstats["send_failures"] > 0, "degraded link did drop transmissions")
    check(lstats["sent"] + lstats["buffered"] + lstats["dropped"] == lstats["events"],
          "every event accounted for (sent + buffered + dropped == published)")
    check(lstats["retransmissions"] > 0, "buffered packets were retransmitted")

    # ------------------------------------------------------------------
    header("5. BOUNDED BUFFER - OLDEST TELEMETRY SHED FIRST")
    # ------------------------------------------------------------------
    small_link = SimulatedTransport(
        inner=LoopbackTransport(), profile=LINK_PROFILES["SATELLITE"], sleep=False, seed=3
    )
    small = TelemetryPublisher(transport=small_link, queue_capacity=5)
    small_link.forced_outage = True
    for i in range(20):
        small.publish_event(make_event(track_id=500 + i), camera, frame=frame, bbox=bbox)
    sstats = small.stats()
    queued_seqs = [p["seq"] for p in small.queue_snapshot()]
    print(f"    capacity=5, published=20 -> buffered={sstats['buffered']}, "
          f"dropped={sstats['dropped']}, held seqs={queued_seqs}")
    check(sstats["buffered"] <= 5, "buffer never exceeds its configured capacity")
    check(sstats["dropped"] == 15, "overflow dropped the 15 oldest packets")
    check(queued_seqs == sorted(queued_seqs) and min(queued_seqs) > 15,
          "the newest telemetry is the telemetry that survived")

    # ------------------------------------------------------------------
    header("6. RETRY EXHAUSTION (LOSSY LINK) VS EXPIRY (DEAD LINK)")
    # ------------------------------------------------------------------
    # 6a. Link is up but hostile: a packet whose retries run out is dead-lettered
    #     with the reason recorded, rather than retried forever.
    hostile = SimulatedTransport(
        inner=LoopbackTransport(), profile=LINK_PROFILES["DEGRADED_SATCOM"],
        sleep=False, seed=23,
    )
    doomed = TelemetryPublisher(
        transport=hostile, max_retries=2, backoff_base_s=0.0, queue_capacity=80,
        max_packet_age_s=900.0,
    )
    for i in range(30):
        doomed.publish_event(make_event(track_id=700 + i), camera)
        doomed.drain_all()
    dstats = doomed.stats()
    reasons = {entry["reason"] for entry in doomed.dead_letters}
    print(f"    6a lossy link, max_retries=2 -> dropped={dstats['dropped']}, "
          f"dead_letters={dstats['dead_letters']}, reasons={reasons or '{}'}")
    check(dstats["dropped"] > 0, "packets are dropped once retries are exhausted")
    check(dstats["dead_letters"] == dstats["dropped"],
          "every drop is recorded in the dead-letter store")
    check("retries_exhausted" in reasons, "drop reason recorded as retries_exhausted")

    # 6b. Link is dead for good: the backlog is RETAINED (never silently lost),
    #     and only becomes expendable once the telemetry itself goes stale.
    fake_now = [1_000_000.0]
    dead_link = SimulatedTransport(
        inner=LoopbackTransport(), profile=LINK_PROFILES["SATELLITE"], sleep=False
    )
    stranded = TelemetryPublisher(
        transport=dead_link, clock=lambda: fake_now[0],
        max_packet_age_s=60.0, queue_capacity=50,
    )
    dead_link.forced_outage = True
    for i in range(3):
        stranded.publish_event(make_event(track_id=900 + i), camera)
    stranded.drain_all()
    held = stranded.stats()
    print(f"    6b total outage, 0 s elapsed -> buffered={held['buffered']}, "
          f"dropped={held['dropped']} (backlog retained, retries not burned)")
    check(held["buffered"] == 3 and held["dropped"] == 0,
          "a dead link retains telemetry instead of discarding it")

    fake_now[0] += 61.0  # Telemetry is now older than max_packet_age_s.
    stranded.drain_all()
    expired = stranded.stats()
    expire_reasons = {entry["reason"] for entry in stranded.dead_letters}
    print(f"    6b +61 s stale -> buffered={expired['buffered']}, "
          f"dropped={expired['dropped']}, reasons={expire_reasons}")
    check(expired["buffered"] == 0, "stale telemetry is released from the buffer")
    check(expired["dropped"] == 3, "stale telemetry counted as dropped, not sent")
    check("expired" in expire_reasons,
          "expiry recorded distinctly from retry exhaustion")

    # ------------------------------------------------------------------
    header("7. LINK PROFILE COVERAGE + BUDGET UTILISATION")
    # ------------------------------------------------------------------
    for name, profile in LINK_PROFILES.items():
        print(f"    {name:<16} {profile.describe()}")

    report = TelemetryPublisher(transport=LoopbackTransport())
    for i, event_type in enumerate(
        ["INTRUSION_ALERT", "VEHICLE_ANPR", "AUTHORIZED_PATROL",
         "FLAGGED_FOR_MANUAL_REVIEW"] * 3
    ):
        report.publish_event(
            make_event(event_type, track_id=i), camera, frame=frame, bbox=bbox
        )
    rstats = report.stats()
    print(f"    {rstats['sent']} packets sent | avg "
          f"{rstats['avg_payload_bytes']} B | utilisation "
          f"{rstats['budget_utilisation']}% of {rstats['max_payload_bytes']} B budget")
    check(rstats["over_budget"] == 0, "no packet ever exceeded the byte budget")
    check(rstats["avg_payload_bytes"] < rstats["max_payload_bytes"],
          "average packet leaves headroom on the link")
    check(rstats["avg_payload_bytes"] > 0, "payloads carry real evidence bytes")

    # ------------------------------------------------------------------
    header("DELIVERED PACKET SAMPLE")
    # ------------------------------------------------------------------
    sample = report.recent_records(limit=6)
    print(f"    {'seq':>4} {'event_type':<26} {'sev':<9} {'topic':<10} {'bytes':>7} "
          f"{'snap':>6} {'delivery':<9}")
    for row in sample:
        print(f"    {row['seq']:>4} {row['event_type']:<26} {row['severity']:<9} "
              f"{row['topic_kind']:<10} {row['bytes']:>7} {row['snapshot_bytes']:>6} "
              f"{row['delivery']:<9}")

    # ------------------------------------------------------------------
    header("8. END-TO-END CONTRACT: EVENT -> LOGGER -> UPLINK -> HQ")
    # ------------------------------------------------------------------
    # Mirrors app.py: a fence/ANPR event is logged locally AND transmitted. The
    # dashboard reads specific dict keys and dataframe columns, so this section
    # pins that contract. (The fence object itself is not used here because it
    # imports OpenCV, which is optional for this verification script.)
    from modules.logger import EventLogger

    alert = {
        "timestamp": "2026-09-22 10:15:00",
        "event_type": "INTRUSION_ALERT",
        "track_id": 42,
        "category": "human",
        "class_name": "person",
        "confidence": 0.83,
        "zone": camera["fence"]["zone_name"],
        "direction": "OUTBOUND (Northbound)",
        "identity": "UNKNOWN PERSON",
        "is_authorized": False,
        "location": "(514, 300)",
        "status": "UNKNOWN_INTRUDER",
    }

    e2e_link = SimulatedTransport(
        inner=LoopbackTransport(), profile=LINK_PROFILES["4G"], sleep=False, seed=5
    )
    e2e = TelemetryPublisher(transport=e2e_link, max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES)
    e2e_logger = EventLogger(csv_path="sample_videos/_phase4_e2e.csv")
    e2e_logger.clear()
    e2e_logger.set_context(node_id=camera["node_id"], camera_id=camera["camera_id"])

    e2e_logger.log_event(
        event_type=alert["event_type"],
        track_id=alert["track_id"],
        category=alert["category"],
        class_name=alert["class_name"],
        confidence=alert["confidence"],
        status=alert["status"],
        details=f"{alert['identity']} | {alert['direction']} at {alert['location']}",
        timestamp=alert["timestamp"],
    )
    result = e2e.publish_event(alert, camera, frame=frame, bbox=bbox)
    print(f"    fence alert -> logger row + uplink packet {result['event_id']} "
          f"({result['bytes']} B, {result['status']})")
    check(result["status"] == "SENT", "fence alert transmitted on a healthy link")
    check(result["snapshot_bytes"] > 0, "fence alert carried an evidence crop")

    logged = e2e_logger.get_dataframe()
    check(len(logged) == 1, "local audit log has exactly one row")
    check(logged.iloc[0]["camera_id"] == camera["camera_id"],
          "audit row attributed to the active camera")
    check(logged.iloc[0]["utc_timestamp"].endswith("Z"),
          "audit row carries a UTC timestamp for HQ reconciliation")

    # Outage: operator toggles the sidebar switch, alerts keep arriving.
    e2e_link.forced_outage = True
    for i in range(4):
        e2e.publish_event(make_event(track_id=60 + i), camera, frame=frame, bbox=bbox)
        e2e_logger.log_event(
            event_type="INTRUSION_ALERT", track_id=60 + i, status="UNKNOWN_INTRUDER"
        )
    backlog = e2e.queue_snapshot()
    required_backlog_keys = {"event_id", "event_type", "severity", "utc",
                             "bytes", "attempts", "age_s"}
    check(len(backlog) == 4, "all four alerts held in the store-and-forward backlog")
    check(required_backlog_keys.issubset(backlog[0].keys()),
          "backlog rows expose the columns the uplink tab renders")

    e2e_link.forced_outage = False
    e2e.drain_all()
    stream = e2e.recent_records(limit=10)
    required_stream_keys = {"utc", "seq", "event_type", "severity", "topic_kind",
                            "bytes", "snapshot_bytes", "delivery", "attempts",
                            "retransmit", "queue_depth"}
    check(required_stream_keys.issubset(stream[0].keys()),
          "packet stream exposes every column the uplink tab renders")
    # The stream is a delivery HISTORY, so a packet appears once as BUFFERED and
    # again as SENT. Every event's most recent record must be SENT, and nothing
    # may be sitting in the dead-letter store.
    # recent_records() is newest-first, so walk it oldest-first to let the most
    # recent state of each event win.
    latest_by_event = {}
    for row in reversed(stream):
        latest_by_event[row["event_id"]] = row["delivery"]
    check(all(state == "SENT" for state in latest_by_event.values()),
          "every event's final delivery state is SENT after link recovery")
    check(len(latest_by_event) == 5, "all five distinct events appear in the stream")
    estats = e2e.stats()
    check(estats["buffered"] == 0 and estats["dropped"] == 0,
          "no telemetry lost across the operator outage")
    check(estats["sent"] == len(logged) + 4,
          "uplink packet count matches the local audit trail (HQ parity)")

    # Live operator retune: tightening the byte budget must take effect at once
    # without discarding the delivery statistics.
    sent_before_retune = e2e.stats()["sent"]
    e2e.set_budget(2048)
    retuned = e2e.publish_event(make_event(track_id=77), camera, frame=frame, bbox=bbox)
    print(f"    budget retuned live to 2048 B -> next packet {retuned['bytes']} B, "
          f"snapshot={retuned['snapshot_bytes']} B")
    check(retuned["bytes"] <= 2048, "retuned budget applies to the very next packet")
    check(e2e.stats()["sent"] == sent_before_retune + 1,
          "retuning the budget preserves the uplink statistics")

    if e2e_logger.csv_path and os.path.exists(e2e_logger.csv_path):
        os.remove(e2e_logger.csv_path)

    # ------------------------------------------------------------------
    header("PHASE 4 RESULT")
    # ------------------------------------------------------------------
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for item in failures:
            print(f"    - {item}")
        print("=" * 72)
        return 1

    print("  ALL CHECKS PASSED - telemetry link is production-shaped:")
    print(f"    * every packet <= {DEFAULT_MAX_PAYLOAD_BYTES} bytes with a real JPEG crop")
    print("    * zero telemetry loss across a total link outage")
    print("    * FIFO retransmission with exponential backoff")
    print("    * bounded buffer sheds oldest data, never the newest")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
