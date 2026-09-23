#!/usr/bin/env python3
"""
AnantaNetra / IBVAP headless edge agent.

Why this exists
---------------
The dashboard is an operator tool: it needs a screen, a browser and a person
sitting in front of it. A Border Out Post needs the opposite - the analytics
running unattended on a small edge box, streaming structured alerts to Sector HQ
and writing a local audit trail, with nobody watching a screen at all.

This daemon runs the SAME pipeline as the dashboard (`modules/pipeline.py`), the
same telemetry uplink and the same event sink. Nothing about the analytics
differs between the two deployments; only the presence of a UI does. That is the
whole point: an alert cannot be "dashboard-only".

What it does
------------
  1. Reads its deployment from IBVAP_* environment variables (or CLI flags).
  2. Starts the threaded ingest grid for its channels.
  3. Routes every event to the local CSV log AND the telemetry uplink.
  4. Emits a periodic HEARTBEAT so Sector HQ can tell "quiet sector" from
     "dead outpost".
  5. Shuts down cleanly on SIGINT/SIGTERM, flushing the store-and-forward queue.

Usage
-----
    python run_edge_daemon.py --channels "Channel 1" --duration 30
    python run_edge_daemon.py --dry-run            # no vision stack needed
    IBVAP_TELEMETRY=mqtt IBVAP_MQTT_HOST=broker python run_edge_daemon.py

--dry-run exercises the full wiring (registry, uplink, queue, logging,
heartbeats, shutdown) with synthetic frames and no inference, so the deployment
path can be validated on a box with no OpenCV, no model weights and no camera.
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime

from modules.ingest import EventBus, StreamManager, source_from_camera
from modules.logger import EventLogger
from modules.nodes import (
    CAMERAS,
    channel_keys,
    deployment_config,
    get_camera,
    node_geo,
    resolve_video_source,
)
from modules.pipeline import VisionModels, heartbeat_event, route_event
from modules.telemetry import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    LINK_PROFILES,
    TelemetryPublisher,
    build_transport,
)


def parse_args(argv=None):
    config = deployment_config()
    parser = argparse.ArgumentParser(
        description="IBVAP headless edge agent (no browser, no UI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--channels", default=None,
        help="Comma-separated registry keys, e.g. 'Channel 1,Channel 2'. "
             "Defaults to IBVAP_CHANNELS, else every registered camera.",
    )
    parser.add_argument("--grid", action="store_true", default=config["grid"],
                        help="Start every registered camera concurrently.")
    parser.add_argument("--telemetry", default=config["telemetry_mode"],
                        choices=["simulated", "mqtt"],
                        help="Uplink transport.")
    parser.add_argument("--mqtt-host", default=config["mqtt_host"])
    parser.add_argument("--mqtt-port", type=int, default=config["mqtt_port"])
    parser.add_argument("--link-profile", default=config["link_profile"],
                        choices=list(LINK_PROFILES.keys()),
                        help="Modelled field link characteristics.")
    parser.add_argument("--budget-kb", type=int, default=config["packet_budget_kb"],
                        help="Hard ceiling for one telemetry packet.")
    parser.add_argument("--data-dir", default=config["data_dir"],
                        help="Where alerts.csv and the biometric index live.")
    parser.add_argument("--node-id", default=config["node_id"])
    parser.add_argument("--conf", type=float, default=config["conf_threshold"])
    parser.add_argument("--heartbeat-s", type=float, default=60.0,
                        help="Seconds between liveness beacons (0 disables).")
    parser.add_argument("--status-s", type=float, default=10.0,
                        help="Seconds between status lines on the console log.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Stop after this many seconds (0 = run forever).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate wiring only: no vision stack, no cameras.")
    parser.add_argument("--no-anpr", action="store_true",
                        help="Skip the ANPR/OCR stack (lighter memory footprint).")
    return parser.parse_args(argv)


def resolve_channels(args) -> list:
    """
    Works out which registry keys to run.

    An unknown key is a configuration error worth failing on: silently running
    the wrong sector's camera is worse than not starting.
    """
    requested = None
    if args.channels:
        requested = [c.strip() for c in str(args.channels).split(",") if c.strip()]
    elif os.environ.get("IBVAP_CHANNELS"):
        requested = [c.strip() for c in os.environ["IBVAP_CHANNELS"].split(",") if c.strip()]

    if args.grid:
        return channel_keys()
    if not requested:
        return channel_keys()

    unknown = [key for key in requested if key not in CAMERAS]
    if unknown:
        raise SystemExit(
            f"Unknown channel(s): {unknown}. Registered: {channel_keys()}"
        )
    return requested


def build_models(args) -> VisionModels:
    """
    Loads the model stack. Only called when NOT in dry-run, so the daemon can be
    smoke-tested on a machine with no vision dependencies at all.
    """
    from modules.anpr import CascadedANPR
    from modules.detector import ObjectDetector
    from modules.enhancer import LowLightEnhancer
    from modules.fence import VirtualFence  # noqa: F401 - imported for parity/side effects
    from modules.frs import FaceIndex, build_frs
    from modules.pose import BEHAVIOR_ORDER, build_pose_engine
    from modules.watchlist import WatchlistDatabase

    detector = ObjectDetector(model_path="models/yolov8n.pt", device="cpu")
    anpr = None
    if not args.no_anpr:
        anpr = CascadedANPR(
            plate_model_path="models/licensePlateDetector.pt",
            device="cpu",
            ocr_conf_threshold=0.40,
            throttle_frames=15,
            min_vehicle_height=50,
        )
    enhancer = LowLightEnhancer(clip_limit=3.0)

    face_index = FaceIndex(db_path=os.path.join(args.data_dir, "frs_index.sqlite"))
    watchlist = WatchlistDatabase(face_index=face_index)
    frs, frs_note = build_frs(
        index=face_index, allow_non_biometric=False, confirm_frames=3, min_face_px=40
    )
    pose_engine, pose_note = build_pose_engine(
        model_path="models/yolov8n-pose.pt",
        device="cpu",
        enabled_behaviors=BEHAVIOR_ORDER,
    )
    print(f"[EDGE] identity: {frs_note}")
    print(f"[EDGE] behaviour: {pose_note}")
    return VisionModels(
        detector=detector, anpr=anpr, enhancer=enhancer,
        watchlist=watchlist, frs=frs, pose=pose_engine,
    )


class EdgeAgent:
    """The unattended outpost agent: ingest grid -> audit log + telemetry uplink."""

    def __init__(self, args):
        self.args = args
        self.channels = resolve_channels(args)
        self.cameras = [get_camera(key) for key in self.channels]
        self.logger = EventLogger(
            csv_path=os.path.join(args.data_dir, "alerts.csv"),
            node_id=args.node_id,
            camera_id=self.cameras[0]["camera_id"] if self.cameras else "CAM-UNKNOWN",
        )
        self.publisher = None
        self.manager = None
        self.running = False
        self.started_at = time.time()
        self.last_heartbeat = 0.0
        self.last_status = 0.0
        self.events_routed = 0

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        os.makedirs(self.args.data_dir, exist_ok=True)

        transport, note = build_transport(
            mode=self.args.telemetry,
            profile=self.args.link_profile,
            sleep=False,
            mqtt_kwargs={"host": self.args.mqtt_host, "port": int(self.args.mqtt_port)},
        )
        self.publisher = TelemetryPublisher(
            transport=transport, max_payload_bytes=self.args.budget_kb * 1024
        )
        print(f"[EDGE] uplink: {note}")

        self.manager = StreamManager(
            event_bus=EventBus(capacity=4000),
            stall_timeout_s=3.0,
            max_concurrent_analysis=2,
            watchdog_interval_s=0.5,
        )

        opts = {
            "conf_thresh": self.args.conf,
            "enable_clahe": False,
            "enable_watchlist": True,
            "enable_frs": True,
            "enable_pose": True,
            "tripwire_y": None,
        }

        if self.args.dry_run:
            print("[EDGE] DRY RUN - no vision stack, no cameras; wiring only.")
        else:
            models = build_models(self.args)

        for camera in self.cameras:
            fence_cfg = camera.get("fence")
            if not self.args.dry_run:
                self.manager.add_camera(
                    camera,
                    process_fn=self._make_process(models, camera, fence_cfg, opts),
                )
            print(
                f"[EDGE] channel '{camera['camera_id']}' node={camera['node_id']} "
                f"source={source_from_camera(camera).describe()}"
            )

        if not self.args.dry_run:
            started = self.manager.start_all()
            print(f"[EDGE] ingest grid started: {started} camera worker(s)")
        self.running = True

    @staticmethod
    def _make_process(models, camera, fence_cfg, opts):
        from modules.pipeline import build_analyser

        return build_analyser(models, camera, fence_cfg, opts)

    def stop(self) -> None:
        self.running = False
        if self.manager is not None:
            self.manager.stop_all()
        if self.publisher is not None:
            # Best effort: push whatever the link outage left in the buffer.
            delivered = self.publisher.drain_all()
            if delivered:
                print(f"[EDGE] flushed {delivered} buffered packet(s) on shutdown")
            stats = self.publisher.stats()
            print(
                f"[EDGE] telemetry: sent={stats['sent']} "
                f"buffered={stats['buffered']} dropped={stats['dropped']} "
                f"retransmits={stats['retransmissions']} "
                f"avg_packet={stats['avg_payload_bytes']}B"
            )
            try:
                self.publisher.transport.close()
            except Exception:
                pass

    # -- runtime ------------------------------------------------------------
    def _camera_for(self, event):
        camera_id = event.get("camera_id")
        for camera in self.cameras:
            if camera["camera_id"] == camera_id:
                return camera
        return self.cameras[0] if self.cameras else {}

    def tick(self) -> int:
        """Drains the grid, routes events, emits heartbeats. Returns events seen."""
        seen = 0
        if self.manager is not None:
            for event in self.manager.drain_events(max_items=200):
                camera = self._camera_for(event)
                frame = self.manager.peek_annotated(camera.get("camera_id")) if camera else None
                route_event(event, camera, frame, self.logger, publisher=self.publisher)
                self.events_routed += 1
                seen += 1

        now = time.time()
        if self.args.heartbeat_s and (now - self.last_heartbeat) >= self.args.heartbeat_s:
            self.last_heartbeat = now
            self.emit_heartbeat()

        if self.args.status_s and (now - self.last_status) >= self.args.status_s:
            self.last_status = now
            self.print_status(now)
        return seen

    def emit_heartbeat(self) -> None:
        """
        A liveness beacon. Without it, Sector HQ cannot distinguish a quiet sector
        from an outpost whose power, link or process has died - and 'no alert' is
        exactly what a successful intrusion looks like.
        """
        if not self.cameras or self.publisher is None:
            return
        camera = self.cameras[0]
        event = heartbeat_event(
            node_id=self.args.node_id,
            note=f"{len(self.cameras)} channel(s) monitored, "
                 f"{self.events_routed} event(s) routed",
            camera_id=camera["camera_id"],
        )
        route_event(event, camera, None, self.logger, publisher=self.publisher)
        self.events_routed += 1

    def print_status(self, now: float) -> None:
        uptime = now - self.started_at
        line = (
            f"[EDGE] up {uptime:.0f}s | events={self.events_routed} | "
            f"link={self.publisher.stats()['link_profile']} "
            f"buffered={self.publisher.stats()['buffered']}"
        )
        if self.manager is not None:
            summary = self.manager.summary()
            line += (
                f" | cams={summary['cameras']} live={summary['live']} "
                f"stalled={summary['stalled']} frames={summary['frames_analysed']}"
            )
        print(line)

    def run(self) -> int:
        self.start()
        deadline = self.started_at + self.args.duration if self.args.duration else None
        try:
            while self.running:
                self.tick()
                if deadline and time.time() >= deadline:
                    print("[EDGE] duration reached, shutting down")
                    break
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("[EDGE] interrupt received")
        finally:
            self.stop()
        print(f"[EDGE] stopped after {time.time() - self.started_at:.0f}s, "
              f"{self.events_routed} event(s) routed to log + uplink")
        return 0


def main(argv=None) -> int:
    args = parse_args(argv)

    def handle_signal(signum, _frame):
        print(f"[EDGE] signal {signum} received - stopping")
        raise KeyboardInterrupt

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, handle_signal)
            except (ValueError, OSError):  # not on the main thread / unsupported
                pass

    print(
        f"[EDGE] node={args.node_id} channels={resolve_channels(args)} "
        f"at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return EdgeAgent(args).run()


if __name__ == "__main__":
    sys.exit(main())
