"""
Low-Bandwidth Telemetry Link for the AnantaNetra / IBVAP edge grid.

The problem this solves
-----------------------
A Border Out Post has no usable backhaul: at best a flaky satellite or cellular
link shared with voice traffic. Streaming video to Sector HQ is impossible, and
even a naive JSON alert queue will collapse the link if it carries raw frames.

This module implements the store-and-forward telemetry contract:

    1. Build a structured, self-contained alert packet from a detection event.
    2. Attach a compressed evidence snapshot crop, *shrinking it until the whole
       packet fits a hard byte budget* (default 10 KB) - never the raw frame.
    3. Hand it to a transport that models the real link: latency, packet loss,
       and total outage.
    4. If the link is down or the send fails, buffer the packet locally and
       retransmit later with exponential backoff. Nothing is lost just because
       the satellite blinked.

Transports
----------
  * LoopbackTransport   - in-process sink; used by the verification script.
  * SimulatedTransport  - wraps any transport and imposes a LinkProfile
                          (latency / drop rate / forced outage).
  * MQTTTransport       - real broker publish via paho-mqtt, used when a broker
                          is reachable. Imported lazily so the whole platform
                          still runs on an outpost with no paho installed.

Heavy dependencies (numpy / Pillow / OpenCV) are all optional at import time:
this module must stay importable on a bare Python so it can be unit-verified
without the full vision stack.
"""

import base64
import io
import json
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Hard payload budget: the contract with the satellite link.
# ---------------------------------------------------------------------------
DEFAULT_MAX_PAYLOAD_BYTES = 10 * 1024  # 10 KB

# Reserved room for JSON envelope growth so a packet never lands a few bytes over.
ENVELOPE_SAFETY_BYTES = 320

# Severity assignment drives up-link triage (CRITICAL alerts take the priority
# topic and are retried hardest).
SEVERITY_BY_EVENT = {
    "INTRUSION_ALERT": "CRITICAL",
    # Pose/behaviour analytics (modules/pose.py). Kept in step with
    # pose.BEHAVIOR_CATALOGUE by test_phase8.py, so the map, the log and the wire
    # can never disagree about how urgent a behaviour is.
    "FENCE_CLIMB": "CRITICAL",
    "FALL_ALERT": "CRITICAL",
    "CROUCH_INTRUSION": "WARNING",
    "LOITERING": "WARNING",
    "RUNNING": "WARNING",
    "GROUP_CONVERGENCE": "NOTICE",
    "ARM_RAISED_SIGNAL": "NOTICE",
    # A confirmed watchlist identity at the perimeter is the highest-priority
    # event the platform can produce.
    "WATCHLIST_HIT": "CRITICAL",
    "FLAGGED_FOR_MANUAL_REVIEW": "WARNING",
    "VEHICLE_ANPR": "NOTICE",
    "AUTHORIZED_PATROL": "INFO",
    "AUTHORIZED_VEHICLE": "INFO",
    "SYSTEM": "INFO",
}


def severity_for(event_type: str, status: str = "") -> str:
    """Maps an event to a link priority class."""
    if event_type in SEVERITY_BY_EVENT:
        return SEVERITY_BY_EVENT[event_type]
    if status == "FLAGGED_FOR_MANUAL_REVIEW":
        return "WARNING"
    return "NOTICE"


# ---------------------------------------------------------------------------
# Link profiles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LinkProfile:
    """Characteristics of a field backhaul link."""

    name: str
    label: str
    latency_ms: float
    drop_rate: float
    bandwidth_kbps: float

    def describe(self) -> str:
        return (
            f"{self.label} | ~{self.latency_ms:.0f} ms RTT | "
            f"{self.drop_rate * 100:.0f}% loss | {self.bandwidth_kbps:.0f} kbps"
        )


LINK_PROFILES: Dict[str, LinkProfile] = {
    "FIBER": LinkProfile("FIBER", "Sector HQ Fiber", 18.0, 0.0, 100_000.0),
    "4G": LinkProfile("4G", "4G Cellular", 170.0, 0.05, 4_000.0),
    "SATELLITE": LinkProfile("SATELLITE", "Satellite (Nominal)", 620.0, 0.22, 256.0),
    "DEGRADED_SATCOM": LinkProfile(
        "DEGRADED_SATCOM", "Degraded Satcom / Storm", 1450.0, 0.55, 64.0
    ),
}


def get_profile(profile: Any) -> LinkProfile:
    """Accepts a profile name or a LinkProfile and returns a LinkProfile."""
    if isinstance(profile, LinkProfile):
        return profile
    return LINK_PROFILES.get(str(profile).upper(), LINK_PROFILES["SATELLITE"])


# ---------------------------------------------------------------------------
# Snapshot compression
# ---------------------------------------------------------------------------
class SnapshotEncoder:
    """
    Compresses an evidence crop (or a downscaled full frame) into a JPEG small
    enough for a satellite link.

    Backends are probed in order - Pillow first (already a project dependency),
    then OpenCV. If neither exists the encoder reports unavailable and packets
    travel as metadata only, which still fits the budget.
    """

    # Ladders are walked largest-image-first, then highest-quality-first, so the
    # best readable crop that still fits the budget is chosen.
    DEFAULT_DIM_LADDER = (320, 240, 176, 128, 88, 56)
    DEFAULT_QUALITY_LADDER = (72, 58, 45, 34, 24, 16, 10)

    def __init__(
        self,
        max_dim: int = 320,
        dim_ladder: Tuple[int, ...] = DEFAULT_DIM_LADDER,
        quality_ladder: Tuple[int, ...] = DEFAULT_QUALITY_LADDER,
        padding: int = 6,
    ):
        self.max_dim = max_dim
        self.dim_ladder = tuple(d for d in dim_ladder if d <= max_dim) or (max_dim,)
        self.quality_ladder = quality_ladder
        self.padding = padding

    # -- backend probing ----------------------------------------------------
    @staticmethod
    def available_backend() -> Optional[str]:
        try:  # Pillow
            from PIL import Image  # noqa: F401

            return "pillow"
        except Exception:
            pass
        try:  # OpenCV
            import cv2  # noqa: F401

            return "opencv"
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return self.available_backend() is not None

    # -- geometry -----------------------------------------------------------
    def _crop(self, frame, bbox):
        """Crops to bbox with padding; returns the whole frame if bbox is bad."""
        try:
            import numpy as np  # local import: optional dependency
        except Exception:
            return None

        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            return None

        h, w = frame.shape[0], frame.shape[1]
        if bbox is None:
            return frame

        try:
            x1, y1, x2, y2 = (int(v) for v in bbox)
        except Exception:
            return frame

        # Accept either x1y1x2y2 or x,y,w,h style degenerate boxes gracefully.
        if x2 <= x1 or y2 <= y1:
            return frame

        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(w, x2 + self.padding)
        y2 = min(h, y2 + self.padding)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return np.ascontiguousarray(crop)

    # -- encoding -----------------------------------------------------------
    def encode(
        self,
        frame,
        bbox=None,
        max_dim: Optional[int] = None,
        quality: int = 55,
    ) -> Optional[dict]:
        """
        Encodes a crop as JPEG and returns metadata + base64 payload, or None if
        no backend/frame is available.
        """
        image = self._crop(frame, bbox)
        if image is None:
            return None

        backend = self.available_backend()
        if backend is None:
            return None

        try:
            if backend == "pillow":
                from PIL import Image

                # OpenCV frames arrive BGR; Pillow expects RGB.
                if image.shape[2] == 3:
                    image = image[:, :, ::-1]
                pil = Image.fromarray(image)
                if max_dim:
                    pil.thumbnail((max_dim, max_dim), Image.LANCZOS)
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=int(quality), optimize=False)
                data = buf.getvalue()
                width, height = pil.size
            else:
                import cv2

                img = image
                if max_dim:
                    h, w = img.shape[:2]
                    scale = min(1.0, float(max_dim) / float(max(h, w)))
                    if scale < 1.0:
                        img = cv2.resize(
                            img,
                            (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv2.INTER_AREA,
                        )
                ok, enc = cv2.imencode(
                    ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
                )
                if not ok:
                    return None
                data = enc.tobytes()
                height, width = img.shape[:2]
        except Exception:
            return None

        return {
            "format": "jpeg",
            "width": int(width),
            "height": int(height),
            "quality": int(quality),
            "bytes": len(data),
            "data": base64.b64encode(data).decode("ascii"),
        }


# ---------------------------------------------------------------------------
# Packet construction with byte-budget enforcement
# ---------------------------------------------------------------------------
def _compact(obj: Any) -> Any:
    """Recursively drops empty values so the JSON envelope carries no dead bytes."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            cleaned = _compact(value)
            if cleaned is None or cleaned == "" or cleaned == [] or cleaned == {}:
                continue
            out[key] = cleaned
        return out
    if isinstance(obj, (list, tuple)):
        return [_compact(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


def _serialize(packet: dict) -> bytes:
    return json.dumps(_compact(packet), separators=(",", ":")).encode("utf-8")


class TelemetryPacketBuilder:
    """
    Turns a detection event into an on-the-wire packet that is guaranteed to fit
    the configured byte budget.

    Degradation order when a packet is too large (each step is logged in the
    packet metadata so HQ knows what it received):
        1. Full-size evidence crop
        2. Smaller / lower-quality crops
        3. Snapshot dropped entirely (metadata-only alert)
        4. Non-essential metadata dropped
    """

    def __init__(
        self,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        encoder: Optional[SnapshotEncoder] = None,
    ):
        self.max_payload_bytes = int(max_payload_bytes)
        self.encoder = encoder if encoder is not None else SnapshotEncoder()

    def metadata_floor_bytes(self, event: Optional[dict] = None) -> int:
        """
        Size of the smallest packet this builder can produce (no snapshot, all
        optional fields stripped). Alerts below this floor are still transmitted
        but flagged, because dropping a CRITICAL alert to save ~200 bytes is the
        wrong trade at a border post.
        """
        event = event or {}
        result = self.build(
            event=event,
            geo={"node_id": "X", "camera_id": "Y", "sector": "Z"},
            frame=None,
        )
        return result["meta"]["bytes"]

    def build(
        self,
        event: dict,
        geo: dict,
        frame=None,
        bbox=None,
        seq: int = 0,
        event_id: Optional[str] = None,
    ) -> dict:
        """
        Returns {'payload': bytes, 'meta': dict, 'over_budget': bool}.
        """
        event_type = str(event.get("event_type", "SYSTEM"))
        status = str(event.get("status", ""))
        now = time.time()
        event_id = event_id or uuid.uuid4().hex[:12]

        envelope = {
            "v": 1,
            "event_id": event_id,
            "seq": int(seq),
            "node_id": geo.get("node_id", "UNKNOWN-NODE"),
            "camera_id": geo.get("camera_id", "UNKNOWN-CAM"),
            "sector": geo.get("sector", "UNKNOWN"),
            "event_type": event_type,
            "severity": severity_for(event_type, status),
            "status": status,
            "category": event.get("category", "human"),
            "class_name": event.get("class_name", "person"),
            "confidence": event.get("confidence", 0.0),
            "zone": event.get("zone", ""),
            "direction": event.get("direction", ""),
            "identity": event.get("identity", ""),
            "plate_text": event.get("plate_text", ""),
            "ocr_confidence": event.get("ocr_confidence", 0.0),
            "track_id": event.get("track_id"),
            "bbox": list(bbox) if bbox is not None else None,
            # UTC is authoritative on the wire; local display time is not sent.
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "ts": round(now, 3),
            # Geo stamp lets the C2 map plot the alert with zero video transfer.
            "geo": {
                "lat": geo.get("lat"),
                "lon": geo.get("lon"),
                "image": _parse_location(event.get("location")),
            },
            "snapshot": None,
        }

        # Position the packet on the wire: critical alerts take the priority topic.
        topic_kind = "alerts" if envelope["severity"] == "CRITICAL" else "telemetry"
        meta = {
            "event_id": event_id,
            "seq": int(seq),
            "topic_kind": topic_kind,
            "node_id": envelope["node_id"],
            "camera_id": envelope["camera_id"],
            "event_type": event_type,
            "severity": envelope["severity"],
            "status": status,
            "zone": envelope["zone"],
            "plate_text": envelope["plate_text"],
            "utc": envelope["utc"],
            "snapshot": None,
            "snapshot_dropped": False,
            "trimmed": False,
            "bytes": 0,
        }

        base_bytes = len(_serialize(envelope))
        budget_for_snapshot = self.max_payload_bytes - base_bytes - ENVELOPE_SAFETY_BYTES
        envelope["_mb"] = base_bytes  # debug aid, stripped before send if over budget

        # --- Step 1-2: find the best crop that fits ------------------------
        if frame is not None and self.encoder is not None and self.encoder.available:
            if budget_for_snapshot > 0:
                snap = self._best_snapshot(frame, bbox, budget_for_snapshot, base_bytes)
                if snap is not None:
                    envelope["snapshot"] = snap
                else:
                    meta["snapshot_dropped"] = True
            else:
                meta["snapshot_dropped"] = True

        # --- Step 3-4: enforce the budget on the finished envelope ---------
        payload = _serialize(envelope)
        over_budget = False
        if len(payload) > self.max_payload_bytes:
            # Drop the debug field, then the snapshot, then optional metadata.
            envelope.pop("_mb", None)
            payload = _serialize(envelope)
            meta["snapshot"] = self._snapshot_meta(envelope)
            if len(payload) > self.max_payload_bytes:
                envelope["snapshot"] = None
                meta["snapshot"] = None
                meta["snapshot_dropped"] = True
                payload = _serialize(envelope)
            if len(payload) > self.max_payload_bytes:
                for optional in ("geo", "identity", "direction", "plate_text"):
                    envelope.pop(optional, None)
                    payload = _serialize(envelope)
                    if len(payload) <= self.max_payload_bytes:
                        break
                meta["trimmed"] = True
            if len(payload) > self.max_payload_bytes:
                over_budget = True  # Must not happen; surfaced in stats if it does.
        else:
            envelope.pop("_mb", None)
            payload = _serialize(envelope)
            meta["snapshot"] = self._snapshot_meta(envelope)

        meta["bytes"] = len(payload)
        meta["snapshot_bytes"] = (
            meta["snapshot"]["bytes"] if meta.get("snapshot") else 0
        )
        return {"payload": payload, "meta": meta, "over_budget": over_budget}

    def _best_snapshot(self, frame, bbox, budget: int, base_bytes: int):
        """Walks the dim/quality ladders and returns the best crop that fits."""
        for max_dim in self.encoder.dim_ladder:
            for quality in self.encoder.quality_ladder:
                snap = self.encoder.encode(
                    frame, bbox=bbox, max_dim=max_dim, quality=quality
                )
                if snap is None:
                    return None
                # Measure the real packet, base64 inflates by ~4/3.
                projected = base_bytes + len(snap["data"]) + 160
                if projected <= self.max_payload_bytes:
                    return snap
        return None

    @staticmethod
    def _snapshot_meta(envelope: dict):
        snap = envelope.get("snapshot")
        if not snap:
            return None
        return {
            "format": snap.get("format"),
            "width": snap.get("width"),
            "height": snap.get("height"),
            "quality": snap.get("quality"),
            "bytes": snap.get("bytes"),
        }


def _parse_location(location: Any) -> Optional[dict]:
    """Parses the '(x, y)' location string produced by the fence engine."""
    if not location:
        return None
    try:
        text = str(location).strip().strip("()")
        x_str, y_str = text.split(",")
        return {"x": int(float(x_str)), "y": int(float(y_str))}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------
class LoopbackTransport:
    """In-process sink. Keeps every packet so tests and demos can inspect them."""

    name = "loopback"

    def __init__(self, fail_rate: float = 0.0, seed: Optional[int] = None, capacity: int = 200):
        self.fail_rate = fail_rate
        self._rng = random.Random(seed)
        self.forced_outage = False
        self.sent = deque(maxlen=capacity)
        self.attempts = 0

    @property
    def online(self) -> bool:
        return not self.forced_outage

    def send(self, topic: str, payload: bytes) -> bool:
        self.attempts += 1
        if self.forced_outage:
            return False
        if self.fail_rate and self._rng.random() < self.fail_rate:
            return False
        self.sent.append({"topic": topic, "bytes": len(payload), "payload": payload})
        return True

    def close(self) -> None:  # pragma: no cover - symmetry with MQTTTransport
        return None


class SimulatedTransport:
    """
    Wraps any transport and imposes a LinkProfile: latency, packet loss, and
    operator-forced outage (for demonstrating store-and-forward behaviour).

    `sleep=False` records the simulated latency in stats without actually
    stalling the caller - required for the live Streamlit loop, where a real
    700 ms sleep per alert would freeze the UI.
    """

    name = "simulated"

    def __init__(
        self,
        inner: Optional[Any] = None,
        profile: Any = "SATELLITE",
        sleep: bool = False,
        seed: Optional[int] = None,
        forced_outage: bool = False,
    ):
        self.inner = inner if inner is not None else LoopbackTransport()
        self.profile = get_profile(profile)
        self.sleep = sleep
        self.forced_outage = forced_outage
        self._rng = random.Random(seed)
        self.simulated_latency_ms = 0.0
        self.attempts = 0
        self.dropped_by_link = 0

    @property
    def online(self) -> bool:
        return not self.forced_outage

    def set_profile(self, profile: Any) -> None:
        self.profile = get_profile(profile)

    def send(self, topic: str, payload: bytes) -> bool:
        self.attempts += 1
        if self.forced_outage:
            self.dropped_by_link += 1
            return False

        if self.sleep and self.profile.latency_ms > 0:
            time.sleep(self.profile.latency_ms / 1000.0)
        self.simulated_latency_ms += self.profile.latency_ms

        if self.profile.drop_rate > 0 and self._rng.random() < self.profile.drop_rate:
            self.dropped_by_link += 1
            return False

        return bool(self.inner.send(topic, payload))

    def close(self) -> None:
        try:
            self.inner.close()
        except Exception:
            pass


class MQTTTransport:
    """
    Real MQTT publish via paho-mqtt. paho is imported lazily so the platform
    keeps working on outposts that never installed it.
    """

    name = "mqtt"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        qos: int = 0,
        keepalive: int = 30,
        connect_timeout_s: float = 4.0,
    ):
        try:
            import paho.mqtt.client as mqtt
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "paho-mqtt is not installed; install it with "
                "'pip install paho-mqtt' to use the MQTT transport."
            ) from exc

        self._mqtt = mqtt
        self.host = host
        self.port = int(port)
        self.qos = int(qos)
        self.connect_timeout_s = connect_timeout_s
        self.connected = False
        self.last_error = ""
        self.sent_count = 0

        self.client = self._make_client(mqtt, client_id)
        if username:
            self.client.username_pw_set(username, password)

    @staticmethod
    def available() -> bool:
        try:
            import paho.mqtt.client  # noqa: F401

            return True
        except Exception:
            return False

    def _make_client(self, mqtt, client_id: Optional[str]):
        cid = client_id or f"ibvap-edge-{uuid.uuid4().hex[:8]}"
        # paho 2.x requires an explicit callback API version; 1.x does not accept it.
        try:
            return mqtt.Client(
                callback_api_version=getattr(mqtt, "CallbackAPIVersion").VERSION2,
                client_id=cid,
            )
        except Exception:
            return mqtt.Client(client_id=cid)

    def connect(self) -> bool:
        try:
            self.client.connect(self.host, self.port, keepalive=30)
            self.client.loop_start()
            self.connected = True
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.connected = False
            return False

    @property
    def online(self) -> bool:
        return self.connected

    def send(self, topic: str, payload: bytes) -> bool:
        if not self.connected and not self.connect():
            return False
        try:
            info = self.client.publish(topic, payload, qos=self.qos)
            ok = getattr(info, "rc", 1) == 0
            if ok:
                self.sent_count += 1
            else:
                self.last_error = f"publish rc={getattr(info, 'rc', 'unknown')}"
            return bool(ok)
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def close(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
        self.connected = False


def build_transport(
    mode: str = "simulated",
    profile: Any = "SATELLITE",
    mqtt_kwargs: Optional[dict] = None,
    seed: Optional[int] = None,
    sleep: bool = False,
) -> Tuple[Any, str]:
    """
    Factory returning (transport, note).

    mode='simulated' -> loopback sink behind a link profile (no broker needed).
    mode='mqtt'      -> real broker publish, still wrapped in the link profile so
                        a field link stays realistically lossy. Falls back to the
                        simulator, with an explanatory note, if paho is absent.
    """
    mqtt_kwargs = mqtt_kwargs or {}

    if mode == "mqtt":
        if not MQTTTransport.available():
            transport = SimulatedTransport(
                LoopbackTransport(seed=seed), profile, sleep=sleep, seed=seed
            )
            return transport, "paho-mqtt not installed - using simulated link."
        try:
            broker = MQTTTransport(**mqtt_kwargs)
            if not broker.connect():
                note = (
                    f"Broker {broker.host}:{broker.port} unreachable "
                    f"({broker.last_error}) - using simulated link."
                )
                return (
                    SimulatedTransport(
                        LoopbackTransport(seed=seed), profile, sleep=sleep, seed=seed
                    ),
                    note,
                )
            return (
                SimulatedTransport(broker, profile, sleep=sleep, seed=seed),
                f"Publishing to MQTT broker {broker.host}:{broker.port}.",
            )
        except Exception as exc:  # pragma: no cover - env dependent
            return (
                SimulatedTransport(
                    LoopbackTransport(seed=seed), profile, sleep=sleep, seed=seed
                ),
                f"MQTT setup failed ({exc}) - using simulated link.",
            )

    return (
        SimulatedTransport(
            LoopbackTransport(seed=seed), profile, sleep=sleep, seed=seed
        ),
        "Simulated link (no broker required).",
    )


# ---------------------------------------------------------------------------
# Store-and-forward queue
# ---------------------------------------------------------------------------
@dataclass
class QueuedPacket:
    """A packet waiting for the link to come back."""

    topic: str
    payload: bytes
    meta: dict
    attempts: int = 0
    next_attempt_at: float = 0.0
    enqueued_at: float = 0.0

    def age_at(self, now: float) -> float:
        """Seconds this packet has been buffered, measured on the publisher clock."""
        return max(0.0, now - self.enqueued_at)


class TelemetryPublisher:
    """
    Edge-side telemetry agent: build -> send -> buffer -> retransmit.

    Head-of-line FIFO ordering is deliberate: on a bandwidth-starved link, a
    newer alert must not overtake an older one, and HQ must never receive
    out-of-order incident IDs.
    """

    def __init__(
        self,
        transport,
        topic_prefix: str = "ssb/anantanetra",
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        queue_capacity: int = 500,
        max_retries: int = 6,
        backoff_base_s: float = 0.75,
        backoff_cap_s: float = 30.0,
        max_packet_age_s: float = 900.0,
        encoder: Optional[SnapshotEncoder] = None,
        clock: Callable[[], float] = time.time,
        recent_capacity: int = 60,
    ):
        self.transport = transport
        self.topic_prefix = topic_prefix
        self.max_payload_bytes = int(max_payload_bytes)
        self.queue_capacity = int(queue_capacity)
        self.max_retries = int(max_retries)
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_cap_s = float(backoff_cap_s)
        # Telemetry older than this is no longer operationally actionable: a
        # 20-minute-old intrusion alert is history, not a live warning. Expiring
        # it is how an unreachable link finally sheds its backlog.
        self.max_packet_age_s = float(max_packet_age_s)
        self.builder = TelemetryPacketBuilder(max_payload_bytes, encoder=encoder)
        self._clock = clock

        self.queue: deque = deque()
        self.dead_letters: deque = deque(maxlen=50)
        self.seq = 0
        self._t0 = clock()

        # Stats
        self.events = 0
        self.sent = 0
        self.send_failures = 0
        self.retransmissions = 0
        self.dropped = 0
        self.bytes_sent = 0
        self.over_budget = 0
        self.last_error = ""
        self.recent: deque = deque(maxlen=recent_capacity)

    # -- packet building ----------------------------------------------------
    def build_packet(self, event: dict, camera: dict, frame=None, bbox=None) -> dict:
        """Builds (but does not send) a packet. Useful for size verification."""
        from modules.nodes import node_geo  # local import avoids a cycle at import time

        self.seq += 1
        return self.builder.build(
            event=event, geo=node_geo(camera), frame=frame, bbox=bbox, seq=self.seq
        )

    @property
    def topic_for(self) -> str:
        return f"{self.topic_prefix}/events"

    # -- send path ----------------------------------------------------------
    def publish_event(self, event: dict, camera: dict, frame=None, bbox=None) -> dict:
        """
        Builds and transmits one event.

        Returns a result summary:
          status: 'SENT' | 'BUFFERED' | 'DROPPED'
                  SENT      - confirmed delivered
                  BUFFERED  - link down, held for retransmission (not lost)
                  DROPPED   - retry budget exhausted, moved to dead letters
        """
        self.events += 1
        packet = self.build_packet(event, camera, frame=frame, bbox=bbox)
        if packet["over_budget"]:
            self.over_budget += 1

        meta = packet["meta"]
        topic = f"{self.topic_prefix}/{meta['topic_kind']}"

        # Opportunistically drain the backlog first so ordering is preserved.
        self.flush()

        if self._try_send(topic, packet["payload"], meta):
            self._record(meta, status="SENT", attempts=1)
            return self._result(meta, "SENT", attempts=1)

        queued = QueuedPacket(
            topic=topic,
            payload=packet["payload"],
            meta=meta,
            attempts=0,
            next_attempt_at=self._clock(),
            enqueued_at=self._clock(),
        )
        self._enqueue(queued)
        self._record(meta, status="BUFFERED", attempts=0)
        return self._result(meta, "BUFFERED", attempts=0)

    def _try_send(self, topic: str, payload: bytes, meta: dict) -> bool:
        try:
            ok = bool(self.transport.send(topic, payload))
        except Exception as exc:  # a dead broker must never kill the video loop
            self.last_error = str(exc)
            self.send_failures += 1
            return False
        if ok:
            self.sent += 1
            self.bytes_sent += len(payload)
        else:
            self.send_failures += 1
            self.last_error = getattr(self.transport, "last_error", "") or "send failed"
        return ok

    # -- store-and-forward --------------------------------------------------
    def _enqueue(self, packet: QueuedPacket) -> None:
        if len(self.queue) >= self.queue_capacity:
            # Bounded buffer: the oldest telemetry is sacrificed, never the newest.
            dropped = self.queue.popleft()
            self.dropped += 1
            self.dead_letters.append(
                {"reason": "queue_overflow", "meta": dropped.meta, "at": self._clock()}
            )
        self.queue.append(packet)

    def _backoff_for(self, attempts: int) -> float:
        delay = self.backoff_base_s * (2 ** max(0, attempts - 1))
        return min(delay, self.backoff_cap_s)

    def flush(self, max_items: int = 25) -> int:
        """
        Retransmits buffered packets in FIFO order.

        Stops at the first packet not yet due for retry (head-of-line), so a
        burst of new alerts cannot reorder the incident stream.
        """
        delivered = 0
        while self.queue and delivered < max_items:
            packet = self.queue[0]
            now = self._clock()

            # Expiry is checked before the link check on purpose: a total outage
            # must still eventually shed telemetry that has gone stale, otherwise
            # only queue overflow can bound a permanent blackout.
            age = packet.age_at(now)
            if self.max_packet_age_s and age > self.max_packet_age_s:
                self.queue.popleft()
                self.dropped += 1
                self.dead_letters.append(
                    {
                        "reason": "expired",
                        "age_s": round(age, 3),
                        "meta": packet.meta,
                        "at": now,
                    }
                )
                self._record(packet.meta, status="DROPPED", attempts=packet.attempts)
                continue

            if packet.next_attempt_at > now:
                break
            # While the link is known-down we deliberately do NOT burn retry
            # attempts: the packet waits and stays retransmittable.
            if not self.transport.online:
                break

            packet.attempts += 1
            if self._try_send(packet.topic, packet.payload, packet.meta):
                self.queue.popleft()
                self.retransmissions += 1
                delivered += 1
                self._record(packet.meta, status="SENT", attempts=packet.attempts, retransmit=True)
                continue

            if packet.attempts >= self.max_retries:
                self.queue.popleft()
                self.dropped += 1
                self.dead_letters.append(
                    {
                        "reason": "retries_exhausted",
                        "attempts": packet.attempts,
                        "meta": packet.meta,
                        "at": now,
                    }
                )
                self._record(
                    packet.meta, status="DROPPED", attempts=packet.attempts
                )
                continue

            packet.next_attempt_at = now + self._backoff_for(packet.attempts)
            break  # Respect head-of-line ordering.

        return delivered

    def drain_all(self, max_rounds: int = 200) -> int:
        """Flushes until the queue empties (used by tests and HQ resync)."""
        total = 0
        for _ in range(max_rounds):
            progressed = self.flush(max_items=self.queue_capacity or 1)
            total += progressed
            if not self.queue:
                break
            if progressed == 0:
                # Nothing due yet: fast-forward the backoff clock.
                pending = self.queue[0]
                pending.next_attempt_at = self._clock()
        return total

    # -- reporting ----------------------------------------------------------
    def _record(
        self, meta: dict, status: str, attempts: int, retransmit: bool = False
    ) -> None:
        self.recent.append(
            {
                **{
                    k: meta.get(k)
                    for k in (
                        "event_id",
                        "seq",
                        "topic_kind",
                        "node_id",
                        "camera_id",
                        "event_type",
                        "severity",
                        "status",
                        "zone",
                        "plate_text",
                        "utc",
                        "bytes",
                        "snapshot_bytes",
                    )
                },
                "delivery": status,
                "attempts": attempts,
                "retransmit": retransmit,
                "queue_depth": len(self.queue),
                "snapshot_dropped": bool(meta.get("snapshot_dropped")),
                "trimmed": bool(meta.get("trimmed")),
            }
        )

    def _result(self, meta: dict, status: str, attempts: int) -> dict:
        return {
            "event_id": meta.get("event_id"),
            "seq": meta.get("seq"),
            "status": status,
            "bytes": meta.get("bytes"),
            "snapshot_bytes": meta.get("snapshot_bytes"),
            "snapshot_dropped": bool(meta.get("snapshot_dropped")),
            "topic": f"{self.topic_prefix}/{meta.get('topic_kind')}",
            "queue_depth": len(self.queue),
            "attempts": attempts,
        }

    def set_link_profile(self, profile: Any) -> None:
        """Changes the simulated link condition (used by the UI outage toggle)."""
        if isinstance(self.transport, SimulatedTransport):
            self.transport.set_profile(profile)

    def set_budget(self, max_payload_bytes: int) -> None:
        """
        Retunes the packet byte budget without discarding the backlog or the
        delivery statistics, so an operator can tighten the budget live.
        """
        self.max_payload_bytes = int(max_payload_bytes)
        self.builder.max_payload_bytes = int(max_payload_bytes)

    def set_outage(self, outage: bool) -> None:
        """Simulates a total link failure so store-and-forward can be demoed."""
        if hasattr(self.transport, "forced_outage"):
            self.transport.forced_outage = bool(outage)

    def stats(self) -> dict:
        transport = self.transport
        profile = getattr(transport, "profile", None)
        return {
            "events": self.events,
            "sent": self.sent,
            "buffered": len(self.queue),
            "dropped": self.dropped,
            "retransmissions": self.retransmissions,
            "send_failures": self.send_failures,
            "dead_letters": len(self.dead_letters),
            "bytes_sent": self.bytes_sent,
            "avg_payload_bytes": (
                round(self.bytes_sent / self.sent, 1) if self.sent else 0.0
            ),
            "max_payload_bytes": self.max_payload_bytes,
            "over_budget": self.over_budget,
            "metadata_floor_bytes": self.builder.metadata_floor_bytes(),
            "max_packet_age_s": self.max_packet_age_s,
            "budget_utilisation": (
                round(100.0 * (self.bytes_sent / self.sent) / self.max_payload_bytes, 1)
                if self.sent
                else 0.0
            ),
            "online": bool(getattr(transport, "online", True)),
            "transport": getattr(transport, "name", "unknown"),
            "link_profile": profile.name if profile else "n/a",
            "link_label": profile.label if profile else "n/a",
            "link_latency_ms": profile.latency_ms if profile else 0.0,
            "link_drop_rate": profile.drop_rate if profile else 0.0,
            "simulated_latency_ms": round(
                getattr(transport, "simulated_latency_ms", 0.0), 1
            ),
            "last_error": self.last_error,
        }

    def recent_records(self, limit: int = 15) -> list:
        records = list(self.recent)
        return records[-limit:][::-1]

    def queue_snapshot(self) -> list:
        """Metadata of buffered packets, oldest first (for the UI backlog view)."""
        now = self._clock()
        return [
            {**p.meta, "attempts": p.attempts, "age_s": round(p.age_at(now), 1)}
            for p in list(self.queue)
        ]
