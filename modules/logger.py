import os
import pandas as pd
from datetime import datetime, timezone


# Behaviour event types produced by the pose analytics layer. Imported lazily by
# consumers so the logger keeps no hard dependency on the pose module.
BEHAVIOR_EVENT_TYPES = (
    "FENCE_CLIMB",
    "FALL_ALERT",
    "CROUCH_INTRUSION",
    "LOITERING",
    "RUNNING",
    "GROUP_CONVERGENCE",
    "ARM_RAISED_SIGNAL",
)


class EventLogger:
    """
    In-memory Alert & Event Logger for Border Surveillance operations.
    Maintains clean tabular logs of intrusions, vehicle movements, and ANPR reads.
    Exports to CSV without external database dependencies.

    Every event carries a UTC timestamp plus node_id / camera_id so records can
    be attributed across a multi-camera, multi-outpost grid and reconciled with
    telemetry transmitted to Sector HQ.
    """
    def __init__(
        self,
        csv_path: str = "alerts.csv",
        node_id: str = "BOP-SECTOR-A",
        camera_id: str = "CAM-UNKNOWN"
    ):
        self.csv_path = csv_path
        self.node_id = node_id
        self.camera_id = camera_id
        self.events = []
        self.columns = [
            "timestamp",
            "utc_timestamp",
            "node_id",
            "camera_id",
            "event_type",
            "track_id",
            "category",
            "class_name",
            "confidence",
            "plate_text",
            "ocr_confidence",
            "status",
            "details"
        ]

    def set_context(self, node_id: str = None, camera_id: str = None):
        """Updates the active outpost/camera context (called when the operator
        switches surveillance sectors)."""
        if node_id:
            self.node_id = node_id
        if camera_id:
            self.camera_id = camera_id

    def log_event(
        self,
        event_type: str,
        track_id: int,
        category: str = "human",
        class_name: str = "person",
        confidence: float = 0.0,
        plate_text: str = "-",
        ocr_confidence: float = 0.0,
        status: str = "VERIFIED",
        details: str = "",
        timestamp: str = None,
        node_id: str = None,
        camera_id: str = None
    ):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        event = {
            "timestamp": timestamp,
            "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "node_id": node_id or self.node_id,
            "camera_id": camera_id or self.camera_id,
            "event_type": event_type,
            "track_id": track_id,
            "category": category,
            "class_name": class_name,
            "confidence": round(float(confidence), 3),
            "plate_text": plate_text,
            "ocr_confidence": round(float(ocr_confidence), 3),
            "status": status,
            "details": details
        }
        self.events.append(event)
        self._append_to_csv(event)
        return event

    def _existing_csv_columns(self):
        """Reads just the header row of an existing log file, or None."""
        try:
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                return None
            with open(self.csv_path, "r", encoding="utf-8") as handle:
                return handle.readline().strip().split(",")
        except Exception:
            return None

    def _migrate_stale_csv(self):
        """
        Rotates a log written under an older schema instead of appending
        mismatched columns. The old file is preserved, never deleted.
        """
        existing = self._existing_csv_columns()
        if existing is None or existing == self.columns:
            return
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(self.csv_path)
            legacy_path = f"{base}.{stamp}.legacy{ext or '.csv'}"
            os.replace(self.csv_path, legacy_path)
            print(
                f"[LOGGER] Event schema changed; previous log preserved as: {legacy_path}"
            )
        except Exception as e:
            print(f"[LOGGER WARNING] Could not rotate stale log: {e}")

    def _append_to_csv(self, event: dict):
        try:
            df = pd.DataFrame([event], columns=self.columns)
            file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
            if file_exists:
                self._migrate_stale_csv()
                file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
            df.to_csv(self.csv_path, mode="a", header=not file_exists, index=False)
        except Exception as e:
            print(f"[LOGGER WARNING] Could not write to CSV: {e}")

    def get_dataframe(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(self.events, columns=self.columns)

    def get_stats(self) -> dict:
        total = len(self.events)
        # A confirmed watchlist identity IS an intrusion event - it must never be
        # excluded from the headline counter.
        intrusions = sum(
            1 for e in self.events
            if e["event_type"] in ("INTRUSION_ALERT", "WATCHLIST_HIT")
        )
        watchlist_hits = sum(
            1 for e in self.events if e["event_type"] == "WATCHLIST_HIT"
        )
        # Pose-derived behaviour alerts are their own class: they describe
        # conduct, not a perimeter breach, and conflating them would corrupt the
        # intrusion headline an operator reads first.
        behavior_alerts = sum(
            1 for e in self.events if e["event_type"] in BEHAVIOR_EVENT_TYPES
        )
        biometric_ids = sum(
            1 for e in self.events if "BIOMETRIC" in str(e.get("details", ""))
        )
        vehicles = sum(1 for e in self.events if e["category"] == "vehicle")
        plates_read = sum(1 for e in self.events if e["plate_text"] != "-" and e["status"] == "VERIFIED")
        flagged_reviews = sum(1 for e in self.events if e["status"] == "FLAGGED_FOR_MANUAL_REVIEW")

        return {
            "total_alerts": total,
            "intrusions": intrusions,
            "watchlist_hits": watchlist_hits,
            "behavior_alerts": behavior_alerts,
            "biometric_identifications": biometric_ids,
            "vehicles_detected": vehicles,
            "plates_read": plates_read,
            "flagged_manual_review": flagged_reviews
        }

    def clear(self):
        self.events = []
        if os.path.exists(self.csv_path):
            try:
                os.remove(self.csv_path)
            except Exception:
                pass
