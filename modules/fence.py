# cv2 is only needed to DRAW the fence overlay; the intrusion logic itself is
# pure geometry. Keeping the import optional means the fence (and therefore the
# whole identity/alert path) can be verified without the vision stack installed.
try:  # pragma: no cover - environment dependent
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

import numpy as np


def ccw(A, B, C):
    """Checks if three points are listed in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(p1, p2, p3, p4):
    """
    Returns True if segment (p1, p2) intersects segment (p3, p4).
    """
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def point_in_polygon(pt, polygon) -> bool:
    """
    Ray-casting point-in-polygon test in pure Python.

    Deliberately not cv2.pointPolygonTest: the intrusion logic must work when
    the drawing backend (and therefore cv2) is absent, exactly like the
    segment-intersection path above.
    """
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


class VirtualFence:
    """
    Virtual Fence / Tripwire intrusion detector for Border Security Posts.
    Monitors object trajectories and raises alerts when unauthorized
    movement breaches the designated boundary.

    Two fence shapes are supported:

    * LINE  - the classic straight tripwire; an alert fires when a track's
      movement segment intersects the line (original behaviour, unchanged).
    * POLYGON - an arbitrary closed zone of 3+ vertices drawn by the operator;
      an alert fires the first time a track ENTERS the zone. This models real
      geometry - an L-shaped yard, a wedge between two paths, a horseshoe
      around a gate - which a single straight line cannot express.

    Both shapes coexist in one code path downstream: consumers only see the
    same alert events, and `p1/p2` always hold the shape's bounding midline so
    the pose layer's climb heuristic keeps working without knowing about zones.
    """

    def __init__(self, line_coords: tuple = None, zone_name: str = "BOP Sector 4 Perimeter",
                 polygon=None):
        self.zone_name = zone_name
        self.intruded_track_ids = set()   # Tracks that have breached the perimeter
        self.watchlist_hit_tracks = set()  # Tracks biometrically matched to a suspect
        self.alert_history = []           # List of triggered alert events
        self.polygon = None

        if polygon is not None and len(polygon) >= 3:
            self.polygon = [(int(p[0]), int(p[1])) for p in polygon]

        if self.polygon is not None:
            self._sync_line_from_polygon()
        elif line_coords is None:
            # Default line tuned for 1080p surveillance video (y=620 tripwire)
            self.p1 = (50, 620)
            self.p2 = (1870, 620)
        else:
            self.p1, self.p2 = line_coords

        self.shape = "polygon" if self.polygon is not None else "line"

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _sync_line_from_polygon(self):
        """Keeps p1/p2 as the polygon's bounding midline (pose heuristic + compat)."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        mid_y = int(sum(ys) / len(ys))
        self.p1 = (int(min(xs)), mid_y)
        self.p2 = (int(max(xs)), mid_y)

    @property
    def fence_y(self) -> int:
        """The shape's representative row - used by the pose climb heuristic."""
        return int((self.p1[1] + self.p2[1]) / 2)

    def points(self) -> list:
        """The fence vertices: polygon corners, or the two line endpoints."""
        if self.polygon is not None:
            return list(self.polygon)
        return [tuple(self.p1), tuple(self.p2)]

    def update_line(self, p1: tuple, p2: tuple):
        """Dynamically update line coordinates (e.g. from a UI control)."""
        self.p1 = p1
        self.p2 = p2
        self.polygon = None
        self.shape = "line"

    def update_polygon(self, points):
        """Dynamically replace the fence with a polygon zone (3+ vertices)."""
        if points is not None and len(points) >= 3:
            self.polygon = [(int(p[0]), int(p[1])) for p in points]
            self._sync_line_from_polygon()
            self.shape = "polygon"

    # ------------------------------------------------------------------
    # Intrusion detection
    # ------------------------------------------------------------------
    def check_intrusions(self, active_tracks: dict, timestamp: str = "", watchlist=None,
                         identity_map: dict = None):
        """
        Evaluates active tracks for boundary crossing with Identity Check.

        `identity_map` carries per-track FRS decisions ({track_id: decision}).
        When a biometric decision is available it takes precedence over the
        DEMO-ONLY simulated-track-ID lookup, and the alert records which source
        identified the subject so the audit trail cannot be misread.

        Returns list of new alert events triggered in this frame.
        """
        new_alerts = []
        identity_map = identity_map or {}

        for track_id, data in active_tracks.items():
            # If already alerted for this track, skip re-alerting
            if track_id in self.intruded_track_ids:
                continue

            traj = data.get("trajectory", [])
            if len(traj) < 2:
                continue

            prev_pt = traj[-2]
            curr_pt = traj[-1]

            if self.polygon is not None:
                # POLYGON ZONE: breach = first transition from outside to inside.
                # Merely touching the boundary is not enough; the track must
                # have been outside on the previous step and be inside now,
                # which is exactly the "crossed into the restricted area" event.
                if not (point_in_polygon(curr_pt, self.polygon)
                        and not point_in_polygon(prev_pt, self.polygon)):
                    continue
                direction = "INBOUND (entered restricted zone)"
            else:
                # LINE: check if recent movement segment intersects the fence line
                if not segments_intersect(prev_pt, curr_pt, self.p1, self.p2):
                    continue
                # Compute crossing direction
                dy = curr_pt[1] - prev_pt[1]
                direction = "INBOUND (Southbound)" if dy >= 0 else "OUTBOUND (Northbound)"

            self.intruded_track_ids.add(track_id)

            # Step 7 & 8: Identity Verification
            is_auth = False
            personnel_info = None
            identity_source = "NONE"
            biometric = identity_map.get(track_id) or {}
            biometric_state = biometric.get("decision")
            similarity = biometric.get("similarity", 0.0)

            if biometric_state == "AUTHORIZED":
                is_auth = True
                identity_source = "BIOMETRIC"
                identity_str = (
                    f"{biometric.get('name') or biometric.get('person_id')} "
                    f"({biometric.get('person_id')}) cos={similarity}"
                )
                personnel_info = {
                    "personnel_id": biometric.get("person_id"),
                    "name": biometric.get("name") or biometric.get("person_id"),
                    "rank": biometric.get("rank", ""),
                    "unit": biometric.get("unit", ""),
                }
                event_type = "AUTHORIZED_PATROL"
                status = "AUTHORIZED_PATROL"
            elif biometric_state == "WATCHLIST_HIT":
                identity_source = "BIOMETRIC"
                identity_str = (
                    f"WATCHLIST SUBJECT: {biometric.get('name') or biometric.get('person_id')} "
                    f"({biometric.get('person_id')}) cos={similarity}"
                )
                event_type = "WATCHLIST_HIT"
                status = "KNOWN_WATCHLIST_SUBJECT"
            elif biometric_state == "REVIEW":
                # Matched something, but not enough to act on: this is exactly
                # the case a human operator must adjudicate.
                identity_source = "BIOMETRIC_UNCONFIRMED"
                identity_str = f"IDENTITY UNCONFIRMED (best cos={similarity})"
                event_type = "INTRUSION_ALERT"
                status = "IDENTITY_UNCONFIRMED"
            else:
                # No usable biometric read: fall back to the DEMO roster and
                # LABEL it as such, so nobody mistakes a demo for biometrics.
                if watchlist is not None:
                    is_auth, personnel_info = watchlist.verify_person(track_id)
                    if is_auth:
                        identity_source = "SIMULATED_DEMO_ID"
                if is_auth and personnel_info:
                    identity_str = (
                        f"{personnel_info['name']} ({personnel_info['personnel_id']})"
                    )
                    event_type = "AUTHORIZED_PATROL"
                    status = "AUTHORIZED_PATROL"
                else:
                    event_type = "INTRUSION_ALERT"
                    status = "UNKNOWN_INTRUDER"
                    identity_str = "UNKNOWN PERSON"

            alert_event = {
                "timestamp": timestamp,
                "event_type": event_type,
                "track_id": track_id,
                "category": data.get("category", "human"),
                "class_name": data.get("class_name", "person"),
                "confidence": data.get("conf", 0.0),
                "zone": self.zone_name,
                "direction": direction,
                "identity": identity_str,
                "identity_source": identity_source,
                "biometric_similarity": similarity,
                "is_authorized": is_auth,
                "location": f"({curr_pt[0]}, {curr_pt[1]})",
                "status": status
            }
            if event_type == "WATCHLIST_HIT":
                self.watchlist_hit_tracks.add(track_id)
            self.alert_history.append(alert_event)
            new_alerts.append(alert_event)

        return new_alerts

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_badge(self, annotated, text: str, cx: int, cy: int):
        """Semi-transparent label badge centred on (cx, cy)."""
        h, w = annotated.shape[:2]
        font_scale = 0.45 if w < 1000 else 0.55
        thickness = 1 if w < 1000 else 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = max(10, cx - int(tw / 2))
        y = max(th + 12, cy)
        cv2.rectangle(annotated, (x - 6, y - th - 6), (x + tw + 6, y + baseline), (0, 0, 180), -1)
        cv2.putText(annotated, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def draw_fence(self, frame: np.ndarray, active_tracks: dict, watchlist=None):
        """
        Draws the fence and overlays intrusion warning markers.
        Renders GREEN for Authorized Patrols and RED for Unknown Intruders.

        The fence itself is drawn EVERY frame in both shapes - a translucent
        red fill for zones plus a thick outline and vertex handles, so the
        restricted area stays continuously visible on the live feed.
        """
        if cv2 is None:  # no drawing backend - never break the pipeline over it
            return frame
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if self.polygon is not None:
            pts = np.array(self.polygon, np.int32).reshape((-1, 1, 2))
            # Translucent red fill: visible but never hides the people in it.
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 180))
            cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, dst=annotated)
            # Thick outline + vertex handles (the same points the editor moves).
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
            for vx, vy in self.polygon:
                cv2.circle(annotated, (vx, vy), 6, (0, 255, 255), -1)
            xs = [p[0] for p in self.polygon]
            ys = [p[1] for p in self.polygon]
            cx, cy = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
            self._draw_badge(
                annotated,
                f"RESTRICTED ZONE - {self.zone_name.upper()}",
                cx, cy,
            )
        else:
            # Draw Fence Line (Neon Red / Crimson)
            fence_color = (0, 0, 255)  # BGR Red
            cv2.line(annotated, self.p1, self.p2, fence_color, 3)

            # Draw line endpoint markers
            cv2.circle(annotated, self.p1, 6, (0, 255, 255), -1)
            cv2.circle(annotated, self.p2, 6, (0, 255, 255), -1)

            mid_x = int((self.p1[0] + self.p2[0]) / 2)
            mid_y = int((self.p1[1] + self.p2[1]) / 2)
            self._draw_badge(
                annotated,
                f"RESTRICTED BORDER VIRTUAL FENCE - {self.zone_name.upper()}",
                mid_x, mid_y - 10,
            )

        # Draw Intrusion highlights on active breached tracks
        for track_id, data in active_tracks.items():
            traj = data.get("trajectory", [])

            is_auth = False
            auth_info = None
            if watchlist is not None:
                is_auth, auth_info = watchlist.verify_person(track_id)
            # A watchlist subject is drawn as a threat even if the demo roster
            # would have whitelisted the track ID.
            if track_id in self.watchlist_hit_tracks:
                is_auth = False
                auth_info = None

            # Trajectory trail color: Green for authorized, Red for breached intruder, Yellow for normal track
            if track_id in self.intruded_track_ids:
                trail_color = (0, 200, 0) if is_auth else (0, 0, 255)
            else:
                trail_color = (0, 255, 255)

            if len(traj) >= 2:
                pts = np.array(traj, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], False, trail_color, 2)

            # Highlight breached track
            if track_id in self.intruded_track_ids:
                x1, y1, x2, y2 = data["bbox"]
                box_color = (0, 200, 0) if is_auth else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

                if is_auth and auth_info:
                    alert_tag = f"AUTH: {auth_info['name'].split()[-1]} ({auth_info['personnel_id']})"
                    bg_color = (0, 150, 0)
                else:
                    alert_tag = f"! INTRUDER #{track_id} !"
                    bg_color = (0, 0, 255)

                (tag_tw, tag_th), _ = cv2.getTextSize(alert_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + tag_tw + 8, y1), bg_color, -1)
                cv2.putText(annotated, alert_tag, (x1 + 4, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated
