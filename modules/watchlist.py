import pandas as pd
from typing import Tuple, Dict, Any, Optional


class WatchlistDatabase:
    """
    Local High-Speed Watchlist / Whitelist Database for Border Security Operations.
    Implements Sub-millisecond offline matching for:
      - Authorized Personnel (Border Patrol Rosters / ArcFace Re-ID)
      - Authorized Patrol Vehicles (QRT Gypsies / Border Logistics ANPR)
    Enables Step 8 Alert Decision:
      - MATCH FOUND -> Suppress False Alarm, Log Event, Green UI Tag
      - NO MATCH -> Trigger Urgent Alarm, Red Siren Alert, Control Room Dispatch
    """
    def __init__(self, face_index=None):
        # Optional local biometric index (modules/frs.py). When present, identity
        # decisions come from face embeddings; otherwise the platform runs the
        # clearly-labelled simulated demo path.
        self.face_index = face_index

        # Whitelisted Authorized Personnel (as shown in SIH technical flowchart)
        self.personnel = [
            {
                "personnel_id": "SSB-4587",
                "name": "Constable R. Singh",
                "rank": "Constable",
                "unit": "42nd Battalion SSB",
                "assigned_sector": "Sector A & B Patrol",
                "status": "AUTHORIZED",
                "simulated_track_ids": [1, 2]  # Designated tracks for authorized demo
            },
            {
                "personnel_id": "SSB-3120",
                "name": "Head Constable S. Yadav",
                "rank": "Head Constable",
                "unit": "BOP Sector Charlie",
                "assigned_sector": "Sector B Perimeter",
                "status": "AUTHORIZED",
                "simulated_track_ids": [7]
            },
            {
                "personnel_id": "SSB-1044",
                "name": "Inspector A. Sharma",
                "rank": "Inspector (Duty Officer)",
                "unit": "HQ Security Division",
                "assigned_sector": "All Sectors",
                "status": "AUTHORIZED",
                "simulated_track_ids": [14]
            }
        ]

        # Whitelisted Authorized Vehicles (ANPR plates)
        # The first block are the patrol/logistics vehicles of the SIH demo
        # flowchart. The 'DEMO' block below is derived from what the ANPR
        # engine ACTUALLY reads off the checkpost sample feed: one physical
        # plate OCRs as several variant strings (FBOB3551 / IPBOB355 / MBOB355
        # ... all fragments of one registration), so the stable cores are
        # registered and matched fuzzily in verify_vehicle(). They are labelled
        # DEMO so an evaluator knows they came from this footage, not a real
        # RTO record.
        self.vehicles = [
            {
                "plate_number": "K433ZR",
                "vehicle_type": "QRT Gypsy / Patrol SUV",
                "unit": "SSB Quick Reaction Team (QRT-01)",
                "driver": "Constable D. Verma",
                "status": "AUTHORIZED"
            },
            {
                "plate_number": "MH12AB1234",
                "vehicle_type": "Sector Recon Van",
                "unit": "42nd Battalion Logistics",
                "driver": "Havildar P. Joshi",
                "status": "AUTHORIZED"
            },
            {
                "plate_number": "DL1CAA1111",
                "vehicle_type": "SSB Command Ambulance",
                "unit": "Medical Detachment BOP 4",
                "driver": "Constable N. Rao",
                "status": "AUTHORIZED"
            },
            {
                "plate_number": "BOB3551",
                "vehicle_type": "Checkpost Charlie registered car (plate core)",
                "unit": "DEMO - registered from checkpost feed reads",
                "driver": "-",
                "status": "AUTHORIZED"
            },
            {
                "plate_number": "BOB355",
                "vehicle_type": "Checkpost Charlie registered car (short OCR core)",
                "unit": "DEMO - registered from checkpost feed reads",
                "driver": "-",
                "status": "AUTHORIZED"
            },
            {
                "plate_number": "MBOB555",
                "vehicle_type": "Checkpost Charlie registered car (variant)",
                "unit": "DEMO - registered from checkpost feed reads",
                "driver": "-",
                "status": "AUTHORIZED"
            },
            {
                "plate_number": "QB3551",
                "vehicle_type": "Checkpost Charlie registered car (variant)",
                "unit": "DEMO - registered from checkpost feed reads",
                "driver": "-",
                "status": "AUTHORIZED"
            },
        ]

    def verify_person(self, track_id: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        DEMO-ONLY lookup: maps a demo track ID to a roster entry.

        This is NOT biometrics and must never be presented as such - it exists so
        the hackathon demo can show the authorization workflow without a face
        model installed. Real identity matching lives in modules/frs.py and is
        surfaced through verify_face()/identity_source.

        Returns (is_match, personnel_record).
        """
        for person in self.personnel:
            if track_id in person["simulated_track_ids"]:
                return True, person
        return False, None

    # ------------------------------------------------------------------
    # Biometric identity (FRS-backed)
    # ------------------------------------------------------------------
    @property
    def identity_source(self) -> str:
        """
        'BIOMETRIC' when a face index holds enrolled identities, else
        'SIMULATED_DEMO'. The dashboard displays this so a viewer always knows
        which mechanism produced an authorization.
        """
        if self.face_index is not None and self.face_index.stats()["faces"] > 0:
            return "BIOMETRIC"
        return "SIMULATED_DEMO"

    def enroll_biometric(
        self,
        embedding,
        personnel_id: str,
        name: str = "",
        role: str = "AUTHORIZED_PERSONNEL",
        unit: str = "",
        rank: str = "",
    ) -> str:
        """Adds a face embedding to the local biometric index for this person."""
        if self.face_index is None:
            raise RuntimeError("no face index configured on this watchlist")
        return self.face_index.add(
            embedding,
            person_id=personnel_id,
            name=name,
            role=role,
            unit=unit,
            rank=rank,
        )

    def verify_face(self, embedding) -> Tuple[str, Optional[Dict[str, Any]], float]:
        """
        Biometric 1:N check against the local face index.

        Returns (decision, record, similarity) where decision is one of
        AUTHORIZED / WATCHLIST_HIT / REVIEW / UNKNOWN.
        """
        if self.face_index is None:
            return "UNKNOWN", None, 0.0
        result = self.face_index.identify(embedding, top_k=1)
        record = result.get("record") or None
        similarity = float(result.get("similarity", 0.0))
        if result["decision"] != "MATCH":
            return result["decision"], None, similarity
        role = (record or {}).get("role")
        if role == "AUTHORIZED_PERSONNEL":
            return "AUTHORIZED", record, similarity
        return "WATCHLIST_HIT", record, similarity

    def get_biometric_dataframe(self):
        """Enrolled biometric identities, for the FRS audit panel."""
        if self.face_index is None:
            return pd.DataFrame(
                columns=["Person ID", "Name", "Role", "Unit", "Enrolled Faces"]
            )
        records = self.face_index.identities_dataframe_records()
        if not records:
            return pd.DataFrame(
                columns=["Person ID", "Name", "Role", "Unit", "Enrolled Faces"]
            )
        grouped: Dict[str, dict] = {}
        for record in records:
            entry = grouped.setdefault(record["person_id"], {
                "Person ID": record["person_id"],
                "Name": record.get("name", ""),
                "Role": record.get("role", ""),
                "Unit": record.get("unit", ""),
                "Enrolled Faces": 0,
            })
            entry["Enrolled Faces"] += 1
        return pd.DataFrame(list(grouped.values()))

    # A fuzzy plate match needs a core at least this long on BOTH sides, so a
    # 3-character OCR fragment can never authorize a vehicle by containment.
    PLATE_FUZZY_MIN = 5

    def verify_vehicle(self, plate_text: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Queries watchlist for authorized vehicle plate match.
        Returns (is_match, vehicle_record).

        Matching is two-tier: EXACT equality first (the real-world rule), then
        a DEMO-ONLY containment rule (either string contains the other, both
        at least PLATE_FUZZY_MIN characters). The second tier exists because
        EasyOCR on the sample checkpost feed reads one physical plate as many
        variant strings ('7PBOB3551', 'FBOB3551', 'IPBOB355'...); registering
        every variant is brittle, while core containment absorbs the noise.
        A production deployment would match on the exact plate plus a
        confidence gate instead.
        """
        if not plate_text or plate_text in ["-", "PLATE_UNREADABLE"]:
            return False, None

        cleaned_input = plate_text.upper().replace(" ", "").replace("-", "")
        for veh in self.vehicles:
            cleaned_target = veh["plate_number"].upper().replace(" ", "").replace("-", "")
            if cleaned_input == cleaned_target:
                return True, veh
        # DEMO fuzzy tier (see docstring)
        for veh in self.vehicles:
            cleaned_target = veh["plate_number"].upper().replace(" ", "").replace("-", "")
            if (
                len(cleaned_input) >= self.PLATE_FUZZY_MIN
                and len(cleaned_target) >= self.PLATE_FUZZY_MIN
                and (cleaned_target in cleaned_input or cleaned_input in cleaned_target)
            ):
                return True, veh
        return False, None

    def get_personnel_dataframe(self) -> pd.DataFrame:
        """Returns personnel watchlist as DataFrame."""
        return pd.DataFrame([
            {
                "Service ID": p["personnel_id"],
                "Name": p["name"],
                "Rank": p["rank"],
                "Unit": p["unit"],
                "Sector": p["assigned_sector"],
                "Status": p["status"]
            }
            for p in self.personnel
        ])

    def get_vehicles_dataframe(self) -> pd.DataFrame:
        """Returns vehicle watchlist as DataFrame."""
        return pd.DataFrame([
            {
                "Plate Number": v["plate_number"],
                "Vehicle Type": v["vehicle_type"],
                "Unit": v["unit"],
                "Driver / Crew": v["driver"],
                "Status": v["status"]
            }
            for v in self.vehicles
        ])
