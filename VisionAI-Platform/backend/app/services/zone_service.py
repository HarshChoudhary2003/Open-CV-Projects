"""
VisionAI Platform - Zone Detection Service
Configurable polygonal restricted/safe zones per camera.
Checks each detected object against all defined zones and raises
intrusion events when someone crosses into a restricted zone.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


ZONE_DB_PATH = Path("zone_db.json")


class Zone:
    """A named, coloured, polygonal region of interest."""

    def __init__(
        self,
        zone_id: str,
        name: str,
        polygon: List[Tuple[int, int]],
        zone_type: str = "restricted",   # restricted | safe | monitor
        camera_id: str = "cam0",
        color: Tuple[int, int, int] = (0, 0, 255),
    ):
        self.zone_id = zone_id
        self.name = name
        self.polygon = np.array(polygon, dtype=np.int32)
        self.zone_type = zone_type
        self.camera_id = camera_id
        self.color = color
        self._intrusion_count: Dict[int, float] = {}   # track_id -> first_seen

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "polygon": self.polygon.tolist(),
            "zone_type": self.zone_type,
            "camera_id": self.camera_id,
            "color": list(self.color),
        }

    def contains_point(self, x: float, y: float) -> bool:
        pt = (float(x), float(y))
        return cv2.pointPolygonTest(self.polygon, pt, False) >= 0

    def check_detection(self, det: dict) -> Optional[dict]:
        """Return an intrusion event dict if the detection is inside this zone."""
        if self.zone_type == "safe":
            return None

        bbox = det.get("bbox", {})
        cx = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2
        cy = (bbox.get("y1", 0) + bbox.get("y2", 0)) / 2

        if not self.contains_point(cx, cy):
            return None

        track_id = det.get("track_id", -1) or -1
        now = time.time()

        if track_id not in self._intrusion_count:
            self._intrusion_count[track_id] = now
            return {
                "zone_id": self.zone_id,
                "zone_name": self.name,
                "zone_type": self.zone_type,
                "class_name": det.get("class_name", "unknown"),
                "track_id": track_id,
                "timestamp": now,
                "description": (
                    f"{det.get('class_name', 'Object')} entered "
                    f"restricted zone '{self.name}'"
                ),
            }
        return None   # already counted

    def draw(self, frame: np.ndarray, show_label: bool = True) -> np.ndarray:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon], (*self.color, 50))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [self.polygon], True, self.color, 2)
        if show_label:
            cx = int(np.mean(self.polygon[:, 0]))
            cy = int(np.mean(self.polygon[:, 1]))
            label = f"⚠ {self.name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (cx - 4, cy - th - 6), (cx + tw + 4, cy + 4),
                          (20, 20, 20), -1)
            cv2.putText(frame, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 1, cv2.LINE_AA)
        return frame


class ZoneService:
    _instance: Optional["ZoneService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def initialise(self):
        if self._initialised:
            return
        self._zones: Dict[str, Zone] = {}   # zone_id -> Zone
        self._load_db()
        self._initialised = True

    # ── Persistence ──────────────────────────────────────────────────

    def _load_db(self):
        if ZONE_DB_PATH.exists():
            try:
                data = json.loads(ZONE_DB_PATH.read_text())
                for zid, z in data.items():
                    self._zones[zid] = Zone(
                        zone_id=z["zone_id"],
                        name=z["name"],
                        polygon=z["polygon"],
                        zone_type=z.get("zone_type", "restricted"),
                        camera_id=z.get("camera_id", "cam0"),
                        color=tuple(z.get("color", [0, 0, 255])),
                    )
            except Exception as e:
                print(f"[ZoneService] Failed to load zone DB: {e}")

    def _save_db(self):
        data = {zid: z.to_dict() for zid, z in self._zones.items()}
        ZONE_DB_PATH.write_text(json.dumps(data, indent=2))

    # ── CRUD ─────────────────────────────────────────────────────────

    def add_zone(
        self,
        zone_id: str,
        name: str,
        polygon: List[Tuple[int, int]],
        zone_type: str = "restricted",
        camera_id: str = "cam0",
        color: Tuple[int, int, int] = (0, 0, 255),
    ) -> Zone:
        z = Zone(zone_id, name, polygon, zone_type, camera_id, color)
        self._zones[zone_id] = z
        self._save_db()
        return z

    def remove_zone(self, zone_id: str) -> bool:
        if zone_id in self._zones:
            del self._zones[zone_id]
            self._save_db()
            return True
        return False

    def list_zones(self, camera_id: Optional[str] = None) -> List[Zone]:
        if camera_id:
            return [z for z in self._zones.values() if z.camera_id == camera_id]
        return list(self._zones.values())

    # ── Frame Processing ─────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[dict],
        camera_id: str = "cam0",
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Draw all zones and check detections for intrusions.
        Returns (annotated_frame, list_of_intrusion_events).
        """
        annotated = frame.copy()
        intrusions: List[dict] = []

        zones = self.list_zones(camera_id)
        for zone in zones:
            zone.draw(annotated)
            for det in detections:
                event = zone.check_detection(det)
                if event:
                    intrusions.append(event)

        return annotated, intrusions


def get_zone_service() -> ZoneService:
    svc = ZoneService()
    svc.initialise()
    return svc
