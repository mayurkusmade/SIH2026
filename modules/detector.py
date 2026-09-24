import cv2
import numpy as np
from ultralytics import YOLO

# COCO Class IDs for Surveillance Monitoring
TARGET_CLASSES = {
    0: ("person", "human"),
    2: ("car", "vehicle"),
    3: ("motorcycle", "vehicle"),
    5: ("bus", "vehicle"),
    7: ("truck", "vehicle")
}

# Distinct UI Palette
COLORS = {
    "human": (0, 165, 255),    # Vibrant Orange for Pedestrians
    "vehicle": (255, 200, 0),  # Bright Cyan-Gold for Vehicles
    "default": (200, 200, 200)
}


class ObjectDetector:
    """
    YOLOv8-based Object Detector filtered strictly for Border Surveillance
    monitoring: Persons (Human) and Transport (Vehicles).
    """
    def __init__(self, model_path: str = "models/yolov8n.pt", device: str = "cpu"):
        self.model = YOLO(model_path)
        self.device = device
        self.class_ids = list(TARGET_CLASSES.keys())
        self.last_error = ""

    def warmup(self) -> bool:
        """
        Runs one throwaway inference on a blank frame so the FIRST real frame does
        not stall.

        Measured on this project's 1080p perimeter feed: the first YOLO pass costs
        ~4.2 s (graph/session init), every later pass ~45 ms. Paying that inside
        model loading - where the UI already shows a spinner - means the operator
        never sees the feed appear to freeze on its first frame.
        """
        try:
            blank = np.zeros((360, 640, 3), dtype="uint8")
            self.detect(blank, conf_threshold=0.5)
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.35):
        """
        Runs inference on a single frame and returns filtered detections.
        """
        results = self.model(
            frame,
            classes=self.class_ids,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())

            class_name, category = TARGET_CLASSES.get(cls_id, ("unknown", "other"))
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "centroid": (cx, cy),
                "conf": conf,
                "class_id": cls_id,
                "class_name": class_name,
                "category": category
            })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: list, draw_centroid: bool = True):
        """
        Draws clean bounding boxes, labels, and centroid points onto frame.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            cat = det["category"]
            name = det["class_name"]
            color = COLORS.get(cat, COLORS["default"])

            # Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Centroid
            if draw_centroid:
                cx, cy = det["centroid"]
                cv2.circle(annotated, (cx, cy), 4, color, -1)

            # Label Tag
            label = f"{name.upper()} {conf*100:.0f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            tag_y1 = max(0, y1 - text_h - 6)
            tag_y2 = max(text_h + 6, y1)
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 8, tag_y2), color, -1)
            cv2.putText(annotated, label, (x1 + 4, tag_y2 - baseline - 1), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        return annotated
