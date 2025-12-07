import time
import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
from ultralytics import YOLO

# ===================== CONFIG =====================

# Small, fast model. You can switch to "yolo11n.pt" if you've got that.
MODEL_PATH = "yolo11n.pt"

# Detection tuning
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45          # NMS inside YOLO
INFER_EVERY_N_FRAMES = 2      # run model every N frames (rest just reuse last result)
TARGET_STRATEGY = "center"    # "center" or "largest"

# Only keep “tabletop” objects (COCO names). Set to None to keep all.
# TARGET_CLASSES = {
#     "cup",
#     "bottle",
#     "wine glass",
#     "cell phone",
#     "remote",
#     "book",
#     "laptop",
#     "keyboard",
#     "mouse",
#     "tv",
#     "tvmonitor",
# }

TARGET_CLASSES = None  # Keep all classes

# Camera index
CAMERA_INDEX = 0

# ===================== DATA STRUCTURES =====================

@dataclass
class Detection:
    label: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    area: int

    def as_dict(self):
        return {
            "label": self.label,
            "conf": self.conf,
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "center": [self.cx, self.cy],
            "area": self.area,
        }


# ===================== CORE LOGIC =====================

def load_model() -> YOLO:
    print(f"[INFO] Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    # Optional: tiny speed improvement
    model.fuse()
    return model


def detect_objects(frame, model: YOLO) -> List[Detection]:
    """Run YOLO on a frame and return structured detections."""
    results = model(
        frame,
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )[0]

    detections: List[Detection] = []

    h, w = frame.shape[:2]

    for box, cls_id, conf in zip(
        results.boxes.xyxy,
        results.boxes.cls,
        results.boxes.conf,
    ):
        conf = float(conf)
        label = model.names[int(cls_id)]

        if TARGET_CLASSES is not None and label not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)

        # Skip tiny boxes that are probably noise
        if area < 0.002 * w * h:
            continue

        detections.append(
            Detection(
                label=label,
                conf=conf,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cx=cx,
                cy=cy,
                area=area,
            )
        )

    return detections


def choose_target(
    detections: List[Detection],
    frame_width: int,
    frame_height: int,
    strategy: str = TARGET_STRATEGY,
) -> Optional[Detection]:
    """Pick a single detection for the robot to act on."""
    if not detections:
        return None

    if strategy == "largest":
        return max(detections, key=lambda d: d.area)

    # Default: closest to image center (good for top-down pick)
    cx_center = frame_width // 2
    cy_center = frame_height // 2

    def dist_to_center(det: Detection) -> float:
        return math.hypot(det.cx - cx_center, det.cy - cy_center)

    return min(detections, key=dist_to_center)


def draw_overlay(frame, detections: List[Detection], target: Optional[Detection]):
    """Draw bounding boxes and a highlight around the chosen target."""
    h, w = frame.shape[:2]

    # Draw all detections
    for det in detections:
        color = (0, 255, 0)
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        label = f"{det.label} {det.conf:.2f}"
        cv2.putText(
            frame,
            label,
            (det.x1, max(20, det.y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        cv2.circle(frame, (det.cx, det.cy), 4, color, -1)

    # Highlight target
    if target is not None:
        color = (0, 0, 255)
        cv2.rectangle(frame, (target.x1, target.y1), (target.x2, target.y2), color, 3)
        cv2.circle(frame, (target.cx, target.cy), 6, color, -1)
        cv2.putText(
            frame,
            f"TARGET: {target.label}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    # Draw center crosshair (useful for top-down alignment)
    cv2.drawMarker(
        frame,
        (w // 2, h // 2),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=1,
    )

    # Count
    cv2.putText(
        frame,
        f"Objects: {len(detections)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    return frame


# ===================== MAIN LOOP =====================

def get_detections_map():
    model = load_model()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    print("[INFO] YOLO fast detector running... press 'q' to quit")

    frame_count = 0
    last_detections: List[Detection] = []
    last_target: Optional[Detection] = None
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera frame not received.")
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # Run detection every N frames
        run_inference = (frame_count % INFER_EVERY_N_FRAMES == 0)

        if run_inference:
            start = time.time()
            last_detections = detect_objects(frame, model)
            last_target = choose_target(last_detections, w, h)
            end = time.time()

            # === This is where you'd pass info to the robot ===
            if last_target is not None:
                # Example robot-friendly print:
                return { det.label: (det.cx, det.cy) for det in last_detections}
                print(
                    {
                        "strategy": TARGET_STRATEGY,
                        "target": last_target.as_dict(),
                        "num_detections": len(last_detections),
                        "latency_ms": round((end - start) * 1000, 1),
                    }
                )
            else:
                print(
                    {
                        "strategy": TARGET_STRATEGY,
                        "target": None,
                        "num_detections": len(last_detections),
                        "latency_ms": round((end - start) * 1000, 1),
                    }
                )

        # # Draw last known detections and target
        frame = draw_overlay(frame, last_detections, last_target)

        # # Show FPS
        # now = time.time()
        # fps = 1.0 / (now - last_time) if now > last_time else 0.0
        # last_time = now
        # cv2.putText(
        #     frame,
        #     f"FPS: {fps:.1f}",
        #     (w - 150, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (0, 255, 0),
        #     2,
        # )

        cv2.imshow("YOLO Fast Tabletop Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
