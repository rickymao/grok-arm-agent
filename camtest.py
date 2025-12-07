import math
from dataclasses import dataclass
from typing import List

import cv2
from ultralytics import YOLO

# ===================== CONFIG =====================

MODEL_PATH = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.30
IOU_THRESHOLD = 0.45
INFER_EVERY_N_FRAMES = 2      # run YOLO every frame for testing
TARGET_CLASSES = None
CAMERA_INDEX = 0

# === MUST MATCH CALIBRATION SCRIPT EXACTLY ===
ROTATE_90_CLOCKWISE = False

CROP_ENABLED = True
CROP_TOP = 0
CROP_BOTTOM = 0
CROP_LEFT = 250
CROP_RIGHT = 600


# ---------- CROPPING (copied from calibration) ----------

def apply_rotation_and_crop(frame):
    """
    Apply the same rotation and crop that you use in calibration.

    Returns:
        cropped_frame
    """
    # 1) Rotate
    if ROTATE_90_CLOCKWISE:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    h, w = frame.shape[:2]

    if CROP_ENABLED:
        x1 = max(0, CROP_LEFT)
        x2 = max(0, w - CROP_RIGHT)
        y1 = max(0, CROP_TOP)
        y2 = max(0, h - CROP_BOTTOM)

        # Make sure bounds are valid
        x1 = min(x1, w - 1)
        x2 = max(x1 + 1, x2)
        y1 = min(y1, h - 1)
        y2 = max(y1 + 1, y2)

        frame_cropped = frame[y1:y2, x1:x2]
        return frame_cropped
    else:
        return frame


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


# ===================== YOLO =====================

def load_model() -> YOLO:
    print(f"[INFO] Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.fuse()
    return model


def detect_objects(frame, model: YOLO) -> List[Detection]:
    results = model(
        frame,
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )[0]

    detections: List[Detection] = []
    h, w = frame.shape[:2]

    for box, cls_id, conf in zip(results.boxes.xyxy,
                                 results.boxes.cls,
                                 results.boxes.conf):
        label = model.names[int(cls_id)]
        x1, y1, x2, y2 = map(int, box.tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        area = (x2 - x1) * (y2 - y1)
        # optional area filter
        if area < 0.002 * w * h:
            continue

        detections.append(
            Detection(label, float(conf), x1, y1, x2, y2, cx, cy, area)
        )

    return detections


# ===================== PUBLIC API =====================

def get_detections_map() -> dict:
    """
    Returns: {label: (cx, cy)} with coordinates IN THE CROPPED IMAGE.
    """
    model = load_model()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return {}

    print("[INFO] Running YOLO detection...")

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera frame not received.")
        cap.release()
        return {}

    # ⭐ Apply EXACT same crop as calibration
    frame = apply_rotation_and_crop(frame)
    print("[DEBUG] Cropped frame shape:", frame.shape)

    # Save what YOLO sees, for sanity
    cv2.imwrite("debug_cropped_cam.jpg", frame)
    print("[DEBUG] Wrote debug_cropped_cam.jpg")

    detections = detect_objects(frame, model)
    print(f"[INFO] Detected labels: {[d.label for d in detections]}")

    cap.release()

    return {d.label: (d.cx, d.cy) for d in detections}


# ===================== TEST HARNESS =====================

if __name__ == "__main__":
    dets = get_detections_map()
    print("Detections map:", dets)
