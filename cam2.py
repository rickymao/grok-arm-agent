import time
import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ===================== CONFIG =====================

# Model path (only used if USE_YOLO = True)
MODEL_PATH = "yolov8n.pt"

# Detection tuning
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45          # NMS inside YOLO
INFER_EVERY_N_FRAMES = 2      # run model every N frames (rest reuse last result)
TARGET_STRATEGY = "center"    # "center" or "largest"

# What to run
USE_YOLO = True
USE_COLOR_SEGMENTATION = True
USE_BACKGROUND_SUBTRACTION = True

# Only keep certain YOLO classes (None = keep all)
TARGET_CLASSES = None

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


# ---------- OPTION 2A: COLOR SEGMENTATION FOR BLOCKS ----------

def segment_colored_blocks(frame) -> List[Detection]:
    """
    Find colored blocks using HSV thresholds.
    You MUST tune the ranges below for your lighting/table.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    detections: List[Detection] = []

    kernel = np.ones((5, 5), np.uint8)

    def add_detections_from_mask(mask, label: str):
        nonlocal detections
        # Clean up mask
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            area = ww * hh
            if area < 0.002 * w * h:
                continue

            cx = x + ww // 2
            cy = y + hh // 2
            detections.append(
                Detection(
                    label=label,
                    conf=1.0,  # fake confidence
                    x1=x,
                    y1=y,
                    x2=x + ww,
                    y2=y + hh,
                    cx=cx,
                    cy=cy,
                    area=area,
                )
            )

    # Example ORANGE (tune!)
    lower_orange = (5, 100, 100)
    upper_orange = (25, 255, 255)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    add_detections_from_mask(mask_orange, "orange_block")

    # Example BLACK (low brightness) (tune!)
    # Often black blocks are just "very dark" regardless of hue.
    lower_black = (0, 0, 0)
    upper_black = (180, 255, 60)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    add_detections_from_mask(mask_black, "black_block")

    # Example WHITE (high brightness, low saturation) (tune!)
    lower_white = (0, 0, 200)
    upper_white = (180, 40, 255)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    add_detections_from_mask(mask_white, "white_block")

    return detections


# ---------- OPTION 2B: BACKGROUND SUBTRACTION ("NOT TABLE") ----------

def capture_background(cap, num_frames: int = 20) -> Optional[np.ndarray]:
    """
    Capture an approximate empty-table background.
    Ask the user to keep the table clear for a second.
    Returns a grayscale background image.
    """
    print("[INFO] Capturing background (please keep table empty)...")

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        time.sleep(0.02)

    if not frames:
        print("[WARN] Could not capture background.")
        return None

    # Average to smooth noise
    bg = np.mean(frames, axis=0).astype(np.uint8)
    print("[INFO] Background captured.")
    return bg


def detect_foreground_objects(frame, background_gray: Optional[np.ndarray]) -> List[Detection]:
    """
    Detect blobs that differ from the background (anything that isn't the table).
    """
    if background_gray is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, background_gray)

    # Threshold: pixels that changed by more than X
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Morphology to clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = frame.shape[:2]
    detections: List[Detection] = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < 0.002 * w * h:
            continue

        cx = x + ww // 2
        cy = y + hh // 2

        detections.append(
            Detection(
                label="foreground_object",
                conf=1.0,
                x1=x,
                y1=y,
                x2=x + ww,
                y2=y + hh,
                cx=cx,
                cy=cy,
                area=area,
            )
        )

    return detections


# ===================== TARGET SELECTION & DRAWING =====================

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

def extract_features_for_detections(frame, detections):
    """
    For each Detection, compute simple geometric + color features
    to send to Grok or use with rule-based logic.
    """
    features = []
    h, w = frame.shape[:2]

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        bw = x2 - x1
        bh = y2 - y1
        aspect = bw / bh if bh > 0 else 0.0
        aspect = max(aspect, 1.0 / aspect) if aspect > 0 else 0.0  # >= 1

        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0:
            mean_bgr = (0, 0, 0)
        else:
            mean_bgr = crop.mean(axis=(0, 1))  # B, G, R

        features.append({
            "id": idx,
            "bbox": [x1, y1, x2, y2],
            "center_px": [det.cx, det.cy],
            "width_px": bw,
            "height_px": bh,
            "area_px": det.area,
            "aspect_ratio": aspect,
            "mean_color_bgr": [float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])],
        })

    return features


# ===================== MAIN LOOP =====================

def main():
    model = load_model() if USE_YOLO else None

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    # Capture background (empty table) once, if enabled
    background_gray = None
    if USE_BACKGROUND_SUBTRACTION:
        background_gray = capture_background(cap)

    print("[INFO] Detector running... press 'q' to quit")

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

        run_inference = (frame_count % INFER_EVERY_N_FRAMES == 0)

        if run_inference:
            start = time.time()

            detections: List[Detection] = []

            if USE_YOLO and model is not None:
                detections.extend(detect_objects(frame, model))

            if USE_COLOR_SEGMENTATION:
                detections.extend(segment_colored_blocks(frame))

            if USE_BACKGROUND_SUBTRACTION:
                detections.extend(detect_foreground_objects(frame, background_gray))

            last_detections = detections
            last_target = choose_target(last_detections, w, h)
            end = time.time()

            if last_target is not None:
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

        # Draw last known detections and target
        frame = draw_overlay(frame, last_detections, last_target)

        # Show FPS
        now = time.time()
        fps = 1.0 / (now - last_time) if now > last_time else 0.0
        last_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (w - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Tabletop Detection (YOLO + Color + BG)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()