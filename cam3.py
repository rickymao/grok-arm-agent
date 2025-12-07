import time
import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

# ===================== CONFIG =====================

CAMERA_INDEX = 0

INFER_EVERY_N_FRAMES = 2      # run foreground detection every N frames
TARGET_STRATEGY = "center"    # "center" or "largest"

# Fraction of image area below which blobs are considered noise
MIN_AREA_FRAC = 0.002

# ---- ROI CROP CONFIG ----
USE_ROI_CROP = True
ROI_X_MIN_FRAC = 0.1
ROI_X_MAX_FRAC = 0.9
ROI_Y_MIN_FRAC = 0.1
ROI_Y_MAX_FRAC = 0.7

# ---- ROTATION CONFIG ----
USE_ROTATE = True
# ROTATE_90_CLOCKWISE or ROTATE_90_COUNTERCLOCKWISE or ROTATE_180
ROTATE_MODE = cv2.ROTATE_90_CLOCKWISE


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


# ===================== UTILS: ROTATE + CROP =====================

def rotate_frame(frame):
    if not USE_ROTATE:
        return frame
    return cv2.rotate(frame, ROTATE_MODE)


def crop_frame(frame):
    """Crop frame to ROI defined by fractional bounds, or return unchanged if disabled."""
    if not USE_ROI_CROP:
        return frame

    h, w = frame.shape[:2]
    x1 = int(w * ROI_X_MIN_FRAC)
    x2 = int(w * ROI_X_MAX_FRAC)
    y1 = int(h * ROI_Y_MIN_FRAC)
    y2 = int(h * ROI_Y_MAX_FRAC)

    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    return frame[y1:y2, x1:x2].copy()


# ===================== FOREGROUND DETECTION =====================

def capture_background(cap, num_frames: int = 20) -> Optional[np.ndarray]:
    """
    Capture an approximate empty-table background.
    Returns a grayscale background image (rotated and cropped).
    """
    print("[INFO] Capturing background (please keep table empty)...")

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = rotate_frame(frame)
        frame = crop_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        time.sleep(0.02)

    if not frames:
        print("[WARN] Could not capture background.")
        return None

    bg = np.mean(frames, axis=0).astype(np.uint8)
    print("[INFO] Background captured.")
    return bg


def detect_foreground_objects(frame, background_gray: Optional[np.ndarray]) -> List[Detection]:
    """
    Detect blobs that differ from the background (anything that isn't the table).
    Assumes `frame` is already rotated + cropped (same as background).
    """
    if background_gray is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, background_gray)

    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = frame.shape[:2]
    min_area = MIN_AREA_FRAC * w * h

    detections: List[Detection] = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < min_area:
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
    if not detections:
        return None

    if strategy == "largest":
        return max(detections, key=lambda d: d.area)

    cx_center = frame_width // 2
    cy_center = frame_height // 2

    def dist_to_center(det: Detection) -> float:
        return math.hypot(det.cx - cx_center, det.cy - cy_center)

    return min(detections, key=dist_to_center)


def draw_overlay(frame, detections: List[Detection], target: Optional[Detection]):
    h, w = frame.shape[:2]

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

    cv2.drawMarker(
        frame,
        (w // 2, h // 2),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=1,
    )

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

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    background_gray = capture_background(cap)

    print("[INFO] Foreground detector (rotated + ROI) running... press 'q' to quit")

    frame_count = 0
    last_detections: List[Detection] = []
    last_target: Optional[Detection] = None
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera frame not received.")
            break

        frame = rotate_frame(frame)
        frame = crop_frame(frame)
        h, w = frame.shape[:2]

        frame_count += 1
        run_inference = (frame_count % INFER_EVERY_N_FRAMES == 0)

        if run_inference:
            start = time.time()

            detections = detect_foreground_objects(frame, background_gray)

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

        frame = draw_overlay(frame, last_detections, last_target)

        now = time.time()
        fps = 1.0 / (now - last_time) if now > last_time else 0.0
        last_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (frame.shape[1] - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Foreground Tabletop Detection (Rotated + Cropped)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()