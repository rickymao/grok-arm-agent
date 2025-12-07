import cv2
import numpy as np

# TODO: make sure this import works in your env
from roarm import RoarmClient

# ---------- Camera & Image Config ----------
CAMERA_ID = 0            # change if needed
SAVE_PATH = "homography.npy"
MIN_POINTS = 4           # at least 4 points required

# Rotate frame 90 degrees clockwise (like your other script)
ROTATE_90_CLOCKWISE = True

# Cropping borders AFTER rotation
# These are in pixels on the rotated image.
# You can tweak these values to adjust the visible/calibrated area.
CROP_ENABLED = True
CROP_TOP = 250        # pixels from top
CROP_BOTTOM = 600     # pixels from bottom
CROP_LEFT = 0       # pixels from left
CROP_RIGHT = 0      # pixels from right.  0 0 250 600 false

# ---------- Globals ----------
img_points = []          # list of (u, v) in *rotated + cropped* pixels
robot_points = []        # list of (X, Y) in robot coords
waiting_for_click = False
current_robot_xy = None

roarm_client = RoarmClient()


def get_robot_xy():
    """
    Read current robot pose and return (X, Y) on table.
    Adjust indexes if your pose_get has a different format.
    """
    pose = roarm_client.pose_get()   # e.g. [x, y, z, t, ...]
    x = float(pose[0])
    y = float(pose[1])
    return x, y


def apply_rotation_and_crop(frame):
    """
    Apply the same rotation and crop that you will use
    in your main detection pipeline.

    Returns:
        cropped_frame, (x_offset, y_offset)
    where offsets are the top-left pixel of the crop in the rotated image
    (useful if you later want global coordinates).
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
        return frame_cropped, (x1, y1)
    else:
        return frame, (0, 0)


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback on the DISPLAYED frame (rotated + cropped).
    (x, y) are coordinates in the cropped image.
    """
    global waiting_for_click, current_robot_xy, img_points, robot_points

    if event == cv2.EVENT_LBUTTONDOWN and waiting_for_click:
        # (x, y) are pixel coordinates in rotated + cropped frame
        img_points.append((x, y))
        robot_points.append(current_robot_xy)

        idx = len(img_points)
        print(f"[{idx}] img (cropped): ({x:.1f}, {y:.1f})  robot: ({current_robot_xy[0]:.3f}, {current_robot_xy[1]:.3f})")

        waiting_for_click = False
        current_robot_xy = None


def compute_and_save_homography():
    if len(img_points) < MIN_POINTS:
        print(f"Need at least {MIN_POINTS} points, currently have {len(img_points)}")
        return None

    img_pts = np.array(img_points, dtype=np.float32)
    robot_pts = np.array(robot_points, dtype=np.float32)

    H, mask = cv2.findHomography(img_pts, robot_pts, method=0)

    if H is None:
        print("Homography computation failed.")
        return None

    print("\nHomography matrix (pixels -> robot XY) in ROTATED+CROPPED coords:")
    print(H)

    np.save(SAVE_PATH, H)
    print(f"\nSaved homography to {SAVE_PATH}")
    return H


def main():
    global waiting_for_click, current_robot_xy

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: cannot open camera.")
        return

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    print("=== Homography Calibration (Rotated + Cropped) ===")
    print("Workflow:")
    print("  1) Move robot tip to a point on the table.")
    print("  2) Press 'c' to capture robot pose (X,Y).")
    print("  3) Click on the robot tip in the image window.")
    print(f"Rotation: 90° clockwise = {ROTATE_90_CLOCKWISE}")
    print(f"Cropping: enabled={CROP_ENABLED}, "
          f"TOP={CROP_TOP}, BOTTOM={CROP_BOTTOM}, LEFT={CROP_LEFT}, RIGHT={CROP_RIGHT}")
    print("Repeat for at least 4 points (more is better).")
    print("Press 'h' to compute & save homography.")
    print("Press 'q' to quit.\n")

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame, (offset_x, offset_y) = apply_rotation_and_crop(frame_raw)

        # Status text
        text1 = f"Points: {len(img_points)}  (need >= {MIN_POINTS})"
        text2 = "Press 'c' to capture pose, then click tip" if not waiting_for_click else "Click tip in image!"

        cv2.putText(frame, text1, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, text2, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw already selected points (in cropped coords)
        for i, (u, v) in enumerate(img_points):
            cv2.circle(frame, (int(u), int(v)), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i + 1), (int(u) + 5, int(v) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c') and not waiting_for_click:
            # Capture robot pose for this point
            try:
                current_robot_xy = get_robot_xy()
                waiting_for_click = True
                print(f"\nMove recorded: robot at X={current_robot_xy[0]:.3f}, Y={current_robot_xy[1]:.3f}")
                print("Now click the robot tip in the image.")
            except Exception as e:
                print(f"Error getting robot pose: {e}")
                current_robot_xy = None
                waiting_for_click = False

        elif key == ord('h'):
            compute_and_save_homography()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
