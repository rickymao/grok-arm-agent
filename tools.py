from langchain_core.tools import tool
from cam import get_detections_map
from roarm import RoarmClient, custom_post_ctrl
import time

roarm_client = RoarmClient()

def check_pose(pose_target: list) -> bool:
    """
    Check if the robot is close to the target pose.
    Args:
        pose_target: The target pose to check.
    Returns:
        True if the robot is close to the target pose, False otherwise.
    """
    count = 0
    while True:
        current_pose = roarm_client.pose_get()
        current_pose = [round(x, 2) for x in current_pose]
        pose_target = [round(x, 2) for x in pose_target]
        if all(abs(a - b) < 20 for a, b in zip(current_pose[:3], pose_target[:3])):
            return True
        else:
            time.sleep(1)
            count += 1
            if count > 50:
                print(f"Failed to move to position {pose_target}, please try again. Current pose: {current_pose}.")
                return False

@tool
def move_to_home_position():
    """Move to the home position"""
    home_radians = [0.0, 0.0, 1.5708, 0.0]
    roarm_client.joints_radian_ctrl(radians=home_radians, speed=500, acc=0)
    return "Moved to home position."

@tool
def pick_up_object(object_name: str):
    """Pick up an object by name
    Args:
        object_name: The name of the object to pick up
    Returns:
        A string indicating that the object has been picked up
    """
    detections_map = get_detections_map()
    if object_name not in detections_map:
        return f"Object {object_name} not found in detections map."
    object_center = detections_map[object_name]
    object_center = [round(x, 2) for x in object_center]
    return f"Picked up {object_name} at center {object_center}."

@tool
def set_led_brightness(brightness: int):
    """"Set the brightness of the LED light between 0 and 255 inclusive, with 0 being off and 255 being full brightness.
    Args:
        brightness (int): Brightness level for the LED light (0-255)
    Returns:
        str: Confirmation message indicating the LED has been turned on with the specified brightness
    """
    roarm_client.led_ctrl(brightness)
    return f"LED turned on with brightness {brightness}."

@tool
def get_all_joint_radians():
    """Get the current radians of all joints
    Returns:
        A list of radians for each joint
        Example: {
            "base_joint": 0.0,
            "shoulder_joint": 0.0,
            "elbow_joint": 1.5708,
            "gripper_joint": 0.0
        }
        mapping:
        - 0: base joint
        - 1: shoulder joint
        - 2: elbow joint
        - 3: gripper joint
    """
    radians = roarm_client.joints_radian_get()
    radians = [round(x, 2) for x in radians]
    return {"base_joint": radians[0], "shoulder_joint": radians[1], "elbow_joint": radians[2], "gripper_joint": radians[3]}

@tool
def move_robot_position(
    up: float = 0.0,
    down: float = 0.0,
    forward: float = 0.0,
    backward: float = 0.0,
    left: float = 0.0,
    right: float = 0.0,
) -> str:
    """
    Move the robot by relative offsets in millimeters along the Cartesian axes.
    Args:
        up: Positive Z movement in mm.
        down: Negative Z movement in mm.
        forward: Positive X movement in mm.
        backward: Negative X movement in mm.
        left: Negative Y movement in mm.
        right: Positive Y movement in mm.
    Returns:
        A string indicating the new target position.
    """
    current_pose = roarm_client.pose_get()
    current_pose = [round(x, 3) for x in current_pose]
    pose = [
        current_pose[0] + forward - backward,
        current_pose[1] + right - left,
        current_pose[2] + up - down,
        current_pose[3],
    ]
    pose = [round(x, 3) for x in pose]
    custom_post_ctrl(pose)
    if not check_pose(pose):
        return f"Failed to move to position {pose}, please try again"
    return f"Moved to position {pose}."

@tool
def set_gripper(is_closed: bool):
    """
    Open or close the gripper using a boolean.

    Args:
        is_closed (bool): 
            True  -> fully close the gripper
            False -> fully open the gripper

    The gripper's physical limits:
        Open  ≈ 1.9 rad
        Close ≈ -0.2 rad
    """
    GRIPPER_OPEN = 1.9
    GRIPPER_CLOSE = -0.2

    target = GRIPPER_CLOSE if is_closed else GRIPPER_OPEN

    roarm_client.gripper_radian_ctrl(target, speed=500, acc=0)

    state = "closed" if is_closed else "opened"
    return f"Gripper {state} (radian={target})."

@tool
def wait(seconds: int):
    """wait for a given number of seconds before issuing the next command
    Args:
        seconds: The number of seconds to wait
    Returns:
        A string indicating that the robot has waited for the given number of seconds
    """
    time.sleep(seconds)
    return "Waited for " + str(seconds) + " seconds."