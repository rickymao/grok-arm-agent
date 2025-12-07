import json
from roarm_sdk.roarm import roarm
from utils import call_serial

port = "/dev/cu.usbserial-10"
class RoarmClient:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = roarm(roarm_type="roarm_m2", port=port, baudrate=115200)

        return cls._instance

def custom_post_ctrl(pose: list):
    """
    Custom post control function for the robot.
    Args:
        pose: The target pose to control the robot to.
    """
    roarm_client = RoarmClient()

    cmd = { "T":104,
           "x":pose[0],
           "y":pose[1],
           "z":pose[2],
           "t":pose[3],
           "spd":0.25}

    call_serial(json.dumps(cmd))

