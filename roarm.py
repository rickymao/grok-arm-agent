from roarm_sdk.roarm import roarm
from utils import call_serial

port = "/dev/ttyUSB0"
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
    roarm_client.pose_ctrl(pose)

