from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

leader_port = "/dev/tty.usbmodem58760432871"
follower_port = "/dev/tty.usbmodem58760434241"

leader_arm = DynamixelMotorsBus(
    port=leader_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (0, "xl330-m077"),
        "shoulder_lift": (1, "xl330-m077"),
        "elbow_flex": (2, "xl330-m077"),
        "wrist_flex": (3, "xl330-m077"),
        "wrist_roll": (4, "xl330-m077"),
        "gripper": (5, "xl330-m077"),
    },
)

follower_arm = DynamixelMotorsBus(
    port=follower_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (0, "xl430-w250"),
        "shoulder_lift": (1, "xl430-w250"),
        "elbow_flex": (2, "xl330-m288"),
        "wrist_flex": (3, "xl330-m288"),
        "wrist_roll": (4, "xl330-m288"),
        "gripper": (5, "xl330-m288"),
    },
)

def main():
    robot = ManipulatorRobot(
        robot_type="koch",
        leader_arms={"main": leader_arm},
        follower_arms={"main": follower_arm},
        calibration_dir=".cache/calibration/koch",
        cameras={
            "laptop": OpenCVCamera(0, fps=30, width=640, height=480),
            "phone": OpenCVCamera(1, fps=30, width=640, height=480),
        },
    )
    robot.connect()

if __name__ == "__main__":
    main()