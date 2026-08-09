from __future__ import annotations

import platform
import sys


def version(module: object) -> str:
    return str(getattr(module, "__version__", "unknown"))


if platform.system() != "Linux":
    raise RuntimeError(f"Expected Linux, got {platform.system()}")

if sys.version_info[:2] != (3, 10):
    raise RuntimeError(f"Expected Python 3.10, got {platform.python_version()}")

import cv2
import gtsam
import gtsam_unstable
import matplotlib
import numpy
import pandas
import rosbag2_py
import rosbags
import yaml
from geometry_msgs.msg import TwistStamped
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage, Imu, NavSatFix, Range
from std_msgs.msg import Float64MultiArray

fixed_lag_classes = (
    "IncrementalFixedLagSmoother",
    "BatchFixedLagSmoother",
)
if not any(hasattr(gtsam_unstable, name) for name in fixed_lag_classes):
    raise RuntimeError("GTSAM fixed-lag smoother bindings are unavailable")

print(f"Python: {platform.python_version()}")
print(f"NumPy: {version(numpy)}")
print(f"OpenCV: {version(cv2)}")
print(f"pandas: {version(pandas)}")
print(f"matplotlib: {version(matplotlib)}")
print(f"rosbags: {version(rosbags)}")
print(f"GTSAM: {version(gtsam)}")
print("ROS 2 Python imports: OK")
print("GTSAM fixed-lag smoother: OK")
