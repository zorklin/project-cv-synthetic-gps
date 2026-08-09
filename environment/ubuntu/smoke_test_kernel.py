from __future__ import annotations

import argparse
from queue import Empty

from jupyter_client import KernelManager


parser = argparse.ArgumentParser()
parser.add_argument("--bag", help="Optional ROS 2 bag directory to open")
arguments = parser.parse_args()

bag_check = ""
if arguments.bag:
    bag_check = f"""
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

reader = SequentialReader()
reader.open(
    StorageOptions(uri={arguments.bag!r}, storage_id="mcap"),
    ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    ),
)
topics = {{item.name: item.type for item in reader.get_all_topics_and_types()}}
assert "/vision/velocity_frd" in topics
first_topic, first_data, first_timestamp = reader.read_next()
print(f"BagTopics={{len(topics)}}")
print(f"FirstMessage={{first_topic}}@{{first_timestamp}}")
"""

code = """
import os
import platform

import cv2
import gtsam
import gtsam_unstable
import numpy
import rosbag2_py

print(f"Python={platform.python_version()}")
print(f"ROS_DISTRO={os.environ.get('ROS_DISTRO')}")
print(f"NumPy={numpy.__version__}")
print(f"OpenCV={cv2.__version__}")
print(f"FixedLag={hasattr(gtsam_unstable, 'IncrementalFixedLagSmoother')}")
""" + bag_check

manager = KernelManager(kernel_name="project-cv-ros-humble")
manager.start_kernel()
client = manager.client()
client.start_channels()

try:
    client.wait_for_ready(timeout=30)
    message_id = client.execute(code)
    errors: list[str] = []

    while True:
        try:
            message = client.get_iopub_msg(timeout=30)
        except Empty as exc:
            raise RuntimeError("Timed out while waiting for kernel output") from exc

        if message.get("parent_header", {}).get("msg_id") != message_id:
            continue

        message_type = message["header"]["msg_type"]
        content = message["content"]

        if message_type == "stream":
            print(content["text"], end="")
        elif message_type == "error":
            errors.extend(content.get("traceback", []))
        elif message_type == "status" and content.get("execution_state") == "idle":
            break

    if errors:
        raise RuntimeError("Kernel execution failed:\n" + "\n".join(errors))
finally:
    client.stop_channels()
    manager.shutdown_kernel(now=True)

print("Jupyter kernel smoke test: OK")
