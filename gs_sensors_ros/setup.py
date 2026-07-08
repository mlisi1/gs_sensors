import os
from glob import glob

from setuptools import find_packages, setup

package_name = "gs_sensors_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "gs_sensor_core"],
    zip_safe=True,
    maintainer="mlisi1",
    maintainer_email="elechim2196@gmail.com",
    description="Phase 1 ROS 2 debug node: renders simulated camera sensor data from a trained 2DGS model at a Gazebo robot's pose.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_debug_node = gs_sensors_ros.camera_debug_node:main",
        ],
    },
)
