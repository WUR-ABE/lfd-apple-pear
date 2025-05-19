import os
from glob import glob
from setuptools import setup, find_packages

package_name = "lfd_executor"
pbdlib_custom = package_name + "/pbdlib_custom"
rtb_model = package_name + "/rtb_model"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name, pbdlib_custom, rtb_model],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share/", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.urdf")),
    ],
    install_requires=["setuptools", "python-math", "numpy", "transforms3d", "pickle-mixin", "roboticstoolbox-python", "pymycobot"],
    zip_safe=True,
    maintainer="agro-legion",
    maintainer_email="robert.vandeven@wur.nl",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fruit_pick_executor = lfd_executor.fruit_pick_executor:main",
        ],
    },
)
