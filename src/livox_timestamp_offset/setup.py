from setuptools import setup

package_name = "livox_timestamp_offset"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/livox_timestamp_offset.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="world",
    maintainer_email="world@example.com",
    description="Apply fixed timestamp offset to Livox PointCloud2 for deskew stability.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "livox_timestamp_offset_node = livox_timestamp_offset.livox_timestamp_offset_node:main",
        ],
    },
)
