from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'semantic_mapper_vslam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='world',
    maintainer_email='jihan1125@gmail.com',
    description='Semantic mapping for RTAB-Map VSLAM with YOLO object detection',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_depth_mapper = semantic_mapper_vslam.yolo_depth_mapper:main',
            'yolo_depth_mapper_v5 = semantic_mapper_vslam.yolo_depth_mapper_v5:main',
            'yolo_rtabmap_fusion = semantic_mapper_vslam.yolo_rtabmap_fusion:main',
            'object_deduplicator = semantic_mapper_vslam.object_deduplicator:main',
            'http_publisher = semantic_mapper_vslam.http_publisher:main',
        ],
    },
)
