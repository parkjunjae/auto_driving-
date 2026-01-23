import os
from glob import glob

from setuptools import setup

package_name = 'rl_local_policy'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='world',
    maintainer_email='jihan1125@gmail.com',
    description='Local policy node that publishes cmd_vel_raw',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_local_policy_node = rl_local_policy.rl_local_policy_node:main',
        ],
    },
)
