import os
from glob import glob

from setuptools import setup

package_name = 'cmd_vel_pid'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='world',
    maintainer_email='jihan1125@gmail.com',
    description='Angular PID cmd_vel bridge',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'angular_pid_cmdvel = cmd_vel_pid.angular_pid_cmdvel:main',
        ],
    },
)
