from setuptools import setup

package_name = 'rl_pid_training'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='world',
    maintainer_email='world@example.com',
    description='RL PID training scaffolding for cmd_vel tracking',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'train_pid = rl_pid_training.train_pid:main',
        ],
    },
)
