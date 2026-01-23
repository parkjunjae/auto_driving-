from setuptools import setup

package_name = 'pointcloud_transform'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='atoz',
    maintainer_email='atoz@todo.todo',
    description='PointCloud2 frame transformer',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'transform_node = pointcloud_transform.transform_node:main'
        ],
    },
)