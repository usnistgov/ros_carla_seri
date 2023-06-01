import os
from glob import glob
from setuptools import setup

package_name = 'commander_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zeid',
    maintainer_email='zeid.kootbally@nist.gov',
    description='Controller for an ego vehicle in the carla simulator',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'commander_py = commander_py.commander_main:main',
        ],
    },
)
