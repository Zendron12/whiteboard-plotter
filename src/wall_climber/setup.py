from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'wall_climber'


def package_files(directory: str, install_root: str):
    paths = []
    for root, _, filenames in os.walk(directory):
        if not filenames:
            continue
        install_path = os.path.join(install_root, os.path.relpath(root, directory))
        files = [os.path.join(root, filename) for filename in filenames]
        paths.append((install_path, files))
    return paths

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Ament index resource marker (REQUIRED for ament_python packages)
        ('share/ament_index/resource_index/packages',
         [os.path.join('resource', package_name)]),

        # Package manifest
        (os.path.join('share', package_name), ['package.xml']),

        # Launch / URDF / Worlds
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        *package_files('fonts', os.path.join('share', package_name, 'fonts')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),

        # Web UI assets
        *package_files('web', os.path.join('share', package_name, 'web')),
    ],
    install_requires=[
        # Keep runtime Python dependencies in sync with package.xml exec_depend entries.
        'setuptools',
        'fastapi',
        'uvicorn',
        'python-multipart',
        'matplotlib',
        'scikit-image',
    ],
    zip_safe=True,
    maintainer='hisham',
    maintainer_email='2144934@std.hu.edu.jo',
    description='Two-cable Webots drawing robot package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'urdf_spawner = wall_climber.urdf_spawner:main',
            'web_server = wall_climber.web_server:main',
        ],
    },
)
