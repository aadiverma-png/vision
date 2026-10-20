from glob import glob

from setuptools import setup

package_name = 'visison_gazebo'

setup(
    name=package_name,
    version='0.1.0',
    # `visison` is a symlink to the algorithm library at the repository root,
    # so the node always runs the same code as the unit tests and the 2-D sim.
    packages=[package_name, 'visison'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/urdf', glob('urdf/*.xacro')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adi',
    maintainer_email='vaadi9902@gmail.com',
    description='Gazebo Harmonic scenario for the Visison obstacle bypass',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bypass_scenario = visison_gazebo.bypass_scenario_node:main',
            'demo_director = visison_gazebo.demo_director:main',
        ],
    },
)
