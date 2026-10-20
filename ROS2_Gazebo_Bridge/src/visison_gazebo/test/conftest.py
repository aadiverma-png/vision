import os
import sys

# make `visison_gazebo` and the symlinked `visison` importable without a colcon build
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
