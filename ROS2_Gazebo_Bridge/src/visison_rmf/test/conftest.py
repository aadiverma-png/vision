import os
import sys

# make `visison_rmf.core` and the `visison` library importable without a colcon build
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "..", "visison_gazebo"))
