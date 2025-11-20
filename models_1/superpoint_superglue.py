import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# SUPERGLUE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "superglue"
# sys.path.append(str(SUPERGLUE_ROOT / "models"))

sys.path.insert(1,'/home/brucewayne/image3d/third_party/superglue/models')

from superglue import SuperGlue
from superpoint import SuperPoint




