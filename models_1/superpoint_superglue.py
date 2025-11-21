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

def load_image_gray(path: Path, resize=None):
    """
    Loads image in grayscale [0,1], optionally resizes.
    Returns numpy array HxW float32.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)

    return (img.astype(np.float32) / 255.0)


#

class SuperGlueMatcher:
    """
    High-level module to run SuperPoint + SuperGlue feature matching.
    """

    def __init__(
        self,
        device="cuda",
        resize=(640, 480),
        sp_config=None,
        sg_config=None
    ):
        # Device handling
        if device == "cuda" and not torch.cuda.is_available():
            print("[SuperGlueMatcher] CUDA not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Default configs
        default_sp = {
            "descriptor_dim": 256,
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": 1024
        }

        default_sg = {
            "weights": "indoor",       # indoor/outdoor
            "sinkhorn_iterations": 20,
            "match_threshold": 0.2
        }

        # Override defaults if user passed custom settings
        if sp_config:
            default_sp.update(sp_config)
        if sg_config:
            default_sg.update(sg_config)

        # Initialize models
        self.superpoint = SuperPoint(default_sp).to(self.device).eval()
        self.superglue  = SuperGlue(default_sg).to(self.device).eval()

        self.resize = resize

   
    def _to_tensor(self, img: np.ndarray):
        return torch.from_numpy(img)[None, None].float().to(self.device)

    def match_pair(self, img0_path: str | Path, img1_path: str | Path):
        """
        Runs SuperPoint + SuperGlue on two images.
        Returns:
            mkpts0  (Nx2): matched keypoints (img0)
            mkpts1  (Nx2): matched keypoints (img1)
            scores   (N):  confidence scores
        """
        img0_path = Path(img0_path)
        img1_path = Path(img1_path)

        # Load images
        img0 = load_image_gray(img0_path, resize=self.resize)
        img1 = load_image_gray(img1_path, resize=self.resize)

        # Convert to torch
        t0 = self._to_tensor(img0)
        t1 = self._to_tensor(img1)

        # Run SuperPoint on each image (it expects a single 'image' entry)
        with torch.no_grad():
            sp0 = self.superpoint({"image": t0})
            sp1 = self.superpoint({"image": t1})

            data = {
                "image0": t0,
                "image1": t1,
                "keypoints0": sp0["keypoints"][0][None],
                "keypoints1": sp1["keypoints"][0][None],
                "scores0": sp0["scores"][0][None],
                "scores1": sp1["scores"][0][None],
                "descriptors0": sp0["descriptors"][0][None],
                "descriptors1": sp1["descriptors"][0][None],
            }

            pred = self.superglue(data)

        # Extract keypoints, matches, scores
        kpts0 = data["keypoints0"][0].cpu().numpy()
        kpts1 = data["keypoints1"][0].cpu().numpy()
        matches0 = pred["matches0"][0].cpu().numpy()
        scores0 = pred["matching_scores0"][0].cpu().numpy()

        # Valid matches
        valid = matches0 > -1

        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches0[valid]]
        mscores = scores0[valid]

        return mkpts0, mkpts1, mscores

