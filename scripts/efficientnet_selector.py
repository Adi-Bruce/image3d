import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EfficientNetPairSelector:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.encoder = backbone.features.to(self.device)
        self.pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        self.encoder.eval()


        self.preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        ),
        ])


    def _encode_image(self, img_path: Path):
        img = Image.open(img_path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.encoder(x)
            pooled = self.pool(feat).view(1, -1)
        return pooled.cpu().numpy().ravel()


    def compute_descriptors(self, image_dir: str):
        image_paths = sorted([p for p in Path(image_dir).glob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        descs = [self._encode_image(p) for p in image_paths]
        return image_paths, np.stack(descs, axis=0)


    def select_topk_pairs(self, image_dir: str, topk_per_image: int = 10):
        image_paths, descs = self.compute_descriptors(image_dir)
        sim = cosine_similarity(descs)
        num_imgs = len(image_paths)


        pairs = set()
        for i in range(num_imgs):
            sims = sim[i]
            idxs = np.argsort(-sims)
        for j in idxs[1:1+topk_per_image]:
            a, b = min(i, j), max(i, j)
            pairs.add((a, b))


        return [(image_paths[i], image_paths[j]) for i, j in sorted(list(pairs))]