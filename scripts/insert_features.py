import argparse
from pathlib import Path
try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None
from PIL import Image
import numpy as np
import torch
import sys

sys.path.insert(1,'/home/brucewayne/image3d/utils')
# sys.path.insert(1,'/home/brucewayne/image3d/third_party/superglue/models')

from colmap_db_writer import ColmapDatabase
from superpoint_superglue import SuperGlueMatcher

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def main(image_dir: str, db_path: str, device: str):
    image_dir = Path(image_dir)
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    # Initialize DB + SuperPoint (we will only use SuperPoint here)
    db = ColmapDatabase(db_path)
    matcher = SuperGlueMatcher(device=device)

    # Fetch image_id mapping from DB
    cur = db.conn.cursor()
    cur.execute("SELECT image_id, name FROM images;")
    name_to_id = {name: image_id for image_id, name in cur.fetchall()}

    print(f"Inserting SuperPoint features for {len(images)} images...")

    for img_path in images:
        img_name = img_path.name
        image_id = name_to_id[img_name]

        # Load grayscale image (OpenCV or Pillow fallback)
        if cv2 is not None:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to load image: {img_path}")
            img = img.astype(np.float32) / 255.0
        else:
            img = Image.open(img_path).convert("L")
            img = np.asarray(img, dtype=np.float32) / 255.0

        # Prepare tensor
        t = torch.from_numpy(img)[None, None].to(matcher.device)

        # Run SuperPoint (expects key 'image')
        with torch.no_grad():
            pred = matcher.superpoint({"image": t})

        keypoints = pred["keypoints"][0].cpu().numpy()      # (N,2)
        descriptors = pred["descriptors"][0].cpu().numpy()  # (256,N)
        descriptors = descriptors.T                           # (N,256)

        if keypoints.shape[0] == 0:
            print(f"[WARN] No keypoints found in {img_name}")
            continue

        # Insert into DB
        db.add_keypoints(image_id, keypoints)
        db.add_descriptors(image_id, descriptors)

        print(f"  {img_name}: {keypoints.shape[0]} keypoints")

    db.commit()
    db.close()
    print("SuperPoint features inserted successfully.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--db_path", default="output/database.db")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    main(args.image_dir, args.db_path, args.device)
