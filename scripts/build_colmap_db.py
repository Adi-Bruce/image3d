# scripts/build_colmap_db.py
import argparse
from pathlib import Path
try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None
from PIL import Image
import sys

sys.path.insert(1,'/home/brucewayne/image3d/utils')

from colmap_db_writer import ColmapDatabase

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}



def main(image_dir: str, db_path: str):
    image_dir = Path(image_dir)
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    # Read first image to get width/height
    if cv2 is not None:
        im0 = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
        if im0 is None:
            raise RuntimeError(f"Cannot read {images[0]}")
        h, w = im0.shape[:2]
    else:
        im0 = Image.open(images[0]).convert("L")
        w, h = im0.size

    # Simple focal guess (works as a baseline)
    # A common heuristic: fx=fy=1.2*max(w,h), cx=w/2, cy=h/2
    f = 1.2 * max(w, h)
    cx, cy = w / 2.0, h / 2.0

    db = ColmapDatabase(db_path)
    cam_id = db.add_camera_pinhole(width=w, height=h, fx=f, fy=f, cx=cx, cy=cy)

    name_to_id = {}
    for img in images:
        img_id = db.add_image(img.name, cam_id)
        name_to_id[img.name] = img_id

    db.commit()
    db.close()

    print(f"DB created: {db_path}")
    print(f"Inserted {len(images)} images with camera_id={cam_id}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--db_path", default="database.db")
    args = ap.parse_args()
    main(args.image_dir, args.db_path)
