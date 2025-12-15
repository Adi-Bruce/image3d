# utils/colmap_db_writer.py
import os
import sqlite3
import numpy as np
from pathlib import Path


# Helpers for COLMAP DB format
def array_to_blob(arr: np.ndarray) -> bytes:
    return arr.tobytes()

def blob_to_array(blob: bytes, dtype, shape):
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)

def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    # COLMAP convention: pair_id = min * 2147483647 + max
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return int(image_id1) * 2147483647 + int(image_id2)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# COLMAP DB schema (minimal)
COLMAP_SCHEMA = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB NOT NULL,
    prior_focal_length INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
);

CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB NOT NULL,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB NOT NULL,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB
);

CREATE UNIQUE INDEX IF NOT EXISTS index_images_name ON images(name);
"""

# Camera models enum (COLMAP)
# We only need PINHOLE for now.
CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}

class ColmapDatabase:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(COLMAP_SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

    def add_camera_pinhole(self, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> int:
        model_id = CAMERA_MODEL_IDS["PINHOLE"]
        params = np.array([fx, fy, cx, cy], dtype=np.float64)
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO cameras(model, width, height, params, prior_focal_length) VALUES(?,?,?,?,?)",
            (model_id, width, height, array_to_blob(params), 1),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_image(self, name: str, camera_id: int) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO images(name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (name, camera_id, None, None, None, None, None, None, None),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_keypoints(self, image_id: int, keypoints_xy: np.ndarray):
        """
        COLMAP keypoints are typically stored as float32 with cols=4: x, y, scale, orientation.
        If you only have x,y, we’ll pad scale=1, orientation=0.
        """
        if keypoints_xy.ndim != 2 or keypoints_xy.shape[1] != 2:
            raise ValueError("keypoints_xy must be (N,2)")

        N = keypoints_xy.shape[0]
        kp = np.zeros((N, 4), dtype=np.float32)
        kp[:, 0:2] = keypoints_xy.astype(np.float32)
        kp[:, 2] = 1.0  # scale
        kp[:, 3] = 0.0  # orientation

        self.conn.execute(
            "INSERT OR REPLACE INTO keypoints(image_id, rows, cols, data) VALUES(?,?,?,?)",
            (image_id, kp.shape[0], kp.shape[1], array_to_blob(kp)),
        )

    def add_descriptors(self, image_id: int, desc: np.ndarray):
        """
        Store descriptors as float32 (SuperPoint outputs float descriptors).
        """
        if desc.ndim != 2:
            raise ValueError("desc must be (N,D)")
        desc = desc.astype(np.float32)
        self.conn.execute(
            "INSERT OR REPLACE INTO descriptors(image_id, rows, cols, data) VALUES(?,?,?,?)",
            (image_id, desc.shape[0], desc.shape[1], array_to_blob(desc)),
        )

    def add_matches(self, image_id1: int, image_id2: int, matches: np.ndarray):
        """
        matches: (M,2) int32 pairs: [idx_in_img1, idx_in_img2]
        """
        if matches.size == 0:
            return
        if matches.ndim != 2 or matches.shape[1] != 2:
            raise ValueError("matches must be (M,2)")
        matches = matches.astype(np.int32)

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        self.conn.execute(
            "INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) VALUES(?,?,?,?)",
            (pair_id, matches.shape[0], matches.shape[1], array_to_blob(matches)),
        )

    def commit(self):
        self.conn.commit()
