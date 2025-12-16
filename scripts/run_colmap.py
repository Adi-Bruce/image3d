import os
import shutil
import subprocess
import argparse
import sqlite3
from pathlib import Path

# Limit threading to avoid shared-memory issues in constrained environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

import numpy as np


MAX_IMAGE_ID = 2147483647


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = pair_id // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def dedup_matches(db_path: str):
    """
    Remove duplicate correspondences per pair to avoid COLMAP warnings and improve initialization.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Ensure table exists so UPDATE statements below do not fail.
    cur.execute(
        """
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
        """
    )
    cur.execute("SELECT pair_id, rows, cols, data FROM matches;")
    rows = cur.fetchall()

    updated = 0
    for pair_id, nrows, ncols, blob in rows:
        matches = np.frombuffer(blob, dtype=np.int32).reshape(nrows, ncols)
        unique = np.unique(matches, axis=0)
        if unique.shape[0] != matches.shape[0]:
            updated += 1
            cur.execute(
                "UPDATE matches SET rows=?, cols=?, data=? WHERE pair_id=?",
                (unique.shape[0], unique.shape[1], unique.astype(np.int32).tobytes(), pair_id),
            )
            # Keep two_view_geometries in sync if it already exists.
            cur.execute(
                "UPDATE two_view_geometries SET rows=?, cols=?, data=? WHERE pair_id=?",
                (unique.shape[0], unique.shape[1], unique.astype(np.uint32).tobytes(), pair_id),
            )

    if updated:
        conn.commit()
        print(f"[INFO] Deduplicated matches for {updated} pair(s).")

    conn.close()


def purge_features(db_path: str):
    """
    Remove existing keypoints/descriptors/matches/two_view_geometries so COLMAP
    can re-extract features cleanly (avoids descriptor shape/type mismatches).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for table in ("matches", "two_view_geometries", "descriptors", "keypoints"):
        cur.execute(f"DELETE FROM {table};")
    conn.commit()
    conn.close()
    print("[INFO] Cleared existing features/matches in database.")


def ensure_two_view_geometries(db_path: str):
    """
    COLMAP's mapper expects entries in two_view_geometries. Mirror the matches
    table so mapper can consume our SuperGlue matches directly.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
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
        """
    )

    cur.execute("SELECT pair_id, rows, cols, data FROM matches;")
    rows = cur.fetchall()

    kp_cache = {}
    if cv2 is not None:
        cur.execute("SELECT image_id, rows, cols, data FROM keypoints;")
        for image_id, k_rows, k_cols, data in cur.fetchall():
            arr = np.frombuffer(data, dtype=np.float32).reshape(k_rows, k_cols)
            kp_cache[int(image_id)] = arr[:, :2]
    else:
        print("[WARN] OpenCV not installed; skipping geometric verification of matches.")

    filled = 0
    for pair_id, m_rows, m_cols, blob in rows:
        matches = np.frombuffer(blob, dtype=np.int32).reshape(m_rows, m_cols)
        config = 2
        F_blob = None

        if cv2 is not None and matches.shape[0] >= 8:
            img1, img2 = pair_id_to_image_ids(pair_id)
            if img1 in kp_cache and img2 in kp_cache:
                pts1 = kp_cache[img1][matches[:, 0]]
                pts2 = kp_cache[img2][matches[:, 1]]
                F, mask = cv2.findFundamentalMat(
                    pts1,
                    pts2,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=2.0,
                    confidence=0.999,
                    maxIters=5000,
                )
                if F is not None and mask is not None:
                    inliers = mask.ravel().astype(bool)
                    if np.any(inliers):
                        matches = matches[inliers]
                        config = 1  # uncalibrated with F
                        F_blob = F.astype(np.float64).tobytes()

        cur.execute(
            """
            INSERT OR REPLACE INTO two_view_geometries
            (pair_id, rows, cols, data, config, F, E, H, qvec, tvec)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                pair_id,
                matches.shape[0],
                matches.shape[1],
                matches.astype(np.uint32).tobytes(),
                config,
                F_blob,
                None,
                None,
                None,
                None,
            ),
        )
        filled += 1

    conn.commit()
    if filled:
        print(f"[INFO] Filled/updated two_view_geometries for {filled} pair(s).")

    conn.close()

def db_feature_stats(db_path: str) -> dict[str, int]:
    """
    Return basic counts for key feature tables to decide whether we need to
    (re)run feature extraction + matching.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    stats = {}
    for table in ("images", "keypoints", "descriptors", "matches", "two_view_geometries"):
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            stats[table] = cur.fetchone()[0] or 0
        except sqlite3.OperationalError:
            stats[table] = 0
    conn.close()
    return stats

def run(cmd):
    print(" ".join(cmd))
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU if no GPU
    subprocess.run(cmd, check=True, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_path", help="Path to COLMAP executable (defaults to COLMAP_BIN or system colmap).")
    parser.add_argument("--db_path", default="output/database.db", help="Path to COLMAP database.")
    parser.add_argument("--image_path", default="datasets/bike", help="Path to images.")
    parser.add_argument("--use_colmap_matching", action="store_true", help="Use COLMAP feature_extractor + exhaustive_matcher to (re)create matches (will clear existing features).")
    args = parser.parse_args()

    colmap_bin = args.colmap_path or os.environ.get("COLMAP_BIN") or shutil.which("colmap")
    if not colmap_bin:
        raise FileNotFoundError(
            "COLMAP executable not found. Install COLMAP or set COLMAP_BIN to its path."
        )

    db_path = Path(args.db_path)
    image_path = Path(args.image_path)

    sparse_dir = Path("output/sparse")
    dense_dir = Path("output/dense")
    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)

    stats = db_feature_stats(str(db_path))
    use_colmap_matching = args.use_colmap_matching
    if not use_colmap_matching and (
        stats.get("matches", 0) == 0 or stats.get("keypoints", 0) == 0 or stats.get("descriptors", 0) == 0
    ):
        print("[INFO] No features/matches found in DB; running COLMAP feature extraction + matching.")
        use_colmap_matching = True

    if use_colmap_matching:
        purge_features(str(db_path))
        run([
            colmap_bin, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(image_path),
            "--FeatureExtraction.use_gpu", "0",
        ])
        run([
            colmap_bin, "exhaustive_matcher",
            "--database_path", str(db_path),
            "--FeatureMatching.use_gpu", "0",
        ])
    else:
        dedup_matches(str(db_path))
        ensure_two_view_geometries(str(db_path))

    run([
        colmap_bin, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_path),
        "--output_path", str(sparse_dir)
    ])

    run([
        colmap_bin, "image_undistorter",
        "--image_path", str(image_path),
        "--input_path", str(sparse_dir / "0"),
        "--output_path", str(dense_dir)
    ])

    run([
        colmap_bin, "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP"
    ])

    run([
        colmap_bin, "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense_dir / "fusion.ply")
    ])

if __name__ == "__main__":
    main()
