"""
Re-extract the top needle-search candidates at their NATIVE Digital Typhoon
resolution (no downsizing to 224). The original search saved 224x224
thumbnails (its own internal scoring resolution), which turned out to be the
wrong thing to feed the real web app: the app upscales a 224px source to 512
in-browser, and the model graph downsizes it back to 224 internally, and that
upscale/downsample round trip visibly changes Gate 2's read (verified
locally: every 224px candidate flips to CYCLONE at 512, while the two
already-native-512 demo images score correctly). Saving native-resolution
frames removes the upscale step entirely -- exactly matches how real DT PNGs
reach the app in the first place.

Targets: the top candidates from needle_search_full.csv, digital_typhoon
source only (TCIR natively is 128x128, not the ~512 DT frames the demo
otherwise uses, so it's skipped here to stay visually consistent with the
rest of the demo set).
"""
import os

import h5py
import numpy as np
import pandas as pd
from PIL import Image

INPUT_ROOT = os.environ.get("CGP_INPUT_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("CGP_OUT_DIR", "/kaggle/working")
TEMP_LO, TEMP_HI = 190.0, 300.0

# (storm_id, frame_index, tag) -- top digital_typhoon candidates from
# needle_search_full.csv, ranked by Model3 score, TCIR entries skipped.
TARGETS = [
    ("201820", 26, "rank00_m3_75"),
    ("200515", 11, "rank01_m3_71"),
    ("199413", 10, "rank02_m3_70"),
    ("200307", 5, "rank03_m3_70"),
    ("201410", 8, "rank04_m3_67"),
    ("201309", 10, "rank05_m3_67"),
    ("200421", 6, "rank06_m3_65"),
    ("200312", 73, "rank09_m3_59"),
]


def find_dt_root():
    cands = []
    for dirpath, dirnames, _ in os.walk(INPUT_ROOT):
        dirnames[:] = [d for d in dirnames if d.lower() not in ("image", "metadata")]
        for d in list(dirnames):
            if "typhoon" in d.lower():
                cands.append(os.path.join(dirpath, d))
                dirnames.remove(d)
    if not cands:
        return None
    wp = [c for c in cands if "western-pacific" in c.lower() or c.lower().rstrip("/\\").endswith("wp")]
    return wp[0] if wp else sorted(cands)[0]


def resolve_dt_dirs(root):
    meta = os.path.join(root, "metadata")
    if os.path.isdir(os.path.join(meta, "metadata")):
        meta = os.path.join(meta, "metadata")
    img = os.path.join(root, "image")
    if os.path.isdir(os.path.join(img, "image")):
        img = os.path.join(img, "image")
    return meta, img


def kelvin_to_display_u8(arr):
    arr = np.clip(arr, TEMP_LO, TEMP_HI)
    arr = 1.0 - (arr - TEMP_LO) / (TEMP_HI - TEMP_LO)
    return (arr * 255.0).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dt_root = find_dt_root()
    if dt_root is None:
        raise SystemExit("Digital Typhoon dataset not found.")
    meta_dir, image_dir = resolve_dt_dirs(dt_root)
    print("meta_dir:", meta_dir, "image_dir:", image_dir)

    for storm_id, frame_index, tag in TARGETS:
        csv_path = os.path.join(meta_dir, f"{storm_id}.csv")
        if not os.path.exists(csv_path):
            print(f"SKIP {storm_id}: no metadata csv")
            continue
        df = pd.read_csv(csv_path)
        if frame_index >= len(df):
            print(f"SKIP {storm_id} f{frame_index}: out of range ({len(df)} rows)")
            continue
        fn = str(df.iloc[frame_index]["file_1"]).strip()
        h5_path = os.path.join(image_dir, storm_id, fn)
        if not os.path.exists(h5_path):
            print(f"SKIP {storm_id} f{frame_index}: h5 not found at {h5_path}")
            continue
        with h5py.File(h5_path, "r") as f:
            arr = np.asarray(f["Infrared"], dtype=np.float32)
        u8 = kelvin_to_display_u8(arr)
        print(f"{storm_id} f{frame_index}: native shape {u8.shape}")
        out_path = os.path.join(OUT_DIR, f"native__{tag}__{storm_id}__f{frame_index}.png")
        Image.fromarray(u8, mode="L").save(out_path)
        print("saved", out_path)

    print("Done.")


if __name__ == "__main__":
    main()
