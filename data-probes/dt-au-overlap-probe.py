"""
Cyclone Genesis Predictor -- probe: does Digital Typhoon AU overlap with Digital Typhoon WP?

Triggered by finding storm 197901's first frame is BYTE-IDENTICAL (same MD5)
between the WP and AU Kaggle mirrors. That could mean:
  (a) the whole AU dataset is largely/fully a re-hosting of WP-visible storms
      (in which case "480 new storms" is false -- most would be duplicates
      already in the training set), or
  (b) it's a handful of border-case/dual-catalogued storms and the other
      ~470-ish AU storms are genuinely new Southern Hemisphere data.

CPU-only, no GPU needed. Root discovery is intentionally SHALLOW and PRUNED --
never descends into an "image" directory while searching for the two dataset
roots, because that subtree contains one folder per storm (hundreds to
thousands) each holding dozens of .h5 files; a blind os.walk there would be
extremely slow. Once a root is confirmed (has both a metadata-ish and an
image-ish subtree), the walk into it stops.
"""
import hashlib
import os
import random

import h5py
import numpy as np
import pandas as pd

INPUT_ROOT = "/kaggle/input"
MAX_DEPTH = 6  # generous bound; real mounts are 2-4 levels deep


def find_dataset_root(keyword, max_depth=MAX_DEPTH):
    """Depth-bounded search for a directory whose path contains `keyword` and
    which itself contains a 'metadata' and an 'image' subdirectory (directly
    or one level down). Never descends INTO 'image' or 'metadata' subtrees --
    those are checked with a cheap listdir, not walked."""
    start_depth = INPUT_ROOT.rstrip("/\\").count(os.sep)

    def walk(path, depth):
        if depth > max_depth:
            return None
        try:
            entries = os.listdir(path)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            return None
        if keyword in path.lower():
            has_meta = "metadata" in entries
            has_img = "image" in entries
            if has_meta and has_img:
                return path
        for e in entries:
            full = os.path.join(path, e)
            if not os.path.isdir(full):
                continue
            if e.lower() in ("image", "metadata"):
                continue  # never descend into the huge per-storm subtree here
            found = walk(full, depth + 1)
            if found:
                return found
        return None

    return walk(INPUT_ROOT, start_depth)


print("Searching for WP root (bounded, pruned)...")
wp_root = find_dataset_root("western-pacific")
print(f"wp_root = {wp_root}")
print("Searching for AU root (bounded, pruned)...")
au_root = find_dataset_root("dataset-au")
print(f"au_root = {au_root}")

if wp_root is None or au_root is None:
    raise SystemExit(f"Could not find both roots. wp_root={wp_root} au_root={au_root}")


def resolve_meta_and_image(root):
    meta_top = os.path.join(root, "metadata")
    img_top = os.path.join(root, "image")
    # handle one extra level of nesting (metadata/metadata, image/image)
    meta_inner = os.path.join(meta_top, "metadata")
    img_inner = os.path.join(img_top, "image")
    meta = meta_inner if os.path.isdir(meta_inner) else meta_top
    img = img_inner if os.path.isdir(img_inner) else img_top
    return meta, img


wp_meta, wp_img = resolve_meta_and_image(wp_root)
au_meta, au_img = resolve_meta_and_image(au_root)
print(f"wp_meta={wp_meta}\nau_meta={au_meta}\nwp_img={wp_img}\nau_img={au_img}")

wp_ids = sorted(f[:-4] for f in os.listdir(wp_meta) if f.endswith(".csv"))
au_ids = sorted(f[:-4] for f in os.listdir(au_meta) if f.endswith(".csv"))
print(f"\nn_wp_storms={len(wp_ids)}  n_au_storms={len(au_ids)}")

wp_set, au_set = set(wp_ids), set(au_ids)
overlap = sorted(wp_set & au_set)
au_only = sorted(au_set - wp_set)
print(f"overlap_by_id={len(overlap)}  au_only={len(au_only)}  wp_only={len(wp_set - au_set)}")


def first_frame_path(img_root, storm_id):
    d = os.path.join(img_root, storm_id)
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.endswith(".h5"))
    return os.path.join(d, files[0]) if files else None


def hash_frame(path):
    with h5py.File(path, "r") as f:
        arr = np.asarray(f["Infrared"], dtype=np.float32)
    return hashlib.md5(arr.tobytes()).hexdigest()


print("\n--- checking overlap IDs: is the underlying image data actually identical? ---")
n_checked = n_identical = n_different = n_missing = 0
CHECK_CAP = 80
for sid in overlap[:CHECK_CAP]:
    wp_p, au_p = first_frame_path(wp_img, sid), first_frame_path(au_img, sid)
    if wp_p is None or au_p is None:
        n_missing += 1
        continue
    try:
        h_wp, h_au = hash_frame(wp_p), hash_frame(au_p)
    except Exception as exc:
        print(f"  {sid}: ERROR {type(exc).__name__}: {exc}")
        continue
    n_checked += 1
    if h_wp == h_au:
        n_identical += 1
    else:
        n_different += 1
        print(f"  {sid}: DIFFERENT first-frame hash (wp={h_wp[:8]} au={h_au[:8]})")
print(f"checked={n_checked} (cap={CHECK_CAP} of {len(overlap)} overlap ids)  "
      f"identical={n_identical}  different={n_different}  missing_dir={n_missing}")

print("\n--- AU-only storms: real distinct tracks? sample lat/lon + grade + wind ---")
random.seed(1337)
sample = random.sample(au_only, min(15, len(au_only)))
for sid in sample:
    try:
        df = pd.read_csv(os.path.join(au_meta, sid + ".csv"))
        lat_min, lat_max = df["lat"].min(), df["lat"].max()
        lon_min, lon_max = df["lng"].min(), df["lng"].max()
        grade_max = df["grade"].max()
        wind_max = df["wind"].max()
        year = df["year"].iloc[0]
        print(f"  {sid}: year={year} lat=[{lat_min:.1f},{lat_max:.1f}] lon=[{lon_min:.1f},{lon_max:.1f}] "
              f"grade_max={grade_max} wind_max={wind_max} n_rows={len(df)}")
    except Exception as exc:
        print(f"  {sid}: ERROR {type(exc).__name__}: {exc}")

print(f"\nn_wp_only(not in AU)={len(wp_set - au_set)} / n_wp_total={len(wp_ids)}")

print("\n--- overlap storms: how many have an early (grade<=2) frame? ---")
n_overlap_has_early = 0
for sid in overlap[:CHECK_CAP]:
    try:
        df = pd.read_csv(os.path.join(au_meta, sid + ".csv"))
        if (pd.to_numeric(df["grade"], errors="coerce") <= 2).any():
            n_overlap_has_early += 1
    except Exception:
        pass
print(f"of first {min(CHECK_CAP, len(overlap))} overlap ids, {n_overlap_has_early} have a grade<=2 (early) row")

print("\nDONE")
