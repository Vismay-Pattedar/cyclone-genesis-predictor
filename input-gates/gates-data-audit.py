"""
Cyclone Genesis Predictor -- data audit for the two NEW gate models, before spending any GPU
quota on a training script that might guess a file layout wrong (the TCIR
'info' table loading bug earlier this project is exactly the failure mode
this audit exists to catch before it happens again).

Gate 1 (satellite vs random image): needs a NEGATIVE source of clearly
non-satellite images. Candidate: prasunroy/natural-images (everyday photos,
8 classes).

Gate 2 (cyclone vs non-cyclone clouds): needs a NEGATIVE source of REAL
satellite IR imagery that is NOT a tracked cyclone. Candidates:
  - troykueh/maritime-continent-satellite-weather (95,036 hourly Himawari-8/9
    B13 frames, 95-115E/5S-10N). VERIFIED against the real IBTrACS best-track
    archive (not assumed): 15 real storms had track points inside this exact
    box between 2015-07 and 2022-02 (the only window checked -- the archive
    file itself stops there), so this is NOT unconditionally clean and needs
    those windows excluded, not just "equatorial = safe" hand-waving.
  - pad1tya/himawari9-b13-thermal-ir (4 labelled Bay-of-Bengal event windows;
    3 of the 4 are non-cyclone weather, useful as HARD negatives -- real
    convective/monsoon weather that still isn't a tracked storm).

This script does NOT train anything. It only confirms real file layout,
column names, image shapes and dtypes for the two new sources (Digital
Typhoon and TCIR's layout is already proven in train_kernel_v5_final/train.py
and train_kernel_intensity_train/train.py -- not re-audited here).
"""
import os
from collections import Counter

import numpy as np
import pandas as pd

INPUT_ROOT = os.environ.get("CGP_INPUT_ROOT", "/kaggle/input")


def walk_report(root, max_depth=3, max_per_dir=12):
    print(f"\n--- directory tree under {root} (depth<={max_depth}) ---")
    root_depth = root.rstrip("/\\").count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip("/\\").count(os.sep) - root_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath) or dirpath}/  "
              f"({len(dirnames)} dirs, {len(filenames)} files)")
        for f in sorted(filenames)[:max_per_dir]:
            print(f"{indent}  {f}")
        if len(filenames) > max_per_dir:
            print(f"{indent}  ... +{len(filenames) - max_per_dir} more files")


def find_dataset_root(name_fragment):
    """Locate a dataset mount by a distinctive path fragment, pruning huge
    unrelated trees (same discipline as find_dt_root/find_tcir_h5 elsewhere
    in this project -- never blind-walk the whole /kaggle/input tree)."""
    hits = []
    for dirpath, dirnames, _ in os.walk(INPUT_ROOT):
        if name_fragment.lower() in dirpath.lower():
            hits.append(dirpath)
    return sorted(hits, key=len)[0] if hits else None


def audit_natural_images():
    print("\n" + "=" * 72)
    print("GATE 1 NEGATIVE SOURCE: prasunroy/natural-images")
    print("=" * 72)
    root = find_dataset_root("natural-images")
    if root is None:
        print("NOT FOUND -- dataset not attached or path fragment wrong.")
        return
    walk_report(root, max_depth=2)
    # Grab one real image file and confirm it opens.
    sample = None
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                sample = os.path.join(dirpath, f)
                break
        if sample:
            break
    if sample:
        from PIL import Image
        im = Image.open(sample)
        print(f"\nSample file: {sample}")
        print(f"  mode={im.mode} size={im.size}")
    else:
        print("No image file found under this root.")


def audit_maritime_continent():
    print("\n" + "=" * 72)
    print("GATE 2 NEGATIVE SOURCE: troykueh/maritime-continent-satellite-weather")
    print("=" * 72)
    root = find_dataset_root("maritime-continent")
    if root is None:
        print("NOT FOUND -- dataset not attached or path fragment wrong.")
        return
    walk_report(root, max_depth=2)

    csvs = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(dirpath, f))
    print(f"\nFound {len(csvs)} CSV files. Sample names: "
          f"{[os.path.basename(c) for c in csvs[:6]]}")

    city_csv = next((c for c in csvs if os.path.basename(c).lower()
                      not in ("regions.csv",)), None)
    if city_csv is None:
        print("No per-city CSV found.")
        return
    df = pd.read_csv(city_csv)
    print(f"\nLoaded {city_csv}")
    print(f"  shape={df.shape} columns={list(df.columns)}")
    print(df.head(3).to_string())

    if "image_path" not in df.columns:
        print("!! No 'image_path' column -- schema differs from the dataset card.")
        return
    has_img = df.get("has_image")
    sample_row = df[df["has_image"] == True].iloc[0] if has_img is not None else df.iloc[0]
    rel_path = str(sample_row["image_path"])
    print(f"\nSample image_path value: {rel_path!r}")

    # Try resolving it a few plausible ways rather than assuming one.
    candidates = [
        os.path.join(root, rel_path),
        os.path.join(os.path.dirname(city_csv), rel_path),
        os.path.join(root, rel_path.lstrip("/\\")),
    ]
    resolved = next((c for c in candidates if os.path.exists(c)), None)
    print("Candidates tried:")
    for c in candidates:
        print(f"  [{'OK' if os.path.exists(c) else 'no'}] {c}")

    if resolved:
        from PIL import Image
        im = Image.open(resolved)
        arr = np.asarray(im)
        print(f"\nResolved: {resolved}")
        print(f"  mode={im.mode} size={im.size} dtype={arr.dtype} "
              f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f}")
        # Confirm the documented calibration formula produces a sane Kelvin range.
        bt = arr.astype(np.float32) / 255.0 * 150.0 + 180.0
        print(f"  BT (documented formula) range: {bt.min():.1f}K - {bt.max():.1f}K "
              f"(expect roughly 180-330K)")
    else:
        print("!! Could not resolve image_path against any candidate root -- "
              "actual layout differs, must fix before training.")


def audit_bay_of_bengal():
    print("\n" + "=" * 72)
    print("GATE 2 HARD-NEGATIVE SOURCE: pad1tya/himawari9-b13-thermal-ir")
    print("=" * 72)
    root = find_dataset_root("himawari9-b13")
    if root is None:
        print("NOT FOUND -- dataset not attached or path fragment wrong.")
        return
    walk_report(root, max_depth=2)

    meta_path = None
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower() == "metadata.csv":
                meta_path = os.path.join(dirpath, f)
    if meta_path is None:
        print("No metadata.csv found.")
        return
    meta = pd.read_csv(meta_path)
    print(f"\nLoaded {meta_path}")
    print(f"  shape={meta.shape} columns={list(meta.columns)}")
    print(f"  label counts:\n{meta['label'].value_counts() if 'label' in meta.columns else 'no label column'}")
    print(meta.head(3).to_string())

    if "filename" not in meta.columns:
        print("!! No 'filename' column -- schema differs from the dataset card.")
        return
    sample_name = str(meta.iloc[0]["filename"])
    candidates = []
    for dirpath, _, filenames in os.walk(root):
        if sample_name in filenames:
            candidates.append(os.path.join(dirpath, sample_name))
    print(f"\nSample filename {sample_name!r} found at: {candidates}")
    if candidates:
        npz = np.load(candidates[0])
        print(f"  npz keys: {list(npz.keys())}")
        if "matrix" in npz:
            m = npz["matrix"]
            print(f"  matrix shape={m.shape} dtype={m.dtype} "
                  f"min={np.nanmin(m):.1f} max={np.nanmax(m):.1f} "
                  f"mean={np.nanmean(m):.1f} (expect Kelvin, ~190-310)")


def main():
    audit_natural_images()
    audit_maritime_continent()
    audit_bay_of_bengal()
    print("\nDone.")


if __name__ == "__main__":
    main()
