"""
Scan all VALIDATION (held-out) storms that eventually reach typhoon strength,
and rank their very first imaged frame by how visually "diffuse" / unorganised
it looks -- i.e. a good candidate for "no visible sign of a cyclone yet, but the
AI still calls it" demo footage.

Diffuseness proxy: after the same display-normalisation the model uses (cold
cloud tops -> bright), take pixels above the 85th brightness percentile (the
coldest/most convective cloud) and measure how SPREAD OUT they are spatially
(mean of std-dev of x and y positions). A tightly organised system has its cold
core clustered in one place (low spread); a genuinely diffuse, not-yet-organised
system has cold pixels scattered across the frame (high spread).

Also runs the trained model on that first frame so we can see, for each
candidate, both "how undramatic does this look" and "how confident is the AI
anyway" -- the best demo candidate is high on both.

Runs as a plain Kaggle CPU script kernel (no GPU needed for this).
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TEMP_LO, TEMP_HI = 190.0, 300.0
IMG_SIZE = 224
NORM_MEAN, NORM_STD = 0.449, 0.226


def find_dir(keyword):
    for dirpath, dirnames, _ in os.walk("/kaggle/input"):
        for d in dirnames:
            if keyword in d.lower():
                return os.path.join(dirpath, d)
    return None


def resolve_dirs(root):
    meta = os.path.join(root, "metadata")
    if os.path.isdir(os.path.join(meta, "metadata")):
        meta = os.path.join(meta, "metadata")
    img = os.path.join(root, "image")
    if os.path.isdir(os.path.join(img, "image")):
        img = os.path.join(img, "image")
    return meta, img


def find_checkpoint(exclude_root):
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if exclude_root and os.path.abspath(dirpath).startswith(os.path.abspath(exclude_root)):
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn == "model_best.pt":
                return os.path.join(dirpath, fn)
    return None


def load_frame_u8(h5_path):
    import h5py
    with h5py.File(h5_path, "r") as f:
        arr = np.asarray(f["Infrared"], dtype=np.float32)
    arr = np.clip(arr, TEMP_LO, TEMP_HI)
    arr = 1.0 - (arr - TEMP_LO) / (TEMP_HI - TEMP_LO)
    return (arr * 255.0).astype(np.uint8)


def diffuseness_score(u8):
    thresh = np.percentile(u8, 85)
    ys, xs = np.where(u8 >= thresh)
    if len(xs) < 50:
        return None  # basically nothing cold in frame -- not a useful candidate either way
    # Guard against the exact failure that fooled this metric last time: storm
    # 201003's top-ranked "diffuseness" turned out to be a saturated/corrupted
    # frame where ~100% of pixels read as "cold" (n_cold_px == 262144, the full
    # 512x512 frame), not genuine scattered convection. A frame that uniform is
    # not a useful candidate either way -- exclude it rather than let it win.
    if len(xs) > 0.90 * u8.size:
        return None
    spread = (xs.std() + ys.std()) / 2.0
    n_cold_px = len(xs)
    return float(spread), int(n_cold_px)


def build_model():
    import torchvision
    net = torchvision.models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


def to_model_input(u8):
    t = torch.from_numpy(u8).float().unsqueeze(0).unsqueeze(0) / 255.0
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return (t - NORM_MEAN) / NORM_STD


def main():
    typhoon_root = find_dir("typhoon")
    meta_dir, image_dir = resolve_dirs(typhoon_root)
    ckpt_path = find_checkpoint(typhoon_root)
    print("meta_dir  =", meta_dir)
    print("image_dir =", image_dir)
    print("checkpoint =", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    val_ids = set(ckpt["val_storm_ids"])
    print(f"{len(val_ids)} storms in validation split")

    net = build_model()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    rows = []
    for i, storm_id in enumerate(sorted(val_ids)):
        csv_path = os.path.join(meta_dir, f"{storm_id}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        grade = pd.to_numeric(df["grade"], errors="coerce")
        if not (grade == 5).any():
            continue  # only interested in storms that genuinely become typhoons

        fname = df["file_1"].fillna("").astype(str).str.strip()
        imaged = df[fname != ""].sort_values(["year", "month", "day", "hour"])
        if imaged.empty:
            continue
        first = imaged.iloc[0]
        h5_path = os.path.join(image_dir, storm_id, str(first["file_1"]).strip())
        if not os.path.exists(h5_path):
            continue

        try:
            u8 = load_frame_u8(h5_path)
        except Exception as exc:
            print(f"  LOAD FAIL {storm_id}: {exc}")
            continue

        ds = diffuseness_score(u8)
        if ds is None:
            continue
        spread, n_cold_px = ds

        with torch.no_grad():
            prob = float(torch.sigmoid(net(to_model_input(u8))).item())

        rows.append({
            "storm_id": storm_id,
            "n_imaged_frames": len(imaged),
            "first_frame_file": str(first["file_1"]).strip(),
            "first_frame_date": f"{int(first['year']):04d}-{int(first['month']):02d}-{int(first['day']):02d} {int(first['hour']):02d}Z",
            "max_wind": float(pd.to_numeric(df["wind"], errors="coerce").max()),
            "diffuseness_spread_px": spread,
            "n_cold_px": n_cold_px,
            "model_prob_frame1": prob,
        })

        if (i + 1) % 50 == 0:
            print(f"  scanned {i + 1}/{len(val_ids)} val storms, {len(rows)} candidates so far")

    result = pd.DataFrame(rows)
    result["year"] = result["storm_id"].str[:4].astype(int)
    # Require a reasonably long track (>=60 imaged frames) so the finished demo
    # isn't too short.
    result = result[result["n_imaged_frames"] >= 60].copy()

    # "Most dramatic AI-calls-it-early" = the frame looks UNDRAMATIC (high
    # diffuseness) AND the model is nonetheless CONFIDENT (high prob). Rank each
    # storm on both axes separately (percentile rank within this candidate pool,
    # so the two very different-scaled quantities combine fairly) and report the
    # combined score -- but this is a shortlist tool, not a final answer. Always
    # visually inspect the actual frame before picking; the diffuseness metric
    # alone was fooled once already (see the corrupted-frame guard above).
    result["diffuseness_pct"] = result["diffuseness_spread_px"].rank(pct=True)
    result["confidence_pct"] = result["model_prob_frame1"].rank(pct=True)
    result["drama_score"] = result["diffuseness_pct"] + result["confidence_pct"]

    result.to_csv("/kaggle/working/candidate_storms.csv", index=False)
    print(f"\nWrote /kaggle/working/candidate_storms.csv, {len(result)} candidates")

    print("\n=== TOP 20 BY COMBINED DRAMA SCORE (diffuse-looking AND confident) ===")
    print(result.sort_values("drama_score", ascending=False).head(20)
          [["storm_id", "year", "max_wind", "n_imaged_frames", "diffuseness_spread_px",
            "n_cold_px", "model_prob_frame1", "drama_score"]].to_string(index=False))

    print("\n=== TOP 15 MOST DIFFUSE FIRST-FRAME CANDIDATES (diffuseness alone) ===")
    print(result.sort_values("diffuseness_spread_px", ascending=False).head(15)
          [["storm_id", "year", "max_wind", "n_imaged_frames", "diffuseness_spread_px",
            "n_cold_px", "model_prob_frame1"]].to_string(index=False))

    print("\n=== TOP 15 BY RAW CONFIDENCE ON FRAME 1 (highest prob alone) ===")
    print(result.sort_values("model_prob_frame1", ascending=False).head(15)
          [["storm_id", "year", "max_wind", "n_imaged_frames", "diffuseness_spread_px",
            "n_cold_px", "model_prob_frame1"]].to_string(index=False))


if __name__ == "__main__":
    main()
