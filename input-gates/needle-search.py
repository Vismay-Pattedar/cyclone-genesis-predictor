"""
Cyclone Genesis Predictor -- dedicated search for a real "Model 3 high, but Gate 2 says NOT
cyclone" frame, i.e. a genuine example that would show a HIGH percentage in
the live susune app rather than short-circuiting to CYCLONE DECLARED.

Manual web-image hunting kept finding one of two failure modes: images
organized enough to score well on Model 3 also tripped Gate 2, or images
weak enough to dodge Gate 2 scored near-zero on Model 3. Rather than keep
guessing at single images, this runs the ACTUAL three checkpoints (uploaded
as a dataset, not re-derived) against a large REAL sample of genuine
pre-formation frames from Digital Typhoon and TCIR -- exactly the
distribution these models were trained on -- and reports what actually
clears both gates, empirically, across thousands of frames, sorted by
Model 3 score. No guessing which real-world photo might work.

Outputs: needle_search_full.csv (every frame scored, all 3 models), the top
candidate PNGs that pass Gate1 AND fail Gate2 (sorted by Model3 desc), and
summary stats on how common that combination even is.
"""
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

INPUT_ROOT = os.environ.get("CGP_INPUT_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("CGP_OUT_DIR", "/kaggle/working")
SEED = 1337
DT_EARLY_GRADE_MAX = 2
TCIR_EARLY_VMAX_KT = 33.0
TEMP_LO, TEMP_HI = 190.0, 300.0
IMG_SIZE = 224
NORM_MEAN, NORM_STD = 0.449, 0.226
MAX_FRAMES_PER_STORM = int(os.environ.get("CGP_FRAMES_PER_STORM", "12"))
TOP_N_TO_SAVE = 20


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


def find_tcir_h5():
    candidates = []
    for dirpath, dirnames, filenames in os.walk(INPUT_ROOT):
        dirnames[:] = [d for d in dirnames if "typhoon" not in d.lower()]
        for fn in filenames:
            if fn.lower().endswith((".h5", ".hdf5")) and ("tcsa" in fn.lower() or "tcir" in fn.lower()):
                candidates.append(os.path.join(dirpath, fn))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return candidates[0]


def find_checkpoints_dir():
    for dirpath, _, filenames in os.walk(INPUT_ROOT):
        if "gate1_satcheck.pt" in filenames:
            return dirpath
    return None


def even_picks(n_eligible, k):
    if n_eligible <= 0:
        return np.zeros(0, dtype=int)
    return np.unique(np.linspace(0, n_eligible - 1, min(k, n_eligible)).round().astype(int))


# ---------------------------------------------------------------- EARLY-ONLY indices
def build_dt_index_early(meta_dir, image_dir):
    csvs = sorted(f for f in os.listdir(meta_dir) if f.endswith(".csv") and f[:-4].isdigit())
    rows = []
    for name in csvs:
        storm_id = name[:-4]
        try:
            df = pd.read_csv(os.path.join(meta_dir, name))
        except Exception:
            continue
        need = {"grade", "file_1", "wind", "year", "month", "day"}
        if need - set(df.columns):
            continue
        if int(storm_id[:4]) < 1990:
            continue
        grade = pd.to_numeric(df["grade"], errors="coerce")
        wind = pd.to_numeric(df["wind"], errors="coerce").fillna(0.0)
        fname = df["file_1"].fillna("").astype(str).str.strip()
        crossed = (grade > DT_EARLY_GRADE_MAX).fillna(False).to_numpy()
        first_cross = int(np.argmax(crossed)) if crossed.any() else len(df)
        if first_cross < 4:
            continue
        eligible = np.arange(0, first_cross)
        eligible = eligible[fname.to_numpy()[eligible] != ""]
        if len(eligible) == 0:
            continue
        keep = eligible[even_picks(len(eligible), MAX_FRAMES_PER_STORM)]
        max_wind = float(wind.to_numpy().max())
        max_grade = float(np.nanmax(grade.to_numpy())) if len(grade) else 0.0
        for p in keep:
            fn = str(df.iloc[int(p)]["file_1"]).strip()
            h5_path = os.path.join(image_dir, storm_id, fn)
            if not os.path.exists(h5_path):
                continue
            rows.append({"source": "digital_typhoon", "storm_id": storm_id, "frame_index": int(p),
                         "h5_path": h5_path, "frame_ref": -1, "frame_wind_kt": float(wind.iloc[int(p)]),
                         "storm_max_wind_kt": max_wind, "storm_max_grade": max_grade})
    print(f"[DT] early-stage candidate frames: {len(rows)}")
    return pd.DataFrame(rows)


def build_tcir_index_early(info):
    rows = []
    vmax = pd.to_numeric(info["Vmax"], errors="coerce")
    work = pd.DataFrame({"frame_ref": np.arange(len(info), dtype=np.int64),
                         "storm_id": info["ID"].astype(str).to_numpy(), "vmax": vmax.to_numpy()})
    for storm_id, g in work.groupby("storm_id", sort=True):
        g = g.sort_values("frame_ref")
        vm = g["vmax"].to_numpy()
        crossed = np.isfinite(vm) & (vm > TCIR_EARLY_VMAX_KT)
        first_cross = int(np.argmax(crossed)) if crossed.any() else len(g)
        if first_cross < 4:
            continue
        eligible = np.arange(0, first_cross)
        if len(eligible) == 0:
            continue
        keep = eligible[even_picks(len(eligible), MAX_FRAMES_PER_STORM)]
        max_wind = float(np.nanmax(vm)) if np.isfinite(vm).any() else 0.0
        refs = g["frame_ref"].to_numpy()
        for p in keep:
            rows.append({"source": "tcir", "storm_id": storm_id, "frame_index": int(p),
                         "h5_path": "", "frame_ref": int(refs[p]),
                         "frame_wind_kt": float(vm[p]) if np.isfinite(vm[p]) else 0.0,
                         "storm_max_wind_kt": max_wind, "storm_max_grade": -1.0})
    print(f"[TCIR] early-stage candidate frames: {len(rows)}")
    return pd.DataFrame(rows)


def kelvin_to_display_u8(arr):
    arr = np.clip(arr, TEMP_LO, TEMP_HI)
    arr = 1.0 - (arr - TEMP_LO) / (TEMP_HI - TEMP_LO)
    return (arr * 255.0).astype(np.uint8)


def load_dt_frame_u8(h5_path):
    import h5py
    with h5py.File(h5_path, "r") as f:
        arr = np.asarray(f["Infrared"], dtype=np.float32)
    return kelvin_to_display_u8(arr)


def load_tcir_frame_u8(images_ds, row):
    frame = np.asarray(images_ds[row], dtype=np.float32)
    arr = frame[..., 0]
    finite = np.isfinite(arr)
    if finite.mean() < 0.5:
        raise ValueError("mostly non-finite")
    arr = np.nan_to_num(arr, nan=TEMP_HI, posinf=TEMP_HI, neginf=TEMP_LO)
    return kelvin_to_display_u8(arr)


def resize_u8(u8, size=IMG_SIZE):
    if u8.shape == (size, size):
        return u8
    t = torch.from_numpy(np.ascontiguousarray(u8)).float()[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze().clamp(0, 255).round().byte().numpy()


def build_resnet18_1ch(regression=False):
    import torchvision
    net = torchvision.models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ckpt_dir = find_checkpoints_dir()
    if ckpt_dir is None:
        raise SystemExit("Checkpoints dataset not found -- check dataset_sources.")
    print("Checkpoints:", ckpt_dir)

    def load_model(fname):
        ckpt = torch.load(os.path.join(ckpt_dir, fname), map_location="cpu", weights_only=False)
        net = build_resnet18_1ch()
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net

    gate1 = load_model("gate1_satcheck.pt")
    gate2 = load_model("gate2_cyclonecheck.pt")
    model3 = load_model("model3_v5final.pt")
    print("Loaded all 3 checkpoints.")

    dt_root = find_dt_root()
    tcir_path = find_tcir_h5()
    parts = []
    if tcir_path:
        info = pd.read_hdf(tcir_path, key="info", mode="r").reset_index(drop=True)
        parts.append(build_tcir_index_early(info))
    if dt_root:
        meta_dir, image_dir = resolve_dt_dirs(dt_root)
        parts.append(build_dt_index_early(meta_dir, image_dir))
    index = pd.concat(parts, ignore_index=True).reset_index(drop=True)
    print(f"Total early-stage candidate frames: {len(index)}")

    # ---- preload (TCIR first, ascending row order -- same discipline as training) ----
    n = len(index)
    imgs = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    ok = np.zeros(n, dtype=bool)
    sources = index["source"].to_numpy()
    paths = index["h5_path"].to_numpy()
    refs = index["frame_ref"].to_numpy()

    tc_pos = np.where(sources == "tcir")[0]
    if len(tc_pos) and tcir_path:
        import h5py
        order = np.argsort(refs[tc_pos], kind="stable")
        tc_pos = tc_pos[order]
        t0 = time.time()
        with h5py.File(tcir_path, "r") as f:
            ds = f["images"]
            for j, pos in enumerate(tc_pos):
                try:
                    imgs[pos] = resize_u8(load_tcir_frame_u8(ds, int(refs[pos])))
                    ok[pos] = True
                except Exception:
                    pass
                if (j + 1) % 1000 == 0:
                    print(f"  [TCIR] {j+1}/{len(tc_pos)} ({time.time()-t0:.0f}s)")
        print(f"[TCIR] preload done in {time.time()-t0:.0f}s")

    dt_pos = np.where(sources == "digital_typhoon")[0]
    t0 = time.time()
    for j, pos in enumerate(dt_pos):
        try:
            imgs[pos] = resize_u8(load_dt_frame_u8(paths[pos]))
            ok[pos] = True
        except Exception:
            pass
        if (j + 1) % 1000 == 0:
            print(f"  [DT] {j+1}/{len(dt_pos)} ({time.time()-t0:.0f}s)")
    print(f"[DT] preload done in {time.time()-t0:.0f}s, total failures={int((~ok).sum())}")

    index = index[ok].reset_index(drop=True)
    imgs = imgs[ok]
    n = len(index)
    print(f"Scoring {n} real, successfully-loaded frames...")

    # ---- batched inference through all 3 models ----
    BATCH = 64
    p1_all, p2_all, p3_all = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, BATCH):
            batch_u8 = imgs[i:i + BATCH]
            x = torch.from_numpy(batch_u8).float().unsqueeze(1) / 255.0
            x = (x - NORM_MEAN) / NORM_STD
            p1 = torch.sigmoid(gate1(x)).squeeze(1).numpy()
            p2 = torch.sigmoid(gate2(x)).squeeze(1).numpy()
            p3 = torch.sigmoid(model3(x)).squeeze(1).numpy()
            p1_all.append(p1); p2_all.append(p2); p3_all.append(p3)
            if (i // BATCH) % 20 == 0:
                print(f"  scored {i+len(batch_u8)}/{n} ({time.time()-t0:.0f}s)")
    index["gate1_prob"] = np.concatenate(p1_all)
    index["gate2_prob"] = np.concatenate(p2_all)
    index["model3_prob"] = np.concatenate(p3_all)
    print(f"Scoring done in {time.time()-t0:.0f}s")

    index.to_csv(os.path.join(OUT_DIR, "needle_search_full.csv"), index=False)

    # ---- the actual search: reaches Model 3 in the live app AND scores high ----
    reaches_model3 = index[(index["gate1_prob"] > 0.5) & (index["gate2_prob"] < 0.5)]
    print(f"\nFrames that would actually reach Model 3 in the live app: "
          f"{len(reaches_model3)} / {n} ({100*len(reaches_model3)/max(1,n):.2f}%)")
    if len(reaches_model3):
        print("Model3 prob distribution among those:")
        print(reaches_model3["model3_prob"].describe())

    top = reaches_model3.sort_values("model3_prob", ascending=False).head(TOP_N_TO_SAVE)
    print(f"\nTop {len(top)} candidates by Model3 score (Gate1>0.5, Gate2<0.5):")
    print(top[["source", "storm_id", "frame_index", "frame_wind_kt", "storm_max_wind_kt",
               "gate1_prob", "gate2_prob", "model3_prob"]].to_string())

    saved = []
    for rank, idx_label in enumerate(top.index):
        u8 = imgs[idx_label]
        row = index.loc[idx_label]
        fname = (f"rank{rank:02d}__{row['source']}__{row['storm_id']}__f{row['frame_index']}"
                 f"__g1_{row['gate1_prob']*100:.0f}__g2_{row['gate2_prob']*100:.0f}"
                 f"__m3_{row['model3_prob']*100:.0f}.png")
        Image.fromarray(u8, mode="L").save(os.path.join(OUT_DIR, fname))
        saved.append(fname)
    print(f"\nSaved {len(saved)} PNGs.")

    # also save the overall best-by-model3 regardless of gate2 (for comparison /
    # to show what Gate2 is correctly catching)
    top_any = index.sort_values("model3_prob", ascending=False).head(5)
    print("\nFor comparison -- top 5 Model3 scores REGARDLESS of Gate2 (most of these "
          "will show CYCLONE in the live app, not a percentage):")
    print(top_any[["source", "storm_id", "frame_index", "storm_max_wind_kt",
                    "gate1_prob", "gate2_prob", "model3_prob"]].to_string())

    print(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
