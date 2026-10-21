"""
Cyclone Genesis Predictor -- Gate 1: "is this real calibrated satellite IR data at all?"

The FIRST of two new gates ahead of the existing predictors. Answers a
narrower question than either the early-stage or intensity model: not
"is there a cyclone here", just "is this the kind of data those models were
ever trained on" -- a single-channel brightness-temperature frame, not an
ordinary photo, a QR code, or noise. Real users kept feeding the demo app
random photos during testing; this replaces the old crude "is it a colour
image" heuristic with a trained classifier.

POSITIVE class: any real IR frame from Digital Typhoon, TCIR, OR the newly
verified troykueh/maritime-continent-satellite-weather Himawari set --
cyclone present or not, ANY stage. Gate 1 does not care whether a cyclone is
in the frame, only whether the frame itself is real satellite data. (Gate 2,
a separate script, is the one that cares about cyclone-vs-not; keeping that
distinction is why this gate's positives are NOT filtered to storms only.)

NEGATIVE class: prasunroy/natural-images (everyday photos, 8 classes) PLUS
synthetic degenerate patterns generated on the fly (solid colours, gradients,
uniform noise, checkerboard/QR-like grids) -- the synthetic half exists
specifically because real users tried exactly this ("random duck images",
"a QR code") and a purely photo-based negative class would not cover it.

Both real sources verified against the ACTUAL Kaggle mount by
train_kernel_gates_audit/audit.py before this script was written (natural
images at natural_images/{class}/{class}_NNNN.jpg; maritime-continent at
images/{year}/{date}/{HHMM}.png, BT = pixel/255*150+180, confirmed to land in
192-303K on a real sample -- inside this project's existing 190-300K
convention, so the same kelvin_to_display_u8 used everywhere else applies
unchanged).

Outputs (under OUT_DIR): model_best.pt, metrics.json, dataset_index.csv.
"""
import datetime
import json
import math
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 1337
MODEL_VERSION = "gate1_satcheck_v1"

INPUT_ROOT = os.environ.get("CGP_INPUT_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("CGP_OUT_DIR", "/kaggle/working")

TEMP_LO, TEMP_HI = 190.0, 300.0
IMG_SIZE = 224
NORM_MEAN, NORM_STD = 0.449, 0.226

# Maritime-continent calibration, verified against a real sample by the audit
# script: BT(K) = pixel/255*150 + 180.
MC_BT_SCALE, MC_BT_OFFSET = 150.0, 180.0

MAX_FRAMES_PER_STORM = int(os.environ.get("CGP_FRAMES_PER_STORM", "20"))
MAX_FRAMES_PER_DAY_MC = int(os.environ.get("CGP_FRAMES_PER_DAY_MC", "3"))
PRELOAD_BUDGET_GB = float(os.environ.get("CGP_PRELOAD_BUDGET_GB", "5.0"))

VAL_FRACTION = 0.2
EPOCHS = int(os.environ.get("CGP_EPOCHS", "10"))
WARMUP_EPOCHS = 1
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 5e-2
EMA_DECAY = 0.999
USE_AMP = True

CKPT_PATH = os.path.join(OUT_DIR, "model_best.pt")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")
INDEX_PATH = os.path.join(OUT_DIR, "dataset_index.csv")


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    return obj


def even_picks(n_eligible, k):
    if n_eligible <= 0:
        return np.zeros(0, dtype=int)
    return np.unique(np.linspace(0, n_eligible - 1, min(k, n_eligible)).round().astype(int))


# ----------------------------------------------------------------------------
# Dataset discovery (Digital Typhoon / TCIR paths verbatim from the proven
# code in train_kernel_v5_final / train_kernel_intensity_train).
# ----------------------------------------------------------------------------
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


def find_dataset_root(fragment):
    hits = [dp for dp, _, _ in os.walk(INPUT_ROOT) if fragment.lower() in dp.lower()]
    return sorted(hits, key=len)[0] if hits else None


# ----------------------------------------------------------------------------
# POSITIVE indices -- ANY STAGE (no early/mature filtering: Gate 1 doesn't
# care about cyclone lifecycle, only "is this a real IR frame").
# ----------------------------------------------------------------------------
def build_dt_index_any(meta_dir, image_dir):
    csvs = sorted(f for f in os.listdir(meta_dir) if f.endswith(".csv") and f[:-4].isdigit())
    rows = []
    audit = Counter()
    for name in csvs:
        storm_id = name[:-4]
        try:
            df = pd.read_csv(os.path.join(meta_dir, name))
        except Exception:
            audit["n_unreadable"] += 1
            continue
        need = {"file_1", "wind", "year", "month", "day"}
        if need - set(df.columns):
            audit["n_bad_columns"] += 1
            continue
        year = int(storm_id[:4])
        if year < 1977:
            continue
        fname = df["file_1"].fillna("").astype(str).str.strip()
        eligible = np.where(fname.to_numpy() != "")[0]
        if len(eligible) == 0:
            continue
        keep = eligible[even_picks(len(eligible), MAX_FRAMES_PER_STORM)]
        for p in keep:
            rec = df.iloc[int(p)]
            h5_path = os.path.join(image_dir, storm_id, str(rec["file_1"]).strip())
            if not os.path.exists(h5_path):
                audit["n_missing_h5"] += 1
                continue
            rows.append({"group_key": f"dt_{storm_id}", "cls": 1, "source": "digital_typhoon",
                         "h5_path": h5_path, "frame_ref": -1})
    print(f"[DT any-stage] storms={len(csvs)} kept_frames={len(rows)} audit={dict(audit)}")
    return pd.DataFrame(rows)


def build_tcir_index_any(info):
    rows = []
    work = pd.DataFrame({
        "frame_ref": np.arange(len(info), dtype=np.int64),
        "storm_id": info["ID"].astype(str).to_numpy(),
    })
    for storm_id, g in work.groupby("storm_id", sort=True):
        idx = g["frame_ref"].to_numpy()
        keep = idx[even_picks(len(idx), MAX_FRAMES_PER_STORM)]
        for fr in keep:
            rows.append({"group_key": f"tcir_{storm_id}", "cls": 1, "source": "tcir",
                         "h5_path": "", "frame_ref": int(fr)})
    print(f"[TCIR any-stage] storms={work['storm_id'].nunique()} kept_frames={len(rows)}")
    return pd.DataFrame(rows)


def build_maritime_continent_index(mc_root, max_per_day=MAX_FRAMES_PER_DAY_MC):
    """Every CSV is one city's view of the SAME hourly frames (image_path is
    the join key, shared across cities) -- so only ONE csv needs reading to
    enumerate frames, not all 23. singapore.csv is picked arbitrarily; any
    city csv lists the identical image_path set per the dataset's own schema.
    No storm filtering here (Gate 1 doesn't care); capped per CALENDAR DAY
    (not per storm -- there is no storm concept here) since hourly frames on
    the same day are highly correlated."""
    csvs = [f for f in os.listdir(mc_root) if f.endswith(".csv") and f != "regions.csv"]
    if not csvs:
        return pd.DataFrame()
    df = pd.read_csv(os.path.join(mc_root, csvs[0]))
    df = df[df["has_image"] == True].copy()
    rows = []
    for date, g in df.groupby("date", sort=True):
        idx = g.index.to_numpy()
        keep = idx[even_picks(len(idx), max_per_day)]
        for i in keep:
            rows.append({"group_key": f"mc_{date}", "cls": 1, "source": "maritime_continent",
                         "h5_path": os.path.join(mc_root, str(df.loc[i, "image_path"])), "frame_ref": -1})
    print(f"[MC any-stage] days={df['date'].nunique()} kept_frames={len(rows)}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# NEGATIVE indices.
# ----------------------------------------------------------------------------
def build_natural_images_index(root):
    rows = []
    seen = set()
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and f not in seen:
                seen.add(f)
                rows.append({"group_key": f"natimg_{f}", "cls": 0, "source": "natural_images",
                             "h5_path": os.path.join(dirpath, f), "frame_ref": -1})
    print(f"[natural-images] distinct files={len(rows)}")
    return pd.DataFrame(rows)


N_SYNTHETIC = int(os.environ.get("CGP_N_SYNTHETIC", "1500"))


def build_synthetic_index():
    # frame_ref encodes a synthetic pattern id; no file on disk.
    rows = [{"group_key": f"synth_{i}", "cls": 0, "source": "synthetic",
             "h5_path": "", "frame_ref": i} for i in range(N_SYNTHETIC)]
    print(f"[synthetic] generated={len(rows)}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Image loading.
# ----------------------------------------------------------------------------
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
        raise ValueError("TCIR IR1 mostly non-finite")
    arr = np.nan_to_num(arr, nan=TEMP_HI, posinf=TEMP_HI, neginf=TEMP_LO)
    return kelvin_to_display_u8(arr)


def load_mc_frame_u8(png_path):
    from PIL import Image
    arr = np.asarray(Image.open(png_path), dtype=np.float32)
    bt_kelvin = arr / 255.0 * MC_BT_SCALE + MC_BT_OFFSET
    return kelvin_to_display_u8(bt_kelvin)


def load_natural_image_u8(path):
    """Real photo -> grayscale display domain. NOT a Kelvin conversion (this
    is not calibrated IR data) -- standard luminance, matching the app's own
    gray_pixels.dart treatment of a non-IR image."""
    from PIL import Image
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.uint8)


_SYNTH_RNG = np.random.RandomState(SEED + 7)


def make_synthetic_u8(pattern_id, size=256):
    kind = pattern_id % 5
    rng = np.random.RandomState(SEED + 1000 + pattern_id)
    if kind == 0:  # solid / near-solid colour
        v = rng.randint(0, 256)
        arr = np.full((size, size), v, dtype=np.float32) + rng.normal(0, 3, (size, size))
    elif kind == 1:  # smooth gradient
        x = np.linspace(rng.randint(0, 256), rng.randint(0, 256), size)
        arr = np.tile(x, (size, 1))
    elif kind == 2:  # uniform random noise
        arr = rng.randint(0, 256, (size, size)).astype(np.float32)
    elif kind == 3:  # checkerboard / QR-like grid
        cell = rng.randint(8, 40)
        yy, xx = np.mgrid[0:size, 0:size]
        arr = (((yy // cell) + (xx // cell)) % 2) * 255.0
        arr = np.abs(arr - rng.normal(0, 10, (size, size)))
    else:  # sparse random blocks (QR-code-ish noise, not a smooth gradient)
        arr = np.full((size, size), 255.0)
        n_blocks = rng.randint(20, 80)
        for _ in range(n_blocks):
            bs = rng.randint(4, 30)
            y0, x0 = rng.randint(0, size - bs), rng.randint(0, size - bs)
            arr[y0:y0 + bs, x0:x0 + bs] = rng.choice([0.0, 255.0])
    return np.clip(arr, 0, 255).astype(np.uint8)


def resize_u8(u8):
    if u8.shape == (IMG_SIZE, IMG_SIZE):
        return u8
    t = torch.from_numpy(np.ascontiguousarray(u8)).float()[None, None]
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return t.squeeze().clamp(0, 255).round().byte().numpy()


def to_unit(u8_batch):
    return u8_batch.unsqueeze(1).float().div_(255.0)


def normalize(x01):
    return (x01 - NORM_MEAN) / NORM_STD


def to_model_input(u8_batch):
    return normalize(to_unit(u8_batch))


def augment_batch(x01, gen, aug):
    p = aug
    n = x01.shape[0]
    dev = x01.device

    def rnd(lo, hi):
        return (torch.rand(n, generator=gen) * (hi - lo) + lo).to(dev)

    zoom = torch.exp(rnd(-p["zoom_log"], p["zoom_log"]))
    tx, ty = rnd(-p["shift"], p["shift"]), rnd(-p["shift"], p["shift"])
    theta = torch.zeros(n, 2, 3, device=dev, dtype=x01.dtype)
    theta[:, 0, 0] = 1.0 / zoom
    theta[:, 1, 1] = 1.0 / zoom
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(x01.shape), align_corners=False)
    x01 = F.grid_sample(x01, grid, mode="bilinear", padding_mode="border", align_corners=False)

    gain = rnd(1.0 - p["contrast"], 1.0 + p["contrast"]).view(n, 1, 1, 1)
    off = rnd(-p["brightness"], p["brightness"]).view(n, 1, 1, 1)
    x01 = (x01 - 0.5) * gain + 0.5 + off
    if p["noise_std"] > 0:
        x01 = x01 + torch.randn(x01.shape, generator=gen).to(dev) * p["noise_std"]
    return x01.clamp_(0.0, 1.0)


AUG = {"brightness": 0.10, "contrast": 0.20, "zoom_log": 0.10, "shift": 0.10, "noise_std": 0.015}


class FrameDataset(Dataset):
    def __init__(self, images_u8, labels, augment):
        self.images = images_u8
        self.labels = np.asarray(labels, dtype=np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augment:
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img = np.rot90(img, k).copy()
            if torch.rand(1).item() < 0.5:  # flips are FINE for this gate -- unlike the
                img = np.fliplr(img).copy()  # storm models, chirality is not the question here
        return torch.from_numpy(img), torch.tensor(self.labels[i], dtype=torch.float32)


def preload_all(index):
    n = len(index)
    print(f"\nAllocating preload buffer: {n} x {IMG_SIZE} x {IMG_SIZE} uint8 = {n * IMG_SIZE * IMG_SIZE / 1e9:.2f} GB")
    out = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    ok = np.ones(n, dtype=bool)
    sources = index["source"].to_numpy()
    paths = index["h5_path"].to_numpy()
    refs = index["frame_ref"].to_numpy()

    tcir_path = find_tcir_h5()
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
                    out[pos] = resize_u8(load_tcir_frame_u8(ds, int(refs[pos])))
                except Exception as exc:
                    print(f"  LOAD FAIL tcir row {refs[pos]}: {exc}")
                    ok[pos] = False
                if (j + 1) % 2000 == 0:
                    print(f"  [TCIR] {j + 1}/{len(tc_pos)} ({time.time() - t0:.0f}s)")
        print(f"[TCIR] preload done in {time.time() - t0:.0f}s")

    dt_pos = np.where(sources == "digital_typhoon")[0]
    t0 = time.time()
    for j, pos in enumerate(dt_pos):
        try:
            out[pos] = resize_u8(load_dt_frame_u8(paths[pos]))
        except Exception as exc:
            print(f"  LOAD FAIL {paths[pos]}: {exc}")
            ok[pos] = False
        if (j + 1) % 2000 == 0:
            print(f"  [DT] {j + 1}/{len(dt_pos)} ({time.time() - t0:.0f}s)")
    if len(dt_pos):
        print(f"[DT] preload done in {time.time() - t0:.0f}s")

    mc_pos = np.where(sources == "maritime_continent")[0]
    t0 = time.time()
    for j, pos in enumerate(mc_pos):
        try:
            out[pos] = resize_u8(load_mc_frame_u8(paths[pos]))
        except Exception as exc:
            print(f"  LOAD FAIL {paths[pos]}: {exc}")
            ok[pos] = False
        if (j + 1) % 5000 == 0:
            print(f"  [MC] {j + 1}/{len(mc_pos)} ({time.time() - t0:.0f}s)")
    if len(mc_pos):
        print(f"[MC] preload done in {time.time() - t0:.0f}s")

    ni_pos = np.where(sources == "natural_images")[0]
    t0 = time.time()
    for j, pos in enumerate(ni_pos):
        try:
            out[pos] = resize_u8(load_natural_image_u8(paths[pos]))
        except Exception as exc:
            print(f"  LOAD FAIL {paths[pos]}: {exc}")
            ok[pos] = False
    if len(ni_pos):
        print(f"[natural-images] preload done in {time.time() - t0:.0f}s")

    sy_pos = np.where(sources == "synthetic")[0]
    for pos in sy_pos:
        out[pos] = resize_u8(make_synthetic_u8(int(refs[pos])))
    if len(sy_pos):
        print(f"[synthetic] generated {len(sy_pos)} patterns")

    print(f"Preload total failures: {int((~ok).sum())}/{n}")
    return out, ok


def build_model():
    import torchvision
    pretrained = True
    try:
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception as exc:
        print(f"!! PRETRAINED WEIGHTS UNAVAILABLE ({exc}) -- using RANDOM INIT !!")
        net = torchvision.models.resnet18(weights=None)
        pretrained = False
    old = net.conv1.weight.data
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.conv1.weight.data = old.mean(dim=1, keepdim=True)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net, pretrained


class ModelEma:
    def __init__(self, model, decay):
        self.decay = decay
        self.n_updates = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.non_float = {k: v.detach().clone() for k, v in model.state_dict().items() if not v.dtype.is_floating_point}

    def update(self, model):
        self.n_updates += 1
        d = min(self.decay, (1.0 + self.n_updates) / (10.0 + self.n_updates))
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.shadow.items():
                v.mul_(d).add_(msd[k].detach().float(), alpha=1.0 - d)
            for k in self.non_float:
                self.non_float[k].copy_(msd[k].detach())

    def state_dict(self, reference):
        ref = reference.state_dict()
        return {k: (self.shadow.get(k, self.non_float.get(k))).detach().clone().to(dtype=v.dtype, device=v.device)
                for k, v in ref.items()}


def binary_metrics(probs, labels, thresh=0.5):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    order = np.argsort(-probs)
    p, l = probs[order], labels[order]
    n_pos, n_neg = l.sum(), len(l) - l.sum()
    if n_pos == 0 or n_neg == 0:
        auc = float("nan")
    else:
        ranks = pd.Series(probs).rank().to_numpy()
        auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    pred = (probs >= thresh).astype(np.float64)
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    acc = (tp + tn) / max(1.0, tp + fp + tn + fn)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"n": int(len(labels)), "auc": float(auc), "accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def amp_autocast(enabled):
    try:
        return torch.amp.autocast("cuda", enabled=enabled, dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    print("=" * 72)
    print(f"Cyclone Genesis Predictor GATE 1: SATELLITE-VS-RANDOM -- {MODEL_VERSION}")
    print("=" * 72)

    dt_root = find_dt_root()
    tcir_path = find_tcir_h5()
    mc_root = find_dataset_root("maritime-continent")
    ni_root = find_dataset_root("natural-images")
    print("DT root:", dt_root, "\nTCIR h5:", tcir_path, "\nMC root:", mc_root, "\nNI root:", ni_root)
    missing = [n for n, v in [("Digital Typhoon", dt_root), ("TCIR", tcir_path),
                               ("maritime-continent", mc_root), ("natural-images", ni_root)] if v is None]
    if missing:
        raise SystemExit(f"Missing dataset mounts: {missing} -- check dataset_sources.")

    parts = []
    info = pd.read_hdf(tcir_path, key="info", mode="r").reset_index(drop=True)
    parts.append(build_tcir_index_any(info))
    meta_dir, image_dir = resolve_dt_dirs(dt_root)
    parts.append(build_dt_index_any(meta_dir, image_dir))
    parts.append(build_maritime_continent_index(mc_root))
    parts.append(build_natural_images_index(ni_root))
    parts.append(build_synthetic_index())
    index = pd.concat(parts, ignore_index=True).reset_index(drop=True)

    print(f"\nClass balance: {index['cls'].value_counts().to_dict()}")
    print(f"By source:\n{index['source'].value_counts()}")

    bytes_per = IMG_SIZE * IMG_SIZE
    need_gb = len(index) * bytes_per / 1e9
    if need_gb > PRELOAD_BUDGET_GB:
        frac = PRELOAD_BUDGET_GB / need_gb
        index = index.groupby("source", group_keys=False).apply(
            lambda g: g.iloc[even_picks(len(g), max(1, int(len(g) * frac)))])
        index = index.reset_index(drop=True)
        print(f"!! Subsampled to fit {PRELOAD_BUDGET_GB} GB budget -> {len(index)} frames")

    # Split at GROUP level (storm / calendar-day / filename), never frame
    # level -- same discipline as every other model in this project.
    groups = index["group_key"].unique()
    rng = np.random.RandomState(SEED)
    rng.shuffle(groups)
    n_val_groups = max(1, int(round(len(groups) * VAL_FRACTION)))
    val_groups = set(groups[:n_val_groups])
    index["split"] = np.where(index["group_key"].isin(val_groups), "val", "train")

    images, ok = preload_all(index)
    index = index[ok].reset_index(drop=True)
    images = images[ok]
    index.to_csv(INDEX_PATH, index=False)

    labels = index["cls"].to_numpy(np.float32)
    is_val = (index["split"] == "val").to_numpy()
    tr_idx, va_idx = np.where(~is_val)[0], np.where(is_val)[0]
    print(f"\nFrames: train={len(tr_idx)} val={len(va_idx)}")
    val_sources = index["source"].to_numpy()[va_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(USE_AMP and device.type == "cuda")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    train_ds = FrameDataset(images[tr_idx], labels[tr_idx], augment=True)
    val_ds = FrameDataset(images[va_idx], labels[va_idx], augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    net, pretrained = build_model()
    net = net.to(device)
    ema = ModelEma(net, EMA_DECAY)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_at(e):
        if e < WARMUP_EPOCHS:
            return (e + 1) / max(1, WARMUP_EPOCHS)
        t = (e - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = make_scaler(use_amp)
    aug_gen = torch.Generator()

    @torch.no_grad()
    def evaluate(loader):
        net.eval()
        probs, labs = [], []
        for imgs_u8, y in loader:
            x = to_model_input(imgs_u8.to(device))
            with amp_autocast(use_amp):
                out = torch.sigmoid(net(x)).squeeze(1).float()
            probs.append(out.cpu().numpy())
            labs.append(y.numpy())
        return np.concatenate(probs), np.concatenate(labs)

    history, best_auc, best_epoch, best_state = [], -1.0, -1, None
    for epoch in range(1, EPOCHS + 1):
        aug_gen.manual_seed(SEED * 1000 + epoch)
        net.train()
        tot_l, n_batches = 0.0, 0
        t0 = time.time()
        for imgs_u8, y in train_dl:
            x01 = to_unit(imgs_u8.to(device, non_blocking=True))
            x = normalize(augment_batch(x01, aug_gen, AUG))
            yt = y.to(device).float()
            opt.zero_grad(set_to_none=True)
            with amp_autocast(use_amp):
                logit = net(x).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logit, yt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(net)
            tot_l += float(loss.detach().item())
            n_batches += 1
        sched.step()

        live = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(ema.state_dict(net))
        vp, vl = evaluate(val_dl)
        m = binary_metrics(vp, vl)
        rec = dict(m)
        rec.update({"epoch": epoch, "train_loss": tot_l / max(1, n_batches)})
        history.append(rec)
        print(f"epoch {epoch:2d}/{EPOCHS} loss={rec['train_loss']:.4f} val_auc={m['auc']:.4f} "
              f"acc={m['accuracy']:.3f} f1={m['f1']:.3f} ({time.time() - t0:.0f}s)")
        if math.isfinite(m["auc"]) and m["auc"] > best_auc:
            best_auc, best_epoch = m["auc"], epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(live)

    if best_state is None:
        raise SystemExit("No epoch produced a scoreable validation result.")
    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    vp, vl = evaluate(val_dl)
    val_metrics = binary_metrics(vp, vl)

    by_source = {}
    for s in sorted(set(val_sources.tolist())):
        m = val_sources == s
        by_source[s] = binary_metrics(vp[m], vl[m])

    print("\n=== BEST (pooled) ===")
    print(json.dumps(jsonable(val_metrics), indent=2))
    print("\nBy source:")
    for s, d in by_source.items():
        print(f"  {s:20s} n={d['n']:6d} auc={d['auc']:.4f} acc={d['accuracy']:.3f}")

    config = jsonable({
        "task": "satellite_vs_random", "temp_lo": TEMP_LO, "temp_hi": TEMP_HI,
        "img_size": IMG_SIZE, "norm_mean": NORM_MEAN, "norm_std": NORM_STD,
        "arch": "resnet18_1ch_binary", "model_version": MODEL_VERSION, "seed": SEED,
        "note": "positive = any real IR frame (DT/TCIR/maritime-continent, any storm "
                "stage, cyclone or not). negative = natural photos + synthetic degenerate "
                "patterns (noise/checkerboard/solid). Does NOT judge cyclone presence.",
    })
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "arch": "resnet18_1ch_binary", "task": "satellite_vs_random",
                "pretrained_backbone": pretrained, "epoch": best_epoch, "config": config,
                "val_metrics": jsonable(val_metrics), "val_metrics_by_source": jsonable(by_source)},
               CKPT_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(jsonable({"config": config, "best_epoch": best_epoch, "val_metrics": val_metrics,
                             "val_metrics_by_source": by_source, "history": history,
                             "class_balance": index["cls"].value_counts().to_dict(),
                             "source_counts": index["source"].value_counts().to_dict()}), f, indent=2)
    print("\nWrote", CKPT_PATH, METRICS_PATH, INDEX_PATH)
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
