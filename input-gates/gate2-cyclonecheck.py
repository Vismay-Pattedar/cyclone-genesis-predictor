"""
Cyclone Genesis Predictor -- Gate 2: "given a real satellite IR frame, is a cyclone actually
present, or is this ordinary weather (including ordinary thunderstorms and
monsoon convection)?"

Runs AFTER Gate 1 (train_kernel_gate1_satcheck) has already confirmed the
frame is real satellite data. This gate does NOT reject non-cyclone storms --
per explicit product direction, ordinary thunderstorms/convection are fine,
only an actual tracked tropical cyclone should trigger the "cyclone" branch.

POSITIVE class: any real tracked-cyclone frame, Digital Typhoon + TCIR, ANY
stage (early or mature -- this gate only asks "cyclone present", not "how
developed").

NEGATIVE class: troykueh/maritime-continent-satellite-weather (Himawari-8/9,
95-115E/5S-10N), restricted to 2015-07-01..2022-02-18 and with 15 REAL,
VERIFIED storm windows excluded (+/-36h buffer). This is NOT an assumption --
every one of these 15 windows was found by downloading the actual global
IBTrACS best-track archive (ncei.noaa.gov, v04r01, since-1980 product) and
checking which storms had a real track point inside this exact geographic
box during this exact date range. The archive itself only covers through
2022-02-18, so frames after that date are NOT used as negatives here (the
dataset's own coverage extends to 2026, but nothing past the verified window
is trustworthy as a confirmed-clean negative -- left for a future pass once a
more current best-track source is checked).

Deliberately NOT used yet: pad1tya/himawari9-b13-thermal-ir. Its non-cyclone
frames are full 5500x5500 Himawari FULL-DISK scans, not pre-cropped to the
Bay of Bengal -- using them correctly needs the geostationary projection math
to crop to the right region, and getting that wrong risks silently including
an unrelated typhoon elsewhere on the same disk (the Western Pacific is
almost always active). Scoped out rather than guessed at.

Outputs (under OUT_DIR): model_best.pt, metrics.json, dataset_index.csv.
"""
import json
import math
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 1337
MODEL_VERSION = "gate2_cyclonecheck_v1"

INPUT_ROOT = os.environ.get("CGP_INPUT_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("CGP_OUT_DIR", "/kaggle/working")

TEMP_LO, TEMP_HI = 190.0, 300.0
IMG_SIZE = 224
NORM_MEAN, NORM_STD = 0.449, 0.226
MC_BT_SCALE, MC_BT_OFFSET = 150.0, 180.0

MAX_FRAMES_PER_STORM = int(os.environ.get("CGP_FRAMES_PER_STORM", "24"))
MAX_FRAMES_PER_DAY_MC = int(os.environ.get("CGP_FRAMES_PER_DAY_MC", "2"))
PRELOAD_BUDGET_GB = float(os.environ.get("CGP_PRELOAD_BUDGET_GB", "5.0"))

VAL_FRACTION = 0.2
EPOCHS = int(os.environ.get("CGP_EPOCHS", "12"))
WARMUP_EPOCHS = 1
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 5e-2
EMA_DECAY = 0.999
USE_AMP = True

CKPT_PATH = os.path.join(OUT_DIR, "model_best.pt")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")
INDEX_PATH = os.path.join(OUT_DIR, "dataset_index.csv")

# ----------------------------------------------------------------------------
# VERIFIED against the real IBTrACS since-1980 archive (v04r01), downloaded
# 2026-09-01 -- see docstring. NOT an estimate. Box: LON in [95,115],
# LAT in [-5,10]. +/-36h buffer applied to each storm's min/max time-in-box.
# ----------------------------------------------------------------------------
MC_SAFE_START = "2015-07-01"
MC_SAFE_END = "2022-02-18"
MC_EXCLUDE_WINDOWS = [
    ("2016-12-10 18:00", "2016-12-13 21:00"),  # 2016347N10110
    ("2017-01-12 18:00", "2017-01-17 12:00"),  # 2017008N08128
    ("2017-10-30 00:00", "2017-11-03 06:00"),  # 2017305N08108
    ("2017-11-01 12:00", "2017-11-08 12:00"),  # 2017307N07102
    ("2017-12-17 21:00", "2017-12-25 06:00"),  # 2017347N11131 KAI-TAK
    ("2017-12-22 12:00", "2017-12-28 00:00"),  # 2017354N08134 TEMBIN
    ("2018-11-15 06:00", "2018-11-22 12:00"),  # 2018321N10113 TORAJI
    ("2018-11-22 06:00", "2018-11-26 12:00"),  # 2018311N07179 USAGI
    ("2018-12-29 03:00", "2019-01-07 00:00"),  # 2018365N08115 PABUK
    ("2019-11-21 00:00", "2019-11-24 00:00"),  # 2019314N14136 KALMAEGI
    ("2019-12-04 18:00", "2019-12-08 00:00"),  # 2019329N09160 KAMMURI
    ("2020-05-16 12:00", "2020-05-19 18:00"),  # 2020139S04095 MANGGA
    ("2020-12-18 21:00", "2020-12-26 18:00"),  # 2020353N06129 KROVANH
    ("2021-11-29 06:00", "2021-12-02 12:00"),  # 2021335N08096 JAWAD
    ("2021-12-13 12:00", "2021-12-18 18:00"),  # 2021349N05108
]


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
        if {"file_1", "year"} - set(df.columns):
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
    work = pd.DataFrame({"frame_ref": np.arange(len(info), dtype=np.int64),
                          "storm_id": info["ID"].astype(str).to_numpy()})
    for storm_id, g in work.groupby("storm_id", sort=True):
        idx = g["frame_ref"].to_numpy()
        keep = idx[even_picks(len(idx), MAX_FRAMES_PER_STORM)]
        for fr in keep:
            rows.append({"group_key": f"tcir_{storm_id}", "cls": 1, "source": "tcir",
                         "h5_path": "", "frame_ref": int(fr)})
    print(f"[TCIR any-stage] storms={work['storm_id'].nunique()} kept_frames={len(rows)}")
    return pd.DataFrame(rows)


def build_maritime_continent_negative_index(mc_root, max_per_day=MAX_FRAMES_PER_DAY_MC):
    """Negative class: real IR frames from a window VERIFIED cyclone-free by
    cross-referencing the actual IBTrACS best-track archive, not assumed from
    geography alone (see MC_EXCLUDE_WINDOWS docstring)."""
    csvs = [f for f in os.listdir(mc_root) if f.endswith(".csv") and f != "regions.csv"]
    if not csvs:
        return pd.DataFrame(), {}
    df = pd.read_csv(os.path.join(mc_root, csvs[0]))
    df = df[df["has_image"] == True].copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df[(df["time"] >= MC_SAFE_START) & (df["time"] <= MC_SAFE_END)]

    excl = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in MC_EXCLUDE_WINDOWS]
    excluded_mask = pd.Series(False, index=df.index)
    for a, b in excl:
        excluded_mask |= (df["time"] >= a) & (df["time"] <= b)
    audit = {"rows_in_safe_range": int(len(df)), "rows_excluded_by_storm_window": int(excluded_mask.sum())}
    df = df[~excluded_mask]

    rows = []
    for date, g in df.groupby("date", sort=True):
        idx = g.index.to_numpy()
        keep = idx[even_picks(len(idx), max_per_day)]
        for i in keep:
            rows.append({"group_key": f"mc_{date}", "cls": 0, "source": "maritime_continent",
                         "h5_path": os.path.join(mc_root, str(df.loc[i, "image_path"])), "frame_ref": -1})
    audit["rows_kept_after_daily_cap"] = len(rows)
    print(f"[MC negatives] {audit}")
    return pd.DataFrame(rows), audit


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


# NOTE: no horizontal flip -- unlike Gate 1, these ARE cyclone frames on the
# positive side, and a mirrored storm reverses its hemisphere-determined
# rotational chirality (same reasoning as every other storm model here).
AUG = {"brightness": 0.10, "contrast": 0.20, "zoom_log": 0.12, "shift": 0.10, "noise_std": 0.015}


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
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
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
    print(f"Cyclone Genesis Predictor GATE 2: CYCLONE-VS-NON-CYCLONE-CLOUDS -- {MODEL_VERSION}")
    print(f"MC negative window: {MC_SAFE_START} to {MC_SAFE_END}, "
          f"{len(MC_EXCLUDE_WINDOWS)} verified storm windows excluded")
    print("=" * 72)

    dt_root = find_dt_root()
    tcir_path = find_tcir_h5()
    mc_root = find_dataset_root("maritime-continent")
    missing = [n for n, v in [("Digital Typhoon", dt_root), ("TCIR", tcir_path),
                               ("maritime-continent", mc_root)] if v is None]
    if missing:
        raise SystemExit(f"Missing dataset mounts: {missing} -- check dataset_sources.")

    info = pd.read_hdf(tcir_path, key="info", mode="r").reset_index(drop=True)
    pos_parts = [build_tcir_index_any(info)]
    meta_dir, image_dir = resolve_dt_dirs(dt_root)
    pos_parts.append(build_dt_index_any(meta_dir, image_dir))
    neg_index, mc_audit = build_maritime_continent_negative_index(mc_root)

    index = pd.concat(pos_parts + [neg_index], ignore_index=True).reset_index(drop=True)
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
        "task": "cyclone_vs_non_cyclone_clouds", "temp_lo": TEMP_LO, "temp_hi": TEMP_HI,
        "img_size": IMG_SIZE, "norm_mean": NORM_MEAN, "norm_std": NORM_STD,
        "arch": "resnet18_1ch_binary", "model_version": MODEL_VERSION, "seed": SEED,
        "note": "positive = any real tracked-cyclone frame (DT/TCIR, any stage). "
                "negative = maritime-continent Himawari frames, IBTrACS-verified "
                "cyclone-free window only (2015-07 to 2022-02), 15 real storm windows "
                "excluded with 36h buffer. Ordinary thunderstorms/monsoon convection "
                "are INTENTIONALLY left in the negative pool -- only tracked cyclones "
                "should score positive here, per product direction.",
        "mc_safe_window": [MC_SAFE_START, MC_SAFE_END],
        "mc_exclude_windows": MC_EXCLUDE_WINDOWS,
        "mc_audit": mc_audit,
        "not_yet_used": "pad1tya/himawari9-b13-thermal-ir (full-disk, needs geos crop -- scoped out)",
    })
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "arch": "resnet18_1ch_binary", "task": "cyclone_vs_non_cyclone_clouds",
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
