"""
Cyclone Genesis Predictor -- Cyclone INTENSITY model, REAL TRAINING RUN.

A SEPARATE model from the shipped early-intensification predictor
(train_kernel_v5_final/train.py, v3 deployed). That model answers "will this
still-weak system later reach 64 kt?" from ONE pre-formation frame. This one
answers the complementary question:

    Given ONE infrared frame of an ALREADY-CONFIRMED, already-mature tropical
    cyclone, what is its CURRENT sustained wind, in knots?

Continuous regression (single linear output, no sigmoid), plus an IMD-style
category bucket derived from the predicted knots via CATEGORY_BINS.

FRAME SELECTION IS NOT REDESIGNED HERE. The index-building logic below is
lifted from train_kernel_intensity/audit.py, which has already RUN FOR REAL on
Kaggle against the real mounts and returned:

    164,821 mature frames / 2,038 storms
      WPAC 119,979 (1,117 storms)   SH   15,564 (346)
      ATLN  12,811 (232)            EPAC 12,799 (243)
      IO     2,575 (82)             CPAC  1,093 (18)
    wind_kt: mean 57.3  std 22.3  min 10  max 185

Mature == at or after the storm's FIRST crossing out of the early stage
(Digital Typhoon: grade > DT_EARLY_GRADE_MAX; TCIR: Vmax > TCIR_EARLY_VMAX_KT).
That is the exact complement of the early-stage model's training set, computed
from the same crossing index, so the two models never share a frame and this
one never sees a pre-formation frame. Do not loosen either rule.

The image decoding, the 190-300 K display normalisation, the resnet18 backbone
with a 1-channel conv1, the (x - 0.449) / 0.226 normalisation and the
storm-level (never frame-level) split all come from
train_kernel_v5_final/train.py verbatim, because those are proven working code
against these two specific datasets, not something to re-derive.

WIND-CONVENTION CAVEAT -- READ THIS BEFORE QUOTING ANY NUMBER FROM THIS MODEL
    Digital Typhoon's `wind` is JMA 10-MINUTE sustained wind.
    TCIR's `Vmax` is JTWC/NHC 1-MINUTE sustained wind.
These are different physical quantities (roughly v_10min ~= 0.88 * v_1min), and
this script deliberately does NOT silently harmonise them: no conversion is
applied to any target, and the two sources ARE pooled for training, which is a
defensible choice (the model learns a blended convention) but an invisible one
unless it is measured. So every row carries a `wind_convention` tag
('JMA10min' / 'JTWC1min'), and every validation metric is reported BOTH pooled
AND broken out by convention/source, so the size of the mixing is an auditable
number in metrics.json rather than a hidden assumption. If a future version
wants to harmonise, the per-convention breakdown here is the before-picture.

Outputs (all under OUT_DIR):
    model_best.pt      state_dict + config (temp_lo/hi, img_size, norm_mean/std,
                       task='intensity_regression', target normalisation
                       constants, wind-convention notes) + val metrics
    metrics.json       full report: overall / per-basin / per-source /
                       per-convention MAE + RMSE, category accuracy, history
    dataset_index.csv  the exact mature index trained on, for auditability

Runs as a plain Kaggle *script* kernel. Needs the GPU (unlike the audit).
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

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SEED = 1337
MODEL_VERSION = "intensity_v1"

# Overridable so the same file can be pointed at a synthetic tree offline. On
# Kaggle neither variable is set and these are the standard paths.
INPUT_ROOT = os.environ.get("CGP_INPUT_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("CGP_OUT_DIR", "/kaggle/working")

# ---- MATURE-frame definition (verbatim from audit.py, itself verbatim from v5)
# "Mature" is the complement of "early": frames at or after the first crossing.
DT_EARLY_GRADE_MAX = 2          # DT mature == grade > this
TCIR_EARLY_VMAX_KT = 33.0       # TCIR mature == Vmax > this

# ---- preprocessing (LOAD-BEARING -- identical to every checkpoint in this
# project, so a frame prepared for the early-stage model is prepared correctly
# for this one too) ----------------------------------------------------------
TEMP_LO, TEMP_HI = 190.0, 300.0
IMG_SIZE = 224
NORM_MEAN, NORM_STD = 0.449, 0.226
TYPHOON_GRADE = 5               # carried in config for downstream compatibility
TCIR_MIN_FINITE_FRAC = 0.50

# ---- dedup (verbatim from audit.py / v5 -- correctness-critical, do not tune)
DEDUP_MATCH_DEG = 1.5
DEDUP_MIN_MATCH_POINTS = 4
DEDUP_USE_ID_KEY_UNION = True
TCIR_ID_RE = re.compile(r"^(\d{4})(\d{2})([A-Za-z])$")

# ---- wind conventions (REPORTING, never applied to a target) ----------------
WIND_CONVENTION_OF_SOURCE = {
    "digital_typhoon": "JMA10min",   # JMA best track, 10-minute sustained
    "tcir": "JTWC1min",              # JTWC/NHC best track, 1-minute sustained
}
# Standard conversion, used ONLY to print how far apart the two scales sit.
# Nothing in the training path multiplies by this.
WIND_10MIN_PER_1MIN = 0.88

# IMD wind-speed category bins (kt). Same table as the audit; applied here as a
# generic display banding, not a basin-specific claim.
CATEGORY_BINS = [
    (0, 34, "Depression"),
    (34, 48, "Cyclonic Storm"),
    (48, 64, "Severe Cyclonic Storm"),
    (64, 90, "Very Severe Cyclonic Storm"),
    (90, 120, "Extremely Severe Cyclonic Storm"),
    (120, 999, "Super Cyclonic Storm"),
]
CATEGORY_NAMES = [name for _, _, name in CATEGORY_BINS]
CATEGORY_INDEX = {name: i for i, name in enumerate(CATEGORY_NAMES)}

# ---- dataset size control ---------------------------------------------------
# The full mature index is 164,821 frames. Preloading all of them at 224x224
# uint8 is ~8.3 GB of RAM plus ~165k HDF5 reads off two network mounts, which is
# neither necessary nor safe inside a Kaggle session. Frames of one storm six
# hours apart are highly correlated, so the marginal value of the 100th frame of
# one typhoon is small -- but unlike the early-stage model this one must NOT
# collapse a storm to a handful of frames, because the whole point is to see the
# same storm at many different intensities. MAX_FRAMES_PER_STORM frames are
# taken EVENLY across each storm's mature track (even_picks, the v3/v5 rule), so
# the sampled frames still span the storm's full intensification-peak-decay
# range rather than clustering at one end.
MAX_FRAMES_PER_STORM = int(os.environ.get("CGP_FRAMES_PER_STORM", "32"))
# Hard RAM ceiling for the preloaded uint8 image array. If the capped index
# still exceeds this, the cap is automatically lowered (and the reduction is
# printed and recorded) rather than letting the kernel OOM mid-run.
PRELOAD_BUDGET_GB = float(os.environ.get("CGP_PRELOAD_BUDGET_GB", "5.0"))

# ---- training ---------------------------------------------------------------
VAL_FRACTION = 0.2
EPOCHS = int(os.environ.get("CGP_EPOCHS", "14"))
WARMUP_EPOCHS = 2
BATCH_SIZE = 32
LR = 3e-4
# v5 measured that regularisation -- not more data -- was the lever that
# actually helped on this imagery (C_reg: +0.0143 storm AUC and a train/val gap
# cut from 0.234 to 0.141). Carrying its weight decay over.
WEIGHT_DECAY = 5e-2
EMA_DECAY = 0.999
USE_AMP = True

# Huber/Smooth-L1 beta, in NORMALISED target units. The target is standardised
# by the train-set mean/std (std ~= 22 kt), so beta = 0.25 is about 5.5 kt: the
# loss is quadratic for the errors a good model actually makes and linear beyond
# them, which stops a handful of extreme-intensity frames (the 150-185 kt tail)
# from dominating the gradient the way plain MSE would.
HUBER_BETA = 0.25

# Predictions are clipped to the physically observed range before any metric is
# computed or any number is displayed. The training data spans 10-185 kt; a
# regression head is free to extrapolate outside that, and such an output is
# never meaningful.
PRED_CLIP_KT = (10.0, 200.0)

# Augmentation, applied in the [0,1] display domain exactly as in v5.
# zoom_log is DELIBERATELY milder than v5's C_reg preset (0.25): for the
# early-stage binary task apparent scale was a nuisance variable worth
# destroying, but for intensity the apparent size/compactness of the cold cloud
# mass is a genuine Dvorak-style intensity cue, so heavy zoom jitter would be
# augmenting away part of the signal. Everything else is kept at C_reg strength.
AUG = {"brightness": 0.10, "contrast": 0.20, "zoom_log": 0.10,
       "shift": 0.12, "noise_std": 0.020, "cutout_frac": 0.15}

# Per-frame loss weighting exponent, w ~ n_basin ** -alpha. 0.0 == uniform,
# identical to v3/v5's default. Kept as a one-constant A/B, off by default.
BASIN_BALANCE_ALPHA = 0.0
# (v2) inverse-frequency loss weighting; 0.0 == intensity_v1 behaviour exactly.
INTENSITY_BALANCE_ALPHA = 0.0
# (v3) which output parameterisation this arm uses. "reg" reproduces v1/v2-a0.
HEAD_MODE = "mt"
DIST_BIN_KT = 5.0          # best-track winds are recorded in 5 kt steps
DIST_SIGMA_KT = 5.0    # Gaussian soft-target width for the dist head
MT_CAT_WEIGHT = 0.3        # weight on the auxiliary IMD-category head

# Size of the fixed, un-augmented training subsample scored each epoch as an
# under/overfit probe (train MAE vs val MAE gap).
TRAIN_PROBE_FRAMES = 2000

CKPT_PATH = os.path.join(OUT_DIR, "model_best.pt")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")
INDEX_PATH = os.path.join(OUT_DIR, "dataset_index.csv")


def wind_convention_note():
    return (
        "Target is the CURRENT sustained wind at the frame's own timestamp. "
        "Digital Typhoon's `wind` is JMA 10-minute sustained; TCIR's `Vmax` is "
        "JTWC/NHC 1-minute sustained. These are DIFFERENT physical quantities "
        "(v_10min ~= 0.88 * v_1min). This model is trained on the two pooled "
        "WITHOUT conversion, so its output is a blended-convention wind speed, "
        "closest to JMA 10-minute in the Western Pacific (where Digital Typhoon "
        "dominates the data) and closest to JTWC/NHC 1-minute in the other "
        "basins. Every validation metric is reported both pooled and split by "
        "wind_convention/source so the size of that mixing is measurable. Do "
        "not quote a single MAE without also quoting the per-convention split."
    )


def mature_rule_note():
    return (
        "Frames at or after the storm's FIRST crossing out of the early stage "
        "(Digital Typhoon: grade > 2; TCIR: Vmax > 33 kt). Post-formation only; "
        "no pre-formation frame is ever included. This is the exact complement "
        "of the early-intensification model's training set, computed from the "
        "same crossing index."
    )


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def jsonable(obj):
    """Recursively cast numpy scalars/arrays to plain Python types, so anything
    written into ckpt['config'] / ckpt['val_metrics'] survives json.dumps()."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None      # NaN/Inf are not valid JSON
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    return obj


def even_picks(n_eligible, k):
    """k frames spread EVENLY across an eligible window, not the first k.
    Verbatim from v5; used here to subsample each storm's mature track without
    biasing the sample toward one end of its life cycle."""
    if n_eligible <= 0:
        return np.zeros(0, dtype=int)
    return np.unique(
        np.linspace(0, n_eligible - 1, min(k, n_eligible)).round().astype(int)
    )


def day_of_year(y, m, d):
    """Day-of-year, tolerant of the odd out-of-range day value in best-track
    data (verbatim from audit.py)."""
    try:
        return int(datetime.date(int(y), int(m), min(28, int(d))).timetuple().tm_yday) \
            + (int(d) - min(28, int(d)))
    except Exception:
        return -1


def category_of(kt):
    for lo, hi, name in CATEGORY_BINS:
        if lo <= kt < hi:
            return name
    return CATEGORY_BINS[-1][2]


def category_index_of(kt):
    for i, (lo, hi, _name) in enumerate(CATEGORY_BINS):
        if lo <= kt < hi:
            return i
    return len(CATEGORY_BINS) - 1


# ----------------------------------------------------------------------------
# Dataset discovery -- verbatim from audit.py (which took it from v5_final,
# where it is already proven: picks the Western Pacific Digital Typhoon mount
# explicitly rather than the first "typhoon"-named dir, and PRUNES the ~86GB
# image tree during the TCIR search instead of walking into it).
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
    if len(cands) > 1:
        print(f"NOTE: {len(cands)} Digital Typhoon mounts found:")
        for c in cands:
            print("   ", c)
    wp = [c for c in cands
          if "western-pacific" in c.lower() or c.lower().rstrip("/\\").endswith("wp")]
    pick = wp[0] if wp else sorted(cands)[0]
    if len(cands) > 1:
        print(f"  -> using {pick}"
              + ("" if wp else "  (NO mount matched 'western-pacific'/'wp'; "
                               "this is a guess, check it)"))
    return pick


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
            if fn.lower().endswith((".h5", ".hdf5")) and (
                "tcsa" in fn.lower() or "tcir" in fn.lower()
            ):
                candidates.append(os.path.join(dirpath, fn))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    if len(candidates) > 1:
        print(f"NOTE: {len(candidates)} TCIR-looking .h5 files found, using the largest:")
        for c in candidates:
            print(f"    {c}  ({os.path.getsize(c) / 1e9:.1f} GB)")
    return candidates[0]


# ----------------------------------------------------------------------------
# MATURE-frame indexing -- audit.py's build_dt_index_mature /
# build_tcir_index_mature, unchanged in what they SELECT. The only additions
# are (a) the wind_convention tag and (b) the per-storm even-pick cap, which is
# applied AFTER the full mature list for a storm is built, so it subsamples the
# selection without ever altering it.
# ----------------------------------------------------------------------------
def build_dt_index_mature(meta_dir, image_dir):
    csvs = sorted(f for f in os.listdir(meta_dir) if f.endswith(".csv") and f[:-4].isdigit())
    print(f"[A] Found {len(csvs)} per-storm metadata CSVs")

    rows = []
    tracks = defaultdict(list)
    audit = Counter()

    for i, name in enumerate(csvs):
        storm_id = name[:-4]
        try:
            df = pd.read_csv(os.path.join(meta_dir, name))
        except Exception:
            audit["n_unreadable"] += 1
            continue

        need = {"grade", "file_1", "wind", "year", "month", "day", "hour", "lat", "lng"}
        if need - set(df.columns):
            audit["n_bad_columns"] += 1
            continue

        year = int(storm_id[:4])
        if year < 1977:
            audit["n_pre_1977_dropped"] += 1
            continue

        grade = pd.to_numeric(df["grade"], errors="coerce")
        wind = pd.to_numeric(df["wind"], errors="coerce").fillna(0.0)

        # track, for dedup -- identical to audit.py / v5_final
        t_y = pd.to_numeric(df["year"], errors="coerce")
        t_m = pd.to_numeric(df["month"], errors="coerce")
        t_d = pd.to_numeric(df["day"], errors="coerce")
        t_h = pd.to_numeric(df["hour"], errors="coerce")
        lat = pd.to_numeric(df["lat"], errors="coerce")
        lng = pd.to_numeric(df["lng"], errors="coerce")
        good = t_y.notna() & t_m.notna() & t_d.notna() & t_h.notna() & lat.notna() & lng.notna()
        if good.any():
            stamps = (t_y[good].astype(int) * 1000000 + t_m[good].astype(int) * 10000
                      + t_d[good].astype(int) * 100 + t_h[good].astype(int))
            for ts, la, lo in zip(stamps.to_numpy(), lat[good].to_numpy(), lng[good].to_numpy()):
                tracks[int(ts)].append((float(la), float(lo), storm_id))

        # MATURE selection: at or after the first crossing OUT of early stage.
        crossed = (grade > DT_EARLY_GRADE_MAX).fillna(False).to_numpy()
        first_cross = int(np.argmax(crossed)) if crossed.any() else len(df)
        if first_cross >= len(df):
            audit["n_never_matures"] += 1
            continue

        fname = df["file_1"].fillna("").astype(str).str.strip()
        mature_pos = np.arange(first_cross, len(df))
        mature_pos = mature_pos[(fname.to_numpy()[mature_pos] != "") &
                                 (wind.to_numpy()[mature_pos] > 0)]
        audit["n_frames_mature_eligible"] += len(mature_pos)
        if len(mature_pos) == 0:
            audit["n_no_mature_frame"] += 1
            continue

        storm_rows = []
        for p in mature_pos:
            rec = df.iloc[int(p)]
            h5_path = os.path.join(image_dir, storm_id, str(rec["file_1"]).strip())
            if not os.path.exists(h5_path):
                audit["n_missing_h5"] += 1
                continue
            storm_rows.append({
                "storm_key": storm_id, "source": "digital_typhoon", "basin": "WPAC",
                "wind_kt": float(wind.iloc[int(p)]),
                "wind_convention": WIND_CONVENTION_OF_SOURCE["digital_typhoon"],
                "h5_path": h5_path, "frame_ref": -1,
                "lat": float(pd.to_numeric(rec["lat"], errors="coerce")),
                "lon": float(pd.to_numeric(rec["lng"], errors="coerce")),
                "doy": day_of_year(rec["year"], rec["month"], rec["day"]),
                "track_pos": int(p),
            })

        # Even-pick cap. Subsamples the mature list built above; never widens it.
        if 0 < MAX_FRAMES_PER_STORM < len(storm_rows):
            keep = even_picks(len(storm_rows), MAX_FRAMES_PER_STORM)
            audit["n_frames_dropped_by_cap"] += len(storm_rows) - len(keep)
            storm_rows = [storm_rows[int(k)] for k in keep]
        rows.extend(storm_rows)

        if (i + 1) % 250 == 0:
            print(f"  [A] indexed {i + 1}/{len(csvs)} storms, {len(rows)} kept frames so far")

    print(f"[A] storms dropped -- unreadable={audit['n_unreadable']} "
          f"bad_columns={audit['n_bad_columns']} pre_1977={audit['n_pre_1977_dropped']} "
          f"never_matures={audit['n_never_matures']} no_mature_frame={audit['n_no_mature_frame']}")
    print(f"[A] frames listed but missing on disk: {audit['n_missing_h5']}")
    print(f"[A] mature frames eligible={audit['n_frames_mature_eligible']} "
          f"dropped_by_per_storm_cap={audit['n_frames_dropped_by_cap']} kept={len(rows)}")
    return pd.DataFrame(rows), tracks, dict(audit)


def build_tcir_index_mature(info):
    rows = []
    audit = Counter()

    vmax = pd.to_numeric(info["Vmax"], errors="coerce")
    time_i = pd.to_numeric(info["time"], errors="coerce")
    lat = pd.to_numeric(info["lat"], errors="coerce")
    lon = pd.to_numeric(info["lon"], errors="coerce")

    work = pd.DataFrame({
        "frame_ref": np.arange(len(info), dtype=np.int64),
        "storm_id": info["ID"].astype(str).to_numpy(),
        "region": info["region"].astype(str).to_numpy(),
        "vmax": vmax.to_numpy(), "time": time_i.to_numpy(),
        "lat": lat.to_numpy(), "lon": lon.to_numpy(),
    })

    for storm_id, g in work.groupby("storm_id", sort=True):
        basin = str(pd.unique(g["region"])[0])
        g = g.sort_values("time", kind="stable")
        vm = g["vmax"].to_numpy()
        crossed = np.isfinite(vm) & (vm > TCIR_EARLY_VMAX_KT)
        first_cross = int(np.argmax(crossed)) if crossed.any() else len(g)
        if first_cross >= len(g):
            audit["n_never_matures"] += 1
            continue

        mature = g.iloc[first_cross:]
        mature = mature[np.isfinite(mature["vmax"]) & (mature["vmax"] > 0)]
        audit["n_frames_mature_eligible"] += len(mature)
        if len(mature) == 0:
            audit["n_no_mature_frame"] += 1
            continue

        storm_rows = []
        for pos, (_, rec) in enumerate(mature.iterrows()):
            ts = rec["time"]
            ts = int(ts) if np.isfinite(ts) else -1
            storm_rows.append({
                "storm_key": storm_id, "source": "tcir", "basin": basin,
                "wind_kt": float(rec["vmax"]),
                "wind_convention": WIND_CONVENTION_OF_SOURCE["tcir"],
                "h5_path": "", "frame_ref": int(rec["frame_ref"]),
                "lat": float(rec["lat"]), "lon": float(rec["lon"]),
                "doy": (day_of_year(ts // 1000000, (ts // 10000) % 100, (ts // 100) % 100)
                        if ts > 0 else -1),
                "track_pos": int(pos),
            })

        if 0 < MAX_FRAMES_PER_STORM < len(storm_rows):
            keep = even_picks(len(storm_rows), MAX_FRAMES_PER_STORM)
            audit["n_frames_dropped_by_cap"] += len(storm_rows) - len(keep)
            storm_rows = [storm_rows[int(k)] for k in keep]
        rows.extend(storm_rows)

    print(f"[B] storms dropped -- never_matures={audit['n_never_matures']} "
          f"no_mature_frame={audit['n_no_mature_frame']}")
    print(f"[B] mature frames eligible={audit['n_frames_mature_eligible']} "
          f"dropped_by_per_storm_cap={audit['n_frames_dropped_by_cap']} kept={len(rows)}")
    return pd.DataFrame(rows), work, dict(audit)


# ----------------------------------------------------------------------------
# Cross-dataset dedup -- Digital Typhoon (JMA) and TCIR (JTWC) both cover the
# Western Pacific and MUST NOT be blindly concatenated: the same storm appears
# in both catalogues under different numbering. Two detectors are UNIONed --
# cheap ID-key match, and authoritative best-track position match. Verbatim
# logic from audit.py / v5_final; the only addition is the returned audit dict.
# ----------------------------------------------------------------------------
def tcir_dedup_key(storm_id):
    m = TCIR_ID_RE.match(str(storm_id))
    return (int(m.group(1)), int(m.group(2))) if m else None


def dt_dedup_key(storm_id):
    s = str(storm_id)
    return (int(s[:4]), int(s[4:6])) if len(s) == 6 and s.isdigit() else None


def dedup_wpac(dt_index, tcir_index, tcir_work, dt_tracks):
    dt_storms = sorted(dt_index["storm_key"].unique())
    tcir_wpac = tcir_index[tcir_index["basin"] == "WPAC"]
    tcir_wpac_storms = sorted(tcir_wpac["storm_key"].unique())

    dt_keys = {}
    for s in dt_storms:
        k = dt_dedup_key(s)
        if k is not None:
            dt_keys.setdefault(k, []).append(s)

    by_key = {s for s in tcir_wpac_storms
              if tcir_dedup_key(s) is not None and tcir_dedup_key(s) in dt_keys}

    dt_storm_set = set(dt_storms)
    tcir_wpac_set = set(tcir_wpac_storms)
    by_track = {}
    tw = tcir_work[tcir_work["region"] == "WPAC"]
    for storm_id, g in tw.groupby("storm_id", sort=True):
        if storm_id not in tcir_wpac_set:
            continue
        votes = Counter()
        for ts, la, lo in zip(g["time"].to_numpy(), g["lat"].to_numpy(), g["lon"].to_numpy()):
            if not (np.isfinite(ts) and np.isfinite(la) and np.isfinite(lo)):
                continue
            for dla, dlo, dsid in dt_tracks.get(int(ts), ()):
                if dsid not in dt_storm_set:
                    continue
                if abs(dla - la) <= DEDUP_MATCH_DEG and abs(dlo - lo) <= DEDUP_MATCH_DEG:
                    votes[dsid] += 1
        if votes and votes.most_common(1)[0][1] >= DEDUP_MIN_MATCH_POINTS:
            by_track[storm_id] = True

    drop = (by_key | set(by_track)) if DEDUP_USE_ID_KEY_UNION else set(by_track)
    print(f"[dedup] dropping {len(drop)} of {len(tcir_wpac_storms)} TCIR WPAC storms "
          f"as Digital Typhoon duplicates")
    audit = {
        "dt_wpac_storms": len(dt_storms),
        "tcir_wpac_storms": len(tcir_wpac_storms),
        "overlap_by_id_key": len(by_key),
        "overlap_by_track_match": len(by_track),
        "overlap_by_both": len(by_key & set(by_track)),
        "n_tcir_wpac_storms_dropped": len(drop),
        "n_tcir_wpac_storms_kept": len(tcir_wpac_storms) - len(drop),
        "used_id_key_union": DEDUP_USE_ID_KEY_UNION,
        "match_deg": DEDUP_MATCH_DEG,
        "min_match_points": DEDUP_MIN_MATCH_POINTS,
    }
    return drop, audit


# ----------------------------------------------------------------------------
# Image loading and the model input pipeline -- verbatim from v5_final. These
# are the REAL, working decode paths for each dataset's file format; nothing
# here is guessed.
# ----------------------------------------------------------------------------
def kelvin_to_display_u8(arr):
    """Brightness temperature (K) -> uint8 display frame, cold = bright.

    Shared by BOTH sources so a Digital Typhoon pixel and a TCIR pixel of the
    same temperature become the same grey level."""
    arr = np.clip(arr, TEMP_LO, TEMP_HI)
    arr = 1.0 - (arr - TEMP_LO) / (TEMP_HI - TEMP_LO)
    return (arr * 255.0).astype(np.uint8)


def load_dt_frame_u8(h5_path):
    """Digital Typhoon: single dataset 'Infrared', (512,512) float64 Kelvin."""
    import h5py  # imported here so a worker process gets its own handle

    with h5py.File(h5_path, "r") as f:
        arr = np.asarray(f["Infrared"], dtype=np.float32)
    return kelvin_to_display_u8(arr)


def load_tcir_frame_u8(images_ds, row):
    """TCIR: images[row] is (128,128,4); channel 0 is IR1 in Kelvin (verified by
    the tcsa_probe run: ch0 spans ~195-301 K, ch1 is water vapour, ch2/ch3 are
    reflectance-scaled). Channels 1-3 are deliberately unused -- this model
    stays 1-channel."""
    frame = np.asarray(images_ds[row], dtype=np.float32)
    arr = frame[..., 0]
    finite = np.isfinite(arr)
    frac = float(finite.mean())
    if frac < TCIR_MIN_FINITE_FRAC:
        raise ValueError(f"IR1 channel only {frac:.1%} finite")
    if frac < 1.0:
        # Missing scan lines -> treat as warm (dark), the project-wide convention.
        arr = np.nan_to_num(arr, nan=TEMP_HI, posinf=TEMP_HI, neginf=TEMP_LO)
    return kelvin_to_display_u8(arr)


def resize_u8(u8):
    """Any HxW uint8 -> IMG_SIZE x IMG_SIZE uint8, bilinear. Downsamples Digital
    Typhoon's 512x512 and upsamples TCIR's 128x128 through the identical path."""
    if u8.shape == (IMG_SIZE, IMG_SIZE):
        return u8
    t = torch.from_numpy(np.ascontiguousarray(u8)).float()[None, None]
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return t.squeeze().clamp(0, 255).round().byte().numpy()


def to_unit(u8_batch):
    """uint8 (N,H,W) tensor -> float (N,1,IMG_SIZE,IMG_SIZE) in [0,1]."""
    t = u8_batch.unsqueeze(1).float().div_(255.0)
    if t.shape[-1] != IMG_SIZE or t.shape[-2] != IMG_SIZE:
        t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return t


def normalize(x01):
    """[0,1] display tensor -> the exact normalisation every tool in this project
    applies at inference."""
    return (x01 - NORM_MEAN) / NORM_STD


def to_model_input(u8_batch):
    """Full inference-path preprocessing. Used for validation and for the train
    probe -- never augmented."""
    return normalize(to_unit(u8_batch))


def augment_batch(x01, gen, aug):
    """Batched augmentation in the [0,1] display domain (verbatim from v5, with
    this model's own strength dict).

    `gen` is a CPU torch.Generator: all randomness is drawn on the CPU and moved
    to x01's device, which keeps the augmentation bit-reproducible from a seed
    regardless of device.

    NOT included: flips. A mirror flip reverses the storm's rotational
    chirality, which is fixed by hemisphere -- a flipped Southern-Hemisphere
    storm is a physically impossible Northern-Hemisphere one.
    """
    p = aug
    n = x01.shape[0]
    dev = x01.device

    def rnd(lo, hi):
        return (torch.rand(n, generator=gen) * (hi - lo) + lo).to(dev)

    # ---- geometric: isotropic zoom + small translation ----
    # affine_grid's theta maps OUTPUT coords back into the INPUT, so the scale
    # entry is 1/zoom.
    zoom = torch.exp(rnd(-p["zoom_log"], p["zoom_log"]))
    tx = rnd(-p["shift"], p["shift"])
    ty = rnd(-p["shift"], p["shift"])
    theta = torch.zeros(n, 2, 3, device=dev, dtype=x01.dtype)
    theta[:, 0, 0] = 1.0 / zoom
    theta[:, 1, 1] = 1.0 / zoom
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(x01.shape), align_corners=False)
    # padding_mode='border' replicates edge pixels; 'zeros' would paint a hard
    # black rim on every zoomed-out frame, a synthetic feature to memorise.
    x01 = F.grid_sample(x01, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)

    # ---- photometric: gain about mid-grey, then offset ----
    gain = rnd(1.0 - p["contrast"], 1.0 + p["contrast"]).view(n, 1, 1, 1)
    off = rnd(-p["brightness"], p["brightness"]).view(n, 1, 1, 1)
    x01 = (x01 - 0.5) * gain + 0.5 + off

    # ---- cutout: one mid-grey square per frame ----
    # Filled with 0.5 rather than 0.0 because 0.0 in the display domain is a hot
    # 300 K surface, i.e. a physically meaningful value the model would read.
    cf = p.get("cutout_frac", 0.0)
    if cf > 0:
        side = max(1, int(round(x01.shape[-1] * cf)))
        cy = torch.randint(0, x01.shape[-2], (n,), generator=gen)
        cx = torch.randint(0, x01.shape[-1], (n,), generator=gen)
        yy = torch.arange(x01.shape[-2]).view(1, -1, 1)
        xx = torch.arange(x01.shape[-1]).view(1, 1, -1)
        my = (yy - cy.view(-1, 1, 1)).abs() < side // 2
        mx = (xx - cx.view(-1, 1, 1)).abs() < side // 2
        box = (my & mx).unsqueeze(1).to(dev)
        x01 = torch.where(box, torch.full_like(x01, 0.5), x01)

    # ---- sensor noise (~2 K of Gaussian at 0.020) ----
    if p["noise_std"] > 0:
        noise = torch.randn(x01.shape, generator=gen).to(dev) * p["noise_std"]
        x01 = x01 + noise

    # Clamp back into the display range: the model must never see a value a real
    # uint8 frame could not produce.
    return x01.clamp_(0.0, 1.0)


class FrameDataset(Dataset):
    """Frames are pre-decoded into RAM once (see preload_all), so this is an
    indexed view plus the cheap CPU-side part of the augmentation.

    Returns (image_u8, target_norm, wind_kt, weight). `target_norm` is the
    standardised regression target the loss is computed on; `wind_kt` is the raw
    truth in knots, carried through so every metric is reported in knots without
    the evaluation code having to know the normalisation.
    """

    def __init__(self, images_u8, target_norm, wind_kt, weights, augment):
        self.images = images_u8
        self.target = np.asarray(target_norm, dtype=np.float32)
        self.wind = np.asarray(wind_kt, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augment:
            # Rotations only -- NOT flips (see augment_batch). A 90-degree
            # rotation of a storm is still the same storm at the same intensity.
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img = np.rot90(img, k).copy()
        # float32 TENSORS, not Python floats. default_collate turns a list of
        # Python floats into a float64 batch; a float64 target makes the loss
        # float64, and autograd's CUDA backward rejects that ("Found dtype
        # Double but expected Float") while CPU autograd silently promotes --
        # i.e. it fails ONLY on the GPU. This exact bug killed a real Kaggle run
        # in the v5 sweep; it is fixed at the source here.
        return (torch.from_numpy(img),
                torch.tensor(self.target[i], dtype=torch.float32),
                torch.tensor(self.wind[i], dtype=torch.float32),
                torch.tensor(self.weights[i], dtype=torch.float32))


def preload_all(index, tcir_h5_path):
    """Decode every frame ONCE, resized, into a single uint8 array in RAM.

    TCIR is done FIRST and deliberately: it is the unknown-latency source
    (random row access into one 6.6GB file), and its rows are read in ASCENDING
    frame order so the access pattern is as close to sequential as it can be. An
    ETA is printed after 100 frames of each pass, so a pathologically slow mount
    is caught in the first minute rather than after half an hour.
    """
    import h5py

    n = len(index)
    print(f"\nAllocating preload buffer: {n} x {IMG_SIZE} x {IMG_SIZE} uint8 "
          f"= {n * IMG_SIZE * IMG_SIZE / 1e9:.2f} GB")
    out = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    ok = np.ones(n, dtype=bool)
    sources = index["source"].to_numpy()

    # ---- pass 1: TCIR ----
    tc_pos = np.where(sources == "tcir")[0]
    if len(tc_pos):
        refs = index["frame_ref"].to_numpy()[tc_pos]
        order = np.argsort(refs, kind="stable")   # ascending row == near-sequential IO
        tc_pos = tc_pos[order]
        print(f"\nPreloading {len(tc_pos)} TCIR frames from {tcir_h5_path}")
        t0 = time.time()
        with h5py.File(tcir_h5_path, "r") as f:
            ds = f["images"]
            for j, pos in enumerate(tc_pos):
                row = int(index["frame_ref"].iat[int(pos)])
                try:
                    out[pos] = resize_u8(load_tcir_frame_u8(ds, row))
                except Exception as exc:
                    # Dropped and counted, never fed to the model as zeros.
                    print(f"  LOAD FAIL tcir row {row}: {type(exc).__name__}: {exc}")
                    ok[pos] = False
                if j == 99:
                    rate = (time.time() - t0) / 100
                    print(f"  [B] preload rate {rate * 1000:.0f} ms/frame "
                          f"-> ETA {rate * len(tc_pos) / 60:.1f} min for TCIR")
                if (j + 1) % 2000 == 0:
                    print(f"  [B] preloaded {j + 1}/{len(tc_pos)} ({time.time() - t0:.0f}s)")
        print(f"[B] TCIR preload done in {time.time() - t0:.0f}s")

    # ---- pass 2: Digital Typhoon ----
    dt_pos = np.where(sources == "digital_typhoon")[0]
    if len(dt_pos):
        paths = index["h5_path"].to_numpy()
        print(f"\nPreloading {len(dt_pos)} Digital Typhoon frames")
        t0 = time.time()
        for j, pos in enumerate(dt_pos):
            try:
                out[pos] = resize_u8(load_dt_frame_u8(paths[pos]))
            except Exception as exc:
                print(f"  LOAD FAIL {paths[pos]}: {type(exc).__name__}: {exc}")
                ok[pos] = False
            if j == 99:
                rate = (time.time() - t0) / 100
                print(f"  [A] preload rate {rate * 1000:.0f} ms/frame "
                      f"-> ETA {rate * len(dt_pos) / 60:.1f} min for Digital Typhoon")
            if (j + 1) % 2000 == 0:
                print(f"  [A] preloaded {j + 1}/{len(dt_pos)} ({time.time() - t0:.0f}s)")
        print(f"[A] Digital Typhoon preload done in {time.time() - t0:.0f}s")

    print(f"Preload total failures: {int((~ok).sum())}/{n}")
    return out, ok


def fit_index_to_ram_budget(index, budget_gb=PRELOAD_BUDGET_GB):
    """Lower the per-storm frame cap until the preload buffer fits the budget.

    Subsampling is per storm and EVEN across its mature track (same rule as the
    initial cap), so shrinking the dataset never biases it toward one part of a
    storm's life. Returns (index, audit)."""
    bytes_per = IMG_SIZE * IMG_SIZE
    need_gb = len(index) * bytes_per / 1e9
    audit = {"frames_before_budget": int(len(index)),
             "estimated_gb_before": float(need_gb),
             "budget_gb": float(budget_gb),
             "cap_applied": None,
             "frames_after_budget": int(len(index))}
    if need_gb <= budget_gb:
        return index, audit

    target_n = int(budget_gb * 1e9 / bytes_per)
    counts = index.groupby("storm_key", sort=False).size().to_numpy()
    cap = MAX_FRAMES_PER_STORM if MAX_FRAMES_PER_STORM > 0 else int(counts.max())
    chosen = None
    for c in range(cap, 3, -1):
        if int(np.minimum(counts, c).sum()) <= target_n:
            chosen = c
            break
    if chosen is None:
        chosen = 4
    print(f"!! preload would need {need_gb:.2f} GB > budget {budget_gb:.2f} GB -- "
          f"lowering the per-storm frame cap to {chosen}")

    keep = []
    for _key, g in index.groupby("storm_key", sort=False):
        pos = g.index.to_numpy()
        keep.append(pos[even_picks(len(pos), chosen)])
    index = index.loc[np.sort(np.concatenate(keep))].reset_index(drop=True)
    audit["cap_applied"] = int(chosen)
    audit["frames_after_budget"] = int(len(index))
    print(f"   -> {audit['frames_after_budget']} frames "
          f"({len(index) * bytes_per / 1e9:.2f} GB)")
    return index, audit


# ----------------------------------------------------------------------------
# Model -- resnet18, 1-channel conv1, SINGLE LINEAR OUTPUT (no sigmoid).
# ----------------------------------------------------------------------------
def build_model():
    """resnet18, ImageNet-pretrained, adapted to 1-channel input, 1 regression
    output.

    Kept as a STOCK torchvision resnet18 (only conv1 and fc replaced) so the
    saved state_dict is strictly loadable by the same one-line builder every
    other tool in this project already uses -- no wrapper module, no renamed
    keys. That is also why there is no dropout layer: inserting one would change
    the state_dict's key names for a regulariser that weight decay 5e-2, the
    augmentation preset and the EMA already cover.

    conv1's pretrained (64,3,7,7) weights are collapsed by MEAN (not sum) across
    the RGB axis: for a grayscale image the pretrained filter's response to a
    3x-replicated channel equals the mean-collapsed filter's response to a
    single channel, which preserves both the learned filters and their scale.
    """
    import torchvision

    pretrained = True
    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=weights)
    except Exception as exc:
        # Requires enable_internet: true in kernel-metadata.json (torchvision
        # fetches the weights from download.pytorch.org). Random init keeps the
        # run alive but will noticeably hurt accuracy -- loud on purpose.
        print(f"!! PRETRAINED WEIGHTS UNAVAILABLE ({exc}) -- using RANDOM INIT !!")
        net = torchvision.models.resnet18(weights=None)
        pretrained = False

    old = net.conv1.weight.data                     # (64,3,7,7)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.conv1.weight.data = old.mean(dim=1, keepdim=True)

    # Regression head: one linear output, NO sigmoid. Predicts the STANDARDISED
    # wind; denorm_kt() maps it back to knots.
    # (v3) Output width is the head's, not always 1. For HEAD_MODE="reg"
    # this is still exactly Linear(512, 1), so the control arm's state_dict is
    # byte-compatible with intensity_v1's.
    net.fc = nn.Linear(net.fc.in_features, head_out_dim())
    return net, pretrained


class ModelEma:
    """Exponential moving average of every floating-point tensor in the model's
    state_dict -- parameters AND BatchNorm running statistics. Verbatim from v5.

    BatchNorm running mean/var are themselves running averages so averaging them
    further is consistent, but `num_batches_tracked` is an integer counter, not a
    statistic, so it is COPIED rather than averaged.

    The decay is ramped as min(decay, (1+t)/(10+t)) over the first steps; without
    that ramp the average stays anchored to the random initialisation for
    ~1/(1-decay) steps and the EMA model is useless for the first several epochs.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.n_updates = 0
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self.non_float = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if not v.dtype.is_floating_point
        }

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
        out = {}
        for k, v in ref.items():
            src = self.shadow.get(k, self.non_float.get(k))
            out[k] = src.detach().clone().to(dtype=v.dtype, device=v.device)
        return out


# ----------------------------------------------------------------------------
# Metrics -- everything is reported in KNOTS, never in normalised units.
# ----------------------------------------------------------------------------
def regression_metrics(pred_kt, true_kt):
    pred = np.asarray(pred_kt, dtype=np.float64)
    true = np.asarray(true_kt, dtype=np.float64)
    n = int(len(true))
    if n == 0:
        return {"n_frames": 0, "mae_kt": None, "rmse_kt": None, "bias_kt": None,
                "median_ae_kt": None, "p90_ae_kt": None, "r2": None,
                "pearson_r": None, "true_mean_kt": None, "pred_mean_kt": None,
                "within_5kt": None, "within_10kt": None, "within_20kt": None}
    err = pred - true
    ae = np.abs(err)
    ss_res = float((err ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    if n > 1 and pred.std() > 0 and true.std() > 0:
        corr = float(np.corrcoef(pred, true)[0, 1])
    else:
        corr = float("nan")
    return {
        "n_frames": n,
        "mae_kt": float(ae.mean()),
        "rmse_kt": float(np.sqrt((err ** 2).mean())),
        "bias_kt": float(err.mean()),          # signed: + means over-predicting
        "median_ae_kt": float(np.median(ae)),
        "p90_ae_kt": float(np.percentile(ae, 90)),
        "r2": r2,
        "pearson_r": corr,
        "true_mean_kt": float(true.mean()),
        "pred_mean_kt": float(pred.mean()),
        "within_5kt": float((ae <= 5.0).mean()),
        "within_10kt": float((ae <= 10.0).mean()),
        "within_20kt": float((ae <= 20.0).mean()),
    }


def category_report(pred_kt, true_kt):
    """IMD-style bucket agreement: predicted knots -> bucket vs true bucket.

    'within_one_bucket' matters as much as exact accuracy here: the bins are
    narrow (14-16 kt wide in the middle of the range) relative to a realistic
    MAE, so a model that is never more than one bucket out is operationally very
    different from one that scatters across the table."""
    pred = np.asarray(pred_kt, dtype=np.float64)
    true = np.asarray(true_kt, dtype=np.float64)
    if len(true) == 0:
        return {"n_frames": 0, "accuracy": None, "within_one_bucket": None,
                "per_true_category": {}, "confusion": {}}
    pi = np.array([category_index_of(v) for v in pred])
    ti = np.array([category_index_of(v) for v in true])

    confusion = {tn: {pn: 0 for pn in CATEGORY_NAMES} for tn in CATEGORY_NAMES}
    for t, p in zip(ti, pi):
        confusion[CATEGORY_NAMES[t]][CATEGORY_NAMES[p]] += 1

    per_true = {}
    for i, name in enumerate(CATEGORY_NAMES):
        m = ti == i
        nb = int(m.sum())
        if nb == 0:
            continue
        per_true[name] = {
            "n_frames": nb,
            "recall": float((pi[m] == i).mean()),
            "within_one_bucket": float((np.abs(pi[m] - i) <= 1).mean()),
            "mae_kt": float(np.abs(pred[m] - true[m]).mean()),
            "bias_kt": float((pred[m] - true[m]).mean()),
        }

    return {
        "n_frames": int(len(true)),
        "accuracy": float((pi == ti).mean()),
        "within_one_bucket": float((np.abs(pi - ti) <= 1).mean()),
        "per_true_category": per_true,
        "confusion": confusion,
    }


def breakdown(pred_kt, true_kt, groups):
    """Per-group val metrics. The per-basin, per-source and per-WIND-CONVENTION
    reports all come from this -- Indian Ocean performance, and the JMA-10min vs
    JTWC-1min gap, both have to be visible on their own rather than hidden
    inside an average dominated by the Western Pacific / Digital Typhoon."""
    out = {}
    groups = np.asarray(groups)
    for g in sorted(set(groups.tolist())):
        m = groups == g
        d = regression_metrics(pred_kt[m], true_kt[m])
        cat = category_report(pred_kt[m], true_kt[m])
        d["category_accuracy"] = cat["accuracy"]
        d["category_within_one_bucket"] = cat["within_one_bucket"]
        out[str(g)] = d
    return out


# ----------------------------------------------------------------------------
# Inference helpers
# ----------------------------------------------------------------------------
def denorm_kt(pred_norm, target_mean_kt, target_std_kt):
    """Standardised model output -> knots, clipped to the physically observed
    range. This is the exact inverse the checkpoint's config documents, so any
    downstream tool can reproduce it from the saved constants alone."""
    kt = np.asarray(pred_norm, dtype=np.float64) * target_std_kt + target_mean_kt
    return np.clip(kt, PRED_CLIP_KT[0], PRED_CLIP_KT[1])


@torch.no_grad()
def predict_kt(net, loader, device, target_mean_kt, target_std_kt):
    """Val-side forward pass. Returns (pred_kt, true_kt), both in knots."""
    net.eval()
    preds, trues = [], []
    for imgs_u8, _t, wind_kt, _w in loader:
        x = to_model_input(imgs_u8.to(device))
        out = net(x).float().cpu().numpy()
        preds.append(out)
        trues.append(wind_kt.numpy())
    raw = np.concatenate(preds).astype(np.float64)
    true = np.concatenate(trues).astype(np.float64)
    pred_kt = head_decode_kt(raw, target_mean_kt, target_std_kt)
    predict_kt.last_raw = raw          # kept for the per-frame dump below
    return pred_kt, true


def basin_balance_weights(basins, alpha):
    """Per-frame loss weight ~ n_basin ** -alpha, renormalised to mean exactly 1
    (so alpha is a pure reweighting knob and does not silently retune the
    effective learning rate). alpha = 0 -> every weight exactly 1.0."""
    basins = np.asarray(basins)
    if alpha == 0.0:
        return np.ones(len(basins), dtype=np.float64)
    counts = Counter(basins.tolist())
    w = np.array([counts[b] ** (-alpha) for b in basins], dtype=np.float64)
    return w * (len(w) / w.sum())


def intensity_balance_weights(wind_kt, alpha):
    """(v2) Per-frame loss weight ~ n_category ** -alpha, renormalised to mean 1.
    alpha = 0.0 returns all-ones, i.e. bit-identical to intensity_v1."""
    wind_kt = np.asarray(wind_kt)
    if alpha == 0.0:
        return np.ones(len(wind_kt), dtype=np.float64)
    cats = np.array([category_index_of(float(w)) for w in wind_kt])
    counts = Counter(cats.tolist())
    w = np.array([counts[c] ** (-alpha) for c in cats], dtype=np.float64)
    return w * (len(w) / w.sum())


# ----------------------------------------------------------------------------
# (v3) OUTPUT PARAMETERISATIONS
# ----------------------------------------------------------------------------
# v2 proved the residual defect is COMPRESSION (predicted-vs-true slope ~0.82)
# and that loss weighting cannot remove it -- a scalar head under a
# squared-error-like loss is trained to emit the conditional MEAN, and shrinking
# toward the centre is the loss-optimal response to an ambiguous image. These
# heads change what the model emits instead of how the same scalar is weighted.
def _dist_grid():
    lo, hi = PRED_CLIP_KT
    edges = np.arange(lo, hi + DIST_BIN_KT, DIST_BIN_KT, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers


DIST_EDGES, DIST_CENTERS = _dist_grid()
N_DIST_BINS = len(DIST_CENTERS)
# Ordinal thresholds sit at bin CENTRES (12.5, 17.5, ... 197.5), deliberately
# OFF the 5-kt reporting grid. 91.3% of best-track winds in this dataset are
# exact multiples of 5 (100% of Digital Typhoon, 79.9% of TCIR), so thresholds
# placed ON the grid would make "kt > edge" ambiguous exactly where the data
# piles up -- a true 25 kt would not count its own 25 kt threshold and the
# decode would land half a bin low on nine frames out of ten. Offset by half a
# bin, count == (kt - 10)/5 exactly for every on-grid value, and off-grid values
# are bounded by the 2.5 kt half-bin resolution, which is the best any 5 kt
# parameterisation can do. Caught by the round-trip test in test_v3_heads.py.
ORDINAL_EDGES = DIST_CENTERS
# IMD category lower boundaries, for torch.bucketize in the multitask head.
CATEGORY_LOWER_EDGES = [float(lo) for lo, _hi, _n in CATEGORY_BINS[1:]]


def head_out_dim():
    if HEAD_MODE == "dist":
        return N_DIST_BINS
    if HEAD_MODE == "ordinal":
        return len(ORDINAL_EDGES)
    if HEAD_MODE == "mt":
        return 1 + len(CATEGORY_BINS)
    return 1


def head_loss(out, t_norm, kt):
    """PER-SAMPLE loss vector, so the existing weighting
    `(loss_vec * wt).sum() / wt.sum()` keeps working unchanged for every head."""
    if HEAD_MODE == "dist":
        centers = torch.as_tensor(DIST_CENTERS, device=out.device,
                                  dtype=torch.float32)
        d = (kt[:, None] - centers[None, :]) / DIST_SIGMA_KT
        # Gaussian soft target, renormalised to a proper distribution. Soft
        # rather than one-hot because best-track winds are recorded in 5 kt
        # steps: a hard label would assert a precision the data does not have.
        soft = torch.softmax(-0.5 * d * d, dim=1)
        return -(soft * torch.log_softmax(out, dim=1)).sum(1)
    if HEAD_MODE == "ordinal":
        edges = torch.as_tensor(ORDINAL_EDGES, device=out.device,
                                dtype=torch.float32)
        tgt = (kt[:, None] > edges[None, :]).float()
        return F.binary_cross_entropy_with_logits(
            out, tgt, reduction="none").sum(1)
    if HEAD_MODE == "mt":
        reg = F.smooth_l1_loss(out[:, 0], t_norm, beta=HUBER_BETA,
                               reduction="none")
        bounds = torch.as_tensor(CATEGORY_LOWER_EDGES, device=out.device,
                                 dtype=kt.dtype)
        cat = torch.bucketize(kt, bounds)
        ce = F.cross_entropy(out[:, 1:], cat, reduction="none")
        return reg + MT_CAT_WEIGHT * ce
    return F.smooth_l1_loss(out[:, 0], t_norm, beta=HUBER_BETA,
                            reduction="none")


def head_decode_kt(out_np, target_mean_kt, target_std_kt):
    """Raw head output (numpy, (N, out_dim)) -> knots, clipped like denorm_kt."""
    lo, hi = PRED_CLIP_KT
    if HEAD_MODE == "dist":
        z = out_np - out_np.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return np.clip(p @ DIST_CENTERS, lo, hi)
    if HEAD_MODE == "ordinal":
        # Expected number of centre-thresholds exceeded -> knots. With the
        # thresholds at bin centres this inverts exactly: an on-grid wind w
        # exceeds precisely (w - 10)/5 of them.
        s = 1.0 / (1.0 + np.exp(-out_np))
        return np.clip(DIST_EDGES[0] + DIST_BIN_KT * s.sum(axis=1), lo, hi)
    return denorm_kt(out_np[:, 0], target_mean_kt, target_std_kt)


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


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    print("=" * 72)
    print(f"Cyclone Genesis Predictor MATURE-CYCLONE INTENSITY REGRESSION -- {MODEL_VERSION}")
    print(f"  EPOCHS={EPOCHS} BATCH={BATCH_SIZE} LR={LR} WD={WEIGHT_DECAY} "
          f"AMP={USE_AMP} frames/storm<={MAX_FRAMES_PER_STORM}")
    print(f"  MATURE RULE: DT grade > {DT_EARLY_GRADE_MAX}, "
          f"TCIR Vmax > {TCIR_EARLY_VMAX_KT} kt (post-formation only)")
    print("=" * 72)

    # ------------------------------------------------------------------ discovery
    dt_root = find_dt_root()
    if dt_root is None:
        raise SystemExit(f"No Digital Typhoon mount under {INPUT_ROOT} -- "
                         f"check dataset_sources.")
    meta_dir, image_dir = resolve_dt_dirs(dt_root)
    print("DT ROOT   =", dt_root)
    print("meta_dir  =", meta_dir)
    print("image_dir =", image_dir)

    tcir_path = find_tcir_h5()
    if tcir_path is None:
        raise SystemExit(
            f"No TCSA/TCIR .h5 under {INPUT_ROOT} -- attach "
            "boyochen/tropical-cyclone-structure-analysis-tcsa as a dataset source"
        )
    print("TCIR h5   =", tcir_path, f"({os.path.getsize(tcir_path) / 1e9:.1f} GB)")

    # ------------------------------------------------------------------ TCIR first
    # Read and validate TCIR's metadata table BEFORE the expensive Digital
    # Typhoon CSV walk, so a wrong assumption fails in the first minute.
    #
    # pd.read_hdf, NOT a raw h5py structured-array read: TCIR's 'info' is a
    # pandas HDF5 table. An earlier draft guessed the h5py form and was wrong;
    # this is the call the working audit and v5 trainer both use.
    print("\nLoading TCIR 'info' table...")
    t0 = time.time()
    info = pd.read_hdf(tcir_path, key="info", mode="r")
    print(f"  info shape={info.shape} columns={list(info.columns)} "
          f"({time.time() - t0:.0f}s)")
    need_b = {"region", "ID", "lon", "lat", "time", "Vmax"}
    missing_b = need_b - set(info.columns)
    if missing_b:
        raise SystemExit(f"TCIR 'info' is missing expected columns {sorted(missing_b)}; "
                         f"got {list(info.columns)}")
    if not isinstance(info.index, pd.RangeIndex):
        print(f"  NOTE: info.index is {type(info.index).__name__}, not a RangeIndex; "
              f"using positional row numbers as image indices (row i of info == images[i])")
    info = info.reset_index(drop=True)

    import h5py
    with h5py.File(tcir_path, "r") as f:
        img_shape = f["images"].shape
    print(f"  images shape={img_shape}")
    if img_shape[0] != len(info):
        raise SystemExit(
            f"TCIR row-alignment broken: images has {img_shape[0]} rows but info has "
            f"{len(info)}. frame_ref indices would point at the wrong storms."
        )
    if len(img_shape) != 4 or img_shape[-1] < 1:
        raise SystemExit(f"Unexpected TCIR images shape {img_shape}; expected (N,128,128,4)")

    tcir_index, tcir_work, tcir_audit = build_tcir_index_mature(info)
    if tcir_index.empty:
        raise SystemExit("TCIR mature index is empty -- layout is not what was expected")

    # ------------------------------------------------------------------ Digital Typhoon
    print()
    dt_index, dt_tracks, dt_audit = build_dt_index_mature(meta_dir, image_dir)
    if dt_index.empty:
        raise SystemExit("Digital Typhoon mature index is empty -- "
                         "layout is not what was expected")

    # ------------------------------------------------------------------ dedup
    drop_ids, dedup_audit = dedup_wpac(dt_index, tcir_index, tcir_work, dt_tracks)
    tcir_index = tcir_index[~tcir_index["storm_key"].isin(drop_ids)].reset_index(drop=True)

    index = pd.concat([dt_index, tcir_index], ignore_index=True)
    index["category"] = index["wind_kt"].apply(category_of)

    # storm_key must be globally unique across the two sources: it is the split
    # key, so a collision would merge two different storms into one split unit.
    # Digital Typhoon ids are 6 digits and TCIR ids end in a letter, so this is
    # structurally impossible -- assert it rather than assume it.
    clashing = index.groupby("storm_key")["source"].nunique()
    clashing = clashing[clashing > 1]
    if len(clashing):
        raise SystemExit(
            f"storm_key collision across sources: {list(clashing.index[:10])}. "
            f"Two different storms would be merged into one split unit; aborting."
        )

    index = index.reset_index(drop=True)
    index, budget_audit = fit_index_to_ram_budget(index)

    print("\n" + "=" * 72)
    print("MATURE INDEX (post-dedup, post-cap)")
    print("=" * 72)
    print(f"Total frames: {len(index)}   distinct storms: {index['storm_key'].nunique()}")
    print(f"\nFrames by source:\n{index['source'].value_counts()}")
    print(f"\nFrames by wind_convention:\n{index['wind_convention'].value_counts()}")
    print(f"\nFrames by basin:\n{index['basin'].value_counts()}")
    print(f"\nStorms by basin:\n{index.groupby('basin')['storm_key'].nunique()}")
    print(f"\nwind_kt distribution:\n{index['wind_kt'].describe()}")
    print(f"\nFrames by IMD category:\n{index['category'].value_counts()}")

    # The mixing this run is deliberately accepting, stated as a number rather
    # than left implicit. REPORT ONLY -- no target is touched.
    print("\n--- wind-convention diagnostic (report only, no conversion applied) ---")
    for conv, g in index.groupby("wind_convention"):
        print(f"  {conv:10s} n={len(g):7d}  mean={g['wind_kt'].mean():6.2f} kt  "
              f"std={g['wind_kt'].std():5.2f}  max={g['wind_kt'].max():.0f}")
    conv_means = index.groupby("wind_convention")["wind_kt"].mean().to_dict()
    if len(conv_means) == 2 and "JTWC1min" in conv_means:
        equiv = conv_means["JTWC1min"] * WIND_10MIN_PER_1MIN
        print(f"  TCIR's mean 1-min {conv_means['JTWC1min']:.2f} kt is "
              f"~{equiv:.2f} kt in JMA 10-min units "
              f"(x{WIND_10MIN_PER_1MIN}); Digital Typhoon's own 10-min mean is "
              f"{conv_means.get('JMA10min', float('nan')):.2f} kt.")

    # ------------------------------------------------------------------ split
    # SPLIT BY STORM, NEVER BY FRAME. Frames of one storm hours apart are nearly
    # the same picture with nearly the same wind; letting a storm span both
    # splits would let the model score by recognising the storm, not by reading
    # intensity. Stratified by (basin, peak-intensity category) so both splits
    # cover the full intensity range in every basin.
    storm_tbl = (index.groupby("storm_key")
                 .agg(basin=("basin", "first"), source=("source", "first"),
                      peak_kt=("wind_kt", "max"))
                 .reset_index())
    storm_tbl["strat"] = storm_tbl["peak_kt"].apply(category_of)
    rng = np.random.RandomState(SEED)
    val_keys = set()
    for (basin, strat), grp in storm_tbl.groupby(["basin", "strat"], sort=True):
        keys = np.array(sorted(grp["storm_key"].tolist()))
        rng.shuffle(keys)
        n_val = int(round(len(keys) * VAL_FRACTION))
        if n_val == 0 and len(keys) >= 2:
            n_val = 1
        val_keys.update(keys[:n_val].tolist())
        print(f"  split {basin:5s} peak={strat:34s}: {len(keys):5d} storms -> "
              f"{n_val:4d} val / {len(keys) - n_val:5d} train")
    all_keys = set(index["storm_key"].unique())
    train_keys = all_keys - val_keys
    index["split"] = np.where(index["storm_key"].isin(val_keys), "val", "train")
    print(f"\nStorms: {len(all_keys)}  train={len(train_keys)}  val={len(val_keys)}")

    # ------------------------------------------------------------------ preload
    images, ok = preload_all(index, tcir_path)
    index = index[ok].reset_index(drop=True)
    images = images[ok]
    n_load_failures = int((~ok).sum())
    index.to_csv(INDEX_PATH, index=False)
    print(f"Wrote {INDEX_PATH} ({len(index)} rows)")

    wind_kt = index["wind_kt"].to_numpy(np.float64)
    is_val = (index["split"] == "val").to_numpy()
    tr_idx = np.where(~is_val)[0]
    va_idx = np.where(is_val)[0]
    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise SystemExit("Empty train or val split after preload; aborting.")
    print(f"Frames: train={len(tr_idx)}  val={len(va_idx)}")

    # (v3) Every number here is compared against v1/v2 runs, which is only valid
    # on the identical split. Deterministic, so it must reproduce exactly.
    _V1 = (1628, 410, 47376, 11953)
    _nts = index.loc[~is_val, "storm_key"].nunique()
    _nvs = index.loc[is_val, "storm_key"].nunique()
    print(f"(v3) REPRODUCTION CHECK: storms {_nts}/{_nvs} (v1 {_V1[0]}/{_V1[1]}), "
          f"frames {len(tr_idx)}/{len(va_idx)} (v1 {_V1[2]}/{_V1[3]})")
    assert (_nts, _nvs) == (_V1[0], _V1[1]), (
        f"(v3) SPLIT CHANGED ({_nts}/{_nvs}) -- comparisons would be invalid.")
    assert abs(len(tr_idx) - _V1[2]) <= 50 and abs(len(va_idx) - _V1[3]) <= 50, (
        f"(v3) FRAME COUNTS DRIFTED ({len(tr_idx)}/{len(va_idx)}).")

    # ------------------------------------------------- target standardisation
    # Fitted on the TRAIN frames ONLY -- using val statistics would leak the val
    # distribution into the model's output scale. Both constants go into the
    # checkpoint config so inference can denormalise back to knots.
    target_mean_kt = float(wind_kt[tr_idx].mean())
    target_std_kt = float(wind_kt[tr_idx].std())
    if not np.isfinite(target_std_kt) or target_std_kt < 1e-6:
        raise SystemExit(f"Degenerate target std {target_std_kt}; aborting.")
    target_norm = ((wind_kt - target_mean_kt) / target_std_kt).astype(np.float32)
    print(f"\nTarget standardisation (train-set only): mean={target_mean_kt:.3f} kt "
          f"std={target_std_kt:.3f} kt")
    print(f"  inverse used at inference: kt = pred * {target_std_kt:.6f} "
          f"+ {target_mean_kt:.6f}, clipped to {PRED_CLIP_KT}")

    val_basins = index["basin"].to_numpy()[va_idx]
    val_sources = index["source"].to_numpy()[va_idx]
    val_convs = index["wind_convention"].to_numpy()[va_idx]
    val_true_kt = wind_kt[va_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(USE_AMP and device.type == "cuda")
    print("\nDevice:", device,
          torch.cuda.get_device_name(0) if device.type == "cuda" else "",
          f"AMP={use_amp}")

    train_w = basin_balance_weights(index["basin"].to_numpy()[tr_idx],
                                    BASIN_BALANCE_ALPHA)
    _int_w = intensity_balance_weights(wind_kt[tr_idx], INTENSITY_BALANCE_ALPHA)
    train_w = train_w * _int_w
    train_w = train_w * (len(train_w) / train_w.sum())
    print(f"\n(v3) HEAD_MODE={HEAD_MODE}  out_dim={head_out_dim()}  "
          f"alpha={INTENSITY_BALANCE_ALPHA}  dist_sigma={DIST_SIGMA_KT}")
    train_ds = FrameDataset(images[tr_idx], target_norm[tr_idx], wind_kt[tr_idx],
                            train_w, augment=True)
    val_ds = FrameDataset(images[va_idx], target_norm[va_idx], wind_kt[va_idx],
                          np.ones(len(va_idx)), augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    probe_sel = np.random.RandomState(SEED + 1).choice(
        len(tr_idx), min(TRAIN_PROBE_FRAMES, len(tr_idx)), replace=False)
    probe_ds = FrameDataset(images[tr_idx[probe_sel]], target_norm[tr_idx[probe_sel]],
                            wind_kt[tr_idx[probe_sel]], np.ones(len(probe_sel)),
                            augment=False)
    probe_dl = DataLoader(probe_ds, batch_size=64, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------ model
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

    # A constant predictor is the honest floor to beat: "always say the mean
    # intensity" already gets a respectable-looking MAE on a distribution this
    # tight, so the model's MAE is only meaningful next to this number.
    baseline_pred = np.full(len(va_idx), target_mean_kt, dtype=np.float64)
    baseline_metrics = regression_metrics(baseline_pred, val_true_kt)
    baseline_cat = category_report(baseline_pred, val_true_kt)
    print(f"\nBaseline (always predict the train mean, {target_mean_kt:.1f} kt): "
          f"MAE={baseline_metrics['mae_kt']:.2f} kt  "
          f"RMSE={baseline_metrics['rmse_kt']:.2f} kt  "
          f"category acc={baseline_cat['accuracy']:.3f}")

    # ------------------------------------------------------------------ train
    history, best_mae, best_epoch, best_state = [], float("inf"), -1, None
    print("\n" + "=" * 72)
    print("TRAINING")
    print("=" * 72)
    for epoch in range(1, EPOCHS + 1):
        aug_gen.manual_seed(SEED * 1000 + epoch)
        net.train()
        tot_l, tot_w = 0.0, 0.0
        t0 = time.time()
        for imgs_u8, t_norm, _kt, w in train_dl:
            x01 = to_unit(imgs_u8.to(device, non_blocking=True))
            x = normalize(augment_batch(x01, aug_gen, AUG))
            t = t_norm.to(device).float()
            wt = w.to(device).float()

            opt.zero_grad(set_to_none=True)
            with amp_autocast(use_amp):
                out = net(x).float()
                # (v3) Per-sample loss vector from the head, so the existing
                # per-frame weighting below is untouched by the head change.
                loss_vec = head_loss(out, t, _kt.to(device).float())
                loss = (loss_vec * wt).sum() / wt.sum()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(net)
            tot_l += float(loss.detach().item()) * float(wt.sum().item())
            tot_w += float(wt.sum().item())
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()

        # Evaluate the EMA weights, then restore the live ones -- the EMA model
        # is what gets selected and shipped.
        live = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(ema.state_dict(net))
        vp, vt = predict_kt(net, val_dl, device, target_mean_kt, target_std_kt)
        m = regression_metrics(vp, vt)
        cat = category_report(vp, vt)
        pp, pt = predict_kt(net, probe_dl, device, target_mean_kt, target_std_kt)
        probe_mae = float(np.abs(pp - pt).mean())
        rec = dict(m)
        rec.update({"epoch": epoch, "lr": float(cur_lr),
                    "train_loss": tot_l / max(1e-9, tot_w),
                    "category_accuracy": cat["accuracy"],
                    "category_within_one_bucket": cat["within_one_bucket"],
                    "train_probe_mae_kt": probe_mae})
        history.append(rec)
        print(f"epoch {epoch:2d}/{EPOCHS}  lr={cur_lr:.2e}  loss={rec['train_loss']:.4f}  "
              f"val_MAE={m['mae_kt']:.2f}kt  val_RMSE={m['rmse_kt']:.2f}kt  "
              f"cat_acc={cat['accuracy']:.3f}  "
              f"[probe] train_MAE={probe_mae:.2f}kt  "
              f"gap={m['mae_kt'] - probe_mae:+.2f}kt  ({time.time() - t0:.0f}s)")

        # Selection on val MAE in KNOTS -- the number the product is judged on.
        if math.isfinite(m["mae_kt"]) and m["mae_kt"] < best_mae:
            best_mae, best_epoch = m["mae_kt"], epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(live)

    if best_state is None:
        raise SystemExit("No epoch produced a scoreable validation result.")
    tail = [h["mae_kt"] for h in history[-5:] if h["mae_kt"] is not None]
    print(f"\nSelection audit: best val MAE {best_mae:.3f} kt at epoch {best_epoch} "
          f"of {EPOCHS}; last-5 mean={np.mean(tail):.3f} sd={np.std(tail):.3f}")

    # ------------------------------------------------------------------ report
    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    pred_kt, true_kt = predict_kt(net, val_dl, device, target_mean_kt, target_std_kt)

    # (v3) THE MULTIPLIER ON THIS BATCH. With per-frame predictions from
    # every arm on the identical val split, ensembling across heads, post-hoc
    # linear recalibration, and alternative decodes of the distributional
    # posterior all become free CPU analysis instead of more GPU sessions.
    _val_idx = index.loc[is_val]
    np.savez_compressed(
        os.path.join(OUT_DIR, "val_predictions.npz"),
        pred_kt=pred_kt, true_kt=true_kt,
        raw=getattr(predict_kt, "last_raw", np.zeros((len(pred_kt), 0))),
        storm_key=_val_idx["storm_key"].to_numpy().astype(str),
        basin=_val_idx["basin"].to_numpy().astype(str),
        source=_val_idx["source"].to_numpy().astype(str),
        head_mode=np.array([HEAD_MODE]),
        alpha=np.array([INTENSITY_BALANCE_ALPHA]),
        dist_centers=DIST_CENTERS,
        target_mean_kt=np.array([target_mean_kt]),
        target_std_kt=np.array([target_std_kt]),
    )
    print(f"(v3) wrote val_predictions.npz ({len(pred_kt)} frames)")

    val_metrics = regression_metrics(pred_kt, true_kt)
    cat_report = category_report(pred_kt, true_kt)
    val_metrics["category_accuracy"] = cat_report["accuracy"]
    val_metrics["category_within_one_bucket"] = cat_report["within_one_bucket"]
    val_metrics["epoch"] = best_epoch

    by_basin = breakdown(pred_kt, true_kt, val_basins)
    by_source = breakdown(pred_kt, true_kt, val_sources)
    by_convention = breakdown(pred_kt, true_kt, val_convs)
    by_true_category = {}
    for name in CATEGORY_NAMES:
        m = np.array([category_of(v) == name for v in true_kt])
        if m.sum():
            by_true_category[name] = regression_metrics(pred_kt[m], true_kt[m])

    print("\n" + "=" * 72)
    print("VALIDATION (best epoch, EMA weights)")
    print("=" * 72)
    print(f"POOLED     MAE={val_metrics['mae_kt']:.2f} kt  "
          f"RMSE={val_metrics['rmse_kt']:.2f} kt  bias={val_metrics['bias_kt']:+.2f} kt  "
          f"r={val_metrics['pearson_r']:.3f}  n={val_metrics['n_frames']}")
    print(f"           category accuracy={cat_report['accuracy']:.3f}  "
          f"within one bucket={cat_report['within_one_bucket']:.3f}")
    print(f"BASELINE   MAE={baseline_metrics['mae_kt']:.2f} kt  "
          f"RMSE={baseline_metrics['rmse_kt']:.2f} kt  (always predict the mean)")

    print("\nBy BASIN:")
    for g, d in sorted(by_basin.items()):
        print(f"  {g:6s} n={d['n_frames']:6d}  MAE={d['mae_kt']:6.2f}  "
              f"RMSE={d['rmse_kt']:6.2f}  bias={d['bias_kt']:+6.2f}  "
              f"cat_acc={d['category_accuracy']:.3f}")
    print("\nBy SOURCE:")
    for g, d in sorted(by_source.items()):
        print(f"  {g:16s} n={d['n_frames']:6d}  MAE={d['mae_kt']:6.2f}  "
              f"RMSE={d['rmse_kt']:6.2f}  bias={d['bias_kt']:+6.2f}  "
              f"cat_acc={d['category_accuracy']:.3f}")
    print("\nBy WIND CONVENTION (the pooling caveat, measured):")
    for g, d in sorted(by_convention.items()):
        print(f"  {g:10s} n={d['n_frames']:6d}  MAE={d['mae_kt']:6.2f}  "
              f"RMSE={d['rmse_kt']:6.2f}  bias={d['bias_kt']:+6.2f}  "
              f"true_mean={d['true_mean_kt']:6.2f}  pred_mean={d['pred_mean_kt']:6.2f}")
    print("  ^ These two rows are DIFFERENT PHYSICAL UNITS (JMA 10-minute vs "
          "JTWC/NHC 1-minute\n    sustained wind). A gap between them is partly "
          "the convention, not only model error.")
    print("\nBy TRUE IMD CATEGORY:")
    for name in CATEGORY_NAMES:
        d = by_true_category.get(name)
        if d:
            print(f"  {name:34s} n={d['n_frames']:6d}  MAE={d['mae_kt']:6.2f}  "
                  f"bias={d['bias_kt']:+6.2f}")

    # ------------------------------------------------------------------ save
    config = jsonable({
        "task": "intensity_regression",
        "task_note": ("single mature IR frame -> current sustained wind (kt), "
                      "continuous; IMD category derived from the predicted knots "
                      "via category_bins"),
        # preprocessing constants, named exactly as in the early-stage
        # checkpoints so a loader written for those finds what it expects
        "temp_lo": TEMP_LO, "temp_hi": TEMP_HI, "img_size": IMG_SIZE,
        "norm_mean": NORM_MEAN, "norm_std": NORM_STD,
        "typhoon_grade": TYPHOON_GRADE,
        "arch": "resnet18_1ch_regression",
        "output": "one linear unit, NO sigmoid, in standardised target units",
        # target normalisation -- everything needed to get back to knots
        "target": {
            "name": "wind_kt",
            "transform": "standardised on TRAIN frames only",
            "target_mean_kt": target_mean_kt,
            "target_std_kt": target_std_kt,
            "inverse": "wind_kt = model_output * target_std_kt + target_mean_kt",
            "pred_clip_kt": list(PRED_CLIP_KT),
            "clip_note": ("predictions are clipped to the observed physical "
                          "range before display or scoring"),
        },
        "category_bins": [[lo, hi, name] for lo, hi, name in CATEGORY_BINS],
        # the mature-frame rule, so the checkpoint carries its own provenance
        "mature_rule": mature_rule_note(),
        "dt_early_grade_max": DT_EARLY_GRADE_MAX,
        "tcir_early_vmax_kt": TCIR_EARLY_VMAX_KT,
        "frames_per_storm_cap": MAX_FRAMES_PER_STORM,
        "frames_per_storm_cap_applied": budget_audit["cap_applied"],
        # the wind-convention caveat, carried in the checkpoint itself
        "wind_convention": wind_convention_note(),
        "wind_convention_by_source": WIND_CONVENTION_OF_SOURCE,
        "wind_10min_per_1min_reference_only": WIND_10MIN_PER_1MIN,
        "sources": ["digital_typhoon", "tcir"],
        "basins": sorted(index["basin"].unique().tolist()),
        "model_version": MODEL_VERSION,
        "seed": SEED,
        "split": "storm-level, stratified by (basin, peak-intensity category)",
        "hyperparameters": {
            "epochs": EPOCHS, "warmup_epochs": WARMUP_EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "ema_decay": EMA_DECAY, "loss": f"SmoothL1 (Huber), beta={HUBER_BETA}",
            "aug": AUG, "basin_balance_alpha": BASIN_BALANCE_ALPHA,
            "intensity_balance_alpha": INTENSITY_BALANCE_ALPHA,
            "head_mode": HEAD_MODE, "head_out_dim": head_out_dim(),
            "dist_bin_kt": DIST_BIN_KT, "dist_sigma_kt": DIST_SIGMA_KT,
            "mt_cat_weight": MT_CAT_WEIGHT,
        },
        "pretrained_backbone": pretrained,
    })
    json.dumps(config)                      # fail here, not after torch.save

    torch.save(
        {
            # A stock torchvision resnet18 with a 1-channel conv1 and
            # fc = Linear(512, 1): strictly loadable by the same builder every
            # other tool in this project uses.
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "arch": "resnet18_1ch_regression",
            "task": "intensity_regression",
            "pretrained_backbone": pretrained,
            "epoch": best_epoch,
            "config": config,
            "val_metrics": jsonable(val_metrics),
            "val_metrics_by_basin": jsonable(by_basin),
            "val_metrics_by_source": jsonable(by_source),
            "val_metrics_by_wind_convention": jsonable(by_convention),
            "val_metrics_by_true_category": jsonable(by_true_category),
            "val_category_report": jsonable(cat_report),
            "baseline_constant_mean": jsonable(baseline_metrics),
            "train_storm_ids": sorted(train_keys),
            "val_storm_ids": sorted(val_keys),
        },
        CKPT_PATH,
    )

    summary = jsonable({
        "task": "mature cyclone IR frame -> current sustained wind (kt) + IMD category",
        "model_version": MODEL_VERSION,
        "mature_rule": mature_rule_note(),
        "wind_convention_caveat": wind_convention_note(),
        "wind_convention_by_source": WIND_CONVENTION_OF_SOURCE,
        "n_storms": len(all_keys),
        "n_train_storms": len(train_keys),
        "n_val_storms": len(val_keys),
        "n_train_frames": int(len(tr_idx)),
        "n_val_frames": int(len(va_idx)),
        "n_load_failures": n_load_failures,
        "index_summary": {
            "frames_total": int(len(index)),
            "frames_by_source": index["source"].value_counts().to_dict(),
            "frames_by_basin": index["basin"].value_counts().to_dict(),
            "frames_by_wind_convention":
                index["wind_convention"].value_counts().to_dict(),
            "frames_by_category": index["category"].value_counts().to_dict(),
            "storms_by_basin": index.groupby("basin")["storm_key"].nunique().to_dict(),
            "wind_kt_describe": index["wind_kt"].describe().to_dict(),
            "wind_kt_mean_by_convention":
                index.groupby("wind_convention")["wind_kt"].mean().to_dict(),
        },
        "frame_selection": {
            "rule": mature_rule_note(),
            "dt_mature_frames_eligible": dt_audit.get("n_frames_mature_eligible", 0),
            "tcir_mature_frames_eligible": tcir_audit.get("n_frames_mature_eligible", 0),
            "per_storm_cap": MAX_FRAMES_PER_STORM,
            "dt_frames_dropped_by_cap": dt_audit.get("n_frames_dropped_by_cap", 0),
            "tcir_frames_dropped_by_cap": tcir_audit.get("n_frames_dropped_by_cap", 0),
            "ram_budget": budget_audit,
        },
        "target_normalisation": {
            "target_mean_kt": target_mean_kt, "target_std_kt": target_std_kt,
            "fitted_on": "train frames only",
            "inverse": "wind_kt = model_output * target_std_kt + target_mean_kt",
            "pred_clip_kt": list(PRED_CLIP_KT),
        },
        "hyperparameters": config["hyperparameters"],
        "dedup": dedup_audit,
        "digital_typhoon_audit": dt_audit,
        "tcir_audit": tcir_audit,
        "best_epoch": best_epoch,
        "selection_audit": {
            "best_val_mae_kt_ema": float(best_mae),
            "last5_mae_mean": float(np.mean(tail)) if tail else None,
            "last5_mae_sd": float(np.std(tail)) if tail else None,
        },
        "baseline_constant_mean": baseline_metrics,
        "baseline_constant_mean_category": baseline_cat,
        "best_val_metrics": val_metrics,
        "best_val_metrics_by_basin": by_basin,
        "best_val_metrics_by_source": by_source,
        "best_val_metrics_by_wind_convention": by_convention,
        "best_val_metrics_by_true_category": by_true_category,
        "best_val_category_report": cat_report,
        "history": history,
    })
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BEST (pooled) ===")
    print(json.dumps(jsonable(val_metrics), indent=2))
    print("\nWrote", CKPT_PATH, METRICS_PATH, INDEX_PATH)
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
