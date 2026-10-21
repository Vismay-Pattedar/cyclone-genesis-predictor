"""
Cyclone Genesis Predictor -- Gate 2 v2: "is a tropical cyclone ACTUALLY PRESENT and organized
in this real satellite IR frame?"

WHY v2 EXISTS (read this before changing anything below)
========================================================
Gate 2 v1 (train_kernel_gate2_cyclonecheck) scored val AUC 0.99995, precision
1.000, ZERO false positives on 953 negatives. That is not a triumph, it is the
signature of a shortcut. Three pieces of evidence, all from this project's own
artifacts, not from theory:

1. v1's OWN metrics.json reports `val_metrics_by_source` with "auc": null for
   EVERY source. That is because each source contained exactly one class:
   digital_typhoon 100% positive, tcir 100% positive, maritime_continent 100%
   negative. A per-source AUC is undefined when a source is single-class. So
   v1 never measured, even once, whether the model can separate cyclone from
   non-cyclone WITHIN a single rendering pipeline. Its headline 0.99995 is
   fully achievable by answering "which of the two loaders produced this
   image?" -- positives went through kelvin_to_display_u8() on real Kelvin
   from .h5, negatives went through PIL -> pixel/255*150+180 -> the same
   kelvin_to_display_u8 on generic 813x801 PNGs. Different sensor, different
   render path, different fixed field of view, ONE negative source.

2. train_kernel_needle_search/output/needle_search_full.csv scored v1's actual
   gate2 checkpoint on 20,447 REAL early-stage (pre-first_cross) frames from
   Digital Typhoon and TCIR -- the exact input domain Model 3 exists to serve.
   Result: mean gate2 prob 0.99497, MEDIAN 1.000000, and only 81 of 20,447
   frames (0.40%) score below 0.5. TCIR early frames: mean 1.000000, i.e.
   every single one. Correlation between gate2 prob and the frame's actual
   wind speed: 0.096, essentially none. The gate does not respond to whether a
   storm has organized at all; it saturates on source identity.

3. Consequence in the shipped product: Model 3 is unreachable. The live app
   routes Gate2>=0.5 to "CYCLONE DECLARED" and never calls Model 3, so 99.6%
   of Model 3's own intended inputs short-circuit before reaching it. The
   reported real-world symptoms (an extratropical December bomb cyclone over
   Japan scoring ~99% on Gate 2; Gate 2 flagging any tracked-storm frame no
   matter how diffuse) are the same defect seen from the outside.

WHAT v2 CHANGES
===============
(a) NEGATIVES NOW COME THROUGH THE POSITIVES' OWN PIPELINE.
    The primary new negative source is Digital Typhoon / TCIR themselves:
    genuinely PRE-ORGANIZATION frames, taken strictly BEFORE the storm first
    crosses out of the early stage (the same `first_cross` definition already
    used by train_kernel_needle_search/search.py and train_kernel_v5_final).
    These are the same .h5 files, the same loader, the same sensor, the same
    storm-centred crop geometry, frequently the SAME STORM as a positive.
    There is no "which pipeline" signal available between a storm's frame 3
    and its frame 60. This is the strongest anti-shortcut control obtainable
    and it costs no new data.

    Is that too close to Gate 2's job? No -- and this is the design decision
    the whole pipeline hinges on. Gate 2's purpose in the 3-stage pipeline is
    to decide whether to SHORT-CIRCUIT to "cyclone declared" or to hand the
    frame to Model 3. Model 3's stated input domain is "ONE early-stage IR
    frame of an already-tracked but still weak/disorganised tropical cyclone."
    If Gate 2 answers "cyclone present" for every tracked-storm frame
    including the weak/disorganised ones, then Model 3 is dead code by
    construction -- which is exactly what evidence (2) and (3) above measured.
    Defining Gate 2's boundary with the SAME `first_cross` that defines Model
    3's training domain makes the two components share one boundary instead of
    overlapping. v1 asked "was this frame taken from a storm that was ever
    tracked?", which is not a question about the image at all. v2 asks "has
    this system organized yet?", which is.

(b) A DEADBAND, so the boundary is honest.
    Storm organization is continuous; a frame 6h before first_cross looks like
    one 6h after. Forcing a label there manufactures noise. So:
      negative = strictly before first_cross AND frame intensity <= 25 kt
      positive = at/after first_cross AND frame intensity >= 35 kt (34 kt is
                 the tropical-storm threshold) AND, for DT, JMA grade in 3/4/5
      anything in between (the 25-35 kt TD/TS grey zone) is DROPPED, not
                 labelled.

(c) EXTRATROPICAL FRAMES ARE NOT POSITIVES.
    DT grade >= 6 is extratropical / post-transition. v1 swept those into the
    positive class, which teaches "large organized non-tropical vortex =
    cyclone" -- precisely the reported Japan bomb-cyclone false positive. v2
    removes them from training by default (EXTRATROPICAL_MODE="drop") and
    instead SCORES them as a held-out diagnostic set, reporting their mean
    probability in metrics.json. That is a direct, honest measurement of the
    reported symptom rather than a guess. Setting EXTRATROPICAL_MODE="negative"
    is the aggressive lever if the product wants strictly-tropical semantics;
    left off by default because an ex-tropical frame often still carries a
    tropical core, and mislabelling those is its own noise.

(d) THE REMAINING NON-STORM SOURCES ARE SCALE-DIVERSIFIED.
    maritime-continent is kept (it is real, IBTrACS-verified IR) but is no
    longer used at one fixed 813x801 field of view. v1's negatives were ALL at
    that single zoom, so "spatial-frequency / field-of-view signature" was a
    second usable shortcut, and it explains why an arbitrary rendered storm
    PNG (the Japan bomb cyclone) landed on the positive side: it simply did
    not look like the one negative texture. v2 takes random-scale sub-crops so
    the negative class spans a range of effective fields of view that overlaps
    the positives' instead of sitting at one point.

(e) OPTIONAL FOURTH SOURCE: pad1tya/himawari9-b13-thermal-ir, RAW KELVIN.
    train_kernel_gates_audit/audit.py already confirmed against the real mount
    that these are .npz files with a float32 'matrix' of shape (5500, 5500),
    values 185.5-317.0 K -- genuine brightness temperature, so the project's
    own kelvin_to_display_u8() applies unchanged. That makes them same-pipeline
    negatives with real non-cyclone weather (94 frames each of
    calm_clear_sky_baseline, pre_monsoon_convective_outbreak and
    southwest_monsoon_onset_surge; the fourth label,
    rotational_vortex_bob01_cyclone, is a real storm and is hard-excluded).

    v1 scoped this out because cropping a 5500x5500 full disk needs
    geostationary projection maths and "getting that wrong risks silently
    including an unrelated typhoon elsewhere on the same disk". Both halves of
    that objection are answered here rather than dodged:
      - The projection is implemented (standard LRIT/HRIT forward transform,
        AHI 2 km grid) and SELF-TESTED at runtime by geos_self_test(). If any
        assertion fails the source is disabled instead of silently producing
        mislocated crops.
      - The typhoon-contamination risk is handled by requiring a real
        best-track cross-check, NOT by assuming. Tiles are excluded if any
        IBTrACS track point within +/-24 h falls within 10 deg of the tile
        centre. If a usable IBTrACS covering the frame dates cannot be
        obtained, the ENTIRE himawari source auto-disables and the reason is
        recorded in metrics.json. It can never ship unverified.

    Note, found while building this: the local archive at
    cgp_scratch/ibtracs_check/ibtracs.since1980.csv is TRUNCATED -- its final
    line carries 120 of 174 columns and has no trailing newline, i.e. a cut-off
    download, not a real end of record. v1's docstring claim that "the archive
    itself only covers through 2022-02-18" is therefore wrong; the DOWNLOAD
    stopped there, the real IBTrACS v04r01 since-1980 product runs to the
    present. load_ibtracs() below explicitly rejects a truncated file for that
    reason. That also means v1's MC_SAFE_END is more conservative than it needs
    to be; extending it is left as a separate, verifiable follow-up rather than
    silently widened here.

(f) METRICS THAT CAN DETECT THE ORIGINAL FAILURE.
    - val_metrics_by_source now produces REAL AUCs for digital_typhoon and
      tcir, because those sources now contain both classes.
    - headline_within_source_auc: AUC over DT+TCIR val frames only. Immune by
      construction to any cross-pipeline shortcut. THIS is the number to quote.
    - source_prior_auc: the AUC obtainable from source identity ALONE (each
      frame scored by its source's positive rate). On v1's dataset this would
      be ~1.0; here it should sit near 0.5 for the DT/TCIR-only subset. It is
      the direct quantitative answer to "how much of the headline number is
      buyable without looking at the image", in the same spirit as the v5
      climatology baseline.
    - reliability_table / ECE, and a specificity-targeted threshold table so
      the app's Gate 2 operating point can be re-derived from THIS checkpoint
      rather than reusing 0.5 blindly.

(g) CLASS BALANCE. v1 ran 58,870 : 4,665 (12.6:1). v2 rebalances by
    even-subsampling the majority class within each source to MAX_IMBALANCE,
    and additionally passes pos_weight to BCE for whatever residual remains.

Checkpoint format is unchanged (plain resnet18, 1-channel conv1, single-logit
fc, saved under "state_dict"), so local_tool/detect.py, the web app and
train_kernel_needle_search/search.py load it with no code changes.

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
MODEL_VERSION = "gate2_cyclonecheck_v2"

INPUT_ROOT = os.environ.get("CGP_INPUT_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("CGP_OUT_DIR", "/kaggle/working")

TEMP_LO, TEMP_HI = 190.0, 300.0
IMG_SIZE = 224
NORM_MEAN, NORM_STD = 0.449, 0.226
MC_BT_SCALE, MC_BT_OFFSET = 150.0, 180.0

# ---------------------------------------------------------------- stage rules
# Identical `first_cross` definition to train_kernel_needle_search/search.py
# and train_kernel_v5_final -- Gate 2's boundary and Model 3's input domain are
# the same boundary on purpose (see docstring (a)).
DT_EARLY_GRADE_MAX = 2          # JMA grade <= 2 is tropical depression or less
TCIR_EARLY_VMAX_KT = 33.0
NEG_MAX_WIND_KT = 25.0          # a negative must be genuinely weak
POS_MIN_WIND_KT = 35.0          # a positive must be at least a tropical storm
DT_POS_GRADES = (3, 4, 5)       # TS / severe TS / typhoon. 6+ is extratropical.
DT_EXTRATROPICAL_GRADE_MIN = 6
EXTRATROPICAL_MODE = os.environ.get("CGP_EXTRATROPICAL_MODE", "drop")  # drop|negative
MIN_PRE_FRAMES = 3              # storms with fewer pre-cross frames give no negatives

MAX_POS_PER_STORM = int(os.environ.get("CGP_POS_PER_STORM", "8"))
MAX_NEG_PER_STORM = int(os.environ.get("CGP_NEG_PER_STORM", "8"))
MAX_DIAG_PER_STORM = int(os.environ.get("CGP_DIAG_PER_STORM", "3"))
MAX_FRAMES_PER_DAY_MC = int(os.environ.get("CGP_FRAMES_PER_DAY_MC", "2"))
PRELOAD_BUDGET_GB = float(os.environ.get("CGP_PRELOAD_BUDGET_GB", "5.0"))
MAX_IMBALANCE = float(os.environ.get("CGP_MAX_IMBALANCE", "1.25"))

VAL_FRACTION = 0.2
EPOCHS = int(os.environ.get("CGP_EPOCHS", "14"))
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
# maritime-continent: VERIFIED against the real IBTrACS since-1980 archive by
# the v1 pass (see v1 docstring). Carried over verbatim -- these 15 windows are
# real storms found in this exact box, not estimates. MC_SAFE_END is limited by
# the archive copy that pass used, which is now known to have been a truncated
# download; widening it needs a fresh verified archive, so it is left alone.
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
# Random-scale sub-crop range for MC, as a fraction of the frame's short side.
MC_CROP_FRAC_RANGE = (0.40, 1.00)

# ----------------------------------------------------------------------------
# himawari9-b13 full-disk source (optional, auto-disables -- see docstring (e))
# ----------------------------------------------------------------------------
USE_HIMAWARI = os.environ.get("CGP_USE_HIMAWARI", "1") not in ("0", "false", "False")
HIMAWARI_CYCLONE_LABELS = ("rotational_vortex_bob01_cyclone",)  # hard-excluded
HIMAWARI_FILES_PER_LABEL = int(os.environ.get("CGP_HW_FILES_PER_LABEL", "30"))
HIMAWARI_TILES_PER_FILE = int(os.environ.get("CGP_HW_TILES_PER_FILE", "20"))
HIMAWARI_TILE_SIZES = (256, 384, 512, 768, 1024)   # 2 km/px -> 512..2048 km
HIMAWARI_MAX_GC_DEG = 55.0        # keep viewing zenith sane, avoid limb smear
HIMAWARI_BT_VALID = (150.0, 350.0)
IBTRACS_EXCLUDE_DEG = 10.0        # tile centre must be this far from any track point
IBTRACS_EXCLUDE_HOURS = 24.0
IBTRACS_URL = ("https://www.ncei.noaa.gov/data/"
               "international-best-track-archive-for-climate-stewardship-ibtracs/"
               "v04r01/access/csv/ibtracs.since1980.list.v04r01.csv")

# AHI full-disk 2 km fixed grid (LRIT/HRIT Global Specification constants).
GEOS_SUB_LON = 140.7
GEOS_H_KM = 42164.0
GEOS_R_EQ = 6378.137
GEOS_R_POL = 6356.7523
GEOS_COFF = GEOS_LOFF = 2750.5
GEOS_CFAC = GEOS_LFAC = 20466275.0
AHI_FULL_DISK_N = 5500

INDEX_COLUMNS = ["group_key", "cls", "source", "sub_source", "storm_id", "h5_path",
                 "frame_ref", "crop_size", "crop_r0", "crop_c0", "tile_lat", "tile_lon",
                 "stage_note"]


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


def as_index_frame(rows):
    """Every builder returns the same canonical schema, so pd.concat can never
    silently misalign columns or invent NaN-typed ones."""
    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    if df.empty:
        return df
    for c, default in (("sub_source", ""), ("storm_id", ""), ("h5_path", ""),
                       ("stage_note", "")):
        df[c] = df[c].fillna(default).astype(str)
    for c in ("frame_ref", "crop_size", "crop_r0", "crop_c0"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(np.int64)
    for c in ("tile_lat", "tile_lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
    df["cls"] = pd.to_numeric(df["cls"], errors="coerce").astype(np.int64)
    return df


# ----------------------------------------------------------------------------
# Dataset discovery -- verbatim from the proven code in gate1/v5_final.
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
# STAGED indices: the actual fix. Same storms supply both classes.
# ----------------------------------------------------------------------------
def build_dt_index_staged(meta_dir, image_dir):
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
        if {"file_1", "grade", "wind", "year"} - set(df.columns):
            audit["n_bad_columns"] += 1
            continue
        if int(storm_id[:4]) < 1977:
            continue
        grade = pd.to_numeric(df["grade"], errors="coerce").to_numpy()
        wind = pd.to_numeric(df["wind"], errors="coerce").fillna(0.0).to_numpy()
        fname = df["file_1"].fillna("").astype(str).str.strip().to_numpy()
        has_img = fname != ""

        crossed = np.nan_to_num(grade, nan=0.0) > DT_EARLY_GRADE_MAX
        first_cross = int(np.argmax(crossed)) if crossed.any() else len(df)
        idx = np.arange(len(df))

        # NEGATIVE: strictly pre-organization. Truncating at first_cross also
        # inherits the v5 Finding-2 fix -- it can never pick up a post-peak
        # decay-tail frame that happens to be graded low again.
        neg_ok = has_img & (idx < first_cross) & (wind <= NEG_MAX_WIND_KT)
        neg_idx = idx[neg_ok]
        if first_cross < MIN_PRE_FRAMES:
            neg_idx = neg_idx[:0]
            audit["n_storms_no_pre_stage"] += 1

        # POSITIVE: organized AND still tropical (grade 3/4/5 only).
        in_pos_grade = np.isin(np.nan_to_num(grade, nan=-1.0).astype(int), DT_POS_GRADES)
        pos_ok = has_img & (idx >= first_cross) & in_pos_grade & (wind >= POS_MIN_WIND_KT)
        pos_idx = idx[pos_ok]

        # DIAGNOSTIC: extratropical / post-transition. Never a positive.
        ex_ok = has_img & (np.nan_to_num(grade, nan=-1.0) >= DT_EXTRATROPICAL_GRADE_MIN)
        ex_idx = idx[ex_ok]

        audit["n_deadband_dropped"] += int(has_img.sum() - len(neg_idx) - len(pos_idx) - len(ex_idx))

        def emit(sel, cap, cls, note):
            for p in sel[even_picks(len(sel), cap)]:
                h5_path = os.path.join(image_dir, storm_id, fname[int(p)])
                if not os.path.exists(h5_path):
                    audit["n_missing_h5"] += 1
                    continue
                rows.append({"group_key": f"dt_{storm_id}", "cls": cls,
                             "source": "digital_typhoon", "sub_source": "wp",
                             "storm_id": storm_id, "h5_path": h5_path, "frame_ref": -1,
                             "crop_size": -1, "crop_r0": -1, "crop_c0": -1,
                             "tile_lat": np.nan, "tile_lon": np.nan, "stage_note": note})

        emit(neg_idx, MAX_NEG_PER_STORM, 0, "pre_organization")
        emit(pos_idx, MAX_POS_PER_STORM, 1, "organized_tropical")
        if EXTRATROPICAL_MODE == "negative":
            emit(ex_idx, MAX_DIAG_PER_STORM, 0, "extratropical_as_negative")
        else:
            emit(ex_idx, MAX_DIAG_PER_STORM, -1, "extratropical_diagnostic")

    df = as_index_frame(rows)
    n_by = df["cls"].value_counts().to_dict() if len(df) else {}
    print(f"[DT staged] storms={len(csvs)} frames={len(df)} by_cls={n_by} audit={dict(audit)}")
    return df, dict(audit)


def build_tcir_index_staged(info):
    vmax = pd.to_numeric(info["Vmax"], errors="coerce").to_numpy()
    region = info["region"].astype(str).to_numpy() if "region" in info.columns else np.array([""] * len(info))
    work = pd.DataFrame({"frame_ref": np.arange(len(info), dtype=np.int64),
                         "storm_id": info["ID"].astype(str).to_numpy(),
                         "vmax": vmax, "region": region})
    rows = []
    audit = Counter()
    for storm_id, g in work.groupby("storm_id", sort=True):
        g = g.sort_values("frame_ref")
        vm = g["vmax"].to_numpy()
        refs = g["frame_ref"].to_numpy()
        basin = str(g["region"].iloc[0])
        crossed = np.isfinite(vm) & (vm > TCIR_EARLY_VMAX_KT)
        first_cross = int(np.argmax(crossed)) if crossed.any() else len(g)
        idx = np.arange(len(g))

        neg_idx = idx[(idx < first_cross) & np.isfinite(vm) & (vm <= NEG_MAX_WIND_KT)]
        if first_cross < MIN_PRE_FRAMES:
            neg_idx = neg_idx[:0]
            audit["n_storms_no_pre_stage"] += 1
        pos_idx = idx[(idx >= first_cross) & np.isfinite(vm) & (vm >= POS_MIN_WIND_KT)]
        audit["n_deadband_dropped"] += int(len(g) - len(neg_idx) - len(pos_idx))

        def emit(sel, cap, cls, note):
            for p in sel[even_picks(len(sel), cap)]:
                rows.append({"group_key": f"tcir_{storm_id}", "cls": cls, "source": "tcir",
                             "sub_source": basin, "storm_id": storm_id, "h5_path": "",
                             "frame_ref": int(refs[int(p)]), "crop_size": -1,
                             "crop_r0": -1, "crop_c0": -1, "tile_lat": np.nan,
                             "tile_lon": np.nan, "stage_note": note})

        emit(neg_idx, MAX_NEG_PER_STORM, 0, "pre_organization")
        emit(pos_idx, MAX_POS_PER_STORM, 1, "organized_tropical")

    df = as_index_frame(rows)
    n_by = df["cls"].value_counts().to_dict() if len(df) else {}
    print(f"[TCIR staged] storms={work['storm_id'].nunique()} frames={len(df)} "
          f"by_cls={n_by} audit={dict(audit)}")
    return df, dict(audit)


def build_maritime_continent_negative_index(mc_root, max_per_day=MAX_FRAMES_PER_DAY_MC):
    """Real IR frames from an IBTrACS-verified cyclone-free window. Unlike v1,
    each row carries a deterministic random-scale sub-crop so this source spans
    a range of fields of view instead of a single fixed one."""
    csvs = [f for f in os.listdir(mc_root) if f.endswith(".csv") and f != "regions.csv"]
    if not csvs:
        return as_index_frame([]), {}
    df = pd.read_csv(os.path.join(mc_root, sorted(csvs)[0]))
    df = df[df["has_image"] == True].copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df[(df["time"] >= MC_SAFE_START) & (df["time"] <= MC_SAFE_END)]

    excluded = pd.Series(False, index=df.index)
    for a, b in MC_EXCLUDE_WINDOWS:
        excluded |= (df["time"] >= pd.Timestamp(a)) & (df["time"] <= pd.Timestamp(b))
    audit = {"rows_in_safe_range": int(len(df)),
             "rows_excluded_by_storm_window": int(excluded.sum())}
    df = df[~excluded]

    rng = np.random.RandomState(SEED + 101)
    rows = []
    for date, g in df.groupby("date", sort=True):
        idx = g.index.to_numpy()
        for i in idx[even_picks(len(idx), max_per_day)]:
            frac = rng.uniform(*MC_CROP_FRAC_RANGE)
            rows.append({"group_key": f"mc_{date}", "cls": 0, "source": "maritime_continent",
                         "sub_source": "himawari8_9_b13_png", "storm_id": "",
                         "h5_path": os.path.join(mc_root, str(df.loc[i, "image_path"])),
                         "frame_ref": -1, "crop_size": int(round(frac * 10000)),
                         "crop_r0": int(rng.randint(0, 10001)),
                         "crop_c0": int(rng.randint(0, 10001)),
                         "tile_lat": np.nan, "tile_lon": np.nan,
                         "stage_note": "verified_cyclone_free_window"})
    audit["rows_kept_after_daily_cap"] = len(rows)
    print(f"[MC negatives] {audit}")
    return as_index_frame(rows), audit


# ----------------------------------------------------------------------------
# Geostationary projection for the Himawari full disks. Standard LRIT/HRIT
# forward transform. Verified numerically offline AND asserted at runtime.
# ----------------------------------------------------------------------------
def latlon_to_ahi_pixel(lat_deg, lon_deg):
    """Geodetic lat/lon -> (column, line) on the AHI 2 km full-disk grid, plus a
    visibility mask. Column increases east, line increases south."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    sub = math.radians(GEOS_SUB_LON)
    ratio = (GEOS_R_POL ** 2) / (GEOS_R_EQ ** 2)
    e2 = 1.0 - ratio
    c_lat = np.arctan(ratio * np.tan(lat))
    r_l = GEOS_R_POL / np.sqrt(1.0 - e2 * np.cos(c_lat) ** 2)
    dlon = lon - sub
    px = r_l * np.cos(c_lat) * np.cos(dlon)
    py = r_l * np.cos(c_lat) * np.sin(dlon)
    pz = r_l * np.sin(c_lat)
    # Visible iff the surface point faces the satellite: dot(P, S - P) > 0.
    visible = (px * GEOS_H_KM) > (r_l * r_l)
    r1 = GEOS_H_KM - px
    rn = np.sqrt(r1 * r1 + py * py + pz * pz)
    x = np.degrees(np.arctan2(py, r1))
    y = np.degrees(np.arcsin(-pz / rn))
    col = GEOS_COFF + x * GEOS_CFAC / 65536.0
    line = GEOS_LOFF + y * GEOS_LFAC / 65536.0
    return col, line, visible


def geos_self_test():
    """If ANY of this fails, the himawari source is disabled rather than
    producing silently mislocated crops. This is the check v1 said it was not
    willing to guess at."""
    problems = []
    c, l, v = latlon_to_ahi_pixel(0.0, GEOS_SUB_LON)
    if not (abs(float(c) - GEOS_COFF) < 1e-6 and abs(float(l) - GEOS_LOFF) < 1e-6 and bool(v)):
        problems.append(f"sub-satellite point did not map to disk centre: {c},{l},{v}")
    ce, _, _ = latlon_to_ahi_pixel(0.0, GEOS_SUB_LON + 5.0)
    cw, _, _ = latlon_to_ahi_pixel(0.0, GEOS_SUB_LON - 5.0)
    if not (float(ce) > GEOS_COFF > float(cw)):
        problems.append("column does not increase eastward")
    _, ln, _ = latlon_to_ahi_pixel(10.0, GEOS_SUB_LON)
    _, ls, _ = latlon_to_ahi_pixel(-10.0, GEOS_SUB_LON)
    if not (float(ln) < GEOS_LOFF < float(ls)):
        problems.append("line does not increase southward")
    lons = np.arange(GEOS_SUB_LON - 60.0, GEOS_SUB_LON + 60.0, 5.0)
    cc, ll, vv = latlon_to_ahi_pixel(np.zeros_like(lons), lons)
    if not bool(np.all(vv)) or not bool(np.all(np.diff(cc) > 0)):
        problems.append("equatorial sweep not visible/monotonic")
    if not bool(np.all((cc >= 0) & (cc < AHI_FULL_DISK_N))):
        problems.append("visible points fall outside the 5500 px grid")
    _, _, v_far = latlon_to_ahi_pixel(0.0, GEOS_SUB_LON + 85.0)
    if bool(v_far):
        problems.append("point beyond the limb reported visible")
    for p in problems:
        print(f"  !! GEOS SELF-TEST FAILURE: {p}")
    print(f"[geos self-test] {'PASS' if not problems else 'FAIL'} "
          f"(sub-sat -> {float(c):.1f},{float(l):.1f}; expect {GEOS_COFF},{GEOS_LOFF})")
    return not problems


def himawari_grid_points():
    """Tile centres over the well-viewed part of the disk. The Bay of Bengal
    (the labels' nominal region) lands near column 480 of 5500 -- extreme limb,
    heavily foreshortened -- so it is deliberately NOT sampled: limb smear
    would itself become a source-identifying texture cue, which is the exact
    class of bug this rewrite exists to remove."""
    pts = []
    for lat in np.arange(-40.0, 40.01, 8.0):
        for lon in np.arange(GEOS_SUB_LON - 50.0, GEOS_SUB_LON + 50.01, 8.0):
            gc = math.degrees(math.acos(max(-1.0, min(1.0,
                math.cos(math.radians(lat)) * math.cos(math.radians(lon - GEOS_SUB_LON))))))
            if gc <= HIMAWARI_MAX_GC_DEG:
                pts.append((round(float(lat), 2), round(float(lon), 2)))
    return pts


def load_ibtracs():
    """Return (DataFrame[time, lat, lon], note). Refuses a truncated file --
    the local cgp_scratch copy was truncated mid-row, which is exactly how v1
    came to believe the archive ended in Feb 2022."""
    path = None
    for dirpath, dirnames, filenames in os.walk(INPUT_ROOT):
        # Never descend into the Digital Typhoon per-storm trees (thousands of
        # directories, no IBTrACS in them) -- same pruning discipline as
        # find_dt_root/find_tcir_h5.
        dirnames[:] = [d for d in dirnames if d.lower() not in ("image", "metadata")]
        for fn in filenames:
            if "ibtracs" in fn.lower() and fn.lower().endswith(".csv"):
                path = os.path.join(dirpath, fn)
                break
        if path:
            break
    note = ""
    if path is None:
        try:
            import tempfile
            import urllib.request
            # /tmp, not OUT_DIR -- a ~200 MB CSV does not belong in the kernel's
            # published output.
            path = os.path.join(tempfile.gettempdir(), "_ibtracs_since1980.csv")
            print(f"[ibtracs] no mount found; downloading {IBTRACS_URL}")
            urllib.request.urlretrieve(IBTRACS_URL, path)
            note = "downloaded_at_runtime"
        except Exception as exc:
            return None, f"unavailable: {exc}"
    else:
        note = f"mounted: {path}"

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            ncol = header.count(",") + 1
            last = ""
            for line in f:
                last = line
        if (last.count(",") + 1) != ncol or not last.endswith("\n"):
            return None, (f"TRUNCATED file ({path}): final row has "
                          f"{last.count(',') + 1}/{ncol} columns, "
                          f"trailing_newline={last.endswith(chr(10))}")
        df = pd.read_csv(path, usecols=["ISO_TIME", "LAT", "LON"], skiprows=[1],
                         low_memory=False)
        df["time"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
        df["lat"] = pd.to_numeric(df["LAT"], errors="coerce")
        df["lon"] = pd.to_numeric(df["LON"], errors="coerce") % 360.0
        df = df.dropna(subset=["time", "lat", "lon"])[["time", "lat", "lon"]]
        return df, f"{note}; rows={len(df)}; covers {df['time'].min()} .. {df['time'].max()}"
    except Exception as exc:
        return None, f"unreadable ({path}): {exc}"


def build_himawari_negative_index(hw_root):
    """Geolocated tiles from non-cyclone-labelled full disks, every tile
    cross-checked against real best-track data. Returns (index, audit). An
    empty index with a stated reason is a normal, expected outcome."""
    audit = {"enabled": True}
    meta_path = None
    for dirpath, _, filenames in os.walk(hw_root):
        for fn in filenames:
            if fn.lower() == "metadata.csv":
                meta_path = os.path.join(dirpath, fn)
    if meta_path is None:
        audit.update(enabled=False, reason="metadata.csv not found")
        return as_index_frame([]), audit
    npz_dir = os.path.dirname(meta_path)

    if not geos_self_test():
        audit.update(enabled=False, reason="geostationary projection self-test FAILED")
        return as_index_frame([]), audit

    meta = pd.read_csv(meta_path)
    if {"filename", "label", "scan_timestamp"} - set(meta.columns):
        audit.update(enabled=False, reason=f"unexpected metadata columns: {list(meta.columns)}")
        return as_index_frame([]), audit
    meta["time"] = pd.to_datetime(meta["scan_timestamp"], errors="coerce")
    meta = meta.dropna(subset=["time"])
    audit["label_counts_all"] = meta["label"].value_counts().to_dict()
    meta = meta[~meta["label"].isin(HIMAWARI_CYCLONE_LABELS)].copy()
    audit["labels_used"] = sorted(meta["label"].unique().tolist())
    if meta.empty:
        audit.update(enabled=False, reason="no non-cyclone-labelled frames")
        return as_index_frame([]), audit

    tracks, ib_note = load_ibtracs()
    audit["ibtracs"] = ib_note
    if tracks is None:
        audit.update(enabled=False,
                     reason=f"no usable IBTrACS, cannot verify these dates are storm-free ({ib_note})")
        print(f"  !! HIMAWARI DISABLED: {audit['reason']}")
        return as_index_frame([]), audit
    t_lo, t_hi = meta["time"].min(), meta["time"].max()
    if tracks["time"].max() < t_hi or tracks["time"].min() > t_lo:
        audit.update(enabled=False,
                     reason=(f"IBTrACS covers {tracks['time'].min()}..{tracks['time'].max()} "
                             f"but frames span {t_lo}..{t_hi} -- cannot verify"))
        print(f"  !! HIMAWARI DISABLED: {audit['reason']}")
        return as_index_frame([]), audit

    points = himawari_grid_points()
    print(f"[himawari] {len(points)} candidate tile centres within "
          f"{HIMAWARI_MAX_GC_DEG} deg of the sub-satellite point")
    rng = np.random.RandomState(SEED + 202)
    rows = []
    n_excluded = 0
    for label, g in meta.groupby("label", sort=True):
        g = g.sort_values("time").reset_index(drop=True)
        pick = even_picks(len(g), HIMAWARI_FILES_PER_LABEL)
        for fi, ri in enumerate(pick):
            rec = g.iloc[int(ri)]
            fpath = os.path.join(npz_dir, str(rec["filename"]))
            if not os.path.exists(fpath):
                continue
            win = tracks[(tracks["time"] >= rec["time"] - pd.Timedelta(hours=IBTRACS_EXCLUDE_HOURS)) &
                         (tracks["time"] <= rec["time"] + pd.Timedelta(hours=IBTRACS_EXCLUDE_HOURS))]
            tlat = win["lat"].to_numpy()
            tlon = win["lon"].to_numpy()
            # rotate the tile-centre list per file so different files sample
            # different places -- decorrelates an otherwise near-identical set
            order = (np.arange(len(points)) + fi * 7) % len(points)
            taken = 0
            for pi in order:
                if taken >= HIMAWARI_TILES_PER_FILE:
                    break
                lat, lon = points[int(pi)]
                if len(tlat):
                    dlon = np.abs(((tlon - (lon % 360.0)) + 180.0) % 360.0 - 180.0)
                    if bool(np.any((np.abs(tlat - lat) <= IBTRACS_EXCLUDE_DEG) &
                                   (dlon <= IBTRACS_EXCLUDE_DEG))):
                        n_excluded += 1
                        continue
                col, line, vis = latlon_to_ahi_pixel(lat, lon)
                if not bool(vis):
                    continue
                size = int(HIMAWARI_TILE_SIZES[rng.randint(len(HIMAWARI_TILE_SIZES))])
                r0 = int(round(float(line))) - size // 2
                c0 = int(round(float(col))) - size // 2
                if r0 < 0 or c0 < 0 or r0 + size > AHI_FULL_DISK_N or c0 + size > AHI_FULL_DISK_N:
                    continue
                rows.append({"group_key": f"hw_{label}_{lat}_{lon}", "cls": 0,
                             "source": "himawari_fulldisk", "sub_source": label,
                             "storm_id": "", "h5_path": fpath, "frame_ref": -1,
                             "crop_size": size, "crop_r0": r0, "crop_c0": c0,
                             "tile_lat": lat, "tile_lon": lon,
                             "stage_note": "ibtracs_verified_tile"})
                taken += 1
    audit["tiles_excluded_by_ibtracs"] = int(n_excluded)
    audit["tiles_kept"] = len(rows)
    print(f"[himawari] {audit}")
    return as_index_frame(rows), audit


# ----------------------------------------------------------------------------
# Image loading -- ONE Kelvin->display conversion for every source.
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
    if np.isfinite(arr).mean() < 0.5:
        raise ValueError("TCIR IR1 mostly non-finite")
    arr = np.nan_to_num(arr, nan=TEMP_HI, posinf=TEMP_HI, neginf=TEMP_LO)
    return kelvin_to_display_u8(arr)


def load_mc_frame_u8(png_path, frac_x10000, r_seed, c_seed):
    """Same Kelvin conversion as v1, plus a deterministic random-scale sub-crop
    so this source no longer sits at exactly one field of view."""
    from PIL import Image
    arr = np.asarray(Image.open(png_path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    bt = arr / 255.0 * MC_BT_SCALE + MC_BT_OFFSET
    if frac_x10000 > 0:
        frac = min(1.0, max(0.05, frac_x10000 / 10000.0))
        side = int(round(frac * min(bt.shape)))
        side = max(32, min(side, min(bt.shape)))
        r0 = int((r_seed / 10000.0) * (bt.shape[0] - side)) if bt.shape[0] > side else 0
        c0 = int((c_seed / 10000.0) * (bt.shape[1] - side)) if bt.shape[1] > side else 0
        bt = bt[r0:r0 + side, c0:c0 + side]
    return kelvin_to_display_u8(bt)


def load_himawari_tile_u8(matrix, r0, c0, size):
    tile = np.asarray(matrix[r0:r0 + size, c0:c0 + size], dtype=np.float32)
    if tile.shape != (size, size):
        raise ValueError(f"tile shape {tile.shape} != {(size, size)}")
    if not np.isfinite(tile).all():
        raise ValueError("tile contains off-disk / non-finite pixels")
    lo, hi = HIMAWARI_BT_VALID
    if float(tile.min()) < lo or float(tile.max()) > hi:
        raise ValueError(f"tile BT out of physical range [{tile.min():.1f},{tile.max():.1f}]")
    return kelvin_to_display_u8(tile)


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


# NOTE: no horizontal flip -- these ARE cyclone frames on the positive side and
# a mirrored storm reverses its hemisphere-determined rotational chirality
# (same reasoning as every other storm model in this project). Zoom range is
# WIDER than v1's 0.12: field of view must not be a usable class cue.
AUG = {"brightness": 0.12, "contrast": 0.25, "zoom_log": 0.22, "shift": 0.12, "noise_std": 0.015}


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
        return torch.from_numpy(np.ascontiguousarray(img)), torch.tensor(self.labels[i], dtype=torch.float32)


def preload_all(index):
    n = len(index)
    print(f"\nAllocating preload buffer: {n} x {IMG_SIZE} x {IMG_SIZE} uint8 = "
          f"{n * IMG_SIZE * IMG_SIZE / 1e9:.2f} GB")
    out = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    ok = np.ones(n, dtype=bool)
    sources = index["source"].to_numpy()
    paths = index["h5_path"].to_numpy()
    refs = index["frame_ref"].to_numpy()
    csize = index["crop_size"].to_numpy()
    cr0 = index["crop_r0"].to_numpy()
    cc0 = index["crop_c0"].to_numpy()

    tcir_path = find_tcir_h5()
    tc_pos = np.where(sources == "tcir")[0]
    if len(tc_pos) and tcir_path:
        import h5py
        tc_pos = tc_pos[np.argsort(refs[tc_pos], kind="stable")]
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
            out[pos] = resize_u8(load_mc_frame_u8(paths[pos], int(csize[pos]),
                                                  int(cr0[pos]), int(cc0[pos])))
        except Exception as exc:
            print(f"  LOAD FAIL {paths[pos]}: {exc}")
            ok[pos] = False
        if (j + 1) % 5000 == 0:
            print(f"  [MC] {j + 1}/{len(mc_pos)} ({time.time() - t0:.0f}s)")
    if len(mc_pos):
        print(f"[MC] preload done in {time.time() - t0:.0f}s")

    # Himawari: group by file so each 121 MB full disk is decompressed once.
    hw_pos = np.where(sources == "himawari_fulldisk")[0]
    if len(hw_pos):
        t0 = time.time()
        by_file = {}
        for pos in hw_pos:
            by_file.setdefault(paths[pos], []).append(pos)
        for fi, (fpath, positions) in enumerate(sorted(by_file.items())):
            try:
                with np.load(fpath) as z:
                    matrix = np.asarray(z["matrix"], dtype=np.float32)
            except Exception as exc:
                print(f"  LOAD FAIL himawari {fpath}: {exc}")
                for pos in positions:
                    ok[pos] = False
                continue
            for pos in positions:
                try:
                    out[pos] = resize_u8(load_himawari_tile_u8(
                        matrix, int(cr0[pos]), int(cc0[pos]), int(csize[pos])))
                except Exception as exc:
                    print(f"  TILE REJECT {os.path.basename(fpath)} "
                          f"r{cr0[pos]} c{cc0[pos]}: {exc}")
                    ok[pos] = False
            del matrix
            if (fi + 1) % 10 == 0:
                print(f"  [HW] {fi + 1}/{len(by_file)} files ({time.time() - t0:.0f}s)")
        print(f"[HW] preload done in {time.time() - t0:.0f}s")

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
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.non_float = {k: v.detach().clone() for k, v in model.state_dict().items()
                          if not v.dtype.is_floating_point}

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


def reliability_table(probs, labels, n_bins=10):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        k = int(m.sum())
        if k == 0:
            rows.append({"lo": lo, "hi": hi, "n": 0, "mean_prob": None, "frac_pos": None})
            continue
        mp, fp_ = float(probs[m].mean()), float(labels[m].mean())
        ece += (k / len(probs)) * abs(mp - fp_)
        rows.append({"lo": lo, "hi": hi, "n": k, "mean_prob": mp, "frac_pos": fp_})
    return {"bins": rows, "ece": float(ece)}


def threshold_table(probs, labels, targets=(0.80, 0.90, 0.95, 0.99)):
    """Operating points by target SPECIFICITY on the negative class, so the
    app's Gate 2 threshold can be re-derived from this checkpoint instead of
    reusing 0.5 blindly."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    neg = probs[labels == 0]
    out = {}
    if len(neg) == 0:
        return out
    for t in targets:
        thr = float(np.quantile(neg, t))
        out[f"specificity_{t:.2f}"] = {"threshold": thr, **binary_metrics(probs, labels, thr)}
    return out


def source_prior_auc(probs, labels, sources):
    """How much AUC is buyable from SOURCE IDENTITY ALONE -- each frame scored
    by its source's own positive rate, no image involved. On v1's dataset this
    would be ~1.0 (which is the whole bug). Near 0.5 means the sources carry no
    label information and the model had to look at the picture."""
    labels = np.asarray(labels, dtype=np.float64)
    sources = np.asarray(sources)
    prior = np.zeros(len(labels), dtype=np.float64)
    for s in np.unique(sources):
        m = sources == s
        prior[m] = labels[m].mean()
    return binary_metrics(prior, labels)["auc"]


def rebalance_index(index, max_ratio=MAX_IMBALANCE):
    """Even-subsample the majority class WITHIN each source, so no source is
    wiped out and the surviving frames stay evenly spread over storms/days."""
    lab = index["cls"].to_numpy()
    n_pos, n_neg = int((lab == 1).sum()), int((lab == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return index, {"skipped": "one class empty"}
    ratio = max(n_pos, n_neg) / min(n_pos, n_neg)
    if ratio <= max_ratio:
        return index, {"n_pos": n_pos, "n_neg": n_neg, "ratio": ratio, "action": "none"}
    major = 1 if n_pos > n_neg else 0
    target = int(round(min(n_pos, n_neg) * max_ratio))
    keep_major = []
    maj = index[index["cls"] == major]
    for _, g in maj.groupby("source", sort=True):
        share = max(1, int(round(target * len(g) / len(maj))))
        keep_major.append(g.iloc[even_picks(len(g), share)])
    out = pd.concat([index[index["cls"] != major]] + keep_major, ignore_index=True)
    audit = {"n_pos": n_pos, "n_neg": n_neg, "ratio": ratio, "action": "subsampled",
             "majority_class": major, "majority_target": target,
             "after": out["cls"].value_counts().to_dict()}
    print(f"[rebalance] {audit}")
    return out.reset_index(drop=True), audit


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
    print(f"Cyclone Genesis Predictor GATE 2 v2: IS A CYCLONE ACTUALLY ORGANIZED HERE -- {MODEL_VERSION}")
    print(f"  negatives now share the positives' pipeline: pre-organization frames")
    print(f"  from the SAME .h5 sources (DT grade<={DT_EARLY_GRADE_MAX} & wind<={NEG_MAX_WIND_KT}kt,")
    print(f"  TCIR Vmax<={NEG_MAX_WIND_KT}kt, both strictly before first_cross)")
    print(f"  deadband: {NEG_MAX_WIND_KT}-{POS_MIN_WIND_KT} kt dropped, not labelled")
    print(f"  extratropical (DT grade>={DT_EXTRATROPICAL_GRADE_MIN}) mode = {EXTRATROPICAL_MODE}")
    print("=" * 72)

    dt_root = find_dt_root()
    tcir_path = find_tcir_h5()
    mc_root = find_dataset_root("maritime-continent")
    hw_root = find_dataset_root("himawari9-b13") if USE_HIMAWARI else None
    print("DT root:", dt_root, "\nTCIR h5:", tcir_path, "\nMC root:", mc_root, "\nHW root:", hw_root)
    missing = [n for n, v in [("Digital Typhoon", dt_root), ("TCIR", tcir_path),
                               ("maritime-continent", mc_root)] if v is None]
    if missing:
        raise SystemExit(f"Missing dataset mounts: {missing} -- check dataset_sources.")

    parts, audits = [], {}
    info = pd.read_hdf(tcir_path, key="info", mode="r").reset_index(drop=True)
    p, a = build_tcir_index_staged(info)
    parts.append(p)
    audits["tcir"] = a
    meta_dir, image_dir = resolve_dt_dirs(dt_root)
    p, a = build_dt_index_staged(meta_dir, image_dir)
    parts.append(p)
    audits["digital_typhoon"] = a
    p, a = build_maritime_continent_negative_index(mc_root)
    parts.append(p)
    audits["maritime_continent"] = a
    if hw_root:
        p, a = build_himawari_negative_index(hw_root)
        parts.append(p)
        audits["himawari_fulldisk"] = a
    else:
        audits["himawari_fulldisk"] = {"enabled": False,
                                       "reason": "mount not found or CGP_USE_HIMAWARI=0"}

    index = pd.concat([x for x in parts if len(x)], ignore_index=True).reset_index(drop=True)
    print(f"\nClass balance (pre-rebalance): {index['cls'].value_counts().to_dict()}  "
          f"(-1 = extratropical diagnostic, not trained on)")
    print(f"By source x class:\n{pd.crosstab(index['source'], index['cls'])}")

    diag = index[index["cls"] == -1].reset_index(drop=True)
    index = index[index["cls"] >= 0].reset_index(drop=True)
    index, audits["rebalance"] = rebalance_index(index)

    bytes_per = IMG_SIZE * IMG_SIZE
    need_gb = (len(index) + len(diag)) * bytes_per / 1e9
    if need_gb > PRELOAD_BUDGET_GB:
        frac = PRELOAD_BUDGET_GB / need_gb
        # Explicit loop rather than groupby().apply() -- pandas 2.x deprecates
        # apply-over-grouping-columns and the semantics have shifted between
        # versions; this is version-proof and does exactly the same thing.
        kept = []
        for _, g in index.groupby(["source", "cls"], sort=True):
            kept.append(g.iloc[even_picks(len(g), max(1, int(len(g) * frac)))])
        index = pd.concat(kept, ignore_index=True).reset_index(drop=True)
        if len(diag):
            diag = diag.iloc[even_picks(len(diag), max(1, int(len(diag) * frac)))].reset_index(drop=True)
        print(f"!! Subsampled to fit {PRELOAD_BUDGET_GB} GB budget -> "
              f"{len(index)} train/val + {len(diag)} diagnostic frames")

    # Split at GROUP level. group_key is the STORM for DT/TCIR, so a storm that
    # supplies BOTH a negative (early) and a positive (mature) frame lands
    # entirely on one side -- no same-storm leakage across the split.
    groups = index["group_key"].unique()
    rng = np.random.RandomState(SEED)
    rng.shuffle(groups)
    n_val_groups = max(1, int(round(len(groups) * VAL_FRACTION)))
    val_groups = set(groups[:n_val_groups])
    index["split"] = np.where(index["group_key"].isin(val_groups), "val", "train")
    both = index.groupby("group_key")["cls"].nunique()
    print(f"\nGroups supplying BOTH classes (the hard, shortcut-proof ones): "
          f"{int((both > 1).sum())}/{len(both)}")

    diag["split"] = "diag"
    all_index = pd.concat([index, diag], ignore_index=True).reset_index(drop=True)
    images, ok = preload_all(all_index)
    all_index = all_index[ok].reset_index(drop=True)
    images = images[ok]
    all_index.to_csv(INDEX_PATH, index=False)

    is_diag = (all_index["split"] == "diag").to_numpy()
    is_val = (all_index["split"] == "val").to_numpy()
    labels = np.clip(all_index["cls"].to_numpy(np.float32), 0.0, 1.0)
    tr_idx = np.where(~is_val & ~is_diag)[0]
    va_idx = np.where(is_val)[0]
    dg_idx = np.where(is_diag)[0]
    print(f"\nFrames: train={len(tr_idx)} val={len(va_idx)} diagnostic={len(dg_idx)}")
    val_sources = all_index["source"].to_numpy()[va_idx]
    val_notes = all_index["stage_note"].to_numpy()[va_idx]

    n_tr_pos = float(labels[tr_idx].sum())
    n_tr_neg = float(len(tr_idx) - n_tr_pos)
    pos_weight_val = (n_tr_neg / max(1.0, n_tr_pos)) if n_tr_pos > 0 else 1.0
    print(f"Train class counts: pos={int(n_tr_pos)} neg={int(n_tr_neg)} "
          f"-> BCE pos_weight={pos_weight_val:.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(USE_AMP and device.type == "cuda")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    pos_weight = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)

    train_ds = FrameDataset(images[tr_idx], labels[tr_idx], augment=True)
    val_ds = FrameDataset(images[va_idx], labels[va_idx], augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    diag_dl = (DataLoader(FrameDataset(images[dg_idx], labels[dg_idx], augment=False),
                          batch_size=64, shuffle=False, num_workers=0) if len(dg_idx) else None)

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
                loss = F.binary_cross_entropy_with_logits(logit.float(), yt, pos_weight=pos_weight)
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
        ws = np.isin(val_sources, ["digital_typhoon", "tcir"])
        within = binary_metrics(vp[ws], vl[ws])["auc"] if ws.any() else float("nan")
        rec = dict(m)
        rec.update({"epoch": epoch, "train_loss": tot_l / max(1, n_batches),
                    "within_source_auc": float(within)})
        history.append(rec)
        print(f"epoch {epoch:2d}/{EPOCHS} loss={rec['train_loss']:.4f} val_auc={m['auc']:.4f} "
              f"within_src_auc={within:.4f} acc={m['accuracy']:.3f} f1={m['f1']:.3f} "
              f"({time.time() - t0:.0f}s)")
        # Select on the SHORTCUT-IMMUNE number, not the pooled one.
        sel = within if math.isfinite(within) else m["auc"]
        if math.isfinite(sel) and sel > best_auc:
            best_auc, best_epoch = sel, epoch
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

    ws = np.isin(val_sources, ["digital_typhoon", "tcir"])
    headline = binary_metrics(vp[ws], vl[ws]) if ws.any() else {}
    prior_all = source_prior_auc(vp, vl, val_sources)
    prior_ws = source_prior_auc(vp[ws], vl[ws], val_sources[ws]) if ws.any() else float("nan")

    diag_report = {"n": 0}
    if diag_dl is not None:
        dp, _ = evaluate(diag_dl)
        diag_report = {"n": int(len(dp)), "mean_prob": float(dp.mean()),
                       "median_prob": float(np.median(dp)),
                       "frac_above_0.5": float((dp >= 0.5).mean()),
                       "note": ("DT grade>=6 extratropical / post-transition frames, held out "
                                "of training. A high number here is the measured version of the "
                                "reported 'Japan bomb cyclone scored 99%' symptom.")}

    print("\n=== BEST (pooled, all sources) ===")
    print(json.dumps(jsonable(val_metrics), indent=2))
    print("\n=== HEADLINE: within-source AUC, DT+TCIR only (shortcut-immune) ===")
    print(json.dumps(jsonable(headline), indent=2))
    print(f"\nsource_prior_auc (AUC buyable from source identity alone): "
          f"all={prior_all:.4f} dt+tcir_only={prior_ws:.4f}")
    print("\nBy source:")
    for s, d in by_source.items():
        a = d["auc"]
        print(f"  {s:20s} n={d['n']:6d} auc={'n/a  ' if not math.isfinite(a) else f'{a:.4f}'} "
              f"acc={d['accuracy']:.3f}")
    if diag_report["n"]:
        print(f"\nExtratropical diagnostic set: n={diag_report['n']} "
              f"mean_prob={diag_report['mean_prob']:.3f} "
              f"frac>=0.5={diag_report['frac_above_0.5']:.3f}")

    by_stage = {}
    for s in sorted(set(val_notes.tolist())):
        m = val_notes == s
        by_stage[s] = {"n": int(m.sum()), "mean_prob": float(vp[m].mean()),
                       "frac_pos_label": float(vl[m].mean())}

    config = jsonable({
        "task": "cyclone_vs_non_cyclone_clouds", "temp_lo": TEMP_LO, "temp_hi": TEMP_HI,
        "img_size": IMG_SIZE, "norm_mean": NORM_MEAN, "norm_std": NORM_STD,
        "arch": "resnet18_1ch_binary", "model_version": MODEL_VERSION, "seed": SEED,
        "note": "positive = tracked-cyclone frames that have ORGANIZED (at/after first_cross, "
                "DT grade 3/4/5 or TCIR Vmax>=35kt). negative = the SAME sources' genuinely "
                "pre-organization frames (strictly before first_cross, <=25kt) plus "
                "IBTrACS-verified non-cyclone IR from maritime-continent and, when a usable "
                "best-track archive is available, geolocated Himawari-9 full-disk tiles. "
                "Every source passes through the identical kelvin_to_display_u8 conversion, "
                "so 'which pipeline produced this image' is not a usable shortcut. The "
                "25-35kt deadband is dropped rather than labelled.",
        "stage_rules": {"dt_early_grade_max": DT_EARLY_GRADE_MAX,
                        "tcir_early_vmax_kt": TCIR_EARLY_VMAX_KT,
                        "neg_max_wind_kt": NEG_MAX_WIND_KT,
                        "pos_min_wind_kt": POS_MIN_WIND_KT,
                        "dt_pos_grades": list(DT_POS_GRADES),
                        "extratropical_mode": EXTRATROPICAL_MODE},
        "mc_safe_window": [MC_SAFE_START, MC_SAFE_END],
        "mc_exclude_windows": MC_EXCLUDE_WINDOWS,
        "mc_crop_frac_range": list(MC_CROP_FRAC_RANGE),
        "source_audits": audits,
        "v1_defect": "v1 scored AUC 0.99995 with every per-source AUC null (each source was "
                     "single-class) and produced mean prob 0.995 / median 1.000 on 20,447 real "
                     "early-stage frames, making Model 3 unreachable for 99.6% of its own "
                     "intended inputs. See train_kernel_needle_search/output/needle_search_full.csv.",
    })
    payload = {
        "config": config, "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "headline_within_source_auc": headline,
        "source_prior_auc": {"all_sources": prior_all, "dt_tcir_only": prior_ws},
        "val_metrics_by_source": by_source,
        "val_mean_prob_by_stage": by_stage,
        "extratropical_diagnostic": diag_report,
        "reliability": reliability_table(vp, vl),
        "recommended_thresholds": threshold_table(vp, vl),
        "history": history,
        "class_balance": all_index[all_index["split"] != "diag"]["cls"].value_counts().to_dict(),
        "source_counts": all_index["source"].value_counts().to_dict(),
    }
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "arch": "resnet18_1ch_binary", "task": "cyclone_vs_non_cyclone_clouds",
                "pretrained_backbone": pretrained, "epoch": best_epoch, "config": config,
                "val_metrics": jsonable(val_metrics),
                "headline_within_source_auc": jsonable(headline),
                "recommended_thresholds": jsonable(payload["recommended_thresholds"]),
                "val_metrics_by_source": jsonable(by_source)},
               CKPT_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(jsonable(payload), f, indent=2)
    print("\nWrote", CKPT_PATH, METRICS_PATH, INDEX_PATH)
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
