"""
Cyclone Genesis Predictor -- Early-intensification prediction, COMBINED two-dataset version (v2).

TASK (identical to v1 -- this is a stronger model for the same problem, not a new
problem):
    Input : ONE early-stage infrared satellite frame (the system is still a weak,
            disorganised depression -- it does not yet look like a cyclone).
    Output: P(this storm will later reach typhoon / hurricane strength).
    Label : built from each storm's own FUTURE best-track history. The model never
            sees future frames, and never sees wind / pressure at inference time --
            only the raw brightness-temperature image.

WHAT IS NEW vs train_kernel/train.py
------------------------------------
1. TWO data sources are merged into one training set:
     A. NII Digital Typhoon (Western Pacific)  -- 512x512 IR, one .h5 per timestamp
        Kaggle: bwandowando/digital-typhoon-dataset-western-pacific-wp
     B. TCIR / TCSA                            -- 128x128x4, ONE 20GB TCSA.h5
        Kaggle: boyochen/tropical-cyclone-structure-analysis-tcsa
        Only channel 0 (IR1, Kelvin) is used, to stay 1-channel and therefore
        drop-in compatible with local_tool/detect.py.
   B adds five basins Digital Typhoon has no coverage of at all -- ATLN, CPAC,
   EPAC, IO, SH -- which is the whole point: IMD cares about the Indian Ocean,
   and v1 had literally zero IO storms in it.

2. CROSS-DATASET DE-DUPLICATION. Both datasets ultimately draw their Western
   Pacific storms from the same real best-track record, so the SAME physical
   storm can appear in both, under two different ID schemes. If that storm landed
   in A's train split and B's val split, val AUC would be inflated by
   memorisation. See dedup_wpac() -- this is the single most important correctness
   property of this script.

3. UNIFIED LABEL RULE: max sustained wind >= 64 kt, applied to Digital Typhoon's
   `wind` column and TCIR's `Vmax` column (both already in knots, no conversion).
   v1 used Digital Typhoon's `grade == 5` instead; TCIR has no grade field, so a
   wind threshold is the only rule expressible on both. 64 kt is each agency's own
   typhoon/hurricane threshold. See label_note() for the caveat.

Everything else -- resnet18 with a 1-channel conv1, mean-collapsed pretrained
weights, BCEWithLogitsLoss(pos_weight), AUC-selected best checkpoint, split by
STORM never by frame, the 190-300 K clip/invert window -- is carried over from v1
unchanged, because it was already reviewed and it already works.

The saved checkpoint keeps v1's exact key layout so local_tool/detect.py and
demo_kernel/render_demo.py both keep working against it with no edits.

Runs as a plain Kaggle *script* kernel with GPU enabled.
"""

import json
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

# Cap frames per storm, per source. Digital Typhoon alone averages ~45 early-stage
# imaged frames per storm; with ~2,300 combined storms, using them all would give
# a six-figure, hugely redundant training set. 4 evenly-spaced frames across the
# early stage keeps the run to a few minutes on a T4 while still covering the
# storm's whole pre-organisation life.
FRAMES_PER_STORM = 4

# ---- label rule (unified across both sources) ------------------------------
TYPHOON_WIND_KT = 64.0

# ---- "early stage" definition, per source ----------------------------------
# Digital Typhoon has a `grade` field; grade <= 2 == depression / pre-organisation.
DT_EARLY_GRADE_MAX = 2
# TCIR has NO grade field, so the early filter has to be expressed on Vmax.
# 33 kt is the standard tropical-depression / tropical-storm boundary, i.e. the
# system is still a depression -- weak and not yet organised. This is a DELIBERATE
# cross-dataset harmonisation choice standing in for DT's `grade <= 2`, not a
# constant handed down by the dataset. It is the closest available equivalent:
# JMA grade 2 (TD) is likewise defined as below tropical-storm strength.
TCIR_EARLY_VMAX_KT = 33.0

# ---- preprocessing (LOAD-BEARING -- do not change) -------------------------
# These exact values are baked into local_tool/detect.py, demo_kernel/render_demo.py
# and the v1 checkpoint. Changing any of them silently invalidates every existing
# deliverable.
TEMP_LO, TEMP_HI = 190.0, 300.0
IMG_SIZE = 224
NORM_MEAN, NORM_STD = 0.449, 0.226

# Written into the checkpoint config purely for compatibility: render_demo.py reads
# `typhoon_grade` to decide when to switch a frame to the CONFIRMED overlay using
# Digital Typhoon's own grade column. It plays no part in this script's labelling.
TYPHOON_GRADE = 5

# ---- dedup ------------------------------------------------------------------
# A TCIR WPAC storm is treated as the same physical storm as a Digital Typhoon
# storm if their best-track positions coincide at >= this many shared synoptic
# timestamps, within this many degrees. See dedup_wpac().
DEDUP_MATCH_DEG = 1.5
DEDUP_MIN_MATCH_POINTS = 4

# Whether the coarse (year, sequence_number) ID key counts towards the drop set,
# in addition to the best-track position match. See dedup_wpac() for why the ID
# key is coarse. Consequence of leaving this True: because Digital Typhoon numbers
# ~26 storms in every year TCIR covers, essentially EVERY TCIR WPAC storm finds an
# ID-key partner, so TCIR ends up contributing only the five basins Digital
# Typhoon has no data for at all (ATLN, CPAC, EPAC, IO, SH). That is a clean,
# explainable division of labour -- Digital Typhoon owns the Western Pacific at
# 512x512 over 1978-2023, TCIR supplies the basins it cannot see -- and it is the
# safe direction to err in. Set False to keep the handful of TCIR WPAC storms that
# only the position match rejects, at the cost of trusting that match completely.
DEDUP_USE_ID_KEY_UNION = True

# Reject the run if fewer than this fraction of TCIR WPAC IDs match the assumed
# YYYYNN<basin letter> shape -- a wrong assumption here would produce wrong dedup
# keys silently, which is exactly the failure mode we cannot tolerate.
TCIR_ID_PATTERN_MIN_MATCH = 0.90
TCIR_ID_RE = re.compile(r"^(\d{4})(\d{2})([A-Za-z])$")

# A TCIR frame with this much of its IR1 channel missing is dropped, not patched.
TCIR_MIN_FINITE_FRAC = 0.50

# ---- training ---------------------------------------------------------------
VAL_FRACTION = 0.2
EPOCHS = 8
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4

OUT_DIR = "/kaggle/working"
CKPT_PATH = os.path.join(OUT_DIR, "model_best.pt")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")
INDEX_PATH = os.path.join(OUT_DIR, "dataset_index.csv")


def label_note():
    return (
        "Positive == max sustained wind >= 64 kt at some point in the storm's life. "
        "CAVEAT: Digital Typhoon's `wind` is JMA 10-minute sustained; TCIR's `Vmax` "
        "is agency 1-minute sustained (JTWC/NHC). 64 kt is each agency's OWN "
        "typhoon/hurricane threshold, so the label reads as 'reached typhoon or "
        "hurricane strength as its own agency classified it' -- consistent as a "
        "category, slightly different as a physical wind speed (a 10-min 64 kt storm "
        "is roughly a 1-min 72 kt storm). This is the only rule expressible on both "
        "datasets; TCIR has no grade field."
    )


# ----------------------------------------------------------------------------
# Dataset discovery
# ----------------------------------------------------------------------------
def find_dt_root():
    """Locate the Digital Typhoon mount.

    The exact /kaggle/input prefix varies between kernel runs, so walk instead of
    hardcoding. Return as soon as a 'typhoon' directory is found -- the image tree
    is ~89GB and must never be walked exhaustively.
    """
    for dirpath, dirnames, _ in os.walk("/kaggle/input"):
        for d in dirnames:
            if "typhoon" in d.lower():
                return os.path.join(dirpath, d)
    return None


def resolve_dt_dirs(root):
    """This Kaggle mirror nests one level deeper than the upstream layout:
    <root>/metadata/metadata/*.csv and <root>/image/image/<storm_id>/*.h5.
    Tolerate both shapes."""
    meta = os.path.join(root, "metadata")
    if os.path.isdir(os.path.join(meta, "metadata")):
        meta = os.path.join(meta, "metadata")
    img = os.path.join(root, "image")
    if os.path.isdir(os.path.join(img, "image")):
        img = os.path.join(img, "image")
    return meta, img


def find_tcir_h5():
    """Locate TCSA.h5. It is one ~20GB file; prefer a filename that actually looks
    like TCSA/TCIR so we cannot accidentally pick up a stray .h5 from the Digital
    Typhoon mount (which contains tens of thousands of them)."""
    candidates = []
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        # PRUNE (not `continue`) the Digital Typhoon mount: its image tree is ~89GB
        # over hundreds of thousands of .h5 files and merely listing it costs
        # minutes. Mutating dirnames in place is what actually stops os.walk from
        # descending; a `continue` here would skip the filenames but still walk
        # every subdirectory underneath.
        dirnames[:] = [d for d in dirnames if "typhoon" not in d.lower()]
        for fn in filenames:
            if fn.lower().endswith((".h5", ".hdf5")) and (
                "tcsa" in fn.lower() or "tcir" in fn.lower()
            ):
                candidates.append(os.path.join(dirpath, fn))
    if not candidates:
        return None
    # Largest wins if the mirror ever ships more than one.
    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    if len(candidates) > 1:
        print(f"NOTE: {len(candidates)} TCIR-looking .h5 files found, using the largest:")
        for c in candidates:
            print(f"    {c}  ({os.path.getsize(c) / 1e9:.1f} GB)")
    return candidates[0]


# ----------------------------------------------------------------------------
# Source A -- Digital Typhoon
# ----------------------------------------------------------------------------
def build_dt_index(meta_dir, image_dir):
    """Index Digital Typhoon early-stage frames.

    Returns (rows_df, tracks, audit) where `tracks` maps an integer timestamp
    YYYYMMDDHH -> list of (lat, lng, storm_id) over the storm's FULL track (not
    just its early stage). tracks is what dedup_wpac() matches TCIR against.
    """
    csvs = sorted(
        f for f in os.listdir(meta_dir) if f.endswith(".csv") and f[:-4].isdigit()
    )
    print(f"[A] Found {len(csvs)} per-storm metadata CSVs")

    rows = []
    tracks = defaultdict(list)
    audit = {
        "n_csvs": len(csvs),
        "n_unreadable": 0,
        "n_bad_columns": 0,
        "n_no_wind_data": 0,
        "n_pre_1977_dropped": 0,
        "n_no_early_frame": 0,
        "n_missing_h5": 0,
        "n_label_agrees_with_v1_grade_rule": 0,
        "n_label_disagrees_with_v1_grade_rule": 0,
        "label_disagreement_examples": [],
    }

    for i, name in enumerate(csvs):
        storm_id = name[:-4]
        try:
            df = pd.read_csv(os.path.join(meta_dir, name))
        except Exception as exc:  # a genuinely unreadable CSV, not an expected case
            print(f"  WARN: cannot read {name}: {exc}")
            audit["n_unreadable"] += 1
            continue

        need = {"grade", "file_1", "wind", "year", "month", "day", "hour", "lat", "lng"}
        missing = need - set(df.columns)
        if missing:
            print(f"  WARN: {name} missing columns {sorted(missing)}, skipping")
            audit["n_bad_columns"] += 1
            continue

        year = int(storm_id[:4])
        # JMA best-track maximum sustained wind only exists from 1977 onwards. For
        # an earlier storm every `wind` cell is 0, which a >=64kt rule would read as
        # "never reached typhoon" -- a systematic false-negative across whole
        # decades. Drop rather than mislabel. (The imagery in this mirror starts
        # ~1978 anyway, so this should catch approximately nothing; it is a guard,
        # and the printed count is how you confirm that.)
        if year < 1977:
            audit["n_pre_1977_dropped"] += 1
            continue

        wind = pd.to_numeric(df["wind"], errors="coerce").fillna(0.0)
        wmax = float(wind.max()) if len(wind) else 0.0
        if wmax <= 0.0:
            # No wind ever recorded for a post-1977 storm. JMA assigns a YYYYNN
            # international number only to systems reaching tropical-storm strength,
            # so this should be rare; labelling it from absent data would be a guess.
            audit["n_no_wind_data"] += 1
            continue

        label = int(wmax >= TYPHOON_WIND_KT)

        # Cross-check against v1's rule so the label change is auditable rather than
        # taken on faith. For Digital Typhoon the two should agree almost perfectly:
        # JMA grade 5 (Typhoon) is *defined* as >= 64 kt 10-minute sustained.
        grade = pd.to_numeric(df["grade"], errors="coerce")
        v1_label = int((grade == TYPHOON_GRADE).any())
        if v1_label == label:
            audit["n_label_agrees_with_v1_grade_rule"] += 1
        else:
            audit["n_label_disagrees_with_v1_grade_rule"] += 1
            if len(audit["label_disagreement_examples"]) < 15:
                audit["label_disagreement_examples"].append(
                    {"storm_id": storm_id, "max_wind_kt": wmax,
                     "v1_grade_rule": v1_label, "v2_wind_rule": label}
                )

        # ---- full track, for the dedup position match ----
        t_y = pd.to_numeric(df["year"], errors="coerce")
        t_m = pd.to_numeric(df["month"], errors="coerce")
        t_d = pd.to_numeric(df["day"], errors="coerce")
        t_h = pd.to_numeric(df["hour"], errors="coerce")
        lat = pd.to_numeric(df["lat"], errors="coerce")
        lng = pd.to_numeric(df["lng"], errors="coerce")
        good = t_y.notna() & t_m.notna() & t_d.notna() & t_h.notna() & lat.notna() & lng.notna()
        if good.any():
            stamps = (
                t_y[good].astype(int) * 1000000
                + t_m[good].astype(int) * 10000
                + t_d[good].astype(int) * 100
                + t_h[good].astype(int)
            )
            for ts, la, lo in zip(stamps.to_numpy(), lat[good].to_numpy(), lng[good].to_numpy()):
                tracks[int(ts)].append((float(la), float(lo), storm_id))

        # ---- early-stage frames ----
        fname = df["file_1"].fillna("").astype(str).str.strip()
        early = df[(grade <= DT_EARLY_GRADE_MAX) & (fname != "")]
        if early.empty:
            audit["n_no_early_frame"] += 1
            continue

        # Evenly spaced through the early stage rather than the first N frames --
        # consecutive frames are 3-6 hours apart and near-identical.
        picks = np.unique(
            np.linspace(0, len(early) - 1, min(FRAMES_PER_STORM, len(early)))
            .round()
            .astype(int)
        )
        for p in picks:
            rec = early.iloc[int(p)]
            h5_path = os.path.join(image_dir, storm_id, str(rec["file_1"]).strip())
            if not os.path.exists(h5_path):
                audit["n_missing_h5"] += 1
                continue
            rows.append(
                {
                    "storm_key": storm_id,
                    "source": "digital_typhoon",
                    "basin": "WPAC",
                    "label": label,
                    "h5_path": h5_path,
                    "frame_ref": -1,
                    "max_wind_kt": wmax,
                }
            )

        if (i + 1) % 250 == 0:
            print(f"  [A] indexed {i + 1}/{len(csvs)} storms, {len(rows)} frames so far")

    print(f"[A] storms dropped -- unreadable={audit['n_unreadable']} "
          f"bad_columns={audit['n_bad_columns']} pre_1977={audit['n_pre_1977_dropped']} "
          f"no_wind_data={audit['n_no_wind_data']} no_early_frame={audit['n_no_early_frame']}")
    print(f"[A] frames listed in metadata but absent on disk: {audit['n_missing_h5']}")
    print(f"[A] label rule cross-check vs v1 (grade==5): "
          f"agree={audit['n_label_agrees_with_v1_grade_rule']} "
          f"disagree={audit['n_label_disagrees_with_v1_grade_rule']}")
    for ex in audit["label_disagreement_examples"]:
        print(f"      disagreement: {ex}")
    return pd.DataFrame(rows), tracks, audit


# ----------------------------------------------------------------------------
# Source B -- TCIR / TCSA
# ----------------------------------------------------------------------------
def check_tcir_id_pattern(info):
    """Validate the assumed TCIR storm-ID shape BEFORE it is used for dedup keys.

    Two ID formats are confirmed by direct inspection of this file: '200401L'
    (ATLN) and '201103I' (IO) -- i.e. YYYY + NN + a one-letter basin code, the ATCF
    convention. The WPAC letter has NOT been directly observed, so it is not
    assumed here: only the *structure* is required, and the actual letters found
    are printed so the log documents reality.

    Raises SystemExit if the structure does not hold, because a mis-parsed
    sequence number would produce wrong dedup keys and cause SILENT train/val
    leakage -- far worse than a loud crash. This check runs before any heavy IO,
    so failing here costs a minute of quota, not an hour.
    """
    print("\n" + "-" * 72)
    print("TCIR storm-ID format check (dedup depends on this)")
    print("-" * 72)

    for region in sorted(info["region"].astype(str).unique()):
        ids = sorted(info.loc[info["region"].astype(str) == region, "ID"].astype(str).unique())
        letters = Counter(i[-1] for i in ids if TCIR_ID_RE.match(i))
        n_ok = sum(1 for i in ids if TCIR_ID_RE.match(i))
        print(f"  {region:5s}: {len(ids):5d} storms, {n_ok:5d} match YYYYNN<letter>, "
              f"trailing letters={dict(letters)}")
        print(f"         sample IDs: {ids[:8]}")

    wpac_ids = sorted(info.loc[info["region"].astype(str) == "WPAC", "ID"].astype(str).unique())
    if not wpac_ids:
        raise SystemExit(
            "No region=='WPAC' rows in TCIR info. Either the region labels differ "
            "from the verified set ['ATLN','CPAC','EPAC','IO','SH','WPAC'], or the "
            f"file is not the expected one. Observed regions: "
            f"{sorted(info['region'].astype(str).unique())}"
        )

    matched = [i for i in wpac_ids if TCIR_ID_RE.match(i)]
    frac = len(matched) / len(wpac_ids)
    print(f"\n  WPAC: {len(matched)}/{len(wpac_ids)} IDs ({frac:.1%}) match "
          f"^YYYY NN <letter>$")
    if frac < TCIR_ID_PATTERN_MIN_MATCH:
        bad = [i for i in wpac_ids if not TCIR_ID_RE.match(i)][:20]
        raise SystemExit(
            "\n" + "!" * 72 + "\n"
            "TCIR WPAC storm-ID FORMAT ASSUMPTION IS WRONG -- refusing to continue.\n"
            f"Only {frac:.1%} of {len(wpac_ids)} WPAC IDs match the assumed\n"
            "'YYYY' + 'NN' + one basin letter shape (as seen in '200401L', '201103I').\n"
            f"Non-matching examples: {bad}\n"
            f"All-WPAC examples:     {wpac_ids[:20]}\n"
            "Fix the parser in check_tcir_id_pattern()/tcir_dedup_key() to the real\n"
            "format before re-running. Continuing would build WRONG dedup keys and\n"
            "leak the same storm across the train/val boundary silently.\n"
            + "!" * 72
        )

    years = sorted({int(TCIR_ID_RE.match(i).group(1)) for i in matched})
    seqs = sorted({int(TCIR_ID_RE.match(i).group(2)) for i in matched})
    print(f"  WPAC parsed years {years[0]}-{years[-1]}, "
          f"sequence numbers {seqs[0]}-{seqs[-1]}")
    if not (1900 <= years[0] and years[-1] <= 2100 and seqs[-1] <= 99):
        raise SystemExit(
            f"TCIR WPAC IDs parse to implausible (year, seq): years={years[:5]}..., "
            f"seqs={seqs[-5:]}. Refusing to build dedup keys from them."
        )
    print("-" * 72 + "\n")
    return True


def tcir_dedup_key(storm_id):
    """('200315W' | whatever the WPAC letter is) -> (2003, 15). None if unparseable."""
    m = TCIR_ID_RE.match(str(storm_id))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def dt_dedup_key(storm_id):
    """Digital Typhoon '199115' -> (1991, 15). None if unparseable."""
    s = str(storm_id)
    if len(s) != 6 or not s.isdigit():
        return None
    return (int(s[:4]), int(s[4:6]))


def build_tcir_index(info):
    """Index TCIR early-stage frames. `frame_ref` is the POSITIONAL row number in
    the h5 'images' dataset -- row i of `info` is images[i]."""
    rows = []
    audit = {
        "n_frames_total": int(len(info)),
        "n_storms_total": int(info["ID"].nunique()),
        "n_no_vmax_data": 0,
        "n_no_early_frame": 0,
        "n_mixed_region": 0,
    }

    vmax = pd.to_numeric(info["Vmax"], errors="coerce")
    time_i = pd.to_numeric(info["time"], errors="coerce")
    lat = pd.to_numeric(info["lat"], errors="coerce")
    lon = pd.to_numeric(info["lon"], errors="coerce")

    work = pd.DataFrame(
        {
            "frame_ref": np.arange(len(info), dtype=np.int64),
            "storm_id": info["ID"].astype(str).to_numpy(),
            "region": info["region"].astype(str).to_numpy(),
            "vmax": vmax.to_numpy(),
            "time": time_i.to_numpy(),
            "lat": lat.to_numpy(),
            "lon": lon.to_numpy(),
        }
    )

    for storm_id, g in work.groupby("storm_id", sort=True):
        regions = pd.unique(g["region"])
        if len(regions) > 1:
            audit["n_mixed_region"] += 1
            print(f"  WARN: TCIR storm {storm_id} spans regions {list(regions)}, "
                  f"using '{regions[0]}'")
        basin = str(regions[0])

        wmax = float(np.nanmax(g["vmax"])) if np.isfinite(g["vmax"]).any() else float("nan")
        if not np.isfinite(wmax):
            audit["n_no_vmax_data"] += 1
            continue
        label = int(wmax >= TYPHOON_WIND_KT)

        # Chronological, so "evenly spaced through the early stage" means what it
        # says. The h5 is already stored in track order but sorting is free here.
        g = g.sort_values("time", kind="stable")
        early = g[g["vmax"] <= TCIR_EARLY_VMAX_KT]
        if early.empty:
            audit["n_no_early_frame"] += 1
            continue

        picks = np.unique(
            np.linspace(0, len(early) - 1, min(FRAMES_PER_STORM, len(early)))
            .round()
            .astype(int)
        )
        for p in picks:
            rec = early.iloc[int(p)]
            rows.append(
                {
                    "storm_key": storm_id,
                    "source": "tcir",
                    "basin": basin,
                    "label": label,
                    "h5_path": "",
                    "frame_ref": int(rec["frame_ref"]),
                    "max_wind_kt": wmax,
                }
            )

    print(f"[B] {audit['n_storms_total']} storms / {audit['n_frames_total']} frames in file")
    print(f"[B] storms dropped -- no_vmax_data={audit['n_no_vmax_data']} "
          f"no_early_frame={audit['n_no_early_frame']} (mixed_region warnings="
          f"{audit['n_mixed_region']})")
    return pd.DataFrame(rows), work, audit


# ----------------------------------------------------------------------------
# Cross-dataset de-duplication (the correctness-critical part)
# ----------------------------------------------------------------------------
def dedup_wpac(dt_index, tcir_index, tcir_work, dt_tracks):
    """Find TCIR WPAC storms that are the same physical storm as a Digital Typhoon
    storm, and return the set of TCIR storm_keys to drop entirely.

    Digital Typhoon wins every collision: 512x512 native resolution vs TCIR's
    128x128, and many more early-stage frames per storm.

    TWO independent detectors are run and their results are UNIONed:

      1. ID KEY  -- (year, sequence_number) parsed out of both ID schemes, as
         specified. Cheap and exact when the two agencies agree on numbering.
         BUT: Digital Typhoon's YYYYNN is the JMA *international* number, assigned
         only to systems reaching tropical-storm strength, while TCIR's WPAC
         numbering is the ATCF/JTWC one, which also numbers systems JMA never
         named. The two therefore DRIFT APART within a season -- JTWC storm 15 of
         1991 is not necessarily JMA storm 15 of 1991. On its own this detector
         would both miss real duplicates and invent fake ones. Because Digital
         Typhoon numbers ~26 storms in every year TCIR covers, expect this
         detector to fire on very nearly every TCIR WPAC storm; a near-100% hit
         rate here is the EXPECTED reading, not a bug.

      2. TRACK MATCH -- both datasets carry best-track lat/lon at synoptic times.
         If a TCIR storm sits within DEDUP_MATCH_DEG degrees of some Digital
         Typhoon storm at >= DEDUP_MIN_MATCH_POINTS shared timestamps, it is the
         same storm regardless of how either agency numbered it. This is the
         detector that actually holds up.

    Union, not intersection: dropping a TCIR storm that was not really a duplicate
    costs a handful of training frames, whereas keeping a real duplicate silently
    inflates val AUC. The asymmetry is the whole reason for running both.
    """
    print("\n" + "=" * 72)
    print("CROSS-DATASET DE-DUPLICATION (Western Pacific)")
    print("=" * 72)

    dt_storms = sorted(dt_index["storm_key"].unique())
    tcir_wpac = tcir_index[tcir_index["basin"] == "WPAC"]
    tcir_wpac_storms = sorted(tcir_wpac["storm_key"].unique())
    print(f"WPAC storms with usable early frames -- "
          f"digital_typhoon={len(dt_storms)}  tcir={len(tcir_wpac_storms)}")

    # ---- detector 1: (year, seq) key ----
    dt_keys = {}
    n_dt_unparsed = 0
    for s in dt_storms:
        k = dt_dedup_key(s)
        if k is None:
            n_dt_unparsed += 1
            print(f"  WARN: Digital Typhoon storm id '{s}' does not parse as YYYYNN")
            continue
        dt_keys.setdefault(k, []).append(s)
    if n_dt_unparsed:
        print(f"  {n_dt_unparsed} Digital Typhoon ids unparseable (excluded from key dedup)")

    by_key = set()
    key_hits = []
    for s in tcir_wpac_storms:
        k = tcir_dedup_key(s)
        if k is None:
            print(f"  WARN: TCIR WPAC id '{s}' does not parse; key dedup skipped for it")
            continue
        if k in dt_keys:
            by_key.add(s)
            key_hits.append((s, k, dt_keys[k][0]))
    print(f"[1] ID-key (year, seq) overlap: {len(by_key)} TCIR WPAC storms")
    for h in key_hits[:10]:
        print(f"      tcir {h[0]} -> key {h[1]} -> digital_typhoon {h[2]}")

    # ---- detector 2: best-track position match ----
    # Only Digital Typhoon storms that actually made it INTO dt_index can be leaked
    # against. dt_tracks also holds storms whose early stage was never imaged; a
    # TCIR copy of one of those is not a duplicate of anything in the training set,
    # so matching it would throw away a usable storm for no safety gain.
    t0 = time.time()
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
        if votes:
            best_sid, best_n = votes.most_common(1)[0]
            if best_n >= DEDUP_MIN_MATCH_POINTS:
                by_track[storm_id] = (best_sid, best_n)
    print(f"[2] Best-track position overlap "
          f"(<= {DEDUP_MATCH_DEG} deg at >= {DEDUP_MIN_MATCH_POINTS} shared timestamps): "
          f"{len(by_track)} TCIR WPAC storms  ({time.time() - t0:.0f}s)")
    for sid, (dsid, n) in list(sorted(by_track.items()))[:10]:
        print(f"      tcir {sid} -> digital_typhoon {dsid} ({n} matching track points)")

    # ---- disagreement between the two detectors, printed for audit ----
    track_set = set(by_track)
    only_key = sorted(by_key - track_set)
    only_track = sorted(track_set - by_key)
    print(f"\nDetector agreement: both={len(by_key & track_set)}  "
          f"ID-key only={len(only_key)}  track only={len(only_track)}")
    if only_key:
        print(f"  ID-key only ({len(only_key)}): the (year, seq) numbers collided but "
              f"the tracks did NOT, so these are most likely DIFFERENT storms that "
              f"merely share a sequence number. Expected to be a large group -- see "
              f"the JMA-vs-JTWC numbering note in this function.")
        print(f"      examples: {only_key[:15]}")
    if only_track:
        print(f"  track only ({len(only_track)}): SAME storm, different agency "
              f"numbering. These are exactly the leaks the ID key alone would have "
              f"MISSED.")
        print(f"      examples: {only_track[:15]}")

    if DEDUP_USE_ID_KEY_UNION:
        drop = by_key | track_set
        print(f"\nDEDUP_USE_ID_KEY_UNION=True -> dropping the UNION of both detectors")
    else:
        drop = set(track_set)
        print(f"\nDEDUP_USE_ID_KEY_UNION=False -> dropping only position-matched storms")
    print(f"Dropping {len(drop)} of {len(tcir_wpac_storms)} TCIR WPAC storms as "
          f"duplicates of Digital Typhoon (Digital Typhoon copy kept: higher "
          f"resolution, more early frames)")
    print("=" * 72 + "\n")

    audit = {
        "dt_wpac_storms": len(dt_storms),
        "tcir_wpac_storms": len(tcir_wpac_storms),
        "overlap_by_id_key": len(by_key),
        "overlap_by_track_match": len(track_set),
        "overlap_by_both": len(by_key & track_set),
        "overlap_id_key_only": len(only_key),
        "overlap_track_only": len(only_track),
        "n_tcir_wpac_storms_dropped": len(drop),
        "n_tcir_wpac_storms_kept": len(tcir_wpac_storms) - len(drop),
        "dropped_tcir_storm_ids": sorted(drop),
        "used_id_key_union": DEDUP_USE_ID_KEY_UNION,
        "match_deg": DEDUP_MATCH_DEG,
        "min_match_points": DEDUP_MIN_MATCH_POINTS,
    }
    return drop, audit


# ----------------------------------------------------------------------------
# Image loading
# ----------------------------------------------------------------------------
def kelvin_to_display_u8(arr):
    """Brightness temperature (K) -> uint8 display frame, cold = bright.

    Shared by BOTH sources so a Digital Typhoon pixel and a TCIR pixel of the same
    temperature become the same grey level -- without this the two datasets would
    be on different scales and the combination would be worse than either alone.
    Inverting so cold (high, convective) cloud tops render bright is the
    conventional IR rendering, is what the demo video shows, and is what
    local_tool/detect.py assumes.
    """
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
    """TCIR: images[row] is (128,128,4); channel 0 is IR1 in Kelvin.

    Channels 1-3 (WV / VIS / PMW) are deliberately unused -- this model stays
    1-channel so it remains loadable by local_tool/detect.py's build_model().
    """
    frame = np.asarray(images_ds[row], dtype=np.float32)
    arr = frame[..., 0]
    finite = np.isfinite(arr)
    frac = float(finite.mean())
    if frac < TCIR_MIN_FINITE_FRAC:
        raise ValueError(f"IR1 channel only {frac:.1%} finite")
    if frac < 1.0:
        # Missing scan lines -> treat as warm (dark), same convention detect.py uses.
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


def to_model_input(u8_batch):
    """uint8 (N,H,W) -> normalised float tensor (N,1,IMG_SIZE,IMG_SIZE)."""
    t = torch.from_numpy(np.ascontiguousarray(u8_batch)).float().div_(255.0).unsqueeze(1)
    if t.shape[-1] != IMG_SIZE or t.shape[-2] != IMG_SIZE:
        t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return (t - NORM_MEAN) / NORM_STD


class FrameDataset(Dataset):
    """Frames are pre-decoded into RAM once (see preload_all), so this is just an
    indexed view plus optional augmentation."""

    def __init__(self, images_u8, labels, augment):
        self.images = images_u8
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augment:
            # Rotations only -- NOT flips. A mirror flip reverses the storm's
            # rotational chirality, which is physically fixed by hemisphere and is
            # exactly the structure the model should key on. (This dataset now
            # contains BOTH chiralities for real, since TCIR's SH basin spins the
            # other way -- which is a reason to keep flips out, not to allow them:
            # a flipped SH storm is a physically impossible NH storm.)
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img = np.rot90(img, k).copy()
        return torch.from_numpy(img), float(self.labels[i])


def preload_all(index, tcir_h5_path):
    """Decode every frame ONCE, resized, into a single uint8 array in RAM.

    ~10k frames at 224x224 uint8 == ~500MB, trivial for Kaggle RAM, and it turns 8
    epochs of HDF5 reads off two network-mounted datasets into one pass.

    TCIR is done FIRST and deliberately: it is the unknown-latency source (random
    row access into one 20GB file, versus Digital Typhoon's many small files which
    v1 already proved fast). Its rows are read in ASCENDING frame order so the
    access pattern is as close to sequential as it can be. An ETA is printed after
    100 frames of each pass, so a pathologically slow mount is caught in the first
    minute rather than after half an hour.
    """
    import h5py

    n = len(index)
    out = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    ok = np.ones(n, dtype=bool)
    sources = index["source"].to_numpy()

    # ---- pass 1: TCIR ----
    tc_pos = np.where(sources == "tcir")[0]
    if len(tc_pos):
        refs = index["frame_ref"].to_numpy()[tc_pos]
        order = np.argsort(refs, kind="stable")  # ascending row order == near-sequential IO
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
                if (j + 1) % 500 == 0:
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
            if (j + 1) % 500 == 0:
                print(f"  [A] preloaded {j + 1}/{len(dt_pos)} ({time.time() - t0:.0f}s)")
        print(f"[A] Digital Typhoon preload done in {time.time() - t0:.0f}s")

    print(f"Preload total failures: {int((~ok).sum())}/{n}")
    return out, ok


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model():
    """resnet18, ImageNet-pretrained, adapted to 1-channel input, 1 logit out.

    Byte-for-byte the same architecture v1 saved, which is what makes this
    checkpoint a drop-in replacement for local_tool/detect.py.
    """
    import torchvision

    pretrained = True
    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=weights)
    except Exception as exc:
        # Requires enable_internet: true in kernel-metadata.json (torchvision
        # fetches the weights from download.pytorch.org). Falling back to random
        # init keeps the run alive but will noticeably hurt AUC -- loud on purpose.
        print(f"!! PRETRAINED WEIGHTS UNAVAILABLE ({exc}) -- using RANDOM INIT !!")
        net = torchvision.models.resnet18(weights=None)
        pretrained = False

    old = net.conv1.weight.data  # (64,3,7,7)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # MEAN (not sum) across the RGB axis: for a grayscale image the pretrained
    # filter's response to a 3x-replicated channel equals the response of the
    # mean-collapsed filter to a single channel. Preserves both the learned
    # edge/texture filters and their output scale.
    net.conv1.weight.data = old.mean(dim=1, keepdim=True)

    net.fc = nn.Linear(net.fc.in_features, 1)
    return net, pretrained


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def evaluate(probs, labels, threshold=0.5):
    from sklearn.metrics import roc_auc_score

    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    acc = (tp + tn) / max(1, len(labels))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")  # only one class present in this split
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def breakdown(probs, labels, groups):
    """Per-group val metrics. Used for the per-basin report -- Indian Ocean
    performance has to be reportable on its own, not hidden inside a global
    average dominated by the Western Pacific."""
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        d = evaluate(probs[m], labels[m])
        d["n_frames"] = int(m.sum())
        d["n_positive_frames"] = int((labels[m] == 1).sum())
        out[str(g)] = d
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ discovery
    dt_root = find_dt_root()
    if not dt_root:
        raise SystemExit("No typhoon dataset under /kaggle/input -- check dataset_sources")
    meta_dir, image_dir = resolve_dt_dirs(dt_root)
    print("DT ROOT   =", dt_root)
    print("meta_dir  =", meta_dir)
    print("image_dir =", image_dir)

    tcir_path = find_tcir_h5()
    if not tcir_path:
        raise SystemExit(
            "No TCSA/TCIR .h5 under /kaggle/input -- attach "
            "boyochen/tropical-cyclone-structure-analysis-tcsa as a dataset source"
        )
    print("TCIR h5   =", tcir_path, f"({os.path.getsize(tcir_path) / 1e9:.1f} GB)")

    # ------------------------------------------------------------------ source B first
    # Read TCIR's metadata table and validate the ID format BEFORE doing the
    # expensive Digital Typhoon CSV walk, so a wrong ID assumption fails in the
    # first minute of the run instead of after the index is built.
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

    with __import__("h5py").File(tcir_path, "r") as f:
        img_shape = f["images"].shape
    print(f"  images shape={img_shape}")
    if img_shape[0] != len(info):
        raise SystemExit(
            f"TCIR row-alignment broken: images has {img_shape[0]} rows but info has "
            f"{len(info)}. frame_ref indices would point at the wrong storms."
        )
    if len(img_shape) != 4 or img_shape[-1] < 1:
        raise SystemExit(f"Unexpected TCIR images shape {img_shape}; expected (N,128,128,4)")

    check_tcir_id_pattern(info)

    tcir_index, tcir_work, tcir_audit = build_tcir_index(info)
    if tcir_index.empty:
        raise SystemExit("TCIR index is empty -- layout is not what was expected")

    # ------------------------------------------------------------------ source A
    print()
    dt_index, dt_tracks, dt_audit = build_dt_index(meta_dir, image_dir)
    if dt_index.empty:
        raise SystemExit("Digital Typhoon index is empty -- layout is not what was expected")

    # ------------------------------------------------------------------ dedup
    drop_ids, dedup_audit = dedup_wpac(dt_index, tcir_index, tcir_work, dt_tracks)
    tcir_index = tcir_index[~tcir_index["storm_key"].isin(drop_ids)].reset_index(drop=True)

    index = pd.concat([dt_index, tcir_index], ignore_index=True)

    # storm_key must be globally unique across the two sources, because it is both
    # the split key AND what gets written into the checkpoint's
    # train_storm_ids/val_storm_ids (which demo_kernel/render_demo.py looks a raw
    # Digital Typhoon storm id up in). Digital Typhoon ids are 6 digits, TCIR ids
    # end in a letter, so a collision is structurally impossible -- but assert it
    # rather than assume it, since a collision would merge two different storms.
    clashing = index.groupby("storm_key")["source"].nunique()
    clashing = clashing[clashing > 1]
    if len(clashing):
        raise SystemExit(
            f"storm_key collision across sources: {list(clashing.index[:10])}. "
            f"Two different storms would be merged into one split unit; aborting."
        )

    storms_by_source_basin = (
        index.drop_duplicates("storm_key")
        .groupby(["source", "basin"])
        .size()
        .to_dict()
    )
    print("Final storms per source per basin (after dedup):")
    for (src, basin), n in sorted(storms_by_source_basin.items()):
        print(f"    {src:16s} {basin:5s} {n:5d}")
    print("Final frames per source per basin:")
    for (src, basin), n in sorted(index.groupby(["source", "basin"]).size().to_dict().items()):
        print(f"    {src:16s} {basin:5s} {n:5d}")

    # ------------------------------------------------------------------ split
    # SPLIT BY STORM, NEVER BY FRAME. Frames of one storm are hours apart, look
    # nearly identical, and all share one label (the label is a property of the
    # storm, not the frame). A random per-row split would put near-duplicates on
    # both sides and inflate val AUC massively.
    #
    # Stratified by (basin, label) rather than a flat shuffle: the Indian Ocean has
    # only ~84 storms, and a flat 80/20 shuffle could easily hand it a val set with
    # almost no positives, making the per-basin IO AUC that this model exists to
    # report meaningless.
    storms = index.drop_duplicates("storm_key")[["storm_key", "basin", "label"]]
    rng = np.random.RandomState(SEED)
    val_keys = set()
    for (basin, label), grp in storms.groupby(["basin", "label"], sort=True):
        keys = np.array(sorted(grp["storm_key"].tolist()))
        rng.shuffle(keys)
        n_val = int(round(len(keys) * VAL_FRACTION))
        if n_val == 0 and len(keys) >= 2:
            n_val = 1  # never leave a stratum with zero val storms
        val_keys.update(keys[:n_val].tolist())
        print(f"  split {basin:5s} label={label}: {len(keys):5d} storms -> "
              f"{n_val:4d} val / {len(keys) - n_val:5d} train")

    index["split"] = np.where(index["storm_key"].isin(val_keys), "val", "train")
    all_keys = set(index["storm_key"].unique())
    train_keys = all_keys - val_keys

    print(f"\nStorms: {len(all_keys)}  train={len(train_keys)}  val={len(val_keys)}")
    print(f"Frames: total={len(index)}  train={(index['split'] == 'train').sum()}  "
          f"val={(index['split'] == 'val').sum()}")
    print("Storm-level label balance:",
          index.drop_duplicates("storm_key")["label"].value_counts().to_dict())

    # ------------------------------------------------------------------ preload
    images, ok = preload_all(index, tcir_path)
    index = index[ok].reset_index(drop=True)
    images = images[ok]
    n_load_failures = int((~ok).sum())
    index.to_csv(INDEX_PATH, index=False)  # written post-filter so it matches training

    tr_mask = (index["split"] == "train").to_numpy()
    va_mask = ~tr_mask
    y = index["label"].to_numpy().astype(np.float32)
    val_basins = index.loc[va_mask, "basin"].to_numpy()
    val_sources = index.loc[va_mask, "source"].to_numpy()

    train_ds = FrameDataset(images[tr_mask], y[tr_mask], augment=True)
    val_ds = FrameDataset(images[va_mask], y[va_mask], augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=False)
    # shuffle=False so the concatenated val probs line up with val_basins/val_sources.
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    net, pretrained = build_model()
    net.to(device)

    n_pos = float((y[tr_mask] == 1).sum())
    n_neg = float((y[tr_mask] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    print(f"train frames: pos={n_pos:.0f} neg={n_neg:.0f} pos_weight={pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # No AMP on purpose: ~10k 224px images means an epoch is seconds of resnet18
    # compute either way, and fp32 avoids torch.cuda.amp / torch.amp API churn.

    history = []
    best_auc = -1.0
    best_epoch = -1
    best_metrics = None
    best_by_basin = None
    best_by_source = None

    for epoch in range(1, EPOCHS + 1):
        net.train()
        total_loss, n_seen = 0.0, 0
        t0 = time.time()
        for imgs_u8, labels in train_dl:
            x = to_model_input(imgs_u8.numpy()).to(device, non_blocking=True)
            t = labels.to(device).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            logits = net(x)
            loss = criterion(logits, t)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(labels)
            n_seen += len(labels)
        sched.step()
        train_loss = total_loss / max(1, n_seen)

        net.eval()
        probs, ys = [], []
        with torch.no_grad():
            for imgs_u8, labels in val_dl:
                x = to_model_input(imgs_u8.numpy()).to(device)
                probs.append(torch.sigmoid(net(x)).squeeze(1).cpu().numpy())
                ys.append(labels.numpy())
        probs = np.concatenate(probs)
        ys = np.concatenate(ys).astype(int)
        m = evaluate(probs, ys)
        m["epoch"] = epoch
        m["train_loss"] = train_loss
        history.append(m)

        print(f"epoch {epoch}/{EPOCHS}  loss={train_loss:.4f}  "
              f"val_acc={m['accuracy']:.3f}  val_prec={m['precision']:.3f}  "
              f"val_rec={m['recall']:.3f}  val_f1={m['f1']:.3f}  val_auc={m['auc']:.3f}  "
              f"({time.time() - t0:.0f}s)")

        # Select on AUC: threshold-free, which matters because the demo displays a
        # raw confidence rather than a hard 0/1 decision.
        score = m["auc"] if m["auc"] == m["auc"] else m["accuracy"]
        if score > best_auc:
            best_auc, best_epoch, best_metrics = score, epoch, m
            best_by_basin = breakdown(probs, ys, val_basins)
            best_by_source = breakdown(probs, ys, val_sources)
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "arch": "resnet18_1ch",
                    "pretrained_backbone": pretrained,
                    "epoch": epoch,
                    "val_metrics": m,
                    # Saved so the demo renderer and the local upload tool can
                    # reproduce preprocessing exactly. Key names and TYPHOON_GRADE
                    # are kept from v1 for drop-in compatibility even though v2
                    # labels on wind, not grade.
                    "config": {
                        "temp_lo": TEMP_LO, "temp_hi": TEMP_HI, "img_size": IMG_SIZE,
                        "norm_mean": NORM_MEAN, "norm_std": NORM_STD,
                        "early_grade_max": DT_EARLY_GRADE_MAX,
                        "typhoon_grade": TYPHOON_GRADE,
                        "typhoon_wind_kt": TYPHOON_WIND_KT,
                        "tcir_early_vmax_kt": TCIR_EARLY_VMAX_KT,
                        "label_rule": label_note(),
                        "sources": ["digital_typhoon", "tcir"],
                        "basins": sorted(index["basin"].unique().tolist()),
                    },
                    # Raw storm ids (Digital Typhoon '201618', TCIR '201103I'), NOT
                    # source-prefixed -- render_demo.py looks a bare Digital Typhoon
                    # storm id up in these lists for its honesty check.
                    "train_storm_ids": sorted(train_keys),
                    "val_storm_ids": sorted(val_keys),
                    "val_metrics_by_basin": best_by_basin,
                    "val_metrics_by_source": best_by_source,
                },
                CKPT_PATH,
            )
            print(f"  -> saved best checkpoint ({CKPT_PATH})")

    summary = {
        "task": "early-stage frame -> P(storm later reaches >= 64 kt / typhoon strength)",
        "label_rule": label_note(),
        "sources": {
            "digital_typhoon": "bwandowando/digital-typhoon-dataset-western-pacific-wp "
                               "(512x512 IR, WPAC only)",
            "tcir": "boyochen/tropical-cyclone-structure-analysis-tcsa "
                    "(128x128 IR1 channel 0 only, all basins)",
        },
        "pretrained_backbone": pretrained,
        "n_storms": len(all_keys),
        "n_train_storms": len(train_keys),
        "n_val_storms": len(val_keys),
        "n_train_frames": int(tr_mask.sum()),
        "n_val_frames": int(va_mask.sum()),
        "frames_per_storm_cap": FRAMES_PER_STORM,
        "n_load_failures": n_load_failures,
        "pos_weight": float(pos_weight.item()),
        "split": "by storm_key, stratified by (basin, label); never by frame",
        "storms_by_source_basin": {f"{k[0]}|{k[1]}": int(v)
                                   for k, v in storms_by_source_basin.items()},
        "dedup": dedup_audit,
        "digital_typhoon_audit": dt_audit,
        "tcir_audit": tcir_audit,
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "best_val_metrics_by_basin": best_by_basin,
        "best_val_metrics_by_source": best_by_source,
        "history": history,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BEST (overall) ===")
    print(json.dumps(best_metrics, indent=2))
    print("\n=== BEST, by basin ===")
    print(json.dumps(best_by_basin, indent=2))
    print("\n=== BEST, by source ===")
    print(json.dumps(best_by_source, indent=2))
    print("\nWrote", CKPT_PATH, METRICS_PATH, INDEX_PATH)


if __name__ == "__main__":
    main()
