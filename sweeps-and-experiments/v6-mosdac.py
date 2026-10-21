"""
Cyclone Genesis Predictor -- v6: does adding MOSDAC/INSAT Indian Ocean data actually help the
Indian Ocean?

FORKED FROM train_kernel_v5_final/train.py. Everything below the "v6" markers is
v5 unchanged, so the baseline arm really is the shipped recipe and not a
re-tuned lookalike. The shipped config C_reg is used for BOTH arms: the ONLY
difference between them is the training data.

WHAT THIS RUN MEASURES, AND WHY IT IS BUILT THIS WAY

The Indian Ocean is the weakest basin (storm AUC 0.681 against WPAC's 0.793 in
the shipped checkpoint) and the diagnosed cause is data scarcity, not modelling:
81 IO storms, of which only ~16 land in validation, which pins IO AUC to roughly
+/-0.13 and gives a measured run-to-run swing of 0.018 on an IDENTICAL config.
Any single-run "IO went up" claim at that noise level is worthless. So:

  * TWO ARMS x THREE SEEDS, not one run each. `baseline` trains on Digital
    Typhoon + TCIR exactly as the shipped model does; `mosdac` adds the INSAT
    frames. Three seeds per arm measure the reproducibility floor IN THIS RUN
    rather than quoting a number from a previous session.

  * IDENTICAL SPLITS across arms. assign_splits() draws the Digital Typhoon and
    TCIR split FIRST, consuming the RandomState in exactly the order v5 did, and
    only then draws the MOSDAC storms from a SEPARATE generator. Without that,
    adding 32 IO storms would reshuffle the IO group and change which TCIR storms
    are held out, and the two arms would be scored on different validation sets.

  * BOTH ARMS SCORED ON THE SAME VAL SET, including the MOSDAC val storms. The
    baseline has simply never seen INSAT imagery, which makes its score on those
    storms a real cross-instrument transfer measurement.

  * THE SOURCE-SHORTCUT PROBE. MOSDAC's storm list is IMD's NAMED-cyclone list,
    and IMD only names a system once it reaches 34 kt, so the MOSDAC storms skew
    positive (19 of 32) while the existing TCIR IO population skews negative (27
    of 81). A model that learns nothing except "this frame looks like INSAT" would
    therefore still score above chance on a POOLED Indian Ocean validation set.
    source_shortcut_probe() computes exactly how much free AUC that shortcut is
    worth, and every IO number is additionally reported WITHIN each source, where
    a source shortcut buys nothing at all. The within-TCIR IO number is the
    headline: it asks whether the storms the model already had got easier.

  * PAIRED BOOTSTRAP OVER STORMS on every one of those slices, because frames of
    one storm share a label and the effective sample size is storms.

MOSDAC frames enter on BOTH SIDES OF THE LABEL by construction (19 positive /
13 negative storms). If they only ever appeared as one class, "came from INSAT"
would be a perfect predictor of that class and the network would take it -- the
same failure as the decay-tail leak this project already fixed once, reached
through a different door.


Trains ONE shipping model using whichever configuration train_kernel_v5/sweep.py
proved out under 4-fold cross-validation, on the same single 80/20 storm split
v3 used, and writes a checkpoint that local_tool/detect.py and
demo_kernel/render_demo.py can load with NO code change.

Read train_kernel_v5/sweep.py's docstring first -- it explains the frame-selection
fix, the aux-metadata branch, the graded target and, most importantly, why the
v3/v4/v4.1 comparison table was measuring noise.

WHAT THIS ADDS ON TOP OF THE SWEEP
  * a single 80/20 storm split (stratified by basin and label, same SEED as v3)
  * Platt calibration with v3's storm-level cross-check gate, folded into fc
  * the operating-point table detect.py's two display constants come from
  * storm-level metrics as well as frame-level
  * an EXACT plain-resnet18 export, verified numerically before saving:
    with the aux mask at zero the aux branch contributes the zero vector, so the
    model IS a stock resnet18 with fc = Linear(512,1). export_plain_resnet18()
    remaps the weights and asserts the two score identically on every validation
    frame. `state_dict` in the checkpoint is that plain model;
    `full_state_dict` is the metadata-capable one.

SET FINAL_CONFIG_NAME (or CGP_FINAL_CONFIG) to the winning sweep config.
SET SHIP_MODE to "image" (calibrate + threshold the image-only mode, which is what
detect.py can feed today) or "meta" (calibrate the mode that also gets
position/date). Both modes are scored either way; only the shipped one is
calibrated.

Runs as a plain Kaggle *script* kernel with GPU enabled (T4). ~15 min.
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

# Overridable so the offline dry-run harness can point the whole script at a
# synthetic /kaggle/input tree. On Kaggle neither variable is set and these are
# exactly the v2 paths.
INPUT_ROOT = os.environ.get("CGP_INPUT_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("CGP_OUT_DIR", "/kaggle/working")

# (A2) Cap frames per storm, per source. v2 used 4. Frames of one storm are hours
# apart and share a label, so extra frames add correlated -- not independent --
# samples; the effective sample size is still bounded by the 2,096 storms. They do
# genuinely help an UNDERFIT model (they are extra views of the same storm across
# its early life, which is what the model has to be invariant to), so 4 -> 6 is a
# deliberate 1.5x on the dominant runtime cost. Set back to 4 to recover v2's
# dataset size and about 6 minutes of GPU time.
FRAMES_PER_STORM = 6

# ---- label rule (unified across both sources) ------------------------------
TYPHOON_WIND_KT = 64.0

# (D2) Audit only -- NOT applied to any label. Digital Typhoon's `wind` is JMA
# 10-minute sustained; TCIR's `Vmax` is JTWC/NHC 1-minute sustained. The standard
# conversion is v_10min ~= 0.88 * v_1min, so TCIR's 64 kt bar corresponds to a
# 56.3 kt bar in Digital Typhoon's units -- i.e. the two sources currently apply
# physically DIFFERENT bars while both calling it "64 kt". This constant exists so
# the run can print how many Digital Typhoon storms would flip label if the bars
# were physically harmonised, which is the number needed to decide whether the fix
# is worth the disruption. See the note in main() before changing anything.
WIND_10MIN_PER_1MIN = 0.88

# ---- "early stage" definition, per source ----------------------------------
DT_EARLY_GRADE_MAX = 2
# TCIR has NO grade field, so the early filter has to be expressed on Vmax.
# 33 kt is the standard tropical-depression / tropical-storm boundary. This is a
# DELIBERATE cross-dataset harmonisation choice standing in for DT's `grade <= 2`.
TCIR_EARLY_VMAX_KT = 33.0

# ---- preprocessing (LOAD-BEARING -- do not change) -------------------------
# These exact values are baked into local_tool/detect.py, demo_kernel/render_demo.py
# and every existing checkpoint. Changing any of them silently invalidates every
# existing deliverable.
TEMP_LO, TEMP_HI = 190.0, 300.0
IMG_SIZE = 224
NORM_MEAN, NORM_STD = 0.449, 0.226

# Written into the checkpoint config purely for compatibility: render_demo.py reads
# `typhoon_grade` to decide when to switch a frame to the CONFIRMED overlay using
# Digital Typhoon's own grade column. It plays no part in this script's labelling.
TYPHOON_GRADE = 5

# ---- dedup (verbatim from v2 -- correctness-critical, do not tune) ----------
DEDUP_MATCH_DEG = 1.5
DEDUP_MIN_MATCH_POINTS = 4
DEDUP_USE_ID_KEY_UNION = True
TCIR_ID_PATTERN_MIN_MATCH = 0.90
TCIR_ID_RE = re.compile(r"^(\d{4})(\d{2})([A-Za-z])$")
TCIR_MIN_FINITE_FRAC = 0.50

# ---- training ---------------------------------------------------------------
VAL_FRACTION = 0.2
EPOCHS = 24               # (A1) v2 used 8 and was still improving at epoch 8.
WARMUP_EPOCHS = 3         # (A1) linear warmup, then cosine over the remainder.
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4

# (A3) EMA decay. At ~310 optimiser steps/epoch, 0.999 averages over roughly the
# last 1,000 steps (~3 epochs) -- long enough to kill the epoch-to-epoch val-AUC
# noise that made v2's "best epoch" close to a coin flip, short enough to still
# track the last third of training.
EMA_DECAY = 0.999

# (B2) Label smoothing. The label is genuinely noisy: the two sources threshold on
# different wind-averaging conventions (see WIND_10MIN_PER_1MIN), and a storm that
# peaked at 63 kt versus 65 kt is a coin flip physically but an opposite label.
# Smoothing caps the target at 0.97/0.03 so the optimiser cannot spend capacity
# driving logits to infinity on examples whose labels are partly arbitrary -- which
# is precisely the mechanism that produces CONFIDENT false positives.
LABEL_SMOOTHING = 0.03

# (B1) Augmentation strengths. All applied in the [0,1] DISPLAY domain -- the same
# domain detect.py feeds the model -- and all kept mild, because the diagnosis is
# underfitting and over-strong augmentation would make that worse.
AUG_BRIGHTNESS = 0.05     # additive, display units (0.05 ~= 5.5 K of scale offset)
AUG_CONTRAST = 0.08       # multiplicative gain about mid-grey, 1 +/- this
AUG_ZOOM_LOG = 0.13       # log-uniform zoom, exp(+/-0.13) => 0.878x .. 1.139x
AUG_SHIFT = 0.06          # translation, normalised grid units (0.06 => 3% of width)
AUG_NOISE_STD = 0.010     # Gaussian, display units (~1.1 K, realistic sensor noise)

# (D3) Basin-balanced loss weighting exponent: frame weight ~ n_basin ** -alpha,
# renormalised to mean 1. alpha=0 is uniform (v2 behaviour, and the DEFAULT);
# alpha=1 gives every basin equal total gradient mass.
#
# OFF BY DEFAULT ON PURPOSE. The "WPAC dominates training" hypothesis is only
# weakly supported by the actual numbers: WPAC is 54% of frames, not 90%, so the
# five other basins already contribute nearly half of every gradient. Turning this
# on would cut WPAC's share from ~54% to ~29% (at alpha=0.5) to buy the Indian
# Ocean a rise from 4% to 10% -- i.e. it trades the ONE basin that currently works
# (auc 0.728) for the ones that do not, on a hypothesis the data does not clearly
# support. It is implemented and unit-tested so it is a one-constant A/B for a
# follow-up run; 0.5 is the sensible value to try.
BASIN_BALANCE_ALPHA = 0.0

# (C) Calibration. Fitting 2 parameters on ~2,500 val frames is statistically
# trivial, but the val set is also what selects the epoch, so the fit is verified
# by a storm-level half-split cross-check before it is trusted. See
# fit_platt_scaling() and calibration_crosscheck().
CALIB_MAX_SLOPE = 50.0    # guards the perfectly-separable case from overflowing fc
CALIB_MAX_BIAS = 50.0
ECE_BINS = 15
# Grouped-by-storm cross-validation folds for that check. 5 means each fit sees 80%
# of the val storms, which is close enough to the final full-data fit for its
# held-out score to be a fair estimate of it.
CALIB_CV_FOLDS = 5

# (C3) Specificity targets for the recommended display thresholds. 0.85 means "at
# most 15% of non-intensifying storms are flagged", against v2's measured 36%.
SPECIFICITY_TARGETS = (0.80, 0.85, 0.90, 0.95)

# (A4) Size of the fixed, un-augmented training subsample used as an under/overfit
# probe each epoch. Small enough to cost ~2 s, large enough for a stable AUC.
TRAIN_PROBE_FRAMES = 2000

# =========================== v5 SWEEP CONFIG ================================
# Everything above this line is v3's config, unchanged, so config A really is a
# reproduction of v3 and not a re-tuned lookalike.

N_FOLDS = 4               # every storm gets exactly one out-of-fold prediction
SWEEP_EPOCHS = 18         # v3 peaked at epoch 14 and only overfitted after; 18 is
                          # enough headroom for the more heavily regularised configs
SWEEP_WARMUP = 2
USE_AMP = True            # fp16 autocast on the T4. ~1.8x, applied identically to
                          # every config, so it cannot bias a comparison.

# Aux-metadata branch (config D onward)
AUX_EMB_DIM = 32
AUX_DROP_P = 0.25         # per-sample probability of zeroing the whole aux branch
                          # during training. Two jobs: regularises the aux path so
                          # it cannot swamp the image path, and makes the trained
                          # checkpoint usable IMAGE-ONLY (mask=0) so it stays a
                          # drop-in for local_tool/detect.py, which has no metadata.
BASINS_ORDER = ("ATLN", "CPAC", "EPAC", "IO", "SH", "WPAC")
AUX_DIM = 8 + len(BASINS_ORDER)

# Physically-graded target (config E onward)
SOFT_LABEL_KT = 8.0       # y_soft = sigmoid((wmax - 64) / 8). 56 kt -> 0.27,
                          # 64 -> 0.50, 72 -> 0.73, 100 -> 0.99. Encodes that the
                          # label is genuinely uncertain within a few kt of the bar
                          # and certain far from it, which flat label smoothing
                          # cannot express.
WIND_AUX_WEIGHT = 0.3     # weight of the auxiliary peak-wind regression loss
WIND_NORM_KT = 100.0
WIND_CLIP_KT = 180.0

# Stronger augmentation used by configs C onward.
AUG_STRONG = {"brightness": 0.10, "contrast": 0.20, "zoom_log": 0.25,
              "shift": 0.12, "noise_std": 0.020, "cutout_frac": 0.18}
# Second, harder rung. C_reg cut the train/val AUC gap from 0.234 to 0.141 AND
# raised AUC, i.e. the model was still on the over-capacity side of the curve at
# AUG_STRONG, so the next question is how much further that goes before it costs
# accuracy. Measured, not assumed.
AUG_STRONGER = {"brightness": 0.14, "contrast": 0.28, "zoom_log": 0.32,
                "shift": 0.16, "noise_std": 0.030, "cutout_frac": 0.25}
AUG_MILD = {"brightness": AUG_BRIGHTNESS, "contrast": AUG_CONTRAST,
            "zoom_log": AUG_ZOOM_LOG, "shift": AUG_SHIFT,
            "noise_std": AUG_NOISE_STD, "cutout_frac": 0.0}

BOOT_REPS = 2000          # paired bootstrap over storms, for delta-AUC CIs

# The ablation. Each row adds exactly ONE change to the row above it, so a
# regression is attributable. `train_sel` is the ONLY thing that changes which
# frames are trained on; evaluation frames are always the clean ones.
SWEEP_CONFIGS = [
    dict(name="A_v3_repro",  train_sel="v3",    dropout=0.0, wd=1e-4,
         aug=AUG_MILD,   aux=False, soft=False, wind_aux=0.0, arch="resnet18"),
    dict(name="B_cleansel",  train_sel="clean", dropout=0.0, wd=1e-4,
         aug=AUG_MILD,   aux=False, soft=False, wind_aux=0.0, arch="resnet18"),
    dict(name="C_reg",       train_sel="clean", dropout=0.3, wd=5e-2,
         aug=AUG_STRONG, aux=False, soft=False, wind_aux=0.0, arch="resnet18"),
    dict(name="D_aux",       train_sel="clean", dropout=0.3, wd=5e-2,
         aug=AUG_STRONG, aux=True,  soft=False, wind_aux=0.0, arch="resnet18"),
    dict(name="E_soft_wind", train_sel="clean", dropout=0.3, wd=5e-2,
         aug=AUG_STRONG, aux=True,  soft=True,  wind_aux=WIND_AUX_WEIGHT,
         arch="resnet18"),
    # F isolates CAPACITY: identical to D_aux except the backbone is truncated
    # after layer3 (2.8M params instead of 11.2M). It deliberately does NOT also
    # change the target -- stacking two changes would make a regression
    # unattributable, which is the trap the A-E chain already risks.
    dict(name="F_small",     train_sel="clean", dropout=0.3, wd=5e-2,
         aug=AUG_STRONG, aux=True,  soft=False, wind_aux=0.0,
         arch="resnet18_l3"),
    # --- second round, chosen from what the first round actually measured ---
    # K: the train/val gap is still +0.137 after C_reg, and regularisation is the
    # only lever so far that improved BOTH the gap and the Indian Ocean. Push it.
    dict(name="K_regmax",    train_sel="clean", dropout=0.5, wd=1e-1,
         aug=AUG_STRONGER, aux=True, soft=False, wind_aux=0.0, arch="resnet18"),
    # J: D_aux scores 0.7529 WITH metadata but only 0.6993 image-only, which is
    # worse than C_reg's 0.7191 -- so the metadata-capable checkpoint is currently
    # a downgrade for local_tool/detect.py, which has no metadata to feed it.
    # Raising the aux-dropout rate forces the image path to stand on its own.
    dict(name="J_auxdrop50", train_sel="clean", dropout=0.3, wd=5e-2,
         aug=AUG_STRONG, aux=True,  soft=False, wind_aux=0.0, arch="resnet18",
         aux_drop=0.50),
]
# Restrict the sweep from the kernel environment without editing the file, e.g.
# CGP_ONLY_CONFIGS="A_v3_repro,B_cleansel". Empty means "all".
ONLY_CONFIGS = [s for s in os.environ.get("CGP_ONLY_CONFIGS", "").split(",") if s]
# Smoke mode: 1 fold, 2 epochs. Used by the offline dry run.
SMOKE = os.environ.get("CGP_SMOKE", "") == "1"

# ============================== v6 (MOSDAC) =================================
# Storm-key prefix for MOSDAC storms. v4 learned this the hard way with the
# Digital Typhoon AU mount, whose storm numbers coincide ~96% with WP's despite
# being an entirely independent counter: namespace the keys so a collision is
# STRUCTURALLY impossible rather than merely unlikely. MOSDAC keys are
# alphabetic ("MOS_FANI") so they cannot collide with Digital Typhoon's 6 digits
# or TCIR's YYYYNN<letter> even by accident.
MOSDAC_KEY_PREFIX = "MOS_"
MOSDAC_BASIN = "IO"

# Seeds for the arm x seed grid. Three is the minimum that gives a spread rather
# than a single difference; the whole point is to measure the noise floor in this
# run instead of importing a number from a previous session.
ARM_SEEDS = [int(s) for s in os.environ.get("CGP_ARM_SEEDS", "1337,2718,3141").split(",")]
ARMS = [a for a in os.environ.get("CGP_ARMS", "baseline,mosdac").split(",") if a]
# Which (arm, seed) produces the saved, calibrated, shippable checkpoint.
SHIP_ARM = os.environ.get("CGP_SHIP_ARM", "mosdac")

# Report-only null test, in the spirit of v4's au_vs_wp_track_null_test(). MOSDAC
# storms are all 2019+ and TCIR's Indian Ocean coverage ends in 2018 (VERIFIED
# against the real dataset_index.csv, whose IO rows are 100% source=='tcir' with
# storm ids spanning 2004..2018), so the expected overlap is ZERO. If the track
# matcher finds any, the assumption is wrong and the run says so loudly instead
# of silently training on the same storm twice under two different keys.
MOSDAC_NULL_MAX_FRAC = 0.02
# ============================================================================

MODEL_VERSION = "v6_mosdac"
# Which sweep config to ship. Overridable from the kernel environment so the
# winner can be selected without editing the file.
# C_reg is the config to ship FIRST: it is the only proven improvement that
# needs NO interface change anywhere downstream. It is image-only, so
# local_tool/detect.py and demo_kernel/render_demo.py keep working exactly as
# they are, and it measured +0.0143 storm AUC over v3's recipe out-of-fold
# (95% CI [+0.0018, +0.0265], P(better) 0.990) plus a much smaller train/val
# gap (+0.141 vs +0.234) and the best Indian Ocean number of any config.
# D_aux scores higher still (storm 0.7753) but only WITH lat/lon/date fed in,
# which detect.py cannot supply today -- its image-only mode is WORSE than
# C_reg (0.6993 vs 0.7191). That is what J_auxdrop50 in the v5c sweep is
# testing; if it lands, re-run this with that config instead.
FINAL_CONFIG_NAME = os.environ.get("CGP_FINAL_CONFIG", "C_reg")
# "image" -> calibration/thresholds are fitted on the image-only mode (drop-in for
# local_tool/detect.py, which has no metadata to supply). "meta" -> fitted on the
# mode that also receives position and date.
SHIP_MODE = os.environ.get("CGP_SHIP_MODE", "image")
FINAL_EPOCHS = int(os.environ.get("CGP_FINAL_EPOCHS", str(SWEEP_EPOCHS)))

RESULTS_PATH = os.path.join(OUT_DIR, "sweep_results.jsonl")
SUMMARY_PATH = os.path.join(OUT_DIR, "sweep_summary.json")
OOF_PATH = os.path.join(OUT_DIR, "oof_predictions.csv")
# ============================================================================

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
        "datasets; TCIR has no grade field. v3 measures the size of the resulting "
        "label disagreement (see dt_audit.n_label_flips_under_1min_equiv_bar) but "
        "deliberately does NOT change the rule."
    )


def jsonable(obj):
    """Recursively cast numpy scalars/arrays to plain Python types.

    detect.py and render_demo.py both do json.dumps() on ckpt['config'] and
    ckpt['val_metrics'] and would crash on a np.float32. Anything that goes into
    those two keys must pass through here.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None  # NaN/Inf are not valid JSON
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    return obj


def even_picks(n_eligible, k):
    """The v3 pick rule, factored out so both selection modes use it verbatim:
    k frames spread evenly across the eligible window, not the first k."""
    if n_eligible <= 0:
        return np.zeros(0, dtype=int)
    return np.unique(
        np.linspace(0, n_eligible - 1, min(k, n_eligible)).round().astype(int)
    )


def selection_flags(elig_v3_pos, elig_clean_pos, k=FRAMES_PER_STORM):
    """Return {positional_row -> (sel_v3, sel_clean)} for the UNION of the frames
    the two selection modes would pick.

    Both modes are indexed once and decoded once, so config A (v3 selection) and
    config B (clean selection) read from the same preloaded image array. Without
    the union the two configs would need two preload passes.

    elig_v3_pos    -- positional rows matching v3's `grade <= 2` / `Vmax <= 33`.
    elig_clean_pos -- the subset of those that occur BEFORE the storm first leaves
                      the early stage (i.e. genuine pre-intensification frames).
    """
    out = {}
    for p in even_picks(len(elig_v3_pos), k):
        out[int(elig_v3_pos[p])] = [True, False]
    for p in even_picks(len(elig_clean_pos), k):
        row = out.setdefault(int(elig_clean_pos[p]), [False, False])
        row[1] = True
    return out


def day_of_year(y, m, d):
    """Day-of-year, tolerant of the odd out-of-range day value in best-track data."""
    try:
        return int(datetime.date(int(y), int(m), min(28, int(d))).timetuple().tm_yday) \
            + (int(d) - min(28, int(d)))
    except Exception:
        return -1


# ----------------------------------------------------------------------------
# Dataset discovery
# ----------------------------------------------------------------------------
def find_dt_root():
    """Locate the Digital Typhoon mount.

    The exact input prefix varies between kernel runs, so walk instead of
    hardcoding. Never descend into the image tree -- it is ~86GB over hundreds of
    thousands of files.

    v5 CHANGE (real bug, caught by the offline dry run): v3 returned the FIRST
    directory whose name contains "typhoon", which is a coin flip as soon as more
    than one Digital Typhoon mount is attached. The v5 dry run has both the WP and
    the AU mounts present, and v3's rule silently indexed AU (40 storms) instead of
    WP (84) -- the run completed and produced plausible-looking numbers for the
    wrong dataset. v4's docstring names this hazard; v3's finder still had it.
    Collect every candidate, prefer the Western Pacific mount explicitly, and print
    what was found either way so a wrong pick is visible in the log.
    """
    cands = []
    for dirpath, dirnames, _ in os.walk(INPUT_ROOT):
        dirnames[:] = [d for d in dirnames if d.lower() not in ("image", "metadata")]
        for d in list(dirnames):
            if "typhoon" in d.lower():
                cands.append(os.path.join(dirpath, d))
                dirnames.remove(d)   # found the root; do not walk inside it
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
    for dirpath, dirnames, filenames in os.walk(INPUT_ROOT):
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
        "n_no_clean_early_frame": 0,
        "n_frames_eligible_v3": 0,
        "n_frames_eligible_clean": 0,
        "n_missing_h5": 0,
        "n_label_agrees_with_v1_grade_rule": 0,
        "n_label_disagrees_with_v1_grade_rule": 0,
        "label_disagreement_examples": [],
        # (D2) how many storms the 1-min-equivalent bar would flip, report only
        "n_label_flips_under_1min_equiv_bar": 0,
        "n_storms_labelled": 0,
    }

    # (D2) A 64 kt 1-minute bar (TCIR's) expressed in Digital Typhoon's 10-minute
    # units. Storms between this and 64 kt are the ones the two sources currently
    # disagree about.
    one_min_equiv_bar = TYPHOON_WIND_KT * WIND_10MIN_PER_1MIN

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
        # decades. Drop rather than mislabel.
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
        audit["n_storms_labelled"] += 1
        if int(wmax >= one_min_equiv_bar) != label:
            audit["n_label_flips_under_1min_equiv_bar"] += 1

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
        # v1-v4 used `grade <= 2` ANYWHERE in the track, which also matches the
        # DECAY tail (2 -> 3 -> 4 -> 5 -> 4 -> 3 -> 2). `first_cross` is the first
        # row at which the storm leaves the early stage; everything at or after it
        # -- including the whole post-peak tail -- is NOT an early-stage frame no
        # matter what its grade later returns to. See the module docstring.
        fname = df["file_1"].fillna("").astype(str).str.strip()
        elig = ((grade <= DT_EARLY_GRADE_MAX) & (fname != "")).to_numpy()
        crossed = (grade > DT_EARLY_GRADE_MAX).fillna(False).to_numpy()
        first_cross = int(np.argmax(crossed)) if crossed.any() else len(df)

        elig_v3_pos = np.where(elig)[0]
        elig_clean_pos = elig_v3_pos[elig_v3_pos < first_cross]
        audit["n_frames_eligible_v3"] += int(len(elig_v3_pos))
        audit["n_frames_eligible_clean"] += int(len(elig_clean_pos))
        if len(elig_v3_pos) == 0:
            audit["n_no_early_frame"] += 1
            continue
        if len(elig_clean_pos) == 0:
            # The storm was already past the early stage on its first imaged row,
            # so it has no honest early-stage frame at all. v3 would have trained
            # AND validated on its decay tail. Dropped from every config so all
            # configs see the same storm set.
            audit["n_no_clean_early_frame"] += 1
            continue

        for p, (sel_v3, sel_clean) in sorted(selection_flags(elig_v3_pos,
                                                             elig_clean_pos).items()):
            rec = df.iloc[int(p)]
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
                    "sel_v3": bool(sel_v3),
                    "sel_clean": bool(sel_clean),
                    # aux metadata -- parsed by v1-v4 for the dedup track match,
                    # never given to the model. Not leaky: position and date are
                    # known at forecast time, unlike wind/pressure.
                    "lat": float(pd.to_numeric(rec["lat"], errors="coerce")),
                    "lon": float(pd.to_numeric(rec["lng"], errors="coerce")),
                    "doy": day_of_year(rec["year"], rec["month"], rec["day"]),
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

    # (D2) Report only. See the block in main() that interprets this number.
    n_lab = max(1, audit["n_storms_labelled"])
    print(f"[A] WIND-CONVENTION AUDIT (report only, labels unchanged): "
          f"{audit['n_label_flips_under_1min_equiv_bar']}/{audit['n_storms_labelled']} "
          f"({audit['n_label_flips_under_1min_equiv_bar'] / n_lab:.1%}) Digital Typhoon "
          f"storms have peak 10-min wind in [{one_min_equiv_bar:.1f}, {TYPHOON_WIND_KT:.1f}) kt, "
          f"i.e. they are NEGATIVE under the current rule but would be POSITIVE if "
          f"the bar were harmonised to TCIR's 1-minute convention.")
    return pd.DataFrame(rows), tracks, audit


# ----------------------------------------------------------------------------
# Source B -- TCIR / TCSA
# ----------------------------------------------------------------------------
def check_tcir_id_pattern(info):
    """Validate the assumed TCIR storm-ID shape BEFORE it is used for dedup keys.

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
            "'YYYY' + 'NN' + one basin letter shape (as seen in '200401W', '201103I').\n"
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
    """'200315W' -> (2003, 15). None if unparseable."""
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
        "n_no_clean_early_frame": 0,
        "n_frames_eligible_v3": 0,
        "n_frames_eligible_clean": 0,
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
        # Same decay-tail fix as the Digital Typhoon side: `Vmax <= 33` matches the
        # end of the track as well as the start, so truncate at the first crossing.
        vm = g["vmax"].to_numpy()
        elig = np.isfinite(vm) & (vm <= TCIR_EARLY_VMAX_KT)
        crossed = np.isfinite(vm) & (vm > TCIR_EARLY_VMAX_KT)
        first_cross = int(np.argmax(crossed)) if crossed.any() else len(g)

        elig_v3_pos = np.where(elig)[0]
        elig_clean_pos = elig_v3_pos[elig_v3_pos < first_cross]
        audit["n_frames_eligible_v3"] += int(len(elig_v3_pos))
        audit["n_frames_eligible_clean"] += int(len(elig_clean_pos))
        if len(elig_v3_pos) == 0:
            audit["n_no_early_frame"] += 1
            continue
        if len(elig_clean_pos) == 0:
            audit["n_no_clean_early_frame"] += 1
            continue

        for p, (sel_v3, sel_clean) in sorted(selection_flags(elig_v3_pos,
                                                             elig_clean_pos).items()):
            rec = g.iloc[int(p)]
            ts = rec["time"]
            ts = int(ts) if np.isfinite(ts) else -1
            rows.append(
                {
                    "storm_key": storm_id,
                    "source": "tcir",
                    "basin": basin,
                    "label": label,
                    "h5_path": "",
                    "frame_ref": int(rec["frame_ref"]),
                    "max_wind_kt": wmax,
                    "sel_v3": bool(sel_v3),
                    "sel_clean": bool(sel_clean),
                    "lat": float(rec["lat"]),
                    "lon": float(rec["lon"]),
                    "doy": (day_of_year(ts // 1000000, (ts // 10000) % 100,
                                        (ts // 100) % 100) if ts > 0 else -1),
                }
            )

    print(f"[B] {audit['n_storms_total']} storms / {audit['n_frames_total']} frames in file")
    print(f"[B] storms dropped -- no_vmax_data={audit['n_no_vmax_data']} "
          f"no_early_frame={audit['n_no_early_frame']} (mixed_region warnings="
          f"{audit['n_mixed_region']})")
    return pd.DataFrame(rows), work, audit


# ----------------------------------------------------------------------------
# Source C -- MOSDAC / INSAT-3DR (v6)
# ----------------------------------------------------------------------------
def find_mosdac_pack():
    """Locate the extracted MOSDAC frame pack in /kaggle/input.

    The pack is a Kaggle dataset holding mosdac_frames.npy (uint8 N x 224 x 224,
    already through kelvin_to_display_u8 + resize_u8) and mosdac_index.csv. The
    decode happens OFF Kaggle because the INSAT source files are ~24 MB each and
    the extracted frames are ~50 KB, so shipping frames keeps the kernel input
    at a few megabytes instead of 3.5 GB.
    """
    for dirpath, _dirnames, filenames in os.walk(INPUT_ROOT):
        if "mosdac_index.csv" in filenames and "mosdac_frames.npy" in filenames:
            return dirpath
    return None


def build_mosdac_index(pack_dir):
    """Read the pack into train.py's index schema and re-derive every label.

    The pack's own `label` column is NOT trusted blindly: it is recomputed here
    from max_wind_kt against TYPHOON_WIND_KT, so the label rule lives in exactly
    one place (this file) for all three sources. A disagreement is a hard error --
    it would mean the extractor and the trainer are applying different bars.

    WIND CONVENTION. max_wind_kt is IMD's 3-minute sustained peak. Digital Typhoon
    is JMA 10-minute and TCIR is JTWC/NHC 1-minute, so this is a THIRD convention.
    64 kt 3-minute is IMD's own Very Severe Cyclonic Storm boundary -- the direct
    analogue of JMA's Typhoon (64 kt 10-min) and JTWC/NHC's Hurricane (64 kt
    1-min) -- so the project's existing rule ("64 kt as each agency's own category
    threshold", see label_note()) transfers unchanged. The size of the resulting
    disagreement is AUDITED and printed, exactly as build_dt_index() does with
    WIND_10MIN_PER_1MIN, and deliberately not silently corrected.
    """
    idx = pd.read_csv(os.path.join(pack_dir, "mosdac_index.csv"))
    frames = np.load(os.path.join(pack_dir, "mosdac_frames.npy"))
    need = {"storm_key", "source", "basin", "label", "frame_ref", "max_wind_kt",
            "lat", "lon", "doy"}
    missing = need - set(idx.columns)
    if missing:
        raise SystemExit(f"MOSDAC pack index is missing columns {sorted(missing)}")
    if len(frames) < len(idx):
        raise SystemExit(
            f"MOSDAC pack is inconsistent: index has {len(idx)} rows but the frame "
            f"array has only {len(frames)}; frame_ref would point past the end.")
    refs = idx["frame_ref"].to_numpy()
    if refs.min() < 0 or refs.max() >= len(frames):
        raise SystemExit(f"MOSDAC frame_ref out of range [0, {len(frames)})")

    wmax = pd.to_numeric(idx["max_wind_kt"], errors="coerce").to_numpy(np.float64)
    recomputed = (wmax >= TYPHOON_WIND_KT).astype(int)
    if not np.array_equal(recomputed, idx["label"].to_numpy().astype(int)):
        bad = idx.loc[recomputed != idx["label"].to_numpy().astype(int),
                      ["storm_key", "max_wind_kt", "label"]].drop_duplicates()
        raise SystemExit(
            "MOSDAC pack labels disagree with this script's own >= "
            f"{TYPHOON_WIND_KT} kt rule:\n{bad.to_string()}")

    idx = idx.copy()
    idx["source"] = "mosdac"
    idx["basin"] = MOSDAC_BASIN
    idx["h5_path"] = ""
    idx["sel_v3"] = True
    idx["sel_clean"] = True
    if not idx["storm_key"].astype(str).str.startswith(MOSDAC_KEY_PREFIX).all():
        raise SystemExit(f"every MOSDAC storm_key must start with "
                         f"{MOSDAC_KEY_PREFIX!r}; the namespace is what makes a "
                         f"collision with a Digital Typhoon or TCIR key impossible")

    storms = idx.drop_duplicates("storm_key")
    bar_1min = TYPHOON_WIND_KT / (1.11 / 1.05)    # 64 kt 1-min, in 3-min units
    bar_10min = TYPHOON_WIND_KT / (1.0 / 1.05)    # 64 kt 10-min, in 3-min units
    sw = pd.to_numeric(storms["max_wind_kt"], errors="coerce").to_numpy(np.float64)
    audit = {
        "n_frames": int(len(idx)),
        "n_storms": int(storms["storm_key"].nunique()),
        "n_positive_storms": int((storms["label"] == 1).sum()),
        "n_negative_storms": int((storms["label"] == 0).sum()),
        "n_positive_frames": int((idx["label"] == 1).sum()),
        "n_negative_frames": int((idx["label"] == 0).sum()),
        "wind_convention": "IMD 3-minute sustained (vs JMA 10-min for Digital "
                           "Typhoon, JTWC/NHC 1-min for TCIR)",
        "bar_applied_kt": TYPHOON_WIND_KT,
        "bar_if_harmonised_to_1min_kt": round(float(bar_1min), 2),
        "bar_if_harmonised_to_10min_kt": round(float(bar_10min), 2),
        "n_storms_in_disputed_band": int(((sw >= bar_1min) & (sw < bar_10min)).sum()),
        "storms_in_disputed_band": sorted(
            storms.loc[(sw >= bar_1min) & (sw < bar_10min), "storm_key"].tolist()),
    }
    print(f"[C] MOSDAC pack: {audit['n_frames']} frames / {audit['n_storms']} storms "
          f"({audit['n_positive_storms']} positive / {audit['n_negative_storms']} "
          f"negative storms; {audit['n_positive_frames']}/{audit['n_negative_frames']} "
          f"frames)")
    print(f"[C] BOTH-SIDES-OF-THE-LABEL CHECK: "
          f"{'PASS' if audit['n_positive_storms'] > 0 and audit['n_negative_storms'] > 0 else 'FAIL'}"
          f" -- MOSDAC frames must never carry only one label, or 'looks like "
          f"INSAT' becomes a free predictor of that class.")
    if audit["n_positive_storms"] == 0 or audit["n_negative_storms"] == 0:
        raise SystemExit("MOSDAC frames are single-class; refusing to train on a "
                         "source that is perfectly correlated with the label.")
    print(f"[C] WIND-CONVENTION AUDIT (report only, labels unchanged): "
          f"{audit['n_storms_in_disputed_band']}/{audit['n_storms']} MOSDAC storms "
          f"have peak 3-min wind in [{bar_1min:.1f}, {bar_10min:.1f}) kt, i.e. the "
          f"three agencies' 64 kt bars disagree about them: "
          f"{audit['storms_in_disputed_band']}")
    cols = ["storm_key", "source", "basin", "label", "h5_path", "frame_ref",
            "max_wind_kt", "sel_v3", "sel_clean", "lat", "lon", "doy"]
    return idx[cols].reset_index(drop=True), frames, audit


def mosdac_overlap_null_test(mos_index, mos_tracks, tcir_index, tcir_work, dt_tracks):
    """Report-only: does any MOSDAC storm match an existing TCIR or Digital
    Typhoon storm by best-track position?

    Modelled on v4's au_vs_wp_track_null_test(). Deliberately position-match ONLY,
    with no ID-key detector: MOSDAC storms are keyed by NAME and the other two by
    number, so there is nothing to key-match, and inventing one would risk
    deleting real storms on a coincidence.

    The expected answer is ZERO -- MOSDAC storms are 2019+ and TCIR's IO coverage
    ends in 2018 -- which makes this a genuine null test of the assumption rather
    than a formality. A non-zero result means the same physical storm would sit in
    the dataset twice under two keys and could straddle the train/val boundary.
    """
    print("\n" + "=" * 72)
    print("MOSDAC OVERLAP NULL TEST (position match only -- expected result: ZERO)")
    print("=" * 72)
    if not mos_tracks:
        print("  no MOSDAC track table shipped; skipping (cannot verify)")
        return {"skipped": True}

    tcir_io = tcir_index[tcir_index["basin"] == "IO"]
    tcir_io_storms = set(tcir_io["storm_key"].unique())
    tw = tcir_work[tcir_work["storm_id"].isin(tcir_io_storms)]
    # timestamp -> [(lat, lon, storm_id)], the same shape dt_tracks uses.
    tcir_ts = defaultdict(list)
    for sid, ts, la, lo in zip(tw["storm_id"].to_numpy(), tw["time"].to_numpy(),
                               tw["lat"].to_numpy(), tw["lon"].to_numpy()):
        if np.isfinite(ts) and np.isfinite(la) and np.isfinite(lo):
            tcir_ts[int(ts)].append((float(la), float(lo), str(sid)))

    hits = {}
    for name, pts in mos_tracks.items():
        votes = Counter()
        for p in pts:
            ts = int(p["stamp"])
            for other, tbl in (("tcir", tcir_ts), ("digital_typhoon", dt_tracks)):
                for la, lo, sid in tbl.get(ts, ()):
                    if (abs(la - p["lat"]) <= DEDUP_MATCH_DEG
                            and abs(lo - p["lon"]) <= DEDUP_MATCH_DEG):
                        votes[(other, sid)] += 1
        if votes:
            best, n = votes.most_common(1)[0]
            if n >= DEDUP_MIN_MATCH_POINTS:
                hits[name] = {"matched": best, "n_points": int(n)}

    n_mos = len(mos_tracks)
    frac = len(hits) / max(1, n_mos)
    print(f"  {len(hits)} of {n_mos} MOSDAC storms position-match an existing "
          f"storm (<= {DEDUP_MATCH_DEG} deg at >= {DEDUP_MIN_MATCH_POINTS} shared "
          f"synoptic timestamps) = {frac:.1%}")
    for k, v in sorted(hits.items())[:15]:
        print(f"      {k} -> {v['matched']} ({v['n_points']} points)")
    if frac > MOSDAC_NULL_MAX_FRAC:
        print("\n" + "!" * 72)
        print("MOSDAC/TCIR SEPARATION ASSUMPTION IS WRONG. The same physical storm")
        print("is present under two different keys and can straddle the train/val")
        print("boundary. Drop the matched MOSDAC storms before trusting this run.")
        print("!" * 72 + "\n")
    else:
        print("  -> null test PASSES: the MOSDAC storms are genuinely new storms, "
              "not re-keyed copies of storms already in the dataset.")
    print("=" * 72 + "\n")
    return {"n_mosdac_storms": n_mos, "n_matched": len(hits),
            "matched_fraction": float(frac), "matches": hits,
            "passes": bool(frac <= MOSDAC_NULL_MAX_FRAC)}


def source_shortcut_probe(val_sources, val_labels, val_storm_keys):
    """How much AUC is available from KNOWING THE SOURCE and nothing else?

    MOSDAC's storm list is IMD's named-cyclone list and IMD only names a system
    once it reaches 34 kt, so MOSDAC storms skew positive relative to the TCIR
    Indian Ocean population. A network that learned only to recognise INSAT's
    sensor signature would therefore still beat chance on a POOLED Indian Ocean
    validation set without having learned anything about storms.

    This scores the degenerate classifier "predict 1 iff the frame came from
    MOSDAC" on the Indian Ocean validation storms. Its AUC is the ceiling on how
    much of any pooled-IO improvement could be pure source leakage. It is the
    reason every IO number in this run is ALSO reported within each source.
    """
    st_ids, st_pos = np.unique(val_storm_keys, return_inverse=True)
    st_y = np.zeros(len(st_ids), dtype=int)
    st_is_mos = np.zeros(len(st_ids), dtype=float)
    for i in range(len(val_labels)):
        st_y[st_pos[i]] = val_labels[i]
        if val_sources[i] == "mosdac":
            st_is_mos[st_pos[i]] = 1.0
    out = {
        "n_io_val_storms": int(len(st_ids)),
        "n_mosdac_val_storms": int(st_is_mos.sum()),
        "n_tcir_val_storms": int((1 - st_is_mos).sum()),
        "positive_rate_mosdac": (float(st_y[st_is_mos == 1].mean())
                                 if (st_is_mos == 1).any() else None),
        "positive_rate_tcir": (float(st_y[st_is_mos == 0].mean())
                               if (st_is_mos == 0).any() else None),
        "auc_from_source_alone": auc_of(st_y, st_is_mos),
    }
    print(f"  SOURCE-SHORTCUT PROBE on {out['n_io_val_storms']} IO val storms: "
          f"positive rate mosdac={out['positive_rate_mosdac']} "
          f"tcir={out['positive_rate_tcir']}  -> a classifier that knows ONLY the "
          f"source scores storm AUC {out['auc_from_source_alone']}")
    print(f"  Any pooled-IO gain smaller than what this buys is not evidence of "
          f"real skill. Read the WITHIN-SOURCE numbers instead.")
    return out


def paired_bootstrap_auc(y, p_a, p_b, reps=BOOT_REPS, seed=SEED):
    """Bootstrap CI on AUC(p_b) - AUC(p_a) over the SAME units (storms).

    Resamples storms, not frames: two frames of one storm share a label and are
    near-duplicates, so bootstrapping frames would understate the interval by
    roughly the frames-per-storm factor. Paired -- both arms are scored on the
    same resample -- so the shared difficulty of a given draw of storms cancels,
    which is what makes a small delta measurable at all.
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or len(y) < 8:
        return {"delta": None, "reason": "too few storms or single class",
                "n": int(len(y))}
    rng = np.random.RandomState(seed)
    base = auc_of(y, p_b) - auc_of(y, p_a)
    d = []
    for _ in range(reps):
        s = rng.randint(0, len(y), len(y))
        if len(np.unique(y[s])) < 2:
            continue
        d.append(auc_of(y[s], p_b[s]) - auc_of(y[s], p_a[s]))
    if not d:
        return {"delta": float(base), "reason": "no usable resamples",
                "n": int(len(y))}
    d = np.asarray(d)
    return {"n_storms": int(len(y)), "auc_a": auc_of(y, p_a),
            "auc_b": auc_of(y, p_b), "delta": float(base),
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "p_better": float((d > 0).mean()), "n_reps": int(len(d))}


# ----------------------------------------------------------------------------
# Cross-dataset de-duplication (the correctness-critical part)
#
# CARRIED OVER FROM v2 UNCHANGED. Do not tune it. Both datasets ultimately draw
# their Western Pacific storms from the same real best-track record, so the SAME
# physical storm can appear in both under two different ID schemes; if that storm
# landed in A's train split and B's val split, val AUC would be inflated by
# memorisation.
# ----------------------------------------------------------------------------
def dedup_wpac(dt_index, tcir_index, tcir_work, dt_tracks):
    """Find TCIR WPAC storms that are the same physical storm as a Digital Typhoon
    storm, and return the set of TCIR storm_keys to drop entirely.

    Digital Typhoon wins every collision: 512x512 native resolution vs TCIR's
    128x128, and many more early-stage frames per storm.

    TWO independent detectors are run and their results are UNIONed:

      1. ID KEY  -- (year, sequence_number) parsed out of both ID schemes.
         Cheap and exact when the two agencies agree on numbering. BUT: Digital
         Typhoon's YYYYNN is the JMA *international* number, assigned only to
         systems reaching tropical-storm strength, while TCIR's WPAC numbering is
         the ATCF/JTWC one, which also numbers systems JMA never named. The two
         therefore DRIFT APART within a season. Because Digital Typhoon numbers ~26
         storms in every year TCIR covers, expect this detector to fire on very
         nearly every TCIR WPAC storm; a near-100% hit rate here is the EXPECTED
         reading, not a bug.

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
# Image loading and the model input pipeline
# ----------------------------------------------------------------------------
def kelvin_to_display_u8(arr):
    """Brightness temperature (K) -> uint8 display frame, cold = bright.

    Shared by BOTH sources so a Digital Typhoon pixel and a TCIR pixel of the same
    temperature become the same grey level. Inverting so cold (high, convective)
    cloud tops render bright is the conventional IR rendering, is what the demo
    video shows, and is what local_tool/detect.py assumes.
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
    """TCIR: images[row] is (128,128,4); channel 0 is IR1 in Kelvin (verified by
    the tcsa_probe run: ch0 spans ~195-301 K, ch1 is water vapour, ch2/ch3 are
    reflectance-scaled).

    Channels 1-3 are deliberately unused -- this model stays 1-channel so it
    remains loadable by local_tool/detect.py's build_model().
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


def to_unit(u8_batch):
    """uint8 (N,H,W) tensor -> float (N,1,IMG_SIZE,IMG_SIZE) in [0,1].

    Split out from normalisation (v2 did both in one step) for two reasons: the
    augmentation has to happen in this [0,1] DISPLAY domain to be physically
    interpretable, and doing the uint8->float cast on whatever device the tensor
    already lives on lets the training loop ship 1 byte/pixel to the GPU instead
    of 4 and do the cast there.
    """
    t = u8_batch.unsqueeze(1).float().div_(255.0)
    if t.shape[-1] != IMG_SIZE or t.shape[-2] != IMG_SIZE:
        t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return t


def normalize(x01):
    """[0,1] display tensor -> the exact normalisation detect.py applies."""
    return (x01 - NORM_MEAN) / NORM_STD


def to_model_input(u8_batch):
    """Full inference-path preprocessing, byte-for-byte what detect.py does.
    Used for validation and for the train-subset probe -- never augmented."""
    return normalize(to_unit(u8_batch))


def augment_batch(x01, gen, aug=None):
    """(B1) Batched augmentation in the [0,1] display domain.

    v5: `aug` is a strength dict (AUG_MILD / AUG_STRONG) so the sweep can vary it
    per config. Passing None reproduces v3's mild settings exactly.

    `gen` is a CPU torch.Generator. All randomness is drawn on the CPU as N-element
    vectors and then moved to x01's device: that is a few hundred bytes per batch,
    and it makes the augmentation bit-reproducible from a seed regardless of
    whether the run is on a GPU or on a laptop, which is what lets it be tested
    offline at all.

    WHY THESE FOUR, given the diagnosis:

    * brightness/contrast -- a linear remap of the display scale is exactly what a
      difference in IR sensor calibration looks like. The two sources are different
      instruments (GMS/MTSAT/Himawari 1978-2023 for Digital Typhoon, a GridSat
      composite for TCIR), and even within Digital Typhoon the satellite changes
      four times. The measured 0.728-vs-0.598 source gap is consistent with the
      model keying on absolute grey level; this forces it to key on structure.

    * zoom -- hedges the leading suspect for the source gap: the two datasets may
      cover different physical areas per frame, which after the common resize to
      224 would put the same storm at different apparent scales. That suspicion is
      NOT yet confirmed (see radial_scale_diagnostic, which measures it this run),
      so rather than apply an unverified correction this makes the model tolerant
      to a scale error of up to about +/-14% in either direction.

    * shift -- kept SMALL. Both datasets are storm-centric crops, so the centre is
      itself signal; this only absorbs centring-convention differences.

    * noise -- ~1.1 K of Gaussian, which is realistic sensor noise, and stops the
      model reading meaning into single-pixel detail that TCIR's 128->224 upsample
      cannot actually resolve.

    NOT included: flips. See FrameDataset.
    """
    p = AUG_MILD if aug is None else aug
    n = x01.shape[0]
    dev = x01.device

    def rnd(lo, hi):
        return (torch.rand(n, generator=gen) * (hi - lo) + lo).to(dev)

    # ---- geometric: isotropic zoom + small translation ----
    # affine_grid's theta maps OUTPUT coordinates back into the INPUT, so the
    # scale entry is 1/zoom: a theta scale below 1 samples a smaller region of the
    # input, which magnifies the storm.
    zoom = torch.exp(rnd(-p["zoom_log"], p["zoom_log"]))
    tx = rnd(-p["shift"], p["shift"])
    ty = rnd(-p["shift"], p["shift"])
    theta = torch.zeros(n, 2, 3, device=dev, dtype=x01.dtype)
    theta[:, 0, 0] = 1.0 / zoom
    theta[:, 1, 1] = 1.0 / zoom
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(x01.shape), align_corners=False)
    # padding_mode='border' replicates the edge pixels. 'zeros' would paint a hard
    # black rim on every zoomed-out frame, which is a synthetic feature the model
    # could learn to detect instead of the storm.
    x01 = F.grid_sample(x01, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)

    # ---- photometric: gain about mid-grey, then offset ----
    gain = rnd(1.0 - p["contrast"], 1.0 + p["contrast"]).view(n, 1, 1, 1)
    off = rnd(-p["brightness"], p["brightness"]).view(n, 1, 1, 1)
    x01 = (x01 - 0.5) * gain + 0.5 + off

    # ---- cutout (v5, off in the mild preset) ----
    # One square of mid-grey per frame. Occlusion regularisation: the model cannot
    # rely on any single region of the cloud field always being visible, which is
    # the standard remedy for a network that memorises specific training storms
    # (v3's train-probe AUC ran to 0.92 against a val AUC of 0.70). Filled with 0.5
    # rather than 0.0 because 0.0 in the display domain is a hot 300 K surface,
    # i.e. a physically meaningful value the model would learn to read.
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

    # ---- sensor noise ----
    if p["noise_std"] > 0:
        # Generated on the CPU and shipped over, like the parameters above, so the
        # whole augmentation is reproducible from one seed on any device. Measured
        # cost: 6 ms per batch of 32, i.e. ~45 s across a full 24-epoch run -- ~4%,
        # which is not worth a separate CUDA generator and a device-dependent path.
        noise = torch.randn(x01.shape, generator=gen).to(dev) * p["noise_std"]
        x01 = x01 + noise

    # Clamp back into the display range: the model must never see a value detect.py
    # could not produce from a real uint8 frame.
    return x01.clamp_(0.0, 1.0)


class FrameDataset(Dataset):
    """Frames are pre-decoded into RAM once (see preload_all), so this is just an
    indexed view plus the cheap CPU-side part of the augmentation.

    Returns (image_u8, label, weight). The weight is the basin-balance weight (all
    1.0 unless BASIN_BALANCE_ALPHA is nonzero) and is unused on the val side.
    """

    def __init__(self, images_u8, labels, weights, augment,
                 aux=None, soft=None, wind=None):
        self.images = images_u8
        self.labels = labels
        self.weights = weights
        self.augment = augment
        n = len(labels)
        # Always present so the collate shape is constant; the aux branch is only
        # consulted when the config asks for it.
        self.aux = np.zeros((n, AUX_DIM), np.float32) if aux is None else aux
        self.soft = labels.astype(np.float32) if soft is None else soft
        self.wind = np.zeros(n, np.float32) if wind is None else wind

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augment:
            # Rotations only -- NOT flips. A mirror flip reverses the storm's
            # rotational chirality, which is physically fixed by hemisphere and is
            # exactly the structure the model should key on. (This dataset contains
            # BOTH chiralities for real, since TCIR's SH basin spins the other way
            # -- which is a reason to keep flips out, not to allow them: a flipped
            # SH storm is a physically impossible NH storm.)
            #
            # Done here rather than in augment_batch because a 90-degree rotation of
            # a 224x224 uint8 array is 50 KB of memcpy, whereas doing it per-sample
            # on the GPU would need the batch split into four groups.
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img = np.rot90(img, k).copy()
        # float32 TENSORS, not Python floats. default_collate turns a list of
        # Python floats into a float64 batch; a float64 target then makes the loss
        # float64, and autograd's CUDA backward rejects that with
        # "Found dtype Double but expected Float". CPU autograd promotes silently
        # instead, so this fails ONLY on the GPU -- which is exactly how it got
        # through the offline dry run and killed a real Kaggle run at config
        # H_soft_only (the first config whose loss included the fp64 wind target).
        return (torch.from_numpy(img),
                torch.tensor(self.labels[i], dtype=torch.float32),
                torch.tensor(self.weights[i], dtype=torch.float32),
                torch.from_numpy(self.aux[i]),
                torch.tensor(self.soft[i], dtype=torch.float32),
                torch.tensor(self.wind[i], dtype=torch.float32))


def preload_all(index, tcir_h5_path, mosdac_frames=None):
    """Decode every frame ONCE, resized, into a single uint8 array in RAM.

    ~12k frames at 224x224 uint8 == ~620MB, fine for Kaggle RAM, and it turns 24
    epochs of HDF5 reads off two network-mounted datasets into one pass.

    TCIR is done FIRST and deliberately: it is the unknown-latency source (random
    row access into one 20GB file). Its rows are read in ASCENDING frame order so
    the access pattern is as close to sequential as it can be. An ETA is printed
    after 100 frames of each pass, so a pathologically slow mount is caught in the
    first minute rather than after half an hour.
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

    # ---- pass 3: MOSDAC / INSAT (v6) ----
    # No decode: the pack already holds 224x224 uint8 frames produced by the SAME
    # kelvin_to_display_u8() + resize_u8() this file defines (mosdac_frame.py
    # imports them from here rather than re-implementing them, so they cannot
    # drift). The shape is asserted rather than trusted -- a silently mis-shaped
    # array would otherwise be broadcast into `out` and train on garbage.
    mo_pos = np.where(sources == "mosdac")[0]
    if len(mo_pos):
        if mosdac_frames is None:
            raise SystemExit("index contains mosdac rows but no frame array was "
                             "passed to preload_all()")
        if mosdac_frames.shape[1:] != (IMG_SIZE, IMG_SIZE):
            raise SystemExit(f"MOSDAC frames are {mosdac_frames.shape[1:]}, "
                             f"expected ({IMG_SIZE}, {IMG_SIZE})")
        if mosdac_frames.dtype != np.uint8:
            raise SystemExit(f"MOSDAC frames are {mosdac_frames.dtype}, expected uint8")
        for pos in mo_pos:
            out[pos] = mosdac_frames[int(index["frame_ref"].iat[int(pos)])]
        print(f"[C] MOSDAC: {len(mo_pos)} frames copied from the pack (already "
              f"through kelvin_to_display_u8 + resize_u8)")

    print(f"Preload total failures: {int((~ok).sum())}/{n}")
    return out, ok


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model():
    """resnet18, ImageNet-pretrained, adapted to 1-channel input, 1 logit out.

    Byte-for-byte the same architecture v1 and v2 saved, which is what makes this
    checkpoint a drop-in replacement for local_tool/detect.py. NOTHING in v3 adds
    a module, a buffer or a parameter -- the calibration is folded into fc's
    existing weight and bias (see fold_calibration_into_fc) precisely so that
    net.load_state_dict(ckpt['state_dict']) stays a STRICT load over an unmodified
    resnet18.
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


def build_aux_features(index):
    """(v5, config D onward) The tabular metadata the pipeline already parsed but
    never used as a model input.

    WHY THESE AND ONLY THESE. The intersection of what BOTH sources carry, minus
    anything that reveals the answer. Digital Typhoon also has `wind`, `pressure`
    and the 30/50 kt wind-radius fields: all four are the storm's CURRENT or FUTURE
    intensity and are correctly excluded, here as everywhere else in the project.
    Position and calendar date are known to a forecaster at the moment of the
    forecast, so they are legitimate inputs, and they stand in for the two
    environmental quantities that dominate real intensity guidance: sea-surface
    temperature (which latitude and season proxy) and basin climatology.

    `year` is deliberately NOT included. It would let the model separate satellite
    generations (GMS/MTSAT/Himawari) and read the dataset's era rather than the
    storm.

    Returns (aux (N, AUX_DIM) float32, ok (N,) bool).
    """
    lat = pd.to_numeric(index["lat"], errors="coerce").to_numpy(np.float64)
    lon = pd.to_numeric(index["lon"], errors="coerce").to_numpy(np.float64)
    doy = pd.to_numeric(index["doy"], errors="coerce").to_numpy(np.float64)
    basin = index["basin"].astype(str).to_numpy()

    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(doy) & (doy > 0)
    lat_s = np.nan_to_num(lat, nan=0.0)
    lon_s = np.nan_to_num(lon, nan=0.0)
    doy_s = np.nan_to_num(doy, nan=1.0)

    lat_r = np.radians(np.clip(lat_s, -70, 70))
    lon_r = np.radians(lon_s)
    ang = 2.0 * np.pi * doy_s / 365.25
    hemi = np.sign(lat_s)
    hemi[hemi == 0] = 1.0

    cols = [
        np.sin(lat_r),                 # signed latitude, smooth through the equator
        np.abs(lat_s) / 40.0,          # distance from the equator == SST proxy
        hemi,                          # hemisphere (fixes the storm's chirality)
        np.sin(lon_r), np.cos(lon_r),  # longitude, circular so 179E ~ 181E
        np.sin(ang), np.cos(ang),      # calendar season
        np.sin(ang) * hemi,            # HEMISPHERIC season: September is peak
                                       # season north of the equator and the quiet
                                       # season south of it, so the raw calendar
                                       # angle alone is ambiguous across basins.
    ]
    for b in BASINS_ORDER:
        cols.append((basin == b).astype(np.float64))
    aux = np.stack(cols, axis=1).astype(np.float32)
    aux[~ok] = 0.0
    return aux, ok


class CycloneNet(nn.Module):
    """resnet18 trunk (+ optional aux-metadata branch, + optional peak-wind head).

    With aux=False and dropout=0 this is arithmetically the same function as v3's
    plain resnet18: `self.fc` is applied to the same 512-d pooled feature vector.
    That is what makes config A a real reproduction rather than a lookalike.

    The aux branch is gated by a per-sample MASK rather than baked in, so a single
    trained checkpoint serves two deployment modes:
        mask=1 -> image + position/date  (best accuracy)
        mask=0 -> image only             (what local_tool/detect.py can supply today)
    Both are scored separately in the sweep so the cost of metadata-free operation
    is a measured number, not an assumption.

    fc stays a single Linear over [image_features | aux_embedding], so
    (a) Platt calibration still folds into fc exactly as in v3, and
    (b) detect.py's CAM (which weights layer4's feature map by fc.weight) still
        works using the first 512 columns of fc.weight.
    """

    def __init__(self, arch="resnet18", use_aux=False, dropout=0.0, use_wind=False):
        super().__init__()
        base, pretrained = build_model()
        self.pretrained = pretrained
        self.arch = arch
        if arch == "resnet18_l3":
            # Truncate after layer3: 2.8M parameters instead of 11.2M. layer4 alone
            # is 8.4M of resnet18's 11.2M, so dropping it is the single biggest
            # capacity cut available without changing the family.
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                      base.layer1, base.layer2, base.layer3)
            feat_dim = 256
        else:
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                      base.layer1, base.layer2, base.layer3, base.layer4)
            feat_dim = 512
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_aux = use_aux
        if use_aux:
            self.aux_mlp = nn.Sequential(
                nn.Linear(AUX_DIM, 64), nn.ReLU(inplace=True),
                nn.Linear(64, AUX_EMB_DIM), nn.ReLU(inplace=True),
            )
            head_in = feat_dim + AUX_EMB_DIM
        else:
            head_in = feat_dim
        self.fc = nn.Linear(head_in, 1)
        self.wind_head = nn.Linear(head_in, 1) if use_wind else None
        self.feat_dim = feat_dim

    def features(self, x, aux, aux_mask):
        f = self.pool(self.stem(x)).flatten(1)
        f = self.drop(f)
        if self.use_aux:
            a = self.aux_mlp(aux)
            # Under autocast the trunk and the aux MLP both emit fp16 but the mask
            # is fp32, and fp16 * fp32 promotes to fp32 -- which makes the cat below
            # a dtype-mismatch error that only ever fires on the GPU. Cast the mask
            # to the branch's own dtype instead of relying on promotion.
            a = a * aux_mask.view(-1, 1).to(a.dtype)
            f = torch.cat([f, a.to(f.dtype)], dim=1)
        return f

    def forward(self, x, aux=None, aux_mask=None):
        if aux is None:
            aux = torch.zeros(x.shape[0], AUX_DIM, device=x.device, dtype=x.dtype)
        if aux_mask is None:
            aux_mask = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        f = self.features(x, aux, aux_mask)
        logit = self.fc(f)
        wind = self.wind_head(f) if self.wind_head is not None else None
        return logit, wind


class ModelEma:
    """(A3) Exponential moving average of every floating-point tensor in the model's
    state_dict -- parameters AND BatchNorm running statistics.

    WHY: v2's val AUC swung 0.620-0.680 between adjacent epochs on an 1,650-frame
    val set, so "best epoch by val AUC" was selecting largely on noise, and going
    from 8 epochs to 24 would have made that worse (best-of-24 draws from a noisy
    distribution is an optimistically biased estimate). Averaging the weights over
    roughly the last three epochs collapses most of that variance, so the selected
    checkpoint is a genuinely better model rather than a luckier evaluation.

    BatchNorm buffers are averaged too. Running mean/var are already running
    averages of batch statistics, so averaging them further is consistent -- but
    `num_batches_tracked` is an integer counter, not a statistic, so it is copied
    rather than averaged (averaging would make it a non-integer and it feeds
    nothing at eval time anyway).

    The decay is ramped in as min(decay, (1+t)/(10+t)) over the first steps.
    Without that ramp the average stays anchored to the random initialisation for
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
        """EMA weights, cast back to the reference model's dtypes/devices so they
        can be loaded straight into it."""
        ref = reference.state_dict()
        out = {}
        for k, v in ref.items():
            src = self.shadow.get(k, self.non_float.get(k))
            out[k] = src.detach().clone().to(dtype=v.dtype, device=v.device)
        return out


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
    # False-positive rate is the user-visible quantity behind the "too many false
    # alarms" complaint, so it is reported directly rather than left to be derived.
    fpr = fp / max(1, fp + tn)
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")  # only one class present in this split
    return {
        "threshold": float(threshold),
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "false_positive_rate": fpr,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def breakdown(probs, labels, groups, threshold=0.5):
    """Per-group val metrics. Used for the per-basin report -- Indian Ocean
    performance has to be reportable on its own, not hidden inside a global
    average dominated by the Western Pacific."""
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        d = evaluate(probs[m], labels[m], threshold)
        d["n_frames"] = int(m.sum())
        d["n_positive_frames"] = int((labels[m] == 1).sum())
        out[str(g)] = d
    return out


def binary_nll(probs, labels, eps=1e-7):
    """Mean negative log-likelihood. The quantity calibration actually minimises,
    and the one that says whether a displayed 85% is honest -- AUC cannot see
    miscalibration at all, because it only looks at the ranking."""
    p = np.clip(probs.astype(np.float64), eps, 1.0 - eps)
    y = labels.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(probs, labels, n_bins=ECE_BINS):
    """Equal-width-bin ECE on the positive-class probability.

    ECE = sum_b (n_b / N) * |mean(y in b) - mean(p in b)|.

    Directly answers "when this tool says 85%, how often is it right?" -- which is
    the property detect.py's LABEL_DETECTED = 0.85 constant silently assumes.
    """
    p = probs.astype(np.float64)
    y = labels.astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins so p == 1.0 lands in the last bin rather than off the end
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        ece += (nb / len(p)) * abs(acc - conf)
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": nb,
                     "mean_predicted": conf, "observed_positive_rate": acc})
    return float(ece), bins


def threshold_table(probs, labels, thresholds):
    return [evaluate(probs, labels, float(t)) for t in thresholds]


def pick_threshold_for_specificity(probs, labels, target_specificity):
    """Lowest threshold whose false-positive rate is at most (1 - target).

    Lowest, not any: among all thresholds meeting the specificity constraint the
    lowest one keeps the most recall, so this is the best operating point that
    still honours the false-alarm budget. Returns None if even a threshold above
    every predicted probability cannot meet it (impossible here, since predicting
    nothing positive gives FPR 0, but returning None rather than asserting keeps
    the caller honest about degenerate inputs).
    """
    neg = probs[labels == 0]
    if len(neg) == 0:
        return None
    max_fpr = 1.0 - target_specificity
    # Candidate thresholds are the distinct predicted probabilities plus a value
    # above all of them; FPR only changes when the threshold crosses a negative's
    # score, so this grid contains every achievable operating point.
    cands = np.unique(np.concatenate([probs, [probs.max() + 1e-6]]))
    for t in cands:
        if float((neg >= t).mean()) <= max_fpr:
            return float(t)
    return None


def basin_balance_weights(basins, alpha):
    """(D3) Per-frame loss weight proportional to n_basin ** -alpha, renormalised
    so the mean weight is exactly 1.

    Renormalisation matters: the loss is a weighted MEAN, so scaling all weights
    scales the effective learning rate. Forcing mean 1 makes alpha a pure
    reweighting knob that does not silently retune the optimiser.

    alpha = 0 -> every weight exactly 1.0, i.e. bit-identical to v2's plain mean.
    alpha = 1 -> every basin contributes the same total gradient mass.
    """
    basins = np.asarray(basins)
    if alpha == 0.0:
        return np.ones(len(basins), dtype=np.float64)
    counts = Counter(basins.tolist())
    w = np.array([counts[b] ** (-alpha) for b in basins], dtype=np.float64)
    return w * (len(w) / w.sum())


# ----------------------------------------------------------------------------
# (C) Probability calibration
# ----------------------------------------------------------------------------
def fit_platt_scaling(logits, labels, max_iter=100, tol=1e-10):
    """Fit p = sigmoid(a * z + b) by maximum likelihood. Returns (a, b, info).

    This is ordinary 2-parameter logistic regression on the model's own logit, so
    the objective is convex and Newton's method with a backtracking line search
    converges in a handful of iterations with no learning rate to tune and no
    randomness -- which is why it is hand-rolled here rather than pulled from
    sklearn: the fit has to be deterministic and inspectable, and the closed-form
    2x2 Hessian makes it 20 lines.

    a = 1/T is temperature scaling; b additionally corrects a prior shift, which
    matters here because label smoothing during training deliberately biases the
    logits toward zero.

    CRITICAL PROPERTY: as long as a > 0 this is a strictly monotone remap of the
    logit, so AUC, the ROC curve and the ranking of every pair of frames are
    EXACTLY preserved. Calibration cannot make the model better or worse at
    discriminating; it only makes the number it prints mean what it says. The
    caller asserts this.
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(z) != len(y):
        raise ValueError(f"logits/labels length mismatch: {len(z)} vs {len(y)}")
    if len(z) < 10:
        raise ValueError(f"refusing to calibrate on {len(z)} points")

    def nll(a, b):
        t = a * z + b
        # log(1+exp(t)) computed stably, so a large `a` on separable data cannot
        # produce inf/nan and silently poison the line search.
        return float(np.mean(np.logaddexp(0.0, t) - y * t))

    a, b = 1.0, 0.0
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        t = a * z + b
        p = 1.0 / (1.0 + np.exp(-np.clip(t, -60.0, 60.0)))
        r = p - y
        g = np.array([np.mean(r * z), np.mean(r)])
        w = p * (1.0 - p)
        # Ridge on the Hessian diagonal: on nearly-separable data w -> 0 and the
        # Hessian becomes singular. 1e-9 is far below any meaningful curvature
        # here, so it regularises the solve without biasing the fit.
        h = np.array([[np.mean(w * z * z) + 1e-9, np.mean(w * z)],
                      [np.mean(w * z), np.mean(w) + 1e-9]])
        try:
            step = -np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        base = nll(a, b)
        s = 1.0
        while s > 1e-8 and nll(a + s * step[0], b + s * step[1]) > base:
            s *= 0.5
        a, b = a + s * step[0], b + s * step[1]
        if max(abs(s * step[0]), abs(s * step[1])) < tol:
            break

    info = {"n_iter": n_iter, "nll": nll(a, b), "clamped": False}
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0.0:
        # a <= 0 means the model ranks val ANTI-correlated with the truth. Flipping
        # the sign would be a silent, unjustifiable "fix", so refuse instead: the
        # uncalibrated model is saved and the run says why.
        print(f"!! CALIBRATION FAILED: fitted slope a={a} is not positive. The model "
              f"ranks the validation set the wrong way round; refusing to fold a "
              f"sign-flipping transform into fc. Saving UNCALIBRATED. !!")
        return 1.0, 0.0, {"n_iter": n_iter, "nll": nll(1.0, 0.0), "failed": True,
                          "clamped": False}
    if a > CALIB_MAX_SLOPE or abs(b) > CALIB_MAX_BIAS:
        # Only reachable on (near-)separable val data, where the MLE slope is
        # unbounded. Clamping keeps fc's weights finite in fp32; at a = 50 the
        # sigmoid is already saturated so the clamp costs nothing real.
        print(f"NOTE: calibration clamped from (a={a:.3f}, b={b:.3f}) to the "
              f"[{1.0 / CALIB_MAX_SLOPE:.3g}, {CALIB_MAX_SLOPE:.3g}] slope range -- "
              f"this means the validation set is nearly separable.")
        a = float(np.clip(a, 1.0 / CALIB_MAX_SLOPE, CALIB_MAX_SLOPE))
        b = float(np.clip(b, -CALIB_MAX_BIAS, CALIB_MAX_BIAS))
        info["clamped"] = True
        info["nll"] = nll(a, b)
    return float(a), float(b), info


def apply_calibration(logits, a, b):
    t = np.clip(a * np.asarray(logits, dtype=np.float64) + b, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-t))


def fold_calibration_into_fc(net, a, b):
    """Bake p = sigmoid(a*z + b) into the final Linear layer.

    fc computes z = W.x + c, so setting W <- a*W and c <- a*c + b makes the network
    output a*z + b directly. The layer is still a Linear(512, 1): no new module, no
    new state_dict key, no architecture change -- which is the whole point.
    local_tool/detect.py builds a plain resnet18 and does a STRICT load_state_dict,
    so a calibration wrapper module would have broken it, and a temperature stored
    in the config would have been ignored by the existing detect.py entirely.
    """
    with torch.no_grad():
        net.fc.weight.mul_(a)
        net.fc.bias.mul_(a).add_(b)


def calibration_crosscheck(logits, labels, storm_keys, seed, n_folds=CALIB_CV_FOLDS):
    """Decide whether the calibration is REAL before folding it into the model.

    The temperature/bias would otherwise be fitted on the same validation set that
    selected the epoch, so "it improves val NLL" is circular -- a 2-parameter fit
    always improves the data it was fitted on. This runs grouped K-fold
    cross-validation instead: split the val STORMS (never frames -- frames of one
    storm are near-duplicates and splitting on them would leak), fit on K-1 folds,
    score the held-out fold, and average.

    Grouped-by-storm and K-fold rather than a single half-split for two reasons
    that a first cut of this got wrong: a half-split fits on only 50% of the data
    and so understates how well the final full-data fit generalises, and a single
    split is itself noisy enough that the verdict flips run to run. K=5 fits on 80%
    (close to the real fit) and averages five estimates.

    Returns (rows, verdict) where verdict["generalises"] is what main() GATES the
    fold on. If the model is already well calibrated there is nothing to correct
    and forcing a fit would inject pure noise into the shipped probabilities, so
    "no change" is a legitimate and expected outcome, not a failure.
    """
    rng = np.random.RandomState(seed)
    keys = np.array(sorted(set(storm_keys.tolist())))
    rng.shuffle(keys)
    fold_of_key = {k: i % n_folds for i, k in enumerate(keys.tolist())}
    fold_id = np.array([fold_of_key[k] for k in storm_keys])

    rows = []
    for f in range(n_folds):
        ev_m = fold_id == f
        fit_m = ~ev_m
        if fit_m.sum() < 20 or ev_m.sum() < 20:
            print(f"  crosscheck fold {f}: too few frames, skipped")
            continue
        if len(set(labels[fit_m].tolist())) < 2 or len(set(labels[ev_m].tolist())) < 2:
            print(f"  crosscheck fold {f}: a fold is single-class, skipped")
            continue
        a, b, _ = fit_platt_scaling(logits[fit_m], labels[fit_m])
        raw = apply_calibration(logits[ev_m], 1.0, 0.0)
        cal = apply_calibration(logits[ev_m], a, b)
        e_raw, _ = expected_calibration_error(raw, labels[ev_m])
        e_cal, _ = expected_calibration_error(cal, labels[ev_m])
        rows.append({
            "fold": int(f), "n_fit": int(fit_m.sum()), "n_eval": int(ev_m.sum()),
            "a": float(a), "b": float(b),
            "nll_before": binary_nll(raw, labels[ev_m]),
            "nll_after": binary_nll(cal, labels[ev_m]),
            "ece_before": float(e_raw), "ece_after": float(e_cal),
        })
        print(f"  crosscheck fold {f}: fit {rows[-1]['n_fit']} / eval {rows[-1]['n_eval']} "
              f"frames  a={a:.3f} b={b:+.3f}  "
              f"NLL {rows[-1]['nll_before']:.4f} -> {rows[-1]['nll_after']:.4f}  "
              f"ECE {e_raw:.4f} -> {e_cal:.4f}")

    if not rows:
        print("  crosscheck: no usable folds -- NOT folding a calibration.")
        return rows, {"generalises": False, "reason": "no usable folds"}

    d_nll = float(np.mean([r["nll_after"] - r["nll_before"] for r in rows]))
    d_ece = float(np.mean([r["ece_after"] - r["ece_before"] for r in rows]))
    n_better = int(sum(1 for r in rows if r["nll_after"] < r["nll_before"]))
    # NLL is the proper scoring rule the calibration actually optimises, so it is
    # the criterion. The majority-of-folds condition stops one lucky fold from
    # carrying a mean that every other fold disagrees with. ECE is reported but not
    # used to decide: it is a binned estimator and is noisy on a few thousand points.
    generalises = (d_nll < 0.0) and (n_better * 2 > len(rows))
    print(f"  crosscheck: mean held-out dNLL={d_nll:+.4f}  dECE={d_ece:+.4f}  "
          f"folds improved {n_better}/{len(rows)}")
    print(f"  crosscheck VERDICT: calibration "
          f"{'GENERALISES -> will be folded in' if generalises else 'DOES NOT help -> will NOT be folded'}")
    return rows, {"generalises": bool(generalises), "mean_delta_nll": d_nll,
                  "mean_delta_ece": d_ece, "folds_improved": n_better,
                  "n_folds_used": len(rows)}


# ----------------------------------------------------------------------------
# (D1) Field-of-view diagnostic
# ----------------------------------------------------------------------------
def radial_scale_diagnostic(images_u8, index, max_per_group=800, seed=SEED):
    """Measure how far the cold-cloud signal extends from frame centre, per source.

    WHY THIS EXISTS: the leading explanation for digital_typhoon auc=0.728 vs tcir
    auc=0.598 is that the two datasets crop DIFFERENT physical areas around the
    storm. Digital Typhoon is documented as 512x512 at ~5 km/pixel; TCIR is a
    128x128 crop on a ~0.07-degree grid. If those cover different spans, then after
    both are resized to 224x224 the SAME storm occupies a different fraction of the
    frame in each source, and a convolutional net trained mostly on one scale
    cannot be expected to transfer to the other.

    It is a HYPOTHESIS, not a fact: six sample frames checked offline (four Digital
    Typhoon, two TCIR) had heavily overlapping cold-cloud radii, but they were
    different storms at different intensities, so they settled nothing. This
    measures it properly, on thousands of intensity-matched early-stage frames.

    Method: per frame, take the "cold" mass w = max(0, pixel/255 - 0.5), i.e. every
    pixel colder than the midpoint of the 190-300 K display window, and report the
    weighted median radius from frame centre in units of the half-width. Frames are
    grouped by (source, label) so a difference cannot be an artefact of one source
    holding stronger storms.

    HOW TO READ THE RESULT: if the r50 ratio between sources is close to 1.0, the
    field-of-view hypothesis is refuted and the source gap is about resolution or
    label noise instead. If Digital Typhoon's r50 is systematically ~0.6x TCIR's
    at matched labels, the hypothesis is supported and the v4 fix is to centre-crop
    Digital Typhoon to the matching span (which would ALSO require the same crop in
    detect.py's Mode A -- it is not a training-only change, which is exactly why
    this run measures rather than acts).
    """
    print("\n" + "=" * 72)
    print("(D1) RADIAL SCALE DIAGNOSTIC -- is the source gap a field-of-view gap?")
    print("=" * 72)

    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    c = (IMG_SIZE - 1) / 2.0
    r = (np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / (IMG_SIZE / 2.0)).ravel()
    order = np.argsort(r)
    r_sorted = r[order]

    rng = np.random.RandomState(seed)
    out = {}
    sources = index["source"].to_numpy()
    labels = index["label"].to_numpy()
    for src in sorted(set(sources.tolist())):
        for lab in (0, 1):
            pos = np.where((sources == src) & (labels == lab))[0]
            if len(pos) == 0:
                continue
            if len(pos) > max_per_group:
                pos = rng.choice(pos, max_per_group, replace=False)
            r50s = []
            for chunk in np.array_split(pos, max(1, len(pos) // 200)):
                w = images_u8[chunk].reshape(len(chunk), -1).astype(np.float32) / 255.0 - 0.5
                np.clip(w, 0.0, None, out=w)
                w = w[:, order]
                cum = np.cumsum(w, axis=1)
                tot = cum[:, -1]
                good = tot > 1e-6  # a frame with no cold cloud at all has no radius
                if not good.any():
                    continue
                half = 0.5 * tot[good][:, None]
                idx = (cum[good] >= half).argmax(axis=1)
                r50s.append(r_sorted[idx])
            if not r50s:
                continue
            r50 = np.concatenate(r50s)
            out[f"{src}|label{lab}"] = {
                "n_frames": int(len(r50)),
                "cold_mass_r50_mean": float(r50.mean()),
                "cold_mass_r50_median": float(np.median(r50)),
                "cold_mass_r50_p25": float(np.percentile(r50, 25)),
                "cold_mass_r50_p75": float(np.percentile(r50, 75)),
            }
            print(f"  {src:16s} label={lab}  n={len(r50):5d}  "
                  f"cold-mass r50 median={np.median(r50):.3f} "
                  f"(IQR {np.percentile(r50, 25):.3f}-{np.percentile(r50, 75):.3f})")

    for lab in (0, 1):
        kd, kt = f"digital_typhoon|label{lab}", f"tcir|label{lab}"
        if kd in out and kt in out:
            ratio = out[kd]["cold_mass_r50_median"] / max(1e-9, out[kt]["cold_mass_r50_median"])
            out[f"dt_over_tcir_r50_ratio_label{lab}"] = float(ratio)
            print(f"  => label={lab}: digital_typhoon / tcir r50 ratio = {ratio:.3f}")
    print("  A ratio near 1.00 REFUTES the field-of-view hypothesis (the gap is then")
    print("  about native resolution or label noise). A ratio near 0.6 SUPPORTS it")
    print("  and points at a centre-crop fix for v4 -- which would also need the")
    print("  same crop added to detect.py's Mode A, so do not apply it blind.")
    print("=" * 72 + "\n")
    return out


# ----------------------------------------------------------------------------
# Inference helpers
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(net, loader, device, tta=False, use_meta=0.0):
    """Val-side forward pass. Returns raw logits (not probabilities) because the
    calibration is fitted on the logit scale.

    tta=True averages the four 90-degree rotations in PROBABILITY space, which is
    the correct way to combine an ensemble of equally-trusted views, and then maps
    back to a logit so the caller's code path is unchanged. REPORT ONLY: detect.py
    does a single un-rotated forward pass, so the saved model and every headline
    metric use tta=False. The TTA number exists to say what a change to detect.py
    would be worth, not to flatter this checkpoint.
    """
    net.eval()
    outs, ys = [], []
    with torch.no_grad():
        for imgs_u8, labels, _w, aux, _soft, _wind in loader:
            x = to_model_input(imgs_u8.to(device))
            a = aux.to(device)
            # use_meta is the deployment switch: 1 -> the model may use position and
            # date, 0 -> image only, exactly what detect.py can supply.
            m = torch.full((x.shape[0],), float(use_meta), device=device)
            if tta:
                p = torch.zeros(x.shape[0], device=device)
                for k in range(4):
                    logit, _ = net(torch.rot90(x, k, dims=(2, 3)), a, m)
                    p += torch.sigmoid(logit.squeeze(1).float())
                p /= 4.0
                p = p.clamp(1e-6, 1 - 1e-6)
                outs.append(torch.log(p / (1 - p)).cpu().numpy())
            else:
                logit, _ = net(x, a, m)
                outs.append(logit.squeeze(1).float().cpu().numpy())
            ys.append(labels.numpy())
    return np.concatenate(outs).astype(np.float64), np.concatenate(ys).astype(int)


def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -60.0, 60.0)))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def assign_groups(index, seed, n_groups, label):
    """Assign every storm to one of n_groups, stratified by (basin, label).

    CRITICAL PROPERTY -- the two arms must get IDENTICAL assignments for every
    storm they share. The Digital Typhoon and TCIR storms are drawn FIRST, from a
    RandomState consumed in exactly the order v5's split loop consumed it, and the
    MOSDAC storms are drawn afterwards from a SEPARATE generator. If MOSDAC storms
    joined the (IO, label) groups instead, adding 32 storms would change the
    shuffle for the Indian Ocean AND, because one RandomState is consumed across
    all groups in sorted order, for every basin sorted after it. The arms would
    then be scored on different validation storms and the comparison would be
    meaningless.

    n_groups=2 with the first group as "val" reproduces v5's 80/20 split shape
    only if VAL_FRACTION is 0.5, so the caller passes n_groups for CV folds and
    uses split_80_20() for the shipping split.
    """
    storms = index.drop_duplicates("storm_key")[["storm_key", "basin", "label",
                                                 "source"]]
    base = storms[storms["source"] != "mosdac"]
    mos = storms[storms["source"] == "mosdac"]

    out = {}
    rng = np.random.RandomState(seed)
    for (basin, lab), grp in base.groupby(["basin", "label"], sort=True):
        keys = np.array(sorted(grp["storm_key"].tolist()))
        rng.shuffle(keys)
        for i, k in enumerate(keys.tolist()):
            out[k] = i % n_groups
    rng_m = np.random.RandomState(seed + 991)
    for lab, grp in mos.groupby("label", sort=True):
        keys = np.array(sorted(grp["storm_key"].tolist()))
        rng_m.shuffle(keys)
        for i, k in enumerate(keys.tolist()):
            out[k] = i % n_groups
    g = index["storm_key"].map(out).to_numpy()
    if pd.isna(g).any():
        raise SystemExit("some storms were not assigned a group")
    print(f"  {label}: {len(out)} storms -> {n_groups} groups "
          f"({Counter(out.values())})")
    return g.astype(int)


def split_80_20(index, seed):
    """v5's exact shipping split, with MOSDAC drawn separately (see assign_groups).

    Returns the set of validation storm keys.
    """
    storms = index.drop_duplicates("storm_key")[["storm_key", "basin", "label",
                                                 "source"]]
    base = storms[storms["source"] != "mosdac"]
    mos = storms[storms["source"] == "mosdac"]
    val = set()
    rng = np.random.RandomState(seed)
    for (basin, lab), grp in base.groupby(["basin", "label"], sort=True):
        keys = np.array(sorted(grp["storm_key"].tolist()))
        rng.shuffle(keys)
        n_val = int(round(len(keys) * VAL_FRACTION))
        if n_val == 0 and len(keys) >= 2:
            n_val = 1
        val.update(keys[:n_val].tolist())
        print(f"  split {basin:5s} label={lab}: {len(keys):5d} storms -> "
              f"{n_val:4d} val / {len(keys) - n_val:5d} train")
    rng_m = np.random.RandomState(seed + 991)
    for lab, grp in mos.groupby("label", sort=True):
        keys = np.array(sorted(grp["storm_key"].tolist()))
        rng_m.shuffle(keys)
        n_val = int(round(len(keys) * VAL_FRACTION))
        if n_val == 0 and len(keys) >= 2:
            n_val = 1
        val.update(keys[:n_val].tolist())
        print(f"  split MOSDAC label={lab}: {len(keys):5d} storms -> "
              f"{n_val:4d} val / {len(keys) - n_val:5d} train")
    return val


def train_one(images, index, aux_all, y, y_soft, wind_t, tr_idx, va_idx, cfg,
              device, seed, epochs, tag, use_amp, keep_per_epoch=True,
              select_best=False):
    """Train one model. Returns (per_epoch_probs, history, net, val_loader).

    select_best=False (the CROSS-VALIDATION arms): no epoch is selected inside
    this function at all. Every epoch's validation probabilities are returned and
    main() picks ONE epoch for both arms, on the baseline arm's curve. Selecting
    per fold on that fold's own held-out data would make each arm's out-of-fold
    prediction optimistically biased, and there is no reason to believe the two
    arms would be biased by the same amount.

    select_best=True (the SHIPPING run): reproduces v5_final's recipe exactly --
    best epoch by validation AUC on the EMA weights. That bias is already priced
    into the shipped pipeline (calibration and the operating-point table are
    fitted on the same validation frames), and keeping it identical is what makes
    this checkpoint comparable with the one v5_final produced.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_w = basin_balance_weights(index["basin"].to_numpy()[tr_idx],
                                    BASIN_BALANCE_ALPHA)
    train_ds = FrameDataset(images[tr_idx], y[tr_idx], train_w, augment=True,
                            aux=aux_all[tr_idx], soft=y_soft[tr_idx],
                            wind=wind_t[tr_idx])
    val_ds = FrameDataset(images[va_idx], y[va_idx], np.ones(len(va_idx)),
                          augment=False, aux=aux_all[va_idx],
                          soft=y_soft[va_idx], wind=wind_t[va_idx])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    net = CycloneNet(arch=cfg["arch"], use_aux=cfg["aux"], dropout=cfg["dropout"],
                     use_wind=cfg["wind_aux"] > 0).to(device)
    ema = ModelEma(net, EMA_DECAY)
    n_pos = float((y[tr_idx] == 1).sum())
    n_neg = float((y[tr_idx] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=cfg["wd"])

    def lr_at(e):
        if e < SWEEP_WARMUP:
            return (e + 1) / max(1, SWEEP_WARMUP)
        t = (e - SWEEP_WARMUP) / max(1, epochs - SWEEP_WARMUP)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = make_scaler(use_amp)
    aug_gen = torch.Generator()
    val_y = y[va_idx].astype(int)

    per_epoch, history = [], []
    best_auc, best_epoch, best_state = -1.0, -1, None
    for epoch in range(1, epochs + 1):
        aug_gen.manual_seed(seed * 1000 + epoch)
        net.train()
        tot_l, tot_w = 0.0, 0.0
        t0 = time.time()
        for imgs_u8, lab, w, aux_b, soft_b, wind_b in train_dl:
            x01 = to_unit(imgs_u8.to(device, non_blocking=True))
            x = normalize(augment_batch(x01, aug_gen, cfg["aug"]))
            a = aux_b.to(device)
            m = (torch.rand(x.shape[0], generator=aug_gen).to(device)
                 >= cfg.get("aux_drop", AUX_DROP_P)).float() * (1.0 if cfg["aux"] else 0.0)
            m = m * (a.abs().sum(dim=1) > 0).float()
            wt = w.to(device).unsqueeze(1)
            if cfg["soft"]:
                t = soft_b.to(device).float().unsqueeze(1)
            else:
                t = lab.to(device).float().unsqueeze(1)
                t = t * (1.0 - 2.0 * LABEL_SMOOTHING) + LABEL_SMOOTHING
            opt.zero_grad(set_to_none=True)
            with amp_autocast(use_amp):
                logit, wpred = net(x, a, m)
                loss = (criterion(logit, t) * wt).sum() / wt.sum()
                if wpred is not None and cfg["wind_aux"] > 0:
                    loss = loss + cfg["wind_aux"] * F.mse_loss(
                        wpred.float(), wind_b.to(device).float().unsqueeze(1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(net)
            tot_l += float(loss.detach().item()) * float(wt.sum().item())
            tot_w += float(wt.sum().item())
        sched.step()

        live = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(ema.state_dict(net))
        vl, _ = predict_logits(net, val_dl, device, use_meta=0.0)
        vp = sigmoid_np(vl)
        if keep_per_epoch:
            per_epoch.append(vp.astype(np.float32))
        auc = auc_of(val_y, vp)
        history.append({"epoch": epoch, "train_loss": tot_l / max(1e-9, tot_w),
                        "val_auc": auc, "secs": round(time.time() - t0, 1)})
        print(f"    [{tag}] epoch {epoch:2d}/{epochs} loss={history[-1]['train_loss']:.4f} "
              f"val_auc={auc:.4f} ({history[-1]['secs']:.0f}s)")
        if select_best and auc == auc and auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if epoch < epochs:
            net.load_state_dict(live)
    if select_best:
        if best_state is None:
            raise SystemExit("no epoch produced a scoreable validation result")
        tail = [h["val_auc"] for h in history[-5:] if h["val_auc"] == h["val_auc"]]
        print(f"    [{tag}] selection audit: best val AUC {best_auc:.4f} at epoch "
              f"{best_epoch}/{epochs}; last-5 mean={np.mean(tail):.4f} "
              f"sd={np.std(tail):.4f}  (a small sd means the pick is stable, not "
              f"a lucky epoch)")
        net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        for h in history:
            h["selected"] = bool(h["epoch"] == best_epoch)
    return (np.stack(per_epoch) if per_epoch else None), history, net, val_dl


def storm_level(probs, labels, storm_keys):
    """Mean probability over a storm's early frames -> (ids, prob, label).

    Storm-level aggregation is worth ~+0.045 AUC free (measured in v5) and, more
    importantly, storms are the independent unit: every CI in this run is computed
    over storms, never frames.
    """
    ids, pos = np.unique(storm_keys, return_inverse=True)
    n = np.bincount(pos).astype(np.float64)
    p = np.bincount(pos, weights=probs) / n
    yy = np.zeros(len(ids), dtype=int)
    for i in range(len(labels)):
        yy[pos[i]] = labels[i]
    return ids, p, yy


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    by_name = {c["name"]: c for c in SWEEP_CONFIGS}
    cfg = by_name[FINAL_CONFIG_NAME]
    print("=" * 72)
    print(f"Cyclone Genesis Predictor {MODEL_VERSION} -- does MOSDAC/INSAT help the Indian Ocean?")
    print(f"  config {FINAL_CONFIG_NAME}: {cfg}")
    print(f"  arms={ARMS}  folds={N_FOLDS}  epochs={SWEEP_EPOCHS}  AMP={USE_AMP}")
    print("=" * 72)

    # ------------------------------------------------------------------ discovery
    dt_root = find_dt_root()
    if not dt_root:
        raise SystemExit(f"No typhoon dataset under {INPUT_ROOT}")
    meta_dir, image_dir = resolve_dt_dirs(dt_root)
    print("DT ROOT   =", dt_root)
    tcir_path = find_tcir_h5()
    if not tcir_path:
        raise SystemExit(f"No TCSA/TCIR .h5 under {INPUT_ROOT}")
    print("TCIR h5   =", tcir_path, f"({os.path.getsize(tcir_path) / 1e9:.1f} GB)")
    pack_dir = find_mosdac_pack()
    if not pack_dir:
        raise SystemExit(
            f"No MOSDAC pack under {INPUT_ROOT}. Attach the dataset holding "
            f"mosdac_frames.npy + mosdac_index.csv, or this run has no treatment "
            f"arm to test.")
    print("MOSDAC    =", pack_dir)

    # ------------------------------------------------------------------ sources
    print("\nLoading TCIR 'info' table...")
    info = pd.read_hdf(tcir_path, key="info", mode="r")
    need_b = {"region", "ID", "lon", "lat", "time", "Vmax"}
    missing_b = need_b - set(info.columns)
    if missing_b:
        raise SystemExit(f"TCIR 'info' is missing {sorted(missing_b)}")
    info = info.reset_index(drop=True)
    with __import__("h5py").File(tcir_path, "r") as f:
        img_shape = f["images"].shape
    if img_shape[0] != len(info):
        raise SystemExit(f"TCIR row-alignment broken: {img_shape[0]} vs {len(info)}")
    check_tcir_id_pattern(info)
    tcir_index, tcir_work, tcir_audit = build_tcir_index(info)

    print()
    dt_index, dt_tracks, dt_audit = build_dt_index(meta_dir, image_dir)

    print()
    mos_index, mos_frames, mos_audit = build_mosdac_index(pack_dir)

    # ------------------------------------------------------------------ dedup
    drop_ids, dedup_audit = dedup_wpac(dt_index, tcir_index, tcir_work, dt_tracks)
    tcir_index = tcir_index[~tcir_index["storm_key"].isin(drop_ids)].reset_index(drop=True)

    mos_tracks = {}
    tpath = os.path.join(pack_dir, "mosdac_tracks_for_dedup.json")
    if os.path.exists(tpath):
        with open(tpath) as f:
            mos_tracks = json.load(f)
    null_audit = mosdac_overlap_null_test(mos_index, mos_tracks, tcir_index,
                                          tcir_work, dt_tracks)

    index = pd.concat([dt_index, tcir_index, mos_index], ignore_index=True)
    clashing = index.groupby("storm_key")["source"].nunique()
    clashing = clashing[clashing > 1]
    if len(clashing):
        raise SystemExit(f"storm_key collision across sources: "
                         f"{list(clashing.index[:10])}")

    print("\nFinal storms per source per basin:")
    for (src, basin), n in sorted(index.drop_duplicates("storm_key")
                                  .groupby(["source", "basin"]).size().to_dict().items()):
        print(f"    {src:16s} {basin:5s} {n:5d}")

    # ------------------------------------------------------------------ preload
    images, ok = preload_all(index, tcir_path, mos_frames)
    index = index[ok].reset_index(drop=True)
    images = images[ok]
    index.to_csv(INDEX_PATH, index=False)

    aux_all, _aux_ok = build_aux_features(index)
    y = index["label"].to_numpy().astype(np.float32)
    wmax = pd.to_numeric(index["max_wind_kt"], errors="coerce").to_numpy(np.float64)
    y_soft = np.clip(1.0 / (1.0 + np.exp(
        -(np.nan_to_num(wmax, nan=64.0) - TYPHOON_WIND_KT) / SOFT_LABEL_KT)),
        0.01, 0.99).astype(np.float32)
    wind_t = (np.clip(np.nan_to_num(wmax, nan=64.0), 0, WIND_CLIP_KT)
              / WIND_NORM_KT).astype(np.float32)
    sel_clean = index["sel_clean"].to_numpy().astype(bool)
    sources = index["source"].to_numpy()
    basins = index["basin"].to_numpy()
    storm_keys = index["storm_key"].to_numpy()
    is_mosdac_frame = sources == "mosdac"

    radial = radial_scale_diagnostic(images, index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(USE_AMP and device.type == "cuda")
    print("Device:", device, f"AMP={use_amp}")

    # ================================================================= PHASE 1
    # Grouped cross-validation. Every storm gets exactly one out-of-fold
    # prediction, so the comparison is made over ~2,069 storms rather than the
    # ~406 a single 80/20 holdout would give -- which is the difference between a
    # standard error of ~0.011 and ~0.025, and therefore the difference between
    # being able to see a real effect and measuring noise. This is the instrument
    # v5 established; reusing it is the point.
    print("\n" + "=" * 72)
    print(f"PHASE 1 -- {N_FOLDS}-fold grouped CV, arms {ARMS}, IDENTICAL folds")
    print("=" * 72)
    fold_of = assign_groups(index, SEED, N_FOLDS, "CV folds")

    n = len(index)
    oof = {a: {"prob": np.full((SWEEP_EPOCHS, n), np.nan, np.float32)}
           for a in ARMS}
    histories = {a: [] for a in ARMS}
    for f in range(N_FOLDS):
        va_idx = np.where(sel_clean & (fold_of == f))[0]
        for arm in ARMS:
            keep = sel_clean & (fold_of != f)
            if arm == "baseline":
                keep = keep & ~is_mosdac_frame
            tr_idx = np.where(keep)[0]
            print(f"\n  fold {f} arm {arm}: train {len(tr_idx)} frames "
                  f"({int(is_mosdac_frame[tr_idx].sum())} mosdac), "
                  f"val {len(va_idx)} frames "
                  f"({int(is_mosdac_frame[va_idx].sum())} mosdac)")
            pe, hist, _net, _dl = train_one(
                images, index, aux_all, y, y_soft, wind_t, tr_idx, va_idx, cfg,
                device, SEED + 17 * f, SWEEP_EPOCHS, f"{arm} f{f}", use_amp)
            oof[arm]["prob"][:, va_idx] = pe
            histories[arm].append({"fold": f, "history": hist})

    # ---- epoch selection, done on the BASELINE arm only --------------------
    # Choosing the epoch that maximises the TREATMENT arm's own score would hand
    # the treatment a free selection advantage over the control. Selecting on the
    # baseline cannot do that: at worst it picks an epoch that suits the baseline,
    # which is conservative for the claim being tested. The delta at EVERY epoch
    # is reported as well, so the headline number can be checked against the whole
    # curve rather than taken on trust.
    covered = sel_clean
    base_curve = []
    for e in range(SWEEP_EPOCHS):
        _i, p, yy = storm_level(oof["baseline"]["prob"][e][covered],
                                y[covered].astype(int), storm_keys[covered])
        base_curve.append(auc_of(yy, p))
    sel_epoch = int(np.nanargmax(base_curve))
    print(f"\nBaseline out-of-fold storm AUC by epoch: "
          f"{[round(v, 4) for v in base_curve]}")
    print(f"Epoch selected on the BASELINE arm only: {sel_epoch + 1}/{SWEEP_EPOCHS} "
          f"(AUC {base_curve[sel_epoch]:.4f})")

    def slice_metrics(mask, name, reps=BOOT_REPS):
        """Storm-level AUC per arm plus the paired bootstrap of the difference."""
        m = covered & mask
        if m.sum() == 0:
            return None
        res = {"slice": name, "n_frames": int(m.sum())}
        probs = {}
        for arm in ARMS:
            ids, p, yy = storm_level(oof[arm]["prob"][sel_epoch][m],
                                     y[m].astype(int), storm_keys[m])
            probs[arm] = (ids, p, yy)
            res[f"auc_{arm}"] = auc_of(yy, p)
            res["n_storms"] = int(len(ids))
            res["n_positive_storms"] = int(yy.sum())
        if len(ARMS) == 2 and "baseline" in probs and "mosdac" in probs:
            ids_a, pa, ya = probs["baseline"]
            ids_b, pb, yb = probs["mosdac"]
            if not np.array_equal(ids_a, ids_b):
                raise SystemExit(f"slice {name}: arms have different storm sets")
            res["paired_bootstrap"] = paired_bootstrap_auc(ya, pa, pb, reps)
        return res

    io = basins == "IO"
    slices = [
        slice_metrics(np.ones(n, bool), "ALL"),
        slice_metrics(io, "IO (pooled -- read against the shortcut probe)"),
        slice_metrics(io & ~is_mosdac_frame, "IO within TCIR only (HEADLINE)"),
        slice_metrics(io & is_mosdac_frame, "IO within MOSDAC only (transfer)"),
    ]
    for b in sorted(set(basins.tolist())):
        if b != "IO":
            slices.append(slice_metrics(basins == b, f"basin {b}"))
    slices.append(slice_metrics(~io, "everything except IO (regression check)"))
    slices = [s for s in slices if s]

    print("\n" + "=" * 72)
    print("PHASE 1 RESULT -- out-of-fold storm-level AUC")
    print("=" * 72)
    print(f"{'slice':46s} {'storms':>6s} {'base':>7s} {'mosdac':>7s} {'delta':>8s}  95% CI")
    for s in slices:
        pb = s.get("paired_bootstrap") or {}
        ci = (f"[{pb['ci_lo']:+.4f}, {pb['ci_hi']:+.4f}] P(better)={pb['p_better']:.3f}"
              if pb.get("ci_lo") is not None else "n/a")
        print(f"{s['slice']:46s} {s.get('n_storms', 0):6d} "
              f"{s.get('auc_baseline', float('nan')):7.4f} "
              f"{s.get('auc_mosdac', float('nan')):7.4f} "
              f"{(pb.get('delta') if pb.get('delta') is not None else float('nan')):+8.4f}  {ci}")

    print("\nSOURCE-SHORTCUT PROBE (Indian Ocean, out-of-fold storms):")
    iom = covered & io
    shortcut = source_shortcut_probe(sources[iom], y[iom].astype(int),
                                     storm_keys[iom])

    print("\nDelta at EVERY epoch (IO within TCIR only), so the headline is not "
          "an epoch cherry-pick:")
    curve = []
    for e in range(SWEEP_EPOCHS):
        m = covered & io & ~is_mosdac_frame
        _i, pa, ya = storm_level(oof["baseline"]["prob"][e][m], y[m].astype(int),
                                 storm_keys[m])
        _i, pb2, _y = storm_level(oof["mosdac"]["prob"][e][m], y[m].astype(int),
                                  storm_keys[m])
        curve.append({"epoch": e + 1, "auc_baseline": auc_of(ya, pa),
                      "auc_mosdac": auc_of(ya, pb2),
                      "delta": auc_of(ya, pb2) - auc_of(ya, pa)})
        print(f"    epoch {e + 1:2d}: base {curve[-1]['auc_baseline']:.4f}  "
              f"mosdac {curve[-1]['auc_mosdac']:.4f}  "
              f"delta {curve[-1]['delta']:+.4f}")

    phase1 = {"selected_epoch": sel_epoch + 1, "slices": slices,
              "source_shortcut_probe": shortcut, "epoch_curve": curve,
              "fold_histories": histories}
    with open(os.path.join(OUT_DIR, "phase1_cv.json"), "w") as fh:
        json.dump(jsonable(phase1), fh, indent=1)

    # ================================================================= PHASE 2
    # One shippable checkpoint on v5's 80/20 split, with MOSDAC included, so the
    # run produces a deliverable and not only a measurement. Calibration, the
    # storm-level cross-check gate, the operating-point table and the exact
    # plain-resnet18 export are all v5's, unchanged, because local_tool/detect.py
    # and the mobile app load this file's `state_dict` with a STRICT load.
    print("\n" + "=" * 72)
    print(f"PHASE 2 -- shipping checkpoint, arm {SHIP_ARM}, v5's 80/20 split")
    print("=" * 72)
    val_keys = split_80_20(index, SEED)
    is_val = index["storm_key"].isin(val_keys).to_numpy()
    keep = sel_clean & ~is_val
    if SHIP_ARM == "baseline":
        keep = keep & ~is_mosdac_frame
    tr_idx = np.where(keep)[0]
    va_idx = np.where(sel_clean & is_val)[0]
    print(f"  train {len(tr_idx)} frames / val {len(va_idx)} frames; "
          f"{len(val_keys)} val storms")

    _pe, hist, net, val_dl = train_one(
        images, index, aux_all, y, y_soft, wind_t, tr_idx, va_idx, cfg, device,
        SEED, FINAL_EPOCHS, f"ship {SHIP_ARM}", use_amp, keep_per_epoch=False,
        select_best=True)
    ship_epoch = next((h["epoch"] for h in hist if h.get("selected")), FINAL_EPOCHS)

    val_y = y[va_idx].astype(int)
    val_basins = basins[va_idx]
    val_sources = sources[va_idx]
    val_storm_keys = storm_keys[va_idx]
    val_logits, _ = predict_logits(net, val_dl, device, use_meta=0.0)
    raw_probs = sigmoid_np(val_logits)
    ece_before, bins_before = expected_calibration_error(raw_probs, val_y)
    print(f"\nbefore calibration: NLL={binary_nll(raw_probs, val_y):.4f} "
          f"ECE={ece_before:.4f}")
    cross, verdict = calibration_crosscheck(val_logits, val_y, val_storm_keys,
                                            SEED + 7)
    if verdict["generalises"]:
        a, b, _ci = fit_platt_scaling(val_logits, val_y)
        print(f"fitted: a={a:.4f} (T={1.0 / a:.4f}) b={b:+.4f}")
    else:
        a, b = 1.0, 0.0
        print("NOT applying a calibration -- the storm-level cross-check says a "
              "fitted remap does not improve held-out likelihood.")
    cal_probs = apply_calibration(val_logits, a, b)
    with torch.no_grad():
        net.fc.weight.mul_(a)
        net.fc.bias.mul_(a).add_(b)
    folded_logits, _ = predict_logits(net, val_dl, device, use_meta=0.0)
    folded_probs = sigmoid_np(folded_logits)
    max_dev = float(np.max(np.abs(folded_probs - cal_probs)))
    if max_dev > 1e-4:
        raise SystemExit(f"calibration fold moved probabilities by {max_dev:.2e}")
    if abs(auc_of(val_y, raw_probs) - auc_of(val_y, folded_probs)) > 1e-6:
        raise SystemExit("AUC changed across calibration; the fold is wrong.")
    ece_after, bins_after = expected_calibration_error(folded_probs, val_y)
    print(f"after : NLL={binary_nll(folded_probs, val_y):.4f} ECE={ece_after:.4f} "
          f"(fold verified, max dev {max_dev:.2e})")

    plain_sd, plain_check = export_plain_resnet18(net, val_dl, device, cfg)
    if plain_check["max_dev"] > 1e-4:
        raise SystemExit("plain-resnet18 export does not reproduce the model")
    print(f"image-only export verified: max dev {plain_check['max_dev']:.2e}")

    print("\nOPERATING POINTS (calibrated):")
    grid = [round(t, 2) for t in np.arange(0.30, 0.91, 0.05)]
    table = threshold_table(folded_probs, val_y, grid)
    recommended = {}
    for target in SPECIFICITY_TARGETS:
        t = pick_threshold_for_specificity(folded_probs, val_y, target)
        if t is None:
            continue
        ev = evaluate(folded_probs, val_y, t)
        recommended[f"specificity_{target:.2f}"] = {"threshold": float(t),
                                                    **jsonable(ev)}
        print(f"  specificity >= {target:.2f} -> threshold {t:.3f}: "
              f"recall={ev['recall']:.3f} precision={ev['precision']:.3f} "
              f"FPR={ev['false_positive_rate']:.3f}")
    print("  RE-DERIVE detect.py's LABEL_WORRYING/LABEL_DETECTED and the mobile "
          "app's kLabelWorrying/kLabelDetected from THIS table before swapping "
          "the checkpoint in. Do NOT reuse v3's 0.50/0.66.")

    _ids, st_prob, st_y = storm_level(folded_probs, val_y, val_storm_keys)
    storm_metrics = {"auc": auc_of(st_y, st_prob), "n_storms": int(len(st_y))}
    for target in SPECIFICITY_TARGETS:
        t = pick_threshold_for_specificity(st_prob, st_y, target)
        if t is not None:
            storm_metrics[f"specificity_{target:.2f}"] = {
                "threshold": float(t), **jsonable(evaluate(st_prob, st_y, t))}
    print(f"\nSTORM-LEVEL held-out: AUC {storm_metrics['auc']:.4f} over "
          f"{len(st_y)} val storms")

    by_basin = breakdown(folded_probs, val_y, val_basins)
    by_source = breakdown(folded_probs, val_y, val_sources)
    io_v = val_basins == "IO"
    io_by_source = {}
    for src in sorted(set(val_sources[io_v].tolist())):
        m = io_v & (val_sources == src)
        _i, p, yy = storm_level(folded_probs[m], val_y[m], val_storm_keys[m])
        io_by_source[src] = {"storm_auc": auc_of(yy, p), "n_storms": int(len(yy)),
                             "n_positive_storms": int(yy.sum()),
                             "frame_auc": auc_of(val_y[m], folded_probs[m]),
                             "n_frames": int(m.sum())}
        print(f"  IO / {src:16s} storm AUC {io_by_source[src]['storm_auc']} "
              f"over {io_by_source[src]['n_storms']} storms")

    best_metrics = evaluate(folded_probs, val_y)
    best_metrics.update({"epoch": ship_epoch,
                         "val_loss": binary_nll(folded_probs, val_y),
                         "val_ece": float(ece_after)})

    config = jsonable({
        "temp_lo": TEMP_LO, "temp_hi": TEMP_HI, "img_size": IMG_SIZE,
        "norm_mean": NORM_MEAN, "norm_std": NORM_STD, "typhoon_grade": TYPHOON_GRADE,
        "early_grade_max": DT_EARLY_GRADE_MAX, "typhoon_wind_kt": TYPHOON_WIND_KT,
        "tcir_early_vmax_kt": TCIR_EARLY_VMAX_KT, "label_rule": label_note(),
        "sources": ["digital_typhoon", "tcir", "mosdac"],
        "mosdac_note": "INSAT-3DR L1C ASIA_MER TIR1 (10.83 um), brightness "
                       "temperature via the file's own IMG_TIR1_TEMP lookup "
                       "table, cropped 128x128 at 0.07 deg (TCIR's geometry) and "
                       "passed through the identical kelvin_to_display_u8 + "
                       "resize_u8 path as the other two sources. Labels are IMD "
                       "3-minute sustained wind >= 64 kt (Very Severe Cyclonic "
                       "Storm), IMD's own category threshold.",
        "basins": sorted(index["basin"].unique().tolist()),
        "model_version": MODEL_VERSION, "sweep_config": cfg,
        "frames_per_storm": FRAMES_PER_STORM,
        "ship_arm": SHIP_ARM, "ship_mode": "image",
        "label_smoothing": LABEL_SMOOTHING,
        "ema_decay": EMA_DECAY, "basin_balance_alpha": BASIN_BALANCE_ALPHA,
        "calibrated": bool(verdict["generalises"]),
        "calibration": {"method": "Platt scaling on val logits, folded into fc",
                        "applied": bool(verdict["generalises"]),
                        "crosscheck_verdict": verdict, "slope_a": a,
                        "temperature": 1.0 / a, "bias_b": b,
                        "ece_before": ece_before, "ece_after": ece_after},
        "recommended_display_thresholds": recommended,
    })
    val_metrics = jsonable(best_metrics)
    json.dumps(config)
    json.dumps(val_metrics)

    torch.save({
        "state_dict": plain_sd, "arch": "resnet18_1ch",
        "full_state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "full_arch": "cyclonenet_v5",
        "full_arch_spec": {"arch": cfg["arch"], "use_aux": bool(cfg["aux"]),
                           "dropout": cfg["dropout"],
                           "use_wind": bool(cfg["wind_aux"] > 0),
                           "aux_dim": AUX_DIM, "aux_emb_dim": AUX_EMB_DIM,
                           "basins_order": list(BASINS_ORDER)},
        "image_only_equivalence_check": plain_check,
        "epoch": ship_epoch, "val_metrics": val_metrics, "config": config,
        "train_storm_ids": sorted(set(index["storm_key"].unique()) - val_keys),
        "val_storm_ids": sorted(val_keys),
        "val_metrics_by_basin": jsonable(by_basin),
        "val_metrics_by_source": jsonable(by_source),
        "val_metrics_io_by_source": jsonable(io_by_source),
        "val_metrics_storm_level": jsonable(storm_metrics),
        "operating_points": jsonable(table),
    }, CKPT_PATH)

    summary = jsonable({
        "task": "early-stage frame -> P(storm later reaches >= 64 kt)",
        "model_version": MODEL_VERSION, "sweep_config": cfg,
        "question": "does adding MOSDAC/INSAT Indian Ocean data help the IO?",
        "phase1_cross_validated_comparison": phase1,
        "mosdac_audit": mos_audit,
        "mosdac_overlap_null_test": null_audit,
        "radial_scale_diagnostic": radial,
        "dedup": dedup_audit, "digital_typhoon_audit": dt_audit,
        "tcir_audit": tcir_audit,
        "n_storms": int(index["storm_key"].nunique()),
        "n_val_storms": len(val_keys),
        "n_train_frames": int(len(tr_idx)), "n_val_frames": int(len(va_idx)),
        "ship_history": hist,
        "calibration": config["calibration"],
        "calibration_crosscheck_folds": cross,
        "reliability_bins_before_calibration": bins_before,
        "reliability_bins_after_calibration": bins_after,
        "recommended_display_thresholds": recommended,
        "best_val_metrics": val_metrics,
        "best_val_metrics_by_basin": by_basin,
        "best_val_metrics_by_source": by_source,
        "val_metrics_io_by_source": io_by_source,
        "val_metrics_storm_level": storm_metrics,
        "operating_points": table,
    })
    with open(METRICS_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== SHIPPED CHECKPOINT, by basin ===")
    print(json.dumps(jsonable(by_basin), indent=2))
    print("\nWrote", CKPT_PATH, METRICS_PATH, INDEX_PATH,
          os.path.join(OUT_DIR, "phase1_cv.json"))



def export_plain_resnet18(net, val_dl, device, cfg):
    """Remap CycloneNet's weights onto a stock torchvision resnet18 and PROVE the
    two score identically in image-only mode.

    Why this is exact rather than approximate: with the aux mask at 0 the aux MLP's
    contribution to fc is `aux_embedding * 0`, i.e. the zero vector, so
        logit = fc.weight[:, :512] @ image_features + fc.bias
    which is precisely what a resnet18 with fc = Linear(512, 1) computes. Dropout is
    an identity at eval. The assertion below is what turns that argument into a
    checked fact before anything is saved.
    """
    import torchvision
    if cfg["arch"] != "resnet18":
        return None, {"skipped": f"arch {cfg['arch']} has no plain-resnet18 form",
                      "max_dev": 0.0, "auc": float("nan")}

    plain = torchvision.models.resnet18(weights=None)
    plain.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    plain.fc = nn.Linear(512, 1)

    src = net.state_dict()
    # net.stem is Sequential(conv1, bn1, relu, maxpool, layer1..layer4)
    name_of = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2",
               "6": "layer3", "7": "layer4"}
    mapped = {}
    for k, v in src.items():
        if k.startswith("stem."):
            head, rest = k[5:].split(".", 1)
            if head not in name_of:
                raise SystemExit(f"unexpected stem key {k}")
            mapped[f"{name_of[head]}.{rest}"] = v.detach().cpu()
    mapped["fc.weight"] = src["fc.weight"][:, :512].detach().cpu().clone()
    mapped["fc.bias"] = src["fc.bias"].detach().cpu().clone()
    missing, unexpected = plain.load_state_dict(mapped, strict=False)
    if unexpected:
        raise SystemExit(f"plain resnet18 export has unexpected keys: {unexpected}")
    if missing:
        raise SystemExit(f"plain resnet18 export is missing keys: {missing}")

    plain.to(device).eval()
    ref, ys = predict_logits(net, val_dl, device, use_meta=0.0)
    outs = []
    with torch.no_grad():
        for imgs_u8, _lab, _w, _a, _s, _wd in val_dl:
            outs.append(plain(to_model_input(imgs_u8.to(device))).squeeze(1)
                        .float().cpu().numpy())
    got = np.concatenate(outs).astype(np.float64)
    dev = float(np.max(np.abs(sigmoid_np(got) - sigmoid_np(ref))))
    plain.cpu()
    return ({k: v.detach().cpu() for k, v in plain.state_dict().items()},
            {"max_dev": dev, "auc": auc_of(ys, sigmoid_np(got)),
             "n_val_frames": int(len(got)),
             "note": "state_dict is a stock resnet18 (1-channel conv1, 1-logit fc) "
                     "that reproduces this model's image-only output exactly; "
                     "full_state_dict is the metadata-capable model."})


# ----------------------------------------------------------------------------
# v5 machinery shared with sweep.py
# ----------------------------------------------------------------------------
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


def auc_of(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(p)))


if __name__ == "__main__":
    main()
