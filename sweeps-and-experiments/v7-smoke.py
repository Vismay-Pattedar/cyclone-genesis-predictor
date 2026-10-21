# ============================================================================
# v7 -- AUC CAMPAIGN round 1 (temporal scale-up).
# Goal: push storm AUC from TEND's 0.7589 toward 0.85 with leak-free techniques.
# Same instrument as v6 (identical folds, seeds, eval rows -- asserted at runtime
# against the attached v6 kernel output), three new configs:
#   TEND5 : 5-ch tendency  [t, d(6h), d(12h), d(18h), d(24h)] -- v6 TEND used 6/12h
#           only; this doubles the temporal reach of the same architecture.
#   SEQ5  : TRUE sequence model -- shared 1-ch ResNet-18 encodes frame t and every
#           VALID past frame (strict lookback, tolerance-banded), masked attention
#           pooling over time, gated residual (init ~1% history; K=1 forward is
#           EXACTLY the single-frame model for any gate value, so the
#           pretrained-init-equivalence constraint is preserved by structure).
#           Frame-dropout keeps K=1 in-distribution.
#   RN50T : ResNet-50 trunk on the proven 3-ch TEND input (capacity lever).
# Plus: per-config epochs, an ENSEMBLE readout of every config's out-of-fold
# predictions, and paired bootstrap of every config directly against v6's REAL
# TEND OOF (kernel_sources attach, alignment-asserted row by row).
# Nothing about the label rule, the folds, the banned-input rules, the demo-storm
# holdout or the climatology floor changes. Every gain must survive the same
# paired-bootstrap instrument that validated TEND itself.
# ============================================================================
"""
Cyclone Genesis Predictor -- Early-intensification prediction, v6 ABLATION SWEEP (TEMPORAL).

Forked from train_kernel_v5c/sweep.py. The measuring instrument (4-fold grouped
CV, identical folds, paired bootstrap) is unchanged and correct; what is new is
the MODEL INPUT.

WHAT v6 ADDS, AND WHY
---------------------
Every model up to v5 sees ONE still frame. Intensification is not a state, it is
a RATE: a cold cloud shield that has been sitting there for two days and one that
consolidated in the last six hours look nearly identical in a single snapshot and
have completely different futures. A single frame is physically incapable of
distinguishing them, which is a hard ceiling no amount of regularisation removes.

Digital Typhoon is HOURLY (verified: 199115.csv has 241 consecutive hourly rows,
each naming its own .h5), so the frames needed to measure that rate already exist
and cost nothing new to obtain. v6 feeds the model a TENDENCY STACK:

    channel 0 : frame at t                      (structure -- what v5 saw)
    channel 1 : frame(t) - frame(t - 6h)        (6-hour tendency)
    channel 2 : frame(t) - frame(t - 12h)       (12-hour tendency)

This is strictly LOOKBACK. It adds no information that a forecaster standing at
time t would not already have on screen, so it does not weaken the "no future
frames, no wind/pressure numbers" honesty claim that defines this task. It is
also what a human analyst actually does: they loop the imagery, they do not stare
at one still.

GRACEFUL DEGRADATION IS A REQUIREMENT, NOT A NICETY
---------------------------------------------------
The shipped deliverables (web app, mobile app, detect.py on a single image) can
supply exactly one frame. So the tendency channels are randomly zeroed during
training with probability TEMPORAL_DROP_P, exactly as the aux branch already is.
One checkpoint therefore serves both:
    tendency present -> best accuracy
    tendency zeroed  -> single-frame mode, still in-distribution, still valid
and BOTH are scored separately below, so the cost of single-frame operation is a
measured number rather than an assumption. Frames with no usable t-6h/t-12h
neighbour (start of track, gaps) get zeroed channels too, which is the same
in-distribution state, not a special case.

RESOLUTION
----------
Digital Typhoon is 512x512 native and v5 trained at 224, discarding ~5x the
pixels on a task whose signal (the first curved band, the first warm core) is
fine-scale. Frames are preloaded at PRELOAD_SIZE and each config picks its own
`img_size`, downsampled on the GPU, so resolution is ablated in the same run
under identical folds instead of across incomparable runs.

DEMO-STORM HOLDOUT (correctness, not tuning)
--------------------------------------------
DEMO_HOLDOUT_STORMS are forced into the validation side of every fold they appear
in and are never trained on. train_kernel_v5_fulldata shipped without this and
swallowed storm 201217 -- the storm the 72-hour headline demo is built on -- into
its TRAIN set, which would have quietly turned "the model had never seen this
storm" into a false claim. Any checkpoint that ships must keep this guarantee.

--- everything below is v5c's rationale, unchanged and still in force ---

This file does NOT ship a model. It answers one question that v3/v4/v4.1 could not:
WHICH changes actually improve the model, as opposed to which changes moved a
number that was inside the noise band all along.

WHY A SWEEP AND NOT ANOTHER SINGLE RUN
--------------------------------------
v3 (0.706), v4 (0.696) and v4.1 (0.690) were each measured on ONE 419-storm
validation split. Frames of a storm share a label, so the effective sample size
of that measurement is 419 storms, not 2,444 frames. The Hanley-McNeil standard
error of an AUC of 0.70 with ~210 positive and ~210 negative storms is about
0.025. Every difference in that table is well under one standard error: on the
evidence available, v3, v4 and v4.1 are statistically indistinguishable, and the
"v4 is worse" / "v4.1 is worse still" conclusions were reading noise.

So this script changes the measuring instrument before changing the model:

  * K-FOLD GROUPED CROSS-VALIDATION (K=4, grouped by storm, stratified by
    (basin,label)). Every one of the ~2,000 storms gets exactly one out-of-fold
    prediction, so the AUC is estimated on ~2,000 storms instead of 419. That
    cuts the standard error to roughly 0.011.
  * IDENTICAL FOLDS FOR EVERY CONFIG, so configs are compared PAIRED. The paired
    bootstrap over storms cancels most of the split-to-split variance and is what
    actually decides whether a change is real.
  * A FIXED EVALUATION FRAME SET for every config (see the frame-selection fix
    below), so the only thing that varies between configs is the model/training.
  * BOTH a best-epoch-selected number AND a fixed-final-epoch number. Selecting
    the best epoch on the same fold you then score is mildly optimistic; the
    final-epoch number has no selection in it at all and is the honest headline.

THE FRAME-SELECTION BUG THIS ALSO FIXES
---------------------------------------
v1-v4 defined an "early-stage" Digital Typhoon frame as ANY row with grade <= 2,
and an "early-stage" TCIR frame as ANY row with Vmax <= 33 kt. A storm's track
goes 2 -> 3 -> 4 -> 5 -> 4 -> 3 -> 2, so those filters also match the DECAY tail
at the end of the storm's life. Verified on real local metadata:

    storm 201519: 25 rows match grade<=2; 7 of them are AFTER the wind peak.
                  2 of the 6 frames v3 actually trained on came from the decay tail.
    storm 197901: 15 rows match; 1 is after the peak, and it IS one of the 6 picks.
    storm 199115: 0 after the peak (it went extratropical, grade 6, and never
                  returned to grade <= 2).

The contamination is ASYMMETRIC. A storm that reaches typhoon strength usually
decays through grade 6 (extratropical) and never returns to grade <= 2, so its
tail is not matched. A storm that never got strong decays 3 -> 2 and its tail IS
matched. So "ragged, decaying, post-peak cloud field" was a feature that
correlated with the NEGATIVE class, learnable in training and absent at
inference -- exactly the kind of shortcut that produces a large train/val gap
(v3 measured 0.919 train-probe vs 0.693 val).

The fix is to truncate each storm at the FIRST time it crosses out of the early
stage, so "early stage" means what the product claims it means: the frames
before the system ever organised. Configs A and B below differ ONLY in this, so
the sweep measures the fix rather than assuming it.

WHAT IS BEING ABLATED (each config adds ONE thing to the one above it)
----------------------------------------------------------------------
  A_v3_repro    v3's exact recipe and v3's frame selection.        [baseline]
  B_cleansel    + the frame-selection fix above.
  C_reg         + real regularisation: dropout, 500x weight decay, much stronger
                  augmentation, cutout. v3's augmentation was deliberately mild
                  because v2's diagnosis was UNDERFITTING; v3's own numbers say
                  the opposite (train probe AUC 0.92 vs val 0.70) and nothing was
                  ever re-tuned after that diagnosis flipped.
  D_aux         + the tabular metadata the pipeline already parses but never fed
                  to the model: latitude, longitude, day-of-year, basin. These
                  are the cheap stand-in for the sea-surface-temperature and
                  seasonal predictors every operational intensity model uses, and
                  they are NOT leaky (a forecaster always knows where and when a
                  storm is; they never know its future wind, which stays excluded).
                  Trained with AUX_DROP_P so the SAME checkpoint still works
                  image-only -- both modes are scored.
  E_soft_wind   + a physically-graded target. The label is "peak wind >= 64 kt",
                  so a storm that peaked at 63 kt and one that peaked at 65 kt get
                  opposite hard labels for a difference that is physically a coin
                  flip, while a 63 kt and a 30 kt storm get the same label. This
                  replaces the flat LABEL_SMOOTHING with sigmoid((wmax-64)/8) and
                  adds an auxiliary head regressing peak wind. Peak wind is a
                  TRAINING TARGET only -- it is never a model input.
  F_small       + a 4x smaller backbone (resnet18 truncated after layer3). Direct
                  test of "11.2M parameters is too much for 1,600 training storms".

Everything is scored at frame level AND at storm level (mean probability over a
storm's early frames), with the false-alarm rate at the specificity operating
points reported for both, because "stop flagging things that are not going to
intensify" is a specificity question, not an AUC question.

RUNTIME: index+preload once (~5 min), then 6 configs x 4 folds x 18 epochs.
Partial results are appended to sweep_results.jsonl after EVERY fold, so a
timeout still leaves every completed config usable.

Runs as a plain Kaggle *script* kernel with GPU enabled (T4).
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

# (v7r2) Stream stdout/stderr line-by-line: Kaggle API-pushed ("Save & Run All")
# sessions only persist logs at version end, and Python block-buffers stdout when
# piped, which left a running kernel completely blind from the outside. Line
# buffering makes every print reach the UI log as it happens. Numerics-neutral.
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(line_buffering=True)
    except Exception:
        pass

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

# ---- v6: temporal tendency stack -------------------------------------------
# Frames are decoded and cached at PRELOAD_SIZE; each config then downsamples to
# its own img_size on the GPU. Preloading at the LARGEST size any config wants is
# what lets 224 and 320 be compared under identical folds in one run. Digital
# Typhoon is 512 native, so 320 is still a downsample, not an upsample.
# (v7) 320 -> 256. The preload buffer is now FIVE channels per frame (see
# TEND_LAGS_H below); at 320px that would be 17,027*5*320*320 = 8.7 GB of RAM,
# which is past what v6 proved fits alongside the rest of the session. At 256px
# the buffer is 5.6 GB -- v6's exact footprint -- and still exceeds the largest
# train-time resolution (224 * 1.139 max zoom = 255.1 px) with one pixel to spare.
# v7 runs no 320px config, so nothing is lost.
PRELOAD_SIZE = 256
# (v7) 6/12 -> 6/12/18/24. v6 proved the temporal signal is real at 6-12h; this
# doubles the lookback reach to a full day. Coverage falls with lag (early-stage
# frames sit near the start of a track), so deeper lags carry LESS information
# per frame -- which is exactly why they are separate configs and per-lag masks,
# not a blind copy of the 6/12 pattern.
TEND_LAGS_H = (6, 12, 18, 24)   # hours of lookback for channels 1..4
# A neighbour is accepted only if its real timestamp is within this many hours of
# the requested lag. Digital Typhoon is hourly but has genuine gaps; silently
# accepting a 30-hour-old frame as "the 6-hour tendency" would be a data bug that
# looks like noise, so it is rejected and the channel is zeroed instead.
TEND_TOL_H = 2
# Tendency channels are differences of [0,1] display values, so they live in
# [-1,1] but concentrate near 0. Dividing by this puts them on roughly the same
# scale as the normalised structure channel instead of being ~10x smaller, which
# would let the trunk ignore them.
TEND_SCALE = 0.15
TEMPORAL_DROP_P = 0.25    # per-sample probability of zeroing BOTH tendency
                          # channels during training -> single-frame inference
                          # stays in-distribution for the shipped one-image apps.
# (v7) SEQ5-only: on top of TEMPORAL_DROP_P (which drops the ENTIRE history),
# each individual past frame is additionally dropped with this probability. The
# current frame is NEVER dropped. Net effect: the model trains on a distribution
# that includes every history length from 1..4, so K=1 deployment (one image, no
# video history -- the mobile app and detect.py single-image mode) remains an
# in-distribution state for these weights, which is the sequence-model version of
# constraint 3 in the multi-frame spec.
SEQ_FRAME_DROP_P = 0.15

# ---- v6: storms that must never be trained on ------------------------------
# 201217 is the storm the 72-hour "AI called it early" demo is built on; the rest
# back the curated demo images. If any of them lands in a training split, the
# demo's central claim ("a storm the model had never seen") becomes false. These
# are forced to the val side of whatever fold they belong to. Held-out storms
# still contribute out-of-fold predictions, so this costs evaluation nothing.
DEMO_HOLDOUT_STORMS = ("201217", "200515", "200312", "200905", "200903")

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
    # ---- v7 round 1. Same folds/seeds/eval rows as v6 (runtime-asserted), so
    # every number below is directly comparable to the published anchors:
    #   L_soft_reg 0.7493 / TEND 0.7589 / TEND_HI 0.7447 (storm AUC, best) and
    #   climatology floor 0.7048. All configs remain IMAGE-ONLY (aux=False):
    # v5 proved the aux-metadata branch reads climatology and degrades the image
    # pathway; that finding is not up for re-litigation here.
    #
    # TEND5 -- same architecture as the validated TEND, but the tendency stack is
    # widened from [d6, d12] to [d6, d12, d18, d24] and each channel is masked by
    # its OWN neighbour's existence (per-lag), so partial histories are used
    # instead of being thrown away. Measures: does a full day of lookback beat
    # 12h, with everything else held fixed?
    dict(name="TEND5", model="cnn", lags=[6, 12, 18, 24], mask="per_lag", tend=True,
         train_sel="clean", dropout=0.3, wd=5e-2,
         aug=AUG_STRONG, aux=False, soft=True, wind_aux=WIND_AUX_WEIGHT,
         arch="resnet18", img_size=224),
    # SEQ5 -- the true sequence model. Same backbone per frame (1-channel
    # ResNet-18, mean-collapsed ImageNet conv1), masked attention pooling over
    # the valid past frames, gated residual starting at ~1% history
    # contribution (the K=1 forward is EXACTLY the single-frame model for ANY
    # gate value -- the equivalence is structural, via the masked softmax).
    # Frame-dropout (TEMPORAL_DROP_P whole-history + SEQ_FRAME_DROP_P per-slot)
    # keeps K=1 deployment in-distribution. Measures: does aggregating genuine
    # frames over time beat collapsing time into difference images?
    dict(name="SEQ5", model="seq", tend=True,
         train_sel="clean", dropout=0.3, wd=5e-2,
         aug=AUG_STRONG, aux=False, soft=True, wind_aux=WIND_AUX_WEIGHT,
         arch="resnet18", img_size=224),
    # RN50T -- pure capacity lever: ResNet-50 trunk on the EXACT validated TEND
    # input AND the exact v6 shared-mask semantics (mask="shared": channels
    # exist only when BOTH the 6h and 12h neighbours exist), so the only change
    # vs published TEND is the backbone. If 2.3x the parameters cannot be
    # converted into AUC on ~12k training frames, that is a real answer about
    # where the ceiling is.
    dict(name="RN50T", model="cnn", lags=[6, 12], mask="shared", tend=True,
         train_sel="clean", dropout=0.3, wd=5e-2,
         aug=AUG_STRONG, aux=False, soft=True, wind_aux=WIND_AUX_WEIGHT,
         arch="resnet50", img_size=224),
]

# Every config key the code reads has a default here, so a config row that omits
# one is well-defined rather than a KeyError halfway through a paid GPU run.
CONFIG_DEFAULTS = dict(img_size=IMG_SIZE, tend=False, aux_drop=AUX_DROP_P,
                       model="cnn", lags=[6, 12], mask="per_lag", epochs=None)


def cfg_get(cfg, key):
    """Config lookup with an explicit default. Unknown keys raise rather than
    silently returning None, so a typo in a config row fails at import."""
    if key not in CONFIG_DEFAULTS:
        raise KeyError(f"{key!r} has no entry in CONFIG_DEFAULTS")
    v = cfg.get(key, CONFIG_DEFAULTS[key])
    return CONFIG_DEFAULTS[key] if v is None else v
# Restrict the sweep from the kernel environment without editing the file, e.g.
# CGP_ONLY_CONFIGS="L_soft_reg,TEND". Empty means "all", which is what v6 wants:
# the whole point of the run is the paired L_soft_reg -> TEND -> TEND_HI chain.
# (This defaulted to v5c's round-2 config NAMES when forked. Those names do not
# exist in v6's table, so the filter selected NOTHING and the run would have spent
# a GPU session training zero models before dying on an IndexError at the summary.
# Caught by the inherited offline suite; the assert below makes it impossible to
# reach a GPU in that state again.)
ONLY_CONFIGS = [s for s in os.environ.get("CGP_ONLY_CONFIGS", "").split(",") if s]
# Smoke mode: 1 fold, 2 epochs. Used by the offline dry run.
SMOKE = os.environ.get("CGP_SMOKE", "1") == "1"
# (v7) SMOKE-ONLY: subsample the storm list to at most this many storms so the
# full data path exercises in minutes. Never read in a real run (see main()).

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

        # ---- (v6) absolute-hour axis for the temporal lookback ----
        # Digital Typhoon is hourly but has real gaps, so a neighbour is located by
        # its actual timestamp, never by "6 rows back" -- row arithmetic would
        # silently hand the model a 30-hour-old frame across a gap and label it a
        # 6-hour tendency. Any row with an unparseable stamp is simply not
        # available as a neighbour.
        abs_h = np.full(len(df), np.nan)
        gm = t_y.notna() & t_m.notna() & t_d.notna() & t_h.notna()
        if gm.any():
            gi = np.where(gm.to_numpy())[0]
            for ii in gi:
                try:
                    dt_ = datetime.datetime(int(t_y.iat[ii]), int(t_m.iat[ii]),
                                            int(t_d.iat[ii]), int(t_h.iat[ii]))
                except ValueError:
                    continue  # a genuinely invalid calendar date, not a gap
                abs_h[ii] = dt_.timestamp() / 3600.0
        fname_arr = fname.to_numpy()

        def neighbour_path(p, lag_h):
            """Path of this storm's frame ~`lag_h` hours BEFORE row p, or "".

            Lookback only: the returned row is always strictly earlier in time than
            p, so this cannot leak anything a forecaster at time p would not have.
            """
            if not np.isfinite(abs_h[p]):
                return ""
            want = abs_h[p] - float(lag_h)
            cand = np.where(np.isfinite(abs_h) & (np.abs(abs_h - want) <= TEND_TOL_H)
                            & (np.arange(len(abs_h)) < p))[0]
            if len(cand) == 0:
                return ""
            best = cand[int(np.argmin(np.abs(abs_h[cand] - want)))]
            nm = str(fname_arr[best]).strip()
            if not nm:
                return ""
            np_path = os.path.join(image_dir, storm_id, nm)
            return np_path if os.path.exists(np_path) else ""

        for p, (sel_v3, sel_clean) in sorted(selection_flags(elig_v3_pos,
                                                             elig_clean_pos).items()):
            rec = df.iloc[int(p)]
            h5_path = os.path.join(image_dir, storm_id, str(rec["file_1"]).strip())
            if not os.path.exists(h5_path):
                audit["n_missing_h5"] += 1
                continue
            lag_paths = [neighbour_path(int(p), lg) for lg in TEND_LAGS_H]
            row = {
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
            # (v7) one tend_path column per configured lag, tend_path_1..N.
            for j in range(len(TEND_LAGS_H)):
                row[f"tend_path_{j + 1}"] = lag_paths[j]
            rows.append(row)

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
                    # TCIR ships one h5 of independently-sampled frames rather than
                    # a dense hourly track, so a trustworthy t-6h/t-12h neighbour
                    # is not reliably available. Left empty on purpose: those
                    # frames simply train and evaluate in single-frame mode, which
                    # TEMPORAL_DROP_P already makes an in-distribution state. This
                    # is a real, reported coverage limit, not a silent fallback --
                    # see the tendency-coverage audit printed in main().
                    "tend_path_1": "",
                    "tend_path_2": "",
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


def resize_u8(u8, size=None):
    """Any HxW uint8 -> size x size uint8, bilinear. Downsamples Digital Typhoon's
    512x512 and upsamples TCIR's 128x128 through the identical path.

    (v6) Defaults to PRELOAD_SIZE, not IMG_SIZE: frames are cached once at the
    largest resolution any config asks for and each config downsamples from that
    cache on the GPU. Caching at 224 and upsampling back to 320 would be a fake
    resolution increase that could only lose the comparison.
    """
    size = PRELOAD_SIZE if size is None else size
    if u8.shape == (size, size):
        return u8
    t = torch.from_numpy(np.ascontiguousarray(u8)).float()[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze().clamp(0, 255).round().byte().numpy()


def to_unit(u8_batch, size=None):
    """uint8 (N,H,W) tensor -> float (N,1,IMG_SIZE,IMG_SIZE) in [0,1].

    Split out from normalisation (v2 did both in one step) for two reasons: the
    augmentation has to happen in this [0,1] DISPLAY domain to be physically
    interpretable, and doing the uint8->float cast on whatever device the tensor
    already lives on lets the training loop ship 1 byte/pixel to the GPU instead
    of 4 and do the cast there.
    """
    # (v6) u8_batch is (N,H,W) for a plain frame or (N,C,H,W) for a tendency
    # stack; both land as (N,C,H,W) here so every downstream step is shared.
    t = (u8_batch.unsqueeze(1) if u8_batch.dim() == 3 else u8_batch).float().div_(255.0)
    if size is None:
        size = IMG_SIZE
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t


def tendency_stack(x01, drop_mask=None):
    """(N,3,H,W) of [0,1] frames [t, t-6h, t-12h] -> the model's 3-channel input.

    channel 0: the frame itself, normalised exactly as every previous model did,
               so with the tendency channels zeroed this reduces to the v5 input.
    channel 1: (frame_t - frame_t-6h)  / TEND_SCALE
    channel 2: (frame_t - frame_t-12h) / TEND_SCALE

    The differences are taken in the [0,1] DISPLAY domain, after augmentation, so
    a zoom/shift/brightness jitter applies identically to all three frames and the
    difference stays a real tendency rather than an augmentation artifact.

    `drop_mask` is 1 for samples whose tendency channels must be zeroed -- either
    because no genuine neighbour existed (start of track, a gap, any TCIR frame)
    or because TEMPORAL_DROP_P selected them for single-frame training. Both
    produce the identical input, which is what makes single-frame inference an
    in-distribution state for the shipped one-image apps rather than a
    degradation the model has never seen.
    """
    cur = x01[:, 0:1]
    d1 = (cur - x01[:, 1:2]) / TEND_SCALE
    d2 = (cur - x01[:, 2:3]) / TEND_SCALE
    if drop_mask is not None:
        keep = (1.0 - drop_mask).view(-1, 1, 1, 1).to(d1.dtype)
        d1 = d1 * keep
        d2 = d2 * keep
    return torch.cat([normalize(cur), d1, d2], dim=1)


def tendency_stack_n(x01, n_lags, drop_lag=None):
    """(v7) Generalisation of tendency_stack to n_lags difference channels.

    x01 is (N, 1+n_lags, H, W) of [0,1] frames [t, lag1, .., lag_n]. Returns
    (N, 1+n_lags, H, W): channel 0 is the normalised frame (identical to every
    previous model), channels 1..n are (frame_t - frame_lag_j) / TEND_SCALE.

    drop_lag is (N, n_lags) with 1 where channel j must be zeroed -- either no
    genuine neighbour existed for that lag, or TEMPORAL_DROP_P selected the
    sample for single-frame training (in which case ALL lag channels drop
    together, exactly the v6 semantics). Unlike v6, a PARTIAL history is valid:
    a frame with a real t-6h neighbour but no t-12h one keeps channel 1 and
    zeros only channel 2. Both are strict-lookback, so neither can leak.
    """
    cur = x01[:, 0:1]
    chans = [normalize(cur)]
    for j in range(1, n_lags + 1):
        d = (cur - x01[:, j:j + 1]) / TEND_SCALE
        if drop_lag is not None:
            # drop_lag == 1 means "zero this channel" (v6 drop_mask convention):
            # multiply by (1 - drop), NEVER by drop itself. Caught by the offline
            # suite: multiplying by drop would zero the channels of frames that
            # HAVE a genuine neighbour and keep them where none exists.
            keep = (1.0 - drop_lag[:, j - 1]).view(-1, 1, 1, 1).to(d.dtype)
            d = d * keep
        chans.append(d)
    return torch.cat(chans, dim=1)


def normalize(x01):
    """[0,1] display tensor -> the exact normalisation detect.py applies."""
    return (x01 - NORM_MEAN) / NORM_STD


def to_model_input(u8_batch, size=None):
    """Full inference-path preprocessing, byte-for-byte what detect.py does.
    Used for validation and for the train-subset probe -- never augmented.

    (v6) A tendency-stacked batch arrives as (N,3,H,W); a non-tendency config must
    see only channel 0, so it is sliced here rather than silently normalising three
    channels into a 1-channel network.
    """
    t = to_unit(u8_batch, size=size)
    if t.shape[1] > 1:
        t = t[:, 0:1]
    return normalize(t)


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
        # (v6) ONE noise field, shared across the tendency channels, not an
        # independent draw per channel. Independent noise would survive the
        # subtraction in tendency_stack as sqrt(2)*noise_std/TEND_SCALE of pure
        # noise -- i.e. it would FABRICATE apparent temporal change where the storm
        # did not actually change, teaching the model to read sensor noise as a
        # tendency. Shared noise perturbs the scene without inventing motion.
        # For a 1-channel config this draws the identical numbers as v5 did (same
        # shape, same generator state), so the control config is unaffected.
        noise = torch.randn((x01.shape[0], 1) + tuple(x01.shape[2:]),
                            generator=gen).to(dev) * p["noise_std"]
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
                 aux=None, soft=None, wind=None, tend_ok=None, ok5=None):
        self.images = images_u8
        # (v6) 1 where a genuine t-6h AND t-12h neighbour was found, else 0. Frames
        # without one are forced into single-frame mode, never fed a fabricated or
        # duplicated "tendency" that would read as zero change when the truth is
        # unknown.
        self.tend_ok = (np.ones(len(labels), np.float32) if tend_ok is None
                        else tend_ok.astype(np.float32))
        # (v7) per-slot validity, shape (N, 1+len(TEND_LAGS_H)). Column 0 is the
        # frame itself (always 1 for rows that survived preload), columns 1..4 are
        # the t-6/12/18/24h neighbours. Supplied by preload_all in v7; the
        # fallback reconstructs the v6-style both-present mask for offline tests.
        if ok5 is None:
            o = np.zeros((len(labels), 1 + len(TEND_LAGS_H)), np.float32)
            o[:, 0] = 1.0
            o[:, 1] = self.tend_ok
            o[:, 2] = self.tend_ok
            self.ok5 = o
        else:
            self.ok5 = ok5.astype(np.float32)
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
                # axes=(-2,-1): a tendency stack is (3,H,W), and the default
                # np.rot90 axes (0,1) would rotate the CHANNEL axis into the image,
                # silently scrambling the frames together.
                img = np.rot90(img, k, axes=(-2, -1)).copy()
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
                torch.tensor(self.wind[i], dtype=torch.float32),
                # (v7) per-slot validity vector instead of the v6 scalar
                torch.from_numpy(self.ok5[i]))


def preload_all(index, tcir_h5_path, want_tend=True):
    """Decode every frame ONCE, resized to PRELOAD_SIZE, into uint8 arrays in RAM.

    Returns (out, ok, tend_ok, ok5):
      out     -- (N,1+L,S,S) uint8 when want_tend, else (N,S,S). Channel 0 is the
                 frame itself; channels 1..L are the t-6h/t-12h/t-18h/t-24h
                 neighbours (L = len(TEND_LAGS_H)); a neighbour that does not
                 exist or failed to decode stays ALL-ZERO -- never a copy of the
                 current frame, which would assert "no change" when the truth is
                 unknown.
      ok      -- False where channel 0 itself failed to decode (those rows are
                 dropped from the run entirely, never fed to the model as zeros).
      tend_ok -- v6-compatible 3-channel coverage: a genuine t-6h AND t-12h
                 neighbour both decoded.
      ok5     -- (N,1+L) per-slot validity. Column 0 mirrors `ok`; column 1+j is
                 True iff the lag TEND_LAGS_H[j] neighbour existed and decoded.
                 This is what TEND5 (per-lag channels) and SEQ5 (per-frame
                 attention masks) read -- partial histories are now first-class:
                 v6 threw away a frame's ONLY 6h neighbour just because its 12h
                 one was missing, which strictly reduced the information
                 available to a strict-lookback model.

    TCIR is done FIRST and deliberately: it is the unknown-latency source (random
    row access into one 6.6GB file), read in ASCENDING frame order so the access
    pattern is as near sequential as it can be. An ETA is printed after 100 frames
    of each pass so a pathologically slow mount is caught in the first minute
    rather than after half an hour.
    """
    import h5py

    n = len(index)
    S = PRELOAD_SIZE
    n_lags = len(TEND_LAGS_H)
    out = (np.zeros((n, 1 + n_lags, S, S), np.uint8) if want_tend
           else np.zeros((n, S, S), np.uint8))
    ok = np.ones(n, dtype=bool)
    tend_ok = np.zeros(n, dtype=bool)
    ok5 = np.zeros((n, 1 + n_lags), dtype=bool)
    sources = index["source"].to_numpy()
    gb = out.nbytes / 1e9
    print(f"\nPreload buffer: {out.shape} uint8 = {gb:.2f} GB "
          f"(PRELOAD_SIZE={S}, tendency={'on' if want_tend else 'off'})")

    def put(pos, arr):
        if want_tend:
            out[pos, 0] = arr
        else:
            out[pos] = arr

    # ---- pass 1: TCIR (channel 0 only -- no dense hourly track, see build_tcir_index)
    tc_pos = np.where(sources == "tcir")[0]
    if len(tc_pos):
        refs = index["frame_ref"].to_numpy()[tc_pos]
        order = np.argsort(refs, kind="stable")
        tc_pos = tc_pos[order]
        print(f"\nPreloading {len(tc_pos)} TCIR frames from {tcir_h5_path}")
        t0 = time.time()
        with h5py.File(tcir_h5_path, "r") as f:
            ds = f["images"]
            for j, pos in enumerate(tc_pos):
                row = int(index["frame_ref"].iat[int(pos)])
                try:
                    put(pos, resize_u8(load_tcir_frame_u8(ds, row)))
                except Exception as exc:
                    print(f"  LOAD FAIL tcir row {row}: {type(exc).__name__}: {exc}")
                    ok[pos] = False
                if j == 99:
                    rate = (time.time() - t0) / 100
                    print(f"  [B] preload rate {rate * 1000:.0f} ms/frame "
                          f"-> ETA {rate * len(tc_pos) / 60:.1f} min for TCIR")
                if (j + 1) % 500 == 0:
                    print(f"  [B] preloaded {j + 1}/{len(tc_pos)} ({time.time() - t0:.0f}s)")
        print(f"[B] TCIR preload done in {time.time() - t0:.0f}s")

    # ---- pass 2: Digital Typhoon, frame + its two lookback neighbours ----
    dt_pos = np.where(sources == "digital_typhoon")[0]
    if len(dt_pos):
        paths = index["h5_path"].to_numpy()
        tp = [(index[f"tend_path_{j}"].to_numpy()
               if f"tend_path_{j}" in index.columns else np.array([""] * n))
              for j in range(1, n_lags + 1)]
        print(f"\nPreloading {len(dt_pos)} Digital Typhoon frames"
              f"{' + tendency neighbours' if want_tend else ''}")
        t0 = time.time()
        n_pair = 0
        for j, pos in enumerate(dt_pos):
            try:
                put(pos, resize_u8(load_dt_frame_u8(paths[pos])))
            except Exception as exc:
                print(f"  LOAD FAIL {paths[pos]}: {type(exc).__name__}: {exc}")
                ok[pos] = False
                continue
            if want_tend:
                # (v7) load EVERY lag independently. v6 required BOTH the 6h and
                # 12h neighbours or gave the frame none at all, which discarded
                # partially-covered frames outright. With per-lag validity a
                # partial history is still a real, strict-lookback history: the
                # model sees exactly the slots that exist, and nothing else.
                got_any = False
                for lj in range(n_lags):
                    p = str(tp[lj][pos])
                    if not p:
                        continue
                    try:
                        out[pos, 1 + lj] = resize_u8(load_dt_frame_u8(p))
                    except Exception as exc:
                        # A neighbour that will not decode is treated exactly
                        # like a neighbour that does not exist: that slot is
                        # masked, never replaced with a copy of the current
                        # frame (which would assert "no change" when the truth
                        # is unknown).
                        print(f"  TEND FAIL lag{TEND_LAGS_H[lj]} {p}: "
                              f"{type(exc).__name__}: {exc}")
                        continue
                    ok5[pos, 1 + lj] = True
                    got_any = True
                if ok5[pos, 1] and ok5[pos, 2]:
                    tend_ok[pos] = True     # v6-compatible 3-channel coverage
                if got_any:
                    n_pair += 1
            if j == 99:
                rate = (time.time() - t0) / 100
                print(f"  [A] preload rate {rate * 1000:.0f} ms/frame "
                      f"-> ETA {rate * len(dt_pos) / 60:.1f} min for Digital Typhoon")
            if (j + 1) % 500 == 0:
                print(f"  [A] preloaded {j + 1}/{len(dt_pos)} ({time.time() - t0:.0f}s)")
        print(f"[A] Digital Typhoon preload done in {time.time() - t0:.0f}s "
              f"({n_pair} frames got at least one genuine neighbour)")

    # Frames whose channel 0 failed are dropped from the run, so they must not be
    # counted as having usable tendency either.
    tend_ok = tend_ok & ok
    ok5[:, 0] = ok
    ok5[~ok] = False
    print(f"Preload total failures: {int((~ok).sum())}/{n}")
    print(f"TENDENCY COVERAGE (v6-compatible, t-6h AND t-12h): "
          f"{int(tend_ok.sum())}/{n} frames ({tend_ok.sum() / max(1, n):.1%}); "
          f"per-lag coverage:")
    for j, lg in enumerate(TEND_LAGS_H):
        print(f"    t-{lg:>2d}h: {int(ok5[:, 1 + j].sum())}/{n} "
              f"({ok5[:, 1 + j].sum() / max(1, n):.1%})")
    return out, ok, tend_ok, ok5


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(arch="resnet18", in_ch=1):
    """resnet18 (or v7: resnet50), ImageNet-pretrained, adapted to `in_ch`
    channels, 1 logit out.

    Byte-for-byte the same architecture v1 and v2 saved, which is what makes this
    checkpoint a drop-in replacement for local_tool/detect.py. NOTHING in v3 adds
    a module, a buffer or a parameter -- the calibration is folded into fc's
    existing weight and bias (see fold_calibration_into_fc) precisely so that
    net.load_state_dict(ckpt['state_dict']) stays a STRICT load over an unmodified
    resnet18.

    (v7) `arch` selects the trunk; both ResNets share the identical conv1 shape
    (64,3,7,7), so the mean-collapse / 0.25-weight channel init below is
    unchanged for either. RN50T is a pure capacity lever on the proven input;
    its fc becomes Linear(2048, 1).
    """
    import torchvision

    if arch == "resnet50":
        ctor = torchvision.models.resnet50
        weights_arg = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    else:
        ctor = torchvision.models.resnet18
        weights_arg = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    pretrained = True
    try:
        net = ctor(weights=weights_arg)
    except Exception as exc:
        # Requires enable_internet: true in kernel-metadata.json (torchvision
        # fetches the weights from download.pytorch.org). Falling back to random
        # init keeps the run alive but will noticeably hurt AUC -- loud on purpose.
        print(f"!! PRETRAINED WEIGHTS UNAVAILABLE ({exc}) -- using RANDOM INIT !!")
        net = ctor(weights=None)
        pretrained = False

    old = net.conv1.weight.data  # (64,3,7,7)
    net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # MEAN (not sum) across the RGB axis: for a grayscale image the pretrained
    # filter's response to a 3x-replicated channel equals the response of the
    # mean-collapsed filter to a single channel. Preserves both the learned
    # edge/texture filters and their output scale.
    collapsed = old.mean(dim=1, keepdim=True)
    if in_ch == 1:
        net.conv1.weight.data = collapsed
    else:
        # (v6) Channel 0 gets the identical mean-collapsed filter, so with the
        # tendency channels zeroed this network is initialised to exactly the
        # 1-channel model -- single-frame mode starts from v5's behaviour and is
        # never a fresh random head. The tendency channels start at a quarter
        # weight: large enough to receive real gradient, small enough that a
        # difference image cannot dominate the structure channel before training
        # has decided how much it is worth.
        w = torch.cat([collapsed] + [collapsed * 0.25] * (in_ch - 1), dim=1)
        net.conv1.weight.data = w

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

    def __init__(self, arch="resnet18", use_aux=False, dropout=0.0, use_wind=False,
                 in_ch=1):
        super().__init__()
        self.in_ch = in_ch
        base, pretrained = build_model(
            "resnet18" if arch == "resnet18_l3" else arch, in_ch)
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
            feat_dim = 2048 if arch == "resnet50" else 512
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


class CycloneSeqNet(nn.Module):
    """(v7) TRUE sequence model: per-frame shared encoder -> masked temporal
    attention pooling -> one logit.

    Architecture (slot 0 = frame at t, slots 1..S-1 = the tolerance-banded
    t-6/12/18/24h lookback frames, all STRICTLY earlier than t):

      f_k = dropout( GAP( ResNet18_1ch(frame_k) ) )        for every VALID slot
      s_k = v^T tanh( W [f_k ; pos_k] )                    temporal attention score
      a   = softmax_k( s_k )                               masked to VALID slots
      h   = sum_k a_k f_k                                  temporally pooled context
      ctx = f_0 + gamma * (h - f_0)                        gated residual
      logit = fc(ctx)                                      (+ peak-wind aux head)

    WHY A GATED RESIDUAL instead of pooling alone -- constraint 7 of the
    multi-frame spec (pretrained-init equivalence). gamma is a scalar gate
    initialised to 0.01, so at initialisation the history contributes ~1% and the
    network is FUNCTIONALLY the proven single-frame ResNet-18 (same mean-collapsed
    conv1, same trunk, fc on the same 512-d pooled feature). This mirrors TEND's
    0.25-weight tendency init: start at (approximately) the proven recipe and let
    training decide how much the new input is worth. gamma is NOT what guarantees
    the single-frame path -- that guarantee is structural and holds for ANY gamma:
    when only slot 0 is valid, the masked softmax puts weight 1.0 on f_0, so
    h == f_0 and ctx = f_0 + gamma*(h - f_0) == f_0 EXACTLY. gamma=0.01 rather
    than a hard 0 for one reason: at exactly 0 the gate blocks ALL gradient to
    the attention module (d loss/d h is multiplied by gamma), and the history
    pathway would stay dead until gamma happened to drift -- the offline suite
    caught exactly that.

    K=1 DEGRADATION (constraint 3). Masked softmax over a single valid slot puts
    weight 1.0 on f_0, so with the whole history invalid the forward pass is the
    plain encoder path regardless of gamma. Training reinforces this: with
    probability TEMPORAL_DROP_P the entire history is dropped, and each surviving
    past frame is then dropped independently with SEQ_FRAME_DROP_P, so the
    weights see every history length 1..S-1 constantly and single-frame inference
    (detect.py one-image mode, the mobile app) stays in-distribution.

    INVALID SLOTS are never encoded-with-garbage: invalid frames still pass
    through the encoder (batching simplicity), but their attention score is set
    to -inf BEFORE the softmax, so they receive exactly 0 weight and cannot
    influence h through any path except gamma*(h - f_0), where their a_k is 0.
    Their features are multiplied by zero weight only in the sum; they never
    touch the current-frame path.

    The encoder encodes all S slots per sample (S<=5 at 224px is cheap on a T4
    with AMP); this keeps the batch shape static for cudnn.benchmark. Compute
    per epoch is therefore ~S/3 of a 3-channel TEND pass per frame -- well
    inside the v6 budget.

    fc stays a plain Linear(512,1) on ctx, so Platt calibration still folds into
    fc exactly as in every previous version (fold_calibration_into_fc).
    """

    def __init__(self, dropout=0.0, use_wind=True, n_slots=5, pe_dim=8):
        super().__init__()
        assert n_slots >= 2, "sequence model needs at least [t, one lookback slot]"
        self.n_slots = n_slots
        base, pretrained = build_model("resnet18", 1)
        self.pretrained = pretrained
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                  base.layer1, base.layer2, base.layer3,
                                  base.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pe = nn.Embedding(n_slots, pe_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(512 + pe_dim, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )
        # Default (small random) init for the score MLP: with the gate at 0.01
        # there is no reason to force uniform attention at init, and a zero-init
        # final projection would block W1's gradient chain on the first backward
        # (second dead-gradient pathology found by the offline suite).
        self.gamma = nn.Parameter(torch.full((1,), 0.01))
        self.fc = nn.Linear(512, 1)
        self.wind_head = nn.Linear(512, 1) if use_wind else None
        self.feat_dim = 512

    def _encode(self, x):
        # x: (B,1,H,W) -> (B,512), dropout applied per frame (train) / off (eval)
        return self.drop(self.pool(self.stem(x)).flatten(1))

    def forward(self, xs, valid):
        # xs:    (B,S,1,H,W) or (B,S,H,W) -- normalised frames, slot 0 == time t
        # valid: (B,S) in {0,1}, slot 0 must be 1 for every sample (asserted).
        if xs.dim() == 5:
            xs = xs[:, :, 0]
        B, S = xs.shape[0], self.n_slots
        assert S == xs.shape[1], f"expected {S} slots, got {xs.shape[1]}"
        assert bool((valid[:, 0] > 0).all().item()), \
            "slot 0 (frame at t) must be valid for every sample"
        f = self._encode(xs.reshape(B * S, 1, *xs.shape[2:])).reshape(B, S, -1)
        pos = self.pe(torch.arange(S, device=xs.device))
        s = self.score_mlp(torch.cat(
            [f, pos.unsqueeze(0).expand(B, -1, -1).to(f.dtype)], dim=2)).squeeze(-1)
        s = s.masked_fill(valid <= 0, float("-inf"))
        a = torch.softmax(s, dim=1)                      # (B,S); invalid slots -> 0
        h = (a.unsqueeze(-1) * f).sum(dim=1)             # (B,512)
        cur = f[:, 0, :]
        ctx = cur + self.gamma.to(f.dtype) * (h - cur)
        logit = self.fc(ctx)
        wind = self.wind_head(ctx) if self.wind_head is not None else None
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
def predict_logits(net, loader, device, tta=False, use_meta=0.0,
                   tend=False, use_tend=1.0, img_size=None, mode=None,
                   readout="full"):
    """Val-side forward pass. Returns raw logits (not probabilities) because the
    calibration is fitted on the logit scale.

    (v7) `mode` selects the input assembly: "seq" (per-frame slots + validity
    masks), "tend5" (5-ch tendency with per-lag masks), "tend" (v6 3-ch, shared
    mask, byte-identical maths) or "plain" (single frame). `readout` is "full"
    (use every temporal input that genuinely exists) or "notend" (single-frame
    deployment readout -- same weights, temporal inputs removed). The default
    mode=None falls back to the v6 tend/plain switch for compatibility.

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
        for imgs_u8, labels, _w, aux, _soft, _wind, tok5 in loader:
            m_kind = mode if mode is not None else ("tend" if tend else "plain")
            if m_kind == "seq":
                x = normalize(to_unit(imgs_u8.to(device), size=img_size))
                valid = tok5.to(device)
                if readout == "notend":
                    # single-frame deployment readout: keep ONLY slot 0
                    valid = valid * torch.cat(
                        [torch.ones_like(valid[:, :1]),
                         torch.zeros_like(valid[:, 1:])], dim=1)
            elif m_kind == "tend5":
                x01 = to_unit(imgs_u8[:, :5].to(device), size=img_size)
                if readout == "notend":
                    drop_lag = torch.ones(x01.shape[0], 4, device=device)
                else:
                    drop_lag = 1.0 - tok5[:, 1:5].to(device)
                x = tendency_stack_n(x01, 4, drop_lag=drop_lag)
            elif m_kind == "tend":
                x01 = to_unit(imgs_u8[:, :3].to(device), size=img_size)
                # v6 semantics: BOTH 3-ch tendency channels share one mask --
                # a frame has them only if a genuine t-6h AND t-12h neighbour
                # existed AND this evaluation mode asks for them. use_tend=0 /
                # readout="notend" is the single-frame deployment readout.
                ut = 0.0 if readout == "notend" else float(use_tend)
                drop = 1.0 - (tok5[:, 1] * tok5[:, 2]).to(device) * ut
                x = tendency_stack(x01, drop_mask=drop)
            else:
                x = to_model_input(imgs_u8.to(device), size=img_size)
            a = aux.to(device)
            # use_meta is the deployment switch: 1 -> the model may use position and
            # date, 0 -> image only, exactly what detect.py can supply.
            m = torch.full((x.shape[0],), float(use_meta), device=device)
            if tta:
                p = torch.zeros(x.shape[0], device=device)
                for k in range(4):
                    xr = torch.rot90(x, k, dims=(2, 3))
                    if m_kind == "seq":
                        logit, _ = net(xr, valid)
                    else:
                        logit, _ = net(xr, a, m)
                    p += torch.sigmoid(logit.squeeze(1).float())
                p /= 4.0
                p = p.clamp(1e-6, 1 - 1e-6)
                outs.append(torch.log(p / (1 - p)).cpu().numpy())
            else:
                if m_kind == "seq":
                    logit, _ = net(x, valid)
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
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    # Input shapes are fixed at (BATCH_SIZE, 1, 224, 224) for every step, which is
    # the case cudnn's autotuner is designed for. Free on a T4.
    torch.backends.cudnn.benchmark = True

    configs = [c for c in SWEEP_CONFIGS
               if (not ONLY_CONFIGS) or c["name"] in ONLY_CONFIGS]
    if not configs:
        raise SystemExit(
            f"No configs selected. CGP_ONLY_CONFIGS={ONLY_CONFIGS!r} matched none "
            f"of {[c['name'] for c in SWEEP_CONFIGS]}. Refusing to start: this "
            f"would otherwise burn a GPU session doing nothing.")
    # Fold ASSIGNMENT is always N_FOLDS wide (so the train fraction is the same in
    # smoke mode as in the real run); only the number of folds actually TRAINED
    # changes. With one fold assigned there would be no training storms at all.
    n_folds = N_FOLDS
    n_folds_run = 1 if SMOKE else N_FOLDS
    n_epochs = 2 if SMOKE else SWEEP_EPOCHS
    print("=" * 72)
    print("Cyclone Genesis Predictor v7 SWEEP (AUC campaign r1: TEND5 / SEQ5 / RN50T)"
          + ("   [SMOKE MODE]" if SMOKE else ""))
    print(f"  {len(configs)} configs x {n_folds_run}/{n_folds} folds x {n_epochs} epochs")
    print(f"  configs: {[c['name'] for c in configs]}")
    print(f"  BATCH={BATCH_SIZE} LR={LR} EMA={EMA_DECAY} AMP={USE_AMP} "
          f"FRAMES_PER_STORM={FRAMES_PER_STORM}")
    print("=" * 72)

    # ------------------------------------------------------------------ discovery
    dt_root = find_dt_root()
    if not dt_root:
        raise SystemExit(f"No typhoon dataset under {INPUT_ROOT} -- check dataset_sources")
    meta_dir, image_dir = resolve_dt_dirs(dt_root)
    print("DT ROOT   =", dt_root)
    print("meta_dir  =", meta_dir)
    print("image_dir =", image_dir)

    tcir_path = find_tcir_h5()
    if not tcir_path:
        raise SystemExit(
            f"No TCSA/TCIR .h5 under {INPUT_ROOT} -- attach "
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

    # (v7) SMOKE-ONLY storm subsample: with CGP_MAX_STORMS set (and active only
    # when SMOKE) keep a deterministic ~1-in-k slice of storms so the whole data
    # path (index -> dedup -> folds -> preload -> train -> summary) exercises in
    # ~15 minutes instead of ~1.5 hours. A real run NEVER enters this branch.
    max_storms = int(os.environ.get("CGP_MAX_STORMS", "400") or 0)
    if SMOKE and max_storms > 0:
        keys_all = sorted(index["storm_key"].astype(str).unique())
        step = max(1, (len(keys_all) + max_storms - 1) // max_storms)
        keep = set(keys_all[::step])
        index = index[index["storm_key"].astype(str).isin(keep)].reset_index(drop=True)
        print(f"[SMOKE] subsampled to {len(keep)} storms "
              f"({len(index)} frames) via CGP_MAX_STORMS={max_storms}")

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

    # ------------------------------------------------------------------ folds
    # GROUPED BY STORM, NEVER BY FRAME (frames of one storm are hours apart, look
    # nearly identical and share one label), and stratified by (basin, label) so
    # the Indian Ocean -- ~84 storms -- cannot land in a fold with no positives.
    #
    # Unlike v3's single 80/20 split, every storm is assigned to exactly one fold
    # and therefore contributes exactly one out-of-fold prediction. The pooled
    # out-of-fold AUC is measured on ALL the storms, not on 20% of them.
    storms = index.drop_duplicates("storm_key")[["storm_key", "basin", "label"]]
    rng = np.random.RandomState(SEED)
    fold_of = {}
    for (basin, label), grp in storms.groupby(["basin", "label"], sort=True):
        keys = np.array(sorted(grp["storm_key"].tolist()))
        rng.shuffle(keys)
        for i, k in enumerate(keys):
            fold_of[k] = i % n_folds       # round robin == near-exactly equal folds
        print(f"  fold-assign {basin:5s} label={label}: {len(keys):5d} storms")
    index["fold"] = index["storm_key"].map(fold_of).astype(int)

    print(f"\nStorms: {index['storm_key'].nunique()}  folds={n_folds}")
    for f in range(n_folds):
        sub = index[index["fold"] == f].drop_duplicates("storm_key")
        print(f"    fold {f}: {len(sub):5d} storms, {int(sub['label'].sum()):4d} positive")
    print("Eligible ROWS per selection rule (before the 6-frames-per-storm pick):")
    for tag, aud in (("digital_typhoon", dt_audit), ("tcir", tcir_audit)):
        ev3 = aud.get("n_frames_eligible_v3", 0)
        evc = aud.get("n_frames_eligible_clean", 0)
        print(f"    {tag:16s} v3 rule {ev3:7d}   clean rule {evc:7d}   "
              f"POST-CROSSING (decay tail) {ev3 - evc:7d} "
              f"({(ev3 - evc) / max(1, ev3):.1%} of what the v3 rule matched)")
        print(f"    {tag:16s} storms with no clean early frame at all: "
              f"{aud.get('n_no_clean_early_frame', 0)}")
    print("Frames eligible under each selection rule:")
    print(f"    v3 rule    (grade<=2 / Vmax<=33 anywhere): {int(index['sel_v3'].sum())}")
    print(f"    clean rule (before the first crossing)   : {int(index['sel_clean'].sum())}")
    both = int((index["sel_v3"] & index["sel_clean"]).sum())
    print(f"    picked by BOTH rules                     : {both}")
    print(f"    picked ONLY by the v3 rule (decay-tail contaminated): "
          f"{int((index['sel_v3'] & ~index['sel_clean']).sum())}")

    # ------------------------------------------------------------------ preload
    want_tend = any(bool(cfg_get(c, "tend")) for c in configs)
    images, ok, tend_ok_all, ok5_all = preload_all(index, tcir_path, want_tend=want_tend)
    index = index[ok].reset_index(drop=True)
    images = images[ok]
    tend_ok_f = tend_ok_all[ok].astype(np.float32)
    ok5_f = ok5_all[ok].astype(np.float32)
    n_load_failures = int((~ok).sum())
    index["tend_ok"] = tend_ok_f.astype(bool)
    # (v7) persist per-slot validity so a crashed run's csv still explains
    # exactly what the models were shown.
    for j in range(1 + len(TEND_LAGS_H)):
        index[f"seq_ok_{j}"] = ok5_f[:, j].astype(bool)
    index.to_csv(INDEX_PATH, index=False)

    aux_all, aux_ok = build_aux_features(index)
    print(f"\nAux metadata parsed for {int(aux_ok.sum())}/{len(index)} frames "
          f"({aux_ok.mean():.2%}); frames without it get a zero vector and mask=0.")

    y = index["label"].to_numpy().astype(np.float32)
    wmax = pd.to_numeric(index["max_wind_kt"], errors="coerce").to_numpy(np.float64)
    y_soft = 1.0 / (1.0 + np.exp(-(np.nan_to_num(wmax, nan=64.0) - TYPHOON_WIND_KT)
                                 / SOFT_LABEL_KT))
    y_soft = np.clip(y_soft, 0.01, 0.99).astype(np.float32)
    wind_t = (np.clip(np.nan_to_num(wmax, nan=64.0), 0, WIND_CLIP_KT)
              / WIND_NORM_KT).astype(np.float32)
    folds = index["fold"].to_numpy()
    sel_clean = index["sel_clean"].to_numpy().astype(bool)
    sel_v3 = index["sel_v3"].to_numpy().astype(bool)

    # ---- DEMO-STORM HOLDOUT (correctness, not tuning) ----------------------
    # Every storm the presentation is built on must be OUT of training in every
    # fold, not merely out of one. Folds are grouped by storm, so a storm already
    # sits in exactly one fold and is therefore already untrained in that fold --
    # what this block does is make that guarantee explicit, verify it, and fail
    # loudly if it ever stops holding. train_kernel_v5_final and
    # train_kernel_v5_fulldata both shipped with storm 201217 -- the storm the
    # 72-hour headline demo is built on -- inside their TRAIN split, which would
    # have quietly falsified "a storm the model had never seen" on stage.
    storm_keys_all = index["storm_key"].to_numpy().astype(str)
    demo_present = [d for d in DEMO_HOLDOUT_STORMS if (storm_keys_all == d).any()]
    print(f"\nDEMO-STORM HOLDOUT: {len(demo_present)}/{len(DEMO_HOLDOUT_STORMS)} "
          f"demo storms present in this dataset: {demo_present}")
    for d in demo_present:
        m = storm_keys_all == d
        fs = sorted(set(folds[m].tolist()))
        assert len(fs) == 1, (
            f"demo storm {d} spans folds {fs}; grouped CV must keep a storm whole "
            f"or it would be trained on in one fold and scored in another")
        print(f"    storm {d}: fold {fs[0]} -> never trained on in that fold, "
              f"and it is the fold that scores it")
    # The guarantee a shipped checkpoint has to satisfy is stronger than the CV
    # one, so it is recorded here for the ship step to honour rather than left to
    # be rediscovered later.
    demo_holdout_note = {
        "storms": list(DEMO_HOLDOUT_STORMS),
        "present": demo_present,
        "fold_of": {d: int(folds[storm_keys_all == d][0]) for d in demo_present},
        "requirement": ("a shipped single-split checkpoint must place these storms "
                        "in VAL; v5_final and v5_fulldata both violated this"),
    }

    # THE EVALUATION SET, FIXED FOR EVERY CONFIG: only genuine pre-intensification
    # frames. Configs differ in what they TRAIN on, never in what they are scored on,
    # so every comparison below is paired on identical rows.
    # In smoke mode only fold 0 is trained, so only fold 0's storms are scored --
    # otherwise the out-of-fold arrays would be full of holes.
    folds_run = list(range(n_folds_run))
    eval_pos = np.where(sel_clean & np.isin(folds, folds_run))[0]
    ev_y = y[eval_pos].astype(int)
    ev_storm = index["storm_key"].to_numpy()[eval_pos]
    ev_basin = index["basin"].to_numpy()[eval_pos]
    ev_source = index["source"].to_numpy()[eval_pos]
    ev_fold = folds[eval_pos]
    # (v6) coverage mask over the SAME rows as every other ev_* array
    ev_tend = tend_ok_f[eval_pos].astype(bool)
    print(f"Evaluation frames (clean selection, folds {folds_run}): {len(eval_pos)} "
          f"from {len(set(ev_storm.tolist()))} storms, {int(ev_y.sum())} positive")

    # storm-level aggregation index (mean probability over a storm's early frames)
    st_ids, st_pos = np.unique(ev_storm, return_inverse=True)
    st_n = np.bincount(st_pos).astype(np.float64)
    st_y = np.zeros(len(st_ids), dtype=int)
    st_basin = np.empty(len(st_ids), dtype=object)
    st_fold = np.zeros(len(st_ids), dtype=int)
    for i in range(len(ev_y)):
        st_y[st_pos[i]] = ev_y[i]
        st_basin[st_pos[i]] = ev_basin[i]
        st_fold[st_pos[i]] = ev_fold[i]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(USE_AMP and device.type == "cuda")
    print("\nDevice:", device,
          torch.cuda.get_device_name(0) if device.type == "cuda" else "",
          f"AMP={use_amp}")

    # ------------------------------------------------- TENDENCY-AVAILABILITY LEAK
    # The tendency channels are strict lookback, so they cannot carry future
    # information. But their AVAILABILITY might: a storm that organises quickly has
    # a SHORT early-stage window, so its early frames are less likely to have 12h
    # of prior imagery behind them. If "tendency was available" correlates with the
    # label, the model could learn to read ZEROED CHANNELS as a prediction -- which
    # is exactly the shape of the shortcut that made Gate 2 v1 score 0.99995 by
    # detecting its own data pipeline. So it is measured, not assumed away.
    #
    # This is the direct analogue of Gate 2 v2's source_prior_auc check: score the
    # nuisance variable ALONE against the label and see how far above chance it is.
    tend_prior_frame = auc_of(ev_y, ev_tend.astype(np.float64))
    st_tend_frac = np.bincount(st_pos, weights=ev_tend.astype(float)) / st_n
    tend_prior_storm = auc_of(st_y, st_tend_frac)
    print("")
    print("TENDENCY-AVAILABILITY LEAK CHECK (does a neighbour existing "
          "predict the label on its own?)")
    print(f"    frame AUC {tend_prior_frame:.4f}   "
          f"storm AUC {tend_prior_storm:.4f}   (0.5 = no leak)")
    print(f"    coverage: {int(ev_tend.sum())}/{len(ev_tend)} eval frames "
          f"({ev_tend.mean():.1%})")
    if max(abs(tend_prior_frame - 0.5), abs(tend_prior_storm - 0.5)) > 0.05:
        print("    !! ABOVE THE 0.05 BAND. Availability carries real signal, so a "
              "TEND-vs-control gain on the POOLED set is not trustworthy on its "
              "own -- read the covered_subset numbers, where every config is "
              "scored on the identical fully-covered frames and this nuisance "
              "variable is constant by construction.")
    else:
        print("    within the 0.05 band -- availability is close to independent of "
              "the label, so the pooled comparison is not confounded by it.")
    tend_leak = {"frame_auc": tend_prior_frame, "storm_auc": tend_prior_storm,
                 "coverage_frac": float(ev_tend.mean()),
                 "band": 0.05,
                 "leaky": bool(max(abs(tend_prior_frame - 0.5),
                                   abs(tend_prior_storm - 0.5)) > 0.05)}

    # ------------------------------------------------------------- climatology ref
    # A logistic regression on the aux metadata ALONE, same folds, no image at all.
    # This is the number every image model has to beat to have earned its existence:
    # a storm's basin, latitude and season already carry a base rate.
    clim = climatology_baseline(aux_all[eval_pos], ev_y, ev_fold, st_pos, st_y)
    print(f"\nCLIMATOLOGY-ONLY REFERENCE (logistic regression on lat/lon/season/basin,"
          f" no image):\n    frame AUC {clim['frame_auc']:.4f}   "
          f"storm AUC {clim['storm_auc']:.4f}")
    print("    Any image model scoring at or below this has learned nothing from "
          "the picture.")

    # ------------------------------------------------- v6 anchor (paired vs REAL TEND)
    # The attached kernel source (vismaypattedar/cyclone-genesis-predictor-v6-sweep) carries the
    # published run's dataset_index.csv and oof_TEND.npy. If present, assert that
    # this run rebuilt the IDENTICAL index (same rows, same folds, same clean
    # selection, same eval order) and then every config and ensemble below gets a
    # paired bootstrap directly against v6's actual TEND out-of-fold predictions
    # -- the strongest possible comparison, no reliance on memory or
    # transcription. A FAILED alignment check aborts the run: a silently
    # misaligned anchor would poison every paired number it produces.
    anchor_logits = None
    anchor_cmp = {}
    if not SMOKE:
        anchor_dir = None
        for dirpath, dirnames, filenames in os.walk(INPUT_ROOT):
            dirnames[:] = [d for d in dirnames if "typhoon" not in d.lower()]
            if "oof_TEND.npy" in filenames and "dataset_index.csv" in filenames:
                anchor_dir = dirpath
                break
        if anchor_dir is None:
            print("!! v6 anchor NOT FOUND -- comparisons vs TEND fall back to "
                  "absolute numbers only (folds are identical by construction).")
        else:
            v6_idx = pd.read_csv(os.path.join(anchor_dir, "dataset_index.csv"))
            if len(v6_idx) != len(index):
                raise SystemExit(f"v6 anchor rows {len(v6_idx)} != v7 rows "
                                 f"{len(index)} -- dataset drift; refusing.")
            for col in ("storm_key", "fold"):
                a = v6_idx[col].to_numpy().astype(str)
                b = index[col].to_numpy().astype(str)
                if not (a == b).all():
                    raise SystemExit(f"v6 anchor '{col}' column differs -- "
                                     "fold/data drift; refusing.")
            for col in ("sel_clean", "tend_ok"):
                a = v6_idx[col].to_numpy().astype(bool)
                b = index[col].to_numpy().astype(bool)
                if not (a == b).all():
                    raise SystemExit(f"v6 anchor '{col}' differs -- refusing.")
            cand = np.load(os.path.join(anchor_dir, "oof_TEND.npy"))
            v6_eval = v6_idx[v6_idx["sel_clean"].astype(bool)]
            if not (len(cand) == len(v6_eval) == len(eval_pos)):
                raise SystemExit(f"anchor OOF n={len(cand)} vs eval n={len(eval_pos)} "
                                 "-- refusing.")
            if not (v6_eval["storm_key"].to_numpy().astype(str)
                    == ev_storm.astype(str)).all():
                raise SystemExit("anchor eval storm ORDER differs -- refusing.")
            anchor_logits = cand.astype(np.float64)
            print(f"\nv6 ANCHOR loaded from {anchor_dir}")
            print(f"    TEND OOF n={len(cand)}; index/folds/eval-order identity "
                  "ASSERTED -- paired comparisons below are exact.")

    # ------------------------------------------------------------------ the sweep
    results = {}
    oof_store = {}          # (v7) keep every config's OOF arrays for the ensembles
    open(RESULTS_PATH, "w").close()
    t_sweep = time.time()
    for ci, cfg in enumerate(configs):
        name = cfg["name"]
        ep = cfg_get(cfg, "epochs") or n_epochs   # (v7) per-config epoch budget
        print("\n" + "=" * 72)
        print(f"CONFIG {ci + 1}/{len(configs)}: {name}  (epochs={ep})")
        print(f"  {cfg}")
        print("=" * 72)
        oof = {k: np.full(len(eval_pos), np.nan)
               for k in ("best", "final", "tta", "best_nometa", "best_notend")}
        fold_logs = []
        for f in folds_run:
            t0 = time.time()
            tr_sel = sel_v3 if cfg["train_sel"] == "v3" else sel_clean
            tr_idx = np.where(tr_sel & (folds != f))[0]
            va_local = np.where(ev_fold == f)[0]        # positions within eval_pos
            va_idx = eval_pos[va_local]

            out = run_one_fold(cfg, f, ep, images, y, y_soft, wind_t, aux_all,
                               aux_ok, index, tr_idx, va_idx, device, use_amp,
                               tend_ok=tend_ok_f, ok5=ok5_f)
            for k in oof:
                oof[k][va_local] = out[k]
            fold_logs.append(out["log"])
            print(f"  fold {f}: best_ep={out['log']['best_epoch']} "
                  f"val_auc(best)={out['log']['best_auc']:.4f} "
                  f"val_auc(final)={out['log']['final_auc']:.4f} "
                  f"train_probe={out['log']['final_probe_auc']:.4f} "
                  f"gap={out['log']['final_probe_auc'] - out['log']['final_auc']:+.3f} "
                  f"({time.time() - t0:.0f}s)")
            with open(RESULTS_PATH, "a") as fh:
                fh.write(json.dumps(jsonable(
                    {"config": name, "fold": f, **out["log"]})) + "\n")

        res = summarise_config(name, cfg, oof, ev_y, ev_basin, ev_source, ev_fold,
                               st_pos, st_y, st_basin, st_n, fold_logs,
                               ev_tend=ev_tend)
        results[name] = res
        oof_store[name] = {k: v.copy() for k, v in oof.items()}
        print_config_summary(res, clim)
        with open(SUMMARY_PATH, "w") as fh:
            json.dump(jsonable({"climatology_reference": clim,
                                "configs": results,
                                "n_folds": n_folds, "n_epochs": n_epochs,
                                "n_eval_frames": len(eval_pos),
                                "n_eval_storms": len(st_ids)}), fh, indent=2)
        np.save(os.path.join(OUT_DIR, f"oof_{name}.npy"), oof["final"])
        print(f"  [elapsed {(time.time() - t_sweep) / 60:.1f} min total]")

    # --------------------------------------------------------- paired comparisons
    print("\n" + "=" * 72)
    print("PAIRED COMPARISON vs the baseline config, storm-level, bootstrap over "
          f"{len(st_ids)} storms")
    print("=" * 72)
    base_name = configs[0]["name"]
    comparisons = {}
    if len(configs) > 1:
        base = results[base_name]["storm_prob_final"]
        for cfg in configs[1:]:
            cur = results[cfg["name"]]["storm_prob_final"]
            comparisons[cfg["name"]] = paired_bootstrap_auc(
                np.asarray(base), np.asarray(cur), st_y, SEED + 99)
            c = comparisons[cfg["name"]]
            print(f"  {cfg['name']:14s} vs {base_name}: dAUC = {c['delta']:+.4f}  "
                  f"95% CI [{c['lo']:+.4f}, {c['hi']:+.4f}]  "
                  f"P(better) = {c['p_better']:.3f}"
                  f"   {'REAL' if c['lo'] > 0 else ('WORSE' if c['hi'] < 0 else 'not distinguishable from noise')}")

    # --------------------------------------------------- paired vs the v6 ANCHOR
    # The question the whole campaign answers: did ANY of this beat the actual
    # published TEND predictions on the same storms, not a recollection of them?
    if anchor_logits is not None:
        a_prob = sigmoid_np(anchor_logits)
        a_storm = np.bincount(st_pos, weights=a_prob) / st_n
        print("\n" + "=" * 72)
        print("PAIRED vs v6 TEND (the published 0.7589 run, real OOF, same folds)")
        print("=" * 72)
        for cfg in configs:
            cur = results[cfg["name"]]["storm_prob_final"]
            c = paired_bootstrap_auc(np.asarray(a_storm), np.asarray(cur),
                                     st_y, SEED + 98)
            anchor_cmp[cfg["name"]] = c
            print(f"  {cfg['name']:14s} vs TEND(v6): dAUC = {c['delta']:+.4f}  "
                  f"95% CI [{c['lo']:+.4f}, {c['hi']:+.4f}]  "
                  f"P(better) = {c['p_better']:.3f}"
                  f"   {'REAL' if c['lo'] > 0 else ('WORSE' if c['hi'] < 0 else 'noise')}")
        np.save(os.path.join(OUT_DIR, "anchor_v6_TEND_oof.npy"), anchor_logits)

    # ------------------------------------------------------------- ensembles (v7)
    # Different architectures make different mistakes, and v6's own numbers
    # already showed a trivial 4-view TTA ensemble buying +0.005. A probability
    # average of OOF predictions is free at deployment (average two model
    # outputs) and is scored here for every 2+ member combination, ALWAYS with a
    # paired bootstrap against the strongest SINGLE member: an ensemble that
    # does not beat its best member is not worth shipping.
    ensembles = {}
    ens_members = {name: sigmoid_np(v["final"]) for name, v in oof_store.items()}
    if anchor_logits is not None:
        ens_members["TEND_v6"] = a_prob
    if len(ens_members) >= 2:
        import itertools
        names_e = list(ens_members.keys())
        combos = []
        for r in range(2, len(names_e) + 1):
            combos.extend(itertools.combinations(names_e, r))
        best_single = max(results.items(),
                          key=lambda kv: kv[1]["storm_auc_final"])[0]
        best_single_storm = np.bincount(
            st_pos, weights=ens_members[best_single]) / st_n
        print("\n" + "=" * 72)
        print(f"ENSEMBLES (probability average of OOF; vs best single = {best_single})")
        print("=" * 72)
        for combo in combos:
            tag = "+".join(combo)
            p_frame = np.mean([ens_members[n] for n in combo], axis=0)
            p_storm = np.bincount(st_pos, weights=p_frame) / st_n
            c = paired_bootstrap_auc(np.asarray(best_single_storm),
                                     np.asarray(p_storm), st_y, SEED + 97)
            ensembles[tag] = {
                "members": list(combo),
                "frame_auc": auc_of(ev_y, p_frame),
                "storm_auc": auc_of(st_y, p_storm),
                "vs_best_single": {"name": best_single, **c},
            }
            e = ensembles[tag]
            print(f"  {tag}: frame {e['frame_auc']:.4f} storm {e['storm_auc']:.4f}"
                  f"   vs {best_single}: dAUC {c['delta']:+.4f} "
                  f"CI [{c['lo']:+.4f},{c['hi']:+.4f}]"
                  f"{'  <-- REAL' if c['lo'] > 0 else ''}")
            safe = tag.replace("+", "_vs_")
            np.save(os.path.join(OUT_DIR, f"ens_prob_{safe}.npy"), p_frame)

    summary = jsonable({
        "note": "v7 AUC-campaign sweep r1 (TEND5/SEQ5/RN50T + ensembles). Not a "
                "shipping model. See sweep_summary.json for full per-config metrics.",
        "n_folds": n_folds, "n_epochs": n_epochs,
        "n_storms": int(index["storm_key"].nunique()),
        "n_eval_frames": len(eval_pos), "n_eval_storms": len(st_ids),
        "n_load_failures": n_load_failures,
        "frame_selection": {
            "v3_rule_frames": int(sel_v3.sum()),
            "clean_rule_frames": int(sel_clean.sum()),
            "v3_only_frames_decay_contaminated": int((sel_v3 & ~sel_clean).sum()),
        },
        "digital_typhoon_audit": dt_audit,
        "tcir_audit": tcir_audit,
        "dedup": dedup_audit,
        "climatology_reference": clim,
        # (v6) Both of these are load-bearing for how the results may be quoted.
        # demo_holdout proves the presentation's storms were never trained on;
        # tendency_coverage says what fraction of frames the temporal configs
        # could actually USE the new input on, so a TEND-vs-control delta is read
        # against real coverage instead of an assumed 100%.
        "demo_holdout": demo_holdout_note,
        "tendency_availability_leak": tend_leak,
        "tendency_coverage": {
            "n_frames_with_full_pair": int(tend_ok_f.sum()),
            "n_frames": int(len(tend_ok_f)),
            "frac": float(tend_ok_f.mean()) if len(tend_ok_f) else 0.0,
            "lags_h": list(TEND_LAGS_H),
            "tol_h": TEND_TOL_H,
            "temporal_drop_p": TEMPORAL_DROP_P,
            "preload_size": PRELOAD_SIZE,
            "note": ("TCIR frames have no dense hourly track and always run "
                     "single-frame; Digital Typhoon frames get a genuine pair "
                     "whenever both neighbours exist within tol_h and decode."),
        },
        "configs": results,
        "paired_vs_baseline": comparisons,
        "baseline_config": base_name,
        # (v7)
        "paired_vs_v6_tend": anchor_cmp,
        "ensembles": ensembles,
        "preload_size": PRELOAD_SIZE,
        "tend_lags_h": list(TEND_LAGS_H),
        "seq_frame_drop_p": SEQ_FRAME_DROP_P,
    })
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)

    # A tidy leaderboard, printed last so it is the first thing visible in the log tail.
    print("\n" + "=" * 78)
    print("LEADERBOARD (out-of-fold, all storms; 'final' = fixed last epoch, no "
          "epoch selection)")
    print("=" * 78)
    print(f"{'config':14s} {'frameAUC':>9s} {'stormAUC':>9s} {'+TTA':>7s} "
          f"{'imgonly':>8s} {'FPR@sp90':>9s} {'prec@sp90':>10s} {'rec@sp90':>9s} "
          f"{'ECE':>6s} {'gap':>6s}")
    print(f"{'climatology':14s} {clim['frame_auc']:9.4f} {clim['storm_auc']:9.4f} "
          f"{'-':>7s} {'-':>8s} {'-':>9s} {'-':>10s} {'-':>9s} {'-':>6s} {'-':>6s}")
    if anchor_logits is not None:
        a_storm_row = np.bincount(st_pos, weights=sigmoid_np(anchor_logits)) / st_n
        print(f"{'TEND(v6anchor)':14s} {auc_of(ev_y, sigmoid_np(anchor_logits)):9.4f} "
              f"{auc_of(st_y, a_storm_row):9.4f}   <- the number to beat")
    for name, r in results.items():
        sp = r["storm_operating_points"].get("specificity_0.90", {})
        print(f"{name:14s} {r['frame_auc_final']:9.4f} {r['storm_auc_final']:9.4f} "
              f"{r['storm_auc_tta']:7.4f} {r['storm_auc_nometa']:8.4f} "
              f"{sp.get('false_positive_rate', float('nan')):9.3f} "
              f"{sp.get('precision', float('nan')):10.3f} "
              f"{sp.get('recall', float('nan')):9.3f} "
              f"{r['frame_ece_calibrated']:6.3f} {r['mean_train_val_gap']:+6.3f}")
    print("\nWrote", SUMMARY_PATH, RESULTS_PATH, INDEX_PATH)


# ----------------------------------------------------------------------------
# v5 sweep machinery
# ----------------------------------------------------------------------------
def amp_autocast(enabled):
    """torch.amp is the current API; torch.cuda.amp is the pre-2.4 one. Kaggle's
    image moves, so support both rather than pinning a version."""
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


def climatology_baseline(aux, y, fold, st_pos, st_y):
    """Cross-validated logistic regression on the aux metadata only.

    The point of this number is honesty about attribution: latitude, season and
    basin carry a real base rate, so part of any gain from config D is climatology
    rather than image understanding. This says exactly how much.
    """
    from sklearn.linear_model import LogisticRegression
    p = np.full(len(y), np.nan)
    for f in sorted(set(fold.tolist())):
        tr, va = fold != f, fold == f
        if len(np.unique(y[tr])) < 2:
            continue
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(aux[tr], y[tr])
        p[va] = lr.predict_proba(aux[va])[:, 1]
    p = np.nan_to_num(p, nan=float(np.mean(y)))
    sp = np.bincount(st_pos, weights=p) / np.bincount(st_pos)
    return {"frame_auc": auc_of(y, p), "storm_auc": auc_of(st_y, sp)}


def run_one_fold(cfg, fold, n_epochs, images, y, y_soft, wind_t, aux_all, aux_ok,
                 index, tr_idx, va_idx, device, use_amp, tend_ok=None, ok5=None):
    """Train one (config, fold) and return out-of-fold logits for the val storms.

    Returns four prediction variants, all on the SAME val frames:
      best        -- EMA weights of the epoch with the best val AUC  (v3's rule;
                     mildly optimistic, because the epoch is chosen on this fold)
      final       -- EMA weights of the last epoch (NO selection at all -> the
                     honest headline number)
      tta         -- final weights, 4-rotation test-time average
      best_nometa -- best weights with the aux branch masked off, i.e. what the
                     same checkpoint scores with no position/date available
    """
    torch.manual_seed(SEED + 100 * fold)
    np.random.seed(SEED + 100 * fold)

    tr_w = basin_balance_weights(index["basin"].to_numpy()[tr_idx],
                                 BASIN_BALANCE_ALPHA)
    aux_tr = aux_all[tr_idx].copy()
    # build_aux_features() zeroes the vector of any frame whose metadata failed to
    # parse, and every parseable frame has a nonzero hemisphere entry and a basin
    # one-hot, so "all-zero aux vector" is an exact, checkable proxy for "no
    # metadata" -- used in the training loop to force mask 0 on those frames.
    assert aux_ok.shape[0] == aux_all.shape[0]

    if tend_ok is None:
        tend_ok = np.ones(len(y), np.float32)
    if ok5 is None:
        ok5 = np.zeros((len(y), 1 + len(TEND_LAGS_H)), np.float32)
        ok5[:, 0] = 1.0
        ok5[:, 1] = tend_ok
        ok5[:, 2] = tend_ok
    use_tend_cfg = bool(cfg_get(cfg, "tend"))
    model_kind = cfg_get(cfg, "model")
    n_lags = len(cfg_get(cfg, "lags"))
    # (v7) which predict_logits path this config uses
    pred_mode = ("seq" if model_kind == "seq"
                 else ("tend5" if use_tend_cfg and n_lags > 2 else
                       ("tend" if use_tend_cfg else "plain")))
    img_size = int(cfg_get(cfg, "img_size"))
    train_ds = FrameDataset(images[tr_idx], y[tr_idx], tr_w, augment=True,
                            aux=aux_tr, soft=y_soft[tr_idx], wind=wind_t[tr_idx],
                            tend_ok=tend_ok[tr_idx], ok5=ok5[tr_idx])
    val_ds = FrameDataset(images[va_idx], y[va_idx], np.ones(len(va_idx)),
                          augment=False, aux=aux_all[va_idx],
                          soft=y_soft[va_idx], wind=wind_t[va_idx],
                          tend_ok=tend_ok[va_idx], ok5=ok5[va_idx])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    probe_sel = np.random.RandomState(SEED + 1).choice(
        len(tr_idx), min(TRAIN_PROBE_FRAMES, len(tr_idx)), replace=False)
    probe_ds = FrameDataset(images[tr_idx[probe_sel]], y[tr_idx[probe_sel]],
                            np.ones(len(probe_sel)), augment=False,
                            aux=aux_tr[probe_sel], soft=y_soft[tr_idx[probe_sel]],
                            wind=wind_t[tr_idx[probe_sel]],
                            tend_ok=tend_ok[tr_idx[probe_sel]],
                            ok5=ok5[tr_idx[probe_sel]])
    probe_dl = DataLoader(probe_ds, batch_size=64, shuffle=False, num_workers=0)

    if model_kind == "seq":
        net = CycloneSeqNet(dropout=cfg["dropout"], use_wind=cfg["wind_aux"] > 0,
                            n_slots=1 + len(TEND_LAGS_H)).to(device)
        print(f"  [v7] SEQ model: slots={1 + len(TEND_LAGS_H)} "
              f"frame_drop={SEQ_FRAME_DROP_P} params="
              f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")
    else:
        net = CycloneNet(arch=cfg["arch"], use_aux=cfg["aux"], dropout=cfg["dropout"],
                         use_wind=cfg["wind_aux"] > 0,
                         in_ch=(1 + n_lags) if use_tend_cfg else 1).to(device)
    ema = ModelEma(net, EMA_DECAY)

    n_pos = float((y[tr_idx] == 1).sum())
    n_neg = float((y[tr_idx] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=cfg["wd"])

    def lr_at(e):
        if e < SWEEP_WARMUP:
            return (e + 1) / max(1, SWEEP_WARMUP)
        t = (e - SWEEP_WARMUP) / max(1, n_epochs - SWEEP_WARMUP)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = make_scaler(use_amp)
    aug_gen = torch.Generator()
    use_meta = 1.0 if cfg["aux"] else 0.0

    best_auc, best_epoch, best_state = -1.0, -1, None
    hist = []
    for epoch in range(1, n_epochs + 1):
        aug_gen.manual_seed(SEED * 1000 + 37 * fold + epoch)
        net.train()
        tot_l, tot_w = 0.0, 0.0
        for imgs_u8, lab, w, aux_b, soft_b, wind_b, tok5 in train_dl:
            if model_kind == "seq":
                # (v7) SEQ path: each slot is normalised INDIVIDUALLY (no
                # differencing); the attention module learns what the history
                # means. Augmentation still happens in the [0,1] display domain,
                # and augment_batch shares one geometric/photometric draw and ONE
                # noise field across all slots of a sample, so a storm's history
                # is perturbed as a single scene, never as independent draws
                # (independent noise across slots would fabricate apparent
                # change between frames -- the same reasoning as v6's shared
                # noise across tendency channels).
                x01 = to_unit(imgs_u8.to(device, non_blocking=True), size=img_size)
                x01 = augment_batch(x01, aug_gen, cfg["aug"])
                x = normalize(x01)
                have = tok5.to(device)                      # (B, 1+n_lags)
                B = x.shape[0]
                keep = torch.ones(B, x.shape[1], device=device)
                dr_all = (torch.rand(B, generator=aug_gen).to(device)
                          < TEMPORAL_DROP_P)
                dr_slot = (torch.rand(B, x.shape[1] - 1, generator=aug_gen)
                           .to(device) < SEQ_FRAME_DROP_P)
                keep[:, 1:] = (~dr_slot).float()
                keep[dr_all, 1:] = 0.0    # whole-history drop -> K=1 in-distribution
                valid = have * keep
                a = aux_b.to(device)
            else:
                x01 = to_unit(imgs_u8.to(device, non_blocking=True), size=img_size)
                # Augment in the [0,1] display domain BEFORE differencing, so the zoom,
                # shift, rotation, brightness and cutout land identically on all three
                # frames and the difference stays a real tendency instead of an
                # augmentation artifact.
                x01 = augment_batch(x01, aug_gen, cfg["aug"])
                if use_tend_cfg and n_lags > 0:
                    # (v7) per-lag masks: channel j is zeroed when ITS OWN
                    # neighbour is missing; when TEMPORAL_DROP_P selects the
                    # sample ALL lag channels drop together (exact v6
                    # single-frame semantics). Partial histories keep their
                    # existing channels -- strict lookback everywhere, so still
                    # zero leakage.
                    have = tok5[:, 1:1 + n_lags].to(device)
                    keep = (torch.rand(x01.shape[0], generator=aug_gen).to(device)
                            >= TEMPORAL_DROP_P).float().unsqueeze(1)
                    if cfg_get(cfg, "mask") == "shared":
                        # v6 semantics: the lag channels exist only when EVERY
                        # neighbour exists (the exact published-TEND rule, kept
                        # for RN50T so the backbone is the ONLY change vs v6).
                        have = have.prod(dim=1, keepdim=True).expand(-1, n_lags)
                    x = tendency_stack_n(x01, n_lags, drop_lag=1.0 - have * keep)
                else:
                    x = normalize(x01[:, 0:1])
                a = aux_b.to(device)
            # aux dropout: zero the whole metadata branch for a random subset, so
            # the same weights work with and without metadata at inference.
            m = (torch.rand(x.shape[0], generator=aug_gen).to(device)
                 >= cfg.get("aux_drop", AUX_DROP_P)).float() * use_meta
            # frames whose metadata never parsed are always masked off
            m = m * (a.abs().sum(dim=1) > 0).float()
            wt = w.to(device).unsqueeze(1)

            if cfg["soft"]:
                t = soft_b.to(device).float().unsqueeze(1)
            else:
                t = lab.to(device).float().unsqueeze(1)
                t = t * (1.0 - 2.0 * LABEL_SMOOTHING) + LABEL_SMOOTHING

            opt.zero_grad(set_to_none=True)
            with amp_autocast(use_amp):
                if model_kind == "seq":
                    logit, wpred = net(x, valid)
                else:
                    logit, wpred = net(x, a, m)
                loss = (criterion(logit, t) * wt).sum() / wt.sum()
                if wpred is not None and cfg["wind_aux"] > 0:
                    wt_target = wind_b.to(device).float().unsqueeze(1)
                    loss = loss + cfg["wind_aux"] * F.mse_loss(wpred.float(), wt_target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(net)
            tot_l += float(loss.detach().item()) * float(wt.sum().item())
            tot_w += float(wt.sum().item())
        sched.step()

        live = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(ema.state_dict(net))
        vl, vy = predict_logits(net, val_dl, device, use_meta=use_meta,
                                mode=pred_mode, img_size=img_size)
        vauc = auc_of(vy, sigmoid_np(vl))
        pl, py = predict_logits(net, probe_dl, device, use_meta=use_meta,
                                mode=pred_mode, img_size=img_size)
        pauc = auc_of(py, sigmoid_np(pl))
        hist.append({"epoch": epoch, "val_auc": vauc, "probe_auc": pauc,
                     "train_loss": tot_l / max(1e-9, tot_w)})
        if vauc == vauc and vauc > best_auc:
            best_auc, best_epoch = vauc, epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(live)

    # ---- final (no epoch selection) ----
    net.load_state_dict(ema.state_dict(net))
    final_logits, _ = predict_logits(net, val_dl, device, use_meta=use_meta,
                                     mode=pred_mode, img_size=img_size)
    tta_logits, _ = predict_logits(net, val_dl, device, tta=True, use_meta=use_meta,
                                   mode=pred_mode, img_size=img_size)
    final_auc = auc_of(y[va_idx].astype(int), sigmoid_np(final_logits))
    fpl, fpy = predict_logits(net, probe_dl, device, use_meta=use_meta,
                              mode=pred_mode, img_size=img_size)
    final_probe = auc_of(fpy, sigmoid_np(fpl))

    # ---- best epoch (v3's selection rule, kept for comparability) ----
    net.load_state_dict(best_state)
    best_logits, _ = predict_logits(net, val_dl, device, use_meta=use_meta,
                                    mode=pred_mode, img_size=img_size)
    nometa_logits, _ = predict_logits(net, val_dl, device, use_meta=0.0,
                                      mode=pred_mode, img_size=img_size)
    # SINGLE-FRAME READOUT: the same trained weights scored with every temporal
    # input removed -- tendency channels zeroed (CNN) or history masked out
    # (SEQ) -- i.e. exactly what the shipped one-image apps would get. This is
    # the number that decides whether temporal input is a free win or a trade,
    # and it is measured rather than assumed. For a non-temporal config it is
    # the same forward pass, so the column stays comparable across the sweep.
    notend_logits, _ = predict_logits(net, val_dl, device, use_meta=use_meta,
                                      mode=pred_mode, readout="notend",
                                      img_size=img_size)

    # (v7) SEQ deployment diagnostics: did the gate open at all, and how much
    # attention mass does history get on frames that HAVE one?
    if model_kind == "seq":
        print(f"  [v7] SEQ gamma (temporal gate) = {float(net.gamma.item()):+.4f}")

    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "best": best_logits, "final": final_logits, "tta": tta_logits,
        "best_nometa": nometa_logits,
        "best_notend": notend_logits,
        "log": {"best_epoch": best_epoch, "best_auc": best_auc,
                "final_auc": final_auc, "final_probe_auc": final_probe,
                "n_train_frames": int(len(tr_idx)), "n_val_frames": int(len(va_idx)),
                "history": hist},
    }


def nested_platt(logits, y, fold):
    """Out-of-sample Platt calibration: each fold is calibrated by a fit made on
    the OTHER folds' out-of-fold logits. Fitting the calibration on the same
    predictions it is then scored on would understate ECE, which is the one number
    that says whether a displayed percentage can be believed."""
    p = np.full(len(y), np.nan)
    for f in sorted(set(fold.tolist())):
        tr, va = fold != f, fold == f
        if len(np.unique(y[tr])) < 2:
            p[va] = sigmoid_np(logits[va])
            continue
        a, b, _ = fit_platt_scaling(logits[tr], y[tr])
        p[va] = apply_calibration(logits[va], a, b)
    return np.nan_to_num(p, nan=0.5)


def summarise_config(name, cfg, oof, ev_y, ev_basin, ev_source, ev_fold,
                     st_pos, st_y, st_basin, st_n, fold_logs, ev_tend=None):
    def storm_mean(p):
        return np.bincount(st_pos, weights=p) / st_n

    probs = {k: sigmoid_np(v) for k, v in oof.items()}
    st = {k: storm_mean(v) for k, v in probs.items()}
    cal = nested_platt(oof["final"], ev_y, ev_fold)
    st_cal = storm_mean(cal)

    # ---- (v6) COVERED-SUBSET readout -------------------------------------
    # Only Digital Typhoon frames with a genuine t-6h AND t-12h neighbour can use
    # the tendency channels at all, and early-stage frames sit near the START of a
    # track, which is exactly where prior imagery is scarcest. If coverage is (say)
    # 40%, then a real +0.05 on the covered frames shows up as only ~+0.02 pooled
    # -- close enough to the noise floor to be called "no effect" when the effect
    # was actually there. So the same paired numbers are reported again restricted
    # to the covered frames (and to storms all of whose eval frames are covered).
    # Both configs are scored on the IDENTICAL subset, so it stays a fair paired
    # comparison and not a cherry-pick: the control is not allowed a different
    # denominator from the treatment.
    cov = {}
    if ev_tend is not None and ev_tend.any():
        cm = ev_tend.astype(bool)
        st_cov_n = np.bincount(st_pos, weights=cm.astype(float))
        st_all_cov = (st_cov_n >= st_n - 1e-9)   # every eval frame of the storm
        cov = {
            "n_frames": int(cm.sum()),
            "frac_frames": float(cm.mean()),
            "n_storms_fully_covered": int(st_all_cov.sum()),
            "frame_auc_best": auc_of(ev_y[cm], probs["best"][cm]),
            "frame_auc_notend": auc_of(ev_y[cm], probs["best_notend"][cm]),
            "storm_auc_best": (auc_of(st_y[st_all_cov], st["best"][st_all_cov])
                               if st_all_cov.sum() > 1 else float("nan")),
            "storm_auc_notend": (auc_of(st_y[st_all_cov], st["best_notend"][st_all_cov])
                                 if st_all_cov.sum() > 1 else float("nan")),
            "note": ("same frames for every config; 'notend' is these weights with "
                     "the tendency channels zeroed, i.e. the single-frame readout "
                     "on the very frames where tendency WAS available -- the "
                     "cleanest within-model measure of what the extra input buys."),
        }

    res = {
        "config": cfg,
        "covered_subset": cov,
        "frame_auc_best": auc_of(ev_y, probs["best"]),
        "frame_auc_final": auc_of(ev_y, probs["final"]),
        "frame_auc_tta": auc_of(ev_y, probs["tta"]),
        "frame_auc_nometa": auc_of(ev_y, probs["best_nometa"]),
        # (v6) the single-frame deployment readout -- same weights, tendency
        # channels zeroed. For a non-tendency config this equals frame_auc_best.
        "frame_auc_notend": auc_of(ev_y, probs["best_notend"]),
        "storm_auc_best": auc_of(st_y, st["best"]),
        "storm_auc_final": auc_of(st_y, st["final"]),
        "storm_auc_tta": auc_of(st_y, st["tta"]),
        "storm_auc_nometa": auc_of(st_y, st["best_nometa"]),
        "storm_auc_notend": auc_of(st_y, st["best_notend"]),
        "frame_ece_raw": expected_calibration_error(probs["final"], ev_y)[0],
        "frame_ece_calibrated": expected_calibration_error(cal, ev_y)[0],
        "frame_nll_calibrated": binary_nll(cal, ev_y),
        "mean_best_epoch": float(np.mean([l["best_epoch"] for l in fold_logs])),
        "mean_train_val_gap": float(np.mean(
            [l["final_probe_auc"] - l["final_auc"] for l in fold_logs])),
        "per_fold": [{k: v for k, v in l.items() if k != "history"}
                     for l in fold_logs],
        "frame_auc_by_basin": {b: auc_of(ev_y[ev_basin == b], cal[ev_basin == b])
                               for b in sorted(set(ev_basin.tolist()))},
        "frame_auc_by_source": {s: auc_of(ev_y[ev_source == s], cal[ev_source == s])
                                for s in sorted(set(ev_source.tolist()))},
        "storm_auc_by_basin": {b: auc_of(st_y[st_basin == b], st_cal[st_basin == b])
                               for b in sorted(set(st_basin.tolist()))},
        "storm_prob_final": st["final"].tolist(),
    }

    # Operating points. THIS is the false-alarm question: at the threshold the tool
    # would actually display, how many quiet systems get flagged?
    res["frame_operating_points"] = {}
    res["storm_operating_points"] = {}
    for target in SPECIFICITY_TARGETS:
        t = pick_threshold_for_specificity(cal, ev_y, target)
        if t is not None:
            res["frame_operating_points"][f"specificity_{target:.2f}"] = {
                "threshold": float(t), **evaluate(cal, ev_y, t)}
        t2 = pick_threshold_for_specificity(st_cal, st_y, target)
        if t2 is not None:
            res["storm_operating_points"][f"specificity_{target:.2f}"] = {
                "threshold": float(t2), **evaluate(st_cal, st_y, t2)}
    res["frame_at_0.50"] = evaluate(cal, ev_y, 0.50)
    res["frame_at_0.66"] = evaluate(cal, ev_y, 0.66)
    res["storm_at_0.50"] = evaluate(st_cal, st_y, 0.50)
    res["threshold_table"] = threshold_table(
        cal, ev_y, [round(t, 2) for t in np.arange(0.30, 0.91, 0.05)])
    return res


def print_config_summary(r, clim):
    print(f"\n  --- {r['config']['name']} out-of-fold results ---")
    print(f"  frame AUC : final {r['frame_auc_final']:.4f}  "
          f"best-epoch {r['frame_auc_best']:.4f}  +TTA {r['frame_auc_tta']:.4f}  "
          f"image-only {r['frame_auc_nometa']:.4f}   "
          f"(climatology-only {clim['frame_auc']:.4f})")
    print(f"  storm AUC : final {r['storm_auc_final']:.4f}  "
          f"best-epoch {r['storm_auc_best']:.4f}  +TTA {r['storm_auc_tta']:.4f}  "
          f"image-only {r['storm_auc_nometa']:.4f}   "
          f"single-frame {r['storm_auc_notend']:.4f}   "
          f"(climatology-only {clim['storm_auc']:.4f})")
    print(f"  overfit gap (train probe - val, final epoch): "
          f"{r['mean_train_val_gap']:+.3f}   ECE (out-of-sample Platt): "
          f"{r['frame_ece_calibrated']:.4f}")
    print("  per-basin frame AUC: " + "  ".join(
        f"{b} {v:.3f}" for b, v in r["frame_auc_by_basin"].items()))
    for k, v in r["frame_operating_points"].items():
        cm = v["confusion_matrix"]
        print(f"  {k}: thr {v['threshold']:.3f} -> FPR {v['false_positive_rate']:.3f} "
              f"({cm['fp']} false alarms of {cm['fp'] + cm['tn']} negatives), "
              f"precision {v['precision']:.3f}, recall {v['recall']:.3f}")


def paired_bootstrap_auc(base_p, cur_p, y, seed, reps=BOOT_REPS):
    """Bootstrap the DIFFERENCE in storm-level AUC, resampling storms.

    Paired: both configs are scored on the same resampled storms in every
    replicate, so the split-to-split variance that dominates the unpaired
    standard error cancels. This is what separates a real improvement from the
    kind of 0.01 wobble that produced the v3/v4/v4.1 table.
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    d0 = auc_of(y, cur_p) - auc_of(y, base_p)
    diffs = []
    for _ in range(reps):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        diffs.append(auc_of(yy, cur_p[idx]) - auc_of(yy, base_p[idx]))
    diffs = np.array(diffs)
    return {"delta": float(d0),
            "lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)),
            "p_better": float((diffs > 0).mean()),
            "n_reps": int(len(diffs))}


if __name__ == "__main__":
    main()
