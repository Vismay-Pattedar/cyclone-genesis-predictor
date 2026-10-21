"""
Cyclone Genesis Predictor -- Early-intensification prediction, v4.

TASK (unchanged from v1/v2/v3 -- this is a better-fed model for the same problem):
    Input : ONE early-stage infrared satellite frame (the system is still a weak,
            disorganised depression -- it does not yet look like a cyclone).
    Output: P(this storm will later reach typhoon / hurricane strength, >= 64 kt).
    Label : built from each storm's own FUTURE best-track history. The model never
            sees future frames, and never sees wind / pressure at inference time --
            only the raw brightness-temperature image.

The checkpoint written here is a drop-in replacement for v3's: same resnet18 with a
1-channel conv1, same required keys, and the SAME SIX BASIN NAMES in config['basins']
(see EXPECTED_BASINS and the hard check in main()), so local_tool/detect.py and
demo_kernel/render_demo.py keep working with NO edits.


WHY v4 EXISTS -- the measured v3 result this is responding to
-------------------------------------------------------------
v3 (24 epochs, 9,856 train / 2,449 val frames, 2,096 storms) produced:

    best val AUC 0.7061 at epoch 14; last-5-epoch AUC mean 0.6983 sd 0.0039
    calibration GENERALISED (5/5 folds): a=0.678 (T=1.48) b=-0.431,
        NLL 0.672 -> 0.625, ECE 0.119 -> 0.023
    epoch 24: train_loss 0.4303, train_probe_auc 0.919 vs val_auc 0.693

That last line is the whole story. v3 fixed v2's underfitting -- and overshot into
the opposite regime: a train-probe AUC of 0.919 against a val AUC of 0.693 is a
0.23 generalisation gap, i.e. the model is now MEMORISING 1,677 training storms.
More epochs or more capacity cannot help that. The only lever that can is MORE
DISTINCT STORMS, which is exactly what v4 adds.

v3's other two diagnostics both came back and both said "do not act":
  * (D1) field-of-view: dt/tcir cold-mass r50 ratio 0.928 (label 0) and 0.877
    (label 1) -- near 1.0, so the field-of-view hypothesis for the source gap is
    REFUTED and no centre-crop is applied here. v4 re-runs it with a much better
    controlled comparison; see D1 below.
  * (D2) wind convention: 56/1116 (5.0%) of Digital Typhoon storms sit in the
    [56.3, 64.0) kt band where the two agencies' bars physically disagree. Small,
    so the labels are STILL not changed -- only re-audited, now over three sources.


WHAT CHANGED, GROUPED
---------------------
E. THE THIRD SOURCE (this is what v4 is for)
   E1. Digital Typhoon AU (bwandowando/digital-typhoon-dataset-au-1979-2024) is
       added as a third mount: 480 Southern Hemisphere storms, identical schema and
       identical 512x512 float64 Kelvin h5 layout to the Western Pacific mirror.
       They are tagged basin="SH" (NOT a new basin -- see EXPECTED_BASINS) and
       source="digital_typhoon", with a finer sub_source of "digital_typhoon_au".
       This roughly TRIPLES the high-resolution Southern Hemisphere storm count and
       adds ~475 genuinely new storms to a model whose measured problem is that it
       has run out of distinct storms.
   E2. Root discovery is now keyword-directed and depth-bounded. v3's find_dt_root()
       returned the FIRST directory whose name contained "typhoon"; with two Digital
       Typhoon mounts attached that is a coin flip, so it had to be replaced. The
       replacement is the pruned search proven by au_overlap_probe/probe.py against
       this exact mount: never descend into a directory named "image" or "metadata"
       (those hold one subdirectory per storm), and confirm a root by requiring both
       of those entries to be present in its listing.
   E3. AU storm keys are NAMESPACED as "AU" + id. This is not cosmetic. A live probe
       found 459 of AU's 480 storm-id strings are also WP storm-id strings, and
       hashing the first frame of 80 such pairs found 0 identical / 80 different:
       AU runs its own per-year counter (BOM's Australian-region numbering), so the
       ~96% ID coincidence is two independent counters colliding, not duplication.
       Un-namespaced, pandas would silently MERGE ~459 pairs of physically different
       storms into one split unit each -- and v3's collision assert would NOT have
       caught it, because it only compared `source`, which is "digital_typhoon" on
       both sides. v4 asserts on `sub_source` instead.
   E4. NO ID-KEY DEDUP IS RUN BETWEEN DT-AU AND DT-WP, for the reason in E3: the ID
       scheme carries zero evidence across those two mirrors. What IS run is the
       physically-grounded best-track position match, as a report-only NULL TEST
       (au_vs_wp_track_null_test). Expected result: zero. The two mounts are
       opposite hemispheres, so a match needs a storm within 1.5 deg of another
       storm's latitude across the equator, which cannot happen. If it ever prints a
       nonzero number, that is a real finding and the run says so loudly.
   E5. dedup_wpac() is generalised into dedup_tcir_against_dt(), parameterised by DT
       source and TCIR region, and called TWICE: (WP, "WPAC") -- which must and does
       reproduce v3's numbers exactly -- and (AU, "SH"), which is the genuine new
       overlap (TCIR's 349 SH-basin storms cover the same real Southern Hemisphere
       storms AU does). For the SH pair the ID-key detector is COMPUTED AND PRINTED
       but NOT unioned into the drop set: TCIR's SH numbering is JTWC's and AU's is
       BOM's, two independent counters, so an ID collision there means exactly what
       it meant in E3 -- nothing. Position matching is authoritative for that pair.
       DT wins every collision, in both pairs (512x512 vs 128x128).
   E6. Radius-field (long30/short30) coverage is audited per sub-source. It plays no
       part in training -- it is only used by the demo's box rendering -- but an
       earlier casual note claimed "near-zero" coverage on AU, so the run now
       measures it instead of anyone assuming either way.

A. Fit the model properly (carried over from v3, one dial moved)
   A1. EPOCHS 24, 3-epoch linear warmup then cosine decay.  UNCHANGED from v3.
   A2. FRAMES_PER_STORM 6 -> 10.  <-- see RUNTIME BUDGET, and the honesty note
       there about what this does and does not buy.
   A3. Exponential moving average of the weights.  UNCHANGED from v3.
   A4. Per-epoch val loss + fixed train-subset AUC probe.  UNCHANGED from v3.

B. Close the cross-source gap / regularise           ALL UNCHANGED FROM v3
   B1. Batched GPU augmentation. Still NO flips -- and now with ~475 real Southern
       Hemisphere storms at full resolution, the reason is no longer hypothetical:
       a mirrored SH storm is a physically impossible NH storm. See FrameDataset.
   B2. Label smoothing (eps=0.03).

C. Make the displayed probability mean what it says   ALL UNCHANGED FROM v3
   C1. Platt/temperature calibration folded into fc, gated on a grouped-by-storm
       5-fold cross-check, and self-verified live after folding.
   C2/C3. Cross-check + operating-point table + recommended display thresholds.

D. Diagnostics
   D1. Radial-scale measurement, now run TWICE and with a real control. v3 could
       only compare digital_typhoon (WPAC only) against tcir (all basins), so a
       source difference and a basin difference were confounded. v4 has high-res
       Digital Typhoon frames in the SH basin, so it also reports the SH-ONLY
       dt_au-vs-tcir ratio -- same basin, same storms' region, different dataset.
       That is the comparison v3's number wanted to be.
   D2. Wind-averaging-convention audit, now per sub-source (report only -- labels
       are NOT changed).  See label_note() for the AU-specific caveat.
   D3. Basin-balanced loss weighting, still implemented and still OFF by default.

Explicitly NOT changed: the two dedup DETECTORS themselves and their thresholds
(only their plumbing was generalised; the WP/WPAC call is a bit-for-bit regression
of v3 and is tested as one), the split-by-storm discipline, the (basin, label)
stratification, SEED, the 190-300 K window, TEMP_LO/TEMP_HI/IMG_SIZE/NORM_MEAN/
NORM_STD (baked into local_tool/detect.py and every existing checkpoint), the
resnet18_1ch architecture and checkpoint key set, EPOCHS/WARMUP_EPOCHS/BATCH_SIZE/
LR/WEIGHT_DECAY/EMA_DECAY/LABEL_SMOOTHING, every augmentation strength, the
no-flips rule, the calibration procedure and its gate and its self-verification,
and the json.dumps() pre-flight on config and val_metrics.


RUNTIME BUDGET (measured from v3's Kaggle log, extrapolated)
-------------------------------------------------------------
MEASURED on v3's T4 run (kernel cyclone-genesis-predictor-intensification-train-v3, 1,136 s wall):
    Digital Typhoon preload   17.3 ms/frame   (6,598 frames in 114 s)
    TCIR preload               3.2 ms/frame   (5,707 frames in  18 s)
    train+val+probe epoch      2.73 ms/frame  (14,305 frames in 39 s)
    everything else            ~70 s          (indexing, dedup, D1, calibration,
                                               TTA probe, checkpoint save)
The 2.73 ms/frame blended epoch cost is a real measurement, not a fit: v2 measured
2.66 ms/frame on the same code path with a different frame count, so the linear
model is confirmed at two points 1.7x apart.

v4 FRAME COUNT at FRAMES_PER_STORM=10. v3 hit 5.95 of its cap of 6 on Digital
Typhoon and 5.78 of 6 on TCIR, i.e. nearly every storm has more early frames than
the cap, so a cap of 10 will land near 10 too (the three storm CSVs checked offline
had 15, 12 and 25 grade<=2 rows). Storms after dedup:
    DT-WP  1,109 (v3 measured)   DT-AU ~475 (480 mounted, ~79/80 have an early frame)
    TCIR   ~870  (v3's 987 minus an ESTIMATED ~120 SH storms newly dropped against
                  AU; this number is a guess and the run prints the real one)
    => ~22,300 frames expected; 24,500 is the hard ceiling (every storm at the cap).

    indexing + dedup + reports  ~2.5 min  (1,596 CSVs vs 1,116; D1 now runs twice)
    preload 14,700 DT frames    ~4.2 min
    preload  7,600 TCIR frames  ~0.4 min
    24 epochs at ~66 s          ~26.6 min
    ------------------------------------------------
    TOTAL                       ~34 min of GPU time (v3 measured ~19 min).
    HARD CEILING if every storm hits the cap: ~37 min. Both are inside the ~45-50
    min soft budget, with the ceiling -- not the estimate -- inside it.

RAM: 24,500 x 224 x 224 uint8 = 1.23 GB for the preloaded array, and the train/val/
probe views are boolean-mask COPIES, so peak is ~3 GB. v3 peaked at ~1.6 GB on the
same code and Kaggle's T4 image has far more than that.

Dials, all single constants, cost very close to linear in each:
    FRAMES_PER_STORM = 6   -> about 23 min total  (recovers v3's frames-per-storm;
                                                   still gets all the new storms)
    EPOCHS = 16            -> about 24 min total
    both of the above      -> about 17 min total
(Only the epoch term scales with EPOCHS; the ~4.5 min of preload and the ~2.5 min
of indexing/dedup/diagnostics are fixed costs either way.)

HONEST NOTE ON WHAT A2 BUYS. Frames of one storm are 3-6 hours apart, look nearly
identical, and share a single label that is a property of the STORM, not the frame.
They are correlated samples; the effective sample size is bounded by the storm
count, not the frame count. Going 6 -> 10 therefore does NOT multiply the
information by 1.67 -- it mostly buys the model more views of the same storms to be
invariant across, which is worth something to an over-parameterised net but is not
the point of this version. The real information gain in v4 is the ~475 new AU
storms (a 23% rise in distinct storms, and a ~2.4x rise in high-resolution Southern
Hemisphere storms). If the runtime ever needs to come down, cut FRAMES_PER_STORM
first and the new source last.

Runs as a plain Kaggle *script* kernel with GPU enabled.
An offline dry run against synthetic fixtures lives in dryrun_test.py next to this
file; it drives this same script through CGP_INPUT_ROOT / CGP_OUT_DIR.
"""

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

# (A2, v4.1 REVERT) v4's first run (FRAMES_PER_STORM=10) came back with overall AUC
# flat-to-down vs v3 (0.696 vs 0.706), and SH -- the basin AU exists to help -- WORSE
# (0.602 vs v3's 0.641), despite 464 real new storms passing every dedup/null-test
# check clean. The per-epoch history explains why: final-epoch generalisation gap
# (train_probe_auc - val_auc) was 0.264, WORSE than v3's 0.226, and the EMA-selected
# best epoch landed at 12/24 with a steep decline after (last-5-epoch mean 0.667 vs
# the 0.696 peak -- a much sharper falloff than v3 showed). Hypothesis: MORE views of
# the SAME storm (10 vs v3's 6) partly cancels the benefit of MORE distinct storms,
# because repeated near-duplicate exposure makes each individual storm faster to
# memorise. This run isolates that one variable -- FRAMES_PER_STORM back to v3's 6,
# nothing else touched -- to test the hypothesis directly against the same 2,431-storm
# v4 dataset. If AUC recovers, the hypothesis holds. If it doesn't, the SH/AU weakness
# is more likely a genuine property of that data (harder physical regime, or older
# AU-era satellite quality) rather than a training-schedule artifact.
FRAMES_PER_STORM = 6

# (E1/E2) The Digital Typhoon mounts, in priority order. `keywords` are matched
# case-insensitively against the FULL candidate path by find_dt_root(); the first
# keyword that resolves wins, and the extra aliases exist only so a renamed Kaggle
# mount does not silently take the whole source out of the run.
#
# `basin` is the basin taxonomy label these storms are filed under, and it is
# deliberately one of EXPECTED_BASINS -- AU is Southern Hemisphere data, so it is
# "SH", the same label TCIR's SH-basin storms already carry. Adding a 7th basin
# would change config['basins'] and therefore the "N basins" string that
# local_tool/detect.py prints on screen; there is no reason to do that here.
#
# `key_prefix` namespaces storm_key. MUST be non-empty and alphabetic for any mount
# whose ids can collide with another mount's -- see E3 in the docstring; AU and WP
# share 459 of AU's 480 id strings while being different storms.
DT_SOURCES = (
    {
        "tag": "wp",
        "sub_source": "digital_typhoon_wp",
        "keywords": ("western-pacific", "typhoon-dataset-wp", "-wp"),
        "basin": "WPAC",
        "key_prefix": "",
        "required": True,
        "desc": "bwandowando/digital-typhoon-dataset-western-pacific-wp "
                "(512x512 IR, WPAC, JMA best track)",
    },
    {
        "tag": "au",
        "sub_source": "digital_typhoon_au",
        "keywords": ("dataset-au", "typhoon-au", "-au-"),
        "basin": "SH",
        "key_prefix": "AU",
        "required": True,
        "desc": "bwandowando/digital-typhoon-dataset-au-1979-2024 "
                "(512x512 IR, Australian region / Southern Hemisphere)",
    },
)

# (E2) Depth bound for the pruned root search. Real Kaggle mounts sit 2-4 levels
# under /kaggle/input; 6 is generous. Proven against both mounts by
# au_overlap_probe/probe.py.
DT_ROOT_MAX_DEPTH = 6

# (E1) The basin taxonomy, frozen. local_tool/detect.py reads config['basins'] at
# runtime and renders "trained on Digital Typhoon + TCIR, N basins", so this set is
# part of the deliverable's user-visible contract. main() asserts the final index
# produces exactly this set and refuses to save otherwise -- a missing basin would
# mean a source silently failed to load, and an extra one would mean detect.py's
# label is now wrong.
EXPECTED_BASINS = ("ATLN", "CPAC", "EPAC", "IO", "SH", "WPAC")

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

# ---- dedup (detectors verbatim from v2/v3 -- correctness-critical, do not tune) --
DEDUP_MATCH_DEG = 1.5
DEDUP_MIN_MATCH_POINTS = 4
TCIR_ID_PATTERN_MIN_MATCH = 0.90
TCIR_ID_RE = re.compile(r"^(\d{4})(\d{2})([A-Za-z])$")
TCIR_MIN_FINITE_FRAC = 0.50

# (E5) Per-pair choice of whether the ID-key detector is UNIONED into the drop set
# or merely computed and printed. The position/track detector always drops.
#
# WP/WPAC = True: this is v3's DEDUP_USE_ID_KEY_UNION verbatim and it must stay
#   True to reproduce v3's measured result (358 by id, 347 by track, 398 dropped of
#   416). Both agencies number the Western Pacific from the same real season, so an
#   ID collision there is weak-but-real evidence, and Digital Typhoon covers 1,109
#   WPAC storms against TCIR's 416 -- dropping an extra TCIR WPAC storm costs
#   almost nothing while a missed duplicate silently inflates val AUC.
#
# AU/SH = False, and this is the single most important judgement call in v4. A live
#   probe hashed the first frame of 80 AU/WP storm pairs that share an id string and
#   found 0 identical: AU numbers storms on BOM's own per-year counter, unrelated to
#   any other agency's. TCIR's SH ids come from JTWC's counter. Two independent
#   per-year counters colliding carries NO evidence about whether two storms are the
#   same, exactly as it carried none between AU and WP. Unioning it here would drop
#   most of TCIR's 349 SH storms on the strength of a coincidence, in the one basin
#   where v4 is trying to ADD data. The ID-key number is still computed and printed
#   so its (expected: poor) agreement with the track detector is visible rather than
#   asserted.
DEDUP_ID_KEY_UNION = {"WPAC": True, "SH": False}

# (E4) AU-vs-WP is a NULL TEST, not a dedup. Report only by default. The two mounts
# are opposite hemispheres, so a position match would need two storms within 1.5
# degrees of latitude of each other across the equator; tropical cyclones do not
# form within ~5 degrees of it, so the physically correct answer is 0. Set True only
# if the run ever prints a nonzero count and it is confirmed to be real duplication.
DEDUP_AU_VS_WP_DROP = False
# If the null test finds more than this fraction of AU storms matching a WP storm,
# something is wrong with an assumption and the run says so in the loudest terms it
# has, rather than quietly training on leaked data.
DEDUP_AU_VS_WP_ALERT_FRAC = 0.02

# ---- training ---------------------------------------------------------------
VAL_FRACTION = 0.2
EPOCHS = 24               # (A1) v2 used 8 and was still improving at epoch 8.
WARMUP_EPOCHS = 3         # (A1) linear warmup, then cosine over the remainder.
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4

# (A3) EMA decay. UNCHANGED from v3. v3 ran ~308 optimiser steps/epoch; v4's larger
# frame count puts it near ~475, so 0.999 now averages over roughly the last 1,000
# steps == ~2.1 epochs rather than v3's ~3.2. Still long enough to kill the
# epoch-to-epoch val-AUC noise it was introduced for (v3's last-5-epoch AUC sd was
# 0.0039, down from v2's ~0.02 swing), still short enough to track the end of
# training. Left alone deliberately: v3's selection behaviour was measured and good,
# and re-tuning it at the same time as changing the dataset would make the two
# effects impossible to separate.
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

# (C) Calibration. UNCHANGED from v3, where it fitted a=0.678 (T=1.48) b=-0.431 and
# took ECE from 0.119 to 0.023 with 5/5 cross-check folds improving. Fitting 2
# parameters on ~4,500 val frames (v3 had ~2,450) is statistically trivial, but the
# val set is also what selects the epoch, so the fit is verified by a storm-level
# grouped cross-check before it is trusted. See
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

CKPT_PATH = os.path.join(OUT_DIR, "model_best.pt")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")
INDEX_PATH = os.path.join(OUT_DIR, "dataset_index.csv")


def label_note():
    return (
        "Positive == max sustained wind >= 64 kt at some point in the storm's life. "
        "CAVEAT 1: Digital Typhoon WP's `wind` is JMA 10-minute sustained; TCIR's "
        "`Vmax` is agency 1-minute sustained (JTWC/NHC). 64 kt is each agency's OWN "
        "typhoon/hurricane threshold, so the label reads as 'reached typhoon or "
        "hurricane strength as its own agency classified it' -- consistent as a "
        "category, slightly different as a physical wind speed (a 10-min 64 kt storm "
        "is roughly a 1-min 72 kt storm). This is the only rule expressible on all "
        "sources; TCIR has no grade field. "
        "CAVEAT 2 (v4, ASSUMPTION -- NOT INDEPENDENTLY VERIFIED): Digital Typhoon "
        "AU's `wind` is TREATED as the same 10-minute-sustained convention as the WP "
        "mirror, because it is the same Digital Typhoon project, the same schema and "
        "the same documented methodology (Kitamoto et al., arXiv:2411.16421). It has "
        "NOT been checked against the Bureau of Meteorology's own best-track "
        "methodology documentation, and BOM is a 10-minute agency by WMO convention, "
        "which is consistent with the assumption but is not the same thing as having "
        "confirmed it. If it turned out to be 1-minute, AU's positives would be "
        "slightly over-called relative to WP's -- the same direction and roughly the "
        "same size as the TCIR mismatch already documented above. "
        "v4 measures the size of the resulting disagreement per sub-source (see "
        "dt_audit.*.n_label_flips_under_1min_equiv_bar) but, as in v3, deliberately "
        "does NOT change the rule."
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


# ----------------------------------------------------------------------------
# Dataset discovery
# ----------------------------------------------------------------------------
def find_dt_root(keyword, max_depth=DT_ROOT_MAX_DEPTH):
    """(E2) Locate ONE Digital Typhoon mount, by keyword, safely.

    Replaces v3's find_dt_root(), which returned the first directory whose NAME
    contained "typhoon". That was fine with one mount and is a coin flip with two,
    so it had to go.

    Two properties matter and both are load-bearing:

      * PRUNED. Never descends into a directory named "image" or "metadata" while
        searching. Those hold one subdirectory per storm (hundreds to thousands),
        each with dozens of .h5 files; a blind os.walk through them costs minutes of
        quota and returns nothing useful. They are checked with a cheap os.listdir
        on the candidate itself instead.
      * CONFIRMED, not guessed. A candidate only counts as a root if `keyword`
        appears in its full path AND its own listing contains both "metadata" and
        "image". A path match alone would happily return the parent of the real
        root, or a stray directory.

    This is au_overlap_probe/probe.py's find_dataset_root(), which was run against
    both live Kaggle mounts and returned the correct root for each -- with ONE fix
    found by dryrun_test.py. The probe seeded its recursion with the ABSOLUTE path
    depth (`INPUT_ROOT.count(os.sep)`) and compared that against max_depth, so the
    budget silently depended on how deep INPUT_ROOT itself sat: it happened to leave
    4 usable levels under "/kaggle/input" and ZERO under the harness's temp
    directory, where the search returned None for every keyword. Depth is counted
    RELATIVE to INPUT_ROOT here, which is what "real mounts are 2-4 levels deep"
    always meant. On Kaggle this is strictly more permissive than the probe was, so
    it still resolves both mounts.
    """
    kw = keyword.lower()

    def walk(path, depth):
        if depth > max_depth:
            return None
        try:
            entries = os.listdir(path)
        except (PermissionError, FileNotFoundError, NotADirectoryError, OSError):
            return None
        if kw in path.lower() and "metadata" in entries and "image" in entries:
            return path
        for e in sorted(entries):
            full = os.path.join(path, e)
            if e.lower() in ("image", "metadata"):
                continue  # never walk the per-storm subtree
            if not os.path.isdir(full):
                continue
            found = walk(full, depth + 1)
            if found:
                return found
        return None

    return walk(INPUT_ROOT, 0)


def discover_dt_sources():
    """Resolve every mount in DT_SOURCES to a concrete (meta_dir, image_dir).

    Returns a list of dicts, each the DT_SOURCES entry plus 'root'/'meta_dir'/
    'image_dir'. Raises SystemExit if a required mount is missing, because the
    alternative -- quietly training a "v4" that is actually v3 plus a bigger frame
    cap -- is the kind of failure nobody notices until the numbers are being
    compared months later.
    """
    resolved = []
    for spec in DT_SOURCES:
        root = None
        hit_kw = None
        for kw in spec["keywords"]:
            root = find_dt_root(kw)
            if root:
                hit_kw = kw
                break
        if root is None:
            msg = (f"Digital Typhoon '{spec['tag']}' mount not found under "
                   f"{INPUT_ROOT}. Tried keywords {list(spec['keywords'])}. "
                   f"Expected dataset: {spec['desc']}")
            if spec.get("required", True):
                raise SystemExit(
                    "\n" + "!" * 72 + "\n" + msg + "\n"
                    "Attach it in kernel-metadata.json -> dataset_sources and re-run. "
                    "Refusing to continue: a silently missing source would produce a "
                    "checkpoint that looks like v4 and is not.\n" + "!" * 72)
            print(f"  NOTE (optional source skipped): {msg}")
            continue
        meta, img = resolve_dt_dirs(root)
        if not os.path.isdir(meta) or not os.path.isdir(img):
            raise SystemExit(
                f"Digital Typhoon '{spec['tag']}' root {root} resolved to "
                f"meta={meta} image={img}, at least one of which is not a directory.")
        entry = dict(spec)
        entry.update({"root": root, "meta_dir": meta, "image_dir": img,
                      "matched_keyword": hit_kw})
        resolved.append(entry)
        print(f"DT[{spec['tag']}] root      = {root}   (matched '{hit_kw}')")
        print(f"DT[{spec['tag']}] meta_dir  = {meta}")
        print(f"DT[{spec['tag']}] image_dir = {img}")

    tags = [e["tag"] for e in resolved]
    if len(set(tags)) != len(tags):
        raise SystemExit(f"Duplicate Digital Typhoon source tags resolved: {tags}")
    roots = [e["root"] for e in resolved]
    if len(set(roots)) != len(roots):
        # Two keyword sets landing on the SAME directory would index the same storms
        # twice under two different key prefixes -- guaranteed near-duplicate leakage
        # across the train/val boundary. Loud, not silent.
        raise SystemExit(
            f"Two Digital Typhoon sources resolved to the same root: {roots}. "
            f"Fix the keywords in DT_SOURCES; indexing one mount twice would put "
            f"identical frames on both sides of the train/val split.")
    return resolved


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
def build_dt_index(meta_dir, image_dir, basin, sub_source, key_prefix="", tag="A"):
    """Index Digital Typhoon early-stage frames from ONE mount.

    v3's version was hardwired to basin="WPAC" and to a bare storm id as the key.
    v4 parameterises exactly three things and nothing else:

      basin       -- taxonomy label ("WPAC" for the WP mirror, "SH" for AU). Must be
                     one of EXPECTED_BASINS.
      sub_source  -- fine-grained provenance ("digital_typhoon_wp"/"..._au"). The
                     coarse `source` stays "digital_typhoon" for BOTH, because they
                     really are the same processing lineage and the same 512x512
                     float64 Kelvin layout -- and because config['sources'] is part
                     of what the existing deliverables read.
      key_prefix  -- namespace for storm_key. "" for WP (so WP keys stay the bare
                     6-digit ids that demo_kernel/render_demo.py looks up), "AU" for
                     AU. NOT optional for AU: 459 of AU's 480 ids are also WP ids
                     while being different storms (docstring E3).

    Returns (rows_df, tracks, storm_tracks, audit):
      tracks       -- {YYYYMMDDHH: [(lat, lng, storm_key), ...]} over FULL tracks,
                      the reference side of the position detector.
      storm_tracks -- {storm_key: [(YYYYMMDDHH, lat, lng), ...]}, the query side.
                      New in v4: the AU-vs-WP null test needs to run the detector
                      with a Digital Typhoon mount on BOTH sides.
    """
    if basin not in EXPECTED_BASINS:
        raise SystemExit(f"build_dt_index called with basin={basin!r}, which is not "
                         f"in EXPECTED_BASINS={list(EXPECTED_BASINS)}. Adding a basin "
                         f"would change config['basins'] and detect.py's on-screen "
                         f"label; do that deliberately or not at all.")
    if key_prefix and not key_prefix.isalpha():
        raise SystemExit(f"key_prefix {key_prefix!r} must be alphabetic: dt_dedup_key() "
                         f"strips a leading alphabetic run to recover the raw id.")

    csvs = sorted(
        f for f in os.listdir(meta_dir) if f.endswith(".csv") and f[:-4].isdigit()
    )
    print(f"[{tag}] Found {len(csvs)} per-storm metadata CSVs "
          f"({sub_source}, basin={basin}, key_prefix={key_prefix or '<none>'})")

    rows = []
    tracks = defaultdict(list)
    storm_tracks = {}
    audit = {
        "sub_source": sub_source,
        "basin": basin,
        "key_prefix": key_prefix,
        "meta_dir": meta_dir,
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
        # (D2) how many storms the 1-min-equivalent bar would flip, report only
        "n_label_flips_under_1min_equiv_bar": 0,
        "n_storms_labelled": 0,
        # (E6) radius-field coverage, report only -- long30/short30 are used by the
        # demo's wind-radius box rendering, never by this classifier.
        "n_rows_scanned": 0,
        "n_rows_long30_present": 0,
        "n_rows_short30_present": 0,
        "n_storms_with_any_radius": 0,
        "n_storms_radius_columns_absent": 0,
        # (E1) sanity: which hemisphere did this mount actually turn out to be?
        "lat_min": None, "lat_max": None, "lng_min": None, "lng_max": None,
        "n_storms_mostly_southern": 0,
    }
    lat_lo = lat_hi = lng_lo = lng_hi = None

    # (D2) A 64 kt 1-minute bar (TCIR's) expressed in Digital Typhoon's 10-minute
    # units. Storms between this and 64 kt are the ones the two sources currently
    # disagree about.
    one_min_equiv_bar = TYPHOON_WIND_KT * WIND_10MIN_PER_1MIN

    for i, name in enumerate(csvs):
        storm_id = name[:-4]
        storm_key = key_prefix + storm_id
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

        # (E6) Radius-field coverage, counted before any drop so the number describes
        # the MOUNT rather than the training subset. Report only.
        if "long30" in df.columns and "short30" in df.columns:
            n_l30 = int(pd.to_numeric(df["long30"], errors="coerce").notna().sum())
            n_s30 = int(pd.to_numeric(df["short30"], errors="coerce").notna().sum())
            audit["n_rows_scanned"] += len(df)
            audit["n_rows_long30_present"] += n_l30
            audit["n_rows_short30_present"] += n_s30
            if n_l30 or n_s30:
                audit["n_storms_with_any_radius"] += 1
        else:
            audit["n_storms_radius_columns_absent"] += 1

        year = int(storm_id[:4])
        # JMA best-track maximum sustained wind only exists from 1977 onwards. For
        # an earlier storm every `wind` cell is 0, which a >=64kt rule would read as
        # "never reached typhoon" -- a systematic false-negative across whole
        # decades. Drop rather than mislabel.
        #
        # (v4) The AU mount starts in 1979 so this filter is inert there, and it is
        # left applying to both mounts unchanged rather than being made
        # source-specific: the guard below (wmax <= 0 -> n_no_wind_data) is what
        # actually catches an era with absent winds, and it catches it for ANY mount
        # without needing a per-agency cutoff nobody has verified. Whatever the AU
        # numbers turn out to be, they are printed in this function's audit.
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
            la_arr = lat[good].to_numpy()
            lo_arr = lng[good].to_numpy()
            pts = []
            for ts, la, lo in zip(stamps.to_numpy(), la_arr, lo_arr):
                tracks[int(ts)].append((float(la), float(lo), storm_key))
                pts.append((int(ts), float(la), float(lo)))
            storm_tracks[storm_key] = pts
            # Hemisphere/extent sanity, so the run PROVES which mount it read rather
            # than trusting the directory name. AU must come out negative-latitude.
            lo_lat, hi_lat = float(la_arr.min()), float(la_arr.max())
            lo_lng, hi_lng = float(lo_arr.min()), float(lo_arr.max())
            lat_lo = lo_lat if lat_lo is None else min(lat_lo, lo_lat)
            lat_hi = hi_lat if lat_hi is None else max(lat_hi, hi_lat)
            lng_lo = lo_lng if lng_lo is None else min(lng_lo, lo_lng)
            lng_hi = hi_lng if lng_hi is None else max(lng_hi, hi_lng)
            if float(np.median(la_arr)) < 0.0:
                audit["n_storms_mostly_southern"] += 1

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
                    "storm_key": storm_key,
                    "source": "digital_typhoon",
                    "sub_source": sub_source,
                    "basin": basin,
                    "label": label,
                    "h5_path": h5_path,
                    "frame_ref": -1,
                    "max_wind_kt": wmax,
                }
            )

        if (i + 1) % 250 == 0:
            print(f"  [{tag}] indexed {i + 1}/{len(csvs)} storms, {len(rows)} frames so far")

    audit["lat_min"], audit["lat_max"] = lat_lo, lat_hi
    audit["lng_min"], audit["lng_max"] = lng_lo, lng_hi

    print(f"[{tag}] storms dropped -- unreadable={audit['n_unreadable']} "
          f"bad_columns={audit['n_bad_columns']} pre_1977={audit['n_pre_1977_dropped']} "
          f"no_wind_data={audit['n_no_wind_data']} no_early_frame={audit['n_no_early_frame']}")
    print(f"[{tag}] frames listed in metadata but absent on disk: {audit['n_missing_h5']}")
    print(f"[{tag}] label rule cross-check vs v1 (grade==5): "
          f"agree={audit['n_label_agrees_with_v1_grade_rule']} "
          f"disagree={audit['n_label_disagrees_with_v1_grade_rule']}")
    for ex in audit["label_disagreement_examples"]:
        print(f"      disagreement: {ex}")

    # (E1) Hemisphere check. Not an assertion -- the Australian region's northern
    # edge reaches the equator and a storm or two could sit just above it -- but a
    # mount tagged basin="SH" whose storms are overwhelmingly northern would mean
    # the keyword search resolved the wrong directory, which is worth shouting about.
    n_tracked = max(1, len(storm_tracks))
    south_frac = audit["n_storms_mostly_southern"] / n_tracked
    print(f"[{tag}] track extent: lat [{lat_lo}, {lat_hi}]  lng [{lng_lo}, {lng_hi}]  "
          f"-- {audit['n_storms_mostly_southern']}/{len(storm_tracks)} storms "
          f"({south_frac:.1%}) have a southern median latitude")
    if basin == "SH" and south_frac < 0.5:
        print(f"  !! WARNING: {sub_source} is tagged basin='SH' but only {south_frac:.1%} "
              f"of its storms are in the Southern Hemisphere. Check that "
              f"find_dt_root() resolved the intended mount.")
    if basin == "WPAC" and south_frac > 0.5:
        print(f"  !! WARNING: {sub_source} is tagged basin='WPAC' but {south_frac:.1%} "
              f"of its storms are in the Southern Hemisphere. Check that "
              f"find_dt_root() resolved the intended mount.")

    # (E6) Radius-field coverage. Report only -- long30/short30 are demo rendering
    # inputs, never classifier inputs, so whatever this says it does not block
    # training. It exists because an earlier casual look claimed near-zero coverage
    # on one storm while a different storm came back 100% covered.
    n_rows = max(1, audit["n_rows_scanned"])
    print(f"[{tag}] RADIUS-FIELD COVERAGE (report only, not used for training): "
          f"long30 present on {audit['n_rows_long30_present']}/{audit['n_rows_scanned']} "
          f"rows ({audit['n_rows_long30_present'] / n_rows:.1%}), short30 on "
          f"{audit['n_rows_short30_present'] / n_rows:.1%}; "
          f"{audit['n_storms_with_any_radius']} storms have at least one radius value, "
          f"{audit['n_storms_radius_columns_absent']} storms lack the columns entirely.")

    # (D2) Report only. See the block in main() that interprets this number.
    n_lab = max(1, audit["n_storms_labelled"])
    print(f"[{tag}] WIND-CONVENTION AUDIT (report only, labels unchanged): "
          f"{audit['n_label_flips_under_1min_equiv_bar']}/{audit['n_storms_labelled']} "
          f"({audit['n_label_flips_under_1min_equiv_bar'] / n_lab:.1%}) {sub_source} "
          f"storms have peak 10-min wind in [{one_min_equiv_bar:.1f}, {TYPHOON_WIND_KT:.1f}) kt, "
          f"i.e. they are NEGATIVE under the current rule but would be POSITIVE if "
          f"the bar were harmonised to TCIR's 1-minute convention.")
    return pd.DataFrame(rows), tracks, storm_tracks, audit


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

    # (E5, v4) The SH region is now a dedup pair too (against Digital Typhoon AU).
    # This check is deliberately SOFT where the WPAC one above is hard, and the
    # asymmetry is the point: for WPAC the ID key is UNIONED into the drop set, so a
    # mis-parsed id there causes real leakage and must abort the run. For SH the ID
    # key is computed for the audit and NEVER dropped on (DEDUP_ID_KEY_UNION["SH"]
    # is False), so a mis-parse there degrades one printed diagnostic and nothing
    # else. Aborting a 30-minute GPU run over a cosmetic column would be the wrong
    # trade.
    sh_ids = sorted(info.loc[info["region"].astype(str) == "SH", "ID"].astype(str).unique())
    if not sh_ids:
        print("\n  NOTE: no region=='SH' rows in TCIR. The Digital Typhoon AU vs "
              "TCIR-SH dedup will have nothing to do; AU storms are still added.")
    else:
        sh_matched = [i for i in sh_ids if TCIR_ID_RE.match(i)]
        sh_frac = len(sh_matched) / len(sh_ids)
        print(f"  SH  : {len(sh_matched)}/{len(sh_ids)} IDs ({sh_frac:.1%}) match "
              f"^YYYY NN <letter>$  (used for the AU dedup AUDIT column only, "
              f"never to drop a storm -- see DEDUP_ID_KEY_UNION)")
        if sh_frac < TCIR_ID_PATTERN_MIN_MATCH:
            print(f"  NOTE: SH id parsing is below {TCIR_ID_PATTERN_MIN_MATCH:.0%}. "
                  f"The 'overlap_by_id_key' number in the SH dedup audit is therefore "
                  f"unreliable and should be read as a lower bound. Nothing else is "
                  f"affected: the SH drop set comes from position matching only. "
                  f"Non-matching examples: {[i for i in sh_ids if not TCIR_ID_RE.match(i)][:10]}")
    print("-" * 72 + "\n")
    return True


def tcir_dedup_key(storm_id):
    """'200315W' -> (2003, 15). None if unparseable."""
    m = TCIR_ID_RE.match(str(storm_id))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def dt_dedup_key(storm_key):
    """Digital Typhoon storm_key -> (year, seq). None if unparseable.

    '199115'   -> (1991, 15)     bare WP key, identical to v3's behaviour
    'AU197901' -> (1979,  1)     namespaced AU key, prefix stripped

    The leading-alphabetic strip is the ONLY change from v3, and it is a strict
    extension: for any input v3 accepted (exactly six digits) there is no prefix to
    strip and the result is bit-identical, and for any input v3 rejected without a
    prefix this still returns None. dryrun_test.py asserts that equivalence rather
    than leaving it as a claim.
    """
    s = re.sub(r"^[A-Za-z]+", "", str(storm_key))
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
                    "sub_source": "tcir",
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
#
# THE TWO DETECTORS AND THEIR THRESHOLDS ARE CARRIED OVER FROM v2/v3 UNCHANGED. Do
# not tune them. Two datasets that ultimately draw on the same real best-track
# record can hold the SAME physical storm under two different ID schemes; if that
# storm landed in one source's train split and the other's val split, val AUC would
# be inflated by memorisation.
#
# What v4 changes is ONLY the plumbing: v3's dedup_wpac() hardcoded "WPAC", the
# Digital Typhoon side, and the union policy. v4 splits out the position detector
# (track_match) so it can also be pointed at Digital Typhoon on both sides, and
# parameterises the caller-facing function by TCIR region and union policy. The
# (WP, "WPAC") call must reproduce v3's numbers bit for bit; dryrun_test.py pins
# that by running v3's own dedup_wpac() and v4's function on identical synthetic
# input and asserting the drop sets and audit counters are equal.
# ----------------------------------------------------------------------------
def track_match(query_points, ref_tracks, ref_allowed,
                match_deg=DEDUP_MATCH_DEG, min_points=DEDUP_MIN_MATCH_POINTS):
    """Best-track position match. The physically grounded detector, and the one
    that -- per v3's own analysis -- actually holds up.

    query_points -- {query_key: iterable of (timestamp, lat, lon)}
    ref_tracks   -- {timestamp: [(lat, lon, ref_key), ...]}, as built by
                    build_dt_index(); timestamps are integer YYYYMMDDHH on both
                    sides, which is what makes the comparison exact rather than
                    approximate in time.
    ref_allowed  -- only these ref_keys can be matched. Callers pass the set of
                    reference storms that actually made it into the training index:
                    a reference storm whose early stage was never imaged is not in
                    the training set, so a duplicate of it cannot leak, and matching
                    it would throw away a usable query storm for no safety gain.

    Returns {query_key: (best_ref_key, n_matching_points)} for every query storm
    that agrees with SOME single reference storm at >= min_points shared synoptic
    timestamps, each within match_deg degrees in BOTH lat and lon.

    Lifted verbatim from v3's dedup_wpac() detector 2, including the
    most_common(1)-then-threshold rule: the count that must clear min_points is the
    count against the single best-matching reference storm, not the total across
    all of them.
    """
    out = {}
    for qkey, pts in query_points.items():
        votes = Counter()
        for ts, la, lo in pts:
            if not (np.isfinite(ts) and np.isfinite(la) and np.isfinite(lo)):
                continue
            for rla, rlo, rkey in ref_tracks.get(int(ts), ()):
                if rkey not in ref_allowed:
                    continue
                if abs(rla - la) <= match_deg and abs(rlo - lo) <= match_deg:
                    votes[rkey] += 1
        if votes:
            best_key, best_n = votes.most_common(1)[0]
            if best_n >= min_points:
                out[qkey] = (best_key, best_n)
    return out


def dedup_tcir_against_dt(dt_index, dt_tracks, tcir_index, tcir_work,
                          tcir_region, use_id_key_union, dt_label, id_key_note):
    """Find TCIR storms in `tcir_region` that are the same physical storm as a
    Digital Typhoon storm from `dt_index`, and return the TCIR storm_keys to drop.

    Digital Typhoon wins every collision, in both pairs: 512x512 native resolution
    against TCIR's 128x128, and many more early-stage frames per storm.

    TWO independent detectors are always COMPUTED. Whether the ID-key one is also
    UNIONED into the drop set is the caller's decision, and it differs by pair --
    see DEDUP_ID_KEY_UNION for the full reasoning:

      1. ID KEY  -- (year, sequence_number) parsed out of both ID schemes.
         WPAC (unioned): Digital Typhoon's YYYYNN is the JMA *international* number,
           assigned only to systems reaching tropical-storm strength, while TCIR's
           WPAC numbering is the ATCF/JTWC one, which also numbers systems JMA never
           named. The two DRIFT APART within a season, and since Digital Typhoon
           numbers ~26 storms in every year TCIR covers, this detector fires on very
           nearly every TCIR WPAC storm. A near-100% hit rate is the EXPECTED
           reading there, not a bug.
         SH (computed, NOT unioned): AU's numbering is BOM's own per-year counter
           and TCIR's SH numbering is JTWC's. Two independent per-year counters, so
           a collision is a coincidence carrying no evidence -- exactly what the
           AU-vs-WP frame-hash probe demonstrated. Printed so its disagreement with
           the track detector is visible.

      2. TRACK MATCH -- both datasets carry best-track lat/lon at synoptic times. If
         a TCIR storm sits within DEDUP_MATCH_DEG degrees of some Digital Typhoon
         storm at >= DEDUP_MIN_MATCH_POINTS shared timestamps, it is the same storm
         regardless of how either agency numbered it. AUTHORITATIVE in both pairs.

    Where the union IS used, union rather than intersection is deliberate: dropping
    a TCIR storm that was not really a duplicate costs a handful of training frames,
    whereas keeping a real duplicate silently inflates val AUC. The asymmetry is the
    whole reason for running both.
    """
    print("\n" + "=" * 72)
    print(f"CROSS-DATASET DE-DUPLICATION -- {dt_label} vs TCIR region '{tcir_region}'")
    print("=" * 72)

    dt_storms = sorted(dt_index["storm_key"].unique())
    tcir_sub = tcir_index[tcir_index["basin"] == tcir_region]
    tcir_storms = sorted(tcir_sub["storm_key"].unique())
    print(f"Storms with usable early frames -- "
          f"{dt_label}={len(dt_storms)}  tcir[{tcir_region}]={len(tcir_storms)}")
    if not tcir_storms:
        print(f"No TCIR storms in region '{tcir_region}'; nothing to dedup.")
        print("=" * 72 + "\n")
        return set(), {
            "dt_label": dt_label, "tcir_region": tcir_region,
            "dt_storms": len(dt_storms), "tcir_storms": 0,
            "overlap_by_id_key": 0, "overlap_by_track_match": 0,
            "overlap_by_both": 0, "overlap_id_key_only": 0, "overlap_track_only": 0,
            "detector_jaccard": None,
            "n_tcir_storms_dropped": 0, "n_tcir_storms_kept": 0,
            "dropped_tcir_storm_ids": [], "used_id_key_union": bool(use_id_key_union),
            "match_deg": DEDUP_MATCH_DEG, "min_match_points": DEDUP_MIN_MATCH_POINTS,
        }

    # ---- detector 1: (year, seq) key ----
    dt_keys = {}
    n_dt_unparsed = 0
    for s in dt_storms:
        k = dt_dedup_key(s)
        if k is None:
            n_dt_unparsed += 1
            print(f"  WARN: {dt_label} storm id '{s}' does not parse as YYYYNN")
            continue
        dt_keys.setdefault(k, []).append(s)
    if n_dt_unparsed:
        print(f"  {n_dt_unparsed} {dt_label} ids unparseable (excluded from key dedup)")

    by_key = set()
    key_hits = []
    for s in tcir_storms:
        k = tcir_dedup_key(s)
        if k is None:
            print(f"  WARN: TCIR {tcir_region} id '{s}' does not parse; "
                  f"key dedup skipped for it")
            continue
        if k in dt_keys:
            by_key.add(s)
            key_hits.append((s, k, dt_keys[k][0]))
    print(f"[1] ID-key (year, seq) overlap: {len(by_key)} TCIR {tcir_region} storms"
          f"{'' if use_id_key_union else '   (AUDIT ONLY -- not dropped)'}")
    for h in key_hits[:10]:
        print(f"      tcir {h[0]} -> key {h[1]} -> {dt_label} {h[2]}")
    print(f"    {id_key_note}")

    # ---- detector 2: best-track position match ----
    t0 = time.time()
    dt_storm_set = set(dt_storms)
    tcir_set = set(tcir_storms)
    tw = tcir_work[tcir_work["region"] == tcir_region]
    query_points = {}
    for storm_id, g in tw.groupby("storm_id", sort=True):
        if storm_id not in tcir_set:
            continue
        query_points[storm_id] = list(zip(g["time"].to_numpy(),
                                          g["lat"].to_numpy(),
                                          g["lon"].to_numpy()))
    by_track = track_match(query_points, dt_tracks, dt_storm_set)
    print(f"[2] Best-track position overlap "
          f"(<= {DEDUP_MATCH_DEG} deg at >= {DEDUP_MIN_MATCH_POINTS} shared timestamps): "
          f"{len(by_track)} TCIR {tcir_region} storms  ({time.time() - t0:.0f}s)"
          f"{'' if use_id_key_union else '   (AUTHORITATIVE for this pair)'}")
    for sid, (dsid, n) in list(sorted(by_track.items()))[:10]:
        print(f"      tcir {sid} -> {dt_label} {dsid} ({n} matching track points)")

    # ---- disagreement between the two detectors, printed for audit ----
    track_set = set(by_track)
    only_key = sorted(by_key - track_set)
    only_track = sorted(track_set - by_key)
    n_both = len(by_key & track_set)
    agree_frac = n_both / max(1, len(by_key | track_set))
    print(f"\nDetector agreement: both={n_both}  "
          f"ID-key only={len(only_key)}  track only={len(only_track)}  "
          f"(Jaccard {agree_frac:.2f})")
    if only_key:
        print(f"  ID-key only ({len(only_key)}): the (year, seq) numbers collided but "
              f"the tracks did NOT, so these are most likely DIFFERENT storms that "
              f"merely share a sequence number."
              f"{' Expected to be a large group -- see the JMA-vs-JTWC numbering note above.' if use_id_key_union else ' This is the group that would be WRONGLY dropped if the ID key were unioned here.'}")
        print(f"      examples: {only_key[:15]}")
    if only_track:
        print(f"  track only ({len(only_track)}): SAME storm, different agency "
              f"numbering. These are exactly the leaks the ID key alone would have "
              f"MISSED.")
        print(f"      examples: {only_track[:15]}")

    if use_id_key_union:
        drop = by_key | track_set
        print(f"\nDEDUP_ID_KEY_UNION['{tcir_region}']=True -> dropping the UNION of "
              f"both detectors")
    else:
        drop = set(track_set)
        print(f"\nDEDUP_ID_KEY_UNION['{tcir_region}']=False -> dropping only "
              f"position-matched storms; the {len(only_key)} ID-key-only hits above "
              f"are KEPT as genuinely distinct storms")
    print(f"Dropping {len(drop)} of {len(tcir_storms)} TCIR {tcir_region} storms as "
          f"duplicates of {dt_label} ({dt_label} copy kept: higher resolution, more "
          f"early frames)")
    print("=" * 72 + "\n")

    audit = {
        "dt_label": dt_label,
        "tcir_region": tcir_region,
        "dt_storms": len(dt_storms),
        "tcir_storms": len(tcir_storms),
        "overlap_by_id_key": len(by_key),
        "overlap_by_track_match": len(track_set),
        "overlap_by_both": n_both,
        "overlap_id_key_only": len(only_key),
        "overlap_track_only": len(only_track),
        "detector_jaccard": float(agree_frac),
        "n_tcir_storms_dropped": len(drop),
        "n_tcir_storms_kept": len(tcir_storms) - len(drop),
        "dropped_tcir_storm_ids": sorted(drop),
        "used_id_key_union": bool(use_id_key_union),
        "match_deg": DEDUP_MATCH_DEG,
        "min_match_points": DEDUP_MIN_MATCH_POINTS,
    }
    return drop, audit


def au_vs_wp_track_null_test(au_storm_tracks, au_index, wp_tracks, wp_index):
    """(E4) Is there ANY real overlap between the two Digital Typhoon mounts?

    THIS IS A NULL TEST, AND THE ONLY DETECTOR IT USES IS THE PHYSICAL ONE. The
    ID-key detector is deliberately not run and not offered: a live probe hashed the
    first frame of 80 storm pairs whose ids coincide across the two mounts and found
    0 identical / 80 different, i.e. AU's YYYYNN is its own per-year counter and the
    459/480 id coincidence is two independent counters colliding. Running an ID-key
    dedup here would delete ~459 real Southern Hemisphere storms -- the entire point
    of v4 -- on the strength of a coincidence.

    The position detector, by contrast, cannot be fooled by a numbering scheme, and
    here it is close to a designed-in null: a match needs two storms within 1.5
    degrees of each other in latitude at four or more shared synoptic times, and the
    two mounts are opposite hemispheres. Tropical cyclones do not form within ~5
    degrees of the equator, so the physically correct answer is 0.

    Report only by default (DEDUP_AU_VS_WP_DROP). Returns (drop_set, audit).
    """
    print("\n" + "=" * 72)
    print("(E4) NULL TEST -- Digital Typhoon AU vs Digital Typhoon WP")
    print("=" * 72)
    au_storms = set(au_index["storm_key"].unique())
    wp_storms = set(wp_index["storm_key"].unique())

    # What the (discredited) ID key WOULD have said, printed so the size of the trap
    # v4 is avoiding is on the record rather than only in this docstring.
    au_keys = {dt_dedup_key(s) for s in au_storms} - {None}
    wp_keys = {dt_dedup_key(s) for s in wp_storms} - {None}
    n_id_coincide = len(au_keys & wp_keys)
    print(f"AU storms={len(au_storms)}  WP storms={len(wp_storms)}")
    print(f"(year, seq) keys that coincide between the mounts: {n_id_coincide} "
          f"of {len(au_keys)} AU keys ({n_id_coincide / max(1, len(au_keys)):.1%})")
    print("  ^ NOT USED FOR ANYTHING. A live probe hashed the first frame of 80 such "
          "pairs:\n    0 identical, 80 different. AU runs BOM's own per-year counter. "
          "This number is\n    printed only to show how many real storms an ID-key "
          "dedup would have destroyed.")

    query = {k: v for k, v in au_storm_tracks.items() if k in au_storms}
    hits = track_match(query, wp_tracks, wp_storms)
    frac = len(hits) / max(1, len(au_storms))
    print(f"\nPosition match (the only detector used here, <= {DEDUP_MATCH_DEG} deg at "
          f">= {DEDUP_MIN_MATCH_POINTS} shared timestamps):")
    print(f"  {len(hits)} of {len(au_storms)} AU storms ({frac:.2%}) match a WP storm")
    for k, (w, n) in list(sorted(hits.items()))[:20]:
        print(f"      AU {k} -> WP {w} ({n} matching track points)")

    if len(hits) == 0:
        print("  => ZERO, which is the expected and physically correct answer: the "
              "two mounts are\n     opposite hemispheres. The 459/480 id coincidence "
              "above is confirmed to be\n     numbering noise, and all AU storms are "
              "added as genuinely new data.")
    elif frac <= DEDUP_AU_VS_WP_ALERT_FRAC:
        print(f"  => {len(hits)} match(es), below the {DEDUP_AU_VS_WP_ALERT_FRAC:.0%} "
              f"alert level. Most likely near-equatorial\n     edge cases. Listed "
              f"above so they can be checked by hand.")
    else:
        print("\n" + "!" * 72)
        print(f"!! {frac:.1%} of AU storms POSITION-MATCH a WP storm. That contradicts "
              f"the\n!! hemisphere-separation assumption this test was built on, and it "
              f"means the\n!! same physical storm may sit on both sides of the "
              f"train/val split.\n!! DO NOT TRUST THIS RUN'S VAL AUC. Investigate, then "
              f"set DEDUP_AU_VS_WP_DROP=True.")
        print("!" * 72)

    drop = set(hits) if DEDUP_AU_VS_WP_DROP else set()
    if DEDUP_AU_VS_WP_DROP and drop:
        print(f"\nDEDUP_AU_VS_WP_DROP=True -> dropping {len(drop)} AU storms "
              f"(WP copy kept: it is the mirror v3's checkpoint and "
              f"render_demo.py's bare-id lookup already use)")
    else:
        print("\nDEDUP_AU_VS_WP_DROP=False -> REPORT ONLY, no AU storm dropped")
    print("=" * 72 + "\n")

    return drop, {
        "n_au_storms": len(au_storms),
        "n_wp_storms": len(wp_storms),
        "n_id_key_coincidences_NOT_used": n_id_coincide,
        "n_position_matches": len(hits),
        "position_match_fraction": float(frac),
        "matches": {str(k): [str(v[0]), int(v[1])] for k, v in sorted(hits.items())},
        "acted_on": bool(DEDUP_AU_VS_WP_DROP),
        "n_au_storms_dropped": len(drop),
        "id_key_dedup_run": False,
    }


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


def augment_batch(x01, gen):
    """(B1) Batched augmentation in the [0,1] display domain.

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
    n = x01.shape[0]
    dev = x01.device

    def rnd(lo, hi):
        return (torch.rand(n, generator=gen) * (hi - lo) + lo).to(dev)

    # ---- geometric: isotropic zoom + small translation ----
    # affine_grid's theta maps OUTPUT coordinates back into the INPUT, so the
    # scale entry is 1/zoom: a theta scale below 1 samples a smaller region of the
    # input, which magnifies the storm.
    zoom = torch.exp(rnd(-AUG_ZOOM_LOG, AUG_ZOOM_LOG))
    tx = rnd(-AUG_SHIFT, AUG_SHIFT)
    ty = rnd(-AUG_SHIFT, AUG_SHIFT)
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
    gain = rnd(1.0 - AUG_CONTRAST, 1.0 + AUG_CONTRAST).view(n, 1, 1, 1)
    off = rnd(-AUG_BRIGHTNESS, AUG_BRIGHTNESS).view(n, 1, 1, 1)
    x01 = (x01 - 0.5) * gain + 0.5 + off

    # ---- sensor noise ----
    if AUG_NOISE_STD > 0:
        # Generated on the CPU and shipped over, like the parameters above, so the
        # whole augmentation is reproducible from one seed on any device. Measured
        # cost: 6 ms per batch of 32, i.e. ~45 s across a full 24-epoch run -- ~4%,
        # which is not worth a separate CUDA generator and a device-dependent path.
        noise = torch.randn(x01.shape, generator=gen).to(dev) * AUG_NOISE_STD
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

    def __init__(self, images_u8, labels, weights, augment):
        self.images = images_u8
        self.labels = labels
        self.weights = weights
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augment:
            # Rotations only -- NOT flips. A mirror flip reverses the storm's
            # rotational chirality, which is physically fixed by hemisphere and is
            # exactly the structure the model should key on. (This dataset contains
            # BOTH chiralities for real, since the SH basin spins the other way --
            # which is a reason to keep flips out, not to allow them: a flipped SH
            # storm is a physically impossible NH storm. v4 makes this stronger
            # still: the AU mount adds ~475 SH storms at 512x512 native resolution,
            # so the counter-clockwise-vs-clockwise distinction is now carried by
            # high-resolution frames rather than only by TCIR's 128x128 upsamples.
            # A flip augmentation here would be teaching the model that a physically
            # impossible storm is a valid example of a real one.)
            #
            # Done here rather than in augment_batch because a 90-degree rotation of
            # a 224x224 uint8 array is 50 KB of memcpy, whereas doing it per-sample
            # on the GPU would need the batch split into four groups.
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img = np.rot90(img, k).copy()
        return torch.from_numpy(img), float(self.labels[i]), float(self.weights[i])


def preload_all(index, tcir_h5_path):
    """Decode every frame ONCE, resized, into a single uint8 array in RAM.

    ~22k frames at 224x224 uint8 == ~1.1GB (v3: 12.3k / 620MB), fine for Kaggle RAM,
    and it turns 24 epochs of HDF5 reads off three network-mounted datasets into one
    pass. The train/val/probe views taken later are boolean-mask COPIES, so peak
    resident is roughly 2.5x this array.

    TCIR is done FIRST and deliberately: it is the unknown-latency source (random
    row access into one 20GB file). Its rows are read in ASCENDING frame order so
    the access pattern is as close to sequential as it can be. An ETA is printed
    after 100 frames of each pass, so a pathologically slow mount is caught in the
    first minute rather than after half an hour.

    Both Digital Typhoon mounts go through the SAME pass and the same loader: they
    are selected on source == "digital_typhoon", which is exactly why the AU mount
    keeps that coarse source name rather than getting one of its own. The frames are
    byte-identical in layout (512x512 float64 Kelvin under the "Infrared" key), so
    there is nothing for a second code path to do.
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
def radial_scale_diagnostic(images_u8, groups, labels, ratio_pairs, title,
                            row_mask=None, max_per_group=800, seed=SEED):
    """Measure how far the cold-cloud signal extends from frame centre, per group.

    WHY THIS EXISTS: the leading explanation for digital_typhoon auc=0.728 vs tcir
    auc=0.598 is that the two datasets crop DIFFERENT physical areas around the
    storm. Digital Typhoon is documented as 512x512 at ~5 km/pixel; TCIR is a
    128x128 crop on a ~0.07-degree grid. If those cover different spans, then after
    both are resized to 224x224 the SAME storm occupies a different fraction of the
    frame in each source, and a convolutional net trained mostly on one scale
    cannot be expected to transfer to the other.

    Method: per frame, take the "cold" mass w = max(0, pixel/255 - 0.5), i.e. every
    pixel colder than the midpoint of the 190-300 K display window, and report the
    weighted median radius from frame centre in units of the half-width. Frames are
    grouped by (group, label) so a difference cannot be an artefact of one group
    holding stronger storms.

    WHAT v4 CHANGES, AND WHY IT MATTERS. v3 ran this once with `groups` = source and
    got 0.928 (label 0) / 0.877 (label 1), reading that as "near 1.0, hypothesis
    refuted". That reading had a real confound v3 could not remove: digital_typhoon
    was WPAC-only and tcir was all six basins, so a source difference and a basin
    difference were measured together. v4 has high-resolution Digital Typhoon frames
    in the SH basin (the AU mount), so main() calls this a SECOND time restricted to
    SH, where dt_au and tcir cover the same region and the comparison is finally
    like-for-like. If the SH-only ratio also lands near 1.0, the hypothesis is
    refuted on a controlled comparison rather than a confounded one.

    HOW TO READ THE RESULT: a ratio near 1.00 refutes the field-of-view hypothesis
    (the gap is then about native resolution or label noise). A ratio near 0.6
    supports it and points at a centre-crop fix -- which would ALSO require the same
    crop in detect.py's Mode A, i.e. it is not a training-only change, which is
    exactly why this measures rather than acts.

    `row_mask` restricts the frames considered (used for the SH-only pass).
    `ratio_pairs` is a list of (numerator_group, denominator_group) to report.
    """
    print("\n" + "=" * 72)
    print(f"(D1) RADIAL SCALE DIAGNOSTIC -- {title}")
    print("=" * 72)

    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    c = (IMG_SIZE - 1) / 2.0
    r = (np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / (IMG_SIZE / 2.0)).ravel()
    order = np.argsort(r)
    r_sorted = r[order]

    rng = np.random.RandomState(seed)
    out = {}
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    base = np.ones(len(groups), dtype=bool) if row_mask is None else np.asarray(row_mask)
    for src in sorted(set(groups[base].tolist())):
        for lab in (0, 1):
            pos = np.where(base & (groups == src) & (labels == lab))[0]
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
            print(f"  {src:20s} label={lab}  n={len(r50):5d}  "
                  f"cold-mass r50 median={np.median(r50):.3f} "
                  f"(IQR {np.percentile(r50, 25):.3f}-{np.percentile(r50, 75):.3f})")

    for num, den in ratio_pairs:
        for lab in (0, 1):
            kn, kd = f"{num}|label{lab}", f"{den}|label{lab}"
            if kn in out and kd in out:
                ratio = (out[kn]["cold_mass_r50_median"]
                         / max(1e-9, out[kd]["cold_mass_r50_median"]))
                out[f"{num}_over_{den}_r50_ratio_label{lab}"] = float(ratio)
                print(f"  => label={lab}: {num} / {den} r50 ratio = {ratio:.3f}")
    print("  A ratio near 1.00 REFUTES the field-of-view hypothesis (the gap is then")
    print("  about native resolution or label noise). A ratio near 0.6 SUPPORTS it")
    print("  and points at a centre-crop fix -- which would also need the same crop")
    print("  added to detect.py's Mode A, so do not apply it blind.")
    print("  v3 measured 0.928 (label 0) / 0.877 (label 1) for digital_typhoon/tcir,")
    print("  confounded by basin. Compare the SH-only pass below for the clean answer.")
    print("=" * 72 + "\n")
    return out


# ----------------------------------------------------------------------------
# Inference helpers
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(net, loader, device, tta=False):
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
    for imgs_u8, labels, _w in loader:
        x = to_model_input(imgs_u8.to(device))
        if tta:
            p = torch.zeros(x.shape[0], device=device)
            for k in range(4):
                p += torch.sigmoid(net(torch.rot90(x, k, dims=(2, 3))).squeeze(1))
            p /= 4.0
            p = p.clamp(1e-6, 1 - 1e-6)
            outs.append(torch.log(p / (1 - p)).cpu().numpy())
        else:
            outs.append(net(x).squeeze(1).cpu().numpy())
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

    print("=" * 72)
    print("Cyclone Genesis Predictor intensification model, v4")
    print(f"  EPOCHS={EPOCHS} (warmup {WARMUP_EPOCHS})  FRAMES_PER_STORM={FRAMES_PER_STORM}  "
          f"BATCH={BATCH_SIZE}  LR={LR}")
    print(f"  EMA_DECAY={EMA_DECAY}  LABEL_SMOOTHING={LABEL_SMOOTHING}  "
          f"BASIN_BALANCE_ALPHA={BASIN_BALANCE_ALPHA}")
    print(f"  augment: brightness+/-{AUG_BRIGHTNESS} contrast+/-{AUG_CONTRAST} "
          f"zoom x{math.exp(-AUG_ZOOM_LOG):.3f}-{math.exp(AUG_ZOOM_LOG):.3f} "
          f"shift+/-{AUG_SHIFT} noise{AUG_NOISE_STD}")
    print("  sources: Digital Typhoon WP + Digital Typhoon AU (new in v4) + TCIR")
    print(f"  dedup id-key union policy: {DEDUP_ID_KEY_UNION}  "
          f"AU-vs-WP drop: {DEDUP_AU_VS_WP_DROP} (null test, report only)")
    print("  expected GPU time ~34 min, ceiling ~37 min (v3 measured ~19 min); see "
          "RUNTIME BUDGET in the module docstring for the dials")
    print("=" * 72)

    # ------------------------------------------------------------------ discovery
    # (E2) Both Digital Typhoon mounts, resolved by keyword. v3's "first directory
    # whose name contains 'typhoon'" would now return either one at random.
    dt_specs = discover_dt_sources()
    by_tag = {e["tag"]: e for e in dt_specs}
    if "wp" not in by_tag:
        raise SystemExit("Digital Typhoon WP mount is mandatory (it is the source of "
                         "v1/v2/v3's WPAC storms) and was not resolved.")

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
    # Both Digital Typhoon mounts, indexed through the SAME function with different
    # basin / sub_source / key_prefix. Nothing about how a frame is read, labelled or
    # preprocessed differs between them -- they are the same dataset in two regions.
    dt_parts, dt_audits = {}, {}
    dt_tracks_by_tag, dt_storm_tracks_by_tag = {}, {}
    for spec in dt_specs:
        print()
        idx, tracks, storm_tracks, aud = build_dt_index(
            spec["meta_dir"], spec["image_dir"],
            basin=spec["basin"], sub_source=spec["sub_source"],
            key_prefix=spec["key_prefix"], tag="A:" + spec["tag"],
        )
        if idx.empty:
            raise SystemExit(
                f"Digital Typhoon '{spec['tag']}' index is empty -- layout is not what "
                f"was expected under {spec['meta_dir']}")
        dt_parts[spec["tag"]] = idx
        dt_tracks_by_tag[spec["tag"]] = tracks
        dt_storm_tracks_by_tag[spec["tag"]] = storm_tracks
        dt_audits[spec["sub_source"]] = aud

    # (E4) NULL TEST first, before anything is dropped: does AU actually overlap WP?
    au_drop = set()
    au_wp_audit = None
    if "au" in dt_parts:
        au_drop, au_wp_audit = au_vs_wp_track_null_test(
            dt_storm_tracks_by_tag["au"], dt_parts["au"],
            dt_tracks_by_tag["wp"], dt_parts["wp"],
        )
        if au_drop:
            dt_parts["au"] = dt_parts["au"][
                ~dt_parts["au"]["storm_key"].isin(au_drop)].reset_index(drop=True)

    # ------------------------------------------------------------------ dedup
    # (E5) The SAME generalised function, called once per (Digital Typhoon mount,
    # TCIR region) pair. The WP/WPAC call is v3's dedup_wpac() with its plumbing
    # pulled out and must reproduce v3's numbers exactly; dryrun_test.py pins that
    # against v3's actual code.
    dedup_audits = {}
    drop_ids = set()

    drop_wpac, aud_wpac = dedup_tcir_against_dt(
        dt_parts["wp"], dt_tracks_by_tag["wp"], tcir_index, tcir_work,
        tcir_region="WPAC",
        use_id_key_union=DEDUP_ID_KEY_UNION["WPAC"],
        dt_label="digital_typhoon_wp",
        id_key_note="Both agencies number the same real WPAC season, so an ID "
                    "collision is weak-but-real evidence here; UNIONED.",
    )
    drop_ids |= drop_wpac
    dedup_audits["wp_vs_tcir_wpac"] = aud_wpac

    if "au" in dt_parts:
        drop_sh, aud_sh = dedup_tcir_against_dt(
            dt_parts["au"], dt_tracks_by_tag["au"], tcir_index, tcir_work,
            tcir_region="SH",
            use_id_key_union=DEDUP_ID_KEY_UNION["SH"],
            dt_label="digital_typhoon_au",
            id_key_note="AU numbers on BOM's own per-year counter and TCIR-SH on "
                        "JTWC's -- two INDEPENDENT counters, so a collision here is "
                        "a coincidence and carries no evidence. AUDIT ONLY.",
        )
        drop_ids |= drop_sh
        dedup_audits["au_vs_tcir_sh"] = aud_sh

    tcir_index = tcir_index[~tcir_index["storm_key"].isin(drop_ids)].reset_index(drop=True)

    index = pd.concat(list(dt_parts.values()) + [tcir_index], ignore_index=True)

    # storm_key must be globally unique across ALL THREE mounts, because it is both
    # the split key AND what gets written into the checkpoint's
    # train_storm_ids/val_storm_ids (which demo_kernel/render_demo.py looks a raw
    # Digital Typhoon storm id up in).
    #
    # v3 checked this on `source` and that check is no longer sufficient: DT-WP and
    # DT-AU share source=="digital_typhoon", and 459 of AU's 480 raw ids are also WP
    # ids while being DIFFERENT storms. Without the "AU" key_prefix, pandas would
    # have merged ~459 pairs into one split unit each and v3's assert would have
    # stayed silent. Checking `sub_source` is what actually catches that class of
    # bug, so both checks are run and the sub_source one is the load-bearing one.
    for col in ("source", "sub_source"):
        clashing = index.groupby("storm_key")[col].nunique()
        clashing = clashing[clashing > 1]
        if len(clashing):
            raise SystemExit(
                f"storm_key collision across {col}: {list(clashing.index[:10])}. "
                f"Two different storms would be merged into one split unit; aborting."
            )
    # Belt and braces on the same failure: every mount's key set must be disjoint.
    seen = {}
    for sub, grp in index.groupby("sub_source"):
        ks = set(grp["storm_key"].unique())
        for other_sub, other_ks in seen.items():
            both = ks & other_ks
            if both:
                raise SystemExit(
                    f"{len(both)} storm_keys appear in BOTH {sub} and {other_sub} "
                    f"(e.g. {sorted(both)[:10]}). Check DT_SOURCES key_prefix.")
        seen[sub] = ks

    storms_by_source_basin = (
        index.drop_duplicates("storm_key")
        .groupby(["sub_source", "basin"])
        .size()
        .to_dict()
    )
    print("Final storms per sub-source per basin (after dedup):")
    for (src, basin), n in sorted(storms_by_source_basin.items()):
        print(f"    {src:20s} {basin:5s} {n:5d}")
    print("Final frames per sub-source per basin:")
    for (src, basin), n in sorted(
            index.groupby(["sub_source", "basin"]).size().to_dict().items()):
        print(f"    {src:20s} {basin:5s} {n:5d}")

    # (E1) The basin taxonomy is a user-visible contract: local_tool/detect.py reads
    # config['basins'] and renders "trained on Digital Typhoon + TCIR, N basins". A
    # missing basin means a source silently failed to load; an extra one means that
    # label is now wrong and detect.py would need a change it is not getting.
    got_basins = tuple(sorted(index["basin"].unique().tolist()))
    if got_basins != tuple(sorted(EXPECTED_BASINS)):
        raise SystemExit(
            f"Basin set changed: got {list(got_basins)}, expected "
            f"{sorted(EXPECTED_BASINS)}. config['basins'] is read at runtime by "
            f"local_tool/detect.py; refusing to ship a checkpoint that silently "
            f"changes what the tool claims it was trained on.")
    print(f"Basin set OK: {list(got_basins)} (unchanged from v3, so detect.py needs "
          f"no edit)")

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
    #
    # Same SEED and same procedure as v2, so v2 and v3 are compared on the SAME
    # validation storms -- the AUC difference is then attributable to the training
    # changes rather than to a luckier split.
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

    # (D1) run on everything that was actually loaded, before the train/val split
    # matters -- it is a property of the data, not of the model.
    #
    # PASS 1 reproduces v3's comparison exactly (source-level, all basins) so the
    # 0.928 / 0.877 numbers stay directly comparable across versions.
    # PASS 2 is the new, CONTROLLED comparison v3 could not run: restricted to the
    # SH basin, where digital_typhoon_au and tcir cover the same region, so a
    # difference cannot be a basin artefact. This is the answer to read.
    idx_basin = index["basin"].to_numpy()
    idx_source = index["source"].to_numpy()
    idx_sub = index["sub_source"].to_numpy()
    idx_label = index["label"].to_numpy()

    fov_diag = radial_scale_diagnostic(
        images, idx_source, idx_label,
        ratio_pairs=[("digital_typhoon", "tcir")],
        title="v3-comparable: per SOURCE, all basins (confounded by basin)")

    fov_diag_sh = radial_scale_diagnostic(
        images, idx_sub, idx_label,
        ratio_pairs=[("digital_typhoon_au", "tcir")],
        title="v4 CONTROLLED: per SUB-SOURCE, SH basin only (same region both sides)",
        row_mask=(idx_basin == "SH"))

    tr_mask = (index["split"] == "train").to_numpy()
    va_mask = ~tr_mask
    y = index["label"].to_numpy().astype(np.float32)
    val_basins = index.loc[va_mask, "basin"].to_numpy()
    val_sources = index.loc[va_mask, "source"].to_numpy()
    val_sub_sources = index.loc[va_mask, "sub_source"].to_numpy()
    val_storm_keys = index.loc[va_mask, "storm_key"].to_numpy()

    # (D3) basin-balance weights are computed from the TRAIN frames only -- letting
    # the val distribution influence training weights would be leakage.
    train_weights = basin_balance_weights(index.loc[tr_mask, "basin"].to_numpy(),
                                          BASIN_BALANCE_ALPHA)
    if BASIN_BALANCE_ALPHA != 0.0:
        wb = pd.DataFrame({"basin": index.loc[tr_mask, "basin"].to_numpy(),
                           "w": train_weights}).groupby("basin")["w"]
        print(f"\nBasin balance alpha={BASIN_BALANCE_ALPHA}: gradient mass share per basin")
        tot = train_weights.sum()
        for basin, s in wb.sum().items():
            print(f"    {basin:5s} weight/frame={wb.mean()[basin]:.3f}  "
                  f"share={s / tot:.1%}")

    train_ds = FrameDataset(images[tr_mask], y[tr_mask], train_weights, augment=True)
    val_ds = FrameDataset(images[va_mask], y[va_mask],
                          np.ones(int(va_mask.sum())), augment=False)
    # drop_last=True: with BatchNorm a trailing batch of size 1 raises at train
    # time, and which sizes occur depends on how many frames survive preload. A
    # different sub-32 subset is dropped every epoch because the loader reshuffles,
    # so no frame is systematically excluded.
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True)
    # shuffle=False so the concatenated val probs line up with val_basins/val_sources.
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # (A4) A fixed, un-augmented subsample of the TRAIN frames. Scoring it every
    # epoch turns "underfit or overfit?" -- the question v2 could only be reasoned
    # about after the fact -- into a number printed next to the val AUC. If train
    # AUC runs far above val AUC the model is memorising; if both stay low it is
    # still underfit and EPOCHS should go up again.
    tr_idx = np.where(tr_mask)[0]
    probe_sel = np.random.RandomState(SEED + 1).choice(
        len(tr_idx), min(TRAIN_PROBE_FRAMES, len(tr_idx)), replace=False)
    probe_ds = FrameDataset(images[tr_idx[probe_sel]], y[tr_idx[probe_sel]],
                            np.ones(len(probe_sel)), augment=False)
    probe_dl = DataLoader(probe_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    net, pretrained = build_model()
    net.to(device)
    ema = ModelEma(net, EMA_DECAY)

    n_pos = float((y[tr_mask] == 1).sum())
    n_neg = float((y[tr_mask] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    print(f"train frames: pos={n_pos:.0f} neg={n_neg:.0f} pos_weight={pos_weight.item():.3f}")

    # reduction='none' so the per-frame basin-balance weights can be applied. With
    # BASIN_BALANCE_ALPHA=0 every weight is exactly 1.0 and the weighted mean below
    # is arithmetically identical to reduction='mean'.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # (A1) Linear warmup then cosine. Warmup because the freshly-attached 1-channel
    # conv1 and the freshly-initialised fc start far from the pretrained trunk's
    # scale, and a full-rate first epoch on 24 rather than 8 epochs has time to do
    # real damage to those pretrained filters. Stepped PER EPOCH (v2 also stepped
    # per epoch), so T_max is in epochs.
    def lr_at(epoch_idx):
        if epoch_idx < WARMUP_EPOCHS:
            return (epoch_idx + 1) / max(1, WARMUP_EPOCHS)
        t = (epoch_idx - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    # No AMP on purpose: ~12k 224px images means an epoch is under a minute of
    # resnet18 compute either way, and fp32 avoids torch.cuda.amp / torch.amp API
    # churn on top of everything else that is new in this version.

    aug_gen = torch.Generator()

    history = []
    best_auc = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # Re-seeded per epoch from a fixed base so the augmentation stream is
        # reproducible for a given (SEED, epoch) but differs between epochs.
        aug_gen.manual_seed(SEED * 1000 + epoch)

        net.train()
        total_loss, total_w = 0.0, 0.0
        t0 = time.time()
        for imgs_u8, labels, w in train_dl:
            x01 = to_unit(imgs_u8.to(device, non_blocking=True))
            x = normalize(augment_batch(x01, aug_gen))
            t = labels.to(device).unsqueeze(1)
            # (B2) Label smoothing: pull both targets in from {0,1} to
            # {eps, 1-eps}. Applied to the TARGET, not the loss, so pos_weight and
            # the per-frame weights keep their meaning.
            t = t * (1.0 - 2.0 * LABEL_SMOOTHING) + LABEL_SMOOTHING
            wt = w.to(device).unsqueeze(1)

            opt.zero_grad(set_to_none=True)
            logits = net(x)
            per_frame = criterion(logits, t)
            loss = (per_frame * wt).sum() / wt.sum()
            loss.backward()
            opt.step()
            ema.update(net)

            total_loss += loss.item() * float(wt.sum().item())
            total_w += float(wt.sum().item())
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        train_loss = total_loss / max(1e-9, total_w)

        # (A3) Everything below is scored on the EMA weights, because that is what
        # gets saved. `net` is left untouched: the EMA state is loaded into a
        # throwaway view of the same module, then the live weights are restored.
        live_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(ema.state_dict(net))

        val_logits, val_y = predict_logits(net, val_dl, device)
        val_probs = sigmoid_np(val_logits)
        m = evaluate(val_probs, val_y)
        m["epoch"] = epoch
        m["lr"] = float(cur_lr)
        m["train_loss"] = train_loss
        m["val_loss"] = binary_nll(val_probs, val_y)
        m["val_ece"] = expected_calibration_error(val_probs, val_y)[0]

        probe_logits, probe_y = predict_logits(net, probe_dl, device)
        pm = evaluate(sigmoid_np(probe_logits), probe_y)
        m["train_probe_auc"] = pm["auc"]
        m["train_probe_loss"] = binary_nll(sigmoid_np(probe_logits), probe_y)

        history.append(m)
        print(f"epoch {epoch:2d}/{EPOCHS}  lr={cur_lr:.2e}  loss={train_loss:.4f}  "
              f"val_loss={m['val_loss']:.4f}  val_acc={m['accuracy']:.3f}  "
              f"val_auc={m['auc']:.3f}  val_ece={m['val_ece']:.3f}  "
              f"[probe] train_auc={m['train_probe_auc']:.3f} "
              f"train_loss={m['train_probe_loss']:.4f}  ({time.time() - t0:.0f}s)")

        # Select on AUC of the EMA model: threshold-free, which matters because the
        # demo displays a raw confidence rather than a hard 0/1 decision, and much
        # less noisy epoch-to-epoch than v2's per-epoch raw weights were.
        score = m["auc"] if m["auc"] == m["auc"] else m["accuracy"]
        if score > best_auc:
            best_auc, best_epoch = score, epoch
            best_state = {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
            # Crash insurance: write a fully valid (but not yet calibrated)
            # checkpoint now, so an interrupted run still leaves something loadable.
            # It is overwritten at the end by the calibrated one.
            torch.save(
                {"state_dict": best_state, "arch": "resnet18_1ch",
                 "pretrained_backbone": pretrained, "epoch": epoch,
                 "val_metrics": jsonable(m),
                 "config": jsonable({"temp_lo": TEMP_LO, "temp_hi": TEMP_HI,
                                     "img_size": IMG_SIZE, "norm_mean": NORM_MEAN,
                                     "norm_std": NORM_STD, "typhoon_grade": TYPHOON_GRADE,
                                     "calibrated": False}),
                 "train_storm_ids": sorted(train_keys),
                 "val_storm_ids": sorted(val_keys)},
                CKPT_PATH,
            )
            print(f"  -> interim best checkpoint saved (uncalibrated)")

        net.load_state_dict(live_state)  # resume training from the live weights

    if best_state is None:
        raise SystemExit("No epoch produced a scoreable validation result; nothing to save.")

    # ------------------------------------------------------------------ selection audit
    # v2 picked the best of 8 noisy val AUCs; v3 picks the best of 24. Best-of-N on
    # a noisy metric is optimistically biased, so print the spread that the pick
    # was made from -- if best_auc sits far above the tail mean, treat the headline
    # number as the top of a range rather than as the model's true skill.
    tail = [h["auc"] for h in history[-5:] if h["auc"] == h["auc"]]
    print(f"\nSelection audit: best val AUC {best_auc:.4f} at epoch {best_epoch} of "
          f"{EPOCHS}; last-5-epoch AUC mean={np.mean(tail):.4f} sd={np.std(tail):.4f}. "
          f"A large gap between the two means the pick is partly selection noise.")

    # ------------------------------------------------------------------ calibration
    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    val_logits, val_y = predict_logits(net, val_dl, device)
    raw_probs = sigmoid_np(val_logits)

    print("\n" + "=" * 72)
    print("(C) PROBABILITY CALIBRATION")
    print("=" * 72)
    ece_before, bins_before = expected_calibration_error(raw_probs, val_y)
    print(f"before: NLL={binary_nll(raw_probs, val_y):.4f}  ECE={ece_before:.4f}  "
          f"prob range [{raw_probs.min():.3f}, {raw_probs.max():.3f}]")

    print(f"storm-level {CALIB_CV_FOLDS}-fold cross-check "
          f"(does the calibration generalise?):")
    cross, verdict = calibration_crosscheck(val_logits, val_y, val_storm_keys, SEED + 7)

    if verdict["generalises"]:
        a, b, calib_info = fit_platt_scaling(val_logits, val_y)
        print(f"fitted on all {len(val_y)} val frames: a={a:.4f} (T={1.0 / a:.4f}) "
              f"b={b:+.4f} in {calib_info['n_iter']} Newton steps")
    else:
        # The cross-check is the gate, not a footnote. If a 2-parameter remap does
        # not improve HELD-OUT likelihood, the model is already about as calibrated
        # as this validation set can tell, and fitting one anyway would fold pure
        # noise into every probability the tool displays. Identity == ship the
        # model's own probabilities unchanged.
        a, b = 1.0, 0.0
        calib_info = {"n_iter": 0, "skipped_by_crosscheck": True}
        print("NOT applying a calibration: the cross-check says a fitted remap does "
              "not improve held-out likelihood. The saved probabilities are the "
              "model's own. The operating-point table below is still valid -- it is "
              "measured empirically on whatever the model actually outputs.")

    cal_probs = apply_calibration(val_logits, a, b)
    ece_after, bins_after = expected_calibration_error(cal_probs, val_y)
    print(f"after : NLL={binary_nll(cal_probs, val_y):.4f}  ECE={ece_after:.4f}  "
          f"prob range [{cal_probs.min():.3f}, {cal_probs.max():.3f}]")

    fold_calibration_into_fc(net, a, b)
    folded_logits, folded_y = predict_logits(net, val_dl, device)
    folded_probs = sigmoid_np(folded_logits)

    # The fold must be an EXACT no-op relative to applying (a, b) numerically. If it
    # is not, the checkpoint would ship probabilities nobody validated, so fail loud.
    max_dev = float(np.max(np.abs(folded_probs - cal_probs)))
    if not np.array_equal(folded_y, val_y):
        raise SystemExit("val label order changed between forward passes; aborting.")
    if max_dev > 1e-4:
        raise SystemExit(
            f"Folding the calibration into fc changed the probabilities by "
            f"{max_dev:.2e}, which it must not. Refusing to save a checkpoint whose "
            f"outputs do not match the calibration that was validated."
        )
    from sklearn.metrics import roc_auc_score
    auc_raw = float(roc_auc_score(val_y, raw_probs))
    auc_folded = float(roc_auc_score(val_y, folded_probs))
    if abs(auc_raw - auc_folded) > 1e-6:
        raise SystemExit(
            f"AUC changed from {auc_raw:.8f} to {auc_folded:.8f} across calibration. "
            f"A positive-slope logit remap is strictly monotone and CANNOT change the "
            f"ranking; this means the fold is wrong."
        )
    print(f"fold verified: max |p_folded - p_calibrated| = {max_dev:.2e}, "
          f"AUC unchanged at {auc_folded:.6f}")
    print("=" * 72 + "\n")

    # ------------------------------------------------------------------ operating points
    print("=" * 72)
    print("(C3) OPERATING POINTS on the CALIBRATED probability")
    print("=" * 72)
    print(f"{'thr':>5s} {'acc':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'FPR':>6s} "
          f"{'tp':>5s} {'fp':>5s} {'tn':>5s} {'fn':>5s}")
    grid = [round(t, 2) for t in np.arange(0.30, 0.91, 0.05)]
    table = threshold_table(folded_probs, val_y, grid)
    for row in table:
        cm = row["confusion_matrix"]
        print(f"{row['threshold']:5.2f} {row['accuracy']:6.3f} {row['precision']:6.3f} "
              f"{row['recall']:6.3f} {row['f1']:6.3f} {row['false_positive_rate']:6.3f} "
              f"{cm['tp']:5d} {cm['fp']:5d} {cm['tn']:5d} {cm['fn']:5d}")

    recommended = {}
    for target in SPECIFICITY_TARGETS:
        t = pick_threshold_for_specificity(folded_probs, val_y, target)
        if t is None:
            continue
        ev = evaluate(folded_probs, val_y, t)
        recommended[f"specificity_{target:.2f}"] = {"threshold": float(t), **jsonable(ev)}
        print(f"  specificity >= {target:.2f}  -> threshold {t:.3f}: "
              f"recall={ev['recall']:.3f} precision={ev['precision']:.3f} "
              f"FPR={ev['false_positive_rate']:.3f} "
              f"(fp {ev['confusion_matrix']['fp']} of "
              f"{ev['confusion_matrix']['fp'] + ev['confusion_matrix']['tn']} negatives)")

    at_050 = evaluate(folded_probs, val_y, 0.50)
    at_085 = evaluate(folded_probs, val_y, 0.85)
    n_neg_val = at_050["confusion_matrix"]["fp"] + at_050["confusion_matrix"]["tn"]
    print(f"\ndetect.py's CURRENT display constants, on THIS saved model:")
    print(f"  LABEL_WORRYING = 0.50 -> {at_050['confusion_matrix']['fp']} false alarms "
          f"of {n_neg_val} negatives (FPR {at_050['false_positive_rate']:.3f}), "
          f"recall {at_050['recall']:.3f}   [v2 measured FPR 0.361]")
    print(f"  LABEL_DETECTED = 0.85 -> {at_085['confusion_matrix']['fp']} false alarms, "
          f"recall {at_085['recall']:.3f}, precision {at_085['precision']:.3f}")
    print("  NOTE: a probability above 0.85 is a claim the tool should only make when")
    print("  it is genuinely that sure. If very few frames now reach 0.85, that is the")
    print("  model telling the truth about its own skill, not a regression -- move")
    print("  detect.py's LABEL_DETECTED down to the specificity_0.90 threshold above")
    print("  rather than reverting anything. Raising LABEL_WORRYING to the")
    print("  specificity_0.85 threshold is the direct fix for the false-alarm rate,")
    print("  and the recall column above is exactly what it costs.")
    print("  (Changing those two constants is a local_tool edit, NOT done here.)")
    print("=" * 72 + "\n")

    # ------------------------------------------------------------------ final metrics
    best_metrics = evaluate(folded_probs, val_y)
    best_metrics["epoch"] = best_epoch
    best_metrics["val_loss"] = binary_nll(folded_probs, val_y)
    best_metrics["val_ece"] = float(ece_after)
    by_basin = breakdown(folded_probs, val_y, val_basins)
    by_source = breakdown(folded_probs, val_y, val_sources)
    # (E1) The finer split. digital_typhoon_au is the new data, so whether it scores
    # like digital_typhoon_wp (0.728 in v2, the source the model is best at) or like
    # tcir (0.598) is the single most informative number this run produces about
    # whether v4's added storms did what they were added to do.
    by_sub_source = breakdown(folded_probs, val_y, val_sub_sources)

    # (12) TTA is REPORT ONLY -- detect.py does a single un-rotated forward pass, so
    # the saved model and every headline number above are non-TTA. This says what a
    # four-rotation average in detect.py would be worth if it were added later.
    tta_logits, tta_y = predict_logits(net, val_dl, device, tta=True)
    tta_metrics = evaluate(sigmoid_np(tta_logits), tta_y)
    print(f"TTA probe (4-rotation average, NOT what detect.py does and NOT saved): "
          f"AUC {tta_metrics['auc']:.4f} vs {best_metrics['auc']:.4f} single-view "
          f"(delta {tta_metrics['auc'] - best_metrics['auc']:+.4f})")

    # ------------------------------------------------------------------ save
    config = jsonable({
        # -- required by detect.py / render_demo.py; do not rename --
        "temp_lo": TEMP_LO, "temp_hi": TEMP_HI, "img_size": IMG_SIZE,
        "norm_mean": NORM_MEAN, "norm_std": NORM_STD,
        "typhoon_grade": TYPHOON_GRADE,
        # -- provenance --
        "early_grade_max": DT_EARLY_GRADE_MAX,
        "typhoon_wind_kt": TYPHOON_WIND_KT,
        "tcir_early_vmax_kt": TCIR_EARLY_VMAX_KT,
        "label_rule": label_note(),
        # `sources` keeps v3's two coarse names on purpose -- it is provenance text
        # that existing consumers may already read. The AU mount is Digital Typhoon,
        # so it belongs under "digital_typhoon"; `sub_sources` carries the detail.
        "sources": ["digital_typhoon", "tcir"],
        "sub_sources": sorted(index["sub_source"].unique().tolist()),
        # Must be exactly EXPECTED_BASINS -- asserted in main() before this point.
        # local_tool/detect.py reads this to build its on-screen source label, so it
        # is a compatibility contract, not just metadata.
        "basins": sorted(index["basin"].unique().tolist()),
        "model_version": "v4.1",
        "frames_per_storm": FRAMES_PER_STORM,
        "label_smoothing": LABEL_SMOOTHING,
        "ema_decay": EMA_DECAY,
        "basin_balance_alpha": BASIN_BALANCE_ALPHA,
        # -- calibration, folded into fc; the raw sigmoid output IS the calibrated
        #    probability, so no consumer has to do anything with these numbers --
        "calibrated": bool(verdict["generalises"]),
        "calibration": {
            "method": "Platt scaling (slope+intercept) on val logits, folded into fc",
            "applied": bool(verdict["generalises"]),
            "crosscheck_verdict": verdict,
            "slope_a": a, "temperature": 1.0 / a, "bias_b": b,
            "fitted_on": "validation frames of the selected EMA checkpoint",
            "ece_before": ece_before, "ece_after": ece_after,
            "nll_before": binary_nll(raw_probs, val_y),
            "nll_after": binary_nll(folded_probs, val_y),
        },
        # -- what detect.py's two display constants SHOULD be, given this model --
        "recommended_display_thresholds": recommended,
    })
    val_metrics = jsonable(best_metrics)

    # detect.py and render_demo.py both json.dumps() these two. Prove it works here
    # rather than discovering it on the judge's laptop.
    json.dumps(config)
    json.dumps(val_metrics)

    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "arch": "resnet18_1ch",
            "pretrained_backbone": pretrained,
            "epoch": best_epoch,
            "val_metrics": val_metrics,
            "config": config,
            # Raw storm ids (Digital Typhoon '201618', TCIR '201103I'), NOT
            # source-prefixed -- render_demo.py looks a bare Digital Typhoon storm
            # id up in these lists for its honesty check.
            "train_storm_ids": sorted(train_keys),
            "val_storm_ids": sorted(val_keys),
            "val_metrics_by_basin": jsonable(by_basin),
            "val_metrics_by_source": jsonable(by_source),
            "val_metrics_by_sub_source": jsonable(by_sub_source),
            "val_metrics_tta_reference_only": jsonable(tta_metrics),
            "operating_points": jsonable(table),
        },
        CKPT_PATH,
    )

    summary = jsonable({
        "task": "early-stage frame -> P(storm later reaches >= 64 kt / typhoon strength)",
        "model_version": "v4.1",
        "label_rule": label_note(),
        "sources": {
            "digital_typhoon_wp": "bwandowando/digital-typhoon-dataset-western-pacific-wp "
                                  "(512x512 IR, WPAC, JMA best track)",
            "digital_typhoon_au": "bwandowando/digital-typhoon-dataset-au-1979-2024 "
                                  "(512x512 IR, Australian region, filed as basin SH; "
                                  "NEW IN v4)",
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
        "hyperparameters": {
            "epochs": EPOCHS, "warmup_epochs": WARMUP_EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "ema_decay": EMA_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "basin_balance_alpha": BASIN_BALANCE_ALPHA,
            "aug": {"brightness": AUG_BRIGHTNESS, "contrast": AUG_CONTRAST,
                    "zoom_log": AUG_ZOOM_LOG, "shift": AUG_SHIFT,
                    "noise_std": AUG_NOISE_STD, "rot90": True, "flips": False},
        },
        "dedup": dedup_audits,
        "au_vs_wp_null_test": au_wp_audit,
        "digital_typhoon_audit": dt_audits,
        "tcir_audit": tcir_audit,
        "field_of_view_diagnostic": fov_diag,
        "field_of_view_diagnostic_sh_only": fov_diag_sh,
        "best_epoch": best_epoch,
        "selection_audit": {
            "best_val_auc_ema": float(best_auc),
            "last5_auc_mean": float(np.mean(tail)) if tail else None,
            "last5_auc_sd": float(np.std(tail)) if tail else None,
        },
        "calibration": config["calibration"],
        "calibration_crosscheck_folds": cross,
        "calibration_crosscheck_verdict": verdict,
        # Reliability-diagram data: per confidence bin, how often the storms in it
        # actually intensified. This is the evidence behind any claim that a
        # displayed percentage is honest, so it is kept rather than reduced to ECE.
        "reliability_bins_before_calibration": bins_before,
        "reliability_bins_after_calibration": bins_after,
        "recommended_display_thresholds": recommended,
        "best_val_metrics": val_metrics,
        "best_val_metrics_by_basin": by_basin,
        "best_val_metrics_by_source": by_source,
        "best_val_metrics_by_sub_source": by_sub_source,
        "val_metrics_tta_reference_only": tta_metrics,
        "operating_points": table,
        "history": history,
    })
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BEST (overall, calibrated) ===")
    print(json.dumps(val_metrics, indent=2))
    print("\n=== BEST, by basin ===")
    print(json.dumps(jsonable(by_basin), indent=2))
    print("\n=== BEST, by source ===")
    print(json.dumps(jsonable(by_source), indent=2))
    print("\n=== BEST, by sub-source (digital_typhoon_au is v4's new data) ===")
    print(json.dumps(jsonable(by_sub_source), indent=2))

    # (A4) The measurement v4 exists to move. v3 ended at train_probe_auc 0.919 vs
    # val_auc 0.693 -- a 0.23 memorisation gap on 1,677 training storms. If that gap
    # has NOT narrowed with ~475 more storms, the next lever is regularisation or a
    # smaller backbone, not more data and not more epochs.
    last = history[-1]
    gap_v3 = 0.919 - 0.693
    gap_v4 = float(last["train_probe_auc"]) - float(last["auc"])
    print("\n=== GENERALISATION GAP (the number v4 exists to move) ===")
    print(f"  final epoch: train_probe_auc={last['train_probe_auc']:.3f}  "
          f"val_auc={last['auc']:.3f}  gap={gap_v4:+.3f}")
    print(f"  v3 measured gap {gap_v3:+.3f} on 1,677 train storms; v4 trained on "
          f"{len(train_keys)}.")
    print(f"  gap change vs v3: {gap_v4 - gap_v3:+.3f}  "
          f"({'NARROWED -- more storms helped' if gap_v4 < gap_v3 else 'NOT narrowed -- more data is no longer the lever; try regularisation or a smaller backbone'})")

    print("\nWrote", CKPT_PATH, METRICS_PATH, INDEX_PATH)


if __name__ == "__main__":
    main()
