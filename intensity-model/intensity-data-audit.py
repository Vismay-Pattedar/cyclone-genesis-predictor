"""
Cyclone Genesis Predictor -- Cyclone INTENSITY model, data-audit pass ONLY.

Companion to train_kernel_v5_final/train.py, but the mirror-image frame
selection: v5's early-stage model trains on frames strictly BEFORE a storm
first crosses out of "early stage" (grade<=2 DT / Vmax<=33kt TCIR). This
model is for a DIFFERENT question -- "how intense is an ALREADY-CONFIRMED
cyclone in this frame" -- so it must train on the complementary set: frames
AT OR AFTER that same crossing point. No overlap with the early-stage
model's training data by construction (same crossing index, opposite side).

This script does NOT train anything. It reuses v5_final's dataset discovery
and cross-dataset dedup (both already correctness-proven -- see that file's
comments) unchanged, builds the "mature" index, and prints exactly the
numbers needed before committing any real compute: how many frames/storms
are available, and what the wind-speed target actually looks like. Full
training is a separate follow-up script once these numbers are sane.

Runs as a plain Kaggle script kernel. CPU is enough for this pass.
"""

import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

INPUT_ROOT = os.environ.get("CGP_INPUT_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("CGP_OUT_DIR", "/kaggle/working")

TYPHOON_GRADE = 5
DT_EARLY_GRADE_MAX = 2          # verbatim from v5 -- "mature" is grade > this
TCIR_EARLY_VMAX_KT = 33.0       # verbatim from v5 -- "mature" is Vmax > this

DEDUP_MATCH_DEG = 1.5
DEDUP_MIN_MATCH_POINTS = 4
DEDUP_USE_ID_KEY_UNION = True
TCIR_ID_PATTERN_MIN_MATCH = 0.90
TCIR_ID_RE = re.compile(r"^(\d{4})(\d{2})([A-Za-z])$")

# IMD wind-speed category bins (kt, 3-minute sustained convention used by
# IMD for the North Indian Ocean; applied here as a generic display banding,
# not a basin-specific claim).
CATEGORY_BINS = [
    (0, 34, "Depression"),
    (34, 48, "Cyclonic Storm"),
    (48, 64, "Severe Cyclonic Storm"),
    (64, 90, "Very Severe Cyclonic Storm"),
    (90, 120, "Extremely Severe Cyclonic Storm"),
    (120, 999, "Super Cyclonic Storm"),
]


def category_of(kt):
    for lo, hi, name in CATEGORY_BINS:
        if lo <= kt < hi:
            return name
    return CATEGORY_BINS[-1][2]


def day_of_year(y, m, d):
    try:
        import datetime
        return int(datetime.date(int(y), int(m), min(28, int(d))).timetuple().tm_yday) \
            + (int(d) - min(28, int(d)))
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Dataset discovery -- verbatim from v5_final/train.py (already proven
# correct: picks the Western Pacific Digital Typhoon mount explicitly rather
# than the first "typhoon"-named dir, and prunes the ~89GB image tree during
# the TCIR search instead of walking into it).
# ---------------------------------------------------------------------------
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
    wp = [c for c in cands
          if "western-pacific" in c.lower() or c.lower().rstrip("/\\").endswith("wp")]
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
            if fn.lower().endswith((".h5", ".hdf5")) and (
                "tcsa" in fn.lower() or "tcir" in fn.lower()
            ):
                candidates.append(os.path.join(dirpath, fn))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# MATURE-frame indexing (mirror of build_dt_index / build_tcir_index in
# v5_final/train.py, but selecting the COMPLEMENT of the early-stage set,
# and keeping the continuous wind value as the target instead of a >=64kt
# threshold).
# ---------------------------------------------------------------------------
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

        # track, for dedup -- identical to v5_final
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

        for p in mature_pos:
            rec = df.iloc[int(p)]
            h5_path = os.path.join(image_dir, storm_id, str(rec["file_1"]).strip())
            if not os.path.exists(h5_path):
                audit["n_missing_h5"] += 1
                continue
            rows.append({
                "storm_key": storm_id, "source": "digital_typhoon", "basin": "WPAC",
                "wind_kt": float(wind.iloc[int(p)]), "h5_path": h5_path, "frame_ref": -1,
                "lat": float(pd.to_numeric(rec["lat"], errors="coerce")),
                "lon": float(pd.to_numeric(rec["lng"], errors="coerce")),
                "doy": day_of_year(rec["year"], rec["month"], rec["day"]),
            })

        if (i + 1) % 250 == 0:
            print(f"  [A] indexed {i + 1}/{len(csvs)} storms, {len(rows)} mature frames so far")

    print(f"[A] storms dropped -- unreadable={audit['n_unreadable']} "
          f"bad_columns={audit['n_bad_columns']} pre_1977={audit['n_pre_1977_dropped']} "
          f"never_matures={audit['n_never_matures']} no_mature_frame={audit['n_no_mature_frame']}")
    print(f"[A] frames listed but missing on disk: {audit['n_missing_h5']}")
    return pd.DataFrame(rows), tracks, audit


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

        for _, rec in mature.iterrows():
            ts = rec["time"]
            ts = int(ts) if np.isfinite(ts) else -1
            rows.append({
                "storm_key": storm_id, "source": "tcir", "basin": basin,
                "wind_kt": float(rec["vmax"]), "h5_path": "", "frame_ref": int(rec["frame_ref"]),
                "lat": float(rec["lat"]), "lon": float(rec["lon"]),
                "doy": (day_of_year(ts // 1000000, (ts // 10000) % 100, (ts // 100) % 100)
                        if ts > 0 else -1),
            })

    print(f"[B] storms dropped -- never_matures={audit['n_never_matures']} "
          f"no_mature_frame={audit['n_no_mature_frame']}")
    return pd.DataFrame(rows), work, audit


# ---------------------------------------------------------------------------
# Dedup -- verbatim logic from v5_final/train.py's dedup_wpac(), trimmed to
# what this audit needs (no printing of full drop lists).
# ---------------------------------------------------------------------------
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
    return drop


def main():
    t0 = time.time()

    dt_root = find_dt_root()
    if dt_root is None:
        raise SystemExit("No Digital Typhoon mount found -- check dataset_sources.")
    meta_dir, image_dir = resolve_dt_dirs(dt_root)
    print(f"Digital Typhoon root: {dt_root}")

    dt_index, dt_tracks, dt_audit = build_dt_index_mature(meta_dir, image_dir)

    tcir_h5 = find_tcir_h5()
    if tcir_h5 is None:
        raise SystemExit("No TCIR/TCSA .h5 found -- check dataset_sources.")
    print(f"TCIR file: {tcir_h5} ({os.path.getsize(tcir_h5) / 1e9:.1f} GB)")

    # Proven method from v5_final/train.py -- TCIR's metadata is a pandas HDF5
    # table, not a raw h5py structured array/group (an earlier draft of this
    # script guessed the latter; verified against the working script instead
    # of trusting that guess).
    info = pd.read_hdf(tcir_h5, key="info", mode="r")
    need_b = {"region", "ID", "lon", "lat", "time", "Vmax"}
    missing_b = need_b - set(info.columns)
    if missing_b:
        raise SystemExit(f"TCIR 'info' is missing expected columns {sorted(missing_b)}; "
                          f"got {list(info.columns)}")
    info = info.reset_index(drop=True)

    tcir_index, tcir_work, tcir_audit = build_tcir_index_mature(info)

    drop = dedup_wpac(dt_index, tcir_index, tcir_work, dt_tracks)
    tcir_index = tcir_index[~tcir_index["storm_key"].isin(drop)]

    full = pd.concat([dt_index, tcir_index], ignore_index=True)
    full["category"] = full["wind_kt"].apply(category_of)

    print("\n" + "=" * 72)
    print("MATURE-FRAME AUDIT (no training run, counts only)")
    print("=" * 72)
    print(f"Total mature frames: {len(full)}")
    print(f"Total distinct storms contributing: {full['storm_key'].nunique()}")
    print(f"\nFrames by source:\n{full['source'].value_counts()}")
    print(f"\nFrames by basin:\n{full['basin'].value_counts()}")
    print(f"\nwind_kt distribution:\n{full['wind_kt'].describe()}")
    print(f"\nFrames by IMD category bin:\n{full['category'].value_counts()}")
    print(f"\nStorms by basin (unique):")
    print(full.groupby('basin')['storm_key'].nunique())

    full.to_csv(os.path.join(OUT_DIR, "mature_index_audit.csv"), index=False)
    print(f"\nWrote {os.path.join(OUT_DIR, 'mature_index_audit.csv')}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
