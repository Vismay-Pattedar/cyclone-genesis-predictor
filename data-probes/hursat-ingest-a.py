"""HURSAT-B1 v06 ingest -- turn NOAA's global storm archive into training frames.

WHY THIS DATASET
----------------
Every measured limit on this project points at data, not modelling:
  * climatology alone scores 0.7048; the best model 0.7589, so ALL image skill
    ever earned is ~0.054 AUC
  * the weakest basins are the data-poor ones -- IO 0.628, EPAC 0.632,
    CPAC 0.644, SH 0.641 against WPAC's 0.749
  * the v5 power analysis concluded IO underperformance is a DATA SCARCITY
    problem: detecting a real +0.03 there needs ~177 IO storms

HURSAT-B1 is global, 1978-2015, ~100-120 storms/year (~4,000 total) against the
current 2,037. It is storm-centred at 8 km on a 301x301 grid (~2,400 km box,
comparable to Digital Typhoon's 512px/~2,560 km), 3-hourly.

VERIFIED BEFORE WRITING THIS (on a real downloaded file):
  IRWIN * 0.01 + 200.0 -> Kelvin, units "Kelvin",
  long_name "Brightness Temperature in (10.2-11.4 um)"
  measured range 186.3-297.0 K -- inside the existing [190,300] clip, so
  kelvin_to_display_u8 applies UNCHANGED.
  WindSpd is in the file, so labels need no external join.
  The filename carries the IBTrACS SID, so dedup against Digital Typhoon /
  TCIR is an EXACT key match rather than fuzzy position matching.

WHAT IS DELIBERATELY NOT READ
-----------------------------
The files also ship odt84 and tnum_icen/acen/vcen -- Dvorak-derived intensity
ESTIMATES. Those are the answer wearing a different hat, exactly like the banned
wind/pressure/wind-radius fields. Only IRWIN is read. CentLat/CentLon/time are
kept because a forecaster knows them at forecast time; WindSpd is used ONLY to
build the label and the early-stage cut, never as a model input.

THE DECAY-TAIL TRAP (v5 Finding 2, do not reintroduce)
------------------------------------------------------
"Early stage" must NOT be "wind <= 33kt anywhere in the track" -- a track runs
2->3->4->5->4->3->2, so that also matches the POST-PEAK decay tail. The
contamination is asymmetric (storms that reach typhoon usually decay through
higher grades and never return), so ragged post-peak cloud becomes a learnable
shortcut for the negative class that does not exist at inference. Each track is
truncated at its FIRST crossing out of the early stage, same as sweep.py.

OUTPUT
------
frames_<range>.npy  uint8 (N, 320, 320)   -- PRELOAD_SIZE, matching sweep.py
index_<range>.csv   one row per frame, with sid / basin / label / time / wind
Frames are kept from track start through the first early-stage exit, so the
6h/12h tendency neighbours exist within the stored history.

Env knobs: HURSAT_YEARS="1978-1995"  HURSAT_SKIP_WP=1  HURSAT_MAX_FRAMES=24
"""

import csv
import io
import os
import re
import sys
import tarfile
import time
import urllib.request

import numpy as np

BASE = "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06"

# Must match sweep.py exactly -- pretraining/ingest in a different input domain
# than training throws away the compatibility this whole exercise depends on.
TEMP_LO, TEMP_HI = 190.0, 300.0
PRELOAD_SIZE = 320

EARLY_MAX_KT = 33.0        # early stage = max sustained wind <= 33 kt
POSITIVE_KT = 64.0         # storm is positive if it EVER reaches 64 kt
MAX_FRAMES = int(os.environ.get("HURSAT_MAX_FRAMES", 24))
SKIP_WP = os.environ.get("HURSAT_SKIP_WP", "1") == "1"
YEARS = os.environ.get("HURSAT_YEARS", "1978-1990")

# SID looks like 2005013N05153 -> year 2005, day 013, N, lat 05, lon 153
SID_RE = re.compile(r"(\d{4})(\d{3})([NS])(\d{2})(\d{3})")
FN_RE = re.compile(r"HURSAT_b1_v06_(\d{7}[NS]\d{5})_([^_]*)_c\d+\.tar\.gz")


def log(m):
    print(f"[hursat] {m}", flush=True)


def basin_of(lat, lon):
    """Coarse basin label from storm genesis position.

    Only used for reporting and for the optional WP skip -- never as a model
    input. Longitudes are 0-360 as encoded in the SID.
    """
    if lat < 0:
        return "SH"
    if 30 <= lon < 100:
        return "IO"
    if 100 <= lon < 180:
        return "WPAC"
    if 180 <= lon < 220:
        return "CPAC"
    if 220 <= lon < 290:
        return "EPAC"
    return "ATLN"


def parse_sid(sid):
    m = SID_RE.match(sid)
    if not m:
        return None
    y, _d, hemi, lat, lon = m.groups()
    lat = float(lat) * (1 if hemi == "N" else -1)
    return int(y), lat, float(lon)


def fetch(url, tries=3, timeout=180):
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as exc:
            last = exc
            time.sleep(2 * (i + 1))
    raise last


def list_year(year):
    html = fetch(f"{BASE}/{year}/", timeout=120).decode("utf-8", "replace")
    return re.findall(r'href="(HURSAT_b1_v06_[^"]+\.tar\.gz)"', html)


def kelvin_to_display_u8(arr):
    """Byte-identical to sweep.py. Do not diverge."""
    arr = np.clip(arr, TEMP_LO, TEMP_HI)
    arr = 1.0 - (arr - TEMP_LO) / (TEMP_HI - TEMP_LO)
    return (arr * 255.0).astype(np.uint8)


def resize_u8(u8, size=PRELOAD_SIZE):
    import torch
    t = torch.from_numpy(u8)[None, None].float()
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear",
                                        align_corners=False)
    return t[0, 0].round().clamp(0, 255).to(torch.uint8).numpy()


def read_member(tf, member):
    """Read one netCDF from the tarball -> (time_h, wind_kt, lat, lon, u8 frame)."""
    import h5py
    data = tf.extractfile(member).read()
    with h5py.File(io.BytesIO(data), "r") as h:
        if "IRWIN" not in h or "WindSpd" not in h:
            return None
        ir = h["IRWIN"]
        raw = np.asarray(ir[0], dtype=np.float64)
        sf = float(np.ravel(ir.attrs.get("scale_factor", [1.0]))[0])
        ao = float(np.ravel(ir.attrs.get("add_offset", [0.0]))[0])
        fill = ir.attrs.get("_FillValue", None)
        k = raw * sf + ao
        if fill is not None:
            bad = raw == float(np.ravel(fill)[0])
            if bad.all():
                return None
            # Missing pixels become warm (clear-sky end) rather than cold, so a
            # data gap is never mistaken for deep convection.
            k[bad] = TEMP_HI
        wind = float(np.ravel(h["WindSpd"][:])[0])
        lat = float(np.ravel(h["CentLat"][:])[0])
        lon = float(np.ravel(h["CentLon"][:])[0])
    # time from the filename: SID.NAME.YYYY.MM.DD.HHMM....nc
    parts = os.path.basename(member.name).split(".")
    try:
        import datetime
        y, mo, d, hm = int(parts[2]), int(parts[3]), int(parts[4]), parts[5]
        t = datetime.datetime(y, mo, d, int(hm[:2]), int(hm[2:])).timestamp() / 3600.0
    except Exception:
        return None
    if not np.isfinite(wind):
        return None
    return t, wind, lat, lon, resize_u8(kelvin_to_display_u8(k))


def process_storm(url):
    """Download one storm, return (rows, frames) for its early-stage window."""
    blob = fetch(url)
    recs = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.name.endswith(".nc"):
                continue
            try:
                r = read_member(tf, m)
            except Exception:
                continue
            if r is not None:
                recs.append(r)
    if len(recs) < 3:
        return [], []

    recs.sort(key=lambda r: r[0])
    winds = np.array([r[1] for r in recs])

    # Label: did this storm EVER reach 64 kt, anywhere in its life?
    label = int(np.nanmax(winds) >= POSITIVE_KT)

    # Early stage, truncated at the FIRST exit -- never the decay tail.
    early = winds <= EARLY_MAX_KT
    if not early[0]:
        return [], []          # already past early stage on its first frame
    cut = len(recs)
    for i in range(len(recs)):
        if not early[i]:
            cut = i
            break
    if cut < 2:
        return [], []

    keep = recs[:cut][-MAX_FRAMES:]   # most recent early frames + their lookback
    return keep, label


def main():
    a, b = YEARS.split("-")
    years = list(range(int(a), int(b) + 1))
    log(f"years {years[0]}-{years[-1]}  skip_wp={SKIP_WP}  max_frames={MAX_FRAMES}")

    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    tag = f"{years[0]}_{years[-1]}"
    frames, rows = [], []
    n_storm = n_skip = n_fail = 0
    t0 = time.time()

    for yr in years:
        try:
            files = list_year(yr)
        except Exception as exc:
            log(f"year {yr}: LISTING FAILED {exc}")
            continue
        log(f"year {yr}: {len(files)} storms")
        for fn in files:
            m = FN_RE.match(fn)
            if not m:
                continue
            sid, name = m.group(1), m.group(2)
            p = parse_sid(sid)
            if p is None:
                continue
            _y, glat, glon = p
            basin = basin_of(glat, glon)
            # WP is already covered by Digital Typhoon at 512px native, so it is
            # skipped by default to spend bandwidth on the data-poor basins.
            if SKIP_WP and basin == "WPAC":
                n_skip += 1
                continue
            try:
                keep, label = process_storm(f"{BASE}/{yr}/{fn}")
            except Exception as exc:
                n_fail += 1
                if n_fail <= 10:
                    log(f"  FAIL {sid}: {type(exc).__name__}: {exc}")
                continue
            if not keep:
                continue
            for fi, (t, wind, lat, lon, u8) in enumerate(keep):
                rows.append({
                    "storm_key": sid, "source": "hursat", "basin": basin,
                    "label": label, "name": name, "t_h": f"{t:.3f}",
                    "wind_kt": f"{wind:.1f}", "lat": f"{lat:.3f}",
                    "lon": f"{lon:.3f}", "frame_idx": fi,
                    "arr_idx": len(frames),
                })
                frames.append(u8)
            n_storm += 1
            if n_storm % 25 == 0:
                el = time.time() - t0
                log(f"  {n_storm} storms, {len(frames)} frames, "
                    f"{el/60:.1f} min, {el/max(n_storm,1):.1f}s/storm")

    if not frames:
        raise SystemExit("no frames ingested -- check network/filters")

    arr = np.stack(frames)
    np.save(os.path.join(out_dir, f"frames_{tag}.npy"), arr)
    with open(os.path.join(out_dir, f"index_{tag}.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sids = {r["storm_key"] for r in rows}
    pos = {r["storm_key"] for r in rows if r["label"] == 1}
    log(f"DONE {tag}: {len(sids)} storms ({len(pos)} positive, "
        f"{len(pos)/max(len(sids),1):.1%}), {len(arr):,} frames, "
        f"{arr.nbytes/1e9:.2f} GB, skipped_wp={n_skip}, failed={n_fail}")
    from collections import Counter
    log("by basin: " + str(Counter(r["basin"] for r in rows
                                   if r["frame_idx"] == 0).most_common()))
    log(f"elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
