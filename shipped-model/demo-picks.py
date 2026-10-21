"""
Cyclone Genesis Predictor -- pick real, in-distribution demo frames for the susune web app,
using the SAME data sources (Digital Typhoon) the models were trained on,
selected against REAL recorded storm outcomes rather than guessed from
random web images. CPU only, no GPU needed -- just picks frames and saves
them as real PNG files using the exact kelvin_to_display_u8 conversion every
checkpoint in this project expects.

Categories picked, each anchored to a real, verifiable number:
  high_dev     -- early-stage frame from a storm that went on to a strong
                  typhoon (max wind >= 100kt). Should read HIGH on Model 3.
  low_dev      -- early-stage frame from a storm that never got past a weak
                  tropical storm (max wind < 40kt). Should read LOW.
  medium_dev   -- early-stage frame from a storm that topped out right near
                  the 64kt threshold (55-64kt). Genuinely ambiguous case.
  weak_mature  -- frame from right at a storm's first crossing into "mature"
                  (grade just above DT_EARLY_GRADE_MAX) -- technically past
                  the early/mature boundary Gate 2 was trained on, but not
                  yet visually dramatic. The "doesn't obviously look like a
                  cyclone, but the pipeline calls it one" case.
  strong_mature-- a frame from deep in a storm's typhoon-grade life (grade 5,
                  well past first_cross) -- the obvious, dramatic eye case,
                  for comparison against weak_mature.

Each category picks ONE representative storm (deterministic, seeded) and
saves ONE frame, named with the real storm id and the real recorded wind at
that frame, so every image comes with a verifiable ground-truth caption.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

INPUT_ROOT = os.environ.get("CGP_INPUT_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("CGP_OUT_DIR", "/kaggle/working")
SEED = 1337
DT_EARLY_GRADE_MAX = 2
TEMP_LO, TEMP_HI = 190.0, 300.0


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


def kelvin_to_display_u8(arr):
    arr = np.clip(arr, TEMP_LO, TEMP_HI)
    arr = 1.0 - (arr - TEMP_LO) / (TEMP_HI - TEMP_LO)
    return (arr * 255.0).astype(np.uint8)


def load_dt_frame_u8(h5_path):
    import h5py
    with h5py.File(h5_path, "r") as f:
        arr = np.asarray(f["Infrared"], dtype=np.float32)
    return kelvin_to_display_u8(arr)


def save_frame(h5_path, out_path):
    u8 = load_dt_frame_u8(h5_path)
    Image.fromarray(u8, mode="L").save(out_path)
    return u8.shape


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dt_root = find_dt_root()
    if dt_root is None:
        raise SystemExit("No Digital Typhoon mount found.")
    meta_dir, image_dir = resolve_dt_dirs(dt_root)
    print("meta_dir:", meta_dir, "image_dir:", image_dir)

    csvs = sorted(f for f in os.listdir(meta_dir) if f.endswith(".csv") and f[:-4].isdigit())
    print(f"Found {len(csvs)} storm CSVs")

    storms = []
    for name in csvs:
        storm_id = name[:-4]
        try:
            df = pd.read_csv(os.path.join(meta_dir, name))
        except Exception:
            continue
        need = {"grade", "file_1", "wind", "year", "month", "day"}
        if need - set(df.columns):
            continue
        year = int(storm_id[:4])
        if year < 1990:  # stick to reasonably modern, well-tracked storms
            continue
        grade = pd.to_numeric(df["grade"], errors="coerce")
        wind = pd.to_numeric(df["wind"], errors="coerce").fillna(0.0)
        fname = df["file_1"].fillna("").astype(str).str.strip()
        valid = (fname.to_numpy() != "")
        if valid.sum() < 8:
            continue
        crossed = (grade > DT_EARLY_GRADE_MAX).fillna(False).to_numpy()
        first_cross = int(np.argmax(crossed)) if crossed.any() else len(df)
        storms.append({
            "storm_id": storm_id, "df": df, "grade": grade.to_numpy(),
            "wind": wind.to_numpy(), "valid": valid, "first_cross": first_cross,
            "max_wind": float(wind.to_numpy()[valid].max()) if valid.any() else 0.0,
            "max_grade": float(np.nanmax(grade.to_numpy())) if len(grade) else 0.0,
            "n": len(df),
        })
    print(f"Usable storms: {len(storms)}")

    rng = np.random.RandomState(SEED)

    def pick_storm(pred, label):
        cands = [s for s in storms if pred(s)]
        print(f"  [{label}] {len(cands)} candidate storms")
        if not cands:
            return None
        cands.sort(key=lambda s: s["storm_id"])
        return cands[rng.randint(len(cands))]

    picks = []

    # HIGH: went on to a strong typhoon (>=100kt), needs a real early portion.
    s = pick_storm(lambda s: s["max_wind"] >= 100 and s["first_cross"] >= 6, "high_dev")
    if s:
        idx = int(s["first_cross"] * 0.6)  # well into the early portion, not frame 0
        idx = max(0, min(idx, s["first_cross"] - 1))
        picks.append(("high_dev", s, idx))

    # LOW: never got past a weak tropical storm.
    s = pick_storm(lambda s: s["max_wind"] < 40 and s["valid"].sum() >= 6, "low_dev")
    if s:
        va = np.where(s["valid"])[0]
        idx = int(va[len(va) // 2])
        picks.append(("low_dev", s, idx))

    # MEDIUM: topped out right near the 64kt threshold.
    s = pick_storm(lambda s: 55 <= s["max_wind"] <= 64 and s["first_cross"] >= 6, "medium_dev")
    if s:
        idx = int(s["first_cross"] * 0.7)
        idx = max(0, min(idx, s["first_cross"] - 1))
        picks.append(("medium_dev", s, idx))

    # WEAK_MATURE: right at the first crossing into "mature".
    s = pick_storm(lambda s: 6 <= s["first_cross"] < s["n"] - 1, "weak_mature")
    if s:
        picks.append(("weak_mature", s, s["first_cross"]))

    # STRONG_MATURE: deep into typhoon grade, obvious eye territory.
    s = pick_storm(lambda s: s["max_grade"] >= 5 and s["first_cross"] < s["n"] - 10, "strong_mature")
    if s:
        peak_idx = int(np.nanargmax(s["wind"]))
        picks.append(("strong_mature", s, peak_idx))

    rows = []
    for category, s, idx in picks:
        rec = s["df"].iloc[idx]
        fname = str(rec["file_1"]).strip()
        h5_path = os.path.join(image_dir, s["storm_id"], fname)
        if not os.path.exists(h5_path):
            print(f"  !! missing file for {category}: {h5_path}")
            continue
        out_name = f"{category}__{s['storm_id']}__frame{idx}__wind{int(s['wind'][idx])}kt.png"
        out_path = os.path.join(OUT_DIR, out_name)
        try:
            shape = save_frame(h5_path, out_path)
        except Exception as exc:
            print(f"  !! failed to save {category}: {exc}")
            continue
        row = {
            "category": category, "storm_id": s["storm_id"], "frame_index": idx,
            "frame_wind_kt": float(s["wind"][idx]), "frame_grade": float(s["grade"][idx]),
            "storm_max_wind_kt": s["max_wind"], "storm_max_grade": s["max_grade"],
            "storm_first_cross_index": s["first_cross"], "storm_n_frames": s["n"],
            "png_shape": shape, "filename": out_name,
        }
        rows.append(row)
        print(f"  saved {category}: {out_name}  (frame wind={s['wind'][idx]:.0f}kt, "
              f"storm max wind={s['max_wind']:.0f}kt, grade at frame={s['grade'][idx]:.0f})")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "picks_manifest.csv"), index=False)
    print("\nWrote picks_manifest.csv")
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
