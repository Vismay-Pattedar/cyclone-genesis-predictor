import os
import glob
import pandas as pd

def walk_shallow(base, depth=4):
    if depth == 0 or not os.path.isdir(base):
        return
    entries = os.listdir(base)
    print(f"{base} -> {entries[:15]}")
    for e in entries[:5]:
        walk_shallow(os.path.join(base, e), depth - 1)

print("=== /kaggle/input tree ===")
walk_shallow("/kaggle/input", depth=4)

# Find any directory anywhere under /kaggle/input whose name contains "typhoon".
ROOT = None
for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
    for d in dirnames:
        if "typhoon" in d.lower():
            ROOT = os.path.join(dirpath, d)
            break
    if ROOT:
        break
if not ROOT:
    raise SystemExit("No typhoon dataset mounted - check dataset_sources in kernel-metadata.json")
print("Using ROOT =", ROOT)
for d in os.listdir(ROOT):
    p = os.path.join(ROOT, d)
    if os.path.isdir(p):
        print(f"--- {d} ---")
        print(os.listdir(p)[:10])

# Locate the metadata dir specifically -- do NOT recursively glob ROOT itself,
# that would also crawl the ~90GB image tree and be extremely slow on a mounted dataset.
meta_root = os.path.join(ROOT, "metadata")
if os.path.isdir(os.path.join(meta_root, "metadata")):
    meta_root = os.path.join(meta_root, "metadata")
print("meta_root =", meta_root)
meta_files = glob.glob(os.path.join(meta_root, "*.csv"))
meta_files = [f for f in meta_files if os.path.basename(f)[:-4].isdigit()]
print(f"\nFound {len(meta_files)} per-storm metadata CSVs")

records = []
for fp in sorted(meta_files):
    storm_id = os.path.basename(fp).replace(".csv", "")
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        print("skip", fp, e)
        continue
    if "grade" not in df.columns:
        continue
    max_grade = df["grade"].max()
    max_wind = df["wind"].max() if "wind" in df.columns else float("nan")
    has_img = df["file_1"].notna() & (df["file_1"] != "")
    imaged = df[has_img]
    early = imaged[imaged["grade"] <= 2]  # TD-stage frames, before real organization
    has_radius = pd.to_numeric(df.get("long30", 0), errors="coerce").fillna(0) > 0
    records.append({
        "id": storm_id,
        "n_rows": len(df),
        "n_imaged": len(imaged),
        "n_early_imaged": len(early),
        "max_grade": max_grade,
        "max_wind": max_wind,
        "reaches_typhoon": int(max_grade >= 5),
        "n_rows_with_wind_radius": int(has_radius.sum()),
    })

survey = pd.DataFrame(records)
survey.to_csv("/kaggle/working/storm_survey.csv", index=False)

print("\n=== SUMMARY ===")
print("Total storms:", len(survey))
print("Storms with >=1 early (TD-stage) imaged frame:", (survey["n_early_imaged"] > 0).sum())
print("Label balance (reaches_typhoon):")
print(survey["reaches_typhoon"].value_counts())
print("Mean early_imaged frames per storm:", survey["n_early_imaged"].mean())
print("Total rows with real wind-radius data across all storms:", survey["n_rows_with_wind_radius"].sum())
print("Total imaged rows across all storms:", survey["n_imaged"].sum())
