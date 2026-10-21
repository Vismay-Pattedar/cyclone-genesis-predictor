import os
import numpy as np
import pandas as pd
import h5py

ROOT = None
for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
    for fn in filenames:
        if fn.lower().endswith(".h5"):
            ROOT = os.path.join(dirpath, fn)
            break
    if ROOT:
        break
print("H5 file:", ROOT)

with h5py.File(ROOT, "r") as f:
    print("Top-level keys:", list(f.keys()))
    for k in f.keys():
        obj = f[k]
        if isinstance(obj, h5py.Dataset):
            print(f"  dataset '{k}': shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"  group '{k}': subkeys={list(obj.keys())[:10]}")

# Try the documented TCIR schema directly.
try:
    info = pd.read_hdf(ROOT, key="info", mode="r")
    print("\n'info' loaded as DataFrame:")
    print("  shape:", info.shape)
    print("  columns:", list(info.columns))
    print(info.head(10).to_string())

    # Look for any Indian Ocean storms specifically (relevant to Cyclone Genesis Predictor / IMD).
    for col in info.columns:
        if "region" in col.lower() or "basin" in col.lower() or "data_set" in col.lower():
            print(f"\nUnique values in '{col}':", info[col].unique()[:20])
except Exception as exc:
    print("Could not read 'info' as DataFrame:", type(exc).__name__, exc)

# The Kaggle mirror uses key 'images' (official TCIR docs say 'matrix') -- try both.
with h5py.File(ROOT, "r") as f:
    key = "images" if "images" in f else ("matrix" if "matrix" in f else None)
    if key is None:
        print("\nNo 'images' or 'matrix' key found; top-level keys were:", list(f.keys()))
    else:
        mat = f[key]
        print(f"\n'{key}' shape:", mat.shape, "dtype:", mat.dtype)

        info = pd.read_hdf(ROOT, key="info", mode="r")
        io_idx = info.index[info["region"] == "IO"]
        print(f"IO (Indian Ocean) region: {len(io_idx)} frames")
        n_storms_io = info.loc[io_idx, "ID"].nunique() if len(io_idx) else 0
        print(f"IO unique storm IDs: {n_storms_io}")

        # Sample one generic frame (index 0) and, if any exist, one real IO frame.
        sample_idxs = [0]
        if len(io_idx):
            sample_idxs.append(int(io_idx[len(io_idx) // 2]))

        for si in sample_idxs:
            frame = np.asarray(mat[si])
            row = info.iloc[si]
            print(f"\nframe idx {si}: region={row['region']} ID={row['ID']} time={row['time']} "
                  f"Vmax={row['Vmax']} MSLP={row['MSLP']} shape={frame.shape}")
            for c in range(frame.shape[-1]):
                ch = frame[..., c]
                finite = ch[np.isfinite(ch)]
                if finite.size:
                    print(f"  channel {c}: min={finite.min():.2f} max={finite.max():.2f} "
                          f"mean={finite.mean():.2f} n_finite={finite.size}/{ch.size}")
                else:
                    print(f"  channel {c}: all non-finite")

            out_name = f"/kaggle/working/sample_{si}_{row['region']}.npy"
            np.save(out_name, frame)
            print("  saved", out_name)

        print("\n=== WPAC ID format check (for cross-dataset dedup) ===")
        wpac = info[info["region"] == "WPAC"]
        print(f"WPAC rows: {len(wpac)}, unique WPAC storm IDs: {wpac['ID'].nunique()}")
        print("Sample WPAC IDs:", sorted(wpac["ID"].unique())[:30])
        print("\n=== other basins' ID formats for comparison ===")
        for b in ["ATLN", "CPAC", "EPAC", "IO", "SH"]:
            sub = info[info["region"] == b]
            print(f"{b}: {sub['ID'].nunique()} storms, sample IDs:", sorted(sub["ID"].unique())[:5])
