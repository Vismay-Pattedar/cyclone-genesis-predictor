"""
Cyclone Genesis Predictor -- Early-intensification prediction for Western Pacific tropical cyclones.

TASK FRAMING (do not confuse this with cyclone *detection*):
    Input : ONE early-stage infrared satellite frame (grade <= 2, i.e. the system is
            already tracked but still a disorganised depression -- it does not yet
            visually look like a cyclone).
    Output: P(this storm will later reach grade >= 5 / typhoon strength).
    Label : built from each storm's own FUTURE best-track history. The model never
            sees future frames, and never sees grade / pressure / wind at inference
            time -- only the raw brightness-temperature image.

Data: NII Digital Typhoon (Western Pacific), Kaggle mirror
      bwandowando/digital-typhoon-dataset-western-pacific-wp

Runs as a plain Kaggle *script* kernel with GPU enabled.
"""

import json
import os
import time

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

# Cap frames per storm. There are ~45 early-stage imaged frames per storm on
# average; using all of them would give ~50k highly-redundant images and a very
# long kernel. 4 evenly-spaced frames across the early stage keeps the whole run
# comfortably under an hour while still covering the storm's pre-organisation life.
FRAMES_PER_STORM = 4

EARLY_GRADE_MAX = 2   # grade <= 2 == depression / pre-organisation stage
TYPHOON_GRADE = 5     # grade >= 5 == typhoon strength (positive class definition)

# Infrared brightness-temperature display/normalisation window, in Kelvin.
# Same window poc_render.py uses, so the demo renderer's visible frame and the
# model's input are literally the same array.
TEMP_LO, TEMP_HI = 190.0, 300.0

IMG_SIZE = 224        # resnet18 native input size
# ImageNet mean/std collapsed to one channel (mean of the 3 per-channel values).
# Correct partner for the "average the RGB conv1 weights" trick used below.
NORM_MEAN, NORM_STD = 0.449, 0.226

VAL_FRACTION = 0.2
EPOCHS = 8
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4

OUT_DIR = "/kaggle/working"
CKPT_PATH = os.path.join(OUT_DIR, "model_best.pt")
METRICS_PATH = os.path.join(OUT_DIR, "metrics.json")
INDEX_PATH = os.path.join(OUT_DIR, "dataset_index.csv")


# ----------------------------------------------------------------------------
# Dataset discovery
# ----------------------------------------------------------------------------
def find_root():
    """Locate the Digital Typhoon mount.

    The exact /kaggle/input prefix varies between kernel runs, so walk instead of
    hardcoding. Break out as soon as a 'typhoon' directory is found -- the image
    tree is ~89GB and must never be walked exhaustively.
    """
    for dirpath, dirnames, _ in os.walk("/kaggle/input"):
        for d in dirnames:
            if "typhoon" in d.lower():
                return os.path.join(dirpath, d)
    return None


def resolve_dirs(root):
    """This Kaggle mirror nests one level deeper than the upstream layout:
    <root>/metadata/metadata/*.csv and <root>/image/image/<storm_id>/*.h5.
    Tolerate both shapes.
    """
    meta = os.path.join(root, "metadata")
    if os.path.isdir(os.path.join(meta, "metadata")):
        meta = os.path.join(meta, "metadata")
    img = os.path.join(root, "image")
    if os.path.isdir(os.path.join(img, "image")):
        img = os.path.join(img, "image")
    return meta, img


def label_for_storm(df):
    """Positive == the storm actually reaches grade == 5 (Typhoon) at some point.

    NOTE: on the RSMC/Digital Typhoon scale, grade 5 is Typhoon but grade 6 is
    Extratropical (a different category, not "stronger than typhoon") and grade 7
    is 'just formed / weak'. A naive max(grade) >= 5 would therefore also mark
    storms that only ever went extratropical without ever reaching true typhoon
    strength as positives -- a real meteorological mislabel. Requiring grade == 5
    to have actually occurred avoids that. (Reviewed 2026-08-29: fixes a bug
    flagged by the code author itself; the originally-surveyed 784/332 balance
    used the looser max(grade) >= 5 rule and will shift slightly once this runs.)
    """
    grade = pd.to_numeric(df["grade"], errors="coerce")
    return int((grade == TYPHOON_GRADE).any())


def build_index(meta_dir, image_dir):
    """Return a DataFrame of (storm_id, label, h5_path) for early-stage frames."""
    csvs = sorted(
        f for f in os.listdir(meta_dir)
        if f.endswith(".csv") and f[:-4].isdigit()
    )
    print(f"Found {len(csvs)} per-storm metadata CSVs")

    rows = []
    n_no_early = 0
    n_missing_file = 0

    for i, name in enumerate(csvs):
        storm_id = name[:-4]
        try:
            df = pd.read_csv(os.path.join(meta_dir, name))
        except Exception as exc:  # a genuinely unreadable CSV, not an expected case
            print(f"  WARN: cannot read {name}: {exc}")
            continue
        if "grade" not in df.columns or "file_1" not in df.columns:
            print(f"  WARN: {name} missing expected columns, skipping")
            continue

        label = label_for_storm(df)

        grade = pd.to_numeric(df["grade"], errors="coerce")
        fname = df["file_1"].fillna("").astype(str).str.strip()
        early = df[(grade <= EARLY_GRADE_MAX) & (fname != "")]
        if early.empty:
            n_no_early += 1
            continue

        # Evenly spaced through the early stage rather than the first N frames --
        # the first frames of a storm are often near-identical to each other.
        picks = np.unique(
            np.linspace(0, len(early) - 1, min(FRAMES_PER_STORM, len(early))).round().astype(int)
        )
        for p in picks:
            rec = early.iloc[int(p)]
            h5_path = os.path.join(image_dir, storm_id, str(rec["file_1"]).strip())
            if not os.path.exists(h5_path):
                n_missing_file += 1
                continue
            rows.append({"storm_id": storm_id, "label": label, "h5_path": h5_path})

        if (i + 1) % 200 == 0:
            print(f"  indexed {i + 1}/{len(csvs)} storms, {len(rows)} frames so far")

    print(f"Storms with no usable early frame: {n_no_early}")
    print(f"Frames whose .h5 was listed in metadata but absent on disk: {n_missing_file}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Image loading
# ----------------------------------------------------------------------------
def load_frame_u8(h5_path):
    """Load one IR frame and return it as uint8 (512,512), display-oriented.

    The HDF5 file holds a single dataset 'Infrared', shape (512,512), float64,
    brightness temperature in Kelvin. Clip to the standard display window and
    INVERT so that cold (high, convective) cloud tops render bright -- this is
    both the conventional IR rendering and what the demo video shows, so the
    model sees exactly the pixels a human sees.
    """
    import h5py  # imported here so a worker process gets its own handle
    with h5py.File(h5_path, "r") as f:
        arr = np.asarray(f["Infrared"], dtype=np.float32)
    arr = np.clip(arr, TEMP_LO, TEMP_HI)
    arr = 1.0 - (arr - TEMP_LO) / (TEMP_HI - TEMP_LO)
    return (arr * 255.0).astype(np.uint8)


def to_model_input(u8_batch):
    """uint8 (N,H,W) -> normalised float tensor (N,1,IMG_SIZE,IMG_SIZE).

    Accepts either the native 512x512 frame or an already-downsized one, so the
    demo renderer can call the identical function on a raw frame.
    """
    t = torch.from_numpy(np.ascontiguousarray(u8_batch)).float().div_(255.0).unsqueeze(1)
    if t.shape[-1] != IMG_SIZE or t.shape[-2] != IMG_SIZE:
        t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return (t - NORM_MEAN) / NORM_STD


def downsize_u8(u8):
    """512x512 uint8 -> IMG_SIZE x IMG_SIZE uint8 (bilinear)."""
    t = torch.from_numpy(u8).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return t.squeeze().clamp(0, 255).round().byte().numpy()


class FrameDataset(Dataset):
    """Frames are pre-decoded into RAM once (see preload below), so this is just
    an indexed view plus optional augmentation."""

    def __init__(self, images_u8, labels, augment):
        self.images = images_u8
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augment:
            # Rotations only -- NOT flips. A mirror flip reverses the cyclone's
            # rotational chirality, which is physically fixed (counter-clockwise
            # in the northern hemisphere) and is exactly the kind of structure
            # the model should be keying on.
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img = np.rot90(img, k).copy()
        return torch.from_numpy(img), float(self.labels[i])


def preload(paths):
    """Decode every frame once, downsized, into a single uint8 array.

    ~4.4k frames at 224x224 uint8 == ~220MB, trivial for Kaggle RAM, and it turns
    8 epochs of HDF5 reads off a network-mounted 89GB dataset into one pass.
    Prints an early ETA so a pathologically slow mount can be caught and killed in
    the first minute instead of after half an hour.
    """
    out = np.zeros((len(paths), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    ok = np.ones(len(paths), dtype=bool)
    t0 = time.time()
    for i, p in enumerate(paths):
        try:
            out[i] = downsize_u8(load_frame_u8(p))
        except Exception as exc:
            # Not silently swallowed: the frame is dropped from the dataset and
            # counted, rather than being fed to the model as zeros.
            print(f"  LOAD FAIL {p}: {exc}")
            ok[i] = False
        if i == 99:
            rate = (time.time() - t0) / 100
            print(f"  preload rate {rate * 1000:.0f} ms/frame -> ETA {rate * len(paths) / 60:.1f} min")
        if (i + 1) % 500 == 0:
            print(f"  preloaded {i + 1}/{len(paths)} ({time.time() - t0:.0f}s)")
    print(f"Preload done in {time.time() - t0:.0f}s; {(~ok).sum()} failures")
    return out, ok


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model():
    """resnet18, ImageNet-pretrained, adapted to 1-channel input, 1 logit out."""
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
    # mean-collapsed filter to a single channel. So this preserves both the
    # learned edge/texture filters and their output scale, while keeping the
    # input tensor 1-channel (3x less memory and IO than replicating).
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


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    root = find_root()
    if not root:
        raise SystemExit("No typhoon dataset under /kaggle/input -- check dataset_sources")
    meta_dir, image_dir = resolve_dirs(root)
    print("ROOT      =", root)
    print("meta_dir  =", meta_dir)
    print("image_dir =", image_dir)

    index = build_index(meta_dir, image_dir)
    if index.empty:
        raise SystemExit("Index is empty -- dataset layout is not what was expected")

    # ---- SPLIT BY STORM ID, NOT BY FRAME -------------------------------------
    # This is the single easiest way to get a dishonest result on this dataset.
    # Frames from one storm are 3-6 hours apart and nearly identical, and every
    # frame of a storm shares the same label (the label is a property of the
    # storm, not the frame). A random per-row split would put near-duplicates of
    # the same storm on both sides and inflate val AUC massively. Splitting on
    # the storm ID means the val storms are genuinely unseen systems.
    storm_ids = np.array(sorted(index["storm_id"].unique()))
    rng = np.random.RandomState(SEED)
    rng.shuffle(storm_ids)
    n_val = int(round(len(storm_ids) * VAL_FRACTION))
    val_ids = set(storm_ids[:n_val])
    train_ids = set(storm_ids[n_val:])
    index["split"] = np.where(index["storm_id"].isin(val_ids), "val", "train")
    index.to_csv(INDEX_PATH, index=False)

    print(f"\nStorms: {len(storm_ids)}  train={len(train_ids)}  val={len(val_ids)}")
    print(f"Frames: total={len(index)}  "
          f"train={(index['split'] == 'train').sum()}  val={(index['split'] == 'val').sum()}")
    print("Storm-level label balance:",
          index.drop_duplicates("storm_id")["label"].value_counts().to_dict())

    print("\nPreloading frames...")
    images, ok = preload(index["h5_path"].tolist())
    index = index[ok].reset_index(drop=True)
    images = images[ok]
    n_load_failures = int((~ok).sum())

    tr_mask = (index["split"] == "train").to_numpy()
    va_mask = ~tr_mask
    y = index["label"].to_numpy().astype(np.float32)

    train_ds = FrameDataset(images[tr_mask], y[tr_mask], augment=True)
    val_ds = FrameDataset(images[va_mask], y[va_mask], augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    net, pretrained = build_model()
    net.to(device)

    # Positives are the MAJORITY here (~70%), so pos_weight comes out BELOW 1.0.
    # The formula n_neg/n_pos is the right one either way; it just down-weights
    # rather than up-weights the positive class in this dataset.
    n_pos = float((y[tr_mask] == 1).sum())
    n_neg = float((y[tr_mask] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    print(f"train frames: pos={n_pos:.0f} neg={n_neg:.0f} pos_weight={pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # No AMP on purpose: the dataset is a few thousand 224px images, so an epoch
    # is seconds of resnet18 compute either way, and fp32 avoids the
    # torch.cuda.amp / torch.amp API churn across torch versions.

    history = []
    best_auc = -1.0
    best_epoch = -1
    best_metrics = None

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

        # Select on AUC: it is threshold-free, which matters because the demo
        # displays a raw confidence rather than a hard 0/1 decision.
        score = m["auc"] if m["auc"] == m["auc"] else m["accuracy"]
        if score > best_auc:
            best_auc, best_epoch, best_metrics = score, epoch, m
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "arch": "resnet18_1ch",
                    "pretrained_backbone": pretrained,
                    "epoch": epoch,
                    "val_metrics": m,
                    # Saved so the demo renderer can reproduce preprocessing exactly
                    # and can warn if the storm it renders was in the training set.
                    "config": {
                        "temp_lo": TEMP_LO, "temp_hi": TEMP_HI, "img_size": IMG_SIZE,
                        "norm_mean": NORM_MEAN, "norm_std": NORM_STD,
                        "early_grade_max": EARLY_GRADE_MAX, "typhoon_grade": TYPHOON_GRADE,
                    },
                    "train_storm_ids": sorted(train_ids),
                    "val_storm_ids": sorted(val_ids),
                },
                CKPT_PATH,
            )
            print(f"  -> saved best checkpoint ({CKPT_PATH})")

    summary = {
        "task": "early-stage frame (grade<=2) -> P(storm later reaches grade>=5)",
        "pretrained_backbone": pretrained,
        "n_storms": int(len(storm_ids)),
        "n_train_storms": len(train_ids),
        "n_val_storms": len(val_ids),
        "n_train_frames": int(tr_mask.sum()),
        "n_val_frames": int(va_mask.sum()),
        "frames_per_storm_cap": FRAMES_PER_STORM,
        "n_load_failures": n_load_failures,
        "pos_weight": float(pos_weight.item()),
        "split": "by storm_id (never by frame -- frames of one storm are near-duplicates)",
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "history": history,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BEST ===")
    print(json.dumps(best_metrics, indent=2))
    print("Wrote", CKPT_PATH, METRICS_PATH, INDEX_PATH)


if __name__ == "__main__":
    main()
