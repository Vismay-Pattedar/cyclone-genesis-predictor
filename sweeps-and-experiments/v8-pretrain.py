"""v8 -- self-supervised temporal pretraining for the cyclone encoder.

WHY
---
Every model in this project starts from ImageNet weights. ImageNet teaches a
network to separate dogs from cars in colour photographs: the first layers learn
edges and gradients (which transfer to anything), but the deeper layers learn
object-part detectors -- wheels, fur, text -- that do not exist in an inverted
grayscale infrared cloud field. Those deep layers are close to dead weight here.

Two measured facts from this project say the bottleneck is DATA, not model size:
  * temporal information is the one lever that demonstrably works
    (TEND vs control: +0.0109 storm AUC, 95% CI [+0.0009, +0.0205])
  * more capacity actively HURTS -- TEND_HI at 320px scored 0.7447 vs the
    224px control's 0.7493, with the train/val gap already at 0.117

The standard fix for a data bottleneck without new labels is to exploit
unlabelled imagery. This project trains on ~11,000 frames while sitting on tens
of gigabytes of Digital Typhoon archive.

THE PRETEXT TASK
----------------
There are no labels on the extra imagery, but every frame carries a TIMESTAMP,
which is free supervision. Given two frames of the same storm, predict how far
apart in time they are. To do that at all, the encoder has to learn what cyclone
EVOLUTION looks like -- how convection reorganises, how banding tightens, what
six hours of development actually changes.

That is the same quantity the v6 tendency channels approximate by hand with a
subtraction. Instead of handing the trunk a difference image and hoping it knows
what to do with it, this trains the trunk to understand change in the first place.

THE LEAK CONSTRAINT -- the thing that decides whether any of this is valid
------------------------------------------------------------------------
Self-supervised pretraining needs no labels, but it is STILL leakage if the
encoder sees imagery from storms that later appear in a validation fold. With
one encoder shared across all four folds, a corpus drawn from the eval storms
would mean the encoder had seen every storm in the study -- the CV would be
meaningless, in the same family as the demo-storm leak that already bit this
project once.

So the corpus is Digital Typhoon **AU** (Southern Hemisphere), which is provably
disjoint from the WP evaluation set: the au_overlap_probe run tested all 464 AU
storms against WP by best-track position and found **0 matches**. It is also
disjoint from the v6/v7 sweeps, which use WP + TCIR only. No fold contamination,
no need to pretrain four separate encoders.

Mature/decay frames of WP storms are deliberately NOT used even though they are
unlabelled: a mature frame of a storm reveals that storm's outcome, so an
encoder that memorised its appearance would carry the answer into the early-stage
val frame of the same storm.

A NOTE ON PRETEXT SHORTCUTS (why this is safe even if imperfect)
---------------------------------------------------------------
Tropical convection has a strong diurnal cycle, so part of the time-gap signal
may be readable from diurnal phase rather than from storm development. That
makes the pretext task less EFFICIENT -- it does not make the result dishonest.
Unlike the v5 D_aux shortcut, which corrupted the evaluation itself, a weak
pretext task only means less benefit transfers. The downstream evaluation is
unchanged: identical folds, paired bootstrap, same climatology floor.

OUTPUT
------
`encoder_pretrained.pt`, a state_dict for a torchvision resnet18 whose conv1 has
been replaced with a 1-channel convolution -- exactly the shape build_model()
produces. It drops in wherever the ImageNet weights are loaded today, so the
existing tendency-channel init trick is untouched.
"""

import datetime
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------- constants
# Identical to the downstream pipeline. Pretraining in a different input domain
# than fine-tuning throws away most of the transfer, so this MUST match
# sweep.py's kelvin_to_display_u8 byte for byte.
TEMP_LO, TEMP_HI = 190.0, 300.0
IMG_SIZE = 224
NORM_MEAN, NORM_STD = 0.449, 0.226

SEED = 1234
EPOCHS = int(os.environ.get("V8_EPOCHS", 12))
BATCH = int(os.environ.get("V8_BATCH", 96))
LR = 3e-4
WD = 1e-4
PAIRS_PER_EPOCH = int(os.environ.get("V8_PAIRS", 60000))

# Time-gap buckets, in hours. Bucketed classification rather than regression:
# regression on a heavy-tailed gap distribution collapses to predicting the mean,
# which teaches the encoder nothing. Bucket edges are geometric because the
# visual difference between 1h and 3h is far larger than between 30h and 32h.
GAP_EDGES = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 1e9)
N_BUCKETS = len(GAP_EDGES) - 1
MAX_GAP_H = 48.0        # pairs wider than this carry little usable structure
MIN_FRAMES_PER_STORM = 8

STAMP_RE = re.compile(r"(\d{10})")


def log(msg):
    print(f"[v8] {msg}", flush=True)


# ------------------------------------------------------------------- data
def find_au_root(base="/kaggle/input"):
    """Locate the Digital Typhoon AU image root without hardcoding a path.

    Never descends into a directory literally named 'image' or 'metadata' while
    searching -- those hold thousands of per-storm subdirectories and walking
    them costs minutes. Prefers an 'au' path explicitly: find_dt_root() in an
    earlier version returned the FIRST match for 'typhoon' and could silently
    index the wrong mount, which is exactly the bug this guards against.
    """
    cands = []
    for root, dirs, _files in os.walk(base):
        depth = root[len(base):].count(os.sep)
        if depth >= 5:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in ("metadata",)]
        low = root.lower()
        if os.path.basename(root) == "image" and "typhoon" in low:
            cands.append(root)
            dirs[:] = []
    if not cands:
        raise SystemExit(f"no Digital Typhoon image root under {base}")
    au = [c for c in cands if "au" in c.lower() or "australia" in c.lower()]
    chosen = (au or cands)[0]
    log(f"image-root candidates: {cands}")
    log(f"CHOSEN (AU preferred): {chosen}")
    if not au:
        log("!! WARNING: no AU-looking mount found. If this is the WP mount, the "
            "corpus OVERLAPS the evaluation storms and the run is INVALID.")
    return chosen


def parse_stamp(name):
    m = STAMP_RE.search(name)
    if not m:
        return None
    s = m.group(1)
    try:
        return datetime.datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]),
                                 int(s[8:10])).timestamp() / 3600.0
    except ValueError:
        return None


def resolve_storm_dir(root):
    """Descend to the level that actually holds per-storm directories.

    The Kaggle mount nests as .../image/image/<storm_id>/<frame>.h5, so the
    first directory named 'image' is one level ABOVE the storm folders. Rather
    than hardcode the depth, probe for the level whose subdirectories contain
    .h5 files -- that survives either layout.
    """
    for cand in (root, os.path.join(root, "image")):
        if not os.path.isdir(cand):
            continue
        try:
            subs = [d for d in sorted(os.listdir(cand))
                    if os.path.isdir(os.path.join(cand, d))]
        except Exception:
            continue
        for d in subs[:15]:
            p = os.path.join(cand, d)
            try:
                if any(f.endswith(".h5") for f in os.listdir(p)[:30]):
                    log(f"storm dirs resolved at: {cand} ({len(subs)} entries)")
                    return cand
            except Exception:
                continue
    log(f"!! could not resolve storm dirs under {root}; using it as-is")
    return root


def build_index(image_root):
    """storm_id -> (paths sorted by time, absolute hours)."""
    image_root = resolve_storm_dir(image_root)
    storms = {}
    for storm_id in sorted(os.listdir(image_root)):
        sdir = os.path.join(image_root, storm_id)
        if not os.path.isdir(sdir):
            continue
        rows = []
        for fn in os.listdir(sdir):
            if not fn.endswith(".h5"):
                continue
            t = parse_stamp(fn)
            if t is not None:
                rows.append((t, os.path.join(sdir, fn)))
        if len(rows) < MIN_FRAMES_PER_STORM:
            continue
        rows.sort()
        storms[storm_id] = (np.array([r[1] for r in rows], dtype=object),
                            np.array([r[0] for r in rows], dtype=np.float64))
    return storms


def kelvin_to_display_u8(arr):
    """Byte-identical to sweep.py. Do not 'improve' this independently."""
    arr = np.clip(arr, TEMP_LO, TEMP_HI)
    arr = 1.0 - (arr - TEMP_LO) / (TEMP_HI - TEMP_LO)
    return (arr * 255.0).astype(np.uint8)


def load_frame_u8(path, size=IMG_SIZE):
    import h5py
    with h5py.File(path, "r") as f:
        key = "Infrared" if "Infrared" in f else list(f.keys())[0]
        arr = np.asarray(f[key], dtype=np.float32)
    u8 = kelvin_to_display_u8(arr)
    t = torch.from_numpy(u8)[None, None].float()
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear",
                                        align_corners=False)
    return t[0, 0].to(torch.uint8).numpy()


def bucket_of(gap_h):
    for i in range(N_BUCKETS):
        if GAP_EDGES[i] <= gap_h < GAP_EDGES[i + 1]:
            return i
    return N_BUCKETS - 1


class GapPairs(Dataset):
    """Random (frame_a, frame_b, bucket(|dt|)) pairs from within one storm."""

    def __init__(self, storms, n_pairs, seed):
        self.keys = sorted(storms.keys())
        self.storms = storms
        self.n = n_pairs
        self.rng = np.random.default_rng(seed)
        # Sample storms proportional to sqrt(length) so a single long track
        # cannot dominate the corpus.
        w = np.array([np.sqrt(len(storms[k][1])) for k in self.keys])
        self.p = w / w.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng((idx * 2654435761) ^ int(self.rng.integers(1 << 30)))
        for _ in range(20):
            k = self.keys[rng.choice(len(self.keys), p=self.p)]
            paths, times = self.storms[k]
            i = int(rng.integers(len(times)))
            near = np.where(np.abs(times - times[i]) <= MAX_GAP_H)[0]
            near = near[near != i]
            if near.size == 0:
                continue
            j = int(rng.choice(near))
            gap = abs(float(times[i] - times[j]))
            try:
                a = load_frame_u8(paths[i])
                b = load_frame_u8(paths[j])
            except Exception:
                continue
            # Order is randomised and only |dt| is predicted, so the task cannot
            # be solved by an arrow-of-time cue alone.
            if rng.random() < 0.5:
                a, b = b, a
            return (torch.from_numpy(a), torch.from_numpy(b),
                    torch.tensor(bucket_of(gap), dtype=torch.long))
        z = torch.zeros(IMG_SIZE, IMG_SIZE, dtype=torch.uint8)
        return z, z, torch.tensor(0, dtype=torch.long)


# ------------------------------------------------------------------ model
def build_encoder(pretrained=True):
    """resnet18 with a 1-channel conv1 -- the exact shape build_model() makes."""
    import torchvision
    if pretrained:
        try:
            net = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as exc:
            log(f"!! ImageNet weights unavailable ({exc}) -- cold start")
            net = torchvision.models.resnet18(weights=None)
    else:
        net = torchvision.models.resnet18(weights=None)
    old = net.conv1.weight.data
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.conv1.weight.data = old.mean(dim=1, keepdim=True)
    net.fc = nn.Identity()
    return net


class GapNet(nn.Module):
    """Siamese encoder + a head that reads the PAIR.

    The encoder is warm-started from ImageNet rather than random: the goal is to
    specialise the deep layers to cyclone structure, not to relearn edge
    detection from 46k frames.
    """

    def __init__(self):
        super().__init__()
        self.enc = build_encoder(pretrained=True)
        d = 512
        self.head = nn.Sequential(
            nn.Linear(4 * d, 512), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(512, N_BUCKETS),
        )

    def embed(self, x):
        return self.enc(x)

    def forward(self, xa, xb):
        ea, eb = self.embed(xa), self.embed(xb)
        h = torch.cat([ea, eb, (ea - eb).abs(), ea * eb], dim=1)
        return self.head(h)


def to_input(u8, gen=None, train=True):
    """uint8 (N,H,W) -> normalised (N,1,H,W), matching the downstream pipeline."""
    x = u8.float().unsqueeze(1) / 255.0
    if train and gen is not None:
        # Photometric jitter only, drawn INDEPENDENTLY per frame so the model
        # cannot read the gap off a shared augmentation. Geometric jitter is
        # deliberately excluded: a shift applied to one frame of the pair would
        # look like real storm motion and fabricate the very signal being taught.
        b = (torch.rand(x.shape[0], 1, 1, 1, generator=gen, device=x.device) - 0.5) * 0.10
        c = 1.0 + (torch.rand(x.shape[0], 1, 1, 1, generator=gen, device=x.device) - 0.5) * 0.20
        x = ((x - 0.5) * c + 0.5 + b).clamp(0, 1)
    return (x - NORM_MEAN) / NORM_STD


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} torch={torch.__version__}")
    if device == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)} n={torch.cuda.device_count()}")

    root = find_au_root()
    storms = build_index(root)
    n_frames = sum(len(v[1]) for v in storms.values())
    log(f"corpus: {len(storms)} storms, {n_frames:,} frames "
        f"(>= {MIN_FRAMES_PER_STORM} frames each)")
    if len(storms) < 50:
        raise SystemExit("corpus too small -- check the mount path")

    ds = GapPairs(storms, PAIRS_PER_EPOCH, SEED)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2,
                    pin_memory=(device == "cuda"), drop_last=True,
                    persistent_workers=True)

    net = GapNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * (PAIRS_PER_EPOCH // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=0.15)
    lossf = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)

    hist = []
    for ep in range(EPOCHS):
        net.train()
        tot = cor = 0
        loss_sum = 0.0
        t0 = time.time()
        for bi, (a, b, y) in enumerate(dl):
            a, b, y = a.to(device), b.to(device), y.to(device)
            xa, xb = to_input(a, gen), to_input(b, gen)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = net(xa, xb)
                loss = lossf(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < steps - 1:
                sched.step()
            loss_sum += float(loss) * y.numel()
            cor += int((out.argmax(1) == y).sum())
            tot += y.numel()
            if bi % 100 == 0:
                log(f"  ep{ep} step{bi}/{len(dl)} loss={loss_sum/max(tot,1):.4f} "
                    f"acc={cor/max(tot,1):.3f}")
        acc = cor / max(tot, 1)
        hist.append({"epoch": ep, "loss": loss_sum / max(tot, 1), "acc": acc,
                     "secs": round(time.time() - t0, 1)})
        # Chance is 1/N_BUCKETS. An accuracy stuck at chance means the encoder is
        # learning nothing from the pretext task and the run is not worth
        # transferring -- that is a real, reportable negative result.
        log(f"epoch {ep}: loss={hist[-1]['loss']:.4f} acc={acc:.3f} "
            f"(chance={1.0/N_BUCKETS:.3f}) [{hist[-1]['secs']}s]")

    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    ckpt_path = os.path.join(out_dir, "encoder_pretrained.pt")
    torch.save({
        "state_dict": net.enc.state_dict(),
        "arch": "resnet18",
        "in_ch": 1,
        "img_size": IMG_SIZE,
        "temp_lo": TEMP_LO, "temp_hi": TEMP_HI,
        "norm_mean": NORM_MEAN, "norm_std": NORM_STD,
        "pretext": "time_gap_bucket_classification",
        "gap_edges": list(GAP_EDGES),
        "corpus": {"root": root, "n_storms": len(storms), "n_frames": int(n_frames)},
        "history": hist,
        "final_acc": hist[-1]["acc"] if hist else None,
        "chance_acc": 1.0 / N_BUCKETS,
        "seed": SEED,
    }, ckpt_path)
    json.dump({"history": hist, "n_storms": len(storms), "n_frames": int(n_frames),
               "chance_acc": 1.0 / N_BUCKETS, "root": root},
              open(os.path.join(out_dir, "pretrain_metrics.json"), "w"), indent=1)

    log(f"saved {ckpt_path}")
    log(f"final pretext acc {hist[-1]['acc']:.3f} vs chance {1.0/N_BUCKETS:.3f}")
    log(f"total {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
