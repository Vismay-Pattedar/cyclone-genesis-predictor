"""
Cyclone Genesis Predictor -- Demo video renderer.

Plays one storm's full infrared satellite sequence in chronological order and
overlays the early-intensification model's live output.

Two clearly-distinguished phases, so the finished video never blurs the line
between what the AI guessed and what the data confirmed:

  PHASE 1 (amber -> orange-red box, "AI PREDICTION"):
      the storm has not yet reached grade >= 5. The model sees ONLY the current
      infrared frame -- no future frames, no grade / pressure / wind -- and emits
      P(this storm will later reach typhoon strength). Within this phase the box
      and text colour ride a confidence sub-gradient, so the prediction visibly
      intensifies as the model gets more sure.

  PHASE 2 (green box, "CONFIRMED BY OBSERVATION"):
      the storm's real best-track grade has reached >= 5. The overlay switches to
      the actual observed grade / pressure / wind, and recaps what the model said
      back at the first early frame.

COLOUR SEMANTICS -- deliberately NOT the same as local_tool/detect.py:
      here the arc is amber -> orange-red -> GREEN, and green means "the AI's
      prediction was confirmed correct". detect.py's arc is green -> crimson and
      means "how worried should you be about this footage". Merging the two would
      make this file's confirmation moment read as an escalation instead of a
      validation, which is the opposite of the point. Nothing green appears on
      the confidence ramp in this file for the same reason: green is reserved for
      confirmation.

EVERYTHING THAT MOVES ON SCREEN IS A MEASURED VALUE. No invented jitter, no
count-up animations, no scanning lines. The per-frame "life" in the picture comes
from the class-activation overlay, which genuinely shifts because the network's
attention genuinely shifts.

Inputs expected on the kernel (see kernel-metadata.json):
  1. bwandowando/digital-typhoon-dataset-western-pacific-wp   (the imagery)
  2. a second dataset containing model_best.pt from the training kernel.
     ASSUMPTION FOR OPS: the checkpoint is published as a Kaggle dataset and
     attached as a second source; this script finds it by walking /kaggle/input
     for a *.pt file outside the typhoon mount, so the dataset's slug and
     internal folder layout do not matter. If the checkpoint is instead attached
     as a kernel output (kernel_sources), it lands under
     /kaggle/input/<kernel-slug>/model_best.pt and the same walk still finds it.

Runs as a plain Kaggle *script* kernel. No package beyond numpy / pandas / torch
/ PIL is required at render time (matplotlib is probed only for its bundled
fonts, and cv2 only as a video-encoding fallback).
"""

import datetime as _dt
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------
# Demo parameters -- change STORM_ID to render a different storm
# ----------------------------------------------------------------------------
# v3 combined Digital Typhoon + TCIR and re-split by (basin, label), which is a
# DIFFERENT split from v1's WPAC-only split. Every storm previously vetted for
# the v1 demo (201618, 201511, 201003, 200620) landed in v3's TRAINING split as
# a result -- confirmed directly against train_kernel_v3/output/model_best.pt's
# train_storm_ids/val_storm_ids before picking a replacement. Do not reuse any
# of those four with the v3 (or later) checkpoint without re-checking the split.
# First replacement tried, 202302, was checked and rejected: frame 1 came back
# only 54.7% confident (a near coin-flip, not a dramatic call).
#
# 201217: peak wind 110 kt, 313 imaged frames, VAL-split storm (genuinely
# unseen). Picked via scan_kernel/scan.py run against v3's real val split (all
# 419 val storms scanned, corrupted-frame guard included after the 201003
# lesson), ranked #1 of 115 candidates on a combined "looks undramatic AND the
# model is confident anyway" score, THEN visually verified by downloading and
# rendering the actual frame -- not trusted from the metric alone. Frame 1 (2012
# -09-20 00Z) genuinely shows two separate, disorganised cloud clusters with no
# visible spiral or eye -- a non-expert would call this "just some clouds" -- yet
# the model is already 84.9% confident on that exact frame.
#
# IMPORTANT: whether this storm is actually held out depends on WHICH checkpoint
# is attached. The split verdict is read from the checkpoint at runtime and
# printed AND rendered on screen verbatim -- it is never assumed. If the
# attached checkpoint has this storm in its TRAINING split, the title card says
# so in red. Check the title card before showing the video to anyone.
STORM_ID = "201217"

FPS = 6                 # slow enough to read the overlay text
SCALE = 2               # render 512 -> 1024 so the overlay text is legible
SAVE_FRAMES = True      # also dump PNGs, as a fallback if video encoding fails

TITLE_SECONDS = 2.5     # opening card
ENDCARD_SECONDS = 6.0   # closing "receipts" card
BEAT_SECONDS = 1.0      # hold on the confirmation frame
CAM_ALPHA = 0.40        # peak opacity of the class-activation overlay
PANEL_ALPHA = 205       # backing-plate opacity (0-255)
TTA_ROTATIONS = 4       # test-time rotations for the uncertainty band (see below)

OUT_DIR = "/kaggle/working"
VIDEO_PATH = os.path.join(OUT_DIR, "demo_video.mp4")
FRAME_DIR = os.path.join(OUT_DIR, "frames")

# Preprocessing defaults -- overridden by the values stored in the checkpoint so
# the model always sees exactly what it saw in training.
TEMP_LO, TEMP_HI = 190.0, 300.0
IMG_SIZE = 224
NORM_MEAN, NORM_STD = 0.449, 0.226
TYPHOON_GRADE = 5

# Bounding box geometry (proven in poc_render.py against a real frame).
KM_PER_PIXEL = 5.0
NM_TO_KM = 1.852
CENTER = 256            # 512x512 storm-centered azimuthal-equidistant crop

GRADE_MAP = {
    2: "Tropical Depression",
    3: "Tropical Storm",
    4: "Severe Tropical Storm",
    5: "Typhoon",
    6: "Extratropical",
    7: "Just formed / weak",
}

# ----------------------------------------------------------------------------
# Colour
# ----------------------------------------------------------------------------
COL_CONFIRM = (0, 220, 90)      # green -- observed ground truth. UNCHANGED.

# "AI PREDICTION" phase sub-gradient: muted amber at low confidence -> strong
# saturated orange-red at high confidence. The prediction phase therefore
# visibly intensifies as the model gets more sure, without touching the meaning
# of the green confirmation switch.
PREDICT_STOPS = [
    (0.00, (176, 132,  52)),   # muted amber
    (0.35, (226, 158,  36)),   # amber
    (0.60, (247, 132,  22)),   # orange
    (0.80, (240,  86,  26)),   # hot orange
    (1.00, (222,  46,  32)),   # orange-red
]

# The FILL gradient for the meter bar and the sparkline dots. Same family as
# PREDICT_STOPS but with genuinely dark endpoints, which are legible as a filled
# bar even though they would be invisible as text on a near-black plate. There
# is deliberately no green anywhere on this ramp: green is reserved for
# "confirmed by observation" and must not double as "low confidence".
METER_STOPS = [
    (0.00, ( 58,  38,  14)),   # very dark amber
    (0.15, (128,  92,  34)),
    (0.35, (226, 158,  36)),   # amber
    (0.60, (247, 132,  22)),   # orange
    (0.80, (238,  84,  26)),
    (1.00, (138,  20,  14)),   # very dark red
]

# Anything rendered as TEXT or as a thin stroke on the near-black plates goes
# through this clamp: the dark endpoints above sit at roughly 1.7:1 contrast
# against black, i.e. functionally invisible. Fills are never clamped -- that is
# exactly where the dark ends are supposed to read.
TEXT_CLAMP = (0.10, 0.92)

# Warm CAM ramp, built by hand so this file needs no colormap library.
CAM_STOPS = [
    (0.00, (  8,  10,  60)),
    (0.35, (120,  16, 140)),
    (0.62, (232,  58,  40)),
    (0.85, (255, 172,  30)),
    (1.00, (255, 250, 200)),
]


def interp_stops(stops, p):
    p = float(np.clip(p, 0.0, 1.0))
    lo_p, lo_c = stops[0]
    if p <= lo_p:
        return lo_c
    for hi_p, hi_c in stops[1:]:
        if p <= hi_p:
            span = hi_p - lo_p
            t = 0.0 if span <= 0 else (p - lo_p) / span
            return tuple(int(round(lo_c[i] + t * (hi_c[i] - lo_c[i]))) for i in range(3))
        lo_p, lo_c = hi_p, hi_c
    return stops[-1][1]


def predict_color(p):
    """Box / text colour during the AI PREDICTION phase (readability-clamped)."""
    return interp_stops(PREDICT_STOPS, float(np.clip(p, *TEXT_CLAMP)))


def meter_color(p):
    """FILL colour -- full range, un-clamped."""
    return interp_stops(METER_STOPS, p)


def rgba(color, a=255):
    if color is None:
        return (255, 255, 255, a)
    return tuple(color) if len(color) == 4 else (color[0], color[1], color[2], a)


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------
def find_typhoon_root():
    for dirpath, dirnames, _ in os.walk("/kaggle/input"):
        for d in dirnames:
            if "typhoon" in d.lower():
                return os.path.join(dirpath, d)
    return None


def resolve_dirs(root):
    """This Kaggle mirror nests one level deeper: metadata/metadata, image/image."""
    meta = os.path.join(root, "metadata")
    if os.path.isdir(os.path.join(meta, "metadata")):
        meta = os.path.join(meta, "metadata")
    img = os.path.join(root, "image")
    if os.path.isdir(os.path.join(img, "image")):
        img = os.path.join(img, "image")
    return meta, img


def find_checkpoint(exclude_root):
    """Find model_best.pt (or any .pt) anywhere under /kaggle/input except inside
    the 89GB imagery mount, which must never be walked."""
    hits = []
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if exclude_root and os.path.abspath(dirpath).startswith(os.path.abspath(exclude_root)):
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.endswith((".pt", ".pth")):
                hits.append(os.path.join(dirpath, fn))
    if not hits:
        return None
    # Prefer the exact filename the training kernel writes.
    for h in hits:
        if os.path.basename(h) == "model_best.pt":
            return h
    return hits[0]


# ----------------------------------------------------------------------------
# Model (must stay byte-for-byte consistent with train.py's build_model)
# ----------------------------------------------------------------------------
def build_model():
    import torchvision

    # weights=None: we immediately overwrite everything with the checkpoint, so
    # there is no download and this kernel needs no internet.
    net = torchvision.models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


def load_frame_u8(h5_path):
    """IR brightness temperature (K) -> display-oriented uint8, cold = bright."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        arr = np.asarray(f["Infrared"], dtype=np.float32)
    arr = np.clip(arr, TEMP_LO, TEMP_HI)
    arr = 1.0 - (arr - TEMP_LO) / (TEMP_HI - TEMP_LO)
    return (arr * 255.0).astype(np.uint8)


def to_model_input(u8_batch):
    t = torch.from_numpy(np.ascontiguousarray(u8_batch)).float().div_(255.0).unsqueeze(1)
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return (t - NORM_MEAN) / NORM_STD


# ----------------------------------------------------------------------------
# Class Activation Mapping -- where the network actually looked
# ----------------------------------------------------------------------------
# This architecture is the textbook CAM case: layer4 -> global average pool -> a
# SINGLE Linear(512, 1), with no hidden layer in between. So
#
#     score = sum_c w[c] * mean_xy(A[c,x,y]) + b
#           = mean_xy( sum_c w[c] * A[c,x,y] ) + b
#
# and the spatial map CAM[x,y] = sum_c w[c]*A[c,x,y] is EXACTLY the per-location
# contribution to the logit. Exact here, not an approximation of Grad-CAM, and
# it needs no backward pass -- just a forward hook. Effectively free.
class CamTap:
    def __init__(self, net):
        self.value = None
        self.handle = net.layer4.register_forward_hook(self._hook)
        self.fc_w = net.fc.weight.detach().clone()

    def _hook(self, _mod, _inp, out):
        self.value = out.detach()

    def maps(self):
        """-> [N, h, w] float32 in [0,1], per-frame min-max normalised at the
        native conv resolution (7x7 for a 224px input). The SPATIAL PATTERN is
        the real, unmodified quantity; the normalisation is the standard CAM
        display convention."""
        if self.value is None:
            return np.zeros((0, 1, 1), np.float32)
        w = self.fc_w.view(1, -1, 1, 1)
        cam = (self.value * w).sum(dim=1).float().cpu().numpy()
        out = np.empty_like(cam)
        for i, c in enumerate(cam):
            lo, hi = float(c.min()), float(c.max())
            out[i] = (c - lo) / (hi - lo) if (hi - lo) > 1e-6 else 0.0
        return out

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


_CAM_LUT = None


def cam_lut():
    global _CAM_LUT
    if _CAM_LUT is None:
        _CAM_LUT = np.array([interp_stops(CAM_STOPS, i / 255.0) for i in range(256)],
                            dtype=np.uint8)
    return _CAM_LUT


def apply_cam(rgb_u8, cam_small, alpha=CAM_ALPHA):
    """Blend a translucent CAM heatmap onto an RGB uint8 frame. The map is
    upsampled bilinearly from the conv resolution to the display resolution.
    Alpha ramps with cam**1.7 so low-attention areas stay essentially clear and
    the underlying satellite imagery is never hidden."""
    h, w = rgb_u8.shape[:2]
    cam = np.asarray(Image.fromarray((np.clip(cam_small, 0, 1) * 255).astype(np.uint8))
                     .resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    heat = cam_lut()[(cam * 255).astype(np.uint8)].astype(np.float32)
    a = (alpha * np.power(cam, 1.7)).astype(np.float32)[..., None]
    return (rgb_u8.astype(np.float32) * (1.0 - a) + heat * a).astype(np.uint8)


# ----------------------------------------------------------------------------
# Overlay drawing
# ----------------------------------------------------------------------------
_FONT_CACHE = {}

# Monospace is load-bearing, not decoration: every numeric readout on screen
# changes frame to frame, and in a proportional font a changing digit changes
# the string's pixel width, which re-flows the backing plate and makes the whole
# overlay jitter horizontally every frame. Monospace digits plus fixed-width
# formats keep the plates rock-steady while the numbers underneath move.
_MONO_NAMES = ("DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf")
_PROP_NAMES = ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf")


def get_font(size, mono=False):
    size = int(max(8, size))
    key = (size, bool(mono))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = []
    try:
        import matplotlib
        fdir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        candidates += [os.path.join(fdir, n) for n in (_MONO_NAMES if mono else _PROP_NAMES)]
    except Exception:
        pass
    if mono:
        candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                       "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                       "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
                       r"C:\Windows\Fonts\consolab.ttf"]
    else:
        candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                       "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                       r"C:\Windows\Fonts\segoeuib.ttf"]
    font = None
    for c in candidates:
        try:
            font = ImageFont.truetype(c, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()  # tiny, but never crashes the render
        print("WARN: no TrueType font found, overlay text will be very small")
    _FONT_CACHE[key] = font
    return font


def _norm_line(item, default_font):
    """A line is (text, color) or (text, color, font)."""
    if len(item) >= 3 and item[2] is not None:
        return item[0], item[1], item[2]
    return item[0], item[1], default_font


def _font_height(f):
    try:
        asc, desc = f.getmetrics()
        return int(asc + desc)
    except Exception:
        bb = f.getbbox("Ag")
        return int(bb[3] - bb[1]) + 4


def _line_height(lines, font):
    """Content-INDEPENDENT line height, from font metrics. A readout whose text
    changes must not change the plate's height any more than its width."""
    fonts = [font] + [it[2] for it in lines if len(it) >= 3 and it[2] is not None]
    return max(_font_height(f) for f in fonts) + 6


def block_size(draw, lines, font, pad=6):
    """Width uses textlength() (the layout ADVANCE, which counts leading spaces)
    rather than the ink bbox, so a right-aligned monospace number like "  9.8%"
    measures the same as "100.0%". The ink bbox would silently defeat that."""
    widths = []
    for item in lines:
        txt, _col, fnt = _norm_line(item, font)
        widths.append(draw.textlength(txt, font=fnt))
    return int(math.ceil(max(widths))) + 2 * pad, _line_height(lines, font) * len(lines) + 2 * pad


def fit_text(draw, txt, font, max_w):
    """Truncate with an ellipsis so a plate cannot run under a pinned tile."""
    if max_w <= 0 or draw.textlength(txt, font=font) <= max_w:
        return txt
    lo, hi = 0, len(txt)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if draw.textlength(txt[:mid] + "...", font=font) <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return txt[:lo].rstrip() + "..."


def fit_lines(draw, lines, default_font, max_w):
    return [(fit_text(draw, t, f, max_w), c, f)
            for t, c, f in (_norm_line(i, default_font) for i in lines)]


def text_block(draw, xy, lines, font, color, pad=6, alpha=PANEL_ALPHA):
    """Draw left-aligned lines on a semi-transparent backing plate."""
    x, y = xy
    w, h = block_size(draw, lines, font, pad)
    line_h = _line_height(lines, font)
    draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0, alpha))
    draw.rectangle([x, y, x + w, y + h], outline=rgba(color), width=2)
    cy = y + pad
    for item in lines:
        txt, col, fnt = _norm_line(item, font)
        draw.text((x + pad, cy), txt, fill=rgba(col or (255, 255, 255)), font=fnt)
        cy += line_h
    return h


def draw_tracker_box(draw, box, color, width, crosshair=True, max_arm=None):
    """Four L-shaped corner brackets + a thin centre crosshair -- the standard
    'a machine is tracking this' grammar -- instead of a plain rectangle.

    Every stroke is laid down twice: once in black at width+4 (the 'casing'),
    then in the accent colour at width, so the accent pops against any
    background. Standard broadcast-graphics trick."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    # Cap the arm length in absolute pixels. A mature storm's 30kt wind radius
    # can span nearly the whole frame, and 22%-of-box arms on a box that big
    # would sprawl across the instrument panels. The BOX itself is never
    # resized -- it is real wind-radius data -- only the bracket arms are.
    arm = max(6.0, 0.22 * min(bw, bh))
    if max_arm:
        arm = min(arm, float(max_arm))
    segs = []
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                           (x0, y1, 1, -1), (x1, y1, -1, -1)):
        segs.append([(cx, cy), (cx + dx * arm, cy)])
        segs.append([(cx, cy), (cx, cy + dy * arm)])
    if crosshair:
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        g = min(max(3.0, 0.05 * min(bw, bh)), arm / 2.0)
        segs += [[(mx - 2 * g, my), (mx - g, my)], [(mx + g, my), (mx + 2 * g, my)],
                 [(mx, my - 2 * g), (mx, my - g)], [(mx, my + g), (mx, my + 2 * g)]]
    for s in segs:
        draw.line(s, fill=(0, 0, 0, 235), width=int(width) + 4)
    for s in segs:
        draw.line(s, fill=rgba(color), width=int(width))


_GRAD_CACHE = {}


def gradient_strip(w, h):
    """Cached strip painted with the full-range FILL gradient (un-clamped)."""
    key = (int(w), int(h))
    if key in _GRAD_CACHE:
        return _GRAD_CACHE[key]
    w_i, h_i = key
    row = np.array([meter_color(i / max(1, w_i - 1)) for i in range(w_i)], dtype=np.uint8)
    img = Image.fromarray(np.repeat(row[None, :, :], h_i, axis=0), "RGB").convert("RGBA")
    _GRAD_CACHE[key] = img
    return img


def draw_meter(layer, draw, box, prob, ticks=(), font=None, title=None):
    """Horizontal confidence meter: full-range gradient fill up to `prob`,
    a visual scale, and a bright needle at the current value."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    w, h = max(2, x1 - x0), max(2, y1 - y0)
    th = (_font_height(font) + 4) if font else 0
    draw.rectangle([x0 - 5, y0 - 5 - (th if title else 0), x1 + 5, y1 + 5 + th],
                   fill=(0, 0, 0, PANEL_ALPHA))
    if title and font:
        draw.text((x0, y0 - th), title, fill=(190, 190, 195, 255), font=font)
    draw.rectangle([x0, y0, x1, y1], fill=(34, 34, 40, 255))
    fill_px = int(round(w * float(np.clip(prob, 0.0, 1.0))))
    if fill_px > 0:
        strip = gradient_strip(w, h).crop((0, 0, fill_px, h))
        layer.paste(strip, (x0, y0), strip)
    draw.rectangle([x0, y0, x1, y1], outline=(205, 205, 210, 220), width=1)
    for tval, tlabel in ticks:
        tx = x0 + w * float(np.clip(tval, 0.0, 1.0))
        draw.line([(tx, y0 - 4), (tx, y1 + 4)], fill=(0, 0, 0, 255), width=4)
        draw.line([(tx, y0 - 4), (tx, y1 + 4)], fill=(232, 232, 236, 255), width=1)
        if tlabel and font:
            draw.text((tx + 3, y1 + 3), tlabel, fill=(178, 178, 184, 255), font=font)
    nx = x0 + w * float(np.clip(prob, 0.0, 1.0))
    draw.line([(nx, y0 - 7), (nx, y1 + 7)], fill=(0, 0, 0, 255), width=7)
    draw.line([(nx, y0 - 7), (nx, y1 + 7)], fill=(255, 255, 255, 255), width=2)


def _dashed_line(draw, p0, p1, fill, width=1, dash=8, gap=7):
    (ax, ay), (bx, by) = p0, p1
    total = math.hypot(bx - ax, by - ay)
    if total <= 0:
        return
    ux, uy = (bx - ax) / total, (by - ay) / total
    t = 0.0
    while t < total:
        e = min(total, t + dash)
        draw.line([(ax + ux * t, ay + uy * t), (ax + ux * e, ay + uy * e)],
                  fill=fill, width=width)
        t = e + gap


def render_axis(size, series, band=None, thresh=None, thresh_label=None,
                marker=None, marker_label=None, font=None, title=None):
    """Pre-render the FULL confidence axis once, as a static RGBA tile.

    This file knows every frame's probability before rendering starts, so unlike
    the streaming local tool it can draw the complete curve up front: every raw
    per-frame prob as a dot, a line through them, the rotation-TTA spread as a
    band, a dashed threshold, and a vertical marker at the real confirmation
    frame. Only the "now" cursor is drawn per frame, on top of this tile."""
    w, h = size
    th = (_font_height(font) + 4) if font else 0
    # Opaque across the WHOLE tile, title strip included: this tile is pasted
    # over the frame with its own alpha as the mask, and a transparent title
    # strip would let the tracker brackets show through the words.
    tile = Image.new("RGBA", (w, h + th), (0, 0, 0, PANEL_ALPHA))
    draw = ImageDraw.Draw(tile)
    if title and font:
        draw.text((0, 0), title, fill=(190, 190, 195, 255), font=font)
    y_off = th
    draw.rectangle([0, y_off, w - 1, y_off + h - 1], fill=(18, 18, 24, 236))
    draw.rectangle([0, y_off, w - 1, y_off + h - 1], outline=(90, 90, 100, 255), width=1)

    n = len(series)
    if n == 0:
        return tile, (lambda i: 0), y_off

    def px(i):
        return (w - 1) * (i / float(max(1, n - 1)))

    def py(p):
        return y_off + (h - 1) - (h - 1) * float(np.clip(p, 0.0, 1.0))

    for gy in (0.25, 0.50, 0.75):
        draw.line([(1, py(gy)), (w - 2, py(gy))], fill=(60, 60, 70, 180), width=1)

    if band is not None:
        lo, hi = band
        pts = ([(px(i), py(hi[i])) for i in range(n)] +
               [(px(i), py(lo[i])) for i in range(n - 1, -1, -1)])
        if len(pts) >= 3:
            draw.polygon(pts, fill=(170, 185, 225, 72))

    if thresh is not None:
        _dashed_line(draw, (1, py(thresh)), (w - 2, py(thresh)), (225, 225, 120, 205), 1)
        if thresh_label and font:
            tw = draw.textlength(thresh_label, font=font)
            draw.text((w - tw - 4, py(thresh) - _font_height(font) - 2), thresh_label,
                      fill=(225, 225, 120, 225), font=font)

    if n >= 2:
        draw.line([(px(i), py(series[i])) for i in range(n)],
                  fill=(232, 232, 238, 215), width=1)
    r = 1 if n > 160 else 2
    for i in range(n):
        cx, cy = px(i), py(series[i])
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgba(meter_color(series[i])))

    if marker is not None and 0 <= marker < n:
        mx = px(marker)
        draw.line([(mx, y_off), (mx, y_off + h - 1)], fill=(0, 0, 0, 225), width=4)
        draw.line([(mx, y_off), (mx, y_off + h - 1)], fill=rgba(COL_CONFIRM), width=2)
        if marker_label and font:
            tw = draw.textlength(marker_label, font=font)
            draw.text((min(mx + 4, w - tw - 3), y_off + 2), marker_label,
                      fill=rgba(COL_CONFIRM), font=font)
    return tile, px, y_off


def draw_tile(layer, draw, xy, img, size, caption_lines, font, accent):
    """A small pinned thumbnail with a caption plate under it. Returns its
    (w, h) footprint so callers can stack tiles."""
    x, y = int(xy[0]), int(xy[1])
    thumb = img.convert("RGB").resize((size, size), Image.BICUBIC).convert("RGBA")
    draw.rectangle([x - 3, y - 3, x + size + 3, y + size + 3], fill=(0, 0, 0, PANEL_ALPHA))
    layer.paste(thumb, (x, y), thumb)
    draw.rectangle([x, y, x + size, y + size], outline=rgba(accent), width=2)
    ch = 0
    if caption_lines:
        # Clip captions to the tile's own width: a tile pinned against the right
        # edge must never grow a caption plate that runs off frame.
        caption_lines = fit_lines(draw, caption_lines, font, size - 2)
        ch = text_block(draw, (x - 3, y + size + 4), caption_lines, font, accent, pad=4)
        ch += 4
    return size + 6, size + 6 + ch


def make_locator(lats, lngs, size, font):
    """Inset locator: the storm's REAL best-track positions on a plain lat/lon
    graticule. Deliberately NO coastlines -- no basemap package (cartopy,
    geopandas, basemap) is installed in this environment or guaranteed on a bare
    Kaggle script kernel, and this must never fetch anything from the network.
    Returns (tile, proj) where proj(lat, lng) -> (x, y) in tile coordinates."""
    pad = 4.0
    la0, la1 = float(np.min(lats)) - pad, float(np.max(lats)) + pad
    ln0, ln1 = float(np.min(lngs)) - pad, float(np.max(lngs)) + pad
    # Keep the aspect square so the track is not distorted.
    span = max(la1 - la0, ln1 - ln0)
    lac, lnc = (la0 + la1) / 2, (ln0 + ln1) / 2
    la0, la1 = lac - span / 2, lac + span / 2
    ln0, ln1 = lnc - span / 2, lnc + span / 2

    tile = Image.new("RGBA", (size, size), (12, 18, 30, 240))
    draw = ImageDraw.Draw(tile)

    def proj(lat, lng):
        return ((float(lng) - ln0) / (ln1 - ln0) * (size - 1),
                (la1 - float(lat)) / (la1 - la0) * (size - 1))

    step = 5.0 if span <= 40 else 10.0
    step_px = size * step / span
    # Label only as often as the labels actually fit, so a small inset never
    # collides its own graticule text.
    lab_w = draw.textlength("000E", font=font) + 6 if font else 1e9
    stride = max(1, int(math.ceil(lab_w / max(1.0, step_px))))
    for axis in ("lng", "lat"):
        lo, hi = (ln0, ln1) if axis == "lng" else (la0, la1)
        g = math.floor(lo / step) * step
        k = 0
        while g <= hi:
            if axis == "lng":
                x, _ = proj(la0, g)
                draw.line([(x, 0), (x, size)], fill=(48, 62, 84, 255), width=1)
                if font and lo < g < hi and k % stride == 0:
                    draw.text((x + 2, size - _font_height(font) - 2), f"{int(g)}E",
                              fill=(110, 130, 155, 255), font=font)
            else:
                _, y = proj(g, ln0)
                draw.line([(0, y), (size, y)], fill=(48, 62, 84, 255), width=1)
                if font and lo < g < hi and k % stride == 0:
                    draw.text((2, y + 1), f"{int(g)}N", fill=(110, 130, 155, 255), font=font)
            g += step
            k += 1

    pts = [proj(a, o) for a, o in zip(lats, lngs)]
    if len(pts) >= 2:
        draw.line(pts, fill=(120, 140, 170, 220), width=1)
    draw.rectangle([0, 0, size - 1, size - 1], outline=(90, 105, 130, 255), width=1)
    return tile, proj, pts


# ----------------------------------------------------------------------------
# Formatting helpers -- all fixed-width, for the monospace plates
# ----------------------------------------------------------------------------
def fmt_pct(p):
    """'  9.8%' .. '100.0%' -- constant advance width in a monospace font."""
    return f"{p * 100.0:5.1f}%"


def fmt_hours(hours):
    if hours is None:
        return "  --  "
    sign = "-" if hours < 0 else ""
    h = abs(float(hours))
    d = int(h // 24)
    return f"{sign}{d}d {h - 24 * d:4.1f}h" if d else f"{sign}{h:5.1f}h"


def row_stamp(row):
    return (f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d} "
            f"{int(row['hour']):02d}Z")


def row_time(row):
    try:
        return _dt.datetime(int(row["year"]), int(row["month"]),
                            int(row["day"]), int(row["hour"]))
    except (ValueError, TypeError, KeyError):
        return None


def model_input_tile(x1):
    """The exact normalised 224x224 tensor the network was handed, linearly
    rescaled from its own [min,max] to 0-255. Nothing cropped, nothing
    re-derived -- this is literally the model's input."""
    a = x1.detach().float().cpu().numpy()
    while a.ndim > 2:
        a = a[0]
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo) if (hi - lo) > 1e-6 else np.zeros_like(a)
    return Image.fromarray((a * 255.0).astype(np.uint8)).convert("RGB")


def radius_to_halfwidth_px(radius_nm):
    if radius_nm and radius_nm > 0:
        return (radius_nm * NM_TO_KM) / KM_PER_PIXEL
    return None


def compute_box(row):
    """Bounding box from the best-track wind-radius fields (real meteorological
    data, not an annotation). Prefer the 30kt radius -- it is the larger one and
    describes the storm's full circulation -- then 50kt, then a fixed fallback.
    Returned in native 512px coordinates."""
    def num(k):
        v = pd.to_numeric(row.get(k, 0), errors="coerce")
        return 0.0 if pd.isna(v) else float(v)

    half_long = radius_to_halfwidth_px(num("long30")) or radius_to_halfwidth_px(num("long50"))
    half_short = radius_to_halfwidth_px(num("short30")) or radius_to_halfwidth_px(num("short50"))
    used_fallback = False
    if not half_long or not half_short:
        used_fallback = True
        half_long = half_short = 60  # ~300 km half-width default

    x0 = max(0, CENTER - half_long)
    y0 = max(0, CENTER - half_short)
    x1 = min(511, CENTER + half_long)
    y1 = min(511, CENTER + half_short)
    return (x0, y0, x1, y1), used_fallback


# ----------------------------------------------------------------------------
# Frame rendering
# ----------------------------------------------------------------------------
def render_frame(u8, row, frame_no, n_frames, confirmed, prob, early_call, ctx):
    """Return an RGB PIL image with box + overlay for one timestep."""
    S = SCALE
    W = 512 * S
    base = Image.fromarray(u8).convert("RGB")
    if S != 1:
        base = base.resize((W, W), Image.BICUBIC)
    cam = ctx["cams"][frame_no - 1]
    if cam is not None:
        base = Image.fromarray(apply_cam(np.asarray(base), cam))

    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    color = COL_CONFIRM if confirmed else predict_color(prob)

    (x0, y0, x1, y1), fallback = compute_box(row)
    draw_tracker_box(draw, (x0 * S, y0 * S, x1 * S, y1 * S), color, 3 * S,
                     max_arm=0.09 * W)

    f_big = get_font(13 * S)
    f_bigm = get_font(13 * S, mono=True)
    f_small = get_font(10 * S)
    f_mono = get_font(10 * S, mono=True)
    f_clock = get_font(17 * S, mono=True)
    f_tiny = get_font(7 * S)
    f_tinym = get_font(7 * S, mono=True)
    m = 10 * S

    tile_px = ctx["tile_px"]
    tile_x = W - tile_px - m - 3
    head_w = tile_x - 2 * m

    g = pd.to_numeric(row.get("grade"), errors="coerce")
    grade = 0 if pd.isna(g) else int(g)
    pressure = pd.to_numeric(row.get("pressure"), errors="coerce")
    wind = pd.to_numeric(row.get("wind"), errors="coerce")
    stamp = row_stamp(row)

    header = [
        (f"Storm {STORM_ID}  |  Himawari/GMS infrared  |  "
         f"center {float(row['lat']):.1f}N {float(row['lng']):.1f}E", (255, 255, 255)),
        (ctx["split_short"], ctx["split_color"]),
    ]
    h = text_block(draw, (m, m), fit_lines(draw, header, f_small, head_w),
                   f_small, (120, 120, 120))

    # Big UTC clock + storm-age counter, promoted out of the small header text.
    age_h = ctx["ages"][frame_no - 1]
    clock = [
        (stamp, (255, 255, 255), f_clock),
        (f"STORM AGE {fmt_hours(age_h)}   FRAME {frame_no:>4}/{n_frames}   "
         f"INFER {ctx['latencies'][frame_no - 1]:5.1f} ms/f", (200, 200, 205), f_mono),
    ]
    y = m + h + 6 * S
    h2 = text_block(draw, (m, y), fit_lines(draw, clock, f_mono, head_w),
                    f_mono, (120, 120, 120))
    y += h2 + 6 * S

    if not confirmed:
        # The model has seen nothing but this single frame's pixels.
        body = [
            ("AI PREDICTION", color, f_big),
            (f"{fmt_pct(prob)} chance of reaching typhoon strength",
             (255, 255, 255), f_bigm),
            ("input: this infrared frame only -- no future frames,", (170, 170, 170), f_small),
            ("no pressure / wind / grade given to the model", (170, 170, 170), f_small),
        ]
        if ctx["countdown"][frame_no - 1] is not None:
            body.insert(2, (f"OBSERVED TYPHOON STATUS IN: "
                            f"{fmt_hours(ctx['countdown'][frame_no - 1])}",
                            (255, 210, 90), f_mono))
    else:
        p_txt = f"{pressure:.0f} hPa" if pd.notna(pressure) and pressure > 0 else "n/a"
        w_txt = f"{wind:.0f} kt" if pd.notna(wind) and wind > 0 else "n/a"
        body = [
            ("CONFIRMED BY OBSERVATION", COL_CONFIRM, f_big),
            (f"{GRADE_MAP.get(grade, 'Grade ' + str(grade))}  |  {p_txt}  |  {w_txt}",
             (255, 255, 255), f_bigm),
        ]
        if early_call is not None:
            body.append((f"AI called it at frame {early_call[0]} "
                         f"({early_call[1]}): {fmt_pct(early_call[2])}",
                         predict_color(early_call[2]), f_mono))
            if ctx["lead_hours"]:
                body.append((f"AI CALLED IT {ctx['lead_hours']:.0f} HOURS EARLY",
                             COL_CONFIRM, f_bigm))
    text_block(draw, (m, y), fit_lines(draw, body, f_big, head_w), f_big, color)

    # ---- footer, then the instrument stack directly above it ----
    foot = [("box: best-track 30/50kt wind radii" if not fallback
             else "box: default 300 km (no wind-radius data this frame)", (200, 200, 200)),
            ("heat overlay = class activation map: where the network looked "
             "(per-frame normalised)", (255, 172, 30))]
    _fw, fh = block_size(draw, foot, f_tiny)
    foot_y = W - fh - m
    text_block(draw, (m, foot_y), foot, f_tiny, (120, 120, 120))

    axis_tile, axis_px, axis_yoff = ctx["axis"]
    ax_x, ax_y = m, foot_y - m - axis_tile.size[1]
    layer.paste(axis_tile, (ax_x, ax_y), axis_tile)
    # Only the "now" cursor is drawn per frame; the curve behind it is static.
    cx = ax_x + axis_px(frame_no - 1)
    cy0 = ax_y + axis_yoff
    cy1 = cy0 + ctx["axis_h"] - 1
    draw.line([(cx, cy0), (cx, cy1)], fill=(255, 255, 255, 150), width=1)
    cy = cy1 - (ctx["axis_h"] - 1) * float(np.clip(prob, 0, 1))
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(0, 0, 0, 255))
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=rgba(color))

    meter_h = 11 * S
    meter_y1 = ax_y - _font_height(f_tinym) - 3 * m
    draw_meter(layer, draw, (m + 2, meter_y1 - meter_h, m + 2 + ctx["axis_w"], meter_y1),
               prob, ticks=((0.25, "25"), (0.50, "50"), (0.75, "75")),
               font=f_tinym, title="model confidence this frame")

    # ---- right column: the receipts ----
    tx = tile_x
    ty = m + 3
    # 1. frame-1 thumbnail, pinned for the whole video
    _tw, th_ = draw_tile(layer, draw, (tx, ty), ctx["frame1_img"], tile_px,
                         [("FRAME 1", (220, 220, 225), f_tinym),
                          (f"AI SAID {fmt_pct(ctx['frame1_prob'])}", (255, 255, 255), f_tinym),
                          (ctx["frame1_stamp"], (175, 175, 180), f_tinym)],
                         f_tinym, predict_color(ctx["frame1_prob"]))
    ty += th_ + 4
    # 2. what the network actually sees
    _tw, th_ = draw_tile(layer, draw, (tx, ty), ctx["inputs"][frame_no - 1], tile_px,
                         [("MODEL INPUT", (220, 220, 225), f_tinym),
                          (f"{IMG_SIZE}x{IMG_SIZE} normalised", (175, 175, 180), f_tinym)],
                         f_tinym, (140, 140, 150))
    ty += th_ + 4
    # 3. locator: real best-track positions on a plain graticule
    loc = ctx["locator"].copy()
    ld = ImageDraw.Draw(loc)
    pts = ctx["locator_pts"]
    if frame_no >= 2:
        ld.line(pts[:frame_no], fill=rgba(color), width=2)
    lx, ly = pts[frame_no - 1]
    ld.ellipse([lx - 4, ly - 4, lx + 4, ly + 4], fill=(0, 0, 0, 255))
    ld.ellipse([lx - 3, ly - 3, lx + 3, ly + 3], fill=rgba(color))
    draw_tile(layer, draw, (tx, ty), loc, tile_px,
              [("TRACK", (220, 220, 225), f_tinym),
               ("best-track lat/lon", (175, 175, 180), f_tinym),
               ("no coastlines", (150, 150, 155), f_tinym)],
              f_tinym, (140, 140, 150))

    return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")


# ----------------------------------------------------------------------------
# Cards and the confirmation beat
# ----------------------------------------------------------------------------
def make_card(size, heading, sublines, accent, tiles=(), footer_lines=()):
    rw, rh = size
    img = Image.new("RGBA", (rw, rh), (8, 9, 12, 255))
    draw = ImageDraw.Draw(img)
    m = 10 * SCALE
    h_size = 17 * SCALE
    f_head = get_font(h_size)
    f_body = get_font(10 * SCALE)
    f_tiny = get_font(7 * SCALE)
    f_tinym = get_font(7 * SCALE, mono=True)

    draw.rectangle([0, 0, rw, 4 * SCALE], fill=rgba(accent))
    y = int(rh * 0.07)
    # Shrink the heading until it fits rather than letting it run off frame.
    while h_size > 9 * SCALE and draw.textlength(heading, font=f_head) > rw - 4 * m:
        h_size = int(h_size * 0.88)
        f_head = get_font(h_size)
    draw.text((2 * m, y), fit_text(draw, heading, f_head, rw - 4 * m),
              fill=rgba(accent), font=f_head)
    y += _font_height(f_head) + m
    for item in sublines:
        txt, col, fnt = _norm_line(item, f_body)
        draw.text((2 * m, y), fit_text(draw, txt, fnt, rw - 4 * m),
                  fill=rgba(col or (218, 218, 224)), font=fnt)
        y += _font_height(fnt) + 6

    # Lay the footer out first so tiles fit into whatever space is really left.
    foot = list(footer_lines)
    foot_h = sum(_font_height(_norm_line(i, f_tiny)[2]) + 4 for i in foot) + 2 * m
    if tiles:
        cap_h = max(block_size(draw, cap, f_tinym, pad=4)[1] for _im, cap in tiles) + 8
        top, bot = y + m, rh - foot_h - m
        tsize = int(min(0.28 * rw, 0.34 * rh, max(56, bot - top - cap_h)))
        ty = max(top, bot - cap_h - tsize)
        tx = 2 * m
        for tile_img, caption in tiles:
            draw_tile(img, draw, (tx, ty), tile_img, tsize, caption, f_tinym, accent)
            tx += tsize + 3 * m

    fy = rh - 2 * m
    for item in reversed(foot):
        txt, col, fnt = _norm_line(item, f_tiny)
        fy -= _font_height(fnt) + 4
        draw.text((2 * m, fy), fit_text(draw, txt, fnt, rw - 4 * m),
                  fill=rgba(col or (150, 150, 158)), font=fnt)
    return img.convert("RGB")


def beat_overlay(img, phase, banner, color):
    """One frame of the confirmation hold: a pulsing border ring plus a single
    restrained banner line. No sound, no klaxon. `phase` in [0,1) is the
    position within the hold, so the pulse runs off the hold's own clock."""
    out = img.convert("RGBA")
    rw, rh = out.size
    layer = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    ring = max(4, int(0.011 * max(rw, rh)))
    if (int(phase * 6.0) % 2) == 0:
        draw.rectangle([ring // 2, ring // 2, rw - 1 - ring // 2, rh - 1 - ring // 2],
                       outline=rgba(color, 240), width=ring)
        draw.rectangle([ring * 2, ring * 2, rw - 1 - ring * 2, rh - 1 - ring * 2],
                       outline=rgba(color, 110), width=max(1, ring // 3))
    f = get_font(14 * SCALE, mono=True)
    w, _h = block_size(draw, [(banner, color, f)], f, pad=10)
    text_block(draw, ((rw - w) // 2, int(rh * 0.55)), [(banner, color, f)], f,
               color, pad=10, alpha=230)
    return Image.alpha_composite(out, layer).convert("RGB")


def credibility_lines(cfg, vm, font=None):
    """Real held-out numbers, straight out of the checkpoint. Nothing recomputed:
    if the checkpoint does not carry a field, it is not claimed."""
    out = []
    auc = vm.get("auc")
    ece = vm.get("val_ece")
    ops = cfg.get("recommended_display_thresholds") or {}
    if auc is not None:
        n_b = len(cfg.get("basins", []) or [])
        out.append((f"held-out AUC          {auc:.3f}"
                    + (f"  ({n_b} basins)" if n_b else ""), (215, 215, 220), font))
    op = ops.get("specificity_0.90") or {}
    if op.get("false_positive_rate") is not None:
        out.append((f"false-positive rate   {op['false_positive_rate'] * 100:.1f}%  "
                    f"at the {op.get('threshold', 0.0):.2f} operating point",
                    (215, 215, 220), font))
    if op.get("precision") is not None:
        out.append((f"precision there       {op['precision'] * 100:.1f}%",
                    (215, 215, 220), font))
    if ece is not None:
        out.append((f"calibration error     {ece:.3f} ECE  (post Platt scaling)",
                    (215, 215, 220), font))
    return out


# ----------------------------------------------------------------------------
# Video writing
# ----------------------------------------------------------------------------
def write_video(frames, path, fps):
    """Try H.264 via imageio-ffmpeg first (plays in browsers and slide decks),
    fall back to OpenCV's mp4v (plays in VLC / most desktop players)."""
    try:
        import imageio.v2 as imageio
        w = imageio.get_writer(path, fps=fps, codec="libx264",
                               quality=8, macro_block_size=1)
        for f in frames:
            w.append_data(np.asarray(f))
        w.close()
        print(f"Wrote {path} via imageio/libx264")
        return True
    except Exception as exc:
        print(f"imageio/libx264 failed ({exc}); falling back to cv2 mp4v")

    try:
        import cv2
        h, w = np.asarray(frames[0]).shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError("cv2.VideoWriter did not open")
        for f in frames:
            vw.write(np.asarray(f)[:, :, ::-1])  # PIL is RGB, OpenCV wants BGR
        vw.release()
        print(f"Wrote {path} via cv2/mp4v")
        return True
    except Exception as exc:
        print(f"cv2 fallback also failed: {exc}")
        return False


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    global TEMP_LO, TEMP_HI, IMG_SIZE, NORM_MEAN, NORM_STD, TYPHOON_GRADE

    os.makedirs(OUT_DIR, exist_ok=True)

    root = find_typhoon_root()
    if not root:
        raise SystemExit("No typhoon dataset under /kaggle/input -- check dataset_sources")
    meta_dir, image_dir = resolve_dirs(root)
    print("ROOT =", root)

    ckpt_path = find_checkpoint(root)
    if not ckpt_path:
        raise SystemExit(
            "No .pt checkpoint found under /kaggle/input. Attach the training "
            "kernel's model_best.pt as a second dataset (or kernel) source."
        )
    print("checkpoint =", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    TEMP_LO = cfg.get("temp_lo", TEMP_LO)
    TEMP_HI = cfg.get("temp_hi", TEMP_HI)
    IMG_SIZE = cfg.get("img_size", IMG_SIZE)
    NORM_MEAN = cfg.get("norm_mean", NORM_MEAN)
    NORM_STD = cfg.get("norm_std", NORM_STD)
    TYPHOON_GRADE = cfg.get("typhoon_grade", TYPHOON_GRADE)
    vm = ckpt.get("val_metrics", {}) or {}
    print("val metrics from checkpoint:", json.dumps(vm, indent=2))

    # Honesty check: a demo shown to judges should use a storm the model never
    # trained on, otherwise the confidence curve is partly memorisation. The
    # verdict is whatever THIS checkpoint says -- it is printed here and rendered
    # verbatim on the title card and on every frame's header, never assumed.
    val_ids = {str(s) for s in (ckpt.get("val_storm_ids") or [])}
    train_ids = {str(s) for s in (ckpt.get("train_storm_ids") or [])}
    if STORM_ID in val_ids:
        split_verdict = f"HELD-OUT: storm {STORM_ID} is in the VALIDATION split -- never seen in training"
        split_short = "HELD-OUT -- validation split, never seen in training"
        split_color = COL_CONFIRM
        print(f"OK: storm {STORM_ID} is in the VALIDATION split (unseen during training)")
    elif STORM_ID in train_ids:
        split_verdict = (f"WARNING: storm {STORM_ID} is in this checkpoint's TRAINING "
                         f"split -- model has seen it")
        split_short = "WARNING: TRAINING SPLIT -- model has seen this storm"
        split_color = (255, 90, 90)
        print(f"WARNING: storm {STORM_ID} was in the TRAINING split. The model has "
              f"seen its early frames. Pick a storm from ckpt['val_storm_ids'] "
              f"for an honest demo.")
    else:
        split_verdict = (f"storm {STORM_ID} is in neither split (no usable early "
                         f"frames) -- never trained on")
        split_short = "not in either split -- never trained on"
        split_color = (200, 200, 205)
        print(f"NOTE: storm {STORM_ID} appears in neither split (no usable early "
              f"frames), so it was never trained on.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_model()
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    cam_tap = CamTap(net)
    print("Device:", device)

    # ---- storm track, chronological ----
    csv_path = os.path.join(meta_dir, f"{STORM_ID}.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"No metadata CSV for storm {STORM_ID} at {csv_path}")
    df = pd.read_csv(csv_path)
    df["file_1"] = df["file_1"].fillna("").astype(str).str.strip()
    df = df[df["file_1"] != ""]
    df = df.sort_values(["year", "month", "day", "hour"]).reset_index(drop=True)
    print(f"Storm {STORM_ID}: {len(df)} imaged frames, "
          f"max grade {pd.to_numeric(df['grade'], errors='coerce').max()}")

    # ---- load frames + run the model, one frame at a time (as trained) ----
    frames_u8, rows = [], []
    for _, r in df.iterrows():
        p = os.path.join(image_dir, STORM_ID, r["file_1"])
        if not os.path.exists(p):
            print(f"  missing image, skipping: {p}")
            continue
        try:
            frames_u8.append(load_frame_u8(p))
            rows.append(r)
        except Exception as exc:
            print(f"  LOAD FAIL {p}: {exc}")
    if not frames_u8:
        raise SystemExit(f"No loadable frames for storm {STORM_ID}")
    print(f"Loaded {len(frames_u8)} frames")

    n = len(frames_u8)
    probs = np.zeros(n, np.float32)
    tta_lo = np.zeros(n, np.float32)
    tta_hi = np.zeros(n, np.float32)
    latencies = np.zeros(n, np.float32)
    cams = [None] * n
    inputs = [None] * n

    # The forward pass runs ONE FRAME AT A TIME so the on-screen latency readout
    # is a genuine per-frame measurement rather than a batch average smeared
    # across 32 frames.
    with torch.no_grad():
        for i in range(n):
            x = to_model_input(np.stack([frames_u8[i]])).to(device)
            t0 = time.perf_counter()
            logit = net(x)
            latencies[i] = (time.perf_counter() - t0) * 1000.0
            probs[i] = float(torch.sigmoid(logit).squeeze().item())
            cams[i] = cam_tap.maps()[0]
            inputs[i] = model_input_tile(x)

            # Test-time-augmentation uncertainty band. ROTATIONS ONLY, NEVER
            # FLIPS: Southern-Hemisphere storms have genuinely opposite
            # chirality, so a flipped storm is a physically different object and
            # flips are banned everywhere else in this project. An exact 90-degree
            # rotation is lossless and chirality-preserving, so it is a fair
            # perturbation. This NEVER changes the number shown on screen -- the
            # displayed probability is always the un-rotated forward above -- it
            # only produces the spread band behind the confidence curve.
            if TTA_ROTATIONS > 1:
                vals = [probs[i]]
                for k in range(1, min(4, TTA_ROTATIONS)):
                    xr = torch.rot90(x, k, dims=(2, 3))
                    vals.append(float(torch.sigmoid(net(xr)).squeeze().item()))
                tta_lo[i], tta_hi[i] = min(vals), max(vals)
            else:
                tta_lo[i] = tta_hi[i] = probs[i]
            if (i + 1) % 50 == 0:
                print(f"  inferred {i + 1}/{n}")
    cam_tap.close()

    # ---- narrative timeline, all from real best-track columns ----
    grades = pd.to_numeric(pd.DataFrame(rows)["grade"], errors="coerce").fillna(0).to_numpy()
    # Latch: once the storm has genuinely reached typhoon grade, every later frame
    # stays in "confirmed" mode (including the extratropical decay tail), so the
    # video never flips back to claiming a prediction about an already-resolved storm.
    confirmed_from = None
    for i, g in enumerate(grades):
        if g >= TYPHOON_GRADE:
            confirmed_from = i
            break

    times = [row_time(r) for r in rows]
    t0 = times[0]
    ages = [((t - t0).total_seconds() / 3600.0 if (t and t0) else None) for t in times]
    confirm_time = times[confirmed_from] if confirmed_from is not None else None
    countdown = [None] * n
    if confirm_time is not None:
        for i, t in enumerate(times):
            if t is not None and i < confirmed_from:
                countdown[i] = (confirm_time - t).total_seconds() / 3600.0
    lead_hours = ((confirm_time - t0).total_seconds() / 3600.0
                  if (confirm_time and t0 and confirm_time > t0) else None)

    # What the model said on the very first (weakest, least organised) frame --
    # this is the "early call" the video is built around.
    early_call = (1, row_stamp(rows[0]), float(probs[0]))
    print(f"Early call (frame 1, {early_call[1]}): {probs[0] * 100:.1f}%")
    if confirmed_from is None:
        print("This storm never reaches grade >= 5 -- the whole video stays in "
              "prediction mode (fine for a negative-example clip).")
    else:
        msg = (f"Real typhoon status reached at frame {confirmed_from + 1} "
               f"({row_stamp(rows[confirmed_from])})")
        if lead_hours:
            msg += f"; lead time from frame 1 = {lead_hours:.0f} h"
        print(msg)
    print(f"Inference: median {np.median(latencies):.1f} ms/frame "
          f"(min {latencies.min():.1f}, max {latencies.max():.1f}) on {device}")
    print(f"Rotation-TTA spread: mean {(tta_hi - tta_lo).mean() * 100:.1f} pp, "
          f"max {(tta_hi - tta_lo).max() * 100:.1f} pp")

    # ---- pre-render everything that is static ----
    W = 512 * SCALE
    m = 10 * SCALE
    f_tinym = get_font(7 * SCALE, mono=True)
    tile_px = int(0.15 * W)
    axis_w = W - 2 * m - tile_px - 3 * m
    axis_h = 11 * SCALE * 5
    axis = render_axis(
        (axis_w, axis_h), probs, band=(tta_lo, tta_hi),
        thresh=0.5, thresh_label="p = 0.50",
        marker=confirmed_from,
        marker_label=("observed typhoon grade" if confirmed_from is not None else None),
        font=f_tinym,
        title="model output, every frame (raw) + rotation-TTA spread")

    lats = [float(r["lat"]) for r in rows]
    lngs = [float(r["lng"]) for r in rows]
    locator, _proj, loc_pts = make_locator(lats, lngs, tile_px, f_tinym)
    loc_pts = [(p[0], p[1]) for p in loc_pts]

    ctx = {
        "cams": cams, "inputs": inputs, "latencies": latencies,
        "ages": ages, "countdown": countdown, "lead_hours": lead_hours,
        "split_verdict": split_verdict, "split_short": split_short,
        "split_color": split_color,
        "axis": axis, "axis_w": axis_w, "axis_h": axis_h,
        "tile_px": tile_px,
        "frame1_img": Image.fromarray(frames_u8[0]).convert("RGB"),
        "frame1_prob": float(probs[0]),
        "frame1_stamp": row_stamp(rows[0]),
        "locator": locator, "locator_pts": loc_pts,
    }

    # ---- render ----
    if SAVE_FRAMES:
        os.makedirs(FRAME_DIR, exist_ok=True)

    f_mono = get_font(10 * SCALE, mono=True)
    f_tiny = get_font(7 * SCALE)

    images = []

    # Opening card: what the viewer is about to see, and the split verdict.
    title_subs = [
        ("live model output on one real storm's full track", (200, 200, 205)),
        (f"Storm {STORM_ID}  |  {len(rows)} infrared frames  |  "
         f"{row_stamp(rows[0])}  ->  {row_stamp(rows[-1])}", (255, 255, 255)),
        (split_verdict, split_color),
        ("", (0, 0, 0)),
        ("PHASE 1  AI PREDICTION -- the model sees ONE infrared frame at a time.",
         (235, 235, 240)),
        ("          No future frames. No pressure, wind or grade. It outputs",
         (185, 185, 190)),
        ("          P(this storm later reaches typhoon strength, wind >= 64 kt).",
         (185, 185, 190)),
        ("", (0, 0, 0)),
        ("PHASE 2  CONFIRMED BY OBSERVATION -- the real best-track record reaches",
         (235, 235, 240)),
        ("          grade >= 5. The overlay switches to observed ground truth and",
         (185, 185, 190)),
        ("          recaps what the AI said back at the very first frame.",
         (185, 185, 190)),
        ("", (0, 0, 0)),
        ("The heat overlay is a class activation map: where the network looked.",
         (255, 172, 30)),
        ("Every number on screen is a measured value. Nothing is animated.",
         (170, 170, 175)),
    ]
    if lead_hours:
        title_subs.insert(3, (f"the AI's first call comes {lead_hours:.0f} hours before "
                              f"observation confirms it", (255, 210, 90), f_mono))
    # Accent = the split verdict, so the very first thing on screen carries the
    # honesty status: green if the storm is genuinely held out, red if not.
    title_card = make_card((W, W), "CYCLONE EARLY-INTENSIFICATION",
                           title_subs, split_color,
                           footer_lines=[("Cyclone Genesis Predictor  |  IMD problem statement  |  "
                                          "resnet18 (1-ch) on Himawari/GMS infrared",
                                          (150, 150, 158), f_tiny)]
                                        + credibility_lines(cfg, vm, f_tinym))
    images += [title_card] * max(1, int(round(FPS * TITLE_SECONDS)))

    for i, (u8, r) in enumerate(zip(frames_u8, rows)):
        confirmed = confirmed_from is not None and i >= confirmed_from
        img = render_frame(u8, r, i + 1, len(rows), confirmed, float(probs[i]),
                           early_call if confirmed else None, ctx)
        images.append(img)
        if SAVE_FRAMES:
            img.save(os.path.join(FRAME_DIR, f"frame_{i:04d}.png"))

        # Threshold-crossing beat: hold ~1 s on the real confirmation frame with
        # a pulsing border and one banner line. No sound.
        if confirmed_from is not None and i == confirmed_from:
            banner = f"INTENSIFICATION CONFIRMED  {row_stamp(r)}"
            n_beat = max(1, int(round(FPS * BEAT_SECONDS)))
            for k in range(n_beat):
                images.append(beat_overlay(img, k / float(n_beat), banner, COL_CONFIRM))
        if (i + 1) % 25 == 0:
            print(f"  rendered {i + 1}/{len(rows)}")

    # Closing card: the receipts.
    peak_i = int(np.nanargmax(pd.to_numeric(pd.DataFrame(rows)["wind"],
                                            errors="coerce").fillna(-1).to_numpy()))
    pr = rows[peak_i]
    pw = pd.to_numeric(pr.get("wind"), errors="coerce")
    pp = pd.to_numeric(pr.get("pressure"), errors="coerce")
    end_subs = [
        (f"first frame       {row_stamp(rows[0])}   AI said {fmt_pct(probs[0])}",
         (235, 235, 240), f_mono),
    ]
    if confirmed_from is not None:
        end_subs.append((f"observed typhoon  {row_stamp(rows[confirmed_from])}   "
                         f"frame {confirmed_from + 1}/{len(rows)}",
                         (235, 235, 240), f_mono))
    end_subs += [
        (f"observed peak     {row_stamp(pr)}   "
         f"{pw:.0f} kt / {pp:.0f} hPa" if pd.notna(pw) and pd.notna(pp) else
         f"observed peak     {row_stamp(pr)}", (235, 235, 240), f_mono),
        (f"median inference  {np.median(latencies):5.1f} ms/frame on "
         f"{str(device).upper()} resnet18 @ {IMG_SIZE}x{IMG_SIZE}",
         (200, 200, 205), f_mono),
        (f"rotation-TTA spread  mean {(tta_hi - tta_lo).mean() * 100:4.1f} pp",
         (200, 200, 205), f_mono),
    ]
    if lead_hours:
        end_subs += [
            ("", (0, 0, 0)),
            (f"AI CALLED IT {lead_hours:.0f} HOURS EARLY", COL_CONFIRM,
                get_font(14 * SCALE)),
            ("hindsight-computed lead time on this storm's own best-track record.",
             (185, 185, 190)),
            ("It is not a validated operational forecast-skill figure, and it is",
             (185, 185, 190)),
            ("one storm, not a score.", (185, 185, 190)),
        ]
    pg = pd.to_numeric(pr.get("grade"), errors="coerce")
    pg_txt = GRADE_MAP.get(int(pg), f"Grade {int(pg)}") if pd.notna(pg) else "grade n/a"
    end_tiles = [(ctx["frame1_img"],
                  [("FRAME 1", (220, 220, 225), f_tinym),
                   (f"AI SAID {fmt_pct(probs[0])}", (255, 255, 255), f_tinym),
                   (row_stamp(rows[0]), (175, 175, 180), f_tinym)]),
                 (Image.fromarray(frames_u8[peak_i]).convert("RGB"),
                  [("OBSERVED PEAK", (220, 220, 225), f_tinym),
                   (pg_txt, (255, 255, 255), f_tinym),
                   (row_stamp(pr), (175, 175, 180), f_tinym)])]
    end_card = make_card((W, W), "RECEIPTS", end_subs,
                         COL_CONFIRM if confirmed_from is not None else (200, 200, 205),
                         tiles=end_tiles,
                         footer_lines=[(split_verdict, split_color, f_tiny),
                                       ("p = P(early-stage system intensifies to typhoon "
                                        "strength). NOT a cyclone-presence detector.",
                                        (150, 150, 158), f_tiny)]
                                      + credibility_lines(cfg, vm, f_tinym))
    images += [end_card] * max(1, int(round(FPS * ENDCARD_SECONDS)))

    ok = write_video(images, VIDEO_PATH, FPS)
    if not ok and not SAVE_FRAMES:
        raise SystemExit("Video encoding failed and frames were not saved.")

    summary = {
        "storm_id": STORM_ID,
        "n_frames": len(rows),
        "n_video_frames": len(images),
        "fps": FPS,
        "split_verdict": split_verdict,
        "early_call_prob": float(probs[0]),
        "early_call_stamp": row_stamp(rows[0]),
        "mean_prob_prediction_phase": float(
            probs[:confirmed_from].mean() if confirmed_from else probs.mean()
        ),
        "confirmed_from_frame": (confirmed_from + 1) if confirmed_from is not None else None,
        "confirmed_stamp": row_stamp(rows[confirmed_from]) if confirmed_from is not None else None,
        "lead_time_hours": lead_hours,
        "lead_time_note": ("hindsight-computed from this storm's own best-track record; "
                           "not a validated operational forecast-skill figure"),
        "inference_ms_per_frame": {
            "median": float(np.median(latencies)),
            "min": float(latencies.min()),
            "max": float(latencies.max()),
            "device": str(device),
            "note": "measured per frame, forward pass only, batch size 1",
        },
        "tta": {
            "rotations": int(min(4, TTA_ROTATIONS)),
            "note": "rotations only, never flips (SH storms have opposite chirality); "
                    "display band only, never changes the reported probability",
            "mean_spread": float((tta_hi - tta_lo).mean()),
            "max_spread": float((tta_hi - tta_lo).max()),
        },
        "probs": [round(float(p), 4) for p in probs],
        "val_metrics": vm,
        "video_written": ok,
    }
    with open(os.path.join(OUT_DIR, "demo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Done. Wrote", VIDEO_PATH, "and demo_summary.json")


if __name__ == "__main__":
    main()
