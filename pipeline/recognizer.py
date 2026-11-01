"""Stage 1 of the grounded pipeline: RECOGNISE surface features in HiRISE before anything is filled in.

A small U-Net reads four co-registered HiRISE products at 6.31 cm (multi-frame restoration, shift-and-average,
single-frame deconvolution, bicubic) and predicts, per pixel:
  * rock-centre heatmaps for two size bands (25-40 cm and >= 50 cm), trained on rocks found in the rover truth;
  * surface class logits (soil, bedrock, sand, big rock), trained where the rover truth carries AI4Mars classes.
Losses are masked to rover-verified ground, so the network is only ever taught by what the rover actually saw.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage as ndi

BANDS = [(25, 40), (50, 100)]                 # rock size bands (cm) -> heatmap channels
BAND_D = [32, 60]                             # representative diameter (cm) reported for a decoded rock
INPUTS = ["restored_multi", "average", "deconv_1frame", "bicubic"]
NCLS = 4


def _std(a):
    a = a.astype(np.float32); return (a - a.mean()) / (a.std() + 1e-6)


def stop_arrays(restored, truth, detect, px_cm):
    """Inputs (C,H,W), rock heatmap targets (2,H,W), class target (H,W; -1 unknown), loss mask (H,W) and the truth rock list."""
    X = np.stack([_std(restored[k]) for k in INPUTS])
    n = X.shape[1]; k = 5.0 / px_cm
    z5 = truth["z"].astype(float)
    zt = ndi.zoom(np.nan_to_num(z5), k, order=1); vt = ndi.zoom(np.isfinite(z5).astype(float), k, order=1) > 0.99
    c5 = truth["cls"].astype(float)
    ct = ndi.zoom(np.nan_to_num(c5, nan=-1), k, order=0)
    t = zt.shape[0]; o = (n - t) // 2
    valid = np.zeros((n, n), bool); valid[o:o + t, o:o + t] = ndi.binary_erosion(vt, iterations=4)
    cls = np.full((n, n), -1, np.int64); cc = np.where(vt & (ct >= 0), ct, -1); cls[o:o + t, o:o + t] = cc.astype(np.int64)
    tz = np.where(vt, zt, np.nanmean(np.where(vt, zt, np.nan)))
    rocks = [(y + o, x + o, D) for y, x, D, _ in detect(tz) if valid[y + o, x + o] and D >= 25]
    H = np.zeros((len(BANDS), n, n), np.float32)
    yy, xx = np.mgrid[0:n, 0:n]
    for y, x, D in rocks:
        b = next(i for i, (lo, hi) in enumerate(BANDS) if D <= hi) if D <= BANDS[-1][1] else len(BANDS) - 1
        s = max(1.5, 0.25 * D / px_cm)
        y0, y1, x0, x1 = max(0, y - 4 * int(s) - 2), min(n, y + 4 * int(s) + 3), max(0, x - 4 * int(s) - 2), min(n, x + 4 * int(s) + 3)
        g = np.exp(-((yy[y0:y1, x0:x1] - y) ** 2 + (xx[y0:y1, x0:x1] - x) ** 2) / (2 * s * s))
        H[b, y0:y1, x0:x1] = np.maximum(H[b, y0:y1, x0:x1], g)
    return X, H, cls, valid, rocks


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                               nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    def forward(self, x): return self.f(x)


class UNet(nn.Module):
    def __init__(self, cin=len(INPUTS), w=16):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = Block(cin, w), Block(w, 2 * w), Block(2 * w, 4 * w), Block(4 * w, 8 * w)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * w, 4 * w, 2, 2), Block(8 * w, 4 * w)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2), Block(4 * w, 2 * w)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * w, w, 2, 2), Block(2 * w, w)
        self.rock = nn.Conv2d(w, len(BANDS), 1); self.cls = nn.Conv2d(w, NCLS, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2)); e4 = self.e4(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1)); d2 = self.d2(torch.cat([self.u2(d3), e2], 1)); d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.rock(d1), self.cls(d1)


def loss_fn(out, H, C, M, pos_w=20.0):
    rock, cls = out
    m = M.unsqueeze(1).float()
    w = 1 + pos_w * H                                      # rocks are rare: up-weight their neighbourhoods
    lr = (F.binary_cross_entropy_with_logits(rock, H, weight=w, reduction="none") * m).sum() / (m.sum() * H.shape[1] + 1e-6)
    lc = F.cross_entropy(cls, C, ignore_index=-1) if (C >= 0).any() else rock.sum() * 0
    return lr + 0.3 * lc


def predict(model, X, tile=256, pad=32):
    """Full-image prediction with overlapping tiles. Returns rock probabilities (2,H,W) and class probabilities (4,H,W)."""
    model.eval(); C, H, W = X.shape
    pr = np.zeros((len(BANDS), H, W), np.float32); pc = np.zeros((NCLS, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    Xp = np.pad(X, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    with torch.no_grad():
        for y in range(0, H, tile - 2 * pad):
            for x in range(0, W, tile - 2 * pad):
                y0, x0 = min(y, H - tile + 2 * pad), min(x, W - tile + 2 * pad)
                inp = torch.from_numpy(Xp[None, :, y0:y0 + tile, x0:x0 + tile].copy())
                r, c = model(inp); r = torch.sigmoid(r)[0].numpy(); c = torch.softmax(c, 1)[0].numpy()
                s = slice(pad, tile - pad)
                pr[:, y0:y0 + tile - 2 * pad, x0:x0 + tile - 2 * pad] += r[:, s, s]
                pc[:, y0:y0 + tile - 2 * pad, x0:x0 + tile - 2 * pad] += c[:, s, s]
                cnt[y0:y0 + tile - 2 * pad, x0:x0 + tile - 2 * pad] += 1
    return pr / np.maximum(cnt, 1), pc / np.maximum(cnt, 1)


def decode(prob, thr, min_dist=4):
    """Rock list (y, x, D_cm, score) from heatmaps: local maxima above thr; one rock per location (largest band wins ties)."""
    out = []
    best = prob.max(0); arg = prob.argmax(0)
    pk = (best == ndi.maximum_filter(best, size=2 * min_dist + 1)) & (best > thr)
    for y, x in zip(*np.nonzero(pk)): out.append((int(y), int(x), BAND_D[arg[y, x]], float(best[y, x])))
    return out
