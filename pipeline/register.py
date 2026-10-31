"""Sub-pixel registration of overlapping frames.

Stage 1: phase correlation gives a coarse (~0.1-0.3 px) estimate, robust to large shifts.
Stage 2: ECC (enhanced correlation coefficient) refines it to ~0.01 px.

Convention (matches synth / fuse): returned d satisfies img(x) ~= ref(x + d).
"""
import numpy as np
import cv2


def _phase(ref, img):
    # cv2.phaseCorrelate applies the window IN PLACE; callers must pass private copies.
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    (sx, sy), _ = cv2.phaseCorrelate(ref, img, win)
    return -sx, -sy  # phaseCorrelate gives b(x) = a(x - s); our d = -s


def _ecc(ref, img, init):
    # findTransformECC's translation warp has the opposite sign to our d.
    w = np.array([[1, 0, -init[0]], [0, 1, -init[1]]], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
    _, w = cv2.findTransformECC(ref, img, w, cv2.MOTION_TRANSLATION, crit, None, 5)
    return -float(w[0, 2]), -float(w[1, 2])


def estimate_shift(ref, img, refine=True):
    ref = np.array(ref, dtype=np.float32, copy=True, order="C")
    img = np.array(img, dtype=np.float32, copy=True, order="C")
    d = _phase(ref.copy(), img.copy())
    if not refine:
        return d
    try:
        return _ecc(ref.copy(), img.copy(), d)
    except cv2.error:
        return d  # ECC failed to converge; keep the coarse estimate


def register_stack(frames, refine=True):
    """Shifts of every frame relative to frames[0]."""
    return [(0.0, 0.0)] + [estimate_shift(frames[0], f, refine) for f in frames[1:]]
