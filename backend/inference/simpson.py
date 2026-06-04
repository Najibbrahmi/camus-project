"""
Biplane Simpson's method of disks — mirrors the metric logic in your notebooks.
EDV = volume(2CH_ED, 4CH_ED); ESV = volume(2CH_ES, 4CH_ES); EF = (EDV-ESV)/EDV*100.
Per-view spacing comes from the NIfTI header; None falls back to SPACING.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.ndimage import label as cc_label
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

SPACING = (0.3, 0.3)   # mm/px fallback (row, col) when no header metadata
NUM_DISKS = 20
Spacing = tuple[float, float]


def get_largest_cc(mask: np.ndarray) -> np.ndarray:
    if not _HAS_SCIPY or mask.sum() == 0:
        return mask
    lbl, n = cc_label(mask)
    if n == 0:
        return mask
    return (lbl == np.bincount(lbl.flat)[1:].argmax() + 1).astype(np.uint8)


def _disk_diameters(mask, spacing, num_disks=NUM_DISKS):
    ys, _ = np.where(mask > 0.5)
    if len(ys) == 0:
        return None, None
    y0, y1 = ys.min(), ys.max()
    h_px = y1 - y0
    if h_px == 0:
        return None, None
    h_mm = (h_px / num_disks) * spacing[0]
    diam = [int((mask[int(y0 + (i + 0.5) * h_px / num_disks), :] > 0.5).sum()) * spacing[1]
            for i in range(num_disks)]
    return diam, h_mm


def biplane_volume(m2, m4, sp2: Optional[Spacing] = None, sp4: Optional[Spacing] = None) -> float:
    d2, h2 = _disk_diameters(m2, sp2 or SPACING)
    d4, h4 = _disk_diameters(m4, sp4 or SPACING)
    if d2 is None or d4 is None:
        return 0.0
    h = 0.5 * (h2 + h4)
    return sum((np.pi / 4) * a * b * h for a, b in zip(d2, d4)) / 1000.0  # mm^3 -> mL


def compute_ef(masks: dict, spacings: Optional[dict] = None) -> dict:
    """masks/spacings keyed by twoch_ed/twoch_es/fourch_ed/fourch_es."""
    sp = spacings or {}
    edv = biplane_volume(masks["twoch_ed"], masks["fourch_ed"], sp.get("twoch_ed"), sp.get("fourch_ed"))
    esv = biplane_volume(masks["twoch_es"], masks["fourch_es"], sp.get("twoch_es"), sp.get("fourch_es"))
    if edv <= 0:
        raise ValueError("Left ventricle not detected in the end-diastole frames.")
    ef = (edv - esv) / edv * 100.0
    return {"ef": float(round(ef, 1)),
            "edv_ml": float(round(edv, 1)),
            "esv_ml": float(round(esv, 1))}
