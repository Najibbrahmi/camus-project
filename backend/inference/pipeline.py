"""
The bridge: load the optimized .pth weights once, segment the 4 frames, run
biplane Simpson -> EF/volumes, and return the masks (base64 PNG) for the UI.

Switch architectures with MODEL_KIND:
  "unet"  -> best_model.pth                       (input 256)   [default]
  "vit"   -> results/mae_finetuned_*.pth + MAE base (input 224)
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from .models import UNet, ViTSegmentationModel
from .preprocessing import preprocess_frame
from .simpson import compute_ef, get_largest_cc

FRAME_ORDER = ("twoch_ed", "twoch_es", "fourch_ed", "fourch_es")

# --- choose the model the bridge serves -----------------------------------
MODEL_KIND = os.getenv("MODEL_KIND", "unet")
UNET_CKPT = os.getenv("UNET_CKPT", "../best_model.pth")
VIT_CKPT = os.getenv("VIT_CKPT", "../results/mae_finetuned_10pct_seed42.pth")
MAE_BASE = os.getenv("MAE_BASE", "../mae_pretrained_full.pth")
INPUT_SIZE = {"unet": 256, "vit": 224}


class InferenceError(RuntimeError):
    pass


def classify_ef(ef: float) -> tuple[str, str]:
    if ef >= 52: return "Normal", "normal"
    if ef >= 41: return "Mildly reduced", "borderline"
    if ef >= 30: return "Moderately reduced", "reduced"
    return "Severely reduced", "severe"


@dataclass
class EFResult:
    ef: float
    confidence: float
    label: str
    severity: str
    edv_ml: Optional[float] = None
    esv_ml: Optional[float] = None
    calibration: str = "default"            # native | mixed | default
    masks: Optional[Dict[str, str]] = None  # frame -> base64 PNG overlay (for the UI)

    def to_dict(self) -> dict:
        return asdict(self)


def _mask_to_png_b64(mask: np.ndarray) -> str:
    """Binary mask -> transparent green PNG (data URI body) for overlay in the UI."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask > 0] = (39, 174, 96, 140)     # green, semi-transparent
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Pipeline:
    def __init__(self) -> None:
        self.model = None
        self.kind = MODEL_KIND
        self.size = INPUT_SIZE[MODEL_KIND]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False

    def load(self) -> None:
        if self.kind == "unet":
            model = UNet(n_classes=2).to(self.device)
            state = torch.load(UNET_CKPT, map_location=self.device, weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
        else:
            model = ViTSegmentationModel(MAE_BASE, n_classes=2).to(self.device)
            model.load_state_dict(torch.load(VIT_CKPT, map_location=self.device, weights_only=True))
        model.eval()
        self.model = model
        self.loaded = True

    @torch.no_grad()
    def _segment(self, raw: bytes):
        x, spacing = preprocess_frame(raw, self.size)
        logits = self.model(x.to(self.device))
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        return get_largest_cc(pred), spacing

    def predict(self, images: Dict[str, bytes]) -> EFResult:
        if self.model is None:
            raise InferenceError("Model is not loaded.")
        try:
            seg = {k: self._segment(images[k]) for k in FRAME_ORDER}
        except Exception as e:
            raise InferenceError(f"Segmentation failed: {e}") from e
        masks = {k: m for k, (m, _) in seg.items()}
        spacings = {k: sp for k, (_, sp) in seg.items()}

        try:
            vols = compute_ef(masks, spacings)
        except ValueError as e:
            raise InferenceError(str(e)) from e

        ef = max(0.0, min(100.0, vols["ef"]))
        label, severity = classify_ef(ef)
        n_header = sum(1 for sp in spacings.values() if sp is not None)
        calibration = "native" if n_header == 4 else "default" if n_header == 0 else "mixed"
        plausible = sum(1 for m in masks.values() if 0.003 < float(m.mean()) < 0.6)
        confidence = round(0.6 + 0.1 * plausible, 2)

        return EFResult(ef=ef, confidence=confidence, label=label, severity=severity,
                        edv_ml=vols["edv_ml"], esv_ml=vols["esv_ml"], calibration=calibration,
                        masks={k: _mask_to_png_b64(m) for k, m in masks.items()})


pipeline = Pipeline()
