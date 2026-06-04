"""
Decode an uploaded echo frame -> model tensor, mirroring the notebook pipeline:
    decode -> resize (bilinear) -> per-image min-max [0,1] -> (1,1,size,size)
Also returns the NIfTI header spacing (scaled to the model grid) so volumes are
physically calibrated; PNG/JPG carry no spacing -> None (Simpson falls back).
Format is sniffed from magic bytes, so filenames don't have to be trusted.
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

NORM_EPS = 1e-8


class PreprocessingError(ValueError):
    """Raised when an upload can't be decoded into a valid 2D image."""


def _decode(raw: bytes) -> tuple[np.ndarray, Optional[tuple[float, float]]]:
    """-> (2D float32 array, native (row,col) mm spacing or None)."""
    if raw[:2] == b"\x1f\x8b":  # gzip magic -> NIfTI (.nii.gz)
        try:
            import SimpleITK as sitk
        except ImportError as e:  # pragma: no cover
            raise PreprocessingError("SimpleITK not installed for .nii.gz files.") from e
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp.write(raw); tmp_path = tmp.name
            img = sitk.ReadImage(tmp_path)
            arr = np.squeeze(sitk.GetArrayFromImage(img))
            if arr.ndim != 2:
                raise PreprocessingError(f"Expected a 2D frame, got shape {arr.shape}.")
            sp = img.GetSpacing()  # (x, y[, z]) -> store as (row=y, col=x)
            spacing = (float(sp[1]), float(sp[0])) if len(sp) >= 2 else None
            return arr.astype(np.float32), spacing
        except PreprocessingError:
            raise
        except Exception as e:
            raise PreprocessingError(f"Could not read NIfTI volume: {e}") from e
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    try:  # raster image (.png/.jpg) -> grayscale; no spacing metadata
        img = Image.open(io.BytesIO(raw)).convert("L")
        return np.asarray(img, dtype=np.float32), None
    except (UnidentifiedImageError, OSError) as e:
        raise PreprocessingError("Uploaded file is not a readable image.") from e


def preprocess_frame(raw: bytes, size: int) -> tuple[torch.Tensor, Optional[tuple[float, float]]]:
    """-> ((1,1,size,size) float32 tensor, effective spacing on the model grid)."""
    arr, native_sp = _decode(raw)
    h, w = arr.shape
    pil = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
    out = np.asarray(pil, dtype=np.float32)
    lo, hi = out.min(), out.max()
    out = (out - lo) / (hi - lo + NORM_EPS)
    tensor = torch.from_numpy(out).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    eff = None
    if native_sp is not None:
        row_sp, col_sp = native_sp
        eff = (row_sp * h / size, col_sp * w / size)  # native -> resized grid
    return tensor, eff
