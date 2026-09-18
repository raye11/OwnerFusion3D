"""Aligned structure-image and mask preprocessing."""

from __future__ import annotations

import numpy as np
from PIL import Image


def structure_crop(pipeline, image: Image.Image) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Remove background and return the exact square crop used by TRELLIS."""
    image = image.convert("RGBA")
    if np.all(np.asarray(image)[:, :, 3] == 255):
        if pipeline.low_vram:
            pipeline.rembg_model.to(pipeline.device)
        image = pipeline.rembg_model(image.convert("RGB"))
        if pipeline.low_vram:
            pipeline.rembg_model.cpu()
    alpha = np.asarray(image)[:, :, 3] > int(0.8 * 255)
    points = np.argwhere(alpha)
    if points.size == 0:
        raise ValueError("The structure image has no foreground after background removal.")
    x0, y0, x1, y1 = points[:, 1].min(), points[:, 0].min(), points[:, 1].max(), points[:, 0].max()
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    size = int(max(x1 - x0, y1 - y0))
    crop = (center_x - size // 2, center_y - size // 2, center_x + size // 2, center_y + size // 2)
    return image, crop


def preprocess_structure_and_mask(pipeline, image: Image.Image, mask: Image.Image):
    """Apply the exact TRELLIS foreground crop to both structure image and mask."""
    image = image.convert("RGBA")
    mask = mask.convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    max_size = max(image.size)
    scale = min(1.0, 1024.0 / max_size)
    if scale < 1.0:
        size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(size, Image.Resampling.LANCZOS)
        mask = mask.resize(size, Image.Resampling.NEAREST)

    image, crop = structure_crop(pipeline, image)
    image = image.crop(crop)
    mask = mask.crop(crop)
    rgba = np.asarray(image).astype(np.float32) / 255.0
    foreground = Image.fromarray(((rgba[:, :, 3] > 0.8) * 255).astype(np.uint8))
    rgb = Image.fromarray((rgba[:, :, :3] * rgba[:, :, 3:4] * 255).astype(np.uint8))
    return rgb, mask, foreground
