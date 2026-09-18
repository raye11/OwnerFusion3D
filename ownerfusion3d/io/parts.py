"""Typed multi-part manifest and aligned label preprocessing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .preprocess import structure_crop


@dataclass(frozen=True)
class PartDefinition:
    name: str
    color: tuple[int, int, int]
    mask_image: Path | None = None
    mask_value: Any = None


@dataclass(frozen=True)
class ReferenceGroup:
    name: str
    content_image: Path | Image.Image
    parts: tuple[str, ...]


@dataclass(frozen=True)
class PartManifest:
    parts: tuple[PartDefinition, ...]
    groups: tuple[ReferenceGroup, ...]

    @classmethod
    def load(cls, path: str | Path) -> "PartManifest":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        parts = tuple(
            PartDefinition(
                name=item["name"],
                color=_parse_color(item["color"]),
                mask_image=(
                    _resolve(path.parent, item.get("mask", item.get("mask_image")))
                    if item.get("mask", item.get("mask_image")) is not None
                    else None
                ),
            )
            for item in data["parts"]
        )
        groups = tuple(
            ReferenceGroup(
                item["name"],
                _resolve(path.parent, item["content_image"]),
                tuple(item["parts"]),
            )
            for item in data["reference_groups"]
        )
        if not parts:
            raise ValueError("The multi-part manifest must define at least one part.")
        if not groups:
            raise ValueError("The multi-part manifest must define at least one reference group.")
        part_names = {part.name for part in parts}
        if len(part_names) != len(parts):
            raise ValueError("Part names must be unique.")
        claimed: set[str] = set()
        for group in groups:
            if not group.parts:
                raise ValueError(f"Reference group {group.name!r} must bind at least one part.")
            unknown = set(group.parts) - part_names
            if unknown:
                raise ValueError(f"Reference group {group.name!r} uses unknown parts: {sorted(unknown)}")
            overlap = set(group.parts) & claimed
            if overlap:
                raise ValueError(f"Parts may belong to only one reference group: {sorted(overlap)}")
            if not group.content_image.is_file():
                raise FileNotFoundError(group.content_image)
            claimed.update(group.parts)
        unclaimed = part_names - claimed
        if unclaimed:
            raise ValueError(f"Every part must belong to exactly one reference group: {sorted(unclaimed)}")
        for part in parts:
            if part.mask_image is not None and not part.mask_image.is_file():
                raise FileNotFoundError(part.mask_image)
        return cls(parts, groups)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _parse_color(value: str | list[int]) -> tuple[int, int, int]:
    if isinstance(value, str):
        text = value.removeprefix("#")
        if len(text) != 6:
            raise ValueError(f"Invalid RGB hex color: {value}")
        return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))
    if len(value) != 3 or any(not 0 <= int(channel) <= 255 for channel in value):
        raise ValueError(f"Invalid RGB color: {value}")
    return tuple(int(channel) for channel in value)


def preprocess_structure_and_labels(pipeline, structure: Image.Image, labels: Image.Image):
    """Apply one shared structure-derived crop to the structure image and labels."""
    structure_rgba = structure.convert("RGBA")
    if labels.size != structure_rgba.size:
        labels = labels.resize(structure_rgba.size, Image.Resampling.NEAREST)
    max_size = max(structure_rgba.size)
    scale = min(1.0, 1024.0 / max_size)
    if scale < 1.0:
        size = (int(structure_rgba.width * scale), int(structure_rgba.height * scale))
        structure_rgba = structure_rgba.resize(size, Image.Resampling.LANCZOS)
        labels = labels.resize(size, Image.Resampling.NEAREST)
    rgba, crop = structure_crop(pipeline, structure_rgba)
    cropped = np.asarray(rgba.crop(crop)).astype(np.float32) / 255.0
    foreground = Image.fromarray(((cropped[:, :, 3] > 0.8) * 255).astype(np.uint8))
    processed = Image.fromarray((cropped[:, :, :3] * cropped[:, :, 3:4] * 255).astype(np.uint8))
    return processed, labels.crop(crop), foreground


def preprocess_structure_and_masks(
    pipeline,
    structure: Image.Image,
    masks: tuple[Image.Image, ...],
):
    """Apply one structure-derived crop to the structure and binary masks."""
    structure_rgba = structure.convert("RGBA")
    prepared_masks = []
    for mask in masks:
        current = mask.convert("L")
        if current.size != structure_rgba.size:
            current = current.resize(structure_rgba.size, Image.Resampling.NEAREST)
        prepared_masks.append(current)
    max_size = max(structure_rgba.size)
    scale = min(1.0, 1024.0 / max_size)
    if scale < 1.0:
        size = (int(structure_rgba.width * scale), int(structure_rgba.height * scale))
        structure_rgba = structure_rgba.resize(size, Image.Resampling.LANCZOS)
        prepared_masks = [mask.resize(size, Image.Resampling.NEAREST) for mask in prepared_masks]
    rgba, crop = structure_crop(pipeline, structure_rgba)
    cropped = np.asarray(rgba.crop(crop)).astype(np.float32) / 255.0
    foreground = Image.fromarray(((cropped[:, :, 3] > 0.8) * 255).astype(np.uint8))
    processed = Image.fromarray((cropped[:, :, :3] * cropped[:, :, 3:4] * 255).astype(np.uint8))
    return processed, tuple(mask.crop(crop) for mask in prepared_masks), foreground


def masks_from_manifest_parts(
    manifest: PartManifest,
) -> tuple[Image.Image, ...]:
    """Load direct binary masks embedded in a multi-part manifest."""
    if not all(part.mask_image is not None or part.mask_value is not None for part in manifest.parts):
        raise ValueError("A direct-mask multi-part run requires a mask for every part.")
    loaded = []
    for part in manifest.parts:
        value = part.mask_value if part.mask_value is not None else part.mask_image
        if isinstance(value, Image.Image):
            loaded.append(value.convert("L"))
        else:
            if value is None:
                raise ValueError(f"Part {part.name!r} has no direct mask.")
            value = Path(value)
            if not value.is_file():
                raise FileNotFoundError(value)
            loaded.append(Image.open(value).convert("L"))
    return tuple(loaded)


def masks_from_labels(label_image: Image.Image, parts: tuple[PartDefinition, ...]) -> dict[str, Image.Image]:
    labels = np.asarray(label_image.convert("RGB"))
    result = {}
    for part in parts:
        mask = np.all(labels == np.asarray(part.color, dtype=np.uint8), axis=-1)
        if not bool(mask.any()):
            raise ValueError(f"Part {part.name!r} color {part.color} is absent from the label image.")
        result[part.name] = Image.fromarray(mask.astype(np.uint8) * 255)
    return result
