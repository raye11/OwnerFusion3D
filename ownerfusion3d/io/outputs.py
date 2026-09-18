"""Stable output layout and ownership readout exporters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh


@dataclass(frozen=True)
class FusionResult:
    mesh_path: Path
    metadata_path: Path
    ownership_directory: Path | None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.tolist() if tensor.numel() <= 256 else {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 256 else {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_owner_ply(
    coordinates: torch.Tensor,
    owner: torch.Tensor,
    path: Path,
    resolution: int,
    *,
    grid_resolution: int | None = None,
) -> None:
    grid = max(int(grid_resolution or (resolution // 16)), 1)
    points = ((coordinates[:, 1:].detach().cpu().numpy().astype(np.float32) + 0.5) / grid) - 0.5
    selected = owner.detach().cpu().numpy().astype(bool, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, active in zip(points, selected):
            color = (230, 126, 34) if active else (125, 132, 145)
            handle.write(f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} {color[0]} {color[1]} {color[2]}\n")


def save_owner_glb(coordinates: torch.Tensor, owner: torch.Tensor, path: Path, resolution: int) -> None:
    """Export a Fuse3D-compatible colored voxel readout, not a final mesh part."""
    coords = coordinates[:, 1:].detach().cpu().numpy().astype(np.int32, copy=False)
    selected = owner.detach().cpu().numpy().astype(bool, copy=False)
    grid = max(int(resolution // 16), 1)
    pitch = 1.0 / grid
    boxes = []
    colors = []
    for coord, active in zip(coords, selected):
        center = ((coord.astype(np.float32) + 0.5) / grid) - 0.5
        box = trimesh.creation.box(extents=(pitch, pitch, pitch))
        box.apply_translation(center)
        boxes.append(box)
        rgba = np.asarray([0, 255, 0, 255] if active else [255, 0, 0, 255], dtype=np.uint8)
        colors.append(np.tile(rgba, (len(box.faces), 1)))
    if not boxes:
        return
    mesh = trimesh.util.concatenate(boxes)
    mesh.visual.face_colors = np.concatenate(colors, axis=0)
    angle = np.deg2rad(-90.0)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    mesh.apply_transform(transform)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def save_categorical_owner_glb(
    coordinates: torch.Tensor,
    owner: torch.Tensor,
    colors: list[tuple[int, int, int]],
    path: Path,
    resolution: int,
    *,
    unassigned_color: tuple[int, int, int] = (140, 140, 140),
) -> None:
    """Export a colored sparse owner field as a voxel GLB.

    Label 0 is rendered gray; positive labels use ``colors[label]``.  The
    artifact is a diagnostic readout and is not the decoded asset mesh.
    """
    coords = coordinates[:, 1:].detach().cpu().numpy().astype(np.int32, copy=False)
    labels = owner.detach().cpu().numpy().astype(np.int64, copy=False)
    if len(coords) == 0:
        return

    grid = max(int(resolution // 16), 1)
    pitch = 1.0 / float(grid)
    boxes: list[trimesh.Trimesh] = []
    face_colors: list[np.ndarray] = []
    for coord, label in zip(coords, labels):
        center = ((coord.astype(np.float32) + 0.5) / float(grid)) - 0.5
        box = trimesh.creation.box(extents=(pitch, pitch, pitch))
        box.apply_translation(center)
        boxes.append(box)
        label_int = int(label)
        color_rgb = colors[label_int] if 0 < label_int < len(colors) else unassigned_color
        rgba = np.asarray([*color_rgb, 255], dtype=np.uint8)
        face_colors.append(np.tile(rgba, (len(box.faces), 1)))

    mesh = trimesh.util.concatenate(boxes)
    mesh.visual.face_colors = np.concatenate(face_colors, axis=0)
    angle = np.deg2rad(-90.0)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    mesh.apply_transform(transform)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def save_categorical_owner_ply(
    coordinates: torch.Tensor,
    owner: torch.Tensor,
    colors: list[tuple[int, int, int]],
    path: Path,
    resolution: int,
) -> None:
    grid = max(int(resolution // 16), 1)
    points = ((coordinates[:, 1:].detach().cpu().numpy().astype(np.float32) + 0.5) / grid) - 0.5
    labels = owner.detach().cpu().numpy().astype(np.int64, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, label in zip(points, labels):
            color = colors[int(label)] if 0 <= int(label) < len(colors) else colors[0]
            handle.write(f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} {color[0]} {color[1]} {color[2]}\n")
