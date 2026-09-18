"""Run the canonical multi-part replay cases with one loaded backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..io.parts import PartDefinition, PartManifest, ReferenceGroup
from .common import add_runtime_arguments, pipeline_from_args


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _parse_assignment(value: str) -> tuple[str, int]:
    name, separator, ordinal = str(value).partition("=")
    if separator != "=" or not name.startswith("p"):
        raise ValueError(f"Invalid part assignment: {value!r}")
    return name, int(ordinal)


def _part_key(value: str) -> str:
    key, separator, _ = str(value).partition("=")
    if separator != "=" or not key.startswith("p"):
        raise ValueError(f"Invalid part binding: {value!r}")
    return key


def _manifest_for_case(case: dict, data_root: Path) -> tuple[Path, Path, PartManifest]:
    structure = _path(data_root, case["structure_image"])
    labels = _path(data_root, case["part_label_image"])
    stats_path = _path(data_root, case["part_index_json"])
    stats = json.loads(stats_path.read_text(encoding="utf-8-sig"))
    raw_parts = stats.get("parts", [])
    colors = [tuple(int(c) for c in item["color_rgb"]) for item in raw_parts]
    part_names = {
        _parse_assignment(item)[1]: f"part_{_parse_assignment(item)[1]:02d}"
        for item in case["fuse_groups"]
    }
    parts = tuple(
        PartDefinition(name=part_names.get(index + 1, f"part_{index + 1:02d}"), color=color)
        for index, color in enumerate(colors)
    )
    groups = []
    for group_spec, image_spec in zip(case["fuse_groups"], case["fuse_images"]):
        group_name = _part_key(group_spec)
        image_name, separator, content_path = image_spec.partition("=")
        if separator != "=" or image_name != group_name:
            raise ValueError(f"Group/reference binding mismatch: {group_spec!r}, {image_spec!r}")
        groups.append(
            ReferenceGroup(
                name=group_name,
                content_image=_path(data_root, content_path),
                parts=(part_names[int(group_spec.split("=")[1])],),
            )
        )
    return structure, labels, PartManifest(parts=parts, groups=tuple(groups))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OwnerFusion3D COR on a replay manifest.")
    parser.add_argument("--manifest", required=True, help="Canonical replay manifest JSON.")
    parser.add_argument("--data-root", required=True, help="Root used to resolve paths in the replay manifest.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-filter", nargs="*", default=[])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    add_runtime_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards).")
    manifest_path = Path(args.manifest).resolve()
    data_root = Path(args.data_root).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    cases = payload["cases"] if isinstance(payload, dict) else payload
    selected = [case for index, case in enumerate(cases) if index % args.num_shards == args.shard_index]
    if args.sample_filter:
        selected = [case for case in selected if any(token.lower() in str(case["name"]).lower() for token in args.sample_filter)]
    if not selected:
        raise RuntimeError("No replay cases selected.")
    pipeline = pipeline_from_args(args)
    output_root = Path(args.output_root)
    for index, case in enumerate(selected, start=1):
        case_id = str(case["name"])
        output_dir = output_root / case_id
        if args.resume and (output_dir / "mesh.glb").is_file() and (output_dir / "metadata.json").is_file():
            print(f"[{index}/{len(selected)}] Reused: {case_id}")
            continue
        structure, labels, part_manifest = _manifest_for_case(case, data_root)
        result = pipeline.fuse_parts(structure, labels, part_manifest, output_dir)
        print(f"[{index}/{len(selected)}] OK: {case_id} -> {result.mesh_path}")


if __name__ == "__main__":
    main()
