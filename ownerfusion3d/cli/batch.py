"""Manifest-driven single-region batch runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import add_runtime_arguments, pipeline_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible OwnerFusion3D batch manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    add_runtime_arguments(parser)
    return parser


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards).")
    manifest_path = Path(args.manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    cases = payload["cases"] if isinstance(payload, dict) else payload
    case_ids = [str(case["id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Batch case IDs must be unique.")
    selected = [case for index, case in enumerate(cases) if index % args.num_shards == args.shard_index]
    pipeline = pipeline_from_args(args)
    root = Path(args.output_root)
    for index, case in enumerate(selected, start=1):
        output = root / str(case["id"])
        if args.resume and (output / "mesh.glb").is_file() and (output / "metadata.json").is_file():
            print(f"[{index}/{len(selected)}] Reused: {case['id']}")
            continue
        result = pipeline.fuse(
            _resolve(manifest_path.parent, case["structure"]),
            _resolve(manifest_path.parent, case["mask"]),
            _resolve(manifest_path.parent, case["content"]),
            output,
        )
        print(f"[{index}/{len(selected)}] OK: {case['id']} -> {result.mesh_path}")


if __name__ == "__main__":
    main()
