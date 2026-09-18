"""Single-region OwnerFusion3D CLI."""

from __future__ import annotations

import argparse

from .common import add_runtime_arguments, pipeline_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OwnerFusion3D on one structure/mask/content triple.")
    parser.add_argument("--structure", required=True, help="Structure image.")
    parser.add_argument("--mask", required=True, help="Binary mask aligned to the structure image.")
    parser.add_argument("--content", required=True, help="Content reference image.")
    parser.add_argument("--output-dir", required=True, help="Case output directory.")
    add_runtime_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = pipeline_from_args(args).fuse(args.structure, args.mask, args.content, args.output_dir)
    print(f"Mesh: {result.mesh_path}")
    print(f"Metadata: {result.metadata_path}")


if __name__ == "__main__":
    main()
