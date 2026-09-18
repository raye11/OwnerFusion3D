"""Unified OwnerFusion3D entry point selected by the number of masks."""

from __future__ import annotations

import argparse

from .common import add_runtime_arguments, pipeline_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OwnerFusion3D. One mask selects MOE+OEHR; multiple masks select COR+OEHR."
    )
    parser.add_argument("--structure", required=True)
    parser.add_argument("--mask", required=True, nargs="+", help="One or more structure-view binary masks.")
    parser.add_argument(
        "--content", required=True, nargs="+", help="One content reference for each mask."
    )
    parser.add_argument("--output-dir", required=True)
    add_runtime_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pipeline = pipeline_from_args(args)
    result = pipeline.fuse_auto(args.structure, args.mask, args.content, args.output_dir)
    print(f"Mesh: {result.mesh_path}")
    print(f"Metadata: {result.metadata_path}")


if __name__ == "__main__":
    main()
