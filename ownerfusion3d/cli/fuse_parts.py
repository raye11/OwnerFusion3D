"""Multi-part OwnerFusion3D CLI."""

from __future__ import annotations

import argparse

from .common import add_runtime_arguments, pipeline_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synchronized multi-part OwnerFusion3D.")
    parser.add_argument("--structure", required=True)
    parser.add_argument(
        "--part-labels",
        default=None,
        help="Optional RGB categorical label image. Omit it when --parts contains one binary mask per part.",
    )
    parser.add_argument("--parts", required=True, help="Part/reference JSON manifest.")
    parser.add_argument("--output-dir", required=True)
    add_runtime_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = pipeline_from_args(args).fuse_parts(args.structure, args.part_labels, args.parts, args.output_dir)
    print(f"Mesh: {result.mesh_path}")
    print(f"Metadata: {result.metadata_path}")


if __name__ == "__main__":
    main()
