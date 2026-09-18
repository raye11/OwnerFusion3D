"""Run the paper-level OwnerFusion3D variants with one loaded backbone."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import AblationVariant
from .common import add_runtime_arguments, pipeline_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OwnerFusion3D module ablations.")
    parser.add_argument("--structure", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--content", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[variant.value for variant in AblationVariant],
        default=[variant.value for variant in AblationVariant],
    )
    add_runtime_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pipeline = pipeline_from_args(args)
    root = Path(args.output_dir)
    for name in args.variants:
        result = pipeline.fuse(
            args.structure,
            args.mask,
            args.content,
            root / name,
            variant=AblationVariant(name),
        )
        print(f"{name}: {result.mesh_path}")


if __name__ == "__main__":
    main()
