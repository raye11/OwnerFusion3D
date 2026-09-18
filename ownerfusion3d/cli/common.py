"""Shared CLI options and pipeline construction."""

from __future__ import annotations

import argparse

from ..config import CORConfig, MOEConfig, MultiPartConfig, OEHRConfig, OwnerFusionConfig
from ..pipeline import OwnerFusionPipeline


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B", help="TRELLIS.2 checkpoint or local directory.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resolution", type=int, default=1024, choices=[1024, 1152, 1280, 1408, 1536])
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument(
        "--attention-chunk-size",
        type=int,
        default=256,
        help=(
            "Number of sparse query tokens processed per regional cross-attention chunk. "
            "Larger values may improve throughput but use more GPU memory."
        ),
    )
    parser.add_argument(
        "--moe-probe-steps",
        type=int,
        default=4,
        choices=range(1, 13),
        metavar="N",
        help="Number of LR Shape SLat Flow steps used to probe MOE ownership evidence.",
    )
    parser.add_argument(
        "--moe-readout",
        choices=("stable_midlate_4", "all_head"),
        default="stable_midlate_4",
        help="MOE readout. The paper method uses stable_midlate_4; all_head is the ablation.",
    )
    parser.add_argument(
        "--cor-mode",
        choices=("full", "score_only"),
        default="full",
        help="Multi-part COR mode; score_only is the controlled w/o-COR ablation.",
    )
    parser.add_argument(
        "--cor-routing-owner-stage",
        choices=("after_core", "after_local", "after_boundary", "after_hole", "after_full"),
        default="after_local",
        help="Owner snapshot used by OEHR; completion statistics still report the full COR owner.",
    )
    parser.add_argument("--no-ownership-output", action="store_true", help="Do not export ownership readouts.")


def config_from_args(args: argparse.Namespace) -> OwnerFusionConfig:
    if int(args.attention_chunk_size) < 1:
        raise ValueError("--attention-chunk-size must be a positive integer.")
    attention_chunk_size = int(args.attention_chunk_size)
    return OwnerFusionConfig(
        model=args.model,
        seed=args.seed,
        output_resolution=args.resolution,
        max_num_tokens=args.max_num_tokens,
        texture_size=args.texture_size,
        save_ownership=not args.no_ownership_output,
        moe=MOEConfig(probe_steps_lr=args.moe_probe_steps, readout=args.moe_readout),
        cor=CORConfig(mode=args.cor_mode, routing_owner_stage=args.cor_routing_owner_stage),
        oehr=OEHRConfig(attention_chunk_size=attention_chunk_size),
        multipart=MultiPartConfig(attention_chunk_size=attention_chunk_size),
    )


def pipeline_from_args(args: argparse.Namespace) -> OwnerFusionPipeline:
    return OwnerFusionPipeline.from_pretrained(config_from_args(args))
