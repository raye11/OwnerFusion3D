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
        default=2048,
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
        "--oehr-attention-backend",
        choices=("dense", "owner_grouped"),
        default="owner_grouped",
        help="OEHR synthesis backend. owner_grouped is the final strict-owner implementation.",
    )
    parser.add_argument("--owner-local-residual-fill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--owner-local-residual-fill-connectivity", type=int, choices=(6, 18, 26), default=18)
    parser.add_argument("--owner-local-residual-fill-min-neighbor-votes", type=int, default=4)
    parser.add_argument("--owner-local-residual-fill-min-neighbor-ratio", type=float, default=0.80)
    parser.add_argument("--owner-local-residual-fill-min-score", type=float, default=0.45)
    parser.add_argument("--owner-local-residual-fill-max-added-ratio", type=float, default=0.10)
    parser.add_argument("--owner-local-residual-fill-max-added-tokens", type=int, default=128)
    parser.add_argument("--owner-component-residual-completion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--owner-component-residual-completion-connectivity", type=int, choices=(6, 18, 26), default=18)
    parser.add_argument("--owner-component-residual-completion-max-component-tokens", type=int, default=24)
    parser.add_argument("--owner-component-residual-completion-min-boundary-votes", type=int, default=4)
    parser.add_argument("--owner-component-residual-completion-min-score", type=float, default=0.35)
    parser.add_argument("--owner-component-residual-completion-min-margin", type=float, default=0.03)
    parser.add_argument("--owner-component-residual-completion-max-added-ratio", type=float, default=0.05)
    parser.add_argument("--owner-component-residual-completion-max-added-tokens", type=int, default=128)
    parser.add_argument("--owner-final-neighbor-assignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--owner-final-neighbor-connectivity", type=int, choices=(6, 18, 26), default=6)
    parser.add_argument("--owner-final-neighbor-min-neighbor-votes", type=int, default=3)
    parser.add_argument("--owner-final-neighbor-min-neighbor-ratio", type=float, default=0.60)
    parser.add_argument(
        "--owner-final-neighbor-force-completion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assign remaining owner-0 tokens by nearest-owned-region propagation while preserving existing owners.",
    )
    parser.add_argument(
        "--owner-conflict-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Correct only ambiguous multi-candidate tokens with decisive score and neighbor support.",
    )
    parser.add_argument("--owner-conflict-correction-min-margin", type=float, default=0.08)
    parser.add_argument("--owner-conflict-correction-min-neighbor-ratio", type=float, default=0.60)
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
        cor=CORConfig(
            mode=args.cor_mode,
            local_residual_fill=args.owner_local_residual_fill,
            local_residual_connectivity=args.owner_local_residual_fill_connectivity,
            local_residual_min_neighbor_votes=args.owner_local_residual_fill_min_neighbor_votes,
            local_residual_min_neighbor_ratio=args.owner_local_residual_fill_min_neighbor_ratio,
            local_residual_min_score=args.owner_local_residual_fill_min_score,
            local_residual_max_added_ratio=args.owner_local_residual_fill_max_added_ratio,
            local_residual_max_added_tokens=args.owner_local_residual_fill_max_added_tokens,
            component_residual_completion=args.owner_component_residual_completion,
            component_residual_connectivity=args.owner_component_residual_completion_connectivity,
            component_residual_max_component_tokens=args.owner_component_residual_completion_max_component_tokens,
            component_residual_min_boundary_votes=args.owner_component_residual_completion_min_boundary_votes,
            component_residual_min_score=args.owner_component_residual_completion_min_score,
            component_residual_min_margin=args.owner_component_residual_completion_min_margin,
            component_residual_max_added_ratio=args.owner_component_residual_completion_max_added_ratio,
            component_residual_max_added_tokens=args.owner_component_residual_completion_max_added_tokens,
            final_neighbor_assignment=args.owner_final_neighbor_assignment,
            final_neighbor_connectivity=args.owner_final_neighbor_connectivity,
            final_neighbor_min_votes=args.owner_final_neighbor_min_neighbor_votes,
            final_neighbor_min_ratio=args.owner_final_neighbor_min_neighbor_ratio,
            final_neighbor_force_completion=args.owner_final_neighbor_force_completion,
            conflict_correction=args.owner_conflict_correction,
            conflict_correction_min_margin=args.owner_conflict_correction_min_margin,
            conflict_correction_min_neighbor_ratio=args.owner_conflict_correction_min_neighbor_ratio,
        ),
        oehr=OEHRConfig(
            attention_chunk_size=attention_chunk_size,
            attention_backend=args.oehr_attention_backend,
        ),
        multipart=MultiPartConfig(
            attention_chunk_size=attention_chunk_size,
            attention_backend=args.oehr_attention_backend,
        ),
    )


def pipeline_from_args(args: argparse.Namespace) -> OwnerFusionPipeline:
    return OwnerFusionPipeline.from_pretrained(config_from_args(args))
