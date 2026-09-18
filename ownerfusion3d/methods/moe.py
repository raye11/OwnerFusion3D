"""MOE: mask-conditioned ownership evidence from frozen cross-attention."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image

from .oehr import AttentionProbeState, majority_smooth_indices, sample_shape_with_regional_attention


def patch_coverage_from_mask_image(mask_image: Image.Image, layout) -> np.ndarray:
    target_size = (layout.grid_w * layout.patch_size, layout.grid_h * layout.patch_size)
    mask = mask_image.convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.BILINEAR)
    mask_np = np.asarray(mask, dtype=np.float32) / 255.0
    patches = mask_np.reshape(layout.grid_h, layout.patch_size, layout.grid_w, layout.patch_size)
    return patches.mean(axis=(1, 3)).reshape(-1).astype(np.float32, copy=False)


def build_patch_token_sets(
    *,
    layout,
    selected_patch_coverage: np.ndarray,
    foreground_patch_coverage: np.ndarray,
    device: torch.device,
    valid_min_coverage: float,
    selected_min_coverage: float,
    background_max_coverage: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Build weighted positive and structure-foreground negative patch sets."""
    selected_patch_coverage = np.asarray(selected_patch_coverage, dtype=np.float32).reshape(-1)
    foreground_patch_coverage = np.asarray(foreground_patch_coverage, dtype=np.float32).reshape(-1)
    if selected_patch_coverage.shape != foreground_patch_coverage.shape:
        raise ValueError("Selected and foreground patch coverage must have the same shape.")

    valid = foreground_patch_coverage >= float(valid_min_coverage)
    selected = (selected_patch_coverage >= float(selected_min_coverage)) & valid
    background = (selected_patch_coverage <= float(background_max_coverage)) & valid
    if not bool(selected.any()):
        selected = (selected_patch_coverage > 0.0) & valid
    if not bool(selected.any()):
        raise ValueError("The mask does not cover any valid structure patch.")

    selected_ids = np.flatnonzero(selected).astype(np.int64)
    background_ids = np.flatnonzero(background).astype(np.int64)
    selected_indices = torch.as_tensor(layout.additional_tokens + selected_ids, device=device, dtype=torch.long)
    background_indices = torch.as_tensor(layout.additional_tokens + background_ids, device=device, dtype=torch.long)
    selected_weights = torch.as_tensor(
        np.clip(selected_patch_coverage[selected_ids], 1e-6, 1.0), device=device, dtype=torch.float32
    )
    background_weights = torch.as_tensor(
        np.clip(foreground_patch_coverage[background_ids] * (1.0 - selected_patch_coverage[background_ids]), 1e-4, 1.0),
        device=device,
        dtype=torch.float32,
    )
    return selected_indices, background_indices, selected_weights, background_weights, {
        "patch_count": int(selected_patch_coverage.size),
        "valid_patch_count": int(valid.sum()),
        "selected_patch_count": int(selected.sum()),
        "background_patch_count": int(background.sum()),
        "ambiguous_patch_count": int((valid & ~selected & ~background).sum()),
    }


def _evidence(state: AttentionProbeState) -> tuple[torch.Tensor, dict]:
    selected = state.scores().float() / max(float(state.selected_token_normalizer), 1e-8)
    background = state.background_scores().float() / max(float(state.background_token_normalizer), 1e-8)
    raw = selected / (selected + background + 1e-8)
    normalized = raw.clamp(0.0, 1.0)
    raw_np = raw.detach().cpu().numpy()
    return normalized, {
        "score_mode": "ratio",
        "probe_layers_seen": int(state.count),
        "raw_score_q50": float(np.quantile(raw_np, 0.50)),
        "raw_score_q90": float(np.quantile(raw_np, 0.90)),
    }


def _expand_sparse_mask_by_score(
    coords: torch.Tensor,
    selected_mask: torch.Tensor,
    score: torch.Tensor,
    *,
    threshold: float,
    iterations: int,
    connectivity: int,
) -> torch.Tensor:
    if int(iterations) <= 0 or not bool(selected_mask.any().item()):
        return selected_mask
    xyz = coords[:, 1:].detach().cpu().numpy().astype(np.int32, copy=False)
    coord_to_index = {tuple(coord.tolist()): index for index, coord in enumerate(xyz)}
    chosen = set(np.flatnonzero(selected_mask.detach().cpu().numpy()).tolist())
    frontier = set(chosen)
    score_np = score.detach().cpu().numpy().astype(np.float32, copy=False)
    if int(connectivity) == 6:
        offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    else:
        offsets = tuple(
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        )
    for _ in range(int(iterations)):
        next_frontier: set[int] = set()
        for index in frontier:
            x, y, z = xyz[index].tolist()
            for dx, dy, dz in offsets:
                neighbor = coord_to_index.get((x + dx, y + dy, z + dz))
                if neighbor is not None and neighbor not in chosen and score_np[neighbor] >= float(threshold):
                    chosen.add(neighbor)
                    next_frontier.add(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier
    out = torch.zeros_like(selected_mask)
    if chosen:
        out[torch.as_tensor(sorted(chosen), device=out.device, dtype=torch.long)] = True
    return out


def postprocess_owner_score(
    *,
    coords: torch.Tensor,
    score: torch.Tensor,
    threshold: float,
    majority_k: int,
    majority_vote_ratio: float,
    expand_iters: int,
    expand_score_delta: float,
    expand_connectivity: int,
    max_selected_ratio: float,
    disable_expand_on_overselect: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    score = score.detach().float().to(coords.device)
    selected_idx = torch.nonzero(score >= float(threshold), as_tuple=False).flatten()
    pre_ratio = float(selected_idx.numel() / max(int(score.numel()), 1))
    effective_threshold = float(threshold)
    overselected = False
    if 0.0 < float(max_selected_ratio) < 1.0 and pre_ratio > float(max_selected_ratio):
        effective_threshold = max(
            effective_threshold,
            float(torch.quantile(score, 1.0 - float(max_selected_ratio)).item()),
        )
        selected_idx = torch.nonzero(score >= effective_threshold, as_tuple=False).flatten()
        max_tokens = max(int(np.ceil(float(max_selected_ratio) * int(score.numel()))), 1)
        if selected_idx.numel() > max_tokens:
            _, order = torch.topk(score[selected_idx], k=max_tokens, largest=True, sorted=False)
            selected_idx = selected_idx[order]
        overselected = True
    if int(majority_k) > 0 and selected_idx.numel() > 0:
        selected_idx = majority_smooth_indices(
            coords[:, 1:], selected_idx, k=int(majority_k), vote_ratio=float(majority_vote_ratio)
        )
    owner = torch.zeros(score.shape[0], dtype=torch.bool, device=score.device)
    owner[selected_idx] = True
    effective_expand_iters = 0 if overselected and bool(disable_expand_on_overselect) else int(expand_iters)
    expand_threshold = max(effective_threshold - max(float(expand_score_delta), 0.0), 0.0)
    before_expand = int(owner.sum().item())
    if effective_expand_iters > 0:
        owner = _expand_sparse_mask_by_score(
            coords,
            owner,
            score,
            threshold=expand_threshold,
            iterations=effective_expand_iters,
            connectivity=int(expand_connectivity),
        )
    strength = torch.where(owner, score, torch.zeros_like(score))
    return owner, strength, {
        "threshold": float(threshold),
        "threshold_effective": float(effective_threshold),
        "pre_suppression_selected_ratio": pre_ratio,
        "max_selected_ratio": float(max_selected_ratio),
        "overselect_suppressed": bool(overselected),
        "majority_k": int(majority_k),
        "majority_vote_ratio": float(majority_vote_ratio),
        "expand_iters": int(effective_expand_iters),
        "expand_threshold": float(expand_threshold),
        "expanded_tokens_added": int(max(int(owner.sum().item()) - before_expand, 0)),
        "expand_skipped_due_to_overselect": bool(overselected and bool(disable_expand_on_overselect)),
        "selected_tokens": int(owner.sum().item()),
        "selected_ratio": float(owner.float().mean().item()),
    }


def _ratio_evidence(
    selected_mass: np.ndarray,
    background_mass: np.ndarray,
    *,
    selected_normalizer: float,
    background_normalizer: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = selected_mass / max(float(selected_normalizer), 1e-8)
    background = background_mass / max(float(background_normalizer), 1e-8)
    evidence = selected / np.maximum(selected + background, 1e-8)
    support = selected + background
    return evidence.astype(np.float32, copy=False), support.astype(np.float32, copy=False)


def _pairwise_consistency(fields: np.ndarray) -> float:
    fields = np.asarray(fields, dtype=np.float32)
    if fields.shape[0] <= 1:
        return 1.0
    centered = fields - fields.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norms, 1e-8)
    sim = normalized @ normalized.T
    tri = sim[np.triu_indices(sim.shape[0], k=1)]
    return float(np.clip(tri.mean(), -1.0, 1.0)) if tri.size else 1.0


def _rank_fraction(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 1:
        return np.ones_like(values, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(values.size, dtype=np.float32)
    return ranks / float(values.size - 1)


def build_reliable_head_candidates(
    *,
    payload: dict[str, object],
    baseline_seed_mask: torch.Tensor,
    threshold: float,
    stable_block_heads: tuple[tuple[int, int], ...],
    reliability_temperature: float,
) -> dict[str, Any]:
    """Rank block-head fields and fuse the fixed stable-midlate-4 set.

    Block and head indices are zero-based internally. Metadata stores one-based
    labels for paper figures and debug tables.
    """
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("stable_midlate_4 requires collected per-head MOE evidence.")
    selected_mass = np.stack([record["selected_mass"].detach().cpu().numpy() for record in records], axis=0).astype(
        np.float32, copy=False
    )
    background_mass = np.stack(
        [record["background_mass"].detach().cpu().numpy() for record in records], axis=0
    ).astype(np.float32, copy=False)
    block_indices = np.asarray([int(record["block_index"]) for record in records], dtype=np.int32)
    if selected_mass.shape != background_mass.shape or selected_mass.ndim != 3:
        raise ValueError("Expected head masses with shape [observations, heads, sparse_tokens].")

    baseline_seed = baseline_seed_mask.detach().cpu().numpy().astype(bool, copy=False).reshape(-1)
    if baseline_seed.shape != (selected_mass.shape[-1],):
        raise ValueError("Baseline seed must contain one value per sparse token.")
    evidence, support = _ratio_evidence(
        selected_mass,
        background_mass,
        selected_normalizer=float(payload.get("selected_token_normalizer", 1.0)),
        background_normalizer=float(payload.get("background_token_normalizer", 1.0)),
    )

    candidates: list[dict[str, Any]] = []
    for block_index in sorted(set(block_indices.tolist())):
        observation_mask = block_indices == int(block_index)
        for head_index in range(evidence.shape[1]):
            field = np.asarray(evidence[observation_mask, head_index], dtype=np.float32).mean(axis=0)
            support_field = np.asarray(support[observation_mask, head_index], dtype=np.float32).mean(axis=0)
            selected = field >= float(threshold)
            selected_mean = float(field[selected].mean()) if bool(selected.any()) else 0.0
            remaining_mean = float(field[~selected].mean()) if bool((~selected).any()) else selected_mean
            union = np.count_nonzero(selected | baseline_seed)
            iou = float(np.count_nonzero(selected & baseline_seed) / union) if union else 1.0
            candidates.append(
                {
                    "block_index_zero_based": int(block_index),
                    "block_label": f"block_{int(block_index) + 1:02d}",
                    "head_index_zero_based": int(head_index),
                    "head_label": f"head_{int(head_index) + 1:02d}",
                    "step_consistency": float(_pairwise_consistency(evidence[observation_mask, head_index])),
                    "target_concentration": float(selected_mean - remaining_mean),
                    "support_focus": float(support_field[selected].mean() / max(float(support_field.mean()), 1e-8))
                    if bool(selected.any())
                    else 0.0,
                    "threshold_selected_ratio": float(selected.mean()),
                    "baseline_seed_iou_diagnostic": iou,
                    "_field": field,
                }
            )

    consistency_rank = _rank_fraction(np.asarray([item["step_consistency"] for item in candidates]))
    concentration_rank = _rank_fraction(np.asarray([item["target_concentration"] for item in candidates]))
    support_rank = _rank_fraction(np.asarray([item["support_focus"] for item in candidates]))
    for index, candidate in enumerate(candidates):
        candidate["reliability_score"] = float(
            0.40 * consistency_rank[index] + 0.35 * concentration_rank[index] + 0.25 * support_rank[index]
        )
    candidates.sort(key=lambda item: float(item["reliability_score"]), reverse=True)

    lookup = {
        (int(item["block_index_zero_based"]), int(item["head_index_zero_based"])): item for item in candidates
    }
    members = [lookup[key] for key in stable_block_heads if key in lookup]
    if not members:
        raise RuntimeError("The fixed stable mid-late block-head set was not observed during probing.")

    temperature = max(float(reliability_temperature), 1e-4)
    reliability = np.asarray([item["reliability_score"] for item in members], dtype=np.float64)
    weights = np.exp((reliability - float(reliability.max())) / temperature)
    weights /= max(float(weights.sum()), 1e-12)
    fused = np.zeros_like(np.asarray(members[0]["_field"], dtype=np.float32))
    public_members: list[dict[str, Any]] = []
    for item, weight in zip(members, weights.tolist()):
        fused += float(weight) * np.asarray(item["_field"], dtype=np.float32)
        public_members.append({key: value for key, value in item.items() if not key.startswith("_")} | {"fusion_weight": float(weight)})

    return {
        "method": "stable_midlate_reliable_head_fusion",
        "threshold": float(threshold),
        "reliability_temperature": temperature,
        "stable_block_heads": [list(map(int, key)) for key in stable_block_heads],
        "members": public_members,
        "score": fused.astype(np.float32, copy=False),
    }


def apply_moe_readout(
    *,
    coords: torch.Tensor,
    baseline_score: torch.Tensor,
    baseline_seed_mask: torch.Tensor,
    head_payload: dict | None,
    readout: str,
    threshold: float,
    majority_k: int,
    majority_vote_ratio: float,
    expand_iters: int,
    expand_score_delta: float,
    expand_connectivity: int,
    max_selected_ratio: float,
    disable_expand_on_overselect: bool,
    stable_block_heads: tuple[tuple[int, int], ...],
    reliability_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    normalized_readout = str(readout).strip().lower()
    score = baseline_score.detach().float().to(coords.device)
    readout_stats: dict[str, Any] = {"method": "all_head_moe", "readout": "all_head", "members": []}
    if normalized_readout == "stable_midlate_4":
        if head_payload is None:
            raise RuntimeError("stable_midlate_4 requires per-head evidence; set collect_head_scores=True.")
        candidates = build_reliable_head_candidates(
            payload=head_payload,
            baseline_seed_mask=baseline_seed_mask,
            threshold=float(threshold),
            stable_block_heads=stable_block_heads,
            reliability_temperature=float(reliability_temperature),
        )
        score = torch.from_numpy(np.asarray(candidates["score"], dtype=np.float32)).to(coords.device)
        readout_stats = {
            "method": "stable_midlate_reliable_head_fusion",
            "readout": "stable_midlate_4",
            "members": candidates["members"],
            "stable_block_heads": candidates["stable_block_heads"],
            "reliability_temperature": candidates["reliability_temperature"],
        }
    elif normalized_readout != "all_head":
        raise ValueError("MOE readout must be 'all_head' or 'stable_midlate_4'.")

    owner, strength, postprocess = postprocess_owner_score(
        coords=coords,
        score=score,
        threshold=float(threshold),
        majority_k=int(majority_k),
        majority_vote_ratio=float(majority_vote_ratio),
        expand_iters=int(expand_iters),
        expand_score_delta=float(expand_score_delta),
        expand_connectivity=int(expand_connectivity),
        max_selected_ratio=float(max_selected_ratio),
        disable_expand_on_overselect=bool(disable_expand_on_overselect),
    )
    readout_stats["postprocess"] = postprocess
    return owner, strength, score, readout_stats


def probe_ownership_evidence(
    *,
    pipeline,
    condition: dict,
    coordinates: torch.Tensor,
    flow_model,
    selected_patch_indices: torch.Tensor,
    background_patch_indices: torch.Tensor,
    selected_patch_weights: torch.Tensor,
    background_patch_weights: torch.Tensor,
    steps: int,
    threshold: float,
    majority_neighbors: int,
    majority_vote_ratio: float,
    max_selected_ratio: float,
    expand_iterations: int,
    expand_score_delta: float,
    expand_connectivity: int,
    disable_expand_on_overselect: bool,
    chunk_size: int,
    readout: str = "stable_midlate_4",
    readout_temperature: float = 0.15,
    stable_block_heads: tuple[tuple[int, int], ...] = ((7, 1), (10, 8), (11, 0), (11, 1)),
    isolate_rng: bool = False,
    probe_runner: Optional[Callable[[AttentionProbeState], None]] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Probe frozen shape-flow attention and return direct MOE ownership."""
    collect_head_scores = str(readout).strip().lower() == "stable_midlate_4"
    state = AttentionProbeState(
        selected_patch_indices,
        background_patch_indices,
        selected_token_weights=selected_patch_weights,
        background_token_weights=background_patch_weights,
        collect_head_scores=collect_head_scores,
    )
    cpu_state = torch.get_rng_state() if isolate_rng else None
    cuda_states = torch.cuda.get_rng_state_all() if isolate_rng and torch.cuda.is_available() else None
    try:
        if probe_runner is None:
            sample_shape_with_regional_attention(
                pipeline=pipeline,
                cond=condition,
                flow_model=flow_model,
                coords=coordinates,
                sampler_params={"steps": int(steps), "guidance_strength": 1.0},
                probe_state=state,
                chunk_size=chunk_size,
            )
        else:
            probe_runner(state)
    finally:
        if cpu_state is not None:
            torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    baseline_score, score_stats = _evidence(state)
    baseline_seed, baseline_strength, baseline_post = postprocess_owner_score(
        coords=coordinates,
        score=baseline_score,
        threshold=float(threshold),
        majority_k=int(majority_neighbors),
        majority_vote_ratio=float(majority_vote_ratio),
        expand_iters=int(expand_iterations),
        expand_score_delta=float(expand_score_delta),
        expand_connectivity=int(expand_connectivity),
        max_selected_ratio=float(max_selected_ratio),
        disable_expand_on_overselect=bool(disable_expand_on_overselect),
    )
    owner, strength, final_score, readout_stats = apply_moe_readout(
        coords=coordinates,
        baseline_score=baseline_score,
        baseline_seed_mask=baseline_seed,
        head_payload=state.head_scores_payload(),
        readout=readout,
        threshold=float(threshold),
        majority_k=int(majority_neighbors),
        majority_vote_ratio=float(majority_vote_ratio),
        expand_iters=int(expand_iterations),
        expand_score_delta=float(expand_score_delta),
        expand_connectivity=int(expand_connectivity),
        max_selected_ratio=float(max_selected_ratio),
        disable_expand_on_overselect=bool(disable_expand_on_overselect),
        stable_block_heads=stable_block_heads,
        reliability_temperature=float(readout_temperature),
    )
    stats = {
        **score_stats,
        "threshold": float(threshold),
        "rng_isolated": bool(isolate_rng),
        "readout": readout_stats,
        "selected_tokens": int(owner.sum().item()),
        "selected_ratio": float(owner.float().mean().item()),
    }
    if str(readout).strip().lower() == "all_head":
        stats["all_head_baseline"] = {
            "selected_tokens": int(baseline_seed.sum().item()),
            "selected_ratio": float(baseline_seed.float().mean().item()),
            "postprocess": baseline_post,
        }
    stats["_baseline_evidence_tensor"] = baseline_score.detach()
    stats["_evidence_tensor"] = final_score.detach()
    stats["_baseline_strength_tensor"] = baseline_strength.detach()
    return owner, strength, stats


def save_evidence_ply(
    coordinates: torch.Tensor,
    evidence: torch.Tensor,
    output: Path,
    resolution: int,
    *,
    grid_resolution: int | None = None,
) -> None:
    """Export continuous MOE evidence for inspection and paper figures."""
    grid = max(int(grid_resolution or (resolution // 16)), 1)
    points = ((coordinates[:, 1:].detach().cpu().numpy().astype(np.float32) + 0.5) / grid) - 0.5
    values = evidence.detach().cpu().numpy().astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, value in zip(points, values):
            t = float(np.clip(value, 0.0, 1.0))
            color = (int(45 + 210 * t), int(100 + 120 * (1 - t)), 55)
            handle.write(f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} {color[0]} {color[1]} {color[2]}\n")
