"""COR: categorical ownership resolution for multi-part fusion.

COR is an internal multi-part operator, not a fourth core OwnerFusion3D module.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

import numpy as np
import torch


class CORSettings(Protocol):
    mode: str
    routing_owner_stage: str
    owner_connectivity: int
    owner_core_quantile: float
    owner_core_min_score: float
    owner_core_min_tokens: int
    owner_boundary_margin: float
    owner_distance_weight: float
    owner_core_neighbor_weight: float
    owner_assigned_neighbor_weight: float
    owner_fill_unassigned_iters: int
    owner_repair_enable: bool
    owner_repair_iters: int
    owner_repair_connectivity: int
    owner_repair_min_neighbor_votes: int
    owner_repair_min_neighbor_ratio: float
    owner_repair_decisive_margin: float
    owner_repair_candidate_bonus: float
    local_residual_fill: bool
    local_residual_connectivity: int
    local_residual_min_neighbor_votes: int
    local_residual_min_neighbor_ratio: float
    local_residual_min_score: float
    local_residual_max_added_ratio: float
    local_residual_max_added_tokens: int
    boundary_completion: bool
    boundary_connectivity: int
    boundary_min_neighbor_votes: int
    boundary_min_neighbor_ratio: float
    boundary_min_score: float
    boundary_min_score_margin: float
    boundary_max_added_ratio: float
    boundary_max_added_tokens: int
    hole_completion: bool
    hole_connectivity: int
    hole_min_neighbor_votes: int
    hole_min_neighbor_ratio: float
    hole_max_competing_ratio: float
    hole_min_score: float
    hole_max_added_ratio: float
    hole_max_added_tokens: int


def _neighbor_offsets(connectivity: int) -> list[tuple[int, int, int]]:
    if int(connectivity) == 6:
        return [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    if int(connectivity) == 18:
        offsets: list[tuple[int, int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    if abs(dx) + abs(dy) + abs(dz) <= 2:
                        offsets.append((dx, dy, dz))
        return offsets
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


def _build_sparse_neighbors(coords: torch.Tensor, connectivity: int) -> list[list[int]]:
    xyz = coords[:, 1:].detach().cpu().numpy().astype(np.int32, copy=False)
    coord_to_index = {tuple(coord.tolist()): idx for idx, coord in enumerate(xyz)}
    offsets = _neighbor_offsets(connectivity)
    neighbors: list[list[int]] = [[] for _ in range(xyz.shape[0])]
    for idx, coord in enumerate(xyz.tolist()):
        x, y, z = coord
        nbrs: list[int] = []
        for dx, dy, dz in offsets:
            nbr = coord_to_index.get((x + dx, y + dy, z + dz))
            if nbr is not None:
                nbrs.append(int(nbr))
        neighbors[idx] = nbrs
    return neighbors


def _build_core_mask(
    score: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    core_quantile: float,
    core_min_score: float,
    core_min_tokens: int,
) -> tuple[np.ndarray, float]:
    core = np.zeros_like(candidate_mask, dtype=bool)
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        return core, float(core_min_score)
    candidate_scores = score[candidate_indices]
    threshold = max(float(core_min_score), float(np.quantile(candidate_scores, float(core_quantile))))
    core = candidate_mask & (score >= threshold)
    min_tokens = max(int(core_min_tokens), 1)
    if int(core.sum()) < min_tokens:
        top_count = min(min_tokens, int(candidate_indices.size))
        order = np.argsort(candidate_scores)
        keep = candidate_indices[order[-top_count:]]
        core[keep] = True
    return core, float(threshold)


def _bfs_distance_to_core(
    neighbors: list[list[int]],
    seed_mask: np.ndarray,
    allowed_mask: np.ndarray,
) -> np.ndarray:
    n = allowed_mask.shape[0]
    inf = np.float32(1e9)
    dist = np.full(n, inf, dtype=np.float32)
    queue: deque[int] = deque()
    seed_indices = np.flatnonzero(seed_mask & allowed_mask)
    for idx in seed_indices.tolist():
        dist[int(idx)] = 0.0
        queue.append(int(idx))
    while queue:
        idx = queue.popleft()
        next_dist = dist[idx] + 1.0
        for nbr in neighbors[idx]:
            if not allowed_mask[nbr]:
                continue
            if next_dist < dist[nbr]:
                dist[nbr] = next_dist
                queue.append(int(nbr))
    return dist


def resolve_categorical_ownership(
    *,
    coords: torch.Tensor,
    score_stack: list[torch.Tensor],
    candidate_masks: list[torch.Tensor],
    config: CORSettings,
) -> tuple[torch.Tensor, dict]:
    if not score_stack or not candidate_masks:
        owner = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        return owner, {"conflict_tokens": 0, "boundary_tokens": 0}

    score_np = torch.stack(score_stack, dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
    candidate_np = torch.stack(candidate_masks, dim=1).detach().cpu().numpy().astype(bool, copy=False)
    num_tokens, num_parts = candidate_np.shape
    neighbors = _build_sparse_neighbors(coords, config.owner_connectivity)

    core_np = np.zeros_like(candidate_np, dtype=bool)
    core_thresholds: list[float] = []
    dist_np = np.full((num_tokens, num_parts), np.float32(1e9), dtype=np.float32)
    for part_idx in range(num_parts):
        core_mask, core_thr = _build_core_mask(
            score_np[:, part_idx],
            candidate_np[:, part_idx],
            core_quantile=config.owner_core_quantile,
            core_min_score=config.owner_core_min_score,
            core_min_tokens=config.owner_core_min_tokens,
        )
        core_np[:, part_idx] = core_mask
        core_thresholds.append(core_thr)
        dist_np[:, part_idx] = _bfs_distance_to_core(neighbors, core_mask, candidate_np[:, part_idx])

    owner_np = np.zeros(num_tokens, dtype=np.int32)
    boundary_np = np.zeros(num_tokens, dtype=bool)
    candidate_count = candidate_np.sum(axis=1)
    core_count = core_np.sum(axis=1)

    unique_candidate = candidate_count == 1
    if np.any(unique_candidate):
        owner_np[unique_candidate] = candidate_np[unique_candidate].argmax(axis=1) + 1

    unique_core_conflict = (owner_np == 0) & (candidate_count > 1) & (core_count == 1)
    if np.any(unique_core_conflict):
        owner_np[unique_core_conflict] = core_np[unique_core_conflict].argmax(axis=1) + 1

    conflict_indices = np.flatnonzero((owner_np == 0) & (candidate_count > 1))
    for token_idx in conflict_indices.tolist():
        part_indices = np.flatnonzero(candidate_np[token_idx])
        combined_scores: list[float] = []
        for part_idx in part_indices.tolist():
            base_score = float(score_np[token_idx, part_idx])
            dist_value = float(dist_np[token_idx, part_idx])
            proximity = 0.0 if dist_value >= 1e8 else 1.0 / (1.0 + dist_value)
            nbr_indices = neighbors[token_idx]
            core_neighbor = 0.0
            assigned_neighbor = 0.0
            if nbr_indices:
                nbr_arr = np.asarray(nbr_indices, dtype=np.int32)
                core_neighbor = float(core_np[nbr_arr, part_idx].mean())
                assigned_neighbor = float((owner_np[nbr_arr] == (part_idx + 1)).mean())
            combined = (
                base_score
                + float(config.owner_distance_weight) * proximity
                + float(config.owner_core_neighbor_weight) * core_neighbor
                + float(config.owner_assigned_neighbor_weight) * assigned_neighbor
            )
            combined_scores.append(combined)
        order = np.argsort(np.asarray(combined_scores, dtype=np.float32))[::-1]
        best_value = float(combined_scores[int(order[0])])
        second_value = float(combined_scores[int(order[1])]) if len(order) > 1 else -1e9
        if best_value - second_value < float(config.owner_boundary_margin):
            boundary_np[token_idx] = True
            continue
        owner_np[token_idx] = int(part_indices[int(order[0])] + 1)

    for _ in range(max(int(config.owner_fill_unassigned_iters), 0)):
        updated = False
        unresolved = np.flatnonzero((owner_np == 0) & (~boundary_np) & (candidate_count > 0))
        for token_idx in unresolved.tolist():
            nbr_indices = neighbors[token_idx]
            if not nbr_indices:
                continue
            votes: dict[int, int] = {}
            for nbr in nbr_indices:
                nbr_owner = int(owner_np[nbr])
                if nbr_owner <= 0:
                    continue
                if candidate_np[token_idx, nbr_owner - 1]:
                    votes[nbr_owner] = votes.get(nbr_owner, 0) + 1
            if not votes:
                continue
            ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)
            best_owner, best_votes = ranked[0]
            second_votes = ranked[1][1] if len(ranked) > 1 else 0
            if best_votes <= second_votes:
                boundary_np[token_idx] = True
                continue
            owner_np[token_idx] = int(best_owner)
            updated = True
        if not updated:
            break

    owner_tensor = torch.as_tensor(owner_np, dtype=torch.long, device=coords.device)
    stats = {
        "conflict_tokens": int(conflict_indices.size),
        "boundary_tokens": int(boundary_np.sum()),
        "unassigned_tokens": int((owner_np == 0).sum()),
        "core_thresholds": {str(part_idx + 1): float(core_thresholds[part_idx]) for part_idx in range(num_parts)},
    }
    return owner_tensor, stats


def repair_categorical_ownership(
    *,
    coords: torch.Tensor,
    owner: torch.Tensor,
    score_stack: list[torch.Tensor],
    candidate_masks: list[torch.Tensor],
    part_strengths: list[torch.Tensor],
    config: CORSettings,
) -> tuple[torch.Tensor, dict]:
    owner_np = owner.detach().cpu().numpy().astype(np.int32, copy=True)
    initial_unassigned = int((owner_np == 0).sum())
    if not bool(config.owner_repair_enable) or initial_unassigned == 0:
        return owner, {
            "enabled": bool(config.owner_repair_enable),
            "initial_unassigned": int(initial_unassigned),
            "final_unassigned": int(initial_unassigned),
            "reassigned_tokens": 0,
            "iterations_run": 0,
            "passes": [],
        }

    score_np = torch.stack(score_stack, dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
    candidate_np = torch.stack(candidate_masks, dim=1).detach().cpu().numpy().astype(bool, copy=False)
    strength_np = torch.stack(part_strengths, dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
    neighbors = _build_sparse_neighbors(coords, int(config.owner_repair_connectivity))

    pass_stats: list[dict] = []
    for _iter in range(max(int(config.owner_repair_iters), 0)):
        unresolved = np.flatnonzero(owner_np == 0)
        if unresolved.size == 0:
            break

        pending: list[tuple[int, int]] = []
        for token_idx in unresolved.tolist():
            nbr_indices = neighbors[int(token_idx)]
            if not nbr_indices:
                continue

            assigned_nbrs = [int(nbr) for nbr in nbr_indices if int(owner_np[nbr]) > 0]
            if not assigned_nbrs:
                continue

            candidate_parts = np.flatnonzero(candidate_np[token_idx]).astype(np.int32) + 1
            allowed_parts = set(int(v) for v in candidate_parts.tolist())
            if not allowed_parts:
                continue

            total_assigned = len(assigned_nbrs)
            metrics: list[tuple[float, int, float, int]] = []
            for part_idx in sorted(allowed_parts):
                vote_count = sum(1 for nbr in assigned_nbrs if int(owner_np[nbr]) == int(part_idx))
                if vote_count <= 0:
                    continue
                vote_ratio = float(vote_count / max(total_assigned, 1))
                local_score = float(score_np[token_idx, part_idx - 1])
                local_strength = float(strength_np[token_idx, part_idx - 1])
                combined = (
                    float(vote_count)
                    + 0.50 * vote_ratio
                    + 0.25 * local_strength
                    + 0.20 * local_score
                    + float(config.owner_repair_candidate_bonus)
                )
                metrics.append((combined, vote_count, vote_ratio, int(part_idx)))

            if not metrics:
                continue

            metrics.sort(key=lambda item: item[0], reverse=True)
            best_combined, best_votes, best_ratio, best_part = metrics[0]
            second_combined = metrics[1][0] if len(metrics) > 1 else -1e9

            if int(best_votes) < int(config.owner_repair_min_neighbor_votes) and float(best_ratio) < float(
                config.owner_repair_min_neighbor_ratio
            ):
                continue
            if (
                (float(best_combined) - float(second_combined)) < float(config.owner_repair_decisive_margin)
                and float(best_ratio) < max(float(config.owner_repair_min_neighbor_ratio), 0.70)
            ):
                continue
            pending.append((int(token_idx), int(best_part)))

        if not pending:
            break

        for token_idx, best_part in pending:
            owner_np[int(token_idx)] = int(best_part)
        pass_stats.append(
            {
                "iteration": int(len(pass_stats) + 1),
                "reassigned_tokens": int(len(pending)),
                "remaining_unassigned": int((owner_np == 0).sum()),
            }
        )

    repaired = torch.as_tensor(owner_np, dtype=torch.long, device=owner.device)
    return repaired, {
        "enabled": True,
        "initial_unassigned": int(initial_unassigned),
        "final_unassigned": int((owner_np == 0).sum()),
        "reassigned_tokens": int(initial_unassigned - int((owner_np == 0).sum())),
        "iterations_run": int(len(pass_stats)),
        "passes": pass_stats,
    }


def resolve_score_only_ownership(
    *,
    score_stack: list[torch.Tensor],
    candidate_masks: list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Controlled w/o-COR ablation: assign every candidate by MOE score."""
    if not score_stack or not candidate_masks:
        owner = torch.zeros(0, dtype=torch.long, device=device)
        return owner, {"mode": "score_only", "candidate_tokens": 0, "conflict_tokens": 0, "unassigned_tokens": 0}
    scores = torch.stack(score_stack, dim=1).to(device=device, dtype=torch.float32)
    candidates = torch.stack(candidate_masks, dim=1).to(device=device, dtype=torch.bool)
    count = candidates.sum(dim=1)
    owner = torch.argmax(scores.masked_fill(~candidates, -torch.inf), dim=1) + 1
    owner = torch.where(count > 0, owner, torch.zeros_like(owner)).long()
    return owner, {
        "mode": "score_only",
        "candidate_tokens": int((count > 0).sum().item()),
        "conflict_tokens": int((count > 1).sum().item()),
        "unassigned_tokens": int((owner == 0).sum().item()),
        "spatial_resolution": False,
    }


def _bounded_completion(
    *,
    coords: torch.Tensor,
    owner: torch.Tensor,
    score_stack: list[torch.Tensor],
    candidate_masks: list[torch.Tensor],
    config: CORSettings,
    kind: str,
) -> tuple[torch.Tensor, dict]:
    """Apply one frozen-neighborhood completion pass for the final COR."""
    owner_np = owner.detach().cpu().numpy().astype(np.int32, copy=True)
    initial = int((owner_np == 0).sum())
    selected_before = int((owner_np > 0).sum())
    enabled = bool(getattr(config, f"{kind}_completion", getattr(config, f"{kind}_residual_fill", False)))
    if kind == "local":
        prefix = "local_residual"
        min_votes = config.local_residual_min_neighbor_votes
        min_ratio = config.local_residual_min_neighbor_ratio
        min_score = config.local_residual_min_score
        connectivity = config.local_residual_connectivity
        max_ratio = config.local_residual_max_added_ratio
        max_tokens = config.local_residual_max_added_tokens
        max_competing = 1.0
        require_candidate = True
    elif kind == "boundary":
        prefix = "boundary"
        min_votes = config.boundary_min_neighbor_votes
        min_ratio = config.boundary_min_neighbor_ratio
        min_score = config.boundary_min_score
        connectivity = config.boundary_connectivity
        max_ratio = config.boundary_max_added_ratio
        max_tokens = config.boundary_max_added_tokens
        max_competing = 1.0
        require_candidate = True
    else:
        prefix = "hole"
        min_votes = config.hole_min_neighbor_votes
        min_ratio = config.hole_min_neighbor_ratio
        min_score = config.hole_min_score
        connectivity = config.hole_connectivity
        max_ratio = config.hole_max_added_ratio
        max_tokens = config.hole_max_added_tokens
        max_competing = config.hole_max_competing_ratio
        require_candidate = False

    stats = {
        "enabled": enabled,
        "initial_unassigned": initial,
        "final_unassigned": initial,
        "selected_before": selected_before,
        "selected_after": selected_before,
        "candidate_tokens": 0,
        "reassigned_tokens": 0,
        "max_added_tokens": 0,
        "capped": False,
    }
    if not enabled or initial == 0 or not score_stack or not candidate_masks:
        return owner, stats

    scores = torch.stack(score_stack, dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
    candidates_mask = torch.stack(candidate_masks, dim=1).detach().cpu().numpy().astype(bool, copy=False)
    neighbors = _build_sparse_neighbors(coords, int(connectivity))
    budget = min(max(int(max_tokens), 0), int(np.ceil(float(max_ratio) * float(max(selected_before, 1)))))
    stats["max_added_tokens"] = int(budget)
    if budget <= 0:
        return owner, stats

    frozen = owner_np.copy()
    proposals: list[tuple[float, int, float, float, int, int]] = []
    for token_idx in np.flatnonzero(frozen == 0).tolist():
        nbrs = [int(n) for n in neighbors[int(token_idx)] if int(frozen[n]) > 0]
        if not nbrs:
            continue
        votes: dict[int, int] = {}
        for nbr in nbrs:
            part = int(frozen[nbr])
            votes[part] = votes.get(part, 0) + 1
        ranked = sorted(votes.items(), key=lambda item: (item[1], -item[0]), reverse=True)
        best_part, best_votes = ranked[0]
        vote_ratio = float(best_votes / max(len(nbrs), 1))
        competing_ratio = float(sum(v for _, v in ranked[1:]) / max(len(nbrs), 1))
        if best_votes < int(min_votes) or vote_ratio < float(min_ratio):
            continue
        if kind == "hole" and competing_ratio > float(max_competing):
            continue
        if require_candidate and not bool(candidates_mask[token_idx, best_part - 1]):
            continue
        local_score = float(scores[token_idx, best_part - 1])
        if local_score < float(min_score) and (not require_candidate or bool(candidates_mask[token_idx, best_part - 1])):
            continue
        priority = vote_ratio * 100.0 + float(best_votes) * 10.0 - competing_ratio + local_score
        proposals.append((priority, int(best_votes), vote_ratio, local_score, int(token_idx), int(best_part)))

    proposals.sort(reverse=True)
    selected = proposals[:budget]
    for _priority, _votes, _ratio, _score, token_idx, part in selected:
        owner_np[token_idx] = part
    stats.update(
        {
            "final_unassigned": int((owner_np == 0).sum()),
            "selected_after": int((owner_np > 0).sum()),
            "candidate_tokens": int(len(proposals)),
            "reassigned_tokens": int(len(selected)),
            "max_added_tokens": int(budget),
            "capped": bool(len(proposals) > len(selected)),
            "connectivity": int(connectivity),
            "min_neighbor_votes": int(min_votes),
            "min_neighbor_ratio": float(min_ratio),
            "min_score": float(min_score),
        }
    )
    if kind == "hole":
        stats["max_competing_ratio"] = float(max_competing)
    return torch.as_tensor(owner_np, dtype=torch.long, device=owner.device), stats


def complete_categorical_ownership(
    *,
    coords: torch.Tensor,
    owner: torch.Tensor,
    score_stack: list[torch.Tensor],
    candidate_masks: list[torch.Tensor],
    config: CORSettings,
) -> tuple[torch.Tensor, dict]:
    """Final bounded local, boundary, and enclosed-hole COR completion."""
    current, _routing_owner, stats = complete_categorical_ownership_with_routing(
        coords=coords,
        owner=owner,
        score_stack=score_stack,
        candidate_masks=candidate_masks,
        config=config,
        routing_owner_stage="after_full",
    )
    return current, stats


def complete_categorical_ownership_with_routing(
    *,
    coords: torch.Tensor,
    owner: torch.Tensor,
    score_stack: list[torch.Tensor],
    candidate_masks: list[torch.Tensor],
    config: CORSettings,
    routing_owner_stage: str = "after_full",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Complete COR and return a separately selected owner for OEHR routing."""
    current = owner
    stage_stats = {}
    routing_owner = owner
    for kind in ("local", "boundary", "hole"):
        current, stage_stats[kind] = _bounded_completion(
            coords=coords,
            owner=current,
            score_stack=score_stack,
            candidate_masks=candidate_masks,
            config=config,
            kind=kind,
        )
        if str(routing_owner_stage) == f"after_{kind}":
            routing_owner = current.clone()
    if str(routing_owner_stage) == "after_full":
        routing_owner = current.clone()
    elif str(routing_owner_stage) == "after_core":
        routing_owner = owner.clone()
    return current, routing_owner, {
        "mode": "bounded_completion",
        "initial_unassigned": int((owner == 0).sum().item()),
        "final_unassigned": int((current == 0).sum().item()),
        "routing_owner_stage": str(routing_owner_stage),
        "stages": stage_stats,
    }


def lift_categorical_ownership(
    *,
    lr_coords: torch.Tensor,
    lr_values: torch.Tensor,
    hr_coords: torch.Tensor,
    lr_resolution: int,
    hr_resolution: int,
    dtype: torch.dtype,
    default_value: float | int,
) -> torch.Tensor:
    lr_grid = max(int(lr_resolution // 16), 1)
    hr_grid = max(int(hr_resolution // 16), 1)
    scale = lr_grid / float(hr_grid)

    lr_np = lr_coords[:, 1:].detach().cpu().numpy().astype(np.int32, copy=False)
    if dtype.is_floating_point:
        lr_value_np = lr_values.detach().cpu().numpy().astype(np.float32, copy=False)
        out_dtype = np.float32
    else:
        lr_value_np = lr_values.detach().cpu().numpy().astype(np.int64, copy=False)
        out_dtype = np.int64
    value_by_coord = {tuple(coord.tolist()): value for coord, value in zip(lr_np, lr_value_np.tolist())}

    hr_np = hr_coords[:, 1:].detach().cpu().numpy().astype(np.int32, copy=False)
    parent = np.floor(hr_np.astype(np.float32) * scale).astype(np.int32, copy=False)
    parent = np.clip(parent, 0, max(lr_grid - 1, 0))
    out_np = np.array(
        [value_by_coord.get(tuple(coord.tolist()), default_value) for coord in parent],
        dtype=out_dtype,
    )
    return torch.as_tensor(out_np, dtype=dtype, device=hr_coords.device)


def build_key_group_index(
    *,
    group_token_counts: list[int],
    structure_token_count: int,
    device: torch.device,
) -> torch.Tensor:
    values: list[int] = []
    for group_idx, token_count in enumerate(group_token_counts, start=1):
        values.extend([group_idx] * int(token_count))
    values.extend([0] * int(structure_token_count))
    return torch.tensor(values, dtype=torch.long, device=device)


def merge_conditions(*conds: dict) -> dict:
    return {
        "cond": torch.cat([cond["cond"] for cond in conds], dim=1),
        "neg_cond": torch.cat([cond["neg_cond"] for cond in conds], dim=1),
    }
