"""OEHR runtime: owner-exclusive cross-attention and hierarchical routing."""

from __future__ import annotations

import gc
import math
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import o_voxel
from trellis2.modules import sparse as sp
from trellis2.modules.utils import manual_cast
from trellis2.pipelines import Trellis2ImageTo3DPipeline


@dataclass(frozen=True)
class CondLayout:
    token_count: int
    additional_tokens: int
    grid_h: int
    grid_w: int
    patch_size: int

    @property
    def patch_count(self) -> int:
        return self.grid_h * self.grid_w


class AttentionProbeState:
    def __init__(
        self,
        selected_token_indices: torch.Tensor,
        background_token_indices: Optional[torch.Tensor] = None,
        selected_token_weights: Optional[torch.Tensor] = None,
        background_token_weights: Optional[torch.Tensor] = None,
        collect_head_scores: bool = False,
    ):
        self.selected_token_indices = selected_token_indices
        self.background_token_indices = background_token_indices
        if selected_token_weights is not None:
            selected_token_weights = selected_token_weights.detach().float().flatten()
            if selected_token_weights.numel() != selected_token_indices.numel():
                raise ValueError("selected_token_weights must match selected_token_indices length.")
        if background_token_weights is not None and background_token_indices is not None:
            background_token_weights = background_token_weights.detach().float().flatten()
            if background_token_weights.numel() != background_token_indices.numel():
                raise ValueError("background_token_weights must match background_token_indices length.")
        self.selected_token_weights = selected_token_weights
        self.background_token_weights = background_token_weights
        self.collect_head_scores = bool(collect_head_scores)
        self.head_score_records: list[dict[str, object]] = []
        self._head_block_call_counts: dict[int, int] = {}
        self.score_sum: Optional[torch.Tensor] = None
        self.background_score_sum: Optional[torch.Tensor] = None
        self.selected_token_count = int(selected_token_indices.numel())
        self.background_token_count = (
            int(background_token_indices.numel())
            if background_token_indices is not None
            else 0
        )
        self.selected_token_normalizer = float(
            selected_token_weights.sum().item()
            if selected_token_weights is not None and selected_token_weights.numel() > 0
            else max(self.selected_token_count, 1)
        )
        self.background_token_normalizer = float(
            background_token_weights.sum().item()
            if background_token_weights is not None and background_token_weights.numel() > 0
            else max(self.background_token_count, 1)
        )
        self.count = 0

    def update_head_scores(
        self,
        selected_mass: torch.Tensor,
        background_mass: Optional[torch.Tensor],
        *,
        block_index: int,
    ) -> None:
        """Record per-head patch mass for the stable MOE readout."""
        if not self.collect_head_scores:
            return
        selected_mass = selected_mass.detach().float()
        if selected_mass.ndim != 2:
            raise ValueError("Per-head selected mass must have shape [heads, sparse_tokens].")
        if background_mass is None:
            background_mass = torch.zeros_like(selected_mass)
        else:
            background_mass = background_mass.detach().float()
        if background_mass.shape != selected_mass.shape:
            raise ValueError("Per-head positive and negative mass tensors must have identical shapes.")

        block_index = int(block_index)
        step_index = int(self._head_block_call_counts.get(block_index, 0))
        self._head_block_call_counts[block_index] = step_index + 1
        self.head_score_records.append(
            {
                "block_index": block_index,
                "step_index": step_index,
                "selected_mass": selected_mass.cpu(),
                "background_mass": background_mass.cpu(),
            }
        )

    def head_scores_payload(self) -> Optional[dict[str, object]]:
        if not self.collect_head_scores or not self.head_score_records:
            return None
        return {
            "records": list(self.head_score_records),
            "selected_token_normalizer": float(self.selected_token_normalizer),
            "background_token_normalizer": float(self.background_token_normalizer),
        }

    def update(
        self,
        full_selected_mass: torch.Tensor,
        full_background_mass: Optional[torch.Tensor] = None,
    ) -> None:
        full_selected_mass = full_selected_mass.detach().float()
        if self.score_sum is None:
            self.score_sum = torch.zeros_like(full_selected_mass)
        self.score_sum += full_selected_mass

        if full_background_mass is not None:
            full_background_mass = full_background_mass.detach().float()
            if self.background_score_sum is None:
                self.background_score_sum = torch.zeros_like(full_background_mass)
            self.background_score_sum += full_background_mass

        self.count += 1

    def scores(self) -> torch.Tensor:
        if self.score_sum is None or self.count == 0:
            raise RuntimeError("No attention scores were collected.")
        return self.score_sum / float(self.count)

    def background_scores(self) -> torch.Tensor:
        base = self.scores()
        if self.background_score_sum is None or self.count == 0:
            return torch.zeros_like(base)
        return self.background_score_sum.to(base.device) / float(self.count)


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _to_long_tensor(values: Iterable[int], device: torch.device) -> torch.Tensor:
    values = list(int(v) for v in values)
    if not values:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(values, dtype=torch.long, device=device)


def infer_patch_size(image_cond_model, resolution: int, token_count: int) -> int:
    patch_size = getattr(getattr(image_cond_model, "model", None), "config", None)
    patch_size = getattr(patch_size, "patch_size", None)
    if isinstance(patch_size, (tuple, list)):
        patch_size = int(patch_size[0])
    elif patch_size is not None:
        patch_size = int(patch_size)

    if patch_size is not None and patch_size > 0:
        patch_count = (resolution // patch_size) ** 2
        if 0 <= token_count - patch_count <= 64:
            return patch_size

    for candidate in (14, 16, 32):
        if resolution % candidate != 0:
            continue
        patch_count = (resolution // candidate) ** 2
        if 0 <= token_count - patch_count <= 64:
            return candidate
    raise ValueError(
        f"Could not infer patch size for resolution={resolution}, tokens={token_count}."
    )


def infer_cond_layout(pipeline, cond: dict, resolution: int) -> CondLayout:
    token_count = int(cond["cond"].shape[1])
    patch_size = infer_patch_size(pipeline.image_cond_model, resolution, token_count)
    grid_h = resolution // patch_size
    grid_w = resolution // patch_size
    additional_tokens = token_count - grid_h * grid_w
    if additional_tokens < 0:
        raise ValueError(
            f"Invalid condition layout: token_count={token_count}, "
            f"grid={grid_h}x{grid_w}."
        )
    return CondLayout(
        token_count=token_count,
        additional_tokens=additional_tokens,
        grid_h=grid_h,
        grid_w=grid_w,
        patch_size=patch_size,
    )


def cat_conds(*conds: dict) -> dict:
    return {
        "cond": torch.cat([cond["cond"] for cond in conds], dim=1),
        "neg_cond": torch.cat([cond["neg_cond"] for cond in conds], dim=1),
    }


def majority_smooth_indices(
    coords_xyz: torch.Tensor,
    selected_indices: torch.Tensor,
    k: int,
    vote_ratio: float,
) -> torch.Tensor:
    if selected_indices.numel() == 0:
        return selected_indices
    points = coords_xyz.float()
    n = int(points.shape[0])
    k = min(max(int(k), 1), n)
    mask = torch.zeros(n, dtype=torch.bool, device=points.device)
    mask[selected_indices.long()] = True

    dist = torch.cdist(points, points)
    knn = dist.topk(k=k, largest=False).indices
    neigh_count = mask[knn].sum(dim=1)
    new_mask = mask.clone()
    new_mask[(~mask) & (neigh_count >= k * vote_ratio)] = True
    new_mask[mask & (neigh_count < k * (1.0 - vote_ratio))] = False
    return torch.nonzero(new_mask, as_tuple=False).flatten()


class RegionalSparseCrossAttention(nn.Module):
    def __init__(
        self,
        parent: nn.Module,
        probe_state: Optional[AttentionProbeState],
        chunk_size: int,
        block_index: int = -1,
    ):
        super().__init__()
        self.parent = parent
        self.probe_state = probe_state
        self.chunk_size = int(chunk_size)
        self.block_index = int(block_index)

    def forward(self, x: sp.SparseTensor, context=None, **kwargs) -> sp.SparseTensor:
        use_regional = (
            kwargs.get("regional_region_mask") is not None
            or kwargs.get("regional_group_owner") is not None
        )
        use_probe = self.probe_state is not None
        if not use_regional and not use_probe:
            return self.parent(x, context)

        if getattr(self.parent, "_type", "cross") != "cross":
            return self.parent(x, context)

        if not isinstance(context, torch.Tensor):
            raise TypeError(
                "Regional attention currently expects dense image condition tensors. "
                "Use a single concatenated condition, not a VarLenTensor list."
            )
        if context.shape[0] != 1:
            raise ValueError("Regional attention currently supports batch size 1.")

        q = self.parent._linear(self.parent.to_q, x)
        q = self.parent._reshape_chs(q, (self.parent.num_heads, -1))
        kv = self.parent._linear(self.parent.to_kv, context)
        kv = self.parent._fused_pre(kv, num_fused=2)

        if self.parent.qk_rms_norm:
            q = self.parent.q_rms_norm(q)
            k, v = kv.unbind(dim=-3)
            k = self.parent.k_rms_norm(k)
            k_feats = k[0].permute(1, 0, 2).contiguous()
            v_feats = v[0].permute(1, 0, 2).contiguous()
        else:
            k_feats = kv[0, :, 0].permute(1, 0, 2).contiguous()
            v_feats = kv[0, :, 1].permute(1, 0, 2).contiguous()

        q_feats = q.feats.permute(1, 0, 2).contiguous()
        head_dim = q_feats.shape[-1]
        token_count = int(q_feats.shape[1])
        key_count = int(k_feats.shape[1])
        chunk_size = max(1, min(self.chunk_size, token_count))

        region_mask = kwargs.get("regional_region_mask")
        local_token_indices = kwargs.get("regional_local_token_indices")
        base_token_indices = kwargs.get("regional_base_token_indices")
        group_owner = kwargs.get("regional_group_owner")
        key_group_index = kwargs.get("regional_key_group_index")
        attention_mode = kwargs.get("regional_attention_mode", "exclusive")
        if region_mask is not None:
            region_mask = region_mask.to(q.device, dtype=torch.bool).flatten()
            if region_mask.shape[0] != token_count:
                raise ValueError(
                    f"Region mask length {region_mask.shape[0]} does not match "
                    f"query token count {token_count}."
                )
            local_token_indices = local_token_indices.to(q.device, dtype=torch.long)
            base_token_indices = base_token_indices.to(q.device, dtype=torch.long)
        if group_owner is not None:
            group_owner = group_owner.to(q.device, dtype=torch.long).flatten()
            if group_owner.shape[0] != token_count:
                raise ValueError(
                    f"Group owner length {group_owner.shape[0]} does not match "
                    f"query token count {token_count}."
                )
        if key_group_index is not None:
            key_group_index = key_group_index.to(q.device, dtype=torch.long).flatten()
            if key_group_index.shape[0] != key_count:
                raise ValueError(
                    f"Key group index length {key_group_index.shape[0]} does not match "
                    f"attention key count {key_count}."
                )
        scale = 1.0 / math.sqrt(float(head_dim))
        selected_probe_indices = None
        background_probe_indices = None
        if self.probe_state is not None:
            selected_probe_indices = self.probe_state.selected_token_indices.to(
                q.device,
                dtype=torch.long,
            )
            selected_probe_indices = selected_probe_indices[
                (selected_probe_indices >= 0) & (selected_probe_indices < key_count)
            ]
            if self.probe_state.background_token_indices is not None:
                background_probe_indices = self.probe_state.background_token_indices.to(
                    q.device,
                    dtype=torch.long,
                )
                background_probe_indices = background_probe_indices[
                    (background_probe_indices >= 0)
                    & (background_probe_indices < key_count)
                ]
            selected_probe_weights = None
            background_probe_weights = None
            if self.probe_state.selected_token_weights is not None:
                selected_probe_weights = self.probe_state.selected_token_weights.to(
                    q.device,
                    dtype=torch.float32,
                )
                if selected_probe_weights.numel() != self.probe_state.selected_token_indices.numel():
                    raise ValueError("selected probe weights length mismatch.")
                selected_probe_weights = selected_probe_weights[
                    (self.probe_state.selected_token_indices.to(q.device) >= 0)
                    & (self.probe_state.selected_token_indices.to(q.device) < key_count)
                ]
            if (
                self.probe_state.background_token_weights is not None
                and self.probe_state.background_token_indices is not None
            ):
                background_probe_weights = self.probe_state.background_token_weights.to(
                    q.device,
                    dtype=torch.float32,
                )
                if background_probe_weights.numel() != self.probe_state.background_token_indices.numel():
                    raise ValueError("background probe weights length mismatch.")
                background_probe_weights = background_probe_weights[
                    (self.probe_state.background_token_indices.to(q.device) >= 0)
                    & (self.probe_state.background_token_indices.to(q.device) < key_count)
                ]

        out_chunks = []
        probe_scores = (
            torch.full((token_count,), float("nan"), dtype=torch.float32, device=q.device)
            if self.probe_state is not None
            else None
        )
        background_scores = (
            torch.full((token_count,), float("nan"), dtype=torch.float32, device=q.device)
            if self.probe_state is not None and background_probe_indices is not None
            else None
        )
        head_probe_scores = (
            torch.full(
                (int(q_feats.shape[0]), token_count),
                float("nan"),
                dtype=torch.float32,
                device=q.device,
            )
            if self.probe_state is not None and self.probe_state.collect_head_scores
            else None
        )
        head_background_scores = (
            torch.full_like(head_probe_scores, float("nan"))
            if head_probe_scores is not None and background_probe_indices is not None
            else None
        )

        for start in range(0, token_count, chunk_size):
            end = min(start + chunk_size, token_count)
            q_chunk = q_feats[:, start:end]
            scores = torch.matmul(q_chunk, k_feats.transpose(-2, -1)) * scale

            if region_mask is not None:
                scores = self._apply_regional_bias(
                    scores=scores,
                    region_mask=region_mask[start:end],
                    local_token_indices=local_token_indices,
                    base_token_indices=base_token_indices,
                )
            if group_owner is not None or key_group_index is not None:
                if group_owner is None or key_group_index is None:
                    raise ValueError(
                        "regional_group_owner and regional_key_group_index must be provided together."
                    )
                scores = self._apply_group_owner_bias(
                    scores=scores,
                    row_group_owner=group_owner[start:end],
                    key_group_index=key_group_index,
                    attention_mode=attention_mode,
                )
            probs = torch.softmax(scores.float(), dim=-1)
            if self.probe_state is not None and selected_probe_indices is not None:
                if selected_probe_indices.numel() > 0:
                    selected_probs = probs[:, :, selected_probe_indices]
                    if selected_probe_weights is not None and selected_probe_weights.numel() > 0:
                        selected_probs = selected_probs * selected_probe_weights.view(1, 1, -1)
                    selected_head_mass = selected_probs.sum(dim=-1)
                    mass = selected_head_mass.mean(dim=0)
                    probe_scores[start:end] = mass
                    if head_probe_scores is not None:
                        head_probe_scores[:, start:end] = selected_head_mass
                if (
                    background_scores is not None
                    and background_probe_indices is not None
                    and background_probe_indices.numel() > 0
                ):
                    background_probs = probs[:, :, background_probe_indices]
                    if background_probe_weights is not None and background_probe_weights.numel() > 0:
                        background_probs = background_probs * background_probe_weights.view(1, 1, -1)
                    background_head_mass = background_probs.sum(dim=-1)
                    background_mass = background_head_mass.mean(dim=0)
                    background_scores[start:end] = background_mass
                    if head_background_scores is not None:
                        head_background_scores[:, start:end] = background_head_mass
            out_chunk = torch.matmul(probs.to(v_feats.dtype), v_feats)
            out_chunks.append(out_chunk.permute(1, 0, 2).contiguous())

        if self.probe_state is not None and probe_scores is not None:
            unset = torch.isnan(probe_scores)
            if unset.any():
                probe_scores[unset] = 0.0
            if background_scores is not None:
                unset = torch.isnan(background_scores)
                if unset.any():
                    background_scores[unset] = 0.0
            if head_probe_scores is not None:
                unset = torch.isnan(head_probe_scores)
                if unset.any():
                    head_probe_scores[unset] = 0.0
                if head_background_scores is not None:
                    unset = torch.isnan(head_background_scores)
                    if unset.any():
                        head_background_scores[unset] = 0.0
                self.probe_state.update_head_scores(
                    head_probe_scores,
                    head_background_scores,
                    block_index=self.block_index,
                )
            self.probe_state.update(
                probe_scores,
                background_scores,
            )

        out_feats = torch.cat(out_chunks, dim=0)
        h = q.replace(out_feats)
        h = self.parent._reshape_chs(h, (-1,))
        h = self.parent._linear(self.parent.to_out, h)
        return h

    @staticmethod
    def _apply_regional_bias(
        scores: torch.Tensor,
        region_mask: torch.Tensor,
        local_token_indices: torch.Tensor,
        base_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        if local_token_indices.numel() == 0 and base_token_indices.numel() == 0:
            return scores

        gated_scores = scores.clone()
        off_value = torch.tensor(-10000.0, dtype=scores.dtype, device=scores.device)
        region_rows = torch.nonzero(region_mask, as_tuple=False).flatten()
        background_rows = torch.nonzero(~region_mask, as_tuple=False).flatten()
        if region_rows.numel() > 0 and base_token_indices.numel() > 0:
            gated_scores[:, region_rows.unsqueeze(1), base_token_indices.unsqueeze(0)] = off_value
        if background_rows.numel() > 0 and local_token_indices.numel() > 0:
            gated_scores[:, background_rows.unsqueeze(1), local_token_indices.unsqueeze(0)] = off_value
        return gated_scores

    @staticmethod
    def _apply_group_owner_bias(
        scores: torch.Tensor,
        row_group_owner: torch.Tensor,
        key_group_index: torch.Tensor,
        attention_mode: str,
    ) -> torch.Tensor:
        row_group_owner = row_group_owner.to(device=scores.device, dtype=torch.long).flatten()
        key_group_index = key_group_index.to(device=scores.device, dtype=torch.long).flatten()

        if attention_mode != "exclusive":
            raise ValueError(
                "regional_group_owner routing currently only supports regional_attention_mode='exclusive'."
            )

        gated_scores = scores.clone()
        off_value = torch.tensor(-10000.0, dtype=scores.dtype, device=scores.device)
        allowed = row_group_owner.unsqueeze(1) == key_group_index.unsqueeze(0)
        gated_scores = gated_scores.masked_fill(~allowed.unsqueeze(0), off_value)
        return gated_scores


@contextmanager
def patched_slat_flow_for_regional_attention(
    flow_model,
    probe_state: Optional[AttentionProbeState] = None,
    chunk_size: int = 768,
):
    old_forward = flow_model.forward
    old_cross_attn = [block.cross_attn for block in flow_model.blocks]
    old_block_forward = [block.forward for block in flow_model.blocks]
    old_block_forward_impl = [block._forward for block in flow_model.blocks]

    def block_forward_impl_with_regional_kwargs(
        self,
        x: sp.SparseTensor,
        mod: torch.Tensor,
        context,
        **kwargs,
    ) -> sp.SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation + mod
            ).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )

        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h

        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context, **kwargs)
        x = x + h

        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def block_forward_with_regional_kwargs(
        self,
        x: sp.SparseTensor,
        mod: torch.Tensor,
        context,
        **kwargs,
    ) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                lambda x_, mod_, context_: self._forward(
                    x_,
                    mod_,
                    context_,
                    **kwargs,
                ),
                x,
                mod,
                context,
                use_reentrant=False,
            )
        return self._forward(x, mod, context, **kwargs)

    def forward_with_regional_kwargs(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond,
        concat_cond: Optional[sp.SparseTensor] = None,
        **kwargs,
    ) -> sp.SparseTensor:
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)

        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        cond = manual_cast(cond, self.dtype)

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond, **kwargs)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h

    try:
        flow_model.forward = types.MethodType(forward_with_regional_kwargs, flow_model)
        for block_index, block in enumerate(flow_model.blocks):
            block._forward = types.MethodType(
                block_forward_impl_with_regional_kwargs,
                block,
            )
            block.forward = types.MethodType(
                block_forward_with_regional_kwargs,
                block,
            )
            block.cross_attn = RegionalSparseCrossAttention(
                block.cross_attn,
                probe_state=probe_state,
                chunk_size=chunk_size,
                block_index=block_index,
            )
        yield
    finally:
        flow_model.forward = old_forward
        for block, old_attn, old_fwd, old_impl in zip(
            flow_model.blocks,
            old_cross_attn,
            old_block_forward,
            old_block_forward_impl,
        ):
            block.cross_attn = old_attn
            block.forward = old_fwd
            block._forward = old_impl


def sample_shape_with_regional_attention(
    pipeline: Trellis2ImageTo3DPipeline,
    cond: dict,
    flow_model,
    coords: torch.Tensor,
    sampler_params: dict,
    probe_state: Optional[AttentionProbeState],
    chunk_size: int,
) -> object:
    with patched_slat_flow_for_regional_attention(
        flow_model,
        probe_state=probe_state,
        chunk_size=chunk_size,
    ):
        return pipeline.sample_shape_slat(cond, flow_model, coords, sampler_params)


def sample_tex_with_regional_attention(
    pipeline: Trellis2ImageTo3DPipeline,
    cond: dict,
    flow_model,
    shape_slat,
    sampler_params: dict,
    probe_state: Optional[AttentionProbeState],
    chunk_size: int,
):
    with patched_slat_flow_for_regional_attention(
        flow_model,
        probe_state=probe_state,
        chunk_size=chunk_size,
    ):
        return pipeline.sample_tex_slat(cond, flow_model, shape_slat, sampler_params)


def build_hr_coords_from_lr_shape(
    pipeline: Trellis2ImageTo3DPipeline,
    lr_slat,
    lr_resolution: int,
    target_resolution: int,
    max_num_tokens: int,
) -> tuple[torch.Tensor, int]:
    if pipeline.low_vram:
        pipeline.models["shape_slat_decoder"].to(pipeline.device)
        pipeline.models["shape_slat_decoder"].low_vram = True

    hr_coords = pipeline.models["shape_slat_decoder"].upsample(lr_slat, upsample_times=4)

    if pipeline.low_vram:
        pipeline.models["shape_slat_decoder"].cpu()
        pipeline.models["shape_slat_decoder"].low_vram = False

    hr_resolution = int(target_resolution)
    while True:
        quant_coords = torch.cat(
            [
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ],
            dim=1,
        )
        coords = quant_coords.unique(dim=0)
        if coords.shape[0] < max_num_tokens or hr_resolution == 1024:
            if hr_resolution != target_resolution:
                print(f"  HR resolution reduced to {hr_resolution} due to token limit.")
            return coords, hr_resolution
        hr_resolution -= 128


def build_hr_mask_from_lr_mask(
    lr_coords: torch.Tensor,
    lr_mask: torch.Tensor,
    hr_coords: torch.Tensor,
    lr_resolution: int,
    hr_resolution: int,
) -> torch.Tensor:
    lr_grid = max(int(lr_resolution // 16), 1)
    hr_grid = max(int(hr_resolution // 16), 1)
    scale = lr_grid / float(hr_grid)

    lr_np = lr_coords[:, 1:].detach().cpu().numpy().astype(np.int32, copy=False)
    selected_lr = {
        tuple(coord.tolist())
        for coord, is_selected in zip(lr_np, lr_mask.detach().cpu().numpy().tolist())
        if bool(is_selected)
    }
    hr_np = hr_coords[:, 1:].detach().cpu().numpy().astype(np.int32, copy=False)
    parent = np.floor(hr_np.astype(np.float32) * scale).astype(np.int32, copy=False)
    parent = np.clip(parent, 0, max(lr_grid - 1, 0))
    hr_mask_np = np.array([tuple(coord.tolist()) in selected_lr for coord in parent])
    return torch.as_tensor(hr_mask_np, dtype=torch.bool, device=hr_coords.device)


def build_regional_sampler_params(
    *,
    region_mask: torch.Tensor,
    local_token_count: int,
    total_token_count: int,
    extra_params: Optional[dict] = None,
) -> dict:
    local_indices = _to_long_tensor(range(local_token_count), region_mask.device)
    base_indices = _to_long_tensor(
        range(local_token_count, total_token_count),
        region_mask.device,
    )
    params = {
        "regional_region_mask": region_mask,
        "regional_local_token_indices": local_indices,
        "regional_base_token_indices": base_indices,
        "regional_attention_mode": "exclusive",
    }
    if extra_params:
        params.update(extra_params)
    return params


def export_glb(mesh, output_path: str, texture_size: int = 64) -> None:
    target_faces = min(int(mesh.faces.shape[0]), 1_000_000)
    if int(mesh.faces.shape[0]) > target_faces:
        mesh.simplify(target_faces)
    _cleanup_cuda()

    base_kwargs = dict(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=texture_size,
        verbose=True,
    )

    try:
        glb = o_voxel.postprocess.to_glb(
            **base_kwargs,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
        )
    except RuntimeError as exc:
        print(f"  export_glb remesh=True failed: {exc}")
        print("  Retrying export with remesh disabled...")
        _cleanup_cuda()
        glb = o_voxel.postprocess.to_glb(
            **base_kwargs,
            remesh=False,
            remesh_band=0,
            remesh_project=0,
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    glb.export(output_path, extension_webp=True)
    del glb
    _cleanup_cuda()
