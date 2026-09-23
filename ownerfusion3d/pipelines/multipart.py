"""Synchronized categorical multi-part, multi-reference OwnerFusion3D."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from ..io.outputs import (
    FusionResult,
    save_categorical_owner_glb,
    save_categorical_owner_ply,
    save_owner_glb,
    write_metadata,
)
from ..io.parts import (
    PartManifest,
    masks_from_labels,
    masks_from_manifest_parts,
    preprocess_structure_and_labels,
    preprocess_structure_and_masks,
)
from ..methods import cor
from ..methods.oehr import (
    build_hr_coords_from_lr_shape,
    export_glb,
    sample_shape_with_regional_attention,
    sample_tex_with_regional_attention,
)
from ..runtime import cleanup_cuda, seed_everything


def _cor_namespace(config) -> SimpleNamespace:
    return SimpleNamespace(
        mode=config.mode,
        owner_connectivity=config.connectivity,
        owner_core_quantile=config.core_quantile,
        owner_core_min_score=config.core_min_score,
        owner_core_min_tokens=config.core_min_tokens,
        owner_boundary_margin=config.boundary_margin,
        owner_distance_weight=config.distance_weight,
        owner_core_neighbor_weight=config.core_neighbor_weight,
        owner_assigned_neighbor_weight=config.assigned_neighbor_weight,
        owner_fill_unassigned_iters=config.fill_unassigned_iterations,
        owner_repair_enable=config.repair_iterations > 0,
        owner_repair_iters=config.repair_iterations,
        owner_repair_connectivity=config.repair_connectivity,
        owner_repair_min_neighbor_votes=config.repair_min_neighbor_votes,
        owner_repair_min_neighbor_ratio=config.repair_min_neighbor_ratio,
        owner_repair_decisive_margin=config.repair_decisive_margin,
        owner_repair_candidate_bonus=config.repair_candidate_bonus,
        local_residual_fill=config.local_residual_fill,
        local_residual_connectivity=config.local_residual_connectivity,
        local_residual_min_neighbor_votes=config.local_residual_min_neighbor_votes,
        local_residual_min_neighbor_ratio=config.local_residual_min_neighbor_ratio,
        local_residual_min_score=config.local_residual_min_score,
        local_residual_max_added_ratio=config.local_residual_max_added_ratio,
        local_residual_max_added_tokens=config.local_residual_max_added_tokens,
        component_residual_completion=config.component_residual_completion,
        component_residual_connectivity=config.component_residual_connectivity,
        component_residual_max_component_tokens=config.component_residual_max_component_tokens,
        component_residual_min_boundary_votes=config.component_residual_min_boundary_votes,
        component_residual_min_score=config.component_residual_min_score,
        component_residual_min_margin=config.component_residual_min_margin,
        component_residual_max_added_ratio=config.component_residual_max_added_ratio,
        component_residual_max_added_tokens=config.component_residual_max_added_tokens,
        final_neighbor_assignment=config.final_neighbor_assignment,
        final_neighbor_connectivity=config.final_neighbor_connectivity,
        final_neighbor_min_votes=config.final_neighbor_min_votes,
        final_neighbor_min_ratio=config.final_neighbor_min_ratio,
        final_neighbor_force_completion=config.final_neighbor_force_completion,
        conflict_correction=config.conflict_correction,
        conflict_correction_min_margin=config.conflict_correction_min_margin,
        conflict_correction_min_neighbor_ratio=config.conflict_correction_min_neighbor_ratio,
    )


class MultiPartOwnerFusionPipeline:
    """Run per-mask MOE, resolve conflicts with COR, then route with OEHR."""

    def __init__(self, ownerfusion_pipeline):
        self.ownerfusion = ownerfusion_pipeline
        self.backbone = ownerfusion_pipeline.backbone
        self.config = ownerfusion_pipeline.config

    @torch.no_grad()
    def fuse(
        self,
        structure_image: str | Path | Image.Image,
        part_label_image: str | Path | Image.Image,
        part_manifest: str | Path | PartManifest,
        output_dir: str | Path,
    ) -> FusionResult:
        manifest = part_manifest if isinstance(part_manifest, PartManifest) else PartManifest.load(part_manifest)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ownership_dir = output_dir / "ownership" if self.config.save_ownership else None
        if ownership_dir:
            ownership_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = output_dir / "mesh.glb"
        metadata_path = output_dir / "metadata.json"

        structure_raw = self.ownerfusion._open_image(structure_image, "RGBA")
        if part_label_image is None:
            raw_masks = masks_from_manifest_parts(manifest)
            structure, processed_masks, foreground = preprocess_structure_and_masks(
                self.backbone, structure_raw, raw_masks
            )
            part_masks = {
                part.name: mask for part, mask in zip(manifest.parts, processed_masks)
            }
        else:
            labels_raw = self.ownerfusion._open_image(part_label_image, "RGB")
            structure, labels, foreground = preprocess_structure_and_labels(
                self.backbone, structure_raw, labels_raw
            )
            part_masks = masks_from_labels(labels, manifest.parts)
        seed_everything(self.config.seed)

        structure_512 = self.backbone.get_cond([structure], resolution=512)
        structure_1024 = self.backbone.get_cond([structure], resolution=1024)
        coordinates_lr = self.backbone.sample_sparse_structure(
            structure_512, self.config.sparse_resolution, 1, {}
        )
        if coordinates_lr.numel() == 0:
            raise RuntimeError("TRELLIS.2 returned an empty sparse structure.")

        scores: list[torch.Tensor] = []
        candidates: list[torch.Tensor] = []
        strengths: list[torch.Tensor] = []
        part_stats: list[dict] = []
        for part in manifest.parts:
            mask = part_masks[part.name]
            patch_data = self.ownerfusion._patch_evidence(mask, foreground, structure_512, 512)
            owner, strength, moe_stats = self.ownerfusion._probe(
                structure_512,
                coordinates_lr,
                self.backbone.models["shape_slat_flow_model_512"],
                patch_data,
            )
            scores.append(moe_stats["_evidence_tensor"].detach())
            candidates.append(owner.detach())
            strengths.append(strength.detach())
            part_stats.append(
                {
                    "name": part.name,
                    "color": part.color,
                    "patches_512": patch_data[-1],
                    "moe": moe_stats,
                }
            )
            cleanup_cuda()

        cor_config = _cor_namespace(self.config.cor)
        if str(cor_config.mode) == "score_only":
            part_owner_lr, cor_stats = cor.resolve_score_only_ownership(
                score_stack=scores,
                candidate_masks=candidates,
                device=coordinates_lr.device,
            )
            repair_stats = {"enabled": False, "ablation_skipped": True}
            completion_stats = {"enabled": False, "ablation_skipped": True}
        else:
            part_owner_lr, cor_stats = cor.resolve_categorical_ownership(
                coords=coordinates_lr, score_stack=scores, candidate_masks=candidates, config=cor_config
            )
            part_owner_lr, repair_stats = cor.repair_categorical_ownership(
                coords=coordinates_lr,
                owner=part_owner_lr,
                score_stack=scores,
                candidate_masks=candidates,
                part_strengths=strengths,
                config=cor_config,
            )
            part_owner_lr, routing_part_owner_lr, completion_stats = cor.complete_categorical_ownership_with_routing(
                coords=coordinates_lr,
                owner=part_owner_lr,
                score_stack=scores,
                candidate_masks=candidates,
                config=cor_config,
            )

        part_index = {part.name: index for index, part in enumerate(manifest.parts, start=1)}
        # The released method routes OEHR from the same fully completed COR
        # field that is reported and exported.  Keep the routing variable for
        # backwards-compatible metadata/output names only.
        routing_part_owner_lr = part_owner_lr
        group_owner_lr = torch.zeros_like(routing_part_owner_lr, dtype=torch.long)
        group_conditions_512 = []
        group_conditions_1024 = []
        group_metadata = []
        for group_index, group in enumerate(manifest.groups, start=1):
            selected = torch.zeros_like(part_owner_lr, dtype=torch.bool)
            for part_name in group.parts:
                index = part_index[part_name]
                current = routing_part_owner_lr == index
                selected |= current
            group_owner_lr[selected] = group_index
            if isinstance(group.content_image, Image.Image):
                content_image = group.content_image.convert("RGBA")
            else:
                content_image = Image.open(group.content_image).convert("RGBA")
            content = self.backbone.preprocess_image(content_image)
            group_conditions_512.append(self.backbone.get_cond([content], resolution=512))
            group_conditions_1024.append(self.backbone.get_cond([content], resolution=1024))
            group_metadata.append({"name": group.name, "content_image": str(group.content_image), "parts": group.parts})

        condition_512 = cor.merge_conditions(*group_conditions_512, structure_512)
        key_groups_512 = cor.build_key_group_index(
            group_token_counts=[int(item["cond"].shape[1]) for item in group_conditions_512],
            structure_token_count=int(structure_512["cond"].shape[1]),
            device=coordinates_lr.device,
        )
        routing_lr = {
            "regional_group_owner": group_owner_lr,
            "regional_key_group_index": key_groups_512,
            "regional_attention_mode": "exclusive",
            "regional_grouped_attention": self.config.multipart.attention_backend == "owner_grouped",
            "steps": self.config.oehr.shape_steps,
        }
        shape_lr = sample_shape_with_regional_attention(
            pipeline=self.backbone,
            cond=condition_512,
            flow_model=self.backbone.models["shape_slat_flow_model_512"],
            coords=coordinates_lr,
            sampler_params=routing_lr,
            probe_state=None,
            chunk_size=self.config.multipart.attention_chunk_size,
        )
        coordinates_hr, resolution = build_hr_coords_from_lr_shape(
            self.backbone, shape_lr, 512, self.config.output_resolution, self.config.max_num_tokens
        )
        # Lift the same final COR field that was used by LR routing.  The
        # duplicated routing variable is retained only for output compatibility.
        part_owner_hr = cor.lift_categorical_ownership(
            lr_coords=coordinates_lr,
            lr_values=part_owner_lr,
            hr_coords=coordinates_hr,
            lr_resolution=512,
            hr_resolution=resolution,
            dtype=torch.long,
            default_value=0,
        )
        routing_part_owner_hr = cor.lift_categorical_ownership(
            lr_coords=coordinates_lr,
            lr_values=routing_part_owner_lr,
            hr_coords=coordinates_hr,
            lr_resolution=512,
            hr_resolution=resolution,
            dtype=torch.long,
            default_value=0,
        )
        group_owner_hr = cor.lift_categorical_ownership(
            lr_coords=coordinates_lr,
            lr_values=group_owner_lr,
            hr_coords=coordinates_hr,
            lr_resolution=512,
            hr_resolution=resolution,
            dtype=torch.long,
            default_value=0,
        )
        condition_1024 = cor.merge_conditions(*group_conditions_1024, structure_1024)
        key_groups_1024 = cor.build_key_group_index(
            group_token_counts=[int(item["cond"].shape[1]) for item in group_conditions_1024],
            structure_token_count=int(structure_1024["cond"].shape[1]),
            device=coordinates_hr.device,
        )
        routing_hr = {
            "regional_group_owner": group_owner_hr,
            "regional_key_group_index": key_groups_1024,
            "regional_attention_mode": "exclusive",
            "regional_grouped_attention": self.config.multipart.attention_backend == "owner_grouped",
            "steps": self.config.oehr.shape_steps,
        }
        shape = sample_shape_with_regional_attention(
            pipeline=self.backbone,
            cond=condition_1024,
            flow_model=self.backbone.models["shape_slat_flow_model_1024"],
            coords=coordinates_hr,
            sampler_params=routing_hr,
            probe_state=None,
            chunk_size=self.config.multipart.attention_chunk_size,
        )
        texture = sample_tex_with_regional_attention(
            pipeline=self.backbone,
            cond=condition_1024,
            flow_model=self.backbone.models["tex_slat_flow_model_1024"],
            shape_slat=shape,
            sampler_params={**routing_hr, "steps": self.config.oehr.texture_steps},
            probe_state=None,
            chunk_size=self.config.multipart.attention_chunk_size,
        )
        mesh = self.backbone.decode_latent(shape, texture, resolution)[0]
        export_glb(mesh, str(mesh_path), texture_size=self.config.texture_size)

        part_colors = [(125, 132, 145), *[part.color for part in manifest.parts]]
        group_colors = [(125, 132, 145), *[manifest.parts[part_index[group.parts[0]] - 1].color for group in manifest.groups]]
        selected_voxel_path = output_dir / f"{output_dir.name}_hr_front_tokens_selected_voxel.glb"
        save_categorical_owner_glb(
            coordinates_hr,
            group_owner_hr,
            group_colors,
            selected_voxel_path,
            resolution,
        )
        if ownership_dir:
            save_categorical_owner_ply(coordinates_lr, part_owner_lr, part_colors, ownership_dir / "part_owner_lr.ply", 512)
            # Keep the owner used by OEHR separate from the fully completed
            # COR readout.  This makes the routing/diagnostic contract
            # inspectable without changing the generated asset.
            save_categorical_owner_ply(
                coordinates_lr,
                routing_part_owner_lr,
                part_colors,
                ownership_dir / "part_owner_lr_routing.ply",
                512,
            )
            save_categorical_owner_ply(coordinates_hr, part_owner_hr, part_colors, ownership_dir / "part_owner_hr.ply", resolution)
            save_categorical_owner_ply(
                coordinates_hr,
                routing_part_owner_hr,
                part_colors,
                ownership_dir / "part_owner_hr_routing.ply",
                resolution,
            )
            save_categorical_owner_ply(coordinates_hr, group_owner_hr, group_colors, ownership_dir / "source_owner_hr.ply", resolution)
            save_owner_glb(coordinates_hr, group_owner_hr > 0, ownership_dir / "source_owner_hr.glb", resolution)

        completion_stages = completion_stats.get("stages", {}) if isinstance(completion_stats, dict) else {}
        conflict_stage = completion_stages.get("conflict_correction", {})
        neighbor_stage = completion_stages.get("final_neighbor", {})
        ownership_audit = {
            "initial_owner_zero": int(completion_stats.get("initial_unassigned", cor_stats.get("unassigned_tokens", 0))),
            "final_owner_zero": int((routing_part_owner_lr == 0).sum().item()),
            "candidate_conflicts": int(cor_stats.get("conflict_tokens", 0)),
            "corrected_tokens": int(conflict_stage.get("corrected_tokens", 0)),
            "forced_tokens": int(neighbor_stage.get("forced_tokens", 0)),
            "cleared_existing_tokens": int(
                conflict_stage.get("cleared_existing_tokens", 0)
                + neighbor_stage.get("cleared_existing_tokens", 0)
            ),
        }

        write_metadata(
            metadata_path,
            {
                "method": "OwnerFusion3D",
                "version": "1.0.0",
                "mode": "multi_part",
                "module_state": {"MOE": True, "OEHR": True, "COR": True},
                "config": self.config.to_dict(),
                "inputs": {
                    "structure_image": str(structure_image),
                    "part_label_image": str(part_label_image),
                    "part_manifest": str(part_manifest) if not isinstance(part_manifest, PartManifest) else "<in-memory>",
                },
                "parts": part_stats,
                "reference_groups": group_metadata,
                "cor": cor_stats,
                "cor_repair": repair_stats,
                "cor_completion": completion_stats,
                "ownership_audit": ownership_audit,
                "cor_routing": {
                    "owner_stage": "after_full",
                    "assigned_lr_tokens": int((routing_part_owner_lr > 0).sum().item()),
                    "unassigned_lr_tokens": int((routing_part_owner_lr == 0).sum().item()),
                    "owner_source": "fully_completed_cor",
                },
                "lr_tokens": int(coordinates_lr.shape[0]),
                "hr_tokens": int(coordinates_hr.shape[0]),
                "resolution": int(resolution),
                "outputs": {
                    "mesh": mesh_path,
                    "ownership": ownership_dir,
                    "hr_front_tokens_selected_voxel": selected_voxel_path,
                },
            },
        )
        return FusionResult(mesh_path, metadata_path, ownership_dir)
