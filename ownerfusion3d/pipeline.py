"""Public OwnerFusion3D pipeline.

The released single-region path is deliberately compact:

1. sample TRELLIS.2 sparse structure from the structure image;
2. construct MOE ownership directly from frozen LR shape-flow cross-attention;
3. reuse that owner field for LR shape, HR shape, and texture through OEHR.

The historical EVO completion path is not part of this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline

from .config import AblationVariant, OwnerFusionConfig
from .io.outputs import FusionResult, save_owner_glb, save_owner_ply, write_metadata
from .io.preprocess import preprocess_structure_and_mask
from .methods.moe import (
    build_patch_token_sets,
    patch_coverage_from_mask_image,
    probe_ownership_evidence,
    save_evidence_ply,
)
from .methods.oehr import (
    build_hr_coords_from_lr_shape,
    build_hr_mask_from_lr_mask,
    build_regional_sampler_params,
    cat_conds,
    export_glb,
    infer_cond_layout,
    sample_shape_with_regional_attention,
    sample_tex_with_regional_attention,
)
from .runtime import cleanup_cuda, configure_background_removal, seed_everything


def _module_state(variant: AblationVariant) -> dict[str, bool]:
    if variant is AblationVariant.FULL:
        return {"MOE": True, "OEHR": True, "COR": False}
    if variant is AblationVariant.WITHOUT_MOE:
        return {"MOE": False, "OEHR": False, "COR": False}
    if variant is AblationVariant.WITHOUT_OEHR:
        return {"MOE": True, "OEHR": False, "COR": False}
    raise ValueError(variant)


class OwnerFusionPipeline:
    """Training-free local 3D fusion built on a frozen TRELLIS.2 pipeline."""

    def __init__(self, backbone: Trellis2ImageTo3DPipeline, config: OwnerFusionConfig | None = None):
        self.backbone = backbone
        self.config = config or OwnerFusionConfig()

    @classmethod
    def from_pretrained(cls, config: OwnerFusionConfig | None = None) -> "OwnerFusionPipeline":
        config = config or OwnerFusionConfig()
        configure_background_removal()
        backbone = Trellis2ImageTo3DPipeline.from_pretrained(config.model)
        backbone.cuda()
        return cls(backbone, config)

    @torch.no_grad()
    def fuse(
        self,
        structure_image: str | Path | Image.Image,
        structure_mask: str | Path | Image.Image,
        content_image: str | Path | Image.Image,
        output_dir: str | Path,
        *,
        variant: AblationVariant | str = AblationVariant.FULL,
    ) -> FusionResult:
        """Run single-region OwnerFusion3D and write a stable case directory."""
        variant = AblationVariant(variant)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ownership_dir = output_dir / "ownership" if self.config.save_ownership else None
        if ownership_dir:
            ownership_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = output_dir / "mesh.glb"
        metadata_path = output_dir / "metadata.json"

        structure = self._open_image(structure_image, "RGBA")
        mask = self._open_image(structure_mask, "L")
        content = self._open_image(content_image, "RGBA")
        metadata = self._run_single(structure, mask, content, mesh_path, ownership_dir, variant)
        metadata.update(
            method="OwnerFusion3D",
            version="1.0.0",
            variant=variant.value,
            module_state=_module_state(variant),
            config=self.config.to_dict(),
            inputs={
                "structure_image": str(structure_image),
                "structure_mask": str(structure_mask),
                "content_image": str(content_image),
            },
            outputs={"mesh": str(mesh_path), "ownership": str(ownership_dir) if ownership_dir else None},
        )
        write_metadata(metadata_path, metadata)
        return FusionResult(mesh_path, metadata_path, ownership_dir)

    @staticmethod
    def _open_image(value: str | Path | Image.Image, mode: str) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert(mode)
        return Image.open(value).convert(mode)

    def fuse_parts(self, structure_image, part_label_image, part_manifest, output_dir) -> FusionResult:
        """Run the synchronized multi-part, multi-reference extension."""
        from .pipelines.multipart import MultiPartOwnerFusionPipeline

        return MultiPartOwnerFusionPipeline(self).fuse(
            structure_image, part_label_image, part_manifest, output_dir
        )

    def fuse_auto(
        self,
        structure_image: str | Path | Image.Image,
        masks: Sequence[str | Path | Image.Image],
        contents: Sequence[str | Path | Image.Image],
        output_dir: str | Path,
    ) -> FusionResult:
        """Select single- or multi-mask routing from the number of masks."""
        masks = tuple(masks)
        contents = tuple(contents)
        if len(masks) == 0:
            raise ValueError("At least one mask is required.")
        if len(masks) != len(contents):
            raise ValueError("The number of masks must equal the number of content references.")
        if len(masks) == 1:
            return self.fuse(structure_image, masks[0], contents[0], output_dir)

        from .io.parts import PartDefinition, PartManifest, ReferenceGroup

        palette = (
            (230, 126, 34),
            (52, 152, 219),
            (46, 204, 113),
            (155, 89, 182),
            (241, 196, 15),
            (231, 76, 60),
            (26, 188, 156),
            (149, 165, 166),
        )
        parts = tuple(
            PartDefinition(
                name=f"part_{index + 1}",
                color=palette[index % len(palette)],
                mask_image=(Path(mask) if not isinstance(mask, Image.Image) else None),
                mask_value=mask,
            )
            for index, mask in enumerate(masks)
        )
        groups = tuple(
            ReferenceGroup(
                name=f"reference_{index + 1}",
                content_image=(Path(content) if not isinstance(content, Image.Image) else content),
                parts=(f"part_{index + 1}",),
            )
            for index, content in enumerate(contents)
        )
        return self.fuse_parts(
            structure_image,
            None,
            PartManifest(parts=parts, groups=groups),
            output_dir,
        )

    def _conditions(self, structure: Image.Image, content: Image.Image):
        structure_512 = self.backbone.get_cond([structure], resolution=512)
        structure_1024 = self.backbone.get_cond([structure], resolution=1024)
        content_512 = self.backbone.get_cond([content], resolution=512)
        content_1024 = self.backbone.get_cond([content], resolution=1024)
        return structure_512, structure_1024, content_512, content_1024

    def _patch_evidence(self, mask, foreground, condition, resolution):
        cfg = self.config.moe
        layout = infer_cond_layout(self.backbone, condition, resolution)
        selected = patch_coverage_from_mask_image(mask, layout)
        valid = patch_coverage_from_mask_image(foreground, layout)
        return build_patch_token_sets(
            layout=layout,
            selected_patch_coverage=selected,
            foreground_patch_coverage=valid,
            device=condition["cond"].device,
            valid_min_coverage=cfg.patch_valid_coverage,
            selected_min_coverage=cfg.patch_selected_coverage,
            background_max_coverage=cfg.patch_background_coverage,
        )

    def _probe(self, condition, coordinates, flow_model, patch_data):
        cfg = self.config.moe
        selected, background, selected_weights, background_weights, _ = patch_data
        return probe_ownership_evidence(
            pipeline=self.backbone,
            condition=condition,
            coordinates=coordinates,
            flow_model=flow_model,
            selected_patch_indices=selected,
            background_patch_indices=background,
            selected_patch_weights=selected_weights,
            background_patch_weights=background_weights,
            steps=cfg.probe_steps_lr,
            threshold=cfg.threshold,
            majority_neighbors=cfg.majority_neighbors,
            majority_vote_ratio=cfg.majority_vote_ratio,
            max_selected_ratio=cfg.max_selected_ratio,
            expand_iterations=cfg.expand_iterations,
            expand_score_delta=cfg.expand_score_delta,
            expand_connectivity=cfg.expand_connectivity,
            disable_expand_on_overselect=cfg.disable_expand_on_overselect,
            chunk_size=self.config.oehr.attention_chunk_size,
            readout=cfg.readout,
            readout_temperature=cfg.readout_temperature,
            stable_block_heads=cfg.stable_block_heads,
            isolate_rng=cfg.isolate_probe_rng,
        )

    def _route_shape(self, condition, flow_model, coordinates, owner, local_count, variant):
        if variant is AblationVariant.WITHOUT_OEHR:
            return self.backbone.sample_shape_slat(
                condition,
                flow_model,
                coordinates,
                {"steps": self.config.oehr.shape_steps},
            )
        params = build_regional_sampler_params(
            region_mask=owner,
            local_token_count=local_count,
            total_token_count=int(condition["cond"].shape[1]),
        )
        return sample_shape_with_regional_attention(
            pipeline=self.backbone,
            cond=condition,
            flow_model=flow_model,
            coords=coordinates,
            sampler_params={**params, "steps": self.config.oehr.shape_steps},
            probe_state=None,
            chunk_size=self.config.oehr.attention_chunk_size,
        )

    def _route_texture(self, condition, flow_model, shape, owner, local_count, variant):
        if variant is AblationVariant.WITHOUT_OEHR:
            return self.backbone.sample_tex_slat(
                condition,
                flow_model,
                shape,
                {"steps": self.config.oehr.texture_steps},
            )
        params = build_regional_sampler_params(
            region_mask=owner,
            local_token_count=local_count,
            total_token_count=int(condition["cond"].shape[1]),
        )
        return sample_tex_with_regional_attention(
            pipeline=self.backbone,
            cond=condition,
            flow_model=flow_model,
            shape_slat=shape,
            sampler_params={**params, "steps": self.config.oehr.texture_steps},
            probe_state=None,
            chunk_size=self.config.oehr.attention_chunk_size,
        )

    def _run_single(self, raw_structure, raw_mask, raw_content, mesh_path, ownership_dir, variant):
        seed_everything(self.config.seed)
        structure, mask, foreground = preprocess_structure_and_mask(self.backbone, raw_structure, raw_mask)
        content = self.backbone.preprocess_image(raw_content)
        structure_512, structure_1024, content_512, content_1024 = self._conditions(structure, content)
        coordinates_lr = self.backbone.sample_sparse_structure(
            structure_512, self.config.sparse_resolution, 1, {}
        )
        if coordinates_lr.numel() == 0:
            raise RuntimeError("TRELLIS.2 returned an empty sparse structure.")

        if variant is AblationVariant.WITHOUT_MOE:
            shape_lr = self.backbone.sample_shape_slat(
                content_512,
                self.backbone.models["shape_slat_flow_model_512"],
                coordinates_lr,
                {"steps": self.config.oehr.shape_steps},
            )
            coordinates_hr, resolution = build_hr_coords_from_lr_shape(
                self.backbone, shape_lr, 512, self.config.output_resolution, self.config.max_num_tokens
            )
            shape = self.backbone.sample_shape_slat(
                content_1024,
                self.backbone.models["shape_slat_flow_model_1024"],
                coordinates_hr,
                {"steps": self.config.oehr.shape_steps},
            )
            texture = self.backbone.sample_tex_slat(
                content_1024,
                self.backbone.models["tex_slat_flow_model_1024"],
                shape,
                {"steps": self.config.oehr.texture_steps},
            )
            mesh = self.backbone.decode_latent(shape, texture, resolution)[0]
            export_glb(mesh, str(mesh_path), texture_size=self.config.texture_size)
            return {
                "ownership": None,
                "routing": "global_content_on_structure_coordinates",
                "lr_tokens": int(coordinates_lr.shape[0]),
                "hr_tokens": int(coordinates_hr.shape[0]),
                "resolution": int(resolution),
            }

        patch_512 = self._patch_evidence(mask, foreground, structure_512, 512)
        owner_lr, strength_lr, moe_stats = self._probe(
            structure_512,
            coordinates_lr,
            self.backbone.models["shape_slat_flow_model_512"],
            patch_512,
        )
        evidence_lr = moe_stats["_evidence_tensor"]
        merged_512 = cat_conds(content_512, structure_512)
        shape_lr = self._route_shape(
            merged_512,
            self.backbone.models["shape_slat_flow_model_512"],
            coordinates_lr,
            owner_lr,
            int(content_512["cond"].shape[1]),
            variant,
        )
        cleanup_cuda()
        coordinates_hr, resolution = build_hr_coords_from_lr_shape(
            self.backbone, shape_lr, 512, self.config.output_resolution, self.config.max_num_tokens
        )
        owner_hr = build_hr_mask_from_lr_mask(coordinates_lr, owner_lr, coordinates_hr, 512, resolution)

        merged_1024 = cat_conds(content_1024, structure_1024)
        shape = self._route_shape(
            merged_1024,
            self.backbone.models["shape_slat_flow_model_1024"],
            coordinates_hr,
            owner_hr,
            int(content_1024["cond"].shape[1]),
            variant,
        )
        texture = self._route_texture(
            merged_1024,
            self.backbone.models["tex_slat_flow_model_1024"],
            shape,
            owner_hr,
            int(content_1024["cond"].shape[1]),
            variant,
        )
        mesh = self.backbone.decode_latent(shape, texture, resolution)[0]
        export_glb(mesh, str(mesh_path), texture_size=self.config.texture_size)

        if ownership_dir:
            save_evidence_ply(coordinates_lr, evidence_lr, ownership_dir / "moe_stable_midlate4_evidence_lr.ply", 512)
            save_owner_ply(coordinates_lr, owner_lr, ownership_dir / "moe_owner_lr.ply", 512)
            save_owner_ply(coordinates_hr, owner_hr, ownership_dir / "owner_hr.ply", resolution)
            save_owner_glb(coordinates_hr, owner_hr, ownership_dir / "owner_hr.glb", resolution)

        return {
            "patches_512": patch_512[-1],
            "moe": moe_stats,
            "routing": "global_shared" if variant is AblationVariant.WITHOUT_OEHR else "owner_exclusive",
            "lr_tokens": int(coordinates_lr.shape[0]),
            "hr_tokens": int(coordinates_hr.shape[0]),
            "owner_lr_tokens": int(owner_lr.sum().item()),
            "owner_hr_tokens": int(owner_hr.sum().item()),
            "owner_lr_mean_strength": float(strength_lr.float().mean().item()),
            "resolution": int(resolution),
        }
