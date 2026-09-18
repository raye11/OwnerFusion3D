"""Versioned public configuration for OwnerFusion3D.

The release configuration intentionally exposes only the paper method that is
kept after the final ablation cycle: MOE direct ownership readout followed by
OEHR source-exclusive routing. Historical EVO completion operators are not part
of the public pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AblationVariant(str, Enum):
    """Paper-level counterfactuals used by the released implementation."""

    FULL = "full"
    WITHOUT_MOE = "without_moe"
    WITHOUT_OEHR = "without_oehr"


@dataclass(frozen=True)
class MOEConfig:
    """Mask-conditioned ownership evidence defaults.

    These values match the final stable setting used for the paper runs:
    ratio evidence, four LR probing steps, fixed tau=0.58, no selected-token
    cap, and the stable mid-late block-head readout.
    """

    probe_steps_lr: int = 4
    score_mode: str = "ratio"
    threshold: float = 0.58
    readout: str = "stable_midlate_4"
    readout_temperature: float = 0.15
    stable_block_heads: tuple[tuple[int, int], ...] = ((7, 1), (10, 8), (11, 0), (11, 1))
    majority_neighbors: int = 16
    majority_vote_ratio: float = 0.60
    max_selected_ratio: float = 1.0
    expand_iterations: int = 1
    expand_score_delta: float = 0.05
    expand_connectivity: int = 6
    disable_expand_on_overselect: bool = True
    patch_valid_coverage: float = 0.05
    patch_selected_coverage: float = 0.15
    patch_background_coverage: float = 0.02
    isolate_probe_rng: bool = True


@dataclass(frozen=True)
class OEHRConfig:
    """Owner-exclusive hierarchical routing defaults."""

    attention_chunk_size: int = 256
    shape_steps: int = 12
    texture_steps: int = 12


@dataclass(frozen=True)
class CORConfig:
    """Categorical ownership resolution for the multi-part extension."""

    mode: str = "full"
    connectivity: int = 6
    core_quantile: float = 0.90
    core_min_score: float = 0.55
    core_min_tokens: int = 16
    boundary_margin: float = 0.06
    distance_weight: float = 0.20
    core_neighbor_weight: float = 0.08
    assigned_neighbor_weight: float = 0.08
    fill_unassigned_iterations: int = 1
    # The final COR release uses conservative spatial resolution followed by
    # bounded completion. Repair is disabled so it cannot silently change the
    # ablation contract.
    repair_iterations: int = 0
    repair_connectivity: int = 18
    repair_min_neighbor_votes: int = 2
    repair_min_neighbor_ratio: float = 0.45
    repair_decisive_margin: float = 0.12
    repair_candidate_bonus: float = 0.05
    local_residual_fill: bool = True
    local_residual_connectivity: int = 26
    local_residual_min_neighbor_votes: int = 2
    local_residual_min_neighbor_ratio: float = 0.55
    local_residual_min_score: float = 0.30
    local_residual_max_added_ratio: float = 0.20
    local_residual_max_added_tokens: int = 512
    boundary_completion: bool = True
    boundary_connectivity: int = 26
    boundary_min_neighbor_votes: int = 1
    boundary_min_neighbor_ratio: float = 0.40
    boundary_min_score: float = 0.25
    boundary_min_score_margin: float = 0.0
    boundary_max_added_ratio: float = 0.30
    boundary_max_added_tokens: int = 512
    hole_completion: bool = True
    hole_connectivity: int = 26
    hole_min_neighbor_votes: int = 2
    hole_min_neighbor_ratio: float = 0.55
    hole_max_competing_ratio: float = 0.45
    hole_min_score: float = 0.10
    hole_max_added_ratio: float = 0.15
    hole_max_added_tokens: int = 512
    # The reported COR owner is fully completed, while OEHR may route from a
    # conservative snapshot to preserve narrow structural gaps.
    routing_owner_stage: str = "after_local"


@dataclass(frozen=True)
class MultiPartConfig:
    """Defaults for COR-based synchronized multi-reference fusion."""

    attention_chunk_size: int = 256


@dataclass(frozen=True)
class OwnerFusionConfig:
    """Stable public configuration for OwnerFusion3D."""

    model: str = "microsoft/TRELLIS.2-4B"
    seed: int = 2026
    sparse_resolution: int = 32
    output_resolution: int = 1024
    max_num_tokens: int = 49152
    texture_size: int = 4096
    save_ownership: bool = True
    moe: MOEConfig = field(default_factory=MOEConfig)
    oehr: OEHRConfig = field(default_factory=OEHRConfig)
    cor: CORConfig = field(default_factory=CORConfig)
    multipart: MultiPartConfig = field(default_factory=MultiPartConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
