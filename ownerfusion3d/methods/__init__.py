"""OwnerFusion3D method modules.

Public method surface:
- MOE: mask-conditioned ownership evidence
- OEHR: owner-exclusive hierarchical routing
- COR: categorical ownership resolution for multi-part fusion
"""

from .moe import build_patch_token_sets, patch_coverage_from_mask_image
from .oehr import AttentionProbeState

__all__ = [
    "AttentionProbeState",
    "build_patch_token_sets",
    "patch_coverage_from_mask_image",
]
