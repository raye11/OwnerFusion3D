"""OwnerFusion3D: training-free local image-to-3D fusion on TRELLIS.2."""

from .config import AblationVariant, OwnerFusionConfig

__all__ = ["AblationVariant", "OwnerFusionConfig", "OwnerFusionPipeline"]
__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "OwnerFusionPipeline":
        from .pipeline import OwnerFusionPipeline

        return OwnerFusionPipeline
    raise AttributeError(name)
