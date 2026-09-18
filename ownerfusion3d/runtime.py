"""Runtime helpers shared by all OwnerFusion3D pipelines."""

from __future__ import annotations

import gc
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure_background_removal() -> None:
    """Use an explicit local RMBG checkpoint when requested."""
    local_path = os.environ.get("OWNERFUSION_RMBG_PATH", "").strip()
    if not local_path:
        return
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    from trellis2.pipelines.rembg.BiRefNet import BiRefNet

    if getattr(BiRefNet, "_ownerfusion_patched", False):
        return

    def initialize(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            local_path, trust_remote_code=True, local_files_only=True
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    BiRefNet.__init__ = initialize
    BiRefNet._ownerfusion_patched = True
