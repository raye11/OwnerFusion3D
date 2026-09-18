import unittest
from types import SimpleNamespace

import torch
from ownerfusion3d.methods.oehr import (
    RegionalSparseCrossAttention,
    build_hr_mask_from_lr_mask,
    patched_slat_flow_for_regional_attention,
)


class OEHRTest(unittest.TestCase):
    def test_exclusive_gate_blocks_incompatible_sources(self):
        scores = torch.zeros((1, 2, 4))
        region = torch.tensor([True, False])
        result = RegionalSparseCrossAttention._apply_regional_bias(
            scores,
            region,
            torch.tensor([0, 1]),
            torch.tensor([2, 3]),
        )
        self.assertTrue(torch.all(result[0, 0, 2:] == -10000))
        self.assertTrue(torch.all(result[0, 1, :2] == -10000))
        self.assertTrue(torch.all(result[0, 0, :2] == 0))
        self.assertTrue(torch.all(result[0, 1, 2:] == 0))

    def test_owner_is_lifted_by_parent_coordinate(self):
        lr = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]])
        owner = torch.tensor([True, False])
        hr = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]])
        lifted = build_hr_mask_from_lr_mask(lr, owner, hr, 512, 1024)
        self.assertEqual(lifted.tolist(), [True, True, False, False])

    def test_categorical_gate_exposes_only_the_owned_source(self):
        scores = torch.zeros((1, 3, 6))
        owner = torch.tensor([0, 1, 2])
        key_groups = torch.tensor([1, 1, 2, 2, 0, 0])
        result = RegionalSparseCrossAttention._apply_group_owner_bias(
            scores,
            owner,
            key_groups,
            "exclusive",
        )
        self.assertTrue(torch.all(result[0, 0, :4] == -10000))
        self.assertTrue(torch.all(result[0, 0, 4:] == 0))
        self.assertTrue(torch.all(result[0, 1, :2] == 0))
        self.assertTrue(torch.all(result[0, 1, 2:] == -10000))
        self.assertTrue(torch.all(result[0, 2, 2:4] == 0))

    def test_attention_patch_is_a_context_manager(self):
        context = patched_slat_flow_for_regional_attention(SimpleNamespace())
        self.assertTrue(hasattr(context, "__enter__"))
        self.assertTrue(hasattr(context, "__exit__"))


if __name__ == "__main__":
    unittest.main()
