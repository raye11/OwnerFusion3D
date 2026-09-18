import unittest

import torch

from ownerfusion3d.methods.moe import build_reliable_head_candidates


class MOETest(unittest.TestCase):
    def test_stable_midlate_readout_uses_fixed_block_heads(self):
        token_count = 5
        records = []
        for block_index in (7, 10, 11):
            selected = torch.zeros((12, token_count))
            background = torch.ones((12, token_count)) * 0.1
            selected[:, :2] = 0.9
            if block_index == 11:
                selected[0, 2] = 0.85
                selected[1, 2] = 0.80
            records.append(
                {
                    "block_index": block_index,
                    "step_index": 0,
                    "selected_mass": selected,
                    "background_mass": background,
                }
            )
        payload = {
            "records": records,
            "selected_token_normalizer": 1.0,
            "background_token_normalizer": 1.0,
        }
        result = build_reliable_head_candidates(
            payload=payload,
            baseline_seed_mask=torch.tensor([True, True, False, False, False]),
            threshold=0.58,
            stable_block_heads=((7, 1), (10, 8), (11, 0), (11, 1)),
            reliability_temperature=0.15,
        )

        self.assertEqual(result["method"], "stable_midlate_reliable_head_fusion")
        self.assertEqual(len(result["members"]), 4)
        self.assertEqual(
            [(m["block_index_zero_based"], m["head_index_zero_based"]) for m in result["members"]],
            [(7, 1), (10, 8), (11, 0), (11, 1)],
        )
        self.assertGreater(float(result["score"][0]), float(result["score"][-1]))


if __name__ == "__main__":
    unittest.main()
