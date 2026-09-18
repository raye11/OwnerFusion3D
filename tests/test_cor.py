import unittest
from types import SimpleNamespace

import torch

from ownerfusion3d.methods.cor import resolve_categorical_ownership


class CORTest(unittest.TestCase):
    def test_overlapping_candidates_resolve_to_one_owner_per_query(self):
        coordinates = torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]], dtype=torch.long
        )
        scores = [torch.tensor([0.9, 0.8, 0.1]), torch.tensor([0.1, 0.7, 0.9])]
        candidates = [torch.tensor([True, True, False]), torch.tensor([False, True, True])]
        config = SimpleNamespace(
            owner_connectivity=6,
            owner_core_quantile=0.9,
            owner_core_min_score=0.55,
            owner_core_min_tokens=1,
            owner_boundary_margin=0.02,
            owner_distance_weight=0.2,
            owner_core_neighbor_weight=0.08,
            owner_assigned_neighbor_weight=0.08,
            owner_fill_unassigned_iters=1,
        )

        owner, stats = resolve_categorical_ownership(
            coords=coordinates,
            score_stack=scores,
            candidate_masks=candidates,
            config=config,
        )

        self.assertEqual(owner[0].item(), 1)
        self.assertEqual(owner[2].item(), 2)
        self.assertTrue(all(value in {0, 1, 2} for value in owner.tolist()))
        self.assertEqual(stats["conflict_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
