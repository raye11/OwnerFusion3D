import unittest

from ownerfusion3d.cli.ablate import build_parser as ablation_parser
from ownerfusion3d.cli.fuse import build_parser as fuse_parser
from ownerfusion3d.cli.fuse_parts import build_parser as parts_parser
from ownerfusion3d.cli.run import build_parser as unified_parser


class CLITest(unittest.TestCase):
    def test_single_region_cli_contract(self):
        args = fuse_parser().parse_args(
            ["--structure", "s.png", "--mask", "m.png", "--content", "c.png", "--output-dir", "out"]
        )
        self.assertEqual(args.seed, 2026)
        self.assertEqual(args.resolution, 1024)
        self.assertEqual(args.moe_probe_steps, 4)
        self.assertEqual(args.moe_readout, "stable_midlate_4")
        self.assertEqual(args.attention_chunk_size, 256)

    def test_probe_step_override(self):
        args = fuse_parser().parse_args(
            [
                "--structure",
                "s.png",
                "--mask",
                "m.png",
                "--content",
                "c.png",
                "--output-dir",
                "out",
                "--moe-probe-steps",
                "8",
            ]
        )
        self.assertEqual(args.moe_probe_steps, 8)

    def test_attention_chunk_size_override(self):
        args = fuse_parser().parse_args(
            [
                "--structure",
                "s.png",
                "--mask",
                "m.png",
                "--content",
                "c.png",
                "--output-dir",
                "out",
                "--attention-chunk-size",
                "1024",
            ]
        )
        self.assertEqual(args.attention_chunk_size, 1024)

    def test_all_head_readout_override(self):
        args = fuse_parser().parse_args(
            [
                "--structure",
                "s.png",
                "--mask",
                "m.png",
                "--content",
                "c.png",
                "--output-dir",
                "out",
                "--moe-readout",
                "all_head",
            ]
        )
        self.assertEqual(args.moe_readout, "all_head")

    def test_multi_part_cli_contract(self):
        args = parts_parser().parse_args(
            ["--structure", "s.png", "--part-labels", "p.png", "--parts", "parts.json", "--output-dir", "out"]
        )
        self.assertEqual(args.parts, "parts.json")

    def test_unified_cli_accepts_one_or_many_masks(self):
        args = unified_parser().parse_args(
            [
                "--structure", "s.png",
                "--mask", "left.png", "right.png",
                "--content", "left_ref.png", "right_ref.png",
                "--output-dir", "out",
            ]
        )
        self.assertEqual(args.mask, ["left.png", "right.png"])
        self.assertEqual(args.content, ["left_ref.png", "right_ref.png"])

    def test_ablation_names_are_paper_names(self):
        args = ablation_parser().parse_args(
            ["--structure", "s.png", "--mask", "m.png", "--content", "c.png", "--output-dir", "out"]
        )
        self.assertEqual(args.variants, ["full", "without_moe", "without_oehr"])


if __name__ == "__main__":
    unittest.main()
