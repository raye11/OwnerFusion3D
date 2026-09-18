import unittest

from ownerfusion3d.config import AblationVariant, MultiPartConfig, OEHRConfig, OwnerFusionConfig
from ownerfusion3d.cli.common import config_from_args
from ownerfusion3d.cli.fuse import build_parser as fuse_parser
from ownerfusion3d.pipeline import _module_state


class ConfigTest(unittest.TestCase):
    def test_paper_module_states(self):
        self.assertEqual(_module_state(AblationVariant.FULL), {"MOE": True, "OEHR": True, "COR": False})
        self.assertEqual(_module_state(AblationVariant.WITHOUT_MOE), {"MOE": False, "OEHR": False, "COR": False})
        self.assertEqual(_module_state(AblationVariant.WITHOUT_OEHR), {"MOE": True, "OEHR": False, "COR": False})

    def test_default_configuration_is_serializable(self):
        payload = OwnerFusionConfig().to_dict()
        self.assertEqual(payload["seed"], 2026)
        self.assertEqual(payload["moe"]["score_mode"], "ratio")
        self.assertEqual(payload["moe"]["readout"], "stable_midlate_4")
        self.assertEqual(payload["moe"]["threshold"], 0.58)
        self.assertEqual(payload["moe"]["max_selected_ratio"], 1.0)
        self.assertEqual(payload["oehr"]["attention_chunk_size"], 256)

    def test_attention_chunk_size_is_shared_by_single_and_multi_routes(self):
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
        config = config_from_args(args)
        self.assertEqual(config.oehr.attention_chunk_size, 1024)
        self.assertEqual(config.multipart.attention_chunk_size, 1024)


if __name__ == "__main__":
    unittest.main()
