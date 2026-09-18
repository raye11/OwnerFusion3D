import json
import unittest

from PIL import Image

from ownerfusion3d.io.parts import PartDefinition, PartManifest, ReferenceGroup


class PartsManifestTest(unittest.TestCase):
    def test_direct_mask_parts_are_supported(self):
        from pathlib import Path

        part = PartDefinition("head", (255, 0, 0), mask_value="head.png")
        group = ReferenceGroup("head_ref", Path("reference.png"), ("head",))
        manifest = PartManifest((part,), (group,))
        self.assertEqual(manifest.parts[0].mask_value, "head.png")

    def test_part_may_not_belong_to_two_groups(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.png").write_bytes(b"x")
            payload = {
                "parts": [{"name": "head", "color": "#ff0000"}],
                "reference_groups": [
                    {"name": "a", "content_image": "a.png", "parts": ["head"]},
                    {"name": "b", "content_image": "a.png", "parts": ["head"]},
                ],
            }
            path = root / "parts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only one reference group"):
                PartManifest.load(path)

    def test_unclaimed_part_is_rejected(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reference.png").write_bytes(b"x")
            payload = {
                "parts": [
                    {"name": "head", "color": "#ff0000"},
                    {"name": "arm", "color": "#00ff00"},
                ],
                "reference_groups": [
                    {"name": "head_ref", "content_image": "reference.png", "parts": ["head"]}
                ],
            }
            path = root / "parts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Every part must belong"):
                PartManifest.load(path)

    def test_direct_masks_share_the_structure_crop(self):
        from ownerfusion3d.io.parts import preprocess_structure_and_masks

        structure = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        for x in range(2, 6):
            for y in range(2, 6):
                structure.putpixel((x, y), (255, 255, 255, 255))
        left = Image.new("L", (8, 8), 0)
        right = Image.new("L", (8, 8), 0)
        left.putpixel((2, 2), 255)
        right.putpixel((3, 3), 255)

        class Pipeline:
            low_vram = False

        processed, masks, foreground = preprocess_structure_and_masks(
            Pipeline(), structure, (left, right)
        )
        self.assertEqual(processed.size, masks[0].size)
        self.assertEqual(processed.size, foreground.size)
        self.assertEqual(len(masks), 2)
        self.assertGreater(masks[0].getbbox()[2], masks[0].getbbox()[0])
        self.assertGreater(masks[1].getbbox()[2], masks[1].getbbox()[0])


if __name__ == "__main__":
    unittest.main()
