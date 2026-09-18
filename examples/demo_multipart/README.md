# Multi-part, multi-reference demo

This demo uses one structure image, one RGB categorical part-label image, and
four content references. The four colors in `part_labels.png` correspond to
`part_01` through `part_04` in `parts_manifest.json`. Run it from the
repository root after installing the external TRELLIS.2 runtime and this
package:

```bash
python -m ownerfusion3d.cli.fuse_parts \
  --structure examples/demo_multipart/structure.png \
  --part-labels examples/demo_multipart/part_labels.png \
  --parts examples/demo_multipart/parts_manifest.json \
  --output-dir outputs/demo_multipart \
  --model microsoft/TRELLIS.2-4B \
  --seed 2026
```

The command resolves the four sparse ownership fields with COR and routes the
reference groups through OEHR. It writes `mesh.glb`, metadata, and optional
ownership diagnostics to `outputs/demo_multipart/`.
