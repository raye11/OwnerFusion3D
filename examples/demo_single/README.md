# Single-mask demo

This demo uses one structure image, one aligned binary mask, and one content
reference. Run it from the repository root after installing the external
TRELLIS.2 runtime and this package:

```bash
python -m ownerfusion3d \
  --structure examples/demo_single/structure.png \
  --mask examples/demo_single/mask.png \
  --content examples/demo_single/content.png \
  --output-dir outputs/demo_single \
  --model microsoft/TRELLIS.2-4B \
  --seed 2026
```

The command writes the generated `mesh.glb`, metadata, and optional ownership
diagnostics to `outputs/demo_single/`.
