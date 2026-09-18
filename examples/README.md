# Examples and input schemas

The JSON files in this directory define the batch and multi-part input
schemas. The two `demo_*` directories additionally provide compact input
examples that can be run after installing the external TRELLIS.2 runtime.

For a single-mask run, provide one structure image, one binary mask in the
same camera view, and one content reference image.

For a multi-part run, `parts_manifest.json` binds each named part to a binary
mask or to an RGB color in a categorical label image, and binds one or more
parts to each reference image. A part belongs to exactly one reference group.
Paths in the manifest are resolved relative to the manifest file.

See `demo_single/README.md` for the single-mask example and
`demo_multipart/README.md` for the multi-part, multi-reference example.
