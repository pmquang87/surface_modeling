# Example drawings produced by `python -m src.drawing`

Rendered pages (PNG at 110 dpi) of the PDF output on public test models, produced
with the default options (A3, first angle, `front,top,right,iso`, section A-A,
sandboxed). The source models are not part of the repository; the licences and
origins are listed so the images can be reused.

| Image | Source model | Route | Licence of the model |
|---|---|---|---|
| `nist_ctc01_step.png` | NIST MBE PMI test case CTC-01 (`nist_ctc_01_asme1_rd.stp`, AP203 geometry only), machined plate with hex boss, slots, through and counterbored holes | STEP → exact HLR | Public domain (US Government, NIST) |
| `nist_ftc06_inch_step.png` | NIST FTC-06 (`nist_ftc_06_asme1_rd.stp`), inch-unit block with bosses, knobs, cones and holes | STEP → exact HLR, inch converted to mm and flagged | Public domain (NIST) |
| `featuretype_step.png` | `featuretype.STEP` from the trimesh test models (SolidWorks 2013 export, 46 cylindrical faces) | STEP → exact HLR | MIT (mikedh/trimesh) |
| `featuretype_stl.png` | `featuretype.STL`, the same part as a binary mesh (3 476 triangles) | STL → classifier → faceted STEP → polyhedral HLR | MIT (mikedh/trimesh) |
| `plate_holes_stl.png` | `plate_holes.STL` (trimesh test models; binary STL whose header starts with `solid`) | STL → faceted STEP → polyhedral HLR | MIT (mikedh/trimesh) |
| `topology_optimised_bracket_stl.png` | `air_brack_TO.stl` from FreeTO, a topology-optimised aircraft bracket (300 920 triangles, inside-out winding) | STL → organic → mesh depth-buffer route | MIT (ooibhadode/FreeTO) |

How to reproduce: download the model from its source, then

```bash
python -m src.drawing nist_ctc_01_asme1_rd.stp out/
```

and render `out/<stem>.pdf` with any PDF viewer. The quality report
`out/<stem>.report.json` lists the route, every dimension placed or dropped and
every flag; see `docs/drawing_module.md`.
