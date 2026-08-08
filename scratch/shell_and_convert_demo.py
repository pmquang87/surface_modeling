"""Shell the real inv part with the new hollow-shell function, convert the
wall to STEP through the full pipeline."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\Users\pmqua\PycharmProjects\surface_modeling")

import numpy as np
import trimesh
from src.io.importers import import_stl
from src.io.exporters import export_step
from src.operations.shell_thicken import shell_solid
from src.reverse_engineering.quad_wrap import QuadWrapper
from src.nurbs.converter import SubDToNURBSConverter

STL = r"E:\foxcore_data\_MITEB\Miteb_Flaechenrueckfuehrung\for_claude\7_LLzugdruck_maxstress_smooth_iso0p3_inv.STL"
OUT = r"E:\foxcore_data\_MITEB\Miteb_Flaechenrueckfuehrung\for_claude\7_LLzugdruck_maxstress_smooth_iso0p3_inv_claude_shell3mm.step"

t0 = time.time()
he = import_stl(STL)
orig_tm = he.to_trimesh()
print(f"[{time.time()-t0:6.1f}s] input: {len(he.faces)} tris, volume {orig_tm.volume:.0f} mm3")

wall = shell_solid(he, thickness=3.0, direction='inward', resolution=192)
wall_tm = wall.to_trimesh()
comps = wall_tm.split(only_watertight=False)
print(f"[{time.time()-t0:6.1f}s] wall mesh: {len(wall_tm.faces)} tris, "
      f"volume {wall_tm.volume:.0f} mm3 ({100*wall_tm.volume/orig_tm.volume:.0f}% of solid), "
      f"components {len(comps)}, watertight {wall_tm.is_watertight}")

# Keep the outer surface and only the meaningful void pockets: micro-voids of
# a few mm^3 cannot be represented by a ~3k-quad cage and are irrelevant.
big = [c for c in comps if len(c.faces) >= 300]
dropped_vol = sum(abs(c.volume) for c in comps if len(c.faces) < 300)
wall_tm = trimesh.util.concatenate(big)
print(f"[{time.time()-t0:6.1f}s] kept {len(big)}/{len(comps)} components "
      f"(dropped micro-voids totalling {dropped_vol:.0f} mm3)")
from src.core.halfedge_mesh import HalfEdgeMesh
wall = HalfEdgeMesh.from_trimesh(wall_tm)

cage = QuadWrapper(target_face_count=3200, smoothing_weight=0.5).wrap(wall)
nq = sum(1 for f in cage.faces if len(cage.get_face_vertices(f)) == 4)
bnd = sum(1 for e in cage.edges if cage.is_boundary_edge(e))
print(f"[{time.time()-t0:6.1f}s] cage: {len(cage.faces)} faces ({nq} quads), boundary edges {bnd}")

conv = SubDToNURBSConverter(continuity='G1', tolerance=1e-4)
res = conv.convert(cage, reference_mesh=wall)
shape = res['shape']
print(f"[{time.time()-t0:6.1f}s] patches: {len(res['patches'])}, shape: {'OK' if shape is not None else 'None'}")
assert shape is not None

from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_EDGE
from OCP.TopoDS import TopoDS
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.BRepCheck import BRepCheck_Analyzer

def count(shp, kind):
    n = 0
    e = TopExp_Explorer(shp, kind)
    while e.More():
        n += 1; e.Next()
    return n

shells = []
e = TopExp_Explorer(shape, TopAbs_SHELL)
while e.More():
    shells.append(TopoDS.Shell_s(e.Current())); e.Next()
fb = ShapeAnalysis_FreeBounds(shape)
free = count(fb.GetClosedWires(), TopAbs_EDGE) + count(fb.GetOpenWires(), TopAbs_EDGE)
print(f"[{time.time()-t0:6.1f}s] B-Rep: solids={count(shape, TopAbs_SOLID)} "
      f"shells={len(shells)} closed={[s.Closed() for s in shells]} free-edges={free} "
      f"valid={BRepCheck_Analyzer(shape).IsValid()}")

export_step(shape, OUT)
import os
print(f"[{time.time()-t0:6.1f}s] wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
