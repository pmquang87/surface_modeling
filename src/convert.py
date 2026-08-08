"""Command-line STL -> STEP reverse-engineering conversion.

Converts a dense triangle mesh (e.g. a TOSCA/topology-optimization STL) into
a STEP solid made of smooth B-spline patches:

    STL -> quad cage (QuadWrapper) -> NURBS patches fitted to the original
    surface (SubDToNURBSConverter) -> sewn, solidified, written as STEP.

Usage (from the repository root):

    python -m src.convert input.stl output.step
    python -m src.convert input.stl output.step --target-faces 2200 --continuity G1

Requires cadquery-ocp (OCP) for the STEP export. The written file is
re-read and audited by default: shell closure, free edges, BRepCheck and
sampled deviation against the input mesh.
"""
import argparse
import os
import sys
import time

if __package__ is None or __package__ == "":  # running as a plain script
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np


def convert(input_path: str, output_path: str, target_faces: int = 2000,
            continuity: str = 'G1', smoothing: float = 0.5,
            fit_reference: bool = True, verify: bool = True) -> int:
    from src.io.importers import import_stl
    from src.io.exporters import export_step
    from src.reverse_engineering.quad_wrap import QuadWrapper
    from src.nurbs.converter import SubDToNURBSConverter

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] loading {input_path}")
    dense = import_stl(input_path)
    print(f"[{time.time()-t0:6.1f}s] loaded: {len(dense.vertices)} vertices, "
          f"{len(dense.faces)} triangles")

    wrapper = QuadWrapper(target_face_count=target_faces, smoothing_weight=smoothing)
    cage = wrapper.wrap(dense)
    n_quads = sum(1 for f in cage.faces if len(cage.get_face_vertices(f)) == 4)
    n_boundary = sum(1 for e in cage.edges if cage.is_boundary_edge(e))
    print(f"[{time.time()-t0:6.1f}s] cage: {len(cage.vertices)} vertices, "
          f"{len(cage.faces)} faces ({n_quads} quads), boundary edges: {n_boundary}")
    if len(cage.faces) == 0 or n_quads == 0:
        print("ERROR: quad wrap produced no quads; aborting")
        return 1

    converter = SubDToNURBSConverter(continuity=continuity, tolerance=1e-4)
    result = converter.convert(cage, reference_mesh=dense if fit_reference else None)
    shape = result['shape']
    print(f"[{time.time()-t0:6.1f}s] patches: {len(result['patches'])}, "
          f"shape: {'OK' if shape is not None else 'None'}")
    if shape is None:
        print("ERROR: NURBS conversion produced no shape (is cadquery-ocp installed?)")
        return 1

    export_step(shape, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[{time.time()-t0:6.1f}s] wrote {output_path} ({size_mb:.1f} MB)")

    if verify:
        return _audit(output_path, input_path, t0)
    return 0


def _audit(step_path: str, stl_path: str, t0: float) -> int:
    """Independent read-back check of the written STEP file."""
    import trimesh
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_EDGE, TopAbs_SOLID
    from OCP.TopoDS import TopoDS
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location

    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        print("AUDIT FAILED: written STEP cannot be re-read")
        return 1
    reader.TransferRoots()
    shape = reader.OneShape()

    def count(kind):
        n = 0
        exp = TopExp_Explorer(shape, kind)
        while exp.More():
            n += 1
            exp.Next()
        return n

    shells = []
    exp = TopExp_Explorer(shape, TopAbs_SHELL)
    while exp.More():
        shells.append(TopoDS.Shell_s(exp.Current()))
        exp.Next()
    closed = [sh.Closed() for sh in shells]
    fb = ShapeAnalysis_FreeBounds(shape)

    def edge_count(compound):
        n = 0
        exp2 = TopExp_Explorer(compound, TopAbs_EDGE)
        while exp2.More():
            n += 1
            exp2.Next()
        return n

    free_edges = edge_count(fb.GetClosedWires()) + edge_count(fb.GetOpenWires())
    valid = BRepCheck_Analyzer(shape).IsValid()
    print(f"[{time.time()-t0:6.1f}s] audit: faces={count(TopAbs_FACE)} "
          f"solids={count(TopAbs_SOLID)} shells={len(shells)} closed={closed} "
          f"free-edges={free_edges} BRepCheck-valid={valid}")

    # sampled deviation of the STEP surface against the input mesh
    BRepMesh_IncrementalMesh(shape, 0.2)
    points = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face, loc)
        if poly:
            for i in range(1, poly.NbNodes() + 1):
                node = poly.Node(i)
                if not loc.IsIdentity():
                    node.Transform(loc.Transformation())
                points.append([node.X(), node.Y(), node.Z()])
        exp.Next()
    points = np.array(points)
    if len(points):
        ref = trimesh.load_mesh(stl_path)
        rng = np.random.default_rng(0)
        sample = points[rng.choice(len(points), min(5000, len(points)), replace=False)]
        _, dist, _ = trimesh.proximity.closest_point(ref, sample)
        print(f"[{time.time()-t0:6.1f}s] deviation STEP->STL: mean={dist.mean():.3f} mm, "
              f"p95={np.percentile(dist, 95):.3f} mm, max={dist.max():.3f} mm")

    ok = valid and free_edges == 0 and all(closed)
    if not ok:
        print("AUDIT WARNING: result is not a fully closed valid solid "
              "(see numbers above) — try a higher --target-faces")
    return 0 if ok else 2


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="python -m src.convert",
        description="Convert a dense STL into a STEP solid of smooth "
                    "B-spline patches (reverse engineering).")
    parser.add_argument("input", help="input STL file")
    parser.add_argument("output", help="output STEP file")
    parser.add_argument("--target-faces", type=int, default=2000,
                        help="approximate quad count of the control cage "
                             "(default 2000; more = finer, slower)")
    parser.add_argument("--continuity", choices=["G0", "G1", "G2", "G3"],
                        default="G1",
                        help="cross-patch smoothness weight; G0 = best fidelity, "
                             "G3 = smoothest (default G1)")
    parser.add_argument("--smoothing", type=float, default=0.5,
                        help="cage relaxation weight 0..1 (default 0.5)")
    parser.add_argument("--no-reference-fit", action="store_true",
                        help="fit the Catmull-Clark limit surface of the cage "
                             "instead of the input mesh surface")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the read-back audit of the written STEP")
    args = parser.parse_args(argv)

    return convert(
        args.input, args.output,
        target_faces=args.target_faces,
        continuity=args.continuity,
        smoothing=args.smoothing,
        fit_reference=not args.no_reference_fit,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    sys.exit(main())
