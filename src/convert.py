"""Command-line STL -> STEP reverse-engineering conversion.

Converts a dense triangle mesh (e.g. a TOSCA/topology-optimization STL) into
a STEP solid made of smooth B-spline patches:

    STL -> quad cage (QuadWrapper) -> NURBS patches fitted to the original
    surface (SubDToNURBSConverter) -> sewn, solidified, written as STEP.

Usage (from the repository root):

    python -m src.convert input.stl output.step
    python -m src.convert input.stl output.step --target-faces 2200 --continuity G1
    python -m src.convert input.stl output.step --faceted      # planar faces, no fitting

Requires cadquery-ocp (OCP) for the STEP export. The written file is
re-read and audited by default (src/io/step_audit.py): shell closure, free
edges, BRepCheck, signed volume vs the mesh, tessellated bbox vs the STL
bbox, sampled deviation, and an OCP-free text census of the file.

Output contract (pattern from BlinkingSun/stl2step, MIT):

* Human progress goes to **stderr** (``--quiet`` silences it).
* The **only** stdout content is the last line ``RESULT {json}`` - the same
  numbers the audit prints, machine-readable, on success and on failure
  (``{"ok": false, "error": ...}``).
* Exit codes: ``0`` clean; ``2`` a STEP was written but the audit found
  something (see ``audit_reasons``); ``1`` failed, nothing usable written
  (usage errors included).
"""
import argparse
import contextlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if __package__ is None or __package__ == "":  # running as a plain script
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class _Progress:
    """Timestamped progress lines on stderr, suppressible."""

    def __init__(self, quiet: bool):
        self.quiet = quiet
        self.t0 = time.time()
        self.warnings: List[str] = []

    def __call__(self, msg: str) -> None:
        if not self.quiet:
            print(f"[{time.time() - self.t0:6.1f}s] {msg}", file=sys.stderr, flush=True)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self("WARNING: " + msg)


def _library_output_to_stderr(quiet: bool):
    """Library modules still ``print()`` progress; keep stdout for RESULT."""
    if quiet:
        return contextlib.redirect_stdout(open(os.devnull, "w"))
    return contextlib.redirect_stdout(sys.stderr)


def convert(input_path: str, output_path: str, target_faces: int = 2000,
            continuity: str = 'G1', smoothing: float = 0.5,
            fit_reference: bool = True, verify: bool = True,
            volume_tol_pct: Optional[float] = None, quiet: bool = False,
            faceted: bool = False, schema: str = "AP214IS",
            ) -> Tuple[int, Dict[str, Any]]:
    """Run the conversion. Returns ``(exit_code, result_dict)``; never raises
    for a bad input - failures come back as ``ok: False`` with ``error``."""
    log = _Progress(quiet)
    result: Dict[str, Any] = {
        "ok": False, "exit_code": 1,
        "input": os.path.abspath(input_path), "output": os.path.abspath(output_path),
        "mode": "faceted" if faceted else "nurbs",
        "warnings": log.warnings,
    }

    def fail(msg: str) -> Tuple[int, Dict[str, Any]]:
        log("ERROR: " + msg)
        result["error"] = msg
        result["exit_code"] = 1
        result["seconds"] = round(time.time() - log.t0, 3)
        return 1, result

    try:
        with _library_output_to_stderr(quiet):
            from src.io.occt_utils import quiet_occt
            quiet_occt()
            from src.io.importers import import_stl
            from src.io.exporters import export_step

            log(f"loading {input_path}")
            try:
                dense = import_stl(input_path)
            except (FileNotFoundError, ValueError) as exc:
                return fail(str(exc))
            result["triangles"] = len(dense.faces)
            result["vertices"] = len(dense.vertices)
            log(f"loaded: {len(dense.vertices)} vertices, {len(dense.faces)} triangles")

            if faceted:
                from src.io.faceted_step import export_step_faceted
                log("faceted export: one planar face per triangle, coplanar merge")
                report = export_step_faceted(dense.to_trimesh(), output_path,
                                             schema=schema, verify=False)
                result["faceted"] = report
                for w in report.get("warnings", []):
                    log.warn(w)
                expected_faces = report.get("faces")
                if not report.get("ok", False):
                    return fail(report.get("error", "faceted export failed"))
                log(f"wrote {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")
            else:
                from src.reverse_engineering.quad_wrap import QuadWrapper
                from src.nurbs.converter import SubDToNURBSConverter

                wrapper = QuadWrapper(target_face_count=target_faces,
                                      smoothing_weight=smoothing)
                cage = wrapper.wrap(dense)
                n_quads = sum(1 for f in cage.faces if len(cage.get_face_vertices(f)) == 4)
                n_boundary = sum(1 for e in cage.edges if cage.is_boundary_edge(e))
                result.update({"cage_vertices": len(cage.vertices),
                               "cage_faces": len(cage.faces), "cage_quads": n_quads,
                               "cage_boundary_edges": n_boundary})
                log(f"cage: {len(cage.vertices)} vertices, {len(cage.faces)} faces "
                    f"({n_quads} quads), boundary edges: {n_boundary}")
                if len(cage.faces) == 0 or n_quads == 0:
                    return fail("quad wrap produced no quads")
                if n_boundary:
                    log.warn(f"cage has {n_boundary} boundary edges; the STEP cannot close")

                converter = SubDToNURBSConverter(continuity=continuity, tolerance=1e-4)
                conv = converter.convert(cage, reference_mesh=dense if fit_reference else None)
                shape = conv['shape']
                result["patches"] = len(conv['patches'])
                expected_faces = len(conv['patches'])
                log(f"patches: {len(conv['patches'])}, shape: {'OK' if shape is not None else 'None'}")
                if shape is None:
                    return fail("NURBS conversion produced no shape (is cadquery-ocp installed?)")

                export_step(shape, output_path, schema=schema)
                log(f"wrote {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")

            result["ok"] = True
            result["exit_code"] = 0
            if verify:
                from src.io.step_audit import measure_step, verdict, format_measurement
                m = measure_step(output_path, reference_stl=input_path,
                                 expected_faces=expected_faces)
                # the census dict is verbose; keep the summary numbers only
                census = m.pop("census", None)
                if isinstance(census, dict) and "surfaces" in census:
                    m["census_surfaces"] = census["surfaces"]
                result["audit"] = m
                log("audit: " + format_measurement(m))
                if m.get("dev_mean_mm") is not None:
                    log(f"deviation STEP->STL: mean={m['dev_mean_mm']:.3f} mm, "
                        f"p95={m['dev_p95_mm']:.3f} mm, max={m['dev_max_mm']:.3f} mm")
                code, reasons = verdict(m, volume_tol_pct=volume_tol_pct)
                result["audit_reasons"] = reasons
                if code == 1:
                    return fail("; ".join(reasons))
                if code == 2:
                    log("AUDIT WARNING: " + "; ".join(reasons) +
                        (" - try a higher --target-faces" if not faceted else ""))
                    result["exit_code"] = 2
    except Exception as exc:  # noqa: BLE001 - the contract is RESULT, not a traceback
        return fail(f"{type(exc).__name__}: {exc}")

    result["seconds"] = round(time.time() - log.t0, 3)
    return result["exit_code"], result


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on usage errors, which collides with our 'written
    with warnings' tier; a usage error means nothing was written -> 1."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        sys.exit(1)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = _Parser(
        prog="python -m src.convert",
        description="Convert a dense STL into a STEP solid of smooth "
                    "B-spline patches (reverse engineering), or into a "
                    "faceted planar-face solid with --faceted.",
        epilog="stdout: exactly one line 'RESULT {json}'. exit 0 clean, "
               "2 written with audit warnings, 1 failed.")
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
    parser.add_argument("--faceted", action="store_true",
                        help="skip retopology and fitting: write the mesh as a "
                             "faceted B-Rep solid (one planar face per triangle, "
                             "coplanar faces merged). Exact geometry, large file.")
    parser.add_argument("--schema", choices=["AP203", "AP214IS", "AP242DIS"],
                        default="AP214IS", help="STEP application protocol")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the read-back audit of the written STEP")
    parser.add_argument("--volume-tol", type=float, default=None, metavar="PCT",
                        help="fail (exit 2) when |V_step - V_mesh| / V_mesh exceeds "
                             "PCT percent. Default: report only. A smooth fit "
                             "differs by ~0.1%% on good parts; --faceted holds "
                             "1e-4 (the coplanar merge shifts the integrator by "
                             "up to 1e-6 relative on identical geometry).")
    parser.add_argument("--quiet", action="store_true",
                        help="no progress on stderr; stdout still gets RESULT")
    args = parser.parse_args(argv)

    code, result = convert(
        args.input, args.output,
        target_faces=args.target_faces,
        continuity=args.continuity,
        smoothing=args.smoothing,
        fit_reference=not args.no_reference_fit,
        verify=not args.no_verify,
        volume_tol_pct=args.volume_tol,
        quiet=args.quiet,
        faceted=args.faceted,
        schema=args.schema,
    )
    print("RESULT " + json.dumps(result, default=str), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
