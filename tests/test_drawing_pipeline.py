"""Report, routing and CLI tests of the drawing module (task C1).

* ``Report``: exit code from the flag severities, quality level from the
  route / hidden lines / section / dimensions, JSON write.
* ``pipeline.run`` in-process on the probe plate (100 x 60 x 15, four Ø6.5
  through holes, one Ø20 hole, a Ø30 x 10 boss): STEP -> exact HLR; the
  plate tessellation as STL -> faceted STEP -> poly HLR; an organic blob ->
  mesh route; forced mesh route; a monkeypatched B-rep failure -> mesh
  fallback; bad inputs -> exit 1 without killing the test process; an inch
  STEP -> ``unit_not_mm``.
* ``python -m src.drawing`` as a subprocess: ``--no-sandbox`` and the
  sandboxed default (one RESULT line, exit codes, sandbox section).

Every defect test asserts its precondition first (repo rule).
"""
import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")


# ---------------------------------------------------------------- fixtures --

def _plate():
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 100.0, 60.0, 15.0).Shape()
    boss = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(78, 30, 15), gp_Dir(0, 0, 1)),
                                    15.0, 10.0).Shape()
    shape = BRepAlgoAPI_Fuse(shape, boss).Shape()
    for (x, y), r in [((8, 8), 3.25), ((92, 8), 3.25), ((8, 52), 3.25),
                      ((92, 52), 3.25), ((50, 30), 10.0)]:
        tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, -5), gp_Dir(0, 0, 1)),
                                        r, 40.0).Shape()
        shape = BRepAlgoAPI_Cut(shape, tool).Shape()
    return shape


def _tessellate(shape, deflection=0.05):
    """OCCT shape -> welded trimesh (outward winding)."""
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(shape, deflection, False, 0.2, True)
    verts, faces = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            off = len(verts)
            trsf = loc.Transformation()
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                verts.append((p.X(), p.Y(), p.Z()))
            rev = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                faces.append((off + a - 1, off + c - 1, off + b - 1) if rev
                             else (off + a - 1, off + b - 1, off + c - 1))
        exp.Next()
    mesh = trimesh.Trimesh(np.array(verts), np.array(faces), process=True)
    mesh.merge_vertices()
    return mesh


@pytest.fixture(scope="module")
def plate_files(tmp_path_factory):
    if not HAS_OCP:
        pytest.skip("cadquery-ocp not installed")
    from src.io.exporters import export_step
    d = tmp_path_factory.mktemp("plate_in")
    shape = _plate()
    step = str(d / "plate.step")
    export_step(shape, step)
    mesh = _tessellate(shape)
    assert mesh.is_watertight                      # precondition of the faceted route
    stl = str(d / "plate.stl")
    mesh.export(stl)
    return {"step": step, "stl": stl, "triangles": int(len(mesh.faces))}


def _blob_stl(path):
    import trimesh
    rng = np.random.default_rng(0)
    blob = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
    blob.vertices = blob.vertices + rng.normal(scale=0.4, size=blob.vertices.shape)
    trimesh.smoothing.filter_taubin(blob, iterations=10)
    blob.export(path)
    return path


def _opts(**kw):
    from src.drawing.options import DrawingOptions
    return DrawingOptions(**kw)


def _codes(rep):
    return {f["code"]: f["severity"] for f in rep["quality"]["flags"]}


# ------------------------------------------------------------------ report --

def _filled_report(tmp_path, pipeline="brep", algorithm="hlr_exact", loops=2, linear=True,
                   **opt_kw):
    from src.drawing.report import Report
    rep = Report(str(tmp_path / "part.step"), str(tmp_path), _opts(**opt_kw))
    rep.data["route"]["pipeline"] = pipeline
    rep.data["views"] = [{"name": "front", "algorithm": algorithm, "visible": 4, "hidden": 2,
                          "seconds": 0.01}]
    rep.data["section"] = {"label": "A", "loops": loops} if loops is not None else None
    if linear:
        rep.data["dimensions"] = [{"kind": "linear", "view": "front", "value_mm": 60.0,
                                   "text": "60", "count": 1}]
    return rep


def test_report_schema_prefilled(tmp_path):
    from src.drawing.report import SCHEMA, Report
    rep = Report(str(tmp_path / "part.stl"), str(tmp_path), _opts())
    for key in ("schema", "ok", "exit_code", "error", "input", "route", "geometry", "views",
                "dimensions", "undimensioned", "section", "sheet", "outputs", "quality",
                "warnings", "timing", "sandbox", "version"):
        assert key in rep.data, key
    assert rep.data["schema"] == SCHEMA
    assert rep.data["sandbox"]["enabled"] is False
    assert rep.data["route"]["stl_to_step"]["mode"] == "auto"
    assert rep.data["version"]["module"] == "0.1.0"
    assert rep.stem == "part"


def test_report_full_level_exit_0(tmp_path):
    rep = _filled_report(tmp_path)
    code, data = rep.finish(ok=True)
    assert code == 0 and data["exit_code"] == 0 and data["ok"] is True
    assert data["quality"]["level"] == "full"
    assert data["error"] is None
    assert data["timing"]["total_s"] >= 0.0


def test_report_warning_flag_gives_exit_2_info_does_not(tmp_path):
    rep = _filled_report(tmp_path)
    rep.flag("approval_blank", "info", "blank")
    assert rep.finish(ok=True)[0] == 0              # control: info flags keep exit 0
    rep2 = _filled_report(tmp_path)
    rep2.flag("section_empty", "warning", "no loops")
    code, data = rep2.finish(ok=True)
    assert code == 2 and data["exit_code"] == 2 and data["ok"] is True
    assert data["quality"]["flags"][0]["code"] == "section_empty"
    with pytest.raises(ValueError):
        rep2.flag("x", "fatal")


def test_report_failure_exit_1(tmp_path):
    rep = _filled_report(tmp_path)
    rep.flag("section_empty", "warning")
    code, data = rep.finish(ok=False, error="x")
    assert code == 1 and data["exit_code"] == 1
    assert data["ok"] is False and data["error"] == "x"
    assert data["quality"]["level"] == "none"


def test_report_levels(tmp_path):
    # precondition: the same data on the B-rep route is "full", so each change below is the cause
    assert _filled_report(tmp_path).finish(True)[1]["quality"]["level"] == "full"
    assert _filled_report(tmp_path, pipeline="mesh",
                          algorithm="depth_buffer").finish(True)[1]["quality"]["level"] == "reduced"
    assert _filled_report(tmp_path, algorithm="hlr_poly").finish(True)[1]["quality"]["level"] == "reduced"
    assert _filled_report(tmp_path, loops=0).finish(True)[1]["quality"]["level"] == "reduced"
    assert _filled_report(tmp_path, hidden_lines=False).finish(True)[1]["quality"]["level"] == "reduced"
    assert _filled_report(tmp_path, loops=None,
                          linear=False).finish(True)[1]["quality"]["level"] == "minimal"


def test_report_write_round_trip(tmp_path):
    rep = _filled_report(tmp_path)
    rep.data["geometry"]["bbox_min"] = np.array([0.0, 1.0, 2.0])
    rep.data["geometry"]["volume_mm3"] = float("nan")
    rep.fallback("brep", "mesh", "boom")
    rep.warn("w1")
    rep.finish(True)
    path = tmp_path / "part.report.json"
    rep.write(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["geometry"]["bbox_min"] == [0.0, 1.0, 2.0]
    assert data["geometry"]["volume_mm3"] is None           # NaN is not JSON
    assert data["route"]["fallbacks"] == [{"from": "brep", "to": "mesh", "reason": "boom"}]
    assert data["warnings"] == ["w1"]
    assert data["outputs"]["report"] == str(path)


# --------------------------------------------------------- pipeline: STEP --

@needs_ocp
def test_run_plate_step_brep_route(plate_files, tmp_path):
    from src.drawing.pipeline import run
    out = tmp_path / "out"
    # orient fixed keeps the file axes (plate lying flat), so the section through the
    # O20 hole axis is the probe's y = 30 cut with its two loops
    code, rep = run(plate_files["step"], str(out), _opts(formats=("svg",), orient="fixed"))
    assert code == 0, (rep["error"], rep["quality"])
    assert rep["ok"] is True and rep["quality"]["level"] == "full"
    assert (out / "plate.svg").is_file() and (out / "plate.report.json").is_file()
    assert not (out / "plate.pdf").exists() and not (out / "plate.dxf").exists()
    assert rep["route"]["pipeline"] == "brep" and rep["route"]["source"] == "step"
    texts = [d["text"] for d in rep["dimensions"]]
    # 25 = the overall height including the 10 mm boss (the plan text said 15: plate only)
    for t in ("100", "60", "25", "4 x Ø6.5", "Ø20"):
        assert texts.count(t) == 1, (t, texts)
    assert rep["section"]["loops"] == 2 and rep["section"]["hatch_segments"] > 0
    assert rep["section"]["plane_point"][1] == pytest.approx(30.0)
    assert [v["name"] for v in rep["views"]] == ["front", "top", "right", "iso"]
    assert all(v["algorithm"] == "hlr_exact" for v in rep["views"])
    front = rep["views"][0]
    assert front["hidden"] > 0                     # hole walls are hidden in the front view
    assert rep["geometry"]["face_types"]["Cylinder"] == 6
    assert rep["geometry"]["features"] == {"holes": 5, "bosses": 1, "fillets": 0}
    assert rep["input"]["detected_format"] == "step" and len(rep["input"]["sha256"]) == 64
    assert rep["sheet"]["size"] == "A3" and rep["sheet"]["projection"] == "first_angle"
    on_disk = json.loads((out / "plate.report.json").read_text(encoding="utf-8"))
    assert on_disk["dimensions"] == rep["dimensions"] and on_disk["exit_code"] == 0
    svg = (out / "plate.svg").read_text(encoding="utf-8")
    assert "4 x Ø6.5" in svg and "A-A" in svg


@needs_ocp
def test_run_plate_step_auto_orientation(plate_files, tmp_path):
    """Default orient=auto: shortest extent (25, the plate thickness + boss) becomes the depth."""
    from src.drawing.pipeline import run
    code, rep = run(plate_files["step"], str(tmp_path / "o"), _opts(formats=("pdf",)))
    assert code == 0, (rep["error"], rep["quality"])
    assert rep["route"]["orientation"]["permutation"] == [0, 2, 1]
    assert rep["geometry"]["extents"] == pytest.approx([100.0, 60.0, 25.0])
    lo, hi = np.array(rep["geometry"]["drawing_bbox_min"]), np.array(rep["geometry"]["drawing_bbox_max"])
    assert hi - lo == pytest.approx([100.0, 25.0, 60.0])
    callouts = [d for d in rep["dimensions"] if d["kind"] == "diameter"]
    assert {d["view"] for d in callouts} == {"front"}   # the holes now look at the viewer
    # the front-parallel section now cuts through the thickness: outer + 5 holes
    assert rep["section"]["loops"] == 6
    assert (tmp_path / "o" / "plate.pdf").read_bytes().startswith(b"%PDF-1.4")


@needs_ocp
def test_run_inch_step_flags_unit(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from src.drawing.pipeline import run
    from src.io.occt_utils import init_step_statics

    p = str(tmp_path / "cube_inch.step")
    try:
        init_step_statics(unit="INCH")
        w = STEPControl_Writer()
        w.Transfer(BRepPrimAPI_MakeBox(25.4, 25.4, 12.7).Shape(), STEPControl_AsIs)
        w.Write(p)
    finally:
        init_step_statics(unit="MM")
    with open(p, encoding="latin-1") as fh:
        assert "INCH" in fh.read()                 # precondition: really an inch file
    code, rep = run(p, str(tmp_path / "o"), _opts(formats=("svg",)))
    assert rep["input"]["source_unit"] == "inch"
    assert _codes(rep).get("unit_not_mm") == "warning"
    assert code == 2 and rep["ok"] is True


@needs_ocp
def test_run_garbage_step_fails_cleanly(tmp_path):
    from src.drawing.pipeline import run
    p = tmp_path / "bad.step"
    p.write_bytes(b"ISO-10303-21;\nthis is garbage, not a STEP file\n")
    code, rep = run(str(p), str(tmp_path / "o"), _opts(formats=("svg",)))
    assert code == 1 and rep["ok"] is False
    assert rep["input"]["detected_format"] == "step"          # it got past the sniff
    assert "STEP" in rep["error"]
    assert (tmp_path / "o" / "bad.report.json").is_file()
    assert not (tmp_path / "o" / "bad.svg").exists()


@needs_ocp
@pytest.mark.parametrize("failing, exc_type, code_flag", [
    ("top", RuntimeError, "view_failed"),
    ("right", TimeoutError, "hlr_timeout"),
])
def test_step_view_failure_skips_the_view(plate_files, tmp_path, monkeypatch, failing,
                                          exc_type, code_flag):
    """STEP input: a failed view is skipped and flagged; no mesh fallback (that is STL-only)."""
    import src.drawing.brep_views as bv
    from src.drawing.pipeline import run
    real = bv.project_brep
    seen = []

    def flaky(shape, spec, *a, **kw):
        seen.append(spec.name)
        if spec.name == failing:
            raise exc_type(f"injected failure of {spec.name}")
        return real(shape, spec, *a, **kw)

    monkeypatch.setattr(bv, "project_brep", flaky)
    out = tmp_path / "o"
    code, rep = run(plate_files["step"], str(out), _opts(formats=("svg",)))
    # precondition: every view was attempted (later calls come from the section's own view)
    assert seen[:4] == ["front", "top", "right", "iso"], seen
    assert rep["ok"] is True, rep["error"]
    names = [v["name"] for v in rep["views"]]
    assert len(names) == 3 and failing not in names, names
    assert _codes(rep).get(code_flag) == "warning"
    other = {"view_failed", "hlr_timeout"} - {code_flag}
    assert not other & set(_codes(rep))
    assert rep["route"]["pipeline"] == "brep" and rep["route"]["source"] == "step"
    assert rep["route"]["fallbacks"] == [] and "fallback_to_mesh" not in _codes(rep)
    assert code == 2
    assert (out / "plate.svg").is_file()
    assert failing not in rep["sheet"]["views_placed"]


@needs_ocp
def test_step_all_views_fail_exit_1(plate_files, tmp_path, monkeypatch):
    import src.drawing.brep_views as bv
    from src.drawing.pipeline import run
    seen = []

    def dead(shape, spec, *a, **kw):
        seen.append(spec.name)
        raise RuntimeError("dead HLR")

    monkeypatch.setattr(bv, "project_brep", dead)
    code, rep = run(plate_files["step"], str(tmp_path / "o"), _opts(formats=("svg",)))
    assert seen == ["front", "top", "right", "iso"]    # precondition: each view was tried
    assert code == 1 and rep["ok"] is False
    assert "no view" in rep["error"], rep["error"]
    assert rep["route"]["pipeline"] == "brep" and rep["route"]["fallbacks"] == []
    assert not (tmp_path / "o" / "plate.svg").exists()
    assert (tmp_path / "o" / "plate.report.json").is_file()


@needs_ocp
def test_dimension_planner_failure_still_writes(plate_files, tmp_path, monkeypatch):
    """Spec 3.10: views exist, so a crash of plan_dimensions must not discard the drawing."""
    import src.drawing.dimensions as dm
    from src.drawing.pipeline import run
    calls = []

    def broken(*a, **kw):
        calls.append(1)
        raise ZeroDivisionError("planner bug")

    monkeypatch.setattr(dm, "plan_dimensions", broken)
    out = tmp_path / "o"
    code, rep = run(plate_files["step"], str(out), _opts(formats=("svg",)))
    assert calls                                       # precondition: the planner was reached
    assert rep["ok"] is True, rep["error"]
    assert code == 2 and _codes(rep).get("dimensions_failed") == "warning"
    assert "planner bug" in " ".join(rep["warnings"])
    assert rep["dimensions"] == [] and len(rep["views"]) == 4
    assert rep["section"]["loops"] >= 1                # the section survived too
    assert rep["quality"]["level"] == "reduced"        # no dimension -> not "full"
    assert (out / "plate.svg").is_file()


def test_run_unknown_input_fails(tmp_path):
    from src.drawing.pipeline import run
    p = tmp_path / "noise.stl"
    p.write_bytes(np.random.default_rng(1).bytes(300))
    code, rep = run(str(p), str(tmp_path / "o"), _opts(formats=("svg",)))
    assert code == 1 and rep["ok"] is False
    assert rep["input"]["detected_format"] == "unknown"
    assert "unsupported" in rep["error"]


# ---------------------------------------------------------- pipeline: STL --

@needs_ocp
def test_run_plate_stl_faceted_route(plate_files, tmp_path):
    from src.drawing.pipeline import run
    out = tmp_path / "out"
    assert plate_files["triangles"] <= 30_000      # precondition: under the faceted cap
    code, rep = run(plate_files["stl"], str(out), _opts(formats=("svg",)))
    assert rep["ok"] is True, rep["error"]
    assert rep["route"]["pipeline"] == "brep"
    assert rep["route"]["source"] == "stl_faceted_step"
    assert rep["route"]["classifier"]["verdict"] == "prismatic"
    assert rep["route"]["stl_to_step"]["ok"] is True
    assert rep["route"]["stl_to_step"]["attempted"] is True
    assert rep["route"]["fallbacks"] == []
    step = out / "plate.step"
    assert step.is_file() and step.read_bytes().startswith(b"ISO-10303-21")
    assert rep["outputs"]["step"] == str(step)
    assert rep["views"][0]["algorithm"] == "hlr_poly"
    assert rep["quality"]["level"] == "reduced"    # poly HLR
    assert rep["section"]["loops"] >= 1
    texts = [d["text"] for d in rep["dimensions"]]
    assert "100" in texts and "60" in texts
    assert (out / "plate.svg").is_file()
    # feature detection is skipped on the faceted body: "unknown", never "0 holes"
    assert rep["geometry"]["features"] == {"holes": None, "bosses": None, "fillets": None}
    # independent audit of the drawn body (measure_shape), next to the exporter's own numbers
    audit = rep["route"]["stl_to_step"]["audit"]
    drawn = audit["drawn_body"]
    assert drawn["faces"] == plate_files["triangles"]          # one face per triangle
    assert drawn["solids"] == 1 and drawn["free_edges"] == 0 and drawn["brepcheck_valid"]
    assert drawn["brep_volume_mm3"] == pytest.approx(audit["volume_mesh_mm3"], rel=1e-9)
    assert abs(drawn["volume_delta_pct"]) < 1e-6
    assert drawn["faces"] > audit["faces"]                     # the exporter merged coplanar faces
    # the clean plate gives info flags only -> exit 0
    assert all(sev == "info" for sev in _codes(rep).values()), rep["quality"]["flags"]
    assert code == 0


@needs_ocp
def test_faceted_audit_mismatch_is_flagged(plate_files, tmp_path, monkeypatch):
    """Negative control of the drawn-body audit: a body that loses volume is flagged."""
    import src.io.step_audit as sa
    from src.drawing.pipeline import run
    real = sa.measure_shape
    seen = []

    def lossy(shape, *a, **kw):
        m = real(shape, *a, **kw)
        seen.append(m["brep_volume_mm3"])
        m["brep_volume_mm3"] = 0.5 * m["brep_volume_mm3"]
        return m

    monkeypatch.setattr(sa, "measure_shape", lossy)
    code, rep = run(plate_files["stl"], str(tmp_path / "o"), _opts(formats=("svg",)))
    assert seen and seen[0] > 9e4                      # precondition: the real body was measured
    assert rep["ok"] is True, rep["error"]
    assert rep["route"]["source"] == "stl_faceted_step"
    assert rep["route"]["stl_to_step"]["audit"]["drawn_body"]["volume_delta_pct"] == \
        pytest.approx(-50.0, abs=1e-6)
    assert _codes(rep).get("faceted_audit_mismatch") == "warning"
    assert code == 2


@needs_ocp
def test_run_organic_blob_mesh_route(tmp_path):
    from src.drawing.pipeline import run
    stl = _blob_stl(str(tmp_path / "blob.stl"))
    code, rep = run(stl, str(tmp_path / "o"), _opts(formats=("svg",)))
    assert rep["ok"] is True, rep["error"]
    assert rep["route"]["classifier"]["planar_big"] == 0.0   # precondition: really organic
    assert rep["route"]["pipeline"] == "mesh" and rep["route"]["source"] == "stl_mesh"
    assert _codes(rep).get("organic_classified") == "info"
    assert rep["route"]["stl_to_step"]["skipped_reason"] == "organic"
    assert rep["route"]["stl_to_step"]["attempted"] is False
    assert rep["outputs"]["step"] is None and not (tmp_path / "o" / "blob.step").exists()
    assert rep["quality"]["level"] == "reduced"
    assert all(v["algorithm"] == "depth_buffer" for v in rep["views"])
    assert (tmp_path / "o" / "blob.svg").is_file()


@needs_ocp
def test_stl_to_step_off_forces_mesh(plate_files, tmp_path):
    from src.drawing.pipeline import run
    code, rep = run(plate_files["stl"], str(tmp_path / "o"), _opts(formats=("svg",),
                                                                  stl_to_step="off"))
    assert rep["route"]["classifier"]["verdict"] == "prismatic"   # precondition
    assert rep["ok"] is True, rep["error"]
    assert rep["route"]["pipeline"] == "mesh"
    assert rep["route"]["stl_to_step"]["skipped_reason"] == "off"
    assert rep["views"][0]["algorithm"] == "depth_buffer"
    assert not (tmp_path / "o" / "plate.step").exists()


@needs_ocp
def test_brep_failure_falls_back_to_mesh(plate_files, tmp_path, monkeypatch):
    import src.drawing.brep_views as bv
    from src.drawing.pipeline import run
    calls = []

    def boom(*a, **kw):
        calls.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(bv, "project_brep", boom)
    code, rep = run(plate_files["stl"], str(tmp_path / "o"), _opts(formats=("svg",)))
    assert calls                                    # precondition: the B-rep stage was entered
    assert rep["ok"] is True, rep["error"]
    assert (tmp_path / "o" / "plate.svg").is_file()
    assert "boom" in rep["route"]["fallbacks"][0]["reason"]
    assert rep["route"]["fallbacks"][0]["from"] == "brep"
    assert _codes(rep).get("fallback_to_mesh") == "warning"
    assert rep["route"]["pipeline"] == "mesh"
    assert code == 2


@needs_ocp
def test_zero_face_stl_fails(tmp_path):
    from src.drawing.pipeline import run
    p = tmp_path / "empty.stl"
    p.write_bytes(b"\0" * 80 + struct.pack("<I", 0))
    code, rep = run(str(p), str(tmp_path / "o"), _opts(formats=("svg",)))
    assert rep["input"]["detected_format"] == "stl_binary"   # precondition: valid header
    assert code == 1 and rep["ok"] is False
    assert "triangles" in rep["error"], rep["error"]           # failed for the right reason
    assert not (tmp_path / "o" / "empty.svg").exists()


# --------------------------------------------------------------------- CLI --

def _cli(args, timeout=300):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-m", "src.drawing", *args], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=env, timeout=timeout)


def _result(proc):
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"stdout must be one RESULT line:\n{proc.stdout}\n{proc.stderr[-2000:]}"
    assert lines[0].startswith("RESULT ")
    return json.loads(lines[0][len("RESULT "):])


@needs_ocp
def test_cli_no_sandbox(plate_files, tmp_path):
    out = tmp_path / "out"
    proc = _cli([plate_files["step"], str(out), "--no-sandbox", "--formats", "svg", "--quiet"])
    res = _result(proc)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert res["ok"] is True and res["exit_code"] == 0
    assert res["sandbox"]["enabled"] is False
    assert (out / "plate.svg").is_file() and not (out / "plate.pdf").exists()


@needs_ocp
def test_cli_sandboxed_default(plate_files, tmp_path):
    pytest.importorskip("src.drawing.sandbox", reason="task C2 (sandbox.py) not merged yet")
    out = tmp_path / "out"
    proc = _cli([plate_files["step"], str(out), "--formats", "svg,dxf", "--quiet"])
    res = _result(proc)
    assert proc.returncode == 0, (res.get("error"), proc.stderr[-2000:])
    box = res["sandbox"]
    assert box["enabled"] is True
    assert box["isolation"] in ("job_object", "psutil_tree", "posix_rlimit")
    assert box["peak_rss_mb"] > 50
    assert res["outputs"]["svg"] == str(out / "plate.svg")
    assert (out / "plate.svg").is_file() and (out / "plate.dxf").is_file()
    assert res["input"]["path"] == os.path.abspath(plate_files["step"])
    on_disk = json.loads((out / "plate.report.json").read_text(encoding="utf-8"))
    assert on_disk["sandbox"]["enabled"] is True
    assert on_disk["sandbox"]["isolation"] == box["isolation"]


@needs_ocp
def test_cli_sandboxed_child_failure_keeps_child_error(tmp_path):
    pytest.importorskip("src.drawing.sandbox", reason="task C2 (sandbox.py) not merged yet")
    p = tmp_path / "bad.step"
    p.write_bytes(b"ISO-10303-21;\nthis is garbage, not a STEP file\n")
    proc = _cli([str(p), str(tmp_path / "out"), "--quiet"])
    res = _result(proc)
    assert res["input"]["detected_format"] == "step"   # precondition: passed the parent sniff
    assert proc.returncode == 1 and res["ok"] is False
    assert res["sandbox"]["enabled"] is True and res["sandbox"]["exit_code"] == 1
    assert "STEP" in res["error"] and not res["error"].startswith("sandbox:")


def _fake_sandbox(monkeypatch, *, exit_code, result, **flags):
    import src.drawing.sandbox as sb
    seen = {}

    def fake_run(argv, work_dir, **kw):
        staging = argv[4]
        seen["argv"], seen["work_dir"], seen["kw"] = argv, work_dir, kw
        svg = os.path.join(staging, "part.svg")
        with open(svg, "w", encoding="utf-8") as fh:
            fh.write("<svg/>")
        res = None
        if result is not None:
            res = dict(result)
            res["outputs"] = {"svg": svg, "pdf": None, "report": None}
        box = {"isolation": "job_object", "timed_out": False, "memory_exceeded": False,
               "cpu_exceeded": False, "output_exceeded": False, "killed": False,
               "result_error": None if res is not None else "no RESULT line",
               "memory_limit_mb": kw.get("memory_limit_mb"), "timeout_s": kw.get("timeout_s"),
               "exit_code": exit_code}
        box.update(flags)
        return sb.SandboxResult(exit_code=exit_code, result=res, sandbox=box,
                                stdout_tail="", stderr_tail="")

    monkeypatch.setattr(sb, "run_sandboxed", fake_run)
    return seen


def _part_stl(tmp_path):
    import trimesh
    p = tmp_path / "part.stl"
    trimesh.creation.box((10, 20, 30)).export(str(p))
    return str(p)


def test_cli_sandbox_parent_moves_outputs_and_merges(tmp_path, monkeypatch):
    pytest.importorskip("src.drawing.sandbox", reason="task C2 (sandbox.py) not merged yet")
    from src.drawing import cli
    seen = _fake_sandbox(monkeypatch, exit_code=0,
                         result={"ok": True, "exit_code": 0, "error": None, "input": {}})
    out = tmp_path / "out"
    code, res = cli._run_sandboxed(_part_stl(tmp_path), str(out), _opts(timeout_s=40.0),
                                   sys.stderr)
    assert seen["argv"][1:3] == ["-m", "src.drawing"] and "--no-sandbox" in seen["argv"]
    assert seen["kw"]["timeout_s"] == 40.0 and seen["kw"]["cpu_limit_s"] == 80.0
    assert code == 0 and res["ok"] is True
    assert res["outputs"]["svg"] == str(out / "part.svg") and (out / "part.svg").is_file()
    assert not os.path.exists(seen["work_dir"])         # work dir removed
    on_disk = json.loads((out / "part.report.json").read_text(encoding="utf-8"))
    assert on_disk["sandbox"]["enabled"] is True and on_disk["sandbox"]["isolation"] == "job_object"


def test_cli_sandbox_breach_with_result_fails(tmp_path, monkeypatch):
    pytest.importorskip("src.drawing.sandbox", reason="task C2 (sandbox.py) not merged yet")
    from src.drawing import cli
    _fake_sandbox(monkeypatch, exit_code=0, memory_exceeded=True,
                  result={"ok": True, "exit_code": 0, "error": None, "input": {}})
    code, res = cli._run_sandboxed(_part_stl(tmp_path), str(tmp_path / "out"), _opts(), sys.stderr)
    assert res["sandbox"]["memory_exceeded"] is True       # precondition: the breach is reported
    assert code == 1 and res["ok"] is False and res["exit_code"] == 1
    assert "memory" in res["error"]


def test_cli_sandbox_no_result_fails(tmp_path, monkeypatch):
    pytest.importorskip("src.drawing.sandbox", reason="task C2 (sandbox.py) not merged yet")
    from src.drawing import cli
    _fake_sandbox(monkeypatch, exit_code=3221225477, result=None,
                  result_error="native crash (input unreadable by OpenCASCADE?)")
    code, res = cli._run_sandboxed(_part_stl(tmp_path), str(tmp_path / "out"), _opts(), sys.stderr)
    assert code == 1 and res["ok"] is False
    assert "native crash" in res["error"]
    assert res["sandbox"]["exit_code"] == 3221225477
    assert (tmp_path / "out" / "part.report.json").is_file()


def test_cli_missing_input(tmp_path):
    out = tmp_path / "out"
    proc = _cli([str(tmp_path / "nope.step"), str(out), "--quiet"])
    res = _result(proc)
    assert proc.returncode == 1
    assert res["ok"] is False and "not found" in res["error"]
    assert (out / "nope.report.json").is_file()


def test_cli_usage_error(tmp_path):
    proc = _cli(["x.step", str(tmp_path), "--sheet", "A5"])
    assert proc.returncode == 1
    assert proc.stdout.strip() == ""
    assert "usage" in proc.stderr.lower() and "A5" in proc.stderr
