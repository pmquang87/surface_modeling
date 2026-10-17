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
    # QA issue 10: the holes run along the front normal, so auto takes the TOP plane
    # through the O20 hole (cutting line in the front view): the two halves of the cut
    assert rep["section"]["reason"] == "hole_axis" and rep["section"]["view"] == "top"
    assert rep["section"]["loops"] == 2
    assert "section_fallback" not in _codes(rep)
    # the boss faces the front viewer already: no depth flip (QA issue 14)
    assert rep["route"]["orientation"]["depth_flipped"] is False
    assert rep["route"]["orientation"]["depth_scores"]["kept"] > \
        rep["route"]["orientation"]["depth_scores"]["flipped"]
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


# ------------------------------------------------- QA round 1 (F6) helpers --

def _compound(shapes):
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    b = BRep_Builder()
    comp = TopoDS_Compound()
    b.MakeCompound(comp)
    for s in shapes:
        b.Add(comp, s)
    return comp


def _write_step(shape, path):
    from src.io.exporters import export_step
    export_step(shape, str(path))
    assert os.path.getsize(str(path)) > 0
    return str(path)


def _write_step_unit(shape, path, unit):
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from src.io.occt_utils import init_step_statics
    try:
        init_step_statics(unit=unit)
        w = STEPControl_Writer()
        w.Transfer(shape, STEPControl_AsIs)
        w.Write(str(path))
    finally:
        init_step_statics(unit="MM")
    return str(path)


def _box_shape(dx, dy, dz, origin=(0.0, 0.0, 0.0)):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    return BRepPrimAPI_MakeBox(gp_Pnt(*origin), dx, dy, dz).Shape()


def _texts(rep):
    return [d["text"] for d in rep["dimensions"]]


def _linear_values(rep):
    return sorted(round(d["value_mm"], 3) for d in rep["dimensions"] if d["kind"] == "linear")


def _box_stl(path, extents, *, process=True):
    import trimesh
    trimesh.creation.box(extents=extents).export(str(path))
    return str(path)


# ---------------------------------------------------- issue 1: loose curves --

@needs_ocp
def test_issue1_loose_curves_and_supplemental_shells_flagged_not_drawn(tmp_path, monkeypatch):
    """Box 100x50x20 + a loose line x = -50..150 (curve set) + a free face at x = 200..210:
    drawn, bounded and dimensioned as the box only, with info flags naming what was left out."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    import src.drawing.brep_load as bl
    from src.drawing.pipeline import run

    edge = BRepBuilderAPI_MakeEdge(gp_Pnt(-50, 0, 0), gp_Pnt(150, 0, 0)).Edge()
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
                                   200.0, 210.0, 0.0, 10.0).Face()
    p = _write_step(_compound([_box_shape(100, 50, 20), edge, face]), tmp_path / "mixed.step")
    text = open(p, encoding="latin-1").read()
    assert "GEOMETRIC_CURVE_SET" in text and "SHELL_BASED_SURFACE_MODEL" in text  # precondition

    code, rep = run(p, str(tmp_path / "o"), _opts(formats=("svg",)))
    assert rep["ok"] is True, rep["error"]
    codes = _codes(rep)
    assert codes.get("loose_curves_ignored") == "info"
    assert codes.get("supplemental_shells_ignored") == "info"
    assert "open_shell" not in codes and "multiple_bodies" not in codes
    assert rep["geometry"]["extents"] == pytest.approx([100.0, 50.0, 20.0])
    assert rep["geometry"]["closed"] is True and rep["geometry"]["free_edges"] == 0
    assert _linear_values(rep) == [20.0, 50.0, 100.0]
    assert code == 0, rep["quality"]["flags"]

    # negative control: without the body selection the curve set widens the part to 260
    monkeypatch.setattr(bl, "select_drawn_shape",
                        lambda shape: (shape, {"drawn": "solids", "dropped": {}}))
    code2, rep2 = run(p, str(tmp_path / "o2"), _opts(formats=("svg",)))
    assert rep2["geometry"]["extents"][0] == pytest.approx(260.0, abs=1e-6)
    assert "loose_curves_ignored" not in _codes(rep2)


# ------------------------------------------------ issue 16: tessellated STEP --

_TESS_PTS = [(0, 0, 0), (20, 0, 0), (20, 10, 0), (0, 10, 0),
             (0, 0, 5), (20, 0, 5), (20, 10, 5), (0, 10, 5)]
_TESS_A = [(1, 3, 2), (1, 4, 3), (5, 6, 7), (5, 7, 8), (1, 2, 6), (1, 6, 5)]
_TESS_B = [(2, 3, 7), (2, 7, 6), (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8)]


def _tess_step_text():
    """Minimal AP242 file: one TESSELLATED_SOLID (two TRIANGULATED_FACEs) of a 20x10x5 box."""
    coords = ",".join("(%.1f,%.1f,%.1f)" % p for p in _TESS_PTS)

    def tl(ts):
        return ",".join("(%d,%d,%d)" % t for t in ts)
    return (
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('tessellated box'),'2;1');\n"
        "FILE_NAME('tess.stp','2026-10-17T00:00:00',(''),(''),'','','');\n"
        "FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF "
        "{ 1 0 10303 442 1 1 4 }'));\nENDSEC;\nDATA;\n"
        "#1=APPLICATION_CONTEXT('Managed model based 3d engineering');\n"
        "#2=APPLICATION_PROTOCOL_DEFINITION('International Standard',"
        "'ap242_managed_model_based_3d_engineering',2011,#1);\n"
        "#3=PRODUCT_CONTEXT(' ',#1,'mechanical');\n"
        "#4=PRODUCT_DEFINITION_CONTEXT('part definition',#1,' ');\n"
        "#5=PRODUCT('tess','tess','',(#3));\n"
        "#6=PRODUCT_DEFINITION_FORMATION('',' ',#5);\n"
        "#7=PRODUCT_DEFINITION('',' ',#6,#4);\n"
        "#8=PRODUCT_DEFINITION_SHAPE(' ',' ',#7);\n"
        "#10=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));\n"
        "#11=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));\n"
        "#12=(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT());\n"
        "#13=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-05),#10,' ',' ');\n"
        "#14=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#13))"
        "GLOBAL_UNIT_ASSIGNED_CONTEXT((#10,#11,#12))REPRESENTATION_CONTEXT(' ',' '));\n"
        f"#20=COORDINATES_LIST('',8,({coords}));\n"
        f"#21=TRIANGULATED_FACE('',#20,8,(),$,(1,2,3,4,5,6,7,8),({tl(_TESS_A)}));\n"
        f"#22=TRIANGULATED_FACE('',#20,8,(),$,(1,2,3,4,5,6,7,8),({tl(_TESS_B)}));\n"
        "#23=TESSELLATED_SOLID('body',(#21,#22),$);\n"
        "#24=TESSELLATED_SHAPE_REPRESENTATION('tess',(#23),#14);\n"
        "#25=SHAPE_DEFINITION_REPRESENTATION(#8,#24);\n"
        "ENDSEC;\nEND-ISO-10303-21;\n")


@needs_ocp
def test_issue16_tessellated_step_goes_to_the_mesh_route(tmp_path):
    from src.drawing.brep_load import TessellatedOnlyError, load_step
    from src.drawing.pipeline import run
    p = tmp_path / "tess.stp"
    p.write_text(_tess_step_text(), encoding="ascii")
    with pytest.raises(TessellatedOnlyError):         # precondition: the B-rep route cannot run
        load_step(str(p))
    code, rep = run(str(p), str(tmp_path / "o"), _opts(formats=("svg",)))
    assert rep["ok"] is True, rep["error"]
    assert rep["route"]["pipeline"] == "mesh"
    assert rep["route"]["source"] == "step_tessellated_mesh"
    codes = _codes(rep)
    assert codes.get("step_tessellated_only") == "warning"
    assert not {"inverted_solid", "brepcheck_invalid", "open_shell"} & set(codes)
    assert rep["input"]["source_unit"] == "mm"
    assert rep["geometry"]["volume_mm3"] == pytest.approx(1000.0, rel=1e-9)
    assert rep["geometry"]["extents"] == pytest.approx([20.0, 10.0, 5.0])
    assert len(rep["views"]) == 4 and code == 2
    assert (tmp_path / "o" / "tess.svg").is_file()


# ------------------------------------------------------------ issue 5: units --

@needs_ocp
def test_issue5_stl_unit_assumed_and_inch_fraction_suspect(tmp_path):
    from src.drawing.pipeline import run
    inch = _box_stl(tmp_path / "inch_part.stl", (5.0, 2.5, 1.375))
    code, rep = run(inch, str(tmp_path / "a"), _opts(formats=("svg",), stl_to_step="off"))
    assert rep["ok"] is True, rep["error"]
    codes = _codes(rep)
    assert rep["input"]["source_unit"] == "assumed_mm"
    assert codes.get("unit_assumed_mm") == "info"
    assert codes.get("unit_suspect") == "warning"
    detail = next(f["detail"] for f in rep["quality"]["flags"] if f["code"] == "unit_suspect")
    assert "in" in detail and "127" in detail
    # control: a plausible mm part keeps the info flag only
    mm = _box_stl(tmp_path / "mm_part.stl", (50.0, 30.0, 20.0))
    code2, rep2 = run(mm, str(tmp_path / "b"), _opts(formats=("svg",), stl_to_step="off"))
    assert _codes(rep2).get("unit_assumed_mm") == "info"
    assert "unit_suspect" not in _codes(rep2)
    # --stl-unit in scales the mesh: 127 x 63.5 x 34.925 mm, recorded in the report
    code3, rep3 = run(inch, str(tmp_path / "c"), _opts(formats=("svg",), stl_to_step="off",
                                                       stl_unit="in"))
    assert rep3["ok"] is True, rep3["error"]
    assert rep3["input"]["source_unit"] == "in" and rep3["input"]["unit_factor"] == 25.4
    assert not {"unit_assumed_mm", "unit_suspect"} & set(_codes(rep3))
    assert rep3["geometry"]["extents"] == pytest.approx([127.0, 63.5, 34.925])
    assert _linear_values(rep3) == pytest.approx([34.925, 63.5, 127.0], abs=1e-3)
    assert rep3["geometry"]["volume_mm3"] == pytest.approx(127.0 * 63.5 * 34.925, rel=1e-9)


@needs_ocp
def test_issue5_step_declared_inch_with_mm_sized_numbers_is_suspect(tmp_path):
    """Raw numbers 200 x 100 x 50 declared INCH read as 5080 mm: implausible (the as1 case)."""
    from src.drawing.pipeline import run
    big = _write_step_unit(_box_shape(200 * 25.4, 100 * 25.4, 50 * 25.4), tmp_path / "big.step",
                           "INCH")
    text = open(big, encoding="latin-1").read()
    assert "INCH" in text and "200." in text           # precondition: raw 200 in an inch file
    code, rep = run(big, str(tmp_path / "a"), _opts(formats=("svg",), section="off"))
    assert rep["input"]["source_unit"] == "inch"
    assert rep["geometry"]["extents"][0] == pytest.approx(5080.0, rel=1e-6)
    assert _codes(rep).get("unit_suspect") == "warning"
    alt = rep["input"]["unit_plausibility"]["alternative"]
    assert alt["unit"] == "mm" and alt["extents_mm"][0] == pytest.approx(200.0, rel=1e-6)
    # control: a 1 inch cube is plausible
    small = _write_step_unit(_box_shape(25.4, 25.4, 12.7), tmp_path / "small.step", "INCH")
    code2, rep2 = run(small, str(tmp_path / "b"), _opts(formats=("svg",), section="off"))
    assert rep2["input"]["source_unit"] == "inch"
    assert "unit_suspect" not in _codes(rep2)


# ------------------------------------------ issues 11/12: STL bodies, volume --

def _two_boxes_mesh(invert_second=False):
    import trimesh
    a = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    b = trimesh.creation.box(extents=(20.0, 10.0, 10.0))
    b.apply_translation((40.0, 0.0, 0.0))
    if invert_second:
        b = trimesh.Trimesh(b.vertices.copy(), b.faces[:, ::-1].copy(), process=False)
    return trimesh.util.concatenate([a, b])


@needs_ocp
@pytest.mark.parametrize("mode, pipeline_name", [("auto", "brep"), ("off", "mesh")])
def test_issue11_multiple_bodies_flagged_on_stl_routes(tmp_path, mode, pipeline_name):
    from src.drawing.mesh_load import mesh_census
    from src.drawing.pipeline import run
    mesh = _two_boxes_mesh()
    assert mesh_census(mesh)["components"] == 2         # precondition: two bodies
    p = str(tmp_path / "two.stl")
    mesh.export(p)
    code, rep = run(p, str(tmp_path / "o"), _opts(formats=("svg",), stl_to_step=mode))
    assert rep["ok"] is True, rep["error"]
    assert rep["route"]["pipeline"] == pipeline_name
    assert _codes(rep).get("multiple_bodies") == "warning"
    assert code == 2


@needs_ocp
def test_issue12_faceted_audit_uses_the_repaired_volume(tmp_path):
    from src.drawing.pipeline import run
    mesh = _two_boxes_mesh(invert_second=True)
    # precondition: closed, but the raw signed sum (1000 - 2000) is meaningless; the old
    # audit compared the drawn body (3000) against abs(mesh.volume) = 1000 and flagged it
    assert mesh.is_watertight
    assert abs(float(mesh.volume)) == pytest.approx(1000.0, rel=1e-6)
    p = str(tmp_path / "wound.stl")
    mesh.export(p)
    code, rep = run(p, str(tmp_path / "o"), _opts(formats=("svg",)))
    assert rep["ok"] is True, rep["error"]
    assert rep["route"]["source"] == "stl_faceted_step", rep["route"]
    assert "faceted_audit_mismatch" not in _codes(rep), rep["quality"]["flags"]
    drawn = rep["route"]["stl_to_step"]["audit"]["drawn_body"]
    assert drawn["mesh_volume_mm3"] == pytest.approx(3000.0, rel=1e-9)
    g = rep["geometry"]
    assert g["volume_mm3"] == pytest.approx(3000.0, rel=1e-6)
    assert g["free_edges"] == 0 and g["closed"] is True
    assert g["face_types"] == {"Plane": g["faces"]}


# -------------------------------------------------------- issue 14: depth sign --

def _plate_with_bosses(side):
    """100 x 10 x 60 plate (depth along Y) with six r4 x 5 bosses on the +Y or -Y face."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    shape = _box_shape(100, 10, 60)
    y0, d = (10.0, 1.0) if side > 0 else (0.0, -1.0)
    for x in (20.0, 50.0, 80.0):
        for z in (15.0, 45.0):
            boss = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y0, z), gp_Dir(0, d, 0)),
                                            4.0, 5.0).Shape()
            shape = BRepAlgoAPI_Fuse(shape, boss).Shape()
    return shape


@needs_ocp
def test_issue14_depth_sign_turns_a_plate_to_its_featured_side(tmp_path):
    from src.drawing.pipeline import run
    far = _write_step(_plate_with_bosses(+1), tmp_path / "far.step")
    near = _write_step(_plate_with_bosses(-1), tmp_path / "near.step")
    o = _opts(formats=("svg",), section="off", views=("front", "top"))
    _, rep_far = run(far, str(tmp_path / "a"), o)
    _, rep_near = run(near, str(tmp_path / "b"), o)
    assert rep_far["ok"] and rep_near["ok"], (rep_far["error"], rep_near["error"])
    # precondition: both get the same permutation, so only the geometry can decide
    assert rep_far["route"]["orientation"]["permutation"] == \
        rep_near["route"]["orientation"]["permutation"] == [0, 1, 2]
    of, on = rep_far["route"]["orientation"], rep_near["route"]["orientation"]
    assert of["depth_flipped"] is True and on["depth_flipped"] is False
    assert of["depth_scores"]["flipped"] > of["depth_scores"]["kept"]
    assert on["depth_scores"]["kept"] > on["depth_scores"]["flipped"]
    assert of["flips"] == [-1, -1, 1]                  # 180 deg about Z
    front = {v["name"]: v for v in rep_far["views"]}["front"]
    assert front["visible"] > front["hidden"]           # the bosses face the viewer
    # orient fixed keeps the file axes
    _, rep_fixed = run(far, str(tmp_path / "c"), _opts(formats=("svg",), section="off",
                                                       views=("front",), orient="fixed"))
    assert rep_fixed["route"]["orientation"]["depth_flipped"] is False


@needs_ocp
def test_issue14_depth_sign_on_the_mesh_route(tmp_path):
    from src.drawing.pipeline import run
    p = str(tmp_path / "far.stl")
    _tessellate(_plate_with_bosses(+1), deflection=0.2).export(p)
    code, rep = run(p, str(tmp_path / "o"), _opts(formats=("svg",), section="off",
                                                  views=("front",), stl_to_step="off"))
    assert rep["ok"] is True, rep["error"]
    o = rep["route"]["orientation"]
    assert o["permutation"] == [0, 1, 2]               # precondition: identity permutation
    assert o["depth_method"] == "mesh_feature_edges"
    assert o["depth_flipped"] is True and o["flips"] == [-1, -1, 1]
    # mirror control: bosses already on the viewer's side -> no turn (an
    # always-flip scorer fails here)
    p2 = str(tmp_path / "near.stl")
    _tessellate(_plate_with_bosses(-1), deflection=0.2).export(p2)
    _, rep2 = run(p2, str(tmp_path / "o2"), _opts(formats=("svg",), section="off",
                                                  views=("front",), stl_to_step="off"))
    assert rep2["ok"] is True, rep2["error"]
    o2 = rep2["route"]["orientation"]
    assert o2["permutation"] == [0, 1, 2] and o2["depth_method"] == "mesh_feature_edges"
    assert o2["depth_flipped"] is False and o2["flips"] == [1, 1, 1]
    assert o2["depth_scores"]["kept"] > o2["depth_scores"]["flipped"]


# --------------------------------------------- issues 19/18/13/3: mesh flags --

@needs_ocp
def test_issue19_triangle_soup_is_not_a_solid(tmp_path):
    import trimesh
    from src.drawing.pipeline import run
    rng = np.random.default_rng(7)
    centres = rng.uniform(0, 50, size=(50, 3))
    tris = centres[:, None, :] + rng.uniform(-2, 2, size=(50, 3, 3))
    soup = trimesh.Trimesh(tris.reshape(-1, 3), np.arange(150).reshape(-1, 3), process=False)
    p = str(tmp_path / "soup.stl")
    soup.export(p)
    code, rep = run(p, str(tmp_path / "o"), _opts(formats=("svg",)))
    assert rep["ok"] is True, rep["error"]
    assert rep["route"]["classifier"]["verdict"] == "not_solid"
    assert _codes(rep).get("not_a_solid") == "warning"
    assert rep["route"]["pipeline"] == "mesh"
    assert rep["route"]["stl_to_step"]["skipped_reason"] == "not_solid"
    assert rep["section"] is None and rep["dimensions"] == []
    assert rep["undimensioned"] == [{"kind": "overall", "reason": "not_a_solid", "view": None}]
    assert rep["quality"]["level"] == "minimal"


@needs_ocp
def test_issue18_inverted_non_manifold_mesh_flags(tmp_path):
    import trimesh
    from src.drawing.mesh_load import mesh_census
    from src.drawing.pipeline import run
    a = trimesh.creation.box(bounds=[(0, 0, 0), (10, 10, 10)])
    b = trimesh.creation.box(bounds=[(10, 10, 0), (20, 20, 10)])   # shares the edge x=y=10
    mesh = trimesh.util.concatenate([a, b])
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[:, ::-1], process=True)   # inside out
    c = mesh_census(mesh)
    assert c["boundary_edges"] == 0 and c["nonmanifold_edges"] > 0 and c["inverted"]  # precondition
    p = str(tmp_path / "inv.stl")
    mesh.export(p)
    code, rep = run(p, str(tmp_path / "o"), _opts(formats=("svg",), stl_to_step="off"))
    assert rep["ok"] is True, rep["error"]
    codes = _codes(rep)
    assert codes.get("inverted_solid") == "warning"
    assert codes.get("mesh_non_manifold") == "warning"
    detail = next(f["detail"] for f in rep["quality"]["flags"] if f["code"] == "mesh_not_watertight")
    assert "non-manifold" in detail and "0 boundary" in detail
    assert rep["geometry"]["volume_mm3"] == pytest.approx(2000.0, rel=1e-6)


@needs_ocp
def test_issue13_interior_silhouettes_omitted_flag(tmp_path, monkeypatch):
    import src.drawing.mesh_views as mv
    from src.drawing.pipeline import run
    stl = _blob_stl(str(tmp_path / "blob.stl"))
    o = _opts(formats=("svg",), section="off", views=("front",))
    _, rep = run(stl, str(tmp_path / "a"), o)
    assert rep["route"]["classifier"]["verdict"] == "organic"      # precondition
    assert "interior_silhouettes_omitted" not in _codes(rep)
    monkeypatch.setattr(mv, "_MAX_ORGANIC_SILHOUETTE_FACES", 10)
    _, rep2 = run(stl, str(tmp_path / "b"), o)
    assert _codes(rep2).get("interior_silhouettes_omitted") == "info"


@needs_ocp
def test_issue3_hidden_lines_dropped_flag(plate_files, tmp_path):
    from src.drawing.pipeline import run
    code, rep = run(plate_files["step"], str(tmp_path / "o"),
                    _opts(formats=("svg",), section="off", views=("top", "iso"),
                          max_hidden_primitives=0))
    assert rep["ok"] is True, rep["error"]
    flags = [f for f in rep["quality"]["flags"] if f["code"] == "hidden_lines_dropped"]
    assert len(flags) == 1 and "top" in flags[0]["detail"]      # iso has no hidden lines
    assert "max_hidden_primitives" in flags[0]["detail"]
    top = {v["name"]: v for v in rep["views"]}["top"]
    assert top["hidden"] == 0 and top["hidden_dropped"] == "max_hidden_primitives"
    assert code == 2
    # control: without the cap the same view keeps its hidden lines (hole walls) and no flag
    _, rep2 = run(plate_files["step"], str(tmp_path / "o2"),
                  _opts(formats=("svg",), section="off", views=("top",)))
    assert rep2["views"][0]["hidden"] > 0 and "hidden_lines_dropped" not in _codes(rep2)


# ------------------------------------------------- issue 10: section fallback --

@needs_ocp
def test_issue10_section_fallback_flags(tmp_path):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    from src.drawing.pipeline import run
    box = _write_step(_box_shape(100, 60, 40), tmp_path / "box.step")
    code, rep = run(box, str(tmp_path / "a"), _opts(formats=("svg",), views=("front", "top")))
    assert rep["ok"] is True, rep["error"]
    assert rep["section"]["reason"] == "bbox_centre_fallback"       # no hole to cut through
    assert _codes(rep).get("section_fallback") == "info"
    assert rep["section"]["loops"] == 1
    # an open surface model: no section at all, and no misleading section_empty
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
                                   0.0, 100.0, 0.0, 50.0).Face()
    sheet = _write_step(face, tmp_path / "sheet.step")
    code2, rep2 = run(sheet, str(tmp_path / "b"), _opts(formats=("svg",), views=("front", "top")))
    assert rep2["ok"] is True, rep2["error"]
    assert rep2["geometry"]["solids"] == 0                          # precondition
    fl = [f for f in rep2["quality"]["flags"] if f["code"] == "section_fallback"]
    assert fl and "no solid" in fl[0]["detail"]
    assert rep2["section"] is None and "section_empty" not in _codes(rep2)


# ----------------------------------------------------- issue 25: bad STL text --

def test_issue25_empty_and_truncated_stl_errors(tmp_path):
    import trimesh
    from src.drawing import cli
    from src.drawing.pipeline import run
    empty = tmp_path / "empty.stl"
    empty.write_bytes(b"solid Empty\nendsolid Empty\n")
    code, rep = run(str(empty), str(tmp_path / "o"), _opts(formats=("svg",)))
    assert rep["input"]["detected_format"] == "stl_ascii"           # precondition: recognised
    assert code == 1 and rep["error"] == "empty STL (0 facets)"
    full = tmp_path / "full.stl"
    trimesh.creation.box((10, 20, 30)).export(str(full))
    data = full.read_bytes()
    trunc = tmp_path / "trunc.stl"
    trunc.write_bytes(data[:-10])
    code2, rep2 = run(str(trunc), str(tmp_path / "o"), _opts(formats=("svg",)))
    assert rep2["input"]["detected_format"] == "stl_binary_truncated"
    assert code2 == 1
    # one source of truth: sniff.unreadable_reason (the triage wording)
    from src.drawing.sniff import sniff, unreadable_reason
    assert rep2["error"] == ("binary STL truncated: header declares 12 triangles (684 bytes), "
                             "file is 674 bytes"), rep2["error"]
    assert rep2["error"] == unreadable_reason(sniff(str(trunc)))
    # the sandbox parent gives the same texts without spawning a child
    c3, r3 = cli._run_sandboxed(str(trunc), str(tmp_path / "p"), _opts(), sys.stderr)
    assert c3 == 1 and r3["error"] == rep2["error"]
    c4, r4 = cli._run_sandboxed(str(empty), str(tmp_path / "p"), _opts(), sys.stderr)
    assert c4 == 1 and r4["error"] == "empty STL (0 facets)"


# ------------------------------------------------------ issue 17: progress --

@needs_ocp
def test_issue17_pipeline_writes_progress_at_stage_boundaries(plate_files, tmp_path,
                                                              monkeypatch):
    import src.drawing.pipeline as pl
    from src.drawing.report import PROGRESS_ENV
    prog = tmp_path / "progress.json"
    monkeypatch.setenv(PROGRESS_ENV, str(prog))
    seen = []
    real = pl.write_progress

    def spy(path, snap):
        seen.append((snap["stage"], snap["view"]))
        if snap["stage"] == "load":
            units.append(snap["source_unit"])
        return real(path, snap)

    units = []

    monkeypatch.setattr(pl, "write_progress", spy)
    code, rep = pl.run(plate_files["step"], str(tmp_path / "o"), _opts(formats=("svg",)))
    assert rep["ok"] is True, rep["error"]
    stages = [s for s, _ in seen]
    for st in ("sniff", "load", "loaded", "views", "section", "dimensions", "layout", "write",
               "done"):
        assert st in stages, (st, stages)
    assert [v for s, v in seen if s == "views"] == ["front", "top", "right", "iso"]
    assert stages.index("loaded") < stages.index("views") < stages.index("section")
    assert units == ["mm"]           # known before the (possibly long) load: text sniff
    last = json.loads(prog.read_text(encoding="utf-8"))
    assert last["stage"] == "done" and last["source_unit"] == "mm"
    assert last["counts"]["views"] == 4 and last["counts"]["section_loops"] == 2
    assert last["geometry"]["faces"] == rep["geometry"]["faces"]
    # control: no environment variable, no file and no snapshot
    monkeypatch.delenv(PROGRESS_ENV)
    seen.clear()
    pl.run(plate_files["step"], str(tmp_path / "o2"), _opts(formats=("svg",), section="off",
                                                             views=("front",)))
    assert seen == []


def test_issue17_cli_merges_progress_when_the_child_is_killed(tmp_path, monkeypatch):
    import src.drawing.sandbox as sb
    from src.drawing import cli
    seen = {}
    progress = {"stage": "views", "view": "top", "t_s": 231.5, "source_unit": "mm",
                "route": {"pipeline": "brep", "source": "step"},
                "geometry": {"faces": 10665, "solids": 18},
                "flags": [{"code": "hlr_poly_used", "severity": "info", "detail": "10665 faces"}],
                "timing": {"load_s": 25.0, "views_s": 200.0}}

    def fake_run(argv, work_dir, **kw):
        seen["kw"] = kw
        box = {"isolation": "job_object", "timed_out": True, "killed": True,
               "memory_exceeded": False, "cpu_exceeded": False, "output_exceeded": False,
               "timeout_s": kw.get("timeout_s"), "wall_s": 240.2, "exit_code": 1,
               "result_error": "no RESULT line", "progress_error": None, "last_stage": "views"}
        return sb.SandboxResult(exit_code=1, result=None, sandbox=box, stdout_tail="",
                                stderr_tail="", progress=progress)

    monkeypatch.setattr(sb, "run_sandboxed", fake_run)
    code, res = cli._run_sandboxed(_part_stl(tmp_path), str(tmp_path / "out"),
                                   _opts(timeout_s=240.0), sys.stderr)
    assert seen["kw"]["progress_file"] == "progress.json"     # the parent asked for progress
    assert code == 1 and res["ok"] is False
    flags = {f["code"]: f for f in res["quality"]["flags"]}
    assert "sandbox_timeout" in flags and "views" in flags["sandbox_timeout"]["detail"]
    assert "hlr_poly_used" in flags
    assert res["input"]["source_unit"] == "mm"
    assert res["quality"]["last_stage"] == "views"
    assert res["timing"]["total_s"] == pytest.approx(240.2)
    assert res["timing"]["views_s"] == pytest.approx(200.0)
    assert res["geometry"]["faces"] == 10665
    on_disk = json.loads((tmp_path / "out" / "part.report.json").read_text(encoding="utf-8"))
    assert on_disk["quality"]["last_stage"] == "views"


# --------------------------------------------- issue 22: every flag is known --

_FLAG_CALL = r"""\.flag\(\s*["']([A-Za-z0-9_]+)["']"""


def _flag_codes_in(text):
    import re
    return set(re.findall(_FLAG_CALL, text))


def test_issue22_every_emitted_flag_is_in_the_table():
    from src.drawing import cli
    from src.drawing.report import FLAGS
    found = set()
    for mod in ("pipeline.py", "cli.py"):
        path = os.path.join(REPO, "src", "drawing", mod)
        found |= _flag_codes_in(open(path, encoding="utf-8").read())
    # precondition: the scan sees the codes (a broken regex would find none)
    assert {"hidden_lines_dropped", "not_a_solid", "unit_suspect", "step_tessellated_only",
            "loose_curves_ignored", "section_fallback", "multiple_bodies"} <= found
    assert len(found) >= 30, sorted(found)
    missing = sorted(c for c in found if c not in FLAGS)
    assert missing == [], missing
    assert {code for _, code in cli._BREACHES} <= set(FLAGS)
    # mutation check of the scanner: an unknown code in the text is reported
    assert "bogus_code" in _flag_codes_in('ctx.rep.flag("bogus_code", "info", "x")') - set(FLAGS)


def test_cli_stl_unit_and_hidden_cap_options():
    from src.drawing.cli import build_parser, options_from_args
    p = build_parser()
    o = options_from_args(p.parse_args(["a.stl", "out", "--stl-unit", "in",
                                        "--max-hidden-primitives", "500"]), p)
    assert o.stl_unit == "in" and o.max_hidden_primitives == 500
    d = options_from_args(p.parse_args(["a.stl", "out"]), p)
    assert d.stl_unit is None and d.max_hidden_primitives == 20000     # default cap
    n = options_from_args(p.parse_args(["a.stl", "out", "--max-hidden-primitives", "none"]), p)
    assert n.max_hidden_primitives is None                            # explicit: no cap
    with pytest.raises(SystemExit):
        p.parse_args(["a.stl", "out", "--max-hidden-primitives", "-3"])
    with pytest.raises(ValueError):
        _opts(stl_unit="furlong").validate()
    with pytest.raises(ValueError):
        _opts(max_hidden_primitives=-1).validate()


# ------------------------------------------- F6 review round: follow-ups --

@needs_ocp
def test_issue17_progress_snapshots_carry_the_stage_timings(plate_files, tmp_path,
                                                            monkeypatch):
    """A killed child's report can only keep timings the snapshots carry."""
    import src.drawing.pipeline as pl
    from src.drawing.report import PROGRESS_ENV
    monkeypatch.setenv(PROGRESS_ENV, str(tmp_path / "progress.json"))
    snaps = []
    real = pl.write_progress

    def spy(path, snap):
        snaps.append(json.loads(json.dumps(snap)))
        return real(path, snap)

    monkeypatch.setattr(pl, "write_progress", spy)
    code, rep = pl.run(plate_files["step"], str(tmp_path / "o"),
                       _opts(formats=("svg",), views=("front", "top")))
    assert rep["ok"] is True, rep["error"]
    by_stage = {}
    for s in snaps:
        by_stage.setdefault((s["stage"], s["view"]), s)
    sec = by_stage[("section", None)]["timing"]
    assert sec["load_s"] is not None and sec["load_s"] > 0.0
    assert sec["views_s"] is not None and sec["views_s"] > 0.0
    # the second view's snapshot already counts the first view's time (live)
    assert by_stage[("views", "top")]["timing"]["views_s"] > 0.0
    assert by_stage[("views", "front")]["timing"]["views_s"] < \
        by_stage[("views", "top")]["timing"]["views_s"]
    lay = by_stage[("layout", None)]["timing"]
    assert lay["section_s"] is not None and lay["section_s"] > 0.0
    # the final report keeps the same totals (no double counting of views_s)
    assert rep["timing"]["views_s"] == pytest.approx(sum(v["seconds"] for v in rep["views"]),
                                                     abs=0.05)


@needs_ocp
def test_issue5_views_tiny_on_a_metre_scale_part(tmp_path):
    import trimesh
    from src.drawing.pipeline import run
    tiny = str(tmp_path / "tiny.stl")
    trimesh.creation.icosphere(subdivisions=3, radius=0.08).export(tiny)
    code, rep = run(tiny, str(tmp_path / "a"), _opts(formats=("svg",)))
    assert rep["ok"] is True, rep["error"]
    assert rep["sheet"]["scale"] == "50:1"              # precondition: the largest scale
    codes = _codes(rep)
    assert codes.get("views_tiny") == "warning"
    detail = next(f["detail"] for f in rep["quality"]["flags"] if f["code"] == "views_tiny")
    assert "check the unit" in detail and "50:1" in detail
    assert codes.get("unit_suspect") == "warning", codes   # the unit check says the same
    # control: a normal part is not tiny
    box = _box_stl(tmp_path / "box.stl", (50, 30, 20))
    _, rep2 = run(box, str(tmp_path / "b"), _opts(formats=("svg",)))
    assert rep2["ok"] is True and "views_tiny" not in _codes(rep2)
    # control: 2.5 mm at 50:1 fills more than 10 % of the A3 frame -> not tiny
    mid = _box_stl(tmp_path / "mid.stl", (2.5, 1.5, 1.0))
    _, rep3 = run(mid, str(tmp_path / "c"), _opts(formats=("svg",)))
    assert rep3["sheet"]["scale"] == "50:1"             # precondition: same scale
    assert "views_tiny" not in _codes(rep3)


@needs_ocp
def test_inverted_verdict_of_an_inconsistently_wound_mesh_is_not_inside_out(tmp_path):
    import trimesh
    from src.drawing.mesh_load import mesh_census
    from src.drawing.pipeline import run
    b = trimesh.creation.box(extents=(40, 20, 10))
    f = b.faces[:, ::-1].copy()                          # inside out ...
    f[:2] = f[:2, ::-1]                                  # ... except one face: conflicts
    mesh = trimesh.Trimesh(b.vertices, f, process=False)
    c = mesh_census(mesh)
    assert c["inverted"] and c["winding_conflicts"] > 0 and c["boundary_edges"] == 0  # precondition
    p = str(tmp_path / "wound.stl")
    mesh.export(p)
    _, rep = run(p, str(tmp_path / "o"), _opts(formats=("svg",), stl_to_step="off",
                                               section="off", views=("front",)))
    assert rep["ok"] is True, rep["error"]
    assert "inverted_solid" not in _codes(rep)
    assert not any(w.startswith("mesh is inside out") for w in rep["warnings"])
    assert any("winding is inconsistent" in w and str(c["winding_conflicts"]) in w
               for w in rep["warnings"]), rep["warnings"]
    # control: the fully inverted (consistently wound) box keeps inverted_solid
    q = str(tmp_path / "inv.stl")
    trimesh.Trimesh(b.vertices, b.faces[:, ::-1], process=False).export(q)
    _, rep2 = run(q, str(tmp_path / "o2"), _opts(formats=("svg",), stl_to_step="off",
                                                 section="off", views=("front",)))
    assert _codes(rep2).get("inverted_solid") == "warning"
    assert not any("winding is inconsistent" in w for w in rep2["warnings"])


@needs_ocp
def test_issue10_thin_part_section_is_flagged_as_fallback(tmp_path):
    from src.drawing.pipeline import run
    plate = _write_step(_box_shape(100, 80, 3), tmp_path / "thin.step")
    _, rep = run(plate, str(tmp_path / "a"), _opts(formats=("svg",), views=("front", "top")))
    assert rep["ok"] is True, rep["error"]
    assert rep["section"]["reason"] == "thin_part"      # precondition: no hole, thin
    fl = [f for f in rep["quality"]["flags"] if f["code"] == "section_fallback"]
    assert len(fl) == 1 and fl[0]["severity"] == "info" and "thin" in fl[0]["detail"]


@needs_ocp
def test_issue17_parent_sniffs_the_step_unit_when_the_child_left_no_progress(tmp_path,
                                                                             monkeypatch):
    import src.drawing.sandbox as sb
    from src.drawing import cli
    step = _write_step_unit(_box_shape(4, 2, 1), tmp_path / "inch.step", "INCH")

    def fake_run(argv, work_dir, **kw):              # killed during the imports
        box = {"isolation": "job_object", "timed_out": True, "killed": True,
               "memory_exceeded": False, "cpu_exceeded": False, "output_exceeded": False,
               "timeout_s": kw.get("timeout_s"), "wall_s": 15.3, "exit_code": 1,
               "result_error": "no RESULT line", "progress_error": None, "last_stage": None}
        return sb.SandboxResult(exit_code=1, result=None, sandbox=box, stdout_tail="",
                                stderr_tail="", progress=None)

    monkeypatch.setattr(sb, "run_sandboxed", fake_run)
    code, res = cli._run_sandboxed(step, str(tmp_path / "out"), _opts(timeout_s=15.0),
                                   sys.stderr)
    assert code == 1 and res["quality"]["last_stage"] is None     # precondition: no progress
    assert res["input"]["detected_format"] == "step"
    assert res["input"]["source_unit"] == "inch"
    # an STL input gets no STEP sniff: the unit stays unknown
    code2, res2 = cli._run_sandboxed(_part_stl(tmp_path), str(tmp_path / "out2"),
                                     _opts(timeout_s=15.0), sys.stderr)
    assert code2 == 1 and res2["input"]["source_unit"] in (None, "unknown")


@needs_ocp
def test_issue3_oriented_bbox_maps_the_file_box_without_a_second_exact_bbox(plate_files,
                                                                           tmp_path,
                                                                           monkeypatch):
    import src.drawing.brep_load as bl
    from src.drawing.brep_load import exact_bbox, load_step, transform_shape
    from src.drawing.pipeline import _permuted_box, run
    calls = []
    real = bl.exact_bbox

    def counting(shape, *a, **k):
        calls.append(1)
        return real(shape, *a, **k)

    monkeypatch.setattr(bl, "exact_bbox", counting)
    _, rep = run(plate_files["step"], str(tmp_path / "o"),
                 _opts(formats=("svg",), section="off", views=("front",)))
    assert rep["ok"] is True, rep["error"]
    o = rep["route"]["orientation"]
    assert o["permutation"] != [0, 1, 2] or o["flips"] != [1, 1, 1]   # precondition: rotated
    assert len(calls) == 1                     # the load's box only, none after the turn
    # the mapped box equals the exact box of the rotated shape
    monkeypatch.setattr(bl, "exact_bbox", real)
    m = load_step(plate_files["step"])
    M = np.zeros((3, 3))
    for i, (p, f) in enumerate(zip(o["permutation"], o["flips"])):
        M[i, p] = f
    lo_x, hi_x = exact_bbox(transform_shape(m.shape, M))
    assert rep["geometry"]["drawing_bbox_min"] == pytest.approx(list(lo_x), abs=1e-6)
    assert rep["geometry"]["drawing_bbox_max"] == pytest.approx(list(hi_x), abs=1e-6)
    # a general rotation is not a permutation: the caller must measure
    c, s = np.cos(0.3), np.sin(0.3)
    assert _permuted_box(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]), [0, 0, 0],
                         [1, 1, 1]) is None
