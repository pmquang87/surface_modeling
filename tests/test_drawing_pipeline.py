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
    # R5: the report names the deciding rule of choose_depth_sign_mesh (QA round 2 issue 4)
    assert o["depth_method"] in ("mesh_cavity", "mesh_feature_edges")
    assert o["depth_flipped"] is True and o["flips"] == [-1, -1, 1]
    # mirror control: bosses already on the viewer's side -> no turn (an
    # always-flip scorer fails here)
    p2 = str(tmp_path / "near.stl")
    _tessellate(_plate_with_bosses(-1), deflection=0.2).export(p2)
    _, rep2 = run(p2, str(tmp_path / "o2"), _opts(formats=("svg",), section="off",
                                                  views=("front",), stl_to_step="off"))
    assert rep2["ok"] is True, rep2["error"]
    o2 = rep2["route"]["orientation"]
    assert o2["permutation"] == [0, 1, 2] and o2["depth_method"] == o["depth_method"]
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
    # QA round 3 issue 16: the mesh route also discloses the features it cannot detect
    assert rep["undimensioned"] == [{"kind": "overall", "reason": "not_a_solid", "view": None},
                                    {"kind": "features", "count": None,
                                     "reason": "route_without_brep_features", "view": None,
                                     "route": "stl_mesh"}]
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


# ------------------------------------------- F7-perf: large assemblies, deadline --

def _plate_row_step(path, n):
    """STEP of ``n`` located copies of the plate in a row along X (one root)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Trsf, gp_Vec
    from src.io.exporters import export_step
    plate = _plate()
    b = BRep_Builder()
    comp = TopoDS_Compound()
    b.MakeCompound(comp)
    for i in range(n):
        t = gp_Trsf()
        t.SetTranslation(gp_Vec(120.0 * i, 0.0, 0.0))
        b.Add(comp, BRepBuilderAPI_Transform(plate, t, False).Shape())
    export_step(comp, str(path))
    return str(path)


@pytest.fixture(scope="module")
def plate_row_25(tmp_path_factory):
    if not HAS_OCP:
        pytest.skip("cadquery-ocp not installed")
    return _plate_row_step(tmp_path_factory.mktemp("row25") / "row25.step", 25)


@needs_ocp
def test_f7_assembly_drawn_without_hidden_lines_unless_forced(plate_row_25, tmp_path):
    """Item C: 25 solids (> 20) -> no hidden lines, flag hidden_lines_omitted_assembly
    (info), level reduced; --hidden-lines-force keeps them."""
    from src.drawing.pipeline import run
    o = dict(formats=("svg",), section="off", views=("top", "iso"))
    code, rep = run(plate_row_25, str(tmp_path / "a"), _opts(**o))
    assert rep["ok"] is True, rep["error"]
    assert rep["geometry"]["solids"] == 25                       # precondition: the assembly
    assert _codes(rep).get("hidden_lines_omitted_assembly") == "info"
    top = {v["name"]: v for v in rep["views"]}["top"]
    assert top["hidden"] == 0 and top["hidden_omitted"] == "assembly"
    assert "hidden_lines_dropped" not in _codes(rep)
    assert rep["quality"]["level"] != "full"
    # control: forced, the same top view has the hidden hole walls and no flag
    _, rep2 = run(plate_row_25, str(tmp_path / "b"), _opts(hidden_lines_force=True, **o))
    assert rep2["ok"] is True, rep2["error"]
    top2 = {v["name"]: v for v in rep2["views"]}["top"]
    assert top2["hidden"] > 0 and "hidden_omitted" not in top2
    assert "hidden_lines_omitted_assembly" not in _codes(rep2)


@needs_ocp
def test_f7_raw_hidden_edge_cap_omits_hidden_lines(plate_files, tmp_path, monkeypatch):
    """Item C, second trigger: a view with more raw hidden HLR edges than
    ``_MAX_HIDDEN_EDGES`` is drawn without them (info flag, not the warning)."""
    import src.drawing.pipeline as pl
    o = dict(formats=("svg",), section="off", views=("top",))
    _, ref = pl.run(plate_files["step"], str(tmp_path / "a"), _opts(**o))
    assert ref["views"][0]["hidden"] > 0                          # precondition
    assert "hidden_lines_omitted_assembly" not in _codes(ref)
    monkeypatch.setattr(pl, "_MAX_HIDDEN_EDGES", 1)
    _, rep = pl.run(plate_files["step"], str(tmp_path / "b"), _opts(**o))
    assert rep["views"][0]["hidden"] == 0
    assert rep["views"][0]["hidden_dropped"] == "max_hidden_edges"
    assert _codes(rep).get("hidden_lines_omitted_assembly") == "info"
    assert "hidden_lines_dropped" not in _codes(rep)
    _, forced = pl.run(plate_files["step"], str(tmp_path / "c"),
                       _opts(hidden_lines_force=True, **o))
    assert forced["views"][0]["hidden"] == ref["views"][0]["hidden"]


@needs_ocp
def test_f7_run_deadline_keeps_the_drawing(plate_files, tmp_path, monkeypatch):
    """Item D: with the run deadline already passed, every view is still drawn
    (visible lines; hidden dropped -> hidden_lines_dropped), the section is
    skipped (section_skipped_budget) and the DXF is written without its audit."""
    import src.drawing.pipeline as pl
    o = dict(formats=("svg", "dxf"), views=("top", "right"))
    code0, ref = pl.run(plate_files["step"], str(tmp_path / "a"), _opts(**o))
    # precondition: with the normal deadline nothing is dropped or skipped
    assert ref["ok"] is True and ref["section"] is not None and ref["section"]["loops"] > 0
    assert all(v["hidden"] > 0 for v in ref["views"])
    assert not {"hidden_lines_dropped", "section_skipped_budget"} & set(_codes(ref))
    assert ref["writers"]["dxf_audited"] is True
    monkeypatch.setattr(pl, "_DEADLINE_FRACTION", 1e-9)     # deadline = run start
    monkeypatch.setattr(pl, "_MIN_VIEW_BUDGET_S", 1e-9)      # tiny time_budget_s per view
    code, rep = pl.run(plate_files["step"], str(tmp_path / "b"), _opts(**o))
    assert rep["ok"] is True, rep["error"]
    assert (tmp_path / "b" / "plate.svg").is_file() and (tmp_path / "b" / "plate.dxf").is_file()
    views = {v["name"]: v for v in rep["views"]}
    assert set(views) == {"top", "right"}
    for name, v in views.items():
        # the visible lines are all there (past the deadline they may stay unmerged)
        assert v["visible"] >= {w["name"]: w for w in ref["views"]}[name]["visible"] > 0, name
        assert v["hidden"] == 0 and v["hidden_dropped"] == "time_budget", name
    assert _codes(rep).get("hidden_lines_dropped") == "warning"
    assert _codes(rep).get("section_skipped_budget") == "warning"
    assert rep["section"] is None
    assert rep["writers"]["dxf_audited"] is False
    assert "deadline" in rep["writers"]["dxf_audit_skipped_reason"]
    assert code == 2


@needs_ocp
def test_f7_dxf_audit_gated_by_sheet_size(plate_files, tmp_path, monkeypatch):
    """Item E: the DXF re-read audit runs for a normal sheet and is skipped above
    ``_DXF_AUDIT_MAX_ITEMS`` sheet items (report writers.dxf_audited)."""
    import src.drawing.pipeline as pl
    import src.drawing.writers as wr
    o = dict(formats=("dxf",), section="off", views=("front",))
    calls = []
    real = wr.write_dxf

    def spy(sheet, path, *a, **k):
        calls.append(k.get("audit"))
        return real(sheet, path, *a, **k)
    monkeypatch.setattr(wr, "write_dxf", spy)
    _, rep = pl.run(plate_files["step"], str(tmp_path / "a"), _opts(**o))
    assert rep["ok"] is True, rep["error"]
    assert calls == [True] and rep["writers"]["dxf_audited"] is True
    n_items = rep["writers"]["dxf_items"]
    assert n_items > 0
    monkeypatch.setattr(pl, "_DXF_AUDIT_MAX_ITEMS", n_items - 1)
    _, rep2 = pl.run(plate_files["step"], str(tmp_path / "b"), _opts(**o))
    assert calls[-1] is False and rep2["writers"]["dxf_audited"] is False
    assert "sheet items" in rep2["writers"]["dxf_audit_skipped_reason"]
    assert (tmp_path / "b" / "plate.dxf").stat().st_size > 0


@needs_ocp
def test_f7_write_dxf_audit_switch(tmp_path, monkeypatch):
    """``write_dxf(audit=False)`` does not re-read the file; the default does."""
    import ezdxf
    from src.drawing.model import LineStyle, Segment, Sheet
    from src.drawing.writers.dxf import write_dxf
    sheet = Sheet("A3", 420.0, 297.0, items=[Segment((10, 10), (100, 50), LineStyle.VISIBLE)])
    reads = []
    real = ezdxf.readfile

    def spy(*a, **k):
        reads.append(1)
        return real(*a, **k)
    monkeypatch.setattr(ezdxf, "readfile", spy)
    on = write_dxf(sheet, str(tmp_path / "on.dxf"))
    assert reads == [1] and on["audited"] is True and on["audit_errors"] == 0
    off = write_dxf(sheet, str(tmp_path / "off.dxf"), audit=False)
    assert reads == [1] and off["audited"] is False and off["audit_errors"] is None
    assert off["entities"] == on["entities"] and off["bytes"] > 0


@needs_ocp
def test_f7_stage_lines_on_stderr_under_quiet(plate_files, tmp_path):
    """Item G: --quiet keeps one 'stage' line per stage boundary on stderr; stdout
    is still exactly the RESULT line."""
    proc = _cli([plate_files["step"], str(tmp_path / "o"), "--no-sandbox", "--quiet",
                 "--formats", "svg"])
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1 and lines[0].startswith("RESULT "), proc.stdout
    stage = [ln for ln in proc.stderr.splitlines() if "src.drawing.stage" in ln]
    text = "\n".join(stage)
    for s in ("stage sniff", "stage load", "stage views view front", "stage section",
              "stage write", "stage done"):
        assert s in text, (s, proc.stderr[-2000:])
    assert all("elapsed" in ln and "rss" in ln for ln in stage)
    # nothing else of the progress chatter under --quiet
    assert "[drawing]" not in proc.stderr


# ================================================= QA round 2 (R5 integration) ==

def _plate_row_compound(n, pitch=120.0):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf, gp_Vec
    plate = _plate()
    parts = []
    for i in range(n):
        t = gp_Trsf()
        t.SetTranslation(gp_Vec(pitch * i, 0.0, 0.0))
        parts.append(BRepBuilderAPI_Transform(plate, t, False).Shape())
    return _compound(parts)


@pytest.fixture(scope="module")
def plate_row_3(tmp_path_factory):
    if not HAS_OCP:
        pytest.skip("cadquery-ocp not installed")
    return _write_step(_plate_row_compound(3), tmp_path_factory.mktemp("row3") / "row3.step")


@needs_ocp
def test_r2_issue5_11_assembly_has_no_hole_dims_and_no_auto_section(plate_row_3, tmp_path,
                                                                     monkeypatch):
    """Issue 5: an assembly above _ASSEMBLY_MIN_SOLIDS gets no hole callouts and no
    position dims (undimensioned reason 'assembly', info flag); issue 11: no auto
    section (info flag) unless --section names a plane; issue 16: n_bodies reaches
    the planner (body_extent / body_position entries)."""
    import src.drawing.pipeline as pl
    o = dict(formats=("svg",), views=("front", "top"))
    _, ref = pl.run(plate_row_3, str(tmp_path / "ref"), _opts(**o))
    assert ref["ok"] is True, ref["error"]
    # precondition: 3 solids below the threshold get callouts, positions and a section
    assert ref["geometry"]["solids"] == 3
    assert any(d["kind"] == "diameter" for d in ref["dimensions"])
    assert ref["section"] is not None and ref["section"]["loops"] > 0
    kinds = {u["kind"] for u in ref["undimensioned"]}
    assert {"body_extent", "body_position"} <= kinds, ref["undimensioned"]
    monkeypatch.setattr(pl, "_ASSEMBLY_MIN_SOLIDS", 2)
    code, rep = pl.run(plate_row_3, str(tmp_path / "asm"), _opts(**o))
    assert rep["ok"] is True, rep["error"]
    assert rep["dimensions"] and all(d["kind"] == "linear" for d in rep["dimensions"])
    ext = sorted(round(v, 3) for v in rep["geometry"]["extents"])
    assert all(round(d["value_mm"], 3) in ext for d in rep["dimensions"]), rep["dimensions"]
    asm = [u for u in rep["undimensioned"] if u.get("reason") == "assembly"]
    assert asm and all(u["kind"] == "hole_group" for u in asm)
    assert sum(int(u["count"]) for u in asm) == 3 * 5            # 3 plates x 5 holes
    codes = _codes(rep)
    assert codes.get("dimensions_omitted_assembly") == "info"
    assert codes.get("section_skipped_assembly") == "info"
    assert rep["section"] is None
    # an explicit plane still draws the section of the assembly
    _, rep2 = pl.run(plate_row_3, str(tmp_path / "asm_front"), _opts(section="front", **o))
    assert rep2["ok"] is True, rep2["error"]
    assert rep2["section"] is not None and rep2["section"]["loops"] > 0
    assert "section_skipped_assembly" not in _codes(rep2)
    assert _codes(rep2).get("dimensions_omitted_assembly") == "info"


@needs_ocp
def test_r2_issue11_section_skipped_when_its_estimated_cost_exceeds_the_budget(
        plate_files, tmp_path, monkeypatch):
    import src.drawing.pipeline as pl
    o = dict(formats=("svg",), views=("front", "top"))
    _, ref = pl.run(plate_files["step"], str(tmp_path / "a"), _opts(**o))
    assert ref["section"] is not None and "section_skipped_budget" not in _codes(ref)
    # the estimate is _SECTION_COST_VIEWS x the mean view time: make it absurd
    monkeypatch.setattr(pl, "_SECTION_COST_VIEWS", 1e9)
    code, rep = pl.run(plate_files["step"], str(tmp_path / "b"), _opts(**o))
    assert rep["ok"] is True, rep["error"]
    assert rep["section"] is None
    fl = [f for f in rep["quality"]["flags"] if f["code"] == "section_skipped_budget"]
    assert len(fl) == 1 and "estimated" in fl[0]["detail"]
    assert code == 2


def _inverted_sphere_one_face_back():
    import trimesh
    s = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
    f = s.faces[:, ::-1].copy()                      # inside out ...
    f[0] = f[0, ::-1]                                # ... except one triangle: 3 conflicts
    return trimesh.Trimesh(s.vertices, f, process=False)


@needs_ocp
def test_r2_issue8_inverted_mesh_with_a_few_conflicts_is_flagged_once(tmp_path):
    from src.drawing.mesh_load import mesh_census
    from src.drawing.pipeline import run
    mesh = _inverted_sphere_one_face_back()
    c = mesh_census(mesh)
    # precondition: the old rule (conflicts == 0) would not flag this mesh
    assert c["inverted"] and c["winding_conflicts"] > 0 and c["inverted_solid"]
    assert c["volume_raw_signed_mm3"] < 0
    p = str(tmp_path / "inv.stl")
    mesh.export(p)
    _, rep = run(p, str(tmp_path / "o"), _opts(formats=("svg",), stl_to_step="off",
                                               section="off", views=("front",)))
    assert rep["ok"] is True, rep["error"]
    fl = {f["code"]: f for f in rep["quality"]["flags"]}
    assert fl["inverted_solid"]["severity"] == "warning"
    assert "-3" in fl["inverted_solid"]["detail"]           # the signed raw sum, -33174
    assert fl["winding_repaired"]["severity"] == "info"
    assert str(c["winding_conflicts"]) in fl["winding_repaired"]["detail"]
    winding = [w for w in rep["warnings"] if "winding" in w]
    assert len(winding) <= 1, rep["warnings"]
    assert not any("winding is inconsistent" in w for w in rep["warnings"])


def test_r2_issue10_text_collision_is_a_warning_moves_are_info(tmp_path):
    from src.drawing.report import ESCALATABLE_FLAGS, FLAGS, Report
    assert FLAGS["layout_text_collision"] == "warning"
    assert FLAGS["layout_text_moved"] == "info" and FLAGS["layout_dims_dropped"] == "info"
    assert "section_gaps_bridged" in ESCALATABLE_FLAGS
    rep = Report(str(tmp_path / "p.step"), str(tmp_path), _opts())
    rep.flag("layout_text_moved", "info", "2 texts moved")
    assert rep.finish(True)[0] == 0
    rep2 = Report(str(tmp_path / "p.step"), str(tmp_path), _opts())
    rep2.flag("layout_text_collision", None, "1 left")
    assert rep2.finish(True)[0] == 2


@needs_ocp
def test_r2_issue21_section_fill_reported_and_blackened_flag(plate_files, tmp_path):
    from src.drawing.pipeline import run
    thin = _write_step(_box_shape(200, 150, 1), tmp_path / "thin.step")
    _, rep = run(thin, str(tmp_path / "a"), _opts(formats=("svg",), views=("front", "top")))
    assert rep["ok"] is True, rep["error"]
    s = rep["section"]
    # precondition: a 1 mm wall at 1:1 or smaller is under the 2.1 mm fill limit
    assert s is not None and s["loops"] > 0 and rep["sheet"]["scale"] in ("1:1", "1:2"), \
        rep["sheet"]["scale"]
    assert s["fill"] == "solid" and s["filled_regions"] >= 1
    assert s["filled_area_fraction"] > 0.5
    # review R5: a 1 mm plate is thin-walled by design (filled at 1:1 too): no flag
    assert s["thin_walled_by_design"] is True and max(s["body_thickness_model_mm"]) < 2.1
    assert "section_blackened" not in _codes(rep)
    # a 4 mm wall at 1:5 (NIST FTC 07 shape of the problem): the scale blackened it
    wall = _write_step(_box_shape(800, 600, 4), tmp_path / "wall.step")
    _, rw = run(wall, str(tmp_path / "w"), _opts(formats=("svg",), views=("front", "top")))
    assert rw["ok"] is True, rw["error"]
    sw = rw["section"]
    assert rw["sheet"]["scale"] == "1:5" and sw["fill"] == "solid", (rw["sheet"], sw)
    assert sw["body_thickness_paper_mm"][0] == pytest.approx(0.8, abs=0.01)
    assert sw["body_thickness_model_mm"][0] == pytest.approx(4.0, abs=0.05)
    assert sw["thin_walled_by_design"] is False
    assert _codes(rw).get("section_blackened") == "info"
    # control: the 15 mm plate is hatched and not flagged
    _, ref = run(plate_files["step"], str(tmp_path / "b"), _opts(formats=("svg",),
                                                                 views=("front", "top")))
    assert ref["section"]["fill"] == "hatch" and ref["section"]["filled_area_fraction"] == 0.0
    assert ref["section"]["hatch_segments"] > 0
    assert "section_blackened" not in _codes(ref)


@needs_ocp
def test_r2_issue21_bridged_mesh_section_gaps_are_flagged(tmp_path):
    import trimesh
    from src.drawing.pipeline import run
    b = trimesh.creation.box(extents=(40.0, 20.0, 30.0))
    keep = ~(b.triangles_center[:, 0] > 19.9)                     # drop the +X face
    open_box = trimesh.Trimesh(b.vertices, b.faces[keep], process=False)
    assert not open_box.is_watertight                             # precondition
    p = str(tmp_path / "open.stl")
    open_box.export(p)
    o = _opts(formats=("svg",), stl_to_step="off", views=("front", "top"))
    _, rep = run(p, str(tmp_path / "o"), o)
    assert rep["ok"] is True, rep["error"]
    assert rep["section"]["bridged_gaps"] >= 1
    fl = [f for f in rep["quality"]["flags"] if f["code"] == "section_gaps_bridged"]
    # a 20-30 mm chord on a 40 mm section: far above 5 % of the extent -> warning
    assert len(fl) == 1 and fl[0]["severity"] == "warning", fl
    # control: the closed box bridges nothing
    q = str(tmp_path / "closed.stl")
    b.export(q)
    _, rep2 = run(q, str(tmp_path / "o2"), o)
    assert rep2["section"]["bridged_gaps"] == 0
    assert "section_gaps_bridged" not in _codes(rep2)


_TESS_EXTRA_R5 = (
    "#30=COORDINATES_LIST('',2,((0.0,0.0,0.0),(1.0,0.0,0.0)));\n"
    "#31=TESSELLATED_CURVE_SET('',#30,((1,2)));\n"
    "#32=TESSELLATED_GEOMETRIC_SET('flatness',(#31));\n"
    "#34=PRESENTATION_STYLE_ASSIGNMENT(());\n"
    "#35=TESSELLATED_ANNOTATION_OCCURRENCE('Flatness.1',(#34),#32);\n"
    "#40=TRIANGULATED_FACE('',#20,8,(),$,(1,2,3,4,5,6,7,8),((1,2,3)));\n"
    "#41=TESSELLATED_SHELL('supp',(#40),$);\n"
    "#42=CONSTRUCTIVE_GEOMETRY_REPRESENTATION('supplemental geometry',(#41),#14);\n")


@needs_ocp
def test_r2_issue24_tessellated_route_reports_what_it_leaves_out(tmp_path):
    from src.drawing.pipeline import run
    plain = tmp_path / "tess.stp"
    plain.write_text(_tess_step_text(), encoding="ascii")
    _, ref = run(str(plain), str(tmp_path / "a"), _opts(formats=("svg",), views=("front",)))
    assert ref["ok"] is True, ref["error"]
    # control: nothing is left out of the plain file
    assert not {"supplemental_shells_ignored", "loose_curves_ignored"} & set(_codes(ref))
    assert ref["route"]["stl_to_step"]["skipped_reason"] == "tessellated_step"
    assert ref["geometry"]["bodies"] is not None
    p = tmp_path / "tess_extra.stp"
    p.write_text(_tess_step_text().replace("ENDSEC;\nEND-ISO", _TESS_EXTRA_R5 + "ENDSEC;\nEND-ISO"),
                 encoding="ascii")
    _, rep = run(str(p), str(tmp_path / "b"), _opts(formats=("svg",), views=("front",)))
    assert rep["ok"] is True, rep["error"]
    assert rep["route"]["source"] == "step_tessellated_mesh"      # precondition
    codes = _codes(rep)
    assert codes.get("supplemental_shells_ignored") == "info"
    assert codes.get("loose_curves_ignored") == "info"
    dropped = rep["geometry"]["bodies"]["dropped"]
    assert dropped.get("tessellated_shells_outside") == 1
    assert dropped.get("annotation_occurrences") == 1
    assert rep["route"]["stl_to_step"]["skipped_reason"] == "tessellated_step"


@needs_ocp
def test_r2_issue30_unit_note_on_the_sheet_when_the_unit_is_doubtful(plate_files, tmp_path):
    from src.drawing.pipeline import run
    inch = _box_stl(tmp_path / "inch_part.stl", (5.0, 2.5, 1.375))
    o = _opts(formats=("svg",), views=("front", "top"), stl_to_step="off")
    _, rep = run(inch, str(tmp_path / "a"), o)
    assert rep["ok"] is True, rep["error"]
    assert _codes(rep).get("unit_suspect") == "warning"            # precondition
    svg = (tmp_path / "a" / "inch_part.svg").read_text(encoding="utf-8")
    assert "unit assumed" in svg
    assert "unit assumed" in (rep["sheet"].get("unit_note") or "")
    # control: a plain mm STEP keeps the plain note
    _, ref = run(plate_files["step"], str(tmp_path / "b"), _opts(formats=("svg",),
                                                                 views=("front",)))
    svg2 = (tmp_path / "b" / "plate.svg").read_text(encoding="utf-8")
    assert "Dimensions in mm" in svg2 and "unit assumed" not in svg2
    assert ref["sheet"].get("unit_note") is None


@needs_ocp
def test_r2_view_entries_carry_the_clean_up_counts_and_writers_dash_adapted(plate_files,
                                                                             tmp_path):
    from src.drawing.pipeline import run
    _, rep = run(plate_files["step"], str(tmp_path / "a"), _opts(formats=("svg", "pdf", "dxf"),
                                                                 views=("front", "top")))
    assert rep["ok"] is True, rep["error"]
    for v in rep["views"]:
        assert isinstance(v.get("hidden_curves_dropped"), int), v
        assert isinstance(v.get("hidden_curves_trimmed"), int), v
    da = rep["writers"]["dash_adapted"]
    assert set(da) == {"svg", "pdf", "dxf"} and all(isinstance(n, int) for n in da.values())
    blob = _blob_stl(str(tmp_path / "blob.stl"))
    _, rep2 = run(blob, str(tmp_path / "b"), _opts(formats=("svg",), views=("front",),
                                                   section="off"))
    assert rep2["route"]["pipeline"] == "mesh"                     # precondition
    v = rep2["views"][0]
    for key in ("welded_ends", "hairpins_removed", "closed_loops_dropped", "dropped_chains",
                "n_sliver_pairs"):
        assert isinstance(v.get(key), int), (key, v)


def _pocket_box(pocket_side):
    """150 x 30 x 75 box, a 100 x 15 x 50 pocket opening to -Y (side -1) or +Y (+1), and
    twelve 3 x 1 x 3 notches on the opposite face (more, smaller faces there)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    shape = _box_shape(150, 30, 75)
    y_pocket = 0.0 if pocket_side < 0 else 15.0
    shape = BRepAlgoAPI_Cut(shape, _box_shape(100, 15, 50, (25, y_pocket, 12.5))).Shape()
    y_notch = 29.0 if pocket_side < 0 else 0.0
    for i in range(6):
        for z in (10.0, 60.0):
            shape = BRepAlgoAPI_Cut(shape, _box_shape(3, 1, 3, (15 + 22 * i, y_notch, z))).Shape()
    return shape


@needs_ocp
def test_r2_issue4_pipeline_depth_sign_follows_the_open_pocket(tmp_path):
    from src.drawing.brep_features import score_depth_sign
    from src.drawing.pipeline import run
    near = _pocket_box(-1)
    plus, minus = score_depth_sign(near, (0.0, -1.0, 0.0))
    assert minus > 1.25 * plus                     # precondition: the face count would turn it
    o = _opts(formats=("svg",), section="off", views=("front",))
    _, rn = run(_write_step(near, tmp_path / "near.step"), str(tmp_path / "a"), o)
    _, rf = run(_write_step(_pocket_box(+1), tmp_path / "far.step"), str(tmp_path / "b"), o)
    assert rn["ok"] and rf["ok"], (rn["error"], rf["error"])
    on, of = rn["route"]["orientation"], rf["route"]["orientation"]
    assert on["permutation"] == of["permutation"] == [0, 1, 2]
    assert on["depth_method"] == of["depth_method"] == "brep_cavity"
    assert on["depth_flipped"] is False and of["depth_flipped"] is True
    for key in ("cavity_kept", "cavity_flipped", "mean_recess_kept", "housing"):
        assert key in on["depth_scores"], on["depth_scores"]


@needs_ocp
def test_r2_hidden_only_holes_are_flagged(tmp_path):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from src.drawing.pipeline import run
    box = _box_shape(50, 20, 30)
    hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(25, 21, 15), gp_Dir(0, -1, 0)), 2.5, 11.0).Shape()
    p = _write_step(BRepAlgoAPI_Cut(box, hole).Shape(), tmp_path / "blind_back.step")
    _, rep = run(p, str(tmp_path / "o"), _opts(formats=("svg",), section="off", views=("front",),
                                                orient="fixed"))
    assert rep["ok"] is True, rep["error"]
    assert rep["geometry"]["features"]["holes"] == 1                # precondition
    assert not any(d["kind"] == "diameter" for d in rep["dimensions"])
    fl = [f for f in rep["quality"]["flags"] if f["code"] == "dimension_hidden_only"]
    assert len(fl) == 1 and fl[0]["severity"] == "info"


def test_r2_issue12_docs_list_every_flag_with_its_severity():
    from src.drawing.report import FLAG_MEANINGS, FLAGS, LEVELS, flag_table_markdown
    assert set(FLAG_MEANINGS) == set(FLAGS)
    assert not [c for c, m in FLAG_MEANINGS.items() if "|" in m or not m]   # table-safe
    path = os.path.join(REPO, "docs", "drawing_module.md")
    text = open(path, encoding="utf-8").read()
    missing = [c for c in FLAGS if f"`{c}`" not in text]
    assert not missing, missing
    # the table in the doc is the generated one (severity, escalation, meaning)
    assert flag_table_markdown() in text
    for level in LEVELS:
        assert f"`{level}`" in text, level
    for value in ("assumed_mm", "stl_binary_truncated", "step_tessellated_mesh",
                  "--stl-unit", "--hidden-lines-force", "--max-hidden-primitives",
                  "tessellated_step", "HIDDEN_ISO"):
        assert value in text, value
    spec = open(os.path.join(REPO, "docs", "superpowers", "specs",
                             "2026-10-17-drawing-module-design.md"), encoding="utf-8").read()
    s38 = spec.split("### 3.8", 1)[1].split("### 3.9", 1)[0]
    missing_spec = [c for c in FLAGS if f"`{c}`" not in s38]
    assert not missing_spec, missing_spec


# ======================================== QA round 2 (R5 review follow-up) ==

def _pinned_plate_row(n=3, pitch=120.0, pin_r=3.0):
    """n plates of _plate() in a row, each with a separate pin solid (radius pin_r)
    in its hole at (8, 8): 2 n solids, n filled holes of radius 3.25."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
    plate = _plate()
    parts = []
    for i in range(n):
        t = gp_Trsf()
        t.SetTranslation(gp_Vec(pitch * i, 0.0, 0.0))
        parts.append(BRepBuilderAPI_Transform(plate, t, False).Shape())
        parts.append(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(8 + pitch * i, 8, 0), gp_Dir(0, 0, 1)),
                                              pin_r, 15.0).Shape())
    return _compound(parts)


def test_r5_assembly_hole_groups_list_filled_holes_too():
    """Review R5: the assembly gate drops 'hole' AND 'filled_hole' features; the filled
    ones must be listed too (reason filled_by_fastener), not vanish."""
    from types import SimpleNamespace as NS
    from src.drawing.pipeline import _assembly_hole_groups
    feats = ([NS(kind="hole", radius=3.25)] * 4 + [NS(kind="filled_hole", radius=3.25)] * 3
             + [NS(kind="filled_hole", radius=5.0)] + [NS(kind="boss", radius=15.0)])
    groups = _assembly_hole_groups(feats)
    assert all(g["kind"] == "hole_group" for g in groups)
    plain = {g["radius"]: g["count"] for g in groups if g["reason"] == "assembly"}
    filled = {g["radius"]: g["count"] for g in groups if g["reason"] == "filled_by_fastener"}
    assert plain == {3.25: 4}
    assert filled == {3.25: 3, 5.0: 1}, groups
    assert sum(g["count"] for g in groups) == 8          # every hole feature disclosed once


@needs_ocp
def test_r5_assembly_filled_holes_are_disclosed_end_to_end(tmp_path, monkeypatch):
    import src.drawing.pipeline as pl
    p = _write_step(_pinned_plate_row(3), tmp_path / "pinned.step")
    monkeypatch.setattr(pl, "_ASSEMBLY_MIN_SOLIDS", 2)
    _, rep = pl.run(p, str(tmp_path / "o"), _opts(formats=("svg",), views=("front", "top")))
    assert rep["ok"] is True, rep["error"]
    feats = rep["geometry"]["features"]
    # precondition: the pins fill one hole per plate (a real filled_hole feature)
    assert rep["geometry"]["solids"] == 6
    assert feats.get("filled_holes") == 3 and feats["holes"] == 3 * 4, feats
    und = rep["undimensioned"]
    filled = [u for u in und if u.get("reason") == "filled_by_fastener"]
    asm = [u for u in und if u.get("reason") == "assembly"]
    assert sum(int(u["count"]) for u in filled) == 3, und
    assert sum(int(u["count"]) for u in asm) == 12, und
    fl = [f for f in rep["quality"]["flags"] if f["code"] == "dimensions_omitted_assembly"]
    # QA round 3 issue 27: the counts are split by reason (the old detail printed the total
    # 15 as 'reason assembly' and the 3 filled holes again after it)
    assert len(fl) == 1 and "12 hole feature(s)" in fl[0]["detail"], fl
    assert "15 hole" not in fl[0]["detail"], fl
    assert "3 in 1 group(s) filled by a fastener" in fl[0]["detail"], fl


def _section_report_for(tmp_path, sl, scale=None):
    from src.drawing.pipeline import _report_section_fill
    from src.drawing.report import Report
    rep = Report(str(tmp_path / "p.step"), str(tmp_path), _opts())
    rep.data["section"] = {"loops": 2}
    if scale is None:
        _report_section_fill(rep, sl)
    else:
        _report_section_fill(rep, sl, scale=scale)
    return rep


def test_r5_section_thickness_is_paper_mm_and_thin_by_design_is_not_flagged(tmp_path):
    """Review R5: body_thickness from the layout is 2A/P ON THE SHEET; the report keys
    say so (paper / model mm), and section_blackened is only raised when the walls
    would be hatched at 1:1 (blackened by the scale, not thin-walled by design)."""
    from src.drawing.style import section_fill_below
    limit = section_fill_below()
    sl = {"fill": "solid", "filled_regions": 1, "filled_area_fraction": 1.0, "bodies": 1,
          "body_thickness": [0.8], "hatch_segments": 0}
    # 4 mm wall drawn at 1:5 -> 0.8 mm on paper: blackened by the scale -> flagged
    rep = _section_report_for(tmp_path, sl, scale=0.2)
    sec = rep.data["section"]
    assert "body_thickness_mm" not in sec
    assert sec["body_thickness_paper_mm"] == [0.8]
    assert sec["body_thickness_model_mm"] == [4.0]
    assert 4.0 > limit and sec["thin_walled_by_design"] is False      # precondition
    assert rep.has_flag("section_blackened")
    # 0.8 mm sheet metal at 1:1: thin by design, filled whatever the scale -> no flag
    rep2 = _section_report_for(tmp_path, sl, scale=1.0)
    sec2 = rep2.data["section"]
    assert sec2["filled_area_fraction"] > 0.5 and sec2["body_thickness_model_mm"] == [0.8]
    assert sec2["thin_walled_by_design"] is True
    assert not rep2.has_flag("section_blackened")
    # no scale known: old behaviour (flag on the area share alone), no model thickness
    rep3 = _section_report_for(tmp_path, sl)
    assert "body_thickness_model_mm" not in rep3.data["section"]
    assert rep3.has_flag("section_blackened")


@needs_ocp
def test_r5_timing_has_features_dimensions_and_layout_buckets(plate_files, tmp_path):
    """Review R5: about 25 s of the RC buggy's child total sat in no timing bucket
    (feature detection, dimension planning, layout); each has its own bucket now."""
    from src.drawing.pipeline import run
    _, rep = run(plate_files["step"], str(tmp_path / "o"),
                 _opts(formats=("svg",), views=("front", "top")))
    assert rep["ok"] is True, rep["error"]
    t = rep["timing"]
    for key in ("features_s", "dimensions_s", "layout_s", "writers_s"):
        assert isinstance(t.get(key), float) and t[key] >= 0.0, (key, t)
    assert t["features_s"] > 0.0                       # cylinder_features ran on the plate
    # write_s stays the whole sheet stage: dimensions + layout + writers
    parts = t["dimensions_s"] + t["layout_s"] + t["writers_s"]
    assert abs(parts - t["write_s"]) < 0.05 + 0.02 * t["write_s"], t


# ======================================================= QA round 3 (task Z1) ==

def _u_channel(thin=2.0, thick=10.0, length=40.0, width=30.0, base=5.0, height=25.0):
    """U channel along x: base 0..base in z, a thin wall at y 0..thin and a thick wall at
    y width-thick..width, both up to z = height. ONE solid; a z plane above the base cuts it
    into two separate regions (the walls)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    s = _box_shape(length, width, height)
    gap = _box_shape(length + 2.0, width - thin - thick, height, origin=(-1.0, thin, base))
    return BRepAlgoAPI_Cut(s, gap).Shape()


@needs_ocp
def test_z1_issue1_one_hatch_spacing_per_body_and_the_report_gives_the_drawn_one(tmp_path):
    """QA round 3 issue 1: the two walls of one solid were hatched at 3 mm and t/3; the
    report gave the nominal value. Now one spacing per body, measured from the DXF HATCH."""
    import ezdxf
    from src.drawing.pipeline import run
    from src.drawing.writers.dxf import ANSI31_SPACING
    p = _write_step(_u_channel(), tmp_path / "u.step")
    _, rep = run(p, str(tmp_path / "o"), _opts(formats=("dxf",), views=("front", "top"),
                                               section="top", orient="fixed"))
    assert rep["ok"] is True, rep["error"]
    sec = rep["section"]
    assert sec["loops"] == 2 and sec.get("bodies") == 1          # precondition: 2 regions, 1 body
    doc = ezdxf.readfile(str(tmp_path / "o" / "u.dxf"))
    spacings = [round(h.dxf.pattern_scale * ANSI31_SPACING, 4)
                for h in doc.modelspace().query("HATCH") if not h.dxf.solid_fill]
    assert len(spacings) == 2, spacings
    assert len(set(spacings)) == 1, spacings                       # one part, one spacing
    assert sec["hatch_spacing_mm"] == pytest.approx(spacings[0], abs=1e-3)
    assert sec["hatch_spacings_mm"] == [pytest.approx(spacings[0], abs=1e-3)]


def test_z1_issue1_report_section_fill_takes_the_drawn_spacing(tmp_path):
    sl = {"fill": "hatch", "hatch_spacing": 1.5, "body_spacing": [3.0, None, 1.5],
          "hatch_segments": 10, "bodies": 3}
    sec = _section_report_for(tmp_path, sl, scale=1.0).data["section"]
    assert sec["hatch_spacing_mm"] == 1.5                           # the drawn minimum
    assert sec["hatch_spacings_mm"] == [3.0, None, 1.5]


def test_z1_issue16_routes_without_brep_features_disclose_the_omission(plate_files, tmp_path):
    from src.drawing.pipeline import run
    ent = {"kind": "features", "reason": "route_without_brep_features"}

    def has(rep):
        return any(all(u.get(k) == v for k, v in ent.items()) for u in rep["undimensioned"])
    _, ref = run(plate_files["step"], str(tmp_path / "s"), _opts(formats=("svg",),
                                                                views=("front", "top")))
    assert ref["ok"] is True and ref["route"]["source"] == "step"
    assert not has(ref)                                            # control: true B-rep route
    for mode, source in (("auto", "stl_faceted_step"), ("off", "stl_mesh")):
        _, rep = run(plate_files["stl"], str(tmp_path / mode),
                     _opts(formats=("svg",), views=("front", "top"), stl_to_step=mode))
        assert rep["ok"] is True, rep["error"]
        assert rep["route"]["source"] == source                    # precondition
        assert rep["geometry"]["features"]["holes"] is None        # nothing was detected
        assert has(rep), rep["undimensioned"]
        e = next(u for u in rep["undimensioned"] if u.get("kind") == "features")
        assert e.get("route") == source


def test_z1_issue25_tessellated_dropped_shell_is_counted_once():
    from src.drawing.pipeline import _tessellated_bodies
    bodies = {"solids": 1, "free_shells": 1, "free_faces": 0, "tessellated_bodies": 0,
              "drawn": "solids"}
    dropped = {"free_shells": 1, "tessellated_shells_outside": 1, "annotation_occurrences": 60}
    out = _tessellated_bodies(bodies, dropped)
    assert out["dropped"] == {"tessellated_shells_outside": 1, "annotation_occurrences": 60}
    assert out["tessellated_bodies"] == 1            # every drawn body of this route is tessellated
    # control: without the folded count the transfer's own keys stay
    out2 = _tessellated_bodies(bodies, {"free_shells": 1})
    assert out2["dropped"] == {"free_shells": 1}
    # a value the loader already filled is kept
    assert _tessellated_bodies(dict(bodies, tessellated_bodies=2), {})["tessellated_bodies"] == 2


@needs_ocp
def test_z1_issue25_tessellated_route_fills_solids_and_tessellated_bodies(tmp_path):
    from src.drawing.pipeline import run
    p = tmp_path / "tess.stp"
    p.write_text(_tess_step_text(), encoding="ascii")
    _, rep = run(str(p), str(tmp_path / "o"), _opts(formats=("svg",), views=("front",)))
    assert rep["ok"] is True and rep["route"]["source"] == "step_tessellated_mesh"
    g = rep["geometry"]
    assert g["bodies"]["solids"] == 1                              # precondition
    assert g["solids"] == 1
    assert g["bodies"]["tessellated_bodies"] >= 1


@needs_ocp
def test_z1_issue26_section_reason_after_the_body_centroid_retry(tmp_path):
    from src.drawing.pipeline import run
    two = _compound([_box_shape(20, 20, 2), _box_shape(20, 20, 2, origin=(0.0, 0.0, 8.0))])
    p = _write_step(two, tmp_path / "two.step")
    _, rep = run(p, str(tmp_path / "o"), _opts(formats=("svg",), views=("front", "top")))
    assert rep["ok"] is True, rep["error"]
    sec = rep["section"]
    assert sec["retried_at_body"] is True                          # precondition: the gap plane
    assert "body_centroid" in sec["reason"], sec["reason"]
    assert abs(sec["retry_mm"]) > 1.0
    fl = [f for f in rep["quality"]["flags"] if f["code"] == "section_fallback"]
    assert len(fl) == 1 and "goes through the bbox centre" not in fl[0]["detail"], fl
    assert "centroid" in fl[0]["detail"]


def test_z1_issue27_assembly_omission_detail_splits_the_reasons():
    from src.drawing.pipeline import _assembly_omission_detail
    omitted = [{"reason": "assembly", "count": 5}, {"reason": "assembly", "count": 3},
               {"reason": "filled_by_fastener", "count": 4}]
    t = _assembly_omission_detail(omitted)
    assert "8 hole feature(s) in 2 radius group(s), reason assembly" in t, t
    assert "4 in 1 group(s) filled by a fastener" in t, t
    assert "12" not in t                                           # no double-counted total
    assert "filled" not in _assembly_omission_detail(omitted[:2])


def _open_misoriented_box(path):
    import trimesh
    m = trimesh.creation.box(extents=(20.0, 10.0, 5.0))
    faces = m.faces[1:].copy()                                     # one face deleted: open
    faces[[1, 2, 3]] = faces[[1, 2, 3]][:, ::-1]                   # three faces flipped
    trimesh.Trimesh(m.vertices.copy(), faces, process=False).export(str(path))
    return str(path)


def test_z1_issue28_open_mesh_with_inconsistent_winding_is_flagged(tmp_path):
    from src.drawing.mesh_load import load_mesh_model
    from src.drawing.pipeline import run
    p = _open_misoriented_box(tmp_path / "open.stl")
    c = load_mesh_model(p).census
    assert c["winding_conflicts"] > 0 and not c["volume_repaired"]   # precondition
    _, rep = run(p, str(tmp_path / "o"), _opts(formats=("svg",), views=("front",),
                                               stl_to_step="off"))
    assert rep["ok"] is True, rep["error"]
    fl = [f for f in rep["quality"]["flags"] if f["code"] == "mesh_winding_inconsistent"]
    assert len(fl) == 1 and fl[0]["severity"] == "info", rep["quality"]["flags"]
    assert str(c["winding_conflicts"]) in fl[0]["detail"]
    assert rep["geometry"]["winding_conflicts"] == c["winding_conflicts"]
    assert "degenerate_faces" in rep["geometry"]
    # control: a closed, consistently wound box raises nothing about the winding
    q = _box_stl(tmp_path / "ok.stl", (20.0, 10.0, 5.0))
    _, ok = run(q, str(tmp_path / "k"), _opts(formats=("svg",), views=("front",),
                                              stl_to_step="off"))
    assert "mesh_winding_inconsistent" not in _codes(ok)
    assert ok["geometry"]["winding_conflicts"] == 0


def test_z1_issue29_never_raised_flags_are_gone_and_fallback_wording():
    from src.drawing.report import FLAG_MEANINGS, FLAGS
    for code in ("loose_curves_drawn", "supplemental_geometry_ignored"):
        assert code not in FLAGS and code not in FLAG_MEANINGS
    m = FLAG_MEANINGS["section_fallback"]
    assert "no solid (no section)" not in m
    assert "surface model" in m and "not_a_solid" in m
    assert FLAGS.get("mesh_winding_inconsistent") == "info"
    path = os.path.join(REPO, "docs", "drawing_module.md")
    text = open(path, encoding="utf-8").read()
    assert "`loose_curves_drawn`" not in text and "`supplemental_geometry_ignored`" not in text


@needs_ocp
def test_z1_issue30_sandboxed_quiet_run_keeps_the_stage_lines_on_stderr(tmp_path):
    p = _write_step(_box_shape(30, 20, 10), tmp_path / "box.step")
    proc = _cli([p, str(tmp_path / "o"), "--quiet", "--formats", "svg", "--views", "front"])
    rep = _result(proc)
    assert rep["sandbox"]["enabled"] is True                       # precondition: sandboxed
    lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
    assert lines, "no stage line reached stderr under --quiet"
    assert all("src.drawing.stage" in ln for ln in lines), lines[:5]
    assert any("stage done" in ln for ln in lines), lines


def test_z1_issue30_docs_name_the_run3_timeout():
    text = open(os.path.join(REPO, "docs", "drawing_module.md"), encoding="utf-8").read()
    for ln in text.splitlines():
        if "default limits" in ln:
            assert "--timeout" in ln, ln
    assert "--timeout 300" in text.split("## Measured numbers", 1)[1].split("HLR (front", 1)[0]


# ---- review Z1: report rounding and wording --------------------------------------------

def test_z1_review_hatch_spacing_is_rounded_like_the_list_and_null_when_all_solid(tmp_path):
    raw = 1.0597614673911924                                      # the unrounded value of FTC 08
    sl = {"fill": "hatch", "hatch_spacing": raw, "body_spacing": [raw], "hatch_segments": 4,
          "bodies": 1}
    sec = _section_report_for(tmp_path, sl, scale=1.0).data["section"]
    assert round(raw, 4) != raw                                   # precondition: needs rounding
    assert sec["hatch_spacing_mm"] == 1.0598 and sec["hatch_spacings_mm"] == [1.0598]
    solid = {"fill": "solid", "hatch_spacing": None, "body_spacing": [None], "hatch_segments": 0,
             "bodies": 1}
    sec2 = _section_report_for(tmp_path, solid, scale=1.0).data["section"]
    assert sec2["hatch_spacing_mm"] is None and sec2["hatch_spacings_mm"] == [None]


def test_z1_review_section_nudged_detail_gives_magnitude_and_side(tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from src.drawing.pipeline import _Ctx, _section_report
    from src.drawing.report import Report

    def detail(nudge):
        rep = Report(str(tmp_path / "p.step"), str(tmp_path), _opts())
        ctx = _Ctx(rep, str(tmp_path / "p.step"), str(tmp_path), _opts(), lambda s: None)
        sec = SimpleNamespace(label="A", spec=SimpleNamespace(name="top"), marker_view="front",
                              plane_point=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]),
                              loops=[np.array([[0, 0], [1, 0], [1, 1]], float)],
                              stats={"nudged": True, "nudge_mm": nudge})
        _section_report(ctx, sec)
        f = [fl for fl in rep.data["quality"]["flags"] if fl["code"] == "section_nudged"]
        assert len(f) == 1                                        # precondition: the flag fires
        return f[0]["detail"]

    neg, pos = detail(-0.003399), detail(0.25)
    assert "-0.0034" not in neg and "0.003399 mm against the plane normal" in neg, neg
    assert "0.25 mm along the plane normal" in pos, pos
