"""Tests of ``src.drawing.brep_load`` and ``src.drawing.brep_features``.

The fixture plate is the one of the brep-analysis probe (REPORT-brep-analysis.md):
a 100x60x15 box, a d30 x 10 boss at (78,30) on top, four d6.5 through holes at
(8,8) (92,8) (8,52) (92,52) and one d20 through hole at (50,30). Every number
asserted here was measured by that probe on OCP 7.9.3.

Test discipline: every negative test asserts its precondition first, and each
test would fail against a constant or no-op implementation.
"""
import functools
import math
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")

PLATE_VOLUME = 90365.210146


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
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


def _box(dx=10.0, dy=10.0, dz=10.0):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    return BRepPrimAPI_MakeBox(dx, dy, dz).Shape()


def _count(shape, kind):
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return m.Extent()


def _write_step(shape, path):
    from src.io.exporters import export_step
    export_step(shape, str(path))
    assert os.path.getsize(path) > 0
    return str(path)


def _continuity_census(shape):
    """Edges with two distinct faces -> list of GeomAbs_Shape ints."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    out = []
    for i in range(1, m.Extent() + 1):
        faces = list(m.FindFromIndex(i))
        if len(faces) != 2:
            continue
        f1, f2 = TopoDS.Face_s(faces[0]), TopoDS.Face_s(faces[1])
        if f1.IsSame(f2):
            continue
        e = TopoDS.Edge_s(m.FindKey(i))
        out.append(int(BRep_Tool.Continuity_s(e, f1, f2)))
    return out


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------

@needs_ocp
def test_plate_census():
    from src.drawing.brep_load import analyze_shape
    model = analyze_shape(_plate())
    c = model.census
    assert c["face_types"] == {"Plane": 7, "Cylinder": 6}
    assert c["faces"] == 13 and c["edges"] == 30 and c["vertices"] == 20
    assert c["solids"] == 1 and c["shells"] == 1
    assert c["brepcheck_valid"] is True
    assert c["closed"] is True
    assert c["free_edges"] == 0
    assert c["inverted"] is False
    assert c["volume_mm3"] == pytest.approx(PLATE_VOLUME, rel=1e-6)
    assert c["area_mm2"] == pytest.approx(19016.393617, rel=1e-6)
    assert np.allclose(c["bbox_min"], [0, 0, 0], atol=1e-6)
    assert np.allclose(c["bbox_max"], [100, 60, 25], atol=1e-6)
    assert model.source_unit == "mm" and model.roots == 1
    assert model.warnings == []


@needs_ocp
def test_bbox_exact_on_plate_and_on_nurbs():
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from src.drawing.brep_load import analyze_shape

    # precondition: on the analytic plate without a mesh Add_s is exact too
    b = Bnd_Box()
    BRepBndLib.Add_s(_plate(), b)
    assert np.allclose(b.Get(), (0, 0, 0, 100, 60, 25), atol=1e-6)

    cyl = BRepPrimAPI_MakeCylinder(10.0, 20.0).Shape()
    nurbs = BRepBuilderAPI_NurbsConvert(cyl, True).Shape()
    b = Bnd_Box()
    BRepBndLib.Add_s(nurbs, b)
    xmin, ymin, _, xmax, ymax, _ = b.Get()
    # precondition: Add_s really over-reports on B-splines (the control hull)
    assert max(xmax - 10, -10 - xmin, ymax - 10, -10 - ymin) > 5
    c = analyze_shape(nurbs).census
    assert np.allclose(c["bbox_min"][:2], [-10, -10], atol=1e-4)
    assert np.allclose(c["bbox_max"][:2], [10, 10], atol=1e-4)
    assert c["bbox_min"][2] == pytest.approx(0, abs=1e-4)
    assert c["bbox_max"][2] == pytest.approx(20, abs=1e-4)


@needs_ocp
def test_reversed_box_is_flagged_inverted():
    from src.drawing.brep_load import analyze_shape
    good = analyze_shape(_box()).census
    assert good["inverted"] is False and good["volume_mm3"] == pytest.approx(1000.0)
    rev = analyze_shape(_box().Reversed())
    assert rev.census["inverted"] is True
    assert rev.census["volume_mm3"] == pytest.approx(-1000.0, rel=1e-9)
    assert any("invert" in w for w in rev.warnings)


@needs_ocp
def test_open_shell_census():
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeShell
    from OCP.Geom import Geom_Plane
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    from src.drawing.brep_load import analyze_shape

    surf = Geom_Plane(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)))
    shell = BRepBuilderAPI_MakeShell(surf, 0.0, 10.0, 0.0, 10.0).Shell()
    m = analyze_shape(shell)
    assert m.census["solids"] == 0
    assert m.census["shells"] == 1
    assert m.census["free_edges"] == 4
    assert m.census["closed"] is False
    assert any("solid" in w for w in m.warnings)
    assert any("free edge" in w for w in m.warnings)


@needs_ocp
def test_fast_census_skips_brepcheck_and_adaptive_volume(monkeypatch):
    """full_check=False must not call BRepCheck nor step_audit.brep_volume
    (both replaced by raising stubs here) and must still give the exact
    volume of a faceted body from the fixed Gauss scheme."""
    import trimesh
    import OCP.BRepCheck
    import src.io.step_audit as step_audit
    from src.drawing.brep_load import analyze_shape, faceted_shape_from_mesh

    mesh = trimesh.creation.icosphere(3, radius=20.0)
    shape = faceted_shape_from_mesh(mesh, 0.35, lambda m: None)
    full = analyze_shape(shape)
    # precondition: the default census runs both and they succeed
    assert full.census["brepcheck_valid"] is True
    assert full.census["volume_mm3"] == pytest.approx(float(mesh.volume), rel=1e-9)

    def boom(*a, **k):
        raise RuntimeError("must not be called")
    monkeypatch.setattr(OCP.BRepCheck, "BRepCheck_Analyzer", boom)
    monkeypatch.setattr(step_audit, "brep_volume", boom)

    # positive control of the stubs: the full census now loses both numbers
    hit = analyze_shape(shape)
    assert hit.census["brepcheck_valid"] is False and hit.census["volume_mm3"] is None

    fast = analyze_shape(shape, full_check=False)
    c = fast.census
    assert c["brepcheck_valid"] is None                 # "not checked"
    assert c["volume_mm3"] == pytest.approx(float(mesh.volume), rel=1e-9)
    for key in ("faces", "edges", "vertices", "solids", "shells", "closed",
                "free_edges", "inverted", "face_types"):
        assert c[key] == full.census[key], key
    assert np.allclose(c["bbox_max"], full.census["bbox_max"], atol=1e-9)
    assert not any("BRepCheck" in w for w in fast.warnings)
    assert not any("volume" in w for w in fast.warnings)


@needs_ocp
def test_null_shape_census_is_empty_not_a_crash():
    from OCP.TopoDS import TopoDS_Shape
    from src.drawing.brep_load import analyze_shape
    m = analyze_shape(TopoDS_Shape())
    assert m.census["faces"] == 0 and m.census["solids"] == 0
    assert m.census["bbox_min"] is None and m.census["brepcheck_valid"] is False


# ---------------------------------------------------------------------------
# STEP reading
# ---------------------------------------------------------------------------

@needs_ocp
def test_load_step_round_trip_same_census(tmp_path):
    from src.drawing.brep_load import analyze_shape, load_step
    path = _write_step(_plate(), tmp_path / "plate.step")
    loaded = load_step(path)
    ref = analyze_shape(_plate()).census
    for key in ("face_types", "faces", "edges", "vertices", "solids", "shells",
                "brepcheck_valid", "closed", "free_edges", "inverted"):
        assert loaded.census[key] == ref[key], key
    assert loaded.census["volume_mm3"] == pytest.approx(PLATE_VOLUME, rel=1e-6)
    assert np.allclose(loaded.census["bbox_max"], [100, 60, 25], atol=1e-6)
    assert loaded.source_unit == "mm" and loaded.roots == 1


def _assert_valid_step_loads(tmp_path):
    """Matched-pair positive control: the same load_step reads a good file."""
    from src.drawing.brep_load import load_step
    good = load_step(_write_step(_box(), tmp_path / "good_box.step"))
    assert good.census["faces"] == 6 and good.census["solids"] == 1
    assert good.census["volume_mm3"] == pytest.approx(1000.0, rel=1e-6)


@needs_ocp
def test_load_step_garbage_raises_value_error(tmp_path):
    from src.drawing.brep_load import load_step
    _assert_valid_step_loads(tmp_path)
    p = tmp_path / "garbage.step"
    p.write_text("garbage\nthis is not a STEP file\n", encoding="ascii")
    assert p.exists() and p.stat().st_size > 0
    with pytest.raises(ValueError):
        load_step(str(p))
    # still alive: OCCT work after the failed read still runs in this process
    from OCP.TopAbs import TopAbs_FACE
    assert _count(_box(), TopAbs_FACE) == 6
    _assert_valid_step_loads(tmp_path)


@needs_ocp
def test_load_step_missing_file_raises(tmp_path):
    from src.drawing.brep_load import load_step
    _assert_valid_step_loads(tmp_path)
    p = tmp_path / "nope.step"
    assert not p.exists()
    with pytest.raises(ValueError):
        load_step(str(p))


@needs_ocp
def test_load_step_garbage_in_a_child_process_exits_cleanly(tmp_path):
    """Independent of the in-process check: a child that first loads a good
    STEP (exit 3 if that fails) and then garbage must exit with Python's own
    status (7, ValueError caught), never 0xC0000005."""
    good = _write_step(_box(), tmp_path / "good_box.step")
    p = tmp_path / "garbage.step"
    p.write_text("garbage\n", encoding="ascii")
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from src.drawing.brep_load import load_step\n"
            "if load_step(%r).census['faces'] != 6:\n    sys.exit(3)\n"
            "try:\n    load_step(%r)\nexcept ValueError:\n    sys.exit(7)\n"
            "sys.exit(0)\n") % (REPO, good, str(p))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          timeout=120, env=env)
    assert proc.returncode == 7, (proc.returncode, proc.stderr[-2000:])


@needs_ocp
def test_load_step_empty_data_raises(tmp_path):
    from src.drawing.brep_load import load_step
    _assert_valid_step_loads(tmp_path)
    p = tmp_path / "empty.step"
    p.write_text(
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\n"
        "FILE_NAME('empty','2026-10-17T00:00:00',(''),(''),'','','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\n"
        "ENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="ascii")
    assert "DATA;" in p.read_text()
    with pytest.raises(ValueError):
        load_step(str(p))


@needs_ocp
def test_load_step_two_roots(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from src.drawing.brep_load import load_step
    from src.io.occt_utils import init_step_statics

    init_step_statics(unit="MM")
    w = STEPControl_Writer()
    w.Transfer(BRepPrimAPI_MakeBox(10, 10, 10).Shape(), STEPControl_AsIs)
    w.Transfer(BRepPrimAPI_MakeBox(gp_Pnt(15, 15, 15), 10, 10, 10).Shape(),
               STEPControl_AsIs)
    p = str(tmp_path / "two.step")
    w.Write(p)
    m = load_step(p)
    assert m.roots == 2
    assert m.census["solids"] == 2
    assert m.census["volume_mm3"] == pytest.approx(2000.0, rel=1e-6)
    assert np.allclose(m.census["bbox_max"], [25, 25, 25], atol=1e-6)
    assert any("solid" in w_ for w_ in m.warnings)


@needs_ocp
def test_load_step_inch_file(tmp_path):
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from src.drawing.brep_load import load_step, step_unit_sniff
    from src.io.occt_utils import init_step_statics

    p = str(tmp_path / "cube_inch.step")
    try:
        init_step_statics(unit="INCH")
        w = STEPControl_Writer()
        w.Transfer(_box(25.4, 25.4, 25.4), STEPControl_AsIs)
        w.Write(p)
    finally:
        init_step_statics(unit="MM")
    with open(p, encoding="latin-1") as fh:
        assert "INCH" in fh.read()           # precondition: really an inch file
    assert step_unit_sniff(p) == "inch"
    m = load_step(p)
    assert m.source_unit == "inch"
    assert np.allclose(m.census["bbox_max"], [25.4, 25.4, 25.4], atol=1e-6)
    assert any("inch" in w_ for w_ in m.warnings)
    assert not any("disagreement" in w_ for w_ in m.warnings)


def _inch_step(tmp_path):
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from src.io.occt_utils import init_step_statics
    p = str(tmp_path / "cube_inch.step")
    try:
        init_step_statics(unit="INCH")
        w = STEPControl_Writer()
        w.Transfer(_box(25.4, 25.4, 25.4), STEPControl_AsIs)
        w.Write(p)
    finally:
        init_step_statics(unit="MM")
    return p


@needs_ocp
def test_load_step_unit_comes_from_file_units_not_the_sniff(tmp_path, monkeypatch):
    """FileUnits is the primary source: with the sniff forced to 'mm' the
    inch file still reports inch, plus a disagreement warning."""
    from src.drawing import brep_load
    p = _inch_step(tmp_path)
    assert brep_load.step_unit_sniff(p) == "inch"          # precondition
    monkeypatch.setattr(brep_load, "step_unit_sniff", lambda path: "mm")
    m = brep_load.load_step(p)
    assert m.source_unit == "inch"
    assert any("disagreement" in w_ and "INCH" in w_.upper() for w_ in m.warnings)


@needs_ocp
def test_load_step_unit_falls_back_to_the_sniff(tmp_path, monkeypatch):
    """When FileUnits gives nothing the text sniff decides, without a
    disagreement warning (there is nothing to disagree with)."""
    from src.drawing import brep_load
    p = _write_step(_box(), tmp_path / "box.step")
    assert brep_load.load_step(p).source_unit == "mm"       # precondition
    monkeypatch.setattr(brep_load, "_file_units", lambda reader: [])
    monkeypatch.setattr(brep_load, "step_unit_sniff", lambda path: "inch")
    m = brep_load.load_step(p)
    assert m.source_unit == "inch"
    assert not any("disagreement" in w_ for w_ in m.warnings)


@needs_ocp
def test_step_unit_sniff_mm_and_unknown(tmp_path):
    from src.drawing.brep_load import step_unit_sniff
    p = _write_step(_box(), tmp_path / "box.step")
    assert step_unit_sniff(p) == "mm"
    g = tmp_path / "g.step"
    g.write_text("garbage", encoding="ascii")
    assert step_unit_sniff(str(g)) == "unknown"


def test_step_unit_sniff_ignores_angle_conversion_unit(tmp_path):
    """A naive first-match would return DEGREE; the rule keeps only records
    with LENGTH_UNIT( and prefers CONVERSION_BASED_UNIT over SI_UNIT."""
    from src.drawing.brep_load import step_unit_sniff
    text = ("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n"
            "#1=( CONVERSION_BASED_UNIT('DEGREE',#2) NAMED_UNIT(*) PLANE_ANGLE_UNIT() );\n"
            "#3=( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );\n"
            "ENDSEC;\nEND-ISO-10303-21;\n")
    p = tmp_path / "deg.step"
    p.write_text(text, encoding="ascii")
    assert step_unit_sniff(str(p)) == "mm"
    p.write_text(text.replace("SI_UNIT(.MILLI.,.METRE.)",
                              "CONVERSION_BASED_UNIT('INCH',#9)"), encoding="ascii")
    assert step_unit_sniff(str(p)) == "inch"


_HEAD = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n"
_DEGREE = "#1=( CONVERSION_BASED_UNIT('DEGREE',#2) NAMED_UNIT(*) PLANE_ANGLE_UNIT() );\n"
_INCH = "#9=( CONVERSION_BASED_UNIT('INCH',#8) LENGTH_UNIT() NAMED_UNIT(*) );\n"
_MM = "#7=( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );\n"
_TAIL = "ENDSEC;\nEND-ISO-10303-21;\n"


def _filler(n_bytes):
    rec = "#100=CARTESIAN_POINT('',(1.25,2.5,3.75));\n"
    return rec * (n_bytes // len(rec) + 1)


def test_step_unit_sniff_finds_units_at_the_end_of_a_large_file(tmp_path):
    """OCCT puts the unit records at the end of DATA; a file larger than any
    fixed read limit (the old sniff read the first 50 MB) must still sniff."""
    from src.drawing.brep_load import step_unit_sniff
    p = tmp_path / "big.step"
    body = _HEAD + _DEGREE + _filler(55_000_000)
    p.write_text(body + _INCH + _TAIL, encoding="ascii")
    assert p.stat().st_size > 55_000_000                   # precondition
    assert len(body) > 50_000_000                          # unit beyond 50 MB
    assert step_unit_sniff(str(p)) == "inch"
    p.write_text(body + _MM + _TAIL, encoding="ascii")
    assert step_unit_sniff(str(p)) == "mm"
    p.write_text(body + _TAIL, encoding="ascii")           # negative control
    assert step_unit_sniff(str(p)) == "unknown"


def test_step_unit_sniff_streams_the_middle_across_chunk_borders(tmp_path, monkeypatch):
    """With small windows the unit record sits in the middle (outside head
    and tail) and is moved across a chunk border: every offset must sniff."""
    from src.drawing import brep_load
    monkeypatch.setattr(brep_load, "_SNIFF_WINDOW", 4096)
    monkeypatch.setattr(brep_load, "_SNIFF_CHUNK", 1024)
    monkeypatch.setattr(brep_load, "_SNIFF_RECORD_MAX", 200)
    # The INCH record (its unit name ~20 characters before LENGTH_UNIT) is
    # slid in 7-byte steps over more than one chunk length, so it crosses at
    # least one chunk border whatever the chunk start; without the chunk
    # overlap some offsets lose either the LENGTH_UNIT token or the unit name.
    p = tmp_path / "mid.step"
    pre = _HEAD + _DEGREE
    n_ok = 0
    for start in range(6000, 6000 + 1100, 7):
        text = (pre + _filler(start)[:start - len(pre)] + _INCH
                + _filler(5000) + _TAIL)
        p.write_text(text, encoding="ascii")
        pos = text.find("LENGTH_UNIT")
        assert 4096 < pos < len(text) - 4096                 # precondition: middle
        assert brep_load.step_unit_sniff(str(p)) == "inch", start
        n_ok += 1
    assert n_ok >= 150
    # negative control: no length unit anywhere -> unknown
    p.write_text(pre + _filler(15_000) + _TAIL, encoding="ascii")
    assert brep_load.step_unit_sniff(str(p)) == "unknown"


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

@needs_ocp
def test_transform_shape_permutes_bbox():
    from src.drawing.brep_load import analyze_shape, transform_shape
    from src.drawing.projection import Orientation

    box = _box(10, 20, 30)
    before = analyze_shape(box).census
    assert np.allclose(before["bbox_max"], [10, 20, 30])
    R = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # det +1
    out = analyze_shape(transform_shape(box, R)).census
    assert np.allclose(np.subtract(out["bbox_max"], out["bbox_min"]), [20, 30, 10], atol=1e-9)
    assert out["volume_mm3"] == pytest.approx(6000.0)
    # the input is not modified (copy=True)
    assert np.allclose(analyze_shape(box).census["bbox_max"], [10, 20, 30])
    # an Orientation is accepted too
    o = Orientation(matrix=R, permutation=(1, 2, 0), flips=(1, 1, 1), mode="auto")
    out2 = analyze_shape(transform_shape(box, o)).census
    assert np.allclose(out2["bbox_max"], out["bbox_max"], atol=1e-9)


@needs_ocp
def test_transform_shape_rejects_improper_matrix():
    from src.drawing.brep_load import analyze_shape, transform_shape
    # matched-pair positive control: a proper 90 deg rotation about z passes
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rot = analyze_shape(transform_shape(_box(10, 20, 30), Rz)).census
    assert np.allclose(rot["bbox_min"], [-20, 0, 0], atol=1e-9)
    assert np.allclose(rot["bbox_max"], [0, 10, 30], atol=1e-9)
    M = np.diag([1.0, 1.0, -1.0])
    assert np.linalg.det(M) == pytest.approx(-1.0)      # precondition: a mirror
    with pytest.raises(ValueError):
        transform_shape(_box(), M)


# ---------------------------------------------------------------------------
# faceted shell from a mesh
# ---------------------------------------------------------------------------

@needs_ocp
def test_faceted_shape_from_mesh_codes_regularity():
    import trimesh
    from OCP.GeomAbs import GeomAbs_C0, GeomAbs_C1
    from OCP.TopAbs import TopAbs_FACE
    from src.drawing.brep_load import faceted_shape_from_mesh
    from src.io.faceted_step import _build_direct, _shell_from_faces

    mesh = trimesh.creation.icosphere(2)
    assert len(mesh.faces) == 320
    # precondition: the plain shell has every edge C0
    raw = _shell_from_faces(_build_direct(np.asarray(mesh.vertices), np.asarray(mesh.faces)))
    raw_c = _continuity_census(raw)
    assert len(raw_c) == 480 and all(c == int(GeomAbs_C0) for c in raw_c)

    warnings = []
    shape = faceted_shape_from_mesh(mesh, 0.35, warnings.append)
    assert _count(shape, TopAbs_FACE) == 320
    cont = _continuity_census(shape)
    assert len(cont) == 480
    assert all(c >= int(GeomAbs_C1) for c in cont)

    # negative control: below the dihedral angle (0.12..0.20 rad) nothing is coded
    low = faceted_shape_from_mesh(mesh, 0.1, warnings.append)
    low_c = _continuity_census(low)
    assert len(low_c) == 480 and all(c == int(GeomAbs_C0) for c in low_c)


@needs_ocp
def test_faceted_shape_closed_is_solid_open_is_shell():
    import trimesh
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from src.drawing.brep_load import analyze_shape, faceted_shape_from_mesh

    mesh = trimesh.creation.icosphere(2, radius=20.0)
    warnings = []
    shape = faceted_shape_from_mesh(mesh, 0.35, warnings.append)
    assert shape.ShapeType() == TopAbs_SOLID
    c = analyze_shape(shape).census
    assert c["solids"] == 1 and c["closed"] is True and c["inverted"] is False
    assert c["volume_mm3"] == pytest.approx(float(mesh.volume), rel=1e-9)

    opened = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[1:], process=False)
    assert not opened.is_watertight                         # precondition
    sh = faceted_shape_from_mesh(opened, 0.35, warnings.append)
    assert sh.ShapeType() == TopAbs_SHELL
    oc = analyze_shape(sh).census
    assert oc["solids"] == 0 and oc["free_edges"] == 3
    assert any("open" in w for w in warnings)


@needs_ocp
def test_faceted_shape_two_components_is_compound_of_solids():
    import trimesh
    from OCP.TopAbs import TopAbs_COMPOUND
    from src.drawing.brep_load import analyze_shape, faceted_shape_from_mesh
    a = trimesh.creation.box((10, 10, 10))
    b = trimesh.creation.box((10, 10, 10))
    b.apply_translation((30, 0, 0))
    both = trimesh.util.concatenate([a, b])
    shape = faceted_shape_from_mesh(both, 0.35, lambda m: None)
    assert shape.ShapeType() == TopAbs_COMPOUND
    c = analyze_shape(shape).census
    assert c["solids"] == 2
    assert c["volume_mm3"] == pytest.approx(2000.0, rel=1e-9)


# ---------------------------------------------------------------------------
# cylinder features
# ---------------------------------------------------------------------------

def _check_plate_features(feats, n_faces):
    holes = [f for f in feats if f.kind == "hole"]
    bosses = [f for f in feats if f.kind == "boss"]
    assert len(feats) == 6
    assert len(holes) == 5 and len(bosses) == 1
    small = [f for f in holes if abs(f.radius - 3.25) < 1e-6]
    big = [f for f in holes if abs(f.radius - 10.0) < 1e-6]
    assert len(small) == 4 and len(big) == 1
    for f in holes:
        assert f.full_circle is True
        assert f.length == pytest.approx(15.0, abs=1e-6)
        assert f.center[2] == pytest.approx(7.5, abs=1e-6)
        assert abs(abs(f.axis[2]) - 1.0) < 1e-9
        assert f.n_faces == n_faces
    xy = sorted((round(float(f.center[0]), 6), round(float(f.center[1]), 6)) for f in small)
    assert xy == [(8, 8), (8, 52), (92, 8), (92, 52)]
    assert np.allclose(big[0].center, [50, 30, 7.5], atol=1e-6)
    boss = bosses[0]
    assert boss.radius == pytest.approx(15.0, abs=1e-6)
    assert boss.length == pytest.approx(10.0, abs=1e-6)
    assert np.allclose(boss.center, [78, 30, 20], atol=1e-6)
    assert boss.full_circle is True and boss.n_faces == n_faces


@needs_ocp
def test_cylinder_features_plate():
    from src.drawing.brep_features import cylinder_features
    from src.drawing.model import CylFeature
    feats = cylinder_features(_plate())
    assert all(isinstance(f, CylFeature) for f in feats)
    _check_plate_features(feats, n_faces=1)


@needs_ocp
def test_cylinder_features_split_faces_are_grouped():
    from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosed
    from OCP.TopAbs import TopAbs_FACE
    from src.drawing.brep_features import cylinder_features
    sd = ShapeUpgrade_ShapeDivideClosed(_plate())
    sd.SetNbSplitPoints(1)
    assert sd.Perform()
    split = sd.Result()
    assert _count(split, TopAbs_FACE) == 19                 # precondition
    _check_plate_features(cylinder_features(split), n_faces=2)


@needs_ocp
def test_cylinder_features_negative_control_without_reversed_flip(monkeypatch):
    from src.drawing import brep_features
    assert brep_features._flip_reversed is True
    monkeypatch.setattr(brep_features, "_flip_reversed", False)
    feats = brep_features.cylinder_features(_plate())
    assert len(feats) == 6
    assert [f.kind for f in feats].count("hole") == 0
    assert [f.kind for f in feats].count("boss") == 6


@needs_ocp
def test_cylinder_features_inverted_solid_keeps_its_holes(monkeypatch):
    """plate.Reversed() is an inverted solid: its holes must stay holes.
    Negative control: without the inversion fix the five holes read as
    bosses and the r 15 boss reads as a hole."""
    from src.drawing import brep_features
    from src.drawing.brep_load import analyze_shape
    inv = _plate().Reversed()
    c = analyze_shape(inv).census
    assert c["inverted"] is True                            # precondition
    assert c["volume_mm3"] == pytest.approx(-PLATE_VOLUME, rel=1e-6)
    _check_plate_features(brep_features.cylinder_features(inv), n_faces=1)

    assert brep_features._heal_inverted is True
    monkeypatch.setattr(brep_features, "_heal_inverted", False)
    feats = brep_features.cylinder_features(inv)
    kinds = [f.kind for f in feats]
    assert len(kinds) == 6 and kinds.count("boss") == 5 and kinds.count("hole") == 1
    assert [f.radius for f in feats if f.kind == "hole"] == pytest.approx([15.0])
    # the un-inverted plate is not affected by the flag
    _check_plate_features(brep_features.cylinder_features(_plate()), n_faces=1)


def _cyl_face(u0, u1, v0, v1, r=10.0, reversed_=False):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Ax3, gp_Cylinder, gp_Dir, gp_Pnt
    cyl = gp_Cylinder(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), r)
    f = BRepBuilderAPI_MakeFace(cyl, u0, u1, v0, v1).Face()
    return f.Reversed() if reversed_ else f


def _compound(shapes):
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    b = BRep_Builder()
    comp = TopoDS_Compound()
    b.MakeCompound(comp)
    for s in shapes:
        b.Add(comp, s)
    return comp


@needs_ocp
def test_cylinder_features_stacked_halves_are_not_a_full_circle():
    """Two half-cylinders on the same side, stacked along the axis (v 0..10
    and 10..20), add up to 360 degrees of U but cover only half the circle at
    every height. Adding the missing other half (v 0..20) closes it."""
    from src.drawing.brep_features import cylinder_features
    pi = math.pi
    for rev, closed_kind, open_kind in ((False, "boss", "boss"), (True, "hole", "fillet")):
        stacked = [_cyl_face(0, pi, 0, 10, reversed_=rev),
                   _cyl_face(0, pi, 10, 20, reversed_=rev)]
        feats = cylinder_features(_compound(stacked))
        assert len(feats) == 1 and feats[0].n_faces == 2    # grouped (precondition)
        f = feats[0]
        assert f.length == pytest.approx(20.0, abs=1e-9)
        assert f.full_circle is False and f.kind == open_kind, rev

        closed = stacked + [_cyl_face(pi, 2 * pi, 0, 20, reversed_=rev)]
        feats = cylinder_features(_compound(closed))
        assert len(feats) == 1 and feats[0].n_faces == 3
        assert feats[0].full_circle is True and feats[0].kind == closed_kind, rev

        # two full rings stacked along the axis are a full circle too
        rings = [_cyl_face(0, 2 * pi, 0, 10, reversed_=rev),
                 _cyl_face(0, 2 * pi, 10, 20, reversed_=rev)]
        feats = cylinder_features(_compound(rings))
        assert len(feats) == 1 and feats[0].n_faces == 2
        assert feats[0].full_circle is True and feats[0].kind == closed_kind, rev


@needs_ocp
def test_cylinder_features_inside_fillet_and_empty():
    """An L-bracket inside fillet is concave but partial -> 'fillet';
    a plain box has no cylinders at all."""
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from src.drawing.brep_features import cylinder_features

    assert cylinder_features(_box()) == []

    base = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 40, 20, 5).Shape()
    wall = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 5, 20, 30).Shape()
    fuse = BRepAlgoAPI_Fuse(base, wall)
    fuse.SimplifyResult()
    lb = fuse.Shape()
    mk = BRepFilletAPI_MakeFillet(lb)
    exp = TopExp_Explorer(lb, TopAbs_EDGE)
    n = 0
    while exp.More():
        e = TopoDS.Edge_s(exp.Current())
        c = BRepAdaptor_Curve(e)
        p0, p1 = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
        # the inside corner edge: x = 5, z = 5, along Y
        if all(abs(p.X() - 5) < 1e-6 and abs(p.Z() - 5) < 1e-6 for p in (p0, p1)):
            mk.Add(5.0, e)
            n += 1
        exp.Next()
    assert n >= 1                                          # precondition: edge found
    shape = mk.Shape()
    feats = cylinder_features(shape)
    assert len(feats) == 1
    f = feats[0]
    assert f.kind == "fillet" and f.full_circle is False
    assert f.radius == pytest.approx(5.0, abs=1e-6)
    assert f.length == pytest.approx(20.0, abs=1e-6)
