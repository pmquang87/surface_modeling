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


# ---------------------------------------------------------------------------
# QA round 1 (F2): drawn bodies, tessellated STEP, open volume, units, counts
# ---------------------------------------------------------------------------

def _edge(p0, p1):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.gp import gp_Pnt
    return BRepBuilderAPI_MakeEdge(gp_Pnt(*p0), gp_Pnt(*p1)).Edge()


def _plane_face(x0, x1, y0, y1, z=0.0):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    return BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, z), gp_Dir(0, 0, 1)),
                                   x0, x1, y0, y1).Face()


def _box_with_supplemental():
    """Box 100x50x20 + a loose line x = -50..150 (a sketch / curve set) + a
    free planar face at x = 200..210 (a supplemental surface)."""
    return _compound([_box(100.0, 50.0, 20.0), _edge((-50, 0, 0), (150, 0, 0)),
                      _plane_face(200.0, 210.0, 0.0, 10.0)])


@needs_ocp
def test_load_step_drops_loose_curves_and_supplemental_shells(tmp_path):
    """Issue 1: a STEP with a solid, a GEOMETRIC_CURVE_SET and a
    SHELL_BASED_SURFACE_MODEL draws, bounds and counts the solid only."""
    from src.drawing.brep_load import analyze_shape, load_step
    p = _write_step(_box_with_supplemental(), tmp_path / "mixed.step")
    text = open(p, encoding="latin-1").read()
    assert "GEOMETRIC_CURVE_SET" in text and "SHELL_BASED_SURFACE_MODEL" in text  # precondition
    m = load_step(p)
    c = m.census
    assert np.allclose(c["bbox_min"], [0, 0, 0], atol=1e-6)
    assert np.allclose(c["bbox_max"], [100, 50, 20], atol=1e-6)
    assert c["closed"] is True and c["free_edges"] == 0
    assert c["solids"] == 1 and c["faces"] == 6 and c["shells"] == 1
    assert c["volume_mm3"] == pytest.approx(100000.0, rel=1e-9)
    assert m.bodies["drawn"] == "solids"
    assert m.bodies["dropped"] == {"free_shells": 1, "loose_edges": 1}
    assert any("supplemental geometry not drawn" in w for w in m.warnings)
    assert not any("free edge" in w for w in m.warnings)
    # negative control: the full transfer (what was drawn before) is wider and open
    raw = analyze_shape(m.original_shape).census
    assert raw["bbox_max"][0] - raw["bbox_min"][0] == pytest.approx(260.0, abs=1e-6)
    assert raw["closed"] is False and raw["free_edges"] == 4


@needs_ocp
def test_select_drawn_shape_rules():
    """Solids win; a surface model keeps shells + faces and drops loose edges;
    a wireframe keeps its edges; a clean solid comes back unchanged."""
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Shell
    from src.drawing.brep_load import exact_bbox, select_drawn_shape

    box = _box(10, 10, 10)
    same, info = select_drawn_shape(box)
    assert same is box and info["dropped"] == {} and info["drawn"] == "solids"

    b = BRep_Builder()
    shell = TopoDS_Shell()
    b.MakeShell(shell)
    b.Add(shell, _plane_face(0, 10, 0, 10))
    surf = _compound([shell, _plane_face(20, 30, 0, 10), _edge((0, 0, 0), (0, 0, 99))])
    drawn, info = select_drawn_shape(surf)
    assert info["drawn"] == "shells_faces"
    assert info["free_shells"] == 1 and info["free_faces"] == 1 and info["loose_edges"] == 1
    assert info["dropped"] == {"loose_edges": 1}
    assert _count(drawn, _ta("FACE")) == 2 and _count(drawn, _ta("EDGE")) == 8
    bb = exact_bbox(drawn)
    assert bb[1][2] - bb[0][2] == pytest.approx(0.0, abs=1e-6)     # the z = 99 line is gone
    assert exact_bbox(surf)[1][2] == pytest.approx(99.0, abs=1e-6)  # precondition

    wire = _compound([_edge((0, 0, 0), (5, 0, 0)), _edge((5, 0, 0), (5, 5, 0))])
    drawn, info = select_drawn_shape(wire)
    assert drawn is wire and info["drawn"] == "wireframe" and info["loose_edges"] == 2


def _ta(name):
    import OCP.TopAbs
    return getattr(OCP.TopAbs, "TopAbs_" + name)


def _tri_face(points, triangles, normals=None):
    """A face with only a Poly_Triangulation (no surface), as OCCT makes
    from a STEP TRIANGULATED_FACE. ``triangles`` are 1-based; ``normals``
    (one per node) are stored with the triangulation when given."""
    from OCP.BRep import BRep_Builder
    from OCP.Poly import Poly_Triangle, Poly_Triangulation
    from OCP.TopoDS import TopoDS_Face
    from OCP.gp import gp_Dir, gp_Pnt
    T = Poly_Triangulation(len(points), len(triangles), False)
    for i, p in enumerate(points, 1):
        T.SetNode(i, gp_Pnt(*map(float, p)))
    for i, t in enumerate(triangles, 1):
        T.SetTriangle(i, Poly_Triangle(*t))
    if normals is not None:
        T.AddNormals()
        for i, nv in enumerate(normals, 1):
            T.SetNormal(i, gp_Dir(*map(float, nv)))
    f = TopoDS_Face()
    BRep_Builder().MakeFace(f, T)
    return f


_TESS_PTS = [(0, 0, 0), (20, 0, 0), (20, 10, 0), (0, 10, 0),
             (0, 0, 5), (20, 0, 5), (20, 10, 5), (0, 10, 5)]
_TESS_A = [(1, 3, 2), (1, 4, 3), (5, 6, 7), (5, 7, 8), (1, 2, 6), (1, 6, 5)]
_TESS_B = [(2, 3, 7), (2, 7, 6), (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8)]


def _tess_step_text(unit="SI_UNIT(.MILLI.,.METRE.)"):
    """Minimal AP242 file: one TESSELLATED_SOLID of two TRIANGULATED_FACEs
    forming a closed 20x10x5 box with outward winding."""
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
        f"#10=(LENGTH_UNIT()NAMED_UNIT(*){unit});\n"
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
def test_load_step_tessellated_only_raises_with_a_mesh(tmp_path):
    """Issue 16: a tessellated-only AP242 file raises TessellatedOnlyError
    (a ValueError naming the cause) that carries a closed, outward mesh."""
    from src.drawing.brep_load import TessellatedOnlyError, load_step
    p = tmp_path / "tess.stp"
    p.write_text(_tess_step_text(), encoding="ascii")
    with pytest.raises(TessellatedOnlyError) as ei:
        load_step(str(p))
    err = ei.value
    assert isinstance(err, ValueError) and "tessellated" in str(err)
    assert err.faces == 2 and err.triangles == 12
    assert err.source_unit == "mm" and err.source_unit_factor == 1.0
    mesh = err.mesh
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(1000.0, rel=1e-9)        # positive: outward winding
    assert np.allclose(mesh.bounds, [[0, 0, 0], [20, 10, 5]])
    # metre file: OCCT scales the nodes, the error reports the unit
    p.write_text(_tess_step_text("SI_UNIT($,.METRE.)"), encoding="ascii")
    with pytest.raises(TessellatedOnlyError) as ei:
        load_step(str(p))
    assert ei.value.source_unit == "m" and ei.value.source_unit_factor == 1000.0
    assert np.allclose(ei.value.mesh.extents, [20000, 10000, 5000])


@needs_ocp
def test_triangulation_mesh_applies_location_and_orientation():
    from OCP.TopLoc import TopLoc_Location
    from OCP.gp import gp_Trsf, gp_Vec
    from src.drawing.brep_load import count_tessellated_faces, triangulation_mesh
    a, b = _tri_face(_TESS_PTS, _TESS_A), _tri_face(_TESS_PTS, _TESS_B)
    mesh, used, n = triangulation_mesh(_compound([a, b]))
    assert used == 2 and n == 12 and mesh.volume == pytest.approx(1000.0)
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(100.0, 0.0, 0.0))
    moved = _compound([a.Moved(TopLoc_Location(tr)), b.Moved(TopLoc_Location(tr))])
    mesh, _, _ = triangulation_mesh(moved)
    assert np.allclose(mesh.bounds, [[100, 0, 0], [120, 10, 5]])
    # reversed faces: the winding flips and the volume turns negative
    mesh, _, _ = triangulation_mesh(_compound([a.Reversed(), b.Reversed()]))
    assert mesh.volume == pytest.approx(-1000.0)
    assert count_tessellated_faces(_compound([a, _box()])) == 1


@needs_ocp
def test_triangulation_mesh_follows_stored_normals():
    """Triangles wound against their stored (outward) node normals, as in the
    NIST FTC 08 file (3370 of 3370 disagree, volume -503281 unflipped), are
    flipped; without stored normals the file winding is kept."""
    from src.drawing.brep_load import triangulation_mesh
    inward_a = [t[::-1] for t in _TESS_A]
    inward_b = [t[::-1] for t in _TESS_B]
    centre = np.mean(_TESS_PTS, axis=0)
    outward = [tuple((np.asarray(p, float) - centre) / np.linalg.norm(np.asarray(p, float) - centre))
               for p in _TESS_PTS]
    bare, _, _ = triangulation_mesh(_compound([_tri_face(_TESS_PTS, inward_a),
                                               _tri_face(_TESS_PTS, inward_b)]))
    assert bare.volume == pytest.approx(-1000.0)                 # precondition: inside out
    voted, _, _ = triangulation_mesh(_compound([_tri_face(_TESS_PTS, inward_a, outward),
                                                _tri_face(_TESS_PTS, inward_b, outward)]))
    assert voted.volume == pytest.approx(1000.0)
    kept, _, _ = triangulation_mesh(_compound([_tri_face(_TESS_PTS, _TESS_A, outward),
                                               _tri_face(_TESS_PTS, _TESS_B, outward)]))
    assert kept.volume == pytest.approx(1000.0)


@needs_ocp
def test_census_counts_tessellated_faces_in_a_mixed_shape():
    from src.drawing.brep_load import analyze_shape
    m = analyze_shape(_compound([_box(), _tri_face(_TESS_PTS, _TESS_A)]))
    assert m.census["tessellated_faces"] == 1
    assert any("tessellated face" in w for w in m.warnings)
    assert analyze_shape(_box()).census["tessellated_faces"] == 0


@needs_ocp
def test_select_drawn_shape_drops_tessellated_body_next_to_an_exact_one():
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Shell, TopoDS_Solid
    from src.drawing.brep_load import select_drawn_shape
    bld = BRep_Builder()
    shell = TopoDS_Shell()
    bld.MakeShell(shell)
    bld.Add(shell, _tri_face(_TESS_PTS, _TESS_A))
    bld.Add(shell, _tri_face(_TESS_PTS, _TESS_B))
    solid = TopoDS_Solid()
    bld.MakeSolid(solid)
    bld.Add(solid, shell)
    drawn, info = select_drawn_shape(_compound([_box(), solid]))
    assert info["solids"] == 2 and info["dropped"] == {"tessellated_bodies": 1}
    assert _count(drawn, _ta("SOLID")) == 1 and _count(drawn, _ta("FACE")) == 6
    # alone, the tessellated solid is kept (load_step routes it to the mesh path)
    drawn, info = select_drawn_shape(_compound([solid]))
    assert info["dropped"] == {} and _count(drawn, _ta("SOLID")) == 1


@needs_ocp
def test_open_model_volume_is_none():
    """Issue 28: an open model reports no volume (the signed number is kept
    for diagnostics); a closed box keeps its volume."""
    from src.drawing.brep_load import analyze_shape
    m = analyze_shape(_plane_face(0, 10, 0, 5))
    assert m.census["solids"] == 0 and m.census["closed"] is False     # precondition
    assert m.census["volume_mm3"] is None
    assert "volume_signed_open" in m.census
    assert m.census["area_mm2"] == pytest.approx(50.0)
    assert any("volume not reported" in w for w in m.warnings)
    assert not any("integration failed" in w for w in m.warnings)
    closed = analyze_shape(_box(10, 10, 10)).census
    assert closed["volume_mm3"] == pytest.approx(1000.0) and "volume_signed_open" not in closed


@needs_ocp
def test_free_face_next_to_a_solid_is_not_in_the_volume():
    """Review F2 (issue 28 follow-up): a box with a loose planar face at
    z = 200 reported closed=True and volume 6571.4 (the free face entered the
    volume integral). The census counts the loose face, calls the model open
    and integrates the volume over the solids only."""
    from src.drawing.brep_load import analyze_shape
    shape = _compound([_box(10, 10, 10), _plane_face(0, 10, 0, 5, z=200.0)])
    m = analyze_shape(shape)
    c = m.census
    assert c["solids"] == 1 and c["faces"] == 7 and c["free_edges"] == 4     # precondition
    assert c["loose_faces"] == 1
    assert c["closed"] is False
    assert c["volume_mm3"] == pytest.approx(1000.0)
    assert c["volume_scope"] == "solids"
    assert any("outside any shell" in w for w in m.warnings)
    assert c["inverted"] is False
    box = analyze_shape(_box(10, 10, 10)).census
    assert box["loose_faces"] == 0 and box["closed"] is True and box["volume_scope"] == "shape"
    assert box["volume_mm3"] == pytest.approx(1000.0)


@needs_ocp
@pytest.mark.parametrize("unit,name,factor", [("M", "m", 1000.0), ("CM", "cm", 10.0),
                                              ("FT", "ft", 304.8)])
def test_load_step_reports_metric_and_imperial_unit_names(tmp_path, unit, name, factor):
    """Issue 31: metre / centimetre / foot files report their unit and its
    factor instead of 'unknown'; OCCT's conversion gives the mm extents."""
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from src.drawing.brep_load import load_step, step_unit_sniff
    from src.io.occt_utils import init_step_statics
    p = str(tmp_path / f"box_{unit}.step")
    try:
        init_step_statics(unit=unit)
        w = STEPControl_Writer()
        w.Transfer(_box(25.0, 10.0, 10.0), STEPControl_AsIs)
        w.Write(p)
    finally:
        init_step_statics(unit="MM")
    assert "MILLI.,.METRE" not in open(p, encoding="latin-1").read() or unit == "FT"
    assert step_unit_sniff(p) == name
    m = load_step(p)
    assert m.source_unit == name and m.source_unit_factor == factor
    assert np.allclose(np.subtract(m.census["bbox_max"], m.census["bbox_min"]), [25, 10, 10],
                       atol=1e-6)
    assert not any("disagreement" in w_ for w_ in m.warnings)


def test_normalize_unit_table():
    from src.drawing.brep_load import normalize_unit, step_unit_sniff
    assert normalize_unit("millimetre") == ("mm", 1.0)
    assert normalize_unit("metre") == ("m", 1000.0)
    assert normalize_unit("M") == ("m", 1000.0)
    assert normalize_unit("INCH") == ("inch", 25.4)
    assert normalize_unit("FOOT") == ("ft", 304.8)
    assert normalize_unit("kilometre") == ("km", 1.0e6)
    assert normalize_unit("micrometre") == ("um", 1.0e-3)
    assert normalize_unit("PARSEC") == ("unknown", None)
    assert normalize_unit(None) == ("unknown", None)
    assert callable(step_unit_sniff)


@needs_ocp
def test_census_reports_instances_and_unique_solids():
    """Issue 32: one box placed twice is 2 solid instances but 1 unique solid."""
    from OCP.TopLoc import TopLoc_Location
    from OCP.gp import gp_Trsf, gp_Vec
    from src.drawing.brep_load import analyze_shape
    box = _box(10, 10, 10)
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(50.0, 0.0, 0.0))
    two = _compound([box, box.Moved(TopLoc_Location(tr))])
    c = analyze_shape(two).census
    assert c["solids"] == 2 and c["solids_unique"] == 1
    assert c["faces"] == 12 and c["faces_unique"] == 6
    # control: two independent boxes are two unique solids
    c2 = analyze_shape(_compound([_box(10, 10, 10), _box(5, 5, 5)])).census
    assert c2["solids"] == 2 and c2["solids_unique"] == 2 and c2["faces_unique"] == 12


def _bossed_slab(side):
    """Slab 100 (X) x 10 (Y, depth) x 60 (Z) with six r4 x 5 bosses on the +Y
    face (side=+1) or on the -Y face (side=-1)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 100.0, 10.0, 60.0).Shape()
    y0 = 10.0 if side > 0 else 0.0
    for x in (20.0, 50.0, 80.0):
        for z in (20.0, 40.0):
            boss = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y0, z), gp_Dir(0, side, 0)),
                                            4.0, 5.0).Shape()
            shape = BRepAlgoAPI_Fuse(shape, boss).Shape()
    return shape


@needs_ocp
def test_score_depth_sign_prefers_the_featured_side():
    """Issue 14: the front viewer stands on +n = -Y. Bosses on +Y face away
    from it (score_minus wins: turn 180 deg); bosses on -Y face it."""
    from src.drawing.brep_features import depth_sign_details, score_depth_sign
    n = (0.0, -1.0, 0.0)                               # VIEWS["front"].n
    back = _bossed_slab(+1)
    plus, minus = score_depth_sign(back, n)
    d = depth_sign_details(back, n)
    assert d["faces_minus"] == 7 and d["faces_plus"] == 1    # slab face + 6 boss tops vs 1
    assert minus > plus + 1.0
    front = _bossed_slab(-1)
    plus, minus = score_depth_sign(front, n)
    assert plus > minus + 1.0
    # an inverted solid gives the same answer (face orientation healed)
    assert score_depth_sign(back.Reversed(), n) == pytest.approx(score_depth_sign(back, n))
    # a through-holed plate has no preferred side
    plus, minus = score_depth_sign(_plate(), (0.0, 0.0, 1.0))
    assert plus > minus                                 # the boss on top
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    holed = BRepAlgoAPI_Cut(_box(50, 50, 10), BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(25, 25, -1), gp_Dir(0, 0, 1)), 5.0, 12.0).Shape()).Shape()
    plus, minus = score_depth_sign(holed, (0.0, 0.0, 1.0))
    assert plus == pytest.approx(minus)
    assert score_depth_sign(None, n) == (0.0, 0.0)


def _cross_bored_slab(xdir):
    """Slab 100 (X) x 10 (Y) x 60 (Z) with five r2 through bores along X at
    y = 5; ``xdir`` is the reference direction of the bore cylinders (a STEP
    file may set it to anything, the geometry is the same)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 100.0, 10.0, 60.0).Shape()
    for z in (10.0, 20.0, 30.0, 40.0, 50.0):
        tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-5, 5, z), gp_Dir(1, 0, 0), gp_Dir(*xdir)),
                                        2.0, 110.0).Shape()
        shape = BRepAlgoAPI_Cut(shape, tool).Shape()
    return shape


@needs_ocp
def test_depth_sign_ignores_the_cylinder_reference_direction(monkeypatch):
    """Review F2: a through bore whose axis is perpendicular to the view
    normal counts for neither side, whatever reference direction its cylinder
    has. With one normal read at the UV-box centre, XDir = +Y gave faces
    (plus 1, minus 6) and XDir = -Y gave (6, 1) for the same slab."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepTools import BRepTools
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from src.drawing.brep_features import depth_sign_details
    n = (0.0, -1.0, 0.0)
    results = []
    for xdir in ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)):
        slab = _cross_bored_slab(xdir)
        # precondition: five full (unsplit) bore faces, each spanning 360 deg of U
        exp, full = TopExp_Explorer(slab, TopAbs_FACE), 0
        while exp.More():
            f = TopoDS.Face_s(exp.Current())
            exp.Next()
            if BRepAdaptor_Surface(f).GetType() == GeomAbs_Cylinder:
                u0, u1, _, _ = BRepTools.UVBounds_s(f)
                full += abs((u1 - u0) - 2 * math.pi) < 1e-6
        assert full == 5
        d = depth_sign_details(slab, n)
        assert (d["faces_plus"], d["faces_minus"]) == (1, 1), (xdir, d)
        assert d["score_plus"] == pytest.approx(d["score_minus"])
        results.append((d["score_plus"], d["score_minus"]))
    assert results[0] == pytest.approx(results[1])
    # negative control: the single centre normal depends on the reference direction
    from src.drawing import brep_features
    monkeypatch.setattr(brep_features, "_mean_normal", False)
    d = depth_sign_details(_cross_bored_slab((0.0, 1.0, 0.0)), n)
    assert (d["faces_plus"], d["faces_minus"]) == (1, 6)


@needs_ocp
def test_depth_sign_counts_half_and_quarter_cylinders_by_their_mean_normal():
    """The mean (area-weighted) normal still counts an edge round that looks
    at the viewer: a box with its two +Y edges rounded (quarter cylinders, mean
    normal 45 deg off +Y, cos 0.64) gets them on the +Y side; a split through
    bore (two half cylinders facing +Y and -Y) counts one face per side."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosed
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from src.drawing.brep_features import depth_sign_details
    box = _box(40, 20, 30)
    mk = BRepFilletAPI_MakeFillet(box)
    edges = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(box, TopAbs_EDGE, edges)
    n_edges = 0
    for i in range(1, edges.Extent() + 1):
        e = TopoDS.Edge_s(edges.FindKey(i))
        c = BRep_Tool.Curve_s(e, 0.0, 1.0)
        p0, p1 = c.Value(0.0), c.Value(1.0)
        # edges along X at y = 20 (the +Y face's top and bottom edges)
        if abs(p0.Y() - 20) < 1e-9 and abs(p1.Y() - 20) < 1e-9 and abs(p0.Z() - p1.Z()) < 1e-9:
            mk.Add(3.0, e)
            n_edges += 1
    assert n_edges == 2                                   # precondition
    rounded = mk.Shape()
    d = depth_sign_details(rounded, (0.0, 1.0, 0.0))
    assert (d["faces_plus"], d["faces_minus"]) == (3, 1)   # +Y face + 2 rounds vs -Y face
    sd = ShapeUpgrade_ShapeDivideClosed(_cross_bored_slab((0.0, 0.0, 1.0)))
    sd.SetNbSplitPoints(1)
    sd.Perform()
    split = sd.Result()
    assert _count(split, _ta("FACE")) == 6 + 10           # precondition: bores split in halves
    d = depth_sign_details(split, (0.0, -1.0, 0.0))
    assert d["faces_plus"] == d["faces_minus"] == 6       # slab face + 5 bore halves per side


def _plate_with_fastener(hole_r=5.0, pin_r=5.0, pin_offset=0.0):
    """Plate 100x20x10 with a hole at (50,10) and a separate pin solid of
    radius ``pin_r`` through it (offset along X by ``pin_offset``)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    plate = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 100.0, 20.0, 10.0).Shape()
    plate = BRepAlgoAPI_Cut(plate, BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(50, 10, -5), gp_Dir(0, 0, 1)), hole_r, 20.0).Shape()).Shape()
    pin = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(50 + pin_offset, 10, -5), gp_Dir(0, 0, 1)),
                                   pin_r, 20.0).Shape()
    return plate, _compound([plate, pin])


@needs_ocp
def test_hole_filled_by_a_fastener_of_another_solid_is_marked(monkeypatch):
    """Issue 34: a hole that a bolt of another solid fills is 'filled_hole'
    (no callout); the plate alone, an off-axis pin and the switch-off keep it
    a 'hole'."""
    from src.drawing import brep_features
    plate, asm = _plate_with_fastener()
    assert [f.kind for f in brep_features.cylinder_features(plate)] == ["hole"]   # precondition
    kinds = sorted(f.kind for f in brep_features.cylinder_features(asm))
    assert kinds == ["boss", "filled_hole"]
    # a clearance hole r5.5 with an r5 bolt is filled too
    _, clr = _plate_with_fastener(hole_r=5.5, pin_r=5.0)
    assert sorted(f.kind for f in brep_features.cylinder_features(clr)) == ["boss", "filled_hole"]
    # a pin beside the hole, or a thin rod (r2 in r5), does not fill it
    _, beside = _plate_with_fastener(pin_offset=30.0)
    assert sorted(f.kind for f in brep_features.cylinder_features(beside)) == ["boss", "hole"]
    _, thin = _plate_with_fastener(pin_r=2.0)
    assert sorted(f.kind for f in brep_features.cylinder_features(thin)) == ["boss", "hole"]
    monkeypatch.setattr(brep_features, "_detect_filled", False)
    assert sorted(f.kind for f in brep_features.cylinder_features(asm)) == ["boss", "hole"]


# ---------------------------------------------------------------------------
# F7-perf: census of large assemblies
# ---------------------------------------------------------------------------

def _located_copies(n=30, dx=10.0, dy=20.0, dz=30.0):
    """A compound of ``n`` located copies (shared TShape) of one box, in a row."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf, gp_Vec
    box = _box(dx, dy, dz)
    copies = []
    for i in range(n):
        t = gp_Trsf()
        t.SetTranslation(gp_Vec(15.0 * i, 0.0, 0.0))
        copies.append(BRepBuilderAPI_Transform(box, t, False).Shape())
    return _compound(copies)


@needs_ocp
def test_fast_census_groups_located_copies_by_tshape(monkeypatch):
    """Above ``fast_above_faces`` the census runs BRepCheck and the volume once
    per unique solid (volume x instances) and the bounded exact bbox; at or
    below it (and with the default None) the exact whole-shape path runs."""
    from OCP.TopAbs import TopAbs_SOLID
    import src.io.step_audit as step_audit
    from src.drawing import brep_load
    asm = _located_copies()
    # precondition: 30 instances (180 faces) of ONE TShape
    assert _count(asm, TopAbs_SOLID) == 30
    calls = []
    real = step_audit.brep_volume

    def spy(shape):
        calls.append(shape.ShapeType())
        return real(shape)
    monkeypatch.setattr(step_audit, "brep_volume", spy)

    fast = brep_load.analyze_shape(asm, fast_above_faces=100).census
    assert fast["census_mode"] == "fast"
    assert fast["solids_unique"] == 1 and fast["solid_instances"] == 30
    assert fast["volume_method"] == "per_unique_solid"
    assert fast["volume_mm3"] == pytest.approx(30 * 6000.0, rel=1e-9)
    assert calls == [TopAbs_SOLID]                     # one integration, of the unique solid
    assert fast["bbox_source"] == "exact_bounded"
    assert np.allclose(fast["bbox_min"], [0, 0, 0], atol=1e-9)
    assert np.allclose(fast["bbox_max"], [15.0 * 29 + 10.0, 20.0, 30.0], atol=1e-9)
    assert fast["brepcheck_valid"] is True

    for gate in (None, 180, 1000):                     # 180 faces: not ABOVE the gate
        calls.clear()
        exact = brep_load.analyze_shape(asm, fast_above_faces=gate).census
        assert exact["census_mode"] == "exact", gate
        assert "solid_instances" not in exact and exact["bbox_source"] == "exact"
        assert calls and calls[0] != TopAbs_SOLID       # the whole compound was integrated
        assert exact["volume_mm3"] == pytest.approx(fast["volume_mm3"], rel=1e-9)
        assert exact["bbox_min"] == fast["bbox_min"] and exact["bbox_max"] == fast["bbox_max"]
        assert exact["brepcheck_valid"] is fast["brepcheck_valid"]


@needs_ocp
def test_bounded_exact_bbox_equals_add_optimal_on_curved_parts():
    """``exact_bbox_bounded`` gives the AddOptimal box bit for bit on a part with
    a NURBS cylinder (where the loose box is 10 mm too large) and on the plate."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from src.drawing.brep_load import exact_bbox, exact_bbox_bounded, loose_bbox
    nurbs = BRepBuilderAPI_NurbsConvert(BRepPrimAPI_MakeCylinder(10.0, 20.0).Shape(), True).Shape()
    for shape in (nurbs, _plate(), _compound([_plate(), nurbs])):
        ref = exact_bbox(shape)
        got = exact_bbox_bounded(shape)
        assert ref is not None and np.array_equal(got, ref)
    # precondition of the point of it: the loose box of the NURBS cylinder is NOT exact
    assert float(np.max(np.abs(loose_bbox(nurbs) - exact_bbox(nurbs)))) > 1.0
