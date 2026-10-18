"""Tests for the mesh route of the drawing module (task B3).

Covers ``src/drawing/mesh_load.py`` (checked STL loading, census,
prismatic/organic classifier), ``src/drawing/mesh_views.py`` (scanline depth
buffer, edge visibility with the 3x3 minimum window, mask outline) and
``src/drawing/mesh_sections.py`` (plane choice, section loops, nudge, body).

Expected values come from the mesh probe (REPORT-mesh.md): trimesh never
raises on bad STL input, the two-box occlusion split lands within a pixel of
z = +-3, organic outlines from the coverage mask are within 1 px, the torus
sections are circles of radius 3 at +-10 and rings 13 / 7, a coplanar plane
gives zero segments like a miss. No OCP import: the whole route is
trimesh + numpy + scikit-image.
"""
import math

import numpy as np
import pytest
import trimesh

from src.drawing import mesh_sections
from src.drawing.geometry2d import polygon_area
from src.drawing.mesh_load import (
    MeshModel,
    classify,
    load_mesh_model,
    load_stl_checked,
    mesh_census,
    transform_mesh,
)
from src.drawing.mesh_sections import choose_section_plane_mesh, section_mesh
from src.drawing.mesh_views import edge_visibility, mask_outline, project_mesh, raster_span
from src.drawing.model import Polyline, Segment
from src.drawing.projection import VIEWS


# ----------------------------------------------------------------- fixtures

def _box(extents=(60.0, 40.0, 20.0), offset=(0.0, 0.0, 0.0)) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents)
    m.apply_translation(offset)
    return m


def _two_boxes(with_front: bool = True) -> trimesh.Trimesh:
    """Back box x 0..30, y 0..10, z -4..6; front box x -5..5, y -10..-8, z -3..3.

    The viewer of the front view stands at -Y, so the front box hides the
    middle of the back box's left vertical edge (x = 0, y = 0, z -4..6).
    """
    back = _box((30.0, 10.0, 10.0), (15.0, 5.0, 1.0))
    if not with_front:
        return back
    front = _box((10.0, 2.0, 6.0), (0.0, -9.0, 0.0))
    return trimesh.util.concatenate([front, back])


def _blob() -> trimesh.Trimesh:
    """Taubin-smoothed noisy icosphere (the probe's organic blob, radius 1)."""
    rng = np.random.default_rng(0)
    blob = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    blob.vertices = blob.vertices + rng.normal(scale=0.02, size=blob.vertices.shape)
    trimesh.smoothing.filter_taubin(blob, iterations=10)
    return blob


def _hollow_box() -> trimesh.Trimesh:
    """60x40x20 box (y -20..20) with a closed 20x10x10 cavity at y 5..15."""
    outer = _box((60.0, 40.0, 20.0))
    cavity = _box((20.0, 10.0, 10.0), (0.0, 10.0, 0.0))
    cavity.invert()
    return trimesh.util.concatenate([outer, cavity])


def _segments(prims):
    return [p for p in prims if isinstance(p, Segment)]


def _on_vertical_line(seg: Segment, x: float, tol: float) -> bool:
    return abs(seg.a[0] - x) <= tol and abs(seg.b[0] - x) <= tol


def _yrange(seg: Segment):
    return tuple(sorted((float(seg.a[1]), float(seg.b[1]))))


def _fit_circle(loop: np.ndarray):
    c = loop.mean(axis=0)
    r = np.linalg.norm(loop - c, axis=1)
    return c, float(r.mean()), float(r.min()), float(r.max())


# --------------------------------------------------------------- mesh_load

def test_load_stl_checked_reads_binary_and_ascii(tmp_path):
    box = _box()
    pb = tmp_path / "box_bin.stl"
    pa = tmp_path / "box_ascii.stl"
    box.export(str(pb))
    box.export(str(pa), file_type="stl_ascii")
    for p in (pb, pa):
        m = load_stl_checked(str(p))
        assert isinstance(m, trimesh.Trimesh)
        assert len(m.faces) == 12
        assert len(m.vertices) == 8          # process=True merged the vertices
        assert len(m.face_adjacency) == 18   # adjacency works only after the merge


def test_load_stl_checked_rejects_empty_file(tmp_path):
    p = tmp_path / "empty.stl"
    p.write_bytes(b"")
    assert p.stat().st_size == 0
    with pytest.raises(ValueError, match="empty"):
        load_stl_checked(str(p))


def test_load_stl_checked_rejects_step_text(tmp_path):
    p = tmp_path / "fake.stl"
    p.write_bytes(b"ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'2;1');\nENDSEC;\nDATA;\nENDSEC;\n"
                  b"END-ISO-10303-21;\n")
    # precondition: trimesh itself does not complain about it
    raw = trimesh.load(str(p), force="mesh", file_type="stl")
    assert len(raw.faces) == 0
    with pytest.raises(ValueError, match="STEP"):
        load_stl_checked(str(p))


def test_load_stl_checked_rejects_truncated_binary(tmp_path):
    good = tmp_path / "good.stl"
    _box().export(str(good))
    data = good.read_bytes()
    p = tmp_path / "trunc.stl"
    p.write_bytes(data[:-10])
    # precondition: trimesh silently returns an empty mesh for it
    raw = trimesh.load(str(p), force="mesh")
    assert len(raw.faces) == 0
    with pytest.raises(ValueError):
        load_stl_checked(str(p))


def test_mesh_census_box():
    c = mesh_census(_box())
    assert c["triangles"] == 12
    assert c["vertices"] == 8
    assert c["watertight"] is True
    assert c["winding_consistent"] is True
    assert c["components"] == 1
    assert c["volume_mm3"] == pytest.approx(48000.0, rel=1e-9)
    assert c["area_mm2"] == pytest.approx(2 * (60 * 40 + 60 * 20 + 40 * 20), rel=1e-9)
    assert np.allclose(c["bbox_min"], (-30, -20, -10))
    assert np.allclose(c["bbox_max"], (30, 20, 10))
    assert c["euler"] == 2
    assert c["degenerate_faces"] == 0
    assert c["duplicate_faces"] == 0
    assert c["inverted"] is False


def test_mesh_census_counts_defects():
    box = _box()
    # duplicate one face and add one degenerate (zero-area) face
    faces = np.vstack([box.faces, box.faces[:1], [[0, 0, 1]]])
    bad = trimesh.Trimesh(box.vertices.copy(), faces, process=True)
    assert len(bad.faces) == 14                      # precondition: trimesh kept both
    c = mesh_census(bad)
    assert c["duplicate_faces"] == 1
    assert c["degenerate_faces"] >= 1

    two = trimesh.util.concatenate([_box(), _box(offset=(100.0, 0.0, 0.0))])
    assert mesh_census(two)["components"] == 2

    open_box = trimesh.Trimesh(box.vertices.copy(), box.faces[1:].copy())
    assert not open_box.is_watertight               # precondition
    co = mesh_census(open_box)
    assert co["watertight"] is False
    assert co["volume_mm3"] is None
    assert co["boundary_edges"] == 3


def test_mesh_census_inside_out_lattice_volume_positive():
    try:
        from tests.test_high_genus_closed_cage import _jungle_gym
        jg = _jungle_gym()
    except Exception as exc:  # pragma: no cover - optional repo fixture
        pytest.skip(f"_jungle_gym unavailable: {exc}")
    assert jg.is_watertight
    assert jg.volume < 0          # precondition: the repo lattice is inside out
    c = mesh_census(jg)
    assert c["volume_mm3"] is not None and c["volume_mm3"] > 0
    assert c["volume_mm3"] == pytest.approx(abs(float(jg.volume)), rel=1e-9)
    assert c["inverted"] is True


def test_classify_prismatic_vs_organic():
    box = classify(_box())
    assert box["planar_big"] >= 0.2
    assert box["verdict"] == "prismatic"
    assert box["threshold"] == pytest.approx(0.10)

    ico = classify(trimesh.creation.icosphere(subdivisions=3))
    assert ico["planar_big"] == 0.0
    assert ico["verdict"] == "organic"

    blob = classify(_blob())
    assert blob["planar_big"] == 0.0
    assert blob["verdict"] == "organic"

    cyl = classify(trimesh.creation.cylinder(radius=5.0, height=20.0, sections=48))
    assert cyl["verdict"] == "prismatic"
    assert 0.0 <= cyl["sharp_turn"] <= 1.0
    assert box["sharp_turn"] == pytest.approx(1.0)      # every turning edge of a box is 90 deg


def test_classify_verdict_follows_threshold():
    c = classify(_box(), threshold=1.01)
    assert c["planar_big"] <= 1.0 + 1e-12                # precondition: cannot reach 1.01
    assert c["verdict"] == "organic"
    assert c["threshold"] == pytest.approx(1.01)


def test_transform_mesh_permutes_axes_and_copies():
    box = _box()
    m = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])   # new x = old z ...
    assert np.linalg.det(m) == pytest.approx(1.0)
    out = transform_mesh(box, m)
    assert np.allclose(out.extents, (20.0, 60.0, 40.0))
    assert np.allclose(box.extents, (60.0, 40.0, 20.0))  # input untouched
    assert out.volume == pytest.approx(48000.0)          # winding still outward


def test_load_mesh_model_reports_warnings(tmp_path):
    box = _box()
    open_box = trimesh.Trimesh(box.vertices.copy(), box.faces[1:].copy())
    p = tmp_path / "open.stl"
    open_box.export(str(p))
    mm = load_mesh_model(str(p), 0.10)
    assert isinstance(mm, MeshModel)
    assert mm.census["watertight"] is False
    assert mm.classifier["verdict"] in ("prismatic", "organic")
    assert any("watertight" in w for w in mm.warnings)

    p2 = tmp_path / "box.stl"
    box.export(str(p2))
    mm2 = load_mesh_model(str(p2), 0.10)
    assert mm2.warnings == []
    assert mm2.classifier["verdict"] == "prismatic"


# --------------------------------------------------------------- raster_span

def _brute_force_raster(xy, depth, faces, grid):
    """Per-triangle barycentric fill at pixel centres (independent checker)."""
    H, W, x0, y0, px = grid["H"], grid["W"], grid["x0"], grid["y0"], grid["px"]
    cx = x0 + (np.arange(W) + 0.5) * px
    cy = y0 + (np.arange(H) + 0.5) * px
    X, Y = np.meshgrid(cx, cy)
    buf = np.full((H, W), -np.inf)
    for f in faces:
        (ax, ay), (bx, by), (qx, qy) = xy[f[0]], xy[f[1]], xy[f[2]]
        den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
        if abs(den) < 1e-15:
            continue
        l1 = ((by - qy) * (X - qx) + (qx - bx) * (Y - qy)) / den
        l2 = ((qy - ay) * (X - qx) + (ax - qx) * (Y - qy)) / den
        l3 = 1.0 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        z = l1 * depth[f[0]] + l2 * depth[f[1]] + l3 * depth[f[2]]
        buf = np.where(inside, np.maximum(buf, z), buf)
    return buf


def test_raster_span_matches_brute_force_fill():
    rng = np.random.default_rng(7)
    xy = rng.uniform(0.0, 10.0, size=(60, 2)) + np.array([math.pi, -math.e])
    depth = rng.uniform(-3.0, 3.0, size=60)
    faces = np.arange(60).reshape(20, 3)
    buf, mask, grid = raster_span(xy, depth, faces, 96)
    assert set(("x0", "y0", "px")) <= set(grid)
    assert buf.shape == mask.shape == (grid["H"], grid["W"])
    ref = _brute_force_raster(xy, depth, faces, grid)
    ref_mask = np.isfinite(ref)
    assert 0 < ref_mask.sum() < ref_mask.size       # precondition: partial coverage
    assert np.array_equal(mask, ref_mask)
    assert np.array_equal(mask, np.isfinite(buf))
    assert np.max(np.abs(buf[mask] - ref[mask])) < 1e-9


def test_raster_span_long_side_and_margin():
    box = _box()
    v = VIEWS["front"].to_view(box.vertices)
    buf, mask, grid = raster_span(v[:, :2], v[:, 2], box.faces, 200, margin_px=4)
    assert max(grid["W"], grid["H"]) <= 200 + 1
    cols = np.nonzero(mask.any(axis=0))[0]
    rows = np.nonzero(mask.any(axis=1))[0]
    x_lo = grid["x0"] + cols[0] * grid["px"]
    x_hi = grid["x0"] + (cols[-1] + 1) * grid["px"]
    y_lo = grid["y0"] + rows[0] * grid["px"]
    y_hi = grid["y0"] + (rows[-1] + 1) * grid["px"]
    assert x_lo == pytest.approx(-30.0, abs=grid["px"])
    assert x_hi == pytest.approx(30.0, abs=grid["px"])
    assert y_lo == pytest.approx(-10.0, abs=grid["px"])     # front view y = world Z
    assert y_hi == pytest.approx(10.0, abs=grid["px"])
    assert np.allclose(buf[mask], 20.0)                      # nearest face y = -20 -> depth +20


# ----------------------------------------------------------- edge_visibility

def test_edge_visibility_splits_a_segment_behind_a_square():
    # square x 0..10, y 0..10 at depth 0
    xy = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [-6, -1], [16, -1]], float)
    depth = np.array([0, 0, 0, 0, -50, -50], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    buf, mask, grid = raster_span(xy, depth, faces, 400)
    px = grid["px"]
    behind = np.array([[[-5.0, 5.0, -5.0], [15.0, 5.0, -5.0]]])
    vis, hid = edge_visibility(behind, buf, grid)
    assert vis.shape[1:] == (2, 2) and hid.shape[1:] == (2, 2)
    assert len(hid) == 1
    hx = sorted(hid[0][:, 0])
    assert hx[0] == pytest.approx(0.0, abs=2.5 * px)
    assert hx[1] == pytest.approx(10.0, abs=2.5 * px)
    assert len(vis) == 2
    total = sum(np.linalg.norm(s[1] - s[0]) for s in vis) + np.linalg.norm(hid[0][1] - hid[0][0])
    assert total == pytest.approx(20.0, abs=1e-9)
    infront = behind.copy()
    infront[:, :, 2] = 5.0
    vis2, hid2 = edge_visibility(infront, buf, grid)
    assert len(hid2) == 0 and len(vis2) == 1


# -------------------------------------------------------------- mask_outline

def test_mask_outline_square_ring_in_view_coordinates():
    mask = np.zeros((60, 80), bool)
    mask[10:50, 10:70] = True       # rows = y, cols = x
    mask[20:40, 30:50] = False
    grid = {"x0": 100.0, "y0": -50.0, "px": 0.5, "W": 80, "H": 60}
    loops = mask_outline(mask, grid)
    assert len(loops) == 2
    for L in loops:
        assert L.shape[1] == 2 and len(L) >= 3
        assert not np.array_equal(L[0], L[-1])          # no repeated end point
    areas = sorted(abs(polygon_area(L)) for L in loops)
    # pixel-centre contours sit half a pixel inside/outside the pixel edges
    assert areas[1] == pytest.approx(30.0 * 20.0, rel=0.05)
    assert areas[0] == pytest.approx(10.0 * 10.0, rel=0.12)
    outer = max(loops, key=lambda L: abs(polygon_area(L)))
    assert outer[:, 0].min() == pytest.approx(100.0 + 10 * 0.5, abs=0.5)
    assert outer[:, 1].min() == pytest.approx(-50.0 + 10 * 0.5, abs=0.5)


# --------------------------------------------------------------- project_mesh

def test_project_mesh_box_front_bbox_and_no_hidden():
    pv = project_mesh(_box(), VIEWS["front"], raster_px=512, feature_angle_deg=30.0, organic=False)
    assert pv.algorithm == "depth_buffer"
    assert pv.spec is VIEWS["front"]
    assert (pv.bbox.xmin, pv.bbox.ymin, pv.bbox.xmax, pv.bbox.ymax) == pytest.approx((-30, -10, 30, 10))
    segs = _segments(pv.visible)
    assert len(segs) == 4                            # back edges deduped onto the front ones
    assert pv.hidden == []


def test_project_mesh_two_boxes_occlusion_split():
    mesh = _two_boxes(True)
    spec = VIEWS["front"]
    # precondition: the front box is nearer and covers x = 0 over z -3..3
    v = spec.to_view(mesh.vertices)
    front_part = v[:8] if v[:8, 2].max() > v[8:, 2].max() else v[8:]
    assert front_part[:, 2].min() > v[:, 2].min()
    assert front_part[:, 0].min() < 0 < front_part[:, 0].max()
    pv = project_mesh(mesh, spec, raster_px=1024, feature_angle_deg=30.0, organic=False)
    px = pv.stats["px"]
    tol = 0.5 * px
    vis = [s for s in _segments(pv.visible) if _on_vertical_line(s, 0.0, tol)]
    hid = [s for s in _segments(pv.hidden) if _on_vertical_line(s, 0.0, tol)]
    assert len(vis) == 2 and len(hid) == 1
    lo_vis, hi_vis = sorted((_yrange(s) for s in vis))
    assert lo_vis[0] == pytest.approx(-4.0, abs=2 * px)
    assert lo_vis[1] == pytest.approx(-3.0, abs=2 * px)
    assert hi_vis[0] == pytest.approx(3.0, abs=2 * px)
    assert hi_vis[1] == pytest.approx(6.0, abs=2 * px)
    h = _yrange(hid[0])
    assert h[0] == pytest.approx(-3.0, abs=2 * px)
    assert h[1] == pytest.approx(3.0, abs=2 * px)


def test_project_mesh_two_boxes_negative_control_without_occluder():
    pv = project_mesh(_two_boxes(False), VIEWS["front"], raster_px=1024, feature_angle_deg=30.0,
                      organic=False)
    tol = 0.5 * pv.stats["px"]
    vis = [s for s in _segments(pv.visible) if _on_vertical_line(s, 0.0, tol)]
    assert len(vis) == 1
    assert _yrange(vis[0]) == pytest.approx((-4.0, 6.0), abs=1e-9)
    assert pv.hidden == []


def _cylinder_with_side_blocks(shift: float) -> trimesh.Trimesh:
    """Vertical 48-gon cylinder r 10, h 40 plus two small blocks far behind it.

    The blocks push the view bbox to irrational x limits so the cylinder's
    silhouette lines (x = +-10) fall at an arbitrary place inside a pixel
    instead of exactly on the raster's pixel boundary (at the bbox limit a
    single-pixel test passes by floating-point luck).
    """
    cyl = trimesh.creation.cylinder(radius=10.0, height=40.0, sections=48)
    left = _box((1.0, 1.0, 1.0), (-10.0 - math.pi - shift, 30.0, 0.0))
    right = _box((1.0, 1.0, 1.0), (10.0 + math.e, 30.0, 0.0))
    return trimesh.util.concatenate([cyl, left, right])


def _cyl_silhouettes(pv):
    tol = 1e-6
    return [s for s in _segments(pv.visible)
            if (_on_vertical_line(s, 10.0, tol) or _on_vertical_line(s, -10.0, tol))
            and _yrange(s) == pytest.approx((-20.0, 20.0), abs=1e-9)]


def test_project_mesh_cylinder_silhouettes_need_the_min_window(monkeypatch):
    from src.drawing import mesh_views

    shifts = (0.0, 0.137, 0.291, 0.443, 0.619, 0.811)
    meshes = [_cylinder_with_side_blocks(s) for s in shifts]
    for mesh in meshes:
        pv = project_mesh(mesh, VIEWS["front"], raster_px=1024, feature_angle_deg=30.0, organic=False)
        assert len(_cyl_silhouettes(pv)) == 2
        assert not [s for s in _segments(pv.hidden) if abs(abs(s.a[0]) - 10.0) < 1e-6]
    # negative control: a single-pixel depth test (nbhd=0) loses a silhouette
    # whenever it lies in the inner half of its pixel (a nearer facet owns the
    # pixel centre); over six raster phases that must happen at least once
    orig = mesh_views.edge_visibility
    monkeypatch.setattr(mesh_views, "edge_visibility",
                        lambda s, b, g, tol_px=1.5, nbhd=1: orig(s, b, g, tol_px=tol_px, nbhd=0))
    found = [len(_cyl_silhouettes(project_mesh(m, VIEWS["front"], raster_px=1024,
                                               feature_angle_deg=30.0, organic=False)))
             for m in meshes]
    assert sum(found) < 2 * len(meshes)


def test_project_mesh_fine_cylinder_side_view_is_a_rectangle():
    cyl = trimesh.creation.cylinder(radius=10.0, height=40.0, sections=400)
    pv = project_mesh(cyl, VIEWS["front"], raster_px=1024, feature_angle_deg=30.0, organic=False)
    segs = _segments(pv.visible)
    assert len(segs) == 4                       # 800 rim edges merged onto 2 lines + 2 silhouettes
    assert pv.hidden == []
    lengths = sorted(math.dist(s.a, s.b) for s in segs)
    assert lengths == pytest.approx([20.0, 20.0, 40.0, 40.0], abs=1e-6)


def test_premerge_collinear_unions_runs_only_on_one_line():
    from src.drawing.mesh_views import premerge_collinear

    segs = np.array([
        [[0, 0], [5, 0]], [[3, 0], [9, 0]], [[9, 0], [12, 0]],     # one run 0..12
        [[20, 0], [25, 0]],                                       # gap -> own run
        [[0, 1], [5, 1]],                                         # parallel line
        [[5, 0], [0, 0]],                                         # reversed duplicate
    ], float)
    out = premerge_collinear(segs, 1e-6)
    spans = sorted((round(float(min(s[:, 0])), 9), round(float(max(s[:, 0])), 9), float(s[0, 1])) for s in out)
    assert spans == [(0.0, 5.0, 1.0), (0.0, 12.0, 0.0), (20.0, 25.0, 0.0)]
    # chords of a circle are never merged (different directions)
    t = np.linspace(0, math.pi / 2, 30)
    pts = np.stack([10 * np.cos(t), 10 * np.sin(t)], axis=1)
    chords = np.stack([pts[:-1], pts[1:]], axis=1)
    assert len(premerge_collinear(chords, 1e-3)) == len(chords)


def test_project_mesh_hidden_flag():
    mesh = _two_boxes(True)
    on = project_mesh(mesh, VIEWS["front"], raster_px=512, feature_angle_deg=30.0, organic=False)
    assert len(on.hidden) > 0                       # precondition
    off = project_mesh(mesh, VIEWS["front"], raster_px=512, feature_angle_deg=30.0, organic=False,
                       hidden=False)
    assert off.hidden == []
    assert len(off.visible) == len(on.visible)


def test_project_mesh_organic_sphere_outline():
    r = 20.0
    ico = trimesh.creation.icosphere(subdivisions=4, radius=r)
    pv = project_mesh(ico, VIEWS["iso"], raster_px=1024, feature_angle_deg=30.0, organic=True)
    polys = [p for p in pv.visible if isinstance(p, Polyline)]
    assert len(polys) == 1 and len(pv.visible) == 1
    assert polys[0].closed
    rad = np.linalg.norm(polys[0].points, axis=1)
    px = pv.stats["px"]
    assert np.all(np.abs(rad - r) <= 1.5 * px)
    assert pv.hidden == []
    assert "outline_s" in pv.stats
    # the organic flag is load-bearing: without it the outline is silhouette segments
    pv2 = project_mesh(ico, VIEWS["iso"], raster_px=1024, feature_angle_deg=30.0, organic=False)
    assert not any(isinstance(p, Polyline) for p in pv2.visible)
    assert len(_segments(pv2.visible)) > 10


def test_project_mesh_organic_torus_top_two_loops():
    torus = trimesh.creation.torus(10.0, 3.0)
    pv = project_mesh(torus, VIEWS["top"], raster_px=1024, feature_angle_deg=30.0, organic=True)
    polys = [p for p in pv.visible if isinstance(p, Polyline) and p.closed]
    assert len(polys) == 2
    radii = sorted(float(np.linalg.norm(p.points, axis=1).mean()) for p in polys)
    assert radii[0] == pytest.approx(7.0, rel=0.02)
    assert radii[1] == pytest.approx(13.0, rel=0.02)
    assert pv.hidden == []


# ------------------------------------- QA round 1 fixes (task F1-views-perf)

def _plate_over_cylinder() -> trimesh.Trimesh:
    """60x60x5 plate on top of a 48-gon cylinder r 10: in the top view both rims are hidden."""
    plate = trimesh.creation.box(extents=(60.0, 60.0, 5.0))
    plate.apply_translation((0.0, 0.0, 2.5))
    cyl = trimesh.creation.cylinder(radius=10.0, height=20.0, sections=48)
    cyl.apply_translation((0.0, 0.0, -10.0))
    return trimesh.util.concatenate([plate, cyl])


def _noisy_sphere(seed: int = 0) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
    m.vertices = m.vertices + np.random.default_rng(seed).normal(0.0, 0.002 * 20.0, m.vertices.shape)
    return m


def _prim_length(p, px):
    from src.drawing.geometry2d import primitives_to_segments
    s = primitives_to_segments([p], px)
    return float(np.linalg.norm(s[:, 1] - s[:, 0], axis=1).sum())


def test_project_mesh_hidden_circle_is_one_polyline(monkeypatch):
    """Issue 6: the 48 hidden rim chords were 48 Segments (no dash pattern)."""
    from src.drawing import mesh_views
    pv = project_mesh(_plate_over_cylinder(), VIEWS["top"], raster_px=1024, feature_angle_deg=30.0,
                      organic=False)
    polys = [p for p in pv.hidden if isinstance(p, Polyline)]
    assert len(pv.hidden) <= 2 and polys
    assert max(len(p.points) for p in polys) >= 40
    rad = np.linalg.norm(np.vstack([p.points for p in polys]), axis=1)
    assert np.allclose(rad, 10.0, atol=3 * pv.stats["px"])
    assert len(_segments(pv.visible)) == 4                 # plate outline stays 4 Segments
    # negative control: without chaining the circle is dozens of chord Segments
    orig = mesh_views.chain_to_primitives
    monkeypatch.setattr(mesh_views, "chain_to_primitives",
                        lambda *a, **k: orig(*a, **dict(k, short_len=0.0, polyline_all=False)))
    pv0 = project_mesh(_plate_over_cylinder(), VIEWS["top"], raster_px=1024, feature_angle_deg=30.0,
                       organic=False)
    assert len(_segments(pv0.hidden)) >= 40


def test_project_mesh_organic_torus_side_has_hidden_inner_silhouettes(monkeypatch):
    """Issue 13: organic meshes drew no interior or hidden silhouettes."""
    from src.drawing import mesh_views
    torus = trimesh.creation.torus(10.0, 3.0)
    pv = project_mesh(torus, VIEWS["front"], raster_px=1024, feature_angle_deg=30.0, organic=True)
    assert pv.stats["interior_silhouettes_omitted"] is False
    assert len(pv.hidden) > 0
    # the hidden inner silhouette lies inside the hole's span |x| < 7 + 3
    pts = np.vstack([np.asarray(p.points if isinstance(p, Polyline) else [p.a, p.b], float)
                     for p in pv.hidden])
    assert np.abs(pts[:, 0]).max() < 10.0
    # the outline is still one closed loop and no silhouette run duplicates it
    loops = [p for p in pv.visible if isinstance(p, Polyline) and p.closed]
    assert len(loops) == 1 and len(pv.visible) == 1
    # skipping silhouettes is disclosed (and then nothing is hidden)
    monkeypatch.setattr(mesh_views, "_MAX_ORGANIC_SILHOUETTE_FACES", 0)
    pv0 = project_mesh(torus, VIEWS["front"], raster_px=1024, feature_angle_deg=30.0, organic=True)
    assert pv0.stats["interior_silhouettes_omitted"] is True and pv0.hidden == []


def test_project_mesh_noisy_sphere_has_no_noise_fragments(monkeypatch):
    """Issue 24: crease flicker and noise gave fragments shorter than 2 px and a staircase outline."""
    from src.drawing import mesh_views
    m = _noisy_sphere()
    pv = project_mesh(m, VIEWS["front"], raster_px=1024, feature_angle_deg=30.0, organic=False)
    px = pv.stats["px"]
    assert len(pv.visible) > 10                            # precondition: silhouettes drawn
    assert not [p for p in pv.visible + pv.hidden if _prim_length(p, px) < 2 * px]
    org = project_mesh(m, VIEWS["front"], raster_px=1024, feature_angle_deg=30.0, organic=True)
    assert len(org.visible) == 1                           # outline only, fold noise removed
    assert len(org.visible[0].points) < 120                # was 171 with the pixel staircase
    # negative control: without the chain-length filter and hysteresis fragments come back
    # (and without the round-3 dangling-stub and fan-chord filters, which also remove them)
    monkeypatch.setattr(mesh_views, "_MIN_CHAIN_PX", 0.0)
    monkeypatch.setattr(mesh_views, "_DANGLING_PAPER_MM", 0.0)
    monkeypatch.setattr(mesh_views, "drop_fan_duplicates", lambda p, t, *a, **k: (list(p), 0))
    orig = mesh_views.edge_visibility
    monkeypatch.setattr(mesh_views, "edge_visibility",
                        lambda s, b, g, tol_px=1.5, nbhd=1: orig(s, b, g, tol_px=tol_px, nbhd=nbhd,
                                                                 hysteresis=False))
    pv0 = project_mesh(m, VIEWS["front"], raster_px=1024, feature_angle_deg=30.0, organic=False)
    assert [p for p in pv0.visible + pv0.hidden if _prim_length(p, px) < 2 * px]


def test_hysteresis_absorbs_single_sample_flips():
    from src.drawing.mesh_views import _hysteresis
    vis = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0], bool)
    rep = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    out = _hysteresis(vis, rep)
    assert out[:5].all()                                  # inner single flip absorbed
    assert not out[5:10].any()
    # segment 2: two samples of each state is a real switch, kept
    assert out[10:14].tolist() == [True, True, False, False]
    # an end sample alone against the next two flips
    out2 = _hysteresis(np.array([0, 1, 1, 1, 1, 0], bool), np.zeros(6, int))
    assert out2.all()
    assert (out != vis).sum() == 2                         # exactly the two lone samples changed


def test_mask_outline_smoothing_removes_the_staircase():
    yy, xx = np.mgrid[0:200, 0:200]
    mask = (yy - 0.37 * xx > 20) & (yy - 0.37 * xx < 120) & (xx > 20) & (xx < 180)
    grid = {"x0": 0.0, "y0": 0.0, "px": 1.0, "W": 200, "H": 200}
    raw = mask_outline(mask, grid, smooth_px=0.0)
    smooth = mask_outline(mask, grid)
    assert len(raw) == len(smooth) == 1
    assert len(raw[0]) > 12                               # precondition: staircase kept
    assert len(smooth[0]) <= 8                            # a quadrilateral, few extra corners
    assert abs(polygon_area(smooth[0])) == pytest.approx(abs(polygon_area(raw[0])), rel=0.03)


def test_project_mesh_time_budget_drops_hidden():
    mesh = _two_boxes(True)
    on = project_mesh(mesh, VIEWS["front"], raster_px=512, feature_angle_deg=30.0, organic=False)
    assert on.hidden and on.stats["hidden_dropped"] is False          # precondition
    pv = project_mesh(mesh, VIEWS["front"], raster_px=512, feature_angle_deg=30.0, organic=False,
                      time_budget_s=1e-9)
    assert pv.hidden == [] and pv.stats["hidden_dropped"] is True
    assert pv.stats["hidden_dropped_reason"] == "time_budget"
    assert len(pv.visible) == len(on.visible)


# ----------------------------------------------------------------- sections

def test_choose_section_plane_mesh_auto_and_modes():
    box = _box(offset=(5.0, 6.0, 7.0))
    p, n, marker = choose_section_plane_mesh(box, "auto")
    assert np.allclose(p, (5.0, 6.0, 7.0))
    assert np.allclose(n, (0.0, -1.0, 0.0))
    assert marker == "top"
    p, n, marker = choose_section_plane_mesh(box, "top")
    assert np.allclose(n, VIEWS["top"].n) and marker == "front"
    p, n, marker = choose_section_plane_mesh(box, "right")
    assert np.allclose(n, VIEWS["right"].n)
    with pytest.raises(ValueError):
        choose_section_plane_mesh(box, "sideways")


def test_section_mesh_torus_through_axis():
    torus = trimesh.creation.torus(10.0, 3.0)
    sv = section_mesh(torus, np.zeros(3), np.array([0.0, -1.0, 0.0]), VIEWS["front"], raster_px=512,
                      feature_angle_deg=30.0, organic=True, depth_extent=26.0)
    assert len(sv.loops) == 2
    assert sv.stats["nudged"] is False
    assert sv.stats["open_wires_dropped"] == 0
    centres = []
    for L in sv.loops:
        assert not np.array_equal(L[0], L[-1])
        c, r, _, _ = _fit_circle(L)
        assert r == pytest.approx(3.0, rel=0.02)
        assert c[1] == pytest.approx(0.0, abs=0.05)
        centres.append(c[0])
    assert sorted(centres) == pytest.approx([-10.0, 10.0], abs=0.05)
    assert np.allclose(sv.plane_normal, VIEWS["front"].n)
    assert sv.body is not None and len(sv.body.visible) > 0


def test_section_mesh_torus_perpendicular_rings():
    torus = trimesh.creation.torus(10.0, 3.0)
    sv = section_mesh(torus, np.zeros(3), np.array([0.0, 0.0, 1.0]), VIEWS["top"], raster_px=512,
                      feature_angle_deg=30.0, organic=True, depth_extent=6.0)
    assert len(sv.loops) == 2
    radii = sorted(_fit_circle(L)[1] for L in sv.loops)
    assert radii[0] == pytest.approx(7.0, rel=0.02)
    assert radii[1] == pytest.approx(13.0, rel=0.02)
    for L in sv.loops:
        assert np.allclose(L.mean(axis=0), 0.0, atol=0.1)    # concentric


def test_section_mesh_coplanar_face_is_nudged():
    from trimesh.intersections import mesh_plane

    box = _box()
    point = np.array([0.0, 0.0, 10.0])
    normal = np.array([0.0, 0.0, 1.0])
    # precondition: the plane on the top face gives no segments (like a miss)
    assert len(mesh_plane(box, plane_normal=normal, plane_origin=point)) == 0
    sv = section_mesh(box, point, normal, VIEWS["top"], raster_px=256, feature_angle_deg=30.0,
                      organic=False, depth_extent=20.0)
    assert sv.stats["nudged"] is True
    assert len(sv.loops) == 1
    assert abs(polygon_area(sv.loops[0])) == pytest.approx(2400.0, rel=1e-6)
    assert sv.plane_point[2] < 10.0                          # moved into the material
    assert sv.plane_point[2] == pytest.approx(10.0 - 0.2, abs=1e-9)


def test_section_mesh_miss_gives_empty_loops():
    box = _box()
    sv = section_mesh(box, np.array([0.0, 500.0, 0.0]), np.array([0.0, -1.0, 0.0]), VIEWS["front"],
                      raster_px=256, feature_angle_deg=30.0, organic=False, depth_extent=40.0)
    assert sv.loops == []
    assert sv.stats["nudged"] is True


def test_section_mesh_box_centre_one_loop_in_view_coords():
    box = _box()
    sv = section_mesh(box, np.zeros(3), np.array([0.0, -1.0, 0.0]), VIEWS["front"], raster_px=256,
                      feature_angle_deg=30.0, organic=False, depth_extent=40.0)
    assert sv.stats["nudged"] is False
    assert len(sv.loops) == 1
    L = sv.loops[0]
    assert (L[:, 0].min(), L[:, 0].max(), L[:, 1].min(), L[:, 1].max()) == pytest.approx((-30, 30, -10, 10))
    assert sv.marker_view == "top"


def _explode(prims):
    """Polylines as their edge Segments (hidden chains are Polylines, QA round 2 issue 9)."""
    out = []
    for p in prims:
        if isinstance(p, Polyline):
            pts = np.asarray(p.points, float).reshape(-1, 2)
            ring = np.vstack([pts, pts[:1]]) if p.closed else pts
            out += [Segment(tuple(a), tuple(b), p.style) for a, b in zip(ring[:-1], ring[1:])]
        else:
            out.append(p)
    return out


def _cavity_wall_segments(prims, px):
    tol = 0.5 * px
    out = []
    for s in _segments(_explode(prims)):
        for x in (-10.0, 10.0):
            if _on_vertical_line(s, x, tol):
                lo, hi = _yrange(s)
                if lo > -5.0 - 2 * px and hi < 5.0 + 2 * px and hi - lo > 5.0:
                    out.append(s)
    return out


def test_section_body_cap_hides_material_behind_cut(monkeypatch):
    mesh = _hollow_box()
    point, normal = np.zeros(3), np.array([0.0, -1.0, 0.0])
    sv = section_mesh(mesh, point, normal, VIEWS["front"], raster_px=512, feature_angle_deg=30.0,
                      organic=False, depth_extent=40.0)
    assert len(sv.loops) == 1                         # the cut misses the cavity (y 5..15)
    px = sv.body.stats["px"]
    hid = _cavity_wall_segments(sv.body.hidden, px)
    vis = _cavity_wall_segments(sv.body.visible, px)
    assert len(hid) == 2 and len(vis) == 0
    # negative control: without the cap the cavity shows through the open cut
    monkeypatch.setattr(mesh_sections, "_CAP_SECTION", False)
    sv2 = section_mesh(mesh, point, normal, VIEWS["front"], raster_px=512, feature_angle_deg=30.0,
                       organic=False, depth_extent=40.0)
    px2 = sv2.body.stats["px"]
    assert len(_cavity_wall_segments(sv2.body.visible, px2)) == 2


# ===================================================== QA round 1 (task F4)

def _edge_sharing_boxes(inverted: bool = False) -> trimesh.Trimesh:
    """Two 10 mm cubes sharing exactly one vertical edge (x = y = 10).

    After the vertex merge that edge carries four faces: one non-manifold
    edge, zero boundary edges. ``inverted`` reverses every face.
    """
    a = _box((10.0, 10.0, 10.0), (5.0, 5.0, 5.0))
    b = _box((10.0, 10.0, 10.0), (15.0, 15.0, 5.0))
    m = trimesh.util.concatenate([a, b])
    faces = m.faces[:, ::-1] if inverted else m.faces
    return trimesh.Trimesh(m.vertices.copy(), np.ascontiguousarray(faces), process=True)


def test_census_components_join_across_nonmanifold_edges():
    # QA issue 11: face_adjacency skips non-manifold edge pairs and split one body
    m = _edge_sharing_boxes()
    old = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(m.faces)),
                                             min_len=1)
    assert len(old) == 2                                  # precondition: the old count
    c = mesh_census(m)
    assert c["nonmanifold_edges"] == 1 and c["boundary_edges"] == 0
    assert c["components"] == 1
    assert c["manifold_components"] == 2
    assert c["closed_components"] == 1 and c["open_components"] == 0
    assert mesh_census(_box())["components"] == 1
    far = trimesh.util.concatenate([_box(), _box(offset=(100.0, 0.0, 0.0))])
    cf = mesh_census(far)
    assert cf["components"] == 2 and cf["manifold_components"] == 2


def test_census_nonmanifold_inside_out_volume_and_warning(tmp_path):
    # QA issue 18: 0 boundary edges + non-manifold: the volume and the
    # inverted check used to be skipped because is_watertight is False
    m = _edge_sharing_boxes(inverted=True)
    assert not m.is_watertight                            # precondition
    assert float(m.volume) == pytest.approx(-2000.0)      # precondition: inside out
    c = mesh_census(m)
    assert c["boundary_edges"] == 0 and c["nonmanifold_edges"] == 1
    assert c["inverted"] is True
    assert c["volume_mm3"] == pytest.approx(2000.0, rel=1e-9)
    assert c["volume_approximate"] is True
    ok = mesh_census(_edge_sharing_boxes(inverted=False))
    assert ok["inverted"] is False and ok["volume_mm3"] == pytest.approx(2000.0, rel=1e-9)
    p = tmp_path / "nm.stl"
    m.export(str(p))
    mm = load_mesh_model(str(p))
    assert mm.census["inverted"] is True
    assert "mesh is not watertight: 0 boundary edge(s), 1 non-manifold edge(s)" in mm.warnings
    assert any("inside out" in w for w in mm.warnings)
    from src.drawing.mesh_load import watertight_detail
    assert watertight_detail(c) == "0 boundary edge(s), 1 non-manifold edge(s)"
    # an open mesh still has no volume (QA issue 28)
    open_box = trimesh.Trimesh(_box().vertices.copy(), _box().faces[1:].copy())
    co = mesh_census(open_box)
    assert co["volume_mm3"] is None and co["volume_approximate"] is False


def test_census_volume_repaired_per_component():
    # QA issue 12: the raw signed sum of a badly wound multi-body mesh is meaningless
    a = _box((20.0, 10.0, 10.0))
    b = _box((10.0, 10.0, 10.0), (50.0, 0.0, 0.0))
    b.invert()                                            # the smaller body inside out
    two = trimesh.util.concatenate([a, b])
    assert two.is_watertight and two.is_winding_consistent        # precondition
    assert abs(float(two.volume)) == pytest.approx(1000.0)        # precondition: 2000 - 1000
    c = mesh_census(two)
    assert c["volume_mm3"] == pytest.approx(3000.0, rel=1e-9)
    assert c["volume_repaired"] is True and c["volume_source"] == "winding_repaired"
    assert c["volume_raw_mm3"] == pytest.approx(1000.0)
    assert c["inverted"] is False                         # the dominant body is outward

    # winding conflicts inside one shell: two faces of a cube reversed
    cube = _box((10.0, 10.0, 10.0))
    faces = cube.faces.copy()
    faces[:2] = faces[:2, ::-1]
    bad = trimesh.Trimesh(cube.vertices.copy(), faces, process=False)
    assert not bad.is_winding_consistent                  # precondition
    assert abs(float(bad.volume)) != pytest.approx(1000.0)        # precondition
    cb = mesh_census(bad)
    assert cb["winding_conflicts"] > 0
    assert cb["volume_mm3"] == pytest.approx(1000.0, rel=1e-9)
    assert cb["volume_repaired"] is True and cb["inverted"] is False

    # a nested void keeps its sign: 48000 - 2000, nothing repaired
    hb = mesh_census(_hollow_box())
    assert hb["volume_mm3"] == pytest.approx(46000.0, rel=1e-9)
    assert hb["volume_repaired"] is False and hb["volume_source"] == "signed"

    from src.drawing.mesh_load import mesh_volume_repaired
    assert mesh_volume_repaired(two) == (pytest.approx(3000.0), True)
    assert mesh_volume_repaired(_box()) == (pytest.approx(48000.0), False)


def _soup(n: int = 50, seed: int = 3) -> trimesh.Trimesh:
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.0, 100.0, (n, 1, 3))
    tris = centres + rng.normal(scale=2.0, size=(n, 3, 3))
    return trimesh.Trimesh(tris.reshape(-1, 3), np.arange(3 * n).reshape(-1, 3), process=True)


def test_classify_triangle_soup_is_not_solid(tmp_path):
    # QA issue 19: disconnected random triangles were classified 'organic'
    soup = _soup()
    cs = mesh_census(soup)
    assert cs["components"] == 50 and cs["closed_components"] == 0      # precondition
    assert classify(soup, 0.10, census=cs)["verdict"] == "not_solid"
    cl = classify(soup)                                   # census computed internally
    assert cl["verdict"] == "not_solid"
    assert cl["surface_verdict"] == "organic" and cl["planar_big"] == 0.0
    assert "50 disconnected open components, 0 closed" in cl["not_solid_reason"]
    p = tmp_path / "soup.stl"
    soup.export(str(p))
    mm = load_mesh_model(str(p))
    assert mm.classifier["verdict"] == "not_solid"
    assert any("not a solid" in w for w in mm.warnings)

    # a single open strip: boundary edges >= triangles
    v = np.array([[i, j, 0.0] for i in range(6) for j in (0.0, 1.0)])
    f = np.array([[2 * i, 2 * i + 2, 2 * i + 1] for i in range(5)] +
                 [[2 * i + 1, 2 * i + 2, 2 * i + 3] for i in range(5)])
    strip = trimesh.Trimesh(v, f, process=True)
    st = mesh_census(strip)
    assert st["components"] == 1 and st["boundary_edges"] >= st["triangles"]   # precondition
    assert classify(strip)["verdict"] == "not_solid"

    # controls: closed meshes and one open body with a hole keep their verdicts
    assert classify(trimesh.creation.icosphere(subdivisions=3))["verdict"] == "organic"
    assert classify(_box())["verdict"] == "prismatic"
    open_box = trimesh.Trimesh(_box().vertices.copy(), _box().faces[1:].copy())
    ob = classify(open_box)
    assert ob["verdict"] == "prismatic" and ob["not_solid_reason"] is None
    one_closed = trimesh.util.concatenate([_box(), _soup(5)])
    assert classify(one_closed)["verdict"] == "prismatic"


def _holed_sphere(r: float = 10.0, n_remove: int = 3) -> trimesh.Trimesh:
    """Icosphere with the ``n_remove`` faces nearest to y = 0 (on the +x side) removed."""
    ico = trimesh.creation.icosphere(subdivisions=3, radius=r)
    c = ico.triangles_center
    score = np.abs(c[:, 1]) + 10.0 * (c[:, 0] < 0.5 * r)          # equator band, +x side
    drop = np.argsort(score)[:n_remove]
    keep = np.setdiff1d(np.arange(len(ico.faces)), drop)
    return trimesh.Trimesh(ico.vertices.copy(), ico.faces[keep].copy(), process=True)


def test_section_open_mesh_bridges_gaps():
    # QA issue 33: an open mesh lost its main contour (open wire dropped)
    from trimesh.intersections import mesh_plane
    from src.drawing.geometry2d import chain_segments

    r = 10.0
    sph = _holed_sphere(r)
    normal = np.array([0.0, -1.0, 0.0])
    segs = mesh_plane(sph, plane_normal=normal, plane_origin=np.zeros(3))
    closed, n_open = chain_segments(np.asarray(segs), 1e-6 * 35.0)
    assert closed == [] and n_open >= 1                   # precondition: the contour is open
    sv = section_mesh(sph, np.zeros(3), normal, VIEWS["front"], raster_px=256,
                      feature_angle_deg=30.0, organic=True, depth_extent=2 * r)
    assert len(sv.loops) == 1
    area = abs(polygon_area(sv.loops[0]))
    assert area == pytest.approx(math.pi * r * r, rel=0.10)
    assert sv.stats["bridged_gaps"] >= 1
    assert sv.stats["open_wires_dropped"] == 0
    assert 0.6 < sv.stats["kept_area_ratio"] <= 1.0      # disc in its bbox: pi / 4
    # negative control: without bridging the contour is dropped again
    sv0 = section_mesh(sph, np.zeros(3), normal, VIEWS["front"], raster_px=256,
                       feature_angle_deg=30.0, organic=True, depth_extent=2 * r, bridge_k=0.0)
    assert sv0.loops == [] and sv0.stats["open_wires_dropped"] >= 1
    assert sv0.stats["bridged_gaps"] == 0
    # a closed sphere needs no bridge
    full = section_mesh(trimesh.creation.icosphere(subdivisions=3, radius=r), np.zeros(3), normal,
                        VIEWS["front"], raster_px=256, feature_angle_deg=30.0, organic=True,
                        depth_extent=2 * r)
    assert full.stats["bridged_gaps"] == 0 and len(full.loops) == 1


def test_section_sliver_left_after_open_contour_is_measurable():
    # the bunny99 case: the main contour is open and dropped, only a small
    # closed loop survives and the report said loops=1
    r = 10.0
    mesh = trimesh.util.concatenate([_holed_sphere(r), _box((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))])
    normal = np.array([0.0, -1.0, 0.0])
    kw = dict(raster_px=128, feature_angle_deg=30.0, organic=True, depth_extent=2 * r)
    sv0 = section_mesh(mesh, np.zeros(3), normal, VIEWS["front"], bridge_k=0.0, **kw)
    assert len(sv0.loops) == 1 and sv0.stats["open_wires_dropped"] >= 1     # precondition: sliver
    assert sv0.stats["kept_length_ratio"] < 0.1
    assert sv0.stats["kept_area_ratio"] < 0.05
    sv = section_mesh(mesh, np.zeros(3), normal, VIEWS["front"], **kw)
    assert len(sv.loops) == 2 and sv.stats["open_wires_dropped"] == 0
    assert sv.stats["kept_length_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert sv.stats["kept_area_ratio"] > 0.6


def test_section_kept_area_ratio_subtracts_holes():
    torus = trimesh.creation.torus(10.0, 3.0)
    sv = section_mesh(torus, np.zeros(3), np.array([0.0, 0.0, 1.0]), VIEWS["top"], raster_px=128,
                      feature_angle_deg=30.0, organic=True, depth_extent=6.0)
    assert len(sv.loops) == 2                             # precondition: ring 13 / 7
    ring = math.pi * (13.0 ** 2 - 7.0 ** 2) / (26.0 * 26.0)       # 0.558, not (169 + 49) pi / 676
    assert sv.stats["kept_area_ratio"] == pytest.approx(ring, rel=0.02)


def test_section_bridge_does_not_join_distant_chains():
    # an open arc whose ends are 19.6 mm apart (far beyond k x median edge
    # length) next to a holed sphere whose small gaps MUST be bridged: a
    # no-op bridging loses the sphere loop, bridging everything closes the arc
    r = 10.0
    sph = _holed_sphere(r, n_remove=0)
    keep = sph.triangles_center[:, 0] > 2.0               # a cap: its cut is an open arc
    cap = trimesh.Trimesh(sph.vertices.copy(), sph.faces[keep].copy(), process=True)
    holed = _holed_sphere(r)
    holed.apply_translation((40.0, 0.0, 0.0))
    both = trimesh.util.concatenate([cap, holed])
    gap = 2.0 * math.sqrt(r * r - 2.0 ** 2)               # chord between the arc ends, ~19.6
    med = float(np.median(both.edges_unique_length))
    assert gap > 3.0 * med                                # precondition: arc gap out of reach
    kw = dict(raster_px=128, feature_angle_deg=30.0, organic=True, depth_extent=2 * r)
    normal = np.array([0.0, -1.0, 0.0])
    sv0 = section_mesh(both, np.zeros(3), normal, VIEWS["front"], bridge_k=0.0, **kw)
    assert sv0.loops == [] and sv0.stats["open_wires_dropped"] >= 2      # precondition: both open
    sv = section_mesh(both, np.zeros(3), normal, VIEWS["front"], **kw)
    assert len(sv.loops) == 1                             # the sphere, not the arc
    assert sv.stats["bridged_gaps"] >= 1 and sv.stats["bridge_rejected"] is False
    assert sv.stats["open_wires_dropped"] == 1            # the arc stays open
    loop = sv.loops[0]
    assert abs(polygon_area(loop)) == pytest.approx(math.pi * r * r, rel=0.10)
    assert float(loop[:, 0].mean()) == pytest.approx(40.0, abs=1.0)
    assert sv.stats["bridge_max_gap"] < 3.0 * med


# ------------------------------------------------------- unit plausibility

def test_unit_plausibility_flags_tiny_huge_and_inch_fractions():
    # QA issue 5: an inch STL, a metre-scale bunny and a 5 m inch STEP were silent
    from src.drawing.mesh_load import unit_plausibility
    v, d, alt = unit_plausibility([50.0, 30.0, 20.0])
    assert v == "ok" and alt is None
    assert unit_plausibility([12.5, 25.0, 6.0])[0] == "ok"        # half-mm sizes are not inch
    v, d, alt = unit_plausibility([5.0, 2.5, 1.375])
    assert v == "suspicious" and alt["unit"] == "in"
    assert alt["extents_mm"] == pytest.approx([127.0, 63.5, 34.925])
    assert "inch fraction" in d
    v, d, alt = unit_plausibility([0.1547, 0.1554, 0.1209])        # bunny99 in metres
    assert v == "suspicious" and alt["unit"] == "m" and alt["factor"] == 1000.0
    assert "below 2 mm" in d and "155" in d
    v, d, alt = unit_plausibility([5080.0, 2540.0, 1270.0], declared_unit="inch")
    assert v == "suspicious" and alt["unit"] == "mm"
    assert alt["extents_mm"] == pytest.approx([200.0, 100.0, 50.0])
    # a declared inch file is not re-flagged for inch fractions
    assert unit_plausibility([127.0, 63.5, 34.925], declared_unit="inch")[0] == "ok"
    assert unit_plausibility([1.0, 0.5, 0.25], declared_unit="mm")[0] == "suspicious"
    assert unit_plausibility([])[0] == "suspicious"


# ----------------------------------------------------- depth sign (mesh)

def _voxel_mesh(occ: np.ndarray, pitch: float) -> trimesh.Trimesh:
    """Closed blocky surface of a boolean voxel array: one quad per exposed voxel face."""
    pad = np.pad(occ, 1)
    tris = []
    for axis in range(3):
        for sgn in (1, -1):
            u, w = [a for a in range(3) if a != axis]
            for idx in np.argwhere(occ):
                nb = idx + 1
                nb[axis] += sgn
                if pad[tuple(nb)]:
                    continue
                base = idx.astype(float)
                if sgn > 0:
                    base[axis] += 1.0
                q = []
                for du, dw in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    p = base.copy()
                    p[u] += du
                    p[w] += dw
                    q.append(p * pitch)
                q = np.array(q)
                n = np.cross(q[1] - q[0], q[2] - q[0])
                if n[axis] * sgn < 0:                    # wind outward
                    q = q[::-1]
                tris += [q[[0, 1, 2]], q[[0, 2, 3]]]
    tris = np.asarray(tris)
    return trimesh.Trimesh(tris.reshape(-1, 3), np.arange(len(tris) * 3).reshape(-1, 3), process=True)


def _bossed_plate(side: int) -> trimesh.Trimesh:
    """Plate 100 x 10 x 60 (depth along Y) with six 10 mm cube bosses on +Y (side=+1) or -Y."""
    occ = np.zeros((10, 3, 6), dtype=bool)
    py = 1 if side > 0 else 2
    occ[:, py, :] = True
    by = 2 if side > 0 else 1
    for i in (1, 4, 7):
        for k in (1, 4):
            occ[i, by, k] = True
    m = _voxel_mesh(occ, 10.0)
    m.apply_translation((0.0, -10.0 - 10.0 * (py - 1), 0.0))
    return m


def test_score_depth_sign_mesh_prefers_the_featured_side():
    # QA issue 14: plates were drawn from their featureless underside
    from src.drawing.mesh_load import choose_depth_sign_mesh, score_depth_sign_mesh
    plate = _bossed_plate(+1)
    assert plate.is_watertight and plate.volume == pytest.approx(60000.0 + 6000.0)   # precondition
    up, down = np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])
    # precondition: the facing AREA cannot tell the sides apart (triage proposal)
    fn, fa = plate.face_normals, plate.area_faces
    proj = [float((fa * np.clip(fn @ d, 0, None)).sum()) for d in (up, down)]
    flat = [float(fa[fn @ d > 0.5].sum()) for d in (up, down)]
    assert proj[0] == pytest.approx(proj[1], rel=1e-12) and flat[0] == pytest.approx(flat[1])
    s_up, s_down = score_depth_sign_mesh(plate, up), score_depth_sign_mesh(plate, down)
    assert s_up == pytest.approx(320.0 + 6 * 80.0)        # plate rim + boss top and foot rims
    assert s_down == pytest.approx(320.0)                 # plate rim only
    # front viewer at -Y sees the plain back: flip; the mirror plate keeps its side
    r = choose_depth_sign_mesh(plate, np.eye(3))
    assert r["flip"] is True
    assert np.allclose(r["matrix"], np.diag([-1.0, -1.0, 1.0]))
    assert np.linalg.det(np.asarray(r["matrix"])) == pytest.approx(1.0)
    mirror = choose_depth_sign_mesh(_bossed_plate(-1), np.eye(3))
    assert mirror["flip"] is False and np.allclose(mirror["matrix"], np.eye(3))
    # a symmetric part keeps the permutation
    assert choose_depth_sign_mesh(_box(), np.eye(3))["flip"] is False


# ------------------------------------------- F4 review: origin-free volume

def _face_sharing_cubes(inverted: bool = False) -> trimesh.Trimesh:
    """Two 10 mm cubes sharing the full face x = 10 (touching bodies of a multi-body STL).

    The four edges of the shared square carry four faces each, so the face
    adjacency splits the closed body into four OPEN shells. ``inverted``
    reverses every face.
    """
    a = _box((10.0, 10.0, 10.0), (5.0, 5.0, 5.0))
    b = _box((10.0, 10.0, 10.0), (15.0, 5.0, 5.0))
    m = trimesh.util.concatenate([a, b])
    faces = m.faces[:, ::-1] if inverted else m.faces
    return trimesh.Trimesh(m.vertices.copy(), np.ascontiguousarray(faces), process=True)


def _shifted(m: trimesh.Trimesh, shift) -> trimesh.Trimesh:
    out = m.copy()
    out.apply_translation(shift)
    return out


_SHIFTS = ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (-100.0, 0.0, 0.0), (-3000.0, 2500.0, 777.0))


def test_census_volume_is_translation_invariant_with_open_shells():
    # review F4 must-fix: open shells (split by non-manifold edges) were
    # flipped or treated as voids from their origin-dependent signed volume:
    # 2000 / 7333.33 / 6000 at x shifts 0 / +100 / -100
    from src.drawing.mesh_load import mesh_topology
    m = _face_sharing_cubes()
    c0 = mesh_census(m)
    assert c0["nonmanifold_edges"] >= 1 and c0["boundary_edges"] == 0       # precondition
    assert c0["manifold_components"] >= 3 and c0["components"] == 1         # precondition
    # precondition: the per-shell signed volumes really depend on the origin
    shell = mesh_topology(m)["shell_label"]
    per_shell = []
    for s in ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)):
        ms = _shifted(m, s)
        tri = ms.triangles
        fv = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
        per_shell.append(np.bincount(shell, weights=fv))
    assert np.abs(per_shell[0] - per_shell[1]).max() > 100.0
    for s in _SHIFTS:
        c = mesh_census(_shifted(m, s))
        assert c["volume_mm3"] == pytest.approx(2000.0, rel=1e-9), s
        assert c["volume_source"] == "signed" and c["volume_repaired"] is False, s
        assert c["inverted"] is False, s
        ci = mesh_census(_shifted(_face_sharing_cubes(inverted=True), s))
        assert ci["volume_mm3"] == pytest.approx(2000.0, rel=1e-9), s
        assert ci["inverted"] is True and ci["volume_source"] == "signed", s
    from src.drawing.mesh_load import mesh_volume_repaired
    assert mesh_volume_repaired(_shifted(m, (100.0, 0.0, 0.0))) == (pytest.approx(2000.0), False)
    # the open shells form one balanced piece (their union is closed)
    assert c0["volume_pieces"] == 1 and c0["volume_unbalanced_pieces"] == 0


def test_census_edge_sharing_one_inverted_is_translation_invariant():
    # two closed shells of equal |V| = 1000, one inside out: the tie must not
    # be broken by float noise of the position (inverted used to flip with it)
    a = _box((10.0, 10.0, 10.0), (5.0, 5.0, 5.0))
    b = _box((10.0, 10.0, 10.0), (15.0, 15.0, 5.0))
    b.invert()
    m0 = trimesh.util.concatenate([a, b])
    m0 = trimesh.Trimesh(m0.vertices.copy(), m0.faces.copy(), process=True)
    tri = m0.triangles
    raw = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0
    assert abs(raw) < 1e-9                                # precondition: raw sum cancels
    assert mesh_census(m0)["nonmanifold_edges"] == 1     # precondition
    for s in _SHIFTS:
        c = mesh_census(_shifted(m0, s))
        assert c["volume_mm3"] == pytest.approx(2000.0, rel=1e-9), s
        assert c["volume_source"] == "winding_repaired", s
        assert c["inverted"] is False, s
    # the all-inverted fixture stays inverted at every position
    for s in _SHIFTS:
        ci = mesh_census(_shifted(_edge_sharing_boxes(inverted=True), s))
        assert ci["inverted"] is True and ci["volume_mm3"] == pytest.approx(2000.0, rel=1e-9)


def _l_block() -> trimesh.Trimesh:
    """L-shaped prism: 2 x 1 x 2 voxels of 10 mm minus the (x 10..20, z 10..20) one; 3000 mm3."""
    occ = np.ones((2, 1, 2), dtype=bool)
    occ[1, 0, 1] = False
    return _voxel_mesh(occ, 10.0)


def test_census_inverted_body_in_a_notch_is_not_a_void():
    # review F4 should-fix: bbox containment alone subtracted a separate
    # inverted body that sits in the notch of an L (inside the bbox, outside
    # the material)
    L = _l_block()
    assert L.is_watertight and L.volume == pytest.approx(3000.0)           # precondition
    small = _box((4.0, 4.0, 4.0), (15.0, 5.0, 15.0))
    small.invert()
    assert np.all(small.bounds[0] >= L.bounds[0]) and np.all(small.bounds[1] <= L.bounds[1])  # precondition
    m = trimesh.util.concatenate([L, small])
    c = mesh_census(m)
    assert c["volume_mm3"] == pytest.approx(3064.0, rel=1e-9)               # 3000 + 64, not 3000 - 64
    assert c["volume_source"] == "winding_repaired" and c["inverted"] is False
    # control: the same small cube inside the material is a void
    void = _box((4.0, 4.0, 4.0), (5.0, 5.0, 5.0))
    void.invert()
    cv = mesh_census(trimesh.util.concatenate([L, void]))
    assert cv["volume_mm3"] == pytest.approx(2936.0, rel=1e-9)
    assert cv["volume_source"] == "signed"


def test_census_many_inverted_bodies_is_fast():
    # review F4 should-fix: the void test was O(n_shells^2) in Python and each
    # repaired shell copied every vertex (2000 cubes, every second inverted:
    # 6.68 s; with two reversed faces in every second cube: 3.24 s)
    import time
    k = 13
    parts = []
    for i in range(2000):
        b = _box((1.0, 1.0, 1.0), (3.0 * (i % k), 3.0 * ((i // k) % k), 3.0 * (i // (k * k))))
        if i % 2:
            b.invert()
        if i % 4 == 2:
            f = b.faces.copy()
            f[:2] = f[:2, ::-1]
            b = trimesh.Trimesh(b.vertices, f, process=False)
        parts.append(b)
    m = trimesh.util.concatenate(parts)
    t = time.perf_counter()
    c = mesh_census(m)
    dt = time.perf_counter() - t
    assert c["manifold_components"] == 2000                # precondition
    assert c["winding_conflicts"] > 0                      # precondition: some shells need a rewind
    assert c["volume_mm3"] == pytest.approx(2000.0, rel=1e-9)
    assert c["volume_source"] == "winding_repaired"
    assert dt < 2.0, f"census of 2000 shells took {dt:.2f} s"


def test_census_zero_volume_closed_mesh_is_silent():
    # review F4 should-fix: mesh.volume (trimesh mass properties) warns
    # "divide by zero" on stderr when the signed sum is 0
    import warnings
    a = _box((10.0, 10.0, 10.0))
    b = _box((10.0, 10.0, 10.0), (50.0, 0.0, 0.0))
    b.invert()
    m = trimesh.util.concatenate([a, b])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        trimesh.Trimesh(m.vertices.copy(), m.faces.copy()).volume
    assert any(issubclass(w.category, RuntimeWarning) for w in rec)        # precondition
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c = mesh_census(trimesh.Trimesh(m.vertices.copy(), m.faces.copy()))
    assert [str(w.message) for w in rec if issubclass(w.category, RuntimeWarning)] == []
    assert c["volume_raw_mm3"] == pytest.approx(0.0, abs=1e-9)
    assert c["volume_mm3"] == pytest.approx(2000.0, rel=1e-9) and c["inverted"] is False


def _dense_soup(n: int = 2000, box: float = 20.0, noise: float = 1.0, seed: int = 3) -> trimesh.Trimesh:
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.0, box, (n, 1, 3))
    tris = centres + rng.normal(scale=noise, size=(n, 3, 3))
    return trimesh.Trimesh(tris.reshape(-1, 3), np.arange(3 * n).reshape(-1, 3), process=True)


def test_section_soup_is_not_bridged_into_loops():
    # review F4 should-fix: bridging turned a triangle soup into junk loops
    # that looked healthy (corpus 22: 75 bridges -> 3 loops, length ratio 1.0)
    from trimesh.intersections import mesh_plane
    from src.drawing.geometry2d import chain_segments
    from src.drawing.mesh_sections import BRIDGE_K, _bridges, _dangling_ends

    soup = _dense_soup()
    centre = 0.5 * (soup.bounds[0] + soup.bounds[1])
    normal = np.array([0.0, -1.0, 0.0])
    weld = 1e-6 * float(np.linalg.norm(soup.extents))
    segs = np.asarray(mesh_plane(soup, plane_normal=normal, plane_origin=centre)).reshape(-1, 2, 3)
    closed, n_open = chain_segments(segs, weld)
    assert closed == [] and n_open > 50                   # precondition: one-segment chains
    tol = BRIDGE_K * float(np.median(soup.edges_unique_length))
    br = _bridges(_dangling_ends(segs, weld), tol)
    junk, _ = chain_segments(np.concatenate([segs, br]), weld)
    assert len(junk) >= 3                                 # precondition: unguarded bridging makes loops
    sv = section_mesh(soup, centre, normal, VIEWS["front"], raster_px=64, feature_angle_deg=30.0,
                      organic=True, depth_extent=20.0)
    assert sv.loops == []
    assert sv.stats["bridge_rejected"] is True and sv.stats["bridged_gaps"] == 0
    assert sv.stats["open_chain_mean_segments"] < 2.0


def test_project_mesh_fine_torus_keeps_hidden_runs_shorter_than_the_outline_cover():
    """F1 review: the 3 px outline subtraction also deleted every visible and hidden run of
    3 px or less before chaining, so a finely tessellated torus (edges ~1.6 px at raster
    256) lost its whole hidden inner silhouette. The length floor is the 0.5 px clean-up
    tolerance; chains are still filtered by their total length afterwards."""
    from src.drawing import mesh_views
    torus = trimesh.creation.torus(10.0, 3.0, major_sections=512, minor_sections=64)
    pv = project_mesh(torus, VIEWS["front"], raster_px=256, feature_angle_deg=30.0, organic=True)
    px = pv.stats["px"]
    # precondition: the major-circle edges are shorter than the 3 px outline cover tolerance
    assert 2.0 * math.pi * 13.0 / 512 < mesh_views._OUTLINE_COVER_PX * px
    assert pv.stats["interior_silhouettes_omitted"] is False
    from src.drawing.geometry2d import primitives_to_segments
    segs = primitives_to_segments(pv.hidden, 0.01) if pv.hidden else np.zeros((0, 2, 2))
    total = float(np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1).sum()) if len(segs) else 0.0
    assert total > 10.0                         # shipped code: 0 (256x32 torus: 14.1)
    pts = segs.reshape(-1, 2)
    assert np.abs(pts[:, 0]).max() < 10.0       # inside the hole's span |x| < 7 + 3
    loops = [p for p in pv.visible if isinstance(p, Polyline) and p.closed]
    assert len(loops) == 1 and len(pv.visible) == 1


# ------------------------------- QA round 2 issue 4: depth sign by open cavity

def _pocketed_plate() -> trimesh.Trimesh:
    """Plate 100 (X) x 12 (Y, depth) x 60 (Z): a 70 x 6 x 30 pocket opens to
    -Y, 36 shallow 5 x 2 x 5 notches on +Y (many feature edges, little
    cavity). Voxels of 1 x 1 x 1 scaled by (5, 2, 5)."""
    occ = np.ones((20, 6, 12), dtype=bool)
    occ[3:17, 0:3, 3:9] = False                       # pocket from the -Y face
    for i in range(1, 19, 2):
        for k in (1, 4, 7, 10):
            occ[i, 5, k] = False                      # notches in the +Y layer
    m = _voxel_mesh(occ, 1.0)
    t = np.diag([5.0, 2.0, 5.0, 1.0])
    m.apply_transform(t)
    return m


def test_depth_sign_mesh_keeps_the_pocket_toward_the_viewer():
    """Round 2 issue 4 (NIST FTC 08: the engraved logo under the tray gave
    5935 vs 5351 mm of feature edges and kept the underside)."""
    from src.drawing.mesh_load import (cavity_depth_sign_mesh, choose_depth_sign_mesh,
                                       score_depth_sign_mesh)
    plate = _pocketed_plate()
    assert plate.is_watertight and plate.volume > 0                       # precondition
    front, back = np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])
    # precondition: the feature-edge rule prefers the notched +Y side (> 1.2 x)
    e_front, e_back = score_depth_sign_mesh(plate, front), score_depth_sign_mesh(plate, back)
    assert e_back > 1.2 * e_front, (e_front, e_back)
    c = cavity_depth_sign_mesh(plate, front)
    assert c["cavity_plus"] == pytest.approx(70.0 * 30.0 * 6.0, rel=1e-9)  # the pocket
    assert c["cavity_minus"] == pytest.approx(36 * 25.0 * 2.0, rel=1e-9)   # the notches
    assert c["bbox_volume"] == pytest.approx(100.0 * 12.0 * 60.0)
    r = choose_depth_sign_mesh(plate, np.eye(3))
    assert r["flip"] is False and r["method"] == "mesh_cavity", r
    assert r["edges_flipped"] > r["edges_kept"]
    # the same plate turned 180 deg about Z: the pocket faces away -> flip
    turned = plate.copy()
    turned.apply_transform(np.diag([-1.0, -1.0, 1.0, 1.0]))
    r2 = choose_depth_sign_mesh(turned, np.eye(3))
    assert r2["flip"] is True and r2["method"] == "mesh_cavity"
    assert np.allclose(r2["matrix"], np.diag([-1.0, -1.0, 1.0]))
    # a plain box has no cavity: the feature-edge rule keeps it
    rb = choose_depth_sign_mesh(_box(), np.eye(3))
    assert rb["flip"] is False and rb["method"] == "mesh_feature_edges"
    assert rb["cavity_kept"] == pytest.approx(0.0, abs=1e-9)
    # an unreliable mesh (the pocket floor missing: open, projected areas 6000 vs
    # 3900) is not decided by the cavity (QA corpus 22, a triangle soup: 195 %)
    fn = plate.face_normals
    ctr = plate.triangles_center
    floor = (fn @ front > 0.99) & (np.abs(ctr[:, 1] - 6.0) < 1e-9)
    assert int(floor.sum()) == 2 * 14 * 6                                 # precondition
    holed = trimesh.Trimesh(plate.vertices, plate.faces[~floor], process=False)
    ch = cavity_depth_sign_mesh(holed, front)
    assert abs(ch["proj_plus"] - ch["proj_minus"]) > 0.3 * ch["proj_minus"]  # precondition
    rh = choose_depth_sign_mesh(holed, np.eye(3))
    assert rh["method"] == "mesh_feature_edges" and "not used" in rh["reason"]


# ------------------- QA round 2 issue 8: inside-out mesh with a few conflicts

def _inverted_sphere_one_face_back() -> trimesh.Trimesh:
    """Icosphere (5120 faces, r 10) wound inside out except one face: 3 winding
    conflicts, the raw signed volume 0.04 % off the repaired one."""
    s = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    f = s.faces[:, ::-1].copy()
    f[0] = f[0, ::-1]
    return trimesh.Trimesh(s.vertices.copy(), f, process=False)


def test_census_inverted_solid_with_a_few_conflicts(tmp_path):
    from src.drawing.mesh_load import inverted_solid_verdict
    m = _inverted_sphere_one_face_back()
    c = mesh_census(m)
    # precondition: closed, conflicts present, more than the count rule allows
    assert c["boundary_edges"] == 0 and c["winding_conflicts"] == 3
    assert c["winding_conflicts"] > max(2, 1e-4 * 1.5 * c["triangles"])
    assert c["inverted"] is True and c["volume_repaired"] is True
    raw = c["volume_raw_signed_mm3"]
    assert raw < 0.0 and abs(abs(raw) - c["volume_mm3"]) < 0.01 * c["volume_mm3"]
    assert c["volume_raw_mm3"] == pytest.approx(abs(raw))          # unsigned key kept
    assert c["inverted_solid"] is True          # the round-1 rule (conflicts == 0) said no
    p = tmp_path / "inv.stl"
    m.export(str(p))
    mm = load_mesh_model(str(p))
    rep = [w for w in mm.warnings if "winding-repaired" in w]
    assert len(rep) == 1 and f"{raw:.6g}" in rep[0] and "-" in rep[0], mm.warnings
    assert "3 winding conflict(s)" in rep[0]
    assert sum("inside out" in w for w in mm.warnings) == 1
    # control: a correctly wound sphere
    ok = mesh_census(trimesh.creation.icosphere(subdivisions=4, radius=10.0))
    assert ok["inverted"] is False and ok["inverted_solid"] is False
    assert ok["volume_raw_signed_mm3"] > 0.0
    # control: a badly wound box (one face of six flipped back) is not inside out
    b = trimesh.creation.box(extents=(40, 20, 10))
    f = b.faces[:, ::-1].copy()
    f[:2] = f[:2, ::-1]
    bad = mesh_census(trimesh.Trimesh(b.vertices, f, process=False))
    assert bad["inverted"] is True and bad["winding_conflicts"] == 4       # precondition
    assert bad["volume_raw_signed_mm3"] == pytest.approx(-8000.0 + 2 * 800.0 * 5.0 / 3.0)
    assert bad["inverted_solid"] is False
    q = tmp_path / "bad.stl"
    trimesh.Trimesh(b.vertices, f, process=False).export(str(q))
    mb = load_mesh_model(str(q))
    assert not any("inside out" in w for w in mb.warnings)
    assert sum("winding-repaired" in w for w in mb.warnings) == 1
    # the count rule: 2 conflicts in 451k edges (the QA bracket) is inside out
    assert inverted_solid_verdict({"inverted": True, "winding_conflicts": 2}, 451_000) is True
    assert inverted_solid_verdict({"inverted": True, "winding_conflicts": 50,
                                   "volume_raw_signed_mm3": 10.0, "volume_mm3": 10.0},
                                  451_000) is False
    assert inverted_solid_verdict({"inverted": False, "winding_conflicts": 0}) is False


# ------------------------------- QA round 2 issue 3: section loop body ids

def _section_front(mesh, y=10.0):
    return section_mesh(mesh, np.array([0.0, y, 0.0]), np.array([0.0, -1.0, 0.0]), VIEWS["front"],
                        raster_px=128, feature_angle_deg=30.0, organic=False, depth_extent=40.0)


def test_section_mesh_loop_body_per_component():
    from src.drawing.mesh_sections import face_components
    two = trimesh.util.concatenate([_box((20.0, 20.0, 20.0), (-20.0, 0.0, 0.0)),
                                    _box((10.0, 20.0, 10.0), (20.0, 0.0, 0.0))])
    assert int(face_components(two).max()) + 1 == 2                     # precondition
    sv = _section_front(two, y=0.0)
    assert len(sv.loops) == 2 and sv.loop_body is not None and len(sv.loop_body) == 2
    by_x = {round(float(L[:, 0].mean())): b for L, b in zip(sv.loops, sv.loop_body)}
    assert by_x[-20] != by_x[20]
    assert sorted(sv.loop_body) == [0, 1]
    # a pin in a clearance hole of a tube (an island) keeps its own body
    tube = trimesh.creation.annulus(r_min=6.0, r_max=10.0, height=20.0)
    pin = trimesh.creation.cylinder(radius=4.0, height=30.0)
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    asm = trimesh.util.concatenate([tube, pin])
    asm.apply_transform(rot)                                             # axes along Y
    sv2 = _section_front(asm, y=0.0)
    assert len(sv2.loops) == 3 and len(set(sv2.loop_body)) == 2


def test_section_mesh_void_shell_joins_its_outer_body(monkeypatch):
    """The cavity of a hollow box is a separate component of the vertex graph
    but its cut loop is a hole of the outer body: one body id (else the
    layout nests it alone and fills the void)."""
    from src.drawing import mesh_sections
    from src.drawing.mesh_sections import component_volumes, face_components
    hb = _hollow_box()
    lab = face_components(hb)
    assert int(lab.max()) + 1 == 2                                      # precondition: 2 shells
    vols = component_volumes(hb, lab)
    assert sorted(np.round(vols, 6)) == [-2000.0, 48000.0]              # the cavity is inward
    sv = _section_front(hb, y=10.0)
    assert len(sv.loops) == 2 and sv.loop_body is not None
    assert sv.loop_body[0] == sv.loop_body[1]
    inner = int(np.argmin([abs(polygon_area(L)) for L in sv.loops]))
    # loops side by side keep their bodies; an exact-fit pin (its loop coincides
    # with the hole of the plate) keeps its body; a body inside the material
    # of another (healed cavity, interference) joins it
    apart = [sv.loops[0], sv.loops[1] + 100.0]
    assert mesh_sections._merge_void_bodies(apart, [0, 1], [10.0, -1.0]) == [0, 1]
    ids = [0, 0]
    ids[inner] = 1
    assert mesh_sections._merge_void_bodies(sv.loops, ids, [10.0, -5.0]) == [0, 0]
    assert mesh_sections._merge_void_bodies(sv.loops, ids, [10.0, 5.0]) == [0, 0]
    pin = [sv.loops[0], sv.loops[inner], sv.loops[inner][::-1].copy()]
    assert mesh_sections._merge_void_bodies(pin, [0, 0, 1], [10.0, 5.0]) == [0, 0, 1]
    # negative control: without the merge the cavity is a body of its own
    monkeypatch.setattr(mesh_sections, "_merge_void_bodies", lambda loops, ids, vol=None: list(ids))
    raw = _section_front(hb, y=10.0)
    assert len(set(raw.loop_body)) == 2


# =========================================== QA round 2 (task R1): issues 19 and 25

def _plate_with_small_hole() -> trimesh.Trimesh:
    """60 x 10 x 40 plate (y -5..5) with a 4 mm through hole along Y at x = z = 0."""
    plate = trimesh.creation.box(extents=(60.0, 10.0, 40.0))
    hole = trimesh.creation.cylinder(radius=2.0, height=30.0, sections=48)
    hole.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0]))
    try:
        m = plate.difference(hole)
    except Exception as exc:  # pragma: no cover - boolean backend missing
        pytest.skip(f"trimesh boolean not available: {exc}")
    if m is None or len(m.faces) == 0:  # pragma: no cover
        pytest.skip("trimesh boolean returned nothing")
    return m


def _loops_near(prims, centre, radius):
    out = []
    for p in prims:
        if isinstance(p, Polyline) and p.closed:
            if np.linalg.norm(np.asarray(p.points).mean(axis=0) - centre) < radius:
                out.append(p)
    return out


def test_iso_keeps_a_small_through_hole_loop(monkeypatch):
    """Issue 25: on the organic iso view the near mouth of a 4 mm hole is a closed loop of
    about 2.5 mm on paper (scale 1:4) and fell under the 3 mm open-chain filter."""
    from src.drawing import mesh_views
    m = _plate_with_small_hole()
    spec = VIEWS["iso"]
    centre = spec.to_view(np.array([[0.0, -5.0, 0.0]]))[0, :2]       # near mouth (y = -5)
    pv = project_mesh(m, spec, raster_px=1024, feature_angle_deg=30.0, organic=True,
                      paper_scale=0.25)
    loops = _loops_near(pv.visible, centre, 1.0)
    assert len(loops) == 1
    per = np.linalg.norm(np.diff(np.vstack([loops[0].points, loops[0].points[:1]]), axis=0),
                         axis=1).sum()
    assert 0.25 * per < 3.0                                       # precondition: below 3 mm
    # negative control: with the open-chain threshold for loops the hole disappears
    monkeypatch.setattr(mesh_views, "_ORGANIC_MIN_LOOP_PAPER_MM", 3.0)
    pv0 = project_mesh(m, spec, raster_px=1024, feature_angle_deg=30.0, organic=True,
                       paper_scale=0.25)
    assert _loops_near(pv0.visible, centre, 1.0) == []
    assert pv0.stats["closed_loops_dropped"] > pv.stats["closed_loops_dropped"]


def _box_with_sliver() -> trimesh.Trimesh:
    """20 x 10 x 20 box whose front face (y = -5) holds a sliver triangle along its
    diagonal: a vertex 1e-5 beside the diagonal midpoint and 1e-4 out of the plane."""
    box = trimesh.creation.box(extents=(20.0, 10.0, 20.0))
    V = np.array(box.vertices, dtype=float)
    F = np.array(box.faces, dtype=np.int64)
    front = [i for i, f in enumerate(F) if np.allclose(V[f, 1], -5.0)]
    assert len(front) == 2
    t0, t1 = F[front[0]], F[front[1]]
    shared = [v for v in t0 if v in t1]                           # the diagonal
    p0, p2 = shared
    p1 = [v for v in t0 if v not in shared][0]
    mid = 0.5 * (V[p0] + V[p2])
    diag = (V[p2] - V[p0]) / np.linalg.norm(V[p2] - V[p0])
    perp = np.cross(diag, [0.0, -1.0, 0.0])
    toward_p1 = np.sign(np.dot(V[p1] - mid, perp)) * perp
    v_new = mid + 1e-5 * toward_p1 + np.array([0.0, -1e-4, 0.0])
    V = np.vstack([V, v_new])
    k = len(V) - 1
    keep = [f for i, f in enumerate(F) if i != front[0]]
    # t0 = (p0, p1, p2) split into (p0, p1, k), (k, p1, p2) and the sliver (p0, k, p2):
    # each is t0 with one vertex replaced by k, so it keeps t0's winding
    new = [[k if v == missing else v for v in t0] for missing in (p2, p0, p1)]
    return trimesh.Trimesh(V, np.vstack([np.array(keep), np.array(new)]), process=False)


def test_sliver_triangle_adds_no_edge_inside_the_box(monkeypatch):
    """Issue 19: a near-degenerate triangle has no reliable normal; its dihedral with
    the face it lies in is spurious and drew a sharp edge across the front face."""
    from src.drawing import mesh_views
    m = _box_with_sliver()
    assert m.is_watertight

    def inside(pv):
        out = []
        for p in pv.visible + pv.hidden:
            pts = np.asarray(p.points if isinstance(p, Polyline) else [p.a, p.b], float)
            mid = 0.5 * (pts[:-1] + pts[1:]) if len(pts) > 1 else pts
            if np.any((np.abs(mid[:, 0]) < 9.9) & (np.abs(mid[:, 1]) < 9.9)):
                out.append(p)
        return out

    monkeypatch.setattr(mesh_views, "_SLIVER_REL", 0.0)               # precondition
    pv0 = project_mesh(m, VIEWS["front"], raster_px=512, feature_angle_deg=30.0, organic=False)
    assert inside(pv0)
    monkeypatch.setattr(mesh_views, "_SLIVER_REL", 1e-5)
    pv = project_mesh(m, VIEWS["front"], raster_px=512, feature_angle_deg=30.0, organic=False)
    assert inside(pv) == []
    assert pv.stats["n_sliver_pairs"] == 3
    assert len(_segments(pv.visible)) == 4                          # the outline only


def test_fine_cylinder_faces_are_not_slivers():
    """The triage's inradius / longest-edge < 0.02 test would flag every side face of a
    256-section cylinder (ratio 0.004); the altitude test flags none."""
    from src.drawing.mesh_views import _degenerate_faces
    cyl = trimesh.creation.cylinder(radius=10.0, height=30.0, sections=256)
    v = cyl.vertices[cyl.faces]
    e = np.linalg.norm(v[:, [1, 2, 0]] - v, axis=2)
    ratio = (2.0 * cyl.area_faces / e.sum(axis=1)) / e.max(axis=1)
    assert np.count_nonzero(ratio < 0.02) >= 512                  # all 512 side faces
    assert not _degenerate_faces(cyl).any()


def test_weld_joins_raster_broken_rim_pieces(monkeypatch):
    """Issue 19: a jittered fine cylinder's rim breaks into pieces a chord or two apart;
    welding dangling ends within 0.3 mm on paper joins them before the chain filter."""
    from src.drawing import mesh_views
    from src.drawing.geometry2d import chain_paths, primitives_to_segments
    cyl = trimesh.creation.cylinder(radius=10.0, height=30.0, sections=256)
    rng = np.random.default_rng(0)
    cyl.vertices = cyl.vertices + rng.normal(0.0, 0.2 * 20.0 / 1024, cyl.vertices.shape)

    def chains(pv):
        o, c = chain_paths(primitives_to_segments(pv.visible, 0.01), 1e-6)
        return len(o) + len(c)

    monkeypatch.setattr(mesh_views, "_DANGLING_PAPER_MM", 0.0)     # round 3: not this test's filter
    monkeypatch.setattr(mesh_views, "_WELD_PAPER_MM", 0.0)
    pv0 = project_mesh(cyl, VIEWS["top"], raster_px=1024, feature_angle_deg=30.0, organic=False,
                       paper_scale=1.0)
    monkeypatch.setattr(mesh_views, "_WELD_PAPER_MM", 0.3)
    pv = project_mesh(cyl, VIEWS["top"], raster_px=1024, feature_angle_deg=30.0, organic=False,
                      paper_scale=1.0)
    assert pv0.stats["welded_ends"] == 0 and pv.stats["welded_ends"] >= 5
    assert chains(pv) <= chains(pv0) - 5


def test_section_mesh_fitted_pin_with_other_tessellation_keeps_its_body():
    """Review of issue 3: an exact-fit pin tessellated differently from its
    hole (48 vs 64 sections, turned 0.05 rad) has a cut loop that differs
    from the hole's by up to the sagitta. It coincides with the hole (the
    tolerance follows the polygons' own sagitta), so it keeps its body
    instead of joining the tube as a void, which would draw it white."""
    tube = trimesh.creation.annulus(r_min=5.0, r_max=10.0, height=20.0, sections=64)
    pin = trimesh.creation.cylinder(radius=5.0, height=30.0, sections=48)
    pin.apply_transform(trimesh.transformations.rotation_matrix(0.05, [0, 0, 1]))
    asm = trimesh.util.concatenate([tube, pin])
    asm.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    sv = _section_front(asm, y=0.0)
    assert len(sv.loops) == 3 and sv.loop_body is not None
    circ = sorted(((abs(polygon_area(L)), L, b) for L, b in zip(sv.loops, sv.loop_body)),
                  key=lambda t: t[0])[:2]
    # precondition: the hole and the pin loops are different polygons of r5
    boxes = [np.concatenate([L.min(axis=0), L.max(axis=0)]) for _, L, _ in circ]
    assert float(np.max(np.abs(boxes[0] - boxes[1]))) > 1e-3
    assert all(abs(a - np.pi * 25.0) < 0.5 for a, _, _ in circ)
    assert circ[0][2] != circ[1][2], sv.loop_body              # pin and tube: two bodies
    assert len(set(sv.loop_body)) == 2


# ------------------------------------------------------------- QA round 3

def _plen(p):
    if isinstance(p, Segment):
        return float(np.linalg.norm(np.subtract(p.b, p.a)))
    pts = np.asarray(p.points, float)
    ring = np.vstack([pts, pts[:1]]) if p.closed else pts
    return float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum())


def _prim_segs(p):
    pts = np.asarray(p.points, float) if isinstance(p, Polyline) else np.array([p.a, p.b], float)
    if isinstance(p, Polyline) and p.closed:
        pts = np.vstack([pts, pts[:1]])
    return np.stack([pts[:-1], pts[1:]], 1)


def _overprint_fraction(p, vis_segs, tol, step=0.25):
    """Share of the samples (``step`` apart) of primitive p within tol of a visible segment."""
    segs = _prim_segs(p)
    pts = []
    for a, b in segs:
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
        pts.append(a + (b - a) * np.linspace(0.0, 1.0, n)[:, None])
    pts = np.vstack(pts)
    a, b = vis_segs[:, 0], vis_segs[:, 1]
    ab = b - a
    t = np.clip(np.einsum("pk,vk->pv", pts, ab) - np.einsum("vk,vk->v", a, ab)[None, :], 0.0, None)
    t = np.minimum(t / np.maximum(np.einsum("vk,vk->v", ab, ab), 1e-300)[None, :], 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d = np.linalg.norm(proj - pts[:, None, :], axis=2).min(axis=1)
    return float(np.mean(d <= tol))


def _plate_flange():
    """Plate 50 x 50 x 3 (z 0..3) on a 49.6 x 49.6 x 1 flange (z -1..0): in the top
    view the flange outline is hidden 0.2 mm inside the plate outline."""
    plate = _box((50.0, 50.0, 3.0), (0.0, 0.0, 1.5))
    flange = _box((49.6, 49.6, 1.0), (0.0, 0.0, -0.5))
    return trimesh.util.concatenate([plate, flange])


def test_mesh_hidden_lines_beside_visible_outline_are_removed(monkeypatch):
    """Issue 10 (09 flange, 25): a hidden edge 0.1-0.3 mm on paper beside a visible
    outline was kept by the 0.5 px clean-up tolerance and printed over it."""
    from src.drawing import mesh_views
    m = _plate_flange()
    vis_segs = lambda pv: np.concatenate([_prim_segs(p) for p in pv.visible])  # noqa: E731
    monkeypatch.setattr(mesh_views, "_HIDDEN_COVER_PAPER_MM", 0.0)        # precondition: old rule
    monkeypatch.setattr(mesh_views, "_HIDDEN_PARALLEL_PAPER_MM", 0.0)
    pv0 = project_mesh(m, VIEWS["top"], raster_px=1024, feature_angle_deg=30.0, organic=False,
                       paper_scale=1.0)
    over0 = [p for p in pv0.hidden if _overprint_fraction(p, vis_segs(pv0), 0.3) > 0.8]
    assert over0                                                          # the flange outline
    monkeypatch.undo()
    pv = project_mesh(m, VIEWS["top"], raster_px=1024, feature_angle_deg=30.0, organic=False,
                      paper_scale=1.0)
    assert [p for p in pv.hidden if _overprint_fraction(p, vis_segs(pv), 0.3) > 0.8] == []
    assert pv.stats["hidden_covered_mm"] > 150.0                          # ~4 x 49.6 mm
    assert len(_segments(pv.visible)) + len([p for p in pv.visible if isinstance(p, Polyline)]) >= 1
    # a hidden edge 1 mm inside the outline is a real edge and stays
    plate = _box((50.0, 50.0, 3.0), (0.0, 0.0, 1.5))
    flange = _box((48.0, 48.0, 1.0), (0.0, 0.0, -0.5))
    pv2 = project_mesh(trimesh.util.concatenate([plate, flange]), VIEWS["top"], raster_px=1024,
                       feature_angle_deg=30.0, organic=False, paper_scale=1.0)
    assert sum(_plen(p) for p in pv2.hidden) > 150.0


def _tube_with_pins(n=400, seed=1):
    """Thin tube (r 40..41, height 30) with n random 1 mm cubes inside: in the front view
    every cube is hidden behind the tube wall."""
    tube = trimesh.creation.annulus(r_min=40.0, r_max=41.0, height=30.0)
    rng = np.random.default_rng(seed)
    pins = []
    for _ in range(n):
        b = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        b.apply_translation((rng.uniform(-25, 25), rng.uniform(-25, 25), rng.uniform(-12, 12)))
        pins.append(b)
    return trimesh.util.concatenate([tube] + pins)


def test_mesh_hidden_density_keeps_only_long_chains(monkeypatch):
    """Issue 2 (25 wheel hub): every hidden chain was drawn, a black mass with no flag."""
    from src.drawing import mesh_views
    m = _tube_with_pins()
    monkeypatch.setattr(mesh_views, "_HIDDEN_DENSITY_RATIO", float("inf"))   # precondition
    monkeypatch.setattr(mesh_views, "_HIDDEN_DENSITY_MAX_SEGMENTS", 10 ** 12)
    pv0 = project_mesh(m, VIEWS["front"], raster_px=512, feature_angle_deg=30.0, organic=False,
                       paper_scale=1.0)
    h0 = sum(_plen(p) for p in pv0.hidden)
    v0 = sum(_plen(p) for p in pv0.visible)
    assert h0 > 3.0 * v0 and pv0.stats["hidden_dropped"] is False
    assert sum(1 for p in pv0.hidden if _plen(p) <= 5.0) > 100
    monkeypatch.undo()
    pv = project_mesh(m, VIEWS["front"], raster_px=512, feature_angle_deg=30.0, organic=False,
                      paper_scale=1.0)
    # Z2 review: a partial thinning is NOT "hidden lines dropped" (that means none are drawn)
    assert pv.stats["hidden_reduced"] is True and pv.stats["hidden_reduced_reason"] == "density"
    assert pv.stats["hidden_dropped"] is False and "hidden_dropped_reason" not in pv.stats
    assert pv.hidden and all(_plen(p) > 5.0 for p in pv.hidden)
    assert len(pv.hidden) < len(pv0.hidden)
    assert pv.stats["hidden_density_removed"] == len(pv0.hidden) - len(pv.hidden)
    assert pv.stats["hidden_density_kept"] == len(pv.hidden)
    # Z2 review: --max-hidden-primitives none (no cap) reaches the mesh route and keeps
    # every hidden chain
    pvn = project_mesh(m, VIEWS["front"], raster_px=512, feature_angle_deg=30.0, organic=False,
                       paper_scale=1.0, max_hidden_primitives=None)
    assert len(pvn.hidden) == len(pv0.hidden) and not pvn.stats.get("hidden_reduced")
    # a small edge cap alone (ratio rule not met) triggers the cut with its own reason
    monkeypatch.setattr(mesh_views, "_HIDDEN_DENSITY_RATIO", float("inf"))
    pvc = project_mesh(m, VIEWS["front"], raster_px=512, feature_angle_deg=30.0, organic=False,
                       paper_scale=1.0, max_hidden_primitives=10)
    assert pvc.stats["hidden_edges"] > 10                                   # precondition
    assert pvc.stats["hidden_reduced"] is True
    assert pvc.stats["hidden_reduced_reason"] == "max_hidden_primitives"
    monkeypatch.undo()
    # the same tube at a scale where the pins are long on paper is not reduced... and a
    # plain box with one hidden pocket is never touched
    hb = _hollow_box()
    pvb = project_mesh(hb, VIEWS["top"], raster_px=512, feature_angle_deg=30.0, organic=False,
                       paper_scale=1.0)
    assert pvb.hidden and pvb.stats["hidden_dropped"] is False and not pvb.stats.get("hidden_reduced")


def test_hidden_density_second_step_drops_locally_dense_long_chains():
    """Issue 2, measured on 25: keeping chains > 5 mm left 88 % of the hub ink (long fold
    silhouettes); chains lying in cells above 1.5 mm of line per mm2 go, a long hidden
    line through sparse paper stays."""
    from src.drawing.mesh_views import _reduce_hidden_density
    from src.drawing.model import LineStyle
    fold = [Segment((0.0, 0.2 * k), (20.0, 0.2 * k + 3.0), LineStyle.HIDDEN) for k in range(40)]
    rim = Segment((-60.0, 40.0), (60.0, 40.0), LineStyle.HIDDEN)
    vis = [Segment((-60.0, -10.0), (60.0, -10.0)), Segment((-60.0, 50.0), (60.0, 50.0))]
    st = {}
    out = _reduce_hidden_density(fold + [rim], vis, 1.0, st)
    assert all(_plen(p) > 5.0 for p in fold + [rim])                      # precondition: all long
    assert st["hidden_reduced"] is True and st["hidden_reduced_reason"] == "density"
    assert out == [rim] and st["hidden_dense_dropped"] == 40
    assert st["hidden_paper_mm_kept"] == pytest.approx(120.0)
    assert not st.get("hidden_dropped")                                   # rim still drawn
    # when every hidden chain goes, the view really has no hidden lines: hidden_dropped
    st3 = {}
    assert _reduce_hidden_density(list(fold), vis, 1.0, st3) == []
    assert st3["hidden_dropped"] is True and st3["hidden_dropped_reason"] == "density"
    assert st3["hidden_density_kept"] == 0 and st3["hidden_density_removed"] == 40
    # below the global ratio nothing happens
    st2 = {}
    assert _reduce_hidden_density([rim], vis, 1.0, st2) == [rim] and "hidden_reduced" not in st2


def _bumped_box():
    """40 x 30 x 10 box subdivided 3 times with the top-face vertex at the origin pulled
    to (0.55, -0.92, 6.12): an oblique bump whose creases end on the face, so the top
    view at 1:20 holds a 0.31 mm crease stub with a free end."""
    m = _box((40.0, 30.0, 10.0))
    for _ in range(3):
        m = m.subdivide()
    v = np.array(m.vertices)
    i = int(np.argmin(np.linalg.norm(v - [0.0, 0.0, 5.0], axis=1)))
    assert np.allclose(v[i], [0.0, 0.0, 5.0])
    v[i] += (0.55, -0.92, 1.12)
    m.vertices = v
    return m


def _dangling_short(prims, others, lim, tol):
    """Open primitives shorter than lim with at least one end farther than tol from
    every other primitive."""
    from src.drawing.geometry2d import _prim_ends_and_length
    out = []
    for p in prims:
        el = _prim_ends_and_length(p)
        if el is None or el[1] >= lim:
            continue
        rest = [q for q in prims + others if q is not p]
        if not rest:
            out.append(p)
            continue
        segs = np.concatenate([_prim_segs(q) for q in rest])
        free = 0
        for e in el[0]:
            a, b = segs[:, 0], segs[:, 1]
            ab = b - a
            t = np.clip(((e - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-300), 0, 1)
            if np.min(np.linalg.norm(a + t[:, None] * ab - e, axis=1)) > tol:
                free += 1
        if free:
            out.append(p)
    return out


def test_mesh_short_dangling_fragments_are_dropped(monkeypatch):
    """Issue 9: sub-millimetre stubs hanging off creases of a jittered tessellation."""
    from src.drawing import mesh_views
    m = _bumped_box()
    scale = 0.05
    monkeypatch.setattr(mesh_views, "_DANGLING_PAPER_MM", 0.0)                # precondition
    pv0 = project_mesh(m, VIEWS["top"], raster_px=1024, feature_angle_deg=30.0, organic=False,
                       paper_scale=scale)
    tol = 0.3 / scale
    assert _dangling_short(pv0.visible, pv0.hidden, 0.5 / scale, tol)
    monkeypatch.undo()
    pv = project_mesh(m, VIEWS["top"], raster_px=1024, feature_angle_deg=30.0, organic=False,
                      paper_scale=scale)
    assert _dangling_short(pv.visible, pv.hidden, 0.5 / scale, tol) == []
    assert _dangling_short(pv.hidden, pv.visible, 0.5 / scale, tol) == []
    assert pv.stats["fragments_dropped"] >= 1 and "fan_dropped" in pv.stats
    assert len(pv.visible) == len(pv0.visible) - pv.stats["fragments_dropped"]
    out = np.concatenate([_prim_segs(p) for p in pv.visible]).reshape(-1, 2)
    assert np.allclose(np.ptp(out, axis=0), (40.0, 30.0), atol=0.05)      # the outline stays


def _bolt_plate():
    """Plate 100 x 60 x 10 with four r5 through holes (32 sections, trimesh boolean)."""
    plate = trimesh.creation.box(extents=(100.0, 60.0, 10.0))
    try:
        for x, y in ((-35, -18), (35, -18), (-35, 18), (35, 18)):
            c = trimesh.creation.cylinder(radius=5.0, height=30.0, sections=32)
            c.apply_translation((x, y, 0))
            plate = plate.difference(c)
    except Exception as exc:  # pragma: no cover - boolean backend missing
        pytest.skip(f"trimesh boolean not available: {exc}")
    return plate


def test_iso_keeps_bolt_circle_rims(monkeypatch):
    """Issue 21 (25, 28): the iso lost its through-holes; here the boolean's sliver
    triangles take single rim edges out of the candidates and the rim breaks into open
    pieces that the organic 3 mm chain filter drops."""
    from src.drawing import mesh_views
    m = _bolt_plate()
    spec = VIEWS["iso"]
    centres = spec.to_view(np.array([[x, y, 5.0] for x, y in ((-35, -18), (35, -18), (-35, 18),
                                                               (35, 18))]))[:, :2]
    monkeypatch.setattr(mesh_views, "_FILL_SLIVER_GAPS", False)                # precondition
    monkeypatch.setattr(mesh_views, "_ISO_WELD_PAPER_MM", mesh_views._WELD_PAPER_MM)
    monkeypatch.setattr(mesh_views, "_ISO_LOOP_CLOSE_PAPER_MM", 0.0)
    pv0 = project_mesh(m, spec, raster_px=1024, feature_angle_deg=30.0, organic=True,
                       hidden=False, paper_scale=0.5)
    assert pv0.stats["n_sliver_pairs"] > 0
    assert sum(1 for c in centres if _loops_near(pv0.visible, c, 2.0)) == 0
    monkeypatch.undo()
    pv = project_mesh(m, spec, raster_px=1024, feature_angle_deg=30.0, organic=True,
                      hidden=False, paper_scale=0.5)
    found = 0
    for c in centres:
        loops = [p for p in _loops_near(pv.visible, c, 2.0) if 25.0 < _plen(p) < 33.0]
        found += bool(loops)                             # an r5 ellipse: 2 pi 5 = 31.4 at most
    # the fourth rim (far left) is split at both extremes by 3-vertex sliver loops of
    # the boolean, which neither the gap fill nor the loop closing bridges
    assert found >= 3
    assert pv.stats["sliver_gaps_filled"] > 0 and pv.stats["loops_closed"] >= 1


def test_depth_sign_mesh_skips_the_cavity_rule_on_organic_meshes():
    """Issue 35 (29 bunny): the cavity rule turned the organic bunny to show its back."""
    from src.drawing.mesh_load import choose_depth_sign_mesh
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    v = np.array(m.vertices)
    sel = v[:, 1] > 0.3
    v[sel, 1] -= 0.6 * (v[sel, 1] - 0.3)
    m.vertices = v
    assert classify(m)["verdict"] == "organic"                              # precondition
    r0 = choose_depth_sign_mesh(m, np.eye(3))
    assert r0["method"] == "mesh_cavity"                                    # precondition
    r = choose_depth_sign_mesh(m, np.eye(3), organic=True)
    assert r["method"] == "mesh_feature_edges" and "organic: cavity rule skipped" in r["reason"]
    # control: a prismatic pocketed plate keeps the cavity rule
    rc = choose_depth_sign_mesh(_pocketed_plate(), np.eye(3), organic=False)
    assert rc["method"] == "mesh_cavity"
    # Z2 review: the flag alone decides - the same plate called organic skips the rule too,
    # and the default (no keyword) is the cavity rule
    assert choose_depth_sign_mesh(_pocketed_plate(), np.eye(3))["method"] == "mesh_cavity"
    rco = choose_depth_sign_mesh(_pocketed_plate(), np.eye(3), organic=True)
    assert rco["method"] != "mesh_cavity" and "organic: cavity rule skipped" in rco["reason"]
