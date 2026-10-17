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


def _cavity_wall_segments(prims, px):
    tol = 0.5 * px
    out = []
    for s in _segments(prims):
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
