"""Foundation of ``src.drawing``: options, 2D model, style table, input sniff.

Test discipline: every negative test asserts its precondition first, and
each behaviour is pinned by a pair of inputs (one accepted, one rejected,
or two different answers) so a no-op or constant implementation fails.
"""
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.drawing.options import DrawingOptions  # noqa: E402
from src.drawing.model import (  # noqa: E402
    Arc, BBox2D, Circle, CylFeature, Dimension, HatchLines, LineStyle, Polygon,
    Polyline, ProjectedView, SectionView, Segment, Sheet, Text, transform_primitive)
from src.drawing.style import (  # noqa: E402
    DEFAULT_STYLE, SHEETS, STANDARD_SCALES, SheetStyle, hatch_spacing, scale_text)
from src.drawing.sniff import SniffResult, check_limits, sniff  # noqa: E402


# --------------------------------------------------------------- options --

def test_default_options_validate():
    DrawingOptions().validate()


@pytest.mark.parametrize("field_name, bad, good", [
    ("sheet", "A5", "A4L"),
    ("projection", "x", "third"),
    ("formats", ("png",), ("svg",)),
    ("timeout_s", 1, 15),
    ("views", ("front", "sideways"), ("front", "iso")),
    ("section", "diagonal", "off"),
    ("stl_to_step", "magic", "nurbs"),
    ("up", "x", "y"),
    ("orient", "random", "fixed"),
    ("text_height", 0.0, 5.0),
    ("raster_px", 4, 2048),
    ("classifier_threshold", 1.5, 0.2),
    ("memory_limit_mb", 16, 512),
    ("formats", (), ("pdf",)),
    ("timeout_s", float("inf"), 60.0),
    ("max_input_mb", float("inf"), 50.0),
    ("hlr_time_budget_s", float("inf"), 30.0),
    ("text_height", float("nan"), 3.5),
])
def test_validate_rejects_bad_field_and_accepts_good(field_name, bad, good):
    # the good value must pass (so a validate() that always raises fails)
    dataclasses.replace(DrawingOptions(), **{field_name: good}).validate()
    opts = dataclasses.replace(DrawingOptions(), **{field_name: bad})
    with pytest.raises(ValueError) as exc:
        opts.validate()
    assert field_name in str(exc.value)


def _all_non_default():
    """Options with EVERY field changed from its default."""
    return DrawingOptions(
        formats=("svg",), views=("front", "top"), section="top", sheet="A4",
        projection="third", orient="fixed", up="y", text_height=5.0,
        hidden_lines=False, hole_positions=False, max_hole_callouts=3,
        stl_to_step="faceted", faceted_max_triangles=1234,
        facet_smooth_angle_deg=12.5, feature_angle_deg=40.0, raster_px=512,
        exact_hlr_max_faces=100, hlr_time_budget_s=30.0,
        classifier_threshold=0.2, max_input_mb=50.0, max_triangles=1000,
        timeout_s=60.0, memory_limit_mb=1024, output_cap_mb=50,
        title="Bracket Ø20", owner="ACME", quiet=True)


def test_json_round_trip_every_field():
    default = DrawingOptions()
    opts = _all_non_default()
    # precondition: every field differs from the default, so a from_json that
    # ignores its input and returns defaults cannot pass
    for f in dataclasses.fields(DrawingOptions):
        assert getattr(opts, f.name) != getattr(default, f.name), f.name
    text = opts.to_json()
    json.loads(text)  # valid JSON
    back = DrawingOptions.from_json(text)
    for f in dataclasses.fields(DrawingOptions):
        assert getattr(back, f.name) == getattr(opts, f.name), f.name
        assert type(getattr(back, f.name)) is type(getattr(opts, f.name)), f.name
    assert back == opts


def test_from_json_rejects_unknown_key():
    with pytest.raises(ValueError):
        DrawingOptions.from_json(json.dumps({"sheet": "A3", "bogus": 1}))


def test_from_json_infinity_token_fails_validation():
    # json.loads accepts the non-standard "Infinity" token; validate must not
    opts = DrawingOptions.from_json('{"timeout_s": Infinity}')
    assert opts.timeout_s == float("inf")   # precondition: it really parsed to inf
    with pytest.raises(ValueError) as exc:
        opts.validate()
    assert "timeout_s" in str(exc.value)


def test_with_overrides_leaves_original_untouched():
    base = DrawingOptions()
    new = base.with_overrides(sheet="A4")
    assert new.sheet == "A4"
    assert base.sheet == "A3"
    assert new is not base
    assert new.views == base.views


# ----------------------------------------------------------------- model --

def test_bbox_of_circle_and_segment():
    bb = BBox2D.of_primitives([Circle((0, 0), 5), Segment((0, 0), (10, 0))])
    assert (bb.xmin, bb.ymin, bb.xmax, bb.ymax) == (-5, -5, 10, 5)
    assert bb.width == 15 and bb.height == 10


def test_bbox_arc_uses_full_circle():
    # a 0..10 degree arc: true extent is tiny in y, the documented rule
    # (full circle) gives the whole circle
    bb = BBox2D.of_primitives([Arc((1, 2), 3, 0.0, 10.0)])
    assert (bb.xmin, bb.ymin, bb.xmax, bb.ymax) == (-2, -1, 4, 5)


def test_bbox_empty_is_none_and_polyline_box():
    assert BBox2D.of_primitives([]) is None
    bb = BBox2D.of_primitives([Polyline(np.array([[1.0, 1.0], [4.0, -2.0], [2.0, 7.0]]))])
    assert (bb.xmin, bb.ymin, bb.xmax, bb.ymax) == (1, -2, 4, 7)


def test_bbox_helpers():
    a = BBox2D(0, 0, 1, 1)
    b = BBox2D(2, -1, 3, 0.5)
    u = a.union(b)
    assert (u.xmin, u.ymin, u.xmax, u.ymax) == (0, -1, 3, 1)
    e = a.expanded(2)
    assert (e.xmin, e.ymin, e.xmax, e.ymax) == (-2, -2, 3, 3)
    assert a.contains((0.5, 0.5)) and not a.contains((1.5, 0.5))
    f = BBox2D.from_points(np.array([[3.0, 4.0], [-1.0, 2.0]]))
    assert (f.xmin, f.ymin, f.xmax, f.ymax) == (-1, 2, 3, 4)


def test_bbox_of_text_uses_estimated_extent():
    h = 3.5
    w = 0.6 * h * len("LONG TEXT")
    bb = BBox2D.of_primitives([Text((0, 0), "LONG TEXT", h)])   # middle, baseline
    np.testing.assert_allclose([bb.xmin, bb.ymin, bb.xmax, bb.ymax],
                               [-w / 2, -0.25 * h, w / 2, h])
    bb = BBox2D.of_primitives([Text((10, 5), "AB", 5.0, anchor="start", valign="middle")])
    np.testing.assert_allclose([bb.xmin, bb.ymin, bb.xmax, bb.ymax], [10, 2.5, 16, 7.5])
    bb = BBox2D.of_primitives([Text((0, 0), "AB", 5.0, anchor="end")])
    np.testing.assert_allclose([bb.xmin, bb.xmax], [-6, 0])
    # rotated 90 degrees: the text runs up the y axis, the height lies along -x
    bb = BBox2D.of_primitives([Text((0, 0), "AB", 5.0, rotation_deg=90, anchor="start")])
    np.testing.assert_allclose([bb.xmin, bb.ymin, bb.xmax, bb.ymax], [-5, 0, 1.25, 6], atol=1e-12)


def test_transform_segment():
    s = transform_primitive(Segment((1, 2), (3, 4), LineStyle.HIDDEN), offset=(10, 10), scale=2)
    assert s.a == (12, 14) and s.b == (16, 18)
    assert s.style is LineStyle.HIDDEN


def test_transform_circle_scales_radius():
    c = transform_primitive(Circle((1, 1), 3), offset=(10, 10), scale=2)
    assert c.center == (12, 12) and c.radius == 6


def test_transform_arc_keeps_angles():
    a = transform_primitive(Arc((0, 0), 2, 30, 120), offset=(5, 0), scale=0.5)
    assert a.center == (5, 0) and a.radius == 1
    assert (a.start_deg, a.end_deg) == (30, 120)


def test_transform_text_height_not_scaled():
    t = transform_primitive(Text((1, 1), "A-A", 5.0, rotation_deg=90), offset=(10, 10), scale=2)
    assert t.position == (12, 12)
    assert t.height == 5.0
    assert t.rotation_deg == 90 and t.text == "A-A"


def test_transform_hatch_and_polyline_and_polygon():
    segs = np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 0.0], [2.0, 3.0]]])
    loop = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]])
    h = transform_primitive(HatchLines(segs, [loop]), offset=(10, 10), scale=2)
    np.testing.assert_allclose(h.segments, segs * 2 + 10)
    np.testing.assert_allclose(h.loops[0], loop * 2 + 10)
    p = transform_primitive(Polyline(loop, closed=True), offset=(1, 0), scale=3)
    np.testing.assert_allclose(p.points, loop * 3 + [1, 0])
    assert p.closed
    g = transform_primitive(Polygon(loop), offset=(0, 1), scale=1)
    np.testing.assert_allclose(g.points, loop + [0, 1])
    # the source is not modified
    np.testing.assert_allclose(segs[0, 1], [1.0, 1.0])


def test_shared_result_objects_construct():
    v = ProjectedView(spec=None, visible=[Segment((0, 0), (1, 0))], hidden=[],
                      bbox=BBox2D(0, 0, 1, 0), algorithm="hlr_exact")
    s = SectionView(spec=None, plane_point=np.zeros(3), plane_normal=np.array([0, -1.0, 0]),
                    loops=[], body=v)
    assert s.label == "A" and s.marker_view == "top" and s.stats == {}
    c = CylFeature(np.zeros(3), np.array([0, 0, 1.0]), 3.25, 10.0, "hole", True, 2)
    assert c.kind == "hole"
    d = Dimension(kind="diameter", text="4 x Ø6.5", value=6.5, count=4)
    assert d.leader_angle_deg == 45.0
    sh = Sheet("A3", 420, 297)
    assert sh.items == [] and sh.dimensions == [] and sh.meta == {}
    assert LineStyle("hidden") is LineStyle.HIDDEN and LineStyle.CUTTING == "cutting"


# ----------------------------------------------------------------- style --

def test_scale_text():
    assert scale_text(0.5) == "1:2"
    assert scale_text(2) == "2:1"
    assert scale_text(1) == "1:1"
    assert scale_text(1 / 100) == "1:100"
    assert scale_text(50) == "50:1"


def test_standard_scales_strictly_decreasing():
    assert len(STANDARD_SCALES) == 12
    assert all(a > b for a, b in zip(STANDARD_SCALES, STANDARD_SCALES[1:]))
    assert 1 in STANDARD_SCALES


def test_hatch_spacing_rule():
    assert hatch_spacing(DEFAULT_STYLE, 15) == 2.0
    assert hatch_spacing(DEFAULT_STYLE, 50) == 3.0
    assert hatch_spacing(DEFAULT_STYLE, 100) == 4.0
    tiny = SheetStyle(hatch_spacing_small=0.3)
    assert hatch_spacing(tiny, 5) == 1.0   # floor


def test_sheets_and_style_numbers():
    assert (SHEETS["A3"].width, SHEETS["A3"].height) == (420, 297)
    assert SHEETS["A4"].orientation == "portrait" and SHEETS["A4"].iso5457
    assert not SHEETS["A4L"].iso5457
    st = DEFAULT_STYLE
    assert st.dash_hidden == (3.0, 0.75)
    assert st.dash_center == (6.0, 0.75, 0.125, 0.75)
    assert st.dash_cutting == (12.0, 1.5, 0.25, 1.5)
    assert st.title_block_h == 4 * st.title_row_h
    assert st.title_cols[-1] == st.title_block_w
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.wide = 1.0


# ----------------------------------------------------------------- sniff --

@pytest.fixture
def tmpdir_path():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _box():
    import trimesh
    return trimesh.creation.box(extents=(10, 20, 30))


def test_sniff_binary_stl(tmpdir_path):
    p = os.path.join(tmpdir_path, "box.stl")
    _box().export(p)
    assert os.path.getsize(p) == 84 + 50 * 12
    s = sniff(p)
    assert isinstance(s, SniffResult)
    assert s.kind == "stl_binary" and s.triangles == 12
    assert s.extension == ".stl" and s.extension_mismatch is False
    assert s.size_bytes == os.path.getsize(p)
    with open(p, "rb") as fh:
        assert s.sha256 == hashlib.sha256(fh.read()).hexdigest()


def test_sniff_binary_with_solid_header(tmpdir_path):
    p = os.path.join(tmpdir_path, "evil.stl")
    _box().export(p)
    with open(p, "r+b") as fh:
        fh.write(b"solid")
    with open(p, "rb") as fh:
        assert fh.read(5) == b"solid"   # precondition
    s = sniff(p)
    assert s.kind == "stl_binary" and s.triangles == 12


def test_sniff_ascii_stl(tmpdir_path):
    p = os.path.join(tmpdir_path, "box_ascii.STL")
    _box().export(p, file_type="stl_ascii")
    with open(p, "rb") as fh:
        assert fh.read(5) == b"solid"   # precondition: really ASCII
    s = sniff(p)
    assert s.kind == "stl_ascii" and s.triangles == 12
    assert s.extension == ".stl" and not s.extension_mismatch


def test_sniff_truncated_binary_is_unknown(tmpdir_path):
    p = os.path.join(tmpdir_path, "trunc.stl")
    _box().export(p)
    full = os.path.getsize(p)
    with open(p, "r+b") as fh:
        fh.truncate(full - 10)
    assert os.path.getsize(p) == full - 10   # precondition
    s = sniff(p)
    assert s.kind == "unknown" and s.triangles is None


def test_sniff_step_text_in_stl_file(tmpdir_path):
    p = os.path.join(tmpdir_path, "fake.stl")
    with open(p, "wb") as fh:
        fh.write(b"ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'2;1');\nENDSEC;\n")
    s = sniff(p)
    assert s.kind == "step" and s.triangles is None
    assert s.extension_mismatch is True
    q = os.path.join(tmpdir_path, "real.stp")
    with open(q, "wb") as fh:
        fh.write(b"ISO-10303-21;\nHEADER;\nENDSEC;\n")
    r = sniff(q)
    assert r.kind == "step" and r.extension == ".stp" and r.extension_mismatch is False


def test_sniff_empty_and_random_are_unknown(tmpdir_path):
    p = os.path.join(tmpdir_path, "empty.stl")
    open(p, "wb").close()
    s = sniff(p)
    assert s.kind == "unknown" and s.size_bytes == 0
    q = os.path.join(tmpdir_path, "junk.stl")
    rng = np.random.default_rng(7)
    data = rng.integers(0, 256, 300, dtype=np.uint8).tobytes()
    assert not data.startswith(b"solid") and not data.startswith(b"ISO")   # precondition
    with open(q, "wb") as fh:
        fh.write(data)
    assert sniff(q).kind == "unknown"


def test_sniff_without_hash(tmpdir_path):
    p = os.path.join(tmpdir_path, "box.stl")
    _box().export(p)
    assert sniff(p, compute_hash=False).sha256 == ""
    assert len(sniff(p).sha256) == 64


def test_sniff_max_bytes_skips_scan_and_hash(tmpdir_path):
    p = os.path.join(tmpdir_path, "box_ascii.stl")
    _box().export(p, file_type="stl_ascii")
    q = os.path.join(tmpdir_path, "box.stl")
    _box().export(q)
    size_p, size_q = os.path.getsize(p), os.path.getsize(q)
    # precondition: below the cap the full scan and hash run
    full = sniff(p, max_bytes=size_p)
    assert full.triangles == 12 and len(full.sha256) == 64
    head = sniff(p, max_bytes=size_p - 1)
    assert head.kind == "stl_ascii" and head.triangles is None and head.sha256 == ""
    assert check_limits(head, max_input_mb=(size_p - 1) / 2 ** 20, max_triangles=10) is not None
    b = sniff(q, max_bytes=100)
    assert size_q > 100   # precondition
    assert b.kind == "stl_binary" and b.triangles == 12 and b.sha256 == ""


def test_sniff_facet_scan_memory_is_bounded(tmpdir_path):
    import tracemalloc
    p = os.path.join(tmpdir_path, "noline.stl")
    n_bytes = 8 * 2 ** 20
    with open(p, "wb") as fh:
        fh.write(b"solid ")
        fh.write(b"\0" * n_bytes)
    assert os.path.getsize(p) > n_bytes   # precondition: no newline anywhere
    tracemalloc.start()
    try:
        s = sniff(p, compute_hash=False)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert s.kind == "unknown"
    # a line-iterating scan reads the whole file as one line (peak > 8 MB)
    assert peak < 1 * 2 ** 20, peak


def test_check_limits(tmpdir_path):
    ok = SniffResult("stl_binary", 1000, 12, "", ".stl", False)
    assert check_limits(ok, max_input_mb=1.0, max_triangles=100) is None
    big = SniffResult("stl_binary", 2 * 2 ** 20, 12, "", ".stl", False)
    assert big.size_bytes > 1.0 * 2 ** 20   # precondition
    msg = check_limits(big, max_input_mb=1.0, max_triangles=100)
    assert isinstance(msg, str) and msg
    many = SniffResult("stl_ascii", 1000, 101, "", ".stl", False)
    msg2 = check_limits(many, max_input_mb=1.0, max_triangles=100)
    assert isinstance(msg2, str) and "101" in msg2
    step = SniffResult("step", 1000, None, "", ".step", False)
    assert check_limits(step, max_input_mb=1.0, max_triangles=100) is None


# --------------------------------------------------------------- package --

def test_package_import_is_light():
    # the sandbox parent imports options/sniff/style before spawning the
    # child: numpy there would cost ~628 MB of OpenBLAS commit
    code = ("import sys; import src.drawing as d; "
            "import src.drawing.options, src.drawing.sniff, src.drawing.style; "
            "print(d.__version__, callable(d.make_drawing), "
            "'trimesh' in sys.modules, 'OCP' in sys.modules, 'numpy' in sys.modules)")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True,
                         text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["0.1.0", "True", "False", "False", "False"]
