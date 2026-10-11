"""Regression tests for the automated bug hunt on commit 4e22413
(4e2241347de43175401df20c6722f38fa70820b4), bug report ``bug-report.md``.

Covered here:

* M10 (cand 23) - ``export_step_faceted`` counted every surface body as an
  open shell, so a clean closed cube exported with ``make_solids=False``
  reported ``open_shells == 1``.
* M21 (cand 26) - ``import_step`` failed under the nested ``OCP.OCP.*``
  package layout when cadquery is absent: ``TopAbs_REVERSED`` was imported
  from the flat ``OCP.TopAbs`` after the fallback block had already chosen
  the nested layout. Round 4 found that no OCP release ships ``OCP.OCP``
  without the flat ``OCP.<Module>`` wrappers (see tests/test_ocp_layout.py),
  so the fallback was removed: the test now pins one flat read path that
  keeps the reversed-face winding fix, and a clear RuntimeError (no
  half-working second path) when the flat names are missing.
* M19 (cand 32) - ``FeatureTree.get_current_mesh`` / ``rebuild`` skipped a last
  feature that evaluated to ``None`` and reported the previous feature's mesh.
* S:nan_volume (report "unverified suspicions") - a NaN volume slipped through
  the volume gates of ``faceted_step.unify_verdict`` and
  ``NURBSSimplifier.simplify``, because every comparison with NaN is False.

Test discipline (the repo's vacuous-test rules): each test first asserts that
the defect / precondition really is in its input, every gate gets an input it
must say "no" to and one it must say "yes" to, and counts are bracketed.
"""
import math
import os
import sys
import types

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")


# --------------------------------------------------------------------------
# M10: open_shells with make_solids=False
# --------------------------------------------------------------------------

def _box(sx=10.0, sy=20.0, sz=30.0):
    return trimesh.creation.box(extents=(sx, sy, sz))


@needs_ocp
def test_m10_closed_surface_body_is_not_counted_as_open_shell(tmp_path):
    """M10 (cand 23): a clean closed cube exported with ``make_solids=False``
    is one CLOSED surface body: ``open_shells`` must be 0 and ``solids`` 0.
    A box with a hole must still count as one open shell."""
    from src.io.faceted_step import export_step_faceted

    box = _box()
    holed = trimesh.Trimesh(vertices=box.vertices.copy(),
                            faces=np.delete(box.faces, 0, axis=0), process=False)
    # preconditions: the box is closed, the holed box is not
    assert box.is_watertight and box.is_winding_consistent
    assert not holed.is_watertight

    report = export_step_faceted(box, str(tmp_path / "surf.step"),
                                 make_solids=False, verify=False)
    assert report["ok"] is True, report.get("error")
    assert len(report["bodies"]) == 1
    assert report["bodies"][0]["closed"] is True
    assert report["bodies"][0]["solid"] is False
    assert report["closed_measured"] == [True]
    assert report["solids"] == 0
    assert report["open_shells"] == 0, report["bodies"]

    # negative control: the same switch on a holed box still reports the hole
    bad = export_step_faceted(holed, str(tmp_path / "holed.step"),
                              make_solids=False, verify=False)
    assert bad["ok"] is True, bad.get("error")
    assert bad["bodies"][0]["closed"] is False
    assert bad["open_shells"] == 1
    assert bad["solids"] == 0

    # control with solids on: closed box -> one solid, no open shell
    solid = export_step_faceted(box, str(tmp_path / "solid.step"), verify=False)
    assert solid["solids"] == 1 and solid["open_shells"] == 0


# --------------------------------------------------------------------------
# M21: import_step reads through the flat OCP names only
# --------------------------------------------------------------------------

#: Every flat OCP module on import_step's read path.
_READ_PATH_MODULES = ("STEPControl", "BRepMesh", "TopExp", "TopAbs", "BRep",
                      "TopLoc", "TopoDS")


def _write_box_step(path, dx=10.0, dy=20.0, dz=30.0):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from src.io.exporters import export_step
    export_step(BRepPrimAPI_MakeBox(dx, dy, dz).Shape(), str(path))
    assert os.path.getsize(path) > 0


def _mesh_volume(result):
    tm = trimesh.Trimesh(vertices=result["vertices"], faces=result["faces"],
                         process=False)
    return float(tm.volume), bool(tm.is_watertight)


def _install_nested_only_layout(monkeypatch):
    """Make only ``OCP.OCP.<Mod>`` importable and hide cadquery.

    Fake modules re-export the real classes under ``OCP.OCP.*``; the flat
    ``OCP.<Mod>`` entries and ``cadquery`` are set to None in sys.modules, which
    makes their import raise ImportError. monkeypatch restores everything.
    """
    import importlib
    real = {name: importlib.import_module("OCP." + name) for name in _READ_PATH_MODULES}
    nested_pkg = types.ModuleType("OCP.OCP")
    nested_pkg.__path__ = []  # a package, so submodules can be imported
    monkeypatch.setitem(sys.modules, "OCP.OCP", nested_pkg)
    for name, mod in real.items():
        fake = types.ModuleType("OCP.OCP." + name)
        fake.__dict__.update({k: v for k, v in vars(mod).items()
                              if not k.startswith("__")})
        monkeypatch.setitem(sys.modules, "OCP.OCP." + name, fake)
        monkeypatch.setitem(sys.modules, "OCP." + name, None)
    monkeypatch.setitem(sys.modules, "cadquery", None)


@needs_ocp
def test_m21_import_step_has_one_flat_ocp_path(tmp_path, monkeypatch):
    """M21 (cand 26), revised in round 4: ``import_step`` reads through the
    flat ``OCP.<Module>`` names only. On the real layout it gives the
    outward-wound 6000 mm3 box (the TopAbs_REVERSED winding fix); with the
    flat names hidden it raises the documented RuntimeError instead of
    running a second, nested-layout code path that no release needs."""
    from src.io.importers import import_step

    path = tmp_path / "box.step"
    _write_box_step(path)

    # the real (flat) layout: a box whose reversed faces were not flipped
    # would not enclose +6000
    flat = import_step(str(path))
    flat_vol, flat_closed = _mesh_volume(flat)
    assert flat_closed and flat_vol == pytest.approx(6000.0, rel=1e-9)
    assert len(flat["faces"]) == 12 and len(flat["vertices"]) == 8

    _install_nested_only_layout(monkeypatch)
    # preconditions: the flat names and cadquery are really gone, the nested
    # ones really work
    with pytest.raises(ImportError):
        from OCP.TopAbs import TopAbs_REVERSED  # noqa: F401
    with pytest.raises(ImportError):
        from OCP.STEPControl import STEPControl_Reader  # noqa: F401
    with pytest.raises(ImportError):
        import cadquery  # noqa: F401
    from OCP.OCP.TopAbs import TopAbs_REVERSED as nested_rev  # noqa: F401
    from OCP.OCP.STEPControl import STEPControl_Reader as nested_reader
    assert nested_reader is not None

    # before round 4 this returned a mesh through the OCP.OCP fallback
    with pytest.raises(RuntimeError, match="STEP support is unavailable"):
        import_step(str(path))


# --------------------------------------------------------------------------
# M19: feature tree with an empty last result
# --------------------------------------------------------------------------

class _FakeMesh:
    def __init__(self, tag=()):
        self.tag = tuple(tag)

    def copy(self):
        return _FakeMesh(self.tag)


def _tree(names, empty=(), calls=None):
    """Tree whose evaluator appends the feature name, or returns None for the
    names in ``empty``. ``calls`` records (name, input tag or None)."""
    from src.core.feature_tree import Feature, FeatureTree
    tree = FeatureTree()
    for name in names:
        tree.features.append(Feature(name, "test", {}))
    calls = [] if calls is None else calls

    def evaluate(feature, mesh):
        calls.append((feature.name, None if mesh is None else mesh.tag))
        if feature.name in empty:
            return None
        base = mesh.tag if mesh is not None else ()
        return _FakeMesh(base + (feature.name,))
    tree.feature_evaluator = evaluate
    return tree, calls


def test_m19_empty_last_feature_reports_no_mesh():
    """M19 (cand 32): A -> mesh, B -> None. ``get_current_mesh()``, the
    ``rebuild`` return value and the rebuild callback must all be None, not
    A's mesh."""
    tree, calls = _tree(["A", "B"], empty={"B"})
    seen = []
    tree.on_rebuild_complete.append(seen.append)

    result = tree.rebuild(0)

    # preconditions: both features really ran, A really produced a mesh
    assert calls == [("A", None), ("B", ("A",))]
    assert tree.features[0].mesh_snapshot is not None
    assert tree.features[0].mesh_snapshot.tag == ("A",)
    assert tree.features[1].mesh_snapshot is None

    assert result is None
    assert tree.get_current_mesh() is None
    assert seen == [None]


def test_m19_controls_non_empty_disabled_and_following_feature():
    """M19 (cand 32) controls: the empty-result rule must not swallow real
    meshes. B producing a mesh -> ('A', 'B'); B empty but disabled -> A's mesh;
    a feature C after an empty B gets None as input and its mesh is current."""
    tree, _ = _tree(["A", "B"])
    assert tree.rebuild(0).tag == ("A", "B")
    assert tree.get_current_mesh().tag == ("A", "B")

    tree, calls = _tree(["A", "B"], empty={"B"})
    tree.features[1].enabled = False
    assert tree.rebuild(0).tag == ("A",)
    assert calls == [("A", None)]

    tree, calls = _tree(["A", "B", "C"], empty={"B"})
    assert tree.rebuild(0).tag == ("C",)
    assert calls[-1] == ("C", None)

    # partial rebuild from C, after B evaluated empty: C must still get None
    del calls[:]
    assert tree.rebuild(2).tag == ("C",)
    assert calls and calls[-1] == ("C", None)
    assert all(inp is None for name, inp in calls if name == "C")

    # empty again at the end after a partial rebuild
    tree, calls = _tree(["A", "B"], empty={"B"})
    tree.rebuild(0)
    del calls[:]
    assert tree.rebuild(1) is None
    assert calls == [("B", ("A",))]
    assert tree.get_current_mesh() is None


# --------------------------------------------------------------------------
# NaN volumes in the faceted unify gate and the simplifier
# --------------------------------------------------------------------------

def test_nan_volume_is_rejected_by_unify_verdict():
    """S:nan_volume: ``unify_verdict`` must reject a NaN (or otherwise
    non-finite) volume on either side and say why; a finite equal volume is
    still accepted (positive control)."""
    from src.io.faceted_step import unify_verdict

    nan = float("nan")
    # positive control
    ok, reason, rel = unify_verdict(12, 6, 1000.0, 1000.0, 1e-6)
    assert ok is True and reason is None and rel == 0.0
    # precondition: the old gate really lets NaN through (NaN > tol is False)
    assert not (nan > 1e-6)

    for before, after in ((1000.0, nan), (nan, 1000.0), (0.0, nan),
                          (math.inf, 1000.0), (1000.0, math.inf)):
        ok, reason, _ = unify_verdict(12, 6, before, after, 1e-6)
        assert ok is False, (before, after)
        assert "volume" in reason, reason
    ok, reason, _ = unify_verdict(12, 6, 1000.0, nan, 1e-6)
    assert "finite" in reason, reason


@needs_ocp
def test_nan_volume_merge_is_discarded_in_the_exporter(tmp_path, monkeypatch):
    """S:nan_volume: when the volume integrator returns NaN for the merged
    body, ``export_step_faceted`` must keep the faceted body (12 faces),
    record the rejection, and keep the report JSON-safe (no NaN delta)."""
    import json
    import src.io.step_audit as step_audit
    from src.io.faceted_step import export_step_faceted

    box = _box()
    good = export_step_faceted(box, str(tmp_path / "good.step"), verify=False)
    assert good["faces"] == 6 and good["bodies"][0]["unify"]["applied"] is True

    real = step_audit.brep_volume
    seen = []

    def nan_after_merge(shape):
        seen.append(1)
        # call 1 is volume_before of the merge, call 2 volume_after
        return float("nan") if len(seen) == 2 else real(shape)
    monkeypatch.setattr(step_audit, "brep_volume", nan_after_merge)

    report = export_step_faceted(box, str(tmp_path / "nan.step"), verify=False)
    assert len(seen) >= 2, "the injected integrator was never called"
    unify = report["bodies"][0]["unify"]
    assert unify["applied"] is False, unify
    assert "finite" in (unify["rejected"] or ""), unify
    assert unify["volume_rel_delta"] is None
    assert report["faces"] == 12
    assert report["solids"] == 1 and report["ok"] is True
    json.dumps(report["bodies"], allow_nan=False)


def _faceted_box_shape(a=10.0):
    """12-triangle OCC box solid built from the faceted exporter's own path."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from src.io.faceted_step import _build_direct, _shell_from_faces
    m = trimesh.creation.box(extents=(a, a, a))
    shell = _shell_from_faces(_build_direct(np.asarray(m.vertices, dtype=np.float64),
                                            np.asarray(m.faces, dtype=np.int64)))
    return BRepBuilderAPI_MakeSolid(shell).Solid()


@needs_ocp
@pytest.mark.parametrize("which,value", [("after", float("nan")),
                                         ("before", float("nan")),
                                         ("before", math.inf)])
def test_nan_volume_is_rejected_by_the_simplifier(monkeypatch, which, value):
    """S:nan_volume: ``NURBSSimplifier.simplify`` must keep the original shape
    when either measured volume is not finite and say so in the report; the
    same harness with real volumes accepts the merge (positive control)."""
    import src.io.step_audit as step_audit
    from src.nurbs.simplifier import NURBSSimplifier

    box = _faceted_box_shape()

    # positive control through the same code path
    ok = NURBSSimplifier()
    merged = ok.simplify(box)
    assert ok.last_report["accepted"] is True, ok.last_report
    assert merged is not box

    real = step_audit.brep_volume
    calls = []

    def injected(shape):
        calls.append(1)
        idx = 1 if which == "before" else 2
        return value if len(calls) == idx else real(shape)
    monkeypatch.setattr(step_audit, "brep_volume", injected)

    simp = NURBSSimplifier()
    out = simp.simplify(box)
    rep = simp.last_report
    assert len(calls) == 2, "both volumes must have been measured"
    # precondition: the injected value really reached the report
    key = "volume_before" if which == "before" else "volume_after"
    assert not math.isfinite(rep[key]), rep
    assert out is box, "a shape with a non-finite volume was accepted"
    assert rep["accepted"] is False, rep
    assert "finite" in (rep["reason"] or ""), rep


def test_nondegenerate_mask_rejects_nan_triangles():
    """S:nan_volume (area gate check): the faceted exporter's degenerate
    triangle gate is written as ``>=`` and so already rejects NaN; a clean
    triangle passes (positive control)."""
    from src.io.faceted_step import _nondegenerate_mask
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [np.nan, 0, 0]], dtype=float)
    f = np.array([[0, 1, 2], [3, 1, 2]], dtype=np.int64)
    assert _nondegenerate_mask(v, f).tolist() == [True, False]
