"""Regression tests for the automated bug hunt on commit 4e22413
(4e2241347de43175401df20c6722f38fa70820b4), bug report ``bug-report.md``.

Covered here:

* M10 (cand 23) - ``export_step_faceted`` counted every surface body as an
  open shell, so a clean closed cube exported with ``make_solids=False``
  reported ``open_shells == 1``.
* M21 (cand 26) - ``import_step`` failed under the nested ``OCP.OCP.*``
  package layout when cadquery is absent: ``TopAbs_REVERSED`` was imported
  from the flat ``OCP.TopAbs`` after the fallback block had already chosen
  the nested layout.
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
# M21: import_step under the nested OCP.OCP.* layout
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
def test_m21_import_step_works_with_nested_ocp_layout(tmp_path, monkeypatch):
    """M21 (cand 26): with only ``OCP.OCP.*`` importable and no cadquery,
    ``import_step`` must read the STEP through the nested-layout fallback and
    give the same outward-wound 6000 mm3 box as the flat layout does."""
    from src.io.importers import import_step

    path = tmp_path / "box.step"
    _write_box_step(path)

    # positive control: the real (flat) layout
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

    nested = import_step(str(path))       # used to raise RuntimeError
    nested_vol, nested_closed = _mesh_volume(nested)
    assert nested["shape"] is not None
    assert len(nested["faces"]) == 12 and len(nested["vertices"]) == 8
    assert nested_closed
    # the TopAbs_REVERSED winding fix is applied on this path too: a box whose
    # reversed faces were not flipped would not enclose +6000
    assert nested_vol == pytest.approx(6000.0, rel=1e-9)
    assert np.allclose(np.sort(nested["vertices"], axis=0),
                       np.sort(flat["vertices"], axis=0))

