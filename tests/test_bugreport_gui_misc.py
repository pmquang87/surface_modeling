"""Regression tests for the GUI viewport / panels / actions entries of the bug
report written by the automated bug hunt on commit
4e2241347de43175401df20c6722f38fa70820b4.

Covers:

* M8 (cand 28)  edge picking took the edge with the nearest MIDPOINT instead of
  the nearest edge (src/gui/viewport.py, MeshViewport._on_click).
* M9 (cand 29)  PropertiesPanel.clear() left the Apply/Cancel buttons of
  set_tool_ui alive: they sit in a nested QHBoxLayout whose item.widget() is
  None (src/gui/panels.py).
* L1 (cand 4)   the Shell/Thicken action's apply() read dlg.spin_thick /
  dlg.combo_dir, which ShellThickenDialog does not have (src/gui/actions.py).
* Suspicion "viewport picking maps PyVista cell ids straight to face indices":
  NOT a bug. HalfEdgeMesh.add_face refuses faces with < 3 vertices, so
  to_pyvista() emits exactly one cell per face, and the dataset pyvista hands
  the mapper (a normals-computing copy under smooth_shading) keeps that cell
  count and order. The guard test below pins that invariant.

Everything here runs with real PySide6 widgets that are never shown, so it
works under QT_QPA_PLATFORM=offscreen. MeshViewport itself needs a live VTK
render window (see test_gui_fixes.py), so _on_click is driven as an unbound
method on a stub that carries only what the method touches, with a stub
vtkCellPicker standing in for the hardware pick.
"""
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ.setdefault('QT_API', 'pyside6')

import numpy as np
import pytest
import vtk

import shiboken6
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from src.core.halfedge_mesh import HalfEdgeMesh
from src.gui import viewport as viewport_module
from src.gui.viewport import MeshViewport
from src.gui.panels import PropertiesPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def _flush_deletes(app):
    """Run pending deleteLater() calls, as the event loop would."""
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


# ----------------------------------------------------------------------------
# M8: nearest edge, not nearest midpoint
# ----------------------------------------------------------------------------

# Triangle of the report: the click (8, 0, 0) lies ON edge 0-1, but the
# midpoint of edge 1-2 (9.5, 0.5, 0) is nearer to it than (5, 0, 0).
_TRI_V = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [9.0, 1.0, 0.0]])
_CLICK = np.array([8.0, 0.0, 0.0])


def _edge_by_ends(mesh, a, b):
    for e in mesh.edges:
        ends = {e.half_edge.vertex.index, e.half_edge.prev.vertex.index}
        if ends == {a, b}:
            return e
    raise AssertionError(f"no edge {a}-{b}")


def _report_triangle():
    mesh = HalfEdgeMesh.from_arrays(_TRI_V, [[0, 1, 2]])
    assert len(mesh.faces) == 1 and len(mesh.edges) == 3
    return mesh


def _midpoint_choice(edges, p):
    """The old rule, kept here only to prove the input really trips it."""
    best, best_d = None, float('inf')
    for e in edges:
        mid = (e.half_edge.vertex.position + e.half_edge.prev.vertex.position) / 2.0
        d = np.linalg.norm(p - mid)
        if d < best_d:
            best, best_d = e.index, d
    return best


def test_m8_point_segment_distance_exact_values():
    """M8 (cand 28): the distance used for edge picking is the point-to-SEGMENT
    distance: zero on the segment, the endpoint distance when the projection
    falls outside [0, 1], the perpendicular distance otherwise, and the plain
    point distance for a zero-length segment."""
    dist = viewport_module.point_segment_distance
    a, b, c = _TRI_V
    # On edge 0-1.
    assert dist(_CLICK, a, b) == pytest.approx(0.0, abs=1e-12)
    # Edge 1-2: projection parameter is exactly 1 -> nearest point is (9,1,0).
    assert dist(_CLICK, b, c) == pytest.approx(np.sqrt(2.0), abs=1e-12)
    # Edge 2-0: interior projection, perpendicular distance 8/sqrt(82).
    assert dist(_CLICK, c, a) == pytest.approx(8.0 / np.sqrt(82.0), abs=1e-12)
    # Projection beyond the end: (5, 0.2) vs segment (0,0)-(1,0) is the
    # distance to (1,0), NOT the 0.2 of the infinite line.
    p = np.array([5.0, 0.2, 0.0])
    assert dist(p, np.zeros(3), np.array([1.0, 0.0, 0.0])) == pytest.approx(
        np.hypot(4.0, 0.2), abs=1e-12)
    # Degenerate segment.
    assert dist(p, np.ones(3), np.ones(3)) == pytest.approx(
        np.linalg.norm(p - np.ones(3)), abs=1e-12)


def test_m8_nearest_edge_helper_picks_edge_under_click():
    """M8 (cand 28): nearest_edge_index returns the edge the point lies on
    (edge 0-1 for the report's triangle and click), where the midpoint rule
    returned edge 1-2; and it clamps, so an edge whose infinite LINE passes
    close but whose segment is far is not chosen."""
    mesh = _report_triangle()
    edges = mesh.get_face_edges(mesh.faces[0])
    e01 = _edge_by_ends(mesh, 0, 1).index
    e12 = _edge_by_ends(mesh, 1, 2).index
    # Precondition: the input really defeats the midpoint rule.
    assert _midpoint_choice(edges, _CLICK) == e12 != e01

    assert viewport_module.nearest_edge_index(edges, _CLICK) == e01

    # Clamping case on a quad (0,0)-(1,0)-(10,1)-(0,1). Point (5, 0.2):
    # line of edge 0-1 is 0.2 away, but the segment is 4.005 away; edge 1-2
    # (segment) is 2.2/sqrt(82) = 0.243 away and must win.
    quad = HalfEdgeMesh.from_arrays(
        np.array([[0, 0, 0], [1, 0, 0], [10, 1, 0], [0, 1, 0]], float),
        [[0, 1, 2, 3]])
    qedges = quad.get_face_edges(quad.faces[0])
    p = np.array([5.0, 0.2, 0.0])
    q01 = _edge_by_ends(quad, 0, 1).index
    q12 = _edge_by_ends(quad, 1, 2).index
    assert viewport_module.point_segment_distance(
        p, quad.vertices[1].position, quad.vertices[2].position
    ) == pytest.approx(2.2 / np.sqrt(82.0), abs=1e-12)
    assert viewport_module.nearest_edge_index(qedges, p) == q12 != q01
    assert viewport_module.nearest_edge_index([], p) is None


class _StubPicker:
    """Stands in for vtk.vtkCellPicker: always hits `cell_id` at `position`."""
    cell_id = 0
    position = (0.0, 0.0, 0.0)

    def SetTolerance(self, tol):
        pass

    def AddPickList(self, actor):
        pass

    def PickFromListOn(self):
        pass

    def Pick(self, x, y, z, renderer):
        return 1

    def GetCellId(self):
        return self.cell_id

    def GetPickPosition(self):
        return self.position


def _click_stub(mesh, mode):
    applied, emitted = [], []
    stub = types.SimpleNamespace(
        current_mesh=mesh,
        selection_mode=mode,
        mesh_actor=None,
        plotter=types.SimpleNamespace(mouse_position=(10, 10), renderer=None),
        _apply_selection_modifier=lambda ids: applied.append(list(ids)),
        edge_selected=types.SimpleNamespace(emit=emitted.append),
        vertex_selected=types.SimpleNamespace(emit=emitted.append),
        face_selected=types.SimpleNamespace(emit=emitted.append),
    )
    return stub, applied, emitted


def test_m8_on_click_edge_mode_selects_edge_under_click(monkeypatch):
    """M8 (cand 28): a click at (8,0,0) on triangle (0,0,0),(10,0,0),(9,1,0)
    in edge mode must select edge 0-1 (the click lies on it), not edge 1-2
    (whose midpoint happens to be nearer). Drives the real _on_click."""
    mesh = _report_triangle()
    e01 = _edge_by_ends(mesh, 0, 1).index
    e12 = _edge_by_ends(mesh, 1, 2).index
    assert _midpoint_choice(mesh.edges, _CLICK) == e12 != e01

    picker = type("P", (_StubPicker,), {"cell_id": 0, "position": tuple(_CLICK)})
    monkeypatch.setattr(vtk, "vtkCellPicker", picker)

    stub, applied, emitted = _click_stub(mesh, 'edge')
    MeshViewport._on_click(stub, None)
    assert applied == [[e01]]
    assert emitted == [e01]

    # The same click near vertex 1 picks edge 1-2 once it is genuinely nearer:
    # (9.6, 0.3, 0) is 0.3 from edge 0-1 but 0.0707 from edge 1-2.
    p = (9.6, 0.3, 0.0)
    assert viewport_module.point_segment_distance(
        np.array(p), _TRI_V[1], _TRI_V[2]) == pytest.approx(0.1 / np.sqrt(2.0), abs=1e-12)
    picker2 = type("P2", (_StubPicker,), {"cell_id": 0, "position": p})
    monkeypatch.setattr(vtk, "vtkCellPicker", picker2)
    stub, applied, emitted = _click_stub(mesh, 'edge')
    MeshViewport._on_click(stub, None)
    assert applied == [[e12]]
    assert emitted == [e12]
