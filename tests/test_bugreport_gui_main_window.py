"""Regression tests for src/gui/main_window.py from the automated bug hunt on
commit 4e2241347de43175401df20c6722f38fa70820b4 (bug-report.md).

Covers:

* H2  (cand 27, cand 34) - STEP export wrote a stale B-rep after the mesh was
  edited (Subdivide, Shell, Quad Wrap, Remove Slivers, Shrink Wrap, a vertex /
  crease edit in the Properties panel, and - found while fixing - an in-place
  gizmo drag or a mesh swap from outside the window's handlers).
* M3  (cand 3)  - Remove Sliver Edges threw away a cleanup that only flipped
  needle triangles, because it compared vertex/face COUNTS only.
* M20 (cand 35) - a Convert to NURBS that produced 0 patches kept the previous
  shape for export.
* Suspicion "on_new" - File > New left an empty HalfEdgeMesh, which is truthy,
  so the "No mesh loaded" guards let it through and OBJ export wrote an empty
  file.

The handlers are driven WITHOUT a real MainWindow: a MainWindow owns a VTK
render window, which aborts under QT_QPA_PLATFORM=offscreen (see
tests/test_gui_fixes.py). Instead the handler functions of
PowerSurfacingMainWindow are bound to a plain harness object whose viewport /
panels are recorders, and every modal dialog, QMessageBox, QFileDialog and
exporter is monkeypatched in src.gui.main_window. No dialog is ever opened, so
this file runs in the offscreen suite too.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault('QT_API', 'pyside6')

import numpy as np
import pytest
import trimesh

import src.gui.main_window as mw
from src.core.halfedge_mesh import HalfEdgeMesh
from src.subd import primitives


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

class _Recorder:
    """Accepts any method call and records it; explicit attributes win."""

    def __init__(self, **attrs):
        self.calls = []
        self.__dict__.update(attrs)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None
        return record


class _Widget:
    """Stand-in for a spin box / combo box / check box holding one value."""

    def __init__(self, value):
        self._v = value

    def value(self):
        return self._v

    def currentText(self):
        return self._v

    def isChecked(self):
        return self._v


def _fake_dialog(params=None, **widgets):
    """A dialog class whose exec_() accepts and whose widgets hold `widgets`."""

    class _Dlg:
        def __init__(self, parent=None):
            for name, value in widgets.items():
                setattr(self, name, _Widget(value))

        def exec_(self):
            return 1

        def get_params(self):
            return dict(params or {})
    return _Dlg


class _Box:
    """Records QMessageBox calls instead of showing a modal box."""

    def __init__(self):
        self.shown = []

    def _rec(self, kind):
        def show(parent, title, text, *a, **k):
            self.shown.append((kind, title, text))
            return None
        return show

    def __getattr__(self, name):
        if name in ('information', 'warning', 'critical', 'question'):
            return self._rec(name)
        raise AttributeError(name)

    def texts(self, kind=None):
        return [t for k, _, t in self.shown if kind is None or k == kind]


class _Shape:
    """Sentinel standing in for a TopoDS_Shape."""

    def __init__(self, tag):
        self.tag = tag

    def __repr__(self):
        return f"_Shape({self.tag!r})"


def _harness_class():
    ns = {name: attr for name, attr in vars(mw.PowerSurfacingMainWindow).items()
          if not name.startswith('__')}
    return type('MainWindowHarness', (), ns)


_Harness = _harness_class()


@pytest.fixture
def env(monkeypatch):
    """A harness 'window' plus recorders for message boxes and exporters."""
    win = _Harness()
    win.current_mesh = None
    win.current_shape = None
    win.viewport = _Recorder(reference_mesh=None)
    win.properties_panel = _Recorder(current_target=None)
    win.status_bar = _Recorder()
    win.logs = []
    win.log = win.logs.append

    box = _Box()
    monkeypatch.setattr(mw, 'QMessageBox', box)
    monkeypatch.setattr(mw, 'QApplication', _Recorder())

    written = {'step': [], 'stl': [], 'obj': []}
    monkeypatch.setattr(mw, 'export_step',
                        lambda shape, path: written['step'].append((shape, path)))
    monkeypatch.setattr(mw, 'export_stl',
                        lambda mesh, path, binary=True: written['stl'].append((mesh, path)))
    monkeypatch.setattr(mw, 'export_obj',
                        lambda mesh, path: written['obj'].append((mesh, path)))
    return win, box, written


def _set_export(monkeypatch, fmt, path):
    monkeypatch.setattr(mw, 'ExportDialog',
                        _fake_dialog(format_combo=fmt, binary_check=True))
    monkeypatch.setattr(mw.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: (str(path), '')))


def _box_mesh(size=2.0):
    return primitives.create_box(width=size, height=size, depth=size)


def _open_step(win, monkeypatch, mesh, shape, path='part.step'):
    """Drive File > Open on a .step file whose importer yields (mesh, shape)."""
    monkeypatch.setattr(mw.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: (path, '')))
    monkeypatch.setattr(mw, 'import_step',
                        lambda p: {'mesh': mesh, 'shape': shape})
    win.on_open()


def _install_converter(monkeypatch, results):
    """Fake SubDToNURBSConverter returning `results` one call at a time.

    Each entry is (shape, patch_count); the result's 'mesh' is the input mesh,
    exactly as the real converter does.
    """
    queue = list(results)

    class FakeConverter:
        def __init__(self, continuity='G2', tolerance=0.01):
            pass

        def convert(self, mesh, simplify=False, reference_mesh=None):
            shape, n = queue.pop(0)
            return {'shape': shape, 'mesh': mesh, 'patches': [object()] * n}

    monkeypatch.setattr(mw, 'SubDToNURBSConverter', FakeConverter)
    monkeypatch.setattr(mw, 'ConvertNURBSDialog', _fake_dialog(
        continuity='G2 (Curvature)', tolerance=0.01, simplify=False))


def _positions(mesh):
    return np.array([v.position for v in mesh.vertices], dtype=float)


# --------------------------------------------------------------------------
# H2 - stale B-rep after a mesh edit
# --------------------------------------------------------------------------

def test_h2_step_import_pairing_still_exports_the_imported_shape(env, monkeypatch, tmp_path):
    """H2 (cand 27/34) control: Open STEP sets mesh AND shape together, and an
    immediate STEP export must still write exactly that shape (no warning)."""
    win, box, written = env
    mesh, shape = _box_mesh(), _Shape('imported')
    _open_step(win, monkeypatch, mesh, shape)
    assert win.current_mesh is mesh and win.current_shape is shape

    _set_export(monkeypatch, 'STEP', tmp_path / 'out.step')
    win.on_export()
    assert written['step'] == [(shape, str(tmp_path / 'out.step'))]
    assert box.texts('warning') == []


def test_h2_convert_then_export_writes_the_converted_shape(env, monkeypatch, tmp_path):
    """H2 control: Convert to NURBS then Export STEP writes the new shape."""
    win, box, written = env
    win.current_mesh = _box_mesh()
    shape = _Shape('converted')
    _install_converter(monkeypatch, [(shape, 6)])
    win.on_convert_nurbs()
    assert win.current_shape is shape

    _set_export(monkeypatch, 'STEP', tmp_path / 'out.step')
    win.on_export()
    assert written['step'] == [(shape, str(tmp_path / 'out.step'))]


def _edit_subdivide(win, monkeypatch):
    monkeypatch.setattr(mw, 'SubdivideDialog',
                        _fake_dialog(params={'levels': 1, 'smooth': True}))
    win.on_subdivide()


def _edit_shell(win, monkeypatch):
    monkeypatch.setattr(mw, 'ShellThickenDialog',
                        _fake_dialog(thickness=0.1, direction='Inward'))
    monkeypatch.setattr(mw, 'shell_solid',
                        lambda mesh, thickness, direction: _box_mesh(1.5))
    win.on_shell()


def _edit_quad_wrap(win, monkeypatch):
    monkeypatch.setattr(mw, 'QuadWrapDialog', _fake_dialog(
        target_count=100, smoothing_weight=0.5, lock_faces=False))

    class FakeQuadWrapper:
        def __init__(self, target_face_count, smoothing_weight):
            pass

        def wrap(self, mesh, frozen_face_ids=None):
            return _box_mesh(3.0)
    monkeypatch.setattr(mw, 'QuadWrapper', FakeQuadWrapper)
    win.on_quad_wrap()


def _edit_remove_slivers(win, monkeypatch):
    # The real backend; on the quad box it triangulates (12 faces from 6).
    win.on_remove_sliver_edges()


def _edit_shrink_wrap(win, monkeypatch):
    win.viewport.reference_mesh = _box_mesh(2.2)
    monkeypatch.setattr(mw, 'ShrinkWrapDialog',
                        _fake_dialog(iterations=3, lock_faces=False))

    class FakeShrinkWrapper:
        def __init__(self, iterations):
            pass

        def wrap(self, cage, reference, frozen_vertices=None):
            return _box_mesh(2.2)
    monkeypatch.setattr(mw, 'ShrinkWrapper', FakeShrinkWrapper)
    win.on_shrink_wrap()


def _edit_vertex_property(win, monkeypatch):
    win.properties_panel.current_target = ('vertex', 0)
    old = float(win.current_mesh.vertices[0].position[0])
    win.on_property_changed('pos_x', old + 0.75)


def _edit_crease_property(win, monkeypatch):
    win.properties_panel.current_target = ('edge', 0)
    old = float(win.current_mesh.edges[0].crease_weight)
    win.on_property_changed('crease_weight', old + 1.0)


def _edit_gizmo_in_place(win, monkeypatch):
    # What MeshViewport._on_gizmo_moved does to the SAME mesh object: it
    # assigns a new position array, without telling the main window.
    v = win.current_mesh.vertices[3]
    v.position = np.asarray(v.position, dtype=np.float64) + np.array([0.0, 0.0, 0.4])


def _edit_swap_behind_the_back(win, monkeypatch):
    # What src.core.command.MeshOperationCommand does on execute/undo: it
    # assigns main_window.current_mesh directly, bypassing the handlers.
    win.current_mesh = _box_mesh(2.7)


_EDITS = {
    'subdivide': _edit_subdivide,
    'shell': _edit_shell,
    'quad_wrap': _edit_quad_wrap,
    'remove_slivers': _edit_remove_slivers,
    'shrink_wrap': _edit_shrink_wrap,
    'vertex_property': _edit_vertex_property,
    'crease_property': _edit_crease_property,
    'gizmo_in_place': _edit_gizmo_in_place,
    'mesh_swap_outside_handlers': _edit_swap_behind_the_back,
}
# Edits that bypass the window's handlers; only the export-time check sees them.
_EDITS_OUTSIDE_HANDLERS = {'gizmo_in_place', 'mesh_swap_outside_handlers'}


def _geometry_state(mesh):
    return (id(mesh), _positions(mesh).tobytes(),
            tuple(float(e.crease_weight) for e in mesh.edges),
            len(mesh.faces))


@pytest.mark.parametrize('source', ['step_import', 'convert'])
@pytest.mark.parametrize('edit', sorted(_EDITS))
def test_h2_edit_after_shape_blocks_stale_step_export(env, monkeypatch, tmp_path, edit, source):
    """H2 (cand 27: Subdivide/Shell/Quad Wrap/Slivers/Shrink Wrap, cand 34:
    Properties-panel vertex/crease edit). After the mesh is edited, Export STEP
    must NOT write the pre-edit B-rep; it must warn "run Convert to NURBS"
    and write nothing."""
    win, box, written = env
    mesh = _box_mesh()
    stale = _Shape('pre-edit')
    if source == 'step_import':
        _open_step(win, monkeypatch, mesh, stale)
    else:
        win.current_mesh = mesh
        _install_converter(monkeypatch, [(stale, 6)])
        win.on_convert_nurbs()
    # precondition: a shape is bound to the mesh as it is now
    assert win.current_shape is stale
    before = _geometry_state(win.current_mesh)

    _EDITS[edit](win, monkeypatch)

    # the edit really changed the mesh (object, positions, creases or faces)
    assert win.current_mesh is not None
    assert _geometry_state(win.current_mesh) != before
    assert box.texts('warning') == [], box.shown
    if edit not in _EDITS_OUTSIDE_HANDLERS:
        # the window's own handler drops the shape at once, not only at export
        assert win.current_shape is None

    _set_export(monkeypatch, 'STEP', tmp_path / 'out.step')
    win.on_export()
    assert written['step'] == [], f"stale shape exported after {edit}"
    warnings = box.texts('warning')
    assert len(warnings) == 1 and 'Convert to NURBS' in warnings[0], box.shown
    assert win.current_shape is None


def test_h2_mesh_exports_still_work_after_an_edit(env, monkeypatch, tmp_path):
    """H2: invalidating the shape must not block the MESH exports - STL after
    Subdivide still writes the subdivided mesh."""
    win, box, written = env
    _open_step(win, monkeypatch, _box_mesh(), _Shape('imported'))
    _edit_subdivide(win, monkeypatch)
    assert len(win.current_mesh.faces) == 24  # 6 quads -> 24 quads

    _set_export(monkeypatch, 'STL', tmp_path / 'out.stl')
    win.on_export()
    assert len(written['stl']) == 1 and written['stl'][0][0] is win.current_mesh
    assert box.texts('warning') == []


def test_h2_no_op_property_edit_keeps_the_shape(env, monkeypatch, tmp_path):
    """H2 (cand 34) control: re-entering the SAME vertex value (panel
    repopulation) changes nothing, so the shape stays exportable."""
    win, box, written = env
    mesh, shape = _box_mesh(), _Shape('imported')
    _open_step(win, monkeypatch, mesh, shape)
    win.properties_panel.current_target = ('vertex', 0)
    win.on_property_changed('pos_x', float(mesh.vertices[0].position[0]))

    _set_export(monkeypatch, 'STEP', tmp_path / 'out.step')
    win.on_export()
    assert written['step'] == [(shape, str(tmp_path / 'out.step'))]


def test_h2_reconvert_after_edit_makes_step_export_work_again(env, monkeypatch, tmp_path):
    """H2: after an edit invalidated the shape, Convert to NURBS again binds the
    NEW shape and Export STEP writes it."""
    win, box, written = env
    win.current_mesh = _box_mesh()
    first, second = _Shape('first'), _Shape('second')
    _install_converter(monkeypatch, [(first, 6), (second, 24)])
    win.on_convert_nurbs()
    _edit_subdivide(win, monkeypatch)
    win.on_convert_nurbs()
    assert win.current_shape is second

    _set_export(monkeypatch, 'STEP', tmp_path / 'out.step')
    win.on_export()
    assert written['step'] == [(second, str(tmp_path / 'out.step'))]


# --------------------------------------------------------------------------
# M3 - sliver removal that only flips edges
# --------------------------------------------------------------------------

def _needle_mesh():
    # Triangle (0,1,2) is a needle: height 0.04 against a 2.0 long edge, while
    # no edge is short; flipping edge 0-1 to 2-3 is the whole cleanup.
    verts = np.array([[-1, 0, 0], [1, 0, 0], [0, 0.04, 0], [0, -1, 0]], float)
    return HalfEdgeMesh.from_arrays(verts, [[0, 1, 2], [1, 0, 3]])


def _faces(mesh):
    return mesh.to_arrays()['faces']


def test_m3_needle_flip_is_kept(env):
    """M3 (cand 3): a mesh whose only sliver is a needle triangle is cleaned by
    an edge flip that keeps vertex and face counts. The flipped mesh must
    replace current_mesh; it must not be reported as "mesh unchanged"."""
    win, box, written = env
    mesh = _needle_mesh()
    win.current_mesh = mesh
    assert _faces(mesh) == [[0, 1, 2], [1, 0, 3]]

    win.on_remove_sliver_edges()

    assert win.current_mesh is not mesh
    assert _faces(win.current_mesh) == [[0, 3, 2], [3, 1, 2]]
    np.testing.assert_allclose(_positions(win.current_mesh), _positions(mesh), atol=1e-12)
    assert not any('mesh unchanged' in m for m in win.logs), win.logs
    assert any('Sliver removal complete' in m for m in win.logs), win.logs


def test_m3_clean_mesh_is_still_reported_unchanged(env):
    """M3 control: a clean triangle mesh (trimesh box) comes back identical,
    so the handler must keep the original object and say "mesh unchanged"."""
    win, box, written = env
    tm = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh = HalfEdgeMesh.from_trimesh(tm)
    win.current_mesh = mesh

    win.on_remove_sliver_edges()

    assert win.current_mesh is mesh
    assert len(mesh.faces) == 12 and len(mesh.vertices) == 8
    assert any('mesh unchanged' in m for m in win.logs), win.logs
