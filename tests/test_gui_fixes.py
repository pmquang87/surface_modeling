"""Regression tests for the verified GUI bugs in src/gui/*.

Windowless, but NOT ``QT_QPA_PLATFORM=offscreen``:

* ``QT_QPA_PLATFORM=offscreen`` gives the QWidget no native window handle, so
  VTK's vtkWin32OpenGLRenderWindow aborts the process with "failed to get valid
  pixel format" as soon as MeshViewport renders.
* ``PYVISTA_OFF_SCREEN=true`` avoids that crash but makes pyvistaqt set
  ``QtInteractor.iren = None`` (see pyvistaqt/plotting.py `_setup_interactor`),
  and the whole viewport — picking, interactor style, modifiers — needs that
  interactor.

So the tests run under the normal Qt platform and simply never call show():
no window ever appears. If the platform is externally forced to offscreen the
GUI tests skip with an explanation instead of aborting the interpreter.

Interaction that genuinely needs a live render window (dragging a rubber band,
holding Shift while clicking) cannot be simulated here, so those paths are
exercised programmatically through the same methods the interactor calls.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ.setdefault('QT_API', 'pyside6')

_FORCED_OFFSCREEN = os.environ.get('QT_QPA_PLATFORM') == 'offscreen'
if _FORCED_OFFSCREEN:
    # Keep VTK away from the Win32 GL path so the process survives; the GUI
    # tests below will skip.
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'

import numpy as np
import pytest
import pyvista as pv
import vtk

if _FORCED_OFFSCREEN:
    pv.OFF_SCREEN = True

# Repeatedly creating/destroying VTK render windows spams wglMakeCurrent errors
# that drown the pytest report; they are noise, not failures.
vtk.vtkObject.GlobalWarningDisplayOff()

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.core.halfedge_mesh import HalfEdgeMesh
from src.subd import primitives
from src.gui.main_window import PowerSurfacingMainWindow
from src.gui import viewport as viewport_module
from src.gui.viewport import MeshViewport, SolidWorksStyle
from src.gui.panels import SelectionPanel, face_normal
from src.gui.dialogs import PrimitiveDialog, ConvertNURBSDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture(scope="session")
def _main_window(qapp):
    """One MainWindow for the whole session.

    Each MainWindow owns a VTK render window / GL context; building dozens of
    them exhausts the driver on Windows, so the window is reused and reset.
    """
    # Re-checked here (not only at import) because other test modules flip these
    # globals at collection time.
    if os.environ.get('QT_QPA_PLATFORM') == 'offscreen' or pv.OFF_SCREEN:
        pytest.skip(
            "No interactive VTK render window: QT_QPA_PLATFORM=offscreen or "
            "pyvista.OFF_SCREEN is set, and pyvistaqt then leaves iren=None. "
            "Run this file on its own."
        )
    try:
        win = PowerSurfacingMainWindow()
    except RuntimeError as exc:  # raised by MeshViewport when iren is None
        pytest.skip(f"No interactive VTK render window available: {exc}")
    yield win
    # Restore the streams LogStream hijacked, otherwise pytest's capture breaks.
    sys.stdout = getattr(win._stdout_stream, 'original', sys.__stdout__)
    sys.stderr = getattr(win._stderr_stream, 'original', sys.__stderr__)
    win.close()
    win.deleteLater()


@pytest.fixture
def window(_main_window):
    win = _main_window
    win.viewport.set_selection_method('pick')
    win.viewport.clear()
    win.viewport.set_selection_mode('none')
    win.viewport.set_selection_modifier('new')
    win.viewport.box_select_through = False
    win.viewport.set_gizmo_enabled(False)
    win.viewport.snap_to_reference = False
    win.viewport.proximity_query = None
    win.current_mesh = None
    win.current_shape = None
    win.properties_panel.clear()
    win.log_panel.clear()
    iren = win.viewport.plotter.iren.interactor
    iren.SetShiftKey(0)
    iren.SetControlKey(0)
    return win


def _box(size=2.0):
    return primitives.create_box(width=size, height=size, depth=size)


def _normal_labels(panel):
    """Read the "Normal X/Y/Z" values back out of the properties form."""
    from PySide6.QtWidgets import QFormLayout, QLabel

    layout = panel.form_layout
    values = []
    for row in range(layout.rowCount()):
        label_item = layout.itemAt(row, QFormLayout.LabelRole)
        field_item = layout.itemAt(row, QFormLayout.FieldRole)
        if label_item is None or field_item is None:
            continue
        label, field = label_item.widget(), field_item.widget()
        if (isinstance(label, QLabel) and isinstance(field, QLabel)
                and label.text().startswith("Normal")):
            values.append(float(field.text()))
    return np.array(values, dtype=float)


def _block_colours(panel, needle):
    """Foreground colours actually stored on the log line containing `needle`.

    ``toPlainText()`` throws every char format away, so it can never see a red
    colour bleeding into the following line — the document has to be walked.
    """
    doc = panel.document()
    for i in range(doc.blockCount()):
        block = doc.findBlockByNumber(i)
        if needle not in block.text():
            continue
        colours = set()
        it = block.begin()
        while not it.atEnd():
            fragment = it.fragment()
            if fragment.isValid():
                colours.add(fragment.charFormat().foreground().color().name())
            it += 1
        return colours
    return None


def _spy_cell_picking(vp, monkeypatch):
    """Record the ``through`` flag on every PyVista picker (re-)arm."""
    calls = []
    real = vp.plotter.enable_cell_picking

    def spy(**kwargs):
        calls.append(kwargs.get('through'))
        return real(**kwargs)

    monkeypatch.setattr(vp.plotter, 'enable_cell_picking', spy)
    return calls


def _count_to_pyvista(mesh):
    """Patch mesh.to_pyvista so the test can see every topology rebuild."""
    calls = []
    original = mesh.to_pyvista

    def counting():
        calls.append(1)
        return original()

    mesh.to_pyvista = counting
    return calls


class _SphereSnap:
    """Nearest point on the unit sphere; records every batch it is asked about.

    Deliberately non-linear: a plane projection cannot tell the accumulating and
    the non-accumulating gizmo implementations apart, a curved surface can.
    """

    def __init__(self):
        self.queries = []

    def on_surface(self, points):
        pts = np.asarray(points, dtype=np.float64)
        self.queries.append(pts.copy())
        lengths = np.linalg.norm(pts, axis=1)
        safe = np.where(lengths == 0.0, 1.0, lengths)
        closest = pts / safe[:, None]
        return closest, np.abs(lengths - 1.0), np.zeros(len(pts), dtype=np.int64)


# ---------------------------------------------------------------------------
# 1. Create > Primitive
# ---------------------------------------------------------------------------

def test_primitive_factories_cover_every_dialog_entry(window):
    dlg = PrimitiveDialog(window)
    names = [dlg.type_combo.itemText(i).lower() for i in range(dlg.type_combo.count())]
    factories = window._primitive_factories()
    assert names, "dialog offers no primitive types"
    for name in names:
        assert name in factories, f"no factory mapped for primitive '{name}'"


def test_every_primitive_builds_at_the_requested_size(window):
    size = 10.0
    for name, factory in window._primitive_factories().items():
        mesh = factory(size)
        assert isinstance(mesh, HalfEdgeMesh)
        assert len(mesh.vertices) > 0 and len(mesh.faces) > 0, f"{name} came out empty"
        pts = np.array([v.position for v in mesh.vertices])
        extent = float((pts.max(axis=0) - pts.min(axis=0)).max())
        # Largest dimension should track the requested size, not the defaults.
        assert 0.5 * size <= extent <= 1.5 * size, f"{name} extent {extent} vs size {size}"


def test_on_create_primitive_runs_without_size_kwarg(window, monkeypatch):
    """The old code called create_fn(size=...) which no factory accepts."""
    monkeypatch.setattr(PrimitiveDialog, 'exec_', lambda self: 1, raising=False)

    errors = []
    monkeypatch.setattr('src.gui.main_window.QMessageBox.warning',
                        lambda *a, **k: errors.append(a[-1]))
    window.on_create_primitive()
    assert not errors, f"primitive creation reported: {errors}"
    assert window.current_mesh is not None
    assert len(window.current_mesh.faces) == 6  # default dialog type is Box


# ---------------------------------------------------------------------------
# 2 + 3. Shrink wrap
# ---------------------------------------------------------------------------

def test_collect_frozen_vertices_uses_half_edge_attribute(window):
    window.current_mesh = _box()
    verts = window._collect_frozen_vertices([0, 1])
    assert len(verts) == 8, verts  # two opposite quads of a cube
    assert all(0 <= v < len(window.current_mesh.vertices) for v in verts)
    # Out-of-range face ids must be ignored, not raise.
    assert window._collect_frozen_vertices([9999]) == []


def test_face_has_no_halfedge_attribute():
    """Guards the exact typo that made 'Lock Selected Faces' abort."""
    mesh = _box()
    assert hasattr(mesh.faces[0], 'half_edge')
    assert not hasattr(mesh.faces[0], 'halfedge')


def test_shrink_wrap_without_reference_reports_instead_of_self_wrapping(window, monkeypatch):
    window.current_mesh = _box()
    window.viewport.reference_mesh = None

    shown = []
    monkeypatch.setattr('src.gui.main_window.QMessageBox.information',
                        lambda *a, **k: shown.append(a[-1]))
    # If the guard fails, the dialog would open — make that an obvious failure.
    monkeypatch.setattr('src.gui.dialogs.ShrinkWrapDialog.exec_',
                        lambda self: pytest.fail("dialog opened without a reference mesh"),
                        raising=False)

    window.on_shrink_wrap()
    assert shown, "no message shown when shrink wrap has no reference mesh"
    assert 'reference' in shown[0].lower()


def test_shrink_wrap_targets_the_reference_mesh(window, monkeypatch):
    cage = _box(2.0)
    reference = _box(4.0)
    window.current_mesh = cage
    window.viewport.reference_mesh = reference

    monkeypatch.setattr('src.gui.dialogs.ShrinkWrapDialog.exec_', lambda self: 1, raising=False)

    seen = {}

    class FakeShrinkWrapper:
        def __init__(self, iterations=5, **kwargs):
            seen['iterations'] = iterations

        def wrap(self, cage_mesh, reference_mesh, frozen_vertices=None):
            seen['cage'] = cage_mesh
            seen['reference'] = reference_mesh
            seen['frozen'] = frozen_vertices
            return cage_mesh

    monkeypatch.setattr('src.gui.main_window.ShrinkWrapper', FakeShrinkWrapper)
    window.on_shrink_wrap()

    assert seen['cage'] is cage
    assert seen['reference'] is reference, "shrink wrap still wraps the mesh onto itself"


# ---------------------------------------------------------------------------
# 4. Stale selection across a mesh swap
# ---------------------------------------------------------------------------

def test_mesh_swap_clears_stale_selection(window):
    vp = window.viewport
    big = primitives.create_sphere(radius=1.0, segments=12, rings=10)
    vp.set_mesh(big)
    vp.set_selection_mode('face')
    high = len(big.faces) - 1
    # Index 0 is valid in BOTH meshes: selecting only an out-of-range index
    # would let plain clamping satisfy the assertion below even if the
    # identity-change reset were gone.
    vp.highlight_selection([0, high], 'face')
    assert sorted(vp.get_selected_faces()) == [0, high]

    small = _box()  # 6 faces — index `high` no longer exists, index 0 does
    vp.update_mesh(small)  # must not raise IndexError
    assert vp.get_selected_faces() == [], "index 0 survived the mesh swap"


def test_highlight_selection_clamps_out_of_range_indices(window):
    vp = window.viewport
    mesh = _box()
    vp.set_mesh(mesh)
    vp.set_selection_mode('vertex')
    vp.highlight_selection([0, 3, 999, -5], 'vertex')  # must not raise
    assert sorted(vp._selected_indices) == [0, 3]


def test_set_mesh_twice_with_shrinking_meshes(window):
    vp = window.viewport
    vp.set_selection_mode('vertex')
    big = primitives.create_sphere(radius=1.0, segments=16, rings=12)
    vp.set_mesh(big)
    vp.highlight_selection(list(range(len(big.vertices))), 'vertex')
    vp.set_mesh(_box())
    assert vp._selected_indices == []


# ---------------------------------------------------------------------------
# 5. Box (area) selection
# ---------------------------------------------------------------------------

def test_extract_original_cell_ids_reads_pyvista_048_key():
    grid = pv.Cube().cast_to_unstructured_grid()
    grid.cell_data['original_cell_ids'] = np.arange(grid.n_cells)
    ids = MeshViewport._extract_original_cell_ids(grid)
    assert ids == list(range(grid.n_cells))


def test_extract_original_cell_ids_handles_multiblock_and_legacy_keys():
    a = pv.Cube().cast_to_unstructured_grid()
    a.cell_data['original_cell_ids'] = np.arange(a.n_cells)
    b = pv.Cube().cast_to_unstructured_grid()
    # Distinct id ranges: with both blocks numbered 0..n the values cannot tell
    # a correct read from counting the first block twice, or from block-local
    # indices being reported instead of global ones.
    b.cell_data['vtkOriginalCellIds'] = np.arange(b.n_cells) + 100
    ids = MeshViewport._extract_original_cell_ids(pv.MultiBlock([a, b]))
    assert len(ids) == a.n_cells + b.n_cells
    assert ids == list(range(a.n_cells)) + list(range(100, 100 + b.n_cells))
    assert MeshViewport._extract_original_cell_ids(None) == []


def test_box_pick_callback_selects_faces(window):
    vp = window.viewport
    mesh = _box()
    vp.set_mesh(mesh)
    vp.set_selection_mode('face')

    picked = mesh.to_pyvista().cast_to_unstructured_grid()
    picked.cell_data['original_cell_ids'] = np.arange(picked.n_cells)
    vp._on_box_picked(picked)

    assert sorted(vp.get_selected_faces()) == list(range(len(mesh.faces)))


def test_box_pick_callback_selects_vertices_of_picked_faces(window):
    vp = window.viewport
    mesh = _box()
    vp.set_mesh(mesh)
    vp.set_selection_mode('vertex')

    grid = mesh.to_pyvista().cast_to_unstructured_grid()
    grid.cell_data['original_cell_ids'] = np.arange(grid.n_cells)
    # Two opposite faces of the cube: a callback that collapses every picked id
    # onto one cell still returns "the right number of vertices" for a single
    # quad, so it has to be asked for a disjoint pair.
    picked = grid.extract_cells([0, 1])  # what the frustum extraction hands back

    expected = sorted({v.index
                       for fid in (0, 1)
                       for v in mesh.get_face_vertices(mesh.faces[fid])})
    assert len(expected) == 8, expected  # precondition: the faces are disjoint

    vp._on_box_picked(picked)

    assert sorted(vp._selected_indices) == expected


# ---------------------------------------------------------------------------
# 6 + 7. Interactor style survival and Select Through toggling
# ---------------------------------------------------------------------------

def test_custom_style_is_active_after_construction(window):
    vp = window.viewport
    assert vp.plotter.iren.interactor.GetInteractorStyle() is vp._custom_style


def test_custom_style_survives_enabling_box_picking(window):
    vp = window.viewport
    vp.set_selection_method('box')
    assert vp._box_picking_enabled
    assert vp.plotter.iren.interactor.GetInteractorStyle() is vp._custom_style, \
        "PyVista's rubber band style replaced the SolidWorks navigation style"
    assert vp._custom_style.rubber_band_enabled


def test_style_supports_rubber_band_and_trackball_navigation():
    # vtkInteractorStyleRubberBandPick already derives from
    # vtkInteractorStyleTrackballCamera, so asserting both base classes is one
    # check, not two. What the viewport actually relies on is the gate that
    # keeps the left button free for single-element picking.
    assert issubclass(SolidWorksStyle, pv._vtk.vtkInteractorStyleRubberBandPick)

    style = SolidWorksStyle(plotter=None)
    assert style.rubber_band_enabled is False   # left button free for picking
    style.set_rubber_band_enabled(True)
    assert style.rubber_band_enabled is True
    style.set_rubber_band_enabled(0)            # truthiness, not identity
    assert style.rubber_band_enabled is False


def test_toggling_select_through_does_not_raise(window, monkeypatch):
    vp = window.viewport
    vp.set_selection_method('box')
    # Spy on the plotter, not on the viewport flag: a setter that only records
    # the flag never re-arms the picker, so `through` never reaches VTK and
    # "does not raise" becomes true for the trivial reason that nothing runs.
    calls = _spy_cell_picking(vp, monkeypatch)

    for through in (False, True, False, True):
        vp.set_box_select_through(through)  # used to raise PyVistaPickingError
        assert vp.box_select_through is through
        assert calls and calls[-1] is through, (
            f"the picker was not re-armed with through={through}: {calls}"
        )
        assert vp._box_picking_enabled
        assert vp.plotter.iren.interactor.GetInteractorStyle() is vp._custom_style

    assert len(calls) == 4, f"expected one re-arm per toggle, got {calls}"


def test_double_enable_without_disable_is_the_original_bug(window):
    """Proves the fix is not vacuous: PyVista still rejects a second picker."""
    from pyvista.plotting.errors import PyVistaPickingError

    vp = window.viewport
    vp.set_selection_method('box')
    with pytest.raises(PyVistaPickingError):
        vp.plotter.enable_cell_picking(callback=vp._on_box_picked, show=False,
                                       show_message=False, through=True)
    # The supported path still works right afterwards.
    vp.set_box_select_through(True)
    assert vp._box_picking_enabled


def test_select_through_checkbox_signal_path(window, monkeypatch):
    """The panel wires cb_through.toggled straight into the viewport."""
    vp = window.viewport
    vp.set_selection_method('box')
    window.selection_panel.cb_through.setChecked(False)
    calls = _spy_cell_picking(vp, monkeypatch)

    window.selection_panel.cb_through.setEnabled(True)
    window.selection_panel.cb_through.setChecked(True)   # must not raise
    assert vp.box_select_through is True
    # The observable downstream effect, not the flag the slot mirrors: a
    # checkbox wired to a dead setter must not pass.
    assert calls and calls[-1] is True, f"picker never re-armed: {calls}"

    window.selection_panel.cb_through.setChecked(False)
    assert vp.box_select_through is False
    assert calls and calls[-1] is False, f"picker never re-armed: {calls}"
    assert len(calls) == 2


def test_switching_back_to_pick_disables_box_picking(window):
    vp = window.viewport
    vp.set_selection_method('box')
    vp.set_selection_method('pick')
    assert not vp._box_picking_enabled
    assert not vp._custom_style.rubber_band_enabled
    assert vp.plotter.iren.interactor.GetInteractorStyle() is vp._custom_style
    # And re-arming afterwards must still work.
    vp.set_selection_method('box')
    assert vp._box_picking_enabled


# ---------------------------------------------------------------------------
# 9. Shift/Ctrl selection modifiers
# ---------------------------------------------------------------------------

def test_shift_and_ctrl_modifiers_come_from_the_interactor(window):
    """Shift = add, Ctrl = remove, driven purely by the VTK interactor state."""
    vp = window.viewport
    mesh = _box()
    vp.set_mesh(mesh)
    vp.set_selection_mode('face')
    vp.set_selection_modifier('new')
    iren = vp.plotter.iren.interactor

    iren.SetShiftKey(0)
    iren.SetControlKey(0)
    vp._apply_selection_modifier([0])
    assert vp.get_selected_faces() == [0]

    iren.SetShiftKey(1)
    vp._apply_selection_modifier([1])
    assert sorted(vp.get_selected_faces()) == [0, 1]

    iren.SetShiftKey(0)
    iren.SetControlKey(1)
    vp._apply_selection_modifier([0])
    assert vp.get_selected_faces() == [1]

    iren.SetControlKey(0)


def test_read_modifiers_reads_the_vtk_interactor(window):
    vp = window.viewport
    iren = vp.plotter.iren.interactor
    iren.SetShiftKey(1)
    iren.SetControlKey(0)
    shift, ctrl = vp._read_modifiers()
    assert shift is True and ctrl is False
    iren.SetShiftKey(0)
    iren.SetControlKey(1)
    shift, ctrl = vp._read_modifiers()
    assert shift is False and ctrl is True
    iren.SetControlKey(0)


def test_modifier_state_is_refreshed_on_every_pick(window):
    """Qt key events never reach MeshViewport; the pick path must re-read."""
    vp = window.viewport
    mesh = _box()
    vp.set_mesh(mesh)
    vp.set_selection_mode('face')
    vp.set_selection_modifier('new')
    vp._shift_pressed = True  # stale value nobody ever cleared

    iren = vp.plotter.iren.interactor
    iren.SetShiftKey(0)
    iren.SetControlKey(0)
    vp._apply_selection_modifier([0])
    vp._apply_selection_modifier([1])
    assert vp.get_selected_faces() == [1], "stale Shift state leaked into the pick"


# ---------------------------------------------------------------------------
# 8. Continuity dropdown
# ---------------------------------------------------------------------------

def test_continuity_is_passed_as_the_string_the_backend_expects(window, monkeypatch):
    window.current_mesh = _box()
    monkeypatch.setattr('src.gui.dialogs.ConvertNURBSDialog.exec_', lambda self: 1, raising=False)

    captured = {}

    class FakeConverter:
        def __init__(self, continuity='G2', tolerance=1e-4):
            captured['continuity'] = continuity
            captured['tolerance'] = tolerance

        def convert(self, mesh, subdivision_levels=3, simplify=False, reference_mesh=None):
            captured['reference_mesh'] = reference_mesh
            return {'shape': None, 'mesh': None, 'patches': []}

    monkeypatch.setattr('src.gui.main_window.SubDToNURBSConverter', FakeConverter)
    monkeypatch.setattr('src.gui.main_window.QMessageBox.warning', lambda *a, **k: None)

    for index, expected in enumerate(['G0', 'G1', 'G2', 'G3']):
        original_init = ConvertNURBSDialog.__init__

        def patched_init(self, parent=None, _i=index, _orig=original_init):
            _orig(self, parent)
            self.continuity.setCurrentIndex(_i)

        monkeypatch.setattr(ConvertNURBSDialog, '__init__', patched_init)
        window.on_convert_nurbs()
        assert captured['continuity'] == expected


class _SpyFitter:
    """Stands in for G3Fitter and records what convert() builds it with."""

    constructed = []      # continuity_weight of every instance
    fitted = []           # quad_mesh_data of every fit_surface call

    def __init__(self, continuity_weight=20.0, **kwargs):
        type(self).constructed.append(continuity_weight)

    def fit_surface(self, quad_mesh_data):
        type(self).fitted.append(quad_mesh_data)
        return []

    @classmethod
    def install(cls, monkeypatch):
        cls.constructed = []
        cls.fitted = []
        # converter.py imports G3Fitter *inside* generate_patches, so the
        # defining module is what has to be patched.
        import src.nurbs.g3_fitter as g3_fitter
        monkeypatch.setattr(g3_fitter, 'G3Fitter', cls)
        return cls


def test_continuity_string_reaches_the_fitter_weight_table(monkeypatch):
    """The converter maps 'G0'..'G3' to distinct fitter weights; ints do not."""
    from src.nurbs.converter import SubDToNURBSConverter

    # Restating the table inside the test proves nothing about src; capture the
    # weight the real converter constructs the fitter with instead.
    spy = _SpyFitter.install(monkeypatch)
    for key in ('G0', 'G1', 'G2', 'G3'):
        SubDToNURBSConverter(continuity=key).convert(_box())

    seen = list(spy.constructed)
    assert len(seen) == 4, seen
    assert len(set(seen)) == 4, f"continuity levels collapsed to {set(seen)}"
    assert seen == sorted(seen), f"stricter continuity must weigh more: {seen}"

    # An int (the old GUI value) is not a key -> it silently falls back to G2.
    spy.constructed.clear()
    SubDToNURBSConverter(continuity=2).convert(_box())
    assert spy.constructed == [seen[2]], spy.constructed


def test_convert_nurbs_forwards_the_loaded_reference_mesh(window, monkeypatch):
    window.current_mesh = _box(2.0)
    reference = _box(4.0)
    window.viewport.reference_mesh = reference

    monkeypatch.setattr('src.gui.dialogs.ConvertNURBSDialog.exec_', lambda self: 1, raising=False)
    captured = {}

    class FakeConverter:
        def __init__(self, continuity='G2', tolerance=1e-4):
            pass

        def convert(self, mesh, subdivision_levels=3, simplify=False, reference_mesh=None):
            captured['reference_mesh'] = reference_mesh
            return {'shape': None, 'mesh': None, 'patches': []}

    monkeypatch.setattr('src.gui.main_window.SubDToNURBSConverter', FakeConverter)
    monkeypatch.setattr('src.gui.main_window.QMessageBox.warning', lambda *a, **k: None)
    window.on_convert_nurbs()
    assert captured['reference_mesh'] is reference


def test_converter_convert_accepts_the_reference_mesh_kwarg(monkeypatch):
    import inspect
    from src.nurbs.converter import SubDToNURBSConverter
    sig = inspect.signature(SubDToNURBSConverter.convert)
    assert 'reference_mesh' in sig.parameters

    # Declaring the parameter is not consuming it. Capture the sample points the
    # converter hands the fitter and check they were projected onto the
    # reference surface instead of the cage's own limit approximation.
    spy = _SpyFitter.install(monkeypatch)

    SubDToNURBSConverter().convert(_box(2.0))
    without = np.array([q['dense_points'] for q in spy.fitted[-1]])

    SubDToNURBSConverter().convert(_box(2.0), reference_mesh=_box(4.0))
    with_ref = np.array([q['dense_points'] for q in spy.fitted[-1]])

    assert with_ref.shape == without.shape
    assert not np.allclose(with_ref, without), "reference_mesh was accepted and ignored"
    # The reference box has half-extent 2.0 and every sample lands on its skin.
    assert np.abs(with_ref).max() == pytest.approx(2.0, abs=1e-6)
    assert np.abs(without).max() < 1.5


def test_set_reference_mesh_stores_the_mesh(window):
    reference = _box(3.0)          # half-extent 1.5
    window.viewport.set_reference_mesh(reference)
    assert window.viewport.reference_mesh is reference
    assert window.viewport.proximity_query is not None

    # `is not None` cannot tell a query built from THIS mesh from one built off
    # unrelated geometry, which would silently break Snap-to-Reference and
    # Shrink Wrap. Ask it something only this box can answer.
    closest, dist, _ = window.viewport.proximity_query.on_surface(
        np.array([[10.0, 0.0, 0.0]])
    )
    assert np.allclose(closest[0], [1.5, 0.0, 0.0], atol=1e-6), closest
    assert dist[0] == pytest.approx(8.5, abs=1e-6)

    window.viewport.set_reference_mesh(None)
    assert window.viewport.reference_mesh is None
    assert window.viewport.proximity_query is None


# ---------------------------------------------------------------------------
# 10. Log HTML escaping
# ---------------------------------------------------------------------------

def test_log_message_with_markup_is_not_swallowed(window):
    payload = "value <b>bold</b> & 3 < 5 in <module>"
    window._append_log(payload)
    text = window.log_panel.toPlainText()
    assert "<b>bold</b>" in text
    assert "3 < 5" in text
    assert "&amp;" not in text


def test_error_log_escapes_markup_and_keeps_styling(window):
    window._append_log("Traceback: expected <class 'int'> & got <str>", error=True)
    text = window.log_panel.toPlainText()
    assert "<class 'int'>" in text
    assert "<str>" in text
    html_text = window.log_panel.toHtml()
    assert "#cc0000" in html_text


def test_error_colour_does_not_bleed_into_the_next_line(window):
    window._append_log("boom", error=True)
    window._append_log("all good")
    assert "boom" in window.log_panel.toPlainText()
    assert "all good" in window.log_panel.toPlainText()

    # The point of the test: the colour, not the text. Both lines exist under a
    # bleeding implementation too.
    error_colours = _block_colours(window.log_panel, "boom")
    normal_colours = _block_colours(window.log_panel, "all good")
    assert error_colours == {"#cc0000"}, error_colours
    assert normal_colours is not None
    assert "#cc0000" not in normal_colours, (
        f"the error colour bled into the following line: {normal_colours}"
    )


# ---------------------------------------------------------------------------
# 11. Property panel write-back
# ---------------------------------------------------------------------------

def test_vertex_position_edit_is_written_to_the_mesh(window):
    mesh = _box(2.0)
    window.current_mesh = mesh
    window.viewport.set_mesh(mesh)
    window.viewport.set_selection_mode('vertex')

    window.properties_panel.set_vertex_properties(3, mesh)
    assert window.properties_panel.current_target == ('vertex', 3)

    window.on_property_changed('pos_y', 7.25)
    assert mesh.vertices[3].position[1] == pytest.approx(7.25)
    # Other axes and other vertices untouched.
    assert mesh.vertices[3].position[0] == pytest.approx(-1.0)
    assert mesh.vertices[0].position[1] == pytest.approx(-1.0)


def test_vertex_spinbox_signal_reaches_the_mesh(window):
    mesh = _box(2.0)
    window.current_mesh = mesh
    window.viewport.set_mesh(mesh)
    window.properties_panel.set_vertex_properties(2, mesh)

    from PySide6.QtWidgets import QDoubleSpinBox
    layout = window.properties_panel.form_layout
    spins = []
    for i in range(layout.count()):
        widget = layout.itemAt(i).widget()
        if isinstance(widget, QDoubleSpinBox):
            spins.append(widget)
    assert len(spins) == 3, "expected X/Y/Z position editors"
    spins[1].setValue(4.5)  # Position Y
    assert mesh.vertices[2].position[1] == pytest.approx(4.5)


def test_edge_crease_edit_is_written_to_the_mesh(window):
    mesh = _box(2.0)
    window.current_mesh = mesh
    window.viewport.set_mesh(mesh)
    window.properties_panel.set_edge_properties(0, mesh)
    assert window.properties_panel.current_target == ('edge', 0)

    window.on_property_changed('crease_weight', 0.75)
    assert mesh.edges[0].crease_weight == pytest.approx(0.75)


def test_populating_the_panel_does_not_emit_edits(window):
    mesh = _box(2.0)
    window.current_mesh = mesh
    window.viewport.set_mesh(mesh)
    mesh.vertices[5].position[0] = 1.234567

    emitted = []
    window.properties_panel.property_changed.connect(lambda *a: emitted.append(a))
    window.properties_panel.set_vertex_properties(5, mesh)

    # Precondition: the editors were really built and show the live value. A
    # panel that populates nothing emits nothing either, and would otherwise
    # satisfy the silence assertion below for the wrong reason.
    from PySide6.QtWidgets import QDoubleSpinBox
    layout = window.properties_panel.form_layout
    spins = [layout.itemAt(i).widget() for i in range(layout.count())
             if isinstance(layout.itemAt(i).widget(), QDoubleSpinBox)]
    assert len(spins) == 3, "panel did not populate the X/Y/Z editors"
    assert spins[0].value() == pytest.approx(1.234567)
    assert window.properties_panel.current_target == ('vertex', 5)

    assert emitted == []
    assert mesh.vertices[5].position[0] == pytest.approx(1.234567)


def test_property_change_without_a_target_is_a_no_op(window):
    mesh = _box(2.0)
    window.current_mesh = mesh
    window.properties_panel.clear()
    window.on_property_changed('pos_x', 99.0)  # must not raise
    assert mesh.vertices[0].position[0] == pytest.approx(-1.0)


def test_stale_vertex_target_after_mesh_swap_is_ignored(window):
    big = primitives.create_sphere(radius=1.0, segments=12, rings=10)
    window.current_mesh = big
    stale_index = len(big.vertices) - 1
    window.properties_panel.set_vertex_properties(stale_index, big)
    # Precondition: the defect really is injected — the panel points at an index
    # the next mesh does not have.
    assert window.properties_panel.current_target == ('vertex', stale_index)

    window.current_mesh = _box()
    assert stale_index >= len(window.current_mesh.vertices)

    before = np.array([v.position.copy() for v in window.current_mesh.vertices])
    window.on_property_changed('pos_x', 5.0)  # index no longer exists
    after = np.array([v.position for v in window.current_mesh.vertices])

    # Any implementation that clamps or wraps the stale index writes 5.0 onto a
    # live vertex; a bound check leaves the mesh untouched.
    assert np.array_equal(before, after), "stale index was clamped onto a live vertex"

    # Positive control: "ignored" must mean the guard fired, not that the whole
    # write-back path is dead.
    window.properties_panel.set_vertex_properties(2, window.current_mesh)
    window.on_property_changed('pos_x', 5.0)
    assert window.current_mesh.vertices[2].position[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Backend alignment checks (already fixed elsewhere — verified, not re-fixed)
# ---------------------------------------------------------------------------

def test_subdivide_accepts_the_smooth_kwarg():
    import inspect
    from src.subd import catmull_clark
    sig = inspect.signature(catmull_clark.subdivide)
    assert 'smooth' in sig.parameters

    cage = np.array([v.position for v in _box().vertices])
    smooth = catmull_clark.subdivide(_box(), 1, smooth=True)
    linear = catmull_clark.subdivide(_box(), 1, smooth=False)

    # One Catmull-Clark level turns 6 quads into 24 either way, so the face
    # count cannot tell a consumed `smooth` from an ignored one - the vertex
    # positions can.
    assert len(smooth.faces) == len(linear.faces) == 24
    ps = np.array([v.position for v in smooth.vertices])
    pl = np.array([v.position for v in linear.vertices])
    assert ps.shape == pl.shape
    assert not np.allclose(ps, pl), "smooth= had no effect on the geometry"

    # Linear subdivision leaves the cage corners exactly where they were;
    # smoothing pulls them in towards the centroid.
    assert np.allclose(pl[:len(cage)], cage)
    assert np.abs(ps[:len(cage)]).max() < 0.9
    assert np.linalg.norm(ps, axis=1).max() < np.linalg.norm(pl, axis=1).max()


def test_export_failure_surfaces_as_a_message_box(window, monkeypatch):
    """Exporters raise now; the GUI must turn that into a critical dialog."""
    window.current_mesh = _box()
    monkeypatch.setattr('src.gui.dialogs.ExportDialog.exec_', lambda self: 1, raising=False)
    monkeypatch.setattr('src.gui.main_window.QFileDialog.getSaveFileName',
                        lambda *a, **k: ('C:/nonexistent-dir-xyz/out.stl', ''))

    def boom(*a, **k):
        raise IOError("disk on fire")

    monkeypatch.setattr('src.gui.main_window.export_stl', boom)

    from src.gui.dialogs import ExportDialog
    original_init = ExportDialog.__init__

    def patched_init(self, parent=None, _orig=original_init):
        _orig(self, parent)
        self.format_combo.setCurrentText("STL")

    monkeypatch.setattr(ExportDialog, '__init__', patched_init)

    reported = []
    monkeypatch.setattr('src.gui.main_window.QMessageBox.critical',
                        lambda *a, **k: reported.append(a[-1]))
    window.on_export()
    assert reported and "disk on fire" in reported[0]


# ---------------------------------------------------------------------------
# 12. Selection is live on startup (viewport default vs. panel default)
# ---------------------------------------------------------------------------

def test_selection_panel_reports_its_initial_state(qapp):
    panel = SelectionPanel()
    try:
        assert panel.current_selection_mode() == 'face'
        assert panel.current_selection_method() == 'pick'
        assert panel.current_selection_modifier() == 'new'
    finally:
        panel.deleteLater()


def test_viewport_default_mode_matches_the_panel_default(qapp):
    panel = SelectionPanel()
    try:
        assert viewport_module.DEFAULT_SELECTION_MODE == panel.current_selection_mode()
    finally:
        panel.deleteLater()
    # The viewport must not fall back to the old dead-on-startup 'none'.
    assert viewport_module.DEFAULT_SELECTION_MODE != 'none'


def test_sync_selection_panel_to_viewport_restores_the_shown_state(window):
    vp = window.viewport
    vp.set_selection_mode('none')
    assert vp.selection_mode == 'none'

    window._sync_selection_panel_to_viewport()

    panel = window.selection_panel
    assert vp.selection_mode == panel.current_selection_mode() == 'face'
    assert vp.selection_method == panel.current_selection_method() == 'pick'
    assert vp.selection_modifier == panel.current_selection_modifier() == 'new'


def test_a_mode_of_none_really_does_swallow_picks(window):
    """Proves the startup fix is not vacuous: 'none' drops every picked id."""
    vp = window.viewport
    mesh = _box()
    vp.set_mesh(mesh)
    vp.set_selection_mode('none')
    vp._apply_selection_modifier([0])
    assert vp._selected_indices == []


def test_first_pick_on_a_fresh_window_selects_something(qapp):
    if os.environ.get('QT_QPA_PLATFORM') == 'offscreen' or pv.OFF_SCREEN:
        pytest.skip("no interactive VTK render window available")

    win = PowerSurfacingMainWindow()
    try:
        vp = win.viewport
        # Nobody has touched the panel yet — it shows "Face", so must the viewport.
        assert vp.selection_mode == win.selection_panel.current_selection_mode() == 'face'

        mesh = _box()
        win.current_mesh = mesh
        vp.set_mesh(mesh)
        vp._apply_selection_modifier([0])  # exactly what a left-click delivers
        assert vp.get_selected_faces() == [0], "the very first click was swallowed"
    finally:
        sys.stdout = getattr(win._stdout_stream, 'original', sys.__stdout__)
        sys.stderr = getattr(win._stderr_stream, 'original', sys.__stderr__)
        win.close()
        win.deleteLater()


# ---------------------------------------------------------------------------
# 13. Face normals in the properties panel
# ---------------------------------------------------------------------------

def test_face_normals_start_out_zero():
    """Precondition of the bug: no import/display path ever fills them in."""
    mesh = _box()
    assert all(np.allclose(f.normal, 0.0) for f in mesh.faces)


def test_face_properties_show_a_real_normal(window):
    mesh = _box()
    window.current_mesh = mesh
    window.properties_panel.set_face_properties(0, mesh)

    shown = _normal_labels(window.properties_panel)
    assert len(shown) == 3, "expected Normal X/Y/Z rows"
    assert not np.allclose(shown, 0.0), "panel still reports (0.000, 0.000, 0.000)"
    assert np.linalg.norm(shown) == pytest.approx(1.0, abs=2e-3)


def test_panel_normal_matches_the_core_computation(window):
    mesh = _box()
    reference = _box()
    reference.compute_face_normals()

    for face_index in range(len(mesh.faces)):
        window.properties_panel.set_face_properties(face_index, mesh)
        shown = _normal_labels(window.properties_panel)
        expected = np.asarray(reference.faces[face_index].normal, dtype=float)
        assert np.allclose(shown, expected, atol=2e-3), (
            f"face {face_index}: panel {shown} vs core {expected}"
        )


def test_face_normal_helper_caches_onto_the_face():
    mesh = _box()
    face = mesh.faces[1]
    assert np.allclose(face.normal, 0.0)
    n = face_normal(mesh, face)
    assert np.linalg.norm(n) == pytest.approx(1.0)
    assert np.allclose(face.normal, n)


def test_face_normal_is_recomputed_after_the_geometry_moves(window):
    mesh = _box()
    mesh.compute_face_normals()
    # Pick a face whose normal points along +/-Y so the rotation below must change it.
    face_index = next(i for i, f in enumerate(mesh.faces) if abs(f.normal[1]) > 0.5)
    stale = np.asarray(mesh.faces[face_index].normal, dtype=float).copy()

    # Rotate every vertex 90 deg about X: (x, y, z) -> (x, -z, y)
    for v in mesh.vertices:
        x, y, z = (float(c) for c in v.position)
        v.position[:] = (x, -z, y)
    expected = np.array([stale[0], -stale[2], stale[1]])

    window.properties_panel.set_face_properties(face_index, mesh)
    shown = _normal_labels(window.properties_panel)
    assert not np.allclose(shown, stale, atol=1e-2), "panel served the stale cached normal"
    assert np.allclose(shown, expected, atol=2e-3)


def test_selection_signal_populates_a_real_face_normal(window):
    mesh = _box()
    window.current_mesh = mesh
    window.viewport.set_mesh(mesh)
    window.viewport.set_selection_mode('face')

    window.viewport._apply_selection_modifier([2])  # emits selection_changed

    shown = _normal_labels(window.properties_panel)
    assert len(shown) == 3
    assert not np.allclose(shown, 0.0)


# ---------------------------------------------------------------------------
# 14. Gizmo drag + "Snap to Reference" offset accumulation
# ---------------------------------------------------------------------------

def _arm_gizmo(vp, mesh, vertex_ids):
    vp.set_mesh(mesh)
    vp.set_selection_mode('vertex')
    vp.highlight_selection(list(vertex_ids), 'vertex')
    vp.set_gizmo_enabled(True)
    assert vp._last_gizmo_pos is not None
    return np.asarray(vp._last_gizmo_pos, dtype=float).copy()


def test_gizmo_drag_without_snap_applies_the_raw_delta(window):
    vp = window.viewport
    mesh = _box()
    start = _arm_gizmo(vp, mesh, [0])
    vp.snap_to_reference = False
    vp.proximity_query = None

    origin = np.asarray(mesh.vertices[0].position, dtype=float).copy()
    for n in range(1, 4):
        vp._on_gizmo_moved(start + np.array([0.3 * n, 0.0, 0.0]))

    assert np.allclose(mesh.vertices[0].position, origin + [0.9, 0.0, 0.0])
    assert np.allclose(vp._last_gizmo_pos, start + [0.9, 0.0, 0.0])


def test_gizmo_snap_does_not_accumulate_an_offset(window):
    vp = window.viewport
    mesh = _box()
    start = _arm_gizmo(vp, mesh, [0, 1])

    snap = _SphereSnap()
    vp.proximity_query = snap
    vp.snap_to_reference = True

    step = np.array([0.0, 0.0, 0.25])
    for n in range(1, 7):
        v_indices = vp._get_selected_vertex_indices()
        before = np.array([mesh.vertices[i].position for i in v_indices], dtype=float)

        widget = start + n * step
        vp._on_gizmo_moved(widget)

        # The sphere widget stays where the user dragged it, so that raw
        # position — not a snap-corrected one — must anchor the next delta.
        assert np.allclose(vp._last_gizmo_pos, widget), (
            f"step {n}: gizmo reference drifted to {vp._last_gizmo_pos}"
        )
        # Each step must offer the snapper exactly "old positions + one raw step".
        assert np.allclose(snap.queries[-1], before + step, atol=1e-9), (
            f"step {n}: the drag delta carried a snap correction"
        )


def test_snap_keeps_the_dragged_vertices_on_the_reference_surface(window):
    vp = window.viewport
    mesh = _box()
    start = _arm_gizmo(vp, mesh, [0, 1])
    vp.proximity_query = _SphereSnap()
    vp.snap_to_reference = True

    for n in range(1, 5):
        vp._on_gizmo_moved(start + np.array([0.0, 0.0, 0.3 * n]))

    for i in vp._get_selected_vertex_indices():
        assert np.linalg.norm(mesh.vertices[i].position) == pytest.approx(1.0, abs=1e-9)


def test_zero_length_gizmo_move_is_a_no_op(window):
    vp = window.viewport
    mesh = _box()
    start = _arm_gizmo(vp, mesh, [0])
    before = np.asarray(mesh.vertices[0].position, dtype=float).copy()
    vp._on_gizmo_moved(start)
    assert np.allclose(mesh.vertices[0].position, before)
    assert np.allclose(vp._last_gizmo_pos, start)


# ---------------------------------------------------------------------------
# 15. Cached PolyData (no full rebuild per click / drag step)
# ---------------------------------------------------------------------------

def test_set_mesh_builds_the_cache_once(window):
    vp = window.viewport
    mesh = _box()
    before = vp._pv_rebuild_count
    vp.set_mesh(mesh)
    assert vp._pv_rebuild_count == before + 1
    assert vp._pv_cache is not None
    assert vp._pv_cache.n_points == len(mesh.vertices)
    assert vp._pv_cache.n_cells == len(mesh.faces)


def test_selection_clicks_do_not_rebuild_the_polydata(window):
    vp = window.viewport
    mesh = _box()
    vp.set_mesh(mesh)
    vp.set_selection_mode('face')

    calls = _count_to_pyvista(mesh)
    rebuilds = vp._pv_rebuild_count
    for face_index in range(len(mesh.faces)):
        vp._apply_selection_modifier([face_index])

    assert calls == [], "to_pyvista ran again on the click path"
    assert vp._pv_rebuild_count == rebuilds


def test_vertex_and_edge_highlighting_reuse_the_cache(window):
    vp = window.viewport
    mesh = _box()
    vp.set_mesh(mesh)
    calls = _count_to_pyvista(mesh)

    vp.set_selection_mode('vertex')
    vp.highlight_selection([0, 1, 2], 'vertex')
    vp.set_selection_mode('edge')
    vp.highlight_selection([0, 1], 'edge')
    vp.set_selection_mode('face')
    vp.highlight_selection([0, 3], 'face')

    assert calls == []


def test_gizmo_drag_does_not_rebuild_the_polydata(window):
    vp = window.viewport
    mesh = _box()
    start = _arm_gizmo(vp, mesh, [0, 1])
    vp.snap_to_reference = False
    vp.proximity_query = None

    calls = _count_to_pyvista(mesh)
    rebuilds = vp._pv_rebuild_count
    for n in range(1, 21):
        vp._on_gizmo_moved(start + np.array([0.05 * n, 0.0, 0.0]))

    assert calls == [], "every drag step rebuilt the PolyData from the half-edge mesh"
    assert vp._pv_rebuild_count == rebuilds


def test_gizmo_drag_actually_moves_the_displayed_points(window):
    vp = window.viewport
    mesh = _box()
    start = _arm_gizmo(vp, mesh, [0])
    vp.snap_to_reference = False
    vp.proximity_query = None

    origin = np.asarray(vp.mesh_actor.mapper.dataset.points[0], dtype=float).copy()
    vp._on_gizmo_moved(start + np.array([1.0, 0.0, 0.0]))

    moved = np.asarray(vp.mesh_actor.mapper.dataset.points[0], dtype=float)
    assert np.allclose(moved, origin + [1.0, 0.0, 0.0])
    assert np.allclose(vp._pv_cache.points[0], moved)


def test_refresh_geometry_updates_points_without_a_rebuild(window):
    vp = window.viewport
    mesh = _box()
    vp.set_mesh(mesh)
    vp.set_selection_mode('vertex')

    mesh.vertices[0].position[0] = 5.0
    calls = _count_to_pyvista(mesh)
    rebuilds = vp._pv_rebuild_count
    vp.refresh_geometry()

    assert calls == []
    assert vp._pv_rebuild_count == rebuilds
    assert vp.mesh_actor.mapper.dataset.points[0][0] == pytest.approx(5.0)
    assert vp._pv_cache.points[0][0] == pytest.approx(5.0)


def test_highlight_never_serves_stale_coordinates(window):
    vp = window.viewport
    mesh = _box()
    vp.set_mesh(mesh)
    vp.set_selection_mode('vertex')

    mesh.vertices[2].position[:] = (7.0, 8.0, 9.0)
    vp.highlight_selection([2], 'vertex')

    assert vp.selection_actors, "no highlight actor was created"
    shown = np.asarray(vp.selection_actors[0].mapper.dataset.points[0], dtype=float)
    assert np.allclose(shown, [7.0, 8.0, 9.0])


def test_topology_change_forces_a_rebuild(window):
    vp = window.viewport
    vp.set_mesh(_box())
    rebuilds = vp._pv_rebuild_count

    sphere = primitives.create_sphere(radius=1.0, segments=10, rings=8)
    vp.update_mesh(sphere)

    assert vp._pv_rebuild_count > rebuilds
    assert vp._pv_cache.n_points == len(sphere.vertices)


def test_changing_the_vertex_count_behind_the_viewport_invalidates_the_cache(window):
    vp = window.viewport
    mesh = _box()
    vp.set_mesh(mesh)
    vp.set_selection_mode('vertex')
    rebuilds = vp._pv_rebuild_count

    mesh.add_vertex([9.0, 9.0, 9.0])  # topology changed without update_mesh()
    vp.highlight_selection([0], 'vertex')

    assert vp._pv_rebuild_count == rebuilds + 1
    assert vp._pv_cache.n_points == len(mesh.vertices)


def test_clear_drops_the_cache(window):
    vp = window.viewport
    vp.set_mesh(_box())
    assert vp._pv_cache is not None
    vp.clear()
    assert vp._pv_cache is None
    assert vp._pv_cache_mesh is None


def test_cache_is_not_shared_between_meshes(window):
    vp = window.viewport
    a = _box(2.0)
    b = _box(4.0)  # same topology, different coordinates
    vp.set_mesh(a)
    vp.set_mesh(b)
    assert vp._pv_cache_mesh is b
    assert float(np.abs(vp._pv_cache.points).max()) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Smoke test of the application entry point
# ---------------------------------------------------------------------------

def test_main_module_constructs_the_window(qapp):
    import importlib
    main_mod = importlib.import_module('src.main')
    assert hasattr(main_mod, 'main')

    if os.environ.get('QT_QPA_PLATFORM') == 'offscreen' or pv.OFF_SCREEN:
        pytest.skip("no interactive VTK render window available")

    # Construct only — never show(), never app.exec().
    win = main_mod.PowerSurfacingMainWindow()
    try:
        qapp.processEvents()
        assert win.windowTitle() == "Python Surfacing"
        assert win.viewport is not None
    finally:
        sys.stdout = getattr(win._stdout_stream, 'original', sys.__stdout__)
        sys.stderr = getattr(win._stderr_stream, 'original', sys.__stderr__)
        win.close()
        win.deleteLater()
