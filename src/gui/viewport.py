import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal, Qt

try:
    from src.core.halfedge_mesh import HalfEdgeMesh
except ImportError:
    HalfEdgeMesh = None

try:
    import vtk
except ImportError:
    vtk = pv._vtk

#: Selection mode a fresh viewport starts in. Kept in sync with the entity
#: radio button SelectionPanel checks by default, so the very first click in a
#: freshly opened window already selects something.
DEFAULT_SELECTION_MODE = 'face'


class SolidWorksStyle(vtk.vtkInteractorStyleRubberBandPick):
    """SolidWorks-like navigation (middle mouse = rotate/pan/dolly).

    Derives from vtkInteractorStyleRubberBandPick (itself a trackball camera
    style) so PyVista's rectangle cell picking keeps working while this style is
    installed. Rubber-band dragging is gated on ``rubber_band_enabled`` which the
    viewport toggles together with box selection, so the left mouse button stays
    free for single-element picking in normal mode.
    """

    def __init__(self, plotter=None):
        super().__init__()
        self.plotter = plotter
        self.rubber_band_enabled = False
        self.AddObserver("MiddleButtonPressEvent", self.on_middle_down)
        self.AddObserver("MiddleButtonReleaseEvent", self.on_middle_up)
        self.AddObserver("LeftButtonPressEvent", self.on_left_down)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_up)

    def set_rubber_band_enabled(self, enabled: bool):
        self.rubber_band_enabled = bool(enabled)

    def on_middle_down(self, obj, event):
        iren = self.GetInteractor()
        if iren.GetShiftKey():
            self.StartDolly()
        elif iren.GetControlKey():
            self.StartPan()
        else:
            # Alt + MMB to choose new center of rotation
            if iren.GetAltKey() and self.plotter is not None:
                clickPos = iren.GetEventPosition()
                picker = vtk.vtkPropPicker()
                picker.Pick(clickPos[0], clickPos[1], 0, self.plotter.renderer)
                if picker.GetActor() is not None:
                    p3d = picker.GetPickPosition()
                    self.plotter.camera.focal_point = p3d
                    self.plotter.render()
            self.StartRotate()
            
    def on_middle_up(self, obj, event):
        state = self.GetState()
        if state == 1: # VTKIS_ROTATE
            self.EndRotate()
        elif state == 2: # VTKIS_PAN
            self.EndPan()
        elif state == 4: # VTKIS_DOLLY
            self.EndDolly()
            
    def on_left_down(self, obj, event):
        # Left button drags the rubber band while box selection is armed,
        # otherwise it is reserved for single-element picking (no rotate).
        if self.rubber_band_enabled:
            self.StartSelect()  # force VTKISRBP_SELECT, independent of the 'r' key
            self.OnLeftButtonDown()

    def on_left_up(self, obj, event):
        if self.rubber_band_enabled:
            self.OnLeftButtonUp()

class MeshViewport(QWidget):
    """3D viewport for interactive mesh visualization and editing."""
    
    vertex_selected = Signal(int)
    edge_selected = Signal(int)  
    face_selected = Signal(int)
    selection_changed = Signal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.setFocusPolicy(Qt.StrongFocus)
        self._shift_pressed = False
        
        # PyVista interactor
        self.plotter = QtInteractor(self)
        if getattr(self.plotter, 'iren', None) is None:
            # pyvistaqt sets iren=None when pyvista.OFF_SCREEN / PYVISTA_OFF_SCREEN
            # is on. Picking, the interactor style and the modifier keys all need
            # it, so fail loudly here instead of deep inside track_click_position.
            raise RuntimeError(
                "MeshViewport requires an interactive VTK render window, but "
                "pyvistaqt returned no interactor. Unset PYVISTA_OFF_SCREEN / "
                "pyvista.OFF_SCREEN before creating the GUI."
            )
        self.layout.addWidget(self.plotter.interactor)

        # Apply SolidWorks interactor style
        self._custom_style = SolidWorksStyle(self.plotter)
        self._apply_custom_interactor_style()

        # Set background
        self.plotter.set_background(color="#f0f0f0", top="#e0e5ea")
        self.plotter.add_axes()
        # Adding a simple grid
        self.plotter.show_grid()
        
        self.mesh_actor = None
        self.ref_actor = None
        self.selection_actors = []

        # Transform Gizmo state
        self._last_gizmo_pos = None
        self.gizmo_enabled = False

        # Snap to Reference state
        self.snap_to_reference = False
        self.proximity_query = None
        self.reference_mesh = None

        self.current_mesh = None
        self.display_mode = 'solid+wireframe'
        # Must agree with SelectionPanel's initially checked entity radio button
        # (Face). Starting at 'none' made every first click a no-op until the
        # user toggled the mode, even though the panel already showed "Face".
        self.selection_mode = DEFAULT_SELECTION_MODE

        self._selected_indices = []

        # Cached PolyData for current_mesh. Rebuilding it from the half-edge
        # mesh is a pure-Python loop over every face, which used to run on every
        # click and every gizmo drag step. The cache is rebuilt only when the
        # topology changes; position-only changes are pushed into the existing
        # point array.
        self._pv_cache = None
        self._pv_cache_mesh = None
        self._pv_cache_sig = None
        self._pv_rebuild_count = 0

        self.selection_method = 'pick'
        self.selection_modifier = 'new'
        self.box_select_through = False
        
        self._ctrl_pressed = False

        # Box picking is only installed while the Box (Area) tool is active,
        # because PyVista's rectangle picking takes over the interactor style.
        self._box_picking_enabled = False

        # Setup left-click picking using custom logic
        self.plotter.track_click_position(self._on_click, side='left')

    def _apply_custom_interactor_style(self):
        """(Re-)install the SolidWorks navigation style on the VTK interactor.

        PyVista's picking helpers call ``enable_rubber_band_style()`` internally,
        which silently replaces the interactor style, so this has to run again
        after every enable/disable of picking.
        """
        try:
            self.plotter.iren.interactor.SetInteractorStyle(self._custom_style)
        except Exception as e:
            print(f"[WARNING] Could not apply custom interactor style: {e}")

    def _enable_box_picking(self):
        """Arm PyVista rectangle picking and keep the custom navigation style."""
        self._disable_box_picking(restore_style=False)
        try:
            self.plotter.enable_cell_picking(
                callback=self._on_box_picked,
                show=False,
                show_message=False,
                through=self.box_select_through,
            )
        except Exception as e:
            print(f"[WARNING] Could not enable box selection: {e}")
            return
        self._box_picking_enabled = True
        # enable_cell_picking swapped in the rubber band style -> put ours back.
        self._apply_custom_interactor_style()
        self._custom_style.set_rubber_band_enabled(True)

    def _disable_box_picking(self, restore_style: bool = True):
        self._custom_style.set_rubber_band_enabled(False)
        if self._box_picking_enabled:
            try:
                self.plotter.disable_picking()
            except Exception as e:
                print(f"[WARNING] Could not disable box selection: {e}")
        self._box_picking_enabled = False
        if restore_style:
            self._apply_custom_interactor_style()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Shift:
            self._shift_pressed = True
        elif event.key() == Qt.Key_Control:
            self._ctrl_pressed = True
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Shift:
            self._shift_pressed = False
        elif event.key() == Qt.Key_Control:
            self._ctrl_pressed = False
        super().keyReleaseEvent(event)

    def _read_modifiers(self):
        """Refresh Shift/Ctrl state at the moment a pick happens.

        Qt key events rarely reach this widget (the VTK render window owns the
        keyboard), so the interactor's latched modifier state is the primary
        source, with Qt's global keyboard state as a fallback.
        """
        shift = False
        ctrl = False
        try:
            iren = self.plotter.iren.interactor
            shift = bool(iren.GetShiftKey())
            ctrl = bool(iren.GetControlKey())
        except Exception:
            pass

        if not shift and not ctrl:
            try:
                from PySide6.QtWidgets import QApplication
                mods = QApplication.keyboardModifiers()
                shift = bool(mods & Qt.ShiftModifier)
                ctrl = bool(mods & Qt.ControlModifier)
            except Exception:
                pass

        self._shift_pressed = shift
        self._ctrl_pressed = ctrl
        return shift, ctrl

    def _element_count(self, element_type: str) -> int:
        if not self.current_mesh:
            return 0
        if element_type == 'vertex':
            return len(self.current_mesh.vertices)
        if element_type == 'edge':
            return len(self.current_mesh.edges)
        if element_type == 'face':
            return len(self.current_mesh.faces)
        return 0

    def _sanitize_indices(self, indices, element_type: str) -> list:
        """Drop indices that no longer address an element of the current mesh."""
        if not indices:
            return []
        n = self._element_count(element_type)
        if n == 0:
            return []
        clean = []
        for i in indices:
            try:
                i = int(i)
            except (TypeError, ValueError):
                continue
            if 0 <= i < n:
                clean.append(i)
        return clean

    def set_selection_method(self, method: str):
        self.selection_method = method.lower()
        if self.selection_method == 'box':
            self._enable_box_picking()
            print("Box selection active: drag a rectangle with the left mouse button.")
        else:
            self._disable_box_picking()

    def set_selection_modifier(self, mod: str):
        self.selection_modifier = mod

    def set_box_select_through(self, through: bool):
        self.box_select_through = bool(through)
        # PyVista refuses to enable a second picker, so only re-arm when the
        # box tool is actually active (and always disable first).
        if self._box_picking_enabled:
            self._enable_box_picking()

    def _apply_selection_modifier(self, new_ids: list):
        self._read_modifiers()

        current = set(self._sanitize_indices(self._selected_indices, self.selection_mode))
        incoming = set(self._sanitize_indices(new_ids, self.selection_mode))

        mod = self.selection_modifier
        if self._shift_pressed and mod == 'new':
            mod = 'add'
        elif self._ctrl_pressed and mod == 'new':
            mod = 'remove'

        if mod == 'new':
            current = incoming
        elif mod == 'add':
            current.update(incoming)
        elif mod == 'remove':
            current.difference_update(incoming)
            
        self._selected_indices = list(current)
        self.selection_changed.emit(self._selected_indices)
        self.highlight_selection(self._selected_indices, self.selection_mode)
        
    def run_selection_operation(self, op: str):
        if not self.current_mesh: return
        if op == 'clear':
            self._selected_indices = []
            self.selection_changed.emit(self._selected_indices)
            self.highlight_selection(self._selected_indices, self.selection_mode)
            return
            
        if not self._selected_indices and op != 'invert': return
        
        new_sel = set(self._selected_indices)
        
        if op == 'adjacent':
            if self.selection_mode == 'face':
                new_sel.update(self.current_mesh.get_adjacent_faces(self._selected_indices))
            elif self.selection_mode == 'vertex':
                new_sel.update(self.current_mesh.get_adjacent_vertices(self._selected_indices))
            elif self.selection_mode == 'edge':
                new_sel.update(self.current_mesh.get_adjacent_edges(self._selected_indices))
                
        elif op == 'connected':
            if self.selection_mode == 'face':
                new_sel.update(self.current_mesh.get_connected_faces(self._selected_indices))
            elif self.selection_mode == 'vertex':
                new_sel.update(self.current_mesh.get_connected_vertices(self._selected_indices))
            elif self.selection_mode == 'edge':
                new_sel.update(self.current_mesh.get_connected_edges(self._selected_indices))
                
        elif op == 'invert':
            if self.selection_mode == 'none':
                return
            if self.selection_mode == 'face':
                all_ids = set(range(len(self.current_mesh.faces)))
            elif self.selection_mode == 'vertex':
                all_ids = set(range(len(self.current_mesh.vertices)))
            elif self.selection_mode == 'edge':
                all_ids = set(range(len(self.current_mesh.edges)))
            new_sel = all_ids - new_sel
            
        self._selected_indices = list(new_sel)
        self.selection_changed.emit(self._selected_indices)
        self.highlight_selection(self._selected_indices, self.selection_mode)

    def _on_click(self, pos):
        if not self.current_mesh or self.selection_mode == 'none': 
            return
            
        # Use robust VTK hardware picking to get the exact cell at the mouse pixel
        import pyvista as pv
        try:
            import vtk
        except ImportError:
            vtk = pv._vtk
            
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        
        # Only pick from our mesh_actor
        if self.mesh_actor:
            picker.AddPickList(self.mesh_actor)
            picker.PickFromListOn()
            
        # Use PyVista's cached mouse position which is already converted to VTK coordinates (bottom-left origin)
        mouse_pos = self.plotter.mouse_position
        if not mouse_pos:
            return
            
        picker.Pick(mouse_pos[0], mouse_pos[1], 0, self.plotter.renderer)
        cell_id = picker.GetCellId()
        
        if cell_id < 0:
            return
            
        p3d = picker.GetPickPosition()
        
        if self.selection_mode == 'face':
            if cell_id < len(self.current_mesh.faces):
                self._apply_selection_modifier([cell_id])
                self.face_selected.emit(cell_id)
                
        elif self.selection_mode == 'vertex':
            # find closest point in the exact picked face to the pick position
            if cell_id < len(self.current_mesh.faces):
                face = self.current_mesh.faces[cell_id]
                vertices = self.current_mesh.get_face_vertices(face)
                min_dist = float('inf')
                closest_vert = None
                
                for v in vertices:
                    dist = np.linalg.norm(v.position - np.array(p3d))
                    if dist < min_dist:
                        min_dist = dist
                        closest_vert = v.index
                        
                if closest_vert is not None:
                    self._apply_selection_modifier([closest_vert])
                    self.vertex_selected.emit(closest_vert)
                    
        elif self.selection_mode == 'edge':
            # find closest edge in the exact picked face to the pick position
            if cell_id < len(self.current_mesh.faces):
                face = self.current_mesh.faces[cell_id]
                edges = self.current_mesh.get_face_edges(face)
                closest_edge = None
                min_dist = float('inf')
                
                for e in edges:
                    v1 = e.half_edge.vertex.position
                    v2 = e.half_edge.prev.vertex.position
                    # simplified distance to segment (using midpoint)
                    mid = (v1 + v2) / 2.0
                    dist = np.linalg.norm(np.array(p3d) - mid)
                    if dist < min_dist:
                        min_dist = dist
                        closest_edge = e.index
                        
                if closest_edge is not None:
                    self._apply_selection_modifier([closest_edge])
                    self.edge_selected.emit(closest_edge)

    @staticmethod
    def _extract_original_cell_ids(picked) -> list:
        """Pull source cell ids out of whatever PyVista's rectangle picker returns.

        PyVista >= 0.44 tags the extracted cells with ``original_cell_ids`` and
        may hand back a MultiBlock when several actors were hit. The older
        ``orig_extract_id`` / ``vtkOriginalCellIds`` keys are still accepted.
        """
        if picked is None:
            return []

        blocks = []
        if isinstance(picked, pv.MultiBlock):
            for block in picked:
                if block is not None:
                    blocks.append(block)
        else:
            blocks.append(picked)

        ids = []
        for block in blocks:
            cell_data = getattr(block, 'cell_data', None)
            if cell_data is None:
                continue
            for key in ("original_cell_ids", "orig_extract_id", "vtkOriginalCellIds"):
                if key in cell_data:
                    ids.extend(int(i) for i in cell_data[key])
                    break
        return ids

    def _on_box_picked(self, picked_mesh):
        if not self.current_mesh or self.selection_mode == 'none' or picked_mesh is None:
            return

        cell_ids = self._extract_original_cell_ids(picked_mesh)

        if not cell_ids:
            return

        if self.selection_mode == 'face':
            valid_ids = [fid for fid in cell_ids if fid < len(self.current_mesh.faces)]
            if valid_ids:
                self._apply_selection_modifier(valid_ids)
                
        elif self.selection_mode == 'vertex':
            vert_ids = set()
            for fid in cell_ids:
                if fid < len(self.current_mesh.faces):
                    face = self.current_mesh.faces[fid]
                    for v in self.current_mesh.get_face_vertices(face):
                        vert_ids.add(v.index)
            if vert_ids:
                self._apply_selection_modifier(list(vert_ids))
                
        elif self.selection_mode == 'edge':
            edge_ids = set()
            for fid in cell_ids:
                if fid < len(self.current_mesh.faces):
                    face = self.current_mesh.faces[fid]
                    for e in self.current_mesh.get_face_edges(face):
                        edge_ids.add(e.index)
            if edge_ids:
                self._apply_selection_modifier(list(edge_ids))

    def get_selected_faces(self) -> list:
        if self.selection_mode == 'face':
            return list(self._selected_indices)
        return []

    def set_mesh(self, mesh, name: str = 'default'):
        self.update_mesh(mesh, name)
        self.reset_camera()

    # ------------------------------------------------------------------
    # Cached PolyData
    # ------------------------------------------------------------------

    @staticmethod
    def _topology_signature(mesh):
        return (len(mesh.vertices), len(mesh.edges), len(mesh.faces))

    def invalidate_geometry_cache(self):
        """Force the next _get_pv_mesh() to rebuild topology from scratch."""
        self._pv_cache = None
        self._pv_cache_mesh = None
        self._pv_cache_sig = None

    def _mesh_points(self):
        """Current vertex positions as an (n, 3) float array, or None."""
        mesh = self.current_mesh
        if mesh is None:
            return None
        try:
            pts = np.asarray([v.position for v in mesh.vertices], dtype=np.float64)
        except Exception:
            return None
        if pts.ndim != 2 or pts.shape[1] != 3:
            return None
        return pts

    def _sync_cache_points(self, pts=None) -> bool:
        """Push positions into the cached PolyData. False -> needs a rebuild."""
        cache = self._pv_cache
        if cache is None or self._pv_cache_mesh is not self.current_mesh:
            return False
        if pts is None:
            pts = self._mesh_points()
        if pts is None or cache.n_points != pts.shape[0]:
            return False
        cache.points = pts
        return True

    def _sync_actor_points(self, pts) -> bool:
        """Push positions into the displayed dataset. False -> needs a rebuild.

        pyvista's ``add_mesh(smooth_shading=True)`` hands the mapper a derived
        copy, so the actor's dataset is usually NOT the cached PolyData.
        """
        dataset = getattr(getattr(self.mesh_actor, 'mapper', None), 'dataset', None)
        if dataset is None or dataset.n_points != pts.shape[0]:
            return False
        dataset.points = pts
        return True

    def _sync_points(self) -> bool:
        """Copy current vertex positions into the cached *and* displayed arrays.

        Returns False when the arrays no longer line up (topology changed under
        us), which tells the caller to do a full rebuild instead.
        """
        pts = self._mesh_points()
        if pts is None:
            return False
        ok_cache = self._sync_cache_points(pts)
        ok_actor = self._sync_actor_points(pts)
        return ok_cache and ok_actor

    def _get_pv_mesh(self, sync_points: bool = False):
        """Return the PolyData for current_mesh, rebuilding only when needed."""
        mesh = self.current_mesh
        if mesh is None:
            self.invalidate_geometry_cache()
            return None

        sig = self._topology_signature(mesh)
        stale = (self._pv_cache is None
                 or self._pv_cache_mesh is not mesh
                 or self._pv_cache_sig != sig)
        if not stale and sync_points:
            stale = not self._sync_cache_points()
        if stale:
            self._pv_cache = mesh.to_pyvista()
            self._pv_cache_mesh = mesh
            self._pv_cache_sig = sig
            self._pv_rebuild_count += 1
        return self._pv_cache

    def update_mesh(self, mesh, name: str = 'default'):
        previous = self.current_mesh
        self.current_mesh = mesh
        if not mesh:
            self.clear()
            return

        if previous is not mesh:
            # A different mesh object is now displayed — stale indices would
            # index out of range in highlight_selection / extract_cells.
            self._reset_selection_state()
        else:
            # Same object, possibly mutated in place: clamp instead of clearing.
            self._selected_indices = self._sanitize_indices(
                self._selected_indices, self.selection_mode
            )

        # Explicit "the mesh changed" entry point: topology may differ without
        # the element counts differing, so always rebuild here. The click and
        # drag paths never come through update_mesh.
        self.invalidate_geometry_cache()
        pv_mesh = self._get_pv_mesh()

        # Remove old main actor
        if self.mesh_actor:
            self.plotter.remove_actor(self.mesh_actor)
            
        show_edges = ('wireframe' in self.display_mode)
        style = 'wireframe' if self.display_mode == 'wireframe' else 'surface'
        
        self.mesh_actor = self.plotter.add_mesh(
            pv_mesh, 
            name=name,
            color="#6699bb", 
            show_edges=show_edges,
            edge_color="#2a2a2a",
            style=style,
            smooth_shading=True,
            pickable=True,
            reset_camera=False
        )
        self.highlight_selection(self._selected_indices, self.selection_mode)
        self.plotter.update()

    def _reset_selection_state(self):
        """Forget the current selection (used whenever the mesh identity changes)."""
        if self._selected_indices:
            self._selected_indices = []
            self.selection_changed.emit(self._selected_indices)
        else:
            self._selected_indices = []
        self._last_gizmo_pos = None

    def clear(self):
        self.plotter.clear_actors()
        self.plotter.clear_sphere_widgets()
        self.mesh_actor = None
        self.ref_actor = None
        self.selection_actors = []
        self._last_gizmo_pos = None
        self.current_mesh = None
        self.reference_mesh = None
        self.proximity_query = None
        self.invalidate_geometry_cache()
        self._reset_selection_state()
        self.plotter.update()

    def set_display_mode(self, mode: str):
        self.display_mode = mode
        if self.current_mesh:
            self.update_mesh(self.current_mesh)

    def set_selection_mode(self, mode: str):
        self.selection_mode = mode
        self._reset_selection_state()
        self.highlight_selection([], mode)

    def refresh_geometry(self):
        """Push changed vertex positions into the existing actor (no rebuild)."""
        if not self.current_mesh or not self.mesh_actor:
            return
        if not self._sync_points():
            # Topology changed under us — fall back to a full rebuild.
            self.update_mesh(self.current_mesh)
            return
        self.highlight_selection(self._selected_indices, self.selection_mode, update_gizmo=False)

    def highlight_selection(self, indices: list, element_type: str, update_gizmo: bool = True):
        # Guard against indices left over from a previous (larger) mesh.
        indices = self._sanitize_indices(indices, element_type)
        self._selected_indices = indices
        # clear previous highlights
        for actor in self.selection_actors:
            self.plotter.remove_actor(actor)
        self.selection_actors.clear()

        if update_gizmo:
            self.plotter.clear_sphere_widgets()

        if not self.current_mesh or not indices:
            self.plotter.update()
            return

        # Cached: only the point coordinates are refreshed here, the (expensive)
        # face loop runs again only when the topology actually changed.
        pv_mesh = self._get_pv_mesh(sync_points=True)
        if pv_mesh is None:
            self.plotter.update()
            return

        if element_type == 'vertex':
            pts = pv_mesh.points[indices]
            pc = pv.PolyData(pts)
            actor = self.plotter.add_mesh(pc, color='red', point_size=12, render_points_as_spheres=True, pickable=False, reset_camera=False, render_lines_as_tubes=True)
            self.selection_actors.append(actor)
            
        elif element_type == 'face':
            extracted = pv_mesh.extract_cells(indices)
            actor = self.plotter.add_mesh(extracted, color='red', show_edges=True, edge_color='red', line_width=3, pickable=False, reset_camera=False)
            self.selection_actors.append(actor)
            
        elif element_type == 'edge':
            lines = []
            pts = []
            pt_idx = 0
            for e_idx in indices:
                e = self.current_mesh.edges[e_idx]
                v1 = e.half_edge.vertex.position
                v2 = e.half_edge.prev.vertex.position
                pts.extend([v1, v2])
                lines.extend([2, pt_idx, pt_idx+1])
                pt_idx += 2
                
            if pts:
                pd = pv.PolyData(np.array(pts), lines=np.array(lines))
                actor = self.plotter.add_mesh(pd, color='red', line_width=6, render_lines_as_tubes=True, pickable=False, reset_camera=False)
                self.selection_actors.append(actor)
                
        if update_gizmo:
            self._update_gizmo()
            
        self.plotter.update()

    def _get_selected_vertex_indices(self) -> list:
        if not self.current_mesh or not self._selected_indices:
            return []
            
        v_indices = set()
        if self.selection_mode == 'vertex':
            v_indices.update(self._selected_indices)
        elif self.selection_mode == 'edge':
            for e_idx in self._selected_indices:
                if e_idx < len(self.current_mesh.edges):
                    e = self.current_mesh.edges[e_idx]
                    v_indices.add(e.half_edge.vertex.index)
                    v_indices.add(e.half_edge.prev.vertex.index)
        elif self.selection_mode == 'face':
            for f_idx in self._selected_indices:
                if f_idx < len(self.current_mesh.faces):
                    face = self.current_mesh.faces[f_idx]
                    for v in self.current_mesh.get_face_vertices(face):
                        v_indices.add(v.index)
        return list(v_indices)

    def set_gizmo_enabled(self, enabled: bool):
        self.gizmo_enabled = enabled
        if enabled:
            self._update_gizmo()
        else:
            self.plotter.clear_sphere_widgets()
            self.plotter.update()

    def set_snap_to_reference(self, enabled: bool):
        self.snap_to_reference = enabled
        # Update immediately if gizmo is being moved? Just setting flag is enough.
        if enabled and self.proximity_query is None:
            print("[WARNING] Snap enabled but no reference mesh loaded.")

    def _update_gizmo(self):
        if not self.gizmo_enabled:
            return
            
        v_indices = self._get_selected_vertex_indices()
        if not v_indices:
            return
            
        pts = np.array([self.current_mesh.vertices[i].position for i in v_indices])
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        self._last_gizmo_pos = np.array(center)
        
        # Calculate a reasonable radius based on mesh bounds
        pv_mesh = self._get_pv_mesh(sync_points=True)
        if pv_mesh is None:
            return
        bounds = pv_mesh.bounds
        diag = np.linalg.norm([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
        radius = max(diag * 0.05, 0.01)
        
        self.plotter.add_sphere_widget(
            callback=self._on_gizmo_moved,
            center=center,
            radius=radius,
            color='yellow',
            test_callback=False
        )

    def _on_gizmo_moved(self, new_center):
        if self._last_gizmo_pos is None or not self.current_mesh:
            return

        widget_pos = np.asarray(new_center, dtype=np.float64)
        delta = widget_pos - self._last_gizmo_pos

        # The sphere widget stays wherever the user dragged it, so the reference
        # for the next step must be that raw widget position. Storing a
        # *corrected* position here (e.g. the snapped centroid) made the next
        # delta carry the correction again, and the selection crept away from
        # the cursor one snap offset per drag step.
        self._last_gizmo_pos = widget_pos

        if np.allclose(delta, 0):
            return

        v_indices = self._get_selected_vertex_indices()
        if not v_indices:
            return

        for i in v_indices:
            self.current_mesh.vertices[i].position = (
                np.asarray(self.current_mesh.vertices[i].position, dtype=np.float64) + delta
            )

        if self.snap_to_reference and self.proximity_query is not None:
            # Snap vertices onto the reference surface. This only moves the
            # vertices; the gizmo reference above is deliberately left alone.
            pts = np.array([self.current_mesh.vertices[i].position for i in v_indices])
            closest_pts, _, _ = self.proximity_query.on_surface(pts)
            for idx, i in enumerate(v_indices):
                self.current_mesh.vertices[i].position = np.asarray(
                    closest_pts[idx], dtype=np.float64
                )

        # Push the new coordinates into the cached PolyData and the actor,
        # without rebuilding the topology.
        if not self._sync_points():
            self.update_mesh(self.current_mesh)
            return

        # Highlight without re-creating the gizmo
        self.highlight_selection(self._selected_indices, self.selection_mode, update_gizmo=False)

    def set_reference_mesh(self, mesh):
        if self.ref_actor:
            self.plotter.remove_actor(self.ref_actor)
            self.ref_actor = None

        if not mesh:
            self.reference_mesh = None
            self.proximity_query = None
            self.plotter.update()
            return

        import trimesh
        self.reference_mesh = mesh
        self.proximity_query = trimesh.proximity.ProximityQuery(mesh.to_trimesh())

        pv_mesh = mesh.to_pyvista()
        self.ref_actor = self.plotter.add_mesh(
            pv_mesh,
            color="blue",
            style="wireframe",
            opacity=0.3,
            pickable=False
        )
        self.plotter.update()

    def reset_camera(self):
        self.plotter.reset_camera()
        if self.current_mesh and len(self.current_mesh.vertices) > 0:
            import numpy as np
            pts = np.array([v.position for v in self.current_mesh.vertices])
            avg_center = np.mean(pts, axis=0)
            
            # Shift the camera to look at the average center instead of bounding box center
            cam = self.plotter.camera
            delta = avg_center - np.array(cam.focal_point)
            cam.position = np.array(cam.position) + delta
            cam.focal_point = avg_center

    def screenshot(self, filepath: str):
        try:
            self.plotter.screenshot(filepath)
        except Exception as e:
            print(f"Failed to save screenshot to {filepath}: {e}")
