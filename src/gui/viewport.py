import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal, Qt

try:
    from src.core.halfedge_mesh import HalfEdgeMesh
except ImportError:
    HalfEdgeMesh = None

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
        self.layout.addWidget(self.plotter.interactor)
        
        # Set background
        self.plotter.set_background(color="#f0f0f0", top="#e0e5ea")
        self.plotter.add_axes()
        # Adding a simple grid
        self.plotter.show_grid()
        
        self.mesh_actor = None
        self.ref_actor = None
        self.selection_actors = []
        
        self.current_mesh = None
        self.display_mode = 'solid+wireframe'
        self.selection_mode = 'none'

        self._selected_indices = []

        self.selection_method = 'pick'
        self.selection_modifier = 'new'
        self.box_select_through = False
        
        self._ctrl_pressed = False
        
        # Setup left-click picking using custom logic
        self.plotter.track_click_position(self._on_click, side='left')

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
        
    def set_selection_method(self, method: str):
        self.selection_method = method
        if method == 'box':
            # PyVista doesn't natively map left drag to box selection easily without overriding the entire interactor style.
            # We will use enable_cell_picking if they press 'r' for now, but log it.
            pass

    def set_selection_modifier(self, mod: str):
        self.selection_modifier = mod

    def set_box_select_through(self, through: bool):
        self.box_select_through = through
        
    def _apply_selection_modifier(self, new_ids: list):
        current = set(self._selected_indices)
        incoming = set(new_ids)
        
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
        picker = pv.vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        
        # Only pick from our mesh_actor
        picker.AddPickList(self.mesh_actor)
        picker.PickFromListOn()
        
        picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
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
                edges = self.current_mesh.get_face_edges(face)
                min_dist = float('inf')
                closest_vert = None
                
                for e in edges:
                    v = e.half_edge.vertex
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

    def get_selected_faces(self) -> list:
        if self.selection_mode == 'face':
            return list(self._selected_indices)
        return []

    def set_mesh(self, mesh, name: str = 'default'):
        self.update_mesh(mesh, name)
        self.reset_camera()

    def update_mesh(self, mesh, name: str = 'default'):
        self.current_mesh = mesh
        if not mesh:
            self.clear()
            return
            
        pv_mesh = mesh.to_pyvista()
        
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

    def clear(self):
        self.plotter.clear_actors()
        self.mesh_actor = None
        self.ref_actor = None
        self.selection_actors = []
        self.current_mesh = None
        self.plotter.update()

    def set_display_mode(self, mode: str):
        self.display_mode = mode
        if self.current_mesh:
            self.update_mesh(self.current_mesh)

    def set_selection_mode(self, mode: str):
        self.selection_mode = mode
        self._selected_indices = []
        self.highlight_selection([], mode)

    def highlight_selection(self, indices: list, element_type: str):
        self._selected_indices = indices
        # clear previous highlights
        for actor in self.selection_actors:
            self.plotter.remove_actor(actor)
        self.selection_actors.clear()

        if not self.current_mesh or not indices:
            self.plotter.update()
            return
            
        pv_mesh = self.current_mesh.to_pyvista()
        
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
                
        self.plotter.update()

    def set_reference_mesh(self, mesh):
        if self.ref_actor:
            self.plotter.remove_actor(self.ref_actor)
            
        if not mesh:
            self.plotter.update()
            return
            
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

    def screenshot(self, filepath: str):
        self.plotter.screenshot(filepath)
