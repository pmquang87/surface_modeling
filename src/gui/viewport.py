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

        # Setup left-click picking using custom logic
        self.plotter.track_click_position(self._on_click, side='left')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Shift:
            self._shift_pressed = True
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Shift:
            self._shift_pressed = False
        super().keyReleaseEvent(event)

    def _on_click(self, pos):
        if not self.current_mesh or self.selection_mode == 'none': 
            return
            
        try:
            p3d = self.plotter.pick_mouse_position()
            if p3d is None: 
                return
        except Exception:
            return
            
        pv_mesh = self.current_mesh.to_pyvista()
        
        if self.selection_mode == 'face':
            face_id = pv_mesh.find_closest_cell(p3d)
            if face_id < 0 or face_id >= len(self.current_mesh.faces): return
            if self._shift_pressed:
                if face_id not in self._selected_indices:
                    self._selected_indices.append(face_id)
            else:
                self._selected_indices = [face_id]
            self.face_selected.emit(face_id)
            self.selection_changed.emit(self._selected_indices)
            self.highlight_selection(self._selected_indices, 'face')
            
        elif self.selection_mode == 'vertex':
            vert_id = pv_mesh.find_closest_point(p3d)
            if vert_id < 0 or vert_id >= len(self.current_mesh.vertices): return
            if self._shift_pressed:
                if vert_id not in self._selected_indices:
                    self._selected_indices.append(vert_id)
            else:
                self._selected_indices = [vert_id]
            self.vertex_selected.emit(vert_id)
            self.selection_changed.emit(self._selected_indices)
            self.highlight_selection(self._selected_indices, 'vertex')
            
        elif self.selection_mode == 'edge':
            face_id = pv_mesh.find_closest_cell(p3d)
            if face_id >= 0 and face_id < len(self.current_mesh.faces):
                face = self.current_mesh.faces[face_id]
                edges = self.current_mesh.get_face_edges(face)
                
                closest_edge = None
                min_dist = float('inf')
                
                for e in edges:
                    v1 = e.half_edge.vertex.position
                    v2 = e.half_edge.prev.vertex.position
                    # mid point approximation for edge picking
                    mid = (v1 + v2) / 2.0
                    dist = np.linalg.norm(np.array(p3d) - mid)
                    if dist < min_dist:
                        min_dist = dist
                        closest_edge = e.index
                        
                if closest_edge is not None:
                    if self._shift_pressed:
                        if closest_edge not in self._selected_indices:
                            self._selected_indices.append(closest_edge)
                    else:
                        self._selected_indices = [closest_edge]
                    self.edge_selected.emit(closest_edge)
                    self.selection_changed.emit(self._selected_indices)
                    self.highlight_selection(self._selected_indices, 'edge')

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
            pickable=True
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
            actor = self.plotter.add_mesh(pc, color='red', point_size=10, render_points_as_spheres=True, pickable=False)
            self.selection_actors.append(actor)
            
        elif element_type == 'face':
            extracted = pv_mesh.extract_cells(indices)
            actor = self.plotter.add_mesh(extracted, color='red', show_edges=True, edge_color='black', pickable=False)
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
                actor = self.plotter.add_mesh(pd, color='red', line_width=5, pickable=False)
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
