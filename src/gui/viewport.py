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
        
        # PyVista interactor
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)
        
        # Set background
        self.plotter.set_background(color="#2b2b2b", top="#1e1e1e")
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

        # Setup picking — deferred to set_selection_mode()
        # Don't enable picking at startup to avoid conflicts

    def _on_cell_picked(self, cell):
        if not self.current_mesh: return
        # A cell in pv.PolyData corresponds to a face
        # We need the index of the face
        pass

    def _on_point_picked(self, point):
        pass

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
            color="#a0a0a0", 
            show_edges=show_edges,
            edge_color="#404040",
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
        
        # Always disable existing picking first
        try:
            self.plotter.disable_picking()
        except Exception:
            pass
        
        if mode == 'face':
            self.plotter.enable_cell_picking(callback=self._on_cell_picked, show_message=False)
        elif mode == 'vertex':
            self.plotter.enable_point_picking(callback=self._on_point_picked, show_message=False)
        # mode == 'none' or 'edge': picking stays disabled

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
            # Extract faces
            pass
            
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
