from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, 
                               QLabel, QFormLayout, QSpinBox, QDoubleSpinBox, QGroupBox,
                               QMenu)
from PySide6.QtCore import Signal, Qt

try:
    from src.core.feature_tree import FeatureTree
except ImportError:
    FeatureTree = None

class FeatureTreePanel(QWidget):
    """Left panel showing the feature history tree."""
    
    feature_selected = Signal(int)
    feature_toggled = Signal(int, bool)
    feature_deleted = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabel("Feature History")
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self._show_context_menu)
        
        self.layout.addWidget(self.tree_widget)
        self.current_tree = None

    def set_feature_tree(self, tree):
        self.current_tree = tree
        self.refresh()

    def refresh(self):
        self.tree_widget.clear()
        if not self.current_tree:
            return
            
        # Mocking items since FeatureTree implementation details might vary
        # Ideally, we'd iterate over current_tree.features
        pass

    def _show_context_menu(self, position):
        item = self.tree_widget.itemAt(position)
        if not item: return
        menu = QMenu()
        edit_action = menu.addAction("Edit")
        delete_action = menu.addAction("Delete")
        toggle_action = menu.addAction("Enable/Disable")
        # Handle actions...
        menu.exec_(self.tree_widget.viewport().mapToGlobal(position))

class PropertiesPanel(QWidget):
    """Right panel showing properties of the current selection/operation."""
    
    property_changed = Signal(str, object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.group_box = QGroupBox("Properties")
        self.form_layout = QFormLayout(self.group_box)
        
        self.layout.addWidget(self.group_box)
        self.layout.addStretch()

    def clear(self):
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.group_box.setTitle("Properties")

    def set_mesh_info(self, mesh):
        self.clear()
        if not mesh: return
        self.group_box.setTitle("Mesh Info")
        self.form_layout.addRow("Vertices:", QLabel(str(len(mesh.vertices))))
        self.form_layout.addRow("Edges:", QLabel(str(len(mesh.edges))))
        self.form_layout.addRow("Faces:", QLabel(str(len(mesh.faces))))

    def set_vertex_properties(self, vertex_index: int, mesh):
        self.clear()
        if not mesh or vertex_index >= len(mesh.vertices): return
        
        v = mesh.vertices[vertex_index]
        self.group_box.setTitle(f"Vertex {vertex_index}")
        
        for i, axis in enumerate(['X', 'Y', 'Z']):
            spin = QDoubleSpinBox()
            spin.setRange(-10000, 10000)
            spin.setValue(v.position[i])
            spin.valueChanged.connect(lambda val, a=axis: self.property_changed.emit(f"pos_{a.lower()}", val))
            self.form_layout.addRow(f"Position {axis}:", spin)

    def set_edge_properties(self, edge_index: int, mesh):
        self.clear()
        if not mesh or edge_index >= len(mesh.edges): return
        
        e = mesh.edges[edge_index]
        self.group_box.setTitle(f"Edge {edge_index}")
        
        spin = QDoubleSpinBox()
        spin.setRange(0, 100)
        spin.setValue(e.crease_weight * 100)
        spin.valueChanged.connect(lambda val: self.property_changed.emit("crease_weight", val / 100.0))
        self.form_layout.addRow("Crease %:", spin)

    def set_face_properties(self, face_index: int, mesh):
        self.clear()
        if not mesh or face_index >= len(mesh.faces): return
        
        f = mesh.faces[face_index]
        self.group_box.setTitle(f"Face {face_index}")
        
        self.form_layout.addRow("Normal X:", QLabel(f"{f.normal[0]:.3f}"))
        self.form_layout.addRow("Normal Y:", QLabel(f"{f.normal[1]:.3f}"))
        self.form_layout.addRow("Normal Z:", QLabel(f"{f.normal[2]:.3f}"))

    def set_feature_properties(self, feature):
        self.clear()
        self.group_box.setTitle("Feature")
        # Add dynamic fields based on feature properties...
