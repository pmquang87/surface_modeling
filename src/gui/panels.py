import numpy as np

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
                               QLabel, QFormLayout, QSpinBox, QDoubleSpinBox, QGroupBox,
                               QMenu, QPushButton)
from PySide6.QtCore import Signal, Qt

try:
    from src.core.feature_tree import FeatureTree
except ImportError:
    FeatureTree = None


def face_normal(mesh, face) -> np.ndarray:
    """Newell normal of a single face, computed on demand and cached on it.

    Importers and the primitive factories leave ``Face.normal`` at (0, 0, 0) —
    only ``HalfEdgeMesh.compute_face_normals()`` ever fills it in, and nothing on
    the display path used to call it, so the properties panel showed
    (0.000, 0.000, 0.000) for every imported mesh. Doing it per displayed face
    keeps this off the O(faces) path (positions can change at any time, so a
    non-zero cached value is no proof it is still current).

    Mirrors ``HalfEdgeMesh.compute_face_normals`` (Newell over every edge), which
    is orientation-correct for reflex corners and warped quads alike.
    """
    try:
        verts = [v.position for v in mesh.get_face_vertices(face)]
    except Exception:
        return np.zeros(3, dtype=np.float64)
    if len(verts) < 3:
        return np.zeros(3, dtype=np.float64)

    P = np.asarray(verts, dtype=np.float64)
    Q = np.roll(P, -1, axis=0)
    d = P - Q
    s = P + Q
    n = np.array([
        np.dot(d[:, 1], s[:, 2]),
        np.dot(d[:, 2], s[:, 0]),
        np.dot(d[:, 0], s[:, 1]),
    ], dtype=np.float64)
    norm = np.linalg.norm(n)
    n = n / norm if norm > 1e-8 else np.zeros(3, dtype=np.float64)
    try:
        face.normal = n
    except Exception:
        pass  # read-only/slotted face implementations
    return n

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
        idx = self.tree_widget.indexOfTopLevelItem(item)
        edit_action.triggered.connect(lambda checked=False, i=idx: self.feature_selected.emit(i))
        delete_action.triggered.connect(lambda checked=False, i=idx: self.feature_deleted.emit(i))
        toggle_action.triggered.connect(lambda checked=False, i=idx: self.feature_toggled.emit(i, True))
        
        menu.exec_(self.tree_widget.viewport().mapToGlobal(position))

class PropertiesPanel(QWidget):
    """Right panel showing properties of the current selection/operation."""
    
    property_changed = Signal(str, object)
    expand_selection_requested = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.group_box = QGroupBox("Properties")
        self.form_layout = QFormLayout(self.group_box)

        # What the currently shown editors write to: (kind, index) with kind in
        # {'vertex', 'edge', 'face', 'mesh'}. Consumers of property_changed need
        # this to know which element an edit belongs to.
        self.current_target = None

        self.layout.addWidget(self.group_box)
        self.layout.addStretch()

    def clear(self):
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.group_box.setTitle("Properties")
        self.current_target = None

    def set_mesh_info(self, mesh):
        self.clear()
        if not mesh: return
        self.current_target = ('mesh', None)
        self.group_box.setTitle("Mesh Info")
        self.form_layout.addRow("Vertices:", QLabel(str(len(mesh.vertices))))
        self.form_layout.addRow("Edges:", QLabel(str(len(mesh.edges))))
        self.form_layout.addRow("Faces:", QLabel(str(len(mesh.faces))))

    def set_vertex_properties(self, vertex_index: int, mesh):
        self.clear()
        if not mesh or vertex_index >= len(mesh.vertices): return

        v = mesh.vertices[vertex_index]
        self.current_target = ('vertex', vertex_index)
        self.group_box.setTitle(f"Vertex {vertex_index}")

        for i, axis in enumerate(['X', 'Y', 'Z']):
            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(6)
            spin.setSingleStep(0.1)
            # setValue before connect: no spurious edit signal on populate.
            spin.setValue(float(v.position[i]))
            spin.valueChanged.connect(lambda val, a=axis: self.property_changed.emit(f"pos_{a.lower()}", val))
            self.form_layout.addRow(f"Position {axis}:", spin)

    def set_edge_properties(self, edge_index: int, mesh):
        self.clear()
        if not mesh or edge_index >= len(mesh.edges): return

        e = mesh.edges[edge_index]
        self.current_target = ('edge', edge_index)
        self.group_box.setTitle(f"Edge {edge_index}")

        spin = QDoubleSpinBox()
        spin.setRange(0, 100)
        spin.setValue(e.crease_weight * 100)
        spin.valueChanged.connect(lambda val: self.property_changed.emit("crease_weight", val / 100.0))
        self.form_layout.addRow("Crease %:", spin)

    def set_face_properties(self, face_indices, mesh):
        self.clear()
        if not mesh or face_indices is None: return

        # `0` is a perfectly good face index; a plain `not face_indices` test
        # used to drop it and leave the panel empty for the first face.
        if isinstance(face_indices, (int, np.integer)):
            face_indices = [int(face_indices)]
        else:
            face_indices = list(face_indices)
        if not face_indices: return

        if len(face_indices) == 1:
            face_index = face_indices[0]
            if face_index >= len(mesh.faces): return
            f = mesh.faces[face_index]
            self.current_target = ('face', face_index)
            self.group_box.setTitle(f"Face {face_index}")

            # Computed here rather than read straight off the face: nothing on
            # the import/display path fills Face.normal in, and a stored value
            # goes stale as soon as a vertex moves.
            normal = face_normal(mesh, f)

            # Face normals are derived from the vertex positions, so they are
            # shown read-only rather than offering an edit that cannot be applied.
            self.form_layout.addRow("Normal X:", QLabel(f"{normal[0]:.3f}"))
            self.form_layout.addRow("Normal Y:", QLabel(f"{normal[1]:.3f}"))
            self.form_layout.addRow("Normal Z:", QLabel(f"{normal[2]:.3f}"))
        else:
            self.current_target = ('face', None)
            self.group_box.setTitle(f"Selected Faces ({len(face_indices)})")

    def set_feature_properties(self, feature):
        self.clear()
        self.group_box.setTitle("Feature")
        # Add dynamic fields based on feature properties...

    def set_tool_ui(self, title: str, tool_widget: QWidget, apply_callback=None, cancel_callback=None):
        self.clear()
        self.group_box.setTitle(title)
        
        # Add the custom tool parameters widget
        self.form_layout.addRow(tool_widget)
        
        # Add Apply / Cancel buttons
        btn_layout = QHBoxLayout()
        if apply_callback:
            apply_btn = QPushButton("Apply")
            apply_btn.setStyleSheet("background-color: #2e7d32; color: white;")
            apply_btn.clicked.connect(apply_callback)
            btn_layout.addWidget(apply_btn)
            
        if cancel_callback:
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(cancel_callback)
            btn_layout.addWidget(cancel_btn)
            
        if apply_callback or cancel_callback:
            self.form_layout.addRow(btn_layout)


from PySide6.QtWidgets import QRadioButton, QButtonGroup, QCheckBox, QHBoxLayout, QGridLayout

class SelectionPanel(QWidget):
    """Unified Selection Tool Panel with modifiers and operations."""
    
    selection_mode_changed = Signal(str)
    selection_method_changed = Signal(str)
    selection_modifier_changed = Signal(str)
    selection_operation_requested = Signal(str)
    tangent_selection_requested = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Method & Entity Type
        grid = QGridLayout()
        
        # Tools
        self.tool_group = QButtonGroup(self)
        self.rb_pick = QRadioButton("Pick")
        self.rb_box = QRadioButton("Box (Area)")
        self.rb_pick.setChecked(True)
        self.tool_group.addButton(self.rb_pick, 1)
        self.tool_group.addButton(self.rb_box, 2)
        grid.addWidget(self.rb_pick, 0, 0)
        grid.addWidget(self.rb_box, 1, 0)
        
        self.cb_through = QCheckBox("Select Through")
        self.cb_through.setToolTip("If unchecked, selects visible only.")
        self.cb_through.setEnabled(False)
        grid.addWidget(self.cb_through, 2, 0)
        self.rb_box.toggled.connect(self.cb_through.setEnabled)
        
        self.tool_group.idClicked.connect(lambda id: self.selection_method_changed.emit('box' if id == 2 else 'pick'))
        
        # Modifiers
        self.mod_group = QButtonGroup(self)
        self.rb_new = QRadioButton("New")
        self.rb_add = QRadioButton("Add")
        self.rb_remove = QRadioButton("Remove")
        self.rb_new.setChecked(True)
        self.mod_group.addButton(self.rb_new, 1)
        self.mod_group.addButton(self.rb_add, 2)
        self.mod_group.addButton(self.rb_remove, 3)
        grid.addWidget(self.rb_new, 0, 1)
        grid.addWidget(self.rb_add, 1, 1)
        grid.addWidget(self.rb_remove, 2, 1)
        
        def mod_emit(id):
            modes = {1: 'new', 2: 'add', 3: 'remove'}
            self.selection_modifier_changed.emit(modes[id])
        self.mod_group.idClicked.connect(mod_emit)
        
        # Entity Type
        self.entity_group = QButtonGroup(self)
        self.rb_vertex = QRadioButton("Vertex")
        self.rb_edge = QRadioButton("Edge")
        self.rb_face = QRadioButton("Face")
        self.rb_face.setChecked(True)
        self.entity_group.addButton(self.rb_vertex, 1)
        self.entity_group.addButton(self.rb_edge, 2)
        self.entity_group.addButton(self.rb_face, 3)
        grid.addWidget(self.rb_vertex, 0, 2)
        grid.addWidget(self.rb_edge, 1, 2)
        grid.addWidget(self.rb_face, 2, 2)
        
        def entity_emit(id):
            entities = {1: 'vertex', 2: 'edge', 3: 'face'}
            self.selection_mode_changed.emit(entities[id])
        self.entity_group.idClicked.connect(entity_emit)
        
        gb = QGroupBox("Selection Method & Type")
        gb.setLayout(grid)
        self.layout.addWidget(gb)
        
        # Operations
        ops_group = QGroupBox("Operations")
        ops_layout = QVBoxLayout(ops_group)
        
        self.btn_adj = QPushButton("Adjacent")
        self.btn_adj.clicked.connect(lambda: self.selection_operation_requested.emit("adjacent"))
        ops_layout.addWidget(self.btn_adj)
        
        self.btn_conn = QPushButton("Connected (Attach)")
        self.btn_conn.clicked.connect(lambda: self.selection_operation_requested.emit("connected"))
        ops_layout.addWidget(self.btn_conn)
        
        self.btn_inv = QPushButton("Reverse (Invert)")
        self.btn_inv.clicked.connect(lambda: self.selection_operation_requested.emit("invert"))
        ops_layout.addWidget(self.btn_inv)
        
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(lambda: self.selection_operation_requested.emit("clear"))
        ops_layout.addWidget(self.btn_clear)
        
        # Tangency Tools
        tangent_layout = QHBoxLayout()
        tangent_layout.addWidget(QLabel("Angle:"))
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.0, 180.0)
        self.tolerance_spin.setValue(15.0)
        tangent_layout.addWidget(self.tolerance_spin)
        ops_layout.addLayout(tangent_layout)
        
        self.btn_tan = QPushButton("Expand Tangent")
        self.btn_tan.clicked.connect(lambda: self.tangent_selection_requested.emit(self.tolerance_spin.value()))
        ops_layout.addWidget(self.btn_tan)
        
        self.layout.addWidget(ops_group)
        self.layout.addStretch()

    # -- current state -----------------------------------------------------
    # The panel is the single source of truth for the initial selection state;
    # MainWindow pushes these into the viewport at startup so the first click
    # already works instead of falling into the viewport's old 'none' default.

    def current_selection_mode(self) -> str:
        if self.rb_vertex.isChecked():
            return 'vertex'
        if self.rb_edge.isChecked():
            return 'edge'
        if self.rb_face.isChecked():
            return 'face'
        return 'none'

    def current_selection_method(self) -> str:
        return 'box' if self.rb_box.isChecked() else 'pick'

    def current_selection_modifier(self) -> str:
        if self.rb_add.isChecked():
            return 'add'
        if self.rb_remove.isChecked():
            return 'remove'
        return 'new'

    def set_selection_mode(self, mode: str):
        self.entity_group.blockSignals(True)
        if mode == 'vertex':
            self.rb_vertex.setChecked(True)
        elif mode == 'edge':
            self.rb_edge.setChecked(True)
        elif mode == 'face':
            self.rb_face.setChecked(True)
        elif mode == 'none':
            self.entity_group.setExclusive(False)
            self.rb_vertex.setChecked(False)
            self.rb_edge.setChecked(False)
            self.rb_face.setChecked(False)
            self.entity_group.setExclusive(True)
        self.entity_group.blockSignals(False)
        self.selection_mode_changed.emit(mode)

