import os
import sys
import io
import datetime
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QSplitter, QMenuBar, QMenu, QToolBar, QStatusBar,
                               QFileDialog, QMessageBox, QApplication, QTextEdit,
                               QComboBox, QLabel, QPushButton)
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QAction, QPalette, QColor, QKeySequence, QIcon, QTextCursor

from src.gui.viewport import MeshViewport
from src.gui.panels import FeatureTreePanel, PropertiesPanel, SelectionPanel
from src.gui.dialogs import (PrimitiveDialog, SubdivideDialog, QuadWrapDialog, 
                             ShrinkWrapDialog, ShellThickenDialog, 
                             ConvertNURBSDialog, ExportDialog)

import traceback

try:
    from src.core.halfedge_mesh import HalfEdgeMesh
    from src.core.feature_tree import FeatureTree
except Exception as e:
    print(f"[WARNING] Could not import core modules: {e}")
    traceback.print_exc()
    HalfEdgeMesh = None
    FeatureTree = None

try:
    from src.io.importers import import_stl, import_obj, import_step
    from src.io.exporters import export_stl, export_obj
except Exception as e:
    print(f"[WARNING] Could not import I/O modules: {e}")
    traceback.print_exc()
    import_stl = None
    import_obj = None
    import_step = None
    export_stl = None
    export_obj = None

try:
    import src.subd.primitives as primitives
    import src.subd.catmull_clark as catmull_clark
except Exception as e:
    print(f"[WARNING] Could not import SubD modules: {e}")
    traceback.print_exc()
    primitives = None
    catmull_clark = None

try:
    from src.reverse_engineering.quad_wrap import QuadWrapper
    from src.reverse_engineering.shrink_wrap import ShrinkWrapper
    from src.reverse_engineering.mesh_tools import smooth_mesh, decimate_mesh, fill_holes
except Exception as e:
    print(f"[WARNING] Could not import RE modules: {e}")
    traceback.print_exc()
    QuadWrapper = None
    ShrinkWrapper = None
    smooth_mesh = None
    decimate_mesh = None
    fill_holes = None

try:
    from src.operations.shell_thicken import shell_solid, thicken_surface
except Exception as e:
    print(f"[WARNING] Could not import operations modules: {e}")
    traceback.print_exc()
    shell_solid = None
    thicken_surface = None

try:
    from src.nurbs.converter import SubDToNURBSConverter
except Exception as e:
    print(f"[WARNING] Could not import NURBS converter: {e}")
    traceback.print_exc()
    SubDToNURBSConverter = None


class LogStream(QObject):
    """Redirects sys.stdout/stderr to a Qt signal for in-GUI display."""
    message = Signal(str)

    def __init__(self, original_stream=None):
        super().__init__()
        self.original = original_stream

    def write(self, text):
        if text.strip():
            self.message.emit(text)
        if self.original:
            self.original.write(text)
            self.original.flush()

    def flush(self):
        if self.original:
            self.original.flush()

class PowerSurfacingMainWindow(QMainWindow):
    """Main application window for Python Surfacing."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Surfacing")
        self.resize(1200, 800)
        
        self.current_mesh = None
        if FeatureTree:
            self.feature_tree = FeatureTree()
        else:
            self.feature_tree = None
            
        self._setup_ui()
        self._setup_light_theme()
        
    def _setup_ui(self):
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Vertical splitter: top = workspace, bottom = log panel
        self.v_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(self.v_splitter)

        # Horizontal splitter for panels
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel (Feature Tree)
        self.feature_panel = FeatureTreePanel()
        self.feature_panel.setMinimumWidth(200)
        
        # Center (Viewport)
        self.viewport = MeshViewport()
        
        # Right Panel (Selection + Properties)
        self.right_panel_widget = QWidget()
        right_layout = QVBoxLayout(self.right_panel_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.selection_panel = SelectionPanel()
        self.properties_panel = PropertiesPanel()
        
        right_layout.addWidget(self.selection_panel)
        right_layout.addWidget(self.properties_panel)
        self.right_panel_widget.setMinimumWidth(250)
        
        # Add to horizontal splitter
        self.splitter.addWidget(self.feature_panel)
        self.splitter.addWidget(self.viewport)
        self.splitter.addWidget(self.right_panel_widget)
        
        # Wire up viewport and panels
        self.viewport.selection_changed.connect(self.on_selection_changed)
        
        # Connect SelectionPanel signals to Viewport
        self.selection_panel.selection_mode_changed.connect(self.viewport.set_selection_mode)
        self.selection_panel.selection_method_changed.connect(self.viewport.set_selection_method)
        self.selection_panel.selection_modifier_changed.connect(self.viewport.set_selection_modifier)
        self.selection_panel.selection_operation_requested.connect(self.viewport.run_selection_operation)
        self.selection_panel.tangent_selection_requested.connect(self.on_expand_selection)
        self.selection_panel.cb_through.toggled.connect(self.viewport.set_box_select_through)
        
        # Set stretch factors (Viewport takes most space)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)

        # Log Panel (bottom)
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumHeight(150)
        self.log_panel.setStyleSheet(
            "QTextEdit { font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; "
            "background-color: #ffffff; color: #1e1e1e; border-top: 1px solid #c0c0c0; }"
        )
        self.log_panel.setPlaceholderText("Application log...")

        # Add to vertical splitter
        self.v_splitter.addWidget(self.splitter)
        self.v_splitter.addWidget(self.log_panel)
        self.v_splitter.setStretchFactor(0, 1)
        self.v_splitter.setStretchFactor(1, 0)
        
        # Redirect stdout/stderr to log panel
        self._stdout_stream = LogStream(sys.stdout)
        self._stderr_stream = LogStream(sys.stderr)
        self._stdout_stream.message.connect(self._append_log)
        self._stderr_stream.message.connect(lambda msg: self._append_log(msg, error=True))
        sys.stdout = self._stdout_stream
        sys.stderr = self._stderr_stream

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Menus & Toolbars
        self._create_actions()
        self._create_menus()
        self._create_toolbar()

    def _append_log(self, text, error=False):
        """Append a message to the in-GUI log panel."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        if error:
            self.log_panel.append(f'<span style="color:#cc0000;">[{timestamp}] {text}</span>')
        else:
            self.log_panel.append(f'[{timestamp}] {text}')
        self.log_panel.moveCursor(QTextCursor.End)

    def log(self, message):
        """Public method to log a message to the GUI log panel."""
        print(message)  # goes through LogStream → _append_log

    def _setup_light_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(30, 30, 30))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ToolTipText, QColor(30, 30, 30))
        palette.setColor(QPalette.Text, QColor(30, 30, 30))
        palette.setColor(QPalette.Button, QColor(230, 230, 230))
        palette.setColor(QPalette.ButtonText, QColor(30, 30, 30))
        palette.setColor(QPalette.BrightText, QColor(200, 30, 30))
        palette.setColor(QPalette.Link, QColor(0, 102, 204))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        palette.setColor(QPalette.Light, QColor(255, 255, 255))
        palette.setColor(QPalette.Midlight, QColor(227, 227, 227))
        palette.setColor(QPalette.Mid, QColor(160, 160, 160))
        palette.setColor(QPalette.Dark, QColor(130, 130, 130))
        palette.setColor(QPalette.Shadow, QColor(105, 105, 105))
        
        app = QApplication.instance()
        if app:
            app.setPalette(palette)
            app.setStyle("Fusion")

    def _create_actions(self):
        # File Actions
        self.act_new = QAction("New", self)
        self.act_new.setShortcut("Ctrl+N")
        self.act_new.triggered.connect(self.on_new)
        
        self.act_open = QAction("Open...", self)
        self.act_open.setShortcut("Ctrl+O")
        self.act_open.triggered.connect(self.on_open)
        
        self.act_save = QAction("Save", self)
        self.act_save.setShortcut("Ctrl+S")
        self.act_save.triggered.connect(self.on_export)
        
        self.act_export = QAction("Export...", self)
        self.act_export.triggered.connect(self.on_export)
        
        self.act_exit = QAction("Exit", self)
        self.act_exit.setShortcut("Ctrl+Q")
        self.act_exit.triggered.connect(self.close)
        
        # Create Actions
        self.act_create_primitive = QAction("Primitive...", self)
        self.act_create_primitive.triggered.connect(self.on_create_primitive)
        
        # View Actions
        self.act_view_solid = QAction("Solid", self)
        self.act_view_solid.triggered.connect(lambda: self.viewport.set_display_mode('solid'))
        self.act_view_wire = QAction("Wireframe", self)
        self.act_view_wire.triggered.connect(lambda: self.viewport.set_display_mode('wireframe'))
        self.act_view_solid_wire = QAction("Solid + Wireframe", self)
        self.act_view_solid_wire.triggered.connect(lambda: self.viewport.set_display_mode('solid+wireframe'))
        self.act_view_reset = QAction("Re-center Camera", self)
        self.act_view_reset.triggered.connect(self.viewport.reset_camera)

        # Camera Views
        self.act_cam_iso = QAction("Isometric", self)
        self.act_cam_iso.triggered.connect(self.viewport.plotter.view_isometric)
        
        self.act_cam_top = QAction("Top (XY)", self)
        self.act_cam_top.triggered.connect(self.viewport.plotter.view_xy)
        
        self.act_cam_bottom = QAction("Bottom", self)
        self.act_cam_bottom.triggered.connect(lambda: self.viewport.plotter.view_xy(negative=True))
        
        self.act_cam_front = QAction("Front (XZ)", self)
        self.act_cam_front.triggered.connect(self.viewport.plotter.view_xz)
        
        self.act_cam_back = QAction("Back", self)
        self.act_cam_back.triggered.connect(lambda: self.viewport.plotter.view_xz(negative=True))
        
        self.act_cam_right = QAction("Right (YZ)", self)
        self.act_cam_right.triggered.connect(self.viewport.plotter.view_yz)
        
        self.act_cam_left = QAction("Left", self)
        self.act_cam_left.triggered.connect(lambda: self.viewport.plotter.view_yz(negative=True))

        # Select Actions
        self.act_sel_vertex = QAction("Select Vertex", self)
        self.act_sel_vertex.triggered.connect(lambda: self.viewport.set_selection_mode('vertex'))
        self.act_sel_edge = QAction("Select Edge", self)
        self.act_sel_edge.triggered.connect(lambda: self.viewport.set_selection_mode('edge'))
        self.act_sel_face = QAction("Select Face", self)
        self.act_sel_face.triggered.connect(lambda: self.viewport.set_selection_mode('face'))
        self.act_sel_none = QAction("Select None", self)
        self.act_sel_none.triggered.connect(lambda: self.viewport.set_selection_mode('none'))

    def _create_menus(self):
        menubar = self.menuBar()
        
        menu_file = menubar.addMenu("&File")
        menu_file.addAction(self.act_new)
        menu_file.addAction(self.act_open)
        menu_file.addAction(self.act_save)
        menu_file.addAction(self.act_export)
        menu_file.addSeparator()
        menu_file.addAction(self.act_exit)
        
        menu_create = menubar.addMenu("&Create")
        menu_create.addAction(self.act_create_primitive)
        
        menu_subd = menubar.addMenu("&SubD")
        menu_subd.addAction("Subdivide...", self.on_subdivide)
        
        menu_ops = menubar.addMenu("&Operations")
        menu_ops.addAction("Shell / Thicken...", self.on_shell)
        menu_ops.addAction("Convert to NURBS...", self.on_convert_nurbs)
        
        menu_rev = menubar.addMenu("&Reverse Engineering")
        menu_rev.addAction("Quad Wrap...", self.on_quad_wrap)
        menu_rev.addAction("Shrink Wrap...", self.on_shrink_wrap)
        
        menu_view = menubar.addMenu("&View")
        menu_view.addAction(self.act_view_solid)
        menu_view.addAction(self.act_view_wire)
        menu_view.addAction(self.act_view_solid_wire)
        menu_view.addSeparator()
        menu_view.addAction(self.act_view_reset)
        menu_view.addSeparator()
        menu_view.addAction(self.act_cam_iso)
        menu_view.addAction(self.act_cam_top)
        menu_view.addAction(self.act_cam_bottom)
        menu_view.addAction(self.act_cam_front)
        menu_view.addAction(self.act_cam_back)
        menu_view.addAction(self.act_cam_right)
        menu_view.addAction(self.act_cam_left)

        menu_select = menubar.addMenu("&Select")
        menu_select.addAction(self.act_sel_vertex)
        menu_select.addAction(self.act_sel_edge)
        menu_select.addAction(self.act_sel_face)
        menu_select.addAction(self.act_sel_none)

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        toolbar.addAction(self.act_new)
        toolbar.addAction(self.act_open)
        toolbar.addAction(self.act_save)
        toolbar.addSeparator()
        toolbar.addAction(self.act_create_primitive)
        toolbar.addSeparator()
        toolbar.addAction(self.act_sel_vertex)
        toolbar.addAction(self.act_sel_edge)
        toolbar.addAction(self.act_sel_face)
        
        toolbar.addSeparator()
        btn_recenter = QPushButton("Re-center")
        btn_recenter.clicked.connect(self.viewport.reset_camera)
        toolbar.addWidget(btn_recenter)
        
        toolbar.addWidget(QLabel("  View: "))
        self.combo_view = QComboBox()
        self.combo_view.addItems(["", "Isometric", "Top", "Bottom", "Front", "Back", "Right", "Left"])
        self.combo_view.currentIndexChanged.connect(self._on_view_combo_changed)
        toolbar.addWidget(self.combo_view)

    def _on_view_combo_changed(self, index):
        if index == 1: self.viewport.plotter.view_isometric()
        elif index == 2: self.viewport.plotter.view_xy()
        elif index == 3: self.viewport.plotter.view_xy(negative=True)
        elif index == 4: self.viewport.plotter.view_xz()
        elif index == 5: self.viewport.plotter.view_xz(negative=True)
        elif index == 6: self.viewport.plotter.view_yz()
        elif index == 7: self.viewport.plotter.view_yz(negative=True)
        self.combo_view.setCurrentIndex(0)

    def on_selection_changed(self, indices):
        if not self.current_mesh: return
        if self.viewport.selection_mode == 'vertex' and len(indices) == 1:
            self.properties_panel.set_vertex_properties(indices[0], self.current_mesh)
        elif self.viewport.selection_mode == 'edge' and len(indices) == 1:
            self.properties_panel.set_edge_properties(indices[0], self.current_mesh)
        elif self.viewport.selection_mode == 'face' and len(indices) > 0:
            self.properties_panel.set_face_properties(indices, self.current_mesh)
            
    def on_expand_selection(self, angle):
        if not self.current_mesh: return
        current_faces = self.viewport.get_selected_faces()
        if not current_faces: return
        
        try:
            new_faces = self.current_mesh.expand_selection_by_angle(current_faces, angle)
            self.viewport.highlight_selection(new_faces, 'face')
            self.properties_panel.set_face_properties(new_faces, self.current_mesh)
        except Exception as e:
            self.log(f"Expand selection failed: {e}")

    def on_new(self):
        self.current_mesh = None
        if HalfEdgeMesh:
            self.current_mesh = HalfEdgeMesh()
        self.viewport.clear()
        self.properties_panel.clear()
        self.status_bar.showMessage("New mesh created.")

    def on_open(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Mesh", "",
            "All Supported (*.stl *.obj *.stp *.step);;STL Files (*.stl);;OBJ Files (*.obj);;STEP Files (*.stp *.step);;All Files (*)"
        )
        if not filepath:
            return

        ext = os.path.splitext(filepath)[1].lower()
        self.status_bar.showMessage(f"Loading {os.path.basename(filepath)}...")
        QApplication.processEvents()  # update UI before heavy I/O

        try:
            mesh = None
            if ext == '.stl':
                if import_stl:
                    mesh = import_stl(filepath)
                else:
                    QMessageBox.warning(self, "Error", "STL importer not available.")
                    return
            elif ext == '.obj':
                if import_obj:
                    mesh = import_obj(filepath)
                else:
                    QMessageBox.warning(self, "Error", "OBJ importer not available.")
                    return
            elif ext in ('.stp', '.step'):
                if import_step:
                    result = import_step(filepath)
                    mesh = result.get('mesh')
                    if mesh is None or len(mesh.vertices) == 0:
                        QMessageBox.warning(
                            self, "STEP Import",
                            "STEP import requires OpenCascade (OCP or cadquery).\n"
                            "Install with: pip install cadquery\n\n"
                            "Alternatively, export to STL and import that."
                        )
                        self.status_bar.showMessage("STEP import failed — OCP/cadquery not installed.")
                        return
                else:
                    QMessageBox.warning(self, "Error", "STEP importer not available.")
                    return
            else:
                QMessageBox.warning(self, "Error", f"Unsupported file format: {ext}")
                return

            if mesh and len(mesh.vertices) > 0:
                self.current_mesh = mesh
                self.viewport.set_mesh(mesh)
                self.properties_panel.set_mesh_info(mesh)
                vcount = len(mesh.vertices)
                fcount = len(mesh.faces)
                self.status_bar.showMessage(
                    f"Loaded {os.path.basename(filepath)} — {vcount:,} vertices, {fcount:,} faces"
                )
            else:
                QMessageBox.warning(self, "Error", "File loaded but mesh is empty.")
                self.status_bar.showMessage("Load failed — empty mesh.")

        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to load file:\n{e}")
            self.status_bar.showMessage(f"Error loading file: {e}")

    def on_export(self):
        if not self.current_mesh:
            QMessageBox.warning(self, "Export", "No mesh to export.")
            return
        dlg = ExportDialog(self)
        if dlg.exec_():
            fmt = dlg.format_combo.currentText().lower()
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export Mesh", "",
                f"{fmt.upper()} Files (*.{fmt});;All Files (*)"
            )
            if not filepath:
                return
            try:
                if fmt == 'stl' and export_stl:
                    export_stl(self.current_mesh, filepath)
                elif fmt == 'obj' and export_obj:
                    export_obj(self.current_mesh, filepath)
                else:
                    self.log(f"Export format '{fmt}' not available.")
                    return
                self.log(f"Exported to {filepath}")
                self.status_bar.showMessage(f"Exported to {os.path.basename(filepath)}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
            
    def on_create_primitive(self):
        dlg = PrimitiveDialog(self)
        if dlg.exec_():
            params = dlg.get_params()
            if primitives and HalfEdgeMesh:
                try:
                    ptype = params['type']
                    create_fn = {
                        'box': primitives.create_box,
                        'cylinder': primitives.create_cylinder,
                        'torus': primitives.create_torus,
                        'cone': primitives.create_cone,
                        'plane': primitives.create_plane,
                        'sphere': primitives.create_sphere,
                    }.get(ptype, primitives.create_box)
                    self.current_mesh = create_fn()
                    self.viewport.set_mesh(self.current_mesh)
                    self.properties_panel.set_mesh_info(self.current_mesh)
                    v = len(self.current_mesh.vertices)
                    f = len(self.current_mesh.faces)
                    self.log(f"Created {ptype} primitive — {v} vertices, {f} faces")
                    self.status_bar.showMessage(f"Created {ptype} primitive.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not create primitive: {e}")
            else:
                QMessageBox.warning(self, "Error", "Backend modules (primitives) not available.")

    def on_subdivide(self):
        if not self.current_mesh:
            QMessageBox.information(self, "Subdivide", "No mesh loaded. Load or create a mesh first.")
            return
        dlg = SubdivideDialog(self)
        if dlg.exec_():
            if catmull_clark:
                try:
                    params = dlg.get_params()
                    v_before = len(self.current_mesh.vertices)
                    f_before = len(self.current_mesh.faces)
                    self.log(f"Subdividing {params['levels']} level(s)... ({v_before} verts, {f_before} faces)")
                    QApplication.processEvents()
                    self.current_mesh = catmull_clark.subdivide(self.current_mesh, params['levels'])
                    v_after = len(self.current_mesh.vertices)
                    f_after = len(self.current_mesh.faces)
                    self.viewport.update_mesh(self.current_mesh)
                    self.properties_panel.set_mesh_info(self.current_mesh)
                    self.log(f"Subdivision complete — {v_after} vertices, {f_after} faces")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Subdivision failed: {e}")
                    self.log(f"ERROR: Subdivision failed: {e}")
            else:
                QMessageBox.warning(self, "Error", "Subdivision backend not available.")

    def on_shell(self):
        if not self.current_mesh:
            QMessageBox.information(self, "Shell/Thicken", "No mesh loaded.")
            return
        dlg = ShellThickenDialog(self)
        if dlg.exec_():
            thickness = dlg.thickness.value()
            direction = dlg.direction.currentText().lower()
            if shell_solid:
                try:
                    self.log(f"Running Shell/Thicken (thickness={thickness}, direction={direction})...")
                    QApplication.processEvents()
                    self.current_mesh = shell_solid(self.current_mesh, thickness=thickness, direction=direction)
                    self.viewport.update_mesh(self.current_mesh)
                    self.properties_panel.set_mesh_info(self.current_mesh)
                    v = len(self.current_mesh.vertices)
                    f = len(self.current_mesh.faces)
                    self.log(f"Shell/Thicken complete — {v} vertices, {f} faces")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Shell/Thicken failed: {e}")
                    self.log(f"ERROR: Shell/Thicken failed: {e}")
            else:
                QMessageBox.warning(self, "Error", "Shell/Thicken backend not available.")

    def on_convert_nurbs(self):
        if not self.current_mesh:
            QMessageBox.information(self, "NURBS", "No mesh loaded.")
            return
        dlg = ConvertNURBSDialog(self)
        if dlg.exec_():
            if SubDToNURBSConverter:
                try:
                    continuity_map = {'G0 (Position)': 0, 'G1 (Tangent)': 1, 'G2 (Curvature)': 2}
                    continuity = continuity_map.get(dlg.continuity.currentText(), 1)
                    tolerance = dlg.tolerance.value()
                    self.log(f"Converting to NURBS (continuity=G{continuity}, tol={tolerance})...")
                    QApplication.processEvents()
                    converter = SubDToNURBSConverter(continuity=continuity, tolerance=tolerance)
                    result = converter.convert(self.current_mesh)
                    patch_count = len(result.get('patches', []))
                    self.log(f"NURBS conversion complete — {patch_count} patches generated")
                    if patch_count == 0:
                        QMessageBox.warning(self, "No Patches Generated", 
                            "0 patches were generated because your mesh contains no quad (4-sided) faces.\n\n"
                            "NURBS conversion only works on quad faces. If you imported an STL, it only contains triangles. "
                            "Please run 'Reverse Engineering -> Quad Wrap' first to convert your mesh into a quad-dominant Sub-D cage.")
                    if result.get('mesh'):
                        self.current_mesh = result['mesh']
                        self.viewport.update_mesh(self.current_mesh)
                        self.properties_panel.set_mesh_info(self.current_mesh)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"NURBS conversion failed: {e}")
                    self.log(f"ERROR: NURBS conversion failed: {e}")
            else:
                QMessageBox.warning(self, "Error", "NURBS converter not available.")

    def on_quad_wrap(self):
        if not self.current_mesh:
            QMessageBox.information(self, "Quad Wrap", "No mesh loaded.")
            return
        dlg = QuadWrapDialog(self)
        if dlg.exec_():
            if QuadWrapper:
                try:
                    target_count = dlg.target_count.value()
                    smooth_weight = dlg.smoothing_weight.value()
                    
                    frozen_faces = None
                    if dlg.lock_faces.isChecked():
                        frozen_faces = self.viewport.get_selected_faces()
                        
                    self.log(f"Running Quad Wrap (target={target_count} faces, smooth={smooth_weight})...")
                    QApplication.processEvents()
                    wrapper = QuadWrapper(target_face_count=target_count, smoothing_weight=smooth_weight)
                    result = wrapper.wrap(self.current_mesh, frozen_face_ids=frozen_faces)
                    v = len(result.vertices)
                    f = len(result.faces)
                    self.current_mesh = result
                    self.viewport.set_mesh(self.current_mesh)
                    self.properties_panel.set_mesh_info(self.current_mesh)
                    self.log(f"Quad Wrap complete — {v} vertices, {f} faces")
                    self.status_bar.showMessage(f"Quad Wrap: {v} verts, {f} faces")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Quad Wrap failed: {e}")
                    self.log(f"ERROR: Quad Wrap failed: {e}")
            else:
                QMessageBox.warning(self, "Error", "Quad Wrap backend not available.")

    def on_shrink_wrap(self):
        if not self.current_mesh:
            QMessageBox.information(self, "Shrink Wrap", "No mesh loaded.")
            return
        dlg = ShrinkWrapDialog(self)
        if dlg.exec_():
            if ShrinkWrapper:
                try:
                    iterations = dlg.iterations.value()
                    
                    frozen_verts = None
                    if dlg.lock_faces.isChecked():
                        selected_faces = self.viewport.get_selected_faces()
                        if selected_faces:
                            vert_set = set()
                            for f_id in selected_faces:
                                if f_id < len(self.current_mesh.faces):
                                    face = self.current_mesh.faces[f_id]
                                    if face and face.halfedge:
                                        he = face.halfedge
                                        start = he
                                        while True:
                                            if he.vertex:
                                                vert_set.add(he.vertex.index)
                                            he = he.next
                                            if he == start or not he:
                                                break
                            frozen_verts = list(vert_set)
                            
                    self.log(f"Running Shrink Wrap ({iterations} iterations)...")
                    QApplication.processEvents()
                    shrinker = ShrinkWrapper(iterations=iterations)
                    result = shrinker.wrap(self.current_mesh, self.current_mesh, frozen_vertices=frozen_verts)
                    v = len(result.vertices)
                    f = len(result.faces)
                    self.current_mesh = result
                    self.viewport.set_mesh(self.current_mesh)
                    self.properties_panel.set_mesh_info(self.current_mesh)
                    self.log(f"Shrink Wrap complete — {v} vertices, {f} faces")
                    self.status_bar.showMessage(f"Shrink Wrap: {v} verts, {f} faces")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Shrink Wrap failed: {e}")
                    self.log(f"ERROR: Shrink Wrap failed: {e}")
            else:
                QMessageBox.warning(self, "Error", "Shrink Wrap backend not available.")
