import os
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QSplitter, QMenuBar, QMenu, QToolBar, QStatusBar,
                               QFileDialog, QMessageBox, QApplication)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QPalette, QColor, QKeySequence, QIcon

from src.gui.viewport import MeshViewport
from src.gui.panels import FeatureTreePanel, PropertiesPanel
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
except Exception as e:
    print(f"[WARNING] Could not import I/O modules: {e}")
    traceback.print_exc()
    import_stl = None
    import_obj = None
    import_step = None

try:
    import src.subd.primitives as primitives
    import src.subd.catmull_clark as catmull_clark
except Exception as e:
    print(f"[WARNING] Could not import SubD modules: {e}")
    traceback.print_exc()
    primitives = None
    catmull_clark = None

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
        
        # Splitter for panels
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Left Panel (Feature Tree)
        self.feature_panel = FeatureTreePanel()
        self.feature_panel.setMinimumWidth(200)
        
        # Center (Viewport)
        self.viewport = MeshViewport()
        
        # Right Panel (Properties)
        self.properties_panel = PropertiesPanel()
        self.properties_panel.setMinimumWidth(250)
        
        # Add to splitter
        self.splitter.addWidget(self.feature_panel)
        self.splitter.addWidget(self.viewport)
        self.splitter.addWidget(self.properties_panel)
        
        # Set stretch factors (Viewport takes most space)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Menus & Toolbars
        self._create_actions()
        self._create_menus()
        self._create_toolbar()

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
        self.act_view_reset = QAction("Reset Camera", self)
        self.act_view_reset.triggered.connect(self.viewport.reset_camera)

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
        dlg = ExportDialog(self)
        if dlg.exec_():
            self.status_bar.showMessage("Export triggered.")
            
    def on_create_primitive(self):
        dlg = PrimitiveDialog(self)
        if dlg.exec_():
            params = dlg.get_params()
            if primitives and HalfEdgeMesh:
                try:
                    ptype = params['type']
                    # Using hypothetical API for primitives
                    if ptype == 'box':
                        self.current_mesh = primitives.create_box()
                    elif ptype == 'cylinder':
                        self.current_mesh = primitives.create_cylinder()
                    elif ptype == 'plane':
                        self.current_mesh = primitives.create_plane()
                    else:
                        self.current_mesh = primitives.create_box() # fallback
                        
                    self.viewport.set_mesh(self.current_mesh)
                    self.properties_panel.set_mesh_info(self.current_mesh)
                    self.status_bar.showMessage(f"Created {ptype} primitive.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not create primitive: {e}")
            else:
                QMessageBox.warning(self, "Error", "Backend modules (primitives) not available.")

    def on_subdivide(self):
        if not self.current_mesh: return
        dlg = SubdivideDialog(self)
        if dlg.exec_():
            if catmull_clark:
                try:
                    params = dlg.get_params()
                    self.current_mesh = catmull_clark.subdivide(self.current_mesh, params['levels'])
                    self.viewport.update_mesh(self.current_mesh)
                    self.properties_panel.set_mesh_info(self.current_mesh)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Subdivision failed: {e}")
            else:
                QMessageBox.warning(self, "Error", "Subdivision backend not available.")

    def on_shell(self):
        dlg = ShellThickenDialog(self)
        dlg.exec_()

    def on_convert_nurbs(self):
        dlg = ConvertNURBSDialog(self)
        dlg.exec_()

    def on_quad_wrap(self):
        dlg = QuadWrapDialog(self)
        dlg.exec_()

    def on_shrink_wrap(self):
        dlg = ShrinkWrapDialog(self)
        dlg.exec_()
