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

try:
    from src.core.halfedge_mesh import HalfEdgeMesh
    from src.core.feature_tree import FeatureTree
except ImportError:
    HalfEdgeMesh = None
    FeatureTree = None

try:
    import src.subd.primitives as primitives
    import src.subd.catmull_clark as catmull_clark
except ImportError:
    primitives = None
    catmull_clark = None

class PowerSurfacingMainWindow(QMainWindow):
    """Main application window for Python Power Surfacing."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Power Surfacing")
        self.resize(1200, 800)
        
        self.current_mesh = None
        if FeatureTree:
            self.feature_tree = FeatureTree()
        else:
            self.feature_tree = None
            
        self._setup_ui()
        self._setup_dark_theme()
        
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

    def _setup_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(43, 43, 43))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(43, 43, 43))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(43, 43, 43))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        
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
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Mesh", "", "Mesh Files (*.obj *.stl *.step)")
        if filepath:
            self.status_bar.showMessage(f"Opened {os.path.basename(filepath)}")
            # Logic to load mesh using importers

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
