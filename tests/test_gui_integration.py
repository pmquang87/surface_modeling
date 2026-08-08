import sys
import os

sys.path.insert(0, r'C:\Users\pmqua\PycharmProjects\surface_modeling')

os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # headless mode
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

import pyvista as pv
pv.OFF_SCREEN = True

from PySide6.QtWidgets import QApplication
from src.core.feature_tree import Feature
from src.gui.main_window import PowerSurfacingMainWindow
from src.io.importers import import_stl
from src.gui.dialogs import PrimitiveDialog, SubdivideDialog, QuadWrapDialog, ShrinkWrapDialog, ShellThickenDialog, ConvertNURBSDialog, ExportDialog

def run_tests():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    print("1. Window construction...")
    try:
        window = PowerSurfacingMainWindow()
        print("PASS: Window created")
    except Exception as e:
        print(f"FAIL: Window creation failed - {e}")
        return

    print("2. Menu structure...")
    try:
        menus = [m.title() for m in window.menuBar().findChildren(type(window.menuBar().addMenu("test")))]
        # Actually window.menuBar().actions() -> text()
        menu_titles = [action.text() for action in window.menuBar().actions()]
        expected_menus = ['&File', '&Create', '&SubD', '&Operations', '&Reverse Engineering', '&View', '&Select']
        # Remove ampersands for checking
        menu_titles_clean = [t.replace('&', '') for t in menu_titles]
        expected_clean = [t.replace('&', '') for t in expected_menus]
        
        missing = [m for m in expected_clean if m not in menu_titles_clean]
        if missing:
            print(f"FAIL: Missing menus: {missing}")
        else:
            print("PASS: All menus present")
    except Exception as e:
        print(f"FAIL: Menu check failed - {e}")

    print("3. Import pipeline...")
    try:
        path = r'E:\foxcore_data\_MITEB\20260728_vorbereitungSLM\7_LLzugdruck_maxstress_smooth_iso03_newtry_red50.STL'
        if not os.path.exists(path):
            print(f"FAIL: Test file does not exist: {path}")
        else:
            mesh = import_stl(path)
            window.current_mesh = mesh
            window.viewport.set_mesh(mesh)
            
            if hasattr(window, 'current_mesh') and window.current_mesh is not None:
                verts = window.current_mesh.n_points
                faces = window.current_mesh.n_cells
                if verts > 0 and faces > 0:
                    print(f"PASS: Import successful. Verts: {verts}, Faces: {faces}")
                else:
                    print("FAIL: Import resulted in empty mesh")
            else:
                print("FAIL: window.current_mesh not set")
    except Exception as e:
        print(f"FAIL: Import pipeline failed - {e}")

    print("4. Primitive creation...")
    try:
        box = pv.Cube()
        window.viewport.set_mesh(box)
        print("PASS: Box created and set on viewport")
    except Exception as e:
        print(f"FAIL: Primitive creation failed - {e}")

    print("5. Subdivision...")
    try:
        if hasattr(box, 'subdivide'):
            sub_box = box.subdivide(1, subfilter='linear')
            window.viewport.set_mesh(sub_box)
            print("PASS: Box subdivided and set on viewport")
        else:
            print("FAIL: box missing subdivide method")
    except Exception as e:
        print(f"FAIL: Subdivision failed - {e}")

    print("6. Properties panel...")
    try:
        window.properties_panel.set_mesh_info(sub_box)
        print("PASS: Properties panel set_mesh_info succeeded")
    except Exception as e:
        print(f"FAIL: Properties panel failed - {e}")

    print("7. Feature tree panel...")
    try:
        window.feature_tree.add_feature(Feature("BoxPrimitive", "primitive", {"size": 1.0}))
        print("PASS: Feature tree panel succeeded")
    except Exception as e:
        print(f"FAIL: Feature tree panel failed - {e}")

    print("8. Viewport methods...")
    try:
        window.viewport.set_display_mode('Wireframe')
        window.viewport.reset_camera()
        window.viewport.clear()
        print("PASS: Viewport methods succeeded")
    except Exception as e:
        print(f"FAIL: Viewport methods failed - {e}")

    print("9. Dialog construction...")
    try:
        dialogs = [
            PrimitiveDialog(window),
            SubdivideDialog(window),
            QuadWrapDialog(window),
            ShrinkWrapDialog(window),
            ShellThickenDialog(window),
            ConvertNURBSDialog(window),
            ExportDialog(window)
        ]
        print("PASS: All dialogs instantiated successfully")
    except Exception as e:
        print(f"FAIL: Dialog construction failed - {e}")

if __name__ == '__main__':
    run_tests()
