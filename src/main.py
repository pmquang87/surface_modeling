"""Python Surfacing — Desktop Application
Entry point for the desktop application.
"""
import sys
import os

# Force Qt API to PySide6 for PyVistaQt before importing it
os.environ["QT_API"] = "pyside6"

# Add project root to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PySide6.QtWidgets import QApplication
from src.gui.main_window import PowerSurfacingMainWindow

def main():
    # Setup QApplication first before any VTK/PyVista windows are created
    app = QApplication(sys.argv)
    app.setApplicationName("Python Surfacing")
    window = PowerSurfacingMainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
