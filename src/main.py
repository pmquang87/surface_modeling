"""Power Surfacing — Python Edition
Entry point for the desktop application.
"""
import sys
import os

# Add project root to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PySide6.QtWidgets import QApplication
from src.gui.main_window import PowerSurfacingMainWindow

def main():
    app = QApplication(sys.argv)
    window = PowerSurfacingMainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
