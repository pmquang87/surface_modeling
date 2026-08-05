from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
                               QComboBox, QSpinBox, QDoubleSpinBox, QDialogButtonBox, 
                               QLabel, QCheckBox)
from PySide6.QtCore import Qt

class PrimitiveDialog(QDialog):
    """Dialog for creating Sub-D primitives."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Primitive")
        self.layout = QVBoxLayout(self)
        
        self.form = QFormLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Box", "Cylinder", "Torus", "Cone", "Plane", "Sphere"])
        self.form.addRow("Type:", self.type_combo)
        
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.1, 1000.0)
        self.size_spin.setValue(10.0)
        self.form.addRow("Size:", self.size_spin)
        
        self.layout.addLayout(self.form)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def get_params(self):
        return {
            'type': self.type_combo.currentText().lower(),
            'size': self.size_spin.value()
        }

class SubdivideDialog(QDialog):
    """Dialog for subdivision settings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Subdivide Mesh")
        self.layout = QVBoxLayout(self)
        
        self.form = QFormLayout()
        self.level_spin = QSpinBox()
        self.level_spin.setRange(1, 5)
        self.level_spin.setValue(1)
        self.form.addRow("Levels:", self.level_spin)
        
        self.smooth_check = QCheckBox()
        self.smooth_check.setChecked(True)
        self.form.addRow("Smooth (Catmull-Clark):", self.smooth_check)
        
        self.layout.addLayout(self.form)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def get_params(self):
        return {
            'levels': self.level_spin.value(),
            'smooth': self.smooth_check.isChecked()
        }

class QuadWrapDialog(QDialog):
    """Dialog for Quad Wrap parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quad Wrap")
        self.layout = QVBoxLayout(self)
        self.form = QFormLayout()
        
        self.target_count = QSpinBox()
        self.target_count.setRange(100, 1000000)
        self.target_count.setValue(2000)
        self.form.addRow("Target Face Count:", self.target_count)
        
        self.align_strength = QDoubleSpinBox()
        self.align_strength.setRange(0.0, 1.0)
        self.align_strength.setValue(0.5)
        self.form.addRow("Alignment Strength:", self.align_strength)
        
        self.layout.addLayout(self.form)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

class ShrinkWrapDialog(QDialog):
    """Dialog for Shrink Wrap parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shrink Wrap")
        self.layout = QVBoxLayout(self)
        self.form = QFormLayout()
        
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 100)
        self.iterations.setValue(10)
        self.form.addRow("Iterations:", self.iterations)
        
        self.layout.addLayout(self.form)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

class ShellThickenDialog(QDialog):
    """Dialog for Shell/Thicken parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shell / Thicken")
        self.layout = QVBoxLayout(self)
        self.form = QFormLayout()
        
        self.thickness = QDoubleSpinBox()
        self.thickness.setRange(0.001, 1000.0)
        self.thickness.setValue(1.0)
        self.form.addRow("Wall Thickness:", self.thickness)
        
        self.direction = QComboBox()
        self.direction.addItems(["Inward", "Outward", "Both"])
        self.form.addRow("Direction:", self.direction)
        
        self.layout.addLayout(self.form)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

class ConvertNURBSDialog(QDialog):
    """Dialog for NURBS conversion settings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Convert to NURBS")
        self.layout = QVBoxLayout(self)
        self.form = QFormLayout()
        
        self.continuity = QComboBox()
        self.continuity.addItems(["G0 (Position)", "G1 (Tangent)", "G2 (Curvature)"])
        self.continuity.setCurrentIndex(1)
        self.form.addRow("Continuity:", self.continuity)
        
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.0001, 1.0)
        self.tolerance.setDecimals(4)
        self.tolerance.setValue(0.01)
        self.form.addRow("Tolerance:", self.tolerance)
        
        self.layout.addLayout(self.form)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

class ExportDialog(QDialog):
    """Dialog for export settings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Settings")
        self.layout = QVBoxLayout(self)
        self.form = QFormLayout()
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["STEP", "STL", "OBJ"])
        self.form.addRow("Format:", self.format_combo)
        
        self.binary_check = QCheckBox()
        self.binary_check.setChecked(True)
        self.form.addRow("Binary (STL only):", self.binary_check)
        
        self.layout.addLayout(self.form)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)
