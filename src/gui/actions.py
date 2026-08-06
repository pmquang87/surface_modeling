from src.gui.action_registry import register_action

# This file will eventually hold all the @register_action decorated functions.
# For now, we will slowly migrate functions from main_window.py here.

def init_actions(main_window):
    """
    Called by main_window to register all actions.
    Passes a reference to the main_window for the commands to manipulate state.
    """
    
    @register_action("Shell / Thicken", "Operations", shortcut="Alt+S")
    def action_shell():
        if not main_window.current_mesh:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(main_window, "Shell/Thicken", "No mesh loaded.")
            return
            
        from src.gui.dialogs import ShellThickenDialog
        from src.operations.shell_thicken import shell_solid
        
        # We convert the Modal Dialog to a Property Manager widget
        dlg = ShellThickenDialog(main_window)
        # Hack to extract the layout from the dialog and place it in the property panel
        widget = dlg.findChild(type(dlg)) # Just a container
        if not widget:
            widget = dlg # Use dialog itself but remove window flags
            widget.setWindowFlags(widget.windowFlags() & ~main_window.windowFlags())
        
        def apply():
            thickness = dlg.spin_thick.value()
            direction = dlg.combo_dir.currentText().lower()
            
            from src.core.command import MeshOperationCommand
            cmd = MeshOperationCommand("Shell", main_window, shell_solid, main_window.current_mesh, thickness=thickness, direction=direction)
            if main_window.command_manager:
                main_window.command_manager.execute_command(cmd)
            main_window.properties_panel.clear()
            
        def cancel():
            main_window.properties_panel.clear()
            
        main_window.properties_panel.set_tool_ui("Shell / Thicken", widget, apply, cancel)
        
    @register_action("Undo", "Edit", shortcut="Ctrl+Z")
    def action_undo():
        if main_window.command_manager and main_window.command_manager.can_undo():
            name = main_window.command_manager.undo()
            main_window.log(f"Undo: {name}")
            
    @register_action("Redo", "Edit", shortcut="Ctrl+Y")
    def action_redo():
        if main_window.command_manager and main_window.command_manager.can_redo():
            name = main_window.command_manager.redo()
            main_window.log(f"Redo: {name}")
