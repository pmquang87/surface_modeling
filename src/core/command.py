from abc import ABC, abstractmethod
from typing import List, Optional

class Command(ABC):
    """
    Abstract base class for all operations that modify the document/mesh state.
    Implements the Command pattern to support Undo/Redo functionality.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self) -> None:
        """Executes the command and applies changes."""
        pass

    @abstractmethod
    def undo(self) -> None:
        """Reverts the changes made by execute()."""
        pass


class CommandManager:
    """
    Manages the execution and history of Commands for Undo/Redo.
    """
    def __init__(self, max_history: int = 50):
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []
        self.max_history = max_history

    def execute_command(self, command: Command) -> None:
        """Executes a command and adds it to the undo stack."""
        command.execute()
        self._undo_stack.append(command)
        
        # Clear the redo stack because a new action branches the history
        self._redo_stack.clear()
        
        # Enforce history limit
        if len(self._undo_stack) > self.max_history:
            self._undo_stack.pop(0)

    def undo(self) -> Optional[str]:
        """Undoes the last command and moves it to the redo stack."""
        if not self._undo_stack:
            return None
            
        command = self._undo_stack.pop()
        command.undo()
        self._redo_stack.append(command)
        return command.name

    def redo(self) -> Optional[str]:
        """Redoes the last undone command and moves it back to the undo stack."""
        if not self._redo_stack:
            return None
            
        command = self._redo_stack.pop()
        command.execute()
        self._undo_stack.append(command)
        return command.name

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0


class MeshOperationCommand(Command):
    """
    A command that saves the state of a HalfEdgeMesh before an operation
    and restores it on undo.
    """
    def __init__(self, name: str, main_window, operation_func, *args, **kwargs):
        super().__init__(name)
        self.main_window = main_window
        self.operation_func = operation_func
        self.args = args
        self.kwargs = kwargs
        
        # State storage
        self.previous_mesh = None
        self.new_mesh = None
        
    def execute(self) -> None:
        import copy
        # Save previous state
        if self.main_window.current_mesh:
            self.previous_mesh = copy.deepcopy(self.main_window.current_mesh)
            
        # Execute the actual operation if we haven't already cached a new_mesh
        if self.new_mesh is None:
            self.operation_func(*self.args, **self.kwargs)
            if self.main_window.current_mesh:
                self.new_mesh = copy.deepcopy(self.main_window.current_mesh)
        else:
            # Re-applying (Redo)
            self.main_window.current_mesh = copy.deepcopy(self.new_mesh)
            self.main_window.viewport.update_mesh(self.main_window.current_mesh, keep_selection=False)
            self.main_window._update_status()

    def undo(self) -> None:
        import copy
        if self.previous_mesh:
            self.main_window.current_mesh = copy.deepcopy(self.previous_mesh)
            self.main_window.viewport.update_mesh(self.main_window.current_mesh, keep_selection=False)
            self.main_window._update_status()

