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
        self._has_run = False

    @staticmethod
    def _snapshot(mesh):
        """Copy a mesh for the history.

        HalfEdgeMesh is a fully cross-linked graph, so copy.deepcopy recurses
        along the half-edge chain and dies well below 100 faces. The mesh
        provides an iterative copy() instead.
        """
        return mesh.copy() if mesh is not None else None

    def _restore(self, mesh, message: str) -> None:
        """Put a stored state back into the document (None means empty document)."""
        self.main_window.current_mesh = self._snapshot(mesh)
        self._notify(message)

    def _notify(self, message: str) -> None:
        """Push the current state to the GUI, using only API the window really has."""
        viewport = getattr(self.main_window, 'viewport', None)
        if viewport is not None:
            mesh = self.main_window.current_mesh
            if mesh is None and hasattr(viewport, 'clear'):
                viewport.clear()
            elif hasattr(viewport, 'update_mesh'):
                viewport.update_mesh(mesh)

        log = getattr(self.main_window, 'log', None)
        if callable(log):
            log(message)

    def execute(self) -> None:
        # Save previous state (None is a valid state: an empty document)
        self.previous_mesh = self._snapshot(self.main_window.current_mesh)

        if not self._has_run:
            # Operations may either mutate the mesh in place or return a new one
            result = self.operation_func(*self.args, **self.kwargs)
            if result is not None:
                self.main_window.current_mesh = result
            self.new_mesh = self._snapshot(self.main_window.current_mesh)
            self._has_run = True
            self._notify(f"{self.name}: applied")
        else:
            # Re-applying (Redo)
            self._restore(self.new_mesh, f"{self.name}: redone")

    def undo(self) -> None:
        self._restore(self.previous_mesh, f"{self.name}: undone")

