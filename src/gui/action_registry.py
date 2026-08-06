from typing import Callable, Dict, Any, List
from dataclasses import dataclass

@dataclass
class ActionDefinition:
    name: str
    menu_path: str
    callback: Callable
    icon: str = ""
    shortcut: str = ""
    tooltip: str = ""
    requires_selection: bool = False

class ActionRegistry:
    """
    Central registry for all GUI tools and operations.
    Decouples the main window from the specific modeling tools.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ActionRegistry, cls).__new__(cls)
            cls._instance._actions = {}
        return cls._instance
        
    def __init__(self):
        # Prevent re-initialization
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._actions: Dict[str, ActionDefinition] = {}
            
    def register(self, action: ActionDefinition) -> None:
        """Registers a new action/tool in the application."""
        self._actions[action.name] = action
        
    def get_action(self, name: str) -> ActionDefinition:
        """Retrieves an action by its name."""
        return self._actions.get(name)
        
    def get_all_actions(self) -> List[ActionDefinition]:
        """Returns all registered actions."""
        return list(self._actions.values())
        
    def get_actions_by_menu(self) -> Dict[str, List[ActionDefinition]]:
        """Groups actions by their designated menu path (e.g., 'Edit Mesh', 'SubD')."""
        grouped = {}
        for action in self._actions.values():
            if action.menu_path not in grouped:
                grouped[action.menu_path] = []
            grouped[action.menu_path].append(action)
        return grouped

# Global singleton accessor
registry = ActionRegistry()

def register_action(name: str, menu_path: str, **kwargs):
    """
    Decorator for registering tools directly on their callback functions.
    """
    def decorator(func):
        action = ActionDefinition(
            name=name,
            menu_path=menu_path,
            callback=func,
            **kwargs
        )
        registry.register(action)
        return func
    return decorator
