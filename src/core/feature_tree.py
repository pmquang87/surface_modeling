from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
from src.core.halfedge_mesh import HalfEdgeMesh

class Feature:
    """
    Represents a single parametric modeling operation in the history tree.
    """
    def __init__(self, name: str, feature_type: str, parameters: Dict[str, Any]):
        self.name: str = name
        self.feature_type: str = feature_type
        self.parameters: Dict[str, Any] = parameters
        self.enabled: bool = True
        self.mesh_snapshot: Optional[HalfEdgeMesh] = None
        self.timestamp: datetime = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'feature_type': self.feature_type,
            'parameters': self.parameters,
            'enabled': self.enabled,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Feature':
        feat = cls(data['name'], data['feature_type'], data['parameters'])
        feat.enabled = data.get('enabled', True)
        if 'timestamp' in data:
            feat.timestamp = datetime.fromisoformat(data['timestamp'])
        return feat


class FeatureTree:
    """
    Parametric feature history tree for non-destructive mesh modeling.
    """
    def __init__(self):
        self.features: List[Feature] = []
        self._undo_stack: List[Tuple[str, Feature, int]] = []
        self._redo_stack: List[Tuple[str, Feature, int]] = []
        
        # Callbacks
        self.on_feature_added: List[Callable[[Feature], None]] = []
        self.on_feature_removed: List[Callable[[int], None]] = []
        self.on_rebuild_complete: List[Callable[[Optional[HalfEdgeMesh]], None]] = []
        
        # Feature evaluation function provided by the engine
        self.feature_evaluator: Optional[Callable[[Feature, Optional[HalfEdgeMesh]], HalfEdgeMesh]] = None

    def _notify_added(self, feature: Feature) -> None:
        for cb in self.on_feature_added:
            cb(feature)

    def _notify_removed(self, index: int) -> None:
        for cb in self.on_feature_removed:
            cb(index)

    def add_feature(self, feature: Feature) -> None:
        self.features.append(feature)
        idx = len(self.features) - 1
        self._undo_stack.append(('ADD', feature, idx))
        self._redo_stack.clear()

        self._notify_added(feature)

        self.rebuild(idx)

    def remove_feature(self, index: int) -> None:
        if 0 <= index < len(self.features):
            feat = self.features.pop(index)
            self._undo_stack.append(('REMOVE', feat, index))
            self._redo_stack.clear()

            self._notify_removed(index)

            self.rebuild(max(0, index - 1))

    def move_feature(self, from_idx: int, to_idx: int) -> None:
        if 0 <= from_idx < len(self.features) and 0 <= to_idx < len(self.features):
            feat = self.features.pop(from_idx)
            self.features.insert(to_idx, feat)
            
            self._undo_stack.append(('MOVE', feat, (to_idx, from_idx)))
            self._redo_stack.clear()
            
            self.rebuild(min(from_idx, to_idx))

    def toggle_feature(self, index: int) -> None:
        if 0 <= index < len(self.features):
            self.features[index].enabled = not self.features[index].enabled
            self._undo_stack.append(('TOGGLE', self.features[index], index))
            self._redo_stack.clear()
            self.rebuild(index)

    def rebuild(self, from_index: int = 0) -> Optional[HalfEdgeMesh]:
        if not self.feature_evaluator:
            return None
            
        start = max(0, min(from_index, len(self.features)))

        # Walk back to the newest usable snapshot. An enabled feature without a
        # snapshot (a freshly loaded tree, an invalidated one) has never been
        # evaluated, so evaluation has to restart there instead of skipping it.
        current_mesh = None
        for i in range(start - 1, -1, -1):
            feat = self.features[i]
            if not feat.enabled:
                continue
            if feat.mesh_snapshot is not None:
                current_mesh = feat.mesh_snapshot.copy()
                break
            start = i

        for i in range(start, len(self.features)):
            feat = self.features[i]
            if not feat.enabled:
                feat.mesh_snapshot = None
                continue
            try:
                current_mesh = self.feature_evaluator(feat, current_mesh)
            except Exception:
                # Snapshots from this feature onwards describe the pre-edit
                # geometry; leaving them would let get_current_mesh() report
                # stale geometry as current.
                for stale in self.features[i:]:
                    stale.mesh_snapshot = None
                raise
            feat.mesh_snapshot = current_mesh.copy() if current_mesh else None

        for cb in self.on_rebuild_complete:
            cb(self.get_current_mesh())
            
        return self.get_current_mesh()

    def get_current_mesh(self) -> Optional[HalfEdgeMesh]:
        for i in range(len(self.features) - 1, -1, -1):
            if self.features[i].enabled and self.features[i].mesh_snapshot:
                return self.features[i].mesh_snapshot.copy()
        return None

    def undo(self) -> None:
        if not self._undo_stack: return
        action, feat, idx = self._undo_stack.pop()
        self._redo_stack.append((action, feat, idx))
        
        if action == 'ADD':
            self.features.pop()
            self._notify_removed(idx)
            self.rebuild(idx)
        elif action == 'REMOVE':
            self.features.insert(idx, feat)
            self._notify_added(feat)
            self.rebuild(idx)
        elif action == 'MOVE':
            to_idx, from_idx = idx
            moved = self.features.pop(to_idx)
            self.features.insert(from_idx, moved)
            self.rebuild(min(from_idx, to_idx))
        elif action == 'TOGGLE':
            self.features[idx].enabled = not self.features[idx].enabled
            self.rebuild(idx)

    def redo(self) -> None:
        if not self._redo_stack: return
        action, feat, idx = self._redo_stack.pop()
        self._undo_stack.append((action, feat, idx))
        
        if action == 'ADD':
            self.features.append(feat)
            self._notify_added(feat)
            self.rebuild(idx)
        elif action == 'REMOVE':
            self.features.pop(idx)
            self._notify_removed(idx)
            self.rebuild(max(0, idx - 1))
        elif action == 'MOVE':
            to_idx, from_idx = idx
            moved = self.features.pop(from_idx)
            self.features.insert(to_idx, moved)
            self.rebuild(min(from_idx, to_idx))
        elif action == 'TOGGLE':
            self.features[idx].enabled = not self.features[idx].enabled
            self.rebuild(idx)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'features': [f.to_dict() for f in self.features]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureTree':
        tree = cls()
        for f_data in data.get('features', []):
            tree.features.append(Feature.from_dict(f_data))
        return tree
