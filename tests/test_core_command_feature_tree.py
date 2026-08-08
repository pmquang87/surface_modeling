"""
Regression tests for the core Command / FeatureTree / T-Spline layers.
GUI collaborators are replaced by light fakes that mirror the real signatures
(src/gui/viewport.py, src/gui/main_window.py) - no Qt is started here.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.command import CommandManager, MeshOperationCommand
from src.core.feature_tree import Feature, FeatureTree
from src.core.halfedge_mesh import HalfEdgeMesh
from src.core.tsplines import TMesh


# ---------------------------------------------------------------- helpers

def make_grid(nx: int, ny: int) -> HalfEdgeMesh:
    """Quad grid with nx*ny faces."""
    mesh = HalfEdgeMesh()
    for i in range(nx + 1):
        for j in range(ny + 1):
            mesh.add_vertex([float(i), float(j), 0.0])

    def idx(i, j):
        return i * (ny + 1) + j

    for i in range(nx):
        for j in range(ny):
            mesh.add_face([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])
    return mesh


class FakeViewport:
    """Same call signatures as MeshViewport.update_mesh / clear."""

    def __init__(self):
        self.mesh = 'never-updated'
        self.update_calls = 0
        self.clear_calls = 0

    def update_mesh(self, mesh, name: str = 'default'):
        self.update_calls += 1
        self.mesh = mesh

    def clear(self):
        self.clear_calls += 1
        self.mesh = None


class FakeMainWindow:
    """Only exposes what PowerSurfacingMainWindow really has: current_mesh, viewport, log."""

    def __init__(self, mesh=None):
        self.current_mesh = mesh
        self.viewport = FakeViewport()
        self.messages = []

    def log(self, message):
        self.messages.append(message)


class FakeMesh:
    """Stand-in for HalfEdgeMesh in FeatureTree tests: identity tag plus copy()."""

    def __init__(self, tag=()):
        self.tag = tuple(tag)

    def copy(self):
        return FakeMesh(self.tag)


def make_evaluator(calls):
    def evaluate(feature, mesh):
        calls.append(feature.name)
        base = mesh.tag if mesh is not None else ()
        return FakeMesh(base + (feature.name,))
    return evaluate


def make_tree(names):
    """Feature tree in the state a freshly loaded document has: features, no snapshots."""
    tree = FeatureTree()
    for name in names:
        tree.features.append(Feature(name, 'test', {}))
    return tree


# ---------------------------------------------------------------- command

class TestMeshOperationCommand:

    def test_execute_snapshots_large_mesh_without_recursion(self):
        # copy.deepcopy blows the recursion limit on a cross-linked half-edge graph
        # well below 100 faces; the iterative HalfEdgeMesh.copy() must be used.
        window = FakeMainWindow(make_grid(10, 10))

        def operation():
            window.current_mesh.vertices[0].position[2] = 5.0

        cmd = MeshOperationCommand("Nudge", window, operation)
        cmd.execute()

        assert cmd.previous_mesh is not None
        assert len(cmd.previous_mesh.faces) == 100
        assert cmd.previous_mesh.vertices[0].position[2] == 0.0

    def test_execute_assigns_returned_mesh(self):
        original = make_grid(2, 2)
        replacement = make_grid(3, 3)
        window = FakeMainWindow(original)

        cmd = MeshOperationCommand("Shell", window, lambda m: replacement, original)
        cmd.execute()

        assert window.current_mesh is replacement
        assert cmd.new_mesh is not None
        assert len(cmd.new_mesh.faces) == 9

    def test_undo_restores_previous_mesh(self):
        original = make_grid(2, 2)
        window = FakeMainWindow(original)

        cmd = MeshOperationCommand("Shell", window, lambda m: make_grid(3, 3), original)
        cmd.execute()
        cmd.undo()

        assert window.current_mesh is not None
        assert len(window.current_mesh.faces) == 4
        assert window.viewport.mesh is window.current_mesh

    def test_undo_uses_supported_viewport_and_status_api(self):
        window = FakeMainWindow(make_grid(2, 2))

        cmd = MeshOperationCommand("Shell", window, lambda m: make_grid(3, 3), window.current_mesh)
        cmd.execute()
        cmd.undo()   # keep_selection= / _update_status() would blow up here

        assert window.viewport.update_calls > 0
        assert window.messages

    def test_undo_restores_empty_document(self):
        window = FakeMainWindow(None)   # nothing loaded yet

        def operation():
            window.current_mesh = make_grid(2, 2)

        cmd = MeshOperationCommand("Create Box", window, operation)
        cmd.execute()
        assert window.current_mesh is not None

        cmd.undo()

        assert window.current_mesh is None
        assert window.viewport.mesh is None

    def test_redo_reuses_cached_mesh(self):
        window = FakeMainWindow(make_grid(2, 2))
        runs = []

        def operation():
            runs.append(1)
            window.current_mesh = make_grid(3, 3)

        manager = CommandManager()
        manager.execute_command(MeshOperationCommand("Shell", window, operation))
        manager.undo()
        manager.redo()

        assert len(runs) == 1                      # operation not re-run
        assert len(window.current_mesh.faces) == 9


# ----------------------------------------------------------- feature tree

class TestFeatureTreeRebuild:

    def test_rebuild_evaluates_features_without_snapshots(self):
        calls = []
        tree = make_tree(['A', 'B', 'C'])
        tree.feature_evaluator = make_evaluator(calls)

        tree.rebuild(2)

        assert calls == ['A', 'B', 'C']
        assert tree.get_current_mesh().tag == ('A', 'B', 'C')

    def test_rebuild_resumes_from_nearest_earlier_snapshot(self):
        calls = []
        tree = make_tree(['A', 'B', 'C'])
        tree.feature_evaluator = make_evaluator(calls)
        tree.features[0].mesh_snapshot = FakeMesh(('A',))

        tree.rebuild(2)

        assert calls == ['B', 'C']
        assert tree.get_current_mesh().tag == ('A', 'B', 'C')

    def test_rebuild_skips_disabled_features(self):
        calls = []
        tree = make_tree(['A', 'B', 'C'])
        tree.feature_evaluator = make_evaluator(calls)
        tree.features[1].enabled = False

        tree.rebuild(0)

        assert calls == ['A', 'C']
        assert tree.features[1].mesh_snapshot is None
        assert tree.get_current_mesh().tag == ('A', 'C')

    def test_failed_rebuild_invalidates_stale_snapshots(self):
        calls = []
        tree = make_tree(['A', 'B', 'C'])
        tree.feature_evaluator = make_evaluator(calls)
        tree.rebuild(0)

        def failing(feature, mesh):
            if feature.name == 'B':
                raise RuntimeError("evaluator blew up")
            return make_evaluator(calls)(feature, mesh)

        tree.feature_evaluator = failing

        with pytest.raises(RuntimeError):
            tree.rebuild(1)

        assert tree.features[1].mesh_snapshot is None
        assert tree.features[2].mesh_snapshot is None
        assert tree.get_current_mesh().tag == ('A',)


class TestFeatureTreeCallbacks:

    def _wire(self, tree):
        added, removed = [], []
        tree.on_feature_added.append(added.append)
        tree.on_feature_removed.append(removed.append)
        return added, removed

    def test_undo_redo_of_add_fires_callbacks(self):
        tree = FeatureTree()
        added, removed = self._wire(tree)
        feat = Feature('A', 'test', {})

        tree.add_feature(feat)
        assert added == [feat]

        tree.undo()
        assert removed == [0]

        tree.redo()
        assert added == [feat, feat]

    def test_undo_redo_of_remove_fires_callbacks(self):
        tree = FeatureTree()
        feat = Feature('A', 'test', {})
        tree.add_feature(feat)

        added, removed = self._wire(tree)

        tree.remove_feature(0)
        assert removed == [0]

        tree.undo()
        assert added == [feat]

        tree.redo()
        assert removed == [0, 0]


# ---------------------------------------------------------------- tsplines

def build_grid_tmesh(n: int = 3) -> dict:
    """n x n regular T-mesh grid; returns {(i, j): TVertex}."""
    mesh = TMesh()
    verts = {}
    for i in range(n):
        for j in range(n):
            boundary = i in (0, n - 1) or j in (0, n - 1)
            verts[(i, j)] = mesh.add_vertex(float(i), float(j), 0.0, is_boundary=boundary)

    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                mesh.add_edge(verts[(i, j)].id, verts[(i + 1, j)].id, 'right', 'left')
            if j + 1 < n:
                mesh.add_edge(verts[(i, j)].id, verts[(i, j + 1)].id, 'up', 'down')
    return mesh, verts


class TestTVertexClassification:

    def test_regular_grid_has_no_extraordinary_vertices(self):
        mesh, verts = build_grid_tmesh(3)

        assert verts[(1, 1)].valence == 4
        assert not verts[(1, 1)].is_extraordinary()
        assert not verts[(1, 1)].is_t_junction()

        corner = verts[(0, 0)]
        assert corner.valence == 2
        assert not corner.is_extraordinary()
        assert not corner.is_t_junction()

        border = verts[(1, 0)]
        assert border.valence == 3
        assert not border.is_t_junction()      # boundary, not a T-junction
        assert not border.is_extraordinary()

    def test_interior_valence_three_is_t_junction(self):
        mesh = TMesh()
        center = mesh.add_vertex(0, 0, 0)
        left = mesh.add_vertex(-1, 0, 0, is_boundary=True)
        right = mesh.add_vertex(1, 0, 0, is_boundary=True)
        up = mesh.add_vertex(0, 1, 0, is_boundary=True)

        mesh.add_edge(center.id, left.id, 'left', 'right')
        mesh.add_edge(center.id, right.id, 'right', 'left')
        mesh.add_edge(center.id, up.id, 'up', 'down')

        assert center.is_t_junction()
        assert not center.is_extraordinary()

    def test_interior_valence_two_is_extraordinary(self):
        mesh = TMesh()
        center = mesh.add_vertex(0, 0, 0)
        left = mesh.add_vertex(-1, 0, 0, is_boundary=True)
        right = mesh.add_vertex(1, 0, 0, is_boundary=True)

        mesh.add_edge(center.id, left.id, 'left', 'right')
        mesh.add_edge(center.id, right.id, 'right', 'left')

        assert not center.is_t_junction()
        assert center.is_extraordinary()


class TestTMeshEdges:

    def test_add_edge_refuses_occupied_slot(self):
        mesh = TMesh()
        v0 = mesh.add_vertex(0, 0, 0)
        v1 = mesh.add_vertex(1, 0, 0)
        v2 = mesh.add_vertex(2, 0, 0)

        mesh.add_edge(v0.id, v1.id, 'right', 'left')
        with pytest.raises(ValueError):
            mesh.add_edge(v0.id, v2.id, 'right', 'left')

        assert v0.edges['right'].v2 is v1     # original edge not orphaned
        assert len(mesh.edges) == 1

    def test_add_edge_rejects_unknown_direction(self):
        mesh = TMesh()
        v0 = mesh.add_vertex(0, 0, 0)
        v1 = mesh.add_vertex(1, 0, 0)

        with pytest.raises(ValueError):
            mesh.add_edge(v0.id, v1.id, 'north', 'left')

        assert 'north' not in v0.edges
        assert len(mesh.edges) == 0

    def test_split_edge_rejects_unregistered_edge(self):
        mesh = TMesh()
        v0 = mesh.add_vertex(0, 0, 0)
        v1 = mesh.add_vertex(2, 0, 0)
        edge = mesh.add_edge(v0.id, v1.id, 'right', 'left')

        v0.edges['right'] = None              # simulate a corrupted direction slot

        with pytest.raises(ValueError):
            mesh.split_edge(edge.id)

        assert None not in v0.edges
        assert None not in v1.edges

    def test_split_edge_keeps_direction_slots_consistent(self):
        mesh = TMesh()
        v0 = mesh.add_vertex(0, 0, 0)
        v1 = mesh.add_vertex(2, 0, 0)
        edge = mesh.add_edge(v0.id, v1.id, 'right', 'left', 2.0)

        new_v = mesh.split_edge(edge.id, alpha=0.5)

        assert new_v.edges['left'] is not None
        assert new_v.edges['right'] is not None
        assert v0.edges['right'].get_other_vertex(v0) is new_v
        assert v1.edges['left'].get_other_vertex(v1) is new_v
        assert None not in new_v.edges
