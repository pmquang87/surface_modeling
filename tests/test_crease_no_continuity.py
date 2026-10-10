"""A creased cage edge gets no cross-boundary continuity rows.

An edge with ``crease_weight > 0`` is an intended sharp feature of the SubD
cage. ``G3Fitter.fit_surface`` learns adjacency only from each quad's
``neighbors`` list, which ``SubDToNURBSConverter.generate_patches`` builds
from the twin faces; before this fix a creased edge was listed like any
other, so G1..G3 added smoothness rows across it and rounded the crease off.
Now the converter lists no neighbour (-1) across a creased edge - any weight
above 0, fractional ones included, counts as sharp for continuity.

Observable used here, independent of the dense-sample grid: the fitter's
least-squares matrix. Data rows involve the variables of one patch only, so
a row with non-zeros in the variable blocks of two different patches is a
continuity row between them.
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import scipy.sparse

from src.nurbs.converter import SubDToNURBSConverter
from src.subd.primitives import create_box

LEVELS = ['G1', 'G2', 'G3']


def _coupling_rows(level, mesh, monkeypatch):
    """{(k, n): number of matrix rows coupling patches k < n} of the fit the
    converter runs for ``level`` on ``mesh`` (no-reference path)."""
    captured = []
    real = scipy.sparse.csr_matrix

    def recording(*args, **kwargs):
        m = real(*args, **kwargs)
        captured.append(m)
        return m

    monkeypatch.setattr(scipy.sparse, 'csr_matrix', recording)
    with contextlib.redirect_stdout(io.StringIO()):
        patches = SubDToNURBSConverter(continuity=level).generate_patches(mesh)
    monkeypatch.setattr(scipy.sparse, 'csr_matrix', real)
    n_vars = 16 * len(patches)
    mats = [m for m in captured if m.shape[1] == n_vars]
    assert len(mats) == 1, [m.shape for m in captured]
    A = mats[0].tocsr()
    pairs = {}
    for r in range(A.shape[0]):
        cols = A.indices[A.indptr[r]:A.indptr[r + 1]]
        vals = A.data[A.indptr[r]:A.indptr[r + 1]]
        blocks = sorted({int(c) // 16 for c, v in zip(cols, vals) if v != 0.0})
        assert len(blocks) <= 2, blocks
        if len(blocks) == 2:
            pairs[tuple(blocks)] = pairs.get(tuple(blocks), 0) + 1
    return pairs


def _adjacent_pairs(mesh):
    """{edge index: (k, n)} for the box's quad faces (patch k = face k: every
    face of create_box is a quad, and generate_patches keeps face order)."""
    out = {}
    for e in mesh.edges:
        he = e.half_edge
        a, b = he.face.index, he.twin.face.index
        out[e.index] = tuple(sorted((a, b)))
    return out


@pytest.mark.parametrize("level", LEVELS)
def test_creased_edge_gets_no_continuity_rows(level, monkeypatch):
    """One edge of the unit box creased (weight 1.0), one at a fractional
    0.3: neither patch pair is coupled by any row, while every one of the
    other ten adjacent pairs is (control: the checker sees rows where they
    exist). Fails on 715f34b: both creased pairs had 4 rows each."""
    box = create_box()
    assert all(len(box.get_face_vertices(f)) == 4 for f in box.faces)
    pairs = _adjacent_pairs(box)
    assert len(pairs) == 12 and len(set(pairs.values())) == 12
    sharp, soft = box.edges[0], box.edges[5]
    sharp.crease_weight, soft.crease_weight = 1.0, 0.3

    rows = _coupling_rows(level, box, monkeypatch)

    assert rows.get(pairs[sharp.index], 0) == 0, (level, rows)
    assert rows.get(pairs[soft.index], 0) == 0, (level, rows)
    controls = [p for i, p in pairs.items() if i not in (sharp.index, soft.index)]
    assert len(controls) == 10
    for p in controls:
        assert rows.get(p, 0) > 0, (level, p, rows)
    # nothing else is coupled: only cage-adjacent pairs get rows
    assert set(rows) <= set(controls), (level, rows)


@pytest.mark.parametrize("level", LEVELS)
def test_uncreased_box_couples_every_adjacent_pair(level, monkeypatch):
    """Control for the test above with no crease at all: all 12 adjacent
    pairs get continuity rows, so 'no rows' above is caused by the crease."""
    box = create_box()
    pairs = _adjacent_pairs(box)
    rows = _coupling_rows(level, box, monkeypatch)
    assert set(rows) == set(pairs.values()), (level, rows)
    assert min(rows.values()) > 0


@pytest.mark.parametrize("level", LEVELS)
def test_fully_creased_box_has_no_continuity_rows(level, monkeypatch):
    """Every edge creased (the sharp cube of the GUI): no row couples any two
    patches, so the fit is the data fit alone at every level."""
    box = create_box()
    for e in box.edges:
        e.crease_weight = 1.0
    rows = _coupling_rows(level, box, monkeypatch)
    assert rows == {}, (level, rows)
