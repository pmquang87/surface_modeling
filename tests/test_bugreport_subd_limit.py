"""Regression tests for the SubD-limit entries of the bug report written by the
automated bug hunt on commit 4e2241347de43175401df20c6722f38fa70820b4.

Covers:
  H1 (cand 2)   the OpenCL limit kernel `compute_limit_surface` used the
                one-step vertex-point rule instead of the limit position

pyopencl is not installed here, so the H1 tests stub ONLY pyopencl: the stub's
Program compiles the kernel source string OpenCLSubdivider hands it (the
repo's own CL_KERNEL_SOURCE) with gcc behind a small C shim and runs every
work item on the CPU. They skip cleanly when gcc is missing.

Limit positions are checked against the closed-form bicubic B-spline stencil
and against the CPU path, catmull_clark.evaluate_limit_surface.
"""
import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.halfedge_mesh import HalfEdgeMesh
from src.subd import gpu_compute
from src.subd.catmull_clark import (subdivide, evaluate_limit_surface,
                                    identify_regular_regions)
from src.subd.primitives import (create_box, create_cone, create_cylinder,
                                 create_plane, create_sphere)


# ---------------------------------------------------------------------------
# pyopencl stub: the repo's kernel source, compiled with gcc, run per work item
# ---------------------------------------------------------------------------

_SHIM_PRELUDE = r"""
#include <stdbool.h>
#include <stddef.h>
#ifdef _WIN32
#define SHIM_EXPORT __declspec(dllexport)
#else
#define SHIM_EXPORT __attribute__((visibility("default")))
#endif
#define __kernel static
#define __global
static size_t cl_shim_gid;
static size_t get_global_id(unsigned int dim) { (void)dim; return cl_shim_gid; }
"""

_KERNEL_RE = re.compile(r"__kernel\s+void\s+(\w+)\s*\(([^)]*)\)", re.S)


def _parse_kernels(source):
    """{kernel name: [(declaration, ctypes type), ...]} for every __kernel."""
    kernels = {}
    for name, params in _KERNEL_RE.findall(source):
        decls = []
        for p in params.split(','):
            p = ' '.join(p.split())
            if '*' in p:
                decls.append((p, ctypes.c_void_p))
            elif re.search(r'\bfloat\b', p):
                decls.append((p, ctypes.c_float))
            elif re.search(r'\bint\b', p):
                decls.append((p, ctypes.c_int))
            else:
                raise AssertionError(f"kernel {name}: unhandled parameter {p!r}")
        kernels[name] = decls
    return kernels


def _shim_source(source, kernels):
    parts = [_SHIM_PRELUDE, source]
    for name, decls in kernels.items():
        names = [re.findall(r'\w+', d)[-1] for d, _ in decls]
        parts.append(
            f"SHIM_EXPORT void shim_{name}(int global_size, {', '.join(d for d, _ in decls)}) {{\n"
            f"    for (cl_shim_gid = 0; cl_shim_gid < (size_t)global_size; ++cl_shim_gid)\n"
            f"        {name}({', '.join(names)});\n}}\n")
    return '\n'.join(parts)


def make_cl_stub(build_dir, gcc):
    cache = {}

    def compile_source(source):
        key = hashlib.sha256(source.encode()).hexdigest()[:16]
        if key in cache:
            return cache[key]
        kernels = _parse_kernels(source)
        assert kernels, "no __kernel found in the source handed to Program()"
        c_file = os.path.join(build_dir, f"cl_shim_{key}.c")
        lib_file = os.path.join(build_dir, f"cl_shim_{key}" + ('.dll' if os.name == 'nt' else '.so'))
        with open(c_file, 'w') as fh:
            fh.write(_shim_source(source, kernels))
        cmd = [gcc, '-std=c99', '-O1', '-shared', '-o', lib_file, c_file]
        cmd += ['-static-libgcc'] if os.name == 'nt' else ['-fPIC']
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"gcc failed on the repo's kernel source:\n{res.stderr}"
        lib = ctypes.CDLL(lib_file)
        cache[key] = (lib, kernels)
        return cache[key]

    class Buffer:
        def __init__(self, ctx, flags, size=0, hostbuf=None):
            if hostbuf is not None:
                self.array = np.ascontiguousarray(hostbuf).copy()
            else:
                self.array = np.zeros(int(size), dtype=np.uint8)

    class Kernel:
        def __init__(self, lib, name, decls):
            self.fn = getattr(lib, 'shim_' + name)
            self.fn.restype = None
            self.fn.argtypes = [ctypes.c_int] + [t for _, t in decls]
            self.decls = decls

        def __call__(self, queue, global_size, local_size, *args):
            assert len(args) == len(self.decls), "wrong kernel argument count"
            conv = []
            for a, (decl, t) in zip(args, self.decls):
                if t is ctypes.c_void_p:
                    assert isinstance(a, Buffer), f"{decl}: expected a Buffer"
                    conv.append(ctypes.c_void_p(a.array.ctypes.data))
                elif t is ctypes.c_float:
                    conv.append(ctypes.c_float(float(a)))
                else:
                    conv.append(ctypes.c_int(int(a)))
            self.fn(int(global_size[0]), *conv)

    class Program:
        def __init__(self, ctx, source):
            self.source = source

        def build(self):
            lib, kernels = compile_source(self.source)
            for name, decls in kernels.items():
                setattr(self, name, Kernel(lib, name, decls))
            return self

    def enqueue_copy(queue, dest, src):
        raw = src.array.view(np.uint8).reshape(-1)
        assert raw.size >= dest.nbytes, "device buffer smaller than the host array"
        np.copyto(dest.view(np.uint8).reshape(-1), raw[:dest.nbytes])

    class _Device:
        type = 'GPU'

    class _Platform:
        def get_devices(self):
            return [_Device()]

    class _Queue:
        def __init__(self, ctx):
            pass

        def finish(self):
            pass

    stub = types.ModuleType('pyopencl')
    stub.Error = RuntimeError
    stub.device_type = types.SimpleNamespace(GPU='GPU', CPU='CPU')
    stub.mem_flags = types.SimpleNamespace(READ_ONLY=1, WRITE_ONLY=2, READ_WRITE=4,
                                           COPY_HOST_PTR=8)
    stub.get_platforms = lambda: [_Platform()]
    stub.Context = lambda devices: object()
    stub.CommandQueue = _Queue
    stub.Buffer = Buffer
    stub.Program = Program
    stub.enqueue_copy = enqueue_copy
    return stub


@pytest.fixture(scope='module')
def cl_stub(tmp_path_factory):
    gcc = shutil.which('gcc')
    if gcc is None:
        pytest.skip("gcc is not on PATH; the OpenCL kernel cannot be compiled")
    return make_cl_stub(str(tmp_path_factory.mktemp('cl_shim')), gcc)


@pytest.fixture
def gpu(cl_stub, monkeypatch):
    """An OpenCLSubdivider running the repo's kernel source through the stub."""
    monkeypatch.setattr(gpu_compute, 'cl', cl_stub, raising=False)
    monkeypatch.setattr(gpu_compute, 'HAS_PYOPENCL', True)
    return gpu_compute.OpenCLSubdivider()


# ---------------------------------------------------------------------------
# mesh helpers
# ---------------------------------------------------------------------------

def mesh_arrays(mesh):
    """(vertices, flat face indices, face offsets) as OpenCLSubdivider takes them."""
    faces = [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]
    flat = np.array([i for f in faces for i in f], dtype=np.int32)
    offsets = np.cumsum([0] + [len(f) for f in faces]).astype(np.int32)
    verts = np.array([v.position for v in mesh.vertices], dtype=np.float64)
    return verts, flat, offsets


def merged(*meshes):
    """Disjoint union of several meshes (each one keeps its own topology)."""
    verts, faces = [], []
    for m in meshes:
        off = len(verts)
        verts.extend(v.position for v in m.vertices)
        faces.extend([v.index + off for v in m.get_face_vertices(f)] for f in m.faces)
    return HalfEdgeMesh.from_arrays(np.array(verts), faces)


def jittered(mesh, scale=0.03, seed=7):
    """Copy with every vertex moved by a fixed pseudo-random offset, so no
    symmetry of the primitive can hide a wrong stencil weight."""
    out = mesh.copy()
    rng = np.random.default_rng(seed)
    for v in out.vertices:
        v.position = v.position + rng.uniform(-scale, scale, 3)
    return out


def quad_grid_with_bump(n=5):
    """n x n vertices on z = 0, (n-1)^2 quads, the centre vertex lifted to z = 1."""
    mesh = HalfEdgeMesh()
    c = n // 2
    for j in range(n):
        for i in range(n):
            mesh.add_vertex([float(i), float(j), 1.0 if (i, j) == (c, c) else 0.0])
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            mesh.add_face([a, a + 1, a + 1 + n, a + n])
    return mesh


def edge_between(mesh, a, b):
    for e in mesh.edges:
        if {e.half_edge.prev.vertex.index, e.half_edge.vertex.index} == {a, b}:
            return e
    raise KeyError(f"no edge {a}-{b}")


def incident_edges(mesh, v_idx):
    return [e for e in mesh.edges
            if v_idx in (e.half_edge.prev.vertex.index, e.half_edge.vertex.index)]


def other_end(e, v_idx):
    s, t = e.half_edge.prev.vertex.index, e.half_edge.vertex.index
    return t if s == v_idx else s


def interior_valences(mesh):
    return {len(mesh.get_vertex_neighbors(v)) for v in mesh.vertices
            if not mesh.is_boundary_vertex(v)}


# box edges, by the vertex pairs of create_box's faces
BACK_LOOP = [(0, 3), (3, 2), (2, 1), (1, 0)]
FRONT_LOOP = [(4, 5), (5, 6), (6, 7), (7, 4)]


def creased_box(pairs=(), weight=1.0, all_edges=None, seed=None):
    box = create_box()
    if seed is not None:
        box = jittered(box, scale=0.08, seed=seed)
    if all_edges is not None:
        for e in box.edges:
            e.crease_weight = all_edges
    for a, b in pairs:
        edge_between(box, a, b).crease_weight = weight
    return box


def smooth_vertex_rule(mesh, v_idx):
    """Catmull-Clark interior vertex point (F + 2R + (n-3)P) / n, written out
    from the face loops independently of src."""
    P = mesh.vertices[v_idx].position
    inc = incident_edges(mesh, v_idx)
    n = len(inc)
    R = np.mean([(P + mesh.vertices[other_end(e, v_idx)].position) / 2.0 for e in inc], axis=0)
    cents = [np.mean([x.position for x in mesh.get_face_vertices(f)], axis=0)
             for f in mesh.faces
             if v_idx in [x.index for x in mesh.get_face_vertices(f)]]
    assert len(cents) == n, "fixture vertex is not an interior manifold vertex"
    return (np.mean(cents, axis=0) + 2.0 * R + (n - 3) * P) / n


def smooth_limit(mesh, v_idx):
    """Closed-form smooth limit ((n-3)P + 4*Rbar + 4*Fbar) / (n+5) (Halstead et
    al. 1993, rewritten for edge midpoints and face centroids)."""
    P = mesh.vertices[v_idx].position
    inc = incident_edges(mesh, v_idx)
    n = len(inc)
    R = np.mean([(P + mesh.vertices[other_end(e, v_idx)].position) / 2.0 for e in inc], axis=0)
    F = np.mean([np.mean([x.position for x in mesh.get_face_vertices(f)], axis=0)
                 for f in mesh.faces
                 if v_idx in [x.index for x in mesh.get_face_vertices(f)]], axis=0)
    return ((n - 3) * P + 4.0 * R + 4.0 * F) / (n + 5.0)


def deep_positions(mesh, levels):
    """Positions of the original vertices after 1..levels subdivision passes."""
    nv = len(mesh.vertices)
    out, deep = [], mesh
    for _ in range(levels):
        deep = subdivide(deep, 1)
        out.append(np.array([deep.vertices[i].position for i in range(nv)]))
    return out


# ---------------------------------------------------------------------------
# H1 (cand 2): the OpenCL limit kernel
# ---------------------------------------------------------------------------

def test_h1_gpu_limit_of_a_bump_on_a_regular_grid_is_the_bicubic_limit(gpu):
    """H1 (cand 2): on a regular quad grid the Catmull-Clark limit is the
    uniform bicubic B-spline, (16 P + 4 sum(edge nbrs) + sum(diagonals)) / 36.
    With only the centre vertex at z = 1 the limit z is 16/36 at the centre,
    4/36 at its edge neighbours and 1/36 at its diagonals; the kernel returned
    0.6875 (the one-step vertex point) at the centre."""
    mesh = quad_grid_with_bump(5)
    verts, flat, offsets = mesh_arrays(mesh)
    out = np.asarray(gpu.evaluate_limit_surface(verts, flat, offsets), dtype=np.float64)

    assert out.shape == (25, 3)
    expected_z = np.zeros(25)
    for j in range(1, 4):
        for i in range(1, 4):
            d = abs(i - 2) + abs(j - 2)
            expected_z[j * 5 + i] = {0: 16.0, 1: 4.0, 2: 1.0}[d] / 36.0
    # x and y of an interior vertex of a uniform grid are reproduced exactly
    assert np.allclose(out[:, :2], verts[:, :2], atol=1e-5)
    assert out[12, 2] == pytest.approx(16.0 / 36.0, abs=1e-6), \
        f"centre limit z = {out[12, 2]}, expected 16/36"
    assert np.allclose(out[:, 2], expected_z, atol=1e-6), \
        f"limit z\n{out[:, 2].reshape(5, 5)}\nexpected\n{expected_z.reshape(5, 5)}"
    # the CPU path is the documented reference and agrees
    cpu, _ = evaluate_limit_surface(mesh)
    assert np.allclose(cpu[:, 2], expected_z, atol=1e-12)


H1_MESHES = {
    # closed, all quads after one pass; interior valences 3 (cone base ring,
    # triangle face points), 4 (edge points), 5 and 6 (apexes, base centres)
    'cones5+6_subdivided': lambda: jittered(merged(subdivide(create_cone(segments=5), 1),
                                                   subdivide(create_cone(segments=6), 1))),
    # raw triangles plus one hexagon: face points of non-quads
    'cone6_raw': lambda: jittered(create_cone(segments=6)),
    # quads plus two octagon caps
    'cylinder_raw': lambda: jittered(create_cylinder(segments=8)),
    # triangle fans at the poles (valence 8), quads elsewhere
    'sphere_raw': lambda: jittered(create_sphere(segments=8, rings=6)),
    # open patch: boundary vertices must stay in place on both paths
    'bent_grid': lambda: jittered(create_plane(subdivisions_x=4, subdivisions_y=4), scale=0.15),
}


@pytest.mark.parametrize('name', sorted(H1_MESHES))
def test_h1_gpu_limit_matches_the_cpu_limit_everywhere(gpu, name):
    """H1 (cand 2): the kernel must give catmull_clark.evaluate_limit_surface's
    limit, ((n-3) P + 4 Rbar + 4 Fbar) / (n+5), for interior vertices of every
    valence and every face type, and leave boundary vertices where they are,
    exactly as the CPU path does."""
    mesh = H1_MESHES[name]()
    verts, flat, offsets = mesh_arrays(mesh)
    out = np.asarray(gpu.evaluate_limit_surface(verts, flat, offsets), dtype=np.float64)
    cpu, _ = evaluate_limit_surface(mesh)

    interior = np.array([not mesh.is_boundary_vertex(v) for v in mesh.vertices])
    assert interior.any()
    # Precondition: the limit really moves the interior, so a kernel that
    # copies its input through cannot pass.
    travel = np.linalg.norm(cpu[interior] - verts[interior], axis=1)
    assert travel.min() > 1e-3, f"fixture interior hardly moves ({travel.min():.2e})"

    assert out.shape == cpu.shape
    err = np.abs(out - cpu).max(axis=1)
    assert err.max() < 1e-5, (
        f"{name}: GPU limit differs from the CPU limit by up to {err.max():.3e} "
        f"at {int((err > 1e-5).sum())} of {len(err)} vertices")
    assert np.allclose(out[~interior], verts[~interior], atol=1e-6)


def test_h1_fixtures_cover_interior_valences_3_to_6():
    """H1 (cand 2) precondition: the comparison above meets interior vertices
    of valence 3, 4, 5 and 6, and at least one open boundary."""
    seen = set()
    for make in H1_MESHES.values():
        seen |= interior_valences(make())
    assert {3, 4, 5, 6} <= seen, f"interior valences covered: {sorted(seen)}"
    grid = H1_MESHES['bent_grid']()
    assert any(grid.is_boundary_vertex(v) for v in grid.vertices)
