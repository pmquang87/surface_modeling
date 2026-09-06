"""Writer-boundary hygiene for export_step / export_stl.

Lessons taken from BlinkingSun/stl2step (MIT): STEP static parameters are
process-global OCCT state and do not even exist until a STEP controller has
been initialised, so setting them must be checked, not assumed; a failed
write must not leave a partial file behind; and the writer's own console
banner must not reach stdout.
"""
import hashlib
import os
import re
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")


def _file_schema(path):
    text = open(path, encoding="utf-8", errors="replace").read(4000)
    m = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", text)
    return m.group(1) if m else None


@needs_ocp
@pytest.mark.parametrize("schema,expected", [
    ("AP203", "CONFIG_CONTROL_DESIGN"),
    ("AP214IS", "AUTOMOTIVE_DESIGN"),
])
def test_export_step_schema_lands_in_a_fresh_process(schema, expected):
    """Run in a FRESH interpreter so export_step is the first STEP operation.

    Regression: Interface_Static.SetCVal_s("write.step.schema", ...) returns
    False and does nothing until STEPControl_Controller::Init has run, which
    the old code only triggered AFTER setting the value. AP214IS masked the
    bug because it is also OCCT's default; AP203 exposes it.
    """
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "box.step")
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox\n"
            "from src.io.exporters import export_step\n"
            "export_step(BRepPrimAPI_MakeBox(10,20,30).Shape(), %r, schema=%r)\n"
        ) % (REPO, out, schema)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, cwd=REPO, timeout=300)
        assert proc.returncode == 0, proc.stderr
        # AP214 is written as "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }"
        assert (_file_schema(out) or "").startswith(expected)
        # the OCCT transfer banner must not be on stdout
        assert "Statistics on Transfer" not in proc.stdout
        assert proc.stdout.strip() == ""


@needs_ocp
def test_export_step_product_name_and_mm_unit_land_in_file():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from src.io.exporters import export_step
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "widget_7.step")
        export_step(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), out)
        text = open(out, encoding="utf-8", errors="replace").read()
        # OCCT appends a per-process counter: PRODUCT('widget_7 1', ...)
        assert re.search(r"PRODUCT\s*\(\s*'widget_7( \d+)?'", text), "product name not written"
        assert ".MILLI." in text and ".METRE." in text
        out2 = os.path.join(d, "other.step")
        export_step(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), out2, product_name="Gegenkoerper")
        assert re.search(r"PRODUCT\s*\(\s*'Gegenkoerper( \d+)?'",
                         open(out2, encoding="utf-8").read())


@needs_ocp
def test_export_step_removes_partial_file_on_failure(monkeypatch):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    import src.io.exporters as exporters

    class _FailingWriter:
        def __init__(self, path):
            self._path = path

        def Transfer(self, shape, mode):
            return 1

        def Write(self, path):
            with open(path, "w") as f:
                f.write("ISO-10303-21;\nHEADER;\n")  # truncated garbage
            return 99  # not IFSelect_RetDone

    monkeypatch.setattr(exporters, "_make_step_writer",
                        lambda: _FailingWriter(None))
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "box.step")
        with pytest.raises(RuntimeError):
            exporters.export_step(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), out)
        assert not os.path.exists(out), "a partial STEP file was left behind"


@needs_ocp
def test_export_step_keeps_a_preexisting_file_when_nothing_was_written(monkeypatch):
    """Failure BEFORE the writer touches the disk must not delete the user's file."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    import src.io.exporters as exporters

    class _TransferFails:
        def Transfer(self, shape, mode):
            return 0

        def Write(self, path):  # pragma: no cover - never reached
            raise AssertionError("Write must not be called")

    monkeypatch.setattr(exporters, "_make_step_writer", lambda: _TransferFails())
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "keep.step")
        marker = b"MARKER-ORIGINAL-CONTENT\n"
        with open(out, "wb") as f:
            f.write(marker)
        before = hashlib.sha256(marker).hexdigest()
        with pytest.raises(RuntimeError):
            exporters.export_step(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), out)
        assert os.path.exists(out)
        assert hashlib.sha256(open(out, "rb").read()).hexdigest() == before


def test_export_stl_binary_size_is_verified_and_count_returned():
    import trimesh
    from src.core.halfedge_mesh import HalfEdgeMesh
    from src.io.exporters import export_stl
    box = trimesh.creation.box(extents=(1, 1, 1))
    he = HalfEdgeMesh.from_trimesh(box)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "box.stl")
        n = export_stl(he, out, binary=True)
        assert n == 12
        assert os.path.getsize(out) == 84 + 50 * 12


def test_export_stl_truncated_write_is_detected(monkeypatch):
    import trimesh
    from src.core.halfedge_mesh import HalfEdgeMesh
    import src.io.exporters as exporters
    box = trimesh.creation.box(extents=(1, 1, 1))
    he = HalfEdgeMesh.from_trimesh(box)

    real_export = trimesh.Trimesh.export

    def truncated(self, file_obj=None, file_type=None, **kw):
        real_export(self, file_obj, file_type=file_type, **kw)
        with open(file_obj, "r+b") as f:
            f.truncate(84 + 50 * 12 - 7)

    monkeypatch.setattr(trimesh.Trimesh, "export", truncated)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "box.stl")
        with pytest.raises(RuntimeError):
            exporters.export_stl(he, out, binary=True)
        assert not os.path.exists(out), "a truncated STL was left behind"
