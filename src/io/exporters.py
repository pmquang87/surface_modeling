import os
from pathlib import Path
from typing import Any, Optional
import trimesh
import logging

from src.core.halfedge_mesh import HalfEdgeMesh

logger = logging.getLogger(__name__)


def _snapshot(filepath: str):
    """(exists, size, mtime_ns) of a path, to tell later whether we touched it."""
    try:
        st = os.stat(filepath)
        return True, st.st_size, st.st_mtime_ns
    except FileNotFoundError:
        return False, None, None


def _remove_if_written_by_us(filepath: str, before) -> None:
    """Delete ``filepath`` if this call created or modified it.

    A failed export must not leave a truncated file that the next audit or a
    CAD import reads as real (pattern from stl2step's removeQuietly). A file
    that existed before and was never touched is left alone.
    """
    after = _snapshot(filepath)
    if not after[0]:
        return
    if not before[0] or after[1:] != before[1:]:
        try:
            os.remove(filepath)
        except OSError as exc:  # pragma: no cover
            logger.warning("could not remove partial file %s: %s", filepath, exc)


def export_stl(mesh: 'HalfEdgeMesh', filepath: str, binary: bool = True) -> int:
    """Export HalfEdgeMesh as STL file and return the triangle count WRITTEN.

    Converts to trimesh first to handle tessellation of arbitrary polygons if
    needed. For binary STL the count is derived from the file itself
    (size == 84 + 50 * n), so a truncated write raises instead of passing.
    """
    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh is empty, cannot export STL to {filepath}")

    before = _snapshot(filepath)
    try:
        t_mesh = mesh.to_trimesh()
        t_mesh.export(filepath, file_type='stl' + ('' if binary else '_ascii'))
        n_expected = len(t_mesh.faces)
        if binary:
            size = os.path.getsize(filepath)
            if size < 84 or (size - 84) % 50 != 0:
                raise RuntimeError(
                    f"binary STL {filepath} has {size} bytes, not 84 + 50*n")
            n_written = (size - 84) // 50
        else:
            with open(filepath, 'r', encoding='ascii', errors='replace') as f:
                n_written = sum(1 for line in f if line.lstrip().startswith('facet normal'))
        if n_written != n_expected:
            raise RuntimeError(
                f"STL {filepath} holds {n_written} triangles, expected {n_expected}")
        logger.info(f"Exported STL to {filepath} ({n_written} triangles)")
        return n_written
    except Exception as e:
        logger.error(f"Failed to export STL: {e}")
        _remove_if_written_by_us(filepath, before)
        raise


def export_obj(mesh: 'HalfEdgeMesh', filepath: str) -> None:
    """Export HalfEdgeMesh as OBJ file.

    Writes vertices and faces manually to preserve quad topologies,
    since trimesh might triangulate them.
    """
    before = _snapshot(filepath)
    try:
        with open(filepath, 'w') as f:
            for v in mesh.vertices:
                f.write(f"v {v.position[0]} {v.position[1]} {v.position[2]}\n")

            for face in mesh.faces:
                f.write("f")
                for v in mesh.get_face_vertices(face):
                    f.write(f" {v.index + 1}")
                f.write("\n")
        logger.info(f"Exported OBJ to {filepath}")
    except Exception as e:
        logger.error(f"Failed to export OBJ: {e}")
        _remove_if_written_by_us(filepath, before)
        raise


def _make_step_writer():
    """Construct the OCCT STEP writer (separate so tests can substitute it)."""
    from OCP.STEPControl import STEPControl_Writer
    return STEPControl_Writer()


def export_step(brep_shape: Any, filepath: str, schema: str = "AP214IS",
                product_name: Optional[str] = None) -> None:
    """Export an OCC shape as STEP file.

    Args:
        brep_shape: OCP TopoDS_Shape or similar
        filepath: output path
        schema: STEP application protocol ("AP203", "AP214IS", "AP242DIS")
        product_name: PRODUCT name written into the file (SolidWorks shows it
            in the feature tree); defaults to the file stem.

    The STEP statics (schema, unit, product name) are process-global OCCT
    state and only exist once the STEP controller is initialised, so they are
    set through ``init_step_statics`` on every call and checked. Output units
    are always millimetres. A failed write removes the partial file.
    """
    if brep_shape is None:
        raise ValueError("No B-Rep shape provided for STEP export.")

    try:
        from OCP.STEPControl import STEPControl_AsIs
        from OCP.IFSelect import IFSelect_RetDone
    except ImportError:
        try:
            import cadquery as cq
            # cadquery exporter
            cq.exporters.export(brep_shape, filepath, "STEP")
            logger.info(f"Exported STEP to {filepath} using CadQuery.")
            return
        except ImportError:
            raise RuntimeError("Neither OCP nor CadQuery are available. Cannot export STEP.")

    from src.io.occt_utils import quiet_occt, init_step_statics

    quiet_occt()
    init_step_statics(schema=schema, unit="MM",
                      product_name=product_name or Path(filepath).stem)

    before = _snapshot(filepath)
    try:
        writer = _make_step_writer()
        status = writer.Transfer(brep_shape, STEPControl_AsIs)
        if status != 1:
            raise RuntimeError("Failed to transfer shape for STEP export.")
        write_status = writer.Write(filepath)
        if write_status != IFSelect_RetDone:
            raise RuntimeError(f"Failed to write STEP file. Return status: {write_status}")
    except Exception:
        _remove_if_written_by_us(filepath, before)
        raise
    logger.info(f"Exported STEP to {filepath}")
