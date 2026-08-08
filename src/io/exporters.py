import os
from typing import Any
import trimesh
import logging

from src.core.halfedge_mesh import HalfEdgeMesh

logger = logging.getLogger(__name__)

def export_stl(mesh: 'HalfEdgeMesh', filepath: str, binary: bool = True) -> None:
    """Export HalfEdgeMesh as STL file.
    
    Converts to trimesh first to handle tessellation of arbitrary polygons if needed,
    and then exports.
    """
    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh is empty, cannot export STL to {filepath}")

    try:
        t_mesh = mesh.to_trimesh()
        t_mesh.export(filepath, file_type='stl' + ('' if binary else '_ascii'))
        logger.info(f"Exported STL to {filepath}")
    except Exception as e:
        logger.error(f"Failed to export STL: {e}")
        raise

def export_obj(mesh: 'HalfEdgeMesh', filepath: str) -> None:
    """Export HalfEdgeMesh as OBJ file.
    
    Writes vertices and faces manually to preserve quad topologies,
    since trimesh might triangulate them.
    """
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
        raise

def export_step(brep_shape: Any, filepath: str) -> None:
    """Export an OCC shape as STEP file.
    
    Args:
        brep_shape: OCP TopoDS_Shape or similar
        filepath: output path
    """
    if brep_shape is None:
        raise ValueError("No B-Rep shape provided for STEP export.")

    try:
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.Interface import Interface_Static
    except ImportError:
        try:
            import cadquery as cq
            # cadquery exporter
            cq.exporters.export(brep_shape, filepath, "STEP")
            logger.info(f"Exported STEP to {filepath} using CadQuery.")
            return
        except ImportError:
            raise RuntimeError("Neither OCP nor CadQuery are available. Cannot export STEP.")

    # schema must be set BEFORE the writer is constructed — the writer's
    # model captures the protocol at creation time
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    writer = STEPControl_Writer()

    status = writer.Transfer(brep_shape, STEPControl_AsIs)
    if status != 1:
        raise RuntimeError("Failed to transfer shape for STEP export.")
    write_status = writer.Write(filepath)
    from OCP.IFSelect import IFSelect_RetDone
    if write_status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to write STEP file. Return status: {write_status}")
    logger.info(f"Exported STEP to {filepath}")
