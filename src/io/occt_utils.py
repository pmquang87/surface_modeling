"""Process-global OpenCascade state helpers shared by import, export and audit.

Two OCCT facts that bit this project (both measured on OCP 7.9; the pattern
of handling them comes from BlinkingSun/stl2step, MIT):

* STEP static parameters (``write.step.schema``, ``write.step.unit``,
  ``write.step.product.name``) do not exist until ``STEPControl_Controller::Init``
  has run. Before that, ``Interface_Static.SetCVal_s`` returns False and does
  nothing - the old exporter set the schema before constructing its first
  writer, i.e. as a silent no-op that only "worked" because it asked for the
  default. ``init_step_statics`` initialises the controller first and refuses
  to continue when a parameter is not accepted.
* OCCT's STEP reader and writer print a multi-line "Statistics on Transfer"
  banner straight to the C++ stdout. It interleaves unpredictably with
  Python's buffered stdout and breaks any "last stdout line is the machine-
  readable result" contract. ``quiet_occt`` removes that printer.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_QUIETED = False


def quiet_occt() -> bool:
    """Detach OCCT's console printer. Idempotent; returns True if detached."""
    global _QUIETED
    if _QUIETED:
        return True
    try:
        from OCP.Message import Message, Message_PrinterOStream
        Message.DefaultMessenger_s().RemovePrinters(
            Message_PrinterOStream.get_type_descriptor_s())
        _QUIETED = True
    except Exception as exc:  # pragma: no cover - depends on the OCP build
        logger.debug("could not silence OCCT printers: %s", exc)
        return False
    return True


def init_step_statics(schema: str = "AP214IS", unit: str = "MM",
                      product_name: Optional[str] = None) -> None:
    """Register the STEP statics and set schema / unit / product name.

    Raises RuntimeError when OCCT rejects a value: a rejected static is a
    check that would otherwise silently pass.
    """
    from OCP.STEPControl import STEPControl_Controller
    from OCP.Interface import Interface_Static

    STEPControl_Controller.Init_s()  # registers write.step.* / read.step.*
    values = [("write.step.schema", schema), ("write.step.unit", unit)]
    if product_name:
        values.append(("write.step.product.name", product_name))
    for key, value in values:
        if not Interface_Static.SetCVal_s(key, str(value)):
            raise RuntimeError(f"OCCT rejected STEP parameter {key}={value!r}")
