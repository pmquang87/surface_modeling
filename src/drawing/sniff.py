"""Detect the input kind by content, before any geometry library is imported.

Rules (sandbox probe item 6, all verified on this machine):

* STEP: the text (after an optional UTF-8 BOM and whitespace) starts with
  ``ISO-10303-21``.
* Binary STL: file size == 84 + 50 * n where n is the little-endian uint32
  at byte 80. This rule wins even when the 80-byte header starts with
  ``solid`` (exporters do that; a header overwritten with ``solid evil``
  still sniffed as binary).
* ASCII STL: starts with ``solid`` and has at least one ``facet normal``
  line; the triangle count is the exact line count (measured: 93 MB /
  327,680 facets scanned in 0.25 s).
* Empty ASCII STL: starts with ``solid``, has an ``endsolid`` line and no
  facet -> ``stl_ascii`` with ``triangles == 0`` and ``empty == True``.
* Truncated binary STL: the header declares n triangles, the file is
  shorter than 84 + 50 n, at least one complete record and at least half
  of the declared data are present, and the first records hold finite
  floats of plausible size -> ``stl_binary_truncated`` with ``triangles``
  = the declared n. Random bytes almost never pass both tests (12 random
  float32 values below 1e9 in magnitude: ~0.3 %, times the length test).
* Anything else -> ``unknown``: 0-byte files, random bytes. This matters
  because ``trimesh.load`` silently returns 0 faces for a truncated binary
  STL instead of raising. ``unreadable_reason`` turns the two new kinds into
  the error text the CLI and pipeline print.

Only the first 4 kB are read for the kind checks; the whole file is read
only for the SHA-256 and the ASCII facet count, and neither happens when
the file is above ``max_bytes`` (pass it from the parent process). The
facet scan reads lines of at most 64 kB, so memory stays bounded even for
a "solid..." file without newlines. No numpy, no trimesh.
"""
from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from typing import Optional

HEAD_BYTES = 4096
_CHUNK = 1 << 20
_MAX_LINE = 1 << 16
_STEP_MAGIC = b"ISO-10303-21"
_EXT_FAMILY = {".step": "step", ".stp": "step", ".p21": "step", ".stl": "stl"}


_PLAUSIBLE_ABS = 1e9               # |float| bound for the truncated-binary record check
_CHECK_RECORDS = 16


@dataclass
class SniffResult:
    # "step" | "stl_binary" | "stl_ascii" | "stl_binary_truncated" | "unknown"
    kind: str
    size_bytes: int
    triangles: Optional[int]       # declared/scanned for STL, None for STEP
    sha256: str
    extension: str                 # lower-case, e.g. ".stp"
    extension_mismatch: bool
    empty: bool = False            # stl_ascii with solid/endsolid and no facet


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_ascii(path: str):
    """(facet count, has an ``endsolid`` line) of an ASCII STL."""
    # readline with a length limit: a "solid..." file without newlines must
    # not be loaded into memory as one giant line
    n = 0
    end = False
    with open(path, "rb") as fh:
        while True:
            line = fh.readline(_MAX_LINE)
            if not line:
                break
            s = line.lstrip()
            if s.startswith(b"facet normal"):
                n += 1
            elif s[:8].lower() == b"endsolid":
                end = True
    return n, end


def _count_facets(path: str) -> int:
    return _scan_ascii(path)[0]


def _truncated_binary(head: bytes, size: int) -> Optional[int]:
    """Declared triangle count when ``head`` looks like a truncated binary STL."""
    if size < 84 + 50 or len(head) < 84:
        return None
    n = int(struct.unpack_from("<I", head, 80)[0])
    full = (size - 84) // 50                           # complete records present
    if n <= 0 or 84 + 50 * n <= size or full < 1 or 2 * full < n:
        return None
    k = min(full, _CHECK_RECORDS, (len(head) - 84) // 50)
    if k < 1:
        return None
    for i in range(k):
        vals = struct.unpack_from("<12f", head, 84 + 50 * i)
        for v in vals:
            if not (v == v) or abs(v) > _PLAUSIBLE_ABS:   # NaN or implausible
                return None
    return n


def _family(kind: str) -> Optional[str]:
    if kind == "step":
        return "step"
    if kind.startswith("stl"):
        return "stl"
    return None


def sniff(path: str, compute_hash: bool = True, max_bytes: Optional[int] = None) -> SniffResult:
    """Classify ``path`` by content. Raises ``OSError`` if it cannot be read.

    ``max_bytes``: when the file is larger, only the 4 kB head is read. The
    kind comes from the head alone (an ASCII STL is ``stl_ascii`` if the
    head holds a ``facet normal`` line), ``triangles`` is the binary header
    count or None, and ``sha256`` is "" (no full-file pass). Callers in the
    unsandboxed parent pass ``int(max_input_mb * 2**20)`` so an oversized
    input is never hashed or scanned before ``check_limits`` rejects it.
    """
    size = os.path.getsize(path)
    oversized = max_bytes is not None and size > int(max_bytes)
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as fh:
        head = fh.read(HEAD_BYTES)
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n")

    kind, triangles, empty = "unknown", None, False
    if stripped.startswith(_STEP_MAGIC):
        kind = "step"
    elif size >= 84 and len(head) >= 84 and 84 + 50 * struct.unpack_from("<I", head, 80)[0] == size:
        kind, triangles = "stl_binary", int(struct.unpack_from("<I", head, 80)[0])
    else:
        if stripped[:5].lower() == b"solid":
            if oversized:
                if b"facet normal" in head:
                    kind = "stl_ascii"
            else:
                n, has_end = _scan_ascii(path)
                if n > 0:
                    kind, triangles = "stl_ascii", n
                elif has_end:
                    kind, triangles, empty = "stl_ascii", 0, True
        if kind == "unknown":
            # after the ASCII test: an ASCII file's bytes 80..83 are text and
            # declare a huge count, but its records fail the float check
            declared = _truncated_binary(head, size)
            if declared is not None:
                kind, triangles = "stl_binary_truncated", declared

    fam = _family(kind)
    mismatch = fam is not None and _EXT_FAMILY.get(ext) != fam
    digest = _sha256(path) if (compute_hash and not oversized) else ""
    return SniffResult(kind=kind, size_bytes=int(size), triangles=triangles, sha256=digest,
                       extension=ext, extension_mismatch=bool(mismatch), empty=bool(empty))


def unreadable_reason(s: SniffResult) -> Optional[str]:
    """Error text for an input that is recognised but holds no usable mesh, else None.

    * empty ASCII STL -> ``"empty STL (0 facets)"``
    * ``stl_binary_truncated`` -> ``"binary STL truncated: header declares N
      triangles (X bytes), file is Y bytes"``
    * ``unknown`` -> the generic ``"unsupported or corrupt input (neither
      STEP nor STL by content)"``
    """
    if s.kind == "stl_ascii" and (s.empty or s.triangles == 0):
        return "empty STL (0 facets)"
    if s.kind == "stl_binary_truncated":
        n = int(s.triangles or 0)
        return (f"binary STL truncated: header declares {n} triangles "
                f"({84 + 50 * n} bytes), file is {int(s.size_bytes)} bytes")
    if s.kind == "unknown":
        return "unsupported or corrupt input (neither STEP nor STL by content)"
    return None


def check_limits(s: SniffResult, max_input_mb: float, max_triangles: int) -> Optional[str]:
    """Error text when the input breaks a cap, else None."""
    cap = float(max_input_mb) * 2 ** 20
    if s.size_bytes > cap:
        return (f"input is {s.size_bytes / 2 ** 20:.1f} MB, above the "
                f"{float(max_input_mb):g} MB limit (max_input_mb)")
    if s.triangles is not None and s.triangles > int(max_triangles):
        return (f"input has {s.triangles} triangles, above the "
                f"{int(max_triangles)} limit (max_triangles)")
    return None
