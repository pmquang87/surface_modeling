#!/usr/bin/env python3
"""Pure-text STEP census and multi-file comparator.

Counts B-Rep topology entities by parsing STEP Part 21 text — no OCCT import.
Robust to AP203/AP214/AP242 formatting and multi-line entities.

Usage:
  step_census.py <file.step>              # JSON census on stdout
  step_census.py <a.step> <b.step> ...    # JSON array + diff table on stderr
  step_census.py --expect <truth.json> <file.step>  # exit 1 on mismatch

-----------------------------------------------------------------------------
Vendored from BlinkingSun/stl2step tests/tools/step_census.py (MIT License,
Copyright (c) 2026 stl2step contributors); modified for surface_modeling:

1. ``split_args`` is index-driven instead of a ``for``-loop. The upstream loop
   could only ``continue`` on the Part 21 ``''`` escape, so it stepped onto the
   second quote of the pair and inverted its own ``in_str`` state. After a name
   like ``'It''s'`` every following comma looked like it was inside a string,
   the argument list collapsed to one part, ``len(parts) >= 3`` failed and the
   ADVANCED_FACE was counted in ``faces`` but silently dropped from the
   ``surfaces`` histogram. The fixed splitter advances by two.
2. ``surfaces_consistent`` / ``curves_consistent`` are reported: the invariant
   ``sum(surfaces.values()) == faces`` turns exactly that silent drop into a
   visible flag, so the class of bug in (1) cannot hide again.
3. Complex (AND-combined) instances are classified. OCCT writes rational
   NURBS as ``#31 = ( BOUNDED_SURFACE() B_SPLINE_SURFACE(..) ..
   RATIONAL_B_SPLINE_SURFACE(..) SURFACE() );`` — an entity whose *type name is
   empty*, which upstream bucketed as ``other``. ``complex_sub_names`` scans the
   top-level sub-entity names and the first one known to SURFACE_TYPES /
   CURVE_TYPES is used.
4. ``census_path`` accepts ``str`` as well as ``Path`` — ``src/io/step_audit.py``
   calls it with the ``str`` path it was handed, and ``str`` has no
   ``read_text``.
5. ``census_text(text, name=...)`` split out of ``census_path`` so the parser
   can be tested on a literal without a temp file.
6. ``main`` reconfigures stdout to UTF-8 (this project's console is cp1252) and
   no longer raises on a missing ``--expect`` file.
7. ``resolve_surface`` carries a ``seen`` set. ``resolve_curve`` always had
   one; its surface twin did not, so an ``OFFSET_SURFACE`` chain that closes
   on itself recursed until ``RecursionError`` escaped ``census_path``. This
   parser exists to read files that are under suspicion, so a malformed
   entity graph has to terminate as ``other``, not as a stack overflow.
8. ``census_path`` never raises: any parse failure comes back as
   ``{"ok": False, "error": ...}``. ``step_audit._census`` swallows an
   exception into a warning, which would have removed the second witness from
   the audit without failing it.

Why the file exists at all: it is the audit's second witness. It shares no code
with the OpenCascade writer or reader, so when ``step_audit.verdict`` finds the
kernel's face count and this text count disagreeing, one of the two is wrong
and the STEP is not trustworthy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Union

# Surface / curve types we classify (STEP entity names).
SURFACE_TYPES = {
    "PLANE": "plane",
    "CYLINDRICAL_SURFACE": "cylinder",
    "CONICAL_SURFACE": "cone",
    "SPHERICAL_SURFACE": "sphere",
    "TOROIDAL_SURFACE": "torus",
    "B_SPLINE_SURFACE": "bspline",
    "B_SPLINE_SURFACE_WITH_KNOTS": "bspline",
    "RATIONAL_B_SPLINE_SURFACE": "bspline",
}

CURVE_TYPES = {
    "LINE": "line",
    "CIRCLE": "circle",
    "B_SPLINE_CURVE": "bspline",
    "B_SPLINE_CURVE_WITH_KNOTS": "bspline",
    "RATIONAL_B_SPLINE_CURVE": "bspline",
}

REF_RE = re.compile(r"#(\d+)\b")
NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?")
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _json_num(v: float) -> Union[float, int]:
    if not math.isfinite(v):
        return 0
    if v == 0.0:
        return 0.0
    r = round(v, 6)
    if abs(r - round(r)) < 1e-9:
        return int(round(r))
    return r


def parse_entities(text: str) -> dict[int, tuple[str, str]]:
    """Return {id: (TYPE, args_string)} for every #id = TYPE(...); entity.

    A complex (AND-combined) instance ``#id = ( A() B() );`` yields an empty
    TYPE and the whole ``A() B()`` body as the argument string; see
    ``complex_sub_names``.
    """
    entities: dict[int, tuple[str, str]] = {}
    i, n = 0, len(text)
    while i < n:
        if text[i] != "#":
            i += 1
            continue
        j = i + 1
        while j < n and text[j].isdigit():
            j += 1
        if j == i + 1:
            i += 1
            continue
        eid = int(text[i + 1 : j])
        while j < n and text[j] in " \t\r\n":
            j += 1
        if j >= n or text[j] != "=":
            i = j
            continue
        j += 1
        while j < n and text[j] in " \t\r\n":
            j += 1
        start = j
        depth = 0
        in_str = False
        while j < n:
            c = text[j]
            if in_str:
                if c == "'" and j + 1 < n and text[j + 1] == "'":
                    j += 2
                    continue
                if c == "'":
                    in_str = False
                j += 1
                continue
            if c == "'":
                in_str = True
                j += 1
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == ";" and depth == 0:
                break
            j += 1
        body = text[start:j].strip()
        p = body.find("(")
        if p < 0:
            etype, args = body, ""
        else:
            etype = body[:p].strip()
            close = len(body) - 1
            while close > p and body[close] in " \t":
                close -= 1
            args = body[p + 1 : close] if body[close] == ")" else body[p + 1 :]
        entities[eid] = (etype, args)
        i = j + 1
    return entities


def split_args(args: str) -> list[str]:
    """Split a STEP argument list on top-level commas.

    Index-driven so the Part 21 ``''`` escape can be skipped whole. The
    upstream ``for``-loop could only ``continue``, i.e. land on the second
    quote of the pair and close the string it was still inside; from there the
    rest of the argument list was swallowed into one part. That turned an
    ADVANCED_FACE with an apostrophe in its name into a face with no surface.
    """
    out: list[str] = []
    depth = 0
    in_str = False
    start = 0
    i, n = 0, len(args)
    while i < n:
        c = args[i]
        if in_str:
            if c == "'":
                if i + 1 < n and args[i + 1] == "'":
                    i += 2          # escaped quote: still inside the string
                    continue
                in_str = False
            i += 1
            continue
        if c == "'":
            in_str = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(args[start:i].strip())
            start = i + 1
        i += 1
    tail = args[start:].strip()
    if tail:
        out.append(tail)
    return out


def complex_sub_names(args: str) -> list[str]:
    """Top-level sub-entity names of an AND-combined (complex) instance.

    ``( BOUNDED_SURFACE() B_SPLINE_SURFACE(5,5,(..)) .. SURFACE() )`` gives
    ``['BOUNDED_SURFACE', 'B_SPLINE_SURFACE', ..., 'SURFACE']``. Only names at
    nesting depth 0 count, so a typed value inside an argument list (e.g.
    ``LENGTH_MEASURE(1.)``) is not mistaken for a sub-entity, and quoted
    strings — including ``REPRESENTATION_ITEM('')`` — are skipped.
    """
    names: list[str] = []
    depth = 0
    in_str = False
    token: str | None = None
    i, n = 0, len(args)
    while i < n:
        c = args[i]
        if in_str:
            if c == "'":
                if i + 1 < n and args[i + 1] == "'":
                    i += 2
                    continue
                in_str = False
            i += 1
            continue
        if c == "'":
            in_str = True
            i += 1
            continue
        if c == "(":
            if depth == 0 and token:
                names.append(token)
            token = None
            depth += 1
            i += 1
            continue
        if c == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0:
            m = NAME_RE.match(args, i)
            if m:
                token = m.group(0)
                i = m.end()
                continue
            if not c.isspace():
                token = None
        i += 1
    return names


def complex_type(args: str, table: dict[str, str]) -> str:
    """First sub-entity name of a complex instance that ``table`` classifies."""
    for name in complex_sub_names(args):
        if name in table:
            return name
    return ""


def first_ref(arg: str) -> int | None:
    m = REF_RE.search(arg)
    return int(m.group(1)) if m else None


def resolve_curve(entities: dict[int, tuple[str, str]], ref: int, seen: set[int]) -> tuple[str, str]:
    if ref in seen:
        return "other", ""
    seen.add(ref)
    etype, args = entities.get(ref, ("", ""))
    if not etype:                                   # complex (AND-combined)
        etype = complex_type(args, CURVE_TYPES)
    if etype in ("SURFACE_CURVE", "SEAM_CURVE"):
        parts = split_args(args)
        for part in parts[:2]:
            inner = first_ref(part)
            if inner is not None:
                inner_type, inner_args = entities.get(inner, ("", ""))
                if not inner_type:
                    inner_type = complex_type(inner_args, CURVE_TYPES)
                if inner_type in CURVE_TYPES or inner_type in ("SURFACE_CURVE", "SEAM_CURVE"):
                    return resolve_curve(entities, inner, seen)
    return etype, args


def resolve_surface(entities: dict[int, tuple[str, str]], ref: int,
                    seen: set[int] | None = None) -> tuple[str, str]:
    """Classify the surface at ``ref``, following OFFSET_SURFACE chains.

    ``seen`` breaks reference cycles the same way ``resolve_curve`` does. An
    ``OFFSET_SURFACE`` pointing at itself (or at a partner that points back)
    is a malformed file, not an unreachable one — and this parser's whole job
    is to read files that may be malformed, so it must terminate on them
    instead of unwinding into RecursionError.
    """
    if seen is None:
        seen = set()
    if ref in seen:
        return "other", ""
    seen.add(ref)
    etype, args = entities.get(ref, ("", ""))
    if not etype:                                   # complex (AND-combined)
        etype = complex_type(args, SURFACE_TYPES)
    if etype == "OFFSET_SURFACE":
        parts = split_args(args)
        if len(parts) >= 2:
            inner = first_ref(parts[1])
            if inner is not None:
                return resolve_surface(entities, inner, seen)
    if etype == "SURFACE_OF_REVOLUTION":
        parts = split_args(args)
        if len(parts) >= 2:
            inner = first_ref(parts[1])
            if inner is not None:
                return resolve_curve(entities, inner, set())
    return etype, args


def cylinder_radius(etype: str, args: str) -> float | None:
    if etype != "CYLINDRICAL_SURFACE":
        return None
    parts = split_args(args)
    for part in reversed(parts):
        m = NUM_RE.fullmatch(part.strip())
        if m:
            return float(m.group(0))
    return None


def circle_radius(etype: str, args: str) -> float | None:
    if etype != "CIRCLE":
        return None
    parts = split_args(args)
    if len(parts) >= 3:
        m = NUM_RE.fullmatch(parts[2].strip())
        if m:
            return float(m.group(0))
    return None


def census_text(text: str, name: str = "<text>") -> dict[str, Any]:
    """Census of STEP Part 21 ``text``; ``name`` only labels the result."""
    if "ISO-10303-21" not in text and "DATA;" not in text:
        return {"ok": False, "input": name, "error": "not a STEP Part 21 file"}

    entities = parse_entities(text)

    surfaces = {k: 0 for k in ("plane", "cylinder", "cone", "sphere", "torus", "bspline", "other")}
    curves = {k: 0 for k in ("line", "circle", "bspline", "other")}
    cyl_radii: list[float] = []
    advanced_faces = 0
    edge_curves = 0

    for eid, (etype, args) in entities.items():
        if etype == "ADVANCED_FACE":
            advanced_faces += 1
            parts = split_args(args)
            if len(parts) >= 3:
                sref = first_ref(parts[2])
                if sref is not None:
                    stype, sargs = resolve_surface(entities, sref)
                    bucket = SURFACE_TYPES.get(stype, "other")
                    surfaces[bucket] += 1
                    r = cylinder_radius(stype, sargs)
                    if r is not None:
                        cyl_radii.append(r)

        elif etype == "EDGE_CURVE":
            edge_curves += 1
            parts = split_args(args)
            if len(parts) >= 4:
                cref = first_ref(parts[3])
                if cref is not None:
                    ctype, cargs = resolve_curve(entities, cref, set())
                    bucket = CURVE_TYPES.get(ctype, "other")
                    curves[bucket] += 1

    cyl_radii.sort()
    cyl_unique = sorted(set(cyl_radii))

    return {
        "ok": True,
        "input": name,
        "faces": advanced_faces,
        "edges": edge_curves,
        "surfaces": surfaces,
        "curves": curves,
        # every ADVANCED_FACE must land in exactly one surface bucket; a
        # mismatch means an argument list did not parse (see split_args)
        "surfaces_consistent": sum(surfaces.values()) == advanced_faces,
        "curves_consistent": sum(curves.values()) == edge_curves,
        "cylinder_radii": [_json_num(r) for r in cyl_radii],
        "cylinder_radii_unique": [_json_num(r) for r in cyl_unique],
    }


def census_path(path: Union[str, Path]) -> dict[str, Any]:
    """Census the STEP file at ``path`` (``str`` or ``Path``).

    Always returns a report dict — never raises. The caller is an audit
    (``src/io/step_audit._census``) running over a file that is under
    suspicion by construction; an exception there costs the audit its second
    witness silently, so a parse failure is reported as ``ok: False`` with the
    error text instead.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"ok": False, "input": str(path), "error": str(e)}
    try:
        census = census_text(text, name=str(path))
    except Exception as e:   # RecursionError on a cyclic entity graph, etc.
        return {"ok": False, "input": str(path),
                "error": f"{type(e).__name__}: {e}"}
    census["input"] = str(path)
    return census


def diff_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    """Human-readable comparison table for N censuses."""
    names = [name for name, _ in rows]
    col_w = max(12, max(len(n) for n in names) + 2)

    def row(metric: str, vals: list[str]) -> str:
        return f"| {metric:<22} |" + "".join(f" {v:>{col_w}} |" for v in vals)

    lines = [
        row("metric", names),
        row("-" * 22, ["-" * col_w for _ in names]),
    ]

    metrics: list[tuple[str, Any]] = [
        ("ADVANCED_FACE", lambda c: c.get("faces", "-")),
        ("EDGE_CURVE", lambda c: c.get("edges", "-")),
    ]
    for key in ("plane", "cylinder", "cone", "sphere", "torus", "bspline", "other"):
        metrics.append((f"surface.{key}", lambda c, k=key: c.get("surfaces", {}).get(k, "-")))
    for key in ("line", "circle", "bspline", "other"):
        metrics.append((f"curve.{key}", lambda c, k=key: c.get("curves", {}).get(k, "-")))
    metrics.append(("cylinder_radii (n)", lambda c: len(c.get("cylinder_radii", []))))
    metrics.append(("surfaces_consistent", lambda c: c.get("surfaces_consistent", "-")))
    metrics.append(("curves_consistent", lambda c: c.get("curves_consistent", "-")))

    for label, fn in metrics:
        vals = []
        for _, c in rows:
            if not c.get("ok"):
                vals.append("ERR")
            else:
                vals.append(str(fn(c)))
        lines.append(row(label, vals))

    return "\n".join(lines)


def compare_expect(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if not actual.get("ok"):
        return [actual.get("error", "census failed")]

    for key in ("faces",):
        if key in expected and actual.get(key) != expected[key]:
            errs.append(f"{key}: got {actual.get(key)} want {expected[key]}")

    for section in ("surfaces", "curves"):
        exp_sec = expected.get(section) or {}
        act_sec = actual.get(section) or {}
        for k, want in exp_sec.items():
            got = act_sec.get(k, 0)
            if got != want:
                errs.append(f"{section}.{k}: got {got} want {want}")

    if "cylinder_radii" in expected:
        exp_r = [_json_num(float(x)) for x in expected["cylinder_radii"]]
        got_r = actual.get("cylinder_radii", [])
        if got_r != exp_r:
            errs.append(f"cylinder_radii: got {got_r} want {exp_r}")

    if not actual.get("surfaces_consistent", True):
        errs.append("surfaces_consistent: an ADVANCED_FACE has no surface bucket")

    return errs


def main(argv: list[str] | None = None) -> int:
    try:  # this project's console is cp1252; STEP names are not
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - captured stdout
        pass

    p = argparse.ArgumentParser(description="STEP Part 21 text census")
    p.add_argument("steps", nargs="+", type=Path, help="STEP file(s)")
    p.add_argument("--expect", type=Path, help="Ground-truth JSON to assert against (last STEP only)")
    p.add_argument("--no-diff", action="store_true", help="Suppress diff table when N>1")
    args = p.parse_args(argv)

    if args.expect and len(args.steps) != 1:
        print("error: --expect requires exactly one STEP file", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    for step in args.steps:
        results.append(census_path(step))

    if len(results) == 1:
        out = results[0]
    else:
        out = {"ok": all(r.get("ok") for r in results), "files": results}

    print(json.dumps(out, separators=(",", ":"), sort_keys=True))

    if len(results) > 1 and not args.no_diff:
        ok_rows = [(Path(r["input"]).name, r) for r in results]
        print("\n" + diff_table(ok_rows), file=sys.stderr)

    all_ok = all(r.get("ok") for r in results)

    if args.expect:
        try:
            expected = json.loads(args.expect.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"error: cannot read --expect {args.expect}: {exc}", file=sys.stderr)
            return 2
        errs = compare_expect(expected, results[0])
        if errs:
            print("EXPECT FAIL:", file=sys.stderr)
            for e in errs:
                print(f"  {e}", file=sys.stderr)
            return 1
        print(f"EXPECT OK vs {args.expect}", file=sys.stderr)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
