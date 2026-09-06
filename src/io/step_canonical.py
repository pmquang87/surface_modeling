"""Canonical form of a STEP file, for comparing two runs byte-for-byte.

Two runs of the same converter on the same input must write the same STEP.
The only thing they are allowed to differ in is the wall-clock timestamp
ISO 10303-21 puts in ``FILE_NAME`` field 2, so that is the only thing this
module rewrites.

Pattern and the ``FILE_NAME``-timestamp regex from
BlinkingSun/stl2step (MIT), ``tests/gates/baseline/canonicalize.py``
(Copyright (c) 2026 stl2step contributors). ``strip_file_name_timestamp``,
``normalize_newlines`` and the ``_read_text`` errors="replace" rationale are
near-verbatim from that file; the section slicing and the public API are a
rewrite. Two deliberate differences from the original:

* the original discards the whole HEADER; here the HEADER is **kept** (with
  the timestamp blanked). A changed schema, unit, or preprocessor version is
  a real change in the product we ship, and dropping the HEADER would hide it.
* section slicing is string-literal aware rather than line-anchored, so a
  ``DATA;`` that occurs *inside* a STEP string cannot be mistaken for the
  start of the DATA section.

What is deliberately **not** done:

* entity ``#N`` ids are never renumbered. Renumbering makes a canonicaliser
  tolerant of exactly the thing it is supposed to catch: a topology change
  that shifts every id downstream of it.
* no whitespace, number formatting, or entity ordering is normalised.

Public API::

    canonical_step_text(text)            -> canonical str
    canonical_step_digest(path)          -> sha256 hex of that str
    step_files_equal(a, b)               -> (bool, first_diff_lines)
    extract_header_section(text)         -> "HEADER; ... ENDSEC;"
    extract_data_section(text)           -> "DATA; ... ENDSEC;"

Standard library only: this module is the instrument, so it must not depend
on OpenCascade (or anything else the thing under test depends on).
"""
from __future__ import annotations

import difflib
import hashlib
import re
from typing import List, Tuple

__all__ = [
    "canonical_step_text",
    "canonical_step_digest",
    "step_files_equal",
    "extract_header_section",
    "extract_data_section",
    "strip_file_name_timestamp",
    "normalize_newlines",
]


# ISO 10303-21 FILE_NAME(name, timestamp, author, organisation,
# preprocessor_version, originating_system, authorization): field 2 is a
# wall-clock timestamp and is the one run-variant field.
#
# The string pattern is written as ``'[^']*(?:''[^']*)*'`` rather than the
# original ``'(?:[^']|'')*'``: same language (a STEP string with '' escapes),
# but unambiguous, so it cannot backtrack exponentially on a malformed file.
_STEP_STRING = r"'[^']*(?:''[^']*)*'"
_FILE_NAME_TS = re.compile(
    r"(FILE_NAME\s*\(\s*" + _STEP_STRING + r"\s*,\s*)" + _STEP_STRING,
    re.IGNORECASE | re.DOTALL,
)


def normalize_newlines(text: str) -> str:
    """CRLF / CR -> LF. Line endings are a platform artefact, not content."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_file_name_timestamp(text: str) -> str:
    """Blank FILE_NAME field 2 (the wall-clock timestamp), nothing else.

    Only the first occurrence is rewritten: a STEP file has exactly one
    FILE_NAME, and a second one appearing in DATA would be content.
    """
    return _FILE_NAME_TS.sub(r"\1''", text, count=1)


def _string_mask(text: str) -> List[bool]:
    """True for every character that sits inside a STEP ``'...'`` literal.

    STEP escapes an apostrophe by doubling it, so ``'it''s'`` is one string.
    The mask marks the quotes themselves as inside, which is all we need:
    a section keyword found at a masked position is not a keyword.
    """
    mask = [False] * len(text)
    inside = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "'":
            if inside and i + 1 < n and text[i + 1] == "'":
                mask[i] = True
                mask[i + 1] = True
                i += 2
                continue
            mask[i] = True
            inside = not inside
            i += 1
            continue
        mask[i] = inside
        i += 1
    return mask


def _find_keyword_line(text: str, mask: List[bool], keyword: str,
                       start: int = 0) -> int:
    """Index of the first line that *is* ``keyword`` (e.g. ``"DATA;"``).

    Token-aware in both directions: the line may carry leading/trailing
    whitespace but nothing else, so ``FOODATA;`` and ``DATA; #1=...`` do not
    match, and the position must not be inside a string literal, so a
    ``DATA;`` line inside a multi-line FILE_DESCRIPTION does not match either.
    Returns -1 when there is no such line.

    The search runs on ``text`` itself with ``re.IGNORECASE`` rather than on
    ``text.upper()``: uppercasing is not length-preserving in Unicode (German
    ``ss`` -> ``SS`` grows by one), and a single such character anywhere ahead
    of the keyword shifts every index of the upper-cased copy out of step with
    ``mask`` and with ``text`` - the section is then mis-sliced, or reported
    missing, on a file that is perfectly well formed.
    """
    n = len(text)
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    for m in pattern.finditer(text, start):
        pos = m.start()
        if mask[pos]:
            continue  # inside a string literal
        line_start = text.rfind("\n", 0, pos) + 1
        if text[line_start:pos].strip() != "":
            continue  # something else on the line before the keyword
        end = m.end()
        line_end = text.find("\n", end)
        if line_end < 0:
            line_end = n
        if text[end:line_end].strip() != "":
            continue  # something else on the line after the keyword
        return pos
    return -1


def _extract_section(text: str, keyword: str) -> str:
    """``keyword ... ENDSEC;`` inclusive, verbatim. Raises ValueError."""
    mask = _string_mask(text)
    start = _find_keyword_line(text, mask, keyword)
    if start < 0:
        raise ValueError("STEP file has no %s section" % keyword.rstrip(";"))
    end = _find_keyword_line(text, mask, "ENDSEC;", start + len(keyword))
    if end < 0:
        raise ValueError(
            "STEP %s section is not terminated by ENDSEC;" % keyword.rstrip(";"))
    return text[start:end + len("ENDSEC;")]


def extract_header_section(text: str) -> str:
    """``HEADER; ... ENDSEC;`` inclusive."""
    return _extract_section(normalize_newlines(text), "HEADER;")


def extract_data_section(text: str) -> str:
    """``DATA; ... ENDSEC;`` inclusive.

    Raises ValueError if the section is missing or unterminated - a file we
    cannot slice must not silently canonicalise to the empty string, or two
    broken files would compare equal.
    """
    return _extract_section(normalize_newlines(text), "DATA;")


def canonical_step_text(text: str) -> str:
    """Canonical comparison form of a STEP file.

    ``HEADER;``-section (FILE_NAME timestamp blanked) followed by the
    ``DATA;`` section, line endings normalised, everything else verbatim.
    ``ISO-10303-21;`` / ``END-ISO-10303-21;`` are dropped: they are constant.
    """
    norm = normalize_newlines(text)
    stripped = strip_file_name_timestamp(norm)
    data = extract_data_section(stripped)
    try:
        header = extract_header_section(stripped)
    except ValueError:
        header = ""  # headerless fragment: compare the DATA section alone
    parts = [p for p in (header, data) if p]
    return "\n".join(p if p.endswith("\n") else p + "\n" for p in parts)


def canonical_step_digest(path: str) -> str:
    """sha256 (hex) of ``canonical_step_text`` of the file at ``path``."""
    canon = canonical_step_text(_read_text(path))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _read_text(path: str) -> str:
    # STEP is ASCII; ``replace`` keeps a truncated write from raising here -
    # a replacement character is itself a difference and will show in the diff.
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def step_files_equal(path_a: str, path_b: str,
                     max_diff_lines: int = 40) -> Tuple[bool, List[str]]:
    """Compare two STEP files in canonical form.

    Returns ``(equal, diff_lines)``. ``diff_lines`` is an empty list when the
    files are equal, otherwise the first ``max_diff_lines`` lines of a
    unified diff (with the two digests on top), ready to paste into an
    assertion message.
    """
    can_a = canonical_step_text(_read_text(path_a))
    can_b = canonical_step_text(_read_text(path_b))
    if can_a == can_b:
        return True, []
    diff = list(difflib.unified_diff(
        can_a.splitlines(), can_b.splitlines(),
        fromfile=path_a, tofile=path_b, lineterm="", n=1))
    head = [
        "  a sha256=" + hashlib.sha256(can_a.encode("utf-8")).hexdigest(),
        "  b sha256=" + hashlib.sha256(can_b.encode("utf-8")).hexdigest(),
    ]
    shown = diff[:max_diff_lines]
    if len(diff) > max_diff_lines:
        shown.append("  ... %d more diff lines" % (len(diff) - max_diff_lines))
    return False, head + shown
