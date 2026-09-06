"""The .gitignore must hide the root build/dist dirs and nothing else.

Unanchored ``build/`` and ``dist/`` patterns match a directory of that name at
ANY depth, so a source or fixture directory called ``build`` becomes invisible
to git -- ``git add`` says nothing, the file is never committed, and the gap is
only found when a clean clone fails. stl2step hit exactly this (FINDINGS-0 D5:
an unanchored ``build*/`` silently hid committed test data), so the patterns
are anchored to the repository root with a leading slash.

The assertions ask git itself (``git check-ignore``), not a regex over the
file, so a pattern that looks anchored but is not still fails. ``--no-index``
lets the paths be hypothetical: nothing has to exist on disk.
"""
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GITIGNORE = os.path.join(REPO, ".gitignore")

needs_git = pytest.mark.skipif(
    shutil.which("git") is None or not os.path.exists(os.path.join(REPO, ".git")),
    reason="git or the repository metadata is unavailable",
)

pytestmark = needs_git


def _ignored(rel_path):
    """True when git would ignore ``rel_path`` (exit 0), False on exit 1."""
    proc = subprocess.run(
        [shutil.which("git"), "check-ignore", "--no-index", "-q", rel_path],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.returncode in (0, 1), (
        f"git check-ignore failed on {rel_path!r}: rc={proc.returncode} "
        f"{proc.stderr.decode('utf-8', 'replace')}"
    )
    return proc.returncode == 0


def test_git_check_ignore_is_actually_usable():
    """Control: the helper can report both answers, so a test can fail."""
    assert _ignored("__pycache__/x.pyc") is True, "the .gitignore is not being read"
    assert _ignored("src/nurbs/simplifier.py") is False, \
        "everything looks ignored - check-ignore is not discriminating"


def test_nested_build_and_dist_directories_are_not_ignored():
    assert _ignored("tests/fixtures/build/x.py") is False, \
        "a nested build/ directory is invisible to git (unanchored pattern)"
    assert _ignored("src/build/foo.py") is False
    assert _ignored("tests/fixtures/dist/x.py") is False
    assert _ignored("src/dist/foo.py") is False


def test_root_build_and_dist_directories_are_still_ignored():
    assert _ignored("build/x.py") is True, "the root build/ output is no longer ignored"
    assert _ignored("dist/x.whl") is True, "the root dist/ output is no longer ignored"


def test_cad_data_patterns_are_untouched():
    """The *.stl / *.step rules are deliberately global; do not anchor them."""
    for path in ("part.stl", "tests/fixtures/part.stl",
                 "out.step", "src/io/out.step", "deep/nest/a.STL"):
        assert _ignored(path) is True, f"{path} should still be ignored"


def test_gitignore_has_no_unanchored_build_or_dist_pattern():
    with open(GITIGNORE, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh]
    for bad in ("build/", "dist/", "build", "dist"):
        assert bad not in lines, f"unanchored pattern {bad!r} is back in .gitignore"
    assert "/build/" in lines and "/dist/" in lines, \
        "the anchored root-only patterns are missing"
