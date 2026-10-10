"""The OCP import contract of ``src``: flat ``OCP.<Module>`` names only.

Round 4 (bug report 2026-10) checked which package layouts exist:

* conda-forge ``ocp`` installs one ``OCP.<abi>.pyd/.so`` straight into
  site-packages; ``OCP.OCP`` does not exist there.
* cadquery-ocp wheels 7.5.3.0 ... 7.7.2 do the same.
* cadquery-ocp / cadquery-ocp-novtk wheels from 7.7.2.2b2 up to 8.0.1.1.0 put
  the native module at ``OCP/OCP.<abi>.pyd`` and ship a thin wrapper package
  ``OCP/<Module>/__init__.py`` (``from ..OCP.<Module> import *``) for every
  module. That is why tracebacks show ``OCP.OCP.<Module>``. Some wrappers add
  compatibility aliases - on 7.9 ``OCP/TopoDS`` adds ``TopoDS.Face_s`` and
  friends, which ``src`` calls in many places.

So no release ships ``OCP.OCP`` without the flat names, and importing
``OCP.OCP.*`` directly would skip the wrapper aliases. The former
``OCP.OCP`` fallback in ``import_step`` was dead code and was removed; these
tests pin the contract for every module in ``src``.

Test discipline: every checker is first shown an input it must reject.
"""
import ast
import importlib
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")


def _src_files():
    out = []
    for dirpath, _dirs, files in os.walk(SRC):
        out += [os.path.join(dirpath, f) for f in files if f.endswith(".py")]
    assert len(out) > 20, out          # the walk really found the package
    return sorted(out)


def _parse(path):
    with open(path, encoding="utf-8-sig") as fh:   # tolerate a BOM
        return ast.parse(fh.read(), filename=path)


def nested_ocp_references(tree):
    """Every place a module tree reaches the native ``OCP.OCP`` module:
    ``import OCP.OCP...``, ``from OCP.OCP... import``, ``from OCP import OCP``
    and a call with a string argument such as ``import_module("OCP.OCP.X")``.
    Comments never reach the AST, so explaining the layout is allowed."""
    hits = []

    def nested(name):
        return name == "OCP.OCP" or name.startswith("OCP.OCP.")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [(node.lineno, a.name) for a in node.names if nested(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if nested(node.module):
                hits.append((node.lineno, node.module))
            elif node.module == "OCP" and any(a.name == "OCP" for a in node.names):
                hits.append((node.lineno, "from OCP import OCP"))
        elif isinstance(node, ast.Call):
            for arg in node.args:
                if (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                        and nested(arg.value)):
                    hits.append((node.lineno, arg.value))
    return hits


def flat_ocp_imports(tree):
    """``(module, name)`` for every ``from OCP.<Module> import name`` and
    ``(module, None)`` for every ``import OCP.<Module>``."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and \
                node.module.startswith("OCP."):
            out |= {(node.module, a.name) for a in node.names}
        elif isinstance(node, ast.Import):
            out |= {(a.name, None) for a in node.names if a.name.startswith("OCP.")}
    return out


def unresolvable(pairs):
    """The ``(module, name)`` pairs that do not import on this install."""
    bad = []
    for module, name in sorted(pairs, key=lambda p: (p[0], p[1] or "")):
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            bad.append((module, name, repr(exc)))
            continue
        if name is not None and not hasattr(mod, name):
            bad.append((module, name, "missing attribute"))
    return bad


def test_nested_reference_checker_rejects_every_form():
    """The checker must flag each way of reaching ``OCP.OCP`` (negative
    controls) and must pass flat imports plus a comment that names it."""
    bad_sources = [
        "from OCP.OCP.TopAbs import TopAbs_FACE\n",
        "import OCP.OCP.STEPControl\n",
        "import OCP.OCP\n",
        "from OCP import OCP\n",
        "import importlib\nimportlib.import_module('OCP.OCP.TopoDS')\n",
        "def f():\n    try:\n        from OCP.TopoDS import TopoDS\n"
        "    except ImportError:\n        from OCP.OCP.TopoDS import TopoDS\n",
    ]
    for source in bad_sources:
        assert nested_ocp_references(ast.parse(source)), source

    good = ("# OCP.OCP is the native module behind the wrappers\n"
            "from OCP.TopoDS import TopoDS\nimport OCP.TopAbs\nimport OCP\n"
            "from OCP import TopoDS as _t\n")
    assert nested_ocp_references(ast.parse(good)) == []


def test_src_never_reaches_the_native_ocp_ocp_module():
    """No module in ``src`` imports ``OCP.OCP.*``: the flat wrappers are the
    public API and carry the compatibility aliases (failed on 1f93473 with
    the seven ``from OCP.OCP.<X>`` lines of ``import_step``)."""
    hits = {os.path.relpath(p, ROOT): nested_ocp_references(_parse(p))
            for p in _src_files()}
    hits = {k: v for k, v in hits.items() if v}
    assert hits == {}, hits


def test_flat_import_checker_rejects_missing_modules_and_names():
    """Negative controls for ``unresolvable``: an unknown module and an
    unknown name in a real module must both be reported, a real pair not."""
    if not HAS_OCP:
        pytest.skip("cadquery-ocp not installed")
    bad = unresolvable({("OCP.NoSuchModule_r4", "X"),
                        ("OCP.TopAbs", "TopAbs_NO_SUCH_NAME"),
                        ("OCP.TopAbs", "TopAbs_FACE")})
    assert {(m, n) for m, n, _ in bad} == {("OCP.NoSuchModule_r4", "X"),
                                          ("OCP.TopAbs", "TopAbs_NO_SUCH_NAME")}


@needs_ocp
def test_every_flat_ocp_name_src_imports_resolves():
    """Every ``from OCP.<Module> import <name>`` in ``src`` resolves on this
    install, so the flat layout really covers the whole code base."""
    pairs = set()
    per_file = {}
    for path in _src_files():
        found = flat_ocp_imports(_parse(path))
        if found:
            per_file[os.path.relpath(path, ROOT)] = len(found)
        pairs |= found
    # bracket: the scan sees the known OCP users and a realistic number of names
    for rel in ("io/importers.py", "io/exporters.py", "io/step_audit.py",
                "io/faceted_step.py", "io/occt_utils.py", "nurbs/converter.py",
                "nurbs/simplifier.py"):
        assert os.path.join("src", os.path.normpath(rel)) in per_file, per_file
    assert 40 <= len(pairs) <= 400, len(pairs)
    assert unresolvable(pairs) == []


def _topods_attributes_used(tree):
    return {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == "TopoDS"}


@needs_ocp
def test_flat_topods_has_every_downcast_src_calls():
    """``src`` calls ``TopoDS.Face_s`` etc.; on cadquery-ocp 7.9 those exist
    only because the flat ``OCP.TopoDS`` wrapper adds them, which is why the
    code must not import ``OCP.OCP.TopoDS`` directly."""
    from OCP.TopoDS import TopoDS

    used = set()
    for path in _src_files():
        used |= _topods_attributes_used(_parse(path))
    assert {"Face_s", "Shell_s"} <= used, used     # the scan sees the calls
    # negative control: the checker reports a name that does not exist
    assert not hasattr(TopoDS, "Bogus_s")
    missing = sorted(a for a in used | {"Bogus_s"} if not hasattr(TopoDS, a))
    assert missing == ["Bogus_s"], missing
