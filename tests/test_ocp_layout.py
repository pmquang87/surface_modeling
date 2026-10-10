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

The native module is also reachable without spelling ``OCP.OCP``: on 7.9
``OCP/__init__.py`` runs ``from OCP.OCP import *``, so ``from OCP import
TopoDS`` and ``import OCP; OCP.TopoDS`` give the NATIVE ``OCP.OCP.TopoDS``
(no ``Face_s``) unless the flat ``OCP.TopoDS`` was imported first. Those
forms are rejected too.

Test discipline: every checker is first shown inputs it must reject.
"""
import ast
import importlib
import importlib.util
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


def _is_native(name):
    return name == "OCP.OCP" or name.startswith("OCP.OCP.")


def _dunder(name):
    return name.startswith("__") and name.endswith("__")


def _import_call_target(call):
    """The module name an ``import_module(...)`` / ``__import__(...)`` call
    imports, resolved against its ``package`` argument; None if not such a
    call or the name is not a string literal."""
    func = call.func
    fname = func.id if isinstance(func, ast.Name) else \
        func.attr if isinstance(func, ast.Attribute) else None
    if fname not in ("import_module", "__import__"):
        return None
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    name = call.args[0] if call.args else kw.get("name")
    if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
        return None
    target = name.value
    if fname == "import_module" and target.startswith("."):
        package = call.args[1] if len(call.args) > 1 else kw.get("package")
        if not (isinstance(package, ast.Constant) and isinstance(package.value, str)):
            return None
        try:
            target = importlib.util.resolve_name(target, package.value)
        except (ImportError, ValueError):
            return None
    return target


def native_ocp_references(tree):
    """Every place a module tree can reach the native ``OCP.OCP`` module:

    * ``import OCP.OCP...`` and ``from OCP.OCP... import ...``;
    * ``from OCP import <name>`` for any non-dunder name (``OCP``,
      ``TopoDS``, ``*``): on the wheel layout the package namespace holds
      the native submodules;
    * ``<pkg>.<Name>`` where ``<pkg>`` is bound by ``import OCP`` /
      ``import OCP as <pkg>`` / ``import OCP.<X>``, unless ``OCP.<Name>``
      itself is imported somewhere in the module (then the attribute is the
      flat wrapper);
    * ``import_module(...)`` / ``__import__(...)`` of an ``OCP.OCP`` name,
      positional or keyword, absolute or relative to a literal package.

    A string that is not an import argument (a log message, a comment) is
    not a reference."""
    hits = []
    pkg_names = set()            # names bound to the top-level OCP package
    flat_imported = set()        # "X" for every "import OCP.X..."
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if _is_native(a.name):
                    hits.append((node.lineno, a.name))
                elif a.name == "OCP" or a.name.startswith("OCP."):
                    if a.name.startswith("OCP."):
                        flat_imported.add(a.name.split(".")[1])
                    if a.asname is None:
                        pkg_names.add("OCP")
                    elif a.name == "OCP":
                        pkg_names.add(a.asname)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if _is_native(node.module):
                hits.append((node.lineno, node.module))
            elif node.module == "OCP":
                hits += [(node.lineno, f"from OCP import {a.name}")
                         for a in node.names if not _dunder(a.name)]
        elif isinstance(node, ast.Call):
            target = _import_call_target(node)
            if target is not None and _is_native(target):
                hits.append((node.lineno, f"import call {target}"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in pkg_names and not _dunder(node.attr)
                and node.attr not in flat_imported):
            hits.append((node.lineno, f"{node.value.id}.{node.attr}"))
    return sorted(set(hits))


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


def topods_attributes_used(tree):
    """Every attribute read on the ``TopoDS`` class, under whatever name the
    module binds it: ``from OCP.TopoDS import TopoDS [as T]`` -> ``T.x``,
    and ``import OCP.TopoDS as M`` / ``import OCP.TopoDS`` ->
    ``M.TopoDS.x`` / ``OCP.TopoDS.TopoDS.x``."""
    cls_names, mod_names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "OCP.TopoDS":
            cls_names |= {a.asname or a.name for a in node.names if a.name == "TopoDS"}
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "OCP.TopoDS":
                    mod_names.add(a.asname or "OCP.TopoDS")

    def dotted(expr):
        parts = []
        while isinstance(expr, ast.Attribute):
            parts.append(expr.attr)
            expr = expr.value
        if isinstance(expr, ast.Name):
            parts.append(expr.id)
            return ".".join(reversed(parts))
        return None

    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            owner = dotted(node.value)
            if owner in cls_names or (owner and owner.endswith(".TopoDS")
                                      and owner[:-len(".TopoDS")] in mod_names):
                used.add(node.attr)
    return used


def missing_topods_attributes(used):
    from OCP.TopoDS import TopoDS
    return sorted(a for a in used if not hasattr(TopoDS, a))


# --------------------------------------------------------------------------
# no route to the native module
# --------------------------------------------------------------------------

#: Sources the scanner must reject, each with the reason it is unsafe.
BAD_SOURCES = [
    "from OCP.OCP.TopAbs import TopAbs_FACE\n",
    "import OCP.OCP.STEPControl\n",
    "import OCP.OCP\n",
    "from OCP import OCP\n",
    # native via the package namespace (``from OCP.OCP import *`` in 7.9)
    "from OCP import TopoDS\n",
    "from OCP import TopoDS as _t\n",
    "from OCP import *\n",
    "import OCP\nOCP.TopoDS.TopoDS.Face_s(x)\n",
    "import OCP as occ\nocc.TopAbs.TopAbs_FACE\n",
    # ``import OCP.TopoDS`` makes OCP.TopoDS flat, but not OCP.TopAbs
    "import OCP.TopoDS\nOCP.TopAbs.TopAbs_FACE\n",
    # dynamic imports
    "import importlib\nimportlib.import_module('OCP.OCP.TopoDS')\n",
    "from importlib import import_module\nimport_module(name='OCP.OCP.TopoDS')\n",
    "import importlib\nimportlib.import_module('.OCP.TopoDS', 'OCP')\n",
    "import importlib\nimportlib.import_module('.OCP.TopoDS', package='OCP')\n",
    "__import__('OCP.OCP.TopAbs')\n",
    "def f():\n    try:\n        from OCP.TopoDS import TopoDS\n"
    "    except ImportError:\n        from OCP.OCP.TopoDS import TopoDS\n",
]

#: Sources the scanner must accept.
GOOD_SOURCES = [
    "# OCP.OCP is the native module behind the wrappers\n"
    "from OCP.TopoDS import TopoDS\nimport OCP.TopAbs\n",
    "import OCP\nfrom OCP import __version__\nprint(OCP.__version__)\n",
    "import OCP.TopoDS\nOCP.TopoDS.TopoDS.Face_s(x)\n",
    "import OCP.TopoDS as T\nT.TopoDS.Face_s(x)\n",
    "import logging\nlogging.info('traceback mentions OCP.OCP.TopoDS')\n",
    "import importlib\nimportlib.import_module('OCP.TopoDS')\n",
    "import importlib\nimportlib.import_module('.TopoDS', 'OCP')\n",
]


def test_nested_reference_checker_rejects_every_form():
    """The scanner flags each way of reaching the native module (negative
    controls) and passes the flat forms, dunder names and plain strings."""
    missed = [s for s in BAD_SOURCES if not native_ocp_references(ast.parse(s))]
    assert missed == []
    flagged = {s: native_ocp_references(ast.parse(s)) for s in GOOD_SOURCES}
    assert {s: h for s, h in flagged.items() if h} == {}


@needs_ocp
def test_rejected_package_forms_really_give_the_native_module():
    """Ground truth for the ``from OCP import <Module>`` rule, in a fresh
    interpreter (this process may already have imported the wrappers):
    on the wheel layout the package attribute is the native module without
    the wrapper aliases, and only the flat import gives the aliases."""
    import subprocess
    code = ("import OCP\n"
            "has_native = hasattr(OCP, 'OCP')\n"
            "from OCP import TopoDS as pkg_attr\n"
            "print(has_native, pkg_attr.__name__, hasattr(pkg_attr.TopoDS, 'Face_s'))\n"
            "import OCP.TopoDS as flat\n"
            "print(flat.__name__, hasattr(flat.TopoDS, 'Face_s'))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=ROOT, timeout=300)
    assert out.returncode == 0, out.stderr
    first, second = out.stdout.split("\n")[:2]
    if not first.startswith("True"):
        pytest.skip("single-module (conda / <=7.7.2) layout: no wrappers")
    assert first.split()[1] == "OCP.OCP.TopoDS", out.stdout
    assert second.split()[0] == "OCP.TopoDS", out.stdout
    assert second.split()[1] == "True", out.stdout


def test_src_never_reaches_the_native_ocp_ocp_module():
    """No module in ``src`` reaches the native module: the flat wrappers are
    the public API and carry the compatibility aliases (failed on 1f93473
    with the seven ``from OCP.OCP.<X>`` lines of ``import_step``)."""
    hits = {os.path.relpath(p, ROOT): native_ocp_references(_parse(p))
            for p in _src_files()}
    hits = {k: v for k, v in hits.items() if v}
    assert hits == {}, hits


# --------------------------------------------------------------------------
# every flat name src uses exists here
# --------------------------------------------------------------------------

@needs_ocp
def test_flat_import_checker_rejects_missing_modules_and_names():
    """Negative controls for ``unresolvable``: an unknown module and an
    unknown name in a real module must both be reported, a real pair not."""
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
    users = []
    for path in _src_files():
        found = flat_ocp_imports(_parse(path))
        if found:
            users.append(os.path.relpath(path, ROOT))
        pairs |= found
    # loose bracket: the scan sees several OCP users and many names
    assert len(users) >= 3, users
    assert 20 <= len(pairs) <= 1000, len(pairs)
    assert unresolvable(pairs) == []


# --------------------------------------------------------------------------
# the TopoDS downcasts src calls exist on the flat TopoDS
# --------------------------------------------------------------------------

@needs_ocp
def test_topods_cast_checker_sees_aliases_and_rejects_unknown_casts():
    """Negative controls: unknown casts are reported under every binding
    the scanner knows (plain, ``as`` alias, module alias, dotted module);
    real casts under the same bindings are not."""
    source = ("from OCP.TopoDS import TopoDS\n"
              "from OCP.TopoDS import TopoDS as T\n"
              "import OCP.TopoDS as M\n"
              "import OCP.TopoDS\n"
              "TopoDS.Nope1_s(a); T.Nope2_s(b); M.TopoDS.Nope3_s(c)\n"
              "OCP.TopoDS.TopoDS.Nope4_s(d)\n"
              "TopoDS.Face_s(a); T.Shell_s(b); M.TopoDS.Edge_s(c)\n"
              "OCP.TopoDS.TopoDS.Vertex_s(d)\n")
    used = topods_attributes_used(ast.parse(source))
    assert used == {"Nope1_s", "Nope2_s", "Nope3_s", "Nope4_s",
                    "Face_s", "Shell_s", "Edge_s", "Vertex_s"}, used
    assert missing_topods_attributes(used) == [
        "Nope1_s", "Nope2_s", "Nope3_s", "Nope4_s"]


@needs_ocp
def test_flat_topods_has_every_downcast_src_calls():
    """``src`` calls ``TopoDS.Face_s`` etc.; on cadquery-ocp 7.9 those exist
    only because the flat ``OCP.TopoDS`` wrapper adds them, which is why the
    code must not reach the native ``OCP.OCP.TopoDS``."""
    used = set()
    for path in _src_files():
        used |= topods_attributes_used(_parse(path))
    assert len(used) >= 2, used          # the scan sees the downcast calls
    assert missing_topods_attributes(used) == []
