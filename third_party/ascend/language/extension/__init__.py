import os
import sys
import types
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

# ---------------------------------------------------------------------------
# Path resolution
#
# ``__file__`` always points to the *installed* location of this module:
#   editable  → …/third_party/ascend/language/extension/__init__.py
#   installed → …/site-packages/triton/language/extra/extension/__init__.py
# In both cases ``os.path.dirname(__file__)`` contains the ``buffer/``
# subdirectory, so the buffer-language redirect works everywhere.
#
# The runtime modules live at ``third_party/ascend/runtime/``, which is
# NOT installed to site-packages.  We locate them by walking up from
# ``triton.__path__[0]`` (editable) or scanning ``sys.path`` (installed).
# ---------------------------------------------------------------------------
_EXTENSION_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_runtime_dir():
    # 1) editable: triton.__path__[0] = …/triton-ascend/python/triton
    triton_pkg = sys.modules.get("triton")
    if triton_pkg is not None and hasattr(triton_pkg, "__path__"):
        root = os.path.dirname(os.path.dirname(os.path.abspath(triton_pkg.__path__[0])))
        candidate = os.path.join(root, "third_party", "ascend", "runtime")
        if os.path.isdir(candidate):
            return candidate

    # 2) installed: search sys.path for a checkout that has third_party/ascend/runtime/
    for p in sys.path:
        candidate = os.path.join(p, "third_party", "ascend", "runtime")
        if os.path.isdir(candidate):
            return candidate

    # 3) absolute fallback (editable without __path__ set yet)
    return os.path.normpath(os.path.join(_EXTENSION_DIR, "..", "..", "..", "runtime"))


_RUNTIME_DIR = _find_runtime_dir()

# ---------------------------------------------------------------------------
# Backward-compat import hooks
#
#   Old import                      → target
#   ──────────────────────────────    ────────────────────────────────────────
#   triton.extension.buffer.language  triton.language.extra.extension.buffer.language
#   triton.runtime.code_cache         triton.ascend_runtime.code_cache
#   triton.runtime.libentry           triton.ascend_runtime.libentry
# ---------------------------------------------------------------------------

_SYNTHETIC = {"triton.extension", "triton.extension.buffer"}

_REDIRECTS: list[tuple[str, str, bool]] = [
    (
        "triton.extension.buffer.language",
        "triton.language.extra.extension.buffer.language",
        True,
    ),
    (
        "triton.runtime.code_cache",
        "triton.ascend_runtime.code_cache",
        False,
    ),
    (
        "triton.runtime.libentry",
        "triton.ascend_runtime.libentry",
        False,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_synth(name, path=None):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__package__ = name
    mod.__path__ = path if path is not None else []
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


class _RedirectLoader(Loader):

    def __init__(self, target):
        self._target = target

    def create_module(self, spec):
        import importlib
        return importlib.import_module(self._target)

    def exec_module(self, module):
        pass


class _CompatFinder(MetaPathFinder):
    _map = {old: (new, pkg) for old, new, pkg in _REDIRECTS}

    def find_spec(self, fullname, path, target=None):
        if fullname in _SYNTHETIC:
            _make_synth(fullname)
            return ModuleSpec(fullname, None, is_package=True)

        pair = self._map.get(fullname)
        if pair is None:
            return None
        target_name, is_pkg = pair
        return ModuleSpec(fullname, _RedirectLoader(target_name), is_package=is_pkg)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
def _install():
    for f in sys.meta_path:
        if isinstance(f, _CompatFinder):
            return

    _make_synth("triton.language.extra.extension", path=[_EXTENSION_DIR])
    _make_synth("triton.ascend_runtime", path=[_RUNTIME_DIR])
    for pkg in _SYNTHETIC:
        _make_synth(pkg)

    sys.meta_path.insert(0, _CompatFinder())


_install()
