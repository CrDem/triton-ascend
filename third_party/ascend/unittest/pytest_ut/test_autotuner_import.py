# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import inspect

# ---------------------------------------------------------------------------
# Module-level check — runs the moment this file is imported (i.e. during
# test collection), *before* any other test module in this directory has
# a chance to trigger ``_patch_autotune()``.
#
# ``_patch_autotune()`` (``backend/runtime/__init__.py``) replaces
# ``triton.autotune`` with an ascend variant that lacks ``cache_results``.
# Capturing the signature here ensures we see the true import-time state.
# ---------------------------------------------------------------------------
from triton.runtime.libentry import libentry  # noqa: F401  exercises the redirect hook
import triton

_sig = inspect.signature(triton.autotune)
_COMMUNITY_AUTOTUNE = "cache_results" in _sig.parameters


def test_community_autotune_has_cache_results():
    """Fail if ``triton.autotune`` has been replaced by an ascend variant
    that does not accept ``cache_results``.

    When ``_backcompat_imports.py`` redirects ``triton.runtime.libentry``
    to ``triton.backends.ascend.runtime.libentry``, the ascend backend's
    ``runtime/__init__.py`` calls ``_patch_autotune()`` which replaces
    ``triton.autotune``.  The ascend variant lacks ``cache_results``,
    so this test catches the regression.

    The signature is captured at module import time so the check is
    immune to execution-order side-effects from other tests that also
    trigger the patch (e.g. ``test_runtime_utils.py`` in parallel runs).
    """
    assert _COMMUNITY_AUTOTUNE, ("triton.autotune was replaced — 'cache_results' is missing. "
                                 f"Signature params: {sorted(_sig.parameters.keys())}"
                                 f"TypeError: autotune() got an unexpected keyword argument 'cache_results'")
