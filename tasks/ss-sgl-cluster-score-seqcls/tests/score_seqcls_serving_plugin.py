# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier-only collision-safe port binding for the pooled-hidden-states HTTP E2E.

Loaded explicitly by ``run_upstream_e2e_group.py`` (never as a conftest) and only
for the group whose file basename is ``test_pooled_hidden_states.py``. It adapts
the vendored maintainer file at COLLECTION time, without editing a byte of it.

``TestPooledHiddenStatesHTTP.setUpClass`` reads the module-global
``DEFAULT_URL_FOR_TEST`` (imported from ``sglang.test.test_utils``) to launch and
address its own server. That global is a fixed port on the shared host network
namespace; a co-located trial can occupy it so a hijacked ``/health`` answers for
another server (the verifier standard proved this cross-talk). This plugin rebinds the module
global to a genuinely free host TCP port from SGLang's own
``sglang.test.test_utils.find_available_port`` before the class sets up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SERVING_MODULE_BASENAME = "test_pooled_hidden_states.py"


def _free_url() -> str:
    from sglang.test.test_utils import find_available_port

    return f"http://127.0.0.1:{find_available_port(21000)}"


def apply_bindings(module: Any) -> str:
    """Rebind the serving module's DEFAULT_URL_FOR_TEST to a free port.

    Returns the bound URL so a regression check can assert it is dynamic.
    """
    url = _free_url()
    module.DEFAULT_URL_FOR_TEST = url
    return url


def pytest_collection_modifyitems(items: list[Any]) -> None:
    modules = {
        item.module
        for item in items
        if Path(str(item.path)).name == SERVING_MODULE_BASENAME
    }
    if not modules:
        return
    if len(modules) != 1:
        raise RuntimeError(
            f"expected exactly one pooled-hidden-states serving module, got {len(modules)}"
        )
    apply_bindings(modules.pop())
