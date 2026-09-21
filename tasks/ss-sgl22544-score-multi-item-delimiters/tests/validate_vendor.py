#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed attestation of the vendored offline wheels (no network permitted)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor"

# name -> sha256; the exact bytes installed offline by prep.sh.
EXPECTED = {
    "distro-1.9.0-py3-none-any.whl": "7bffd925d65168f85027d8da9af6bddab658135b840670a223589bc0c8ef02b2",
}


def main() -> int:
    present = {p.name for p in VENDOR.glob("*.whl")}
    if present != set(EXPECTED):
        raise SystemExit(f"VERIFIER ERROR: vendored wheel set drift: {sorted(present)} != {sorted(EXPECTED)}")
    for name, want in EXPECTED.items():
        got = hashlib.sha256((VENDOR / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"VERIFIER ERROR: vendored wheel hash drift for {name}: {got} != {want}")
    print(f"validated {len(EXPECTED)} pinned offline wheel(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
