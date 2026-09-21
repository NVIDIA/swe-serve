# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail
export PYTHONPATH=/code/python
export HOME=/tmp
mkdir -p /tmp/.cache/flashinfer /tmp/.cache/torch_extensions 2>/dev/null || true
