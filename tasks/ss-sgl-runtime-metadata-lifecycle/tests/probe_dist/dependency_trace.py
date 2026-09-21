# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small directed graph for verifier-owned CUDA dependency observations."""

from __future__ import annotations


class DependencyTrace:
    """Record stream order and event record/wait edges without wall-clock timing."""

    def __init__(self) -> None:
        self._next_node = 0
        self._edges: dict[int, set[int]] = {}
        self._stream_tail: dict[str, int] = {}
        self._event_record: dict[int, int] = {}
        self._markers: dict[tuple[str, int], int] = {}
        self.unresolved_waits: list[int] = []

    def _operation(
        self,
        stream: str,
        *,
        marker: tuple[str, int] | None = None,
    ) -> int:
        node = self._next_node
        self._next_node += 1
        self._edges[node] = set()
        previous = self._stream_tail.get(stream)
        if previous is not None:
            self._edges[previous].add(node)
        self._stream_tail[stream] = node
        if marker is not None:
            if marker in self._markers:
                raise RuntimeError(f"duplicate dependency marker: {marker!r}")
            self._markers[marker] = node
        return node

    def record_event(
        self,
        event: int,
        stream: str,
        *,
        marker: tuple[str, int] | None = None,
    ) -> int:
        node = self._operation(stream, marker=marker)
        self._event_record[event] = node
        return node

    def wait_event(self, event: int, stream: str) -> int:
        node = self._operation(stream)
        record = self._event_record.get(event)
        if record is None:
            self.unresolved_waits.append(node)
        else:
            self._edges[record].add(node)
        return node

    def marker(self, kind: str, iteration: int) -> int | None:
        return self._markers.get((kind, iteration))

    def has_path(self, start: int, destination: int) -> bool:
        if start == destination:
            return True
        pending = [start]
        visited = {start}
        while pending:
            node = pending.pop()
            for successor in self._edges.get(node, ()):
                if successor == destination:
                    return True
                if successor not in visited:
                    visited.add(successor)
                    pending.append(successor)
        return False

    @property
    def node_count(self) -> int:
        return self._next_node

    @property
    def edge_count(self) -> int:
        return sum(len(successors) for successors in self._edges.values())
