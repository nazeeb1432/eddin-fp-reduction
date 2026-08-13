"""Incremental sliding-window transaction graph.

SlidingGraph keeps only the last `window` days of edges in memory and
supports fast in/out adjacency lookups without ever rebuilding a networkx
graph. It is fed one edge at a time (in non-decreasing day order, as a
chronological transaction ledger naturally is) and advanced day by day.

Eviction is O(1) amortized per edge, not O(n) per day: edges are appended
to a global deque and to per-node deques in the same relative order, so
expiring edges just means popping off the front of a few deques until the
cutoff is reached, regardless of how many nodes or edges are outside the
window.

window is a plain integer window (in days). To support different windows
per node type (Stage 9), instantiate multiple SlidingGraph objects -- each
is independent and self-contained, so nothing here needs to change to
support that later.
"""

from __future__ import annotations

from collections import defaultdict, deque


class SlidingGraph:
    def __init__(self, window: int):
        self.window = window
        self.current_day: int | None = None

        # Global edge queue in insertion (day) order -- the source of truth
        # for what's still "in window".
        self._edges: deque[tuple[int, str, str, float]] = deque()

        # dict of deques, not a networkx graph: out_adj[node] / in_adj[node]
        # hold (day, neighbor, amount) tuples in the same relative order as
        # self._edges, which is what makes O(1) amortized eviction possible.
        self.out_adj: dict[str, deque[tuple[int, str, float]]] = defaultdict(deque)
        self.in_adj: dict[str, deque[tuple[int, str, float]]] = defaultdict(deque)

    def add_edge(self, day: int, sender: str, receiver: str, amount: float) -> None:
        if self._edges and day < self._edges[-1][0]:
            raise ValueError(
                f"add_edge called out of order: day={day} is before the last-added day "
                f"{self._edges[-1][0]}. Edges must be added in non-decreasing day order."
            )
        self._edges.append((day, sender, receiver, amount))
        self.out_adj[sender].append((day, receiver, amount))
        self.in_adj[receiver].append((day, sender, amount))

    def advance_to(self, day: int) -> None:
        """Drop every edge older than (day - window) from all adjacency
        structures. Edges with day == (day - window) are kept (the window
        is inclusive of its start)."""
        self.current_day = day
        cutoff = day - self.window
        while self._edges and self._edges[0][0] < cutoff:
            edge_day, sender, receiver, amount = self._edges.popleft()
            self.out_adj[sender].popleft()
            self.in_adj[receiver].popleft()

    def out_neighbors(self, node: str) -> set[str]:
        return {neighbor for _, neighbor, _ in self.out_adj.get(node, ())}

    def in_neighbors(self, node: str) -> set[str]:
        return {neighbor for _, neighbor, _ in self.in_adj.get(node, ())}

    def out_degree(self, node: str) -> int:
        """Number of distinct out-neighbors currently in window."""
        return len(self.out_neighbors(node))

    def in_degree(self, node: str) -> int:
        """Number of distinct in-neighbors currently in window."""
        return len(self.in_neighbors(node))

    def out_weight(self, node: str) -> float:
        """Sum of amounts on this node's currently-in-window outgoing edges."""
        return sum(amount for _, _, amount in self.out_adj.get(node, ()))

    def in_weight(self, node: str) -> float:
        """Sum of amounts on this node's currently-in-window incoming edges."""
        return sum(amount for _, _, amount in self.in_adj.get(node, ()))

    def n_edges(self) -> int:
        return len(self._edges)
