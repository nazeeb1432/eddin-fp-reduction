"""Hand-traceable tests for compute_guilty_walker_features.

Graph (9 nodes), built once and shared by all tests:

    T1 -- ILLICIT                          (distance 1, T1's only edge)
    START -- A -- B -- C -- ILLICIT        (distance 4, a plain chain)
    D -- E                                 (unrelated, padding)

Plus a node that never appears in any edge, standing in for a fully
isolated account (the all-walks-fail case).
"""

from __future__ import annotations

import math

from features.graph import SlidingGraph
from features.guilty_walker import compute_guilty_walker_features

ILLICIT_SET = {"ILLICIT"}


def _build_graph() -> SlidingGraph:
    graph = SlidingGraph(window=1000)
    edges = [
        ("T1", "ILLICIT"),
        ("START", "A"),
        ("A", "B"),
        ("B", "C"),
        ("C", "ILLICIT"),
        ("D", "E"),
    ]
    for sender, receiver in edges:
        graph.add_edge(day=0, sender=sender, receiver=receiver, amount=1.0)
    graph.advance_to(0)
    return graph


def test_direct_neighbor_is_fully_deterministic():
    """T1's only edge is directly to ILLICIT, so every walk is forced to
    reach it in exactly 1 step, regardless of the seed -- there's no other
    neighbor to choose."""
    graph = _build_graph()
    feats = compute_guilty_walker_features(
        graph, "T1", ILLICIT_SET, n_walks=50, max_steps=20, seed=0
    )
    assert feats["gw_min"] == 1.0
    assert feats["gw_max"] == 1.0
    assert feats["gw_mean"] == 1.0
    assert feats["gw_std"] == 0.0
    assert feats["gw_hit_rate"] == 1.0
    assert feats["gw_n_distinct_illicit"] == 1


def test_chain_shortest_path_matches_hand_computed_distance():
    """START -> A -> B -> C -> ILLICIT is a hand-countable 4-hop shortest
    path. gw_min can never be shorter than the true graph distance, and
    empirically (verified across many seeds during development) at least
    one of 50 walks with max_steps=20 always finds the shortest path on
    this small a graph, so gw_min == 4 is a safe exact assertion -- not
    just a lower bound.
    """
    graph = _build_graph()
    feats = compute_guilty_walker_features(
        graph, "START", ILLICIT_SET, n_walks=50, max_steps=20, seed=42
    )
    assert feats["gw_min"] == 4.0
    # hit_rate is genuinely random here (branching at every interior node
    # of the chain), so this is checked against the fixed seed=42 result,
    # not an independently hand-derived probability.
    assert feats["gw_hit_rate"] == 0.76
    assert feats["gw_n_distinct_illicit"] == 1
    assert feats["gw_max"] <= 20.0


def test_all_walks_fail_gracefully_on_isolated_node():
    """A node that never appears in any edge has no neighbors at all --
    every walk dead-ends on step 1. Must not crash, and must return NaN
    length stats with hit_rate/n_distinct_illicit at 0."""
    graph = _build_graph()
    feats = compute_guilty_walker_features(
        graph, "NEVER_SEEN", ILLICIT_SET, n_walks=50, max_steps=20, seed=0
    )
    assert feats["gw_hit_rate"] == 0.0
    assert feats["gw_n_distinct_illicit"] == 0
    for key in ("gw_min", "gw_max", "gw_mean", "gw_median", "gw_std", "gw_p25", "gw_p75"):
        assert math.isnan(feats[key])


def test_account_already_illicit_short_circuits():
    graph = _build_graph()
    feats = compute_guilty_walker_features(
        graph, "ILLICIT", ILLICIT_SET, n_walks=50, max_steps=20, seed=0
    )
    assert feats["gw_min"] == 0.0
    assert feats["gw_hit_rate"] == 1.0
    assert feats["gw_n_distinct_illicit"] == 1
