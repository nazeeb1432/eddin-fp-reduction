"""Degree-based features over a SlidingGraph snapshot."""

from __future__ import annotations

import numpy as np

from features.graph import SlidingGraph


def compute_degree_features(graph: SlidingGraph, account_id: str) -> dict[str, float]:
    """Degree features for account_id as of the graph's current window.

    - in_degree, out_degree: count of distinct in/out neighbors.
    - weighted_in_degree, weighted_out_degree: summed in/out edge amounts.
    - neighbor_{in,out}_degree_{mean,min,max}: unweighted in/out-degree of
      account_id's 1-hop neighbors (union of in- and out-neighbors). This is
      what surfaces a smurfing mule: its own degree is tiny, but one of its
      neighbors (the collector) has a very high in-degree.
    - neighbor_weighted_{in,out}_degree_{mean,min,max}: same, using each
      neighbor's weighted in/out-degree (summed amounts) instead of count.

    All neighbor aggregates are NaN when account_id has no 1-hop neighbors
    in the current window.
    """
    neighbors = graph.out_neighbors(account_id) | graph.in_neighbors(account_id)

    neighbor_in_degrees = [graph.in_degree(n) for n in neighbors]
    neighbor_out_degrees = [graph.out_degree(n) for n in neighbors]
    neighbor_weighted_in = [graph.in_weight(n) for n in neighbors]
    neighbor_weighted_out = [graph.out_weight(n) for n in neighbors]

    features = {
        "in_degree": graph.in_degree(account_id),
        "out_degree": graph.out_degree(account_id),
        "weighted_in_degree": graph.in_weight(account_id),
        "weighted_out_degree": graph.out_weight(account_id),
    }
    features.update(_agg("neighbor_in_degree", neighbor_in_degrees))
    features.update(_agg("neighbor_out_degree", neighbor_out_degrees))
    features.update(_agg("neighbor_weighted_in_degree", neighbor_weighted_in))
    features.update(_agg("neighbor_weighted_out_degree", neighbor_weighted_out))
    return features


def _agg(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {f"{prefix}_mean": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }
