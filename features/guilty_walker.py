"""GuiltyWalker features: random-walk distance from an account to a set of
known-illicit accounts, on a SlidingGraph snapshot.

Also includes the delay-aware ("GWd") pipeline: in production, a recent
account-day's true SAR label isn't known yet -- investigators take days to
weeks to confirm an alert. GWd handles this by pseudo-labeling the unsettled
recent window with a model trained on older, already-labeled data, then
building a "hybrid" illicit set (real labels where they exist, pseudo
labels where they don't yet) for the walker to reference.
"""

from __future__ import annotations

import random

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.graph import SlidingGraph


def compute_guilty_walker_features(
    graph: SlidingGraph,
    account_id: str,
    illicit_set: set[str],
    n_walks: int = 50,
    max_steps: int = 20,
    seed: int | None = None,
) -> dict[str, float]:
    """Run n_walks random walks from account_id, moving along in/out edges
    in either direction (undirected adjacency), stopping early on hitting a
    node in illicit_set. A walk fails if it runs out of neighbors (dead
    end) or exhausts max_steps without reaching illicit_set.

    Returns gw_min/max/mean/median/std/p25/p75 of walk length among
    successful walks, gw_hit_rate (successful / n_walks), and
    gw_n_distinct_illicit (distinct illicit nodes reached across all
    walks). If account_id is itself in illicit_set, every walk trivially
    "succeeds" at length 0. If no walk succeeds, the length stats are NaN
    and gw_hit_rate/gw_n_distinct_illicit are 0 -- no crash.
    """
    if account_id in illicit_set:
        return _summarize([0] * n_walks, {account_id}, n_walks)

    rng = random.Random(seed)
    walk_lengths: list[int] = []
    reached_illicit: set[str] = set()

    for _ in range(n_walks):
        current = account_id
        for step in range(1, max_steps + 1):
            neighbors = graph.out_neighbors(current) | graph.in_neighbors(current)
            if not neighbors:
                break
            # sorted(), not list(): set iteration order depends on string
            # hash randomization (PYTHONHASHSEED), which varies between
            # process runs. Without sorting, the same `seed` would pick a
            # different node across runs, breaking reproducibility.
            current = rng.choice(sorted(neighbors))
            if current in illicit_set:
                walk_lengths.append(step)
                reached_illicit.add(current)
                break

    return _summarize(walk_lengths, reached_illicit, n_walks)


def _summarize(walk_lengths: list[int], reached_illicit: set[str], n_walks: int) -> dict[str, float]:
    if not walk_lengths:
        return {
            "gw_min": np.nan,
            "gw_max": np.nan,
            "gw_mean": np.nan,
            "gw_median": np.nan,
            "gw_std": np.nan,
            "gw_p25": np.nan,
            "gw_p75": np.nan,
            "gw_hit_rate": 0.0,
            "gw_n_distinct_illicit": 0,
        }

    arr = np.array(walk_lengths, dtype=float)
    return {
        "gw_min": float(arr.min()),
        "gw_max": float(arr.max()),
        "gw_mean": float(arr.mean()),
        "gw_median": float(np.median(arr)),
        "gw_std": float(arr.std()),
        "gw_p25": float(np.percentile(arr, 25)),
        "gw_p75": float(np.percentile(arr, 75)),
        "gw_hit_rate": len(walk_lengths) / n_walks,
        "gw_n_distinct_illicit": len(reached_illicit),
    }


def train_stage1_scorer(train_a_df: pd.DataFrame, feature_cols: list[str]) -> lgb.LGBMClassifier:
    """Train a LightGBM scorer on train_a_df's profile + degree features
    (feature_cols must not include any guilty-walker feature -- GWd exists
    specifically to avoid needing an illicit_set to score recent activity,
    so the scorer that produces pseudo-labels can't depend on one either).
    """
    X = train_a_df[feature_cols]
    y = train_a_df["label"].astype(int)
    model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=5,
        num_leaves=31,
        class_weight="balanced",
        verbosity=-1,
        random_state=42,
    )
    model.fit(X, y)
    return model


def generate_pseudo_labels(
    stage1_model: lgb.LGBMClassifier, train_b_df: pd.DataFrame, delay_days: int, threshold: float
) -> pd.Series:
    """Score every account-day in train_b_df within the most recent
    delay_days window (relative to train_b_df's own latest date) using
    stage1_model, and label True where the predicted probability exceeds
    threshold.

    Returns a bool Series indexed like the recent-window subset of
    train_b_df (rows outside the window aren't included -- those are
    real_labels' job, in build_hybrid_illicit_set).
    """
    dates = pd.to_datetime(train_b_df["date"])
    cutoff = dates.max()
    window_start = cutoff - pd.Timedelta(days=delay_days)
    recent = train_b_df.loc[dates >= window_start]

    feature_cols = stage1_model.feature_name_
    scores = stage1_model.predict_proba(recent[feature_cols])[:, 1]
    return pd.Series(scores > threshold, index=recent.index, name="pseudo_label")


def build_hybrid_illicit_set(
    train_b_df: pd.DataFrame,
    real_labels: pd.Series,
    pseudo_labels: pd.Series,
    delay_days: int,
) -> set[str]:
    """Accounts flagged illicit in train_b_df, using real_labels for
    account-days older than delay_days from train_b_df's cutoff (settled,
    in a real deployment) and pseudo_labels for account-days within the
    delay window (not yet settled).
    """
    dates = pd.to_datetime(train_b_df["date"])
    cutoff = dates.max()
    window_start = cutoff - pd.Timedelta(days=delay_days)
    is_recent = dates >= window_start

    real_true = real_labels.reindex(train_b_df.index, fill_value=False).astype(bool)
    older_illicit = set(train_b_df.loc[~is_recent & real_true, "account_id"])

    pseudo_true = pseudo_labels.reindex(train_b_df.index, fill_value=False).astype(bool)
    recent_illicit = set(train_b_df.loc[is_recent & pseudo_true, "account_id"])

    return older_illicit | recent_illicit


def compute_guilty_walker_delay_features(
    graph: SlidingGraph,
    account_id: str,
    hybrid_illicit_set: set[str],
    n_walks: int = 50,
    max_steps: int = 20,
    seed: int | None = None,
) -> dict[str, float]:
    """Identical mechanics to compute_guilty_walker_features -- given a
    distinct name because hybrid_illicit_set mixes real and pseudo labels
    (Stage 8), not ground truth, and the notebook needs to tell GW and GWd
    features apart when comparing them.
    """
    return compute_guilty_walker_features(graph, account_id, hybrid_illicit_set, n_walks, max_steps, seed)
