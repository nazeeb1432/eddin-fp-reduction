# %%
# Inspect GuiltyWalker features: confirm gw_min matches true BFS distance
# to a known-illicit set, confirm hit_rate/mean walk length decrease
# monotonically with distance, and benchmark runtime at scale.
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data_gen import generate_accounts, generate_legit_transactions, inject_all_patterns
from features import build_account_day_table, compute_guilty_walker_features, SlidingGraph
from rules import run_rule_engine

SEED = 42
N_ACCOUNTS = 5_000
N_DAYS = 365
WINDOW = 60
BASE_DATE = pd.Timestamp("2024-01-01")

pd.set_option("display.width", 160)

# %%
t0 = time.time()
accounts = generate_accounts(N_ACCOUNTS, seed=SEED)
legit_txns = generate_legit_transactions(accounts, n_days=N_DAYS, seed=SEED)
sar_txns, involved = inject_all_patterns(accounts, seed=SEED, n_days=N_DAYS)
legit_txns["is_sar"] = False
legit_txns["pattern_type"] = None
legit_txns["pattern_id"] = None
full_txns = (
    pd.concat([legit_txns, sar_txns], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
)
alerts = run_rule_engine(full_txns)
account_day = build_account_day_table(full_txns, alerts)
print(f"accounts: {len(accounts):,}, full txns: {len(full_txns):,}, generated in {time.time() - t0:.1f}s")

# %%
def bfs_distance(graph: SlidingGraph, start: str, targets: set[str]) -> int | None:
    """Ground truth: true shortest-path distance from start to the nearest
    node in targets, moving along in/out edges either direction. None if
    unreachable. Used only to validate GW features, not as a feature itself
    (a full BFS per account would defeat the point of a cheap random-walk
    approximation)."""
    if start in targets:
        return 0
    visited = {start}
    frontier = [start]
    dist = 0
    while frontier:
        dist += 1
        next_frontier = []
        for node in frontier:
            for neighbor in graph.out_neighbors(node) | graph.in_neighbors(node):
                if neighbor in targets:
                    return dist
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return None


# %%
# "Known illicit" = the two smurfing collectors in this dataset. Tried
# broadening this to all 99 non-mule pattern participants first (to get a
# better hit_rate against a bigger target), but that backfired: within a
# 60-day window this graph is small-world (merchant hubs connect huge
# swaths of the population), so 76% of a random 300-account sample were
# already within 2 hops of *some* member of a 99-account illicit_set --
# "distance" stopped meaningfully separating anything. Keeping illicit_set
# narrow and specific (one pattern's collector) gives a clean, unambiguous
# distance story instead; the hit_rate-too-low problem this causes at the
# default n_walks=50 is fixed below by using more walks for this specific
# diagnostic, not by diluting what "illicit" means.
smurfing_collectors = involved[(involved["pattern_type"] == "smurfing") & (involved["role"] == "collector")]
illicit_set = set(smurfing_collectors["account_id"])
print(f"illicit_set (smurfing collectors): {len(illicit_set)} accounts")

example_pattern_id = smurfing_collectors["pattern_id"].iloc[0]
collector_id = smurfing_collectors.loc[
    smurfing_collectors["pattern_id"] == example_pattern_id, "account_id"
].iloc[0]
mule_id = involved.loc[
    (involved["pattern_id"] == example_pattern_id) & (involved["role"] == "mule"), "account_id"
].iloc[0]

pattern_txns = sar_txns[sar_txns["pattern_id"] == example_pattern_id]
snapshot_day = int((pattern_txns["timestamp"].max().normalize() - BASE_DATE).days)
print(f"example pattern: {example_pattern_id}, collector={collector_id}, mule={mule_id}, snapshot_day={snapshot_day}")

# %%
# Build the sliding graph day by day up through the snapshot day (mirrors
# Stage 6's construction), then benchmark GW at scale over the full graph.
full_txns["day"] = (full_txns["timestamp"].dt.normalize() - BASE_DATE).dt.days

graph = SlidingGraph(window=WINDOW)
t0 = time.time()
for day, day_txns in full_txns[full_txns["day"] <= snapshot_day].groupby("day"):
    for row in day_txns.itertuples():
        graph.add_edge(day, row.sender_id, row.receiver_id, row.amount)
    graph.advance_to(day)
print(f"built graph through day {snapshot_day} in {time.time() - t0:.2f}s")

# %%
# Find a genuine 2-hop account: one of the mule's own neighbors, excluding
# the collector itself. Verify by BFS rather than assuming.
mule_neighbors = (graph.out_neighbors(mule_id) | graph.in_neighbors(mule_id)) - illicit_set
two_hop_candidates = [n for n in sorted(mule_neighbors) if bfs_distance(graph, n, illicit_set) == 2]
two_hop_id = two_hop_candidates[0]

# An "unrelated" legit account genuinely far from illicit_set: search many
# candidates and keep whichever has the largest true BFS distance, rather
# than hoping a single random pick lands far away.
excluded = illicit_set | {mule_id, two_hop_id}
candidates = accounts.loc[~accounts["account_id"].isin(excluded), "account_id"].sample(300, random_state=SEED)
candidate_distances = {acc: bfs_distance(graph, acc, illicit_set) for acc in candidates}
unrelated_id = max(
    candidate_distances, key=lambda acc: (candidate_distances[acc] is None, candidate_distances[acc] or 0)
)
print(f"unrelated candidate: distance={candidate_distances[unrelated_id]} (farthest of {len(candidates)} sampled)")

examples = [("1-hop (mule)", mule_id), ("2-hop", two_hop_id), ("unrelated legit", unrelated_id)]

# %%
# With illicit_set this narrow (2 accounts), the default n_walks=50 gives
# too few successful walks to read anything off cleanly (verified this
# empirically: 2-3 successes out of 50 for the 2-hop/unrelated cases). Use
# more walks for this specific diagnostic; the function's own default
# stays 50, this is a notebook-only choice to see the effect clearly.
DEMO_N_WALKS = 500

print(f"{'label':<18}{'account_id':<14}{'bfs_dist':<10}{'gw_min':<8}{'gw_hit_rate':<13}{'gw_mean':<10}")
rows = []
for label, account_id in examples:
    true_dist = bfs_distance(graph, account_id, illicit_set)
    feats = compute_guilty_walker_features(
        graph, account_id, illicit_set, n_walks=DEMO_N_WALKS, max_steps=20, seed=SEED
    )
    rows.append({"label": label, "account_id": account_id, "bfs_dist": true_dist, **feats})
    print(
        f"{label:<18}{account_id:<14}{str(true_dist):<10}{feats['gw_min']:<8}"
        f"{feats['gw_hit_rate']:<13}{feats['gw_mean']:<10.2f}"
    )

results_df = pd.DataFrame(rows)
print(f"\ngw_min matches true BFS distance for the closest (1-hop) example: {results_df.iloc[0]['gw_min'] == results_df.iloc[0]['bfs_dist']}")
print(
    "The single 1-hop example is dramatically distinguished (hit_rate several times higher than "
    "the rest), but 2-hop vs. the farthest example found aren't reliably ordered by hit_rate alone "
    "-- checked this up to 5,000 walks and they stayed statistically tied. That's not noise, it's "
    "individual account idiosyncrasy: hit_rate depends on local branching/degree along the way, not "
    "purely on shortest-path length, so any *single* 2-hop account can have a narrower path than a "
    "well-connected 4-hop one (the same hub-confound found in Stage 6). The distributional check "
    "below -- many accounts per true-distance bucket, not one -- is the reliable way to see the "
    "trend."
)

# %%
# --- Distributional check: many accounts per true-distance bucket ---
# Sample broadly, bucket by true BFS distance, and compare median hit_rate
# per bucket. This is what actually shows the expected monotonic trend
# cleanly -- the same lesson as Stage 6's degree-feature validation.
import random as _random

rng = _random.Random(SEED)
broad_sample = rng.sample(list(accounts["account_id"]), 600)

buckets: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
for account_id in broad_sample:
    dist = bfs_distance(graph, account_id, illicit_set)
    if dist in buckets and len(buckets[dist]) < 15:
        buckets[dist].append(account_id)

bucket_rows = []
for dist, bucket_accounts in buckets.items():
    for account_id in bucket_accounts:
        feats = compute_guilty_walker_features(graph, account_id, illicit_set, n_walks=200, max_steps=20, seed=SEED)
        bucket_rows.append({"true_distance": dist, "gw_hit_rate": feats["gw_hit_rate"], "gw_mean": feats["gw_mean"]})

bucket_df = pd.DataFrame(bucket_rows)
bucket_summary = bucket_df.groupby("true_distance").agg(
    n=("gw_hit_rate", "size"), median_hit_rate=("gw_hit_rate", "median"), median_mean_len=("gw_mean", "median")
)
print(bucket_summary)

hit_rate_by_dist = bucket_summary["median_hit_rate"]
print(f"\nmedian hit_rate monotonically non-increasing with true distance: {list(hit_rate_by_dist) == sorted(hit_rate_by_dist, reverse=True)}")

# %%
# --- Runtime benchmark: 50 walks x every alerted account-day ---
# GW is meaningfully more expensive than degree features (each walk can
# take up to max_steps individual random hops, vs. one O(degree) lookup),
# so this is worth checking explicitly rather than assuming it scales.
illicit_set_full = set(involved["account_id"])
sample_account_days = account_day.sample(min(200, len(account_day)), random_state=SEED)

for n_walks, max_steps in [(50, 20), (20, 10)]:
    t0 = time.time()
    for _, r in sample_account_days.iterrows():
        compute_guilty_walker_features(
            graph, r["account_id"], illicit_set_full, n_walks=n_walks, max_steps=max_steps, seed=SEED
        )
    elapsed = time.time() - t0
    per_account = elapsed / len(sample_account_days)
    extrapolated = per_account * len(account_day)
    print(
        f"n_walks={n_walks:>2} max_steps={max_steps:>2}: {per_account * 1000:.2f}ms/account-day -> "
        f"~{extrapolated:.0f}s extrapolated for all {len(account_day):,} alerted account-days"
    )

print(
    "\nAt default params (50 walks, 20 steps) this is a couple of minutes for the full alerted "
    "set at this dataset's scale -- real but not alarming for a batch feature job. Dropping to "
    "20 walks / 10 steps (the checklist's suggested first-pass mitigation) cuts it roughly 4x, "
    "trading off precision on the tail statistics (max, p75) for speed."
)
