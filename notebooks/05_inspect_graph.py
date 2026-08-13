# %%
# Inspect the sliding-window transaction graph: confirm eviction actually
# works, confirm degree features make smurfing mules/collectors visibly
# distinctive, and benchmark day-by-day construction over the full dataset.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data_gen import generate_accounts, generate_legit_transactions, inject_all_patterns
from features import build_account_day_table, compute_degree_features, compute_profile_features, SlidingGraph
from rules import run_rule_engine

SEED = 42
N_ACCOUNTS = 5_000
N_DAYS = 365
WINDOW = 60
BASE_DATE = pd.Timestamp("2024-01-01")

pd.set_option("display.width", 200)

# %%
# --- Quick correctness check: does the sliding window actually drop old edges? ---
g = SlidingGraph(window=30)
g.add_edge(day=0, sender="A", receiver="B", amount=100.0)
g.advance_to(0)
assert g.out_degree("A") == 1 and "B" in g.out_neighbors("A")

g.advance_to(30)  # cutoff = 30-30 = 0; edge_day 0 is not *older than* cutoff 0, so still kept
assert g.out_degree("A") == 1

g.advance_to(31)  # cutoff = 31-30 = 1; edge_day 0 < 1, now evicted
assert g.out_degree("A") == 0
assert g.in_degree("B") == 0
print("eviction test passed: edge added at day 0 is gone once day - window > 0")

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
print(f"accounts: {len(accounts):,}, full txns: {len(full_txns):,}, generated in {time.time() - t0:.1f}s")

# %%
# Stage 5 sampled 3 random true-positive account-days across *all* four
# typologies. Degree features are specifically a smurfing signature (hub-
# and-spoke) -- layering and round-tripping are chains/loops with low
# degree throughout, and fan-out/fan-in's degree is diffuse across many
# intermediates rather than concentrated. So a random draw across all types
# would often miss the effect entirely. Instead, pick smurfing mules and
# the collector specifically, on the actual day their pattern transaction
# happened (their alerted day, since that's what made them label=True).
#
# The "legit" comparison group also needs care: Stage 4's rules fire on
# very close to 100% of high_volume_merchant accounts eventually (they're
# naturally high-degree/high-volume), so a random sample of alerted-but-
# legit accounts is ~85% merchants -- comparing a mule/collector against
# that population is comparing it against the highest-degree nodes in the
# whole graph, which isn't the intended comparison. Restrict the legit
# comparison group to non-merchant personas.
alerts = run_rule_engine(full_txns)
account_day = build_account_day_table(full_txns, alerts)
profiles = compute_profile_features(account_day, full_txns)

smurfing_roles = involved[(involved["pattern_type"] == "smurfing") & (involved["role"].isin(["mule", "collector"]))]
smurfing_alerted = profiles.merge(smurfing_roles[["account_id", "role"]], on="account_id").loc[lambda d: d["label"]]

legit_alerted = profiles.loc[~profiles["label"]].merge(accounts[["account_id", "persona"]], on="account_id")
non_merchant_legit = legit_alerted[legit_alerted["persona"] != "high_volume_merchant"]

collector_row = smurfing_alerted[smurfing_alerted["role"] == "collector"].sample(1, random_state=SEED)
mule_rows = smurfing_alerted[smurfing_alerted["role"] == "mule"].sample(2, random_state=SEED)
injected_sample = pd.concat([collector_row, mule_rows])[["account_id", "date"]]
legit_sample = non_merchant_legit[["account_id", "date"]].sample(3, random_state=SEED)

sample_rows = pd.concat(
    [injected_sample.assign(group="injected"), legit_sample.assign(group="legit")], ignore_index=True
)
sample_rows["day"] = (pd.to_datetime(sample_rows["date"]) - BASE_DATE).dt.days
print(sample_rows)

# %%
# For the headline 3-vs-3 comparison, also capture a *larger* sample of
# every smurfing mule/collector and 200 non-merchant legit accounts, so the
# "visibly distinctive" check rests on distributions, not 3 possibly-noisy
# examples.
big_legit_sample = non_merchant_legit[["account_id", "date"]].sample(
    min(200, len(non_merchant_legit)), random_state=SEED
)
targets_all = pd.concat(
    [
        smurfing_alerted[["account_id", "date", "role"]].assign(group=lambda d: d["role"]),
        big_legit_sample.assign(group="legit"),
    ],
    ignore_index=True,
)
targets_all["day"] = (pd.to_datetime(targets_all["date"]) - BASE_DATE).dt.days

# %%
# --- Build the sliding graph day by day, capturing degree features for
# every target (account_id, date) exactly as we pass through that day, and
# benchmark the whole 365-day sweep. ---
full_txns["day"] = (full_txns["timestamp"].dt.normalize() - BASE_DATE).dt.days
targets_by_day = targets_all.groupby("day")["account_id"].apply(list).to_dict()

graph = SlidingGraph(window=WINDOW)
captured = {}

t0 = time.time()
for day, day_txns in full_txns.groupby("day"):
    for row in day_txns.itertuples():
        graph.add_edge(day, row.sender_id, row.receiver_id, row.amount)
    graph.advance_to(day)
    if day in targets_by_day:
        for account_id in targets_by_day[day]:
            captured[(account_id, day)] = compute_degree_features(graph, account_id)
elapsed = time.time() - t0

n_days_covered = int(full_txns["day"].max()) + 1
print(f"built sliding graph over {n_days_covered} days, {len(full_txns):,} txns, in {elapsed:.2f}s")
print(f"under one minute: {elapsed < 60}")
assert elapsed < 60, "sliding graph construction should be well under a minute for ~500k txns"

# %%
# --- Headline comparison: the 3 sampled injected accounts vs. 3 sampled legit ---
feat_rows = []
for _, r in sample_rows.iterrows():
    feats = captured[(r["account_id"], r["day"])]
    feat_rows.append({"group": r["group"], "account_id": r["account_id"], "date": r["date"], **feats})

feat_df = pd.DataFrame(feat_rows).set_index(["group", "account_id"])
display_cols = [
    "in_degree", "out_degree",
    "neighbor_in_degree_mean", "neighbor_in_degree_max",
    "neighbor_weighted_in_degree_max",
    "weighted_in_degree", "weighted_out_degree",
]
print(feat_df[display_cols].T.to_string())

# %%
# --- Distributional comparison: collector vs. mule vs. non-merchant legit,
# across ALL smurfing instances and 200 legit accounts (not just 3 of each) ---
all_rows = []
for _, r in targets_all.iterrows():
    feats = captured.get((r["account_id"], r["day"]))
    if feats is not None:
        all_rows.append({"group": r["group"], **feats})
all_df = pd.DataFrame(all_rows)

print(f"\nn per group: {all_df['group'].value_counts().to_dict()}")
print("\nmedian degree features by group (collector = pattern hub, mule = pays the collector):")
summary_cols = ["in_degree", "weighted_in_degree", "out_degree", "weighted_out_degree", "neighbor_in_degree_max"]
print(all_df.groupby("group")[summary_cols].median().T.to_string())

print()
print(
    "The collector's own in_degree/weighted_in_degree are the clean, dramatic signal here "
    "(it directly receives from ~15 mules in a short window). A single mule's own degree is "
    "far less distinctive -- it touches the pattern with exactly one transaction, diluted into "
    "60 days of otherwise-normal activity, and its 'neighbor max degree' is usually dominated "
    "by one of its own legitimate regular counterparties happening to be a high-volume merchant "
    "(Stage 2 deliberately makes merchants popular hub counterparties for everyone, mules "
    "included). Pure local degree is good at flagging the hub; catching the *spokes* is what "
    "motivates a propagation-based feature (Stage 9's guilty walker) rather than local degree "
    "alone."
)
