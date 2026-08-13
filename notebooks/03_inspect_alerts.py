# %%
# Inspect the rule engine: overall false-positive rate, which rules catch
# which injected typologies, and eyeball a few false-positive alerts to
# confirm they look like plausible legitimate activity rather than
# degenerate edge cases.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data_gen import generate_accounts, generate_legit_transactions, inject_all_patterns
from rules import RULES, run_rule_engine

SEED = 42
N_ACCOUNTS = 20_000
N_DAYS = 365

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
    pd.concat([legit_txns, sar_txns], ignore_index=True)
    .sort_values("timestamp")
    .reset_index(drop=True)
)
print(f"accounts: {len(accounts):,}, legit txns: {len(legit_txns):,}, sar txns: {len(sar_txns):,}")
print(f"generated in {time.time() - t0:.1f}s")

# %%
t0 = time.time()
alerts = run_rule_engine(full_txns)
print(f"rule engine ran in {time.time() - t0:.1f}s")

n_alerts = len(alerts)
n_true_positive = int(alerts["any_true_positive"].sum())
fp_rate = 1 - alerts["any_true_positive"].mean()

print(f"total alerts (account, date) pairs: {n_alerts:,}")
print(f"true-positive alerts: {n_true_positive:,} ({alerts['any_true_positive'].mean():.2%})")
print(f"false-positive rate: {fp_rate:.2%}  (target: ~90%+, ideally close to ~97%)")

# %%
# Per-rule breakdown: raw hit volume, unique (account, date) alerts, and how
# many distinct accounts each rule fires on.
print("=== per-rule volume ===")
for name, fn in RULES.items():
    hits = fn(full_txns).drop_duplicates(["account_id", "date"])
    print(
        f"{name:30s} alerts={len(hits):7,d}  unique_accounts={hits['account_id'].nunique():6,d}"
    )

print()
print("=== alerts by number of rules triggered together ===")
print(alerts["triggered_rules"].apply(len).value_counts().sort_index())

# %%
# Which rules catch which injected typologies? Build one row per
# (pattern_id, account_id, date) the pattern actually touches, then join
# against the alerts table.
sar_only = full_txns[full_txns["is_sar"] == True]  # noqa: E712
sender_side = sar_only[["pattern_id", "pattern_type", "sender_id", "timestamp"]].rename(
    columns={"sender_id": "account_id"}
)
receiver_side = sar_only[["pattern_id", "pattern_type", "receiver_id", "timestamp"]].rename(
    columns={"receiver_id": "account_id"}
)
pattern_days = pd.concat([sender_side, receiver_side], ignore_index=True)
pattern_days["date"] = pattern_days["timestamp"].dt.date
pattern_days = pattern_days.drop_duplicates(["pattern_id", "account_id", "date"])

pattern_alerts = pattern_days.merge(
    alerts[["account_id", "date", "triggered_rules"]], on=["account_id", "date"], how="left"
)
pattern_alerts["caught"] = pattern_alerts["triggered_rules"].notna()

print("=== fraction of pattern INSTANCES caught by at least one rule (target: 100%) ===")
caught_per_instance = pattern_alerts.groupby(["pattern_type", "pattern_id"])["caught"].any()
print(caught_per_instance.groupby("pattern_type").mean())

missed = caught_per_instance[~caught_per_instance]
if len(missed) > 0:
    print(f"\n{len(missed)} pattern instance(s) caught by NO rule -- inspect before trusting the % above:")
    for pattern_type, pattern_id in missed.index:
        instance_txns = sar_only[sar_only["pattern_id"] == pattern_id]
        print(f"  {pattern_id} ({pattern_type}): amounts = {instance_txns['amount'].tolist()}")

# %%
print("=== which rule(s) fire on which pattern type (count of caught account-days) ===")
caught_rows = pattern_alerts[pattern_alerts["caught"]].explode("triggered_rules")
print(pd.crosstab(caught_rows["pattern_type"], caught_rows["triggered_rules"]))

# %%
# Does high_daily_velocity specifically catch smurfing collectors, the
# accounts it was designed to catch (many mules paying in over a short
# window)?
collectors = set(involved.loc[involved["role"] == "collector", "account_id"])
collector_alerts = pattern_alerts[
    pattern_alerts["account_id"].isin(collectors) & pattern_alerts["caught"]
].explode("triggered_rules")
print("rules firing on smurfing collectors specifically:")
print(collector_alerts["triggered_rules"].value_counts())

# %%
# Sanity check: pull one false-positive alert per rule (where that rule was
# the *sole* trigger) and print the account's full transaction context for
# that day, to eyeball whether it looks like a plausible legitimate event
# (e.g. a one-off large payment) rather than a degenerate edge case.
false_positive_alerts = alerts[~alerts["any_true_positive"]]

for rule_name in RULES:
    only_this_rule = false_positive_alerts[
        false_positive_alerts["triggered_rules"].apply(lambda triggered: triggered == [rule_name])
    ]
    if only_this_rule.empty:
        print(f"--- no solo false positives found for {rule_name} ---\n")
        continue

    example = only_this_rule.sample(1, random_state=SEED).iloc[0]
    acc_id, date = example["account_id"], example["date"]
    persona = accounts.loc[accounts["account_id"] == acc_id, "persona"].iloc[0]

    print(f"--- false positive: {acc_id} ({persona}) on {date} -- solely triggered {rule_name} ---")
    day_txns = full_txns[
        (full_txns["timestamp"].dt.date == date)
        & ((full_txns["sender_id"] == acc_id) | (full_txns["receiver_id"] == acc_id))
    ]
    print(
        day_txns[["txn_id", "sender_id", "receiver_id", "amount", "timestamp", "direction"]]
        .sort_values("timestamp")
        .to_string(index=False)
    )
    print()
