"""Rule engine: scans the full (legit + injected) transaction set and
raises alerts the way a traditional, "dumb" AML monitoring system would --
each rule is a simple threshold, deliberately noisy, producing a high
false-positive rate. This is the baseline the triage model in models/ will
learn to clean up.

Each rule function returns a DataFrame of (account_id, date, rule_name)
hits. combine_alerts() merges all rule hits into one alerts-per-account-day
table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LARGE_TXN_THRESHOLD = 9_000.0
ROUND_AMOUNT_LOW = 8_500.0
ROUND_AMOUNT_HIGH = 9_999.0
VELOCITY_THRESHOLD = 5
SPIKE_MULTIPLIER = 5.0
SPIKE_WINDOW_DAYS = 30


def large_single_txn(txns: pd.DataFrame, threshold: float = LARGE_TXN_THRESHOLD) -> pd.DataFrame:
    """Sent amount in a single transaction > threshold."""
    hits = txns.loc[txns["amount"] > threshold, ["sender_id", "timestamp"]].copy()
    hits["date"] = hits["timestamp"].dt.date
    hits["rule_name"] = "large_single_txn"
    return hits.rename(columns={"sender_id": "account_id"})[["account_id", "date", "rule_name"]]


def round_amount_near_threshold(
    txns: pd.DataFrame, low: float = ROUND_AMOUNT_LOW, high: float = ROUND_AMOUNT_HIGH
) -> pd.DataFrame:
    """Any (sent) transaction landing just under a reporting threshold --
    the classic structuring fingerprint."""
    mask = (txns["amount"] >= low) & (txns["amount"] <= high)
    hits = txns.loc[mask, ["sender_id", "timestamp"]].copy()
    hits["date"] = hits["timestamp"].dt.date
    hits["rule_name"] = "round_amount_near_threshold"
    return hits.rename(columns={"sender_id": "account_id"})[["account_id", "date", "rule_name"]]


def high_daily_velocity(txns: pd.DataFrame, threshold: int = VELOCITY_THRESHOLD) -> pd.DataFrame:
    """Count of distinct counterparties (sent-to OR received-from) in a
    calendar day > threshold. Counting both directions is what lets this
    catch a smurfing collector -- its outgoing txn count is tiny, but it
    receives from many mules in a short window.
    """
    sent = txns[["sender_id", "receiver_id", "timestamp"]].rename(
        columns={"sender_id": "account_id", "receiver_id": "counterparty"}
    )
    received = txns[["receiver_id", "sender_id", "timestamp"]].rename(
        columns={"receiver_id": "account_id", "sender_id": "counterparty"}
    )
    both = pd.concat([sent, received], ignore_index=True)
    both["date"] = both["timestamp"].dt.date

    degree = both.groupby(["account_id", "date"])["counterparty"].nunique()
    hits = degree[degree > threshold].reset_index(name="n_counterparties")
    hits["rule_name"] = "high_daily_velocity"
    return hits[["account_id", "date", "rule_name"]]


def sudden_volume_spike(
    txns: pd.DataFrame,
    multiplier: float = SPIKE_MULTIPLIER,
    window_days: int = SPIKE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Daily sent amount > multiplier x the account's trailing window_days
    average daily sent amount, where the average is over the account's own
    *active* sending days in that window (not diluted by zero-fill on
    inactive calendar days). Averaging over calendar days instead was tried
    first and was a bad idea: for a low-frequency retail account, a single
    prior transaction spread over 30 mostly-inactive days produces a tiny
    baseline, so almost *every* transaction from almost every account reads
    as a "spike" -- a degenerate rule that discriminates nothing. Averaging
    over active days only gives a stable per-account baseline, so this rule
    flags genuine outlier days relative to an account's own typical
    transacting pattern. Accounts with no prior active day in the trailing
    window have an undefined (NaN) average and are naturally never flagged.
    """
    daily_sent = (
        txns.assign(date=txns["timestamp"].dt.normalize())
        .groupby(["sender_id", "date"])["amount"]
        .sum()
        .rename("daily_sent")
        .reset_index()
        .sort_values(["sender_id", "date"])
    )

    grouped = daily_sent.groupby("sender_id", group_keys=False)
    trailing_avg = grouped.apply(
        lambda g: g.set_index("date")["daily_sent"]
        .rolling(f"{window_days}D", closed="left")
        .mean(),
        include_groups=False,
    )
    daily_sent["trailing_avg"] = trailing_avg.to_numpy()

    is_spike = daily_sent["daily_sent"] > multiplier * daily_sent["trailing_avg"]

    hits = daily_sent.loc[is_spike, ["sender_id", "date"]].copy()
    hits["date"] = hits["date"].dt.date
    hits["rule_name"] = "sudden_volume_spike"
    return hits.rename(columns={"sender_id": "account_id"})[["account_id", "date", "rule_name"]]


RULES = {
    "large_single_txn": large_single_txn,
    "round_amount_near_threshold": round_amount_near_threshold,
    "high_daily_velocity": high_daily_velocity,
    "sudden_volume_spike": sudden_volume_spike,
}


def run_rule_engine(txns: pd.DataFrame) -> pd.DataFrame:
    """Run every rule over txns and combine hits into one alerts table.

    Returns a DataFrame with columns:
        account_id, date, triggered_rules (list[str]), any_true_positive
    any_true_positive is True iff the account appears as sender OR receiver
    in an is_sar=True transaction on that date.
    """
    hits = pd.concat([rule_fn(txns) for rule_fn in RULES.values()], ignore_index=True)

    alerts = (
        hits.groupby(["account_id", "date"])["rule_name"]
        .agg(list)
        .reset_index()
        .rename(columns={"rule_name": "triggered_rules"})
    )

    sar_days = _sar_account_days(txns)
    alerts["any_true_positive"] = list(
        zip(alerts["account_id"], alerts["date"])
    )
    alerts["any_true_positive"] = alerts["any_true_positive"].isin(sar_days)

    return alerts.sort_values(["date", "account_id"]).reset_index(drop=True)


def _sar_account_days(txns: pd.DataFrame) -> set[tuple[str, object]]:
    if "is_sar" not in txns.columns:
        return set()
    sar_txns = txns.loc[txns["is_sar"].fillna(False)]
    dates = sar_txns["timestamp"].dt.date
    sender_days = set(zip(sar_txns["sender_id"], dates))
    receiver_days = set(zip(sar_txns["receiver_id"], dates))
    return sender_days | receiver_days
