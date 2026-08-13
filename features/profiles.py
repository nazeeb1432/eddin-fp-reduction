"""Account-day aggregation and rolling profile features.

build_account_day_table() collapses transactions down to one row per
alerted (account_id, date). compute_profile_features() then computes
rolling sent/received statistics over several trailing windows, using the
*full* transaction history (not just alerted activity) as lookback context
-- every window for a given (account_id, date) row only looks at data
timestamped on or before that date, so there is no lookahead leakage.
select_top_profile_features() reduces the resulting feature set down to the
smallest subset that explains most of a model's permutation importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

DEFAULT_WINDOWS = {"1d": 1, "1w": 7, "2w": 14, "1mo": 30, "2mo": 60}


def build_account_day_table(transactions_df: pd.DataFrame, alerts_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (account_id, date) for every account-day present in
    alerts_df, aggregating that day's activity (sent + received) out of
    transactions_df.

    Columns: account_id, date, total_sent, total_received, n_txns,
    counterparties (list[str]), direction_summary (dict[str, int]), label
    (True iff any transaction touching the account that day was is_sar=True).
    """
    txns = transactions_df.assign(date=transactions_df["timestamp"].dt.date)

    sent = txns.rename(columns={"sender_id": "account_id", "receiver_id": "counterparty"})
    sent["role"] = "sent"
    received = txns.rename(columns={"receiver_id": "account_id", "sender_id": "counterparty"})
    received["role"] = "received"
    both = pd.concat([sent, received], ignore_index=True)

    target_days = alerts_df[["account_id", "date"]].drop_duplicates()
    both = both.merge(target_days, on=["account_id", "date"], how="inner")

    if "is_sar" in both.columns:
        both["is_sar"] = both["is_sar"].fillna(False)
    else:
        both["is_sar"] = False

    grouped = both.groupby(["account_id", "date"])

    total_sent = both.loc[both["role"] == "sent"].groupby(["account_id", "date"])["amount"].sum()
    total_received = (
        both.loc[both["role"] == "received"].groupby(["account_id", "date"])["amount"].sum()
    )
    n_txns = grouped.size()
    counterparties = grouped["counterparty"].agg(lambda s: sorted(set(s)))
    direction_summary = grouped["direction"].agg(lambda s: dict(s.value_counts()))
    label = grouped["is_sar"].any()

    account_day = target_days.set_index(["account_id", "date"]).copy()
    account_day["total_sent"] = total_sent
    account_day["total_received"] = total_received
    account_day[["total_sent", "total_received"]] = account_day[
        ["total_sent", "total_received"]
    ].fillna(0.0)
    account_day["n_txns"] = n_txns
    account_day["counterparties"] = counterparties
    account_day["direction_summary"] = direction_summary
    account_day["label"] = label.fillna(False)

    return account_day.reset_index()


def compute_profile_features(
    account_day_df: pd.DataFrame,
    full_transactions_df: pd.DataFrame,
    windows: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Rolling sent/received profile features for every (account_id, date)
    row in account_day_df, computed from full_transactions_df.

    For each window, and for sent_amount/received_amount separately:
    sum, mean, min, max, count -- where mean/min/max/count are exact
    transaction-level statistics (reconstructed from daily sub-aggregates,
    not diluted by inactive calendar days). Also adds ratio and difference
    features comparing the 1-day window to each longer window, plus a
    "burst" ratio (today's total vs. the longer window's typical
    transaction size) matching the intuition behind Stage 4's volume-spike
    rule.

    Every window for a row is a rolling window ending ON that row's own
    date (closed='right'): it can never see a transaction timestamped
    after that date, regardless of what later activity exists elsewhere in
    full_transactions_df for the same account.
    """
    if windows is None:
        windows = DEFAULT_WINDOWS

    target_accounts = account_day_df["account_id"].unique()
    txns = full_transactions_df[
        full_transactions_df["sender_id"].isin(target_accounts)
        | full_transactions_df["receiver_id"].isin(target_accounts)
    ].copy()
    txns["date"] = txns["timestamp"].dt.normalize()

    sent_daily = (
        txns.loc[txns["sender_id"].isin(target_accounts)]
        .groupby(["sender_id", "date"])["amount"]
        .agg(["sum", "count", "min", "max"])
    )
    received_daily = (
        txns.loc[txns["receiver_id"].isin(target_accounts)]
        .groupby(["receiver_id", "date"])["amount"]
        .agg(["sum", "count", "min", "max"])
    )

    target_dates_by_account = (
        account_day_df.assign(date=pd.to_datetime(account_day_df["date"]))
        .groupby("account_id")["date"]
        .apply(list)
    )

    sent_features = _all_accounts_window_features(sent_daily, target_dates_by_account, windows, "sent")
    received_features = _all_accounts_window_features(
        received_daily, target_dates_by_account, windows, "received"
    )

    base = account_day_df[["account_id", "date", "label"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    result = (
        base.merge(sent_features.reset_index(), on=["account_id", "date"], how="left")
        .merge(received_features.reset_index(), on=["account_id", "date"], how="left")
    )

    ratio_diff_sent = _ratio_diff_features(result, "sent", windows)
    ratio_diff_received = _ratio_diff_features(result, "received", windows)
    result = pd.concat([result, ratio_diff_sent, ratio_diff_received], axis=1)

    result["date"] = result["date"].dt.date
    return result


def _account_window_features(
    daily: pd.DataFrame, target_dates: list[pd.Timestamp], windows: dict[str, int]
) -> pd.DataFrame:
    """daily: this account's (date-indexed) sum/count/min/max for one
    direction. Returns a DataFrame indexed by target_dates with columns
    {stat}_{window_name}, where every window is a trailing window ending on
    (and including) that index date.
    """
    idx = pd.DatetimeIndex(sorted(set(daily.index) | set(target_dates)))
    g = daily.reindex(idx)
    g["sum"] = g["sum"].fillna(0.0)
    g["count"] = g["count"].fillna(0.0)
    # min/max stay NaN on inactive days -- a real transaction can never be
    # NaN, so this correctly signals "no data" rather than "amount 0".

    out = pd.DataFrame(index=idx)
    for window_name, window_days in windows.items():
        rolled = g.rolling(f"{window_days}D", closed="right").agg(
            {"sum": "sum", "count": "sum", "min": "min", "max": "max"}
        )
        out[f"sum_{window_name}"] = rolled["sum"]
        out[f"count_{window_name}"] = rolled["count"]
        out[f"min_{window_name}"] = rolled["min"]
        out[f"max_{window_name}"] = rolled["max"]
        out[f"mean_{window_name}"] = rolled["sum"] / rolled["count"].replace(0, np.nan)

    return out.loc[pd.DatetimeIndex(target_dates)]


def _all_accounts_window_features(
    daily: pd.DataFrame,
    target_dates_by_account: pd.Series,
    windows: dict[str, int],
    prefix: str,
) -> pd.DataFrame:
    accounts_with_data = set(daily.index.get_level_values(0))
    empty = pd.DataFrame(columns=["sum", "count", "min", "max"], dtype=float)

    pieces = []
    for account_id, target_dates in target_dates_by_account.items():
        group = daily.xs(account_id, level=0) if account_id in accounts_with_data else empty
        feats = _account_window_features(group, target_dates, windows)
        feats.insert(0, "account_id", account_id)
        pieces.append(feats)

    out = pd.concat(pieces)
    out.index.name = "date"
    out = out.reset_index().set_index(["account_id", "date"])
    out.columns = [f"{prefix}_{c}" for c in out.columns]
    return out


def _ratio_diff_features(result: pd.DataFrame, prefix: str, windows: dict[str, int]) -> pd.DataFrame:
    stats = ["sum", "mean", "min", "max", "count"]
    longer_windows = [w for w in windows if w != "1d"]
    out = pd.DataFrame(index=result.index)

    for stat in stats:
        col_1d = f"{prefix}_{stat}_1d"
        for window_name in longer_windows:
            col_w = f"{prefix}_{stat}_{window_name}"
            denom = result[col_w].replace(0, np.nan)
            out[f"{prefix}_{stat}_ratio_1d_vs_{window_name}"] = result[col_1d] / denom
            out[f"{prefix}_{stat}_diff_1d_vs_{window_name}"] = result[col_1d] - result[col_w]

    sum_1d = result[f"{prefix}_sum_1d"]
    for window_name in longer_windows:
        mean_w = result[f"{prefix}_mean_{window_name}"].replace(0, np.nan)
        out[f"{prefix}_burst_ratio_{window_name}"] = sum_1d / mean_w

    return out


def select_top_profile_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model,
    cumulative_importance_threshold: float = 0.9,
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: str = "neg_log_loss",
) -> list[str]:
    """Fit model on X_train/y_train, compute permutation importance, and
    return the smallest set of feature names (most important first) whose
    cumulative importance share reaches cumulative_importance_threshold.

    scoring defaults to log loss rather than sklearn's default accuracy:
    with a rare positive class (SAR labels are typically <1% of alerted
    account-days), accuracy barely moves when a feature is permuted, so
    almost every feature reads as "zero importance" and the ranking becomes
    unstable. Log loss is sensitive to probability shifts everywhere, not
    just at the decision boundary, and gives a far less degenerate ranking.
    """
    model.fit(X_train, y_train)

    result = permutation_importance(
        model,
        X_train,
        y_train,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring=scoring,
    )
    importances = pd.Series(result.importances_mean, index=X_train.columns).clip(lower=0)

    if importances.sum() == 0:
        return list(X_train.columns)

    ranked = importances.sort_values(ascending=False)
    cumulative_share = ranked.cumsum() / ranked.sum()
    n_features = int((cumulative_share < cumulative_importance_threshold).sum()) + 1
    n_features = min(n_features, len(ranked))
    return ranked.index[:n_features].tolist()
