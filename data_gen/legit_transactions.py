"""Synthetic legitimate transaction stream generation.

No laundering typologies here -- this models the "background" activity a
rule engine and later typology injectors will be layered on top of.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_gen.accounts import MAX_REGULAR_COUNTERPARTIES

REGULAR_COUNTERPARTY_PROB = 0.8
BASE_DATE = pd.Timestamp("2024-01-01")


def generate_legit_transactions(
    accounts_df: pd.DataFrame, n_days: int, seed: int
) -> pd.DataFrame:
    """Generate a legitimate transaction ledger over accounts_df.

    Each account initiates a Poisson-distributed number of transactions
    over n_days, driven by its persona's weekly_txn_rate. Counterparties
    are drawn 80% of the time from the sender's fixed "regular"
    counterparties (skewing the degree distribution toward hubs) and 20%
    of the time uniformly at random from the whole population.

    Columns: txn_id, sender_id, receiver_id, amount, timestamp, direction

    Pure function: identical (accounts_df, n_days, seed) always yields an
    identical DataFrame.
    """
    rng = np.random.default_rng(seed)

    n_accounts = len(accounts_df)
    account_id = accounts_df["account_id"].to_numpy()
    account_type = accounts_df["account_type"].to_numpy()
    weekly_txn_rate = accounts_df["weekly_txn_rate"].to_numpy(dtype=float)
    amount_mu = accounts_df["amount_mu"].to_numpy(dtype=float)
    amount_sigma = accounts_df["amount_sigma"].to_numpy(dtype=float)

    regular_matrix, regular_counts = _build_regular_matrix(accounts_df, account_id)

    n_weeks = n_days / 7.0
    lam = weekly_txn_rate * n_weeks
    txn_counts = rng.poisson(lam)

    sender_idx = np.repeat(np.arange(n_accounts), txn_counts)
    n_txns = len(sender_idx)

    if n_txns == 0:
        return pd.DataFrame(
            columns=["txn_id", "sender_id", "receiver_id", "amount", "timestamp", "direction"]
        )

    use_regular = (rng.random(n_txns) < REGULAR_COUNTERPARTY_PROB) & (
        regular_counts[sender_idx] > 0
    )

    safe_counts = np.maximum(regular_counts[sender_idx], 1)
    regular_col = (rng.random(n_txns) * safe_counts).astype(int)
    regular_col = np.minimum(regular_col, safe_counts - 1)
    regular_receiver = regular_matrix[sender_idx, regular_col]

    random_receiver = rng.integers(0, n_accounts, size=n_txns)
    self_mask = random_receiver == sender_idx
    while self_mask.any():
        random_receiver[self_mask] = rng.integers(0, n_accounts, size=self_mask.sum())
        self_mask = random_receiver == sender_idx

    receiver_idx = np.where(use_regular, regular_receiver, random_receiver)

    amount = rng.lognormal(mean=amount_mu[sender_idx], sigma=amount_sigma[sender_idx])

    day_offset = rng.integers(0, n_days, size=n_txns)
    seconds_offset = rng.integers(0, 86400, size=n_txns)
    timestamp = (
        BASE_DATE
        + pd.to_timedelta(day_offset, unit="D")
        + pd.to_timedelta(seconds_offset, unit="s")
    )

    sender_type = account_type[sender_idx]
    receiver_type = account_type[receiver_idx]
    direction = np.select(
        [
            (sender_type == "internal") & (receiver_type == "internal"),
            (sender_type == "internal") & (receiver_type == "external"),
            (sender_type == "external") & (receiver_type == "internal"),
        ],
        ["internal", "outbound", "inbound"],
        default="external",
    )

    txn_id = np.char.add("TXN", np.char.zfill(np.arange(n_txns).astype(str), 9))

    txns = pd.DataFrame(
        {
            "txn_id": txn_id,
            "sender_id": account_id[sender_idx],
            "receiver_id": account_id[receiver_idx],
            "amount": np.round(amount, 2),
            "timestamp": timestamp,
            "direction": direction,
        }
    )
    return txns.sort_values("timestamp").reset_index(drop=True)


def _build_regular_matrix(
    accounts_df: pd.DataFrame, account_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pad each account's regular_counterparties list into a fixed-width
    index matrix (n_accounts, MAX_REGULAR_COUNTERPARTIES) for vectorized
    lookup, plus a parallel array of each account's actual list length.
    """
    n_accounts = len(accounts_df)
    id_to_idx = {aid: i for i, aid in enumerate(account_id)}

    regular_matrix = np.zeros((n_accounts, MAX_REGULAR_COUNTERPARTIES), dtype=int)
    regular_counts = np.zeros(n_accounts, dtype=int)

    for i, counterparties in enumerate(accounts_df["regular_counterparties"]):
        idxs = [id_to_idx[c] for c in counterparties]
        regular_counts[i] = len(idxs)
        if idxs:
            regular_matrix[i, : len(idxs)] = idxs

    return regular_matrix, regular_counts
