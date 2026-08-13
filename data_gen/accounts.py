"""Synthetic account population generation.

Accounts are assigned a "persona" cluster that drives both their legitimate
transaction frequency and typical amount distribution, plus a small fixed
set of "regular" counterparties they preferentially transact with. Merchant
personas are given a higher chance of being picked as someone else's regular
counterparty, which produces a heavy-tailed (hub-like) degree distribution
rather than a uniform one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERSONAS = ("retail", "small_business", "high_volume_merchant")

# weekly_txn_rate: expected legitimate transactions initiated per week.
# amount_mu/amount_sigma: parameters of the underlying normal distribution
#   for the persona's lognormal transaction amount (mu = log of the median).
# popularity_weight: relative likelihood of being picked as someone else's
#   "regular" counterparty -- higher for merchants, which makes them hubs.
PERSONA_PARAMS = {
    "retail": dict(
        share=0.70,
        weekly_txn_rate=0.3,
        amount_mu=np.log(50.0),
        amount_sigma=0.7,
        popularity_weight=1.0,
    ),
    "small_business": dict(
        share=0.20,
        weekly_txn_rate=2.0,
        amount_mu=np.log(400.0),
        amount_sigma=0.9,
        popularity_weight=4.0,
    ),
    "high_volume_merchant": dict(
        share=0.10,
        weekly_txn_rate=15.0,
        amount_mu=np.log(120.0),
        amount_sigma=1.3,
        popularity_weight=25.0,
    ),
}

ACCOUNT_TYPE_SHARES = {"internal": 0.8, "external": 0.2}

MIN_REGULAR_COUNTERPARTIES = 1
MAX_REGULAR_COUNTERPARTIES = 5


def generate_accounts(n_accounts: int, seed: int) -> pd.DataFrame:
    """Generate a synthetic account population.

    Columns:
        account_id: str, e.g. "ACC0000042"
        account_type: "internal" | "external" (~80/20)
        persona: "retail" | "small_business" | "high_volume_merchant"
        weekly_txn_rate: expected legitimate txns/week for this account
        amount_mu, amount_sigma: lognormal amount params for this account
        regular_counterparties: list[str] of 1-5 account_ids this account
            preferentially transacts with

    Pure function: identical (n_accounts, seed) always yields an identical
    DataFrame.
    """
    rng = np.random.default_rng(seed)

    account_id = np.array([f"ACC{i:07d}" for i in range(n_accounts)])

    account_type = rng.choice(
        list(ACCOUNT_TYPE_SHARES.keys()),
        size=n_accounts,
        p=list(ACCOUNT_TYPE_SHARES.values()),
    )

    persona = rng.choice(
        PERSONAS,
        size=n_accounts,
        p=[PERSONA_PARAMS[p]["share"] for p in PERSONAS],
    )

    base_weekly_txn_rate = np.array([PERSONA_PARAMS[p]["weekly_txn_rate"] for p in persona])
    amount_mu = np.array([PERSONA_PARAMS[p]["amount_mu"] for p in persona])
    amount_sigma = np.array([PERSONA_PARAMS[p]["amount_sigma"] for p in persona])
    base_popularity_weight = np.array(
        [PERSONA_PARAMS[p]["popularity_weight"] for p in persona]
    )

    # Per-account jitter so accounts within a persona aren't clones of each
    # other -- without it, degree clusters into one tight band per persona
    # instead of a smooth heavy tail.
    weekly_txn_rate = base_weekly_txn_rate * rng.lognormal(
        mean=0.0, sigma=0.4, size=n_accounts
    )
    popularity_weight = base_popularity_weight * rng.lognormal(
        mean=0.0, sigma=0.6, size=n_accounts
    )

    regular_idx = _assign_regular_counterparties(
        n_accounts=n_accounts, popularity_weight=popularity_weight, rng=rng
    )
    regular_counterparties = [
        [str(account_id[c]) for c in row] for row in regular_idx
    ]

    return pd.DataFrame(
        {
            "account_id": account_id,
            "account_type": account_type,
            "persona": persona,
            "weekly_txn_rate": weekly_txn_rate,
            "amount_mu": amount_mu,
            "amount_sigma": amount_sigma,
            "regular_counterparties": regular_counterparties,
        }
    )


def _assign_regular_counterparties(
    n_accounts: int, popularity_weight: np.ndarray, rng: np.random.Generator
) -> list[list[int]]:
    """Assign each account 1-5 "regular" counterparty indices, sampled
    without replacement and weighted by the counterparty's popularity
    (merchants are much more likely to be picked, creating hubs).
    """
    if n_accounts <= 1:
        return [[] for _ in range(n_accounts)]

    weights_int = np.maximum(popularity_weight.astype(int), 1)
    pool = np.repeat(np.arange(n_accounts), weights_int)

    max_k = min(MAX_REGULAR_COUNTERPARTIES, n_accounts - 1)
    min_k = min(MIN_REGULAR_COUNTERPARTIES, max_k)
    k_per_account = rng.integers(min_k, max_k + 1, size=n_accounts)

    result: list[list[int]] = []
    for i in range(n_accounts):
        k = int(k_per_account[i])
        seen = {i}
        uniq: list[int] = []
        attempts = 0
        max_attempts = max(50, k * 20)
        while len(uniq) < k and attempts < max_attempts:
            batch = pool[rng.integers(0, len(pool), size=k * 3)]
            for c in batch:
                c = int(c)
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)
                    if len(uniq) == k:
                        break
            attempts += 1
        if len(uniq) < k:
            for c in range(n_accounts):
                if c not in seen:
                    uniq.append(c)
                    seen.add(c)
                if len(uniq) == k:
                    break
        result.append(uniq)
    return result
