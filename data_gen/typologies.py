"""Laundering typology injection.

Each injector samples existing accounts out of accounts_df and generates new,
labeled (is_sar=True) transaction rows layered on top of the legitimate
stream. Every transaction's txn_id is prefixed with its pattern_id (e.g.
"SMURF00003_004") so a pattern instance's rows can be found and traced by
hand directly in the DataFrame.

Accounts are never reused across pattern instances: within a single call,
each injector tracks the accounts it has already placed, and callers can
chain calls together via `exclude_accounts` (inject_all_patterns does this)
so the combined output never merges two patterns -- or a pattern and an
unrelated one -- into an unintended shared connected component.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_gen.legit_transactions import BASE_DATE

REPORTING_THRESHOLD = 10_000.0

_STRUCTURING_MIN = 9_000.0
_STRUCTURING_MAX = 9_900.0


def inject_smurfing(
    accounts_df: pd.DataFrame,
    n_patterns: int,
    seed: int,
    n_days: int = 365,
    exclude_accounts: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hub-and-spoke structuring: 8-15 mules each send a collector an amount
    just under the reporting threshold within a 2-4 day window; the
    collector then forwards ~90% of the pooled total to one destination
    account 1-2 days later.
    """
    rng = np.random.default_rng(seed)
    account_ids = accounts_df["account_id"].to_numpy()
    account_type = accounts_df.set_index("account_id")["account_type"]
    used = set(exclude_accounts) if exclude_accounts else set()

    txn_rows: list[dict] = []
    account_rows: list[dict] = []

    for i in range(n_patterns):
        pattern_id = f"SMURF{i:05d}"
        n_mules = int(rng.integers(8, 16))

        collector = _sample_unused_accounts(account_ids, 1, rng, used)[0]
        mules = _sample_unused_accounts(account_ids, n_mules, rng, used)
        destination = _sample_unused_accounts(account_ids, 1, rng, used)[0]

        pattern_start = _pattern_start(rng, n_days, buffer_days=8)
        window_days = rng.uniform(2, 4)
        mule_amounts = rng.uniform(_STRUCTURING_MIN, _STRUCTURING_MAX, size=n_mules)
        mule_offsets = rng.uniform(0, window_days, size=n_mules)
        mule_timestamps = [pattern_start + pd.Timedelta(days=d) for d in mule_offsets]

        for j in range(n_mules):
            txn_rows.append(
                _txn_row(
                    txn_id=f"{pattern_id}_{j:03d}",
                    sender_id=mules[j],
                    receiver_id=collector,
                    amount=mule_amounts[j],
                    timestamp=mule_timestamps[j],
                    account_type=account_type,
                    pattern_id=pattern_id,
                    pattern_type="smurfing",
                )
            )

        pooled_total = mule_amounts.sum()
        forward_amount = pooled_total * rng.uniform(0.85, 0.95)
        forward_delay = rng.uniform(1, 2)
        forward_timestamp = max(mule_timestamps) + pd.Timedelta(days=forward_delay)
        txn_rows.append(
            _txn_row(
                txn_id=f"{pattern_id}_{n_mules:03d}",
                sender_id=collector,
                receiver_id=destination,
                amount=forward_amount,
                timestamp=forward_timestamp,
                account_type=account_type,
                pattern_id=pattern_id,
                pattern_type="smurfing",
            )
        )

        account_rows.append(_account_row(pattern_id, "smurfing", collector, "collector"))
        for m in mules:
            account_rows.append(_account_row(pattern_id, "smurfing", m, "mule"))
        account_rows.append(_account_row(pattern_id, "smurfing", destination, "destination"))

    return _finalize(txn_rows, account_rows)


def inject_layering(
    accounts_df: pd.DataFrame,
    n_patterns: int,
    seed: int,
    n_days: int = 365,
    exclude_accounts: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chain of 3-6 accounts A1 -> A2 -> ... -> An, each hop within hours to
    2 days of the previous, retaining 90-97% of the amount per hop.
    """
    rng = np.random.default_rng(seed)
    account_ids = accounts_df["account_id"].to_numpy()
    account_type = accounts_df.set_index("account_id")["account_type"]
    used = set(exclude_accounts) if exclude_accounts else set()

    txn_rows: list[dict] = []
    account_rows: list[dict] = []

    for i in range(n_patterns):
        pattern_id = f"LAYER{i:05d}"
        chain_len = int(rng.integers(3, 7))
        chain = _sample_unused_accounts(account_ids, chain_len, rng, used)

        pattern_start = _pattern_start(rng, n_days, buffer_days=12)
        timestamp = pattern_start
        running_amount = rng.uniform(5_000, 50_000)

        for hop in range(chain_len - 1):
            gap_hours = rng.uniform(1, 48)
            timestamp = timestamp + pd.Timedelta(hours=gap_hours)
            txn_rows.append(
                _txn_row(
                    txn_id=f"{pattern_id}_{hop:03d}",
                    sender_id=chain[hop],
                    receiver_id=chain[hop + 1],
                    amount=running_amount,
                    timestamp=timestamp,
                    account_type=account_type,
                    pattern_id=pattern_id,
                    pattern_type="layering",
                )
            )
            retention = rng.uniform(0.90, 0.97)
            running_amount *= retention

        for pos, acc in enumerate(chain):
            role = "origin" if pos == 0 else ("destination" if pos == chain_len - 1 else "intermediate")
            account_rows.append(_account_row(pattern_id, "layering", acc, role))

    return _finalize(txn_rows, account_rows)


def inject_fan_out_fan_in(
    accounts_df: pd.DataFrame,
    n_patterns: int,
    seed: int,
    n_days: int = 365,
    exclude_accounts: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One source fans out to 5-10 intermediates, which converge back into
    1-3 destinations within a similarly short window (an "hourglass").
    """
    rng = np.random.default_rng(seed)
    account_ids = accounts_df["account_id"].to_numpy()
    account_type = accounts_df.set_index("account_id")["account_type"]
    used = set(exclude_accounts) if exclude_accounts else set()

    txn_rows: list[dict] = []
    account_rows: list[dict] = []

    for i in range(n_patterns):
        pattern_id = f"FANIO{i:05d}"
        n_intermediates = int(rng.integers(5, 11))
        n_destinations = int(rng.integers(1, 4))

        source = _sample_unused_accounts(account_ids, 1, rng, used)[0]
        intermediates = _sample_unused_accounts(account_ids, n_intermediates, rng, used)
        destinations = _sample_unused_accounts(account_ids, n_destinations, rng, used)

        pattern_start = _pattern_start(rng, n_days, buffer_days=8)
        window_days = rng.uniform(2, 4)
        total_amount = rng.uniform(10_000, 80_000)
        shares = rng.dirichlet(np.ones(n_intermediates))
        fan_out_amounts = total_amount * shares
        fan_out_offsets = rng.uniform(0, window_days, size=n_intermediates)
        fan_out_timestamps = [pattern_start + pd.Timedelta(days=d) for d in fan_out_offsets]

        for j in range(n_intermediates):
            txn_rows.append(
                _txn_row(
                    txn_id=f"{pattern_id}_out{j:03d}",
                    sender_id=source,
                    receiver_id=intermediates[j],
                    amount=fan_out_amounts[j],
                    timestamp=fan_out_timestamps[j],
                    account_type=account_type,
                    pattern_id=pattern_id,
                    pattern_type="fan_out_fan_in",
                )
            )

        # Guarantee every declared destination receives at least one hop --
        # pure random assignment can otherwise leave a destination account
        # "involved" in the pattern but with no actual transaction.
        dest_choice = np.empty(n_intermediates, dtype=int)
        dest_choice[:n_destinations] = np.arange(n_destinations)
        if n_intermediates > n_destinations:
            dest_choice[n_destinations:] = rng.integers(
                0, n_destinations, size=n_intermediates - n_destinations
            )
        rng.shuffle(dest_choice)
        retention = rng.uniform(0.90, 0.97, size=n_intermediates)
        delay_hours = rng.uniform(1, 48, size=n_intermediates)
        for j in range(n_intermediates):
            txn_rows.append(
                _txn_row(
                    txn_id=f"{pattern_id}_in{j:03d}",
                    sender_id=intermediates[j],
                    receiver_id=destinations[int(dest_choice[j])],
                    amount=fan_out_amounts[j] * retention[j],
                    timestamp=fan_out_timestamps[j] + pd.Timedelta(hours=delay_hours[j]),
                    account_type=account_type,
                    pattern_id=pattern_id,
                    pattern_type="fan_out_fan_in",
                )
            )

        account_rows.append(_account_row(pattern_id, "fan_out_fan_in", source, "source"))
        for acc in intermediates:
            account_rows.append(_account_row(pattern_id, "fan_out_fan_in", acc, "intermediate"))
        for acc in destinations:
            account_rows.append(_account_row(pattern_id, "fan_out_fan_in", acc, "destination"))

    return _finalize(txn_rows, account_rows)


def inject_round_tripping(
    accounts_df: pd.DataFrame,
    n_patterns: int,
    seed: int,
    n_days: int = 365,
    exclude_accounts: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A -> B -> ... -> A: a loop of 2-4 accounts. Intermediate hops happen
    within hours to 2 days; the final hop back to the origin is delayed
    days to weeks, and the amount is slightly reduced by the time it
    returns.
    """
    rng = np.random.default_rng(seed)
    account_ids = accounts_df["account_id"].to_numpy()
    account_type = accounts_df.set_index("account_id")["account_type"]
    used = set(exclude_accounts) if exclude_accounts else set()

    txn_rows: list[dict] = []
    account_rows: list[dict] = []

    for i in range(n_patterns):
        pattern_id = f"RTRIP{i:05d}"
        loop_len = int(rng.integers(2, 5))
        loop = _sample_unused_accounts(account_ids, loop_len, rng, used)

        pattern_start = _pattern_start(rng, n_days, buffer_days=35)
        timestamp = pattern_start
        running_amount = rng.uniform(5_000, 50_000)

        for hop in range(loop_len):
            sender = loop[hop]
            receiver = loop[(hop + 1) % loop_len]
            is_return_hop = hop == loop_len - 1
            if is_return_hop:
                timestamp = timestamp + pd.Timedelta(days=rng.uniform(3, 21))
            else:
                timestamp = timestamp + pd.Timedelta(hours=rng.uniform(1, 48))

            txn_rows.append(
                _txn_row(
                    txn_id=f"{pattern_id}_{hop:03d}",
                    sender_id=sender,
                    receiver_id=receiver,
                    amount=running_amount,
                    timestamp=timestamp,
                    account_type=account_type,
                    pattern_id=pattern_id,
                    pattern_type="round_tripping",
                )
            )
            retention = rng.uniform(0.90, 0.97)
            running_amount *= retention

        for pos, acc in enumerate(loop):
            role = "origin" if pos == 0 else "intermediate"
            account_rows.append(_account_row(pattern_id, "round_tripping", acc, role))

    return _finalize(txn_rows, account_rows)


def inject_all_patterns(
    accounts_df: pd.DataFrame,
    seed: int,
    n_days: int = 365,
    target_suspicious_share: float = 0.025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all four injectors with pattern counts tuned so that roughly
    target_suspicious_share (default 2.5%, i.e. the middle of the 2-3%
    target band) of accounts_df's accounts end up involved in at least one
    laundering pattern. Accounts are never reused across pattern types or
    instances, so components stay isolated.

    Returns (sar_txns_df, accounts_involved_df) covering all four typologies.
    """
    total_accounts = len(accounts_df)
    target_accounts = total_accounts * target_suspicious_share
    budget_per_type = target_accounts / 4

    # Rough expected accounts-consumed-per-instance for each typology, used
    # only to convert an account budget into a pattern count.
    avg_accounts_per_instance = {
        "smurfing": 1 + 11.5 + 1,
        "layering": 4.5,
        "fan_out_fan_in": 1 + 7.5 + 2,
        "round_tripping": 3.0,
    }

    def n_patterns_for(typology: str) -> int:
        return max(1, round(budget_per_type / avg_accounts_per_instance[typology]))

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(4)

    used: set[str] = set()
    txns_parts: list[pd.DataFrame] = []
    accounts_parts: list[pd.DataFrame] = []

    injectors = [
        ("smurfing", inject_smurfing),
        ("layering", inject_layering),
        ("fan_out_fan_in", inject_fan_out_fan_in),
        ("round_tripping", inject_round_tripping),
    ]

    for (typology, injector), child_seed in zip(injectors, child_seeds):
        txns, involved = injector(
            accounts_df,
            n_patterns_for(typology),
            int(child_seed.generate_state(1)[0]),
            n_days=n_days,
            exclude_accounts=used,
        )
        used |= set(involved["account_id"])
        txns_parts.append(txns)
        accounts_parts.append(involved)

    sar_txns = pd.concat(txns_parts, ignore_index=True)
    accounts_involved = pd.concat(accounts_parts, ignore_index=True)
    return sar_txns, accounts_involved


def _sample_unused_accounts(
    account_ids: np.ndarray, k: int, rng: np.random.Generator, used: set[str]
) -> list[str]:
    chosen: list[str] = []
    attempts = 0
    max_attempts = max(500, k * 50)
    while len(chosen) < k and attempts < max_attempts:
        candidate = str(account_ids[rng.integers(0, len(account_ids))])
        if candidate not in used:
            used.add(candidate)
            chosen.append(candidate)
        attempts += 1
    if len(chosen) < k:
        raise ValueError(
            f"Could not find {k} unused accounts (only found {len(chosen)}); "
            "accounts_df may be too small for the requested n_patterns."
        )
    return chosen


def _pattern_start(rng: np.random.Generator, n_days: int, buffer_days: int) -> pd.Timestamp:
    max_start = max(1, n_days - buffer_days)
    return BASE_DATE + pd.Timedelta(days=rng.uniform(0, max_start))


def _txn_row(
    txn_id: str,
    sender_id: str,
    receiver_id: str,
    amount: float,
    timestamp: pd.Timestamp,
    account_type: pd.Series,
    pattern_id: str,
    pattern_type: str,
) -> dict:
    sender_type = account_type[sender_id]
    receiver_type = account_type[receiver_id]
    if sender_type == "internal" and receiver_type == "internal":
        direction = "internal"
    elif sender_type == "internal" and receiver_type == "external":
        direction = "outbound"
    elif sender_type == "external" and receiver_type == "internal":
        direction = "inbound"
    else:
        direction = "external"

    return {
        "txn_id": txn_id,
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "amount": round(float(amount), 2),
        "timestamp": timestamp,
        "direction": direction,
        "is_sar": True,
        "pattern_type": pattern_type,
        "pattern_id": pattern_id,
    }


def _account_row(pattern_id: str, pattern_type: str, account_id: str, role: str) -> dict:
    return {
        "pattern_id": pattern_id,
        "pattern_type": pattern_type,
        "account_id": account_id,
        "role": role,
    }


def _finalize(
    txn_rows: list[dict], account_rows: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    txn_columns = [
        "txn_id",
        "sender_id",
        "receiver_id",
        "amount",
        "timestamp",
        "direction",
        "is_sar",
        "pattern_type",
        "pattern_id",
    ]
    account_columns = ["pattern_id", "pattern_type", "account_id", "role"]

    txns_df = pd.DataFrame(txn_rows, columns=txn_columns).sort_values("timestamp").reset_index(
        drop=True
    )
    accounts_df = pd.DataFrame(account_rows, columns=account_columns)
    return txns_df, accounts_df
