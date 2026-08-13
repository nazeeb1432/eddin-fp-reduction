"""Run the full pipeline at a small, inspection-friendly scale and dump every
intermediate table to CSV under data/, so the generated (in-memory-only, by
default) synthetic dataset can actually be opened and read.

Not part of the pipeline itself -- this is a debugging/inspection aid.
Uses the same seed as the real experiment (models.run_experiment.SEED) so
the data matches what the pipeline actually trains on, just at a scale
small enough to skim by eye.

Usage:
    python -m scripts.export_dataset
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from data_gen import generate_accounts, generate_legit_transactions, inject_all_patterns
from features import build_account_day_table, compute_profile_features
from models.run_experiment import (
    BASE_DATE,
    GRAPH_WINDOW,
    GW_MAX_STEPS,
    GW_N_WALKS,
    SEED,
    build_degree_and_gw_features,
    build_gwd_features,
    build_hybrid_illicit_set_for_experiment,
)
from rules import run_rule_engine

N_ACCOUNTS = 800
N_DAYS = 365

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"


def save(df: pd.DataFrame, name: str) -> None:
    path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  {name:<22} {df.shape[0]:>8,} rows x {df.shape[1]:>3} cols -> {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    accounts = generate_accounts(N_ACCOUNTS, seed=SEED)
    legit_txns = generate_legit_transactions(accounts, n_days=N_DAYS, seed=SEED)
    sar_txns, involved = inject_all_patterns(accounts, seed=SEED, n_days=N_DAYS)

    legit_txns = legit_txns.copy()
    legit_txns["is_sar"] = False
    legit_txns["pattern_type"] = None
    legit_txns["pattern_id"] = None
    full_txns = (
        pd.concat([legit_txns, sar_txns], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    )
    full_txns["day"] = (full_txns["timestamp"].dt.normalize() - BASE_DATE).dt.days

    alerts = run_rule_engine(full_txns)
    account_day = build_account_day_table(full_txns, alerts)
    account_day["day"] = (pd.to_datetime(account_day["date"]) - BASE_DATE).dt.days

    profiles = compute_profile_features(account_day, full_txns)
    profiles["day"] = (pd.to_datetime(profiles["date"]) - BASE_DATE).dt.days
    profile_cols = [c for c in profiles.columns if c not in ("account_id", "date", "day", "label")]

    oracle_illicit_set = set(account_day.loc[account_day["label"], "account_id"])
    degree_gw = build_degree_and_gw_features(account_day, full_txns, oracle_illicit_set)
    degree_cols = [c for c in degree_gw.columns if c not in ("account_id", "day") and not c.startswith("gw_")]

    profile_degree = profiles.merge(degree_gw[["account_id", "day"] + degree_cols], on=["account_id", "day"], how="inner")
    hybrid_illicit_set = build_hybrid_illicit_set_for_experiment(profile_degree, profile_cols, degree_cols)
    gwd = build_gwd_features(account_day, full_txns, hybrid_illicit_set)

    combined = (
        profiles.merge(degree_gw, on=["account_id", "day"], how="inner")
        .merge(gwd, on=["account_id", "day"], how="inner")
    )

    print(f"generated in {time.time() - t0:.1f}s, saving to {OUTPUT_DIR}/\n")
    save(accounts, "01_accounts")
    save(legit_txns, "02_legit_transactions")
    save(sar_txns, "03_sar_transactions")
    save(involved, "04_involved_accounts")
    save(full_txns, "05_full_transactions")
    save(alerts, "06_alerts")
    save(account_day, "07_account_day")
    save(profiles, "08_profile_features")
    save(degree_gw, "09_degree_and_gw_features")
    save(gwd, "10_gwd_features")
    save(combined, "11_combined_features")

    print(f"\ndone ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
