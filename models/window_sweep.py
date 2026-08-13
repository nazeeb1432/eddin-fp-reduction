"""Stage 9 window sweep: reproduce the paper's Figure 5-style experiment.

Degree features are normally computed from one SlidingGraph window. Here,
each account-day instead uses one of two windows depending on its own
*known* label: TWL ("time window legit") for legit account-days, TWS
("time window suspicious") for suspicious ones. This is only meaningful
offline, on already-labeled training data -- it's an ablation asking
"what lookback window best characterizes each class's behavior", not a
deployable inference-time policy (an unlabeled account's window couldn't
be chosen this way in production; Stage 8's GWd is what a delay-aware,
label-free version of this idea would look like).

Retrains ("retrain" = refit, not re-tune) the *same* profiles+degrees
model architecture whose hyperparameters were already chosen by one
Optuna search, swapping in freshly computed degree features for each of
the 6x6 (TWL, TWS) combinations, and reports recall_at_fpr(0.20) as a
heatmap.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import SlidingGraph, compute_degree_features
from models.metrics import recall_at_fpr
from models.run_experiment import BASE_DATE, GRAPH_WINDOW, SEED, TARGET_FPR, TRAIN_FRAC, VAL_FRAC, build_dataset, build_profile_features
from models.train import temporal_split, train_and_tune

WINDOW_GRID = [1, 7, 14, 30, 60, 90]
N_ACCOUNTS = 3_000
N_TUNE_TRIALS = 30

OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"


def build_dual_window_degree_features(
    account_day: pd.DataFrame, full_txns: pd.DataFrame, twl: int, tws: int
) -> pd.DataFrame:
    """Two SlidingGraphs built in lockstep from the same edge stream (one
    windowed at twl, one at tws). Each account-day's degree features come
    from whichever graph matches its own true label.
    """
    targets_by_day = account_day.groupby("day")["account_id"].apply(list).to_dict()
    label_by_key = account_day.set_index(["account_id", "day"])["label"].to_dict()

    graph_legit = SlidingGraph(window=twl)
    graph_suspicious = SlidingGraph(window=tws)
    rows = []
    for day, day_txns in full_txns.groupby("day"):
        for row in day_txns.itertuples():
            graph_legit.add_edge(day, row.sender_id, row.receiver_id, row.amount)
            graph_suspicious.add_edge(day, row.sender_id, row.receiver_id, row.amount)
        graph_legit.advance_to(day)
        graph_suspicious.advance_to(day)
        if day in targets_by_day:
            for account_id in targets_by_day[day]:
                is_suspicious = bool(label_by_key.get((account_id, day), False))
                graph = graph_suspicious if is_suspicious else graph_legit
                rows.append({"account_id": account_id, "day": day, **compute_degree_features(graph, account_id)})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    full_txns, account_day, accounts = build_dataset(n_accounts=N_ACCOUNTS)
    print(f"accounts={len(accounts):,} txns={len(full_txns):,} account-days={len(account_day):,} "
          f"positives={account_day['label'].sum()} ({time.time() - t0:.1f}s)")

    profiles = build_profile_features(account_day, full_txns)
    profile_cols = [c for c in profiles.columns if c not in ("account_id", "date", "day", "label")]

    # One reference architecture: tune profiles+degrees at the default
    # symmetric window (TWL=TWS=GRAPH_WINDOW), then reuse those same
    # hyperparameters (refit, not re-tune) for every grid cell below.
    print(f"tuning reference profiles+degrees architecture at TWL=TWS={GRAPH_WINDOW}... ({time.time() - t0:.1f}s)")
    reference_degrees = build_dual_window_degree_features(account_day, full_txns, GRAPH_WINDOW, GRAPH_WINDOW)
    degree_cols = [c for c in reference_degrees.columns if c not in ("account_id", "day")]
    reference_combined = profiles.merge(reference_degrees, on=["account_id", "day"], how="inner")
    ref_train, ref_val, _ = temporal_split(reference_combined, TRAIN_FRAC, VAL_FRAC)
    feature_cols = profile_cols + degree_cols
    reference_model = train_and_tune(
        ref_train[feature_cols], ref_train["label"].astype(int),
        ref_val[feature_cols], ref_val["label"].astype(int),
        model_type="lightgbm", n_trials=N_TUNE_TRIALS, target_fpr=TARGET_FPR, seed=SEED, timeout=180,
    )
    best_params = reference_model.get_params()
    print(f"reference tuned ({time.time() - t0:.1f}s)")

    heatmap = pd.DataFrame(index=WINDOW_GRID, columns=WINDOW_GRID, dtype=float)
    for twl in WINDOW_GRID:
        for tws in WINDOW_GRID:
            cell_t0 = time.time()
            if twl == GRAPH_WINDOW and tws == GRAPH_WINDOW:
                degree_df = reference_degrees
            else:
                degree_df = build_dual_window_degree_features(account_day, full_txns, twl, tws)
            combined = profiles.merge(degree_df, on=["account_id", "day"], how="inner")
            train, val, test = temporal_split(combined, TRAIN_FRAC, VAL_FRAC)

            model = type(reference_model)(**best_params)
            model.fit(train[feature_cols], train["label"].astype(int))
            scores = model.predict_proba(test[feature_cols])[:, 1]
            recall = recall_at_fpr(test["label"], scores, TARGET_FPR)
            heatmap.loc[tws, twl] = recall
            print(f"TWL={twl:>2} TWS={tws:>2}: recall@{TARGET_FPR:.0%}FPR={recall:.3f} ({time.time() - cell_t0:.1f}s)")

    heatmap.to_csv(OUTPUT_DIR / "window_sweep_heatmap.csv")
    print(f"\nsaved {OUTPUT_DIR / 'window_sweep_heatmap.csv'}")

    plt.figure(figsize=(7, 6))
    im = plt.imshow(heatmap.values, cmap="viridis", aspect="auto", origin="lower")
    plt.colorbar(im, label=f"recall@{TARGET_FPR:.0%}FPR")
    plt.xticks(range(len(WINDOW_GRID)), WINDOW_GRID)
    plt.yticks(range(len(WINDOW_GRID)), WINDOW_GRID)
    plt.xlabel("TWL (legit window, days)")
    plt.ylabel("TWS (suspicious window, days)")
    plt.title("recall@20%FPR by (TWL, TWS)")
    for i, tws in enumerate(WINDOW_GRID):
        for j, twl in enumerate(WINDOW_GRID):
            plt.text(j, i, f"{heatmap.loc[tws, twl]:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "window_sweep_heatmap.png", dpi=150)
    plt.close()
    print(f"saved {OUTPUT_DIR / 'window_sweep_heatmap.png'}")

    best_tws, best_twl = (int(x) for x in heatmap.stack().idxmax())
    print(f"\nbest (TWS, TWL): ({best_tws}, {best_twl}), recall={heatmap.stack().max():.3f}")
    print(f"total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
