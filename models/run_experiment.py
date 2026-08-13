"""Stage 9: assemble the full pipeline (Stages 2-8), train models on five
increasing feature sets, and evaluate on a held-out temporal test set. This
is the reproduction of the paper's core experiment: does each added
feature family (degrees, GuiltyWalker, GuiltyWalker-delay) actually help?

Feature sets (each is profiles + one addition, not cumulative chaining --
"+degrees+GW" is its own explicit combination so the ablation can show
diminishing/overlapping lift, not just a growing pile):
    1. profiles                  (baseline)
    2. profiles + degrees
    3. profiles + GW             (oracle illicit_set: real labels, used
                                   immediately -- the theoretical ceiling
                                   for a graph-distance feature)
    4. profiles + degrees + GW
    5. profiles + degrees + GWd  (delay=7, production-realistic pseudo-
                                   labeling from Stage 8 -- no future info)

A 6th combination (profiles + GWd, no degrees) is also computed, purely to
validate the paper's claim that GWd's disadvantage vs. plain GW shrinks
once degrees are added -- it isn't one of the "5 models" from the spec, so
it's reported in the summary but left out of the saved plots.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from data_gen import generate_accounts, generate_legit_transactions, inject_all_patterns
from features import (
    SlidingGraph,
    build_account_day_table,
    build_hybrid_illicit_set,
    compute_degree_features,
    compute_guilty_walker_delay_features,
    compute_guilty_walker_features,
    compute_profile_features,
    generate_pseudo_labels,
    train_stage1_scorer,
)
from models.metrics import recall_at_fpr
from models.train import temporal_split, train_and_tune
from rules import run_rule_engine

SEED = 42
N_ACCOUNTS = 10_000
N_DAYS = 365
GRAPH_WINDOW = 60
BASE_DATE = pd.Timestamp("2024-01-01")
GW_N_WALKS = 20
GW_MAX_STEPS = 15
GWD_DELAY_DAYS = 7
GWD_THRESHOLD = 0.25
N_TUNE_TRIALS = 50
TUNE_TIMEOUT_SECONDS = 180
TARGET_FPR = 0.20
TRAIN_FRAC = 0.6
VAL_FRAC = 0.1

OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"


def build_dataset(
    n_accounts: int = N_ACCOUNTS, n_days: int = N_DAYS, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accounts = generate_accounts(n_accounts, seed=seed)
    legit_txns = generate_legit_transactions(accounts, n_days=n_days, seed=seed)
    sar_txns, involved = inject_all_patterns(accounts, seed=seed, n_days=n_days)
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
    return full_txns, account_day, accounts


def build_profile_features(account_day: pd.DataFrame, full_txns: pd.DataFrame) -> pd.DataFrame:
    profiles = compute_profile_features(account_day, full_txns)
    profiles["day"] = (pd.to_datetime(profiles["date"]) - BASE_DATE).dt.days
    return profiles


def build_degree_and_gw_features(
    account_day: pd.DataFrame, full_txns: pd.DataFrame, oracle_illicit_set: set[str]
) -> pd.DataFrame:
    """One day-by-day graph pass computing both degree features and GW
    features (oracle illicit_set = real labels, used immediately)."""
    targets_by_day = account_day.groupby("day")["account_id"].apply(list).to_dict()
    graph = SlidingGraph(window=GRAPH_WINDOW)
    rows = []
    for day, day_txns in full_txns.groupby("day"):
        for row in day_txns.itertuples():
            graph.add_edge(day, row.sender_id, row.receiver_id, row.amount)
        graph.advance_to(day)
        if day in targets_by_day:
            for account_id in targets_by_day[day]:
                degree_feats = compute_degree_features(graph, account_id)
                gw_feats = compute_guilty_walker_features(
                    graph,
                    account_id,
                    oracle_illicit_set - {account_id},
                    n_walks=GW_N_WALKS,
                    max_steps=GW_MAX_STEPS,
                    seed=SEED,
                )
                rows.append({"account_id": account_id, "day": day, **degree_feats, **gw_feats})
    return pd.DataFrame(rows)


def build_gwd_features(
    account_day: pd.DataFrame, full_txns: pd.DataFrame, hybrid_illicit_set: set[str]
) -> pd.DataFrame:
    """A second day-by-day graph pass computing GWd features (hybrid
    illicit_set from Stage 8). Column names are prefixed gwd_ instead of
    gw_ so they can coexist with the oracle GW columns after merging."""
    targets_by_day = account_day.groupby("day")["account_id"].apply(list).to_dict()
    graph = SlidingGraph(window=GRAPH_WINDOW)
    rows = []
    for day, day_txns in full_txns.groupby("day"):
        for row in day_txns.itertuples():
            graph.add_edge(day, row.sender_id, row.receiver_id, row.amount)
        graph.advance_to(day)
        if day in targets_by_day:
            for account_id in targets_by_day[day]:
                gwd_feats = compute_guilty_walker_delay_features(
                    graph,
                    account_id,
                    hybrid_illicit_set - {account_id},
                    n_walks=GW_N_WALKS,
                    max_steps=GW_MAX_STEPS,
                    seed=SEED,
                )
                renamed = {f"gwd_{key[3:]}": value for key, value in gwd_feats.items()}
                rows.append({"account_id": account_id, "day": day, **renamed})
    return pd.DataFrame(rows)


def build_hybrid_illicit_set_for_experiment(profile_degree_df: pd.DataFrame, profile_cols: list[str], degree_cols: list[str]) -> set[str]:
    """Stage 8's pipeline, reusing this experiment's own temporal ordering:
    the "train" split acts as train_a (stage1 scorer, no future info), and
    val+test act as train_b (where GWd's delay_days=7 window applies)."""
    train_a, val_a, test_a = temporal_split(profile_degree_df, TRAIN_FRAC, VAL_FRAC)
    train_b = pd.concat([val_a, test_a], ignore_index=True)

    feature_cols = profile_cols + degree_cols
    stage1_model = train_stage1_scorer(train_a, feature_cols)
    pseudo_labels = generate_pseudo_labels(stage1_model, train_b, GWD_DELAY_DAYS, GWD_THRESHOLD)
    return build_hybrid_illicit_set(train_b, train_b["label"], pseudo_labels, GWD_DELAY_DAYS)


def train_and_evaluate(
    combined: pd.DataFrame, feature_sets: dict[str, list[str]]
) -> tuple[dict[str, dict], dict[str, object]]:
    train, val, test = temporal_split(combined, TRAIN_FRAC, VAL_FRAC)
    print(
        f"train/val/test sizes: {len(train):,}/{len(val):,}/{len(test):,}  "
        f"positives: {train['label'].sum()}/{val['label'].sum()}/{test['label'].sum()}"
    )

    results = {}
    models = {}
    for name, feature_cols in feature_sets.items():
        t0 = time.time()
        model = train_and_tune(
            train[feature_cols],
            train["label"].astype(int),
            val[feature_cols],
            val["label"].astype(int),
            model_type="lightgbm",
            n_trials=N_TUNE_TRIALS,
            target_fpr=TARGET_FPR,
            seed=SEED,
            timeout=TUNE_TIMEOUT_SECONDS,
        )
        test_scores = model.predict_proba(test[feature_cols])[:, 1]
        test_recall = recall_at_fpr(test["label"], test_scores, TARGET_FPR)
        fpr, tpr, _ = roc_curve(test["label"], test_scores)

        results[name] = {"test_recall_at_fpr": test_recall, "fpr": fpr, "tpr": tpr, "n_features": len(feature_cols)}
        models[name] = model
        print(f"[{name}] {len(feature_cols)} features, tuned in {time.time() - t0:.1f}s, test recall@{TARGET_FPR:.0%}FPR = {test_recall:.3f}")

    return results, models


def plot_recall_vs_fpr(results: dict[str, dict], plot_names: list[str], path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for name in plot_names:
        r = results[name]
        plt.plot(r["fpr"], r["tpr"], label=f"{name} (recall@{TARGET_FPR:.0%}={r['test_recall_at_fpr']:.2f})")
    plt.axvline(TARGET_FPR, color="gray", linestyle="--", linewidth=1, label=f"target FPR={TARGET_FPR:.0%}")
    plt.plot([0, 1], [0, 1], color="lightgray", linestyle=":", linewidth=1, label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Recall (True Positive Rate)")
    plt.title("Recall vs. FPR by feature set (test set)")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_cols: list[str], path: Path, top_n: int = 25) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, 8))
    importances.iloc[::-1].plot.barh()
    plt.xlabel("LightGBM feature importance (split count)")
    plt.title("Top feature importances, best model")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    full_txns, account_day, accounts = build_dataset()
    print(f"accounts={len(accounts):,} txns={len(full_txns):,} account-days={len(account_day):,} "
          f"positives={account_day['label'].sum()} ({time.time() - t0:.1f}s)")

    profiles = build_profile_features(account_day, full_txns)
    profile_cols = [c for c in profiles.columns if c not in ("account_id", "date", "day", "label")]
    print(f"profile features: {len(profile_cols)} columns ({time.time() - t0:.1f}s)")

    oracle_illicit_set = set(account_day.loc[account_day["label"], "account_id"])
    degree_gw = build_degree_and_gw_features(account_day, full_txns, oracle_illicit_set)
    degree_cols = [c for c in degree_gw.columns if c not in ("account_id", "day") and not c.startswith("gw_")]
    gw_cols = [c for c in degree_gw.columns if c.startswith("gw_")]
    print(f"degree+GW features: {len(degree_cols)}+{len(gw_cols)} columns ({time.time() - t0:.1f}s)")

    profile_degree = profiles.merge(degree_gw[["account_id", "day"] + degree_cols], on=["account_id", "day"], how="inner")
    hybrid_illicit_set = build_hybrid_illicit_set_for_experiment(profile_degree, profile_cols, degree_cols)
    print(f"hybrid illicit set (delay={GWD_DELAY_DAYS}d): {len(hybrid_illicit_set)} accounts ({time.time() - t0:.1f}s)")

    gwd = build_gwd_features(account_day, full_txns, hybrid_illicit_set)
    gwd_cols = [c for c in gwd.columns if c.startswith("gwd_")]
    print(f"GWd features: {len(gwd_cols)} columns ({time.time() - t0:.1f}s)")

    combined = (
        profiles.merge(degree_gw, on=["account_id", "day"], how="inner")
        .merge(gwd, on=["account_id", "day"], how="inner")
    )
    print(f"combined table: {combined.shape} ({time.time() - t0:.1f}s)")

    feature_sets = {
        "profiles": profile_cols,
        "profiles+degrees": profile_cols + degree_cols,
        "profiles+GW": profile_cols + gw_cols,
        "profiles+degrees+GW": profile_cols + degree_cols + gw_cols,
        "profiles+degrees+GWd": profile_cols + degree_cols + gwd_cols,
        "profiles+GWd": profile_cols + gwd_cols,  # validation-only, not one of the 5 headline models
    }

    results, models = train_and_evaluate(combined, feature_sets)

    baseline = results["profiles"]["test_recall_at_fpr"]
    print(f"\n=== recall@{TARGET_FPR:.0%}FPR by feature set (delta vs. profiles-only baseline) ===")
    for name in feature_sets:
        r = results[name]
        delta = r["test_recall_at_fpr"] - baseline
        print(f"{name:<22} recall={r['test_recall_at_fpr']:.3f}  delta={delta:+.3f}  n_features={r['n_features']}")

    headline = ["profiles", "profiles+degrees", "profiles+GW", "profiles+degrees+GW", "profiles+degrees+GWd"]
    plot_recall_vs_fpr(results, headline, OUTPUT_DIR / "recall_vs_fpr.png")
    print(f"\nsaved {OUTPUT_DIR / 'recall_vs_fpr.png'}")

    best_name = max(headline, key=lambda name: results[name]["test_recall_at_fpr"])
    best_model = models[best_name]
    with open(OUTPUT_DIR / "best_model.pkl", "wb") as f:
        pickle.dump({"name": best_name, "model": best_model, "feature_cols": feature_sets[best_name]}, f)
    print(f"saved {OUTPUT_DIR / 'best_model.pkl'} (best: {best_name}, recall={results[best_name]['test_recall_at_fpr']:.3f})")

    plot_feature_importance(best_model, feature_sets[best_name], OUTPUT_DIR / "feature_importance.png")
    print(f"saved {OUTPUT_DIR / 'feature_importance.png'}")

    print(f"\ntotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
