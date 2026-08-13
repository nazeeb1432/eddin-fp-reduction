"""Temporal train/val/test split and hyperparameter-tuned model training,
selected by recall_at_fpr(0.20) on a held-out validation set.
"""

from __future__ import annotations

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.metrics import recall_at_fpr

optuna.logging.set_verbosity(optuna.logging.WARNING)

# The paper's hyperparameter ranges.
DEFAULT_PARAM_GRIDS = {
    "glm": {"C": (0.01, 0.09), "standardize": [True, False]},
    "random_forest": {"max_depth": (10, 40), "n_estimators": (100, 200), "min_samples_split": (10, 50)},
    "lightgbm": {"num_leaves": (200, 500), "min_child_samples": (100, 200), "learning_rate": (0.01, 0.09)},
}


def temporal_split(
    account_day_df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split account_day_df into (train, val, test) by date, oldest first,
    no shuffling -- train_frac/val_frac/remaining-as-test are fractions of
    ROW COUNT, not of the date range (rows aren't evenly spread over time).
    account_id is a secondary sort key purely to make the split
    deterministic when many rows share the same date.
    """
    df = account_day_df.sort_values(["date", "account_id"], kind="mergesort").reset_index(drop=True)
    n = len(df)
    train_end = int(round(n * train_frac))
    val_end = train_end + int(round(n * val_frac))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def _sample_params(model_type: str, param_grid: dict, trial: optuna.Trial) -> dict:
    if model_type == "glm":
        return {
            "C": trial.suggest_float("C", *param_grid["C"]),
            "standardize": trial.suggest_categorical("standardize", param_grid["standardize"]),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }
    if model_type == "random_forest":
        return {
            "max_depth": trial.suggest_int("max_depth", *param_grid["max_depth"]),
            "n_estimators": trial.suggest_int("n_estimators", *param_grid["n_estimators"]),
            "min_samples_split": trial.suggest_int("min_samples_split", *param_grid["min_samples_split"]),
        }
    if model_type == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", *param_grid["num_leaves"]),
            "min_child_samples": trial.suggest_int("min_child_samples", *param_grid["min_child_samples"]),
            "learning_rate": trial.suggest_float("learning_rate", *param_grid["learning_rate"]),
        }
    raise ValueError(f"unknown model_type: {model_type!r}")


def _build_model(model_type: str, params: dict, seed: int):
    if model_type == "glm":
        clf = LogisticRegression(
            C=params["C"],
            penalty=params["penalty"],
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
            max_iter=1000,
        )
        if params["standardize"]:
            return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        return clf
    if model_type == "random_forest":
        return RandomForestClassifier(
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            min_samples_split=params["min_samples_split"],
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_type == "lightgbm":
        return lgb.LGBMClassifier(
            num_leaves=params["num_leaves"],
            min_child_samples=params["min_child_samples"],
            learning_rate=params["learning_rate"],
            class_weight="balanced",
            random_state=seed,
            verbosity=-1,
            # LightGBM auto-detects row-wise vs col-wise histogram building
            # per fit based on data shape, and that heuristic can pick a
            # dramatically slower strategy in a way that's hard to predict
            # (observed one fit-configuration take 20-40x longer than
            # others on otherwise-comparable data). Forcing row-wise --
            # the documented recommendation for datasets in the low tens
            # of thousands of rows, which is this project's scale
            # throughout -- avoids relying on that heuristic at all.
            force_row_wise=True,
        )
    raise ValueError(f"unknown model_type: {model_type!r}")


def train_and_tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str,
    param_grid: dict | None = None,
    n_trials: int = 50,
    target_fpr: float = 0.20,
    seed: int = 42,
    timeout: float | None = None,
):
    """Tune model_type ("glm" | "random_forest" | "lightgbm") over
    param_grid (defaults to the paper's ranges) via Optuna, selecting the
    trial with the best recall_at_fpr(target_fpr) on (X_val, y_val).
    Refits the winning hyperparameters on X_train/y_train and returns that
    model.

    timeout (seconds, optional) caps the whole study as a safety net --
    Optuna will stop issuing new trials once elapsed time exceeds it, so a
    handful of unusually slow fits can't make the run open-ended. Search
    quality degrades gracefully (fewer completed trials), it doesn't fail.
    """
    param_grid = param_grid or DEFAULT_PARAM_GRIDS[model_type]

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(model_type, param_grid, trial)
        model = _build_model(model_type, params, seed)
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        return recall_at_fpr(y_val, scores, target_fpr)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_model = _build_model(model_type, study.best_params, seed)
    best_model.fit(X_train, y_train)
    return best_model
