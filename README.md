# aml-fp-pipeline

ML pipeline for reducing false positives in AML (Anti-Money Laundering) transaction monitoring alerts, based on the EDDIN triage-graph approach.

## Pipeline stages

1. **`data_gen/`** — Synthetic ledger generator and laundering typology injectors. Produces a base population of "normal" transactions plus injected patterns (structuring, layering, smurfing, rapid movement, etc.) with ground-truth labels.
   - `accounts.py` — `generate_accounts()`: persona-clustered account population (retail/small_business/high_volume_merchant) with fixed "regular" counterparties per account.
   - `legit_transactions.py` — `generate_legit_transactions()`: legitimate transaction stream driven by persona frequency/amount distributions.
   - `typologies.py` — `inject_smurfing()`, `inject_layering()`, `inject_fan_out_fan_in()`, `inject_round_tripping()`, and `inject_all_patterns()`: laundering typology injectors that layer labeled (`is_sar=True`) transactions onto existing accounts, tracking used accounts so pattern instances never overlap.

2. **`rules/`** — Rule engine that scans the synthetic (or real) transaction ledger and raises alerts, mimicking a traditional AML monitoring system. This is the noisy, high-false-positive baseline the rest of the pipeline learns to triage.
   - `engine.py` — `large_single_txn`, `round_amount_near_threshold`, `high_daily_velocity`, `sudden_volume_spike`, and `run_rule_engine()` which combines all four into one alerts table (`account_id`, `date`, `triggered_rules`, `any_true_positive`).

3. **`features/`** — Feature engineering over alerted entities/accounts:
   - `profiles.py` — `build_account_day_table()` collapses transactions to one row per alerted (account_id, date); `compute_profile_features()` computes rolling sent/received sum/mean/min/max/count over several trailing windows (1d/1w/2w/1mo/2mo) plus ratio/diff features vs. the 1-day window, using full transaction history for lookback with no lookahead leakage; `select_top_profile_features()` reduces the feature set via permutation importance.
   - `graph.py` — `SlidingGraph`: incremental sliding-window transaction graph (dict-of-deques adjacency, O(1) amortized edge eviction, no per-day networkx rebuild).
   - `degree_features.py` — `compute_degree_features()`: a node's own in/out-degree and weighted in/out-degree (summed amounts), plus mean/min/max in/out-degree (weighted and unweighted) of its 1-hop neighbors.
   - `guilty_walker.py` — `compute_guilty_walker_features()`: runs random walks from an account toward a known-illicit set, moving along in/out edges either direction; returns walk-length distribution stats (min/max/mean/median/std/p25/p75), hit_rate, and distinct-illicit-nodes-reached. Also the delay-aware ("GWd") pipeline: `train_stage1_scorer()` (profile+degree features only, no illicit_set needed), `generate_pseudo_labels()` for the not-yet-settled recent window, `build_hybrid_illicit_set()` (real labels where settled, pseudo labels where not), and `compute_guilty_walker_delay_features()`.

4. **`models/`** — Model training and evaluation:
   - `metrics.py` — `recall_at_fpr()`: recall at a target false-positive rate, interpolated off `sklearn.roc_curve`.
   - `train.py` — `temporal_split()` (date-sorted train/val/test, no shuffling) and `train_and_tune()`: Optuna-tuned GLM/RandomForest/LightGBM, model selected by validation `recall_at_fpr(0.20)`, using the paper's hyperparameter ranges.
   - `run_experiment.py` — assembles the full pipeline (Stages 2-8), trains LightGBM on five increasing feature sets (profiles → +degrees → +GW → +degrees+GW → +degrees+GWd), reports test-set `recall_at_fpr(0.20)` deltas vs. the profiles-only baseline, and saves a recall-vs-FPR plot, the best model, and a feature-importance plot to `models/artifacts/`.
   - `window_sweep.py` — reproduces the paper's Figure-5-style sweep: refits the tuned profiles+degrees architecture across a 6×6 grid of (TWL, TWS) sliding-window sizes and saves a `recall_at_fpr(0.20)` heatmap.

5. **`configs/`** — YAML configuration for window sizes, rule thresholds, and hyperparameter search ranges, plus `config_loader.py` to load them into Python.

6. **`notebooks/`** — Exploratory scripts using `# %%` cell markers (VSCode Jupyter-style), for ad hoc analysis outside the pipeline proper.

7. **`tests/`** — Unit tests for the above.

## Status

- `data_gen/` — implemented (legit transactions + laundering typology injection). See `notebooks/01_inspect_legit_data.py` and `notebooks/02_inspect_patterns.py`.
- `rules/` — implemented (rule engine → alerts). See `notebooks/03_inspect_alerts.py`.
- `features/` — implemented: account-day aggregation + rolling profile features (`notebooks/04_inspect_profiles.py`), sliding-window graph + degree features (`notebooks/05_inspect_graph.py`), GuiltyWalker random-walk features (`notebooks/06_inspect_guilty_walker.py`, `tests/test_guilty_walker.py`), and the delay-aware GWd pseudo-labeling pipeline (`notebooks/07_inspect_gwd.py`).
- `models/` — implemented: metrics, temporal split + tuning, and the end-to-end experiment (`python -m models.run_experiment`) and window sweep (`python -m models.window_sweep`) scripts. Both scripts run the full data-generation-through-evaluation pipeline and take real time (tens of minutes at their default account counts).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
