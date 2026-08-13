# %%
# Inspect synthetic legitimate transaction data: degree distribution,
# per-persona amount distribution, and summary stats.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_gen import generate_accounts, generate_legit_transactions

SEED = 42
N_ACCOUNTS = 50_000
N_DAYS = 365

# %%
t0 = time.time()
accounts = generate_accounts(N_ACCOUNTS, seed=SEED)
txns = generate_legit_transactions(accounts, n_days=N_DAYS, seed=SEED)
print(f"generated {len(accounts):,} accounts, {len(txns):,} txns in {time.time() - t0:.1f}s")

accounts.head()

# %%
txns.head()

# %%
# Degree = number of distinct counterparties per account (sender or receiver side).
both_sides = pd.concat(
    [
        txns[["sender_id", "receiver_id"]].rename(
            columns={"sender_id": "account_id", "receiver_id": "counterparty"}
        ),
        txns[["receiver_id", "sender_id"]].rename(
            columns={"receiver_id": "account_id", "sender_id": "counterparty"}
        ),
    ]
)
degree = both_sides.groupby("account_id")["counterparty"].nunique()
degree = degree.reindex(accounts["account_id"], fill_value=0)

degree_counts = degree[degree > 0].value_counts().sort_index()

plt.figure(figsize=(6, 5))
plt.loglog(degree_counts.index, degree_counts.values, marker="o", linestyle="none", alpha=0.6)
plt.xlabel("degree (distinct counterparties)")
plt.ylabel("number of accounts")
plt.title("Degree distribution (log-log)")
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "degree_distribution.png", dpi=150)
plt.show()

# %%
# Amount distribution per persona (sender's persona).
txns_with_persona = txns.merge(
    accounts[["account_id", "persona"]], left_on="sender_id", right_on="account_id", how="left"
)

plt.figure(figsize=(7, 5))
for persona, group in txns_with_persona.groupby("persona"):
    plt.hist(
        np.log10(group["amount"]),
        bins=60,
        alpha=0.5,
        label=persona,
        density=True,
    )
plt.xlabel("log10(amount)")
plt.ylabel("density")
plt.title("Transaction amount distribution by persona")
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "amount_distribution_by_persona.png", dpi=150)
plt.show()

# %%
# Summary stats
txns_per_account_as_sender = txns.groupby("sender_id").size().reindex(
    accounts["account_id"], fill_value=0
)

print("=== Summary stats ===")
print(f"accounts: {len(accounts):,}")
print(f"txns: {len(txns):,}")
print(f"avg txns/account (as sender): {txns_per_account_as_sender.mean():.2f}")
print(f"avg counterparties/account (degree): {degree.mean():.2f}")
print(f"median degree: {degree.median():.1f}, max degree: {degree.max():.0f}")
print()
print("txns/account (as sender) by persona:")
print(
    txns_per_account_as_sender.groupby(accounts.set_index("account_id")["persona"]).mean()
)
print()
print("degree by persona:")
print(degree.groupby(accounts.set_index("account_id")["persona"]).mean())

# %%
# Determinism check: regenerating with the same seed must be identical.
accounts_repeat = generate_accounts(N_ACCOUNTS, seed=SEED)
txns_repeat = generate_legit_transactions(accounts_repeat, n_days=N_DAYS, seed=SEED)
print("accounts identical:", accounts.equals(accounts_repeat))
print("txns identical:", txns.equals(txns_repeat))
