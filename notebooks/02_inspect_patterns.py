# %%
# Inspect laundering typology injectors: draw each pattern's subgraph in
# isolation, print its transactions/accounts for manual tracing, and
# validate suspicious-account share + connected-component isolation at
# scale.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from data_gen import (
    generate_accounts,
    inject_all_patterns,
    inject_fan_out_fan_in,
    inject_layering,
    inject_round_tripping,
    inject_smurfing,
)

SEED = 7
N_ACCOUNTS = 50
OUT_DIR = Path(__file__).resolve().parent

ROLE_COLORS = {
    "collector": "tab:red",
    "source": "tab:red",
    "origin": "tab:red",
    "mule": "tab:blue",
    "intermediate": "tab:blue",
    "destination": "tab:green",
}

# %%
accounts = generate_accounts(N_ACCOUNTS, seed=SEED)
accounts[["account_id", "persona", "account_type"]]


# %%
def _chain_order(txns: pd.DataFrame) -> list[str]:
    """Reconstruct account visit order along a simple chain/loop by walking
    transactions in timestamp order (dedupes the repeated origin at the end
    of a round-trip loop)."""
    txns_sorted = txns.sort_values("timestamp")
    order = [txns_sorted.iloc[0]["sender_id"], *txns_sorted["receiver_id"]]
    seen: set[str] = set()
    uniq: list[str] = []
    for node in order:
        if node not in seen:
            uniq.append(node)
            seen.add(node)
    return uniq


def smurfing_layout(involved: pd.DataFrame, txns: pd.DataFrame) -> dict:
    collector = involved.loc[involved["role"] == "collector", "account_id"].iloc[0]
    mules = involved.loc[involved["role"] == "mule", "account_id"].tolist()
    destination = involved.loc[involved["role"] == "destination", "account_id"].iloc[0]

    pos = {collector: np.array([0.0, 0.0])}
    n = len(mules)
    for k, m in enumerate(mules):
        angle = 2 * np.pi * k / n
        pos[m] = np.array([np.cos(angle), np.sin(angle)])
    pos[destination] = np.array([2.2, 0.0])
    return pos


def layering_layout(involved: pd.DataFrame, txns: pd.DataFrame) -> dict:
    chain = _chain_order(txns)
    return {node: np.array([float(i), 0.0]) for i, node in enumerate(chain)}


def fan_out_fan_in_layout(involved: pd.DataFrame, txns: pd.DataFrame) -> dict:
    source = involved.loc[involved["role"] == "source", "account_id"].iloc[0]
    intermediates = involved.loc[involved["role"] == "intermediate", "account_id"].tolist()
    destinations = involved.loc[involved["role"] == "destination", "account_id"].tolist()

    pos = {source: np.array([0.0, 2.0])}
    n_int = len(intermediates)
    for k, node in enumerate(intermediates):
        pos[node] = np.array([(k - (n_int - 1) / 2) * 1.0, 1.0])
    n_dest = len(destinations)
    for k, node in enumerate(destinations):
        pos[node] = np.array([(k - (n_dest - 1) / 2) * 1.8, 0.0])
    return pos


def round_tripping_layout(involved: pd.DataFrame, txns: pd.DataFrame) -> dict:
    chain = _chain_order(txns)
    n = len(chain)
    pos = {}
    for i, node in enumerate(chain):
        angle = 2 * np.pi * i / n
        pos[node] = np.array([np.cos(angle), np.sin(angle)])
    return pos


def draw_pattern(txns: pd.DataFrame, involved: pd.DataFrame, layout_fn, title: str, filename: str):
    G = nx.DiGraph()
    role_by_account = dict(zip(involved["account_id"], involved["role"]))
    for acc, role in role_by_account.items():
        G.add_node(acc, role=role)
    for _, row in txns.iterrows():
        G.add_edge(row["sender_id"], row["receiver_id"], amount=row["amount"])

    pos = layout_fn(involved, txns)
    colors = [ROLE_COLORS.get(G.nodes[n]["role"], "gray") for n in G.nodes]

    plt.figure(figsize=(7, 6))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=7)
    nx.draw_networkx_edges(
        G, pos, arrowstyle="-|>", arrowsize=15, connectionstyle="arc3,rad=0.08"
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150)
    plt.show()
    return G


# %%
# --- Smurfing: expect a star / hub-and-spoke shape ---
smurf_txns, smurf_involved = inject_smurfing(accounts, n_patterns=1, seed=SEED, n_days=60)
print("=== smurfing: accounts involved ===")
print(smurf_involved.to_string(index=False))
print("\n=== smurfing: transactions ===")
print(smurf_txns[["txn_id", "sender_id", "receiver_id", "amount", "timestamp"]].to_string(index=False))
draw_pattern(smurf_txns, smurf_involved, smurfing_layout, "Smurfing (expect hub-and-spoke)", "pattern_smurfing.png")

# %%
# --- Layering: expect a straight chain ---
layer_txns, layer_involved = inject_layering(accounts, n_patterns=1, seed=SEED, n_days=60)
print("=== layering: accounts involved ===")
print(layer_involved.to_string(index=False))
print("\n=== layering: transactions ===")
print(layer_txns[["txn_id", "sender_id", "receiver_id", "amount", "timestamp"]].to_string(index=False))
draw_pattern(layer_txns, layer_involved, layering_layout, "Layering (expect a chain)", "pattern_layering.png")

# %%
# --- Fan-out/fan-in: expect an hourglass ---
fan_txns, fan_involved = inject_fan_out_fan_in(accounts, n_patterns=1, seed=SEED, n_days=60)
print("=== fan_out_fan_in: accounts involved ===")
print(fan_involved.to_string(index=False))
print("\n=== fan_out_fan_in: transactions ===")
print(fan_txns[["txn_id", "sender_id", "receiver_id", "amount", "timestamp"]].to_string(index=False))
draw_pattern(
    fan_txns, fan_involved, fan_out_fan_in_layout, "Fan-out/fan-in (expect an hourglass)", "pattern_fan_out_fan_in.png"
)

# %%
# --- Round-tripping: expect a loop back to the origin ---
rtrip_txns, rtrip_involved = inject_round_tripping(accounts, n_patterns=1, seed=SEED, n_days=60)
print("=== round_tripping: accounts involved ===")
print(rtrip_involved.to_string(index=False))
print("\n=== round_tripping: transactions ===")
print(rtrip_txns[["txn_id", "sender_id", "receiver_id", "amount", "timestamp"]].to_string(index=False))
draw_pattern(
    rtrip_txns, rtrip_involved, round_tripping_layout, "Round-tripping (expect a loop)", "pattern_round_tripping.png"
)

# %%
# --- Validation at scale: suspicious account share + component isolation ---
VALIDATION_N_ACCOUNTS = 20_000
big_accounts = generate_accounts(VALIDATION_N_ACCOUNTS, seed=SEED)
sar_txns, sar_involved = inject_all_patterns(big_accounts, seed=SEED, n_days=365)

suspicious_share = sar_involved["account_id"].nunique() / len(big_accounts)
print(f"accounts: {len(big_accounts):,}")
print(f"sar txns: {len(sar_txns):,}")
print(f"unique accounts involved: {sar_involved['account_id'].nunique():,}")
print(f"suspicious account share: {suspicious_share:.3%} (target: 2-3%)")

print("\npattern instance counts:")
print(sar_involved.groupby("pattern_type")["pattern_id"].nunique())

dup_accounts = sar_involved["account_id"].value_counts()
dup_accounts = dup_accounts[dup_accounts > 1]
print(f"\naccounts reused across >1 pattern instance: {len(dup_accounts)} (should be 0)")

# %%
# Connected components of the full SAR-transaction graph: each pattern
# instance should be its own isolated component -- num components should
# equal num pattern instances, and the largest components should be no
# bigger than the largest single pattern instance (a 15-mule smurfing ring:
# collector + mules + destination = 17 accounts).
G_all = nx.from_pandas_edgelist(sar_txns, "sender_id", "receiver_id", create_using=nx.Graph())
components = list(nx.connected_components(G_all))
sizes = sorted((len(c) for c in components), reverse=True)

n_pattern_instances = sar_involved["pattern_id"].nunique()
print(f"connected components: {len(components)}")
print(f"pattern instances: {n_pattern_instances} (should match component count)")
print(f"largest 10 component sizes: {sizes[:10]}")
print(f"graph nodes: {G_all.number_of_nodes()} vs involved accounts: {sar_involved['account_id'].nunique()} (should match)")

# %%
# Determinism check: regenerating with the same seed must be identical.
big_accounts_repeat = generate_accounts(VALIDATION_N_ACCOUNTS, seed=SEED)
sar_txns_repeat, sar_involved_repeat = inject_all_patterns(big_accounts_repeat, seed=SEED, n_days=365)
print("sar_txns identical:", sar_txns.equals(sar_txns_repeat))
print("sar_involved identical:", sar_involved.equals(sar_involved_repeat))
