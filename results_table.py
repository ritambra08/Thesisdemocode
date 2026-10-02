"""results_table.py - prints the real result tables (Markdown) from results/*.csv, to paste into the report/slides."""
import json
import pandas as pd

df = pd.read_csv("results/e1_main_all_seeds.csv")
g = df.groupby("method", sort=False)
t = g[["J2C_P@10", "J2C_NDCG@10", "J2C_MAP@100", "C2J_P@5", "C2J_NDCG@5", "time_s"]].mean()
t["NDCG@10 sd"] = g["J2C_NDCG@10"].std(); t["NDCG@5 sd"] = g["C2J_NDCG@5"].std()
print("## Table 4.2 Main comparison (mean over 5 seeds)\n"); print(t.round(3).to_markdown()); print()

for f, c, title in [("e4_alias_rate.csv", "alias", "NDCG@10 vs alias rate"), ("e5_sparsity.csv", "listing", "NDCG@10 vs listing rate")]:
    x = pd.read_csv("results/" + f).groupby([c, "method"])["J2C_NDCG@10"].mean().unstack()
    print(f"## {title}\n"); print(x.round(3).T.to_markdown()); print()

b = pd.read_csv("results/e7_bipartite_all_seeds.csv").groupby("method", sort=False)[["greedy", "hungarian"]].mean()
b["difference"] = b.hungarian - b.greedy
print("## Table 4.3 Bipartite assignment (share of jobs given a relevant candidate)\n"); print((b * 100).round(1).to_markdown()); print()

s = pd.read_csv("results/e6_scalability.csv"); s = s[s.n_candidates == 8000][["method", "time_s"]]
print("## Scoring time, 8000 candidates x 300 jobs\n"); print(s.round(3).to_markdown(index=False)); print()

p = df.pivot(index="seed", columns="method", values="J2C_NDCG@10")
for m in ["G-C Centrality-weighted overlap", "G-E Personalised PageRank"]:
    print(f"{m} - B3 per-seed NDCG@10 difference:", (p[m] - p["B3 Keyword + normalisation"]).round(4).tolist())
print("\n", json.dumps(json.load(open("results/summary.json"))["louvain"]["NMI_all_seeds"][0]))
