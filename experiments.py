"""
experiments.py - runs every experiment and writes results/*.csv and results/summary.json.
Usage:  python experiments.py          (about 2-4 minutes)
"""
import json
import os
import time

import numpy as np
import pandas as pd

import gm

OUT = "results"
os.makedirs(OUT, exist_ok=True)
SEEDS5 = [0, 1, 2, 3, 4]
SEEDS3 = [0, 1, 2]


def e1_main():
    rows = []
    for s in SEEDS5:
        d = gm.generate(seed=s); ctx = gm.build_ctx(d)
        for name, r in gm.run_all(d, ctx, seed=s).items():
            rows.append({"seed": s, "method": name, **r})
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/e1_main_all_seeds.csv", index=False)
    agg = df.groupby("method", sort=False).agg(["mean", "std"])
    agg.to_csv(f"{OUT}/e1_main_summary.csv")
    return df


def sweep(param, values, fname):
    rows = []
    for v in values:
        for s in SEEDS3:
            d = gm.generate(seed=s, **{param: v}); ctx = gm.build_ctx(d)
            for name, r in gm.run_all(d, ctx, seed=s).items():
                rows.append({param: v, "seed": s, "method": name, **r})
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/{fname}", index=False)
    return df


def e6_scalability():
    rows = []
    for n in [500, 1000, 2000, 4000, 8000]:
        d = gm.generate(n_cand=n, seed=0); ctx = gm.build_ctx(d)
        for name, fn in gm.METHODS.items():
            best = 1e9
            for _ in range(3):                      # best of 3 to reduce timing noise
                t0 = time.perf_counter(); fn(d, ctx); best = min(best, time.perf_counter() - t0)
            rows.append({"n_candidates": n, "method": name, "time_s": best, "graph_build_s": ctx["t_build"]})
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/e6_scalability.csv", index=False)
    return df


def e7_bipartite():
    """300 jobs filled from a pool of 360 candidates, greedy vs Hungarian (Algorithm A)."""
    rows = []
    for s in SEEDS5:
        d = gm.generate(seed=s); ctx = gm.build_ctx(d)
        pool = np.random.default_rng(100 + s).choice(d.n, 360, replace=False)
        for name, fn in gm.METHODS.items():
            M = fn(d, ctx)
            g, h = gm.assignment_experiment(d, M, pool, seed=s)
            rows.append({"seed": s, "method": name, "greedy": g, "hungarian": h})
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/e7_bipartite_all_seeds.csv", index=False)
    return df


def structure_summary():
    d = gm.generate(seed=0); ctx = gm.build_ctx(d)
    out = {
        "dataset_seed0": gm.dataset_stats(d),
        "hetero_graph": {"nodes": ctx["G"].number_of_nodes(), "edges": ctx["G"].number_of_edges()},
        "skill_graph": {"nodes": ctx["Gs"].number_of_nodes(), "edges": ctx["Gs"].number_of_edges()},
        "louvain": {"communities": len(ctx["communities"]), "modularity": ctx["modularity"],
                    "NMI_vs_domains": gm.community_nmi(ctx)},
        "top_degree_centrality": gm.top_centrality(ctx),
        "top_betweenness_skill_graph": gm.betweenness_top(ctx),
        "n_aliases": sum(len(v) for v in gm.ALIASES.values()),
    }
    # NMI over all seeds
    out["louvain"]["NMI_all_seeds"] = []
    for s in SEEDS5:
        c = gm.build_ctx(gm.generate(seed=s))
        out["louvain"]["NMI_all_seeds"].append(
            {"seed": s, "communities": len(c["communities"]), "modularity": c["modularity"], "NMI": gm.community_nmi(c)})
    return out


if __name__ == "__main__":
    t0 = time.time()
    print("E1 main comparison ...");       df1 = e1_main()
    print("E4 alias-rate sweep ...");      sweep("alias", [0.0, 0.2, 0.35, 0.5, 0.7], "e4_alias_rate.csv")
    print("E5 listing-rate sweep ...");    sweep("listing", [0.3, 0.45, 0.6, 0.8, 1.0], "e5_sparsity.csv")
    print("E6 scalability ...");           e6_scalability()
    print("E7 bipartite matching ...");    e7_bipartite()
    print("Structure summary ...")
    summ = structure_summary()
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=2, default=float)

    # paired comparison G-C vs B3 over seeds (per-seed difference)
    p = df1.pivot(index="seed", columns="method", values="J2C_NDCG@10")
    diff = p["G-C Centrality-weighted overlap"] - p["B3 Keyword + normalisation"]
    print("\nG-C minus B3 NDCG@10 per seed:", diff.round(4).tolist())
    print(f"done in {time.time() - t0:.0f}s")
