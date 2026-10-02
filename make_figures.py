"""
make_figures.py - produces all report/presentation figures into figures/ (PNG, 300 dpi).
Run after experiments.py and explain_example.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gm
from explain_example import select_case

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": .3})
BASE = ["B1", "B2", "B3", "B4"]
GREEN, GREY = "#0f766e", "#9ca3af"
SHORT = {"B1 Exact keyword (Jaccard)": "Keyword (exact)", "B2 TF-IDF cosine": "TF-IDF cosine",
         "B3 Keyword + normalisation": "Keyword + normalisation", "B4 LSA semantic vectors": "LSA",
         "G-B Skill-graph similarity": "Graph skill-graph similarity (B)",
         "G-C Centrality-weighted overlap": "Graph centrality-weighted (C)",
         "G-D Community profile": "Graph community profile (D)",
         "G-E Personalised PageRank": "Graph personalised PageRank (E)"}
STYLE = {m: dict(color=c, ls="--" if m.startswith("B") else "-", marker="o", ms=3.5) for m, c in zip(
    SHORT, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#9467bd", "#e377c2", "#7f7f7f"])}


def fig_architecture():
    fig, ax = plt.subplots(figsize=(9, 1.7)); ax.axis("off")
    labels = ["1. Data\nprocessing", "2. Skill extraction\n& normalisation", "3. Heterogeneous\ngraph construction",
              "4. Graph analysis\n(centrality, PPR,\ncommunities)", "5. Matching &\nrecommendation", "6. Evaluation"]
    for k, t in enumerate(labels):
        x = k * 1.6
        ax.add_patch(plt.Rectangle((x, 0), 1.3, 1, fc="#e0f2f1" if k < 5 else "#fef3c7", ec=GREEN, lw=1.2))
        ax.text(x + .65, .5, t, ha="center", va="center", fontsize=7.5)
        if k < 5: ax.annotate("", xy=(x + 1.6, .5), xytext=(x + 1.3, .5), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-.1, 9.6); ax.set_ylim(-.1, 1.1)
    fig.savefig("figures/fig3_1_architecture.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_example_graph():
    d = gm.generate(seed=0); ctx = gm.build_ctx(d)
    j, i = select_case(d, ctx)
    W = ctx["W"]; nd = gm.N_DOMAIN_SKILLS
    jm = [(k, w) for k, w in d.job_mentions[j] if k < nd]
    cm = [(k, w) for k, w in d.cand_mentions[i] if k < nd]
    req = [k for k, _ in jm]
    skills = sorted(set(req) | set(k for k, _ in cm), key=lambda k: (k not in req, k))
    ys = np.linspace(len(skills) - 1, 0, len(skills)) * 1.3
    sx = 5.0
    pos = {k: (sx, y) for k, y in zip(skills, ys)}
    jp, cp = (0.0, ys.mean()), (10.0, ys.mean())
    fig, ax = plt.subplots(figsize=(8.5, 5.2)); ax.axis("off")
    def lab(p, q, t, col, frac):
        x = p[0] + (q[0] - p[0]) * frac; y = p[1] + (q[1] - p[1]) * frac
        ax.text(x, y, t, color=col, fontsize=6.5, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=.6), zorder=5)
    for k, w in jm:
        ax.plot(*zip(jp, pos[k]), color="#dc2626", lw=1.3, zorder=1); lab(jp, pos[k], f"'{w}'", "#dc2626", .62)
    for k, w in cm:
        ax.plot(*zip(cp, pos[k]), color="#2563eb", lw=1.3, zorder=1); lab(cp, pos[k], f"'{w}'", "#2563eb", .62)
    for a_ in skills:
        for b_ in skills:
            if a_ < b_ and W[a_, b_] >= 0.3 and ((a_ in req) != (b_ in req)):
                (x1, y1), (x2, y2) = pos[a_], pos[b_]
                ax.annotate("", xy=(x2 + .45, y2), xytext=(x1 + .45, y1),
                            arrowprops=dict(arrowstyle="-", ls=":", color="#374151", lw=1.2, connectionstyle="arc3,rad=-0.45"), zorder=2)
                ax.text(sx + 1.05 + abs(y1 - y2) * .12, (y1 + y2) / 2, f"w={W[a_, b_]:.2f}", fontsize=6.5, color="#374151", zorder=5)
    for k, (x, y) in pos.items():
        ax.scatter(x, y, s=1500, color="#6ee7b7", ec="k", lw=.6, zorder=3)
        ax.text(x, y, gm.ALL[k].replace(" ", "\n", 1) if len(gm.ALL[k]) > 10 else gm.ALL[k], ha="center", va="center", fontsize=6.5, zorder=4)
    ax.scatter(*jp, s=2200, color="#fca5a5", ec="k", zorder=3); ax.text(*jp, f"J{j:04d}\n(job)", ha="center", va="center", fontsize=7.5, zorder=4)
    ax.scatter(*cp, s=2200, color="#93c5fd", ec="k", zorder=3); ax.text(*cp, f"C{i:05d}\n(candidate)", ha="center", va="center", fontsize=7, zorder=4)
    ax.plot([], [], color="#dc2626", label="job requires (as written)"); ax.plot([], [], color="#2563eb", label="candidate lists (as written)")
    ax.plot([], [], color="#374151", ls=":", label="skill co-occurrence edge (weight w)")
    ax.legend(loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(.5, -.07))
    ax.set_xlim(-1.3, 11.3); ax.set_ylim(-1.0, ys.max() + .9)
    fig.savefig("figures/fig3_2_example_graph.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_main():
    df = pd.read_csv("results/e1_main_all_seeds.csv")
    g = df.groupby("method", sort=False)["J2C_NDCG@10"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = list(g.index)[::-1]
    ax.barh([SHORT[n] for n in names], g.loc[names, "mean"], xerr=g.loc[names, "std"],
            color=[GREY if n.startswith("B") else GREEN for n in names], capsize=2)
    for y, n in enumerate(names): ax.text(g.loc[n, "mean"] + .02, y, f"{g.loc[n, 'mean']:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.08); ax.set_xlabel("NDCG@10, job → candidate ranking (mean ± s.d., 5 random seeds)")
    fig.savefig("figures/fig4_1_main_ndcg.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_sweep(fname, col, xlabel, out):
    df = pd.read_csv(f"results/{fname}").groupby([col, "method"], sort=False)["J2C_NDCG@10"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m in SHORT: ax.plot(df.index, df[m], label=SHORT[m], **STYLE[m])
    ax.set_xlabel(xlabel); ax.set_ylabel("NDCG@10"); ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.22), frameon=False)
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_scale():
    df = pd.read_csv("results/e6_scalability.csv")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m in SHORT:
        s = df[df.method == m]; ax.plot(s.n_candidates, s.time_s, label=SHORT[m], **STYLE[m])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Number of candidates (300 jobs)")
    ax.set_ylabel("Scoring time for all pairs (s)"); ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.22), frameon=False)
    fig.savefig("figures/fig4_4_scalability.png", dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_architecture(); fig_example_graph(); fig_main()
    fig_sweep("e4_alias_rate.csv", "alias", "Share of skills written with an alias (e.g. 'ML' instead of 'Machine Learning')", "figures/fig4_2_alias.png")
    fig_sweep("e5_sparsity.csv", "listing", "Share of a candidate's real skills that appear on the profile (listing rate)", "figures/fig4_3_sparsity.png")
    fig_scale()
    print(sorted(os.listdir("figures")))
