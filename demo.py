"""
demo.py - small live demo for the viva (runs in about 2 seconds).
  python demo.py
"""
import numpy as np

import gm

d = gm.generate(n_cand=200, n_jobs=30, seed=7)
ctx = gm.build_ctx(d)
print(f"Dataset : {d.n} candidates, {d.m} jobs, {len(gm.SKILLS)} canonical skills")
print(f"Hetero graph : {ctx['G'].number_of_nodes()} nodes, {ctx['G'].number_of_edges()} edges")
print(f"Skill graph  : {ctx['Gs'].number_of_nodes()} nodes, {ctx['Gs'].number_of_edges()} weighted edges")
print(f"Communities  : {len(ctx['communities'])}  (modularity {ctx['modularity']:.3f}, NMI vs domains {gm.community_nmi(ctx):.2f})\n")

j = 3
print(f"JOB J{j:04d} ({d.job_role[j]}) lists: {[w for _, w in d.job_mentions[j]]}\n")
for label, fn in [("Exact keyword (B1)", gm.b1_keyword), ("Centrality-weighted graph (G-C)", gm.g_c_centrality)]:
    M = fn(d, ctx)
    top = np.argsort(-M[j])[:5]
    print(f"Top-5 candidates by {label}:")
    for i in top:
        print(f"   C{i:05d}  score={M[j, i]:.3f}  truth coverage={d.cov[j, i]:.0%}  relevance={d.rel[j, i]}  lists={[w for _, w in d.cand_mentions[i]]}")
    print()

M = gm.g_c_centrality(d, ctx)
i = int(np.argmax(M[j]))
print("Automatic explanation for the top graph recommendation:")
print("  " + gm.explain(d, ctx, j, i))
