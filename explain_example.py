"""
explain_example.py - finds a benchmark case where exact keyword matching fails but the graph method
succeeds, and generates the automatic explanation. Writes results/explanation_example.json.
"""
import json
import os

import numpy as np

import gm


def ranks_of(M, j, i):
    s = M[j]
    return int((s > s[i]).sum() + ((s == s[i]).sum() + 1) / 2)       # expected rank under random ties


def select_case(d, ctx, seed_search=True):
    M1, MC = gm.b1_keyword(d, ctx), gm.g_c_centrality(d, ctx)
    best, best_gain = None, -1
    for j in range(d.m):
        for i in np.nonzero(d.cov[j] >= 1.0 - 1e-9)[0]:
            if M1[j, i] > 0:                      # want: no exact shared keyword at all
                continue
            matched = [k for k in np.nonzero(d.B[j])[0] if d.A[i, k] > 0 and k < gm.N_DOMAIN_SKILLS]
            nreq = int(sum(1 for k in np.nonzero(d.B[j])[0] if k < gm.N_DOMAIN_SKILLS))
            if len(matched) < 2 or len(matched) >= nreq:   # need >=2 matches AND a gap for the graph to bridge
                continue
            rc = ranks_of(MC, j, i)
            if rc > 15:
                continue
            gain = ranks_of(M1, j, i) - rc
            if gain > best_gain:
                best, best_gain = (j, int(i)), gain
    return best


def main(seed=0):
    d = gm.generate(seed=seed); ctx = gm.build_ctx(d)
    j, i = select_case(d, ctx)
    M1, MC = gm.b1_keyword(d, ctx), gm.g_c_centrality(d, ctx)
    out = {
        "seed": seed, "job": f"J{j:04d}", "candidate": f"C{i:05d}",
        "job_role": d.job_role[j],
        "job_lists": [w for _, w in d.job_mentions[j]],
        "candidate_lists": [w for _, w in d.cand_mentions[i]],
        "shared_exact_keywords": sorted(set(w.lower() for _, w in d.job_mentions[j]) &
                                        set(w.lower() for _, w in d.cand_mentions[i])),
        "ground_truth_coverage": float(d.cov[j, i]),
        "rank_exact_keyword": ranks_of(M1, j, i),
        "rank_centrality_weighted": ranks_of(MC, j, i),
        "n_candidates": d.n,
        "explanation": gm.explain(d, ctx, j, i),
    }
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/explanation_example.json", "w"), indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
