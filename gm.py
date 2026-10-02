"""
gm.py - Graph-based skill matching and candidate-job recommendation.

Core library:
  * skill taxonomy, synthetic benchmark generator (known ground truth)
  * heterogeneous recruitment graph + weighted skill co-occurrence graph
  * four baselines (B1-B4) and graph methods G-B..G-E (G-A = bipartite assignment)
  * ranking metrics (P@K, NDCG@K, MAP@K)
  * rule-based explanation generator
"""
import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score

# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------
DOMAINS = {
    "AI/ML": ["Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Computer Vision",
              "Natural Language Processing", "Transformers", "Statistical Modelling",
              "Feature Engineering", "Reinforcement Learning", "Python", "Scikit-learn"],
    "Data Engineering": ["SQL", "Apache Spark", "Kafka", "Airflow", "ETL", "Data Warehousing",
                         "Hadoop", "Snowflake", "dbt", "Data Modelling", "Scala"],
    "Analytics": ["Tableau", "Power BI", "Excel", "Data Visualization", "Statistics", "A/B Testing",
                  "Business Analysis", "Data Analysis", "R", "Looker"],
    "Web Frontend": ["JavaScript", "TypeScript", "React", "Angular", "Vue", "HTML", "CSS", "Redux",
                     "Webpack", "Responsive Design", "UI Design"],
    "Backend": ["Java", "Spring Boot", "Node.js", "REST APIs", "Microservices", "Django",
                "PostgreSQL", "MongoDB", "Redis", "GraphQL", "System Design"],
    "DevOps/Cloud": ["Docker", "Kubernetes", "AWS", "Terraform", "Jenkins", "CI/CD", "Linux",
                     "Ansible", "Prometheus", "Grafana", "Azure"],
    "Mobile": ["Android", "Kotlin", "iOS", "Swift", "Flutter", "React Native", "Mobile UI",
               "Firebase", "Dart"],
    "Security": ["Network Security", "Penetration Testing", "Cryptography", "SIEM",
                 "Incident Response", "Vulnerability Assessment", "Firewalls", "Threat Modelling",
                 "OWASP", "Identity Management"],
}
GENERIC = ["Git", "Agile", "Communication", "Teamwork", "Problem Solving", "Time Management"]

# role -> (domain, core skills)   (23 roles)
ROLES = {
    "Machine Learning Engineer": ("AI/ML", ["Machine Learning", "Python", "Scikit-learn", "Feature Engineering",
                                             "Statistical Modelling", "TensorFlow", "Deep Learning"]),
    "NLP Engineer": ("AI/ML", ["Natural Language Processing", "Transformers", "Python", "PyTorch",
                               "Deep Learning", "Machine Learning"]),
    "Computer Vision Engineer": ("AI/ML", ["Computer Vision", "Deep Learning", "PyTorch", "Python",
                                           "TensorFlow", "Machine Learning"]),
    "Deep Learning Researcher": ("AI/ML", ["Deep Learning", "PyTorch", "TensorFlow", "Reinforcement Learning",
                                           "Python", "Transformers"]),
    "Data Engineer": ("Data Engineering", ["Python", "SQL", "Apache Spark", "Airflow", "ETL", "Data Warehousing"]),
    "Big Data Engineer": ("Data Engineering", ["Apache Spark", "Hadoop", "Kafka", "Scala", "SQL", "Data Modelling"]),
    "ETL Developer": ("Data Engineering", ["ETL", "SQL", "Snowflake", "dbt", "Data Warehousing", "Data Modelling"]),
    "BI Analyst": ("Analytics", ["SQL", "Tableau", "Power BI", "Data Visualization", "Excel", "Data Analysis"]),
    "Data Analyst": ("Analytics", ["SQL", "Python", "Data Analysis", "Statistics", "Excel", "Data Visualization"]),
    "Product Analyst": ("Analytics", ["SQL", "A/B Testing", "Statistics", "Looker", "Business Analysis", "Data Analysis"]),
    "Frontend Developer": ("Web Frontend", ["JavaScript", "HTML", "CSS", "TypeScript", "Responsive Design", "React"]),
    "React Developer": ("Web Frontend", ["React", "Redux", "JavaScript", "TypeScript", "Webpack", "HTML"]),
    "UI Engineer": ("Web Frontend", ["UI Design", "CSS", "HTML", "JavaScript", "Responsive Design", "Vue"]),
    "Java Backend Developer": ("Backend", ["Java", "Spring Boot", "REST APIs", "Microservices", "PostgreSQL", "System Design"]),
    "Node Backend Developer": ("Backend", ["Node.js", "JavaScript", "REST APIs", "MongoDB", "GraphQL", "Redis"]),
    "Python Backend Developer": ("Backend", ["Python", "Django", "REST APIs", "PostgreSQL", "Redis", "System Design"]),
    "DevOps Engineer": ("DevOps/Cloud", ["Docker", "Kubernetes", "Jenkins", "CI/CD", "Linux", "Ansible"]),
    "Site Reliability Engineer": ("DevOps/Cloud", ["Kubernetes", "Prometheus", "Grafana", "Linux", "Terraform", "AWS"]),
    "Cloud Engineer": ("DevOps/Cloud", ["AWS", "Azure", "Terraform", "Docker", "Kubernetes", "Linux"]),
    "Android Developer": ("Mobile", ["Android", "Kotlin", "Firebase", "Mobile UI", "REST APIs", "Java"]),
    "iOS Developer": ("Mobile", ["iOS", "Swift", "Mobile UI", "Firebase", "REST APIs", "React Native"]),
    "Security Analyst": ("Security", ["SIEM", "Incident Response", "Network Security", "Firewalls",
                                      "Vulnerability Assessment", "Threat Modelling"]),
    "Penetration Tester": ("Security", ["Penetration Testing", "OWASP", "Vulnerability Assessment",
                                        "Cryptography", "Network Security", "Linux"]),
}

ALIASES = {
    "Machine Learning": ["ML"], "Deep Learning": ["DL", "Neural Networks"], "TensorFlow": ["TF"],
    "PyTorch": ["Torch"], "Computer Vision": ["CV"], "Natural Language Processing": ["NLP"],
    "Statistical Modelling": ["Statistical Modeling"], "Reinforcement Learning": ["RL"],
    "Scikit-learn": ["sklearn"], "Feature Engineering": ["Feature Eng"], "Python": ["Python3", "Py"],
    "SQL": ["Structured Query Language"], "Apache Spark": ["Spark", "PySpark"], "Kafka": ["Apache Kafka"],
    "Airflow": ["Apache Airflow"], "ETL": ["Extract Transform Load"], "Data Warehousing": ["DWH"],
    "Hadoop": ["HDFS"], "Data Modelling": ["Data Modeling"], "Power BI": ["PowerBI"], "Excel": ["MS Excel"],
    "Data Visualization": ["Data Viz"], "A/B Testing": ["Split Testing"], "Data Analysis": ["Data Analytics"],
    "Business Analysis": ["BA"], "JavaScript": ["JS"], "TypeScript": ["TS"], "React": ["ReactJS"],
    "Angular": ["AngularJS"], "Vue": ["Vue.js"], "Responsive Design": ["Responsive Web Design"],
    "UI Design": ["User Interface Design"], "Node.js": ["NodeJS"], "REST APIs": ["RESTful APIs"],
    "Microservices": ["Microservice Architecture"], "PostgreSQL": ["Postgres"], "MongoDB": ["Mongo"],
    "Spring Boot": ["Spring"], "Kubernetes": ["K8s"], "Docker": ["Containerization"],
    "AWS": ["Amazon Web Services"], "Azure": ["Microsoft Azure"], "CI/CD": ["Continuous Integration"],
    "Terraform": ["IaC"], "Android": ["Android SDK"], "iOS": ["iOS Development"],
    "Mobile UI": ["Mobile UI Design"], "Network Security": ["NetSec"],
    "Penetration Testing": ["Pen Testing", "Pentesting"], "Incident Response": ["IR"],
    "Vulnerability Assessment": ["VA"], "Threat Modelling": ["Threat Modeling"],
    "Identity Management": ["IAM"], "OWASP": ["OWASP Top 10"], "Git": ["Version Control"],
    "Agile": ["Scrum"], "Communication": ["Communication Skills"], "Teamwork": ["Collaboration"],
    "Problem Solving": ["Problem-Solving"],
}

SKILLS = [s for d in DOMAINS.values() for s in d]          # 85 canonical skills
ALL = SKILLS + GENERIC                                      # 91 columns
IDX = {s: i for i, s in enumerate(ALL)}
K = len(ALL)
N_DOMAIN_SKILLS = len(SKILLS)
DOM_NAMES = list(DOMAINS)
DOM_OF = {s: d for d, l in DOMAINS.items() for s in l}
ROLE_NAMES = list(ROLES)

# every written form -> canonical index  (the normalisation dictionary)
NORM = {}
for _s in ALL:
    NORM[_s.lower()] = IDX[_s]
    for _a in ALIASES.get(_s, []):
        NORM[_a.lower()] = IDX[_s]
VOCAB = sorted(NORM)
VIDX = {w: i for i, w in enumerate(VOCAB)}

assert len(SKILLS) == 85 and len(ROLES) == 23
for _r, (_d, _core) in ROLES.items():
    assert _d in DOMAINS and all(c in IDX for c in _core), _r


# --------------------------------------------------------------------------
# Synthetic benchmark with known ground truth
# --------------------------------------------------------------------------
@dataclass
class Data:
    n: int
    m: int
    A: np.ndarray          # n x K  candidate visible skills (canonical)
    B: np.ndarray          # m x K  job listed skills (canonical)
    H: np.ndarray          # n x K  candidate hidden (true) skills
    Rq: np.ndarray         # m x K  job truly required skills
    Ar: np.ndarray         # n x V  raw surface forms
    Br: np.ndarray         # m x V
    rel: np.ndarray        # m x n  graded relevance 0/1/2
    cov: np.ndarray        # m x n  coverage of job requirements by hidden skills
    job_cat: np.ndarray    # m      domain index of job
    cand_role: list
    job_role: list
    cand_mentions: list    # per candidate: list[(skill_idx, surface)]
    job_mentions: list
    params: dict


def generate(n_cand=2000, n_jobs=300, seed=0, listing=0.55, alias=0.35, p_core=0.85, p_second=0.25):
    rng = np.random.default_rng(seed)

    def surface(s):
        al = ALIASES.get(s)
        if al and rng.random() < alias:
            return str(al[rng.integers(len(al))])
        return s

    H = np.zeros((n_cand, K)); A = np.zeros((n_cand, K)); Ar = np.zeros((n_cand, len(VOCAB)))
    cand_roles, cand_mentions = [], []
    for i in range(n_cand):
        r1 = ROLE_NAMES[rng.integers(len(ROLE_NAMES))]
        roles = [r1]
        if rng.random() < p_second:
            r2 = ROLE_NAMES[rng.integers(len(ROLE_NAMES))]
            if r2 != r1:
                roles.append(r2)
        hidden = set()
        for ri, rr in enumerate(roles):
            dom, core = ROLES[rr]
            pc = p_core if ri == 0 else 0.6
            hidden.update(s for s in core if rng.random() < pc)
            extras = [s for s in DOMAINS[dom] if s not in core]
            ne = min(len(extras), rng.poisson(0.6 if ri == 0 else 0.3))
            if ne:
                hidden.update(str(x) for x in rng.choice(extras, ne, replace=False))
        if not hidden:
            hidden.add(ROLES[r1][1][0])
        hidden = sorted(hidden)
        shown = [s for s in hidden if rng.random() < listing]
        if not shown:
            shown = [hidden[rng.integers(len(hidden))]]
        ng = rng.integers(0, 3)
        if ng:
            shown += [str(x) for x in rng.choice(GENERIC, ng, replace=False)]
        ment = [(IDX[s], surface(s)) for s in shown]
        for s in hidden:
            H[i, IDX[s]] = 1
        for k, w in ment:
            A[i, k] = 1
            Ar[i, VIDX[w.lower()]] = 1
        cand_roles.append(roles); cand_mentions.append(ment)

    Rq = np.zeros((n_jobs, K)); B = np.zeros((n_jobs, K)); Br = np.zeros((n_jobs, len(VOCAB)))
    job_roles, job_cat, job_mentions = [], [], []
    for j in range(n_jobs):
        r = ROLE_NAMES[rng.integers(len(ROLE_NAMES))]
        dom, core = ROLES[r]
        kk = rng.integers(4, min(6, len(core)) + 1)
        req = [str(x) for x in rng.choice(core, kk, replace=False)]
        if rng.random() < 0.3:
            extras = [s for s in DOMAINS[dom] if s not in core]
            if extras:
                req.append(str(extras[rng.integers(len(extras))]))
        listed = list(req)
        if rng.random() < 0.4:                       # boilerplate generic skill (not part of the truth)
            listed.append(GENERIC[rng.integers(len(GENERIC))])
        ment = [(IDX[s], surface(s)) for s in listed]
        for s in req:
            Rq[j, IDX[s]] = 1
        for k, w in ment:
            B[j, k] = 1
            Br[j, VIDX[w.lower()]] = 1
        job_roles.append(r); job_cat.append(DOM_NAMES.index(dom)); job_mentions.append(ment)

    cov = (Rq @ H.T) / Rq.sum(1, keepdims=True)
    rel = np.where(cov >= 0.8 - 1e-9, 2, np.where(cov >= 0.6 - 1e-9, 1, 0)).astype(int)
    return Data(n_cand, n_jobs, A, B, H, Rq, Ar, Br, rel, cov, np.array(job_cat),
                cand_roles, job_roles, cand_mentions, job_mentions,
                dict(seed=seed, listing=listing, alias=alias))


def dataset_stats(d):
    return {
        "candidates": d.n, "jobs": d.m, "roles": len(ROLES), "canonical_skills": len(SKILLS),
        "generic_skills": len(GENERIC), "surface_forms": len(VOCAB),
        "avg_listed_per_candidate": float(d.A.sum(1).mean()),
        "avg_hidden_per_candidate": float(d.H.sum(1).mean()),
        "avg_required_per_job": float(d.Rq.sum(1).mean()),
        "relevant_per_job": float((d.rel > 0).sum(1).mean()),
        "relevant_pair_share": float((d.rel > 0).mean()),
    }


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------
def build_ctx(d):
    """Build the heterogeneous graph, skill co-occurrence graph and everything derived from them."""
    t0 = time.perf_counter()
    n, m = d.n, d.m
    # heterogeneous graph G = (V, E, phi, psi)
    G = nx.Graph()
    for i in range(n): G.add_node(("c", i), type="candidate")
    for j in range(m): G.add_node(("j", j), type="job")
    for k in range(K): G.add_node(("s", k), type="skill")
    for t in range(len(DOM_NAMES)): G.add_node(("t", t), type="category")
    ci, ck = np.nonzero(d.A)
    G.add_edges_from((("c", int(i)), ("s", int(k)), {"type": "possesses"}) for i, k in zip(ci, ck))
    ji, jk = np.nonzero(d.B)
    G.add_edges_from((("j", int(j)), ("s", int(k)), {"type": "requires"}) for j, k in zip(ji, jk))
    G.add_edges_from((("j", j), ("t", int(d.job_cat[j])), {"type": "belongs_to"}) for j in range(m))
    dc = nx.degree_centrality(G)
    dc_skill = np.array([dc[("s", k)] for k in range(K)])

    # skill co-occurrence graph: w(a,b) = n_ab / sqrt(n_a n_b), kept if n_ab >= 3
    X = np.vstack([d.A, d.B])
    C = X.T @ X
    na = np.diag(C).copy()
    W = C / np.sqrt(np.outer(na, na) + 1e-12)
    W[C < 3] = 0
    np.fill_diagonal(W, 0)
    Gs = nx.Graph()
    Gs.add_nodes_from(range(K))
    a_, b_ = np.nonzero(np.triu(W, 1))
    Gs.add_weighted_edges_from((int(a), int(b), float(W[a, b])) for a, b in zip(a_, b_))

    comms = nx.community.louvain_communities(Gs, weight="weight", seed=0)
    comm = np.zeros(K, dtype=int)
    for ci_, members in enumerate(comms):
        for k in members: comm[k] = ci_
    modularity = nx.community.modularity(Gs, comms, weight="weight")

    # sparse transition matrix for personalised PageRank (node order: cands, jobs, skills, categories)
    N = n + m + K + len(DOM_NAMES)
    rows, cols = [], []
    for i, k in zip(ci, ck): rows.append(i); cols.append(n + m + k)
    for j, k in zip(ji, jk): rows.append(n + j); cols.append(n + m + k)
    for j in range(m): rows.append(n + j); cols.append(n + m + K + d.job_cat[j])
    rows, cols = np.array(rows), np.array(cols)
    Adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    Adj = (Adj + Adj.T).tocsr()
    deg = np.asarray(Adj.sum(0)).ravel()
    P = (Adj @ sp.diags(1.0 / np.maximum(deg, 1))).tocsr()      # column-stochastic

    return dict(G=G, Gs=Gs, W=W, dc_skill=dc_skill, comm=comm, communities=comms,
                modularity=modularity, P=P, N=N, t_build=time.perf_counter() - t0)


# --------------------------------------------------------------------------
# Scoring methods: each returns M (jobs x candidates)
# --------------------------------------------------------------------------
def _cos(X, Y):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return Xn @ Yn.T


def _jaccard(Bm, Am):
    inter = Bm @ Am.T
    union = Bm.sum(1)[:, None] + Am.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def b1_keyword(d, ctx):          return _jaccard(d.Br, d.Ar)

def b2_tfidf(d, ctx):
    raw_c = [[w.lower() for _, w in m] for m in d.cand_mentions]
    raw_j = [[w.lower() for _, w in m] for m in d.job_mentions]
    tv = TfidfVectorizer(analyzer=lambda x: x)
    X = tv.fit_transform(raw_c + raw_j)
    Xc, Xj = X[:d.n], X[d.n:]
    return (Xj @ Xc.T).toarray()                      # rows are L2-normalised -> cosine

def b3_keyword_norm(d, ctx):     return _jaccard(d.B, d.A)

def b4_lsa(d, ctx, dim=16):
    X = np.vstack([d.Ar, d.Br])
    Z = TruncatedSVD(dim, random_state=0).fit_transform(X)
    return _cos(Z[d.n:], Z[:d.n])

def g_b_skillgraph(d, ctx):
    Wi = ctx["W"] + np.eye(K)
    return _cos(d.B @ Wi, d.A @ Wi)

def g_c_centrality(d, ctx):
    w = np.log1p(1.0 / np.maximum(ctx["dc_skill"], 1e-9))
    return _cos(d.B * w, d.A * w)

def g_d_community(d, ctx):
    oh = np.eye(ctx["comm"].max() + 1)[ctx["comm"]]
    return _cos(d.B @ oh, d.A @ oh)

def g_e_ppr(d, ctx, alpha=0.85, iters=30):
    n, m, N, P = d.n, d.m, ctx["N"], ctx["P"]
    E = np.zeros((N, m)); E[n + np.arange(m), np.arange(m)] = 1.0
    R = E.copy()
    for _ in range(iters):
        R = alpha * (P @ R) + (1 - alpha) * E
    return R[:n].T

METHODS = {
    "B1 Exact keyword (Jaccard)": b1_keyword,
    "B2 TF-IDF cosine": b2_tfidf,
    "B3 Keyword + normalisation": b3_keyword_norm,
    "B4 LSA semantic vectors": b4_lsa,
    "G-B Skill-graph similarity": g_b_skillgraph,
    "G-C Centrality-weighted overlap": g_c_centrality,
    "G-D Community profile": g_d_community,
    "G-E Personalised PageRank": g_e_ppr,
}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _rank(M, rng, top):
    noise = rng.random(M.shape) * 1e-10                   # random tie-breaking
    return np.argsort(-(M + noise), axis=1)[:, :top]


def _ndcg_p(M, R, K_, rng):
    order = _rank(M, rng, K_)
    rel = np.take_along_axis(R, order, axis=1)
    disc = 1.0 / np.log2(np.arange(2, K_ + 2))
    dcg = ((2.0 ** rel - 1) * disc).sum(1)
    ideal = -np.sort(-R, axis=1)[:, :K_]
    idcg = ((2.0 ** ideal - 1) * disc).sum(1)
    ok = idcg > 0
    ndcg = (dcg[ok] / idcg[ok]).mean()
    prec = (rel[ok] > 0).mean()
    return prec, ndcg, order, ok


def evaluate(M, rel, seed=0):
    """M: jobs x candidates. Returns J2C (P@10, NDCG@10, MAP@100) and C2J (P@5, NDCG@5)."""
    rng = np.random.default_rng(seed)
    p10, n10, order, ok = _ndcg_p(M, rel, 10, rng)
    order100 = _rank(M, np.random.default_rng(seed), 100)
    r100 = (np.take_along_axis(rel, order100, axis=1) > 0).astype(float)
    cum = np.cumsum(r100, axis=1) / np.arange(1, 101)
    nrel = np.minimum((rel > 0).sum(1), 100)
    okm = nrel > 0
    ap = (cum * r100).sum(1)[okm] / nrel[okm]
    p5, n5, _, _ = _ndcg_p(M.T, rel.T, 5, np.random.default_rng(seed + 1))
    return {"J2C_P@10": p10, "J2C_NDCG@10": n10, "J2C_MAP@100": ap.mean(),
            "C2J_P@5": p5, "C2J_NDCG@5": n5}


def run_all(d, ctx, seed=0, methods=None):
    out = {}
    for name, fn in (methods or METHODS).items():
        t0 = time.perf_counter()
        M = fn(d, ctx)
        dt = time.perf_counter() - t0
        res = evaluate(M, d.rel, seed)
        res["time_s"] = dt
        out[name] = res
    return out


# --------------------------------------------------------------------------
# Weighted bipartite matching (Algorithm A)
# --------------------------------------------------------------------------
def assignment_experiment(d, M, pool, seed=0):
    """Fill every job from a limited candidate pool, one distinct candidate per job.
    Returns (greedy, hungarian) share of jobs that received a relevant candidate."""
    rng = np.random.default_rng(seed)
    S = M[:, pool]
    relp = d.rel[:, pool] > 0
    # greedy: random job order, best remaining candidate
    taken = np.zeros(len(pool), bool); hit = 0
    for j in rng.permutation(d.m):
        s = np.where(taken, -np.inf, S[j])
        c = int(np.argmax(s + rng.random(len(pool)) * 1e-12))
        taken[c] = True; hit += relp[j, c]
    greedy = hit / d.m
    r, c = linear_sum_assignment(-S)
    hung = relp[r, c].sum() / d.m
    return greedy, hung


# --------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------
def community_nmi(ctx):
    skills = list(range(N_DOMAIN_SKILLS))
    truth = [DOM_NAMES.index(DOM_OF[ALL[k]]) for k in skills]
    pred = [int(ctx["comm"][k]) for k in skills]
    return normalized_mutual_info_score(truth, pred)


def top_centrality(ctx, top=6):
    order = np.argsort(-ctx["dc_skill"])[:top]
    return [(ALL[k], float(ctx["dc_skill"][k])) for k in order]


def betweenness_top(ctx, top=6):
    Gd = nx.Graph()
    for a, b, w in ctx["Gs"].edges(data="weight"):
        Gd.add_edge(a, b, dist=1.0 / w)
    bc = nx.betweenness_centrality(Gd, weight="dist")
    order = sorted(bc, key=bc.get, reverse=True)[:top]
    return [(ALL[k], float(bc[k])) for k in order]


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------
def explain(d, ctx, j, i):
    """Rule-based natural-language explanation of why candidate i fits job j (uses skill graph)."""
    W = ctx["W"]
    req = [k for k in np.nonzero(d.B[j])[0] if k < N_DOMAIN_SKILLS]
    surf = {k: w for k, w in d.cand_mentions[i]}
    have = [k for k in req if d.A[i, k] > 0]
    miss = [k for k in req if d.A[i, k] == 0]
    cand_sk = [k for k in np.nonzero(d.A[i])[0] if k < N_DOMAIN_SKILLS]
    parts = []
    if have:
        desc = []
        for k in have:
            s = ALL[k]
            desc.append(s if surf[k].lower() == s.lower() else f"{s} (listed as '{surf[k]}')")
        joined = desc[0] if len(desc) == 1 else ", ".join(desc[:-1]) + " and " + desc[-1]
        parts.append(f"the candidate has {joined}, "
                     f"which match {len(have)} of the {len(req)} requirements after normalisation")
    links = []
    for k in miss:
        if cand_sk:
            best = max(cand_sk, key=lambda c: W[k, c])
            if W[k, best] > 0:
                links.append((ALL[best], ALL[k], float(W[k, best])))
    if links:
        by_src = {}
        for src, tgt, w in links:
            by_src.setdefault(src, []).append((tgt, w))
        for src, lst in by_src.items():
            tl = ", ".join(f"{t} (w = {w:.2f})" for t, w in sorted(lst, key=lambda x: -x[1]))
            parts.append(f"{src} is strongly linked in the skill graph to the remaining requirements {tl}")
    if not parts:
        return f"No strong graph evidence for J{j:04d} / C{i:05d}."
    return f"Recommended for J{j:04d} because " + ", and ".join(parts) + "."
