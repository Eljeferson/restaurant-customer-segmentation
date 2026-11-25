import json
import os
import re
import time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, Birch, HDBSCAN, KMeans, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score as ARI, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# ================= AJUSTAR =================
DATA = r"C:\Users\USER\Desktop\Articulos Restaurante\ARTICULO RESTAURANTE\Predicción de Preferencias Gastronómicas\sentiment_spanish_benchmark_50k.csv"
HIST = r"C:\Users\USER\Desktop\Articulos Restaurante\ARTICULO RESTAURANTE\Predicción de Preferencias Gastronómicas\reviews.csv"
C = dict(
    date="date",
    rest="service",
    user="id_nick",
    platform="platform",
    stars="score",
    score_norm="score_norm",
    sent="sent_robertuito",
    likes="likes",
    freq_log="freq_log",
    label="cluster_roberta"
)
SEED = 42
# ===========================================

print("Loading benchmark sample...")
d = pd.read_csv(DATA).reset_index(drop=True)

# Function to parse relative dates if needed
def parse_relative_date(val, ref_date=pd.Timestamp("2024-10-01")):
    if pd.isna(val): return pd.NaT
    s = str(val).strip().lower()
    m_mo = re.search(r"(\d+)\s+month", s)
    if m_mo: return ref_date - pd.DateOffset(months=int(m_mo.group(1)))
    m_yr = re.search(r"(\d+)\s+year", s)
    if m_yr: return ref_date - pd.DateOffset(years=int(m_yr.group(1)))
    return pd.to_datetime(val, errors="coerce")

# Fast mapping of dates
unique_dates = {v: parse_relative_date(v) for v in d[C["date"]].unique()}
d["date_"] = d[C["date"]].map(unique_dates)

d["score_norm"], d["sent"] = d[C["score_norm"]], d[C["sent"]]
d["sat"] = 0.5 * d["score_norm"] + 0.5 * ((d["sent"] + 1) / 2)          # Eq. (1)
d["likes_log"], d["freq_log"] = np.log1p(d[C["likes"]]), d[C["freq_log"]]
F5 = ["score_norm", "sent", "sat", "likes_log", "freq_log"]
F4 = ["score_norm", "sent", "sat", "freq_log"]                     # sin likes_log

# Ensure cluster_roberta labels match the official RoBERTuito K-means partition
if C["label"] not in d.columns:
    X_init = StandardScaler().fit_transform(d[F5])
    km_init = KMeans(3, n_init=10, max_iter=300, tol=1e-4, random_state=SEED).fit(X_init)
    label_map = {2: 0, 1: 1, 0: 2}  # 0: Social (3,148), 1: Passive (37,972), 2: Critical (8,880)
    d[C["label"]] = [label_map[x] for x in km_init.labels_]

ref = d[C["label"]].to_numpy()
rng = np.random.default_rng(SEED)
nan = float("nan")

def km(X, s=SEED): return KMeans(3, n_init=10, max_iter=300, tol=1e-4, random_state=s).fit(X)
def agree(a, b):
    ct = pd.crosstab(np.asarray(a), np.asarray(b)).to_numpy()
    r, c = linear_sum_assignment(-ct); return ct[r, c].sum() / ct.sum()
def sil(X, lab, n=10000):
    i = rng.choice(len(X), min(n, len(X)), replace=False)
    return silhouette_score(X[i], lab[i]) if len(np.unique(lab[i])) > 1 else nan
def shares(lab): return (100 * pd.Series(lab).value_counts(normalize=True).sort_index()).round(2).to_dict()

R = {}

print("1. Reference & Sanity check...")
# Referencia: re-ajuste del modelo publicado
X5 = StandardScaler().fit_transform(d[F5]); m5 = km(X5)
R["sanity_refit_vs_label"] = dict(ari=ARI(ref, m5.labels_), agree=agree(ref, m5.labels_))
print("Sanity refit vs label:", R["sanity_refit_vs_label"])

# 1) R2-2: K-means por plataforma y modelo armonizado sin likes_log
print("2. Platform-specific K-means & Harmonized 4F...")
R["platform"] = {}
for p, g in d.groupby(C["platform"]):
    f = [x for x in F5 if g[x].std() > 0]                          # TripAdvisor: likes_log constante
    Xp = StandardScaler().fit_transform(g[f]); mp = km(Xp)
    R["platform"][str(p)] = dict(n=len(g), features=f, shares=shares(mp.labels_),
        silhouette=sil(Xp, mp.labels_), ari_vs_global=ARI(ref[g.index], mp.labels_),
        agree_vs_global=agree(ref[g.index], mp.labels_),
        means=g.groupby(mp.labels_)[F5].mean().round(3).to_dict(orient="index"))

X4 = StandardScaler().fit_transform(d[F4]); m4 = km(X4)
R["harmonized_4f"] = dict(shares=shares(m4.labels_), silhouette=sil(X4, m4.labels_),
    ari_vs_global=ARI(ref, m4.labels_), agree_vs_global=agree(ref, m4.labels_),
    crosstab_global_x_4f=pd.crosstab(ref, m4.labels_).to_dict(),
    by_platform=pd.crosstab(d[C["platform"]], m4.labels_, normalize="index").round(4).to_dict(),
    means=d.groupby(m4.labels_)[F5].mean().round(3).to_dict(orient="index"))

# 2) R1-3: partición temporal y GroupKFold (scaler y K-means ajustados solo en train)
print("3. Temporal & Grouped holdouts...")
def eval_split(tr, te, feats, fcol=None):
    D = d[feats].copy()
    if fcol is not None and "freq_log" in feats: D["freq_log"] = fcol
    sc = StandardScaler().fit(D.iloc[tr]); Xtr, Xte = sc.transform(D.iloc[tr]), sc.transform(D.iloc[te])
    m = km(Xtr); ytr, yte = m.labels_, m.predict(Xte)
    rf = RandomForestClassifier(200, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    mt = km(Xte)
    return dict(n_train=len(tr), n_test=len(te), rf_fidelity_test=(rf.predict(Xte) == yte).mean(),
        ari_train_model_vs_test_refit=ARI(yte, mt.labels_), agree_train_model_vs_test_refit=agree(yte, mt.labels_),
        ari_vs_global=ARI(ref[te], yte), silhouette_test=sil(Xte, yte), shares_test=shares(yte))

ok = d["date_"].notna()
if HIST:   # frecuencia previa usando todo el historial del corpus
    print("Reading full history from HIST...")
    h = pd.read_csv(HIST, usecols=[C["user"], C["date"]])
    unique_dates_hist = {v: parse_relative_date(v) for v in h[C["date"]].unique()}
    h["t"] = h[C["date"]].map(unique_dates_hist)
    hd = h.dropna().groupby(C["user"])["t"].apply(lambda s: np.sort(s.values.astype("datetime64[ns]")))
    empty = np.array([], dtype="datetime64[ns]")
    past = np.array([np.searchsorted(hd.get(u, empty), t) if ok_ else 0
                     for u, t, ok_ in zip(d[C["user"]], d["date_"].values, ok)])
else:      # frecuencia previa dentro de la muestra de 50k
    past = d.sort_values("date_").groupby(C["user"]).cumcount().reindex(d.index).to_numpy()

fp = pd.Series(np.log1p(past + 1), index=d.index)

idx = d.index[ok].to_numpy(); idx = idx[np.argsort(d.loc[idx, "date_"].values, kind="stable")]
cut = int(0.8 * len(idx)); tr, te = idx[:cut], idx[cut:]
R["temporal"] = dict(cutoff=str(d.loc[idx[cut], "date_"].date()), n_dated=int(ok.sum()),
    five_features_past_freq=eval_split(tr, te, F5, fp),
    four_features_no_likes_past_freq=eval_split(tr, te, F4, fp),
    five_features_full_freq=eval_split(tr, te, F5))

print("Running GroupKFold by restaurant and user...")
def grouped(col):
    out = [eval_split(a, b, F5) for a, b in GroupKFold(5).split(d, groups=d[col])]
    ks = ["rf_fidelity_test", "ari_train_model_vs_test_refit", "agree_train_model_vs_test_refit", "ari_vs_global", "silhouette_test"]
    return {k: dict(mean=np.mean([o[k] for o in out]), sd=np.std([o[k] for o in out], ddof=1)) for k in ks}

R["grouped_restaurant"], R["grouped_user"] = grouped(C["rest"]), grouped(C["user"])

tr, te = train_test_split(np.arange(len(d)), test_size=0.2, random_state=SEED, stratify=ref)
R["random_split_reference"] = eval_split(tr, te, F5)

# 3) R2-6 y R2-5: margen Δd vs. Random Forest, abstención y regla simple
print("4. Voronoi margin Δd vs Random Forest...")
Dm = np.sort(m5.transform(X5), axis=1); dd = (Dm[:, 1] - Dm[:, 0])[te]
rf = RandomForestClassifier(200, random_state=SEED, n_jobs=-1).fit(X5[tr], m5.labels_[tr])
P = rf.predict_proba(X5[te]); pmax = P.max(1); yhat = rf.classes_[P.argmax(1)]; yc = m5.labels_[te]
low, lowp = dd < 0.20, pmax < 0.90
R["margin_vs_rf"] = dict(n_test=len(te), share_dd_lt_020=low.mean(),
    share_lowmargin_with_p_gt_095=(pmax[low] > 0.95).mean() if low.any() else nan,
    spearman_dd_pmax=spearmanr(dd, pmax)[0],
    rf_disagreement_low_margin=(yhat[low] != yc[low]).mean() if low.any() else nan,
    rf_disagreement_high_margin=(yhat[~low] != yc[~low]).mean(),
    share_p_lt_090=lowp.mean(), jaccard_low_margin_vs_low_p=(low & lowp).sum() / max((low | lowp).sum(), 1),
    abstention={str(t): dict(coverage=(dd >= t).mean(),
        rf_centroid_agreement_retained=(yhat[dd >= t] == yc[dd >= t]).mean()) for t in (0.1, 0.2, 0.5)})

stars, likes = d[C["stars"]].to_numpy(), d[C["likes"]].to_numpy()
hi_neg, lo_pos = (stars >= 4) & (d["sent"] < 0), (stars <= 2) & (d["sent"] > 0.5)
for c2 in (2, 3):   # regla: estrellas <= c2 → Crítico; si no, likes ≥ 1 → Socially Endorsed; si no, Pasivo
    r = np.where(stars <= c2, 2, np.where(likes >= 1, 0, 1)); dis = r != ref
    R[f"rule_stars_le_{c2}"] = dict(accuracy_vs_kmeans=(r == ref).mean(), ari=ARI(ref, r),
        crosstab=pd.crosstab(ref, r).to_dict(),
        share_disagreements_with_text_rating_conflict=(dis & (hi_neg | lo_pos)).sum() / max(dis.sum(), 1))

R["text_rating_conflicts"] = dict(high_star_negative_text=int(hi_neg.sum()), high_star_negative_profiles=shares(ref[hi_neg]),
    low_star_positive_text=int(lo_pos.sum()), low_star_positive_profiles=shares(ref[lo_pos]))

# 4) R1-13: Tabla 13 con 5 corridas repetidas en n = 10,000
print("5. Table 13 6-algorithm repeated benchmark...")
Xs = X5[rng.choice(len(d), 10000, replace=False)]
algos = {
    "KMeans": lambda s: KMeans(3, n_init=10, max_iter=300, tol=1e-4, random_state=s),
    "MiniBatchKMeans": lambda s: MiniBatchKMeans(3, n_init=10, max_iter=300, tol=1e-4, batch_size=1024, random_state=s),
    "Agglomerative_Ward": lambda s: AgglomerativeClustering(3, linkage="ward"),
    "GMM": lambda s: GaussianMixture(3, covariance_type="full", n_init=1, max_iter=100, tol=1e-3, random_state=s),
    "BIRCH": lambda s: Birch(n_clusters=3, branching_factor=50, threshold=0.5),
    "HDBSCAN": lambda s: HDBSCAN(min_cluster_size=200)
}
R["table13_repeated"] = {}
for name, mk in algos.items():
    print(f"  Benchmarking {name}...")
    ts, ss, ns = [], [], []
    for s in range(5):
        t0 = time.perf_counter(); lab = mk(s).fit_predict(Xs); ts.append(time.perf_counter() - t0)
        k = lab >= 0; ns.append(1 - k.mean())
        ss.append(silhouette_score(Xs[k], lab[k]) if len(np.unique(lab[k])) > 1 else nan)
    R["table13_repeated"][name] = dict(time_s_mean=np.mean(ts), time_s_sd=np.std(ts, ddof=1),
        silhouette_mean=np.nanmean(ss), silhouette_sd=np.nanstd(ss, ddof=1), noise_share=np.mean(ns))

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if np.isnan(o) else round(float(o), 4)
    return o

out_path = r"C:\Users\USER\Desktop\Articulos Restaurante\blockC_results.json"
json.dump(clean(R), open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nFINISHED! Results saved to blockC_results.json")
print(json.dumps(clean(R), indent=1, ensure_ascii=False))
