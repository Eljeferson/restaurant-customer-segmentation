import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8')

# Configure IJIES publication style: Times New Roman, 9-10 pt, 600 DPI
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.dpi": 600,
    "savefig.dpi": 600,
})

p_dir = os.path.dirname(os.path.abspath(__file__))
f_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
os.makedirs(f_dir, exist_ok=True)

df_50k = pd.read_csv(os.path.join(p_dir, "sentiment_spanish_benchmark_50k.csv"), low_memory=False)
df_50k["sat_robertuito"] = 0.5 * df_50k["score_norm"] + 0.5 * (df_50k["sent_robertuito"] + 1) / 2

FEATS = ["score_norm", "sent_robertuito", "sat_robertuito", "likes_log", "freq_log"]
SHORT_LABELS = ["score_norm", "sentiment", "satisfaction", "likes_log", "freq_log"]

scaler = StandardScaler()
X_rob = scaler.fit_transform(df_50k[FEATS].values)

km3 = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
raw_labs = km3.fit_predict(X_rob)
df_50k["raw_c"] = raw_labs
means_raw = df_50k.groupby("raw_c")[FEATS].mean()
c_adv = means_raw["likes_log"].idxmax()
c_crit = means_raw["score_norm"].idxmin()
c_pass = [c for c in [0, 1, 2] if c not in [c_adv, c_crit]][0]
remap = {c_adv: 0, c_pass: 1, c_crit: 2}
df_50k["cluster_rob"] = df_50k["raw_c"].map(remap)
y_rob = df_50k["cluster_rob"].values

def save_both(fig, names_pdir, names_fdir):
    for n in names_pdir:
        fig.savefig(os.path.join(p_dir, n), dpi=600, bbox_inches="tight")
    for n in names_fdir:
        fig.savefig(os.path.join(f_dir, n), dpi=600, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------
# FIG 1: Data Sources (Raw vs Clean by Platform)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.35, 2.3))
platforms = ["Google Places", "TripAdvisor"]
raw_counts = [1194114, 64321]
clean_counts = [690745, 64321]
x = np.arange(len(platforms))
w = 0.34
b1 = ax.bar(x - w/2, [c/1000 for c in raw_counts], w, label="Raw Corpus (N=1,258k)", color="#6baed6", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w/2, [c/1000 for c in clean_counts], w, label="Clean Corpus (N=755k)", color="#08519c", edgecolor="black", linewidth=0.5)
ax.set_ylabel("Number of Reviews (×1,000)")
ax.set_title("Review Distribution by Platform", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(platforms)
ax.set_ylim(0, 1380)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.annotate(f"{h:,.1f}k", (bar.get_x() + bar.get_width()/2, h),
                ha="center", va="bottom", fontsize=7.5, xytext=(0, 2), textcoords="offset points")
ax.legend(loc="upper right", frameon=True)
ax.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
save_both(fig, ["fig_fuentes.png", "fig_fuentes_hr.png"], ["Fig_01_Data_Sources.png"])
print("Saved Fig 1")

# ---------------------------------------------------------
# FIG 2: Feature Distributions (Histograms 3x2 compact)
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(3.45, 4.1))
axes = axes.flatten()
titles_f2 = [
    "(a) score_norm", "(b) sentiment (RoBERTuito)",
    "(c) satisfaction_score", "(d) likes_log",
    "(e) freq_log"
]
colors_f2 = ["#2b5c8f", "#2e8b57", "#d95f02", "#7570b3", "#e7298a"]
for i, (col, t_lbl, c_col) in enumerate(zip(FEATS, titles_f2, colors_f2)):
    ax = axes[i]
    vals = df_50k[col].values
    ax.hist(vals, bins=35, color=c_col, edgecolor="white", linewidth=0.3, alpha=0.88)
    m_v = vals.mean()
    ax.axvline(m_v, color="red", linestyle="--", linewidth=1.0, label=f"μ={m_v:.2f}")
    ax.set_title(t_lbl, fontweight="bold", pad=2, fontsize=8.5)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else f"{int(v)}"))
    ax.legend(loc="upper left" if i in [0, 1, 2] else "upper right", fontsize=7.0, handlelength=1.2)
    ax.grid(alpha=0.25, linestyle=":")
axes[5].axis("off")
plt.tight_layout(pad=0.5)
save_both(fig, ["fig_histogramas.png", "fig_histogramas_hr.png"], ["Fig_06_Outlier_Detection_Boxplots.png"])
print("Saved Fig 2")

# ---------------------------------------------------------
# FIG 3: Pearson Correlation Matrix (annot_kws={"size": 9})
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.35, 2.7))
corr_df = df_50k[FEATS].copy()
corr_df.columns = SHORT_LABELS
corr_mat = corr_df.corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
sns.heatmap(
    corr_mat, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, vmin=-1, vmax=1,
    linewidths=0.6, cbar_kws={"shrink": 0.82},
    annot_kws={"size": 8.5, "weight": "bold"}, ax=ax
)
ax.set_title("Pearson Correlation Matrix (Spanish NLP)", fontweight="bold", pad=6)
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
save_both(fig, ["fig_correlacion_matrix.png", "fig_correlacion_matrix_hr.png"], ["Fig_08_Correlation_Matrix.png"])
print("Saved Fig 3")

# ---------------------------------------------------------
# FIG 4: Optimal k Selection (k = 2..8, 2x2 grid)
# ---------------------------------------------------------
k_vals = [2, 3, 4, 5, 6, 7, 8]
inertias = [153.1, 112.3, 84.8, 70.5, 62.8, 56.4, 50.8]  # x10^3
sils = [0.4808, 0.5146, 0.3596, 0.3673, 0.3654, 0.3708, 0.3826]
dbs = [0.9260, 0.8114, 0.9537, 0.9401, 0.9762, 0.9566, 0.9339]
chs = [31.64, 30.65, 32.44, 31.82, 29.80, 28.62, 28.03]  # x10^3

fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.1))
axes[0, 0].plot(k_vals, inertias, "o-", color="#1f77b4", linewidth=1.4, markersize=4)
axes[0, 0].axvline(3, color="red", linestyle="--", linewidth=0.9)
axes[0, 0].set_title("(a) Inertia (×10³)", fontweight="bold", fontsize=8.5)
axes[0, 0].set_xlabel("k", fontsize=8)
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(k_vals, sils, "s-", color="#ff7f0e", linewidth=1.4, markersize=4)
axes[0, 1].axvline(3, color="red", linestyle="--", linewidth=0.9, label="Opt k=3")
axes[0, 1].set_title("(b) Silhouette Score", fontweight="bold", fontsize=8.5)
axes[0, 1].set_xlabel("k", fontsize=8)
axes[0, 1].legend(fontsize=7, loc="upper right")
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(k_vals, dbs, "^-", color="#2ca02c", linewidth=1.4, markersize=4)
axes[1, 0].axvline(3, color="red", linestyle="--", linewidth=0.9, label="Min k=3")
axes[1, 0].set_title("(c) Davies-Bouldin", fontweight="bold", fontsize=8.5)
axes[1, 0].set_xlabel("k", fontsize=8)
axes[1, 0].legend(fontsize=7, loc="lower right")
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(k_vals, chs, "D-", color="#9467bd", linewidth=1.4, markersize=4)
axes[1, 1].axvline(3, color="red", linestyle="--", linewidth=0.9)
axes[1, 1].set_title("(d) Calinski-Harabasz (×10³)", fontweight="bold", fontsize=8.5)
axes[1, 1].set_xlabel("k", fontsize=8)
axes[1, 1].grid(alpha=0.3)

for ax in axes.flatten():
    ax.set_xticks(k_vals)
    ax.tick_params(labelsize=7.5)

plt.tight_layout(pad=0.5)
save_both(fig, ["fig_seleccion_k.png"], ["Fig_09_Optimal_k_Selection.png"])
print("Saved Fig 4")

# ---------------------------------------------------------
# FIG 5: PCA Cluster Visualization (2D projection)
# ---------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_rob)
var_exp = pca.explained_variance_ratio_ * 100
tot_var = var_exp.sum()
print(f"PCA variance explained (RoBERTuito): PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%, Total={tot_var:.1f}%")

vis_idx = np.random.RandomState(42).choice(len(X_pca), 6000, replace=False)
fig, ax = plt.subplots(figsize=(3.35, 2.6))
c_colors = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728"}
c_names = {
    0: "C0: Socially Endorsed (6.3%)",
    1: "C1: Passive (75.9%)",
    2: "C2: Critical (17.8%)"
}
for c in [1, 2, 0]:  # plot largest first so smaller clusters stay visible
    m = y_rob[vis_idx] == c
    ax.scatter(X_pca[vis_idx][m, 0], X_pca[vis_idx][m, 1],
               s=5, alpha=0.45, color=c_colors[c], label=c_names[c], rasterized=True)

# Plot centroids in canonical order
inv_remap = {v: k for k, v in remap.items()}
cent_ordered = np.array([km3.cluster_centers_[inv_remap[c]] for c in [0, 1, 2]])
cent_pca = pca.transform(cent_ordered)
ax.scatter(cent_pca[:, 0], cent_pca[:, 1], c="black", s=35, marker="X", zorder=6, label="Centroids")

ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)")
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)")
ax.set_title(f"Review-Level Profile Projection (Total: {tot_var:.1f}%)", fontweight="bold")
ax.legend(loc="upper left", markerscale=1.1, frameon=True, fontsize=7.5)
ax.grid(alpha=0.25, linestyle="--")
plt.tight_layout()
save_both(fig, ["fig_pca_clusters.png", "fig_pca_clusters_hr.png"], ["Fig_12_PCA_Cluster_Visualization.png"])
print("Saved Fig 5")

# ---------------------------------------------------------
# FIG 6: Boxplots of Features by Cluster (3x2 grid)
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(3.45, 4.1))
axes = axes.flatten()
for i, (col, t_lbl) in enumerate(zip(FEATS, titles_f2)):
    ax = axes[i]
    data_groups = [df_50k[df_50k["cluster_rob"] == c][col].values for c in [0, 1, 2]]
    bp = ax.boxplot(data_groups, tick_labels=["C0", "C1", "C2"], patch_artist=True,
                    widths=0.45, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, c in zip(bp["boxes"], [0, 1, 2]):
        patch.set_facecolor(c_colors[c])
        patch.set_alpha(0.75)
    ax.set_title(t_lbl, fontweight="bold", fontsize=8.5, pad=2)
    ax.tick_params(labelsize=8.0)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
axes[5].axis("off")
plt.tight_layout(pad=0.5)
save_both(fig, ["fig_boxplots_cluster.png", "fig_boxplots_cluster_hr.png"], ["Fig_14_Feature_Distribution_Clusters.png"])
print("Saved Fig 6")

# ---------------------------------------------------------
# FIG 7: Segment Distribution by Platform
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.35, 2.35))
plat_ct = df_50k.groupby(["platform", "cluster_rob"]).size().unstack(fill_value=0)
plat_pct = plat_ct.div(plat_ct.sum(axis=1), axis=0) * 100
plat_pct.index = ["Google Places", "TripAdvisor"]

bottom = np.zeros(2)
for c in [0, 1, 2]:
    vals = plat_pct[c].values
    bars = ax.bar(plat_pct.index, vals, width=0.45, bottom=bottom,
                  color=c_colors[c], edgecolor="white", linewidth=0.6, label=c_names[c])
    for j, v in enumerate(vals):
        if v > 5.0:
            ax.text(j, bottom[j] + v/2, f"{v:.1f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=8.0)
    bottom += vals

ax.set_ylabel("Profile Share (%)")
ax.set_ylim(0, 108)
ax.set_title("Review-Level Profiles Across Platforms", fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=7.5)
ax.grid(axis="y", alpha=0.25, linestyle="--")
plt.tight_layout()
save_both(fig, ["fig_clusters_plataforma.png", "fig_clusters_plataforma_hr.png"], ["Fig_15_Segments_by_Platform.png"])
print("Saved Fig 7")

# ---------------------------------------------------------
# FIG 8: Confusion Matrix & Feature Importance (Vertical Stack)
# ---------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(X_rob, y_rob, test_size=0.20, random_state=42, stratify=y_rob)
rf_rob = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_rob.fit(X_tr, y_tr)
y_pr = rf_rob.predict(X_te)

fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.2), gridspec_kw={"height_ratios": [1.15, 1.0]})
cm = confusion_matrix(y_te, y_pr)
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[0],
            xticklabels=["C0", "C1", "C2"], yticklabels=["C0", "C1", "C2"],
            annot_kws={"size": 8.5, "weight": "bold"}, linewidths=0.5, linecolor="gray")
axes[0].set_title("(a) Confusion Matrix (Hold-out Test, n=10,000)", fontweight="bold", fontsize=9.0)
axes[0].set_xlabel("Predicted Segment")
axes[0].set_ylabel("K-means Segment")

imp_s = pd.Series(rf_rob.feature_importances_, index=SHORT_LABELS).sort_values(ascending=True)
bars = axes[1].barh(imp_s.index, imp_s.values, color="#2b5c8f", edgecolor="black", height=0.55, linewidth=0.5)
axes[1].set_title("(b) Random Forest Gini Feature Importance", fontweight="bold", fontsize=9.0)
axes[1].set_xlabel("Relative Importance")
axes[1].set_xlim(0, 0.50)
for b in bars:
    w_val = b.get_width()
    axes[1].annotate(f"{w_val:.3f}", (w_val, b.get_y() + b.get_height()/2),
                     ha="left", va="center", fontsize=8.0, xytext=(3, 0), textcoords="offset points")
axes[1].grid(axis="x", alpha=0.3, linestyle="--")

plt.tight_layout(pad=0.6)
save_both(fig, ["fig_clasificador.png"], ["Fig_18_Classifier_Confusion_Matrix.png"])
print("Saved Fig 8")

# ---------------------------------------------------------
# FIG 9: Marginal Feature Contribution (RoBERTuito vs TextBlob)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.55))
stages = ["Rating\nOnly", "+ Engagement\n(Likes, Freq)", "+ NLP\nSentiment", "Complete\n(5 Features)"]
acc_rob = [86.72, 92.74, 100.00, 100.00]
acc_tb = [92.89, 99.55, 99.97, 99.98]

x = np.arange(len(stages))
w = 0.36
b_r = ax.bar(x - w/2, acc_rob, w, label="RoBERTuito Pipeline", color="#1b9e77", edgecolor="black", linewidth=0.5)
b_t = ax.bar(x + w/2, acc_tb, w, label="TextBlob Baseline", color="#7570b3", edgecolor="black", linewidth=0.5)

ax.set_ylabel("Partition Replication Accuracy (%)")
ax.set_title("Marginal Feature Contribution to Assignment", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=7.8)
ax.set_ylim(80, 105.5)
for bar in b_r:
    h = bar.get_height()
    lbl = "100%" if h >= 99.99 else f"{h:.1f}%"
    ax.annotate(lbl, (bar.get_x() + bar.get_width()/2, h),
                ha="center", va="bottom", fontsize=6.8, xytext=(-1, 2), textcoords="offset points")
for bar in b_t:
    h = bar.get_height()
    lbl = f"{h:.1f}%"
    ax.annotate(lbl, (bar.get_x() + bar.get_width()/2, h),
                ha="center", va="bottom", fontsize=6.8, xytext=(1, 2), textcoords="offset points")
ax.legend(loc="lower right", frameon=True, fontsize=7.5)
ax.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
save_both(fig, ["feature_ablation.png"], ["Fig_19_Feature_Ablation_Study.png"])
print("Saved Fig 9")

# ---------------------------------------------------------
# FIG 10: NLP Model Comparison & Real-Time Segment Assignment
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.85))

# Panel (a): Sentiment Model Benchmark
models_lbl = ["TextBlob\n(Lexicon)", "RoBERTuito\n(Spanish)", "XLM-RoBERTa\n(Multilingual)"]
f1_vals = [0.138, 0.795, 0.813]
r_vals = [0.102, 0.621, 0.655]
x_m = np.arange(len(models_lbl))
w_m = 0.32
b_f1 = axes[0].bar(x_m - w_m/2, f1_vals, w_m, label="Macro F1 vs. Stars", color="#2b5c8f", edgecolor="black", linewidth=0.5)
b_pr = axes[0].bar(x_m + w_m/2, r_vals, w_m, label="Pearson r with Rating", color="#e6550d", edgecolor="black", linewidth=0.5)
axes[0].set_ylabel("Metric Value")
axes[0].set_title("(a) Spanish Sentiment Model Benchmark (n=50,000)", fontweight="bold", fontsize=8.8)
axes[0].set_xticks(x_m)
axes[0].set_xticklabels(models_lbl, fontsize=8.0)
axes[0].set_ylim(0, 1.22)
for b in list(b_f1) + list(b_pr):
    h = b.get_height()
    axes[0].annotate(f"{h:.2f}", (b.get_x() + b.get_width()/2, h),
                     ha="center", va="bottom", fontsize=7.2, xytext=(0, 1.5), textcoords="offset points")
axes[0].legend(loc="upper left", ncol=2, fontsize=7.0, frameon=True)
axes[0].grid(axis="y", alpha=0.3, linestyle="--")

# Panel (b): Real-time inference probabilities on 3 Spanish test cases
cases_lbl = ["Case 1: Positive\n(Score=5, Likes=12)", "Case 2: Critical\n(Score=2.5, Likes=0)", "Case 3: Passive\n(Score=4.5, Likes=0)"]
p_c0 = [0.996, 0.000, 0.002]
p_c1 = [0.004, 0.001, 0.998]
p_c2 = [0.000, 0.999, 0.000]
y_pos = np.arange(len(cases_lbl))
axes[1].barh(y_pos, p_c0, color=c_colors[0], edgecolor="white", height=0.5, label="C0: Socially Endorsed")
axes[1].barh(y_pos, p_c1, left=p_c0, color=c_colors[1], edgecolor="white", height=0.5, label="C1: Passive")
axes[1].barh(y_pos, p_c2, left=np.array(p_c0)+np.array(p_c1), color=c_colors[2], edgecolor="white", height=0.5, label="C2: Critical")
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(cases_lbl, fontsize=7.8)
axes[1].set_xlabel("Random Forest surrogate score (uncalibrated)")
axes[1].set_xlim(0, 1.0)
axes[1].set_title("(b) Surrogate partition scores (illustrative reviews)", fontweight="bold", fontsize=8.8)
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3, fontsize=7.0, frameon=True)
axes[1].grid(axis="x", alpha=0.3, linestyle="--")

plt.tight_layout(pad=0.6)
save_both(fig, ["fig_correlacion_nlp_rating.png"], ["Fig_20_Prediction_System.png", "Fig_02_Feature_Engineering_NLP.png"])
print("Saved Fig 10")
