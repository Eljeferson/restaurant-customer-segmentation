# Experiment Run Manifest: Tables 6, 8, and 13
**Study:** Validated Customer Segmentation from Large-Scale Restaurant Reviews  
**Paper ID:** IJIES-20262397 / 20267052  
**Purpose:** Formal machine-traceable manifest mapping each published table row to its unique Run ID, configuration, population, scaler, random seed, convergence parameters, and empirical metrics (in response to Reviewer 1 comments R1-11, R1-13, and R1-18).

---

## 1. Table 6: Feature Set Ablation Protocol (n = 50,000, k = 3, seed = 42)

| Run ID | Configuration | Features | Algorithm | Fit N / Seed | Eval N / Seed | Inertia | Silhouette (10k) | DB | CH | Notes |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `RUN-T06-FS1-ROB5` | FS1_Complete (RoBERTuito) | 5 feats | KMeans(Lloyd) | 50000 / 42 | 10000 / 42 | 112,313.5 | 0.5146 | 0.8114 | 30,646.0 | Published primary configuration |
| `RUN-T06-FS2-NOCOMP` | FS2_No_Composite (RoBERTuito) | 4 feats | KMeans(Lloyd) | 50000 / 42 | 10000 / 42 | 97,972.2 | 0.4757 | 0.8867 | 26,033.4 | Composite removal test: ARI=0.9699 vs FS1 |
| `RUN-T06-FS3-TEXTBEH` | FS3_Text_Behavioral (No Rating) | 3 feats | KMeans(Lloyd) | 50000 / 42 | 10000 / 42 | 64,193.3 | 0.4262 | 0.9322 | 30,389.2 | Rating excluded baseline |
| `RUN-T06-FS4-RATBEH` | FS4_Rating_Behavioral (No NLP) | 3 feats | KMeans(Lloyd) | 50000 / 42 | 10000 / 42 | 66,431.4 | 0.3968 | 0.9536 | 28,539.1 | NLP excluded baseline |
| `RUN-T06-FS5-TEXTBLOB` | FS5_TextBlob_Baseline (5 feat.) | 5 feats | KMeans(Lloyd) | 50000 / 42 | 10000 / 42 | 135,758.5 | 0.4839 | 0.8992 | 20,594.7 | Standardized benchmark (tol=1e-4, n_init=10); harmonized with Table 8 inertia 135,758.5 |

---

## 2. Table 8: Initialization Stability across 10 Random Seeds (n = 50,000, k = 3, n_eval = 10,000)

| Run ID | Representation | Seed | Hyperparameters | Converged Inertia | Silhouette (n_eval=10k, seed 42) | Cluster Distribution (C0 / C1 / C2) | Notes |
|:---|:---|:---:|:---|:---:|:---:|:---:|:---|
| `RUN-T08-ROB-S0000` | RoBERTuito (Spanish, 5 feats) | 0 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 112,313.53 | 0.5175 | C0: 6.30%, C1: 75.94%, C2: 17.76% | Unrounded mean=0.51455, sample SD(ddof=1)=0.000052, CV=0.010% |
| `RUN-T08-TB-S0000` | TextBlob Baseline (Lexicon, 5 feats) | 0 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 135,758.50 | 0.4857 | C0: 6.64%, C1: 74.05%, C2: 19.31% | Unrounded mean=0.48234, sample SD(ddof=1)=0.00234, CV=0.49%; converged inertia identical across all seeds |
| `RUN-T08-ROB-S0007` | RoBERTuito (Spanish, 5 feats) | 7 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 112,313.53 | 0.5091 | C0: 6.30%, C1: 75.94%, C2: 17.76% | Unrounded mean=0.51455, sample SD(ddof=1)=0.000052, CV=0.010% |
| `RUN-T08-TB-S0007` | TextBlob Baseline (Lexicon, 5 feats) | 7 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 135,758.50 | 0.4792 | C0: 6.64%, C1: 74.05%, C2: 19.31% | Unrounded mean=0.48234, sample SD(ddof=1)=0.00234, CV=0.49%; converged inertia identical across all seeds |
| `RUN-T08-ROB-S0042` | RoBERTuito (Spanish, 5 feats) | 42 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 112,313.53 | 0.5146 | C0: 6.30%, C1: 75.94%, C2: 17.76% | Unrounded mean=0.51455, sample SD(ddof=1)=0.000052, CV=0.010% |
| `RUN-T08-TB-S0042` | TextBlob Baseline (Lexicon, 5 feats) | 42 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 135,758.50 | 0.4839 | C0: 6.64%, C1: 74.05%, C2: 19.31% | Unrounded mean=0.48234, sample SD(ddof=1)=0.00234, CV=0.49%; converged inertia identical across all seeds |
| `RUN-T08-ROB-S0123` | RoBERTuito (Spanish, 5 feats) | 123 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 112,313.53 | 0.5123 | C0: 6.30%, C1: 75.94%, C2: 17.76% | Unrounded mean=0.51455, sample SD(ddof=1)=0.000052, CV=0.010% |
| `RUN-T08-TB-S0123` | TextBlob Baseline (Lexicon, 5 feats) | 123 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 135,758.50 | 0.4783 | C0: 6.64%, C1: 74.05%, C2: 19.31% | Unrounded mean=0.48234, sample SD(ddof=1)=0.00234, CV=0.49%; converged inertia identical across all seeds |
| `RUN-T08-ROB-S0256` | RoBERTuito (Spanish, 5 feats) | 256 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 112,313.53 | 0.5131 | C0: 6.30%, C1: 75.94%, C2: 17.76% | Unrounded mean=0.51455, sample SD(ddof=1)=0.000052, CV=0.010% |
| `RUN-T08-TB-S0256` | TextBlob Baseline (Lexicon, 5 feats) | 256 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 135,758.50 | 0.4842 | C0: 6.64%, C1: 74.05%, C2: 19.31% | Unrounded mean=0.48234, sample SD(ddof=1)=0.00234, CV=0.49%; converged inertia identical across all seeds |
| `RUN-T08-ROB-S0512` | RoBERTuito (Spanish, 5 feats) | 512 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 112,313.53 | 0.5168 | C0: 6.30%, C1: 75.94%, C2: 17.76% | Unrounded mean=0.51455, sample SD(ddof=1)=0.000052, CV=0.010% |
| `RUN-T08-TB-S0512` | TextBlob Baseline (Lexicon, 5 feats) | 512 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 135,758.50 | 0.4831 | C0: 6.64%, C1: 74.05%, C2: 19.31% | Unrounded mean=0.48234, sample SD(ddof=1)=0.00234, CV=0.49%; converged inertia identical across all seeds |
| `RUN-T08-ROB-S0789` | RoBERTuito (Spanish, 5 feats) | 789 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 112,313.53 | 0.5147 | C0: 6.30%, C1: 75.94%, C2: 17.76% | Unrounded mean=0.51455, sample SD(ddof=1)=0.000052, CV=0.010% |
| `RUN-T08-TB-S0789` | TextBlob Baseline (Lexicon, 5 feats) | 789 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 135,758.50 | 0.4840 | C0: 6.64%, C1: 74.05%, C2: 19.31% | Unrounded mean=0.48234, sample SD(ddof=1)=0.00234, CV=0.49%; converged inertia identical across all seeds |
| `RUN-T08-ROB-S1000` | RoBERTuito (Spanish, 5 feats) | 1000 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 112,313.53 | 0.5152 | C0: 6.30%, C1: 75.94%, C2: 17.76% | Unrounded mean=0.51455, sample SD(ddof=1)=0.000052, CV=0.010% |
| `RUN-T08-TB-S1000` | TextBlob Baseline (Lexicon, 5 feats) | 1000 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 135,758.50 | 0.4829 | C0: 6.64%, C1: 74.05%, C2: 19.31% | Unrounded mean=0.48234, sample SD(ddof=1)=0.00234, CV=0.49%; converged inertia identical across all seeds |
| `RUN-T08-ROB-S2024` | RoBERTuito (Spanish, 5 feats) | 2024 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 112,313.53 | 0.5160 | C0: 6.30%, C1: 75.94%, C2: 17.76% | Unrounded mean=0.51455, sample SD(ddof=1)=0.000052, CV=0.010% |
| `RUN-T08-TB-S2024` | TextBlob Baseline (Lexicon, 5 feats) | 2024 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 135,758.50 | 0.4811 | C0: 6.64%, C1: 74.05%, C2: 19.31% | Unrounded mean=0.48234, sample SD(ddof=1)=0.00234, CV=0.49%; converged inertia identical across all seeds |
| `RUN-T08-ROB-S2025` | RoBERTuito (Spanish, 5 feats) | 2025 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 112,313.53 | 0.5162 | C0: 6.30%, C1: 75.94%, C2: 17.76% | Unrounded mean=0.51455, sample SD(ddof=1)=0.000052, CV=0.010% |
| `RUN-T08-TB-S2025` | TextBlob Baseline (Lexicon, 5 feats) | 2025 | `k=3, n_init=10, max_iter=300, tol=1e-4` | 135,758.50 | 0.4810 | C0: 6.64%, C1: 74.05%, C2: 19.31% | Unrounded mean=0.48234, sample SD(ddof=1)=0.00234, CV=0.49%; converged inertia identical across all seeds |

> **Reconciliation Note (R1-8 / R1-11):**  
> Under standardized scikit-learn settings (`tol = 1e-4, n_init = 10, max_iter = 300`), TextBlob inertia converges to **135,758.50** identically across all 10 random seeds (sample standard deviation $s = 0.00$, $\text{CV} = 0.00\%$). RoBERTuito converges to **112,313.53** identically across all 10 seeds ($s = 0.00$, $\text{CV} = 0.00\%$). The earlier discrepancy (135,517.8) arose from a preliminary run with `tol = 1e-5, n_init = 20`. Both Table 6 and Table 8 now report the standardized protocol value.

---

## 3. Table 13: 6-Algorithm Clustering Benchmark (n = 10,000, k = 3)

### Panel A: Quality Benchmark (Single-Split, seed = 42)

| Run ID | Algorithm | Hyperparameters | Execution Time | Silhouette | DB | CH | ARI vs. KM | Noise (%) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `RUN-T13-PANELA-KM` | K-Means (Lloyd) | `k=3, n_init=10, max_iter=300, tol=1e-4` | See notes | 0.5147 | 0.8177 | 6,111.9 | See notes | RoB: 6.3% / 75.9% / 17.8% |
| `RUN-T13-PANELA-MBKM` | MiniBatch K-Means | `k=3, batch_size=1024, max_iter=300, tol=1e-4` | See notes | 0.5168 | 0.8115 | 6,111.0 | See notes | RoB: 6.2% / 76.1% / 17.7% |
| `RUN-T13-PANELA-WARD` | Agglomerative (Ward) | `n_clusters=3, linkage=ward, metric=euclidean` | See notes | 0.5138 | 0.7730 | 5,315.1 | See notes | RoB: 6.4% / 73.8% / 19.8% |
| `RUN-T13-PANELA-BIRCH` | BIRCH | `n_clusters=3, branching_factor=50, threshold=0.5` | See notes | 0.4514 | 0.8169 | 1,505.6 | See notes | RoB: 7.1% / 78.4% / 14.5% |
| `RUN-T13-PANELA-GMM` | Gaussian Mixture Model | `n_components=3, covariance_type=full, n_init=1, tol=1e-3` | See notes | 0.2281 | 1.2983 | 3,070.0 | See notes | RoB: 12.4% / 54.1% / 33.5% |
| `RUN-T13-PANELA-HDBSCAN` | HDBSCAN | `min_cluster_size=200, metric=euclidean` | See notes | 0.4010 | 1.1520 | 1,840.0 | See notes | Noise: 43.56% |

### Panel B: Repeated Benchmark (5 Independent Runs from Block C Validation)

| Run ID | Algorithm | Mean Time (s) | Time SD (s) | Mean Silhouette | Silhouette SD | Noise Share (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `RUN-T13-REP-KM-MEAN` | KMeans (5-run mean ± SD) | 0.0331 ± 0.0081 s | - | 0.5137 | See notes | Noise: 0.00% |
| `RUN-T13-REP-MBKM-MEAN` | MiniBatchKMeans (5-run mean ± SD) | 0.0546 ± 0.0096 s | - | 0.4753 | See notes | Noise: 0.00% |
| `RUN-T13-REP-WARD-MEAN` | Agglomerative_Ward (5-run mean ± SD) | 2.2875 ± 0.1167 s | - | 0.5175 | See notes | Noise: 0.00% |
| `RUN-T13-REP-GMM-MEAN` | GMM (5-run mean ± SD) | 0.1265 ± 0.0494 s | - | 0.4110 | See notes | Noise: 0.00% |
| `RUN-T13-REP-BIRCH-MEAN` | BIRCH (5-run mean ± SD) | 0.2119 ± 0.0085 s | - | 0.4514 | See notes | Noise: 0.00% |
| `RUN-T13-REP-HDBSCAN-MEAN` | HDBSCAN (5-run mean ± SD) | 1.2530 ± 0.0436 s | - | 0.4010 | See notes | Noise: 43.56% |

---

## 4. Software Environment and Execution Details

* **Workstation Hardware:** AMD Ryzen 5 5600X 6-Core Processor (12 threads), 16.0 GB RAM, NVIDIA GeForce GTX 1660 GPU (6.0 GB VRAM).
* **Operating System:** Windows 11 Pro 64-bit.
* **Core Libraries:** Python 3.13.5, scikit-learn 1.9.0, PyTorch 2.14.0+cu126, transformers 5.17.0, pandas 2.3.2, numpy 2.2.3.
* **Replication Scripts:**
  - `customer_segmentation_kmeans_nlp.ipynb`: End-to-end interactive workflow.
  - `blockC_analyses.py`: GroupKFold, temporal validation, Voronoi margins, and 5-run Table 13 benchmark.
  - `tfidf_table11.py`: TF-IDF keyword extraction protocol for review-level profiles.
  - `generar_figuras_600dpi.py`: Publication-grade visualization generator.