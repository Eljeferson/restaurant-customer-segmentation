# Customer Segmentation from Large-Scale Restaurant Reviews

Replication package and code repository for the empirical study:
**"Validated Customer Segmentation from Large-Scale Restaurant Reviews: Integrating K-means Clustering with Complementary Multilingual NLP Sentiment Features for Business Intelligence"**

---

## Repository Structure

```
├── customer_segmentation_kmeans_nlp.ipynb   # Complete analysis workflow (preprocessing, NLP benchmark, K-means, ablation, surrogate RF)
├── generar_figuras_600dpi.py                # Publication-grade (600 DPI) visualization generation script
├── requirements.txt                         # Required Python libraries and dependencies
├── data/
│   ├── sentiment_spanish_benchmark_50k.csv  # Stratified benchmark corpus (50,000 reviews with RoBERTuito, XLM-R, TextBlob features)
│   └── reviews.zip                          # Full raw & clean review dataset (755,066 reviews, compressed)
└── README.md                                # Replication guide and documentation
```

---

## Dataset Overview

* **Full Clean Corpus (N = 755,066):** Collected from Google Places (91.48%) and TripAdvisor (8.52%) across 9,258 restaurants in Peru. Pseudonymized user IDs (`id_nick`) and anonymized establishment codes (`service`).
* **Stratified Benchmark Sample (n = 50,000):** Used for Spanish Transformer inference benchmarking (RoBERTuito and XLM-RoBERTa vs. baseline TextBlob), bootstrap stability, and feature ablation.

To extract the full dataset:
```python
import zipfile
with zipfile.ZipFile('data/reviews.zip', 'r') as zip_ref:
    zip_ref.extractall('data/')
```

---

## Reproduction Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/Eljeferson/restaurant-customer-segmentation.git
cd restaurant-customer-segmentation
pip install -r requirements.txt
```

### 2. Run the Analysis
Open and execute `customer_segmentation_kmeans_nlp.ipynb` in Jupyter Notebook, Google Colab, or VS Code.

### 3. Generate High-Resolution Figures
```bash
python generar_figuras_600dpi.py
```

---

## Key Experimental Findings

1. **Spanish Transformer Sentiment vs. English Lexicon:** TextBlob yields 93.91% exact zero polarities on Spanish feedback, whereas native RoBERTuito eliminates zero-inflation (0.00% exact zeros, Macro F1 = 0.7946, Pearson r = 0.6213 with ratings).
2. **Optimal Cluster Structure (k = 3):** Convergent internal validation across Silhouette, Davies-Bouldin (0.8114), and Calinski-Harabasz (30,646.0).
3. **Cluster Robustness & Ablation:** Removing the composite satisfaction feature (FS2) maintains ARI = 0.9699 (98.99% label alignment, 1.01% reassigned), confirming cluster geometry is not artificially induced.
4. **Replication Fidelity:** Native Spanish sentiment provides a +7.26 pp gain in partition replication fidelity over rating-and-engagement baselines, compared to +0.43 pp for TextBlob.

---

## License
This repository is released under the MIT License for academic and non-commercial research use.
