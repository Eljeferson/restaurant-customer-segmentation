import os
import pandas as pd
import numpy as np
import unicodedata
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# Path relative to repository root
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sentiment_spanish_benchmark_50k.csv")
if not os.path.exists(DATA_PATH):
    # Fallback to local
    DATA_PATH = "sentiment_spanish_benchmark_50k.csv"

print(f"Loading benchmark sample from: {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# Compute RoBERTuito 5D clustering partition (Table 10 and 11 reference)
sat_rob = 0.5 * df['score_norm'] + 0.5 * ((df['sent_robertuito'] + 1) / 2)
df['sat_rob'] = sat_rob
f5 = ['score_norm', 'sent_robertuito', 'sat_rob', 'likes_log', 'freq_log']
X1 = StandardScaler().fit_transform(df[f5])
km1 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X1)

# Align labels to official manuscript names:
# km1.labels_ == 2 -> 0: Socially Endorsed (3,148)
# km1.labels_ == 1 -> 1: Passive Satisfied (37,972)
# km1.labels_ == 0 -> 2: Critical (8,880)
label_map = {2: 0, 1: 1, 0: 2}
df['cluster_roberta'] = [label_map[x] for x in km1.labels_]

TEXT, LABEL = "review", "cluster_roberta"
NAMES = {0: "Socially Endorsed", 1: "Passive Satisfied", 2: "Critical"}

def strip_acc(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

texts = df[TEXT].fillna("").astype(str)
sin_tildes = not texts.str.contains(r"[áéíóú]", regex=True).any()
sw = set(stopwords.words("spanish")) | {"comida", "lugar", "restaurante", "bien", "buen"}
if sin_tildes:
    sw = {strip_acc(w) for w in sw}

vec = TfidfVectorizer(
    sublinear_tf=True,
    ngram_range=(1, 1),
    min_df=5,
    max_df=0.50,
    lowercase=True,
    stop_words=sorted(sw),
    token_pattern=r"(?u)\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]{3,}\b"
)
X = vec.fit_transform(texts)
vocab = np.array(vec.get_feature_names_out())

print(f"\nn = {len(df)}, vocabulario = {len(vocab)}, NLTK {nltk.__version__}, texto sin tildes: {sin_tildes}")
print(f"'estaba' en vocabulario: {'estaba' in set(vocab)}")

for k in sorted(df[LABEL].unique()):
    m = (df[LABEL] == k).values
    mean_w = np.asarray(X[m].mean(axis=0)).ravel()
    top = mean_w.argsort()[::-1][:5]
    print(f"\nCluster {k} – {NAMES.get(k, k)} (n = {m.sum()}):")
    for i in top:
        print(f"  {vocab[i]:<15} {mean_w[i]:.4f}")
