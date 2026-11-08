"""
EP2 - ACH2118 - Análise do item 9 (interpretação global, local e análise de erros).
Classificador simples: o modelo-base TF-IDF (1-2 gramas) + Regressão Logística do stacking.

Uso:  pip install eli5 scikit-learn pandas openpyxl
      python analise_eli5.py train.xlsx
"""
import sys
import numpy as np
import pandas as pd
import eli5
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

L = ["c1", "c234", "c5"]
label2id = {c: i for i, c in enumerate(L)}


def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(C=4, max_iter=2000)),
    ])


df = pd.read_excel(sys.argv[1] if len(sys.argv) > 1 else "train.xlsx")
df["resp_text"] = df["resp_text"].astype(str)
df["label"] = df["clarity"].map(label2id)
df["text_group"] = pd.factorize(df["resp_text"])[0]
y = df["label"].values

# ---------------- OOF com os mesmos folds do notebook ----------------
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(df, y, groups=df["text_group"]))
oof = np.zeros((len(df), 3))
fold_models, fold_of = {}, np.zeros(len(df), int)
for k, (tr, va) in enumerate(folds):
    m = make_model().fit(df["resp_text"].iloc[tr], y[tr])
    oof[va] = m.predict_proba(df["resp_text"].iloc[va])
    fold_models[k], fold_of[va] = m, k
pred = oof.argmax(1)
print(f"OOF TF-IDF + RegLog: acc={accuracy_score(y, pred):.4f}  f1_macro={f1_score(y, pred, average='macro'):.4f}")
print(confusion_matrix(y, pred))

# ---------------- (i) interpretação global ----------------
full = make_model().fit(df["resp_text"], y)
vec, clf = full.named_steps["tfidf"], full.named_steps["clf"]
glob = eli5.format_as_dataframe(eli5.explain_weights(clf, vec=vec, top=(25, 15), target_names=L))
glob.to_csv("eli5_global.csv", index=False)
for c in L:
    s = glob[glob.target == c]
    print(f"\n== {c} | positivos:", ", ".join(f"{f}({w:.2f})" for f, w in zip(s.feature, s.weight) if w > 0))
    print(f"== {c} | negativos:", ", ".join(f"{f}({w:.2f})" for f, w in zip(s.feature, s.weight) if w < 0))

# ---------------- (ii) interpretação local ----------------
# índices usados no relatório (podem variar se a ordem do train.xlsx for diferente)
EXEMPLOS = [14793, 6562, 14996, 3830]
for i in EXEMPLOS:
    m = fold_models[fold_of[i]]
    txt = df["resp_text"].iloc[i]
    p = m.predict_proba([txt])[0]
    alvo = L[p.argmax()]
    ex = eli5.format_as_dataframe(eli5.explain_prediction(
        m.named_steps["clf"], txt, vec=m.named_steps["tfidf"], target_names=L, targets=[alvo], top=10))
    print(f"\n#{i} real={df['clarity'].iloc[i]} previsto={alvo} ({p.max():.0%})\n  {txt.strip()[:300]}")
    print("  ", ", ".join(f"{f} {w:+.2f}" for f, w in zip(ex.feature, ex.weight) if f != "<BIAS>"))

# ---------------- (iii) análise de erros ----------------
df["pred"] = np.array(L)[pred]
df["conf"] = oof.max(1)
df["ok"] = df["pred"] == df["clarity"]
df["len"] = df["resp_text"].str.len()
g = df.groupby("text_group")["clarity"].nunique()
df["conflito"] = df["text_group"].isin(g[g > 1].index)
err = df[~df["ok"]]
extremos = ((err.clarity == "c1") & (err.pred == "c5")) | ((err.clarity == "c5") & (err.pred == "c1"))
print(f"\nErros: {len(err)} | trocas entre extremos: {extremos.mean():.1%}")
print("Acurácia em textos com rótulo conflitante:", round(df[df.conflito].ok.mean(), 4),
      "| demais:", round(df[~df.conflito].ok.mean(), 4))
print(df.groupby(pd.cut(df.conf, [0, .4, .5, .6, .7, 1.0]), observed=True).ok.agg(["mean", "size"]))
print(df.groupby(pd.qcut(df.len, 4), observed=True).ok.agg(["mean", "size"]))
print("\nErros de alta confiança (textos curtos) para seleção manual:")
for i, r in err[err.len < 450].sort_values("conf", ascending=False).head(25).iterrows():
    print(f"\n#{i} real={r.clarity} previsto={r.pred} conf={r.conf:.2f} conflito={r.conflito}\n  {r.resp_text.strip()[:450]}")
