"""
Calcula a acurácia das previsões (arquivos rotulados pelo modelo) contra o gabarito.

Como usar no VS Code:
  1. Instale as dependências (uma vez), no terminal:
       pip install pandas openpyxl scikit-learn scipy
  2. Ajuste os caminhos em CONFIGURAÇÃO abaixo.
  3. Rode:  python calcular_acuracia.py   (ou o botão "Run Python File")
"""
from pathlib import Path

import numpy as np
import pandas as pd
from itertools import combinations

from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# ============================ CONFIGURAÇÃO ============================
# Gabarito: arquivo com os rótulos verdadeiros (colunas resp_text e clarity)
GABARITO = "test1-rotulado-para-teste-estatistico.xlsx"

# Previsões a avaliar: nome curto -> caminho (os arquivos gerados pelo notebook)
# Para testar a vantagem do seu modelo sobre um baseline, inclua aqui também o arquivo
# rotulado pelo baseline (ex.: "tfidf": "test_ep1_rotulado_tfidf.xlsx"). Todos os pares são comparados.
PREVISOES = {
    "trabalho anterior": "test1_rotulado.xlsx",
    "média simples": "test_ep1_rotulado_stacking_v4_media_simples.xlsx",
    "stacking": "test_ep1_rotulado_stacking_v4_stacking.xlsx",
}

COL_TEXTO = "resp_text"
COL_ROTULO = "clarity"
CLASSES = ["c1", "c234", "c5"]
ALFA = 0.05                    # nível de significância do teste de McNemar
SALVAR_ERROS = True            # salva uma planilha com as linhas erradas de cada modelo
# ======================================================================


def ler(caminho):
    df = pd.read_excel(caminho)
    faltam = {COL_TEXTO, COL_ROTULO} - set(df.columns)
    assert not faltam, f"{caminho}: faltam as colunas {faltam} (colunas: {list(df.columns)})"
    df = df[[COL_TEXTO, COL_ROTULO]].copy()
    df[COL_TEXTO] = df[COL_TEXTO].astype(str).str.strip()
    df[COL_ROTULO] = df[COL_ROTULO].astype(str).str.strip()
    return df


def alinhar(gab, prev, nome):
    """Alinha previsão e gabarito. Usa a ordem das linhas se os textos coincidirem;
    senão, casa pelo texto (tratando textos repetidos pela ordem de ocorrência)."""
    if len(gab) == len(prev) and (gab[COL_TEXTO].values == prev[COL_TEXTO].values).all():
        return gab[COL_ROTULO].values, prev[COL_ROTULO].values, gab[COL_TEXTO].values

    print(f"[{nome}] ordem das linhas diferente do gabarito -> casando pelo texto")
    g = gab.assign(_n=gab.groupby(COL_TEXTO).cumcount())
    p = prev.assign(_n=prev.groupby(COL_TEXTO).cumcount())
    m = g.merge(p, on=[COL_TEXTO, "_n"], how="inner", suffixes=("_true", "_pred"))
    sem_par = len(gab) - len(m)
    if sem_par:
        print(f"[{nome}] AVISO: {sem_par} linhas do gabarito sem correspondência na previsão (ficam fora do cálculo)")
    return m[f"{COL_ROTULO}_true"].values, m[f"{COL_ROTULO}_pred"].values, m[COL_TEXTO].values


def ic_wilson(acertos, n, z=1.96):
    """Intervalo de confiança de 95% (Wilson) para a acurácia."""
    p = acertos / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    meia = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centro - meia, centro + meia


gab = ler(GABARITO)
print(f"Gabarito: {len(gab)} linhas | distribuição: {gab[COL_ROTULO].value_counts().to_dict()}\n")

acertos_por_modelo = {}
erros_planilhas = {}

for nome, caminho in PREVISOES.items():
    if not Path(caminho).exists():
        print(f"[{nome}] arquivo não encontrado: {caminho} (pulado)\n")
        continue
    y_true, y_pred, textos = alinhar(gab, ler(caminho), nome)

    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=CLASSES)
    lo, hi = ic_wilson(int((y_true == y_pred).sum()), n)

    print("=" * 70)
    print(f"{nome.upper()}  ({caminho})")
    print("=" * 70)
    print(f"Acurácia : {acc:.4f}  ({int((y_true == y_pred).sum())}/{n})   IC 95%: [{lo:.4f}, {hi:.4f}]")
    print(f"F1 macro : {f1:.4f}")
    print("\n" + classification_report(y_true, y_pred, labels=CLASSES, digits=4, zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=CLASSES),
                      index=[f"real {c}" for c in CLASSES], columns=[f"prev {c}" for c in CLASSES])
    print("Matriz de confusão:\n", cm, "\n")

    acertos_por_modelo[nome] = pd.Series(y_true == y_pred, index=textos)
    erros_planilhas[nome[:31]] = pd.DataFrame({COL_TEXTO: textos, "real": y_true, "previsto": y_pred})[y_true != y_pred]

# ---- Teste de McNemar: compara cada par de modelos nas mesmas linhas ----
#   b = só o modelo A acertou | c = só o modelo B acertou (linhas em que os dois concordam não entram)
#   estatística (correção de continuidade): chi2 = (|b - c| - 1)^2 / (b + c), 1 grau de liberdade
#   p-valor exato (binomial): recomendado quando b + c < 25
def mcnemar(a_ok, b_ok):
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    if b + c == 0:
        return b, c, 0.0, 1.0, 1.0
    est = (abs(b - c) - 1) ** 2 / (b + c)
    p_chi2 = float(chi2.sf(est, df=1))
    p_exato = float(binomtest(b, b + c, 0.5).pvalue)
    return b, c, est, p_chi2, p_exato


for (n1, a), (n2, bm) in combinations(acertos_por_modelo.items(), 2):
    if not (len(a) == len(bm) and (a.index == bm.index).all()):
        print(f"\n[McNemar {n1} x {n2}] linhas não alinhadas, teste pulado")
        continue
    b, c, est, p_chi2, p_exato = mcnemar(a.values, bm.values)
    acc1, acc2 = a.mean(), bm.mean()
    melhor, pior = (n1, n2) if acc1 >= acc2 else (n2, n1)
    print("\n" + "=" * 70)
    print(f"McNEMAR: {n1} (acc={acc1:.4f}) x {n2} (acc={acc2:.4f})")
    print("=" * 70)
    print(f"Tabela de discordância: só '{n1}' acertou (b) = {b} | só '{n2}' acertou (c) = {c}")
    print(f"Estatística McNemar (chi2, correção de continuidade, gl=1) = {est:.4f}")
    print(f"p-valor (qui-quadrado) = {p_chi2:.4g} | p-valor (exato binomial) = {p_exato:.4g}")
    sig = p_chi2 < ALFA


if SALVAR_ERROS and erros_planilhas:
    with pd.ExcelWriter("erros_por_modelo.xlsx") as w:
        for aba, d in erros_planilhas.items():
            d.to_excel(w, sheet_name=aba, index=False)
    print("\nLinhas erradas salvas em: erros_por_modelo.xlsx")
