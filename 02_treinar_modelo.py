"""
=============================================================
MODELO DE RISCO DE INADIMPLÊNCIA
Script 2 de 3 — Treinamento e Avaliação do Modelo
=============================================================

O que este script faz:
  1. Lê o arquivo gerado pelo script 01
  2. Treina um modelo Random Forest
  3. Avalia a performance com métricas e gráficos
  4. Salva o modelo treinado para uso no script 03

Dependências:
  pip install pandas scikit-learn matplotlib seaborn openpyxl joblib
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

ARQUIVO_ENTRADA = "base_modelagem.csv"    # gerado pelo script 01
ARQUIVO_MODELO  = "modelo_inadimplencia.pkl"
ARQUIVO_SCALER  = "scaler.pkl"
ARQUIVO_FEATURES = "features_utilizadas.txt"

# Colunas que NÃO entram como variáveis do modelo
COLUNAS_EXCLUIR = [
    "cnpj",
    "inadimplente",   # variável alvo
    "plano",          # já virou dummy
    "total_inadimplentes",  # vazamento de dados (leakage)
    "taxa_inadimplencia",   # idem — derivada diretamente do alvo
]

# ─────────────────────────────────────────────
# 1. LEITURA E PREPARAÇÃO
# ─────────────────────────────────────────────

print("=" * 60)
print("ETAPA 1 — Carregando dados...")
print("=" * 60)

df = pd.read_csv(ARQUIVO_ENTRADA)

print(f"  Clientes na base : {len(df):,}")
print(f"  Inadimplentes    : {df['inadimplente'].sum():,} ({df['inadimplente'].mean():.1%})")

# Define variáveis X (features) e y (alvo)
colunas_excluir_existentes = [c for c in COLUNAS_EXCLUIR if c in df.columns]
X = df.drop(columns=colunas_excluir_existentes)
y = df["inadimplente"]

# Garante que tudo é numérico
X = X.select_dtypes(include=[np.number])

# Salva a lista de features usadas
with open(ARQUIVO_FEATURES, "w") as f:
    f.write("\n".join(X.columns.tolist()))

print(f"\n  Features utilizadas ({len(X.columns)}):")
for col in X.columns:
    print(f"    - {col}")

# ─────────────────────────────────────────────
# 2. DIVISÃO TREINO / TESTE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("ETAPA 2 — Dividindo treino e teste...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% para teste
    random_state=42,
    stratify=y            # mantém a proporção de inadimplentes nos dois conjuntos
)

# Normaliza as features numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"  Treino : {len(X_train):,} clientes")
print(f"  Teste  : {len(X_test):,} clientes")

# ─────────────────────────────────────────────
# 3. TREINAMENTO DO MODELO
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("ETAPA 3 — Treinando o modelo Random Forest...")
print("  (pode levar alguns minutos dependendo do tamanho da base)")
print("=" * 60)

# class_weight="balanced" corrige automaticamente o desbalanceamento
# (clientes inadimplentes são minoria na base — isso é esperado)
modelo = RandomForestClassifier(
    n_estimators=300,        # número de árvores
    max_depth=12,            # profundidade máxima de cada árvore
    min_samples_leaf=20,     # mínimo de clientes por folha (evita overfitting)
    class_weight="balanced", # penaliza mais os erros nos inadimplentes
    random_state=42,
    n_jobs=-1                # usa todos os núcleos disponíveis
)

modelo.fit(X_train_scaled, y_train)
print("  ✓ Modelo treinado!")

# ─────────────────────────────────────────────
# 4. AVALIAÇÃO — VALIDAÇÃO CRUZADA
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("ETAPA 4 — Avaliando performance (validação cruzada 5 folds)...")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_auc = cross_val_score(modelo, X_train_scaled, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
scores_f1  = cross_val_score(modelo, X_train_scaled, y_train, cv=cv, scoring="f1", n_jobs=-1)

print(f"\n  ROC-AUC (médio) : {scores_auc.mean():.3f} ± {scores_auc.std():.3f}")
print(f"  F1-Score (médio): {scores_f1.mean():.3f} ± {scores_f1.std():.3f}")
print("\n  Interpretação do ROC-AUC:")
print("    0.90–1.00 → Excelente  |  0.80–0.90 → Bom")
print("    0.70–0.80 → Aceitável  |  < 0.70    → Fraco")

# ─────────────────────────────────────────────
# 5. MÉTRICAS NO CONJUNTO DE TESTE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("ETAPA 5 — Resultado no conjunto de teste (20% da base)...")
print("=" * 60)

y_pred       = modelo.predict(X_test_scaled)
y_pred_proba = modelo.predict_proba(X_test_scaled)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
ap  = average_precision_score(y_test, y_pred_proba)

print(f"\n  ROC-AUC          : {auc:.3f}")
print(f"  Average Precision: {ap:.3f}")
print("\n  Relatório completo:")
print(classification_report(y_test, y_pred, target_names=["Adimplente", "Inadimplente"]))

# ─────────────────────────────────────────────
# 6. GRÁFICOS
# ─────────────────────────────────────────────

print("\nETAPA 6 — Gerando gráficos...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Avaliação do Modelo de Inadimplência", fontsize=14, fontweight="bold")

# Gráfico 1: Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color="#0F6E56", lw=2, label=f"AUC = {auc:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
axes[0].fill_between(fpr, tpr, alpha=0.1, color="#0F6E56")
axes[0].set_xlabel("Taxa de Falsos Positivos")
axes[0].set_ylabel("Taxa de Verdadeiros Positivos")
axes[0].set_title("Curva ROC")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Greens", ax=axes[1],
    xticklabels=["Adimplente", "Inadimplente"],
    yticklabels=["Adimplente", "Inadimplente"]
)
axes[1].set_xlabel("Predito")
axes[1].set_ylabel("Real")
axes[1].set_title("Matriz de Confusão")

# Gráfico 3: Importância das variáveis (top 15)
importancias = pd.Series(modelo.feature_importances_, index=X.columns)
importancias = importancias.sort_values(ascending=True).tail(15)

importancias.plot(kind="barh", ax=axes[2], color="#0F6E56", alpha=0.8)
axes[2].set_title("Variáveis mais importantes")
axes[2].set_xlabel("Importância")
axes[2].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("avaliacao_modelo.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Gráfico salvo: avaliacao_modelo.png")

# ─────────────────────────────────────────────
# 7. ANÁLISE DE IMPORTÂNCIA DAS VARIÁVEIS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Importância das variáveis (ranking completo):")
print("=" * 60)

ranking = (
    pd.DataFrame({
        "variavel": X.columns,
        "importancia": modelo.feature_importances_
    })
    .sort_values("importancia", ascending=False)
    .assign(importancia_pct=lambda x: (x["importancia"] * 100).round(2))
)

print(ranking[["variavel", "importancia_pct"]].to_string(index=False))

# ─────────────────────────────────────────────
# 8. SALVAR MODELO
# ─────────────────────────────────────────────

joblib.dump(modelo,  ARQUIVO_MODELO)
joblib.dump(scaler,  ARQUIVO_SCALER)

print(f"\n✓ Modelo salvo: {ARQUIVO_MODELO}")
print(f"✓ Scaler salvo: {ARQUIVO_SCALER}")
print("\n  Próximo passo: execute o script 03_gerar_scores.py")
