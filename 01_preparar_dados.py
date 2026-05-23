"""
=============================================================
MODELO DE RISCO DE INADIMPLÊNCIA
Script 1 de 3 — Preparação e Engenharia de Variáveis
Versão 3 — inclui: campanha comercial + desconto
=============================================================

Variáveis esperadas na planilha (uma linha por cobrança):

  - cnpj               : identificador do cliente
  - mes_cobranca       : mês/ano de referência da cobrança
  - data_vencimento    : data de vencimento
  - data_pagamento     : data do pagamento (vazio = não pago)
  - dias_em_atraso     : dias de atraso (0 = pagou em dia)
  - status_cobranca    : status ("pago", "inadimplente", "em aberto"...)
  - plano              : produto/plano do cliente
  - data_entrada_base  : quando o cliente entrou na base
  - health_score       : score de saúde (0 a 100)
  - estado             : UF do cliente (ex: "SP", "MG")
  - campanha           : nome da campanha de aquisição (ex: "Black Friday",
                         "Indicação", "Sem campanha")
  - teve_campanha      : se o cliente veio via alguma campanha (Sim/Não)
  - teve_desconto      : se o cliente teve desconto na contratação (Sim/Não)
=============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

ARQUIVO_ENTRADA = "dados_clientes.xlsx"
ABA_PLANILHA    = 0

DIAS_ATRASO_INADIMPLENTE = 30
STATUS_INADIMPLENTE      = ["inadimplente", "em aberto", "vencido", "cancelado"]

FAIXAS_HEALTH = {
    "Crítico" : (0,  40),
    "Atenção" : (40, 70),
    "Saudável": (70, 101),
}

COLUNAS = {
    "cnpj"             : "cnpj",
    "mes_cobranca"     : "mes_cobranca",
    "data_vencimento"  : "data_vencimento",
    "data_pagamento"   : "data_pagamento",
    "dias_em_atraso"   : "dias_em_atraso",
    "status_cobranca"  : "status_cobranca",
    "plano"            : "plano",
    "data_entrada_base": "data_entrada_base",
    "health_score"     : "health_score",
    "estado"           : "estado",
    "campanha"         : "campanha",       # ← NOVO: nome da campanha
    "teve_campanha"    : "teve_campanha",  # ← NOVO: Sim/Não
    "teve_desconto"    : "teve_desconto",  # ← NOVO: Sim/Não
}

# ─────────────────────────────────────────────
# 1. LEITURA
# ─────────────────────────────────────────────

print("=" * 60)
print("ETAPA 1 — Lendo a planilha...")
print("=" * 60)

df = pd.read_excel(ARQUIVO_ENTRADA, sheet_name=ABA_PLANILHA)
df = df.rename(columns={v: k for k, v in COLUNAS.items()})

print(f"  Registros carregados : {len(df):,}")
print(f"  Clientes únicos      : {df['cnpj'].nunique():,}")

for col in ["data_vencimento", "data_pagamento", "data_entrada_base"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

if "mes_cobranca" in df.columns:
    df["mes_cobranca"] = pd.to_datetime(df["mes_cobranca"], errors="coerce")
else:
    df["mes_cobranca"] = df["data_vencimento"].dt.to_period("M").dt.to_timestamp()

if "estado" in df.columns:
    df["estado"] = df["estado"].astype(str).str.strip().str.upper()

# Padroniza campanha e desconto
if "campanha" in df.columns:
    df["campanha"] = df["campanha"].astype(str).str.strip().str.title().fillna("Sem Campanha")
    df["campanha"] = df["campanha"].replace({"Nan": "Sem Campanha", "None": "Sem Campanha"})

for col_bin in ["teve_campanha", "teve_desconto"]:
    if col_bin in df.columns:
        df[col_bin] = df[col_bin].astype(str).str.strip().str.lower()
        df[col_bin] = df[col_bin].map({"sim": 1, "não": 0, "nao": 0,
                                        "yes": 1, "no": 0, "1": 1, "0": 0}).fillna(0).astype(int)

# ─────────────────────────────────────────────
# 2. DEFINIÇÃO DE INADIMPLÊNCIA
# ─────────────────────────────────────────────

print("\nETAPA 2 — Definindo inadimplência...")

inad_dias   = df["dias_em_atraso"] >= DIAS_ATRASO_INADIMPLENTE
inad_status = df["status_cobranca"].str.lower().str.contains(
    "|".join(STATUS_INADIMPLENTE), na=False
)
df["cobranca_inadimplente"] = (inad_dias | inad_status).astype(int)
df["cobranca_em_dia"]       = (df["cobranca_inadimplente"] == 0).astype(int)

print(f"  Cobranças em dia        : {df['cobranca_em_dia'].sum():,}")
print(f"  Cobranças inadimplentes : {df['cobranca_inadimplente'].sum():,} ({df['cobranca_inadimplente'].mean():.1%})")

# ─────────────────────────────────────────────
# 3. FEATURES POR CLIENTE
# ─────────────────────────────────────────────

print("\nETAPA 3 — Calculando variáveis por cliente...")

hoje = pd.Timestamp(datetime.today().date())

features = df.groupby("cnpj").agg(
    total_cobranças         = ("cobranca_inadimplente", "count"),
    total_em_dia            = ("cobranca_em_dia", "sum"),
    total_inadimplentes     = ("cobranca_inadimplente", "sum"),
    media_dias_atraso       = ("dias_em_atraso", "mean"),
    max_dias_atraso         = ("dias_em_atraso", "max"),
    mediana_dias_atraso     = ("dias_em_atraso", "median"),
    cobranças_acima_10_dias = ("dias_em_atraso", lambda x: (x >= 10).sum()),
    cobranças_acima_30_dias = ("dias_em_atraso", lambda x: (x >= 30).sum()),
    cobranças_acima_60_dias = ("dias_em_atraso", lambda x: (x >= 60).sum()),
    cobranças_acima_90_dias = ("dias_em_atraso", lambda x: (x >= 90).sum()),
    plano                   = ("plano", "last"),
    data_entrada_base       = ("data_entrada_base", "min"),
    health_score            = ("health_score", "last"),
    estado                  = ("estado", "last"),
    primeiro_mes            = ("mes_cobranca", "min"),
    ultimo_mes              = ("mes_cobranca", "max"),
    # ── Campanha e desconto (fixos por cliente — pega o primeiro valor) ──
    campanha                = ("campanha", "first"),
    teve_campanha           = ("teve_campanha", "first"),
    teve_desconto           = ("teve_desconto", "first"),
).reset_index()

# Variável alvo
features["inadimplente"]       = (features["total_inadimplentes"] > 0).astype(int)
features["taxa_inadimplencia"] = features["total_inadimplentes"] / features["total_cobranças"]
features["taxa_em_dia"]        = features["total_em_dia"] / features["total_cobranças"]
features["prop_atraso_grave"]  = features["cobranças_acima_30_dias"] / features["total_cobranças"]
features["meses_na_base"]      = ((hoje - features["data_entrada_base"]).dt.days / 30).round(1)

# ─────────────────────────────────────────────
# 4. TENDÊNCIA RECENTE
# ─────────────────────────────────────────────

print("\nETAPA 4 — Calculando tendência recente...")

hoje_ts = pd.Timestamp(datetime.today().date())
corte   = hoje_ts - pd.DateOffset(months=3)

df_recente   = df[df["mes_cobranca"] >= corte]
df_historico = df[df["mes_cobranca"] <  corte]

taxa_rec = (df_recente.groupby("cnpj")["cobranca_inadimplente"]
    .mean().reset_index().rename(columns={"cobranca_inadimplente": "taxa_inadim_recente"}))
taxa_his = (df_historico.groupby("cnpj")["cobranca_inadimplente"]
    .mean().reset_index().rename(columns={"cobranca_inadimplente": "taxa_inadim_historica"}))

features = features.merge(taxa_rec, on="cnpj", how="left")
features = features.merge(taxa_his, on="cnpj", how="left")
features["delta_tendencia"] = (features["taxa_inadim_recente"] - features["taxa_inadim_historica"]).fillna(0)

# ─────────────────────────────────────────────
# 5. HEALTH SCORE
# ─────────────────────────────────────────────

print("\nETAPA 5 — Analisando Health Score x Inadimplência...")

def faixa_health(score):
    for nome, (low, high) in FAIXAS_HEALTH.items():
        if low <= score < high:
            return nome
    return "Crítico"

features["faixa_health"] = features["health_score"].apply(faixa_health)

taxa_health = (features.groupby("faixa_health")["inadimplente"]
    .mean().reset_index().rename(columns={"inadimplente": "taxa_inadim_health"}))
features = features.merge(taxa_health, on="faixa_health", how="left")

print("\n  Inadimplência por faixa de Health Score:")
print(features.groupby("faixa_health")
    .agg(clientes=("cnpj","count"), inadimplentes=("inadimplente","sum"))
    .assign(taxa_pct=lambda x: (x["inadimplentes"]/x["clientes"]*100).round(1))
    .reindex(["Crítico","Atenção","Saudável"]).to_string())

# ─────────────────────────────────────────────
# 6. ESTADO
# ─────────────────────────────────────────────

print("\nETAPA 6 — Analisando Estado x Inadimplência...")

taxa_estado = (features.groupby("estado")["inadimplente"]
    .mean().reset_index().rename(columns={"inadimplente": "taxa_inadim_estado"}))
features = features.merge(taxa_estado, on="estado", how="left")

# ─────────────────────────────────────────────
# 7. PLANO
# ─────────────────────────────────────────────

print("\nETAPA 7 — Inadimplência por plano...")

taxa_plano = (features.groupby("plano")["inadimplente"]
    .mean().reset_index().rename(columns={"inadimplente": "taxa_inadim_plano"}))
features = features.merge(taxa_plano, on="plano", how="left")

# ─────────────────────────────────────────────
# 8. CAMPANHA — análise e taxa de inadimplência
# ─────────────────────────────────────────────

print("\nETAPA 8 — Analisando Campanha x Inadimplência...")

# Taxa de inadimplência por campanha
taxa_campanha = (features.groupby("campanha")["inadimplente"]
    .mean().reset_index().rename(columns={"inadimplente": "taxa_inadim_campanha"}))
features = features.merge(taxa_campanha, on="campanha", how="left")

print("\n  Inadimplência por campanha:")
resumo_campanha = (
    features.groupby("campanha")
    .agg(clientes=("cnpj","count"), inadimplentes=("inadimplente","sum"))
    .assign(taxa_pct=lambda x: (x["inadimplentes"]/x["clientes"]*100).round(1))
    .sort_values("taxa_pct", ascending=False)
)
print(resumo_campanha.to_string())

# Taxa de inadimplência por desconto
print("\n  Inadimplência por desconto:")
resumo_desconto = (
    features.groupby("teve_desconto")
    .agg(clientes=("cnpj","count"), inadimplentes=("inadimplente","sum"))
    .assign(taxa_pct=lambda x: (x["inadimplentes"]/x["clientes"]*100).round(1))
)
resumo_desconto.index = resumo_desconto.index.map({1: "Com desconto", 0: "Sem desconto"})
print(resumo_desconto.to_string())

# Taxa de inadimplência por teve_campanha
print("\n  Inadimplência por campanha (binário):")
resumo_teve_campanha = (
    features.groupby("teve_campanha")
    .agg(clientes=("cnpj","count"), inadimplentes=("inadimplente","sum"))
    .assign(taxa_pct=lambda x: (x["inadimplentes"]/x["clientes"]*100).round(1))
)
resumo_teve_campanha.index = resumo_teve_campanha.index.map({1: "Com campanha", 0: "Sem campanha"})
print(resumo_teve_campanha.to_string())

# ─────────────────────────────────────────────
# 9. DUMMIES PARA O MODELO
# ─────────────────────────────────────────────

for col in ["plano", "estado", "faixa_health", "campanha"]:
    dummies = pd.get_dummies(features[col], prefix=col, drop_first=False)
    features = pd.concat([features, dummies], axis=1)

# teve_campanha e teve_desconto já são 0/1 — não precisam de dummy

# ─────────────────────────────────────────────
# 10. SALVAMENTO
# ─────────────────────────────────────────────

print("\nETAPA 9 — Salvando...")

features = features.drop(columns=["data_entrada_base", "primeiro_mes", "ultimo_mes"])
features = features.fillna(0)
features.to_csv("base_modelagem.csv", index=False)

print(f"\n✓ Arquivo 'base_modelagem.csv' salvo!")
print(f"  Clientes        : {len(features):,}")
print(f"  Inadimplentes   : {features['inadimplente'].sum():,} ({features['inadimplente'].mean():.1%})")
print(f"  Variáveis       : {len(features.columns)} colunas")
print(f"  Estados únicos  : {features['estado'].nunique()}")
print(f"  Campanhas únicas: {features['campanha'].nunique()}")
print(f"  Com desconto    : {features['teve_desconto'].sum():,} ({features['teve_desconto'].mean():.1%})")
print("\n  Próximo passo: execute o script 02_treinar_modelo.py")
