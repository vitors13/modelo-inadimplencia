"""
=============================================================
GERADOR DE BASE SIMULADA — Modelo de Risco de Inadimplência
=============================================================

Gera uma planilha Excel com dados fictícios de cobranças
para testar o pipeline de Machine Learning.

Saída: dados_clientes.xlsx
       300 clientes / ~7.000 cobranças
=============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────
# CONFIGURAÇÕES — ajuste se quiser
# ─────────────────────────────────────────────

N_CLIENTES  = 300
PLANOS      = ["Plano Básico", "Plano Pro", "Plano Enterprise"]
ESTADOS     = ["SP", "MG", "RJ", "RS", "PR", "BA", "GO", "SC", "PE", "CE", "ES", "MT"]
CAMPANHAS   = ["Black Friday", "Indicação", "Google Ads", "Sem Campanha", "Parceiro", "Evento"]

# ─────────────────────────────────────────────
# FUNÇÕES
# ─────────────────────────────────────────────

def gerar_cnpj(i):
    """Gera um CNPJ fictício no formato padrão."""
    return f"{i:02d}.{i*3:03d}.{i*7:03d}/0001-{(i % 99):02d}"

def gerar_cliente(i):
    """
    Gera todas as cobranças de um cliente com base no seu perfil.
    Perfis:
      - bom   (55%): paga em dia, Health Score alto
      - medio (30%): algum atraso, risco médio
      - ruim  (15%): muitos atrasos, Health Score baixo
    """
    perfil   = np.random.choice(["bom", "medio", "ruim"], p=[0.55, 0.30, 0.15])
    plano    = np.random.choice(PLANOS,    p=[0.50, 0.35, 0.15])
    estado   = np.random.choice(ESTADOS)
    campanha = np.random.choice(CAMPANHAS, p=[0.15, 0.20, 0.25, 0.25, 0.10, 0.05])
    teve_campanha = "Não" if campanha == "Sem Campanha" else "Sim"
    meses_na_base = np.random.randint(3, 48)
    data_entrada  = datetime(2026, 5, 1) - timedelta(days=meses_na_base * 30)

    # Desconto — clientes de campanha ou Plano Básico têm mais chance
    prob_desconto = 0.60 if teve_campanha == "Sim" else 0.20
    if plano == "Plano Enterprise":
        prob_desconto += 0.20
    teve_desconto = "Sim" if np.random.random() < prob_desconto else "Não"

    # Perfil de pagamento
    if perfil == "bom":
        health_score = np.random.randint(65, 100)
        prob_atraso  = np.random.uniform(0.00, 0.12)
    elif perfil == "medio":
        health_score = np.random.randint(35, 70)
        prob_atraso  = np.random.uniform(0.15, 0.45)
    else:
        health_score = np.random.randint(5, 40)
        prob_atraso  = np.random.uniform(0.45, 0.90)

    linhas = []
    for m in range(meses_na_base):
        mes  = data_entrada + timedelta(days=m * 30)
        venc = mes + timedelta(days=15)

        # Tendência: clientes ruins pioram no final, bons melhoram
        if perfil == "ruim" and m >= meses_na_base - 3:
            p = min(prob_atraso * 1.40, 0.95)
        elif perfil == "bom" and m >= meses_na_base - 3:
            p = max(prob_atraso * 0.60, 0.0)
        else:
            p = prob_atraso

        atrasou = np.random.random() < p

        if atrasou:
            dias_atraso = int(np.random.choice(
                [5, 10, 15, 20, 30, 45, 60, 90],
                p=[0.10, 0.15, 0.20, 0.15, 0.20, 0.10, 0.07, 0.03]
            ))
            data_pag = venc + timedelta(days=dias_atraso)
            status   = "inadimplente" if dias_atraso >= 30 else "pago"
        else:
            dias_atraso = 0
            data_pag    = venc - timedelta(days=np.random.randint(0, 3))
            status      = "pago"

        linhas.append({
            "cnpj"             : gerar_cnpj(i),
            "mes_cobranca"     : mes.strftime("%Y-%m-%d"),
            "data_vencimento"  : venc.strftime("%Y-%m-%d"),
            "data_pagamento"   : data_pag.strftime("%Y-%m-%d") if status == "pago" else "",
            "dias_em_atraso"   : dias_atraso,
            "status_cobranca"  : status,
            "plano"            : plano,
            "data_entrada_base": data_entrada.strftime("%Y-%m-%d"),
            "health_score"     : health_score,
            "estado"           : estado,
            "campanha"         : campanha,
            "teve_campanha"    : teve_campanha,
            "teve_desconto"    : teve_desconto,
        })
    return linhas

# ─────────────────────────────────────────────
# GERAR A BASE
# ─────────────────────────────────────────────

print("Gerando base simulada...")

todas_linhas = []
for i in range(1, N_CLIENTES + 1):
    todas_linhas.extend(gerar_cliente(i))

df = pd.DataFrame(todas_linhas)
df.to_excel("dados_clientes.xlsx", index=False)

# ─────────────────────────────────────────────
# RESUMO
# ─────────────────────────────────────────────

print(f"\n✓ Arquivo 'dados_clientes.xlsx' gerado com sucesso!")
print(f"\n  Clientes    : {N_CLIENTES:,}")
print(f"  Cobranças   : {len(df):,}")
print(f"  Colunas     : {list(df.columns)}")
print(f"  Período     : {df['mes_cobranca'].min()} a {df['mes_cobranca'].max()}")
print(f"\n  Campanhas:")
for c, n in df.groupby("campanha")["cnpj"].nunique().items():
    print(f"    {c:<20}: {n} clientes")
print(f"\n  Com desconto: {df.groupby('cnpj')['teve_desconto'].first().eq('Sim').sum()} clientes")
print(f"\n  Próximo passo: execute o script 01_preparar_dados.py")
