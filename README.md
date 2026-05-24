# Modelo Preditivo de Risco de Inadimplência

Projeto desenvolvido para a disciplina de Engenharia de Prompt e Fundamentos de Inteligência Artificial

## O que é

Pipeline de Machine Learning que gera um score de risco de inadimplência (0–100) para cada cliente de uma empresa SaaS B2B, combinando:

- Modelo Random Forest treinado com histórico de cobranças
- IA Generativa (Claude) para geração de pareceres em linguagem natural
- Dashboard web interativo para visualização dos resultados

## Como rodar

### Pré-requisitos

```bash
pip install pandas scikit-learn matplotlib seaborn openpyxl joblib anthropic
```

### Execução

```bash
python 01_preparar_dados.py
python 02_treinar_modelo.py
python 03_gerar_scores.py
```

### Dashboard

Abra o arquivo `index.html` no navegador — funciona offline, sem instalação.

## Colunas esperadas na planilha de entrada (`dados_clientes.xlsx`)

| Coluna | Tipo | Descrição |
|---|---|---|
| `cnpj` | Texto | Identificador do cliente |
| `mes_cobranca` | Data | Mês de referência |
| `data_vencimento` | Data | Vencimento da cobrança |
| `data_pagamento` | Data | Pagamento (vazio = não pago) |
| `dias_em_atraso` | Número | 0 = pagou em dia |
| `status_cobranca` | Texto | pago / inadimplente / em aberto |
| `plano` | Texto | Produto contratado |
| `data_entrada_base` | Data | Quando entrou na base |
| `health_score` | Número | Score de saúde (0–100) |
| `estado` | Texto | UF do cliente |
| `campanha` | Texto | Nome da campanha de aquisição |
| `teve_campanha` | Sim/Não | Veio via campanha? |
| `teve_desconto` | Sim/Não | Teve desconto na contratação? |

## Stack

- Python 3.10+
- scikit-learn (Random Forest)
- pandas, openpyxl, joblib
- Claude API (Anthropic) — pareceres em linguagem natural
- HTML + JavaScript (dashboard)
- Microsoft Power BI (visualização final)

## Grupo

Álvaro · Felipe Bertoldo · Felipe Miranda · Gabriel Vieira · João Pedro de Almeida · João Vitor Silva
