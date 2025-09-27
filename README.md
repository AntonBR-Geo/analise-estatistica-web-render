# 📊 Análise Estatística e PCA Web

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black?logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Uma **aplicação web completa** para análise estatística avançada e **Análise de Componentes Principais (PCA)**, com suporte total ao **formato brasileiro** de dados (separador `;`, decimal com `,`).

Ideal para pesquisadores, analistas e estudantes que precisam de uma ferramenta **rápida, visual e profissional** para explorar dados e preparar índices sintéticos para AHP.

---

## ✨ Funcionalidades

### 📈 Análise Estatística Básica
- **Estatística descritiva** completa (média, mediana, variância, skewness, kurtosis)
- **Boxplots** interativos (mostrados em lotes de 5 para grandes datasets)
- **Testes de normalidade**: Shapiro-Wilk e Kolmogorov-Smirnov
- **Matriz de correlação** com heatmap e tabela

### 🔍 Adequação para Análise Fatorial
- **Teste KMO** (Kaiser-Meyer-Olkin) com MSA por variável e geral
- **Teste de Bartlett** de esfericidade

### 🧮 Análise de Componentes Principais (PCA)
- **Extração de componentes**:
  - Critério de Kaiser (autovalor > 1) ✅ **padrão**
  - Número fixo de componentes (definido pelo usuário)
- **Métodos de rotação**:
  - **Varimax** (ortogonal) ✅ **padrão**
  - **Equamax** (ortogonal)
  - **Promax** (oblíqua)
- **Saídas completas**:
  - Gráfico de escarpa (scree plot)
  - Tabela de variância explicada (solução inicial e rotacionada)
  - Matriz de componente rotativa com comunalidades e singularidades
  - Matriz de transformação de componente
  - Número de iterações até convergência (máx. 25)

### 🇧🇷 Especificidades Brasileiras
- ✅ Leitura de CSV com separador `;` e decimal `,`
- ✅ Formatação de números com vírgula como separador decimal
- ✅ Casas decimais inteligentes (inteiros sem casas, decimais com 3 casas)

### 🎯 Preparação para AHP
- Cálculo de **índice sintético ponderado** baseado nas componentes
- Dados prontos para uso em matrizes de comparação pareada

---

## 🚀 Como Usar

### 1. Acesse a aplicação online
➡️ **[Clique aqui para usar agora](https://seu-app.onrender.com)** *(substitua pelo seu link após deploy)*

### 2. Ou execute localmente

#### Pré-requisitos
- Python 3.9+
- pip

#### Instalação
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/analise-estatistica-web.git
cd analise-estatistica-web

# Instale as dependências
pip install -r requirements.txt

# Execute a aplicação
python app.py
