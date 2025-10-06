import io
import base64
import traceback
import hashlib
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kstest, norm
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')

# Novas importações para exportação
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.drawing.image import Image as ExcelImage
import tempfile
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'chave_secreta_analise_estatistica'
app.config['TEMPLATES_AUTO_RELOAD'] = False

# Variável global para armazenar resultados
resultados_globais = {}

class AnaliseEstatistica:
    def __init__(self, df, variaveis_selecionadas=None):
        self.df = df
        self.variaveis_numericas = self._identificar_variaveis_numericas()
        self.variaveis_selecionadas = variaveis_selecionadas or self.variaveis_numericas
        self.resultados = {}
    
    def _identificar_variaveis_numericas(self):
        """Identifica variáveis numéricas no dataframe"""
        numericas = []
        for coluna in self.df.columns:
            try:
                serie_temp = pd.to_numeric(
                    self.df[coluna].astype(str).str.replace(',', '.'), 
                    errors='coerce'
                )
                if serie_temp.notna().sum() >= 2:
                    numericas.append(coluna)
            except:
                continue
        return numericas
    
    def identificar_variaveis_nao_numericas(self):
        """Identifica variáveis não numéricas (categóricas)"""
        todas_colunas = set(self.df.columns)
        numericas = set(self.variaveis_numericas)
        return list(todas_colunas - numericas)
    
    def converter_para_numerico(self):
        """Converte variáveis selecionadas para formato numérico"""
        for coluna in self.variaveis_selecionadas:
            if coluna in self.variaveis_numericas:
                self.df[coluna] = pd.to_numeric(
                    self.df[coluna].astype(str).str.replace(',', '.'), 
                    errors='coerce'
                )
        self.df = self.df.dropna(subset=self.variaveis_selecionadas)
    
    def validar_variaveis_selecionadas(self):
        """Valida se as variáveis selecionadas são adequadas para análise"""
        if len(self.variaveis_selecionadas) < 2:
            return False, "Selecione pelo menos 2 variáveis para análise"
        
        for var in self.variaveis_selecionadas:
            if var not in self.df.columns:
                return False, f"Variável '{var}' não encontrada no dataset"
            
            dados_validos = self.df[var].notna().sum()
            if dados_validos < 3:
                return False, f"Variável '{var}' tem poucos dados válidos ({dados_validos})"
        
        return True, "Variáveis válidas"
    
    def calcular_matriz_anti_imagem(self, dados):
        """Calcula a matriz de correlação anti-imagem usando factor_analyzer"""
        try:
            from factor_analyzer.factor_analyzer import calculate_kmo
            # Calcular KMO para obter as MAS individuais
            kmo_all, kmo_model = calculate_kmo(dados)
            # Matriz de correlação
            R = np.corrcoef(dados, rowvar=False)
            # Inversa da matriz de correlação
            R_inv = np.linalg.inv(R)
            # Calcular a matriz anti-imagem
            D = np.diag(1 / np.sqrt(np.diag(R_inv)))
            A = D @ R_inv @ D
            # Matriz anti-imagem final:
            # Diagonal = MAS (valores KMO individuais)
            # Fora da diagonal = A_ij (correlações parciais)
            matriz_anti_imagem = A.copy()
            np.fill_diagonal(matriz_anti_imagem, kmo_all)
            return matriz_anti_imagem, A, kmo_all
        except Exception as e:
            print(f"Erro no cálculo da matriz anti-imagem: {e}")
            n_variaveis = dados.shape[1]
            matriz_zeros = np.zeros((n_variaveis, n_variaveis))
            kmo_fallback = np.ones(n_variaveis) * 0.5
            return matriz_zeros, np.eye(n_variaveis), np.ones(n_variaveis)

    def criar_heatmap_anti_imagem(self, matriz_anti_imagem, variaveis):
        """Cria heatmap para a matriz anti-imagem, mostrando explicitamente os valores da diagonal (MAS)"""
        n = len(variaveis)
        plt.figure(figsize=(max(10, n * 0.8), max(8, n * 0.8)))  # Tamanho adaptativo
        matriz_plot = matriz_anti_imagem.copy()
    
        ax = sns.heatmap(
            matriz_plot,
            annot=False,
            cmap='RdBu_r',
            center=0,
            square=True,
            cbar_kws={"shrink": 0.8},
            xticklabels=variaveis,
            yticklabels=variaveis,
            linewidths=0.5,
            linecolor='white'
        )
    
        # Adicionar todos os textos manualmente com fonte maior
        for i in range(n):
            for j in range(n):
                val = matriz_plot[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j + 0.5, i + 0.5, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=12,  # 👈 Fonte dos números dentro das células
                        color=color, weight='bold')
    
        plt.title('Matriz de Correlação Anti-Imagem\n(Correlações Parciais - Diagonal mostra MAS)',
                fontsize=16, fontweight='bold')  # 👈 Título também aumentado
        plt.xticks(rotation=45, ha='right', fontsize=14)  # 👈 Rótulos do eixo X
        plt.yticks(rotation=0, fontsize=14)               # 👈 Rótulos do eixo Y
        plt.tight_layout(pad=1.5)
    
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=200, bbox_inches='tight', pad_inches=0.3)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url
    
  
    def estatisticas_descritivas(self):
        """Calcula estatísticas descritivas apenas para variáveis selecionadas"""
        stats_data = []
        for coluna in self.variaveis_selecionadas:
            if coluna in self.variaveis_numericas:
                dados = self.df[coluna].dropna()
                if len(dados) > 0:
                    stats_data.append({
                        'Variável': coluna,
                        'Média': dados.mean(),
                        'Mediana': dados.median(),
                        'Desvio Padrão': dados.std(),
                        'Variância': dados.var(),
                        'Assimetria (Skewness)': stats.skew(dados),
                        'Curtose (Kurtosis)': stats.kurtosis(dados),
                        'Q1': dados.quantile(0.25),
                        'Q3': dados.quantile(0.75),
                        'Mínimo': dados.min(),
                        'Máximo': dados.max(),
                        'N': len(dados)
                    })
        return pd.DataFrame(stats_data)
    
    def criar_boxplot(self):
        """Cria boxplot para variáveis selecionadas"""
        plt.figure(figsize=(12, 8))
        dados_plot = self.df[self.variaveis_selecionadas]
        dados_plot.boxplot()
        plt.title('Boxplot das Variáveis Selecionadas')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url
    
    def teste_normalidade(self):
        """Realiza testes de normalidade para variáveis selecionadas"""
        resultados = []
        for coluna in self.variaveis_selecionadas:
            if coluna in self.variaveis_numericas:
                dados = self.df[coluna].dropna()
                if len(dados) > 3:
                    stat_sw, p_sw = shapiro(dados)
                    stat_ks, p_ks = kstest(dados, 'norm', args=(dados.mean(), dados.std()))
                    
                    resultados.append({
                        'Variável': coluna,
                        'Shapiro-Wilk (Estatística)': stat_sw,
                        'Shapiro-Wilk (p-valor)': p_sw,
                        'Shapiro-Wilk (Normal)': p_sw > 0.05,
                        'Kolmogorov-Smirnov (Estatística)': stat_ks,
                        'Kolmogorov-Smirnov (p-valor)': p_ks,
                        'Kolmogorov-Smirnov (Normal)': p_ks > 0.05
                    })
        return pd.DataFrame(resultados)
    
    def matriz_correlacao(self):
        """Calcula matriz de correlação para variáveis selecionadas"""
        corr_matrix = self.df[self.variaveis_selecionadas].corr()
    
        # Aumentar o tamanho da figura para acomodar melhor os textos
        n = len(self.variaveis_selecionadas)
        plt.figure(figsize=(max(10, n * 0.8), max(8, n * 0.8)))
    
        ax = sns.heatmap(
            corr_matrix,
            annot=False,  # Desativar anotação automática
            cmap='coolwarm',
            center=0,
            square=True,
            cbar_kws={"shrink": 0.8},
            xticklabels=self.variaveis_selecionadas,
            yticklabels=self.variaveis_selecionadas,
            linewidths=0.5,
            linecolor='white'
        )
    
        # Adicionar todos os textos manualmente com fonte maior
        for i in range(n):
            for j in range(n):
                val = corr_matrix.iloc[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j + 0.5, i + 0.5, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=12,  # 👈 Tamanho da fonte dos números (ajuste conforme necessário)
                        color=color, weight='bold')
    
        plt.title('Matriz de Correlação de Pearson - Variáveis Selecionadas', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=14)  # 👈 Tamanho dos rótulos do eixo X
        plt.yticks(rotation=0, fontsize=14)               # 👈 Tamanho dos rótulos do eixo Y
        plt.tight_layout(pad=1.5)
    
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=200, bbox_inches='tight', pad_inches=0.3)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return corr_matrix, plot_url
    
    def teste_kmo_bartlett(self):
        """Realiza testes KMO e Bartlett para variáveis selecionadas"""
        dados = self.df[self.variaveis_selecionadas].dropna()
        
        chi_square, p_value = calculate_bartlett_sphericity(dados)
        kmo_all, kmo_model = calculate_kmo(dados)
        
        return {
            'Bartlett_Chi_Quadrado': chi_square,
            'Bartlett_p_valor': p_value,
            'Bartlett_Significativo': p_value < 0.05,
            'KMO_Geral': kmo_model,
            'KMO_Variaveis': dict(zip(self.variaveis_selecionadas, kmo_all))
        }
    
    def calcular_matriz_transformacao_spss(self, cargas_iniciais, cargas_rotacionadas, rotacao):
        """
        Calcula a matriz de transformação no padrão SPSS
        """
        try:
            if rotacao in ['varimax', 'quartimax', 'equamax']:
                A_ini = cargas_iniciais
                A_rot = cargas_rotacionadas
                
                if np.linalg.matrix_rank(A_ini) == A_ini.shape[1]:
                    T = np.linalg.solve(A_ini.T @ A_ini, A_ini.T @ A_rot)
                else:
                    T = np.linalg.pinv(A_ini.T @ A_ini) @ A_ini.T @ A_rot
                
                return T
                
            elif rotacao in ['promax', 'oblimin']:
                try:
                    fa_obliqua = FactorAnalyzer(n_factors=cargas_iniciais.shape[1], 
                                              rotation=rotacao, method='principal')
                    from factor_analyzer.utils import correlation_matrix
                    R = correlation_matrix(cargas_iniciais)
                    fa_obliqua.fit(R)
                    if hasattr(fa_obliqua, 'phi_') and fa_obliqua.phi_ is not None:
                        return fa_obliqua.phi_
                except:
                    pass
                
                T = np.corrcoef(cargas_rotacionadas.T)
                return T
                
            else:
                n_fatores = cargas_iniciais.shape[1]
                return np.eye(n_fatores)
                
        except Exception as e:
            print(f"Erro no cálculo da matriz de transformação: {e}")
            n_fatores = cargas_iniciais.shape[1]
            return np.eye(n_fatores)
    
    def analise_fatorial_spss(self, n_fatores=None, rotacao='varimax'):
        """Análise fatorial seguindo padrão SPSS para variáveis selecionadas"""
        dados = self.df[self.variaveis_selecionadas].dropna()
        n_observacoes, n_variaveis = dados.shape
    
        if n_observacoes < n_variaveis:
            raise ValueError(f"Número de observações ({n_observacoes}) menor que número de variáveis ({n_variaveis})")
    
        # Padronizar dados
        scaler = StandardScaler()
        dados_padronizados = scaler.fit_transform(dados)
    
        # Matriz de correlações
        matriz_correlacao = np.corrcoef(dados_padronizados, rowvar=False)
    
        # Matriz anti-imagem - CORREÇÃO: passar os valores KMO
        matriz_anti_imagem, matriz_imagem, mas = self.calcular_matriz_anti_imagem(dados_padronizados)
        heatmap_anti_imagem = self.criar_heatmap_anti_imagem(matriz_anti_imagem, self.variaveis_selecionadas)
        
        # PCA para critério de Kaiser
        pca = PCA()
        pca.fit(dados_padronizados)
        autovalores_pca = pca.explained_variance_
        
        if n_fatores is None:
            n_fatores = sum(autovalores_pca > 1)
            n_fatores = max(1, min(n_fatores, n_variaveis))
        
        # ANÁLISE FATORIAL - método PCA como no SPSS
        fa = FactorAnalyzer(n_factors=n_variaveis, rotation=None, method='principal')
        fa.fit(dados_padronizados)
        
        # Autovalores iniciais
        autovalores_iniciais = fa.get_eigenvalues()[0]
        variancia_inicial = autovalores_iniciais / n_variaveis * 100
        variancia_acumulada_inicial = np.cumsum(variancia_inicial)
        
        # Extrair apenas n_fatores
        autovalores_iniciais = autovalores_iniciais[:n_fatores]
        variancia_inicial = variancia_inicial[:n_fatores]
        variancia_acumulada_inicial = variancia_acumulada_inicial[:n_fatores]
        
        # Cargas fatoriais iniciais
        cargas_iniciais = fa.loadings_[:, :n_fatores]
        
        # Aplicar rotação
        if rotacao != 'none':
            fa_rotacao = FactorAnalyzer(n_factors=n_fatores, rotation=rotacao, method='principal')
            fa_rotacao.fit(dados_padronizados)
            cargas_rotacionadas = fa_rotacao.loadings_
            n_iteracoes = getattr(fa_rotacao, 'n_iter_', 'N/A')
            
            if rotacao in ['varimax', 'quartimax', 'equamax']:
                soma_quadrados = np.sum(cargas_rotacionadas**2, axis=0)
                variancia_rotacionada = soma_quadrados / n_variaveis * 100
                variancia_acumulada_rotacionada = np.cumsum(variancia_rotacionada)
            else:
                soma_quadrados = np.sum(cargas_rotacionadas**2, axis=0)
                variancia_rotacionada = soma_quadrados / n_variaveis * 100
                variancia_acumulada_rotacionada = np.cumsum(variancia_rotacionada)
        else:
            cargas_rotacionadas = cargas_iniciais
            variancia_rotacionada = variancia_inicial
            variancia_acumulada_rotacionada = variancia_acumulada_inicial
            n_iteracoes = 'N/A'
        
        # MATRIZ DE TRANSFORMAÇÃO DE COMPONENTES
        matriz_transformacao = self.calcular_matriz_transformacao_spss(
            cargas_iniciais, cargas_rotacionadas, rotacao
        )
        
        # COMUNALIDADES
        comunalidades = np.sum(cargas_rotacionadas**2, axis=1)
        
        # MATRIZ DE CARGAS
        cargas_rotacionadas_df = pd.DataFrame(
            cargas_rotacionadas,
            index=self.variaveis_selecionadas,
            columns=[f'Fator_{i+1}' for i in range(n_fatores)]
        )
        
        comunalidades_df = pd.DataFrame({
            'Inicial': np.sum(cargas_iniciais**2, axis=1),
            'Extração': comunalidades
        }, index=self.variaveis_selecionadas)
        comunalidades_df['Singularidade'] = 1 - comunalidades_df['Extração']
        
        # MATRIZ DE TRANSFORMAÇÃO
        matriz_transformacao_df = pd.DataFrame(
            matriz_transformacao,
            index=[f'Fator_{i+1}' for i in range(n_fatores)],
            columns=[f'Fator_{i+1}' for i in range(n_fatores)]
        )
        
        # SCREE PLOT
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(autovalores_pca) + 1), autovalores_pca, 'bo-', label='Autovalores')
        plt.axhline(y=1, color='r', linestyle='--', label='Autovalor = 1')
        plt.title('Scree Plot - Análise Fatorial')
        plt.xlabel('Número do Fator')
        plt.ylabel('Autovalor')
        plt.legend()
        plt.grid(True)
        
        img_scree = io.BytesIO()
        plt.savefig(img_scree, format='png', dpi=300, bbox_inches='tight')
        img_scree.seek(0)
        scree_url = base64.b64encode(img_scree.getvalue()).decode()
        plt.close()
        
        # SCORES
        try:
            R = matriz_correlacao
            L = cargas_rotacionadas
            pesos_scores = np.linalg.solve(R, L)
            
            scores = dados_padronizados @ pesos_scores
            scores_df = pd.DataFrame(
                scores,
                columns=[f'Score_Fator_{i+1}' for i in range(n_fatores)],
                index=dados.index
            )
        except:
            if rotacao != 'none':
                scores = fa_rotacao.transform(dados_padronizados)
            else:
                scores = fa.transform(dados_padronizados)[:, :n_fatores]
            
            scores_df = pd.DataFrame(
                scores,
                columns=[f'Score_Fator_{i+1}' for i in range(n_fatores)],
                index=dados.index
            )
        
        return {
            'n_fatores': n_fatores,
            'rotacao': rotacao,
            'n_variaveis': n_variaveis,
            'n_observacoes': n_observacoes,
            'n_iteracoes': n_iteracoes,
            
            # Variância Explicada
            'variancia_inicial': {
                'autovalores': autovalores_iniciais,
                'percentual': variancia_inicial,
                'acumulado': variancia_acumulada_inicial
            },
            'variancia_rotacionada': {
                'soma_quadrados': np.sum(cargas_rotacionadas**2, axis=0),
                'percentual': variancia_rotacionada,
                'acumulado': variancia_acumulada_rotacionada
            },
            
            # Matrizes
            'cargas_rotacionadas': cargas_rotacionadas_df,
            'comunalidades': comunalidades_df,
            'matriz_transformacao': matriz_transformacao_df,
            'matriz_correlacao': matriz_correlacao,
            'matriz_anti_imagem': matriz_anti_imagem,
            'heatmap_anti_imagem': heatmap_anti_imagem,
            
            # Gráficos e Scores
            'scree_plot': scree_url,
            'scores': scores_df,
            
            # Informações
            'autovalores_totais': autovalores_pca
        }

# CLASSE ExportadorResultados CORRIGIDA - SEM ESTILOS DINÂMICOS
class ExportadorResultados:
    def __init__(self, resultados):
        self.resultados = resultados
    
    def exportar_csv_completo(self):
        """Exporta todos os resultados em formato CSV (arquivo ZIP)"""
        import zipfile
        
        buffer_zip = io.BytesIO()
        
        with zipfile.ZipFile(buffer_zip, 'w') as zip_file:
            # 1. Estatísticas Descritivas
            stats_df = pd.DataFrame(self.resultados['estatisticas'])
            zip_file.writestr('estatisticas_descritivas.csv', stats_df.to_csv(index=False, sep=';', decimal=','))
            
            # 2. Testes de Normalidade
            norm_df = pd.DataFrame(self.resultados['normalidade'])
            zip_file.writestr('testes_normalidade.csv', norm_df.to_csv(index=False, sep=';', decimal=','))
            
            # 3. Matriz de Correlação
            corr_df = pd.DataFrame(self.resultados['correlacao']['matriz'])
            zip_file.writestr('matriz_correlacao.csv', corr_df.to_csv(sep=';', decimal=','))
            
            # 4. Teste KMO por Variável
            kmo_variaveis_df = pd.DataFrame([
                {'Variável': var, 'KMO': kmo_value}
                for var, kmo_value in self.resultados['kmo_bartlett']['KMO_Variaveis'].items()
            ])
            zip_file.writestr('kmo_por_variavel.csv', kmo_variaveis_df.to_csv(index=False, sep=';', decimal=','))
            
            # 5. Variância Explicada - INICIAL E ROTACIONADA
            variancia_data = []
            for i in range(self.resultados['fatorial']['n_fatores']):
                variancia_data.append({
                    'Fator': f'Fator {i+1}',
                    'Autovalor_Inicial': self.resultados['fatorial']['variancia_inicial']['autovalores'][i],
                    'Variância_Inicial_Percentual': self.resultados['fatorial']['variancia_inicial']['percentual'][i],
                    'Variância_Inicial_Acumulada': self.resultados['fatorial']['variancia_inicial']['acumulado'][i],
                    'Soma_Quadrados_Rotacionada': self.resultados['fatorial']['variancia_rotacionada']['soma_quadrados'][i],
                    'Variância_Rotacionada_Percentual': self.resultados['fatorial']['variancia_rotacionada']['percentual'][i],
                    'Variância_Rotacionada_Acumulada': self.resultados['fatorial']['variancia_rotacionada']['acumulado'][i]
                })
            variancia_df = pd.DataFrame(variancia_data)
            zip_file.writestr('variancia_explicada.csv', variancia_df.to_csv(index=False, sep=';', decimal=','))
            
            # 6. Cargas Fatoriais
            cargas_df = self.resultados['fatorial']['cargas_rotacionadas']
            zip_file.writestr('cargas_fatoriais.csv', cargas_df.to_csv(sep=';', decimal=','))
            
            # 7. Comunalidades
            comun_df = self.resultados['fatorial']['comunalidades']
            zip_file.writestr('comunalidades.csv', comun_df.to_csv(sep=';', decimal=','))
            
            # 8. Matriz de Transformação
            transf_df = self.resultados['fatorial']['matriz_transformacao']
            zip_file.writestr('matriz_transformacao.csv', transf_df.to_csv(sep=';', decimal=','))
        
        buffer_zip.seek(0)
        return buffer_zip
    
    def exportar_excel_completo(self):
        """Exporta todos os resultados em formato Excel"""
        buffer_excel = io.BytesIO()
        
        with pd.ExcelWriter(buffer_excel, engine='openpyxl') as writer:
            # 1. Resumo da Análise
            resumo_data = {
                'Parâmetro': [
                    'Número de Variáveis', 'Número de Observações', 'Número de Fatores',
                    'Método de Rotação', 'KMO Geral', 'Bartlett (p-valor)'
                ],
                'Valor': [
                    self.resultados['fatorial']['n_variaveis'],
                    self.resultados['fatorial']['n_observacoes'],
                    self.resultados['fatorial']['n_fatores'],
                    self.resultados['fatorial']['rotacao'].title(),
                    f"{self.resultados['kmo_bartlett']['KMO_Geral']:.4f}",
                    f"{self.resultados['kmo_bartlett']['Bartlett_p_valor']:.6f}"
                ]
            }
            resumo_df = pd.DataFrame(resumo_data)
            resumo_df.to_excel(writer, sheet_name='Resumo Análise', index=False)
            
            # 2. Estatísticas Descritivas
            stats_df = pd.DataFrame(self.resultados['estatisticas'])
            stats_df.to_excel(writer, sheet_name='Estatísticas Descritivas', index=False)
            
            # 3. Testes de Normalidade
            norm_df = pd.DataFrame(self.resultados['normalidade'])
            norm_df.to_excel(writer, sheet_name='Testes Normalidade', index=False)
            
            # 4. Matriz de Correlação
            corr_df = pd.DataFrame(self.resultados['correlacao']['matriz'])
            corr_df.to_excel(writer, sheet_name='Matriz Correlação', index=True)
            
            # 5. Teste KMO por Variável
            kmo_variaveis_df = pd.DataFrame([
                {'Variável': var, 'KMO': kmo_value}
                for var, kmo_value in self.resultados['kmo_bartlett']['KMO_Variaveis'].items()
            ])
            kmo_variaveis_df.to_excel(writer, sheet_name='KMO por Variável', index=False)
            
            # 6. Variância Explicada - INICIAL
            variancia_inicial_data = []
            for i in range(self.resultados['fatorial']['n_fatores']):
                variancia_inicial_data.append({
                    'Fator': f'Fator {i+1}',
                    'Autovalor': self.resultados['fatorial']['variancia_inicial']['autovalores'][i],
                    'Variância (%)': self.resultados['fatorial']['variancia_inicial']['percentual'][i],
                    'Variância Acumulada (%)': self.resultados['fatorial']['variancia_inicial']['acumulado'][i]
                })
            variancia_inicial_df = pd.DataFrame(variancia_inicial_data)
            variancia_inicial_df.to_excel(writer, sheet_name='Variância Inicial', index=False)
            
            # 7. Variância Explicada - ROTACIONADA
            variancia_rotacionada_data = []
            for i in range(self.resultados['fatorial']['n_fatores']):
                variancia_rotacionada_data.append({
                    'Fator': f'Fator {i+1}',
                    'Soma dos Quadrados': self.resultados['fatorial']['variancia_rotacionada']['soma_quadrados'][i],
                    'Variância (%)': self.resultados['fatorial']['variancia_rotacionada']['percentual'][i],
                    'Variância Acumulada (%)': self.resultados['fatorial']['variancia_rotacionada']['acumulado'][i]
                })
            variancia_rotacionada_df = pd.DataFrame(variancia_rotacionada_data)
            variancia_rotacionada_df.to_excel(writer, sheet_name='Variância Rotacionada', index=False)
            
            # 8. Cargas Fatoriais
            cargas_df = self.resultados['fatorial']['cargas_rotacionadas']
            cargas_df.to_excel(writer, sheet_name='Cargas Fatoriais', index=True)
            
            # 9. Comunalidades
            comun_df = self.resultados['fatorial']['comunalidades']
            comun_df.to_excel(writer, sheet_name='Comunalidades', index=True)
            
            # 10. Matriz de Transformação
            transf_df = self.resultados['fatorial']['matriz_transformacao']
            transf_df.to_excel(writer, sheet_name='Matriz Transformação', index=True)
        
        buffer_excel.seek(0)
        return buffer_excel
    
    def exportar_pdf_completo(self):
        """Exporta relatório completo em PDF seguindo mesma sequência do Excel"""
        buffer_pdf = io.BytesIO()
        doc = SimpleDocTemplate(buffer_pdf, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Título
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("RELATÓRIO DE ANÁLISE ESTATÍSTICA - StatAnalyzer", title_style))
        story.append(Spacer(1, 20))
        
        # 1. RESUMO DA ANÁLISE
        story.append(Paragraph("RESUMO DA ANÁLISE", styles['Heading2']))
        resumo_data = [
            ['Parâmetro', 'Valor'],
            ['Número de Variáveis', str(self.resultados['fatorial']['n_variaveis'])],
            ['Número de Observações', str(self.resultados['fatorial']['n_observacoes'])],
            ['Número de Fatores', str(self.resultados['fatorial']['n_fatores'])],
            ['Método de Rotação', self.resultados['fatorial']['rotacao'].title()],
            ['KMO Geral', f"{self.resultados['kmo_bartlett']['KMO_Geral']:.4f}"],
            ['Bartlett (p-valor)', f"{self.resultados['kmo_bartlett']['Bartlett_p_valor']:.6f}"]
        ]
        
        resumo_table = Table(resumo_data, colWidths=[3*inch, 3*inch])
        resumo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(resumo_table)
        story.append(Spacer(1, 20))
        
        # 2. VARIÂNCIA EXPLICADA - SOLUÇÃO INICIAL
        story.append(Paragraph("VARIÂNCIA EXPLICADA - SOLUÇÃO INICIAL", styles['Heading2']))
        variancia_inicial_data = [['Fator', 'Autovalor', 'Variância (%)', 'Variância Acumulada (%)']]
        
        for i in range(self.resultados['fatorial']['n_fatores']):
            variancia_inicial_data.append([
                f'Fator {i+1}',
                f"{self.resultados['fatorial']['variancia_inicial']['autovalores'][i]:.4f}",
                f"{self.resultados['fatorial']['variancia_inicial']['percentual'][i]:.2f}",
                f"{self.resultados['fatorial']['variancia_inicial']['acumulado'][i]:.2f}"
            ])
        
        variancia_inicial_table = Table(variancia_inicial_data, colWidths=[1.2*inch, 1.2*inch, 1.3*inch, 1.8*inch])
        variancia_inicial_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(variancia_inicial_table)
        story.append(Spacer(1, 15))
        
        # 3. VARIÂNCIA EXPLICADA - SOLUÇÃO ROTACIONADA
        story.append(Paragraph("VARIÂNCIA EXPLICADA - SOLUÇÃO ROTACIONADA", styles['Heading2']))
        variancia_rotacionada_data = [['Fator', 'Soma dos Quadrados', 'Variância (%)', 'Variância Acumulada (%)']]
        
        for i in range(self.resultados['fatorial']['n_fatores']):
            variancia_rotacionada_data.append([
                f'Fator {i+1}',
                f"{self.resultados['fatorial']['variancia_rotacionada']['soma_quadrados'][i]:.4f}",
                f"{self.resultados['fatorial']['variancia_rotacionada']['percentual'][i]:.2f}",
                f"{self.resultados['fatorial']['variancia_rotacionada']['acumulado'][i]:.2f}"
            ])
        
        variancia_rotacionada_table = Table(variancia_rotacionada_data, colWidths=[1.2*inch, 1.5*inch, 1.3*inch, 1.8*inch])
        variancia_rotacionada_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(variancia_rotacionada_table)
        
        # Legenda da variância
        legend_style = ParagraphStyle(
            'LegendStyle',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey
        )
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "💡 A solução rotacionada redistribui a variância explicada entre os fatores para melhor interpretação", 
            legend_style
        ))
        story.append(Spacer(1, 20))
        
        # 4. TESTE KMO POR VARIÁVEL
        story.append(Paragraph("MEDIDA DE ADEQUAÇÃO AMOSTRAL (KMO) POR VARIÁVEL", styles['Heading2']))
        kmo_data = [['Variável', 'KMO', 'Adequação']]
        
        for var_name, kmo_value in self.resultados['kmo_bartlett']['KMO_Variaveis'].items():
            if kmo_value >= 0.8:
                adequacao = 'Excelente'
            elif kmo_value >= 0.7:
                adequacao = 'Boa'
            elif kmo_value >= 0.6:
                adequacao = 'Regular'
            else:
                adequacao = 'Ruim'
            
            kmo_data.append([
                var_name,
                f"{kmo_value:.4f}",
                adequacao
            ])
        
        kmo_table = Table(kmo_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        kmo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(kmo_table)
        
        # Legenda KMO
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "💡 Interpretação KMO: >0.9 Maravilhoso | 0.8-0.9 Meritório | 0.7-0.8 Médio | 0.6-0.7 Medíocre | <0.6 Inaceitável", 
            legend_style
        ))
        story.append(Spacer(1, 20))
        
        # 5. CARGAS FATORIAIS (PRIMEIRAS 15 VARIÁVEIS)
        story.append(Paragraph("CARGAS FATORIAIS ROTACIONADAS (PRIMEIRAS 15 VARIÁVEIS)", styles['Heading2']))
        cargas_df = self.resultados['fatorial']['cargas_rotacionadas']
        
        # Preparar dados da tabela
        cargas_data = [['Variável'] + [f'Fator {i+1}' for i in range(self.resultados['fatorial']['n_fatores'])]]
        
        # Pegar apenas as primeiras 15 variáveis
        variaveis = cargas_df.index[:15]
        for var_name in variaveis:
            row = [var_name]
            for i in range(self.resultados['fatorial']['n_fatores']):
                carga = cargas_df.loc[var_name, f'Fator_{i+1}']
                row.append(f"{carga:.4f}")
            cargas_data.append(row)
        
        cargas_table = Table(cargas_data, colWidths=[1.5*inch] + [1*inch] * self.resultados['fatorial']['n_fatores'])
        cargas_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(cargas_table)
        
        # Legenda das cargas
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "💡 Cargas ≥ |0.5| são consideradas significativas", 
            legend_style
        ))
        story.append(Spacer(1, 20))
        
        # 6. COMUNALIDADES (PRIMEIRAS 15 VARIÁVEIS)
        story.append(Paragraph("COMUNALIDADES (PRIMEIRAS 15 VARIÁVEIS)", styles['Heading2']))
        comun_df = self.resultados['fatorial']['comunalidades']
        
        comun_data = [['Variável', 'Inicial', 'Extração', 'Singularidade']]
        
        for var_name in variaveis:
            comun_data.append([
                var_name,
                f"{comun_df.loc[var_name, 'Inicial']:.4f}",
                f"{comun_df.loc[var_name, 'Extração']:.4f}",
                f"{comun_df.loc[var_name, 'Singularidade']:.4f}"
            ])
        
        comun_table = Table(comun_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.2*inch])
        comun_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(comun_table)
        
        # Legenda das comunalidades
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "💡 Comunalidades > 0.6 indicam boa representação da variável pelo modelo fatorial", 
            legend_style
        ))
        
        doc.build(story)
        buffer_pdf.seek(0)
        return buffer_pdf

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/selecionar_variaveis', methods=['POST'])
def selecionar_variaveis():
    try:
        if 'arquivo' not in request.files:
            return render_template('erro.html', erro="Nenhum arquivo enviado")
        
        arquivo = request.files['arquivo']
        if arquivo.filename == '':
            return render_template('erro.html', erro="Nenhum arquivo selecionado")
        
        if not arquivo.filename.lower().endswith('.csv'):
            return render_template('erro.html', erro="Arquivo deve ser CSV")
        
        try:
            df = pd.read_csv(arquivo, sep=';', decimal=',', encoding='ISO-8859-1')
        except:
            try:
                df = pd.read_csv(arquivo, sep=';', decimal=',', encoding='latin-1')
            except Exception as e:
                return render_template('erro.html', erro=f"Erro ao ler arquivo: {str(e)}")
        
        if len(df.columns) < 2:
            return render_template('erro.html', erro="Arquivo deve ter pelo menos 2 colunas")
        
        # Identificar variáveis
        analise = AnaliseEstatistica(df)
        variaveis_numericas = analise.variaveis_numericas
        variaveis_nao_numericas = analise.identificar_variaveis_nao_numericas()
        
        if len(variaveis_numericas) < 2:
            return render_template('erro.html', 
                                 erro=f"Encontradas apenas {len(variaveis_numericas)} variáveis numéricas. São necessárias pelo menos 2.")
        
        # Salvar dataframe temporariamente - apenas nomes das colunas para economizar memória
        import hashlib
        import json
        session_id = hashlib.md5(str(pd.Timestamp.now()).encode()).hexdigest()
        
        # Converter DataFrame para JSON serializável
        df_dict = {}
        for coluna in df.columns:
            df_dict[coluna] = df[coluna].fillna('').astype(str).tolist()
        
        resultados_globais[session_id] = {
            'df': df_dict,
            'colunas': df.columns.tolist(),
            'variaveis_numericas': variaveis_numericas,
            'variaveis_nao_numericas': variaveis_nao_numericas
        }
        
        return render_template('selecionar_variaveis.html', 
                             variaveis_numericas=variaveis_numericas,
                             variaveis_nao_numericas=variaveis_nao_numericas,
                             session_id=session_id)
    
    except Exception as e:
        traceback.print_exc()
        return render_template('erro.html', erro=f"Erro durante o processamento: {str(e)}")

@app.route('/analisar', methods=['POST'])
def analisar():
    try:
        session_id = request.form.get('session_id')
        if not session_id or session_id not in resultados_globais:
            return render_template('erro.html', erro="Sessão expirada. Faça upload do arquivo novamente.")
        
        dados_session = resultados_globais[session_id]
        
        # Reconstruir DataFrame a partir do dicionário
        df = pd.DataFrame(dados_session['df'])
        
        # Converter colunas numéricas de volta para numérico
        for coluna in dados_session['variaveis_numericas']:
            if coluna in df.columns:
                df[coluna] = pd.to_numeric(df[coluna].replace('', np.nan), errors='coerce')
        
        # Obter variáveis selecionadas
        variaveis_selecionadas = request.form.getlist('variaveis_selecionadas')
        
        if not variaveis_selecionadas:
            return render_template('erro.html', erro="Nenhuma variável selecionada. Selecione pelo menos 2 variáveis numéricas.")
        
        # Inicializar análise com variáveis selecionadas
        analise = AnaliseEstatistica(df, variaveis_selecionadas)
        
        # Validar variáveis selecionadas
        valido, mensagem = analise.validar_variaveis_selecionadas()
        if not valido:
            return render_template('erro.html', erro=mensagem)
        
        analise.converter_para_numerico()
        
        # Coletar parâmetros do formulário
        n_fatores = request.form.get('n_fatores')
        n_fatores = int(n_fatores) if n_fatores and n_fatores.isdigit() else None
        rotacao = 'varimax'  # Forçar sempre Varimax
        
        resultados = {}
        
        # Análises básicas
        resultados['estatisticas'] = analise.estatisticas_descritivas().to_dict('records')
        resultados['boxplot'] = analise.criar_boxplot()
        resultados['normalidade'] = analise.teste_normalidade().to_dict('records')
        
        corr_matrix, heatmap = analise.matriz_correlacao()
        resultados['correlacao'] = {
            'matriz': corr_matrix.to_dict(),
            'heatmap': heatmap
        }
        
        resultados['kmo_bartlett'] = analise.teste_kmo_bartlett()
        
        # Análise fatorial no padrão SPSS
        resultados['fatorial'] = analise.analise_fatorial_spss(n_fatores, rotacao)
        
        resultados['dados_originais'] = df.head(10).to_dict('records')
        resultados['variaveis_numericas'] = analise.variaveis_selecionadas
        resultados['variaveis_nao_selecionadas'] = list(set(analise.variaveis_numericas) - set(analise.variaveis_selecionadas))
        
        # Atualizar session com resultados
        resultados_globais[session_id] = {
            'scores': resultados['fatorial']['scores'].to_dict(),
            'variaveis_selecionadas': analise.variaveis_selecionadas,
            'resultados_completos': resultados  # Salvar resultados completos para exportação
        }
        
        resultados['session_id'] = session_id
        
        return render_template('resultados_completo.html', resultados=resultados)
    
    except Exception as e:
        traceback.print_exc()
        return render_template('erro.html', erro=f"Erro durante a análise: {str(e)}")

@app.route('/download_scores', methods=['POST'])
def download_scores():
    try:
        session_id = request.form.get('session_id')
        if not session_id or session_id not in resultados_globais:
            return "Nenhum resultado disponível para download. Execute uma análise primeiro."
        
        resultados = resultados_globais[session_id]
        scores_df = pd.DataFrame(resultados['scores'])
        
        output = io.BytesIO()
        scores_df.to_csv(output, index=True, sep=';', decimal=',', encoding='utf-8')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='scores_fatoriais.csv'
        )
    
    except Exception as e:
        return f"Erro ao gerar download: {str(e)}"

# NOVAS ROTAS PARA EXPORTAÇÃO
@app.route('/exportar_csv', methods=['POST'])
def exportar_csv():
    try:
        session_id = request.form.get('session_id')
        if not session_id or session_id not in resultados_globais:
            return "Nenhum resultado disponível para exportação."
        
        resultados = resultados_globais[session_id]['resultados_completos']
        exportador = ExportadorResultados(resultados)
        zip_buffer = exportador.exportar_csv_completo()
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='resultados_analise_completo.zip'
        )
    
    except Exception as e:
        return f"Erro ao exportar CSV: {str(e)}"

@app.route('/exportar_excel', methods=['POST'])
def exportar_excel():
    try:
        session_id = request.form.get('session_id')
        if not session_id or session_id not in resultados_globais:
            return "Nenhum resultado disponível para exportação."
        
        resultados = resultados_globais[session_id]['resultados_completos']
        exportador = ExportadorResultados(resultados)
        excel_buffer = exportador.exportar_excel_completo()
        
        return send_file(
            excel_buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='resultados_analise_completo.xlsx'
        )
    
    except Exception as e:
        return f"Erro ao exportar Excel: {str(e)}"

@app.route('/exportar_pdf', methods=['POST'])
def exportar_pdf():
    try:
        session_id = request.form.get('session_id')
        if not session_id or session_id not in resultados_globais:
            return "Nenhum resultado disponível para exportação."
        
        resultados = resultados_globais[session_id]['resultados_completos']
        exportador = ExportadorResultados(resultados)
        pdf_buffer = exportador.exportar_pdf_completo()
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='relatorio_analise.pdf'
        )
    
    except Exception as e:
        return f"Erro ao exportar PDF: {str(e)}"

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)