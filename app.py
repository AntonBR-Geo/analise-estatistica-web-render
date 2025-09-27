import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np
import base64
from scipy.stats import shapiro, kstest, chi2
import matplotlib.pyplot as plt
import seaborn as sns
import io
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer

app = Flask(__name__)
# Use uma chave secreta forte em produção (defina no Render como variável de ambiente)
app.secret_key = 'sua_chave_secreta_aqui'

def format_brazilian_number(x):
    """Formata número no estilo brasileiro"""
    if pd.isna(x) or x is None:
        return '-'
    if isinstance(x, (int, float)):
        if x == int(x):
            return f"{int(x):,}".replace(",", ".")
        else:
            return f"{x:,.3f}".replace(".", "X").replace(",", ".").replace("X", ",")
    return str(x)

def kmo_and_bartlett(df):
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if numeric_df.shape[1] < 2:
        return {
            'error': f'Menos de 2 variáveis numéricas. Encontradas: {numeric_df.shape[1]}'
        }, {
            'error': f'Menos de 2 variáveis numéricas. Encontradas: {numeric_df.shape[1]}'
        }

    variances = numeric_df.var()
    numeric_df = numeric_df.loc[:, variances > 1e-10]
    removed_columns = [col for col in variances.index if variances[col] <= 1e-10]
    
    if removed_columns:
        print(f"⚠️ Colunas removidas por serem constantes: {removed_columns}")

    if numeric_df.shape[1] < 2:
        return {
            'error': f'Após remover {len(removed_columns)} colunas constantes, restaram apenas {numeric_df.shape[1]} variáveis.'
        }, {
            'error': f'Após remover {len(removed_columns)} colunas constantes, restaram apenas {numeric_df.shape[1]} variáveis.'
        }

    original_rows = len(numeric_df)
    numeric_df = numeric_df.dropna(how='all')
    removed_rows = original_rows - len(numeric_df)
    
    if removed_rows > 0:
        print(f"⚠️ Removidas {removed_rows} linhas totalmente vazias.")

    if len(numeric_df) < 2:
        return {
            'error': f'Após limpeza, restaram apenas {len(numeric_df)} linhas válidas (mínimo: 2).'
        }, {
            'error': f'Após limpeza, restaram apenas {len(numeric_df)} linhas válidas (mínimo: 2).'
        }

    try:
        kmo_individual, kmo_overall = calculate_kmo(numeric_df)
        var_names = numeric_df.columns.tolist()

        if hasattr(kmo_overall, '__len__') and len(kmo_overall) == 1:
            kmo_overall = float(kmo_overall[0])
        elif hasattr(kmo_overall, 'item'):
            kmo_overall = kmo_overall.item()
        else:
            kmo_overall = float(kmo_overall)

        if np.isscalar(kmo_individual):
            kmo_individual = [kmo_individual]
        else:
            kmo_individual = np.array(kmo_individual).flatten().tolist()

        msa_individual = [format_brazilian_number(x) for x in kmo_individual]
        overall_msa = format_brazilian_number(kmo_overall)
        indexed_vars = list(enumerate(zip(var_names, msa_individual)))

        kmo_result = {
            'overall_msa': overall_msa,
            'msa_individual': msa_individual,
            'var_names': var_names,
            'indexed_vars': indexed_vars,
        }

        chi2_stat, p_value = calculate_bartlett_sphericity(numeric_df)
        df_bartlett = int(numeric_df.shape[1] * (numeric_df.shape[1] - 1) / 2)

        bartlett_result = {
            'chi2': format_brazilian_number(chi2_stat),
            'df': df_bartlett,
            'p': format_brazilian_number(p_value)
        }

        return kmo_result, bartlett_result

    except Exception as e:
        error_msg = f'Erro no cálculo: {str(e)}'
        return {'error': error_msg}, {'error': error_msg}

def pca_analysis(df, extraction_method='kaiser', n_components_manual=2, rotation_method='varimax'):
    try:
        df_clean = df.loc[:, df.var() > 1e-10].copy()
        if df_clean.shape[1] < 2:
            return {'error': f'PCA: Menos de 2 variáveis válidas após remover constantes. Restaram: {df_clean.shape[1]}'}

        n_vars = df_clean.shape[1]
        var_names = df_clean.columns.tolist()

        corr_matrix = df_clean.corr().values

        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        if extraction_method == 'manual' and n_components_manual is not None:
            n_comp = min(n_components_manual, len(eigenvals))
        else:
            n_comp = sum(eigenvals > 1)
            if n_comp < 1:
                n_comp = 1

        # Scree plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1, len(eigenvals) + 1), eigenvals, 'bo-', label='Autovalores')
        ax.axhline(y=1, color='r', linestyle='--', label='Critério Kaiser (λ=1)')
        ax.set_xlabel('Componente')
        ax.set_ylabel('Autovalor')
        ax.set_title('Gráfico de Escarpa (Scree Plot)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img_scree = io.BytesIO()
        plt.savefig(img_scree, format='png', bbox_inches='tight')
        plt.close()
        
        img_data = img_scree.getvalue()
        scree_url = f"data:image/png;base64,{base64.b64encode(img_data).decode('utf-8')}"
        
        if len(scree_url) < 1000:
            raise ValueError("Base64 muito curto")

        total_variance = sum(eigenvals)
        variance_explained = eigenvals / total_variance
        cumulative_variance = np.cumsum(variance_explained)

        loadings = eigenvecs * np.sqrt(eigenvals)

        transformation_matrix = None
        try:
            fa = FactorAnalyzer(
                n_factors=n_comp,
                rotation=rotation_method,
                method='principal',
                rotation_kwargs={'max_iter': 25}
            )
            fa.fit(df_clean.values)
            rotated_loadings = fa.loadings_
            n_iterations = fa.n_iter_ if hasattr(fa, 'n_iter_') else 0
            
            if hasattr(fa, 'rotation_matrix_'):
                transformation_matrix = fa.rotation_matrix_
            else:
                transformation_matrix = np.eye(n_comp)
                
        except Exception as e:
            print(f"⚠️ Fallback PCA sem rotação: {str(e)}")
            rotated_loadings = loadings[:, :n_comp]
            n_iterations = 0
            transformation_matrix = np.eye(n_comp)

        rotated_sum_squares = np.sum(rotated_loadings**2, axis=0)
        rotated_variance_explained = rotated_sum_squares / total_variance
        rotated_cumulative_variance = np.cumsum(rotated_variance_explained)

        communalities = np.sum(rotated_loadings**2, axis=1)
        uniquenesses = 1 - communalities

        transformation_matrix_list = [
            [format_brazilian_number(x) for x in row] 
            for row in transformation_matrix
        ] if transformation_matrix is not None else None

        pca_results = {
            'scree_plot': scree_url,
            'eigenvalues': [format_brazilian_number(x) for x in eigenvals],
            'variance_explained': [format_brazilian_number(x * 100) + '%' for x in variance_explained],
            'cumulative_variance': [format_brazilian_number(x * 100) + '%' for x in cumulative_variance],
            'rotated_sum_squares': [format_brazilian_number(x) for x in rotated_sum_squares],
            'rotated_variance_explained': [format_brazilian_number(x * 100) + '%' for x in rotated_variance_explained],
            'rotated_cumulative_variance': [format_brazilian_number(x * 100) + '%' for x in rotated_cumulative_variance],
            'loadings': [[format_brazilian_number(x) for x in row] for row in loadings],
            'rotated_loadings': [[format_brazilian_number(x) for x in row] for row in rotated_loadings],
            'communalities': [format_brazilian_number(x) for x in communalities],
            'uniquenesses': [format_brazilian_number(x) for x in uniquenesses],
            'n_iterations': int(n_iterations),
            'n_components': int(n_comp),
            'rotation_method': rotation_method,
            'extraction_method': extraction_method,
            'method': extraction_method,
            'var_names': var_names,
            'transformation_matrix': transformation_matrix_list
        }

        return pca_results

    except Exception as e:
        error_msg = f'Erro na PCA: {str(e)}'
        fallback_svg = "image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0id2hpdGUiIC8+PGxpbmUgeDE9IjAiIHkxPSIwIiB4Mj0iMTAwIiB5Mj0iMTAwIiBzdHJva2U9ImJsYWNrIiAvPjwvc3ZnPg=="
        return {
            'scree_plot': fallback_svg,
            'error': error_msg,
            'eigenvalues': [],
            'variance_explained': [],
            'cumulative_variance': [],
            'rotated_sum_squares': [],
            'rotated_variance_explained': [],
            'rotated_cumulative_variance': [],
            'loadings': [],
            'rotated_loadings': [],
            'communalities': [],
            'uniquenesses': [],
            'n_iterations': 0,
            'n_components': 0,
            'rotation_method': rotation_method,
            'extraction_method': extraction_method,
            'method': extraction_method,
            'var_names': [],
            'transformation_matrix': None
        }

def analyze_data_basic(df):
    results = {}

    desc = df.describe(include='all').T
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        desc.loc[numeric_cols, 'median'] = df[numeric_cols].median()
        desc.loc[numeric_cols, 'variance'] = df[numeric_cols].var()
        desc.loc[numeric_cols, 'skewness'] = df[numeric_cols].skew()
        desc.loc[numeric_cols, 'kurtosis'] = df[numeric_cols].kurtosis()
    
    desc_formatted = desc.applymap(format_brazilian_number)
    results['descritiva'] = desc_formatted.to_html(classes='table table-striped')

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        img_parts = []
        cols = numeric_df.columns.tolist()
        
        for i in range(0, len(cols), 5):
            batch = cols[i:i+5]
            plt.figure(figsize=(10, 6))
            numeric_df[batch].boxplot()
            plt.title(f"Boxplots das Variáveis ({i+1} a {min(i+5, len(cols))})")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            img_data = img.getvalue()
            img_url = f"data:image/png;base64,{base64.b64encode(img_data).decode('utf-8')}"
            plt.close()
            img_parts.append(img_url)
        
        results['boxplot'] = img_parts
    else:
        results['boxplot'] = None

    normalidade = []
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 3:
            continue
        try:
            shapiro_stat, shapiro_p = shapiro(data)
            ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
            
            def format_pvalue(x):
                if x == int(x):
                    return f"{int(x):,}".replace(",", ".")
                elif x < 0.001:
                    return "< 0,001"
                else:
                    return f"{x:,.3f}".replace(".", "X").replace(",", ".").replace("X", ",")
            
            normalidade.append({
                'variavel': col,
                'shapiro_stat': format_pvalue(shapiro_stat),
                'shapiro_p': format_pvalue(shapiro_p),
                'ks_stat': format_pvalue(ks_stat),
                'ks_p': format_pvalue(ks_p),
                'shapiro_normal': 'Sim' if shapiro_p > 0.05 else 'Não',
                'ks_normal': 'Sim' if ks_p > 0.05 else 'Não'
            })
        except Exception as e:
            normalidade.append({
                'variavel': col,
                'shapiro_stat': '-', 'shapiro_p': '-', 'ks_stat': '-', 'ks_p': '-',
                'shapiro_normal': f'Erro: {str(e)}', 'ks_normal': ''
            })
    results['normalidade'] = normalidade

    if not numeric_df.empty and numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        corr_formatted = corr.applymap(lambda x: f"{x:,.3f}".replace(".", "X").replace(",", ".").replace("X", ","))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, mask=mask, annot=corr.values, fmt=".3f", cmap='coolwarm', center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title("Matriz de Correlação (Pearson)")
        img_corr = io.BytesIO()
        plt.savefig(img_corr, format='png', bbox_inches='tight')
        img_corr.seek(0)
        img_data = img_corr.getvalue()
        corr_url = f"data:image/png;base64,{base64.b64encode(img_data).decode('utf-8')}"
        plt.close()
        results['correlacao_img'] = corr_url
        results['correlacao_table'] = corr_formatted.to_html(classes='table table-striped')
    else:
        results['correlacao_img'] = None
        results['correlacao_table'] = "<p>Não há variáveis numéricas suficientes para correlação.</p>"

    return results

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "❌ Nenhum arquivo enviado", 400

    file = request.files['file']
    if file.filename == '':
        return "❌ Arquivo sem nome", 400

    if file and file.filename.endswith('.csv'):
        try:
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None

            for encoding in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(
                        file,
                        sep=';',
                        decimal=',',
                        encoding=encoding,
                        on_bad_lines='skip',
                        header=0,
                        skipinitialspace=True
                    )
                    
                    valid_names = []
                    for col in df.columns:
                        col_str = str(col).strip()
                        if (col_str and 
                            not col_str.startswith('Unnamed:') and 
                            not col_str.replace('.', '').replace('-', '').isdigit()):
                            valid_names.append(True)
                        else:
                            valid_names.append(False)
                    
                    if sum(valid_names) >= len(valid_names) * 0.8:
                        df.columns = [str(col).strip() for col in df.columns]
                        break
                    else:
                        df = None
                        continue
                        
                except Exception as e:
                    continue

            if df is None:
                file.seek(0)
                df = pd.read_csv(
                    file,
                    sep=';',
                    decimal=',',
                    encoding='latin-1',
                    on_bad_lines='skip',
                    header=None
                )
                df.columns = [f"IPSA{i+1}" for i in range(df.shape[1])]

            if df.empty:
                return "❌ Arquivo vazio.", 400

            # Armazena na sessão (em memória)
            session['df'] = df.to_dict('records')
            session['columns'] = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            session['numeric_cols'] = numeric_cols

            return render_template('select_vars.html', columns=numeric_cols)

        except Exception as e:
            return f"<h2>Erro no upload</h2><pre>{str(e)}</pre>", 500

    return "❌ Formato inválido. Envie .csv", 400

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'df' not in session:
            return "❌ Dados não encontrados. Faça novo upload.", 400

        # Reconstrói o DataFrame da sessão
        df = pd.DataFrame(session['df'])
        selected_vars = request.form.getlist('selected_vars')
        extraction_method = request.form.get('extraction_method', 'kaiser')
        n_components_manual = request.form.get('n_components', type=int)
        rotation_method = request.form.get('rotation_method', 'varimax')

        if not selected_vars:
            return "❌ Selecione pelo menos uma variável numérica.", 400

        if len(selected_vars) < 2:
            return "❌ Selecione pelo menos 2 variáveis para PCA.", 400

        if extraction_method == 'manual' and n_components_manual is None:
            n_components_manual = 2
        if n_components_manual and n_components_manual < 1:
            n_components_manual = 1

        df_selected = df[selected_vars].copy()
        numeric_df = df_selected.select_dtypes(include=[np.number])

        if numeric_df.empty or numeric_df.shape[1] < 2:
            return "❌ Variáveis selecionadas inválidas.", 400

        kmo_result, bartlett_result = kmo_and_bartlett(numeric_df)
        pca_results = pca_analysis(
            numeric_df,
            extraction_method=extraction_method,
            n_components_manual=n_components_manual,
            rotation_method=rotation_method
        )

        results = {
            'descritiva': None,
            'boxplot': None,
            'normalidade': [],
            'correlacao_img': None,
            'correlacao_table': None,
            'kmo': kmo_result,
            'bartlett': bartlett_result,
            'pca': pca_results
        }

        results.update(analyze_data_basic(numeric_df))

        return render_template('result.html', results=results, columns=selected_vars)

    except Exception as e:
        return f"<h2>Erro na análise</h2><pre>{str(e)}</pre>", 500

if __name__ == '__main__':
    app.run()