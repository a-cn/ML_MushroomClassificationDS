# app/streamlit_app.py
# ---------------------------------------------------------
# Classificação de Cogumelos — PyCaret + XAI + Análise Completa
# Com aba "Entenda os Resultados" + Exportação HTML Único + Validação
# Persistência por aba (session_state) e sem reprocessar desnecessariamente
# ---------------------------------------------------------

import os
import io
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from dotenv import load_dotenv
from ydata_profiling import ProfileReport

# PyCaret (classificação)
from pycaret.classification import (
    setup, load_model, predict_model, plot_model, pull, get_config,
    compare_models, create_model
)

# Métricas adicionais
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder

# SHAP opcional
SHAP_OK = True
try:
    import shap
    import matplotlib.pyplot as plt
except Exception:
    SHAP_OK = False

# =========================================
# Setup
# =========================================
st.set_page_config(page_title="Classificação de Cogumelos — PyCaret", page_icon="🍄", layout="wide")
load_dotenv()

# Tema dark padrão
pio.templates.default = "plotly_dark"

# Constantes globais
RANDOM_STATE = 42

# Configurar logging
logging.basicConfig(
    filename="logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("App iniciado.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Caminhos padrão
DATASET_PATH = (PROJECT_ROOT / "doc" / "mushrooms.csv").resolve()
MODEL_BASENAME_DEFAULT = (PROJECT_ROOT / "models" / "modelo_classificacao_cogumelos").as_posix()

# =========================================
# Helpers de HTML/Export e Visualizações
# =========================================
def _fig_to_html_snippet(fig):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)

def _mpl_fig_to_base64(fig):
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=160)
    bio.seek(0)
    b64 = base64.b64encode(bio.read()).decode("utf-8")
    return f"<img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto;border:1px solid #2a2a2a;border-radius:8px'/>"

def _md_to_inline_html(md: str) -> str:
    if md is None:
        return ""
    html = md.replace("**", "<b>").replace("</b><b>", "")
    html = html.replace("\n", "<br>")
    return f"<p>{html}</p>"

# =========================================
# Explicações das métricas e gráficos (modo didático)
# =========================================
EXPLANATIONS_MD = {
    "auc_roc": """
**Curva ROC e AUC-ROC**  
- **ROC (Receiver Operating Characteristic)**: mostra a relação entre taxa de verdadeiros positivos (TPR) e taxa de falsos positivos (FPR) ao variar o threshold.  
- **AUC-ROC**: área sob a curva ROC; mede a capacidade do modelo de distinguir entre classes.  
- **Interpretação**: AUC = 0.5 (aleatório), AUC = 1.0 (perfeito), AUC > 0.8 (bom), AUC > 0.9 (excelente).
""",
    "ap_pr": """
**Curva Precisão × Revocação (PR) e AP (PR AUC)**  
- **Precisão (Precision)**: entre as predições de classe positiva, a fração que é realmente positiva.  
- **Revocação (Recall)**: entre as amostras realmente positivas, a fração que foi corretamente identificada.  
- **AP** (*Average Precision*): área sob a curva PR; resume o trade-off *precisão×revocação* ao varrer o *threshold*.  
- **Uso**: em bases desbalanceadas, **AP** é mais informativo que ROC-AUC.
""",
    "threshold": """
**Threshold (limiar)**  
Converte probabilidade em classe (0/1). *Threshold* maior → mais conservador (↑Precisão, ↓Recall).  
**Dica**: ajuste conforme o custo operacional de **falso positivo** vs **falso negativo**.
""",
    "cm": """
**Matriz de Confusão**  
- **TP**: amostras positivas previstas como positivas (acerto).  
- **FP**: amostras negativas previstas como positivas (falso positivo).  
- **TN**: amostras negativas previstas como negativas (acerto).  
- **FN**: amostras positivas previstas como negativas (falso negativo).  
**Foco**: olhe **FN** se perder classe positiva é crítico; **FP** se falso positivo custa caro.
""",
    "clf_report": """
**Relatório de Classificação**  
Inclui *precision*, *recall* e *f1-score* por classe, além de *support* (n° de exemplos por classe).  
O **F1** é a média harmônica entre *precision* e *recall*: F1 = 2 × (precision × recall) / (precision + recall).
""",
    "feat_importance": """
**Importância de Variáveis**  
Mede relevância de cada *feature* para o modelo, mas o significado varia conforme o método utilizado.
""",
    "feat_importance_model": """
**Importância de Features (Treinamento do Modelo)**  
Reflete a importância calculada durante o treinamento do estimador (ex.: ganho de impureza em árvores).  
- Árvores/Florestas: soma do ganho de impureza (Gini/Entropia) ao usar a feature nos splits.  
- Interpretação: valores maiores indicam que a feature ajudou mais a separar as classes ao treinar.  
- Cuidados: pode favorecer variáveis com mais níveis; correlações podem “dividir” importância entre variáveis similares.
""",
    "feat_importance_shap": """
**Importância de Features por SHAP**  
Baseada na média do valor absoluto dos efeitos SHAP por feature (impacto médio no log-odds/probabilidade).  
- Quanto maior, maior o efeito médio da feature nas predições (independe de sinal).  
- Direção (aumentar/diminuir probabilidade) é vista nos gráficos SHAP detalhados; aqui o foco é a força média.  
- Vantagens: consistente entre modelos; lida melhor com correlações do que importâncias por impureza.  
- Use em conjunto com os plots SHAP globais/locais para entender sinais e faixas de valores.
""",
    "learning_curve": """
**Curva de Aprendizado**  
Mostra como a performance (acurácia/precisão) evolui com o tamanho do dataset de treino.  
**Interpretação**: 
- Curva ascendente = modelo pode melhorar com mais dados
- Curva achatada = modelo já convergiu ou está limitado por viés
""",
    "calibration": """
**Calibração**  
Ajuste que aproxima as probabilidades do modelo da frequência real observada.  
**Probabilidades calibradas** facilitam escolha de *threshold* alinhada ao risco real.
""",
    "manifold": """
**Redução de Dimensionalidade (PCA/t-SNE)**  
Visualiza como as amostras se agrupam no espaço de features reduzido.  
- **PCA**: preserva variância máxima
- **t-SNE**: preserva estruturas locais (mais visual)
""",
    "val_class_dist": """
**Distribuição das Classes Preditas**  
Mostra quantas amostras foram classificadas em cada classe.  
- Picos desbalanceados podem indicar viés do modelo.  
- Compare com a distribuição esperada do problema para verificar se há tendência excessiva a uma classe.
""",
    "val_prob_dist": """
**Distribuição das Probabilidades**  
Histograma das probabilidades preditas para a classe positiva (venenoso).  
- Picos perto de 0 ou 1 indicam predições mais confiantes; concentração em torno de 0.5 indica incerteza.  
- Útil para escolher thresholds e avaliar calibragem das probabilidades.
""",
    "eda": """
**Análise Exploratória de Dados (EDA)**  
Exame inicial dos dados para entender padrões, distribuições e relações.  
Inclui estatísticas descritivas, visualizações e detecção de outliers.
""",
    "shap_global": """
**SHAP — Global (summary plot)**  
Cada ponto é o efeito de uma amostra em uma *feature*:  
- Posição no eixo X = impacto no *logit* (ou probabilidade).  
- Cor (geralmente) = valor da *feature* (baixo→alto).  
Dá visão de direção e força média do efeito de cada variável.
""",
    "shap_local_pos": """
**SHAP — Classe Positiva (Venenoso)**  
Este sumário foca amostras previstas/rotuladas como **venenosas**.  
- **Eixo X**: impacto SHAP médio (absoluto) por feature; quanto mais à direita, maior a influência para a classe venenosa.  
- **Cores**: valores altos da feature (vermelho) vs baixos (azul) e como empurram a predição para **venenoso**.  
- **Leitura**: barras mais extensas e vermelhas à direita indicam que valores altos dessa feature aumentam a probabilidade de ser venenoso.
""",
    "shap_local_neg": """
**SHAP — Classe Negativa (Comestível)**  
Este sumário foca amostras previstas/rotuladas como **comestíveis**.  
- **Eixo X**: impacto SHAP médio (absoluto) por feature; mais à esquerda/direita indica força da influência, aqui direcionada à classe comestível.  
- **Cores**: valores altos (vermelho) e baixos (azul) e sua relação em puxar a predição para **comestível**.  
- **Leitura**: barras proeminentes onde tons azuis dominam à direita sugerem que valores baixos daquela feature favorecem a classe comestível.
""",
    "validation": """
**Validação com Dados Sem Rótulo**  
Teste o modelo com novos dados para verificar sua performance em cenários reais.  
- **Distribuição das Predições**: Verifica se o modelo está balanceado
- **Análise de Confiança**: Identifica predições com alta/baixa confiança
- **Detecção de Drift**: Compara distribuições entre treino e validação
- **Validação Cruzada**: Testa robustez com diferentes amostras
""",
    "pipeline": """
**Pipeline Plot**  
Mostra o fluxo de processamento dos dados através do pipeline de machine learning.  
- **Transformações**: Pré-processamento, normalização, encoding
- **Seleção de Features**: Quais variáveis foram selecionadas
- **Modelo**: Algoritmo de classificação utilizado
- **Validação**: Como os dados foram divididos para treino/teste
""",
    "error": """
**Prediction Error**  
Análise dos erros de predição do modelo.  
- **Resíduos**: Diferença entre valores reais e preditos
- **Distribuição dos Erros**: Padrões nos erros de classificação
- **Outliers**: Predições com maior erro
- **Bias-Variance**: Identifica se o modelo tem viés ou variância alta
""",
    "tree": """
**Decision Tree**  
Visualização da árvore de decisão do modelo.  
- **Nós**: Pontos de decisão baseados em features
- **Ramos**: Condições que levam a diferentes classificações
- **Folhas**: Classes finais (comestível/venenoso)
- **Profundidade**: Complexidade da árvore
- **Importância**: Quais features são mais decisivas
"""
}

def render_expander_md(title: str, key: str):
    with st.expander(f"Como interpretar — {title}"):
        st.markdown(EXPLANATIONS_MD.get(key, "_(sem conteúdo)_"))

# =========================================
# Funções de visualização avançadas
# =========================================
def plot_pr_curve_advanced(y_true, y_proba, threshold=None):
    """Curva Precision-Recall com threshold destacado"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, 
        mode="lines", 
        name=f"PR Curve (AP={ap:.4f})",
        line=dict(color="#1f77b4", width=3)
    ))
    
    if threshold is not None and len(thresholds) > 0:
        idx = np.argmin(np.abs(thresholds - threshold))
        fig.add_trace(go.Scatter(
            x=[recall[idx]], y=[precision[idx]], 
            mode="markers",
            marker=dict(size=12, color="red", symbol="diamond"),
            name=f"Threshold={threshold:.3f}"
        ))
    
    fig.update_layout(
        title="Curva Precision-Recall",
        xaxis_title="Recall (Sensibilidade)",
        yaxis_title="Precision (Precisão)",
        height=500,
        showlegend=True
    )
    return fig

def plot_roc_curve_advanced(y_true, y_proba, threshold=None):
    """Curva ROC com threshold destacado"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, 
        mode="lines", 
        name=f"ROC Curve (AUC={auc:.4f})",
        line=dict(color="#1f77b4", width=3)
    ))
    
    # Linha diagonal (classificador aleatório)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], 
        mode="lines", 
        name="Random Classifier",
        line=dict(color="red", width=2, dash="dash")
    ))
    
    if threshold is not None and len(thresholds) > 0:
        idx = np.argmin(np.abs(thresholds - threshold))
        fig.add_trace(go.Scatter(
            x=[fpr[idx]], y=[tpr[idx]], 
            mode="markers",
            marker=dict(size=12, color="green", symbol="diamond"),
            name=f"Threshold={threshold:.3f}"
        ))
    
    fig.update_layout(
        title="Curva ROC",
        xaxis_title="Taxa de Falsos Positivos (1 - Especificidade)",
        yaxis_title="Taxa de Verdadeiros Positivos (Sensibilidade)",
        height=500,
        showlegend=True
    )
    return fig

def plot_confusion_matrix_advanced(y_true, y_pred, class_names=None):
    """Matriz de confusão interativa"""
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f"Classe {i}" for i in range(len(cm))]
    
    # Normalizar para percentuais
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=class_names,
        y=class_names,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale="Reds",
        showscale=True,
        colorbar=dict(title="Frequência")
    ))
    
    fig.update_layout(
        title="Matriz de Confusão",
        xaxis_title="Predição",
        yaxis_title="Valor Real",
        height=400
    )
    
    return fig, cm

def plot_feature_importance_advanced(importance_dict, top_n=20):
    """Gráfico de importância de features"""
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_features)
    
    fig = go.Figure(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Top Features Mais Importantes",
        xaxis_title="Importância",
        yaxis_title="Feature",
        height=max(400, len(features) * 25)
    )
    
    return fig

def plot_calibration_curve_advanced(y_true, y_proba, n_bins=10):
    """Curva de calibração"""
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins
    )
    
    fig = go.Figure()
    
    # Curva de calibração
    fig.add_trace(go.Scatter(
        x=mean_predicted_value, y=fraction_of_positives,
        mode='lines+markers',
        name='Modelo',
        line=dict(color='blue', width=3)
    ))
    
    # Linha de calibração perfeita
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Calibração Perfeita',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Curva de Calibração",
        xaxis_title="Probabilidade Média Predita",
        yaxis_title="Fração de Positivos",
        height=500
    )
    
    return fig

# =========================================
# Funções de interpretação do modelo
# =========================================
def generate_shap_analysis(model, X_sample, y_sample=None):
    """Gera análise SHAP para interpretação do modelo"""
    if not SHAP_OK:
        return None, None, None
    
    try:
        # Criar explainer SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Processar valores SHAP
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_neg = shap_values[0]
            shap_values_pos = shap_values[1]
        else:
            shap_values_pos = shap_values
            shap_values_neg = -shap_values_pos
        
        # Summary plot global
        fig_global = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_pos, X_sample, show=False)
        plt.title("SHAP Summary Plot - Global")
        plt.tight_layout()
        shap_global_img = _mpl_fig_to_base64(fig_global)
        plt.close(fig_global)
        
        # Análise por classe se y_sample disponível
        shap_pos_img = None
        shap_neg_img = None
        if y_sample is not None:
            y_array = np.array(y_sample)
            idx_pos = np.where(y_array == 1)[0]
            idx_neg = np.where(y_array == 0)[0]
            
            if len(idx_pos) > 0:
                fig_pos = plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values_pos[idx_pos],
                    X_sample.iloc[idx_pos],
                    show=False
                )
                plt.title("SHAP Summary Plot - Classe Positiva (Venenoso)")
                plt.tight_layout()
                shap_pos_img = _mpl_fig_to_base64(fig_pos)
                plt.close(fig_pos)
            
            if len(idx_neg) > 0:
                fig_neg = plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values_neg[idx_neg],
                    X_sample.iloc[idx_neg],
                    show=False
                )
                plt.title("SHAP Summary Plot - Classe Negativa (Comestível)")
                plt.tight_layout()
                shap_neg_img = _mpl_fig_to_base64(fig_neg)
                plt.close(fig_neg)
        
        return shap_global_img, shap_pos_img, shap_neg_img
        
    except Exception as e:
        st.warning(f"Erro ao gerar análise SHAP: {e}")
        return None, None, None

def plot_feature_importance_shap(shap_values, feature_names, top_n=20):
    """Gráfico de importância de features baseado em SHAP"""
    if not SHAP_OK:
        return None
    
    try:
        # Calcular importância média absoluta
        importance = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True).tail(top_n)
        
        fig = go.Figure(go.Bar(
            x=feature_importance['importance'],
            y=feature_importance['feature'],
            orientation='h',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Top Features por Importância SHAP",
            xaxis_title="Importância SHAP (média absoluta)",
            yaxis_title="Feature",
            height=max(400, len(feature_importance) * 25)
        )
        
        return fig
    except Exception as e:
        st.warning(f"Erro ao gerar gráfico de importância SHAP: {e}")
        return None

# =========================================
# Funções de validação e threshold
# =========================================
def choose_threshold_by_precision(y_true, probs, target_precision=0.9):
    """Escolhe threshold baseado em precisão alvo"""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    chosen = None
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= target_precision:
            chosen = (t, p, r)
            break
    if chosen is None:
        f1s = [(2*p*r)/(p+r) if (p+r)>0 else 0 for p,r,_ in zip(precision[:-1], recall[:-1], thresholds)]
        idx = int(np.argmax(f1s))
        return float(thresholds[idx]), float(precision[idx]), float(recall[idx]), True
    t, p, r = chosen
    return float(t), float(p), float(r), False

def plot_threshold_analysis(y_true, y_proba, threshold=None):
    """Análise de threshold com métricas"""
    if threshold is None:
        threshold, precision, recall, _ = choose_threshold_by_precision(y_true, y_proba)
    
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Criar gráfico de métricas vs threshold
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1s = []
    
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_true, y_pred_t, zero_division=0))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precisions, mode='lines', name='Precision', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls, mode='lines', name='Recall', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=thresholds, y=f1s, mode='lines', name='F1-Score', line=dict(color='green')))
    
    # Marcar threshold escolhido
    fig.add_vline(x=threshold, line_dash="dash", line_color="black", 
                  annotation_text=f"Threshold={threshold:.3f}")
    
    fig.update_layout(
        title="Métricas vs Threshold",
        xaxis_title="Threshold",
        yaxis_title="Score",
        height=500
    )
    
    return fig, {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# =========================================
# Carregamento de dados (cache)
# =========================================
@st.cache_data(show_spinner=False)
def load_mushrooms_csv(path: Path = None, uploaded_file = None) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    elif path is not None and path.exists():
        df = pd.read_csv(path)
        return df
    else:
        # Fallback para o caminho padrão
        default_path = DATASET_PATH
        if not default_path.exists():
            raise FileNotFoundError(f"CSV não encontrado em: {default_path}")
        df = pd.read_csv(default_path)
        return df

@st.cache_resource(show_spinner=True)
def load_pycaret_model(model_base_no_pkl: str):
    pkl_path = Path(model_base_no_pkl).with_suffix(".pkl")
    if not pkl_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {pkl_path}")
    return load_model(model_base_no_pkl)

def load_model_from_upload(uploaded_model):
    """Carrega modelo a partir de arquivo uploadado"""
    if uploaded_model is None:
        return None
    # Salvar temporariamente o arquivo uploadado
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        tmp_file.write(uploaded_model.getvalue())
        tmp_path = tmp_file.name
    try:
        # Remover a extensão .pkl do caminho antes de passar para load_model
        model_path = tmp_path.replace('.pkl', '')
        model = load_model(model_path)
        return model
    finally:
        # Limpar arquivo temporário
        os.unlink(tmp_path)

def check_model_available():
    """Verifica se o modelo está disponível"""
    try:
        return uploaded_model is not None
    except:
        return False

def get_available_model():
    """Retorna o modelo disponível e sua fonte"""
    if st.session_state.training_completed and st.session_state.trained_model is not None:
        return st.session_state.trained_model, "treinado na sessão"
    elif check_model_available():
        return load_model_from_upload(uploaded_model), "via upload"
    return None, None

# =========================================
# UI — Sidebar e Tabs
# =========================================
st.title("🍄 Classificação de Cogumelos — Treinamento + Upload + XAI + Análise Completa")

# Sidebar
st.sidebar.header("⚙️ Configurações")

# 1. Fonte de dados (com padrão do seu projeto lembrado)
default_local = os.environ.get("MUSHROOM_LOCAL", "./doc/mushrooms.csv")
fonte = st.sidebar.radio(
    "Fonte de dados",
    options=["Local (ENV/Path)", "Upload (CSV)"],
    index=0
)

# Upload de CSV aparece logo abaixo das opções radio
uploaded_file = None
if fonte == "Upload (CSV)":
    uploaded_file = st.sidebar.file_uploader("Envie um CSV", type=["csv"])
else:
    local_path = st.sidebar.text_input("Caminho local do dataset", value=default_local)

# 2. Definição de variável resposta (fixa em "class")
target_col = "class"
st.sidebar.text_input("Coluna alvo (target)", value="class", disabled=True)

# 3. Configurações de treinamento
st.sidebar.subheader("Treinamento")
acc_goal = st.sidebar.slider("🎯 Alvo de Precisão (threshold)", 0.5, 1.0, 0.8, 0.01,
                             help="Meta desejada de *Accuracy* para destacar se o modelo atingiu a precisão mínima.")
folds = st.sidebar.slider("🔁 K-Folds (cross-validation)", min_value=3, max_value=20, value=10, step=1,
                         help="Número de divisões para validação cruzada. Valores maiores reduzem variância mas aumentam tempo de processamento.")
st.sidebar.caption("💡 Dica: ajuste os Folds para controle de variância; use o Alvo de Precisão para checar se o modelo atingiu sua meta.")

# 4. Avisos
st.sidebar.markdown("---")
st.sidebar.warning("⚠️ **AVISO**: Este modelo é apenas para fins educacionais. NUNCA use para determinar se um cogumelo é comestível na vida real!")

# Informação sobre funcionalidades
st.sidebar.info("ℹ️ **Funcionalidades**: Este aplicativo permite treinar novos modelos (aba Treino & Métricas) ou carregar modelos existentes (.pkl) para análise.")

# 5. Possibilidade de envio de modelo em pkl
st.sidebar.markdown("---")
st.sidebar.header("🤖 Modelo")
uploaded_model = st.sidebar.file_uploader(
    "Envie um modelo (.pkl)",
    type=["pkl"],
    help="Selecione um arquivo de modelo PyCaret (.pkl) para usar nas abas de análise"
)
model_basename = None

# Abas
tab_explain, tab_eda, tab_training, tab_evaluate, tab_interpret, tab_validate, tab_predict, tab_export = st.tabs([
    "Orientações",
    "EDA (ydata-profiling)",
    "Treino & Métricas",
    "Evaluate Model",
    "Interpretação (SHAP)",
    "Validação (Dados Sem Rótulo)",
    "Predição Interativa",
    "Exportar Relatório"
])

# =========================================
# Aba 2 — EDA (ydata-profiling)
# =========================================
with tab_eda:
    st.subheader("Análise Exploratória de Dados (ydata-profiling)")
    try:
        # Determinar caminho dos dados
        if fonte == "Upload (CSV)" and uploaded_file is not None:
            df = load_mushrooms_csv(uploaded_file=uploaded_file)
        else:
            data_path = Path(local_path) if local_path else DATASET_PATH
            df = load_mushrooms_csv(path=data_path)
        st.write("Amostra do dataset:", df.head())
        
        if st.button("🔍 Gerar relatório (profiling)", use_container_width=False):
            with st.spinner("Gerando relatório..."):
                profile = ProfileReport(df, title="Mushrooms — Profiling", minimal=False)
                html_str = profile.to_html()
            st.success("Relatório gerado!")
        
        # Relatório ocupa toda a largura (fora das colunas)
        if 'html_str' in locals():
            st.components.v1.html(html_str, height=900, scrolling=True)

            # Botão de download (sem salvar automaticamente)
        if 'html_str' in locals():
            st.download_button(
                "💾 Baixar relatório (HTML)", 
                data=html_str.encode("utf-8"),
                file_name="pandas_profiling_mushrooms.html",
                mime="text/html", 
                use_container_width=False,
                help="Clique para baixar o relatório de profiling"
            )
    except Exception as e:
        st.error(f"Falha no EDA: {e}")

# =========================================
# Aba 3 — Treino & Métricas
# =========================================
with tab_training:
    st.subheader("Treino & Métricas")
    st.markdown("Clique para **carregar** o dataset, **treinar** e **visualizar** os resultados principais (Leaderboard e plots).")

    # Estados em sessão para treinamento
    if "training_data" not in st.session_state:
        st.session_state.training_data = None
    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None
    if "training_leaderboard" not in st.session_state:
        st.session_state.training_leaderboard = None
    if "training_setup_params" not in st.session_state:
        st.session_state.training_setup_params = {}
    if "training_setup_info" not in st.session_state:
        st.session_state.training_setup_info = None
    if "training_model_info" not in st.session_state:
        st.session_state.training_model_info = None
    if "training_completed" not in st.session_state:
        st.session_state.training_completed = False
    if "training_plots" not in st.session_state:
        st.session_state.training_plots = {}
    if "training_accuracy_goal_met" not in st.session_state:
        st.session_state.training_accuracy_goal_met = None


    # Botão para executar o treinamento
    if st.button("🚀 Carregar, Treinar e Visualizar", use_container_width=False):
        try:
            # Carregar dados
            if fonte == "Upload (CSV)" and uploaded_file is not None:
                df = load_mushrooms_csv(uploaded_file=uploaded_file)
            else:
                data_path = Path(local_path) if local_path else DATASET_PATH
                df = load_mushrooms_csv(path=data_path)
            st.session_state.training_data = df.copy()

            # Features
            if target_col in df.columns:
                feature_cols = [c for c in df.columns if c != target_col]
            else:
                st.warning(f"Target '{target_col}' não encontrado no dataset. Verifique o nome da coluna.")
                st.stop()

            # === SETUP EXATAMENTE COMO NO SEU NOTEBOOK ===
            setup_params = dict(
                data=df,
                target=target_col,
                session_id=1,
                feature_selection=True,
                feature_selection_method='classic',
                remove_multicollinearity=True,
                multicollinearity_threshold=0.9,
                low_variance_threshold=0.05,
                fold=folds,  # controlado pela sidebar
            )
            st.session_state.training_setup_params = {k: v for k, v in setup_params.items() if k != "data"}

            with st.spinner("Executando setup() com seus parâmetros..."):
                s = setup(**setup_params)

            # Captura e salva a tabela resultante do setup
            setup_info = pull()
            st.session_state.training_setup_info = setup_info
            st.success("Setup concluído ✅")

            # Leaderboard por AUC (compare_models) — TABELA
            with st.spinner("Comparando modelos (ordenando por AUC)..."):
                _ = compare_models(sort='AUC')        # só para formar o leaderboard ordenado
                leaderboard = pull()                  # captura a tabela
            st.session_state.training_leaderboard = leaderboard

            # === MODELO PRINCIPAL: DECISION TREE ===
            with st.spinner("Criando modelo Decision Tree (dt)..."):
                dt_model = create_model("dt")  # classificador principal
            
            # Captura e salva a tabela resultante do create_model
            model_info = pull()
            st.session_state.training_model_info = model_info
            
            st.session_state.trained_model = dt_model

            # Marca que o treinamento foi concluído
            st.session_state.training_completed = True
            st.success("Treino concluído ✅")
            
            # Checagem de meta de precisão (se houver coluna de Accuracy)
            acc_col = None
            for candidate in ["Accuracy", "Accuracy "]:
                if candidate in leaderboard.columns:
                    acc_col = candidate
                    break
            if acc_col is not None:
                top_acc = leaderboard.iloc[0][acc_col]
                if pd.notnull(top_acc) and float(top_acc) >= acc_goal:
                    st.session_state.training_accuracy_goal_met = True
                    st.success(f"🎉 Meta de precisão atingida! {float(top_acc):.4f} ≥ {acc_goal:.4f}")
                else:
                    st.session_state.training_accuracy_goal_met = False
                    st.warning(f"Meta de precisão **não** atingida. Top Accuracy: {float(top_acc):.4f} < {acc_goal:.4f}")
            else:
                st.session_state.training_accuracy_goal_met = None
                st.info("Coluna de Accuracy não localizada no leaderboard.")

        except Exception as e:
            st.error(f"Erro durante o treinamento: {e}")
            st.exception(e)

    # Exibir resultados se o treinamento foi concluído
    if st.session_state.training_completed and st.session_state.training_leaderboard is not None:
        st.markdown("---")
        # Informações do Setup
        if st.session_state.training_setup_info is not None:
            st.markdown("#### 📋 Informações do Setup")
            st.dataframe(st.session_state.training_setup_info, use_container_width=True)
            st.markdown("---")
        
        # Leaderboard
        st.markdown("#### 📊 Leaderboard (compare_models | sort='AUC')")
        st.dataframe(st.session_state.training_leaderboard, use_container_width=True)
        st.markdown("---")
        
        # Informações do Modelo
        if st.session_state.training_model_info is not None:
            st.markdown("#### 🌳 Informações do Modelo Decision Tree")
            st.dataframe(st.session_state.training_model_info, use_container_width=True)

        # Botão para limpar resultados
        if st.button("🗑️ Limpar Resultados", use_container_width=False, key="btn_clear_eval_results"):
            st.session_state.training_completed = False
            st.session_state.training_leaderboard = None
            st.session_state.trained_model = None
            st.session_state.training_plots = {}
            st.session_state.training_setup_info = None
            st.session_state.training_model_info = None
            st.session_state.training_accuracy_goal_met = None
            st.rerun()

# =========================================
# Aba 4 — Evaluate Model
# =========================================
with tab_evaluate:

    # Verificar se há modelo treinado disponível
    if st.session_state.training_completed and st.session_state.trained_model is not None:
        model_source_type = "treinado na sessão"
        st.info(f"ℹ️ Usando o modelo {model_source_type}")
        
        # Exibir métricas principais do leaderboard
        if st.session_state.training_leaderboard is not None:
            st.markdown("### 📊 Métricas Principais do Modelo")
            
            # Obter métricas do primeiro modelo (melhor) do leaderboard
            leaderboard = st.session_state.training_leaderboard
            if len(leaderboard) > 0:
                # Buscar colunas de métricas disponíveis
                available_metrics = {}
                
                # Mapear nomes de colunas possíveis para métricas
                metric_mapping = {
                    'Accuracy': 'Acurácia',
                    'Accuracy ': 'Acurácia',
                    'Precision': 'Precisão', 
                    'Precision ': 'Precisão',
                    'Recall': 'Recall',
                    'Recall ': 'Recall',
                    'F1': 'F1-Score',
                    'F1 ': 'F1-Score',
                    'AUC': 'AUC-ROC',
                    'AUC ': 'AUC-ROC',
                    'AP': 'AP (PR-AUC)',
                    'AP ': 'AP (PR-AUC)'
                }
                
                # Encontrar métricas disponíveis no leaderboard
                for col in leaderboard.columns:
                    if col in metric_mapping:
                        value = leaderboard.iloc[0][col]
                        if pd.notnull(value):
                            available_metrics[metric_mapping[col]] = float(value)
                
                # Exibir métricas em cartões
                if available_metrics:
                    # Organizar em 4 colunas para as métricas principais
                    metric_keys = list(available_metrics.keys())
                    cols = st.columns(4)
                    
                    for i, (metric_name, value) in enumerate(available_metrics.items()):
                        if i < 4:  # Primeiras 4 métricas em 4 colunas
                            with cols[i]:
                                st.metric(metric_name, f"{value:.4f}")
                    
                    # Métricas adicionais em 2 colunas se houver mais de 4
                    if len(available_metrics) > 4:
                        additional_cols = st.columns(2)
                        for i, (metric_name, value) in enumerate(available_metrics.items()):
                            if i >= 4:  # Métricas adicionais
                                col_idx = (i - 4) % 2
                                with additional_cols[col_idx]:
                                    st.metric(metric_name, f"{value:.4f}")
                else:
                    st.info("Métricas não disponíveis no leaderboard")
            else:
                st.info("Leaderboard vazio")
            
            st.markdown("---")
        
        # Plots do modelo usando evaluate_model
        st.markdown("### 📈 Avaliação do Modelo (evaluate_model)")
        
        # Lista de plots, na ordem solicitada
        plots_requested = [
            ("Pipeline Plot", "pipeline", "pipeline"),
            ("Feature Importance", "feature", "feat_importance_model"),
            ("Learning Curve", "learning", "learning_curve"),
            ("Validation Curve", "vc", "validation"),
            ("AUC", "auc", "auc_roc"),
            ("Threshold", "threshold", "threshold"),
            ("Confusion Matrix", "confusion_matrix", "cm"),
            ("Class Report", "class_report", "clf_report"),
            ("Precision Recall", "pr", "ap_pr"),
            ("Prediction Error", "error", "error"),
            ("Calibration Curve", "calibration", "calibration"),
            ("Manifold Learning", "manifold", "manifold"),
            ("Decision Tree", "tree", "tree"),
        ]

        # Exibir plots em uma única coluna
        # Cache leve para evitar recomputar predições várias vezes em fallbacks
        _fallback_preds = None
        _fallback_y_true_numeric = None
        _fallback_y_pred_numeric = None
        _fallback_y_proba = None
        total_plots = len(plots_requested)
        for i, (label, plt_name, explanation_key) in enumerate(plots_requested):
            try:
                st.markdown(f"#### {label}")
                # Gera o plot do PyCaret e exibe diretamente
                plot_model(st.session_state.trained_model, plot=plt_name, display_format='streamlit')
                
                # Marca como plot gerado mas não salvo
                st.session_state.training_plots[label] = None
                
            except Exception as e:
                st.warning(f"Plot '{label}' não pôde ser gerado: {e}")
                # ===== FALLBACKS ESPECÍFICOS =====
                try:
                    # Pipeline Plot — fallback com Graphviz (DOT via st.graphviz_chart)
                    if plt_name == "pipeline":
                        dot_lines = [
                            "digraph G {",
                            "rankdir=LR;",
                            "node [shape=box, style=filled, color=gray30, fillcolor=gray15, fontcolor=white];",
                            "Dados [label=\"Dados\"];",
                            "Pre [label=\"Pré-processamento\"];",
                            "FS [label=\"Seleção de Features\"];",
                            "Model [label=\"Modelo: Decision Tree\"];",
                            f"CV [label=\"Validação (K-Fold={folds})\"];",
                            "Dados -> Pre -> FS -> Model -> CV;",
                            "}"
                        ]
                        st.graphviz_chart("\n".join(dot_lines))

                    # Feature Importance — fallback com Plotly
                    if plt_name == "feature":
                        model = st.session_state.trained_model
                        if hasattr(model, "feature_importances_"):
                            try:
                                feature_names = get_config("X_train_transformed").columns.tolist()
                            except Exception:
                                feature_names = [f"f{i}" for i in range(len(getattr(model, "feature_importances_", [])))]
                            importances = getattr(model, "feature_importances_", None)
                            if importances is not None and len(importances) == len(feature_names):
                                importance_dict = dict(zip(feature_names, importances))
                                fig = plot_feature_importance_advanced(importance_dict, top_n=20)
                                st.plotly_chart(fig, use_container_width=True)
                        st.info("Importâncias de features indisponíveis para este modelo.")

                    # Calibration Curve — fallback com nossa função
                    if plt_name == "calibration":
                        try:
                            if _fallback_preds is None:
                                if st.session_state.training_data is None:
                                    st.info("Dados de treino indisponíveis para gerar calibração.")
                                else:
                                    df_train = st.session_state.training_data.copy()
                                    _fallback_preds = predict_model(st.session_state.trained_model, data=df_train)
                                    from sklearn.preprocessing import LabelEncoder as _LE
                                    _le = _LE()
                                    _fallback_y_true_numeric = _le.fit_transform(df_train[target_col])
                                    _fallback_y_pred_numeric = _le.transform(_fallback_preds['prediction_label'])
                                    _fallback_y_proba = _fallback_preds['prediction_score']
                            if _fallback_y_true_numeric is not None and _fallback_y_proba is not None:
                                fig = plot_calibration_curve_advanced(_fallback_y_true_numeric, _fallback_y_proba)
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as _e_cal:
                            st.info(f"Fallback de calibração também falhou: {_e_cal}")

                    # Decision Tree — fallback com matplotlib.plot_tree
                    if plt_name == "tree":
                        try:
                            import matplotlib.pyplot as _plt
                            from sklearn import tree as _sk_tree
                            model = st.session_state.trained_model
                            # Tentar obter nomes das features
                            try:
                                feature_names = get_config("X_train_transformed").columns.tolist()
                            except Exception:
                                feature_names = None
                            # Nomes de classes
                            if st.session_state.training_data is not None and target_col in st.session_state.training_data.columns:
                                classes = sorted(st.session_state.training_data[target_col].unique().tolist())
                            else:
                                classes = None
                            fig = _plt.figure(figsize=(14, 8))
                            _sk_tree.plot_tree(
                                model,
                                feature_names=feature_names,
                                class_names=classes,
                                filled=True,
                                rounded=True,
                                fontsize=8
                            )
                            _plt.tight_layout()
                            st.pyplot(fig, clear_figure=True)
                        except Exception as _e_tree:
                            st.info(f"Fallback de Decision Tree também falhou: {_e_tree}")

                except Exception as _e_fb:
                    st.info(f"Fallback não disponível para '{label}': {_e_fb}")
            
            # ===== RENDERIZAÇÃO ADICIONAL: garantir exibição dos 4 solicitados =====
            try:
                if plt_name == "pipeline":
                    dot_lines = [
                        "digraph G {",
                        "rankdir=LR;",
                        "node [shape=box, style=filled, color=gray30, fillcolor=gray15, fontcolor=white];",
                        "Dados [label=\"Dados\"];",
                        "Pre [label=\"Pré-processamento\"];",
                        "FS [label=\"Seleção de Features\"];",
                        "Model [label=\"Modelo: Decision Tree\"];",
                        f"CV [label=\"Validação (K-Fold={folds})\"];",
                        "Dados -> Pre -> FS -> Model -> CV;",
                        "}"
                    ]
                    st.graphviz_chart("\n".join(dot_lines))
                elif plt_name == "feature":
                    model = st.session_state.trained_model
                    if hasattr(model, "feature_importances_"):
                        try:
                            feature_names = get_config("X_train_transformed").columns.tolist()
                        except Exception:
                            feature_names = [f"f{i}" for i in range(len(getattr(model, "feature_importances_", [])))]
                        importances = getattr(model, "feature_importances_", None)
                        if importances is not None and len(importances) == len(feature_names):
                            importance_dict = dict(zip(feature_names, importances))
                            fig = plot_feature_importance_advanced(importance_dict, top_n=20)
                            st.plotly_chart(fig, use_container_width=True)
                elif plt_name == "calibration":
                    if _fallback_preds is None and st.session_state.training_data is not None:
                        df_train = st.session_state.training_data.copy()
                        _fallback_preds = predict_model(st.session_state.trained_model, data=df_train)
                        from sklearn.preprocessing import LabelEncoder as _LE
                        _le = _LE()
                        _fallback_y_true_numeric = _le.fit_transform(df_train[target_col])
                        _fallback_y_pred_numeric = _le.transform(_fallback_preds['prediction_label'])
                        _fallback_y_proba = _fallback_preds['prediction_score']
                    if _fallback_y_true_numeric is not None and _fallback_y_proba is not None:
                        fig = plot_calibration_curve_advanced(_fallback_y_true_numeric, _fallback_y_proba)
                        st.plotly_chart(fig, use_container_width=True)
                elif plt_name == "tree":
                    try:
                        import matplotlib.pyplot as _plt
                        from sklearn import tree as _sk_tree
                        model = st.session_state.trained_model
                        try:
                            feature_names = get_config("X_train_transformed").columns.tolist()
                        except Exception:
                            feature_names = None
                        if st.session_state.training_data is not None and target_col in st.session_state.training_data.columns:
                            classes = sorted(st.session_state.training_data[target_col].unique().tolist())
                        else:
                            classes = None
                        fig = _plt.figure(figsize=(14, 8))
                        _sk_tree.plot_tree(
                            model,
                            feature_names=feature_names,
                            class_names=classes,
                            filled=True,
                            rounded=True,
                            fontsize=8
                        )
                        _plt.tight_layout()
                        st.pyplot(fig, clear_figure=True)
                    except Exception:
                        pass
            except Exception:
                pass

            # Explicação sempre após o(s) gráfico(s)
            with st.expander(f"Como interpretar — {label}"):
                st.markdown(EXPLANATIONS_MD.get(explanation_key, "_(Explicação não disponível para este gráfico)_"))
            # Linha divisória entre gráficos (exceto após o último)
            if i < total_plots - 1:
                st.markdown("---")
            
    else:
        st.warning("⚠️ Nenhum modelo treinado disponível!")
        st.info("💡 Treine um modelo na aba 'Treino & Métricas' primeiro.")

# =========================================
# Aba 5 — Interpretação do Modelo (SHAP)
# =========================================
with tab_interpret:
    st.subheader("Interpretação do Modelo (SHAP)")
    
    # Verificar se há modelo disponível (treinado ou pré-treinado)
    model_to_use, model_source_type = get_available_model()
    
    if not SHAP_OK:
        st.warning("⚠️ SHAP não está disponível. Instale com: `pip install shap matplotlib`")
    elif model_to_use is None:
        st.error("❌ Nenhum modelo disponível!")
        st.info("💡 Treine um modelo na aba 'Treino & Métricas' ou carregue um modelo (.pkl) na sidebar.")
    else:
        st.info(f"ℹ️ Usando o modelo {model_source_type}")
        try:
            # Carregar dados
            if fonte == "Upload (CSV)" and uploaded_file is not None:
                df = load_mushrooms_csv(uploaded_file=uploaded_file)
            else:
                data_path = Path(local_path) if local_path else DATASET_PATH
                df = load_mushrooms_csv(path=data_path)
            if target_col not in df.columns:
                st.error(f"A coluna alvo '{target_col}' não existe no CSV.")
            else:
                # Setup mínimo apenas para obter configurações necessárias
                _ = setup(
                    data=df,
                    target=target_col,
                    session_id=1,
                    feature_selection=True,
                    feature_selection_method="classic",
                    remove_multicollinearity=True,
                    multicollinearity_threshold=0.9,
                    low_variance_threshold=0.05,
                    verbose=False
                )
                
                # Carregar modelo baseado na fonte
                if model_source_type == "treinado na sessão":
                    model = model_to_use
                else:  # via upload
                    model = model_to_use
                
                # Mensagem de sucesso removida para evitar redundância
                
                # Obter dados transformados
                X_train_transformed = get_config("X_train_transformed")
                y_train = get_config("y_train")
                
                # Amostra para análise SHAP (limitar para performance) — manter alinhamento por rótulo
                sample_size = min(1000, len(X_train_transformed))
                X_sample = X_train_transformed.sample(n=sample_size, random_state=RANDOM_STATE)
                y_sample = y_train.reindex(X_sample.index) if hasattr(y_train, 'reindex') else np.array(y_train)[:len(X_sample)]
                # Remover possíveis NaN após reindex para manter tamanhos iguais
                if hasattr(y_sample, 'isna'):
                    if y_sample.isna().any():
                        valid_idx = y_sample.dropna().index
                        X_sample = X_sample.loc[valid_idx]
                        y_sample = y_sample.loc[valid_idx]
                
                # Converter labels para numérico
                le = LabelEncoder()
                y_sample_numeric = le.fit_transform(y_sample)
                
                clicked_shap = st.button("🔍 Gerar Análise SHAP", use_container_width=False, key="btn_generate_shap")
                if clicked_shap:
                    with st.spinner("Gerando análise SHAP (pode demorar alguns minutos)..."):
                        shap_global, shap_pos, shap_neg = generate_shap_analysis(model, X_sample, y_sample_numeric)
                        if shap_global:
                            # Persistir resultados na sessão (sem exibir imediatamente para evitar duplicação)
                            st.session_state.shap_global = shap_global
                            st.session_state.shap_pos = shap_pos
                            st.session_state.shap_neg = shap_neg
                            # Calcular e persistir importância SHAP (se possível)
                            try:
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer.shap_values(X_sample)
                                if isinstance(shap_values, list):
                                    shap_values = shap_values[1]
                                feature_names = X_sample.columns.tolist()
                                fig_shap_imp = plot_feature_importance_shap(shap_values, feature_names, top_n=20)
                                if fig_shap_imp:
                                    st.session_state.fig_shap_imp = fig_shap_imp
                            except Exception:
                                st.session_state.fig_shap_imp = None
                            st.rerun()
                        else:
                            st.error("Falha ao gerar análise SHAP")
                
                # Exibir resultados persistidos, se existirem
                if st.session_state.get("shap_global"):
                    st.markdown("### 📊 SHAP Summary Plot - Global")
                    st.markdown(st.session_state.shap_global, unsafe_allow_html=True)
                    st.markdown("")
                    render_expander_md("SHAP Global", "shap_global")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.session_state.get("shap_pos"):
                            st.markdown("### ☠️ SHAP - Classe Positiva (Venenoso)")
                            st.markdown(st.session_state.shap_pos, unsafe_allow_html=True)
                            st.markdown("")
                            render_expander_md("SHAP Local — Classe Positiva", "shap_local_pos")
                    with col2:
                        if st.session_state.get("shap_neg"):
                            st.markdown("### 🍽️ SHAP - Classe Negativa (Comestível)")
                            st.markdown(st.session_state.shap_neg, unsafe_allow_html=True)
                            st.markdown("")
                            render_expander_md("SHAP Local — Classe Negativa", "shap_local_neg")
                    if st.session_state.get("fig_shap_imp") is not None:
                        st.plotly_chart(st.session_state.fig_shap_imp, use_container_width=True)
                        st.markdown("")
                        render_expander_md("Importância de Features por SHAP", "feat_importance_shap")

                    # Botão para limpar resultados persistidos
                    if st.button("🗑️ Limpar Resultados", use_container_width=False, key="btn_clear_shap_results"):
                        for _k in ["shap_global", "shap_pos", "shap_neg", "fig_shap_imp"]:
                            st.session_state.pop(_k, None)
                        st.success("Resultados SHAP limpos.")
                        st.rerun()
                        
        except Exception as e:
            st.error(f"Falha na interpretação: {e}")

# =========================================
# Aba 6 — Validação com Dados Sem Rótulo
# =========================================
with tab_validate:
    st.subheader("Validação com Dados Sem Rótulo")
    st.caption("Teste o modelo com novos dados e analise a distribuição das predições")
    
    # Verificar se há modelo disponível (treinado ou pré-treinado)
    model_to_use, model_source_type = get_available_model()
    
    if model_to_use is None:
        st.error("❌ Nenhum modelo disponível!")
        st.info("💡 Treine um modelo na aba 'Treino & Métricas' ou carregue um modelo (.pkl) na sidebar.")
    else:
        st.info(f"ℹ️ Usando o modelo {model_source_type}")
        try:
            # Carregar modelo baseado na fonte
            if model_source_type == "treinado na sessão":
                model = model_to_use
            else:  # via upload
                model = load_pycaret_model(model_to_use)
            # Carregar dados
            if fonte == "Upload (CSV)" and uploaded_file is not None:
                df = load_mushrooms_csv(uploaded_file=uploaded_file)
            else:
                data_path = Path(local_path) if local_path else DATASET_PATH
                df = load_mushrooms_csv(path=data_path)
            
            # Upload de arquivo CSV
            st.markdown("#### 📁 Upload de Dados para Validação")
            uploaded_file = st.file_uploader(
                "Envie um arquivo CSV com dados de cogumelos (mesmas colunas do dataset original, SEM a coluna 'class')",
                type=['csv'],
                help="O arquivo deve ter as mesmas colunas do dataset original, exceto a coluna 'class'"
            )
        
            if uploaded_file is not None:
                try:
                    # Carregar dados de validação
                    df_validate = pd.read_csv(uploaded_file)
                    st.success(f"Arquivo carregado com sucesso! {len(df_validate)} amostras encontradas.")
                    
                    # Verificar colunas
                    expected_cols = [col for col in df.columns if col != target_col]
                    missing_cols = [col for col in expected_cols if col not in df_validate.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Colunas obrigatórias ausentes: {missing_cols}")
                    else:
                        # Mostrar prévia dos dados
                        st.markdown("#### Prévia dos Dados")
                        st.dataframe(df_validate.head(10), use_container_width=True)
                        
                        if st.button("Executar Validação", use_container_width=False):
                            with st.spinner("Executando predições..."):
                                # Executar predições
                                predictions = predict_model(model, data=df_validate)
                                
                                # Análise de distribuição
                                st.markdown("### 📊 Análise das Predições")
                            
                                # Distribuição das classes preditas
                                class_dist = predictions['prediction_label'].value_counts()
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("#### Distribuição das Classes Preditas")
                                    fig_dist = go.Figure(data=[
                                        go.Bar(x=class_dist.index, y=class_dist.values, 
                                              marker_color=['lightcoral' if x == 'p' else 'lightgreen' for x in class_dist.index])
                                    ])
                                    fig_dist.update_layout(
                                        title="Distribuição das Predições",
                                        xaxis_title="Classe Predita",
                                        yaxis_title="Frequência",
                                        height=400
                                    )
                                    st.plotly_chart(fig_dist, use_container_width=True)
                                    st.markdown("")
                                    render_expander_md("Distribuição das Classes Preditas", "val_class_dist")
                                
                                with col2:
                                    st.markdown("#### Distribuição das Probabilidades")
                                    fig_prob = go.Figure(data=[
                                        go.Histogram(x=predictions['prediction_score'], 
                                                   nbinsx=30, marker_color='lightblue')
                                    ])
                                    fig_prob.update_layout(
                                        title="Distribuição das Probabilidades",
                                        xaxis_title="Probabilidade de Ser Venenoso",
                                        yaxis_title="Frequência",
                                        height=400
                                    )
                                    st.plotly_chart(fig_prob, use_container_width=True)
                                    st.markdown("")
                                    render_expander_md("Distribuição das Probabilidades", "val_prob_dist")
                                
                                # Download dos resultados
                                csv_data = predictions.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "💾 Baixar Resultados (CSV)",
                                    data=csv_data,
                                    file_name="validacao_cogumelos.csv",
                                    mime="text/csv",
                                    use_container_width=False
                                )
                                
                except Exception as e:
                    st.error(f"Erro ao processar arquivo: {e}")
        
        except Exception as e:
            st.error(f"Falha na validação: {e}")

# =========================================
# Aba 7 — Predição Interativa
# =========================================
with tab_predict:
    st.subheader("Predição Interativa")
    
    # Verificar se há modelo disponível (treinado ou pré-treinado)
    model_to_use, model_source_type = get_available_model()
    
    if model_to_use is None:
        st.error("❌ Nenhum modelo disponível!")
        st.info("💡 Treine um modelo na aba 'Treino & Métricas' ou carregue um modelo (.pkl) na sidebar.")
    else:
        st.info(f"ℹ️ Usando o modelo {model_source_type}")
        try:
            # Carregar modelo baseado na fonte
            if model_source_type == "treinado na sessão":
                model = model_to_use
            else:  # via upload
                model = load_pycaret_model(model_to_use)
            # Carregar dados
            if fonte == "Upload (CSV)" and uploaded_file is not None:
                df = load_mushrooms_csv(uploaded_file=uploaded_file)
            else:
                data_path = Path(local_path) if local_path else DATASET_PATH
                df = load_mushrooms_csv(path=data_path)

            # Predição única (form dinâmico com categorias do CSV)
            st.markdown("#### 📍 Predição única")
            cats = {c: sorted(df[c].dropna().unique().tolist())
                    for c in df.columns if c != target_col}
            cols = list(cats.keys())
            
            # Organizar em colunas com validação
            col1, col2, col3 = st.columns(3)
            values = {}
            validation_errors = []
            
            for i, c in enumerate(cols):
                with (col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3):
                    values[c] = st.selectbox(
                        f"{c} *", 
                        cats[c], 
                        key=f"one_{c}",
                        help=f"Escolha um valor para {c}"
                    )
                    # Validação básica
                    if values[c] is None or values[c] == "":
                        validation_errors.append(f"{c} é obrigatório")

            # Validação adicional
            st.markdown("#### 🔍 Validação de Entrada")
            if validation_errors:
                for error in validation_errors:
                    st.error(f"❌ {error}")
            else:
                st.success("✅ Todos os campos preenchidos corretamente")

            if st.button("🔮 Prever (amostra única)", use_container_width=False, disabled=len(validation_errors) > 0):
                try:
                    df_one = pd.DataFrame([values])

                    # Divisória entre o botão e os resultados subsequentes
                    st.markdown("---")

                    # Validação adicional dos dados
                    st.markdown("#### 📋 Validação dos Dados")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Dados enviados:**")
                        st.dataframe(df_one, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Verificações:**")
                        # Verificar se todas as colunas estão presentes
                        missing_cols = [col for col in df.columns if col != target_col and col not in df_one.columns]
                        if missing_cols:
                            st.error(f"❌ Colunas ausentes: {missing_cols}")
                        else:
                            st.success("✅ Todas as colunas presentes")
                        # Verificar compatibilidade (loader temporário)
                        with st.spinner("Verificando compatibilidade..."):
                            pass

                    # Executar predição
                    with st.spinner("Executando predição..."):
                        out = predict_model(model, data=df_one)
                        label = out.loc[0, "prediction_label"]
                        score = float(out.loc[0, "prediction_score"])
                
                    # Resultados com análise detalhada
                    st.markdown("#### 📊 Resultados da Predição")
                    
                    # Métricas principais
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Classe Prevista", str(label), 
                                 delta="Venenoso" if label == 'p' else "Comestível")
                    with col2:
                        st.metric("Probabilidade", f"{score:.3f}", 
                                 delta=f"{(score-0.5)*100:+.1f}% vs aleatório")
                    with col3:
                        confidence = "Alta" if score > 0.8 or score < 0.2 else "Média" if score > 0.6 or score < 0.4 else "Baixa"
                        st.metric("Confiança", confidence)
                
                    # Análise de confiança
                    st.markdown("#### 🛡️ Análise de Confiança")
                    if score > 0.8:
                        st.success("🟢 **Alta Confiança**: O modelo está muito confiante na predição")
                    elif score < 0.2:
                        st.success("🟢 **Alta Confiança**: O modelo está muito confiante na predição")
                    elif score > 0.6 or score < 0.4:
                        st.warning("🟡 **Confiança Média**: O modelo tem confiança moderada")
                    else:
                        st.error("🔴 **Baixa Confiança**: O modelo não está confiante na predição")
                    
                    # Avisos de segurança aprimorados
                    st.markdown("#### ⚠️ Avisos de Segurança")
                    if label == 'p':
                        st.error("""
                        🚨 **ATENÇÃO CRÍTICA**: 
                        - O modelo classificou como **VENENOSO**
                        - **NUNCA** consuma este cogumelo baseado apenas neste modelo
                        - **SEMPRE** consulte um micologista especialista
                        - **ERRO** pode ser **FATAL**
                        """)
                    else:
                        st.warning("""
                        ⚠️ **ATENÇÃO IMPORTANTE**: 
                        - Mesmo classificado como comestível, **NUNCA** consuma baseado apenas neste modelo
                        - **SEMPRE** consulte um micologista especialista
                        - Este modelo é apenas para fins educacionais
                        - **ERRO** pode ser **FATAL**
                        """)

                    # Histórico de predições (session state)
                    if 'prediction_history' not in st.session_state:
                        st.session_state.prediction_history = []
                    
                    # Adicionar à história
                    st.session_state.prediction_history.append({
                        'timestamp': pd.Timestamp.now(),
                        'features': values.copy(),
                        'prediction': label,
                        'probability': score
                    })
                    
                    # Mostrar histórico
                    if len(st.session_state.prediction_history) > 1:
                        st.markdown("#### ⏳ Histórico de Predições")
                        history_df = pd.DataFrame(st.session_state.prediction_history)
                        st.dataframe(history_df[['timestamp', 'prediction', 'probability']], use_container_width=True)
                        
                        if st.button("🗑️ Limpar Histórico"):
                            st.session_state.prediction_history = []
                            st.rerun()
                    
                except Exception as e:
                    st.error(f"Erro na predição: {e}")
                    st.exception(e)

            st.markdown("---")
            st.markdown("#### 📦 Predição em lote (CSV)")
            file = st.file_uploader("Envie um CSV com as **mesmas colunas originais** (sem a coluna alvo).", type=["csv"])
            if file is not None:
                try:
                    df_in = pd.read_csv(file)
                    st.write("#### Prévia dos Dados", df_in.head())
                    if st.button("🔮 Prever lote", use_container_width=False):
                        preds = predict_model(model, data=df_in.copy())
                        preds = preds.rename(columns={
                            "prediction_label": "pred_label",
                            "prediction_score": "pred_score",
                        })
                        st.success("Predições geradas!")
                        st.dataframe(preds.head(50), use_container_width=True)
                        st.download_button(
                            "💾 Baixar resultados (CSV)",
                            data=preds.to_csv(index=False).encode("utf-8"),
                            file_name="predicoes_mushrooms.csv",
                            mime="text/csv",
                            use_container_width=False
                        )
                except Exception as e:
                    st.error(f"Falha ao processar CSV: {e}")

        except Exception as e:
            st.error(f"Falha na predição: {e}")

# =========================================
# Aba 1 — Orientações
# =========================================
with tab_explain:
    # Passo a passo de uso
    st.markdown("""
    ### 📚 Como Usar o Aplicativo
    1. **Configuração inicial (Sidebar)**
       - Selecione a fonte de dados: `Local (ENV/Path)` ou `Upload (CSV)`.
       - Opcional: envie um modelo `.pkl` na seção "Modelo" para usar nas abas de análise.

    2. **EDA (ydata-profiling)**
       - Opcional: clique em "Gerar relatório (profiling)" para um resumo exploratório do dataset.
       - Baixe o HTML se quiser compartilhar ou documentar.

    3. **Treino & Métricas**
       - Clique em "Carregar, Treinar e Visualizar" para executar o `setup`, comparar modelos e treinar o melhor modelo (`Decision Tree`).
       - Veja o `Leaderboard`, informações de `setup` e do modelo treinado.

    4. **Evaluate Model**
       - Visualize os gráficos/diagramas de avaliação do modelo na ordem sugerida (pipeline, importância, curvas, relatórios, etc.).
       - Use os expanders "Como interpretar" para entender cada visualização.

    5. **Interpretação (SHAP)**
       - Clique em "Gerar Análise SHAP" para visualizar `Global`, `Classe Positiva` (venenoso) e `Classe Negativa` (comestível).
       - As visualizações ficam persistidas até você limpar os resultados.

    6. **Validação (Dados Sem Rótulo)**
       - Envie um CSV sem a coluna `class` e clique em "Executar Validação".
       - Veja a distribuição de classes preditas e probabilidades e baixe os resultados em CSV.

    7. **Predição Interativa**
       - Preencha os campos com os valores de uma amostra e clique em "Prever (amostra única)".
       - Para lote, envie um CSV com as mesmas colunas originais (sem `class`) e clique em "Prever lote".

    8. **Exportar Relatório**
       - Gere um **HTML completo** com métricas e visualizações e faça o download.

    9. **Dicas**
       - Caso altere a fonte de dados, reexecute as seções conforme necessário.
       - Alguns gráficos dependem de bibliotecas opcionais; mensagens orientarão se algo estiver faltando.
       - Lembre-se: este app é apenas educacional. Não utilize para decisões reais sobre cogumelos.
    """)

    # Aviso de segurança principal
    st.error("⚠️ **AVISO IMPORTANTE DE SEGURANÇA**")
    st.markdown("""
    Este modelo de classificação de cogumelos é **APENAS para fins educacionais e de demonstração**.
    
    **NUNCA use este modelo para determinar se um cogumelo é comestível na vida real!**
    
    - ❌ **Erro na classificação de cogumelos venenosos pode ser FATAL**;
    - ✅ **SEMPRE consulte especialistas (micologistas) para identificação segura**;
    - ✅ **Use apenas para aprendizado de Machine Learning e análise de dados**.
    """)

# =========================================
# Aba 8 — Exportação HTML Completa
# =========================================
def export_everything_html(
    path_html="doc/resultados/relatorio_completo_cogumelos.html",
    figs_metricas: dict | None = None,
    ap_value=None, auc_value=None, accuracy_value=None,
    precision_value=None, recall_value=None, f1_value=None
):
    """Exporta relatório completo em HTML"""
    Path(path_html).parent.mkdir(parents=True, exist_ok=True)
    sections = []
    
    # Cabeçalho
    sections.append(f"""
    <div class="card">
      <h1>🍄 Relatório de Classificação de Cogumelos</h1>
      <p><b>Análise completa do modelo de classificação</b> — Relatório gerado automaticamente</p>
      <div class="warning">
        <strong>⚠️ AVISO DE SEGURANÇA:</strong> Este modelo é apenas para fins educacionais. 
        NUNCA use para determinar se um cogumelo é comestível na vida real!
      </div>
    </div>
    """)
    
    # Resumo das métricas
    if any([ap_value, auc_value, accuracy_value, precision_value, recall_value, f1_value]):
        metrics_html = "<div class='card'><h2>📊 Resumo das Métricas</h2><div class='metrics-grid'>"
        if accuracy_value: metrics_html += f"<div class='metric'><b>Acurácia:</b> {accuracy_value:.4f}</div>"
        if precision_value: metrics_html += f"<div class='metric'><b>Precisão:</b> {precision_value:.4f}</div>"
        if recall_value: metrics_html += f"<div class='metric'><b>Recall:</b> {recall_value:.4f}</div>"
        if f1_value: metrics_html += f"<div class='metric'><b>F1-Score:</b> {f1_value:.4f}</div>"
        if auc_value: metrics_html += f"<div class='metric'><b>AUC-ROC:</b> {auc_value:.4f}</div>"
        if ap_value: metrics_html += f"<div class='metric'><b>AP (PR-AUC):</b> {ap_value:.4f}</div>"
        metrics_html += "</div></div>"
        sections.append(metrics_html)
    
    # Gráficos
    if figs_metricas:
        for title, fig in figs_metricas.items():
            try:
                sections.append(f"<div class='card'><h2>{title}</h2>{_fig_to_html_snippet(fig)}</div>")
            except Exception as e:
                sections.append(f"<div class='card'><h2>{title}</h2><p>Erro ao gerar gráfico: {e}</p></div>")
    
    # Glossário
    glossary = f"""
    <div class="card">
      <h2>📚 Glossário de Métricas</h2>
      <ul>
        <li><b>AUC-ROC</b>: {_md_to_inline_html(EXPLANATIONS_MD['auc_roc'])}</li>
        <li><b>AP (PR-AUC)</b>: {_md_to_inline_html(EXPLANATIONS_MD['ap_pr'])}</li>
        <li><b>Matriz de Confusão</b>: {_md_to_inline_html(EXPLANATIONS_MD['cm'])}</li>
        <li><b>Importância de Features (Treinamento)</b>: {_md_to_inline_html(EXPLANATIONS_MD['feat_importance_model'])}</li>
        <li><b>Calibração</b>: {_md_to_inline_html(EXPLANATIONS_MD['calibration'])}</li>
        <li><b>EDA</b>: {_md_to_inline_html(EXPLANATIONS_MD['eda'])}</li>
      </ul>
    </div>
    """
    sections.append(glossary)

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
    <meta charset="utf-8">
    <title>Relatório de Classificação de Cogumelos</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
     body{{margin:20px; font-family:Inter, system-ui, Arial; color:#eaeaea; background:#111}}
     .card{{background:#161616;border:1px solid #2a2a2a;border-radius:12px;padding:16px;margin:18px 0}}
     .metrics-grid{{display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin:16px 0}}
     .metric{{background:#2a2a2a;padding:12px;border-radius:8px;text-align:center}}
     .warning{{background:#ff4444;color:white;padding:12px;border-radius:8px;margin:16px 0}}
     a, a:visited{{color:#7ab9ff}}
     h1,h2,h3{{color:#f5f5f5}}
     ul{{line-height:1.6}}
    </style>
    </head>
    <body>
      {"".join(sections)}
    </body>
    </html>
    """
    with open(path_html, "w", encoding="utf-8") as f:
        f.write(html)
    return path_html

with tab_export:
    st.subheader("Exportar Relatório Completo")
    st.markdown("Gere um **relatório único (HTML)** com todas as métricas e visualizações.")
    
    # Verificar se há modelo disponível (treinado ou pré-treinado)
    model_to_use, model_source_type = get_available_model()
    
    if model_to_use is None:
        st.error("❌ Nenhum modelo disponível!")
        st.info("💡 Treine um modelo na aba 'Treino & Métricas' ou carregue um modelo (.pkl) na sidebar.")
    else:
        st.info(f"ℹ️ Usando o modelo {model_source_type}")
        if st.button("📦 Gerar Relatório HTML Completo", use_container_width=False):
            try:
                # Carregar dados e modelo para exportação
                # Carregar dados
                if fonte == "Upload (CSV)" and uploaded_file is not None:
                    df = load_mushrooms_csv(uploaded_file=uploaded_file)
                else:
                    data_path = Path(local_path) if local_path else DATASET_PATH
                    df = load_mushrooms_csv(path=data_path)
                # Setup mínimo apenas para obter configurações necessárias
                _ = setup(
                    data=df,
                    target=target_col,
                    session_id=1,
                    feature_selection=True,
                    feature_selection_method="classic",
                    remove_multicollinearity=True,
                    multicollinearity_threshold=0.9,
                    low_variance_threshold=0.05,
                    verbose=False
                )
                
                # Carregar modelo baseado na fonte
                if model_source_type == "treinado na sessão":
                    model = model_to_use
                else:  # via upload
                    model = model_to_use
                
                # Obter predições e métricas
                predictions = predict_model(model, data=df)
                y_true = df[target_col]
                y_pred = predictions['prediction_label']
                y_proba = predictions['prediction_score']
                
                # Converter labels para valores numéricos para as métricas
                le = LabelEncoder()
                y_true_numeric = le.fit_transform(y_true)
                y_pred_numeric = le.transform(y_pred)
                
                # Calcular métricas
                accuracy = accuracy_score(y_true_numeric, y_pred_numeric)
                precision = precision_score(y_true_numeric, y_pred_numeric, average='weighted')
                recall = recall_score(y_true_numeric, y_pred_numeric, average='weighted')
                f1 = f1_score(y_true_numeric, y_pred_numeric, average='weighted')
                auc_roc = roc_auc_score(y_true_numeric, y_proba)
                ap = average_precision_score(y_true_numeric, y_proba)
                
                # Gerar gráficos para exportação
                figs_metricas = {}
                try:
                    figs_metricas["Curva ROC"] = plot_roc_curve_advanced(y_true_numeric, y_proba)
                    figs_metricas["Curva Precision-Recall"] = plot_pr_curve_advanced(y_true_numeric, y_proba)
                    # Curva de Calibração (consistente com Evaluate Model)
                    figs_metricas["Curva de Calibração"] = plot_calibration_curve_advanced(y_true_numeric, y_proba)
                    class_names = sorted(y_true.unique())
                    figs_metricas["Matriz de Confusão"], _ = plot_confusion_matrix_advanced(y_true_numeric, y_pred_numeric, class_names)
                    
                    # Importância de features
                    if hasattr(model, 'feature_importances_'):
                        feature_names = get_config("X_train_transformed").columns.tolist()
                        importances = model.feature_importances_
                        importance_dict = dict(zip(feature_names, importances))
                        figs_metricas["Importância de Features"] = plot_feature_importance_advanced(importance_dict, top_n=20)
                except Exception as e:
                    st.warning(f"Erro ao gerar alguns gráficos: {e}")
                
                # Gerar relatório HTML
                html_path = export_everything_html(
                    figs_metricas=figs_metricas,
                    ap_value=ap,
                    auc_value=auc_roc,
                    accuracy_value=accuracy,
                    precision_value=precision,
                    recall_value=recall,
                    f1_value=f1
                )
                
                st.success(f"Relatório HTML gerado com sucesso!")
                st.info(f"Arquivo salvo em: {html_path}")
                
                # Download do relatório
                with open(html_path, "rb") as f:
                    st.download_button(
                        "💾 Baixar Relatório HTML Completo",
                        data=f.read(),
                        file_name="relatorio_completo_cogumelos.html",
                        mime="text/html",
                        use_container_width=False
                    )
                
                # Mostrar prévia do relatório
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                
                st.markdown("### Prévia do Relatório")
                st.components.v1.html(html_content, height=600, scrolling=True)
                
            except Exception as e:
                st.error(f"Falha na exportação: {e}")

# =========================================
# Rodapé
# =========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🍄 <strong>Classificação de Cogumelos</strong> — Treinamento + Upload + XAI + Análise Completa</p>
    <p><em>Desenvolvido para fins educacionais e de demonstração</em></p>
</div>
""", unsafe_allow_html=True)