# app.py
# -*- coding: utf-8 -*-
"""
Streamlit – Classificação (PyCaret)
Abrange:
- Treino & Métricas (carregar dataset, setup, compare_models, plots do evaluate/plot_model)
- Profiling (ydata-profiling + download HTML)
- Previsão Interativa (form dinâmico + upload CSV)
- Entenda os Resultados (explicação em texto)
- Validação (sem rótulo) (upload CSV sem target)
- Exportar (relatório HTML com métricas/plots)
"""

import os
import io
import glob
import json
import base64
import textwrap
import streamlit as st
import pandas as pd
from datetime import datetime

# PyCaret 3.x – classificação
from pycaret.classification import setup, compare_models, pull, create_model, tune_model, finalize_model, predict_model, plot_model, get_config, set_config

# ydata-profiling
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html as st_html

# --------------------------
# Configurações básicas
# --------------------------
st.set_page_config(
    page_title="Pipeline de Classificação - PyCaret",
    page_icon="🧪",
    layout="wide"
)

# Diretórios auxiliares
ARTIFACTS_DIR = "artifacts"
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
EXPORTS_DIR = os.path.join(ARTIFACTS_DIR, "exports")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

# --------------------------
# Funções utilitárias
# --------------------------
@st.cache_data(show_spinner=False)
def load_dataframe(
    source: str,
    local_path: str = "",
    sep: str = ",",
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Carrega o dataset conforme a fonte.
    - "local": lê de um caminho local (env MUSHROOM_LOCAL como padrão).
    - "upload": usa o arquivo carregado via UI (tratar fora).
    - "kaggle": (opcional) você pode adaptar para baixar via API se desejar.
    """
    if source == "local":
        if not local_path or not os.path.exists(local_path):
            raise FileNotFoundError(f"Arquivo local não encontrado: {local_path}")
        return pd.read_csv(local_path, sep=sep, encoding=encoding)
    else:
        raise ValueError("Fonte de dados inválida para load_dataframe (use 'local' ou trate 'upload' separadamente).")

def b64_download(data_bytes: bytes, filename: str, mime: str = "text/html"):
    b64 = base64.b64encode(data_bytes).decode()
    href = f'<a download="{filename}" href="data:{mime};base64,{b64}">Baixar: {filename}</a>'
    return href

def _latest_plot(pattern: str = "*.png") -> str | None:
    files = sorted(glob.glob(os.path.join(PLOTS_DIR, pattern)), key=os.path.getmtime)
    return files[-1] if files else None

def _save_current_plot(name_hint: str):
    """
    Alguns plots de PyCaret salvam com nomes fixos.
    Depois do plot, tenta achar a imagem mais recente e renomear com um padrão amigável.
    """
    latest = _latest_plot("*.png")
    if latest:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = os.path.join(PLOTS_DIR, f"{name_hint}_{stamp}.png")
        try:
            os.replace(latest, new_name)
            return new_name
        except Exception:
            return latest
    return None

def _show_plot_image(path: str, caption: str):
    if path and os.path.exists(path):
        st.image(path, use_column_width=True, caption=caption)
    else:
        st.info(f"Plot não encontrado ainda ({caption}).")

def _export_html_report(
    dataset_name: str,
    target_col: str,
    setup_params: dict,
    leaderboard_df: pd.DataFrame | None,
    plot_paths: dict[str, str],
    notes: str = ""
) -> bytes:
    """
    Gera um HTML simples consolidando:
    - Metadados (dataset, target, params de setup)
    - Tabela de leaderboard (compare_models/pull)
    - Referências às imagens dos plots gerados
    - Notas/explicações
    """
    # Serializa setup params de forma legível
    params_repr = "<pre>" + textwrap.indent(json.dumps(setup_params, indent=2, ensure_ascii=False), "  ") + "</pre>"

    # Tabela (HTML) do leaderboard
    leaderboard_html = ""
    if leaderboard_df is not None and not leaderboard_df.empty:
        leaderboard_html = leaderboard_df.to_html(index=False, escape=False)
    else:
        leaderboard_html = "<p><em>Leaderboard não disponível.</em></p>"

    # Lista de imagens
    images_html = []
    for label, p in plot_paths.items():
        if p and os.path.exists(p):
            # Converte a imagem para base64 embutida no HTML
            with open(p, "rb") as f:
                b64img = base64.b64encode(f.read()).decode()
            images_html.append(f"<h3>{label}</h3><img src='data:image/png;base64,{b64img}' style='max-width:100%;'/>")
        else:
            images_html.append(f"<h3>{label}</h3><p><em>Plot exibido apenas na interface (não salvo como arquivo).</em></p>")

    html_doc = f"""
<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8"/>
<title>Relatório de Classificação – PyCaret</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; }}
h1, h2, h3 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 8px; font-size: 14px; }}
pre {{ background: #f6f8fa; padding: 12px; overflow-x: auto; }}
.section {{ margin-bottom: 32px; }}
hr {{ margin: 32px 0; }}
</style>
</head>
<body>
  <h1>Relatório de Classificação – PyCaret</h1>
  <div class="section">
    <h2>Resumo</h2>
    <p><strong>Dataset:</strong> {dataset_name}</p>
    <p><strong>Target:</strong> {target_col}</p>
    <h3>Parâmetros do setup()</h3>
    {params_repr}
  </div>

  <div class="section">
    <h2>Leaderboard (compare_models)</h2>
    {leaderboard_html}
  </div>

  <div class="section">
    <h2>Gráficos</h2>
    {"".join(images_html)}
  </div>

  <div class="section">
    <h2>Notas/Explicações</h2>
    <p>{notes}</p>
  </div>
</body>
</html>
"""
    return html_doc.encode("utf-8")

# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("⚙️ Configurações")

# Fonte de dados (com padrão do seu projeto lembrado)
default_local = os.environ.get("MUSHROOM_LOCAL", "./doc/mushrooms.csv")
fonte = st.sidebar.radio(
    "Fonte de dados",
    options=["Local (ENV/Path)", "Upload (CSV)"],
    index=0
)
local_path = st.sidebar.text_input("Caminho local (ENV MUSHROOM_LOCAL ou outro)", value=default_local)

# Coluna alvo (target) - estática
st.sidebar.text_input("Coluna alvo (target)", value="class", disabled=True)

st.sidebar.markdown("---")
acc_goal = st.sidebar.slider("🎯 Alvo de Precisão (threshold)", 0.5, 1.0, 0.8, 0.01,
                             help="Meta desejada de *Accuracy* para destacar se o modelo atingiu a precisão mínima.")
folds = st.sidebar.number_input("🔁 K-Folds (cross-validation)", min_value=3, max_value=20, value=10, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("💡 Dica: ajuste os Folds para controle de variância; use o Alvo de Precisão para checar se o melhor modelo atingiu sua meta.")

# Upload opcional
uploaded_file = None
if fonte == "Upload (CSV)":
    uploaded_file = st.sidebar.file_uploader("Envie um CSV", type=["csv"])

# --------------------------
# Cabeçalho
# --------------------------
st.title("🧪 Classificação com PyCaret – Dashboard Interativo")

# --------------------------
# Tabs
# --------------------------
tabs = st.tabs([
    "Treino & Métricas",
    "Profiling",
    "Previsão Interativa",
    "Entenda os Resultados",
    "Validação (sem rótulo)",
    "Exportar"
])

# Estados em sessão
if "data" not in st.session_state:
    st.session_state.data = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = None
if "setup_params" not in st.session_state:
    st.session_state.setup_params = {}
if "profiling_html" not in st.session_state:
    st.session_state.profiling_html = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = []
if "plots" not in st.session_state:
    st.session_state.plots = {}  # label -> path
if "training_completed" not in st.session_state:
    st.session_state.training_completed = False
if "setup_info" not in st.session_state:
    st.session_state.setup_info = None
if "model_info" not in st.session_state:
    st.session_state.model_info = None
if "accuracy_goal_met" not in st.session_state:
    st.session_state.accuracy_goal_met = None

# =========================================================
# TAB 1 – Treino & Métricas
# =========================================================
# ===== SUBSTITUA apenas o conteúdo da aba "Treino & Métricas" pelo bloco abaixo =====
with tabs[0]:
    st.subheader("Treino & Métricas")
    st.markdown("Clique para **carregar** o dataset, **treinar** e **visualizar** os resultados principais (Leaderboard e plots).")

    # Botão para executar o treinamento
    if st.button("🚀 Carregar, Treinar e Visualizar", use_container_width=True):
        # Carregar dados
        if fonte.startswith("Local"):
            df = load_dataframe("local", local_path)
        else:
            if uploaded_file is None:
                st.error("Envie um CSV na barra lateral.")
                st.stop()
            df = pd.read_csv(uploaded_file)

        st.session_state.data = df.copy()

        # Features
        target_col = "class"  # Target fixo
        if target_col in df.columns:
            st.session_state.feature_cols = [c for c in df.columns if c != target_col]
        else:
            st.warning(f"Target '{target_col}' não encontrado no dataset. Verifique o nome da coluna.")
            st.stop()

        # === SETUP EXATAMENTE COMO NO SEU NOTEBOOK ===
        setup_params = dict(
            data=df,
            target="class",
            session_id=1,
            feature_selection=True,
            feature_selection_method='classic',
            remove_multicollinearity=True,
            multicollinearity_threshold=0.9,
            low_variance_threshold=0.05,
            fold=folds,  # controlado pela sidebar
        )
        st.session_state.setup_params = {k: v for k, v in setup_params.items() if k != "data"}

        with st.spinner("Executando setup() com seus parâmetros..."):
            s = setup(**setup_params)

        # Captura e exibe a tabela resultante do setup
        setup_info = pull()
        st.session_state.setup_info = setup_info
        st.markdown("#### 📋 Informações do Setup")
        st.dataframe(setup_info, use_container_width=True)
        
        st.success("Setup concluído ✅")

        # Leaderboard por AUC (compare_models) — TABELA
        with st.spinner("Comparando modelos (ordenando por AUC)..."):
            _ = compare_models(sort='AUC')        # só para formar o leaderboard ordenado
            leaderboard = pull()                  # captura a tabela
        st.session_state.leaderboard = leaderboard

        # === SEU MODELO PRINCIPAL: DECISION TREE ===
        with st.spinner("Criando modelo Decision Tree (dt)..."):
            dt_model = create_model("dt")  # seu classificador principal
        
        # Captura e exibe a tabela resultante do create_model
        model_info = pull()
        st.session_state.model_info = model_info
        st.markdown("#### 🌳 Informações do Modelo Decision Tree")
        st.dataframe(model_info, use_container_width=True)
        
        st.session_state.best_model = dt_model

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
                st.session_state.accuracy_goal_met = True
                st.success(f"🎉 Meta de precisão atingida! {float(top_acc):.4f} ≥ {acc_goal:.4f}")
            else:
                st.session_state.accuracy_goal_met = False
                st.warning(f"Meta de precisão **não** atingida. Top Accuracy: {float(top_acc):.4f} < {acc_goal:.4f}")
        else:
            st.session_state.accuracy_goal_met = None
            st.info("Coluna de Accuracy não localizada no leaderboard.")

    # Exibir resultados se o treinamento foi concluído
    if st.session_state.training_completed and st.session_state.leaderboard is not None:
        st.markdown("---")
        st.markdown("### 📊 Resultados do Treinamento")
        
        # Informações do Setup
        if st.session_state.setup_info is not None:
            st.markdown("#### 📋 Informações do Setup")
            st.dataframe(st.session_state.setup_info, use_container_width=True)
            st.markdown("---")
        
        # Leaderboard
        st.markdown("#### Leaderboard (compare_models | sort='AUC')")
        st.dataframe(st.session_state.leaderboard, use_container_width=True)
        st.markdown("---")
        
        # Informações do Modelo
        if st.session_state.model_info is not None:
            st.markdown("#### 🌳 Informações do Modelo Decision Tree")
            st.dataframe(st.session_state.model_info, use_container_width=True)
            st.markdown("---")
        
        # Meta de Precisão
        if st.session_state.accuracy_goal_met is not None:
            if st.session_state.accuracy_goal_met:
                st.success("🎉 Meta de precisão foi atingida!")
            else:
                st.warning("⚠️ Meta de precisão não foi atingida.")
            st.markdown("---")

        # Plots do modelo
        if st.session_state.best_model is not None:
            st.markdown("#### Gráficos do Modelo (Decision Tree)")
            plots_requested = [
                ("Pipeline", "pipeline"),
                ("AUC", "auc"),
                ("Matriz de Confusão", "confusion_matrix"),
                ("Threshold", "threshold"),
                ("Importância de Features", "feature"),
            ]

            cols = st.columns(2)
            idx = 0
            for label, plt_name in plots_requested:
                try:
                    # Exibe o plot diretamente no Streamlit sem salvar arquivo
                    with cols[idx % 2]:
                        # Gera o plot do PyCaret e exibe diretamente
                        plot_model(st.session_state.best_model, plot=plt_name, display_format='streamlit')
                    # Marca como plot gerado mas não salvo
                    st.session_state.plots[label] = None
                    idx += 1
                except Exception as e:
                    st.info(f"Plot '{label}' não pôde ser gerado automaticamente ({e}).")

            # Árvore de Decisão (visual)
            try:
                # Exibe o plot diretamente no Streamlit sem salvar arquivo
                with cols[idx % 2]:
                    # Gera o plot do PyCaret e exibe diretamente
                    plot_model(st.session_state.best_model, plot="tree", display_format='streamlit')
                # Marca como plot gerado mas não salvo
                st.session_state.plots["Decision Tree (DT)"] = None
            except Exception as e:
                st.info(f"Plot 'Decision Tree' não pôde ser gerado ({e}).")

        # Botão para limpar resultados
        if st.button("🗑️ Limpar Resultados", use_container_width=True):
            st.session_state.training_completed = False
            st.session_state.leaderboard = None
            st.session_state.best_model = None
            st.session_state.plots = {}
            st.session_state.setup_info = None
            st.session_state.model_info = None
            st.session_state.accuracy_goal_met = None
            st.rerun()

# =========================================================
# TAB 2 – Profiling
# =========================================================
with tabs[1]:
    st.subheader("Profiling (ydata-profiling)")
    st.caption("Gera um relatório exploratório completo do dataset.")
    
    if st.session_state.data is None:
        st.info("Carregue e treine na aba **Treino & Métricas** primeiro.")
    else:
        # Botão para gerar/atualizar profiling
        if st.button("📊 Gerar/Atualizar Profiling", use_container_width=True):
            with st.spinner("Gerando ydata-profiling..."):
                prof = ProfileReport(st.session_state.data, title="Data Profiling", minimal=False)
                report_html = prof.to_html()
                st.session_state.profiling_html = report_html
            st.success("Profiling gerado ✅")
            st.rerun()
        
        # Status informativo abaixo do botão
        if st.session_state.profiling_html:
            st.success("✅ Profiling disponível para visualização e download")
        else:
            st.info("ℹ️ Clique no botão para gerar o profiling")

        # Exibir profiling se disponível
        if st.session_state.profiling_html:
            st.markdown("---")
            st.markdown("### 📈 Relatório de Profiling")
            st_html(st.session_state.profiling_html, height=800, scrolling=True)
            
            # Botão de download (HTML)
            st.markdown("---")
            st.download_button(
                "⬇️ Baixar Profiling (HTML)",
                data=st.session_state.profiling_html.encode("utf-8"),
                file_name="profiling_report.html",
                mime="text/html",
                use_container_width=True
            )

# =========================================================
# TAB 3 – Previsão Interativa
# =========================================================
with tabs[2]:
    st.subheader("Previsão Interativa")
    if st.session_state.best_model is None or st.session_state.data is None:
        st.info("Treine um modelo na aba **Treino & Métricas** primeiro.")
    else:
        st.markdown("Preencha os campos para uma previsão unitária **ou** envie um CSV.")
        df = st.session_state.data
        feats = st.session_state.feature_cols
        form_vals = {}
        with st.form("form_predict"):
            for c in feats:
                if pd.api.types.is_numeric_dtype(df[c]):
                    val = st.number_input(f"{c}", value=float(df[c].dropna().median()) if df[c].dropna().size else 0.0)
                else:
                    # Categóricas – tenta pegar categoria mais comum
                    common = df[c].mode().iloc[0] if not df[c].mode().empty else ""
                    val = st.text_input(f"{c}", value=str(common))
                form_vals[c] = val
            submitted = st.form_submit_button("🔮 Prever (unitário)")
        if submitted:
            X = pd.DataFrame([form_vals])
            pred = predict_model(st.session_state.best_model, data=X)
            st.write("**Resultado:**")
            st.dataframe(pred)

        st.markdown("---")
        up = st.file_uploader("Ou envie um CSV com as colunas de features", type=["csv"], key="predict_csv")
        if up:
            Xcsv = pd.read_csv(up)
            st.write("Prévia do arquivo enviado:", Xcsv.head(10))
            with st.spinner("Gerando previsões..."):
                predcsv = predict_model(st.session_state.best_model, data=Xcsv)
            st.success("Previsões concluídas.")
            st.dataframe(predcsv.head(100), use_container_width=True)
            st.download_button(
                "⬇️ Baixar previsões (CSV)",
                data=predcsv.to_csv(index=False).encode("utf-8"),
                file_name="predicoes.csv",
                mime="text/csv"
            )

# =========================================================
# TAB 4 – Entenda os Resultados
# =========================================================
with tabs[3]:
    st.subheader("Entenda os Resultados")
    st.markdown("""
**O que você está vendo neste app:**

- **Setup do PyCaret**: configura pré-processamentos (normalização, seleção de variáveis, remoção de multicolinearidade, *k-folds*, etc.).
- **compare_models()**: treina e avalia diversos algoritmos sob validação cruzada, exibindo o *leaderboard* (métricas como Accuracy, AUC, Recall, etc.). O melhor modelo (topo) é selecionado.
- **Plots principais**:
  - **Pipeline**: visão do pipeline (pré-processamento + algoritmo).
  - **AUC**: curva ROC e respectiva área (capacidade do modelo em discriminar classes).
  - **Matriz de Confusão**: acertos/erros por classe.
  - **Threshold**: análise de limiar de decisão para calibrar *trade-off* entre métricas (p.ex., Precision vs Recall).
  - **Importância de Features**: quais variáveis mais impactam as decisões do modelo.
  - **Árvore de Decisão**: visualização explicável (usando um modelo DT como exemplo).
- **Previsão Interativa**: insira valores ou envie CSV para obter predições com o modelo treinado.
- **Validação (sem rótulo)**: processo de *scoring* em dados sem a coluna alvo.
- **Exportar**: gera um HTML com parâmetros, leaderboard e imagens dos plots para documentação/entrega.

> **Dica:** Use o *Alvo de Precisão* (na barra lateral) apenas como meta para destacar se a acurácia atingiu seu patamar desejado. Não é um limiar aplicado ao classificador.
""")

# =========================================================
# TAB 5 – Validação (sem rótulo)
# =========================================================
with tabs[4]:
    st.subheader("Validação (sem rótulo)")
    if st.session_state.best_model is None or st.session_state.data is None:
        st.info("Treine um modelo na aba **Treino & Métricas** primeiro.")
    else:
        st.caption("Envie um CSV **sem** a coluna alvo; o app aplicará o modelo e retornará as predições.")
        up2 = st.file_uploader("CSV para validação (sem rótulo)", type=["csv"], key="unlabeled_csv")
        if up2:
            X_unlabeled = pd.read_csv(up2)
            st.write("Prévia:", X_unlabeled.head(10))
            with st.spinner("Pontuando dados..."):
                scored = predict_model(st.session_state.best_model, data=X_unlabeled)
            st.success("Scoring concluído.")
            st.dataframe(scored.head(200), use_container_width=True)
            st.download_button(
                "⬇️ Baixar scoring (CSV)",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="scoring_sem_rotulo.csv",
                mime="text/csv"
            )

# =========================================================
# TAB 6 – Exportar
# =========================================================
with tabs[5]:
    st.subheader("Exportar Relatório HTML")
    if st.session_state.data is None or st.session_state.best_model is None:
        st.info("Execute o treino na aba **Treino & Métricas** primeiro.")
    else:
        notes = st.text_area("Notas/explicações adicionais (opcional)", value="", height=150)

        if st.button("🧾 Gerar Relatório Completo (HTML)", use_container_width=True):
            dataset_name = os.path.basename(local_path) if fonte.startswith("Local") else (uploaded_file.name if uploaded_file else "dataset.csv")
            lb = st.session_state.leaderboard if isinstance(st.session_state.leaderboard, pd.DataFrame) else None
            html_bytes = _export_html_report(
                dataset_name=dataset_name,
                target_col="class",  # Target fixo
                setup_params=st.session_state.setup_params,
                leaderboard_df=lb,
                plot_paths=st.session_state.plots,
                notes=notes
            )
            out_name = f"relatorio_classificacao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            out_path = os.path.join(EXPORTS_DIR, out_name)
            with open(out_path, "wb") as f:
                f.write(html_bytes)
            st.success(f"Relatório gerado: {out_name}")
            st.markdown(b64_download(html_bytes, out_name), unsafe_allow_html=True)
