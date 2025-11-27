import streamlit as st
import pickle
import cv2
import numpy as np
import sys
import tempfile
import os
import pandas as pd
import shutil
import time
import plotly.express as px
from pathlib import Path
from PIL import Image

# Adiciona o diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.face_segmentation import FaceSegmenter
from src.feature_extraction import extract_features_from_array
from src.parallel_pipeline import ParallelPipeline  # Importando a pipeline paralela

# Caminhos dos modelos
MODELS_DIR = ROOT_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'model.pkl'
SCALER_PATH = MODELS_DIR / 'scaler.pkl'

# Configuração da página
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide"  # Layout wide para acomodar a tabela e gráficos do batch
)

# Título
st.title("🔍 Deepfake Detector")

# Carrega o modelo e scaler (com cache)
@st.cache_resource
def load_model_and_scaler():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Arquivo não encontrado: {e}. Execute `python scripts/main.py` primeiro.")
        return None, None

# Carrega segmentador
@st.cache_resource
def load_segmenter():
    return FaceSegmenter()

# --- FUNÇÃO DE CLASSIFICAÇÃO INDIVIDUAL ---
def classify_single_image(image, model, scaler, segmenter):
    img_array = np.array(image)
    # Verifica canais
    if len(img_array.shape) == 2:  # Grayscale
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:  # RGBA
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    face = segmenter.segment_face(img_bgr)
    
    if face is None:
        raise ValueError("Nenhuma face detectada na imagem")
    
    features = extract_features_from_array(face)
    features_scaled = scaler.transform([features])
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability, face

# --- FUNÇÃO DE CLASSIFICAÇÃO EM BATCH (PARALELA) ---
def process_batch_upload(uploaded_files, model, scaler):
    # Cria diretório temporário
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    file_map = {}  # path -> original filename

    try:
        # Salva arquivos no diretório temporário
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_paths.append(temp_path)
            file_map[temp_path] = uploaded_file.name
        
        # Inicializa pipeline paralela
        # Usa 2 workers por padrão para não travar a UI
        pipeline = ParallelPipeline(model=model, scaler=scaler)
        
        start_time = time.time()
        # Processa
        results = pipeline.process_images(image_paths, num_workers=2)
        pipeline.stop()
        elapsed_time = time.time() - start_time
        
        # Formata resultados
        processed_data = []
        for res in results:
            filename = file_map.get(res['path'], "Unknown")
            
            if 'error' in res:
                processed_data.append({
                    "Arquivo": filename,
                    "Predição": "Erro",
                    "Confiança": 0.0,
                    "Status": "❌ Erro",
                    "Detalhes": res['error']
                })
            elif 'prediction' in res:
                pred = res['prediction']
                conf = res['probability'][pred]
                label = "REAL" if pred == 1 else "FAKE"
                status = "🟩 REAL" if pred == 1 else "🟥 FAKE"
                
                processed_data.append({
                    "Arquivo": filename,
                    "Predição": label,
                    "Confiança": conf,
                    "Status": status,
                    "Detalhes": f"Prob. Fake: {res['probability'][0]:.1%}"
                })
            else:
                 processed_data.append({
                    "Arquivo": filename,
                    "Predição": "N/A",
                    "Confiança": 0.0,
                    "Status": "⚠️ Sem Predição",
                    "Detalhes": "Features extraídas mas sem predição"
                })
                
        return processed_data, elapsed_time
        
    finally:
        # Limpa diretório temporário
        shutil.rmtree(temp_dir)


# --- INTERFACE PRINCIPAL ---
model, scaler = load_model_and_scaler()

if model is not None and scaler is not None:
    segmenter = load_segmenter()
    
    # Abas
    tab1, tab2 = st.tabs(["🖼️ Classificação Individual", "📚 Classificação em Batch (Paralela)"])
    
    # --- ABA 1: INDIVIDUAL ---
    with tab1:
        st.header("Análise de Imagem Única")
        uploaded_file = st.file_uploader("Escolha uma imagem", type=['jpg', 'jpeg', 'png'], key="single")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Imagem Original", use_container_width=True)
            
            with st.spinner("Analisando..."):
                try:
                    prediction, probability, face = classify_single_image(image, model, scaler, segmenter)
                    
                    with col2:
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        st.image(face_rgb, caption="Face Detectada", use_container_width=True)
                    
                    st.markdown("---")
                    if prediction == 0:
                        st.error("🟥 **FAKE (Deepfake Detectado)**")
                        confidence = probability[0]
                    else:
                        st.success("🟩 **REAL (Imagem Autêntica)**")
                        confidence = probability[1]
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Classificação", "FAKE" if prediction == 0 else "REAL")
                    m2.metric("Confiança", f"{confidence:.1%}")
                    m3.metric("Fake Score", f"{probability[0]:.1%}")
                    
                    st.progress(float(confidence))
                    
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")

    # --- ABA 2: BATCH ---
    with tab2:
        st.header("Análise em Lote (Paralela)")
        st.markdown("Utilize a **Pipeline Paralela** para classificar múltiplas imagens simultaneamente.")
        
        uploaded_files = st.file_uploader("Escolha as imagens", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key="batch")
        
        if uploaded_files:
            st.info(f"📂 {len(uploaded_files)} imagens selecionadas.")
            
            if st.button("🚀 Iniciar Processamento Paralelo"):
                with st.spinner(f"Processando {len(uploaded_files)} imagens em paralelo..."):
                    
                    # Barra de progresso simulada (já que o pipeline bloqueia)
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01) # Feedback visual
                        progress_bar.progress(i + 1)
                    
                    results_data, elapsed_time = process_batch_upload(uploaded_files, model, scaler)
                    progress_bar.progress(100)
                    
                    if results_data:
                        df = pd.DataFrame(results_data)
                        
                        # Métricas Gerais
                        total = len(df)
                        real_count = len(df[df["Predição"] == "REAL"])
                        fake_count = len(df[df["Predição"] == "FAKE"])
                        errors = len(df[df["Predição"] == "Erro"])
                        
                        st.markdown("### 📊 Estatísticas do Lote")
                        
                        # Cards de Métricas
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Imagens", total)
                        c2.metric("⏱️ Tempo Total", f"{elapsed_time:.2f}s")
                        c3.metric("⚡ Velocidade", f"{total/elapsed_time:.1f} img/s")
                        c4.metric("Taxa de Erro", f"{errors/total:.1%}")
                        
                        col_chart, col_df = st.columns([1, 2])
                        
                        with col_chart:
                            # Gráfico de Rosca
                            st.markdown("#### Distribuição")
                            if total > 0:
                                fig = px.pie(
                                    names=["REAL", "FAKE", "Erro"],
                                    values=[real_count, fake_count, errors],
                                    color_discrete_map={"REAL": "green", "FAKE": "red", "Erro": "gray"},
                                    hole=0.5
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col_df:
                            st.markdown("#### Resultados Detalhados")
                            # Estilização da tabela
                            def color_status(val):
                                color = 'red' if 'FAKE' in val else 'green' if 'REAL' in val else 'black'
                                return f'color: {color}; font-weight: bold'
                            
                            st.dataframe(
                                df.style.applymap(color_status, subset=['Status']),
                                use_container_width=True,
                                height=400
                            )
                        
                        # Download CSV
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Baixar Relatório CSV",
                            csv,
                            "relatorio_deepfake.csv",
                            "text/csv",
                            key='download-csv'
                        )
else:
    st.warning("⚠️ Treine o modelo primeiro executando: `python scripts/main.py`")
