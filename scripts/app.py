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

import logging

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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
st.title("Deepfake Detector")

# Carrega o modelo e scaler (com cache)
@st.cache_resource
def load_model_and_scaler():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Modelo e Scaler carregados com sucesso.")
        return model, scaler
    except FileNotFoundError as e:
        logger.error(f"Erro ao carregar modelos: {e}")
        st.error(f"❌ Arquivo não encontrado: {e}. Execute `python scripts/main.py` primeiro.")
        return None, None

# Carrega segmentador
@st.cache_resource
def load_segmenter():
    logger.info("Carregando FaceSegmenter...")
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

import zipfile

# --- FUNÇÃO AUXILIAR DE UPLOAD (ZIP & IMAGENS) ---
def save_and_extract_files(uploaded_files, temp_dir):
    """
    Salva arquivos enviados e extrai ZIPs se houver.
    Retorna lista de caminhos de imagens encontradas.
    """
    image_paths = []
    file_map = {}
    
    logger.info(f"Processando upload de {len(uploaded_files)} arquivos/zips...")
    
    for uploaded_file in uploaded_files:
        # Se for ZIP
        if uploaded_file.name.lower().endswith('.zip'):
            logger.info(f"Extraindo ZIP: {uploaded_file.name}")
            zip_path = os.path.join(temp_dir, uploaded_file.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                # Opcional: remover o zip original para limpar
                os.remove(zip_path)
            except zipfile.BadZipFile:
                logger.error(f"Arquivo ZIP inválido: {uploaded_file.name}")
                continue
        
        # Se for Imagem
        else:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    # Varre o diretório recursivamente buscando imagens
    # (Isso pega tanto as imagens soltas quanto as que estavam dentro do ZIP)
    logger.info("Varrendo diretório temporário em busca de imagens...")
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(valid_extensions) and not file.startswith('.'):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
                file_map[full_path] = file  # Mapeia caminho completo -> nome do arquivo
    
    logger.info(f"Total de imagens encontradas: {len(image_paths)}")
    return image_paths, file_map

# --- FUNÇÃO DE CLASSIFICAÇÃO EM BATCH (PARALELA) ---
def process_batch_upload(uploaded_files, model, scaler):
    logger.info(f"Iniciando processamento batch.")
    # Cria diretório temporário
    temp_dir = tempfile.mkdtemp()

    try:
        # Usa a nova função auxiliar
        image_paths, file_map = save_and_extract_files(uploaded_files, temp_dir)
        
        if not image_paths:
            st.warning("Nenhuma imagem válida encontrada nos arquivos enviados.")
            return [], 0

        # Inicializa pipeline paralela
        pipeline = ParallelPipeline(model=model, scaler=scaler)
        
        start_time = time.time()
        # Processa
        logger.info("Executando pipeline com 2 workers...")
        results = pipeline.process_images(image_paths, num_workers=2)
        pipeline.stop()
        elapsed_time = time.time() - start_time
        logger.info(f"Processamento concluído em {elapsed_time:.2f}s")
        
        # Formata resultados
        processed_data = []
        for res in results:
            filename = file_map.get(res['path'], os.path.basename(res['path']))
            
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

# --- FUNÇÃO DE BENCHMARK DE ESCALABILIDADE ---
def run_scalability_benchmark(uploaded_files, model, scaler):
    logger.info(f"Iniciando Benchmark de Escalabilidade.")
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Usa a nova função auxiliar
        image_paths, _ = save_and_extract_files(uploaded_files, temp_dir)
        
        if not image_paths:
            st.warning("Nenhuma imagem válida encontrada para benchmark.")
            return None
        
        pipeline = ParallelPipeline(model=model, scaler=scaler)
        thread_counts = [1, 2, 4, 8]
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        base_time = None
        
        for i, n_threads in enumerate(thread_counts):
            msg = f"Iniciando iteração de benchmark: {n_threads} threads..."
            logger.info(msg)
            status_text.markdown(f"🔄 Rodando teste com **{n_threads} threads** ({len(image_paths)} imagens)...")
            
            start_time = time.time()
            try:
                _ = pipeline.process_images(image_paths, num_workers=n_threads)
            except Exception as e:
                logger.error(f"ERRO CRÍTICO na execução com {n_threads} threads: {e}", exc_info=True)
                st.error(f"Erro na execução com {n_threads} threads: {e}")
                raise e

            elapsed = time.time() - start_time
            logger.info(f"Finalizado {n_threads} threads em {elapsed:.4f}s")
            
            if n_threads == 1:
                base_time = elapsed
                speedup = 1.0
            else:
                speedup = base_time / elapsed if elapsed > 0 else 0.0
            
            efficiency = speedup / n_threads
            
            results.append({
                "Threads": n_threads,
                "Tempo (s)": elapsed,
                "Speedup": speedup,
                "Eficiência": efficiency
            })
            
            progress_bar.progress((i + 1) / len(thread_counts))
            
        pipeline.stop()
        logger.info("Benchmark completo com sucesso.")
        status_text.text("✅ Benchmark concluído!")
        return pd.DataFrame(results)

    except Exception as e:
        logger.error(f"Erro geral no benchmark: {e}", exc_info=True)
        st.error(f"Falha no benchmark: {e}")
        return None

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)



# --- INTERFACE PRINCIPAL ---
model, scaler = load_model_and_scaler()

if model is not None and scaler is not None:
    segmenter = load_segmenter()
    
    # Abas
    tab1, tab2 = st.tabs(["Classificação Individual", "Classificação em Batch (Parale la)"])
    
    # --- ABA 1: INDIVIDUAL ---
    with tab1:
        st.header("Análise de Imagem Única")
        uploaded_file = st.file_uploader("Escolha uma imagem", type=['jpg', 'jpeg', 'png'], key="single")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2, col3, col4 = st.columns([0.5, 0.5, 0.2, 0.5])
            
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
        st.header("Processamento em Lote e Escalabilidade")
        
        uploaded_files = st.file_uploader(
            "Escolha as imagens ou arquivo ZIP", 
            type=['jpg', 'jpeg', 'png', 'zip'], # ACEITA ZIP AGORA
            accept_multiple_files=True, 
            key="batch"
        )
        
        if uploaded_files:
            # Pré-cálculo da quantidade real de imagens (abrindo ZIPs se necessário)
            total_images_count = 0
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
            
            for up_file in uploaded_files:
                if up_file.name.lower().endswith('.zip'):
                    try:
                        with zipfile.ZipFile(up_file) as z:
                            # Conta arquivos dentro do zip que têm extensões válidas
                            # Filtra __MACOSX e arquivos ocultos para contagem precisa
                            count = sum(1 for f in z.namelist() if f.lower().endswith(valid_extensions) and not f.startswith('__MACOSX') and not os.path.basename(f).startswith('.'))
                            total_images_count += count
                    except Exception as e:
                        logger.warning(f"Erro ao pré-ler ZIP: {e}")
                else:
                    total_images_count += 1
            
            st.info(f"📂 {total_images_count} imagens identificadas (incluindo conteúdo de ZIPs).")
            
            col_btn1, col_btn2 = st.columns(2)
            
            # Botão 1: Classificação Normal (agora com 8 threads)
            run_classification = col_btn1.button("Classificar Imagens (8 threads)")
            
            # Botão 2: Teste de Escalabilidade
            run_benchmark = col_btn2.button("⚡ Executar Teste de Escalabilidade Forte (1, 2, 4, 8 Threads)")
            
            # --- LÓGICA CLASSIFICAÇÃO ---
            if run_classification:
                with st.spinner(f"Processando {total_images_count} imagens..."):
                    # Altera para usar 8 workers
                    results_data, elapsed_time = process_batch_upload(uploaded_files, model, scaler, num_workers=8)
                    
                    if results_data:
                        df = pd.DataFrame(results_data)
                        total = len(df)
                        real_count = len(df[df["Predição"] == "REAL"])
                        fake_count = len(df[df["Predição"] == "FAKE"])
                        errors = len(df[df["Predição"] == "Erro"])
                        
                        st.markdown("### Estatísticas do Lote")
                        
                        # Cards de Métricas
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total", total)
                        c2.metric("Tempo", f"{elapsed_time:.2f}s")
                        c3.metric("Throughput", f"{total/elapsed_time:.1f} img/s")
                        c4.metric("Erros", errors)
                        
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
            
            # --- LÓGICA BENCHMARK ---
            if run_benchmark:
                st.markdown("### ⚡ Resultados do Teste de Escalabilidade")
                df_bench = run_scalability_benchmark(uploaded_files, model, scaler)
                
                if df_bench is not None:
                    # Exibir Tabela
                    st.dataframe(df_bench.style.format("{:.2f}", subset=["Tempo (s)", "Speedup", "Eficiência"]))
                    
                    # Gráficos Lado a Lado
                    g1, g2 = st.columns(2)
                    
                    with g1:
                        fig_time = px.line(
                            df_bench, x="Threads", y="Tempo (s)", 
                            markers=True, title="Tempo de Execução vs Threads (Menor é melhor)"
                        )
                        st.plotly_chart(fig_time, use_container_width=True)
                        
                    with g2:
                        # Adiciona linha ideal
                        df_bench['Speedup Ideal'] = df_bench['Threads']
                        fig_speed = px.line(
                            df_bench, x="Threads", y=["Speedup", "Speedup Ideal"], 
                            markers=True, title="Speedup vs Threads (Maior é melhor)"
                        )
                        st.plotly_chart(fig_speed, use_container_width=True)
                    
                    st.success(f"Teste finalizado! O speedup máximo alcançado foi de {df_bench['Speedup'].max():.2f}x com {df_bench.loc[df_bench['Speedup'].idxmax(), 'Threads']} threads.")
else:
    st.warning("⚠️ Treine o modelo primeiro executando: `python scripts/main.py`")
