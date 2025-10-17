import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image
from face_segmentation import FaceSegmenter
from feature_extraction import extract_features_from_array

# Configuração da página
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="centered"
)

# Título
st.title("🔍 Deepfake Detector")
st.markdown("Envie uma imagem para verificar se é real ou fake")

# Carrega o modelo e scaler (com cache)
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Arquivo não encontrado: {e}. Execute `python main.py` primeiro.")
        return None, None

# Carrega segmentador
@st.cache_resource
def load_segmenter():
    return FaceSegmenter()

# Processa a imagem
def classify_image(image, model, scaler, segmenter):
    # Converte PIL para OpenCV
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Segmenta face
    face = segmenter.segment_face(img_bgr)
    
    # Extrai features
    features = extract_features_from_array(face)
    
    # Normaliza features
    features_scaled = scaler.transform([features])
    
    # Classifica
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability, face

# Interface principal
model, scaler = load_model_and_scaler()

if model is not None and scaler is not None:
    segmenter = load_segmenter()
    
    # Upload de imagem
    uploaded_file = st.file_uploader("Escolha uma imagem", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Mostra a imagem
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Imagem Original", use_column_width=True)
        
        # Classifica
        with st.spinner("Analisando..."):
            try:
                prediction, probability, face = classify_image(image, model, scaler, segmenter)
                
                # Mostra face detectada
                with col2:
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    st.image(face_rgb, caption="Face Detectada", use_column_width=True)
                
                # Resultado
                st.markdown("---")
                
                if prediction == 0:
                    st.error("🟥 **FAKE (Deepfake Detectado)**")
                    confidence = probability[0]
                else:
                    st.success("🟩 **REAL (Imagem Autêntica)**")
                    confidence = probability[1]
                
                # Métricas
                col1, col2, col3 = st.columns(3)
                col1.metric("Classificação", "FAKE" if prediction == 0 else "REAL")
                col2.metric("Confiança", f"{confidence:.1%}")
                col3.metric("Fake Score", f"{probability[0]:.1%}")
                
                # Barra de progresso
                st.progress(float(confidence))
                
                # Detalhes
                with st.expander("📊 Detalhes da Classificação"):
                    st.write(f"**Probabilidade FAKE:** {probability[0]:.2%}")
                    st.write(f"**Probabilidade REAL:** {probability[1]:.2%}")
                    st.write(f"**Classe predita:** {prediction}")
                
            except Exception as e:
                st.error(f"❌ Erro ao processar imagem: {str(e)}")
    
    else:
        # Instruções
        st.info("👆 Faça upload de uma imagem para começar")
        
        with st.expander("ℹ️ Como usar"):
            st.markdown("""
            1. Clique em "Browse files" acima
            2. Selecione uma imagem (JPG, JPEG ou PNG)
            3. Aguarde a análise
            4. Veja o resultado: FAKE ou REAL
            
            **Nota:** A imagem deve conter um rosto visível.
            """)

else:
    st.warning("⚠️ Treine o modelo primeiro executando: `python main.py`")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido com ❤️ usando Streamlit")
