"""Script para debug de uma única imagem"""
import cv2
import numpy as np
from face_segmentation import FaceSegmenter
from feature_extraction import extract_features_from_array
import pickle

# Testa a imagem
image_path = "val/real/00000026_30.png"

print(f"1. Carregando imagem: {image_path}")
img = cv2.imread(image_path)
if img is None:
    print("   ❌ Falha ao carregar imagem!")
    exit(1)
print(f"   ✓ Imagem carregada: shape={img.shape}")

print("\n2. Segmentando face...")
segmenter = FaceSegmenter()
face = segmenter.segment_face(image_path)
if face is None:
    print("   ❌ Nenhuma face detectada!")
    print("   Verificando se há faces na imagem...")
    # Tenta detectar diretamente
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = segmenter.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"   Número de faces detectadas: {len(faces)}")
    if len(faces) > 0:
        print(f"   Faces: {faces}")
    exit(1)
print(f"   ✓ Face segmentada: shape={face.shape}")

print("\n3. Extraindo features...")
features = extract_features_from_array(face)
if features is None:
    print("   ❌ Falha ao extrair features!")
    exit(1)
print(f"   ✓ Features extraídas: shape={features.shape}, dtype={features.dtype}")
print(f"   Features min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")

print("\n4. Carregando scaler...")
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print(f"   ✓ Scaler carregado")

print("\n5. Normalizando features...")
features_reshaped = features.reshape(1, -1)
print(f"   Features antes: {features_reshaped[0][:5]}...")
features_normalized = scaler.transform(features_reshaped)
print(f"   Features depois: {features_normalized[0][:5]}...")

print("\n6. Carregando modelo...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
print(f"   ✓ Modelo carregado")

print("\n7. Fazendo predição...")
prediction = model.predict(features_normalized)
probabilities = model.predict_proba(features_normalized)
print(f"   ✓ Predição: {prediction[0]} (0=fake, 1=real)")
print(f"   ✓ Probabilidades: fake={probabilities[0][0]:.4f}, real={probabilities[0][1]:.4f}")

print("\n✅ Teste completo com sucesso!")
