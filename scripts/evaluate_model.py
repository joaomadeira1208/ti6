"""
Script para avaliar o modelo e gerar métricas completas
"""
import pickle
import json
from parallel_pipeline import ParallelPipeline
from sklearn.metrics import classification_report, confusion_matrix
import os

print("="*80)
print("AVALIAÇÃO COMPLETA DO MODELO")
print("="*80)

# Carrega modelo e scaler
print("\n1. Carregando modelo e scaler...")
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("   ✓ Modelo e scaler carregados")

# Coleta todas as imagens de validação
print("\n2. Coletando imagens de validação...")
val_fake_dir = "data/val/fake"
val_real_dir = "data/val/real"

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

fake_images = []
for root, dirs, files in os.walk(val_fake_dir):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            fake_images.append(os.path.join(root, file))

real_images = []
for root, dirs, files in os.walk(val_real_dir):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            real_images.append(os.path.join(root, file))

print(f"   ✓ {len(fake_images)} imagens fake")
print(f"   ✓ {len(real_images)} imagens real")
print(f"   ✓ Total: {len(fake_images) + len(real_images)} imagens")

# Processa todas as imagens em batches para mostrar progresso
print("\n3. Processando imagens...")
pipeline = ParallelPipeline(model=model, scaler=scaler)

def process_in_batches(image_list, batch_size=100, label=""):
    results = []
    total = len(image_list)
    for i in range(0, total, batch_size):
        batch = image_list[i:i+batch_size]
        print(f"   Processing {label} batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({min(i+batch_size, total)}/{total})...")
        # Usando 1 worker para garantir estabilidade e evitar deadlocks no macOS
        batch_results = pipeline.process_images(batch, num_workers=1)
        results.extend(batch_results)
    return results

print("   --- Iniciando processamento FAKE ---")
fake_results = process_in_batches(fake_images, batch_size=200, label="FAKE")

print("\n   --- Iniciando processamento REAL ---")
real_results = process_in_batches(real_images, batch_size=200, label="REAL")

pipeline.stop()
print("\n   ✓ Processamento concluído")

# Coleta predições e labels verdadeiros
y_true = []
y_pred = []

# Fake = 0
for result in fake_results:
    if 'prediction' in result:
        y_true.append(0)
        y_pred.append(result['prediction'])

# Real = 1
for result in real_results:
    if 'prediction' in result:
        y_true.append(1)
        y_pred.append(result['prediction'])

print(f"\n4. Imagens processadas com sucesso: {len(y_true)}")
print(f"   - Fake processadas: {sum(1 for r in fake_results if 'prediction' in r)}/{len(fake_results)}")
print(f"   - Real processadas: {sum(1 for r in real_results if 'prediction' in r)}/{len(real_results)}")

# Calcula métricas
print("\n" + "="*80)
print("MÉTRICAS DE DESEMPENHO")
print("="*80)

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
print("\nMatriz de Confusão:")
print("                Predito")
print("                FAKE    REAL")
print(f"Real FAKE      {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"     REAL      {cm[1][0]:4d}    {cm[1][1]:4d}")

# Métricas detalhadas
print("\n" + "-"*80)
print(classification_report(y_true, y_pred, target_names=['fake', 'real'], digits=4))

# Estatísticas adicionais
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision_fake = tn / (tn + fn) if (tn + fn) > 0 else 0
precision_real = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_fake = tn / (tn + fp) if (tn + fp) > 0 else 0
recall_real = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_fake = 2 * (precision_fake * recall_fake) / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0
f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0

print("\n" + "="*80)
print("ANÁLISE DETALHADA")
print("="*80)

print(f"\n📊 Acurácia Geral: {accuracy:.2%}")
print(f"\n🎯 Classe FAKE (0):")
print(f"   - Verdadeiros Negativos (TN): {tn}")
print(f"   - Falsos Negativos (FN): {fn}")
print(f"   - Precisão: {precision_fake:.2%} (de todas as predições FAKE, quantas estavam corretas)")
print(f"   - Recall: {recall_fake:.2%} (de todas as imagens FAKE reais, quantas foram detectadas)")
print(f"   - F1-Score: {f1_fake:.2%}")

print(f"\n🎯 Classe REAL (1):")
print(f"   - Verdadeiros Positivos (TP): {tp}")
print(f"   - Falsos Positivos (FP): {fp}")
print(f"   - Precisão: {precision_real:.2%} (de todas as predições REAL, quantas estavam corretas)")
print(f"   - Recall: {recall_real:.2%} (de todas as imagens REAL reais, quantas foram detectadas)")
print(f"   - F1-Score: {f1_real:.2%}")

print(f"\n⚖️ Balanço:")
print(f"   - Diferença de Recall: {abs(recall_fake - recall_real):.2%}")
print(f"   - Diferença de Precisão: {abs(precision_fake - precision_real):.2%}")

# Análise de viés
if recall_fake > recall_real:
    bias_direction = "FAKE"
    bias_amount = recall_fake - recall_real
else:
    bias_direction = "REAL"
    bias_amount = recall_real - recall_fake

print(f"\n⚠️ Viés do Modelo:")
print(f"   O modelo tem viés de {bias_amount:.2%} para classificar como {bias_direction}")

if bias_amount > 0.15:
    print("   🔴 ATENÇÃO: Viés significativo detectado (>15%)")
elif bias_amount > 0.10:
    print("   🟡 Viés moderado detectado (>10%)")
else:
    print("   🟢 Viés aceitável (<10%)")

# Erros de processamento
fake_errors = sum(1 for r in fake_results if 'error' in r)
real_errors = sum(1 for r in real_results if 'error' in r)
total_errors = fake_errors + real_errors

print(f"\n❌ Erros de Processamento:")
print(f"   - Fake: {fake_errors}/{len(fake_results)} ({fake_errors/len(fake_results)*100:.1f}%)")
print(f"   - Real: {real_errors}/{len(real_results)} ({real_errors/len(real_results)*100:.1f}%)")
print(f"   - Total: {total_errors}/{len(fake_results)+len(real_results)} ({total_errors/(len(fake_results)+len(real_results))*100:.1f}%)")

print("\n" + "="*80)
print("FIM DA AVALIAÇÃO")
print("="*80)
