"""Teste da pipeline com uma imagem"""
import pickle
from parallel_pipeline import ParallelPipeline

# Carrega modelo e scaler
print("Carregando modelo e scaler...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✓ Modelo e scaler carregados")

# Imagem de teste
image_path = "val/real/00000026_30.png"

print(f"\nProcessando imagem: {image_path}")
pipeline = ParallelPipeline(model=model, scaler=scaler)
results = pipeline.process_images([image_path], num_workers=1)  # Usar apenas 1 worker
pipeline.stop()

print(f"\nResultados: {len(results)}")
if results:
    result = results[0]
    print(f"Resultado: {result}")
    if 'prediction' in result:
        print(f"\n✓ Predição: {result['prediction']} (0=fake, 1=real)")
        print(f"✓ Probabilidades: fake={result['probability'][0]:.4f}, real={result['probability'][1]:.4f}")
    else:
        print(f"\n❌ Sem predição")
        if 'error' in result:
            print(f"Erro: {result['error']}")
else:
    print("❌ Nenhum resultado retornado!")
