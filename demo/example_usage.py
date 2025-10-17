#!/usr/bin/env python3
"""
Script de exemplo demonstrando o uso da pipeline paralelizada
"""

import os
import time
from parallel_pipeline import ParallelPipeline, process_images_parallel
import pickle


def exemplo_basico():
    """Exemplo básico: processar algumas imagens sem modelo (apenas features)"""
    print("="*80)
    print("EXEMPLO 1: Extração de Features (sem classificação)")
    print("="*80)
    
    # Pega algumas imagens de teste
    image_paths = []
    test_dir = 'test/fake'
    if os.path.exists(test_dir):
        for img in os.listdir(test_dir)[:3]:  # Apenas 3 imagens
            if img.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(test_dir, img))
    
    if not image_paths:
        print("⚠️  Nenhuma imagem encontrada em test/fake/")
        return
    
    print(f"\nProcessando {len(image_paths)} imagens...")
    start_time = time.time()
    
    # Processa sem modelo (apenas extrai features)
    results = process_images_parallel(image_paths, model=None, num_workers=2)
    
    elapsed = time.time() - start_time
    
    # Mostra resultados
    for i, result in enumerate(results):
        print(f"\n[{i+1}] {result['path']}")
        if 'features' in result and result['features'] is not None:
            print(f"   ✓ Features extraídas: {len(result['features'])} dimensões")
            print(f"   ✓ Primeiras 5 features: {result['features'][:5]}")
    
    print(f"\n⏱️  Tempo total: {elapsed:.2f}s ({elapsed/len(image_paths):.2f}s por imagem)")


def exemplo_com_modelo():
    """Exemplo com modelo: classificação completa"""
    print("\n" + "="*80)
    print("EXEMPLO 2: Classificação Completa com Modelo")
    print("="*80)
    
    # Verifica se modelo existe
    if not os.path.exists('model.pkl'):
        print("⚠️  Modelo não encontrado. Execute 'python main.py' primeiro para treinar.")
        return
    
    # Carrega o modelo
    print("\nCarregando modelo...")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Pega imagens de fake e real
    image_paths = []
    for class_dir in ['test/fake', 'test/real']:
        if os.path.exists(class_dir):
            for img in os.listdir(class_dir)[:2]:  # 2 de cada classe
                if img.endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, img))
    
    if not image_paths:
        print("⚠️  Nenhuma imagem encontrada em test/")
        return
    
    print(f"\nProcessando {len(image_paths)} imagens com classificação...")
    start_time = time.time()
    
    # Processa com modelo
    results = process_images_parallel(image_paths, model=model, num_workers=2)
    
    elapsed = time.time() - start_time
    
    # Mostra resultados
    label_map = {0: 'FAKE', 1: 'REAL'}
    for i, result in enumerate(results):
        print(f"\n[{i+1}] {result['path']}")
        if 'prediction' in result:
            pred = result['prediction']
            proba = result['probability']
            print(f"   🎯 Predição: {label_map[pred]} (classe {pred})")
            print(f"   📊 Confiança: {proba[pred]:.2%}")
            print(f"   📈 Probabilidades: FAKE={proba[0]:.2%}, REAL={proba[1]:.2%}")
        elif 'error' in result:
            print(f"   ❌ Erro: {result['error']}")
    
    print(f"\n⏱️  Tempo total: {elapsed:.2f}s ({elapsed/len(image_paths):.2f}s por imagem)")
    
    # Estatísticas
    fake_count = sum(1 for r in results if r.get('prediction') == 0)
    real_count = sum(1 for r in results if r.get('prediction') == 1)
    
    print(f"\n📊 ESTATÍSTICAS:")
    print(f"   Total: {len(results)} imagens")
    print(f"   FAKE: {fake_count} ({fake_count/len(results)*100:.1f}%)")
    print(f"   REAL: {real_count} ({real_count/len(results)*100:.1f}%)")


def exemplo_comparacao_workers():
    """Exemplo comparando diferentes números de workers"""
    print("\n" + "="*80)
    print("EXEMPLO 3: Comparação de Performance (diferentes números de workers)")
    print("="*80)
    
    if not os.path.exists('model.pkl'):
        print("⚠️  Modelo não encontrado. Execute 'python main.py' primeiro.")
        return
    
    # Carrega modelo
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Pega mais imagens para teste
    image_paths = []
    test_dir = 'test/fake'
    if os.path.exists(test_dir):
        for img in os.listdir(test_dir)[:10]:  # 10 imagens
            if img.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(test_dir, img))
    
    if len(image_paths) < 5:
        print("⚠️  Poucas imagens disponíveis para teste de performance")
        return
    
    print(f"\nTestando com {len(image_paths)} imagens...\n")
    
    # Testa com diferentes números de workers
    for num_workers in [1, 2, 4]:
        print(f"🔧 Testando com {num_workers} worker(s) por estágio...")
        
        start_time = time.time()
        results = process_images_parallel(image_paths, model=model, num_workers=num_workers)
        elapsed = time.time() - start_time
        
        print(f"   ⏱️  Tempo: {elapsed:.2f}s ({elapsed/len(image_paths):.3f}s por imagem)")
        print(f"   ✓ Processadas: {len([r for r in results if 'prediction' in r])} imagens\n")


def exemplo_pipeline_objeto():
    """Exemplo usando objeto Pipeline diretamente"""
    print("\n" + "="*80)
    print("EXEMPLO 4: Uso do Objeto Pipeline Diretamente")
    print("="*80)
    
    if not os.path.exists('model.pkl'):
        print("⚠️  Modelo não encontrado.")
        return
    
    # Carrega modelo
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Cria pipeline
    pipeline = ParallelPipeline(model=model, max_queue_size=5)
    
    # Pega algumas imagens
    image_paths = []
    test_dir = 'test/fake'
    if os.path.exists(test_dir):
        for img in os.listdir(test_dir)[:3]:
            if img.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(test_dir, img))
    
    if not image_paths:
        print("⚠️  Nenhuma imagem encontrada")
        pipeline.stop()
        return
    
    print(f"\nProcessando {len(image_paths)} imagens com objeto Pipeline...")
    
    # Processa
    results = pipeline.process_images(image_paths, num_workers=2)
    
    # Mostra resultados
    for i, result in enumerate(results):
        print(f"\n[{i+1}] {os.path.basename(result['path'])}")
        if 'prediction' in result:
            label = 'FAKE' if result['prediction'] == 0 else 'REAL'
            conf = result['probability'][result['prediction']]
            print(f"   → {label} (confiança: {conf:.2%})")
    
    # Para a pipeline
    pipeline.stop()
    print("\n✓ Pipeline finalizada")


def main():
    """Executa todos os exemplos"""
    print("\n🚀 EXEMPLOS DE USO DA PIPELINE PARALELIZADA\n")
    
    # Exemplo 1: Apenas features
    exemplo_basico()
    
    # Exemplo 2: Com modelo
    exemplo_com_modelo()
    
    # Exemplo 3: Comparação de workers
    exemplo_comparacao_workers()
    
    # Exemplo 4: Uso direto do objeto
    exemplo_pipeline_objeto()
    
    print("\n" + "="*80)
    print("✅ TODOS OS EXEMPLOS CONCLUÍDOS!")
    print("="*80)
    print("\nPara mais informações, consulte PIPELINE_README.md")


if __name__ == '__main__':
    main()
