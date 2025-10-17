#!/usr/bin/env python3
"""
Visualização da pipeline paralelizada processando imagens
Mostra o progresso em tempo real de cada estágio
"""

import os
import time
import threading
from collections import deque
from parallel_pipeline import ParallelPipeline
import pickle


class PipelineMonitor:
    """Monitor para visualizar o progresso da pipeline em tempo real"""
    
    def __init__(self):
        self.segmentation_count = 0
        self.feature_count = 0
        self.prediction_count = 0
        self.total = 0
        self.lock = threading.Lock()
        self.running = False
    
    def update(self, stage):
        """Atualiza contadores por estágio"""
        with self.lock:
            if stage == 'segmentation':
                self.segmentation_count += 1
            elif stage == 'features':
                self.feature_count += 1
            elif stage == 'prediction':
                self.prediction_count += 1
    
    def display(self):
        """Exibe o progresso"""
        while self.running:
            with self.lock:
                seg = self.segmentation_count
                feat = self.feature_count
                pred = self.prediction_count
                total = self.total
            
            # Limpa a linha e imprime progresso
            print(f"\r🔄 Segmentação: {seg}/{total} | "
                  f"📊 Features: {feat}/{total} | "
                  f"🎯 Predição: {pred}/{total}", end='', flush=True)
            
            time.sleep(0.1)
    
    def start(self, total):
        """Inicia o monitor"""
        self.total = total
        self.running = True
        self.display_thread = threading.Thread(target=self.display)
        self.display_thread.start()
    
    def stop(self):
        """Para o monitor"""
        self.running = False
        if hasattr(self, 'display_thread'):
            self.display_thread.join()
        print()  # Nova linha após o progresso


def visualize_pipeline(image_paths, model=None, num_workers=2):
    """
    Processa imagens e visualiza o progresso
    
    Args:
        image_paths: lista de caminhos de imagens
        model: modelo para classificação (opcional)
        num_workers: número de workers por estágio
    """
    print("="*80)
    print("VISUALIZAÇÃO DA PIPELINE PARALELIZADA")
    print("="*80)
    print(f"\n📁 Total de imagens: {len(image_paths)}")
    print(f"👷 Workers por estágio: {num_workers}")
    print(f"🎯 Modo: {'Classificação completa' if model else 'Apenas extração de features'}")
    print("\n🚀 Iniciando processamento...\n")
    
    # Cria pipeline
    pipeline = ParallelPipeline(model=model)
    
    # Inicia monitoramento
    start_time = time.time()
    
    # Processa imagens
    results = pipeline.process_images(image_paths, num_workers=num_workers)
    
    elapsed = time.time() - start_time
    pipeline.stop()
    
    # Mostra resultados finais
    print(f"\n✅ Processamento concluído em {elapsed:.2f}s")
    print(f"⚡ Velocidade: {len(image_paths)/elapsed:.2f} imagens/segundo")
    print(f"📊 Tempo médio por imagem: {elapsed/len(image_paths):.3f}s")
    
    return results


def demo_visual():
    """Demo com visualização"""
    print("\n" + "🎨"*40)
    print("DEMONSTRAÇÃO VISUAL DA PIPELINE PARALELIZADA")
    print("🎨"*40)
    
    # Verifica modelo
    if not os.path.exists('model.pkl'):
        print("\n⚠️  Modelo não encontrado. Executando apenas extração de features.")
        print("   Execute 'python main.py' primeiro para ter classificação completa.\n")
        model = None
    else:
        print("\n✓ Modelo encontrado. Executando classificação completa.\n")
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    
    # Coleta imagens
    image_paths = []
    for class_dir in ['test/fake', 'test/real']:
        if os.path.exists(class_dir):
            for img in os.listdir(class_dir)[:5]:  # 5 de cada classe
                if img.endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, img))
    
    if not image_paths:
        print("❌ Nenhuma imagem encontrada em test/fake ou test/real")
        return
    
    # Visualiza processamento
    results = visualize_pipeline(image_paths, model=model, num_workers=2)
    
    # Mostra alguns resultados
    print("\n" + "="*80)
    print("AMOSTRA DE RESULTADOS (primeiras 5 imagens)")
    print("="*80)
    
    label_map = {0: '🟥 FAKE', 1: '🟩 REAL'}
    
    for i, result in enumerate(results[:5]):
        basename = os.path.basename(result['path'])
        print(f"\n[{i+1}] {basename}")
        
        if 'prediction' in result:
            pred = result['prediction']
            proba = result['probability']
            print(f"   Resultado: {label_map[pred]}")
            print(f"   Confiança: {proba[pred]:.1%}")
            
            # Barra de confiança visual
            bar_length = 30
            filled = int(bar_length * proba[pred])
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"   [{bar}] {proba[pred]:.1%}")
        
        elif 'features' in result and result['features'] is not None:
            print(f"   ✓ {len(result['features'])} features extraídas")
        
        elif 'error' in result:
            print(f"   ❌ Erro: {result['error']}")
    
    # Estatísticas finais
    if model:
        print("\n" + "="*80)
        print("📊 ESTATÍSTICAS GERAIS")
        print("="*80)
        
        fake_count = sum(1 for r in results if r.get('prediction') == 0)
        real_count = sum(1 for r in results if r.get('prediction') == 1)
        error_count = sum(1 for r in results if 'error' in r)
        
        total = len(results)
        print(f"\n   Total processado: {total} imagens")
        print(f"   🟥 FAKE: {fake_count} ({fake_count/total*100:.1f}%)")
        print(f"   🟩 REAL: {real_count} ({real_count/total*100:.1f}%)")
        if error_count > 0:
            print(f"   ⚠️  Erros: {error_count}")
        
        # Distribuição de confiança
        confidences = [r['probability'][r['prediction']] for r in results if 'prediction' in r]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            
            print(f"\n   Confiança média: {avg_conf:.1%}")
            print(f"   Confiança mínima: {min_conf:.1%}")
            print(f"   Confiança máxima: {max_conf:.1%}")
    
    print("\n" + "="*80)
    print("✨ FIM DA DEMONSTRAÇÃO")
    print("="*80)
    print("\n💡 Dica: Use 'python inference.py <diretório>' para processar mais imagens!")


def compare_sequential_vs_parallel():
    """Compara processamento sequencial vs paralelo"""
    print("\n" + "⚡"*40)
    print("COMPARAÇÃO: SEQUENCIAL vs PARALELO")
    print("⚡"*40)
    
    if not os.path.exists('model.pkl'):
        print("\n⚠️  Necessário modelo treinado para esta demo.")
        return
    
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Coleta imagens
    image_paths = []
    test_dir = 'test/fake'
    if os.path.exists(test_dir):
        for img in os.listdir(test_dir)[:10]:
            if img.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(test_dir, img))
    
    if len(image_paths) < 5:
        print("\n⚠️  Poucas imagens disponíveis para comparação")
        return
    
    print(f"\n📁 Processando {len(image_paths)} imagens...\n")
    
    # Teste sequencial (1 worker)
    print("1️⃣  Modo SEQUENCIAL (1 worker):")
    start = time.time()
    results_seq = visualize_pipeline(image_paths, model=model, num_workers=1)
    time_seq = time.time() - start
    
    print("\n")
    
    # Teste paralelo (2 workers)
    print("2️⃣  Modo PARALELO (2 workers):")
    start = time.time()
    results_par = visualize_pipeline(image_paths, model=model, num_workers=2)
    time_par = time.time() - start
    
    # Comparação
    print("\n" + "="*80)
    print("📊 COMPARAÇÃO DE PERFORMANCE")
    print("="*80)
    print(f"\n   Sequencial (1 worker):  {time_seq:.2f}s ({time_seq/len(image_paths):.3f}s/imagem)")
    print(f"   Paralelo (2 workers):   {time_par:.2f}s ({time_par/len(image_paths):.3f}s/imagem)")
    
    speedup = time_seq / time_par
    improvement = (1 - time_par/time_seq) * 100
    
    print(f"\n   ⚡ Speedup: {speedup:.2f}x")
    print(f"   📈 Melhoria: {improvement:.1f}% mais rápido")
    
    if speedup > 1.5:
        print("\n   🎉 Excelente ganho com paralelização!")
    elif speedup > 1.1:
        print("\n   ✓ Bom ganho com paralelização")
    else:
        print("\n   ℹ️  Ganho modesto (pode ser devido ao overhead de threading)")


def main():
    """Menu interativo"""
    print("\n" + "="*80)
    print("🎬 VISUALIZAÇÃO DA PIPELINE PARALELIZADA")
    print("="*80)
    print("\nEscolha uma opção:")
    print("  1. Demonstração visual completa")
    print("  2. Comparação sequencial vs paralelo")
    print("  3. Ambos")
    print("  0. Sair")
    
    try:
        choice = input("\nOpção: ").strip()
        
        if choice == '1':
            demo_visual()
        elif choice == '2':
            compare_sequential_vs_parallel()
        elif choice == '3':
            demo_visual()
            input("\n⏸️  Pressione Enter para continuar para a comparação...")
            compare_sequential_vs_parallel()
        elif choice == '0':
            print("\n👋 Até logo!")
        else:
            print("\n❌ Opção inválida")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")


if __name__ == '__main__':
    main()
