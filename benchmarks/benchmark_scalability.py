#!/usr/bin/env python3
"""
Script de Benchmark para Testes de Escalabilidade
==================================================

Testa escalabilidade forte e fraca da pipeline paralela.

Métricas:
- Tempo de execução
- Speedup
- Eficiência

Uso:
    python benchmark_scalability.py --mode strong --images 100 --max-workers 8
    python benchmark_scalability.py --mode weak --images-per-worker 50 --max-workers 8
    python benchmark_scalability.py --mode both
"""

import argparse
import time
import os
import pickle
import json
import platform
import psutil
from pathlib import Path
from datetime import datetime
from parallel_pipeline import ParallelPipeline


def get_system_info():
    """Coleta informações do sistema"""
    return {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'python_version': platform.python_version(),
        'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2)
    }


def load_test_images(directory='val/real', max_images=None):
    """Carrega lista de imagens para teste"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    
    if not os.path.exists(directory):
        print(f"⚠️  Diretório {directory} não existe, tentando val/fake...")
        directory = 'val/fake'
    
    for file in os.listdir(directory):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            images.append(os.path.join(directory, file))
            if max_images and len(images) >= max_images:
                break
    
    return images


def benchmark_pipeline(images, model, scaler, num_workers):
    """
    Executa benchmark da pipeline
    
    Returns:
        dict com métricas de tempo
    """
    print(f"  Testando com {num_workers} worker(s) e {len(images)} imagens...")
    
    # Aquece (warm-up) - 3 imagens
    warmup_images = images[:min(3, len(images))]
    pipeline = ParallelPipeline(model=model, scaler=scaler)
    _ = pipeline.process_images(warmup_images, num_workers=num_workers)
    pipeline.stop()
    
    # Benchmark real
    start_time = time.time()
    pipeline = ParallelPipeline(model=model, scaler=scaler)
    results = pipeline.process_images(images, num_workers=num_workers)
    pipeline.stop()
    elapsed_time = time.time() - start_time
    
    # Contabiliza erros
    errors = sum(1 for r in results if 'error' in r)
    successful = len(results) - errors
    
    return {
        'num_workers': num_workers,
        'num_images': len(images),
        'elapsed_time': elapsed_time,
        'successful': successful,
        'errors': errors,
        'images_per_second': successful / elapsed_time if elapsed_time > 0 else 0
    }


def test_strong_scaling(images, model, scaler, max_workers=8):
    """
    Teste de Escalabilidade Forte
    
    Problema fixo, aumenta recursos (workers)
    """
    print(f"\n{'='*80}")
    print("TESTE DE ESCALABILIDADE FORTE (Strong Scaling)")
    print(f"{'='*80}")
    print(f"Imagens: {len(images)} (fixo)")
    print(f"Workers: 1 até {max_workers}")
    print(f"{'='*80}\n")
    
    results = []
    worker_counts = [1, 2, 4] + ([8] if max_workers >= 8 else [])
    worker_counts = [w for w in worker_counts if w <= max_workers]
    
    baseline_time = None
    
    for num_workers in worker_counts:
        result = benchmark_pipeline(images, model, scaler, num_workers)
        
        if baseline_time is None:
            baseline_time = result['elapsed_time']
            result['speedup'] = 1.0
            result['efficiency'] = 1.0
        else:
            result['speedup'] = baseline_time / result['elapsed_time']
            result['efficiency'] = result['speedup'] / num_workers
        
        results.append(result)
        
        print(f"    ✓ {num_workers} worker(s): {result['elapsed_time']:.2f}s "
              f"| Speedup: {result['speedup']:.2f}x "
              f"| Eficiência: {result['efficiency']:.1%}")
    
    return {
        'type': 'strong_scaling',
        'baseline_time': baseline_time,
        'results': results
    }


def test_weak_scaling(all_images, model, scaler, images_per_worker=50, max_workers=8):
    """
    Teste de Escalabilidade Fraca
    
    Problema cresce proporcionalmente aos recursos
    """
    print(f"\n{'='*80}")
    print("TESTE DE ESCALABILIDADE FRACA (Weak Scaling)")
    print(f"{'='*80}")
    print(f"Imagens por worker: {images_per_worker} (fixo)")
    print(f"Workers: 1 até {max_workers}")
    print(f"{'='*80}\n")
    
    results = []
    worker_counts = [1, 2, 4] + ([8] if max_workers >= 8 else [])
    worker_counts = [w for w in worker_counts if w <= max_workers]
    
    baseline_time = None
    
    for num_workers in worker_counts:
        # Número de imagens cresce com workers
        num_images = images_per_worker * num_workers
        
        if num_images > len(all_images):
            print(f"  ⚠️  Não há imagens suficientes para {num_workers} workers "
                  f"({num_images} necessárias, {len(all_images)} disponíveis)")
            break
        
        images = all_images[:num_images]
        result = benchmark_pipeline(images, model, scaler, num_workers)
        
        if baseline_time is None:
            baseline_time = result['elapsed_time']
            result['efficiency'] = 1.0
        else:
            # Em weak scaling, eficiência é T(1)/T(p)
            result['efficiency'] = baseline_time / result['elapsed_time']
        
        results.append(result)
        
        print(f"    ✓ {num_workers} worker(s) x {images_per_worker} imgs = {num_images} total: "
              f"{result['elapsed_time']:.2f}s | Eficiência: {result['efficiency']:.1%}")
    
    return {
        'type': 'weak_scaling',
        'images_per_worker': images_per_worker,
        'baseline_time': baseline_time,
        'results': results
    }


def print_summary(strong_results=None, weak_results=None):
    """Imprime resumo dos resultados"""
    print(f"\n{'='*80}")
    print("RESUMO DOS RESULTADOS")
    print(f"{'='*80}\n")
    
    if strong_results:
        print("📊 ESCALABILIDADE FORTE:")
        print(f"   Tempo baseline (1 worker): {strong_results['baseline_time']:.2f}s")
        best = max(strong_results['results'], key=lambda x: x['speedup'])
        print(f"   Melhor speedup: {best['speedup']:.2f}x com {best['num_workers']} workers")
        print(f"   Melhor eficiência: {max(r['efficiency'] for r in strong_results['results']):.1%}")
        print()
    
    if weak_results:
        print("📊 ESCALABILIDADE FRACA:")
        print(f"   Tempo baseline (1 worker): {weak_results['baseline_time']:.2f}s")
        print(f"   Imagens por worker: {weak_results['images_per_worker']}")
        worst = min(weak_results['results'], key=lambda x: x['efficiency'])
        print(f"   Pior eficiência: {worst['efficiency']:.1%} com {worst['num_workers']} workers")
        print()
    
    print("💡 RECOMENDAÇÕES:")
    if strong_results:
        # Encontra melhor configuração (eficiência > 70% e maior speedup)
        good_configs = [r for r in strong_results['results'] if r['efficiency'] > 0.7]
        if good_configs:
            best_config = max(good_configs, key=lambda x: x['speedup'])
            print(f"   - Para produção: {best_config['num_workers']} workers "
                  f"(speedup {best_config['speedup']:.2f}x, eficiência {best_config['efficiency']:.1%})")
        else:
            print(f"   - Para produção: 1-2 workers (melhor custo/benefício)")
    
    print(f"   - Para desenvolvimento: 1 worker (debug mais fácil)")
    print()


def save_results(output_file, system_info, strong_results=None, weak_results=None):
    """Salva resultados em JSON"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'system_info': system_info,
        'strong_scaling': strong_results,
        'weak_scaling': weak_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Resultados salvos em: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark de Escalabilidade da Pipeline Paralela',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s --mode strong --images 100 --max-workers 4
  %(prog)s --mode weak --images-per-worker 50 --max-workers 8
  %(prog)s --mode both --output results.json
        """
    )
    
    parser.add_argument('--mode', choices=['strong', 'weak', 'both'], default='both',
                        help='Tipo de teste de escalabilidade')
    parser.add_argument('--images', type=int, default=100,
                        help='Número de imagens para strong scaling')
    parser.add_argument('--images-per-worker', type=int, default=50,
                        help='Imagens por worker para weak scaling')
    parser.add_argument('--max-workers', type=int, default=8,
                        help='Número máximo de workers')
    parser.add_argument('--model', default='model.pkl',
                        help='Caminho para o modelo')
    parser.add_argument('--scaler', default='scaler.pkl',
                        help='Caminho para o scaler')
    parser.add_argument('--output', default='benchmark_results.json',
                        help='Arquivo de saída para resultados')
    parser.add_argument('--data-dir', default='val/real',
                        help='Diretório com imagens de teste')
    
    args = parser.parse_args()
    
    # Banner
    print(f"\n{'='*80}")
    print("🔬 BENCHMARK DE ESCALABILIDADE - PIPELINE PARALELA")
    print(f"{'='*80}")
    
    # Informações do sistema
    print("\n📋 INFORMAÇÕES DO SISTEMA:")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"   {key}: {value}")
    
    # Carrega modelo e scaler
    print(f"\n⚙️  Carregando modelo e scaler...")
    try:
        with open(args.model, 'rb') as f:
            model = pickle.load(f)
        with open(args.scaler, 'rb') as f:
            scaler = pickle.load(f)
        print("   ✓ Modelo e scaler carregados")
    except FileNotFoundError as e:
        print(f"   ❌ Erro: {e}")
        print("   Execute 'python main.py' para treinar o modelo primeiro")
        return 1
    
    # Carrega imagens
    max_images_needed = args.images
    if args.mode in ['weak', 'both']:
        max_images_needed = max(max_images_needed, args.images_per_worker * args.max_workers)
    
    print(f"\n📁 Carregando imagens de {args.data_dir}...")
    all_images = load_test_images(args.data_dir, max_images_needed)
    print(f"   ✓ {len(all_images)} imagens carregadas")
    
    if len(all_images) < args.images:
        print(f"   ⚠️  Apenas {len(all_images)} imagens disponíveis, "
              f"ajustando tamanho do teste")
        args.images = len(all_images)
    
    # Executa testes
    strong_results = None
    weak_results = None
    
    if args.mode in ['strong', 'both']:
        images = all_images[:args.images]
        strong_results = test_strong_scaling(images, model, scaler, args.max_workers)
    
    if args.mode in ['weak', 'both']:
        weak_results = test_weak_scaling(all_images, model, scaler, 
                                        args.images_per_worker, args.max_workers)
    
    # Resumo
    print_summary(strong_results, weak_results)
    
    # Salva resultados
    save_results(args.output, system_info, strong_results, weak_results)
    
    print(f"{'='*80}")
    print("✅ BENCHMARK COMPLETO!")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
