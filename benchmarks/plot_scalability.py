#!/usr/bin/env python3
"""
Script de Visualização de Resultados de Escalabilidade
========================================================

Gera gráficos e tabelas dos resultados de benchmark.

Uso:
    python plot_scalability.py benchmark_results.json
"""

import argparse
import json
import sys
from datetime import datetime


def load_results(filename):
    """Carrega resultados do benchmark"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo '{filename}' não encontrado")
        print("   Execute 'python benchmark_scalability.py' primeiro")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Erro: Arquivo '{filename}' não é um JSON válido")
        sys.exit(1)


def print_system_info(system_info):
    """Imprime informações do sistema"""
    print(f"\n{'='*80}")
    print("📋 INFORMAÇÕES DO SISTEMA")
    print(f"{'='*80}")
    for key, value in system_info.items():
        print(f"  {key:20s}: {value}")
    print()


def plot_ascii_chart(data, title, xlabel, ylabel, max_width=60):
    """Plota gráfico ASCII simples"""
    if not data:
        return
    
    print(f"\n{title}")
    print("─" * max_width)
    
    max_val = max(data)
    min_val = min(data)
    
    if max_val == min_val:
        max_val = min_val + 1
    
    for i, val in enumerate(data):
        bar_len = int((val - min_val) / (max_val - min_val) * (max_width - 20))
        bar = "█" * bar_len
        print(f"  {i+1:2d} │{bar} {val:.2f}")
    
    print("─" * max_width)
    print(f"  {xlabel} │ {ylabel}")
    print()


def plot_strong_scaling(results):
    """Visualiza resultados de strong scaling"""
    if not results:
        return
    
    print(f"\n{'='*80}")
    print("📊 ESCALABILIDADE FORTE (Strong Scaling)")
    print(f"{'='*80}")
    print(f"Baseline: {results['baseline_time']:.2f}s com 1 worker\n")
    
    # Tabela
    print(f"{'Workers':<10} {'Tempo (s)':<12} {'Speedup':<10} {'Eficiência':<12} {'Imgs/s':<10}")
    print("─" * 70)
    
    data = results['results']
    for r in data:
        print(f"{r['num_workers']:<10} "
              f"{r['elapsed_time']:<12.2f} "
              f"{r['speedup']:<10.2f}x "
              f"{r['efficiency']*100:<11.1f}% "
              f"{r['images_per_second']:<10.1f}")
    
    # Gráfico de Speedup
    workers = [r['num_workers'] for r in data]
    speedups = [r['speedup'] for r in data]
    
    print(f"\n📈 Speedup vs Workers")
    print("─" * 70)
    max_speedup = max(speedups)
    for w, s in zip(workers, speedups):
        bar_len = int(s / max_speedup * 40)
        bar = "█" * bar_len
        ideal = w  # Speedup ideal = número de workers
        marker = "↑ ideal" if abs(s - ideal) < 0.1 else ""
        print(f"  {w} workers │{bar} {s:.2f}x {marker}")
    
    # Gráfico de Eficiência
    efficiencies = [r['efficiency'] for r in data]
    
    print(f"\n💯 Eficiência vs Workers")
    print("─" * 70)
    for w, e in zip(workers, efficiencies):
        bar_len = int(e * 40)
        bar = "█" * bar_len
        color = "🟢" if e > 0.8 else "🟡" if e > 0.6 else "🔴"
        print(f"  {w} workers │{bar} {e:.1%} {color}")
    
    # Análise
    print(f"\n🔍 ANÁLISE:")
    best_speedup = max(data, key=lambda x: x['speedup'])
    print(f"  • Melhor speedup: {best_speedup['speedup']:.2f}x com {best_speedup['num_workers']} workers")
    
    best_efficiency = max(data, key=lambda x: x['efficiency'])
    print(f"  • Melhor eficiência: {best_efficiency['efficiency']:.1%} com {best_efficiency['num_workers']} worker(s)")
    
    # Calcula fração serial (Lei de Amdahl)
    if len(data) >= 2:
        p = data[-1]['num_workers']
        s = data[-1]['speedup']
        # S = 1 / (f + (1-f)/p) => f ≈ (p/S - 1) / (p - 1)
        f = (p/s - 1) / (p - 1)
        max_theoretical_speedup = 1 / f if f > 0 else float('inf')
        print(f"  • Fração serial estimada: {f:.1%}")
        if f > 0:
            print(f"  • Speedup máximo teórico: {max_theoretical_speedup:.2f}x (Lei de Amdahl)")
    
    print()


def plot_weak_scaling(results):
    """Visualiza resultados de weak scaling"""
    if not results:
        return
    
    print(f"\n{'='*80}")
    print("📊 ESCALABILIDADE FRACA (Weak Scaling)")
    print(f"{'='*80}")
    print(f"Baseline: {results['baseline_time']:.2f}s com 1 worker")
    print(f"Imagens por worker: {results['images_per_worker']}\n")
    
    # Tabela
    print(f"{'Workers':<10} {'Imagens':<10} {'Tempo (s)':<12} {'Eficiência':<12} {'Imgs/s':<10}")
    print("─" * 70)
    
    data = results['results']
    for r in data:
        print(f"{r['num_workers']:<10} "
              f"{r['num_images']:<10} "
              f"{r['elapsed_time']:<12.2f} "
              f"{r['efficiency']*100:<11.1f}% "
              f"{r['images_per_second']:<10.1f}")
    
    # Gráfico de Tempo
    workers = [r['num_workers'] for r in data]
    times = [r['elapsed_time'] for r in data]
    
    print(f"\n⏱️  Tempo vs Workers (ideal: constante)")
    print("─" * 70)
    baseline = times[0]
    for w, t in zip(workers, times):
        bar_len = int(t / max(times) * 40)
        bar = "█" * bar_len
        diff = ((t - baseline) / baseline * 100)
        marker = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
        print(f"  {w} workers │{bar} {t:.2f}s ({marker})")
    
    # Gráfico de Eficiência
    efficiencies = [r['efficiency'] for r in data]
    
    print(f"\n💯 Eficiência vs Workers (ideal: 100%)")
    print("─" * 70)
    for w, e in zip(workers, efficiencies):
        bar_len = int(e * 40)
        bar = "█" * bar_len
        color = "🟢" if e > 0.9 else "🟡" if e > 0.7 else "🔴"
        print(f"  {w} workers │{bar} {e:.1%} {color}")
    
    # Análise
    print(f"\n🔍 ANÁLISE:")
    time_increase = (times[-1] - times[0]) / times[0] * 100
    print(f"  • Aumento de tempo: {time_increase:.1f}% ({times[0]:.2f}s → {times[-1]:.2f}s)")
    
    worst_efficiency = min(data, key=lambda x: x['efficiency'])
    print(f"  • Pior eficiência: {worst_efficiency['efficiency']:.1%} com {worst_efficiency['num_workers']} workers")
    
    if len(data) >= 2:
        avg_efficiency = sum(e for e in efficiencies[1:]) / (len(efficiencies) - 1)
        print(f"  • Eficiência média (exceto baseline): {avg_efficiency:.1%}")
    
    print()


def generate_report(data):
    """Gera relatório completo"""
    print(f"\n{'='*80}")
    print("📝 RELATÓRIO DE ESCALABILIDADE")
    print(f"{'='*80}")
    print(f"Data: {data['timestamp']}")
    
    print_system_info(data['system_info'])
    
    if data.get('strong_scaling'):
        plot_strong_scaling(data['strong_scaling'])
    
    if data.get('weak_scaling'):
        plot_weak_scaling(data['weak_scaling'])
    
    # Recomendações finais
    print(f"{'='*80}")
    print("💡 RECOMENDAÇÕES FINAIS")
    print(f"{'='*80}")
    
    if data.get('strong_scaling'):
        results = data['strong_scaling']['results']
        good_configs = [r for r in results if r['efficiency'] > 0.7]
        
        if good_configs:
            best = max(good_configs, key=lambda x: x['speedup'])
            print(f"  🎯 Configuração Recomendada: {best['num_workers']} workers")
            print(f"     - Speedup: {best['speedup']:.2f}x")
            print(f"     - Eficiência: {best['efficiency']:.1%}")
            print(f"     - Throughput: {best['images_per_second']:.1f} imagens/segundo")
        else:
            print(f"  🎯 Configuração Recomendada: 1-2 workers (melhor custo/benefício)")
    
    print()
    print("  📌 Uso Sugerido:")
    print("     - Desenvolvimento/Debug: 1 worker")
    print("     - Produção (poucos dados): 2 workers")
    print("     - Produção (muitos dados): 4 workers")
    print("     - Acima de 4 workers: retorno diminui significativamente")
    
    print(f"\n{'='*80}\n")


def export_csv(data, output_file):
    """Exporta resultados para CSV"""
    try:
        with open(output_file, 'w') as f:
            if data.get('strong_scaling'):
                f.write("# Strong Scaling Results\n")
                f.write("workers,images,elapsed_time,speedup,efficiency,images_per_second\n")
                for r in data['strong_scaling']['results']:
                    f.write(f"{r['num_workers']},{r['num_images']},{r['elapsed_time']:.4f},"
                           f"{r['speedup']:.4f},{r['efficiency']:.4f},{r['images_per_second']:.4f}\n")
                f.write("\n")
            
            if data.get('weak_scaling'):
                f.write("# Weak Scaling Results\n")
                f.write("workers,images,elapsed_time,efficiency,images_per_second\n")
                for r in data['weak_scaling']['results']:
                    f.write(f"{r['num_workers']},{r['num_images']},{r['elapsed_time']:.4f},"
                           f"{r['efficiency']:.4f},{r['images_per_second']:.4f}\n")
        
        print(f"✓ Dados exportados para: {output_file}")
    except Exception as e:
        print(f"❌ Erro ao exportar CSV: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualiza resultados de benchmark de escalabilidade'
    )
    
    parser.add_argument('input', nargs='?', default='benchmark_results.json',
                       help='Arquivo JSON com resultados')
    parser.add_argument('--csv', help='Exporta resultados para CSV')
    
    args = parser.parse_args()
    
    # Carrega resultados
    data = load_results(args.input)
    
    # Gera relatório
    generate_report(data)
    
    # Exporta CSV se solicitado
    if args.csv:
        export_csv(data, args.csv)
    
    return 0


if __name__ == '__main__':
    exit(main())
