#!/usr/bin/env python3
"""
Quick Demo - Pipeline Paralelizada
Demonstração rápida de 30 segundos
"""

import os
import sys


def print_banner():
    """Imprime banner bonito"""
    print("\n" + "="*80)
    print("🚀 PIPELINE PARALELIZADA - DEMO RÁPIDA".center(80))
    print("="*80 + "\n")


def check_requirements():
    """Verifica se todas as dependências estão instaladas"""
    print("🔍 Verificando dependências...")
    
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'pandas': 'pandas',
        'skimage': 'scikit-image'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} - FALTANDO")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Dependências faltando: {', '.join(missing)}")
        print(f"   Instale com: pip install {' '.join(missing)}")
        return False
    
    print("   ✓ Todas as dependências instaladas!\n")
    return True


def check_data():
    """Verifica se há dados para testar"""
    print("📁 Verificando dados...")
    
    test_dirs = ['test/fake', 'test/real']
    found_images = []
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            found_images.extend([os.path.join(test_dir, img) for img in images[:3]])
            print(f"   ✓ {test_dir}: {len(images)} imagens")
        else:
            print(f"   ⚠️  {test_dir}: não encontrado")
    
    if not found_images:
        print("\n⚠️  Nenhuma imagem encontrada para teste")
        print("   Certifique-se de ter imagens em test/fake/ e test/real/\n")
        return None
    
    print(f"   ✓ Total: {len(found_images)} imagens para demo\n")
    return found_images[:5]  # Limita a 5 para demo rápida


def check_model():
    """Verifica se o modelo está treinado"""
    print("🤖 Verificando modelo...")
    
    if os.path.exists('model.pkl'):
        print("   ✓ Modelo encontrado (model.pkl)\n")
        return True
    else:
        print("   ✗ Modelo não encontrado")
        print("   Execute primeiro: python main.py\n")
        return False


def run_quick_demo():
    """Executa demonstração rápida"""
    print_banner()
    
    # Verificações
    if not check_requirements():
        return False
    
    images = check_data()
    if not images:
        return False
    
    has_model = check_model()
    
    # Demo
    print("🎬 Executando demonstração...\n")
    
    try:
        from parallel_pipeline import process_images_parallel
        import pickle
        import time
        
        # Carrega modelo se disponível
        model = None
        if has_model:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
        
        # Processa imagens
        print(f"Processando {len(images)} imagens com pipeline paralelizada...")
        print("Estágios: Segmentação → Features → Classificação\n")
        
        start_time = time.time()
        results = process_images_parallel(images, model=model, num_workers=2)
        elapsed = time.time() - start_time
        
        # Mostra resultados
        print("\n" + "="*80)
        print("RESULTADOS")
        print("="*80 + "\n")
        
        label_map = {0: '🟥 FAKE', 1: '🟩 REAL'}
        
        for i, result in enumerate(results):
            basename = os.path.basename(result['path'])
            print(f"[{i+1}] {basename}")
            
            if 'prediction' in result:
                pred = result['prediction']
                conf = result['probability'][pred]
                print(f"    → {label_map[pred]} (confiança: {conf:.1%})\n")
            elif 'features' in result and result['features'] is not None:
                print(f"    → {len(result['features'])} features extraídas\n")
            else:
                print(f"    → Erro no processamento\n")
        
        # Estatísticas
        print("="*80)
        print(f"⏱️  Tempo total: {elapsed:.2f}s ({elapsed/len(images):.2f}s por imagem)")
        print(f"⚡ Velocidade: {len(images)/elapsed:.2f} imagens/segundo")
        
        if has_model:
            fake_count = sum(1 for r in results if r.get('prediction') == 0)
            real_count = sum(1 for r in results if r.get('prediction') == 1)
            print(f"📊 Classificação: {fake_count} fakes, {real_count} reais")
        
        print("="*80 + "\n")
        
        # Próximos passos
        print("✅ DEMO CONCLUÍDA COM SUCESSO!\n")
        print("📚 Próximos passos:")
        print("   1. python example_usage.py      - Mais exemplos")
        print("   2. python visualize_pipeline.py - Visualização interativa")
        print("   3. python test_pipeline.py      - Executar testes")
        print("   4. python inference.py <dir>    - Processar seus dados")
        print("\n💡 Leia PIPELINE_README.md para documentação completa\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro durante demo: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_menu():
    """Mostra menu de opções"""
    print_banner()
    print("Escolha uma opção:\n")
    print("  1. Demo rápida (30s)")
    print("  2. Verificar apenas dependências")
    print("  3. Verificar apenas dados")
    print("  4. Verificar apenas modelo")
    print("  5. Verificar tudo")
    print("  0. Sair\n")
    
    choice = input("Opção: ").strip()
    
    if choice == '1':
        run_quick_demo()
    elif choice == '2':
        check_requirements()
    elif choice == '3':
        check_data()
    elif choice == '4':
        check_model()
    elif choice == '5':
        check_requirements()
        check_data()
        check_model()
    elif choice == '0':
        print("\n👋 Até logo!\n")
    else:
        print("\n❌ Opção inválida\n")


def main():
    """Função principal"""
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Modo automático (sem menu)
        run_quick_demo()
    else:
        # Modo interativo (com menu)
        try:
            show_menu()
        except KeyboardInterrupt:
            print("\n\n👋 Interrompido pelo usuário\n")


if __name__ == '__main__':
    main()
