#!/usr/bin/env python3
"""
Script de verificação da estrutura do projeto
Testa se todos os módulos e imports estão funcionando corretamente
"""

import sys
from pathlib import Path

# Cores para output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def test_structure():
    """Verifica se a estrutura de diretórios está correta"""
    print(f"\n{BLUE}{'='*60}")
    print("🔍 VERIFICANDO ESTRUTURA DO PROJETO")
    print(f"{'='*60}{RESET}\n")
    
    root = Path(__file__).parent
    required_dirs = ['src', 'scripts', 'demo', 'benchmarks', 'models', 'data', 'results', 'docs']
    
    all_good = True
    for dir_name in required_dirs:
        dir_path = root / dir_name
        if dir_path.exists():
            print(f"{GREEN}✓{RESET} {dir_name}/ existe")
        else:
            print(f"{RED}✗{RESET} {dir_name}/ NÃO encontrado")
            all_good = False
    
    return all_good

def test_imports():
    """Verifica se os imports estão funcionando"""
    print(f"\n{BLUE}{'='*60}")
    print("🔧 TESTANDO IMPORTS")
    print(f"{'='*60}{RESET}\n")
    
    # Adiciona o diretório raiz ao path
    root = Path(__file__).parent
    sys.path.insert(0, str(root))
    
    tests = [
        ("src.face_segmentation", "FaceSegmenter"),
        ("src.feature_extraction", "extract_features"),
        ("src.parallel_pipeline", "ParallelPipeline"),
        ("src.data", "generate_csv"),
    ]
    
    all_good = True
    for module_name, obj_name in tests:
        try:
            module = __import__(module_name, fromlist=[obj_name])
            obj = getattr(module, obj_name)
            print(f"{GREEN}✓{RESET} {module_name}.{obj_name}")
        except Exception as e:
            print(f"{RED}✗{RESET} {module_name}.{obj_name}: {str(e)}")
            all_good = False
    
    return all_good

def test_files():
    """Verifica se arquivos importantes existem"""
    print(f"\n{BLUE}{'='*60}")
    print("📄 VERIFICANDO ARQUIVOS IMPORTANTES")
    print(f"{'='*60}{RESET}\n")
    
    root = Path(__file__).parent
    required_files = [
        'README.md',
        'requirements.txt',
        '.gitignore',
        'src/__init__.py',
        'scripts/main.py',
        'scripts/inference.py',
        'docs/PARALELISMO.md',
        'docs/ESTRUTURA_PROJETO.md',
    ]
    
    all_good = True
    for file_path in required_files:
        full_path = root / file_path
        if full_path.exists():
            print(f"{GREEN}✓{RESET} {file_path}")
        else:
            print(f"{RED}✗{RESET} {file_path} NÃO encontrado")
            all_good = False
    
    return all_good

def test_models():
    """Verifica se modelos treinados existem"""
    print(f"\n{BLUE}{'='*60}")
    print("🤖 VERIFICANDO MODELOS")
    print(f"{'='*60}{RESET}\n")
    
    root = Path(__file__).parent
    model_files = [
        'models/model.pkl',
        'models/scaler.pkl',
        'models/label_map.json',
    ]
    
    all_good = True
    for file_path in model_files:
        full_path = root / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            size_mb = size / (1024 * 1024)
            print(f"{GREEN}✓{RESET} {file_path} ({size_mb:.2f} MB)")
        else:
            print(f"{RED}✗{RESET} {file_path} NÃO encontrado (precisa treinar)")
            all_good = False
    
    return all_good

def main():
    """Executa todos os testes"""
    print(f"\n{BLUE}{'#'*60}")
    print("#  VERIFICAÇÃO COMPLETA DO PROJETO")
    print(f"{'#'*60}{RESET}\n")
    
    results = []
    
    # Testa estrutura
    results.append(("Estrutura de Diretórios", test_structure()))
    
    # Testa imports
    results.append(("Imports de Módulos", test_imports()))
    
    # Testa arquivos
    results.append(("Arquivos Importantes", test_files()))
    
    # Testa modelos
    results.append(("Modelos Treinados", test_models()))
    
    # Resumo
    print(f"\n{BLUE}{'='*60}")
    print("📊 RESUMO")
    print(f"{'='*60}{RESET}\n")
    
    all_passed = True
    for test_name, passed in results:
        status = f"{GREEN}PASSOU{RESET}" if passed else f"{RED}FALHOU{RESET}"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    
    if all_passed:
        print(f"\n{GREEN}✅ TODOS OS TESTES PASSARAM!{RESET}")
        print(f"\n{BLUE}O projeto está corretamente organizado e pronto para uso.{RESET}\n")
        return 0
    else:
        print(f"\n{RED}❌ ALGUNS TESTES FALHARAM{RESET}")
        print(f"\n{BLUE}Verifique os erros acima e corrija os problemas.{RESET}\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
