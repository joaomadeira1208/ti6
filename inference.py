import os
import pickle
import numpy as np
from parallel_pipeline import ParallelPipeline
import json


def load_model(model_path='model.pkl', scaler_path='scaler.pkl'):
    """Carrega o modelo XGBoost treinado e o scaler"""
    print(f"Carregando modelo de {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Scaler carregado de {scaler_path}")
    else:
        print(f"⚠️  Scaler não encontrado em {scaler_path}")
    
    return model, scaler


def load_label_map(label_map_path='label_map.json'):
    """Carrega o mapa de labels"""
    with open(label_map_path, 'r') as f:
        label_map_str = json.load(f)
    # Converte chaves de string para int
    label_map = {int(k): v for k, v in label_map_str.items()}
    return label_map


def predict_single_image(image_path, model, scaler=None, num_workers=2):
    """
    Faz predição de uma única imagem usando a pipeline paralelizada
    
    Args:
        image_path: caminho para a imagem
        model: modelo XGBoost treinado
        scaler: scaler para normalização (opcional)
        num_workers: número de workers por estágio
        
    Returns:
        dicionário com resultado da predição
    """
    pipeline = ParallelPipeline(model=model, scaler=scaler)
    results = pipeline.process_images([image_path], num_workers=num_workers)
    pipeline.stop()
    
    if results and len(results) > 0:
        return results[0]
    return None


def predict_batch(image_paths, model, scaler=None, num_workers=2):
    """
    Faz predição de múltiplas imagens usando a pipeline paralelizada
    
    Args:
        image_paths: lista de caminhos para imagens
        model: modelo XGBoost treinado
        scaler: scaler para normalização (opcional)
        num_workers: número de workers por estágio
        
    Returns:
        lista de dicionários com resultados
    """
    pipeline = ParallelPipeline(model=model, scaler=scaler)
    results = pipeline.process_images(image_paths, num_workers=num_workers)
    pipeline.stop()
    
    return results


def predict_directory(directory_path, model, scaler=None, num_workers=2):
    """
    Faz predição de todas as imagens em um diretório
    
    Args:
        directory_path: caminho para o diretório
        model: modelo XGBoost treinado
        scaler: scaler para normalização (opcional)
        num_workers: número de workers por estágio
        
    Returns:
        lista de dicionários com resultados
    """
    # Extensões de imagem suportadas
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Lista todas as imagens no diretório
    image_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_paths.append(os.path.join(root, file))
    
    print(f"Encontradas {len(image_paths)} imagens em {directory_path}")
    
    # Processa em batch
    return predict_batch(image_paths, model, scaler=scaler, num_workers=num_workers)


def print_results(results, label_map=None):
    """
    Imprime os resultados de forma formatada
    
    Args:
        results: lista de dicionários com resultados
        label_map: mapa de labels (opcional)
    """
    if label_map is None:
        label_map = {0: 'fake', 1: 'real'}
    
    print("\n" + "="*80)
    print("RESULTADOS DA CLASSIFICAÇÃO")
    print("="*80)
    
    for i, result in enumerate(results):
        print(f"\n[{i+1}] {result['path']}")
        
        if 'error' in result:
            print(f"   ❌ ERRO: {result['error']}")
        elif 'prediction' in result:
            pred = result['prediction']
            proba = result['probability']
            label = label_map.get(pred, f'classe_{pred}')
            
            print(f"   ✓ Predição: {label} (classe {pred})")
            print(f"   ✓ Confiança: {proba[pred]:.2%}")
            print(f"   ✓ Probabilidades: fake={proba[0]:.2%}, real={proba[1]:.2%}")
        else:
            print("   ⚠️  Features extraídas (sem predição)")
    
    print("\n" + "="*80)


def main():
    """Exemplo de uso da inferência"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inferência com pipeline paralelizada')
    parser.add_argument('input', help='Caminho para imagem ou diretório')
    parser.add_argument('--model', default='model.pkl', help='Caminho para o modelo')
    parser.add_argument('--workers', type=int, default=1, help='Número de workers por estágio')
    parser.add_argument('--label-map', default='label_map.json', help='Caminho para label_map.json')
    
    args = parser.parse_args()
    
    # Carrega o modelo e scaler
    model, scaler = load_model(args.model)
    
    # Carrega o label map se existir
    label_map = None
    if os.path.exists(args.label_map):
        label_map = load_label_map(args.label_map)
    
    # Verifica se é arquivo ou diretório
    if os.path.isfile(args.input):
        print(f"Processando imagem: {args.input}")
        result = predict_single_image(args.input, model, scaler=scaler, num_workers=args.workers)
        
        if result is None:
            print("\n❌ Erro: Não foi possível processar a imagem.")
            print("   Possíveis causas:")
            print("   - Imagem corrompida ou em formato inválido")
            print("   - Nenhuma face detectada na imagem")
            print("   - Erro na extração de features")
            return
        
        print_results([result], label_map)
        
    elif os.path.isdir(args.input):
        print(f"Processando diretório: {args.input}")
        results = predict_directory(args.input, model, scaler=scaler, num_workers=args.workers)
        print_results(results, label_map)
        
        # Estatísticas
        total = len(results)
        fake_count = sum(1 for r in results if r.get('prediction') == 0)
        real_count = sum(1 for r in results if r.get('prediction') == 1)
        error_count = sum(1 for r in results if 'error' in r)
        
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   Total de imagens: {total}")
        print(f"   Classificadas como FAKE: {fake_count} ({fake_count/total*100:.1f}%)")
        print(f"   Classificadas como REAL: {real_count} ({real_count/total*100:.1f}%)")
        if error_count > 0:
            print(f"   Erros: {error_count}")
    
    else:
        print(f"❌ Erro: {args.input} não é um arquivo ou diretório válido")


if __name__ == '__main__':
    main()
