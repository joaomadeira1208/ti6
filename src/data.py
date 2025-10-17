import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Adiciona o diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.feature_extraction import extract_features, feature_names

# Caminhos relativos ao diretório raiz
DATA_DIR = ROOT_DIR / 'data'
RESULTS_DIR = ROOT_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def generate_csv():
    # Pastas que contêm imagens
    folders = {
        'train': DATA_DIR / 'train',
        'test': DATA_DIR / 'test',
        'val': DATA_DIR / 'val'
    }

    # Inicializa listas para armazenar features e labels
    all_features = []
    all_labels = []

    # Primeiro, conta o total de imagens
    print("🔍 Contando imagens...")
    total_images = 0
    image_files = []
    
    for folder_type, folder_path in folders.items():
        for class_name, label in [('fake', 0), ('real', 1)]:
            class_path = os.path.join(folder_path, class_name)
            
            if not os.path.exists(class_path):
                continue
            
            for img_name in os.listdir(class_path):
                # Ignora arquivos ocultos e não-imagens
                if img_name.startswith('.'):
                    continue
                
                # Verifica se é uma imagem
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                    continue
                
                img_path = os.path.join(class_path, img_name)
                image_files.append((img_path, label))
                total_images += 1
    
    print(f"✓ Total de imagens encontradas: {total_images}\n")
    
    # Percorre cada imagem com barra de progresso
    print("🚀 Extraindo features...")
    processed = 0
    errors = 0
    
    for img_path, label in tqdm(image_files, desc="Processando imagens", unit="img", ncols=80):
        try:
            features = extract_features(img_path)
            all_features.append(features)
            all_labels.append(label)
            processed += 1
        except Exception as e:
            errors += 1
            tqdm.write(f"❌ Erro em {os.path.basename(img_path)}: {str(e)[:50]}")

    # Converte para arrays NumPy
    X_all = np.array(all_features)
    y_all = np.array(all_labels)

    # Cria DataFrame com nomes das features + coluna label
    df_all = pd.DataFrame(X_all, columns=feature_names)
    df_all['label'] = y_all

    # Salva em CSV
    csv_path = RESULTS_DIR / 'all_features.csv'
    df_all.to_csv(csv_path, index=False)

    print(f"\n{'='*80}")
    print(f"✅ CSV criado com sucesso!")
    print(f"   Total processado: {processed}/{total_images}")
    if errors > 0:
        print(f"   Erros: {errors}")
    print(f"   Fake: {sum(y_all == 0)}")
    print(f"   Real: {sum(y_all == 1)}")
    print(f"   Features por imagem: {len(df_all.columns) - 1}")
    print(f"   Arquivo: {csv_path}")
    print(f"{'='*80}\n")

