# Pipeline Paralelizada para Classificação de Imagens

Este projeto implementa uma pipeline paralelizada para classificação de imagens fake/real com três estágios:

1. **Segmentação de Face**: Detecta e extrai faces das imagens
2. **Extração de Features**: Extrai características da imagem (cor, textura, bordas)
3. **Classificação**: Classifica a imagem usando XGBoost

## Arquivos Criados

### 1. `face_segmentation.py`
Módulo responsável pela segmentação de faces usando Haar Cascade Classifier.

**Classe Principal**: `FaceSegmenter`
- `segment_face(image)`: Segmenta face de uma imagem (pode receber path ou numpy array)
- Adiciona margem de 10% ao redor da face detectada
- Retorna imagem original se não detectar face

### 2. `parallel_pipeline.py`
Pipeline paralelizada usando threads e queues.

**Classe Principal**: `ParallelPipeline`
- Usa 3 filas (segmentation_queue, feature_queue, result_queue)
- Workers executam em paralelo:
  - `_segmentation_worker`: Segmenta faces
  - `_feature_extraction_worker`: Extrai features
  - `_prediction_worker`: Faz predições (se modelo fornecido)

**Função de Conveniência**: `process_images_parallel(image_paths, model=None, num_workers=2)`

### 3. `feature_extraction.py` (atualizado)
Agora suporta duas formas de uso:
- `extract_features_from_array(img)`: Extrai features de numpy array (nova)
- `extract_features(image_path)`: Extrai features de caminho (mantida para compatibilidade)

### 4. `inference.py`
Script para fazer inferência usando a pipeline paralelizada.

**Funções**:
- `predict_single_image(image_path, model)`: Predição de uma imagem
- `predict_batch(image_paths, model)`: Predição em lote
- `predict_directory(directory_path, model)`: Predição de todas as imagens em um diretório

### 5. `main.py` (atualizado)
Agora salva o modelo treinado e o label_map para uso posterior.

## Como Usar

### 1. Treinar o Modelo

```bash
python main.py
```

Isso irá:
- Gerar/carregar o CSV com features
- Treinar o modelo XGBoost
- Salvar o modelo em `model.pkl`
- Salvar o label_map em `label_map.json`

### 2. Fazer Inferência com Pipeline Paralelizada

**Uma única imagem:**
```bash
python inference.py test/fake/0.jpg
```

**Um diretório inteiro:**
```bash
python inference.py test/fake/
```

**Opções adicionais:**
```bash
python inference.py test/fake/ --workers 4 --model model.pkl
```

Parâmetros:
- `--workers N`: Número de workers por estágio (padrão: 2)
- `--model PATH`: Caminho para o modelo (padrão: model.pkl)
- `--label-map PATH`: Caminho para label_map.json (padrão: label_map.json)

## Vantagens da Pipeline Paralelizada

1. **Processamento Concorrente**: Enquanto uma imagem está sendo segmentada, outra está tendo features extraídas, e outra está sendo classificada.

2. **Escalabilidade**: Fácil ajustar o número de workers por estágio conforme recursos disponíveis.

3. **Eficiência**: Reduz tempo total de processamento em lotes grandes de imagens.

4. **Modularidade**: Cada estágio é independente e pode ser modificado sem afetar os outros.

## Estrutura da Pipeline

```
Imagens → [Queue 1] → Segmentação → [Queue 2] → Features → [Queue 3] → Classificação → Resultados
          (N workers)              (N workers)             (1 worker)
```

## Exemplo de Saída

```
================================================================================
RESULTADOS DA CLASSIFICAÇÃO
================================================================================

[1] test/fake/0.jpg
   ✓ Predição: fake (classe 0)
   ✓ Confiança: 87.34%
   ✓ Probabilidades: fake=87.34%, real=12.66%

[2] test/real/1.jpg
   ✓ Predição: real (classe 1)
   ✓ Confiança: 92.15%
   ✓ Probabilidades: fake=7.85%, real=92.15%

================================================================================

📊 ESTATÍSTICAS:
   Total de imagens: 2
   Classificadas como FAKE: 1 (50.0%)
   Classificadas como REAL: 1 (50.0%)
```

## Uso Programático

Você também pode usar a pipeline diretamente em seu código:

```python
from parallel_pipeline import ParallelPipeline
import pickle

# Carregar modelo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Criar pipeline
pipeline = ParallelPipeline(model=model)

# Processar imagens
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = pipeline.process_images(image_paths, num_workers=2)

# Usar resultados
for result in results:
    print(f"{result['path']}: {result['prediction']}")

pipeline.stop()
```

## Requisitos

Certifique-se de ter todas as dependências instaladas:
- OpenCV (cv2)
- NumPy
- scikit-image
- scikit-learn
- XGBoost
- pandas

```bash
pip install -r requirements.txt
# ou para macOS:
pip install -r requirements-macos.txt
```
