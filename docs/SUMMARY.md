# 🚀 Pipeline Paralelizada - Resumo das Alterações

## 📦 Arquivos Criados

### 1. **face_segmentation.py**
Módulo para segmentação de faces usando Haar Cascade Classifier do OpenCV.

**Características:**
- Detecta faces automaticamente
- Adiciona margem de 10% ao redor da face
- Retorna imagem original se não detectar face
- Suporta entrada como caminho ou numpy array

### 2. **parallel_pipeline.py** ⭐
Implementação da pipeline paralelizada com 3 estágios.

**Arquitetura:**
```
Imagens → [Queue] → Segmentação → [Queue] → Features → [Queue] → Predição → Resultados
           N workers            N workers           1 worker
```

**Componentes:**
- `ParallelPipeline`: Classe principal
- `process_images_parallel()`: Função de conveniência
- Threading com queues para comunicação entre estágios
- Suporte a múltiplos workers por estágio

### 3. **feature_extraction.py** (Atualizado)
Agora suporta duas interfaces:

- `extract_features_from_array(img)`: Nova função para arrays numpy
- `extract_features(image_path)`: Mantida para compatibilidade

### 4. **inference.py** ⭐
Script completo para inferência com a pipeline.

**Funcionalidades:**
- Inferência de imagem única
- Inferência em lote
- Inferência de diretório inteiro
- Interface CLI com argparse
- Estatísticas detalhadas

**Uso:**
```bash
# Imagem única
python inference.py imagem.jpg

# Diretório
python inference.py test/fake/ --workers 4

# Com modelo customizado
python inference.py imagens/ --model meu_modelo.pkl
```

### 5. **main.py** (Atualizado)
Agora salva o modelo e label_map após treinamento:

- `model.pkl`: Modelo XGBoost treinado
- `label_map.json`: Mapa de classes (0: fake, 1: real)

### 6. **example_usage.py**
Script com 4 exemplos práticos:

1. Extração de features (sem modelo)
2. Classificação completa (com modelo)
3. Comparação de performance (diferentes workers)
4. Uso direto do objeto Pipeline

### 7. **visualize_pipeline.py**
Visualização interativa da pipeline em tempo real:

- Demonstração visual completa
- Comparação sequencial vs paralelo
- Barra de progresso
- Menu interativo

### 8. **test_pipeline.py**
Suite de testes unitários:

- Testes de segmentação
- Testes de extração de features
- Testes da pipeline
- Testes com modelo (se disponível)

### 9. **PIPELINE_README.md**
Documentação completa:

- Explicação da arquitetura
- Instruções de uso
- Exemplos de código
- Detalhes técnicos

### 10. **SUMMARY.md** (este arquivo)
Resumo de todas as alterações.

---

## 🎯 Como a Pipeline Funciona

### Estágio 1: Segmentação de Faces
```python
# Worker 1, 2, ... N
for cada imagem:
    carregar imagem
    detectar face com Haar Cascade
    extrair região da face com margem
    → enviar para fila de features
```

### Estágio 2: Extração de Features
```python
# Worker 1, 2, ... N
for cada imagem segmentada:
    redimensionar para 128x128
    extrair features de cor (6 features)
    extrair LBP (58 features)
    extrair Haralick/GLCM (5 features)
    extrair bordas (1 feature)
    → enviar 70 features para fila de predição
```

### Estágio 3: Classificação
```python
# Worker único (modelo não é thread-safe)
for cada vetor de features:
    fazer predição com XGBoost
    calcular probabilidades
    → enviar resultado final
```

---

## 🔄 Fluxo de Execução

### Treinamento
```bash
python main.py
```
1. Carrega/gera CSV com features
2. Treina modelo XGBoost
3. Avalia performance
4. **Salva model.pkl e label_map.json**

### Inferência com Pipeline
```bash
python inference.py <imagem_ou_diretório>
```
1. Carrega modelo
2. Cria pipeline paralelizada
3. Processa imagens em 3 estágios paralelos
4. Exibe resultados e estatísticas

---

## 📊 Vantagens da Paralelização

### Sem Paralelização
```
Imagem 1: [Seg] → [Feat] → [Pred] = 3s
Imagem 2: [Seg] → [Feat] → [Pred] = 3s
Imagem 3: [Seg] → [Feat] → [Pred] = 3s
Total: 9s
```

### Com Paralelização (3 imagens)
```
t=0s:  Img1[Seg] | ------ | ------
t=1s:  Img2[Seg] | Img1[Feat] | ------
t=2s:  Img3[Seg] | Img2[Feat] | Img1[Pred]
t=3s:  ------    | Img3[Feat] | Img2[Pred]
t=4s:  ------    | ------      | Img3[Pred]
Total: 5s (1.8x mais rápido)
```

**Ganhos:**
- ⚡ 1.5x - 2x mais rápido em lotes
- 🔄 Melhor uso de CPU
- 📈 Escalável (mais workers = mais rápido)
- 🎯 Processamento contínuo (pipeline nunca para)

---

## 🎮 Guia Rápido de Uso

### 1. Instalar Dependências
```bash
pip install -r requirements.txt  # ou requirements-macos.txt
```

### 2. Treinar o Modelo
```bash
python main.py
```

### 3. Testar a Pipeline
```bash
# Testes unitários
python test_pipeline.py

# Exemplos práticos
python example_usage.py

# Visualização interativa
python visualize_pipeline.py
```

### 4. Usar para Inferência
```bash
# Imagem única
python inference.py test/fake/0.jpg

# Diretório
python inference.py test/fake/

# Com mais workers
python inference.py test/ --workers 4
```

### 5. Uso Programático
```python
from parallel_pipeline import process_images_parallel
import pickle

# Carregar modelo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Processar imagens
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = process_images_parallel(images, model=model, num_workers=2)

# Ver resultados
for result in results:
    print(f"{result['path']}: {result['prediction']}")
```

---

## 🧪 Testando

### Testes Unitários
```bash
python test_pipeline.py
```

### Exemplos
```bash
python example_usage.py
```

### Visualização
```bash
python visualize_pipeline.py
```

---

## 📝 Configuração

### Número de Workers
```python
# Menos workers: mais sequencial, menos overhead
results = process_images_parallel(images, num_workers=1)

# Mais workers: mais paralelo, mais CPU
results = process_images_parallel(images, num_workers=4)
```

### Tamanho das Filas
```python
# Filas maiores: mais memória, buffer maior
pipeline = ParallelPipeline(model=model, max_queue_size=20)

# Filas menores: menos memória, processamento mais controlado
pipeline = ParallelPipeline(model=model, max_queue_size=5)
```

---

## 🎓 Conceitos Técnicos

### Threading vs Multiprocessing
- **Escolhido**: Threading
- **Por quê**: Python GIL não é problema aqui pois:
  - OpenCV libera GIL em operações C++
  - XGBoost libera GIL em inferência
  - I/O (leitura de imagens) não é bloqueado
  - Menor overhead que multiprocessing

### Queues
- `queue.Queue`: Thread-safe
- `maxsize`: Evita consumo excessivo de memória
- `task_done()`: Sincronização entre estágios

### Thread Safety
- Cada worker tem seu próprio contexto
- Queues garantem ordem de processamento
- Modelo: apenas 1 worker (XGBoost não é thread-safe)

---

## 🚀 Próximos Passos (Melhorias Futuras)

1. **Batch Prediction**: Agrupar múltiplas imagens em um batch para o modelo
2. **GPU Support**: Usar GPU para segmentação e features
3. **Cache**: Cachear faces segmentadas
4. **Métricas**: Adicionar logging detalhado de performance
5. **API REST**: Criar API web para inferência
6. **Docker**: Containerizar a aplicação

---

## 📚 Estrutura Final do Projeto

```
ti6/
├── data.py                    # Geração de CSV (original)
├── feature_extraction.py      # Extração de features (atualizado)
├── face_segmentation.py       # ✨ Novo: Segmentação de faces
├── parallel_pipeline.py       # ✨ Novo: Pipeline paralelizada
├── main.py                    # Treinamento (atualizado)
├── inference.py               # ✨ Novo: Script de inferência
├── example_usage.py           # ✨ Novo: Exemplos de uso
├── visualize_pipeline.py      # ✨ Novo: Visualização
├── test_pipeline.py           # ✨ Novo: Testes
├── PIPELINE_README.md         # ✨ Novo: Documentação
├── SUMMARY.md                 # ✨ Novo: Este arquivo
├── label_map.json            # Gerado: Mapa de classes
├── model.pkl                 # Gerado: Modelo treinado
├── all_features.csv          # Gerado: Features extraídas
└── test/                     # Dados de teste
    ├── fake/
    └── real/
```

---

## ✅ Checklist de Implementação

- [x] Módulo de segmentação de faces
- [x] Pipeline paralelizada com threads
- [x] Atualização do feature_extraction
- [x] Script de inferência completo
- [x] Salvamento do modelo
- [x] Exemplos de uso
- [x] Visualização interativa
- [x] Testes unitários
- [x] Documentação completa

---

## 🎉 Conclusão

A pipeline paralelizada foi implementada com sucesso! Agora você tem:

✅ **3 estágios paralelos** trabalhando simultaneamente
✅ **Segmentação automática de faces**
✅ **Inferência rápida e escalável**
✅ **Scripts prontos para uso**
✅ **Testes e exemplos**
✅ **Documentação completa**

**Próximo passo**: Execute `python visualize_pipeline.py` para ver a mágica acontecer! 🚀
