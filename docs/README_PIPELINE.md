# 🚀 Pipeline Paralelizada para Classificação de Imagens Fake/Real

Sistema completo de classificação de imagens utilizando pipeline paralelizada com 3 estágios: **Segmentação de Faces**, **Extração de Features** e **Classificação**.

## ✨ Características

- 🔄 **Pipeline Paralelizada**: 3 estágios trabalhando simultaneamente
- 👤 **Detecção Automática de Faces**: Usando Haar Cascade Classifier
- 📊 **70 Features Extraídas**: Cor, textura (LBP, GLCM) e bordas
- 🤖 **Classificação com XGBoost**: Modelo de alta performance
- ⚡ **Processamento Rápido**: 1.5x-2x mais rápido que sequencial
- 🎯 **Interface CLI Completa**: Scripts prontos para uso
- 🧪 **Testes Incluídos**: Suite completa de testes unitários

## 🎬 Quick Start (30 segundos)

```bash
# 1. Demo rápida
python quick_demo.py --auto

# 2. Ou menu interativo
python quick_demo.py
```

## 📦 Instalação

### 1. Clonar/Navegar para o Projeto
```bash
cd /caminho/para/ti6
```

### 2. Criar Ambiente Virtual
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# ou
.venv\Scripts\activate     # Windows
```

### 3. Instalar Dependências
```bash
# macOS
pip install -r requirements-macos.txt

# Linux/Windows
pip install -r requirements.txt
```

## 🎯 Uso Básico

### 1. Treinar o Modelo
```bash
python main.py
```

### 2. Fazer Inferência
```bash
# Imagem única
python inference.py test/fake/0.jpg

# Diretório completo
python inference.py test/fake/

# Com mais workers (mais rápido)
python inference.py test/ --workers 4
```

### 3. Exemplos e Visualizações
```bash
# Exemplos práticos
python example_usage.py

# Visualização interativa
python visualize_pipeline.py

# Testes
python test_pipeline.py
```

## 🏗️ Arquitetura da Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE PARALELIZADA                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Imagens  →  [Queue 1]  →  🔍 Segmentação  →  [Queue 2]         │
│                             (N workers)                          │
│                                                                  │
│              →  [Queue 2]  →  📊 Features  →  [Queue 3]         │
│                             (N workers)                          │
│                                                                  │
│              →  [Queue 3]  →  🎯 Classificação  →  Resultados   │
│                             (1 worker)                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Estágios

1. **🔍 Segmentação de Faces**
   - Detecta faces usando Haar Cascade
   - Extrai região da face com margem
   - Suporta múltiplos workers

2. **📊 Extração de Features**
   - Features de cor (6)
   - LBP - Local Binary Patterns (58)
   - GLCM - Haralick (5)
   - Bordas - Sobel (1)
   - **Total: 70 features**

3. **🎯 Classificação**
   - Modelo XGBoost
   - Predição com probabilidades
   - Thread-safe

## 📁 Estrutura do Projeto

```
ti6/
├── 🎯 Scripts Principais
│   ├── main.py                    # Treinamento do modelo
│   ├── inference.py               # Inferência com pipeline
│   ├── data.py                    # Geração de CSV
│   
├── 🔧 Módulos da Pipeline
│   ├── face_segmentation.py       # Segmentação de faces
│   ├── parallel_pipeline.py       # Pipeline paralelizada
│   ├── feature_extraction.py      # Extração de features
│   
├── 🎨 Scripts de Demo/Teste
│   ├── quick_demo.py              # Demo rápida
│   ├── example_usage.py           # Exemplos de uso
│   ├── visualize_pipeline.py      # Visualização interativa
│   ├── test_pipeline.py           # Testes unitários
│   
├── 📚 Documentação
│   ├── README.md                  # Este arquivo
│   ├── PIPELINE_README.md         # Documentação da pipeline
│   ├── SETUP.md                   # Guia de instalação
│   └── SUMMARY.md                 # Resumo das alterações
│   
├── 📊 Dados (gerados)
│   ├── model.pkl                  # Modelo treinado
│   ├── label_map.json            # Mapa de classes
│   └── all_features.csv          # Features extraídas
│   
└── 🖼️ Datasets
    ├── train/fake/                # Treino - fakes
    ├── train/real/                # Treino - reais
    ├── test/fake/                 # Teste - fakes
    ├── test/real/                 # Teste - reais
    ├── val/fake/                  # Validação - fakes
    └── val/real/                  # Validação - reais
```

## 🎮 Guia de Comandos

### Treinamento
```bash
# Treinar modelo
python main.py

# Resultado:
# - model.pkl (modelo treinado)
# - label_map.json (mapa de classes)
# - all_features.csv (dataset)
```

### Inferência
```bash
# Básico
python inference.py <imagem_ou_diretório>

# Com opções
python inference.py test/fake/ --workers 4 --model model.pkl

# Ajuda
python inference.py --help
```

### Demonstrações
```bash
# Demo rápida (30s)
python quick_demo.py --auto

# Exemplos variados
python example_usage.py

# Visualização com menu
python visualize_pipeline.py
```

### Testes
```bash
# Executar testes
python test_pipeline.py

# Resultado esperado:
# ✓ Testes executados: XX
# 🎉 TODOS OS TESTES PASSARAM!
```

## 💻 Uso Programático

```python
from parallel_pipeline import process_images_parallel
import pickle

# Carregar modelo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Processar imagens
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = process_images_parallel(
    images, 
    model=model, 
    num_workers=2
)

# Analisar resultados
for result in results:
    if 'prediction' in result:
        label = 'FAKE' if result['prediction'] == 0 else 'REAL'
        conf = result['probability'][result['prediction']]
        print(f"{result['path']}: {label} ({conf:.1%})")
```

## 📊 Performance

### Benchmarks (10 imagens)

| Workers | Tempo | Velocidade | Speedup |
|---------|-------|------------|---------|
| 1       | 10s   | 1 img/s    | 1.0x    |
| 2       | 6s    | 1.7 img/s  | 1.7x    |
| 4       | 5s    | 2 img/s    | 2.0x    |

*Valores aproximados, variam com hardware*

### Vantagens

- ✅ Processamento paralelo de estágios
- ✅ Melhor uso de CPU multi-core
- ✅ Throughput aumentado em lotes
- ✅ Escalável (mais workers = mais rápido)

## 🎓 Exemplos de Saída

### Inferência Simples
```
================================================================================
RESULTADOS DA CLASSIFICAÇÃO
================================================================================

[1] test/fake/0.jpg
   🎯 Predição: FAKE (classe 0)
   📊 Confiança: 87.34%
   📈 Probabilidades: FAKE=87.34%, REAL=12.66%

[2] test/real/1.jpg
   🎯 Predição: REAL (classe 1)
   📊 Confiança: 92.15%
   📈 Probabilidades: FAKE=7.85%, REAL=92.15%

================================================================================

📊 ESTATÍSTICAS:
   Total de imagens: 2
   Classificadas como FAKE: 1 (50.0%)
   Classificadas como REAL: 1 (50.0%)
```

## 🔧 Configuração Avançada

### Ajustar Workers
```python
# Mais workers = mais rápido (até limite de CPU)
pipeline = ParallelPipeline(model=model)
results = pipeline.process_images(images, num_workers=4)
```

### Ajustar Tamanho das Filas
```python
# Maior = mais memória, menor = mais controlado
pipeline = ParallelPipeline(model=model, max_queue_size=20)
```

### Ajustar Detecção de Faces
Edite `face_segmentation.py`, linha ~28:
```python
faces = self.face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,     # Sensibilidade
    minNeighbors=5,      # Confiança
    minSize=(30, 30)     # Tamanho mínimo
)
```

## 🐛 Solução de Problemas

### Erro: "No module named 'xxx'"
```bash
pip install opencv-python numpy scikit-image scikit-learn xgboost pandas
```

### Erro: "Nenhuma imagem encontrada"
```bash
# Verifique a estrutura de pastas
ls -la test/fake test/real
```

### Performance Lenta
```bash
# Reduza workers
python inference.py imagens/ --workers 1

# Ou processe menos imagens por vez
```

### Modelo não encontrado
```bash
# Treine primeiro
python main.py
```

## 📚 Documentação Completa

- 📖 **[PIPELINE_README.md](PIPELINE_README.md)** - Documentação técnica detalhada
- 🔧 **[SETUP.md](SETUP.md)** - Guia completo de instalação e configuração
- 📝 **[SUMMARY.md](SUMMARY.md)** - Resumo de todas as alterações

## 🧪 Testando

### Checklist Rápido
```bash
# 1. Verificar instalação
python quick_demo.py

# 2. Executar testes
python test_pipeline.py

# 3. Testar inferência
python inference.py test/fake/
```

## 🎯 Próximos Passos

1. **Primeiro Uso**
   ```bash
   python quick_demo.py --auto
   ```

2. **Treinar Modelo**
   ```bash
   python main.py
   ```

3. **Explorar Exemplos**
   ```bash
   python example_usage.py
   python visualize_pipeline.py
   ```

4. **Usar em Produção**
   ```bash
   python inference.py meus_dados/ --workers 4
   ```

## 🤝 Contribuindo

Melhorias futuras planejadas:
- [ ] Batch prediction no estágio de classificação
- [ ] Suporte a GPU
- [ ] Cache de faces segmentadas
- [ ] API REST
- [ ] Docker container

## 📄 Licença

[Sua licença aqui]

## 👨‍💻 Autor

João Madeira - PUC-CC 6º Período

---

## 🚀 Comece Agora!

```bash
# Setup completo em 3 comandos
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python quick_demo.py --auto
```

**Boa sorte com seu projeto! 🎉**
