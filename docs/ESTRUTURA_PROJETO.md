# Estrutura do Projeto - Guia Completo

## 📁 Organização Final

O projeto foi organizado seguindo boas práticas de estruturação de código Python:

```
ti6/
├── 📦 src/                       # Código fonte (módulos principais)
│   ├── __init__.py              # Exports dos módulos
│   ├── face_segmentation.py     # Detecção e segmentação de faces
│   ├── feature_extraction.py    # Extração de 70 features
│   ├── parallel_pipeline.py     # Pipeline paralela (3 estágios)
│   └── data.py                  # Geração de dataset CSV
│
├── 🚀 scripts/                   # Scripts executáveis principais
│   ├── main.py                  # Treinamento do modelo
│   ├── inference.py             # Inferência via linha de comando
│   ├── app.py                   # Interface web Streamlit
│   └── evaluate_model.py        # Avaliação e métricas
│
├── 🎮 demo/                      # Scripts de demonstração e teste
│   ├── test_single_image.py    # Debug passo-a-passo
│   ├── test_pipeline.py        # Teste da pipeline completa
│   ├── quick_demo.py           # Demo rápido
│   ├── example_usage.py        # Exemplos de uso da API
│   └── visualize_pipeline.py   # Visualização do fluxo
│
├── ⚡ benchmarks/                # Testes de desempenho
│   ├── benchmark_scalability.py # Testes de escalabilidade
│   ├── plot_scalability.py      # Gráficos dos resultados
│   └── benchmark_results*.json  # Resultados dos testes
│
├── 🤖 models/                    # Modelos treinados
│   ├── model.pkl                # XGBoost classifier
│   ├── scaler.pkl               # StandardScaler
│   └── label_map.json           # Mapeamento fake/real
│
├── 💾 data/                      # Datasets de imagens
│   ├── train/                   # Dados de treinamento
│   │   ├── fake/
│   │   └── real/
│   ├── val/                     # Dados de validação
│   │   ├── fake/
│   │   └── real/
│   └── test/                    # Dados de teste
│       ├── fake/
│       └── real/
│
├── 📊 results/                   # Resultados de experimentos
│   ├── all_features.csv         # Features extraídas
│   └── metrics.json             # Métricas do modelo
│
├── 📚 docs/                      # Documentação completa
│   ├── PARALELISMO.md           # Análise de paralelismo
│   ├── MODEL_IMPROVEMENTS.md    # Melhorias do modelo
│   ├── PIPELINE_README.md       # Documentação da pipeline
│   ├── SETUP.md                 # Guia de instalação
│   └── ...                      # Outros documentos
│
├── 📄 README.md                  # Documentação principal
├── 📋 requirements.txt           # Dependências Python
├── 📋 requirements-macos.txt     # Dependências macOS
└── 🔧 .gitignore                 # Arquivos ignorados pelo Git
```

---

## 🎯 Princípios de Organização

### 1. Separação de Responsabilidades

**`src/`** - Código reutilizável
- Módulos puros, sem side effects
- Importáveis por outros scripts
- Testáveis isoladamente

**`scripts/`** - Código executável
- Scripts que fazem algo quando rodados
- Podem ter side effects (treinar modelo, salvar arquivos)
- Usam os módulos de `src/`

**`demo/`** - Código de exemplo
- Scripts para demonstração
- Testes exploratórios
- Exemplos de uso

### 2. Paths Absolutos vs Relativos

Todos os scripts agora usam **paths absolutos** baseados no diretório raiz:

```python
from pathlib import Path

# Diretório raiz do projeto
ROOT_DIR = Path(__file__).parent.parent

# Caminhos absolutos
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
RESULTS_DIR = ROOT_DIR / 'results'
```

**Vantagens**:
- Scripts funcionam de qualquer diretório
- Não depende de `cd` para diretório específico
- Mais robusto e profissional

### 3. Imports Relativos

Módulos em `src/` usam imports relativos:

```python
# Em src/parallel_pipeline.py
from .face_segmentation import FaceSegmenter
from .feature_extraction import extract_features_from_array
```

Scripts externos importam de `src`:

```python
# Em scripts/inference.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.parallel_pipeline import ParallelPipeline
```

---

## 🚀 Como Usar Após a Reorganização

### 1. Treinamento

```bash
# De qualquer lugar do projeto:
python scripts/main.py

# Arquivos gerados:
# - models/model.pkl
# - models/scaler.pkl
# - models/label_map.json
# - results/all_features.csv
# - results/metrics.json
```

### 2. Inferência

```bash
# Imagem única
python scripts/inference.py data/val/real/00000026_30.png

# Diretório
python scripts/inference.py data/val/fake/ --workers 2

# Interface web
streamlit run scripts/app.py
```

### 3. Avaliação

```bash
python scripts/evaluate_model.py
```

### 4. Benchmarks

```bash
cd benchmarks
python benchmark_scalability.py --mode both
python plot_scalability.py benchmark_results_complete.json
```

### 5. Demos

```bash
# Teste detalhado de uma imagem
python demo/test_single_image.py

# Teste da pipeline
python demo/test_pipeline.py

# Demo rápido
python demo/quick_demo.py
```

---

## 🔄 Migrations (O Que Mudou)

### Imports

**Antes**:
```python
from parallel_pipeline import ParallelPipeline
from feature_extraction import extract_features
```

**Depois**:
```python
from src.parallel_pipeline import ParallelPipeline
from src.feature_extraction import extract_features
```

### Paths de Arquivos

**Antes**:
```python
model_path = 'model.pkl'
csv_path = 'all_features.csv'
```

**Depois**:
```python
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
model_path = ROOT_DIR / 'models' / 'model.pkl'
csv_path = ROOT_DIR / 'results' / 'all_features.csv'
```

### Execução de Scripts

**Antes**:
```bash
cd /path/to/ti6
python main.py
python inference.py image.png
```

**Depois (funciona de qualquer lugar)**:
```bash
python /path/to/ti6/scripts/main.py
python /path/to/ti6/scripts/inference.py /path/to/image.png
```

---

## 📦 Pacotes e Módulos

### `src/` como Pacote Python

O diretório `src/` agora é um pacote Python com `__init__.py`:

```python
from src import (
    FaceSegmenter,
    extract_features,
    ParallelPipeline
)
```

### Estrutura Modular

Cada módulo tem responsabilidade única:

- **`face_segmentation.py`**: Detecta e extrai faces
- **`feature_extraction.py`**: Calcula features numéricas
- **`parallel_pipeline.py`**: Orquestra o pipeline paralelo
- **`data.py`**: Processa datasets e gera CSV

---

## 🧹 Arquivos Removidos

Foram removidos arquivos temporários e redundantes:

- ❌ `__pycache__/` (bytecode Python)
- ❌ `*.pyc` (arquivos compilados)
- ❌ Arquivos duplicados na raiz

---

## 📝 Convenções de Nomenclatura

### Diretórios

- **Minúsculas**: `src`, `scripts`, `models`
- **Plurais para collections**: `models`, `results`, `benchmarks`
- **Singular para contexto**: `src`, `demo`

### Arquivos

- **Snake_case**: `parallel_pipeline.py`, `benchmark_scalability.py`
- **Descritivos**: `test_single_image.py` melhor que `test.py`
- **Verbos para scripts**: `evaluate_model.py`, `plot_scalability.py`

### Módulos

- **Substantivos**: `FaceSegmenter`, `ParallelPipeline`
- **Funções verbos**: `extract_features()`, `generate_csv()`

---

## 🎓 Boas Práticas Aplicadas

### 1. ✅ Estrutura Clara
- Fácil encontrar qualquer arquivo
- Propósito de cada diretório é óbvio
- Novos desenvolvedores se orientam rapidamente

### 2. ✅ Escalabilidade
- Fácil adicionar novos módulos em `src/`
- Fácil adicionar novos scripts
- Benchmarks separados da lógica principal

### 3. ✅ Testabilidade
- Módulos em `src/` são importáveis e testáveis
- Scripts de demo servem como testes exploratórios
- Benchmarks medem desempenho real

### 4. ✅ Manutenibilidade
- Código organizado é mais fácil de manter
- Separação clara de responsabilidades
- Documentação próxima do código

### 5. ✅ Profissionalismo
- Estrutura similar a projetos open-source
- Pronto para ser compartilhado
- README claro e completo

---

## 🔍 Verificação Rápida

Para verificar se tudo está funcionando após a reorganização:

```bash
# 1. Teste imports
python -c "from src import ParallelPipeline; print('✓ Imports OK')"

# 2. Teste script
python scripts/inference.py data/val/real/00000026_30.png

# 3. Teste demo
python demo/test_pipeline.py

# 4. Verifique estrutura
tree -L 2 -I '.venv|data|__pycache__'
```

---

## 📚 Referências

- [Python Packaging Guide](https://packaging.python.org/)
- [Real Python - Project Structure](https://realpython.com/python-application-layouts/)
- [The Hitchhiker's Guide to Python - Structuring Your Project](https://docs.python-guide.org/writing/structure/)

---

**Documento criado em**: 16 de outubro de 2025  
**Projeto**: Sistema de Classificação de Imagens Fake/Real  
**Autor**: TI6 - PUC-CC 6º Período
