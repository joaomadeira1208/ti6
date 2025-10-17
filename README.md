# Sistema de Classificação de Imagens Fake/Real

Sistema paralelo para detecção de imagens falsas usando extração de features e XGBoost.

## 📁 Estrutura do Projeto

```
ti6/
├── src/                          # Código fonte principal
│   ├── __init__.py              # Exports dos módulos
│   ├── face_segmentation.py     # Segmentação de faces (Haar Cascade)
│   ├── feature_extraction.py    # Extração de 70 features
│   ├── parallel_pipeline.py     # Pipeline paralela (3 estágios)
│   └── data.py                  # Geração de dataset CSV
│
├── scripts/                      # Scripts executáveis
│   ├── main.py                  # Treinamento do modelo
│   ├── inference.py             # Inferência via CLI
│   ├── app.py                   # Interface Streamlit
│   └── evaluate_model.py        # Avaliação completa do modelo
│
├── demo/                         # Scripts de demonstração
│   ├── test_single_image.py    # Teste passo-a-passo
│   ├── test_pipeline.py        # Teste da pipeline
│   ├── quick_demo.py           # Demo rápido
│   ├── example_usage.py        # Exemplos de uso
│   └── visualize_pipeline.py   # Visualização da pipeline
│
├── benchmarks/                   # Testes de desempenho
│   ├── benchmark_scalability.py # Testes de escalabilidade
│   ├── plot_scalability.py      # Visualização dos resultados
│   └── benchmark_results*.json  # Resultados dos testes
│
├── models/                       # Modelos treinados
│   ├── model.pkl                # Modelo XGBoost
│   ├── scaler.pkl               # StandardScaler
│   └── label_map.json           # Mapeamento de classes
│
├── data/                         # Datasets
│   ├── train/                   # Imagens de treino
│   │   ├── fake/
│   │   └── real/
│   ├── val/                     # Imagens de validação
│   │   ├── fake/
│   │   └── real/
│   └── test/                    # Imagens de teste
│       ├── fake/
│       └── real/
│
├── results/                      # Resultados de experimentos
│   ├── all_features.csv         # Features extraídas
│   └── metrics.json             # Métricas do modelo
│
├── docs/                         # Documentação
│   ├── PARALELISMO.md           # Análise completa de paralelismo
│   ├── MODEL_IMPROVEMENTS.md    # Melhorias do modelo
│   ├── PIPELINE_README.md       # Documentação da pipeline
│   ├── SETUP.md                 # Guia de instalação
│   └── ...                      # Outros documentos
│
├── requirements.txt             # Dependências Python
├── requirements-macos.txt       # Dependências macOS
└── .gitignore                   # Arquivos ignorados pelo Git
```

## 🚀 Quick Start

### 1. Instalação

```bash
# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt
# ou para macOS:
pip install -r requirements-macos.txt
```

### 2. Treinar Modelo

```bash
cd scripts
python main.py
```

### 3. Fazer Inferência

```bash
# Imagem única
python scripts/inference.py data/val/real/00000026_30.png

# Diretório completo
python scripts/inference.py data/val/fake/ --workers 2

# Interface web (Streamlit)
streamlit run scripts/app.py
```

### 4. Avaliar Modelo

```bash
python scripts/evaluate_model.py
```

### 5. Testar Escalabilidade

```bash
cd benchmarks
python benchmark_scalability.py --mode both
python plot_scalability.py benchmark_results_complete.json
```

## 📊 Resultados

- **Acurácia**: ~89% no conjunto de validação
- **Speedup**: 1.34x com 2 workers, 1.74x com 4 workers
- **Throughput**: Até 79 imagens/segundo (paralelo)
- **Features**: 70 features extraídas (cor, LBP, GLCM, edges)

## 🏗️ Arquitetura

### Pipeline Paralela (3 Estágios)

```
Imagens → [Segmentação] → [Extração Features] → [Classificação] → Resultados
           (N workers)       (N workers)           (1 worker)
```

**Estágios**:
1. **Segmentação**: Detecção de faces usando Haar Cascade
2. **Features**: Extração de 70 features (cor, LBP, GLCM, edges)
3. **Classificação**: XGBoost + StandardScaler

Veja [`docs/PARALELISMO.md`](docs/PARALELISMO.md) para análise completa.

## 📚 Documentação

- **[PARALELISMO.md](docs/PARALELISMO.md)**: Análise completa do paralelismo, testes de escalabilidade
- **[MODEL_IMPROVEMENTS.md](docs/MODEL_IMPROVEMENTS.md)**: Melhorias implementadas no modelo
- **[PIPELINE_README.md](docs/PIPELINE_README.md)**: Documentação da pipeline paralela
- **[SETUP.md](docs/SETUP.md)**: Guia completo de instalação

## 🔧 Tecnologias

- **Python 3.12**
- **OpenCV**: Processamento de imagens e detecção de faces
- **NumPy**: Operações numéricas
- **scikit-learn**: Normalização e métricas
- **XGBoost**: Classificação
- **Streamlit**: Interface web
- **Threading**: Paralelização

## 📈 Desempenho

### Escalabilidade Forte (50 imagens)

| Workers | Tempo | Speedup | Eficiência |
|---------|-------|---------|------------|
| 1       | 0.77s | 1.00x   | 100%       |
| 2       | 0.57s | 1.34x   | 67%        |
| 4       | 0.44s | 1.74x   | 44%        |

### Configuração Recomendada

- **Desenvolvimento**: 1 worker (debug mais fácil)
- **Produção (poucos dados)**: 2 workers (melhor custo/benefício)
- **Produção (muitos dados)**: 4 workers (máximo throughput)

## 🧪 Demos e Testes

```bash
# Teste passo-a-passo de uma imagem
python demo/test_single_image.py

# Teste da pipeline completa
python demo/test_pipeline.py

# Demo rápido
python demo/quick_demo.py

# Visualizar pipeline
python demo/visualize_pipeline.py
```

## 🤝 Contribuindo

Este é um projeto acadêmico (TI6 - PUC-CC 6º Período).

## 📄 Licença

Projeto acadêmico - PUC Minas

## 👥 Autores

- **João Madeira** - TI6 - PUC-CC 6º Período

---

**Última atualização**: 16 de outubro de 2025
