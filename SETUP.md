# 🚀 Setup e Instalação - Pipeline Paralelizada

## ⚙️ Configuração do Ambiente

### 1. Verificar Python
```bash
python --version  # Deve ser Python 3.8+
```

### 2. Criar Ambiente Virtual (Recomendado)
```bash
# Criar venv
python -m venv .venv

# Ativar (macOS/Linux)
source .venv/bin/activate

# Ativar (Windows)
.venv\Scripts\activate
```

### 3. Instalar Dependências

#### macOS
```bash
pip install -r requirements-macos.txt
```

#### Linux/Windows
```bash
pip install -r requirements.txt
```

#### Instalação Manual (se necessário)
```bash
pip install opencv-python numpy scikit-image scikit-learn xgboost pandas
```

### 4. Verificar Instalação
```bash
python -c "import cv2, numpy, sklearn, xgboost; print('✓ Todas as dependências instaladas!')"
```

---

## 🎯 Passo a Passo de Uso

### Primeira Vez - Treinar o Modelo

```bash
# 1. Garantir que as pastas de dados existem
ls -la train/fake train/real test/fake test/real val/fake val/real

# 2. Treinar o modelo
python main.py
```

**Saída esperada:**
```
Processando imagens...
Treinando modelo...
Accuracy: 0.XXXX
...
✓ Modelo salvo em 'model.pkl'
✓ Label map salvo em 'label_map.json'
✅ Treinamento concluído!
```

**Arquivos gerados:**
- `all_features.csv` - Dataset com features extraídas
- `model.pkl` - Modelo XGBoost treinado
- `label_map.json` - Mapa de classes

---

### Uso Diário - Inferência

#### Opção 1: Script de Inferência
```bash
# Uma imagem
python inference.py test/fake/0.jpg

# Diretório
python inference.py test/fake/

# Com mais workers (mais rápido)
python inference.py test/ --workers 4
```

#### Opção 2: Exemplos Interativos
```bash
# Executa 4 exemplos diferentes
python example_usage.py
```

#### Opção 3: Visualização
```bash
# Menu interativo com visualização em tempo real
python visualize_pipeline.py
```

---

## 🧪 Testes

### Executar Todos os Testes
```bash
python test_pipeline.py
```

**Saída esperada:**
```
EXECUTANDO TESTES DA PIPELINE PARALELIZADA
...
test_extract_features_from_array ... ok
test_process_single_image ... ok
...
🎉 TODOS OS TESTES PASSARAM!
```

---

## 🐛 Solução de Problemas

### Erro: "No module named 'cv2'"
```bash
pip install opencv-python
```

### Erro: "No module named 'skimage'"
```bash
pip install scikit-image
```

### Erro: "No module named 'xgboost'"
```bash
pip install xgboost
```

### Erro: "Haar Cascade não encontrado"
```python
# O OpenCV já inclui os cascades
# Mas se der erro, baixe manualmente:
# https://github.com/opencv/opencv/tree/master/data/haarcascades
```

### Erro: "Nenhuma imagem encontrada"
```bash
# Verifique se as pastas existem e contêm imagens
ls -la test/fake/*.jpg test/real/*.jpg
```

### Performance Lenta
```bash
# Reduza o número de workers
python inference.py imagens/ --workers 1

# Ou reduza o tamanho do lote
# Processar menos imagens por vez
```

---

## 📊 Benchmarks

### Hardware de Teste
- CPU: [seu processador]
- RAM: [sua memória]
- Python: 3.x

### Performance Esperada

| Imagens | Workers | Tempo | Velocidade |
|---------|---------|-------|------------|
| 10      | 1       | ~10s  | 1 img/s    |
| 10      | 2       | ~6s   | 1.7 img/s  |
| 10      | 4       | ~5s   | 2 img/s    |

**Nota:** Performance varia com hardware e tamanho das imagens.

---

## 🔧 Configuração Avançada

### Ajustar Número de Workers
```python
# Em inference.py ou programaticamente
pipeline = ParallelPipeline(model=model)
results = pipeline.process_images(images, num_workers=4)  # 4 workers por estágio
```

### Ajustar Tamanho das Filas
```python
# Mais memória, buffer maior
pipeline = ParallelPipeline(model=model, max_queue_size=20)

# Menos memória, controle mais fino
pipeline = ParallelPipeline(model=model, max_queue_size=5)
```

### Ajustar Detecção de Faces
```python
# Em face_segmentation.py, linha 28
faces = self.face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,     # Diminuir para detectar mais faces (mais lento)
    minNeighbors=5,      # Aumentar para menos falsos positivos
    minSize=(30, 30)     # Tamanho mínimo da face
)
```

---

## 📂 Estrutura de Dados Esperada

```
ti6/
├── train/
│   ├── fake/       # Imagens fake para treino
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── real/       # Imagens reais para treino
│       ├── img1.jpg
│       └── ...
├── test/
│   ├── fake/       # Imagens fake para teste
│   └── real/       # Imagens reais para teste
├── val/
│   ├── fake/       # Imagens fake para validação
│   └── real/       # Imagens reais para validação
└── [scripts aqui]
```

---

## 🎓 Uso Programático Avançado

### Exemplo Completo
```python
#!/usr/bin/env python3
import pickle
from parallel_pipeline import ParallelPipeline
import glob

# 1. Carregar modelo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Coletar imagens
images = glob.glob('minhas_imagens/*.jpg')

# 3. Criar pipeline customizada
pipeline = ParallelPipeline(
    model=model,
    batch_size=4,
    max_queue_size=10
)

# 4. Processar
print(f"Processando {len(images)} imagens...")
results = pipeline.process_images(images, num_workers=3)

# 5. Analisar resultados
fake_count = 0
real_count = 0

for result in results:
    if 'prediction' in result:
        if result['prediction'] == 0:
            fake_count += 1
            print(f"FAKE: {result['path']} ({result['probability'][0]:.1%})")
        else:
            real_count += 1
            print(f"REAL: {result['path']} ({result['probability'][1]:.1%})")
    elif 'error' in result:
        print(f"ERRO: {result['path']} - {result['error']}")

# 6. Finalizar
pipeline.stop()

print(f"\nResumo: {fake_count} fakes, {real_count} reais")
```

---

## 📝 Logs e Debug

### Habilitar Logs Detalhados
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Verificar Progresso
```python
# A pipeline já imprime erros automaticamente
# Verifique a saída do terminal para:
# "Erro ao segmentar <imagem>: ..."
# "Erro ao extrair features de <imagem>: ..."
```

---

## 🎯 Checklist de Setup

- [ ] Python 3.8+ instalado
- [ ] Ambiente virtual criado e ativado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Pastas de dados existem (train/, test/, val/)
- [ ] Modelo treinado (`python main.py`)
- [ ] Testes passando (`python test_pipeline.py`)
- [ ] Inferência funcionando (`python inference.py test/fake/`)

---

## 🆘 Suporte

### Erros Comuns

1. **ImportError**: Instale as dependências faltantes
2. **FileNotFoundError**: Verifique os caminhos das imagens
3. **MemoryError**: Reduza `max_queue_size` ou `num_workers`
4. **Lentidão**: Reduza número de workers ou tamanho de imagens

### Dicas de Performance

✅ **DO:**
- Use SSD para armazenamento de imagens
- Use 2-4 workers por estágio
- Processe em lotes grandes (50-100 imagens)

❌ **DON'T:**
- Não use muitos workers (>4) se CPU for limitada
- Não processe imagens muito grandes sem redimensionar
- Não use num_workers=1 para lotes grandes

---

## 🚀 Pronto para Começar!

```bash
# Setup completo em 3 comandos
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Treinar
python main.py

# Testar
python visualize_pipeline.py
```

**Divirta-se! 🎉**
