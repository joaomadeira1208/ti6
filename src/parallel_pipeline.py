import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
import os
import sys
import time # Adicionado import time
from pathlib import Path

# Necessário para garantir imports corretos nos processos filhos
try:
    from .face_segmentation import FaceSegmenter
    from .feature_extraction import extract_features_from_array
except ImportError:
    # Fallback para execução direta
    sys.path.append(str(Path(__file__).parent))
    from face_segmentation import FaceSegmenter
    from feature_extraction import extract_features_from_array

# Variáveis globais para os workers (inicializadas uma vez por processo)
_worker_model = None
_worker_scaler = None
_worker_segmenter = None

def init_worker(model, scaler):
    """
    Inicializa o worker carregando o segmentador e recebendo modelo/scaler.
    Isso roda uma vez por Processo, evitando overhead de pickling repetido.
    Também desativa threads internas de bibliotecas para evitar contenção.
    """
    global _worker_model, _worker_scaler, _worker_segmenter
    
    # Desativar multithreading interno do OpenCV e outras libs numéricas
    # para evitar competição de recursos (oversubscription) quando rodando em multiprocessos
    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass
        
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    _worker_model = model
    _worker_scaler = scaler
    # Instancia o segmentador uma vez por processo
    _worker_segmenter = FaceSegmenter()

def dummy_task(_):
    """Função auxiliar para warm-up do processo."""
    return None

def process_single_image_full(image_path):
    """
    Função 'Monolítica' que realiza todo o pipeline para uma imagem.
    Executada isoladamente em um processo.
    """
    global _worker_model, _worker_scaler, _worker_segmenter
    
    try:
        # 1. Segmentação
        face_image = _worker_segmenter.segment_face(image_path)
        
        if face_image is None:
             return {
                'path': image_path,
                'features': None,
                'error': 'No face detected'
            }

        # 2. Extração de Features
        features = extract_features_from_array(face_image)
        
        # 3. Classificação
        if features is not None and _worker_model is not None:
            features_reshaped = features.reshape(1, -1)
            
            if _worker_scaler is not None:
                features_reshaped = _worker_scaler.transform(features_reshaped)
            
            prediction = _worker_model.predict(features_reshaped)[0]
            proba = _worker_model.predict_proba(features_reshaped)[0]
            
            return {
                'path': image_path,
                'features': features,
                'prediction': prediction,
                'probability': proba
            }
        else:
            return {
                'path': image_path,
                'features': features,
                'error': 'Model not loaded or features failed'
            }

    except Exception as e:
        return {
            'path': image_path,
            'error': str(e)
        }

class ParallelPipeline:
    """
    Nova implementação usando Multiprocessing (ProcessPoolExecutor).
    Escalabilidade Forte Real (Bypassing GIL).
    """
    
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler
        self.executor = None
        self.execution_time = 0.0 # Armazena o tempo da última execução

    def process_images(self, image_paths, num_workers=None):
        """
        Processa imagens em paralelo usando processos.
        """
        total = len(image_paths)
        
        # Se num_workers não for definido ou for 1, evitar overhead de processos
        if num_workers == 1:
            # Execução serial para benchmark de base preciso (sem overhead de pool)
            init_worker(self.model, self.scaler)
            results = []
            
            # Medição Serial: Começa APÓS o primeiro item para consistência com a lógica paralela
            # Mas como é serial, vamos medir tudo e descontar um delta fixo se necessário, 
            # ou simplesmente medir o loop puro.
            
            t_start = time.time()
            for i, path in enumerate(image_paths):
                results.append(process_single_image_full(path))
            t_end = time.time()
            
            self.execution_time = t_end - t_start
            return results

        # Execução Paralela
        print(f"🔥 Iniciando Pool com {num_workers} Processos...", flush=True)
        
        results = []
        
        # WARM-UP & PROCESSAMENTO no mesmo Pool
        # Isso garante que os workers inicializados no warm-up sejam os mesmos usados no processamento
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(self.model, self.scaler)) as executor:
            # 1. WARM-UP: Forçar inicialização dos workers
            # Garante que todos os processos carreguem o segmentador antes de cronometrar
            list(executor.map(dummy_task, range(num_workers)))
            print(f"✅ Pool de Processos com {num_workers} workers aquecido.", flush=True)
            
            # 2. PROCESSAMENTO REAL
            # OTIMIZAÇÃO: Usar executor.map com chunksize
            # Isso reduz drasticamente o overhead de comunicação IPC (Inter-Process Communication)
            chunksize = max(1, total // (num_workers * 4)) 
            
            t_start_processing = time.time() # Marca inicio real
            
            # map aplica a função em cada item do iterável
            result_iterator = executor.map(process_single_image_full, image_paths, chunksize=chunksize)
            
            completed = 0
            for res in result_iterator:
                results.append(res)
                completed += 1
            
            t_end_processing = time.time()
            
            # Cálculo de tempo TOTAL
            self.execution_time = t_end_processing - t_start_processing
        
        return results

    def stop(self):
        # No ProcessPoolExecutor (context manager), o shutdown é automático, 
        # mas mantemos o método para compatibilidade com a interface anterior.
        pass