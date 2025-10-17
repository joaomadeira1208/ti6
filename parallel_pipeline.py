import threading
import queue
import cv2
import numpy as np
from face_segmentation import FaceSegmenter
from feature_extraction import extract_features_from_array


class ParallelPipeline:
    """
    Pipeline paralelizada para classificação de imagens com 3 estágios:
    1. Segmentação de faces
    2. Extração de features
    3. Classificação/Predição
    """
    
    def __init__(self, model=None, batch_size=4, max_queue_size=10):
        """
        Args:
            model: modelo XGBoost treinado (opcional, para inferência)
            batch_size: número de imagens a processar em batch
            max_queue_size: tamanho máximo das filas entre estágios
        """
        self.model = model
        self.batch_size = batch_size
        
        # Filas para comunicação entre estágios
        self.segmentation_queue = queue.Queue(maxsize=max_queue_size)
        self.feature_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        
        # Inicializa o segmentador de faces
        self.face_segmenter = FaceSegmenter()
        
        # Flags de controle
        self.stop_flag = threading.Event()
        self.threads = []
    
    def _segmentation_worker(self):
        """Worker que processa segmentação de faces"""
        while not self.stop_flag.is_set():
            try:
                # Timeout para permitir verificar stop_flag periodicamente
                item = self.segmentation_queue.get(timeout=0.5)
                
                if item is None:  # Sinal de finalização
                    self.feature_queue.put(None)
                    break
                
                idx, image_path = item
                
                # Segmenta a face
                try:
                    face_image = self.face_segmenter.segment_face(image_path)
                    self.feature_queue.put((idx, face_image, image_path))
                except Exception as e:
                    print(f"Erro ao segmentar {image_path}: {e}")
                    # Coloca None para indicar erro
                    self.feature_queue.put((idx, None, image_path))
                
                self.segmentation_queue.task_done()
                
            except queue.Empty:
                continue
    
    def _feature_extraction_worker(self):
        """Worker que processa extração de features"""
        while not self.stop_flag.is_set():
            try:
                item = self.feature_queue.get(timeout=0.5)
                
                if item is None:  # Sinal de finalização
                    self.result_queue.put(None)
                    break
                
                idx, face_image, image_path = item
                
                # Extrai features
                try:
                    if face_image is not None:
                        features = extract_features_from_array(face_image)
                        self.result_queue.put((idx, features, image_path))
                    else:
                        self.result_queue.put((idx, None, image_path))
                except Exception as e:
                    print(f"Erro ao extrair features de {image_path}: {e}")
                    self.result_queue.put((idx, None, image_path))
                
                self.feature_queue.task_done()
                
            except queue.Empty:
                continue
    
    def _prediction_worker(self, results_dict):
        """Worker que processa predições (se modelo fornecido)"""
        while not self.stop_flag.is_set():
            try:
                item = self.result_queue.get(timeout=0.5)
                
                if item is None:  # Sinal de finalização
                    break
                
                idx, features, image_path = item
                
                # Faz predição se modelo disponível
                try:
                    if features is not None and self.model is not None:
                        features_reshaped = features.reshape(1, -1)
                        prediction = self.model.predict(features_reshaped)[0]
                        proba = self.model.predict_proba(features_reshaped)[0]
                        results_dict[idx] = {
                            'path': image_path,
                            'features': features,
                            'prediction': prediction,
                            'probability': proba
                        }
                    elif features is not None:
                        results_dict[idx] = {
                            'path': image_path,
                            'features': features
                        }
                    else:
                        results_dict[idx] = {
                            'path': image_path,
                            'features': None,
                            'error': 'Failed to extract features'
                        }
                except Exception as e:
                    print(f"Erro ao processar resultado de {image_path}: {e}")
                    results_dict[idx] = {
                        'path': image_path,
                        'error': str(e)
                    }
                
                self.result_queue.task_done()
                
            except queue.Empty:
                continue
    
    def process_images(self, image_paths, num_workers=2):
        """
        Processa uma lista de imagens através da pipeline paralelizada
        
        Args:
            image_paths: lista de caminhos para as imagens
            num_workers: número de workers por estágio
            
        Returns:
            lista de dicionários com resultados ordenados pelos índices
        """
        results_dict = {}
        
        # Inicia workers
        self.stop_flag.clear()
        self.threads = []
        
        # Workers de segmentação
        for _ in range(num_workers):
            t = threading.Thread(target=self._segmentation_worker)
            t.start()
            self.threads.append(t)
        
        # Workers de extração de features
        for _ in range(num_workers):
            t = threading.Thread(target=self._feature_extraction_worker)
            t.start()
            self.threads.append(t)
        
        # Worker de predição (apenas 1, pois modelo não é thread-safe em geral)
        t = threading.Thread(target=self._prediction_worker, args=(results_dict,))
        t.start()
        self.threads.append(t)
        
        # Adiciona imagens à fila de segmentação
        for idx, image_path in enumerate(image_paths):
            self.segmentation_queue.put((idx, image_path))
        
        # Envia sinais de finalização
        for _ in range(num_workers):
            self.segmentation_queue.put(None)
        
        # Aguarda conclusão
        for t in self.threads:
            t.join()
        
        # Ordena resultados pelo índice
        results = [results_dict[i] for i in sorted(results_dict.keys())]
        
        return results
    
    def stop(self):
        """Para todos os workers"""
        self.stop_flag.set()
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=1.0)


def process_images_parallel(image_paths, model=None, num_workers=2):
    """
    Função de conveniência para processar imagens com pipeline paralelizada
    
    Args:
        image_paths: lista de caminhos de imagens
        model: modelo XGBoost treinado (opcional)
        num_workers: número de workers por estágio
        
    Returns:
        lista de resultados
    """
    pipeline = ParallelPipeline(model=model)
    results = pipeline.process_images(image_paths, num_workers=num_workers)
    pipeline.stop()
    return results
