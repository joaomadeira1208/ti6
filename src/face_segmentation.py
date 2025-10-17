import cv2
import numpy as np
import os

class FaceSegmenter:
    """Classe para segmentar faces em imagens usando Haar Cascade"""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Erro ao carregar Haar Cascade para detecção de faces")
    
    def segment_face(self, image):
        """
        Segmenta a face em uma imagem
        
        Args:
            image: numpy array (BGR) ou caminho para imagem
            
        Returns:
            numpy array com a face segmentada ou imagem original se não detectar face
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Não foi possível carregar a imagem: {image}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            
            margin = int(0.1 * min(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            
            face_img = image[y1:y2, x1:x2]
            return face_img
        else:
            return image
    
    def segment_face_from_path(self, image_path):
        """
        Segmenta face de uma imagem a partir do caminho
        
        Args:
            image_path: caminho para a imagem
            
        Returns:
            numpy array com a face segmentada
        """
        return self.segment_face(image_path)
