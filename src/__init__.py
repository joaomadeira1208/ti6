"""
Módulos principais do sistema de classificação de imagens fake/real
"""

from .face_segmentation import FaceSegmenter
from .feature_extraction import extract_features, extract_features_from_array
from .parallel_pipeline import ParallelPipeline

__all__ = [
    'FaceSegmenter',
    'extract_features',
    'extract_features_from_array',
    'ParallelPipeline',
    'process_images_parallel',
]
