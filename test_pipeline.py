#!/usr/bin/env python3
"""
Testes para a pipeline paralelizada
"""

import os
import unittest
import numpy as np
import cv2
from face_segmentation import FaceSegmenter
from feature_extraction import extract_features, extract_features_from_array
from parallel_pipeline import ParallelPipeline, process_images_parallel


class TestFaceSegmentation(unittest.TestCase):
    """Testes para o módulo de segmentação de faces"""
    
    def setUp(self):
        self.segmenter = FaceSegmenter()
    
    def test_segmenter_initialization(self):
        """Testa se o segmentador inicializa corretamente"""
        self.assertIsNotNone(self.segmenter.face_cascade)
    
    def test_segment_face_with_array(self):
        """Testa segmentação com numpy array"""
        # Cria uma imagem simples
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.segmenter.segment_face(img)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[2], 3)  # BGR
    
    def test_segment_face_with_path(self):
        """Testa segmentação com caminho de arquivo"""
        # Procura uma imagem de teste
        test_dir = 'test/fake'
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
            if images:
                img_path = os.path.join(test_dir, images[0])
                result = self.segmenter.segment_face_from_path(img_path)
                self.assertIsNotNone(result)
                self.assertGreater(result.shape[0], 0)
                self.assertGreater(result.shape[1], 0)


class TestFeatureExtraction(unittest.TestCase):
    """Testes para extração de features"""
    
    def test_extract_features_from_array(self):
        """Testa extração de features de array"""
        # Cria imagem aleatória
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        features = extract_features_from_array(img)
        
        self.assertIsNotNone(features)
        self.assertEqual(len(features), 70)  # 3+3+58+5+1 features
        self.assertTrue(np.all(np.isfinite(features)))  # Sem NaN ou Inf
    
    def test_extract_features_from_path(self):
        """Testa extração de features de caminho"""
        test_dir = 'test/fake'
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
            if images:
                img_path = os.path.join(test_dir, images[0])
                features = extract_features(img_path)
                
                self.assertIsNotNone(features)
                self.assertEqual(len(features), 70)
                self.assertTrue(np.all(np.isfinite(features)))
    
    def test_features_consistency(self):
        """Testa consistência entre as duas formas de extração"""
        test_dir = 'test/fake'
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
            if images:
                img_path = os.path.join(test_dir, images[0])
                
                # Extrai de duas formas
                features1 = extract_features(img_path)
                
                img = cv2.imread(img_path)
                features2 = extract_features_from_array(img)
                
                # Devem ser iguais (ou muito próximas)
                np.testing.assert_array_almost_equal(features1, features2, decimal=5)


class TestParallelPipeline(unittest.TestCase):
    """Testes para a pipeline paralelizada"""
    
    def setUp(self):
        self.pipeline = ParallelPipeline(model=None)
    
    def tearDown(self):
        self.pipeline.stop()
    
    def test_pipeline_initialization(self):
        """Testa inicialização da pipeline"""
        self.assertIsNotNone(self.pipeline.segmentation_queue)
        self.assertIsNotNone(self.pipeline.feature_queue)
        self.assertIsNotNone(self.pipeline.result_queue)
        self.assertIsNotNone(self.pipeline.face_segmenter)
    
    def test_process_single_image(self):
        """Testa processamento de uma única imagem"""
        test_dir = 'test/fake'
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
            if images:
                img_path = os.path.join(test_dir, images[0])
                results = self.pipeline.process_images([img_path], num_workers=1)
                
                self.assertEqual(len(results), 1)
                self.assertIn('path', results[0])
                self.assertIn('features', results[0])
                self.assertIsNotNone(results[0]['features'])
    
    def test_process_multiple_images(self):
        """Testa processamento de múltiplas imagens"""
        test_dir = 'test/fake'
        if os.path.exists(test_dir):
            images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                     if f.endswith(('.jpg', '.png'))][:3]
            
            if len(images) > 1:
                results = self.pipeline.process_images(images, num_workers=2)
                
                self.assertEqual(len(results), len(images))
                for result in results:
                    self.assertIn('path', result)
                    self.assertIn('features', result)
    
    def test_process_images_parallel_function(self):
        """Testa a função de conveniência"""
        test_dir = 'test/fake'
        if os.path.exists(test_dir):
            images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                     if f.endswith(('.jpg', '.png'))][:2]
            
            if images:
                results = process_images_parallel(images, model=None, num_workers=1)
                
                self.assertEqual(len(results), len(images))
                for result in results:
                    self.assertIn('features', result)


class TestPipelineWithModel(unittest.TestCase):
    """Testes para pipeline com modelo (se disponível)"""
    
    def setUp(self):
        self.model = None
        if os.path.exists('model.pkl'):
            import pickle
            with open('model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        
        self.pipeline = ParallelPipeline(model=self.model) if self.model else None
    
    def tearDown(self):
        if self.pipeline:
            self.pipeline.stop()
    
    def test_prediction_with_model(self):
        """Testa predição com modelo"""
        if not self.model:
            self.skipTest("Modelo não disponível")
        
        test_dir = 'test/fake'
        if os.path.exists(test_dir):
            images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                     if f.endswith(('.jpg', '.png'))][:2]
            
            if images:
                results = self.pipeline.process_images(images, num_workers=1)
                
                for result in results:
                    self.assertIn('prediction', result)
                    self.assertIn('probability', result)
                    self.assertIn(result['prediction'], [0, 1])
                    self.assertEqual(len(result['probability']), 2)
                    self.assertAlmostEqual(sum(result['probability']), 1.0, places=5)


def run_tests():
    """Executa todos os testes"""
    print("="*80)
    print("EXECUTANDO TESTES DA PIPELINE PARALELIZADA")
    print("="*80)
    
    # Cria test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Adiciona testes
    suite.addTests(loader.loadTestsFromTestCase(TestFaceSegmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestParallelPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineWithModel))
    
    # Executa
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Sumário
    print("\n" + "="*80)
    print("SUMÁRIO DOS TESTES")
    print("="*80)
    print(f"✓ Testes executados: {result.testsRun}")
    print(f"✓ Sucessos: {result.testsRun - len(result.failures) - len(result.errors)}")
    if result.failures:
        print(f"✗ Falhas: {len(result.failures)}")
    if result.errors:
        print(f"✗ Erros: {len(result.errors)}")
    if result.skipped:
        print(f"⊘ Pulados: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n🎉 TODOS OS TESTES PASSARAM!")
    else:
        print("\n⚠️  ALGUNS TESTES FALHARAM")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
