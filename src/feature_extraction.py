import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def extract_features_from_array(img):
    """
    Extrai features de uma imagem numpy array
    
    Args:
        img: numpy array BGR
        
    Returns:
        numpy array com features extraídas
    """
    img = cv2.resize(img, (128, 128))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mean_color = np.mean(img, axis=(0, 1))
    std_color = np.std(img, axis=(0, 1))  
    
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(59), density=True)
    
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    features = np.hstack([
        mean_color, std_color, lbp_hist,
        [contrast, dissimilarity, homogeneity, energy, correlation, edge_strength]
    ])
    return features


def extract_features(image_path):
    """
    Extrai features de uma imagem a partir do caminho
    (mantido para compatibilidade com código existente)
    
    Args:
        image_path: caminho para a imagem
        
    Returns:
        numpy array com features extraídas
    """
    img = cv2.imread(image_path)
    return extract_features_from_array(img)


feature_names = []

# Cor
feature_names += ['mean_B', 'mean_G', 'mean_R']
feature_names += ['std_B', 'std_G', 'std_R']

# LBP
feature_names += [f'lbp_bin_{i}' for i in range(58)]

# Haralick
feature_names += ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

# Bordas
feature_names += ['edge_strength']
