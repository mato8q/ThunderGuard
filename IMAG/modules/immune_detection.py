import numpy as np

class ImmuneDetector:
    def __init__(self, threshold_T=0.5, k=5):
        self.T = threshold_T
        self.k = k          
    def _cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _get_top_k_activations(self, h_x, memory_bank_vectors):
        similarities = [self._cosine_similarity(h_x, v) for v in memory_bank_vectors]
        top_k_indices = np.argsort(similarities)[-self.k:]
        return np.array([memory_bank_vectors[i] for i in top_k_indices])

    def _get_svd_reference_vector(self, H_matrix):
       
        U, S, Vh = np.linalg.svd(H_matrix, full_matrices=False)
        return Vh[0]

    def detect(self, h_x, memory_a, memory_b):
      
        # 1. Retrieve Top-K
        X_a = self._get_top_k_activations(h_x, memory_a)
        X_b = self._get_top_k_activations(h_x, memory_b) 

        # 2. Extract Reference Vectors using SVD
        h_a = self._get_svd_reference_vector(X_a) 
        h_b = self._get_svd_reference_vector(X_b) 

     
        s_a = np.linalg.norm(h_a - h_x)
        s_b = np.linalg.norm(h_b - h_x)

        if s_b - s_a > self.T:
            return "attack", s_a, s_b
        elif s_a - s_b > self.T:
            return "benign", s_a, s_b
        else:
            return "candidate", s_a, s_b 