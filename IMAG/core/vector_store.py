import numpy as np
import os

class VectorStore:
    def __init__(self, file_path):
        self.file_path = file_path

    def save(self, memory_a, memory_b):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        np.savez(self.file_path, attack=memory_a, benign=memory_b)

    def load(self):
        if os.path.exists(self.file_path):
            data = np.load(self.file_path)
            return data['attack'].tolist(), data['benign'].tolist()
        return [], []