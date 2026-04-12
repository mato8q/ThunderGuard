import numpy as np

class MemoryBank:
    def __init__(self, storage_path=None):
        self.long_term_a = []
        self.long_term_b = []
        self.short_term = []

    def add_attack(self, h_x):
        self.long_term_a.append(h_x)

    def add_benign(self, h_x):
        self.long_term_b.append(h_x)

    def update_short_term(self, h_x, predicted_label):
        self.short_term.append({"activation": h_x, "label": predicted_label})

    def commit_to_long_term(self):
      
        for item in self.short_term:
            if item["label"] == "attack":
                self.long_term_a.append(item["activation"]) 
            elif item["label"] == "benign":
                self.long_term_b.append(item["activation"])
        
        self.short_term = []
        
    def get_memories(self):
        return np.array(self.long_term_a), np.array(self.long_term_b)