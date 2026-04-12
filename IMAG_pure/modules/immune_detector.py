"""
Immune Detector — Stage 1 ของ IMAG

Algorithm:
1. Normalise h_x and all memory vectors to unit sphere
2. Compute centroid of top-K nearest attack vectors  → c_a
3. Compute centroid of top-K nearest benign vectors  → c_b
4. Cosine similarity: sim_a = cos(h_x, c_a),  sim_b = cos(h_x, c_b)
5. ตัดสิน:
     sim_a - sim_b > T  → ATTACK   (closer to attack centroid)
     sim_b - sim_a > T  → BENIGN   (closer to benign centroid)
     else               → CANDIDATE
"""

import numpy as np


class ImmuneDetector:
    def __init__(self, threshold_T=0.05, top_k=5):
        """
        threshold_T : minimum cosine similarity gap to make a decision
        top_k       : number of nearest neighbours used to build each centroid
        """
        self.T = threshold_T
        self.k = top_k

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0

    def _top_k_centroid(self, h_x: np.ndarray, memory: np.ndarray) -> np.ndarray:
        """
        Select top-K memory vectors most similar to h_x,
        return their normalised centroid.
        """
        sims = [self._cosine_sim(h_x, v) for v in memory]
        k = min(self.k, len(memory))
        top_idx = np.argsort(sims)[-k:]
        top_vecs = np.array([self._normalize(memory[i]) for i in top_idx])
        centroid = top_vecs.mean(axis=0)
        return self._normalize(centroid)

    # ── Main ──────────────────────────────────────────────────────────────────

    def detect(self, h_x: np.ndarray, memory_attack: np.ndarray, memory_benign: np.ndarray):
        """
        Parameters
        ----------
        h_x           : activation vector of input prompt  [hidden_dim]
        memory_attack : numpy array [n, hidden_dim]
        memory_benign : numpy array [n, hidden_dim]

        Returns
        -------
        label  : "ATTACK" | "BENIGN" | "CANDIDATE"
        sim_a  : cosine similarity to attack centroid
        sim_b  : cosine similarity to benign centroid
        """
        h_x_norm = self._normalize(h_x)

        c_a = self._top_k_centroid(h_x, memory_attack)   # attack centroid
        c_b = self._top_k_centroid(h_x, memory_benign)   # benign centroid

        sim_a = self._cosine_sim(h_x_norm, c_a)
        sim_b = self._cosine_sim(h_x_norm, c_b)

        gap = sim_a - sim_b   # positive → closer to attack

        if gap > self.T:
            return "ATTACK", sim_a, sim_b
        elif -gap > self.T:
            return "BENIGN", sim_a, sim_b
        else:
            return "CANDIDATE", sim_a, sim_b
