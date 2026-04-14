"""
Immune Detector — Stage 1 of IMAG
Faithful implementation of Eq. 1–4 from the paper.

Algorithm (paper §3.3):
1. Top-K retrieval by cosine similarity  (Eq. 1)
2. Stack top-K vectors into matrix H_a / H_b  (Eq. 3)
3. SVD rank-1 → dominant direction h_a, h_b  (Eq. 3)
4. Euclidean distance s_a = ‖h_a − h_lt(x)‖₂,  s_b = ‖h_b − h_lt(x)‖₂  (Eq. 4)
5. Classify:
     s_a − s_b > T  →  ATTACK   (closer to attack = smaller Euclidean distance from attack ref)
     s_b − s_a > T  →  BENIGN
     else           →  CANDIDATE
"""

import numpy as np


class ImmuneDetector:
    def __init__(self, threshold_T: float = 0.5, top_k: int = 5):
        """
        threshold_T : min Euclidean distance gap to make a confident decision.
                      Note: this is on Euclidean scale (~0–5 for unit vectors),
                      so default is larger than the old cosine-based 0.01.
        top_k       : number of nearest neighbours used to build each SVD matrix.
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
        """Used only for Top-K retrieval (Eq. 1)."""
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0

    def _top_k_indices(self, h_x: np.ndarray, memory: np.ndarray) -> np.ndarray:
        """Return indices of top-K memory vectors most similar to h_x (Eq. 1)."""
        sims = np.array([self._cosine_sim(h_x, v) for v in memory])
        k = min(self.k, len(memory))
        return np.argsort(sims)[-k:]

    def _svd_rank1(self, matrix: np.ndarray) -> np.ndarray:
        """
        Apply SVD to matrix H ∈ ℝ^{k × d}, return rank-1 dominant direction.
        (Eq. 3): SVD(H) → keep only the first right-singular vector (row of V^T).
        This captures the primary characteristic direction of the activation set.
        """
        # matrix shape: [k, hidden_dim]
        # np.linalg.svd returns U [k,k], S [min(k,d)], Vh [d,d]
        # First row of Vh is the dominant right-singular vector
        _, _, Vh = np.linalg.svd(matrix, full_matrices=False)
        dominant = Vh[0]  # shape [hidden_dim] — rank-1 direction
        return self._normalize(dominant)

    # ── Main ──────────────────────────────────────────────────────────────────

    def detect(
        self,
        h_x: np.ndarray,
        memory_attack: np.ndarray,
        memory_benign: np.ndarray,
    ) -> tuple[str, float, float]:
        """
        Parameters
        ----------
        h_x           : activation vector of input prompt  [hidden_dim]
        memory_attack : numpy array [n, hidden_dim]
        memory_benign : numpy array [n, hidden_dim]

        Returns
        -------
        label  : "ATTACK" | "BENIGN" | "CANDIDATE"
        s_a    : Euclidean distance to attack reference vector
        s_b    : Euclidean distance to benign reference vector
        """
        # Step 1: Top-K retrieval (Eq. 1)
        idx_a = self._top_k_indices(h_x, memory_attack)
        idx_b = self._top_k_indices(h_x, memory_benign)

        H_a = memory_attack[idx_a]   # [k, hidden_dim]
        H_b = memory_benign[idx_b]   # [k, hidden_dim]

        # Step 2: SVD rank-1 to get reference vectors h_a, h_b (Eq. 3)
        h_a = self._svd_rank1(H_a)   # dominant attack direction
        h_b = self._svd_rank1(H_b)   # dominant benign direction

        # Step 3: Euclidean distance (Eq. 4)
        # s_a = ‖h_a − h_lt(x)‖₂,  s_b = ‖h_b − h_lt(x)‖₂
        s_a = float(np.linalg.norm(h_a - h_x))
        s_b = float(np.linalg.norm(h_b - h_x))

        # Step 4: Classification (Eq. 4)
        # ATTACK  if s_a − s_b < -T  (h_x closer to attack ref = smaller s_a)
        # BENIGN  if s_b − s_a < -T  (h_x closer to benign ref = smaller s_b)
        # CANDIDATE otherwise
        #
        # Equivalently: gap = s_b - s_a  (positive → closer to attack)
        gap = s_b - s_a   # positive means s_a < s_b → h_x closer to attack

        if gap > self.T:
            return "ATTACK", s_a, s_b
        elif -gap > self.T:
            return "BENIGN", s_a, s_b
        else:
            return "CANDIDATE", s_a, s_b
