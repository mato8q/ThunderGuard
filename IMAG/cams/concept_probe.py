"""
ConceptProbe — JBShield-style SVD concept direction for CAMS.

Option B design: calibrated on harmful/harmless data only (no attack-specific
jailbreak JSON). The toxic_vector is the rank-1 SVD direction of the
paired difference matrix (harmful - harmless embeddings).

Score convention:
  score > 0  →  h is on the harmful side of concept space
  score < 0  →  h is on the harmless side
"""

import numpy as np


class ConceptProbe:
    """
    Compute a general safety concept direction via SVD rank-1 decomposition.

    Parameters
    ----------
    harmful_embeddings  : list of np.ndarray [hidden_dim]  (at critical layer)
    harmless_embeddings : list of np.ndarray [hidden_dim]  (at critical layer)
    """

    def __init__(
        self,
        harmful_embeddings: list[np.ndarray],
        harmless_embeddings: list[np.ndarray],
    ):
        n = min(len(harmful_embeddings), len(harmless_embeddings))
        if n == 0:
            raise ValueError("ConceptProbe needs at least one harmful/harmless pair.")

        harmful_mat  = np.stack(harmful_embeddings[:n]).astype(np.float32)   # [n, d]
        harmless_mat = np.stack(harmless_embeddings[:n]).astype(np.float32)  # [n, d]

        self.mean_harmful  = harmful_mat.mean(axis=0)   # [d]
        self.mean_harmless = harmless_mat.mean(axis=0)  # [d]

        # SVD rank-1 direction of paired difference matrix
        diff = harmful_mat - harmless_mat  # [n, d]
        _, _, Vh = np.linalg.svd(diff, full_matrices=False)
        v = Vh[0].copy()

        # Align sign: mean(harmful - harmless) should project positively onto v
        mean_diff = self.mean_harmful - self.mean_harmless
        if np.dot(mean_diff, v) < 0:
            v = -v
        self.toxic_vector: np.ndarray = v / (np.linalg.norm(v) + 1e-8)

        self._n_cal = n
        self.delta = float(np.dot(mean_diff, self.toxic_vector))
        print(f"  [ConceptProbe] n_cal={n}  delta={self.delta:.4f}")

    # ── Scoring ────────────────────────────────────────────────────────────────

    def score(self, h: np.ndarray) -> float:
        """
        Toxic concept score for a single hidden state.

        score = cosine_sim(h - mean_harmless, toxic_vector)
        """
        centered = (h - self.mean_harmless).astype(np.float32)
        norm = np.linalg.norm(centered)
        if norm < 1e-8:
            return 0.0
        return float(np.dot(centered / norm, self.toxic_vector))

    def score_batch(self, vectors: list[np.ndarray]) -> list[float]:
        return [self.score(h) for h in vectors]

    def is_jailbreak(self, h: np.ndarray, tau: float = 0.3) -> bool:
        """True if concept score ≥ tau."""
        return self.score(h) >= tau

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def calibration_summary(self) -> dict:
        return {
            "n_cal": self._n_cal,
            "delta": round(self.delta, 6),
            "vector_norm": round(float(np.linalg.norm(self.toxic_vector)), 6),
        }
