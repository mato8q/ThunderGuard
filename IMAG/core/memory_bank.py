"""
Policy-Driven Memory Bank — ตามเปเปอร์ JailDAM §2.3 และ §2.5

- สร้าง memory จาก harmful concept descriptions (ไม่ต้องใช้ harmful data จริง)
- คำนวณ attention score: Z = softmax(E(S) · P^T / sqrt(d))   [Eq. 6]
- Test-time adaptation: แทนที่ memory ที่ใช้น้อยสุดด้วย residual  [Eq. 8, 9]
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.harmful_concepts import HARMFUL_CONCEPTS


class MemoryBank:
    def __init__(self, encoder, concepts_dict=None, gamma=0.8, top_k=5):
        """
        encoder   : CLIPTextEncoder
        gamma     : threshold สำหรับ trigger test-time adaptation  [Eq. 8]
        top_k     : จำนวน top-K memories ที่ใช้คำนวณ residual  [Eq. 9]
        """
        self.encoder = encoder
        self.gamma = gamma
        self.top_k = top_k

        if concepts_dict is None:
            concepts_dict = HARMFUL_CONCEPTS

        print("Building policy-driven memory bank...")
        self._build(concepts_dict)

    def _build(self, concepts_dict):
        all_embeddings = []
        for category, concepts in concepts_dict.items():
            embs = self.encoder.encode(concepts)   # [n_concepts, 512]
            all_embeddings.append(embs)

        self.memories = np.vstack(all_embeddings).astype(np.float32)  # [N, 512]
        self.frequencies = np.zeros(len(self.memories), dtype=np.int32)
        print(f"  Memory bank: {len(self.memories)} concepts, dim={self.memories.shape[1]}")

    @property
    def size(self):
        return len(self.memories)

    def compute_attention(self, embedding):
        """
        Z = softmax(E(S) · P^T / sqrt(d))  — Eq. 6
        คืน attention vector ขนาด [N] และ raw scores
        """
        d = embedding.shape[-1]
        scores = (embedding @ self.memories.T) / np.sqrt(d)   # [N]

        # Stable softmax
        scores_shifted = scores - scores.max()
        exp_scores = np.exp(scores_shifted)
        attention = exp_scores / exp_scores.sum()              # [N]

        # อัพเดท frequency counter ของ memory ที่ถูกใช้มากสุด
        self.frequencies[np.argmax(attention)] += 1

        return attention, scores

    def adapt(self, embedding, attention):
        """
        Test-time adaptation — Eq. 8 & 9

        ถ้า max softmax > γ:
          1. คำนวณ residual R = S - Σ w_i * P_i  (top-K weighted sum)
          2. แทนที่ memory ที่ใช้น้อยสุดด้วย R
        """
        max_softmax = attention.max()
        if max_softmax <= self.gamma:
            return False

        # Top-K memories ที่ใกล้เคียงที่สุด
        top_k_idx = np.argsort(attention)[-self.top_k:]
        top_k_attn = attention[top_k_idx]

        # Softmax weights สำหรับ top-K
        exp_w = np.exp(top_k_attn - top_k_attn.max())
        weights = exp_w / exp_w.sum()                          # Eq. 9

        # Residual representation
        weighted_sum = (weights[:, None] * self.memories[top_k_idx]).sum(axis=0)
        residual = embedding - weighted_sum

        # Normalize
        norm = np.linalg.norm(residual)
        if norm > 1e-8:
            residual = residual / norm

        # แทนที่ memory ที่ถูกใช้น้อยสุด  — Eq. 8
        least_used_idx = np.argmin(self.frequencies)
        self.memories[least_used_idx] = residual.astype(np.float32)
        self.frequencies[least_used_idx] = 0

        return True
