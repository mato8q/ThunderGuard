"""
JailDAM Detector — ตามเปเปอร์ JailDAM §2.4 และ §2.5

Pipeline:
1. CLIP encode input → embedding
2. Attention Z = softmax(E(S) · P^T / sqrt(d))   [Eq. 6]
3. Autoencoder reconstruction error H(S) = MSE(D(Z), Z)   [Eq. 5]
4. H(S) > τ → ATTACK, ไม่เกิน → BENIGN
5. Test-time adaptation ถ้า max softmax > γ   [Eq. 8]
"""

import torch
import numpy as np


class JailDAM:
    def __init__(self, encoder, memory_bank, autoencoder, threshold_tau=0.01, device="cpu"):
        """
        threshold_tau : เส้นตัดสินใจ reconstruction error  (τ ในเปเปอร์)
        """
        self.encoder = encoder
        self.memory = memory_bank
        self.autoencoder = autoencoder
        self.tau = threshold_tau
        self.device = device

    def detect(self, text):
        """
        ตรวจสอบว่า text เป็น jailbreak หรือ benign

        Returns dict:
          is_unsafe          : bool
          label              : "ATTACK" | "BENIGN"
          reconstruction_error : float  (H(S) ในเปเปอร์)
          threshold          : float  (τ)
          max_softmax        : float  (ค่าสูงสุดใน attention vector)
          adapted            : bool   (มีการ update memory หรือไม่)
        """
        # Step 1: CLIP encode
        embedding = self.encoder.encode(text)[0]   # [512]

        # Step 2: Attention features Z = softmax(E(S)·P^T / sqrt(d))  Eq. 6
        attention, _ = self.memory.compute_attention(embedding)   # [N_concepts]

        # Step 3: Test-time adaptation  Eq. 8 & 9
        adapted = self.memory.adapt(embedding, attention)

        # Step 4: Reconstruction error H(S)  Eq. 5
        z = torch.tensor(attention, dtype=torch.float32).unsqueeze(0).to(self.device)
        error = self.autoencoder.reconstruction_error(z).item()

        # Step 5: Decision
        is_unsafe = error > self.tau

        return {
            "is_unsafe": is_unsafe,
            "label": "ATTACK" if is_unsafe else "BENIGN",
            "reconstruction_error": round(error, 6),
            "threshold": self.tau,
            "max_softmax": round(float(attention.max()), 4),
            "adapted": adapted,
        }
