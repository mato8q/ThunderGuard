"""
jailbreak_detector.py – Embedding-based jailbreak detection (RAD layer).

Uses cosine similarity between the incoming prompt and stored jailbreak
signatures.  A prompt is flagged if its max similarity to any known
jailbreak exceeds the threshold.

Based on:
  - JBFuzz §4.5: e5-base-v2 + MLP outperforms GPT-4o evaluator 16×
  - Galinkin & Sablotny (2024): Random Forest / NN on top of embeddings
    achieves F1 = 0.96 on JailbreakHub with FPR < 1%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from vector_store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    is_jailbreak: bool
    score: float          # highest cosine similarity found (0–1)
    nearest_text: str     # closest known jailbreak example


class JailbreakDetector:
    """
    Checks an input prompt against the FAISS jailbreak vector store.

    Decision rule (RAD – Retrieval-Augmented Defence):
      max_sim = max cosine_similarity(prompt, jailbreak_i)
      if max_sim >= threshold  →  is_jailbreak = True
    """

    def __init__(
        self,
        vector_store: VectorStore,
        threshold: float = 0.85,
        top_k: int = 5,
    ) -> None:
        self.vector_store = vector_store
        self.threshold = threshold
        self.top_k = top_k

        log.info("JailbreakDetector ready (threshold=%.2f, k=%d, store_size=%d)",
                 threshold, top_k, vector_store.size)

    def check(self, text: str) -> DetectionResult:
        """
        Returns a DetectionResult.

        If the store is empty (cold start), the prompt is allowed through
        so the system degrades gracefully before seed data is loaded.
        """
        if self.vector_store.size == 0:
            log.warning("JailbreakDetector: store is empty – cold start, allowing prompt.")
            return DetectionResult(is_jailbreak=False, score=0.0, nearest_text="")

        results = self.vector_store.search(text, k=self.top_k)

        # Filter to only labelled-jailbreak neighbours
        jailbreak_hits = [r for r in results if r["label"] == "jailbreak"]

        if not jailbreak_hits:
            return DetectionResult(is_jailbreak=False, score=0.0, nearest_text="")

        top = max(jailbreak_hits, key=lambda r: r["score"])
        is_jb = top["score"] >= self.threshold

        log.debug("JailbreakDetector: max_sim=%.4f (threshold=%.2f) → %s",
                  top["score"], self.threshold, "BLOCK" if is_jb else "pass")

        return DetectionResult(
            is_jailbreak=is_jb,
            score=top["score"],
            nearest_text=top["text"],
        )
