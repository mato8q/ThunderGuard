"""
vector_store.py – FAISS-backed vector store for RAD (Retrieval-Augmented Defence).

Stores labelled embeddings (jailbreak / benign) and supports cosine similarity
nearest-neighbour search.  Uses intfloat/e5-base-v2 as the encoder, consistent
with the JBFuzz evaluator finding (16× faster than GPT-4o, higher accuracy).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

_MODEL_NAME = "intfloat/e5-base-v2"
_DIM = 768   # e5-base-v2 output dimension


class VectorStore:
    """
    Persistent FAISS index with metadata sidecar (JSON).

    Embeddings are L2-normalised before insertion so that inner-product
    search equals cosine similarity.
    """

    def __init__(self, store_path: str = "jailbreak_store") -> None:
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self._index_file = self.store_path / "index.faiss"
        self._meta_file  = self.store_path / "meta.json"

        log.info("Loading encoder: %s", _MODEL_NAME)
        self._encoder = SentenceTransformer(_MODEL_NAME)

        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(_DIM)
        self._meta: list[dict] = []   # parallel list to index rows

        self._load()

    # ── Encode ────────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Return L2-normalised 768-d embedding for `text`."""
        # e5 models expect a "query: " prefix for retrieval tasks
        vec = self._encoder.encode(f"query: {text}", normalize_embeddings=True)
        return vec.astype(np.float32).reshape(1, -1)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, text: str, k: int = 5) -> list[dict]:
        """
        Return the top-k nearest neighbours with their cosine similarity scores.

        Returns [] if the index is empty.
        """
        if self._index.ntotal == 0:
            return []

        vec = self.encode(text)
        k = min(k, self._index.ntotal)
        distances, indices = self._index.search(vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(dist),
                "label": self._meta[idx]["label"],
                "text":  self._meta[idx]["text"],
            })
        return results

    # ── Add ───────────────────────────────────────────────────────────────────

    def add(self, text: str, label: str = "jailbreak") -> None:
        """Embed and store a new example; persist to disk immediately."""
        vec = self.encode(text)
        self._index.add(vec)
        self._meta.append({"label": label, "text": text})
        self._save()
        log.info("VectorStore: added '%s' example (total=%d)", label, self._index.ntotal)

    def seed(self, examples: list[dict]) -> None:
        """
        Bulk-load initial seed examples.

        Args:
            examples: list of {"text": str, "label": str}
        """
        for ex in examples:
            self.add(ex["text"], ex.get("label", "jailbreak"))

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        faiss.write_index(self._index, str(self._index_file))
        self._meta_file.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2))

    def _load(self) -> None:
        if self._index_file.exists() and self._meta_file.exists():
            self._index = faiss.read_index(str(self._index_file))
            self._meta  = json.loads(self._meta_file.read_text())
            log.info("VectorStore: loaded %d entries from %s", self._index.ntotal, self.store_path)
        else:
            log.info("VectorStore: starting fresh (no existing index at %s)", self.store_path)

    @property
    def size(self) -> int:
        return self._index.ntotal
