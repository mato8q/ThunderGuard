"""
Memory Bank — stores activation vectors for attack and benign prompts.
Faithful implementation of §3.5 (Eq. 8–10) from the paper.

Two-tier architecture:
  Short-term memory M_S : buffer for high-confidence Stage-1 results within
                          the current detection cycle.  Not yet persisted.
  Long-term memory M_a / M_b : permanent store updated from short-term.
                                Only Stage-2-verified entries reach here.

Update flow (Algorithm 1, lines 11-12):
  Stage 1 confident → short-term  (Eq. 8)
  Stage 2 verified  → long-term   (Eq. 9–10)
  evaluate() flushes short-term → long-term at end of cycle.
"""

import os
import numpy as np


class MemoryBank:
    def __init__(self, save_path: str = "data/memory.npz"):
        self.save_path = save_path

        # Long-term memory (M_a, M_b) — Eq. 9–10
        self._lt_attack: list[np.ndarray] = []
        self._lt_benign: list[np.ndarray] = []

        # Short-term memory (M_S) — Eq. 8
        # Each entry: (vector, label) where label ∈ {"attack", "benign"}
        self._st_buffer: list[tuple[np.ndarray, str]] = []

        self._load_if_exists()

    # ── Short-term (Stage 1 → buffer) ────────────────────────────────────────

    def stage1_add(self, h_x: np.ndarray, label: str):
        """
        Store Stage-1 confident result in short-term memory (Eq. 8).
        Called after a confident ATTACK or BENIGN decision from ImmuneDetector.
        label: "attack" | "benign"
        """
        self._st_buffer.append((h_x.flatten(), label))

    def flush_short_term(self):
        """
        Promote all short-term entries to long-term memory (Eq. 9–10).
        Called at the end of a detection cycle (after Stage 2 if needed).
        """
        for vec, label in self._st_buffer:
            if label == "attack":
                self._lt_attack.append(vec)
            else:
                self._lt_benign.append(vec)
        self._st_buffer.clear()
        self.save()

    # ── Long-term (Stage 2 verified → permanent) ──────────────────────────────

    def add_attack(self, h_x: np.ndarray):
        """Directly add a Stage-2-verified attack to long-term memory (Eq. 9)."""
        self._lt_attack.append(h_x.flatten())

    def add_benign(self, h_x: np.ndarray):
        """Directly add a Stage-2-verified benign to long-term memory (Eq. 10)."""
        self._lt_benign.append(h_x.flatten())

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_attack(self) -> np.ndarray:
        """Return long-term attack memory [n, hidden_dim] or empty array."""
        return np.array(self._lt_attack) if self._lt_attack else np.array([])

    def get_benign(self) -> np.ndarray:
        """Return long-term benign memory [n, hidden_dim] or empty array."""
        return np.array(self._lt_benign) if self._lt_benign else np.array([])

    @property
    def is_ready(self) -> bool:
        """Ready for immune detection when both long-term banks are non-empty."""
        return len(self._lt_attack) > 0 and len(self._lt_benign) > 0

    @property
    def short_term_count(self) -> int:
        return len(self._st_buffer)

    def stats(self) -> str:
        return (
            f"Long-term: {len(self._lt_attack)} attack, {len(self._lt_benign)} benign | "
            f"Short-term buffer: {len(self._st_buffer)} pending"
        )

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self):
        os.makedirs(
            os.path.dirname(self.save_path) if os.path.dirname(self.save_path) else ".",
            exist_ok=True,
        )
        np.savez(
            self.save_path,
            attack=np.array(self._lt_attack) if self._lt_attack else np.zeros((0,)),
            benign=np.array(self._lt_benign) if self._lt_benign else np.zeros((0,)),
        )

    def _load_if_exists(self):
        if not os.path.exists(self.save_path):
            return
        data = np.load(self.save_path, allow_pickle=True)
        if data["attack"].ndim == 2 and data["attack"].shape[0] > 0:
            self._lt_attack = list(data["attack"])
        if data["benign"].ndim == 2 and data["benign"].shape[0] > 0:
            self._lt_benign = list(data["benign"])
        print(f"  Memory loaded: {len(self._lt_attack)} attack, {len(self._lt_benign)} benign")
