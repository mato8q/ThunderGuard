"""
Memory Bank — เก็บ activation vectors ของ attack และ benign prompts

memory_attack : รายการ hidden-state vectors ของ jailbreak prompts
memory_benign : รายการ hidden-state vectors ของ safe prompts
"""

import numpy as np
import os


class MemoryBank:
    def __init__(self, save_path="data/memory.npz"):
        self.save_path = save_path
        self.memory_attack: list[np.ndarray] = []
        self.memory_benign: list[np.ndarray] = []
        self._load_if_exists()

    # ── Add ───────────────────────────────────────────────────────────────────

    def add_attack(self, h_x: np.ndarray):
        self.memory_attack.append(h_x.flatten())

    def add_benign(self, h_x: np.ndarray):
        self.memory_benign.append(h_x.flatten())

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_attack(self) -> np.ndarray:
        """คืน numpy array [n, hidden_dim] หรือ [] ถ้าว่าง"""
        return np.array(self.memory_attack) if self.memory_attack else np.array([])

    def get_benign(self) -> np.ndarray:
        return np.array(self.memory_benign) if self.memory_benign else np.array([])

    @property
    def is_ready(self) -> bool:
        """พร้อมใช้งาน Immune Detection เมื่อมี memory ทั้ง 2 ฝั่ง"""
        return len(self.memory_attack) > 0 and len(self.memory_benign) > 0

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self):
        os.makedirs(os.path.dirname(self.save_path) if os.path.dirname(self.save_path) else ".", exist_ok=True)
        np.savez(
            self.save_path,
            attack=np.array(self.memory_attack),
            benign=np.array(self.memory_benign),
        )

    def _load_if_exists(self):
        if not os.path.exists(self.save_path):
            return
        data = np.load(self.save_path, allow_pickle=True)
        if data["attack"].ndim == 2:
            self.memory_attack = list(data["attack"])
        if data["benign"].ndim == 2:
            self.memory_benign = list(data["benign"])
        print(f"  Memory loaded: {len(self.memory_attack)} attack, {len(self.memory_benign)} benign")
