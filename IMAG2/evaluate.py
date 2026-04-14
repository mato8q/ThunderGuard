"""
IMAG — Dataset Evaluation (faithful to Algorithm 1)
Loads attack prompts from HarmBench CSV + jailbreak val set,
benign prompts from XSTest / JailbreakBench / WildJailbreak (paper Table 1).

Usage:
    python evaluate.py
"""

import csv
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from core.llm_interface import TargetLLM
from data.seed_prompts import SEED_ATTACKS, SEED_BENIGNS
from modules.immune_detector import ImmuneDetector
from modules.memory_bank import MemoryBank

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME    = "mistralai/Mistral-7B-Instruct-v0.2"
TOP_K         = 5
THRESHOLD_T   = 0.5
POOLING       = "last"
MEMORY_PATH   = "data/eval_memory.npz"
LAYER_CACHE   = "data/critical_layer.txt"   # shared with main.py
DATASETS_DIR  = "/Users/keira/datasets"     # แก้ path ตรงนี้

RANDOM_SEED   = 42
MAX_ATTACK    = 200
MAX_BENIGN    = 200

# ── Benign dataset paths (paper Table 1) ─────────────────────────────────────
# TODO: แก้ path ด้านล่างให้ตรงกับที่วางไฟล์ไว้ในเครื่อง
#
# วิธีโหลดแต่ละ dataset:
#   XSTest        : huggingface-cli download paul-rottger/xstest --repo-type dataset
#   JailbreakBench: https://github.com/JailbreakBench/jailbreakbench → benign_behaviors.csv
#   WildJailbreak : huggingface-cli download allenai/wildjailbreak --repo-type dataset
#
# วางไฟล์ตาม structure นี้:
#   datasets/
#   └── benign/
#       ├── xstest.csv                  ← column: "prompt", กรอง type == "safe"
#       ├── jailbreakbench_benign.csv   ← column: "Goal"
#       └── wildjailbreak_benign.csv    ← column: "vanilla", กรอง data_type == "benign"

XSTEST_PATH          = "/path/to/xstest.csv"                # TODO: แก้ path
JAILBREAKBENCH_PATH  = "/path/to/jailbreakbench_benign.csv" # TODO: แก้ path
WILDJAILBREAK_PATH   = "/path/to/wildjailbreak_benign.csv"  # TODO: แก้ path

# ── Benign loader ─────────────────────────────────────────────────────────────

def load_benign_texts(max_samples: int, rng: random.Random) -> list[str]:
    texts = []

    # XSTest — 200 benign (type == "safe")
    if os.path.exists(XSTEST_PATH):
        with open(XSTEST_PATH, newline="", encoding="utf-8") as f:
            rows = [r["prompt"].strip() for r in csv.DictReader(f)
                    if r.get("type", "").strip() == "safe" and r.get("prompt", "").strip()]
        texts.extend(rows)
        print(f"  Loaded {len(rows)} benign from XSTest")
    else:
        print(f"  [SKIP] XSTest not found: {XSTEST_PATH}")

    # JailbreakBench — 100 benign
    if os.path.exists(JAILBREAKBENCH_PATH):
        with open(JAILBREAKBENCH_PATH, newline="", encoding="utf-8") as f:
            rows = [r["Goal"].strip() for r in csv.DictReader(f)
                    if r.get("Goal", "").strip()]
        texts.extend(rows)
        print(f"  Loaded {len(rows)} benign from JailbreakBench")
    else:
        print(f"  [SKIP] JailbreakBench not found: {JAILBREAKBENCH_PATH}")

    # WildJailbreak — 200 benign (data_type == "benign")
    if os.path.exists(WILDJAILBREAK_PATH):
        with open(WILDJAILBREAK_PATH, newline="", encoding="utf-8") as f:
            rows = [r["vanilla"].strip() for r in csv.DictReader(f)
                    if r.get("data_type", "").strip() == "benign" and r.get("vanilla", "").strip()]
        texts.extend(rows)
        print(f"  Loaded {len(rows)} benign from WildJailbreak")
    else:
        print(f"  [SKIP] WildJailbreak not found: {WILDJAILBREAK_PATH}")

    if not texts:
        raise FileNotFoundError(
            "No benign datasets found. Please set XSTEST_PATH, "
            "JAILBREAKBENCH_PATH, WILDJAILBREAK_PATH correctly."
        )

    rng.shuffle(texts)
    return texts[:max_samples]

# ── Attack loader ─────────────────────────────────────────────────────────────

def load_attack_texts(datasets_dir: str, max_samples: int, rng: random.Random) -> list[str]:
    texts = []

    csv_path = os.path.join(datasets_dir, "behavior_datasets",
                            "harmbench_behaviors_text_all.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = [r["Behavior"].strip() for r in csv.DictReader(f)
                    if r.get("Behavior", "").strip()]
        texts.extend(rows)
        print(f"  Loaded {len(rows)} from harmbench_behaviors_text_all.csv")

    json_path = os.path.join(datasets_dir, "classifier_val_sets",
                             "text_behaviors_val_set.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            val_data = json.load(f)
        jailbreaks = [e["test_case"].strip()
                      for entries in val_data.values()
                      for e in entries
                      if e.get("test_case", "").strip()]
        texts.extend(jailbreaks)
        print(f"  Loaded {len(jailbreaks)} from text_behaviors_val_set.json")

    rng.shuffle(texts)
    return texts[:max_samples]

# ── Layer helper ──────────────────────────────────────────────────────────────

def get_critical_layer(llm: TargetLLM) -> int:
    if os.path.exists(LAYER_CACHE):
        with open(LAYER_CACHE) as f:
            layer = int(f.read().strip())
        print(f"  Critical layer (cached): {layer}")
        return layer
    layer = llm.find_critical_layer(
        SEED_ATTACKS[:min(len(SEED_ATTACKS), len(SEED_BENIGNS))],
        SEED_BENIGNS[:min(len(SEED_ATTACKS), len(SEED_BENIGNS))],
        pooling=POOLING,
    )
    os.makedirs("data", exist_ok=True)
    with open(LAYER_CACHE, "w") as f:
        f.write(str(layer))
    return layer

# ── Memory seeding ─────────────────────────────────────────────────────────────

def seed_memory(llm: TargetLLM, memory: MemoryBank, critical_layer: int):
    print(f"  Encoding seed prompts (layer {critical_layer})...")
    for p in SEED_ATTACKS:
        h = llm.extract_hidden_states(p, target_layer=critical_layer, pooling=POOLING)
        memory.add_attack(h)
    for p in SEED_BENIGNS:
        h = llm.extract_hidden_states(p, target_layer=critical_layer, pooling=POOLING)
        memory.add_benign(h)
    memory.save()
    print(f"  Seeded: {memory.stats()}")

# ── Single-prompt detection (memory frozen) ────────────────────────────────────

def detect_one(
    prompt: str,
    llm: TargetLLM,
    detector: ImmuneDetector,
    memory: MemoryBank,
    critical_layer: int,
) -> tuple[str, float, float]:
    h_x = llm.extract_hidden_states(prompt, target_layer=critical_layer, pooling=POOLING)
    label, s_a, s_b = detector.detect(h_x, memory.get_attack(), memory.get_benign())

    if label == "CANDIDATE":
        label = "ATTACK" if (s_b - s_a) >= 0 else "BENIGN"

    return label, s_a, s_b

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(RANDOM_SEED)
    print("=" * 65)
    print("  IMAG — Dataset Evaluation (Algorithm 1 faithful)")
    print("=" * 65)

    print(f"\n[1] Loading LLM: {MODEL_NAME}")
    llm = TargetLLM(MODEL_NAME, device="cuda")

    print("\n[2] Finding critical layer (Eq. 2)...")
    critical_layer = get_critical_layer(llm)

    print("\n[3] Building eval memory bank...")
    if os.path.exists(MEMORY_PATH):
        os.remove(MEMORY_PATH)
    memory = MemoryBank(save_path=MEMORY_PATH)
    seed_memory(llm, memory, critical_layer)

    print(f"\n[4] Loading datasets from {os.path.normpath(DATASETS_DIR)}")
    attack_texts = load_attack_texts(DATASETS_DIR, MAX_ATTACK, rng)
    benign_texts = load_benign_texts(MAX_BENIGN, rng)
    print(f"  Test set: {len(attack_texts)} attack, {len(benign_texts)} benign")

    detector = ImmuneDetector(threshold_T=THRESHOLD_T, top_k=TOP_K)

    print("\n[5] Running detection (memory frozen)...")
    print("-" * 65)

    results = []
    total = len(attack_texts) + len(benign_texts)
    t_start = time.time()

    for i, text in enumerate(attack_texts):
        pred, s_a, s_b = detect_one(text, llm, detector, memory, critical_layer)
        results.append(("ATTACK", pred))
        elapsed = time.time() - t_start
        avg = elapsed / (i + 1)
        remaining = avg * (total - i - 1)
        print(f"  [{i+1:>3}/{total}] ATTACK  → {pred:<6}  "
              f"gap={s_b-s_a:+.4f}  ETA {remaining:.0f}s", end="\r")

    for i, text in enumerate(benign_texts):
        pred, s_a, s_b = detect_one(text, llm, detector, memory, critical_layer)
        results.append(("BENIGN", pred))
        done = len(attack_texts) + i + 1
        elapsed = time.time() - t_start
        avg = elapsed / done
        remaining = avg * (total - done)
        print(f"  [{done:>3}/{total}] BENIGN  → {pred:<6}  "
              f"gap={s_b-s_a:+.4f}  ETA {remaining:.0f}s", end="\r")

    print()

    print("\n" + "=" * 65)
    tp = sum(1 for t, p in results if t == "ATTACK" and p == "ATTACK")
    tn = sum(1 for t, p in results if t == "BENIGN" and p == "BENIGN")
    fp = sum(1 for t, p in results if t == "BENIGN" and p == "ATTACK")
    fn = sum(1 for t, p in results if t == "ATTACK" and p == "BENIGN")

    accuracy  = (tp + tn) / len(results) if results else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0)

    total_time = time.time() - t_start
    print(f"  Samples  : {len(results)} total  "
          f"({len(attack_texts)} attack, {len(benign_texts)} benign)")
    print(f"  Layer    : {critical_layer}  |  Threshold: {THRESHOLD_T}")
    print(f"  Confusion: TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"\n  Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n  Total time: {total_time:.1f}s  ({total_time/len(results):.2f}s/sample)")
    print("=" * 65)


if __name__ == "__main__":
    main()
