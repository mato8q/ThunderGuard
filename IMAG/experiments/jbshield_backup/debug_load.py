import traceback, sys
sys.path.insert(0, '.')

# Replicate run_detection.py imports to find which one causes the crash
print("importing torch...")
import torch
print("importing numpy...")
import numpy as np
print("importing sklearn...")
from sklearn.metrics import roc_curve
print("importing detection...")
from detection import find_critical_layer, find_optimal_threshold, detection_judge
print("importing utils extra...")
from utils import get_sentence_embeddings, interpret_difference_matrix, cosine_similarity
print("all imports OK")

try:
    from config import path_harmful_test, path_harmless_test, path_harmful_calibration, path_harmless_calibration
    from utils import load_ori_prompts, get_jailbreak_prompts

    print("Testing load_ori_prompts (test)...")
    a, b = load_ori_prompts(path_harmful_test, path_harmless_test)
    print(f"  OK: {len(a)} harmful, {len(b)} harmless")

    print("Testing load_ori_prompts (cal)...")
    c, d = load_ori_prompts(path_harmful_calibration, path_harmless_calibration)
    print(f"  OK: {len(c)} harmful, {len(d)} harmless")

    print("Testing get_jailbreak_prompts (calibration)...")
    jb = get_jailbreak_prompts("mistral", ["zulu"], split="calibration")
    print(f"  OK: {len(jb['zulu'])} zulu prompts")

    print("Testing get_jailbreak_prompts (calibration) all attacks...")
    jb_all = ["autodan", "zulu", "base64", "pair", "drattack", "gcg", "ijp", "saa", "puzzler"]
    for jb in jb_all:
        cal = get_jailbreak_prompts("mistral", [jb], split="calibration")
        test = get_jailbreak_prompts("mistral", [jb], split="test")
        print(f"  {jb}: cal={len(cal[jb])}  test={len(test[jb])}")

    print("All OK!")

except Exception as e:
    traceback.print_exc()