"""
IMAG evaluation on JBShield's own test data — for fair comparison with JBShield-D.

Attack test : baselines/jbshield/data/jailbreak/{attack}/mistral_test.json  (jailbreak field)
Benign test : baselines/jbshield/data/harmless.csv  rows 51-250 (Alpaca)

JBShield run_detection.py uses the same test + benign data (OUR_TEST_DATASETS = {}).

Usage:
    cd IMAG
    python evaluate_jbshield.py --seeds handcrafted --no-active-immunity
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import gc

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from core.llm_interface import TargetLLM
from data.seed_prompts import SEED_ATTACKS, SEED_BENIGNS
from modules.active_immunity import ActiveImmunity
from modules.immune_detector import ImmuneDetector
from modules.memory_bank import MemoryBank

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_NAME  = "mistralai/Mistral-7B-Instruct-v0.2"
TOP_K       = 5
THRESHOLD_T = 0.1
POOLING     = "last"
N_SEEDS     = 30

_HERE        = os.path.dirname(os.path.abspath(__file__))
JBSHIELD_DIR = os.path.join(_HERE, "..", "baselines", "jbshield")
ADVBENCH_PATH = os.path.join(_HERE, "..", "adversarial-prompt", "data", "original", "harmful_behaviors.csv")

ATTACK_DATASETS = ["autodan", "zulu", "base64", "drattack", "pair", "gcg", "ijp", "saa", "puzzler"]
BENIGN_DATASETS = ["jbshield_harmless"]
ALL_DATASETS    = ATTACK_DATASETS + BENIGN_DATASETS

# ── Data loaders ───────────────────────────────────────────────────────────────

def _load_jbshield_attack(attack: str, model: str = "mistral") -> list[str]:
    path = os.path.join(JBSHIELD_DIR, "data", "jailbreak", attack, f"{model}_test.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"JBShield test JSON not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["jailbreak"] for item in data if item.get("jailbreak", "").strip()]
    print(f"  [JBShield JSON] {attack}: {len(texts)} prompts")
    return texts


def _load_jbshield_harmless(skip: int = 50, cap: int = 200) -> list[str]:
    """Alpaca harmless rows skip..skip+cap — same split as run_original.py test set."""
    path = os.path.join(JBSHIELD_DIR, "data", "harmless.csv")
    with open(path, encoding="utf-8") as f:
        rows = [r["prompt"].strip() for r in csv.DictReader(f) if r.get("prompt", "").strip()]
    subset = rows[skip: skip + cap]
    print(f"  [JBShield harmless.csv] rows {skip}–{skip + cap}: {len(subset)} prompts")
    return subset

# ── Seed builder ───────────────────────────────────────────────────────────────

def build_seed_vectors(
    llm: TargetLLM,
    critical_layer: int,
    n_seeds: int,
    seed_strategy: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:

    def _enc(texts, label):
        out = []
        for i, t in enumerate(texts, 1):
            out.append(llm.extract_hidden_states(t, target_layer=critical_layer, pooling=POOLING))
            print(f"  {label} {i}/{len(texts)}", end="\r")
        print()
        return out

    if seed_strategy == "advbench":
        if not os.path.exists(ADVBENCH_PATH):
            print("  [WARN] AdvBench not found — falling back to handcrafted seeds")
            atk_texts = SEED_ATTACKS[:n_seeds]
        else:
            with open(ADVBENCH_PATH, encoding="utf-8") as f:
                atk_texts = [r["goal"].strip() for r in csv.DictReader(f) if r.get("goal", "").strip()][:n_seeds]
    else:
        atk_texts = SEED_ATTACKS[:n_seeds]

    ben_texts = SEED_BENIGNS[:n_seeds]
    print(f"  Encoding {len(atk_texts)} attack seeds ({seed_strategy})...")
    seed_atk = _enc(atk_texts, "atk-seed")
    print(f"  Encoding {len(ben_texts)} benign seeds...")
    seed_ben = _enc(ben_texts, "ben-seed")
    return seed_atk, seed_ben

# ── Encode dataset ─────────────────────────────────────────────────────────────

def encode_dataset(
    dataset: str,
    llm: TargetLLM,
    critical_layer: int,
    rng: random.Random,
) -> dict:

    def _enc(texts, label):
        import torch
        out = []
        for i, t in enumerate(texts, 1):
            out.append(llm.extract_hidden_states(t, target_layer=critical_layer, pooling=POOLING))
            print(f"    {label} {i}/{len(texts)}", end="\r")
            if i % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        if texts:
            print()
        return out

    if dataset == "jbshield_harmless":
        all_texts = _load_jbshield_harmless()
        rng.shuffle(all_texts)
        seed_texts = all_texts[:N_SEEDS]
        eval_texts = all_texts[N_SEEDS:]
        print(f"\n  {dataset}: {len(all_texts)} benign ({N_SEEDS} seeds + {len(eval_texts)} eval) — encoding...")
        seed_h = _enc(seed_texts, "seed")
        eval_h = _enc(eval_texts, "eval")
        return {
            "ds_seed_atk": [], "ds_seed_ben": seed_h,
            "atk_texts": [],   "atk_h": [],
            "ben_texts": eval_texts, "ben_h": eval_h,
        }
    else:
        all_texts = _load_jbshield_attack(dataset)
        rng.shuffle(all_texts)
        print(f"\n  {dataset}: {len(all_texts)} attack (all eval) — encoding...")
        eval_h = _enc(all_texts, "eval")
        return {
            "ds_seed_atk": [], "ds_seed_ben": [],
            "atk_texts": all_texts, "atk_h": eval_h,
            "ben_texts": [],        "ben_h": [],
        }

# ── Metrics ────────────────────────────────────────────────────────────────────

def _compute_metrics(results: list, dataset: str, elapsed: float) -> dict:
    total = len(results)
    tp = sum(1 for t, p in results if t == "ATTACK" and p == "ATTACK")
    tn = sum(1 for t, p in results if t == "BENIGN" and p == "BENIGN")
    fp = sum(1 for t, p in results if t == "BENIGN" and p == "ATTACK")
    fn = sum(1 for t, p in results if t == "ATTACK" and p == "BENIGN")
    acc  = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    fpr  = fp / (fp + tn) if (fp + tn) else 0
    if dataset in BENIGN_DATASETS:
        safe = tn + fp
        if safe:
            print(f"  Pass-through: {tn/safe*100:.1f}%  FPR: {fpr*100:.1f}%")
    else:
        if (tp + fn):
            print(f"  Acc={acc:.3f}  F1={f1:.3f}  Detection={rec*100:.1f}%  Miss={fn/(tp+fn)*100:.1f}%")
    return {
        "dataset": dataset, "total": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4), "fpr": round(fpr, 4),
        "time_s": round(elapsed, 1),
    }


def classify_cached(
    dataset: str,
    attack_texts: list, attack_h: list,
    benign_texts: list, benign_h: list,
    seed_atk: list, seed_ben: list,
    threshold: float, no_ai: bool, ai=None,
) -> dict:
    memory = MemoryBank.__new__(MemoryBank)
    memory.save_path = None
    memory._lt_attack = [v.copy() for v in seed_atk]
    memory._lt_benign = [v.copy() for v in seed_ben]
    memory._st_buffer = []

    detector = ImmuneDetector(threshold_T=threshold, top_k=TOP_K)
    results = []
    gap_records = []
    t_start = time.time()
    total = len(attack_h) + len(benign_h)

    def _step(h_x, text, true_label, idx):
        label, s_a, s_b = detector.detect(h_x, memory.get_attack(), memory.get_benign())
        gap = s_b - s_a
        if label == "CANDIDATE":
            if no_ai:
                label = "BENIGN"
            else:
                label, *_ = ai.evaluate(text)
                if label == "ATTACK":
                    memory._lt_attack.append(h_x)
                else:
                    memory._lt_benign.append(h_x)
        else:
            if label == "ATTACK":
                memory._lt_attack.append(h_x)
            else:
                memory._lt_benign.append(h_x)
        results.append((true_label, label))
        gap_records.append((true_label, gap))
        print(f"  [{idx:>3}/{total}] {true_label} → {label:<8}  gap={gap:+.4f}", end="\r")

    for i, (text, h_x) in enumerate(zip(attack_texts, attack_h), 1):
        _step(h_x, text, "ATTACK", i)
    for i, (text, h_x) in enumerate(zip(benign_texts, benign_h), 1):
        _step(h_x, text, "BENIGN", len(attack_h) + i)
    print()

    metrics = _compute_metrics(results, dataset, time.time() - t_start)
    metrics["_gap_records"] = gap_records
    return metrics

# ── Save ───────────────────────────────────────────────────────────────────────

def save_results(rows: list[dict], model_name: str, threshold: float, no_ai: bool) -> str:
    from datetime import datetime
    os.makedirs("results", exist_ok=True)
    model_short = model_name.split("/")[-1]
    ai_tag = "noAI" if no_ai else "fullAI"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"results/{model_short}_jbshield_t{threshold}_{ai_tag}_{ts}.csv"
    fields = ["dataset", "total", "tp", "tn", "fp", "fn",
              "acc", "precision", "recall", "f1", "fpr", "time_s"]
    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Results saved → {fname}")
    return fname

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global N_SEEDS, POOLING
    parser = argparse.ArgumentParser(description="IMAG evaluated on JBShield test data")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--threshold", type=float, default=THRESHOLD_T)
    parser.add_argument("--no-active-immunity", action="store_true")
    parser.add_argument("--seeds", choices=["advbench", "handcrafted"], default="handcrafted",
                        help="Seed strategy: 'handcrafted' (fair, no leakage) or 'advbench'")
    parser.add_argument("--seed-size", type=int, default=N_SEEDS)
    parser.add_argument("--pooling", choices=["last", "mean"], default=None)
    args = parser.parse_args()

    N_SEEDS = args.seed_size
    if args.pooling:
        POOLING = args.pooling

    rng = random.Random(42)
    model_name = args.model

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print(f"  IMAG — JBShield Test Data Evaluation")
    print(f"  Model    : {model_name}")
    print(f"  Attacks  : {ATTACK_DATASETS}")
    print(f"  Benign   : JBShield harmless.csv rows 51-250 (Alpaca)")
    print(f"  Threshold: {args.threshold}")
    print(f"  Seeds    : {args.seeds}  (n={args.seed_size})")
    print(f"  Active Immunity: {'OFF' if args.no_active_immunity else 'ON'}")
    print("=" * 65)

    print(f"\n[1] Loading model: {model_name}")
    llm = TargetLLM(model_name, device=device)

    critical_layer = 31
    print(f"\n[2] Critical layer: {critical_layer} (Mistral-7B default; use evaluate.py --find-critical-layer to override)")

    print("\n[3] Encoding seed prompts...")
    seed_atk, seed_ben = build_seed_vectors(llm, critical_layer, args.seed_size, args.seeds)

    print(f"\n[4] Encoding {len(ALL_DATASETS)} dataset(s)...")
    encodings: dict[str, dict] = {}
    for ds in ALL_DATASETS:
        try:
            encodings[ds] = encode_dataset(ds, llm, critical_layer, rng)
        except FileNotFoundError as e:
            print(f"  SKIP {ds}: {e}")

    actual_seed_ben = []
    if "jbshield_harmless" in encodings:
        actual_seed_ben = encodings["jbshield_harmless"]["ds_seed_ben"]
    ref_ben = actual_seed_ben if actual_seed_ben else seed_ben
    print(f"\n  Benign reference: {len(ref_ben)} seeds from JBShield harmless.csv")

    hidden_dim = llm.model.config.hidden_size
    atk_mb  = round(len(seed_atk) * hidden_dim * 2 / (1024 ** 2), 4)
    ben_mb  = round(len(ref_ben)  * hidden_dim * 2 / (1024 ** 2), 4)

    print(f"\n[5] Classifying  (T={args.threshold})...")
    all_results = []
    t_total = time.time()

    for ds, enc in encodings.items():
        print(f"\n{'='*65}")
        print(f"  Dataset: {ds}")
        ds_seed_atk = seed_atk
        ds_seed_ben = enc["ds_seed_ben"] if ds in BENIGN_DATASETS and enc["ds_seed_ben"] else ref_ben

        ai = ActiveImmunity(agent_llm=llm) if not args.no_active_immunity else None
        row = classify_cached(
            ds, enc["atk_texts"], enc["atk_h"], enc["ben_texts"], enc["ben_h"],
            ds_seed_atk, ds_seed_ben,
            args.threshold, args.no_active_immunity, ai,
        )
        all_results.append(row)

    print("\n" + "=" * 65)
    print(f"  SUMMARY  (IMAG on JBShield data  |  {model_name.split('/')[-1]})  T={args.threshold}")
    print(f"  {'Dataset':<16} {'Acc':>6} {'F1':>6} {'Recall':>8} {'FPR':>6}")
    print("  " + "-" * 46)
    for r in all_results:
        if r["dataset"] in BENIGN_DATASETS:
            print(f"  {r['dataset']:<16} {'—':>6} {'—':>6} {'—':>8} {r['fpr']*100:>5.1f}%")
        else:
            print(f"  {r['dataset']:<16} {r['acc']:>6.3f} {r['f1']:>6.3f} {r['recall']*100:>7.1f}% {'—':>6}")

    atk_rows = [r for r in all_results if r["dataset"] in ATTACK_DATASETS]
    if atk_rows:
        avg_acc = sum(r["acc"] for r in atk_rows) / len(atk_rows)
        avg_f1  = sum(r["f1"]  for r in atk_rows) / len(atk_rows)
        print("  " + "-" * 46)
        print(f"  {'Average':<16} {avg_acc:>6.3f} {avg_f1:>6.3f}")

    print(f"\n  Total wall time: {time.time()-t_total:.1f}s")
    print(f"\n  Memory Bank  ({hidden_dim}d, fp16):")
    print(f"    Attack seeds : {len(seed_atk):>4} vecs = {atk_mb:.4f} MB")
    print(f"    Benign seeds : {len(ref_ben):>4} vecs = {ben_mb:.4f} MB")
    print(f"    Total        :            {atk_mb + ben_mb:.4f} MB")
    print("=" * 65)

    save_results(all_results, model_name, args.threshold, args.no_active_immunity)


if __name__ == "__main__":
    main()
