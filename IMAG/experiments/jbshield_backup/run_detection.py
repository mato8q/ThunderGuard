"""
JBShield detection wrapper — runs detection.py logic and prints a summary
table in the same style as IMAG's evaluate.py.

Usage:
    cd baselines/jbshield
    python run_detection.py --model mistral
"""

import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve

from config import model_paths
from config import path_harmful_test, path_harmless_test
from config import path_harmful_calibration, path_harmless_calibration
path_harmless_calibration = "data/harmless.csv"  # JBShield original (Alpaca)
path_harmful_calibration  = "data/harmful.csv"   # JBShield original
from utils import load_model, get_jailbreak_prompts
from utils import get_sentence_embeddings, interpret_difference_matrix, cosine_similarity

# pandas crashes on Windows after torch is imported — use pure-Python CSV loader instead
import csv as _csv, random as _random
def load_ori_prompts(path_harmful, path_harmless):
    def _read(path):
        with open(path, encoding="utf-8") as f:
            return [r["prompt"].strip() for r in _csv.DictReader(f) if r.get("prompt", "").strip()]
    harmful  = _read(path_harmful)
    harmless = _read(path_harmless)
    _random.seed(0)
    _random.shuffle(harmless)
    return harmful, harmless[:len(harmful)]
from utils import get_sentence_embeddings, interpret_difference_matrix, cosine_similarity

from detection import (
    find_critical_layer,
    find_optimal_threshold,
    detection_judge,
)

JAILBREAKS = ["autodan", "zulu", "base64", "pair", "drattack", "gcg", "ijp", "saa", "puzzler"]

# Use JBShield's own test JSON for all attacks (fair comparison with IMAG)
OUR_TEST_DATASETS = {}


def load_test_prompts(model_name: str) -> dict:
    """Load test prompts: our CSV for attacks we have, JBShield JSON for the rest."""
    import csv
    result = {}
    for jb in JAILBREAKS:
        if jb in OUR_TEST_DATASETS:
            path = OUR_TEST_DATASETS[jb]
            with open(path, encoding="utf-8") as f:
                prompts = [r["prompt"].strip() for r in csv.DictReader(f) if r.get("prompt", "").strip()]
            print(f"    [our data] {jb}: {len(prompts)} prompts")
            result[jb] = prompts
        else:
            jb_json = get_jailbreak_prompts(model_name, [jb], split="test")
            result[jb] = jb_json[jb]
            print(f"    [jbshield] {jb}: {len(result[jb])} prompts")
    return result


def run(model_name: str):
    t0 = time.time()
    print("=" * 65)
    print(f"  JBShield-D  —  Detection Evaluation")
    print(f"  Model   : {model_name}")
    print(f"  Attacks : {JAILBREAKS}")
    print("=" * 65)

    # ── Load prompts first (before model — avoids pandas/torch memory conflict) ──
    print("\n[1] Loading prompts...")
    _, harmless_prompts_test = load_ori_prompts(path_harmful_test, path_harmless_test)
    harmful_prompts_cal, harmless_prompts_cal = load_ori_prompts(
        path_harmful_calibration, path_harmless_calibration
    )
    jb_cal  = get_jailbreak_prompts(model_name, JAILBREAKS, split="calibration")
    jb_test = load_test_prompts(model_name)
    print(f"  Harmless test : {len(harmless_prompts_test)} prompts")
    print(f"  Calibration   : {len(harmful_prompts_cal)} harmful + {len(harmless_prompts_cal)} harmless")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[2] Loading model: {model_name}")
    model, tokenizer = load_model(model_name, model_paths)

    # ── Embeddings (calibration) ───────────────────────────────────────────────
    print("\n[3] Encoding calibration embeddings...")
    cal_harmless_emb = get_sentence_embeddings(harmless_prompts_cal, model, model_name, tokenizer)
    cal_harmful_emb  = get_sentence_embeddings(harmful_prompts_cal,  model, model_name, tokenizer)

    mean_harmful_emb  = [torch.mean(torch.stack(e), dim=0) for e in cal_harmful_emb]
    mean_harmless_emb = [torch.mean(torch.stack(e), dim=0) for e in cal_harmless_emb]

    cal_jb_emb = {}
    for jb in JAILBREAKS:
        print(f"    calibration: {jb}", end="\r")
        cal_jb_emb[jb] = get_sentence_embeddings(jb_cal[jb], model, model_name, tokenizer)
    print()

    # ── Embeddings (test) ─────────────────────────────────────────────────────
    print("\n[4] Encoding test embeddings...")
    test_harmless_emb = get_sentence_embeddings(harmless_prompts_test, model, model_name, tokenizer)
    test_jb_emb = {}
    for jb in JAILBREAKS:
        print(f"    test: {jb}", end="\r")
        test_jb_emb[jb] = get_sentence_embeddings(jb_test[jb], model, model_name, tokenizer)
    print()

    # ── Critical layers ───────────────────────────────────────────────────────
    print("\n[5] Finding critical layers...")
    _, safety_layer = find_critical_layer(cal_harmful_emb, cal_harmless_emb)
    jb_layers = {}
    for jb in JAILBREAKS:
        _, jb_layers[jb] = find_critical_layer(cal_jb_emb[jb], cal_harmful_emb)

    # ── Calibration vectors ───────────────────────────────────────────────────
    safety_vec, safety_delta = interpret_difference_matrix(
        model, tokenizer,
        cal_harmful_emb[safety_layer],
        cal_harmless_emb[safety_layer],
        return_tokens=False,
    )
    print(f"  safety_layer={safety_layer}  safety_delta={safety_delta:.4f}")

    jb_vecs = {}
    for jb in JAILBREAKS:
        vec, delta = interpret_difference_matrix(
            model, tokenizer,
            cal_jb_emb[jb][jb_layers[jb]],
            cal_harmful_emb[jb_layers[jb]],
            return_tokens=False,
        )
        jb_vecs[jb] = vec
        print(f"  {jb}: layer={jb_layers[jb]}  delta={delta:.4f}")
    # ── Classify ──────────────────────────────────────────────────────────────
    print("\n[6] Classifying...")
    rows = []

    for jb in tqdm(JAILBREAKS):
        layer = jb_layers[jb]

        thr_safety = find_optimal_threshold(
            model, tokenizer,
            cal_jb_emb[jb][safety_layer],
            cal_harmless_emb[safety_layer],
            mean_harmless_emb[safety_layer],
            safety_vec,
        )
        thr_jb = find_optimal_threshold(
            model, tokenizer,
            cal_jb_emb[jb][layer],
            cal_harmful_emb[layer],
            mean_harmful_emb[layer],
            jb_vecs[jb],
        )

        n = len(test_jb_emb[jb][safety_layer])

        print(f"  [{jb}] layer={layer} safety_layer={safety_layer} thr_safety={thr_safety:.4f} thr_jb={thr_jb:.4f}")

        res_jb_safety = detection_judge(model, tokenizer, test_jb_emb[jb][safety_layer],
                                        mean_harmless_emb[safety_layer], safety_vec, thr_safety)
        res_jb_jb     = detection_judge(model, tokenizer, test_jb_emb[jb][layer],
                                        mean_harmful_emb[layer], jb_vecs[jb], thr_jb)
        res_bn_safety = detection_judge(model, tokenizer, test_harmless_emb[safety_layer][:n],
                                        mean_harmless_emb[safety_layer], safety_vec, thr_safety)
        res_bn_jb     = detection_judge(model, tokenizer, test_harmless_emb[layer][:n],
                                        mean_harmful_emb[layer], jb_vecs[jb], thr_jb)

        # per-check stats
        atk_s = sum(res_jb_safety)/len(res_jb_safety)*100
        atk_j = sum(res_jb_jb)/len(res_jb_jb)*100
        bn_s  = sum(res_bn_safety)/len(res_bn_safety)*100
        bn_j  = sum(res_bn_jb)/len(res_bn_jb)*100
        print(f"    atk pass safety={atk_s:.1f}%  jb={atk_j:.1f}%  |  benign pass safety={bn_s:.1f}%  jb={bn_j:.1f}%")
        labels_jb = [1.0 if s == 1.0 and j == 1.0 else 0.0 for s, j in zip(res_jb_safety, res_jb_jb)]
        labels_bn = [1.0 if s == 1.0 and j == 1.0 else 0.0 for s, j in zip(res_bn_safety, res_bn_jb)]

        tp = sum(labels_jb);  fp = sum(labels_bn)
        fn = len(labels_jb) - tp;  tn = len(labels_bn) - fp

        acc  = (tp + tn) / (len(labels_jb) + len(labels_bn))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0

        rows.append({
            "dataset": jb, "total": len(labels_jb) + len(labels_bn),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "acc": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4), "fpr": round(fpr, 4),
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  SUMMARY  (JBShield-D  |  model={model_name})")
    print(f"  {'Dataset':<12} {'Acc':>6} {'Precision':>10} {'Recall':>8} {'F1':>6} {'FPR':>6}")
    print("  " + "-" * 50)
    for r in rows:
        print(f"  {r['dataset']:<12} {r['acc']:>6.3f} {r['precision']:>10.3f} "
              f"{r['recall']:>8.3f} {r['f1']:>6.3f} {r['fpr']*100:>5.1f}%")
    print("  " + "-" * 50)
    avg_f1  = sum(r["f1"]  for r in rows) / len(rows)
    avg_acc = sum(r["acc"] for r in rows) / len(rows)
    print(f"  {'Average':<12} {avg_acc:>6.3f} {'':>10} {'':>8} {avg_f1:>6.3f}")
    print(f"\n  Total wall time: {time.time()-t0:.1f}s")
    print("=" * 65)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    import csv, os
    from datetime import datetime
    os.makedirs("results", exist_ok=True)
    fname = f"results/jbshield_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset","total","tp","fp","fn","tn",
                                               "acc","precision","recall","f1","fpr"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Results saved → {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JBShield-D with summary table")
    parser.add_argument("--model", default="mistral", help="Target model name")
    args = parser.parse_args()
    run(args.model)