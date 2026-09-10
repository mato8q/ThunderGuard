"""
Validate JBShield-D on its OWN original data (data/harmful.csv, data/harmless.csv,
data/jailbreak/<attack>/mistral_test.json).

If F1 > 0.90 here -> algorithm is correct, our test data is just harder.
If F1 still low here -> there is a remaining implementation bug.

Usage:
    cd baselines/jbshield
    python run_original.py --model mistral
"""

import argparse
import csv
import json
import random
import time

import torch
from tqdm import tqdm

from config import model_paths
from utils import load_model, get_sentence_embeddings, interpret_difference_matrix, cosine_similarity
from detection import find_critical_layer, find_optimal_threshold, detection_judge

CALIBRATION_N = 50
JAILBREAKS = ["autodan", "zulu", "base64", "pair", "drattack", "gcg", "ijp", "saa", "puzzler"]


def _read_csv_col(path, col):
    with open(path, encoding="utf-8") as f:
        return [r[col].strip() for r in csv.DictReader(f) if r.get(col, "").strip()]


def _read_json_jailbreaks(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [item["jailbreak"] for item in data]


def run(model_name: str):
    t0 = time.time()
    print("=" * 65)
    print(f"  JBShield-D  —  ORIGINAL DATA Validation")
    print(f"  Model   : {model_name}")
    print("=" * 65)

    # ── Load prompts ──────────────────────────────────────────────────────────
    print("\n[1] Loading prompts...")
    all_harmful  = _read_csv_col("data/harmful.csv",  "prompt")
    all_harmless = _read_csv_col("data/harmless.csv", "prompt")

    # Calibration: first CALIBRATION_N rows
    cal_harmful  = all_harmful[:CALIBRATION_N]
    cal_harmless = all_harmless[:CALIBRATION_N]

    # Test harmless: next rows (cap at 200 to match paper §4.2)
    test_harmless = all_harmless[CALIBRATION_N:CALIBRATION_N + 200]

    print(f"  Cal harmful={len(cal_harmful)}  Cal harmless={len(cal_harmless)}")
    print(f"  Test harmless={len(test_harmless)}")

    # Load jailbreak calibration and test from JSON
    jb_cal, jb_test = {}, {}
    for jb in JAILBREAKS:
        cal_path  = f"data/jailbreak/{jb}/{model_name}_calibration.json"
        test_path = f"data/jailbreak/{jb}/{model_name}_test.json"
        jb_cal[jb]  = _read_json_jailbreaks(cal_path)
        jb_test[jb] = _read_json_jailbreaks(test_path)
        print(f"  {jb}: cal={len(jb_cal[jb])}  test={len(jb_test[jb])}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[2] Loading model: {model_name}")
    model, tokenizer = load_model(model_name, model_paths)

    # ── Embeddings (calibration) ──────────────────────────────────────────────
    print("\n[3] Encoding calibration embeddings...")
    cal_harmless_emb = get_sentence_embeddings(cal_harmless, model, model_name, tokenizer)
    cal_harmful_emb  = get_sentence_embeddings(cal_harmful,  model, model_name, tokenizer)

    mean_harmful_emb  = [torch.mean(torch.stack(e), dim=0) for e in cal_harmful_emb]
    mean_harmless_emb = [torch.mean(torch.stack(e), dim=0) for e in cal_harmless_emb]

    cal_jb_emb = {}
    for jb in JAILBREAKS:
        print(f"    cal: {jb}", end="\r")
        cal_jb_emb[jb] = get_sentence_embeddings(jb_cal[jb], model, model_name, tokenizer)
    print()

    # ── Embeddings (test) ─────────────────────────────────────────────────────
    print("\n[4] Encoding test embeddings...")
    test_harmless_emb = get_sentence_embeddings(test_harmless, model, model_name, tokenizer)
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
    print(f"  safety_layer={safety_layer}  delta={safety_delta:.4f}")

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
        print(f"\n  [{jb}] layer={layer}  thr_safety={thr_safety:.4f}  thr_jb={thr_jb:.4f}")

        res_jb_safety = detection_judge(model, tokenizer, test_jb_emb[jb][safety_layer],
                                        mean_harmless_emb[safety_layer], safety_vec, thr_safety)
        res_jb_jb     = detection_judge(model, tokenizer, test_jb_emb[jb][layer],
                                        mean_harmful_emb[layer], jb_vecs[jb], thr_jb)
        res_bn_safety = detection_judge(model, tokenizer, test_harmless_emb[safety_layer][:n],
                                        mean_harmless_emb[safety_layer], safety_vec, thr_safety)
        res_bn_jb     = detection_judge(model, tokenizer, test_harmless_emb[layer][:n],
                                        mean_harmful_emb[layer], jb_vecs[jb], thr_jb)

        atk_s = sum(res_jb_safety) / len(res_jb_safety) * 100
        atk_j = sum(res_jb_jb)     / len(res_jb_jb)     * 100
        bn_s  = sum(res_bn_safety) / len(res_bn_safety)  * 100
        bn_j  = sum(res_bn_jb)     / len(res_bn_jb)      * 100
        print(f"    atk  pass safety={atk_s:.1f}%  jb={atk_j:.1f}%")
        print(f"    benign pass safety={bn_s:.1f}%  jb={bn_j:.1f}%")

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

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  SUMMARY  (JBShield-D ORIGINAL  |  model={model_name})")
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

    import os
    from datetime import datetime
    os.makedirs("results", exist_ok=True)
    fname = f"results/jbshield_original_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset","total","tp","fp","fn","tn",
                                               "acc","precision","recall","f1","fpr"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Results saved -> {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral")
    args = parser.parse_args()
    run(args.model)