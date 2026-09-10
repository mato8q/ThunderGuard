"""
CAMS — Concept-Guided Adaptive Memory Shield — Evaluation Script.

Pipeline per prompt:
  hidden state h
    │
    ├─[Concept Probe]─ score = cosine_sim(h - mean_harmless, toxic_vector)
    │                       score ≥ tau_detect?
    │                         YES → ATTACK  (+ filtered memory update if score ≥ tau_memory)
    │                         NO  ↓
    └─[Memory Bank]── IMAG-style gap detection
                            gap > T  → ATTACK
                            gap ≤ T  → BENIGN

Calibration data (Option B — no attack-specific JSON):
  Concept probe : data/harmful.csv + data/harmless.csv  (first CAL_N rows)
  Memory seeds  : data/jailbreak/{attack}/mistral_calibration.json (all attacks combined)
  Benign seeds  : data/harmless.csv rows CAL_N .. CAL_N+SEED_N
  Test          : data/jailbreak/{attack}/mistral_test.json  (never used above)

Usage:
    cd IMAG
    python cams/evaluate_cams.py --tau-detect 0.3 --tau-memory 0.5
    python cams/evaluate_cams.py --no-concept-probe   # ablation: memory bank only
"""

import argparse
import csv
import gc
import json
import os
import random
import sys
import time

import numpy as np

# Make IMAG root importable from cams/ subdirectory
_CAMS_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAG_DIR = os.path.dirname(_CAMS_DIR)
sys.path.insert(0, _IMAG_DIR)

from cams.concept_probe import ConceptProbe
from core.llm_interface import TargetLLM
from modules.immune_detector import ImmuneDetector
from modules.memory_bank import MemoryBank

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_NAME  = "mistralai/Mistral-7B-Instruct-v0.2"
TOP_K       = 5
THRESHOLD_T = 0.1
POOLING     = "last"
CAL_N       = 50    # harmful/harmless rows used for concept probe calibration
SEED_N      = 30    # rows used for initial memory bank seeds

JBSHIELD_DIR = os.path.join(_IMAG_DIR, "..", "baselines", "jbshield")

ATTACK_DATASETS = ["autodan", "zulu", "base64", "drattack", "pair",
                   "gcg", "ijp", "saa", "puzzler"]

# ── Data loaders ───────────────────────────────────────────────────────────────

def _read_csv_prompts(path: str, col: str = "prompt", limit: int = None) -> list[str]:
    with open(path, encoding="utf-8") as f:
        rows = [r[col].strip() for r in csv.DictReader(f) if r.get(col, "").strip()]
    return rows[:limit] if limit else rows


def _load_jbshield_attack_json(attack: str, split: str = "test", model: str = "mistral") -> list[str]:
    path = os.path.join(JBSHIELD_DIR, "data", "jailbreak", attack, f"{model}_{split}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [item["jailbreak"] for item in data if item.get("jailbreak", "").strip()]


def load_calibration_data() -> tuple[list[str], list[str]]:
    """First CAL_N rows of harmful.csv + harmless.csv for concept probe."""
    harmful_path  = os.path.join(JBSHIELD_DIR, "data", "harmful.csv")
    harmless_path = os.path.join(JBSHIELD_DIR, "data", "harmless.csv")
    harmful  = _read_csv_prompts(harmful_path,  limit=CAL_N)
    harmless = _read_csv_prompts(harmless_path, limit=CAL_N)
    print(f"  [Concept probe cal] {len(harmful)} harmful + {len(harmless)} harmless")
    return harmful, harmless


def load_memory_seeds() -> tuple[list[str], list[str]]:
    """
    Attack seeds: JBShield calibration JSON (all attacks combined, capped at SEED_N total).
    Benign seeds: harmless.csv rows CAL_N .. CAL_N+SEED_N.
    """
    harmless_path = os.path.join(JBSHIELD_DIR, "data", "harmless.csv")
    all_harmless  = _read_csv_prompts(harmless_path)
    benign_seeds  = all_harmless[CAL_N: CAL_N + SEED_N]

    atk_seeds = []
    per_attack = max(1, SEED_N // len(ATTACK_DATASETS))
    for attack in ATTACK_DATASETS:
        try:
            prompts = _load_jbshield_attack_json(attack, split="calibration")
            atk_seeds.extend(prompts[:per_attack])
        except FileNotFoundError:
            pass
    atk_seeds = atk_seeds[:SEED_N]
    print(f"  [Memory seeds] {len(atk_seeds)} attack + {len(benign_seeds)} benign")
    return atk_seeds, benign_seeds


def load_test_data() -> tuple[dict[str, list[str]], list[str]]:
    """
    Attack test: mistral_test.json for each attack.
    Benign test: harmless.csv rows CAL_N+SEED_N .. CAL_N+SEED_N+200.
    """
    harmless_path = os.path.join(JBSHIELD_DIR, "data", "harmless.csv")
    all_harmless  = _read_csv_prompts(harmless_path)
    benign_test   = all_harmless[CAL_N + SEED_N: CAL_N + SEED_N + 200]

    attack_test = {}
    for attack in ATTACK_DATASETS:
        try:
            prompts = _load_jbshield_attack_json(attack, split="test")
            attack_test[attack] = prompts
            print(f"  [Test] {attack}: {len(prompts)} prompts")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
    print(f"  [Test] benign: {len(benign_test)} prompts (harmless.csv rows {CAL_N+SEED_N}–{CAL_N+SEED_N+200})")
    return attack_test, benign_test

# ── Encoding ───────────────────────────────────────────────────────────────────

def encode_prompts(prompts: list[str], llm: TargetLLM, critical_layer: int,
                   label: str = "") -> list[np.ndarray]:
    import torch
    out = []
    for i, t in enumerate(prompts, 1):
        out.append(llm.extract_hidden_states(t, target_layer=critical_layer, pooling=POOLING))
        print(f"    {label}{i}/{len(prompts)}", end="\r")
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    if prompts:
        print()
    return out

# ── CAMS Classification ────────────────────────────────────────────────────────

def _compute_metrics(results: list, dataset: str, elapsed: float) -> dict:
    total = len(results)
    tp = sum(1 for t, p in results if t == "ATTACK" and p == "ATTACK")
    tn = sum(1 for t, p in results if t == "BENIGN" and p == "BENIGN")
    fp = sum(1 for t, p in results if t == "BENIGN" and p == "ATTACK")
    fn = sum(1 for t, p in results if t == "ATTACK" and p == "BENIGN")
    acc  = (tp + tn) / total if total else 0
    prec = tp / (tp + fp)   if (tp + fp) else 0
    rec  = tp / (tp + fn)   if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    fpr  = fp / (fp + tn)   if (fp + tn) else 0
    return {
        "dataset": dataset, "total": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4), "fpr": round(fpr, 4),
        "time_s": round(elapsed, 1),
    }


def classify_cams(
    dataset: str,
    attack_h: list[np.ndarray],
    benign_h: list[np.ndarray],
    probe: "ConceptProbe | None",
    memory: MemoryBank,
    detector: ImmuneDetector,
    tau_detect: float,
    tau_memory: float,
) -> dict:
    """
    CAMS classification loop.

    For each prompt:
      1. Concept probe check  → if score ≥ tau_detect: ATTACK (+ filtered memory update)
         (skipped when probe is None — ablation mode)
      2. Memory bank check    → IMAG gap detection
      3. Fallback             → BENIGN
    """
    results = []
    total   = len(attack_h) + len(benign_h)
    t_start = time.time()

    probe_hits = 0
    memory_hits = 0

    def _step(h_x: np.ndarray, true_label: str, idx: int):
        nonlocal probe_hits, memory_hits

        # ── Stage 1: Concept Probe ────────────────────────────────────────────
        ts = None
        if probe is not None:
            ts = probe.score(h_x)
            if ts >= tau_detect:
                if ts >= tau_memory:
                    memory._lt_attack.append(h_x)
                results.append((true_label, "ATTACK"))
                probe_hits += 1
                print(f"  [{idx:>4}/{total}] {true_label} → ATTACK  (probe score={ts:+.3f})", end="\r")
                return

        # ── Stage 2: Memory Bank ──────────────────────────────────────────────
        atk_vecs = memory.get_attack()
        ben_vecs = memory.get_benign()
        if len(atk_vecs) > 0 and len(ben_vecs) > 0:
            label, s_a, s_b = detector.detect(h_x, atk_vecs, ben_vecs)
            gap = s_b - s_a
            if label == "ATTACK":
                memory._lt_attack.append(h_x)
                results.append((true_label, "ATTACK"))
                memory_hits += 1
                print(f"  [{idx:>4}/{total}] {true_label} → ATTACK  (memory gap={gap:+.3f})", end="\r")
                return
            elif label == "BENIGN":
                memory._lt_benign.append(h_x)
                results.append((true_label, "BENIGN"))
                print(f"  [{idx:>4}/{total}] {true_label} → BENIGN  (memory gap={gap:+.3f})", end="\r")
                return
            # CANDIDATE → fall through to BENIGN

        # ── Fallback: BENIGN ──────────────────────────────────────────────────
        results.append((true_label, "BENIGN"))
        ts_str = f"probe={ts:+.3f}" if ts is not None else "probe=off"
        print(f"  [{idx:>4}/{total}] {true_label} → BENIGN  ({ts_str}, no memory hit)", end="\r")

    for i, h_x in enumerate(attack_h, 1):
        _step(h_x, "ATTACK", i)
    for i, h_x in enumerate(benign_h, 1):
        _step(h_x, "BENIGN", len(attack_h) + i)
    print()

    metrics = _compute_metrics(results, dataset, time.time() - t_start)
    metrics["probe_hits"]  = probe_hits
    metrics["memory_hits"] = memory_hits
    print(f"  Acc={metrics['acc']:.3f}  F1={metrics['f1']:.3f}  "
          f"Recall={metrics['recall']*100:.1f}%  "
          f"[probe={probe_hits}  memory={memory_hits}]")
    return metrics

# ── Save ───────────────────────────────────────────────────────────────────────

def save_results(rows: list[dict], model_name: str, tau_detect: float,
                 tau_memory: float, threshold_t: float,
                 no_concept_probe: bool = False) -> str:
    from datetime import datetime
    os.makedirs(os.path.join(_IMAG_DIR, "results"), exist_ok=True)
    model_short = model_name.split("/")[-1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant = "memory_only" if no_concept_probe else f"cams_td{tau_detect}_tm{tau_memory}"
    fname = os.path.join(
        _IMAG_DIR, "results",
        f"{model_short}_{variant}_t{threshold_t}_{ts}.csv"
    )
    fields = ["dataset", "total", "tp", "tn", "fp", "fn",
              "acc", "precision", "recall", "f1", "fpr",
              "time_s", "probe_hits", "memory_hits"]
    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Results saved → {fname}")
    return fname

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global CAL_N, SEED_N, POOLING

    parser = argparse.ArgumentParser(description="CAMS evaluation on JBShield test data")
    parser.add_argument("--model",       default=MODEL_NAME)
    parser.add_argument("--threshold",   type=float, default=THRESHOLD_T,
                        help="Memory bank gap threshold T (default: 0.1)")
    parser.add_argument("--tau-detect",  type=float, default=0.3,
                        help="Concept probe detection threshold (default: 0.3)")
    parser.add_argument("--tau-memory",  type=float, default=0.5,
                        help="Concept score threshold for filtered memory update (default: 0.5)")
    parser.add_argument("--cal-n",       type=int,   default=CAL_N,
                        help=f"Rows of harmful/harmless used for concept probe (default: {CAL_N})")
    parser.add_argument("--seed-n",      type=int,   default=SEED_N,
                        help=f"Attack + benign prompts for memory bank init (default: {SEED_N})")
    parser.add_argument("--pooling",          choices=["last", "mean"], default=None)
    parser.add_argument("--no-concept-probe", action="store_true",
                        help="Ablation: skip concept probe, use memory bank only")
    args = parser.parse_args()

    CAL_N  = args.cal_n
    SEED_N = args.seed_n
    if args.pooling:
        POOLING = args.pooling

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_probe = not args.no_concept_probe

    print("=" * 65)
    print(f"  CAMS — Concept-Guided Adaptive Memory Shield")
    print(f"  Model      : {args.model}")
    if use_probe:
        print(f"  tau_detect : {args.tau_detect}  tau_memory : {args.tau_memory}")
    else:
        print(f"  [ABLATION] Concept probe DISABLED — memory bank only")
    print(f"  threshold T: {args.threshold}  pooling    : {POOLING}")
    if use_probe:
        print(f"  Concept probe cal: {CAL_N} harmful + {CAL_N} harmless (from JBShield data/)")
    print(f"  Memory seeds     : {SEED_N} attack (all attacks) + {SEED_N} benign")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    if use_probe:
        cal_harmful_prompts, cal_harmless_prompts = load_calibration_data()
    atk_seed_prompts, ben_seed_prompts = load_memory_seeds()
    attack_test, benign_test           = load_test_data()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[2] Loading model: {args.model}")
    llm = TargetLLM(args.model, device=device)
    critical_layer = 31
    print(f"  Critical layer: {critical_layer} (Mistral-7B default)")

    # ── Encode calibration (skipped in ablation) ──────────────────────────────
    probe = None
    if use_probe:
        print("\n[3] Encoding concept probe calibration...")
        cal_harmful_h  = encode_prompts(cal_harmful_prompts,  llm, critical_layer, "harmful-cal ")
        cal_harmless_h = encode_prompts(cal_harmless_prompts, llm, critical_layer, "harmless-cal ")
        probe = ConceptProbe(cal_harmful_h, cal_harmless_h)
    else:
        print("\n[3] Concept probe SKIPPED (ablation mode).")

    # ── Encode memory seeds ───────────────────────────────────────────────────
    print("\n[4] Encoding memory bank seeds...")
    atk_seed_h = encode_prompts(atk_seed_prompts, llm, critical_layer, "atk-seed ")
    ben_seed_h = encode_prompts(ben_seed_prompts, llm, critical_layer, "ben-seed ")

    # ── Encode test data (all attacks + benign) ────────────────────────────────
    print(f"\n[5] Encoding test data ({len(attack_test)} attacks + benign)...")
    attack_test_h: dict[str, list[np.ndarray]] = {}
    for attack, prompts in attack_test.items():
        print(f"  Encoding {attack} ({len(prompts)} prompts)...")
        attack_test_h[attack] = encode_prompts(prompts, llm, critical_layer, f"{attack} ")
    print(f"  Encoding benign ({len(benign_test)} prompts)...")
    benign_test_h = encode_prompts(benign_test, llm, critical_layer, "benign ")

    # ── Classify ──────────────────────────────────────────────────────────────
    print(f"\n[6] Classifying (tau_detect={args.tau_detect}  tau_memory={args.tau_memory}  T={args.threshold})...")
    detector = ImmuneDetector(threshold_T=args.threshold, top_k=TOP_K)
    all_results = []
    t_total = time.time()
    rng = random.Random(42)

    for attack in ATTACK_DATASETS:
        if attack not in attack_test_h:
            continue
        print(f"\n{'='*65}")
        print(f"  Attack: {attack}  ({len(attack_test_h[attack])} prompts)")

        # Fresh memory bank per attack (seeded from combined attack seeds + benign seeds)
        memory = MemoryBank.__new__(MemoryBank)
        memory.save_path   = None
        memory._lt_attack  = [v.copy() for v in atk_seed_h]
        memory._lt_benign  = [v.copy() for v in ben_seed_h]
        memory._st_buffer  = []

        # Shuffle benign test to interleave with attacks for realistic online scenario
        benign_subset = list(benign_test_h)
        rng.shuffle(benign_subset)
        benign_n = len(attack_test_h[attack])
        benign_eval = benign_subset[:benign_n]

        row = classify_cams(
            attack,
            attack_test_h[attack],
            benign_eval,
            probe, memory, detector,
            args.tau_detect, args.tau_memory,
        )
        all_results.append(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    variant_label = "Memory Bank Only (ablation)" if not use_probe else "CAMS"
    print("\n" + "=" * 65)
    print(f"  SUMMARY  ({variant_label}  |  {args.model.split('/')[-1]})")
    if use_probe:
        print(f"  tau_detect={args.tau_detect}  tau_memory={args.tau_memory}  T={args.threshold}")
    else:
        print(f"  T={args.threshold}  (concept probe disabled)")
    print(f"  {'Dataset':<14} {'Acc':>6} {'Prec':>6} {'Recall':>8} {'F1':>6} {'FPR':>6}  probe/memory hits")
    print("  " + "-" * 62)
    for r in all_results:
        print(f"  {r['dataset']:<14} {r['acc']:>6.3f} {r['precision']:>6.3f} "
              f"{r['recall']*100:>7.1f}% {r['f1']:>6.3f} {r['fpr']*100:>5.1f}%  "
              f"[{r.get('probe_hits',0)}/{r.get('memory_hits',0)}]")

    atk_rows = all_results
    if atk_rows:
        avg_acc = sum(r["acc"] for r in atk_rows) / len(atk_rows)
        avg_f1  = sum(r["f1"]  for r in atk_rows) / len(atk_rows)
        print("  " + "-" * 62)
        print(f"  {'Average':<14} {avg_acc:>6.3f} {'':>6} {'':>8} {avg_f1:>6.3f}")

    print(f"\n  Total wall time: {time.time()-t_total:.1f}s")
    print("=" * 65)

    save_results(all_results, args.model, args.tau_detect, args.tau_memory, args.threshold,
                 no_concept_probe=args.no_concept_probe)


if __name__ == "__main__":
    main()
