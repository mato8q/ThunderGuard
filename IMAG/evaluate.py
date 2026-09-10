"""
IMAG — Dataset Evaluation (faithful to Algorithm 1)
Attack  : HEX-Phi, AdvBench, AutoDAN, Zulu, Base64, DrAttack
Benign  : XSTest, JailbreakBench, WildJailbreak

Usage:
    python evaluate.py --dataset all
"""

import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from core.llm_interface import TargetLLM
from data.seed_prompts import SEED_ATTACKS, SEED_BENIGNS
from modules.active_immunity import ActiveImmunity
from modules.immune_detector import ImmuneDetector
from modules.memory_bank import MemoryBank

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME       = "mistralai/Mistral-7B-Instruct-v0.2"
TOP_K            = 5
THRESHOLD_T      = 0.1   # paper uses small T; 0.5 causes CANDIDATE cascade corrupting memory
POOLING          = "last"   # overridden by --pooling flag at runtime
MEMORY_PATH      = "data/eval_memory.npz"
LAYER_CACHE      = "data/critical_layer.txt"   # shared with main.py
N_DATASET_SEEDS  = 30   # samples from each dataset reserved to seed the memory bank

# Root of the datasets folder (resolved relative to this file)
_HERE              = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR       = os.path.join(_HERE, "..", "datasets")
TRANSFORMED_DIR    = os.path.join(_HERE, "..", "data", "transformed")

RANDOM_SEED   = 42

# ── Dataset paths ─────────────────────────────────────────────────────────────

_ORIG_DIR = os.path.join(_HERE, "..", "adversarial-prompt", "data", "original")

# Attack datasets
HEX_PHI_PATH  = os.path.join(_ORIG_DIR, "hex_phi.csv")
ADVBENCH_PATH = os.path.join(_ORIG_DIR, "harmful_behaviors.csv")

# Benign datasets
XSTEST_PATH         = os.path.join(_ORIG_DIR, "xstest_benign.csv")
JAILBREAKBENCH_PATH = os.path.join(_ORIG_DIR, "jailbreakbench_benign.csv")
WILDJAILBREAK_PATH  = os.path.join(_ORIG_DIR, "wildjailbreak_benign.csv")

# ── Benign loader ─────────────────────────────────────────────────────────────

def _load_benign_csv(path: str, name: str) -> list[str]:
    import csv
    if not os.path.exists(path):
        print(f"  [SKIP] {name} not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        rows = [r["prompt"].strip() for r in csv.DictReader(f) if r.get("prompt", "").strip()]
    print(f"  Loaded {len(rows)} benign from {name}")
    return rows


def load_benign_texts(dataset: str, rng: random.Random) -> list[str]:
    texts = []

    if dataset == "xstest":
        rows = _load_benign_csv(XSTEST_PATH, "XSTest")
        # Paper Table 1: XSTest uses 200 benign prompts for FPR evaluation
        # (CSV contains 250 rows; cap to match paper §4.2)
        if len(rows) > 200:
            rng.shuffle(rows)
            rows = rows[:200]
            print(f"  Sampled 200 from XSTest (paper §4.2)")
        # texts.extend(_load_benign_csv(XSTEST_PATH, "XSTest"))
        texts.extend(rows)

    elif dataset == "jailbreakbench":
        texts.extend(_load_benign_csv(JAILBREAKBENCH_PATH, "JailbreakBench"))

    elif dataset == "wildjailbreak":
        rows = _load_benign_csv(WILDJAILBREAK_PATH, "WildJailbreak")
        # Paper Table 1: WildJailbreak uses 200 benign prompts for FPR evaluation
        if len(rows) > 200:
            rng.shuffle(rows)
            rows = rows[:200]
            print(f"  Sampled 200 from WildJailbreak (paper §4.2)")
        texts.extend(rows)

    if not texts:
        raise FileNotFoundError(f"No benign samples loaded for dataset='{dataset}'.")

    rng.shuffle(texts)
    return texts

# ── Attack loader ─────────────────────────────────────────────────────────────

def _load_csv(path: str, col: str, label: str) -> list[str]:
    import csv
    if not os.path.exists(path):
        print(f"  [SKIP] Not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        rows = [r[col].strip() for r in csv.DictReader(f) if r.get(col, "").strip()]
    print(f"  Loaded {len(rows)} from {label}")
    return rows


def _load_transformed_csv(fname: str, prompt_col: str) -> list[str]:
    return _load_csv(os.path.join(TRANSFORMED_DIR, fname), prompt_col, fname)


def _load_autodan_texts(fname: str = "autodan_prompts.csv") -> list[str]:
    """
    AutoDAN CSVs store the template with literal '[REPLACE]' where the harmful
    goal should appear.  Substitute it before encoding hidden states so the
    prompt actually contains the harmful intent (otherwise it's just a generic
    roleplay template that looks identical to benign creative-writing prompts).
    """
    import csv as _csv
    path = os.path.join(TRANSFORMED_DIR, fname)
    if not os.path.exists(path):
        print(f"  [SKIP] Not found: {path}")
        return []
    texts = []
    n_replaced = 0
    with open(path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            prompt = row.get("autodan_prompt", "").strip() or row.get("prompt", "").strip()
            goal   = row.get("goal", "").strip()
            if not prompt:
                continue
            if "[REPLACE]" in prompt and goal:
                prompt = prompt.replace("[REPLACE]", goal)
                n_replaced += 1
            texts.append(prompt)
    print(f"  Loaded {len(texts)} from {fname}  ({n_replaced} goal substitutions)")
    return texts


def _load_drattack_texts(model_name: str = "") -> list[str]:
    """Resolve DrAttack CSV per-model (drattack_results_<model-short>.csv → fallback
    drattack_results_mistral.csv) and load the `best_adversarial_prompt` column."""
    model_short = model_name.split("/")[-1].lower().replace("_", "-") if model_name else ""
    candidate = f"drattack_results_{model_short}.csv" if model_short else ""
    candidate_path = os.path.join(TRANSFORMED_DIR, candidate) if candidate else ""
    if candidate_path and os.path.exists(candidate_path):
        fname = candidate
        print(f"  Using model-specific DrAttack file: {fname}")
    else:
        fname = "drattack_results_mistral-7b.csv"
        if model_short and "mistral" not in model_short:
            print(f"  [WARN] No DrAttack file for '{model_short}' — falling back to mistral version")
    return _load_transformed_csv(fname, "best_adversarial_prompt")



def _load_pair_texts(model_name: str = "") -> list[str]:
    """Load PAIR results. Prefer model-specific 850-prompt file, fallback to mistral-7b_850."""
    model_short = model_name.split("/")[-1].lower().replace("_", "-") if model_name else ""
    for fname in [
        f"pair_results_{model_short}_850.csv" if model_short else "",
        "pair_results_mistral-7b_850.csv",
        f"pair_results_{model_short}.csv" if model_short else "",
        "pair_results_mistral-7b.csv",
    ]:
        if fname and os.path.exists(os.path.join(TRANSFORMED_DIR, fname)):
            print(f"  Using PAIR file: {fname}")
            return _load_transformed_csv(fname, "best_adversarial_prompt")
    print("  [WARN] No PAIR file found")
    return []


def load_attack_texts(dataset: str, rng: random.Random, model_name: str = "") -> list[str]:
    """
    dataset: "hex_phi" | "advbench" | "zulu" | "base64" | "drattack" | "autodan"
    model_name: used to find model-specific DrAttack files (e.g. drattack_results_llama2-7b-chat.csv)
    """
    texts = []

    if dataset == "hex_phi":
        texts.extend(_load_csv(HEX_PHI_PATH, "goal", "HEX-Phi"))

    elif dataset == "advbench":
        texts.extend(_load_csv(ADVBENCH_PATH, "goal", "AdvBench"))

    elif dataset == "autodan":
        texts.extend(_load_autodan_texts())

    elif dataset == "zulu":
        texts.extend(_load_transformed_csv("zulu_prompts.csv", "prompt"))

    elif dataset == "base64":
        texts.extend(_load_transformed_csv("base64_prompts.csv", "prompt"))

    elif dataset == "drattack":
        texts.extend(_load_drattack_texts(model_name))

    elif dataset == "pair":
        texts.extend(_load_pair_texts(model_name))

    if not texts:
        raise FileNotFoundError(f"No attack samples found for dataset='{dataset}'.")

    rng.shuffle(texts)
    return texts

# ── Layer helper ──────────────────────────────────────────────────────────────

def _layer_cache_path(model_name: str) -> str:
    """Cache path is model-specific to avoid cross-model contamination."""
    safe = model_name.replace("/", "_").replace("\\", "_")
    return os.path.join("data", f"critical_layer_{safe}.txt")


def get_critical_layer(llm: TargetLLM, model_name: str) -> int:
    cache_path = _layer_cache_path(model_name)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            layer = int(f.read().strip())
        print(f"  Critical layer (cached for {model_name.split('/')[-1]}): {layer}")
        return layer
    layer = llm.find_critical_layer(
        SEED_ATTACKS[:min(len(SEED_ATTACKS), len(SEED_BENIGNS))],
        SEED_BENIGNS[:min(len(SEED_ATTACKS), len(SEED_BENIGNS))],
        pooling=POOLING,
    )
    os.makedirs("data", exist_ok=True)
    with open(cache_path, "w") as f:
        f.write(str(layer))
    return layer

# ── Seed encoding (run once, shared across all T values) ──────────────────────

def build_seed_vectors(
    llm: TargetLLM,
    critical_layer: int,
    n_seeds: int = 30,
    seed_strategy: str = "advbench",
    model_name: str = "",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Encode attack seeds for the memory bank.

    seed_strategy:
      "advbench"    — 30 AdvBench direct harmful behaviors (paper §5.2; best when
                      eval attacks were generated from the SAME model)
      "handcrafted" — 30 diverse seeds from seed_prompts.py (DAN/roleplay/direct/
                      evasion mix; better when eval attacks are obfuscated/DAN-style
                      generated against a DIFFERENT model)
      "mixed"       — AdvBench + handcrafted combined (~60 seeds total)
      "autodan" | "drattack" | "zulu" | "base64"
                    — cross-attack: seed the memory bank from one attack dataset
                      to test generalization onto a different attack dataset
                      (see results/seed_overlap_report.md). Reuses the same CSV
                      loaders as --dataset, so text preprocessing (e.g. AutoDAN
                      [REPLACE] substitution) stays identical.

    NOTE: AdvBench seeds give 0% detection when the eval attack CSVs were generated
    for a different model (e.g. Mistral-generated attacks evaluated on LLaMA2).
    In that case, use --seeds handcrafted or run with the correct target model.
    """
    import csv as _csv

    def _load_goals(path: str, col: str) -> list[str]:
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [r[col].strip() for r in _csv.DictReader(f) if r.get(col, "").strip()]

    texts_advbench = _load_goals(ADVBENCH_PATH, "goal")[:n_seeds]
    texts_hexphi   = _load_goals(HEX_PHI_PATH,  "goal")[:n_seeds]

    # Paper §5.2: seeds drawn from AdvBench + HexPHI combined
    paper_seeds = list(dict.fromkeys(texts_advbench + texts_hexphi))  # dedup, preserve order

    cross_attack_loaders = {
        "autodan":  lambda: _load_autodan_texts(),
        "zulu":     lambda: _load_transformed_csv("zulu_prompts.csv", "prompt"),
        "base64":   lambda: _load_transformed_csv("base64_prompts.csv", "prompt"),
        "drattack": lambda: _load_drattack_texts(model_name),
        "pair":     lambda: _load_pair_texts(model_name),
    }

    if seed_strategy in cross_attack_loaders:
        loaded = cross_attack_loaders[seed_strategy]()
        attack_texts = loaded[:n_seeds]
        src_label = f"{seed_strategy} cross-attack"
        if not attack_texts:
            raise FileNotFoundError(
                f"No prompts loaded for --seeds {seed_strategy} (CSV missing or empty)."
            )
    elif seed_strategy == "handcrafted" or not paper_seeds:
        attack_texts = SEED_ATTACKS[:n_seeds]
        src_label = "handcrafted"
        if not paper_seeds and seed_strategy != "handcrafted":
            print(f"  [WARN] Neither AdvBench nor HexPHI found — falling back to handcrafted seeds.")
    elif seed_strategy == "mixed":
        attack_texts = list(dict.fromkeys(paper_seeds + SEED_ATTACKS))
        src_label = (f"mixed({len(texts_advbench)} AdvBench + {len(texts_hexphi)} HexPHI "
                     f"+ {len(SEED_ATTACKS)} handcrafted = {len(attack_texts)} total)")
    else:
        attack_texts = paper_seeds
        parts = []
        if texts_advbench: parts.append(f"{len(texts_advbench)} AdvBench")
        if texts_hexphi:   parts.append(f"{len(texts_hexphi)} HexPHI")
        if not parts:      parts.append("handcrafted")
        src_label = " + ".join(parts)

    print(f"  Encoding {len(attack_texts)} {src_label} attack seeds + "
          f"{len(SEED_BENIGNS)} benign global seeds (layer {critical_layer})...")

    attack_vecs = [
        llm.extract_hidden_states(p, target_layer=critical_layer, pooling=POOLING)
        for p in attack_texts
    ]
    benign_vecs = [
        llm.extract_hidden_states(p, target_layer=critical_layer, pooling=POOLING)
        for p in SEED_BENIGNS
    ]
    avail = []
    if texts_advbench: avail.append("AdvBench")
    if texts_hexphi:   avail.append("HexPHI")
    if not avail:      avail.append("handcrafted")
    print(f"  Seeds ready: {len(attack_vecs)} attack ({', '.join(avail)}), {len(benign_vecs)} benign")
    return attack_vecs, benign_vecs


# ── Classification (fast CPU step — reused for every T in a sweep) ────────────

def classify_cached(
    dataset: str,
    attack_texts: list[str],
    attack_h: list[np.ndarray],
    benign_texts: list[str],
    benign_h: list[np.ndarray],
    seed_attack_vecs: list[np.ndarray],
    seed_benign_vecs: list[np.ndarray],
    threshold: float,
    no_active_immunity: bool,
    active_immunity=None,
) -> dict:
    """
    Classify pre-encoded hidden states at the given threshold.
    Memory is seeded fresh from seed_*_vecs for each call (so T sweep is independent).
    Stage 2 (active_immunity) is invoked for CANDIDATE prompts when not suppressed.
    """
    # Fresh in-memory bank seeded from pre-encoded vectors (no disk I/O)
    memory = MemoryBank.__new__(MemoryBank)
    memory.save_path = None
    memory._lt_attack = [v.copy() for v in seed_attack_vecs]
    memory._lt_benign = [v.copy() for v in seed_benign_vecs]
    memory._st_buffer = []

    detector = ImmuneDetector(threshold_T=threshold, top_k=TOP_K)
    results = []
    gap_records = []   # (true_label, gap) for each sample — used for plots & multi-run
    t_start = time.time()
    total = len(attack_h) + len(benign_h)

    def _step(h_x, text, true_label, idx):
        label, s_a, s_b = detector.detect(h_x, memory.get_attack(), memory.get_benign())
        gap = s_b - s_a
        is_candidate = (label == "CANDIDATE")

        if is_candidate:
            if no_active_immunity:
                label = "BENIGN"
                # Paper Eq.8: only I_known (high-confidence immune detections) enter
                # short-term memory.  CANDIDATE prompts skipped by active immunity are
                # NOT added to memory — prevents attack samples from corrupting the
                # benign reference when Stage 2 is disabled.
            else:
                label, *_ = active_immunity.evaluate(text)
                # Stage-2-verified result IS confident → update memory (Eq. 9–10)
                if label == "ATTACK":
                    memory._lt_attack.append(h_x)
                else:
                    memory._lt_benign.append(h_x)
        else:
            # Confident immune detection (ATTACK or BENIGN) → update memory (Eq. 8)
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
    metrics["_gap_records"] = gap_records   # per-sample (true_label, gap) — stripped before CSV save
    return metrics

# ── Main ──────────────────────────────────────────────────────────────────────

ATTACK_DATASETS = ["autodan", "zulu", "base64", "drattack", "pair"]
BENIGN_DATASETS = ["xstest", "jailbreakbench", "wildjailbreak"]
ALL_DATASETS    = ATTACK_DATASETS + BENIGN_DATASETS


def encode_dataset(
    dataset: str,
    llm,
    critical_layer: int,
    rng: random.Random,
    model_name: str = "",
) -> dict:
    """
    Load texts and pre-encode all prompts to hidden states (slow GPU step).

    The first N_DATASET_SEEDS samples are reserved as dataset-specific seeds
    to calibrate the memory bank for this attack/benign style.  This is
    critical because obfuscated attacks (Base64, Zulu, AutoDAN, DrAttack)
    have completely different hidden state patterns than the handcrafted seed
    prompts and need their own reference direction.

    Returns a dict with keys:
      ds_seed_atk  : list[np.ndarray]  — N_DATASET_SEEDS attack vectors from THIS dataset
      ds_seed_ben  : list[np.ndarray]  — N_DATASET_SEEDS benign vectors from THIS dataset
      atk_texts    : list[str]         — remaining attack texts for evaluation
      atk_h        : list[np.ndarray]
      ben_texts    : list[str]         — remaining benign texts for evaluation
      ben_h        : list[np.ndarray]
    """
    is_benign_only = dataset in BENIGN_DATASETS
    n_seeds = N_DATASET_SEEDS

    def encode_list(texts, label=""):
        out = []
        for i, t in enumerate(texts, 1):
            out.append(llm.extract_hidden_states(t, target_layer=critical_layer, pooling=POOLING))
            print(f"    {label}{i}/{len(texts)}", end="\r")
        if texts:
            print()
        return out

    if is_benign_only:
        all_texts = load_benign_texts(dataset, rng)
        seed_texts = all_texts[:n_seeds]
        eval_texts = all_texts[n_seeds:]
        print(f"\n  {dataset}: {len(all_texts)} benign "
              f"({n_seeds} seeds + {len(eval_texts)} eval) — encoding...")
        seed_h = encode_list(seed_texts, "seed ")
        eval_h = encode_list(eval_texts, "eval ")
        return {
            "ds_seed_atk": [],   "ds_seed_ben": seed_h,
            "atk_texts": [],     "atk_h": [],
            "ben_texts": eval_texts, "ben_h": eval_h,
        }
    else:
        all_texts = load_attack_texts(dataset, rng, model_name=model_name)
        # Attack memory is seeded from AdvBench (build_seed_vectors), not from
        # this dataset — all samples here go to evaluation (paper §5.2 setup).
        print(f"\n  {dataset}: {len(all_texts)} attack (all eval, AdvBench seeds) — encoding...")
        eval_h = encode_list(all_texts, "eval ")
        return {
            "ds_seed_atk": [],       "ds_seed_ben": [],
            "atk_texts": all_texts,  "atk_h": eval_h,
            "ben_texts": [],         "ben_h": [],
        }


def _print_separation_diagnostic(
    ds: str,
    seed_atk: list,
    seed_ben: list,
    eval_h: list,
    n_samples: int = 5,
):
    """
    Print key metrics to diagnose why gap ≈ 0.

    The main question: do attack and benign SVD directions collapse to the
    same unit vector at this layer?  If cos(h_a, h_b) ≈ 1.0, the model's
    hidden states don't separate attacks from benign prompts at this layer —
    i.e., representation collapse.  Try --pooling mean or a different layer.
    """
    print(f"\n  [DIAG] ===== {ds} separation analysis =====")

    atk = np.array(seed_atk)   # [n_seeds, d]
    ben = np.array(seed_ben)   # [n_seeds, d]

    # ── 1. Mean directions ────────────────────────────────────────────────────
    mu_a = atk.mean(0)
    mu_b = ben.mean(0)
    norm_a, norm_b = np.linalg.norm(mu_a), np.linalg.norm(mu_b)
    mu_a_u = mu_a / (norm_a + 1e-8)
    mu_b_u = mu_b / (norm_b + 1e-8)
    cos_mean = float(np.dot(mu_a_u, mu_b_u))
    print(f"  [DIAG] cos(mean_attack, mean_benign)  = {cos_mean:+.6f}")
    print(f"  [DIAG] ‖mean_attack‖ = {norm_a:.4f},  ‖mean_benign‖ = {norm_b:.4f}")

    # ── 2. SVD rank-1 directions ──────────────────────────────────────────────
    k = min(TOP_K, len(atk), len(ben))
    _, _, Vh_a = np.linalg.svd(atk[:k], full_matrices=False)
    _, _, Vh_b = np.linalg.svd(ben[:k], full_matrices=False)
    h_a = Vh_a[0] / (np.linalg.norm(Vh_a[0]) + 1e-8)
    h_b = Vh_b[0] / (np.linalg.norm(Vh_b[0]) + 1e-8)
    cos_svd = float(np.dot(h_a, h_b))
    print(f"  [DIAG] cos(SVD_h_a,   SVD_h_b)       = {cos_svd:+.6f}")
    print(f"  [DIAG]   → 1.0 = identical refs (collapse)  0.0 = orthogonal (good)")

    # ── 3. Variance within each cluster ───────────────────────────────────────
    var_a = float(np.mean([np.linalg.norm(v - mu_a)**2 for v in atk]))
    var_b = float(np.mean([np.linalg.norm(v - mu_b)**2 for v in ben]))
    print(f"  [DIAG] within-cluster variance: attack={var_a:.4f}  benign={var_b:.4f}")

    # ── 4. Per-sample gaps ────────────────────────────────────────────────────
    n = min(n_samples, len(eval_h))
    print(f"  [DIAG] First {n} eval samples (using SVD refs from seeds):")
    for i, h_x in enumerate(eval_h[:n]):
        h_u = h_x / (np.linalg.norm(h_x) + 1e-8)
        sa = float(np.linalg.norm(h_a - h_u))
        sb = float(np.linalg.norm(h_b - h_u))
        cos_xa = float(np.dot(h_a, h_u))
        cos_xb = float(np.dot(h_b, h_u))
        print(f"  [DIAG]   [{i+1}] s_a={sa:.4f}  s_b={sb:.4f}  gap={sb-sa:+.4f} "
              f" cos(hA,x)={cos_xa:+.4f}  cos(hB,x)={cos_xb:+.4f}")

    print(f"  [DIAG] =========================================")


def _compute_metrics(results, dataset, total_time):
    total = len(results)
    tp = sum(1 for t, p in results if t == "ATTACK" and p == "ATTACK")
    tn = sum(1 for t, p in results if t == "BENIGN" and p == "BENIGN")
    fp = sum(1 for t, p in results if t == "BENIGN" and p == "ATTACK")
    fn = sum(1 for t, p in results if t == "ATTACK" and p == "BENIGN")
    acc  = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
    if dataset in BENIGN_DATASETS:
        safe = tn + fp
        print(f"  Pass-through: {tn/safe*100:.1f}%  FPR: {fpr*100:.1f}%" if safe else "")
    else:
        print(f"  Acc={acc:.3f}  F1={f1:.3f}  Detection={rec*100:.1f}%  Miss={fn/(tp+fn)*100:.1f}%" if (tp+fn) else "")
    return {
        "dataset": dataset, "total": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4), "fpr": round(fpr, 4),
        "time_s": round(total_time, 1),
        "s_per_sample": round(total_time / total, 2) if total > 0 else 0,
    }


def save_results(rows: list[dict], model_name: str, datasets: list[str], threshold: float, no_ai: bool,
                 mem_cost: dict | None = None, seed_size: int | None = None):
    import csv
    from datetime import datetime
    os.makedirs("results", exist_ok=True)
    model_short = model_name.split("/")[-1]
    tag = "all" if set(datasets) == set(ALL_DATASETS) else "_".join(datasets)
    ai_tag = "noAI" if no_ai else "fullAI"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"results/{model_short}_{tag}_t{threshold}_{ai_tag}_{ts}.csv"

    for row in rows:
        if seed_size is not None:
            row.setdefault("seed_size", seed_size)
        if mem_cost is not None:
            row.setdefault("memory_atk_mb", mem_cost["atk_mb"])
            row.setdefault("memory_total_mb", mem_cost["total_mb"])

    fieldnames = ["dataset", "total", "tp", "tn", "fp", "fn",
                  "acc", "precision", "recall", "f1", "fpr", "time_s", "s_per_sample",
                  "seed_size", "memory_atk_mb", "memory_total_mb"]

    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Save per-sample gaps for plotting (gap_distribution, ROC)
    gaps_fname = fname.replace(".csv", "_gaps.csv")
    with open(gaps_fname, "w", newline="", encoding="utf-8") as f:
        gw = csv.writer(f)
        gw.writerow(["dataset", "true_label", "gap"])
        for r in rows:
            for true_label, gap in r.get("_gap_records", []):
                gw.writerow([r["dataset"], true_label, round(gap, 6)])

    print(f"\n  Results saved → {fname}")
    print(f"  Gap data  saved → {gaps_fname}")
    return fname


def _compute_memory_cost(n_atk: int, n_ben: int, hidden_dim: int) -> dict:
    """Memory footprint of seed vectors stored in the memory bank (fp16, 2 bytes/element)."""
    fp16_bytes = 2
    atk_mb = n_atk * hidden_dim * fp16_bytes / (1024 ** 2)
    ben_mb = n_ben * hidden_dim * fp16_bytes / (1024 ** 2)
    return {
        "n_atk": n_atk, "n_ben": n_ben, "hidden_dim": hidden_dim,
        "atk_mb": round(atk_mb, 4), "ben_mb": round(ben_mb, 4),
        "total_mb": round(atk_mb + ben_mb, 4),
    }


def main():
    global N_DATASET_SEEDS
    parser = argparse.ArgumentParser(description="IMAG dataset evaluation")
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"HuggingFace model name or path (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        metavar="DATASET",
        help=(
            "Dataset(s) to evaluate. Use 'all' for everything, or pick from: "
            + ", ".join(ALL_DATASETS)
            + "  (e.g. --dataset zulu base64 drattack)"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD_T,
        help=f"Euclidean distance gap threshold (default: {THRESHOLD_T})",
    )
    parser.add_argument(
        "--threshold-sweep",
        nargs="+",
        type=float,
        default=None,
        metavar="T",
        help="Run evaluation at multiple T values (e.g. --threshold-sweep 0.1 0.3 0.5 0.7). "
             "Encodes prompts once, reruns classification for each T. Saves one CSV per T.",
    )
    parser.add_argument(
        "--critical-layer",
        type=int,
        default=31,
        help="Critical layer index. Paper reports 31 for Mistral-7B (default). "
             "Use --find-critical-layer to auto-search via Eq. 2.",
    )
    parser.add_argument(
        "--find-critical-layer",
        action="store_true",
        help="Auto-search the best critical layer via Eq. 2 (overrides --critical-layer).",
    )
    parser.add_argument(
        "--no-active-immunity",
        action="store_true",
        help="Skip Stage 2; treat CANDIDATE as BENIGN",
    )
    parser.add_argument(
        "--pooling",
        choices=["last", "mean"],
        default=None,
        help="Hidden-state pooling: 'last' (final token, paper default) or 'mean' (avg all tokens). "
             "Try 'mean' if gap≈0 with chat/instruct models.",
    )
    parser.add_argument(
        "--seeds",
        choices=["advbench", "handcrafted", "mixed", "autodan", "drattack", "zulu", "base64", "pair"],
        default="advbench",
        help=(
            "Seed strategy for the attack memory bank. "
            "'advbench' (default): 30 AdvBench direct behaviors — best when eval attacks were "
            "generated against THIS model from AdvBench. "
            "'handcrafted': 30 diverse DAN/roleplay/evasion seeds from seed_prompts.py — "
            "better when eval attacks are obfuscated or generated against a DIFFERENT model. "
            "'mixed': combine both (~60 seeds). "
            "'autodan' | 'drattack' | 'zulu' | 'base64': cross-attack seeds — use the first "
            "30 prompts from one attack dataset to detect a different attack dataset "
            "(generalization test; see results/seed_overlap_report.md)."
        ),
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print separation diagnostics (cos similarity between attack/benign SVD directions). "
             "Useful for debugging gap≈0 issues.",
    )
    parser.add_argument(
        "--seed-size",
        type=int,
        default=N_DATASET_SEEDS,
        metavar="M",
        help=(
            f"Number of seed vectors per class stored in the memory bank (default: {N_DATASET_SEEDS}). "
            "More seeds → better recall but higher RAM. "
            "fp16 cost per class ≈ M × hidden_dim × 2 bytes "
            "(e.g. M=30, d=4096 → 0.23 MB attack + benign each)."
        ),
    )
    args = parser.parse_args()

    N_DATASET_SEEDS = args.seed_size

    # Resolve dataset list
    if "all" in args.dataset:
        datasets = ALL_DATASETS
    else:
        invalid = [d for d in args.dataset if d not in ALL_DATASETS]
        if invalid:
            parser.error(f"Unknown dataset(s): {invalid}. Choose from: {ALL_DATASETS}")
        datasets = list(args.dataset)

    # Auto-add benign datasets for FPR evaluation when only attack datasets are specified
    atk_in = [d for d in datasets if d in ATTACK_DATASETS]
    ben_in = [d for d in datasets if d in BENIGN_DATASETS]
    if atk_in and not ben_in:
        datasets = atk_in + BENIGN_DATASETS

    thresholds = args.threshold_sweep if args.threshold_sweep else [args.threshold]

    # Override module-level POOLING if --pooling is given
    global POOLING
    if args.pooling is not None:
        POOLING = args.pooling

    rng = random.Random(RANDOM_SEED)
    model_name = args.model

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print(f"  IMAG — Evaluation")
    print(f"  Model       : {model_name}")
    print(f"  Datasets    : {datasets}")
    if atk_in and not ben_in:
        print(f"  (auto-added benign datasets for FPR evaluation)")
    print(f"  Threshold(s): {thresholds}")
    print(f"  Pooling     : {POOLING}")
    print(f"  Seeds       : {args.seeds}  (seed-size={args.seed_size})")
    print(f"  Active Immunity: {'OFF' if args.no_active_immunity else 'ON'}")
    print("=" * 65)

    print(f"\n[1] Loading LLM: {model_name}  (device={device})")
    llm = TargetLLM(model_name, device=device)

    print("\n[2] Finding critical layer...")
    if args.find_critical_layer:
        print("  Searching via Eq. 2 (slow — all layers × seed pairs)...")
        critical_layer = get_critical_layer(llm, model_name)
    else:
        critical_layer = args.critical_layer
        print(f"  Critical layer: {critical_layer}  (use --find-critical-layer to auto-search)")

    # ── Phase A: encode everything once (GPU — expensive) ─────────────────────
    print("\n[3] Encoding seed prompts...")
    seed_atk, seed_ben = build_seed_vectors(
        llm, critical_layer, n_seeds=args.seed_size, seed_strategy=args.seeds, model_name=model_name
    )

    print(f"\n[4] Pre-encoding {len(datasets)} dataset(s) (GPU — done once for all T)...")
    print(f"     First {N_DATASET_SEEDS} samples per dataset → memory seeds; rest → evaluation.")
    encodings: dict[str, dict] = {}   # ds → encode_dataset() dict
    for ds in datasets:
        try:
            encodings[ds] = encode_dataset(ds, llm, critical_layer, rng, model_name=model_name)
        except FileNotFoundError as e:
            print(f"  SKIP {ds}: {e}")

    # ── Aggregate actual benign seeds from evaluation datasets ───────────────────
    # Use XSTest/JailbreakBench/WildJailbreak benign prompts as the benign memory
    # reference for attack datasets.  These are the same benchmark prompts the paper
    # uses.  The global handcrafted seed_ben ("How to bake bread?") looks too similar
    # to obfuscated attacks at deep layers — both appear as "user message" to the model.
    actual_seed_ben: list[np.ndarray] = []
    for ds in BENIGN_DATASETS:
        if ds in encodings and encodings[ds]["ds_seed_ben"]:
            actual_seed_ben.extend(encodings[ds]["ds_seed_ben"])
    if actual_seed_ben:
        print(f"\n  Benign reference: {len(actual_seed_ben)} actual seeds "
              f"from {[d for d in BENIGN_DATASETS if d in encodings]}")
    else:
        print(f"\n  Benign reference: {len(seed_ben)} global handcrafted seeds (fallback)")

    # Memory cost is constant across all T — compute once here
    hidden_dim = llm.model.config.hidden_size
    n_ben_used = len(actual_seed_ben) if actual_seed_ben else len(seed_ben)
    mem_cost = _compute_memory_cost(len(seed_atk), n_ben_used, hidden_dim)

    # ── Phase B: classify at each threshold (CPU numpy — fast) ────────────────
    for threshold in thresholds:
        if len(thresholds) > 1:
            print(f"\n{'#'*65}")
            print(f"  THRESHOLD = {threshold}")
            print(f"{'#'*65}")

        print(f"\n[5] Classifying  (T={threshold})...")
        all_results = []
        t_total = time.time()

        for ds, enc in encodings.items():
            print(f"\n{'='*65}")
            print(f"  Dataset: {ds}")
            # Attack datasets: use dataset-specific attack seeds + actual benign seeds
            # Benign datasets: use global attack seeds + dataset-specific benign seeds
            if ds in ATTACK_DATASETS:
                ds_seed_atk = enc["ds_seed_atk"] if enc["ds_seed_atk"] else seed_atk
                ds_seed_ben = actual_seed_ben if actual_seed_ben else seed_ben
            else:
                ds_seed_atk = enc["ds_seed_atk"] if enc["ds_seed_atk"] else seed_atk
                ds_seed_ben = enc["ds_seed_ben"] if enc["ds_seed_ben"] else seed_ben

            if args.diagnose and ds_seed_atk and ds_seed_ben:
                eval_for_diag = enc["atk_h"] if ds in ATTACK_DATASETS else enc["ben_h"]
                _print_separation_diagnostic(ds, ds_seed_atk, ds_seed_ben, eval_for_diag)

            ai = ActiveImmunity(agent_llm=llm) if not args.no_active_immunity else None
            row = classify_cached(
                ds, enc["atk_texts"], enc["atk_h"], enc["ben_texts"], enc["ben_h"],
                ds_seed_atk, ds_seed_ben,
                threshold, args.no_active_immunity, ai,
            )
            all_results.append(row)

        # Summary table
        print("\n" + "=" * 65)
        print(f"  SUMMARY  ({model_name.split('/')[-1]})  T={threshold}")
        print(f"  {'Dataset':<14} {'Acc':>6} {'F1':>6} {'Recall':>8} {'FPR':>6}")
        print("  " + "-" * 45)
        for r in all_results:
            if r["dataset"] in BENIGN_DATASETS:
                print(f"  {r['dataset']:<14} {'—':>6} {'—':>6} {'—':>8} {r['fpr']*100:>5.1f}%")
            else:
                print(f"  {r['dataset']:<14} {r['acc']:>6.3f} {r['f1']:>6.3f} "
                      f"{r['recall']*100:>7.1f}% {'—':>6}")

        attack_rows = [r for r in all_results if r["dataset"] in ATTACK_DATASETS]
        if attack_rows:
            avg_acc = sum(r["acc"] for r in attack_rows) / len(attack_rows)
            avg_f1  = sum(r["f1"]  for r in attack_rows) / len(attack_rows)
            print("  " + "-" * 45)
            print(f"  {'Average':<14} {avg_acc:>6.3f} {avg_f1:>6.3f}")

        print(f"\n  Total wall time: {time.time()-t_total:.1f}s")
        print(f"\n  Memory Bank  ({mem_cost['hidden_dim']}d, fp16):")
        print(f"    Attack seeds : {mem_cost['n_atk']:>4} vecs = {mem_cost['atk_mb']:.4f} MB")
        print(f"    Benign seeds : {mem_cost['n_ben']:>4} vecs = {mem_cost['ben_mb']:.4f} MB")
        print(f"    Total        :            {mem_cost['total_mb']:.4f} MB")
        print("=" * 65)

        if all_results:
            save_results(all_results, model_name, datasets, threshold, args.no_active_immunity,
                         mem_cost=mem_cost, seed_size=args.seed_size)


if __name__ == "__main__":
    main()
