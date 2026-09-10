"""
Convert ThunderGuard datasets to JBShield format (prompt, label columns).
Output goes to data_thunderguard/ — does not touch JBShield's original data/.

Usage:
    python prepare_data.py
"""

import csv
import os

HERE         = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(HERE, "data_thunderguard")
TRANSFORMED  = os.path.join(HERE, "..", "..", "data", "transformed")
ORIGINAL     = os.path.join(HERE, "..", "..", "adversarial-prompt", "data", "original")

CALIBRATION_N = 50   # prompts per class for calibration


def write_jbshield_csv(path: str, prompts: list[str], label: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "label"])
        for p in prompts:
            w.writerow([p, label])
    print(f"  Wrote {len(prompts):>4} rows → {os.path.basename(path)}")


def load_col(path: str, col: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [r[col].strip() for r in csv.DictReader(f) if r.get(col, "").strip()]


def load_autodan(path: str) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt = row.get("prompt", row.get("autodan_prompt", "")).strip()
            goal   = row.get("goal", "").strip()
            if not prompt:
                continue
            if "[REPLACE]" in prompt and goal:
                prompt = prompt.replace("[REPLACE]", goal)
            texts.append(prompt)
    return texts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output dir: {OUT_DIR}\n")

    # ── Attack datasets ───────────────────────────────────────────────────────

    datasets = {
        "zulu":     lambda: load_col(os.path.join(TRANSFORMED, "zulu_prompts.csv"), "prompt"),
        "base64":   lambda: load_col(os.path.join(TRANSFORMED, "base64_prompts.csv"), "prompt"),
        "pair":     lambda: load_col(os.path.join(TRANSFORMED, "pair_results_mistral-7b_850.csv"), "best_adversarial_prompt"),
        "autodan":  lambda: load_autodan(os.path.join(TRANSFORMED, "autodan_prompts.csv")),
        "drattack": lambda: load_col(os.path.join(TRANSFORMED, "drattack_results_mistral-7b.csv"), "best_adversarial_prompt"),
    }

    for name, loader in datasets.items():
        prompts = loader()
        write_jbshield_csv(os.path.join(OUT_DIR, f"{name}_harmful_test.csv"), prompts, "harmful")

    # ── Benign dataset (XSTest, cap 200 to match paper §4.2) ─────────────────

    benign = load_col(os.path.join(ORIGINAL, "xstest_benign.csv"), "prompt")[:200]
    write_jbshield_csv(os.path.join(OUT_DIR, "xstest_harmless_test.csv"), benign, "harmless")

    # ── Calibration sets (AdvBench + XSTest subsets) ─────────────────────────

    cal_harmful  = load_col(os.path.join(ORIGINAL, "harmful_behaviors.csv"), "goal")[:CALIBRATION_N]
    cal_harmless = benign[:CALIBRATION_N]
    write_jbshield_csv(os.path.join(OUT_DIR, "harmful_calibration.csv"),  cal_harmful,  "harmful")
    write_jbshield_csv(os.path.join(OUT_DIR, "harmless_calibration.csv"), cal_harmless, "harmless")

    print("\nDone.")


if __name__ == "__main__":
    main()