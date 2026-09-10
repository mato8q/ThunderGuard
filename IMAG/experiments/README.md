# Experiment Results Archive

All CSV files from `IMAG/results/` and `results/` (both gitignored) are copied
here for version-controlled backup. Files marked **[KEY]** are used directly in
the paper. Files marked `[SWEEP]` or `[EARLY]` are intermediate runs kept for
reference only.

---

## [KEY] Files used in paper

### IMAG Baseline — Mistral-7B (our datasets)
- `Mistral-7B-Instruct-v0.2_all_t0.1_noAI_20260611_133123.csv`
  IMAG without active immunity, T=0.1, all attack datasets + benign (our data).
  Used as the IMAG "no AI" row in the main comparison table.

- `Mistral-7B-Instruct-v0.2_all_t0.1_fullAI_20260611_124002.csv`
  IMAG with active immunity (Stage 2 LLM judge), T=0.1, all datasets.
  Used as the IMAG "full AI" row in the main comparison table.

### IMAG Baseline — Llama-2-7B
- `Llama-2-7b-chat-hf_all_t0.1_fullAI_20260521_155739.csv`
  IMAG with active immunity, T=0.1, all datasets, Llama-2-7B-chat target.
  Used to show IMAG performance on a second model.

### Fair Comparison — IMAG on JBShield test data
- `Mistral-7B-Instruct-v0.2_jbshield_t0.1_noAI_20260611_193813.csv`
  IMAG evaluated on JBShield's own test split (9 attacks, mistral_test.json).
  avg F1 = 0.180. Used as the apples-to-apples IMAG baseline vs JBShield/CAMS.

### CAMS — Full System
- `Mistral-7B-Instruct-v0.2_cams_td0.3_tm0.5_t0.1_20260612_193721.csv`
  CAMS with concept probe (tau_detect=0.3, tau_memory=0.5) + JBShield seeds.
  avg F1 = 0.967. Main CAMS result row.

### CAMS — Ablation: Memory Bank Only (no concept probe)
- `Mistral-7B-Instruct-v0.2_memory_only_t0.1_20260820_014344.csv`
  Same as CAMS but concept probe disabled (--no-concept-probe).
  avg F1 = 0.966. Shows concept probe contributes ~0.001 F1 — memory bank
  does all the work. Key ablation finding.

---

## [SWEEP] Threshold sweeps — AutoDAN only (Mistral, 2026-05-18)

Early per-dataset threshold exploration. Not used in paper directly but
informed the choice of T=0.1.

- `Mistral-7B-Instruct-v0.2_autodan_*_t0.1_noAI_*.csv` (multiple runs)
- `Mistral-7B-Instruct-v0.2_autodan_*_t0.2_noAI_*.csv`
- `Mistral-7B-Instruct-v0.2_autodan_*_t0.24_noAI_*.csv`
- `Mistral-7B-Instruct-v0.2_zulu_*_t0.1_noAI_*.csv`
- `Mistral-7B-Instruct-v0.2_base64_*_t0.1_noAI_*.csv`

Corresponding `_gaps.csv` files contain per-prompt gap scores from those runs.

---

## [SWEEP] Threshold sweeps — Llama-2-7B (top-level results/)

- `Llama-2-7b-chat-hf_all_t0.2_noAI.csv`
- `Llama-2-7b-chat-hf_all_t0.3_noAI.csv`
- `Llama-2-7b-chat-hf_all_t0.5_noAI.csv`
- `Llama-2-7b-chat-hf_all_t0.7_noAI.csv`
- `Llama-2-7b-chat-hf_autodan_*_t*.csv` (6 files, T=0.0–0.5)

---

## [EARLY] Partial-dataset runs (2026-06-10)

Runs on subsets of attack datasets during debugging. Not used in paper.

- `Mistral-7B-Instruct-v0.2_autodan_drattack_pair_*_20260610_125755.csv`
- `Mistral-7B-Instruct-v0.2_autodan_zulu_base64_pair_*_20260610_131309.csv`

---

## Still needed (not yet run)

- **JBShield baseline on JBShield test data** — run `baselines/jbshield/run_original.py`
  to get the F1 numbers JBShield reports on their own test split.
- **CAMS ablation with fair/handcrafted seeds** — run `cams/evaluate_cams.py`
  with handcrafted seeds instead of JBShield calibration JSON, to get honest
  numbers without data leakage.
