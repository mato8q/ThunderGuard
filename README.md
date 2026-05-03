# Adversarial Prompt Dataset

Base64, Zulu, PAIR, and DrAttack attacks on 820 harmful prompts (AdvBench + HEx-PHI).
**All models run fully local. Zero API cost.**

## Attacks

| Attack | Mechanism | Cost | Time | Runner |
|---|---|---|---|---|
| **Base64** | Encodes the harmful prompt in Base64 — model decodes before refusing, bypassing filters | Free | <1s | — |
| **Zulu** | Translates the prompt to Zulu (low-resource language) to evade English-trained safety filters | Free | <1s | — |
| **PAIR** | Attacker LLM (Qwen 2.5 7B) iteratively rewrites the prompt based on judge feedback (Mistral 7B) over up to 20 rounds | Free | 2–3 hrs | `run_pair.py` |
| **DrAttack** | Decomposes the harmful prompt into 3 innocent sub-prompts via Qwen 2.5 7B, ranks by semantic similarity (nomic-embed-text), scores with Mistral 7B | Free | 4–6 hrs | `run_drattack.py` |

---

## Structure

```
exports/
├── README.md
├── pyproject.toml              ← dependencies (uv)
├── requirements.txt            ← dependencies (pip)
├── run_pair.py                 ← PAIR attack runner
├── run_drattack.py             ← DrAttack runner
└── data/
    ├── original/
    │   ├── harmful_behaviors.csv   ← AdvBench (520 prompts)
    │   └── hex_phi.csv             ← HEx-PHI (300 prompts)
    └── transformed/
        ├── base64_prompts.csv              ← ready to paste into any model
        ├── zulu_prompts.csv                ← ready to paste into any model
        ├── pair_results_mistral.csv        ← PAIR output
        └── drattack_results_mistral.csv    ← DrAttack output
```

---

## Original Datasets (`data/original/`)

Both files: `goal` (harmful prompt), `target` (expected jailbroken response start), `category` (HEx-PHI only).

```python
import pandas as pd
df = pd.read_csv("data/original/harmful_behaviors.csv")
print(df["goal"][0])
```

---

## Base64 & Zulu (`data/transformed/`)

No setup needed. Open the CSV, copy any `prompt` value, paste into a model.

```python
import pandas as pd
df = pd.read_csv("data/transformed/base64_prompts.csv")
print(df["prompt"][0])   # paste into ChatGPT / Claude / etc.
```

---

## PAIR (`run_pair.py`)

Attacker LLM iteratively rewrites a jailbreak prompt using feedback from a judge LLM (score 1–10). Runs K=3 parallel streams and returns the best result. Based on Chao et al. 2023. **Now fully local.**

**Setup:**
```bash
pip install openai  # for OpenAI client abstraction (uses Ollama backend)
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama serve       # in another terminal
```

**Run:**
```bash
python run_pair.py --n 1          # smoke test (~10 min)
python run_pair.py --n 10         # pilot (~30 min)
python run_pair.py                # full run (~2–3 hrs on RTX 4070)
```

**Options:**
```
--target MODEL   Ollama target model (default: mistral)
--n N            Limit to first N prompts
--iters N        Max iterations per stream (default: 20)
--k N            Parallel streams (default: 3)
--output PATH    Output CSV path
```

**Config:**

| Role | Model | VRAM | Where |
|---|---|---|---|
| Attacker | `qwen2.5:7b` (Q4_K_M) | 4.2 GB | Ollama (local) |
| Judge | `mistral:7b` (Q4_K_M) | 4.2 GB | Ollama (local) |
| Target | `mistral` (default) | 4.2 GB | Ollama (local) |
| **Total (sequential)** | — | **~4–5 GB active** | Ollama (local) |

---

## DrAttack (`run_drattack.py`)

Decomposes the harmful prompt into 3 sub-prompts that each appear innocuous, then uses in-context learning to guide the target model into implicitly reassembling the harmful intent. Iteratively refines sub-prompts via synonym substitution ranked by semantic similarity. Based on Li et al. 2024. **Now fully local.**

**Setup:**
```bash
pip install openai requests  # for OpenAI client abstraction + Ollama embeddings API
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull nomic-embed-text
ollama serve               # in another terminal
```

**Run:**
```bash
python run_drattack.py --n 1      # smoke test (~15 min)
python run_drattack.py --n 10     # pilot (~1 hr)
python run_drattack.py            # full run (~4–6 hrs on RTX 4070)
```

**Options:**
```
--target MODEL   Ollama target model (default: mistral)
--n N            Limit to first N prompts
--iters N        Max combinations to test (default: 10)
--output PATH    Output CSV path
```

**Config:**

| Role | Model | VRAM | Where |
|---|---|---|---|
| Decomposer | `qwen2.5:7b` (Q4_K_M) | 4.2 GB | Ollama (local) |
| Embeddings | `nomic-embed-text` (Q4_K_M) | ~1.0 GB | Ollama (local) |
| Judge | `mistral:7b` (Q4_K_M) | 4.2 GB | Ollama (local) |
| Target | `mistral` (default) | 4.2 GB | Ollama (local) |
| **Total (sequential)** | — | **~4–5 GB active** | Ollama (local) |

---

## Output Schema (PAIR & DrAttack)

| Column | Description |
|---|---|
| `id` | Prompt ID |
| `source` | `harmful_behaviors` or `hex_phi` |
| `category` | Harm category |
| `goal` | Original harmful prompt |
| `target_str` | Expected jailbroken response prefix |
| `best_adversarial_prompt` | Best jailbreak prompt found |
| `best_response` | Target model's response |
| `best_score` | Judge score 1–10 (10 = jailbreak) |
| `iterations` | Iterations used |

---

## Hardware Requirements

All runners are optimized for **RTX 4070 (12GB VRAM)**. Sequential execution (load model → run → unload) fits comfortably.

| Scenario | VRAM | GPU | Feasible |
|----------|------|-----|----------|
| PAIR + target (sequential) | ~4–5 GB | RTX 4070 (12GB) | ✅ Yes (~10 min per prompt) |
| DrAttack + target (sequential) | ~4–5 GB | RTX 4070 (12GB) | ✅ Yes (~15 min per prompt) |
| Concurrent (all 3 loaded) | ~15 GB | RTX 4070 | ❌ No (exceeds VRAM) |
| Concurrent (all 3 loaded) | 24+ GB | RTX 4090, A100 | ✅ Yes (but overkill) |

**Note:** Quantization level (Q4_K_M vs Q5_K_M) trades quality for memory. Q4_K_M retains ~95% quality at half VRAM. Recommended for RTX 4070.

---

## References

- Zou et al. 2023 — AdvBench
- Qi et al. 2023 — HEx-PHI
- Wei et al. 2023 — Base64 / Zulu (Jailbroken)
- Chao et al. 2023 — PAIR (Jailbreaking Black Box LLMs in Twenty Queries)
- Li et al. 2024 — DrAttack (Prompt Decomposition and Reconstruction)
- Nomic AI — nomic-embed-text (efficient local embeddings)
