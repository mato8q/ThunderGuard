# Adversarial Prompt Dataset

Base64, Zulu, PAIR, and DrAttack attacks on 820 harmful prompts (AdvBench + HEx-PHI).

## Attacks

| Attack | Mechanism | Cost | Runner |
|---|---|---|---|
| **Base64** | Encodes the harmful prompt in Base64 — model decodes before refusing, bypassing filters | Free | — |
| **Zulu** | Translates the prompt to Zulu (low-resource language) to evade English-trained safety filters | Free | — |
| **PAIR** | Attacker LLM iteratively rewrites the prompt based on judge feedback over up to 20 rounds | ~$80–120 / 820 prompts | `run_pair.py` |
| **DrAttack** | Decomposes the harmful prompt into 3 innocent sub-prompts, then uses in-context learning to reconstruct the intent inside the target model | ~$24 / 820 prompts | `run_drattack.py` |

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

Attacker LLM iteratively rewrites a jailbreak prompt using feedback from a judge LLM (score 1–10). Runs K=3 parallel streams and returns the best result. Based on Chao et al. 2023.

**Setup:**
```bash
pip install openai
export OPENAI_API_KEY=sk-...
ollama pull mistral
```

**Run:**
```bash
python run_pair.py --n 1          # smoke test (~$0.10)
python run_pair.py --n 10         # pilot (~$1–2)
python run_pair.py                # full run (~$80–120, ~17 hrs)
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

| Role | Model | API |
|---|---|---|
| Attacker | `gpt-4o-mini` | OpenAI |
| Judge | `gpt-4o` | OpenAI |
| Target | `mistral` | Ollama (free) |

---

## DrAttack (`run_drattack.py`)

Decomposes the harmful prompt into 3 sub-prompts that each appear innocuous, then uses in-context learning to guide the target model into implicitly reassembling the harmful intent. Iteratively refines sub-prompts via synonym substitution. Based on Li et al. 2024.

**Setup:**
```bash
pip install openai
export OPENAI_API_KEY=sk-...
ollama pull mistral
```

**Run:**
```bash
python run_drattack.py --n 1      # smoke test (~$0.02)
python run_drattack.py --n 10     # pilot (~$0.20)
python run_drattack.py            # full run (~$24, ~8–10 hrs)
```

**Options:**
```
--target MODEL   Ollama target model (default: mistral)
--n N            Limit to first N prompts
--iters N        Max synonym refinement iterations (default: 10)
--output PATH    Output CSV path
```

**Config:**

| Role | Model | API |
|---|---|---|
| Decomposer | `gpt-4o` | OpenAI |
| Synonym refiner | `gpt-4o` | OpenAI |
| Judge | `gpt-4o` | OpenAI |
| Target | `mistral` | Ollama (free) |

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

## References

- Zou et al. 2023 — AdvBench
- Qi et al. 2023 — HEx-PHI
- Wei et al. 2023 — Base64 / Zulu (Jailbroken)
- Chao et al. 2023 — PAIR
- Li et al. 2024 — DrAttack
