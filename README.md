# Adversarial Prompt Dataset

Base64, Zulu, PAIR, and DrAttack attacks on 820 harmful prompts (AdvBench + HEx-PHI).
**All models run fully local. Zero API cost.**

## Attacks

| Attack | Mechanism | Cost | Time | Runner |
|---|---|---|---|---|
| **Base64** | Encodes the harmful prompt in Base64 — model decodes before refusing, bypassing filters | Free | <1s | — |
| **Zulu** | Translates the prompt to Zulu (low-resource language) to evade English-trained safety filters | Free | <1s | — |
| **PAIR** | Attacker LLM (Qwen 3.5 9B) iteratively rewrites the prompt based on judge feedback (Qwen 3.5 9B) over up to 20 rounds | Free | 2–3 hrs | `run_pair.py` |
| **DrAttack** | Decomposes the harmful prompt into 3 innocent sub-prompts via Qwen 3.5 9B, ranks by semantic similarity (nomic-embed-text), scores with Qwen 3.5 9B | Free | 4–6 hrs | `run_drattack.py` |

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
ollama pull qwen3.5:9b
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

**Config (this implementation — fully local):**

| Role | Model | VRAM | Where |
|---|---|---|---|
| Attacker | `qwen3.5:9b` (Q4_K_M) | 6.6 GB | Ollama (local) |
| Judge | `qwen3.5:9b` (Q4_K_M) | 6.6 GB | Ollama (local) |
| Target | `mistral` (default, Q4_K_M) | 4.2 GB | Ollama (local) |
| **Active (sequential)** | — | **~6–7 GB** | Ollama (local) |

**Original paper config (Chao et al. 2023):**

| Role | Paper model | This impl. | Notes |
|---|---|---|---|
| Attacker | `Vicuna-13B-v1.5` | `qwen3.5:9b` | Paper used Vicuna because Llama-2 over-refuses harmless prompts; Qwen 3.5 9B is the modern permissive equivalent |
| Judge | `GPT-4` | `qwen3.5:9b` | Qwen 3.5 9B Q4 hits 89.2% IFEval (vs GPT-4 ~85%); reliable `Rating: [[N]]` formatting |
| Targets evaluated | GPT-3.5, GPT-4, Vicuna-13B, Llama-2-7B-Chat, Claude-1/2, PaLM-2, Gemini-Pro | any Ollama model | swap via `--target` |

---

## DrAttack (`run_drattack.py`)

Decomposes the harmful prompt into 3 sub-prompts that each appear innocuous, then uses in-context learning to guide the target model into implicitly reassembling the harmful intent. Iteratively refines sub-prompts via synonym substitution ranked by semantic similarity. Based on Li et al. 2024. **Now fully local.**

**Setup:**
```bash
pip install openai requests  # for OpenAI client abstraction + Ollama embeddings API
ollama pull qwen3.5:9b
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

**Config (this implementation — fully local):**

| Role | Model | VRAM | Where |
|---|---|---|---|
| Decomposer | `qwen3.5:9b` (Q4_K_M) | 6.6 GB | Ollama (local) |
| Embeddings | `nomic-embed-text` (Q4_K_M) | ~1.0 GB | Ollama (local) |
| Judge | `qwen3.5:9b` (Q4_K_M) | 6.6 GB | Ollama (local) |
| Target | `mistral` (default, Q4_K_M) | 4.2 GB | Ollama (local) |
| **Active (sequential)** | — | **~6–7 GB** | Ollama (local) |

**Original paper config (Li et al. 2024 EMNLP):**

| Role | Paper model | This impl. | Notes |
|---|---|---|---|
| Decomposer / synonym generator | `GPT-4` | `qwen3.5:9b` | Paper used GPT-4 to build the parsing tree and generate synonyms; Qwen 3.5 9B has comparable IFEval for JSON output |
| Embeddings | `text-embedding-ada-002` (OpenAI) | `nomic-embed-text` (Ollama) | Used to rank synonym candidates by cosine similarity to the original sub-prompt |
| Judge | `GPT-3.5-turbo` | `qwen3.5:9b` | Paper used GPT-3.5; the local 9B exceeds GPT-3.5 on most benchmarks |
| Targets evaluated | GPT-3.5, GPT-4, Llama-2, Vicuna | any Ollama model | swap via `--target`. Paper reported 80% ASR on GPT-4 |

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

| Scenario | Active VRAM | GPU | Feasible |
|----------|------|-----|----------|
| PAIR + target (sequential, qwen3.5:9b) | ~6–7 GB | RTX 4070 (12GB) | ✅ Yes (~10 min per prompt) |
| DrAttack + target (sequential, qwen3.5:9b) | ~6–7 GB | RTX 4070 (12GB) | ✅ Yes (~15 min per prompt) |
| Concurrent (qwen3.5:9b + target loaded) | ~11 GB | RTX 4070 (12GB) | ⚠️ Tight — leave sequential |
| Concurrent (all loaded) | ~18 GB | RTX 4090, A100 | ✅ Yes (but overkill) |

**Note:** Quantization level (Q4_K_M vs Q5_K_M) trades quality for memory. Q4_K_M retains ~95% quality at half VRAM. `qwen3.5:9b` at Q4_K_M is the recommended sweet spot for RTX 4070 — it matches GPT-4o-mini on MMLU-Pro and IFEval while fitting in 6.6 GB.

---

## References

- Zou et al. 2023 — AdvBench
- Qi et al. 2023 — HEx-PHI
- Wei et al. 2023 — Base64 / Zulu (Jailbroken)
- Chao et al. 2023 — PAIR ([Jailbreaking Black Box LLMs in Twenty Queries](https://arxiv.org/abs/2310.08419)) — paper used Vicuna-13B-v1.5 attacker + GPT-4 judge
- Li et al. 2024 — DrAttack ([Prompt Decomposition and Reconstruction](https://arxiv.org/abs/2402.16914), EMNLP Findings) — paper used GPT-4 decomposer + text-embedding-ada-002 + GPT-3.5-turbo judge
- Qwen Team 2026 — [Qwen 3.5 9B](https://qwen.ai/blog?id=qwen3.5) (open-weight, Apache 2.0, March 2026)
- Nomic AI — nomic-embed-text (efficient local embeddings)
