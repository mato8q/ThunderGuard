# Adversarial Prompt Dataset

Base64, Zulu, PAIR, and DrAttack attacks on 820 harmful prompts (AdvBench + HEx-PHI).
**All models run fully local. Zero API cost.**

## Attacks

| Attack | Mechanism | Cost | Time | Runner |
|---|---|---|---|---|
| **Base64** | Encodes the harmful prompt in Base64 — model decodes before refusing, bypassing filters | Free | <1s | — |
| **Zulu** | Translates the prompt to Zulu (low-resource language) to evade English-trained safety filters | Free | <1s | — |
| **PAIR** | Attacker LLM (Qwen 3.5 9B) iteratively rewrites the prompt based on judge feedback (Qwen 3.5 9B) over up to 20 rounds | Free | ~2 min for 10 / ~3 hrs for 820 | `run_pair.py` |
| **DrAttack** | Decomposes the harmful prompt into 3 innocent sub-prompts via Qwen 3.5 9B, ranks by semantic similarity (nomic-embed-text), scores with Qwen 3.5 9B | Free | ~2.5 min for 10 / ~3.5 hrs for 820 | `run_drattack.py` |

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
        ├── pair_results_mistral-7b.csv        ← PAIR output
        └── drattack_results_mistral-7b.csv    ← DrAttack output
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

## Prerequisites

Both scripts connect to Ollama at `http://localhost:1234` (configured via `OLLAMA_HOST=0.0.0.0:1234` on this machine). If your Ollama uses the default port, update `OLLAMA_HOST` at the top of each script to `http://localhost:11434`.

Run `ollama list` to confirm the three required models are present: `qwen3.5:9b`, `mistral:7b`, `nomic-embed-text`.

---

## PAIR (`run_pair.py`)

Attacker LLM iteratively rewrites a jailbreak prompt using feedback from a judge LLM (score 1–10). Runs K=3 parallel streams and returns the best result. Based on Chao et al. 2023. **Now fully local.**

**Setup:**
```bash
ollama pull qwen3.5:9b
ollama pull mistral:7b
ollama serve       # in another terminal (skip if already running)
uv sync            # or: pip install ollama requests tqdm
```

**Run:**
```bash
uv run python run_pair.py --n 1                    # smoke test (~3 min)
uv run python run_pair.py --n 10                   # pilot (~30 min)
uv run python run_pair.py                          # full run (~25 hrs for 820 prompts)
```

> **Live ETA**: tqdm shows `[elapsed<remaining, sec/prompt]` after the first prompt — most accurate estimate.

**Options:**
```
--target MODEL   Ollama target model (default: mistral:7b)
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
ollama pull qwen3.5:9b
ollama pull mistral:7b
ollama pull nomic-embed-text
ollama serve               # in another terminal (skip if already running)
uv sync                    # or: pip install ollama requests tqdm
```

**Run:**
```bash
uv run python run_drattack.py --n 1      # smoke test (~15 sec)
uv run python run_drattack.py --n 10     # pilot (~25 min)
uv run python run_drattack.py            # full run (~22 hrs for 820 prompts)
```

> **Live ETA**: tqdm shows `[elapsed<remaining, sec/prompt]` after the first prompt — most accurate estimate.

**Options:**
```
--target MODEL   Ollama target model (default: mistral:7b)
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

## Expected Results (Paper Baselines)

These are the headline ASR numbers reported in the original papers. Use them to sanity-check whether your local run is producing reasonable output. **Local Qwen 3.5 9B will trail these numbers — RLHF on closed models is hard to match.**

### PAIR — Chao et al. 2023 (Table 1)

Evaluated on AdvBench harmful_behaviors subset (~50 prompts), K=3 streams, max 20 iters, GPT-4 judge:

| Target model | Paper ASR | Avg queries |
|---|---|---|
| Vicuna-13B-v1.5 | **88%** | 11.9 |
| GPT-3.5-Turbo | **60%** | 15.6 |
| GPT-4 | **50%** | 16.6 |
| Gemini-Pro | **73%** | 14.6 |
| Llama-2-7B-Chat | **10%** | 33.8 |
| Claude-1 / Claude-2 | **0–4%** | 18+ |

**Local expectations** with `qwen3.5:9b` attacker + judge, target = `mistral:7b`:
- Mistral lacks the heavy RLHF of GPT-4/Llama-2-Chat → expect ASR closer to Vicuna's range
- Realistic target: **40–70% ASR** (lower than paper because the local attacker is weaker than the paper's Vicuna-13B + GPT-4 judge combo)

### DrAttack — Li et al. 2024 EMNLP (Table 1)

Evaluated on AdvBench harmful_behaviors subset (~52 prompts), N=3 sub-prompts, K=5 variants, GPT-3.5-turbo judge, **human evaluation**:

| Target model | Paper ASR | Avg queries |
|---|---|---|
| Vicuna-7B | **98.1%** | 7.6 |
| GPT-3.5-Turbo | **86.2%** | 12.4 |
| GPT-4 | **84.6%** | 12.9 |
| Llama-2-13B-Chat | **44.2%** | — |
| Llama-2-7B-Chat | **38.5%** | 16.1 |

**Local expectations** with `qwen3.5:9b` decomposer + judge, target = `mistral:7b`:
- DrAttack's wordgame mode is target-agnostic — should generalize well
- Realistic target: **60–80% ASR**, average ~5–8 iterations per prompt

### Why your local numbers will differ from the paper

1. **Weaker attacker/judge** — Qwen 3.5 9B Q4 is strong, but GPT-4 still has more nuanced adversarial prose generation
2. **Different target** — paper evaluated GPT-4, Vicuna, Llama-2; you're attacking `mistral:7b` (different alignment, different refusal patterns)
3. **Judge calibration drift** — local judge may be more lenient or stricter than GPT-4
4. **Smaller test set** — paper used ~50 prompts; you have 820 → wider distribution, smoother ASR

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

All runners are optimized for **RTX 4090 (24 GB VRAM)**. All three models fit concurrently with room to spare.

| Model | VRAM |
|-------|------|
| `qwen3.5:9b` (Q4_K_M) | 6.6 GB |
| `mistral:7b` (Q4_K_M) | 4.4 GB |
| `nomic-embed-text` | 0.3 GB |
| **Total (all concurrent)** | **~11 GB** |

All models stay resident in VRAM simultaneously — no swapping between attacker → target → judge steps. This is why DrAttack completes in ~15 sec/prompt and PAIR in ~14 sec/prompt.

**On a smaller GPU (e.g. RTX 4070, 12 GB):** reduce to one PAIR stream (`--k 1`) to avoid Ollama evicting models mid-run.

---

## References

- Zou et al. 2023 — AdvBench
- Qi et al. 2023 — HEx-PHI
- Wei et al. 2023 — Base64 / Zulu (Jailbroken)
- Chao et al. 2023 — PAIR ([Jailbreaking Black Box LLMs in Twenty Queries](https://arxiv.org/abs/2310.08419)) — paper used Vicuna-13B-v1.5 attacker + GPT-4 judge
- Li et al. 2024 — DrAttack ([Prompt Decomposition and Reconstruction](https://arxiv.org/abs/2402.16914), EMNLP Findings) — paper used GPT-4 decomposer + text-embedding-ada-002 + GPT-3.5-turbo judge
- Qwen Team 2026 — [Qwen 3.5 9B](https://qwen.ai/blog?id=qwen3.5) (open-weight, Apache 2.0, March 2026)
- Nomic AI — nomic-embed-text (efficient local embeddings)
