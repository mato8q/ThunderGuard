"""
LLM Interface — loads target LLM for IMAG.

Capabilities:
  1. extract_hidden_states()   — activation vector from a specific layer
  2. find_critical_layer()     — layer search via Eq. 2 (NEW: was hardcoded before)
  3. generate()                — text generation for Active Immunity
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TargetLLM:
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        self.device = device
        print(f"Loading model: {model_name_or_path} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers

    # ── Hidden States ──────────────────────────────────────────────────────────

    def extract_hidden_states(
        self,
        prompt: str,
        target_layer: int = -1,
        pooling: str = "mean",
    ) -> np.ndarray:
        """
        Extract hidden state from the specified layer.

        Parameters
        ----------
        target_layer : int
            Layer index. Negative indexing supported (-1 = last layer).
            Paper uses the critical layer found by find_critical_layer().
        pooling : "mean" | "last"
            "last"  — final token only  (paper §3.1 description)
            "mean"  — mean over all tokens  (more stable for short prompts)

        Returns
        -------
        np.ndarray  shape [hidden_dim]
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        all_hidden = outputs.hidden_states   # tuple: (n_layers+1, 1, seq, hidden)
        layer_hidden = all_hidden[target_layer]  # [1, seq_len, hidden_dim]

        if pooling == "last":
            # Paper §3.1: "hidden states are extracted from the final token"
            pooled = layer_hidden[:, -1, :]   # [1, hidden_dim]
        else:
            # Mean-pool — more robust for variable-length prompts
            pooled = layer_hidden.mean(dim=1)  # [1, hidden_dim]

        return pooled.to(torch.float32).squeeze(0).cpu().numpy()  # [hidden_dim]

    # ── Critical Layer Search (Eq. 2) ─────────────────────────────────────────

    def find_critical_layer(
        self,
        attack_prompts: list[str],
        benign_prompts: list[str],
        pooling: str = "last",
    ) -> int:
        """
        Find the layer l_t that maximally separates attack from benign activations.
        Paper Eq. 2:
            l_t = argmin_l  (1/n) Σ cos( h_l(x_i^a),  h_l(x_i^b) )

        The layer with MINIMUM average cosine similarity between paired attack/benign
        hidden states is where the representations are most discriminative.

        Parameters
        ----------
        attack_prompts : list of jailbreak prompts (x_i^a)
        benign_prompts : list of safe prompts      (x_i^b)
            Lengths should be equal; pairs are (attack[i], benign[i]).
        pooling : "last" | "mean"  — same choice used during inference

        Returns
        -------
        int : critical layer index (1-indexed, matching HuggingFace hidden_states tuple)
              Use this directly as target_layer in extract_hidden_states().
        """
        n = min(len(attack_prompts), len(benign_prompts))
        assert n > 0, "Need at least one pair of prompts"

        # hidden_states tuple has n_layers+1 entries (index 0 = embedding layer)
        # We search layers 1..n_layers (skip embedding layer 0)
        n_layers = self.num_layers   # transformer layers only
        layer_scores = np.zeros(n_layers + 1)  # index 0 unused

        print(f"  Layer search over {n_layers} layers with {n} prompt pairs...")

        for i in range(n):
            a_inputs = self.tokenizer(attack_prompts[i], return_tensors="pt").to(
                self.model.device
            )
            b_inputs = self.tokenizer(benign_prompts[i], return_tensors="pt").to(
                self.model.device
            )

            with torch.no_grad():
                a_out = self.model(**a_inputs, output_hidden_states=True)
                b_out = self.model(**b_inputs, output_hidden_states=True)

            for l_idx in range(1, n_layers + 1):
                h_a = a_out.hidden_states[l_idx]
                h_b = b_out.hidden_states[l_idx]

                if pooling == "last":
                    va = h_a[0, -1, :].float().cpu().numpy()
                    vb = h_b[0, -1, :].float().cpu().numpy()
                else:
                    va = h_a[0].mean(0).float().cpu().numpy()
                    vb = h_b[0].mean(0).float().cpu().numpy()

                # cosine similarity between this attack/benign pair at layer l
                denom = np.linalg.norm(va) * np.linalg.norm(vb)
                cos = float(np.dot(va, vb) / denom) if denom > 1e-8 else 0.0
                layer_scores[l_idx] += cos

        # Average over n pairs
        layer_scores[1:] /= n

        # argmin: layer where attack/benign representations are most different
        # (lowest cosine similarity = largest angular separation)
        critical_layer = int(np.argmin(layer_scores[1:]) + 1)  # 1-indexed

        print(f"  Critical layer: {critical_layer}  "
              f"(avg cosine = {layer_scores[critical_layer]:.4f})")
        return critical_layer

    # ── Text Generation ────────────────────────────────────────────────────────

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Text generation for Active Immunity stage."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
