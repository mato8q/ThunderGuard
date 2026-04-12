"""
LLM Interface — โหลด target LLM สำหรับ IMAG

มีสองความสามารถ:
1. extract_hidden_states()  → ดึง activation vector จาก layer ที่กำหนด
2. generate()               → สร้างคำตอบจาก prompt (ใช้ใน Active Immunity)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TargetLLM:
    def __init__(self, model_name_or_path, device="cuda"):
        self.device = device
        print(f"Loading model: {model_name_or_path} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    # ── Hidden States ──────────────────────────────────────────────────────────

    def extract_hidden_states(self, prompt, target_layer=-1):
        """
        ดึง hidden state จาก layer ที่ระบุ แล้ว mean-pool ทุก token positions
        target_layer=-1  → เลเยอร์สุดท้าย (แนะนำ)
        คืน numpy array ขนาด [hidden_dim]
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        all_hidden = outputs.hidden_states       # tuple: (n_layers+1, batch, seq, hidden)
        layer_hidden = all_hidden[target_layer]  # [1, seq_len, hidden_dim]
        # Mean-pool over all token positions — more stable than last-token only
        pooled = layer_hidden.mean(dim=1)        # [1, hidden_dim]
        return pooled.to(torch.float32).squeeze(0).cpu().numpy()  # [hidden_dim]

    # ── Text Generation ────────────────────────────────────────────────────────

    def generate(self, prompt, max_new_tokens=256):
        """สร้างคำตอบจาก prompt — ใช้ใน Active Immunity stage"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # ตัดส่วน prompt ออก เหลือแค่ generated text
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
