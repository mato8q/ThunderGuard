"""
CLIP Text Encoder — ตามเปเปอร์ JailDAM §2.4.1
ใช้ CLIP encode input text → embedding ขนาด 512 มิติ
"""

import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel


class CLIPTextEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.device = device
        print(f"Loading CLIP encoder: {model_name}")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.dim = 512

    def encode(self, texts):
        """แปลง text เป็น normalized embedding ขนาด [batch, 512]"""
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.pooler_output  # [batch, 512]

        # L2 normalize — ตรงกับ paper ที่ใช้ cosine similarity
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
