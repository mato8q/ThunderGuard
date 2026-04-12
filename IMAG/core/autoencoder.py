"""
Autoencoder — ตามเปเปอร์ JailDAM §2.4.2

Train บน safe samples เท่านั้น (ไม่ต้องการ harmful data)
Loss = MSE(reconstruct(Z), Z)  — Eq. 7

ตอน inference:
- reconstruction_error สูง → input "แปลก" จาก safe distribution → ATTACK
- reconstruction_error ต่ำ → BENIGN
"""

import torch
import torch.nn as nn
import numpy as np
import os


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.decoder_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder_net(x)
        return self.decoder_net(z)

    def reconstruction_error(self, x):
        """คำนวณ MSE error ระหว่าง input กับ reconstruction"""
        with torch.no_grad():
            x_hat = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=-1)   # [batch]


class AutoencoderTrainer:
    def __init__(self, autoencoder, lr=1e-3, device="cpu"):
        self.model = autoencoder.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, attention_features, epochs=200, batch_size=16):
        """
        Train บน attention features ของ safe samples — Eq. 7
        attention_features: numpy array [n_safe, N_concepts]
        """
        self.model.train()
        X = torch.tensor(attention_features, dtype=torch.float32).to(self.device)

        for epoch in range(epochs):
            idx = torch.randperm(len(X))
            X = X[idx]

            total_loss = 0.0
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output, batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} — Loss: {total_loss:.6f}")

        self.model.eval()

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"  Autoencoder saved → {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"  Autoencoder loaded ← {path}")
