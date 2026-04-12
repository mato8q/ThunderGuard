import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class MemoryNetwork(nn.Module):
    def __init__(self, clip_model, concept_embeddings, device, embedding_dim=768,
                 max_memory_size=1300, num_classes=14, learning_rate=0.001):
        """
        Text-only memory-based classification model with soft attention.
        - Uses CLIP text encoder only (no image input).
        - Memory concepts are 768-dim (text space).
        - MLP classifies using the text memory output.
        """
        super(MemoryNetwork, self).__init__()

        self.clip_model = clip_model
        self.device = device
        self.embedding_dim = embedding_dim
        self.max_memory_size = max_memory_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Trainable memory concepts: (max_memory_size, 768)
        self.memory_concepts = nn.Parameter(concept_embeddings)

        # MLP Classifier: text memory output (768) → num_classes
        self.mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )

    def entropy_loss(self):
        p = F.softmax(self.memory_concepts, dim=0)
        entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1).mean()
        return -entropy

    def completeness_loss(self, memory_output, text_embedding):
        loss = torch.norm(memory_output - text_embedding, p=2, dim=-1).mean()
        return loss

    def encode_text(self, input_ids, attention_mask):
        """Encode text using CLIP text encoder."""
        with torch.no_grad():
            out = self.clip_model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
            text_embedding = out if isinstance(out, torch.Tensor) else out.pooler_output
        return F.normalize(text_embedding, p=2, dim=-1)  # (batch, 768)

    def attention_memory_lookup(self, text_embedding):
        """
        Compute weighted sum of memory using soft attention over text embeddings.
        Returns memory output (batch, 768) and attention weights.
        """
        attention_scores = torch.matmul(text_embedding, self.memory_concepts.T)  # (batch, memory_size)
        attention_weights = F.softmax(attention_scores, dim=-1)                   # (batch, memory_size)
        memory_output = torch.matmul(attention_weights, self.memory_concepts)     # (batch, 768)
        return memory_output, attention_weights

    def forward(self, text_input_ids, text_attention_mask):
        """
        Forward pass (text only).
        Returns: logits, memory_output, text_embedding, attention_weights
        """
        text_embedding = self.encode_text(text_input_ids, text_attention_mask)  # (batch, 768)
        memory_output, attention_weights = self.attention_memory_lookup(text_embedding)
        logits = self.mlp(memory_output)
        return logits, memory_output, text_embedding, attention_weights
