"""
Classifier neural network.

Key design: exposes intermediate embeddings (32-dim) so the adversary can
work on the representation space rather than just the final scalar prediction.
This lets the adversary detect proxy features hidden in learned representations.

Architecture:
    Input → FC(64) → ReLU → Dropout
          → FC(32) → ReLU → Dropout     ← embedding exposed here
          → FC(32) → ReLU               ← "embedding layer"
          → FC(1)  → Sigmoid            ← prediction
"""

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, n_features: int, embedding_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Named separately so we can extract it cleanly
        self.embedding_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 32-dim representation — used by adversary and proxy detection."""
        return self.embedding_layer(self.encoder(x))

    def forward(self, x: torch.Tensor):
        embedding = self.get_embedding(x)
        return self.head(embedding), embedding
