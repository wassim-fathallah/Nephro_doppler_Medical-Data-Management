"""
Adversary neural network.

Operates on the classifier's 32-dim EMBEDDING (not the scalar prediction).
This is the key difference from the original code: by seeing the full
representation vector, the adversary can detect proxy correlations that
survive even when the sensitive attribute is explicitly dropped.

The adversary tries to predict ALL sensitive attributes simultaneously
from a single shared embedding — outputs one logit per sensitive attr.
"""

import torch
import torch.nn as nn


class Adversary(nn.Module):
    def __init__(self, embedding_dim: int = 32, n_sensitive: int = 2, n_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),   # one logit per sensitive attr
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (N, n_sensitive) — apply BCEWithLogitsLoss."""
        return self.net(embedding)
