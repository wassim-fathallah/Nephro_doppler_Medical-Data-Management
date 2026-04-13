"""Quick diagnostic to check tensor shapes"""
import torch
import torch.nn as nn

class MockClassifier(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def extract_features(self, x):
        """Return the 32-dim hidden representation for adversarial training."""
        return self.feature_extractor(x)
    
    def forward(self, x):
        hidden = self.extract_features(x)
        return self.classifier_head(hidden)


class MockAdversary(nn.Module):
    def __init__(self, n_sensitive: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 16),  # Takes classifier's hidden layer
            nn.ReLU(),
            nn.Linear(16, n_sensitive)
        )
    
    def forward(self, hidden):
        return self.net(hidden)


# Test with same dimensions as in integration_example.py
n_features = 20
n_sensitive = 2
batch_size = 64

classifier = MockClassifier(n_features)
adversary = MockAdversary(n_sensitive)

# Create sample batch
X = torch.randn(batch_size, n_features)

print("Testing shapes:")
print(f"Input X shape: {X.shape}")

# Extract features
hidden = classifier.extract_features(X)
print(f"Hidden features shape: {hidden.shape}")

# Classifier output
output = classifier(X)
print(f"Classifier output shape: {output.shape}")

# Adversary prediction
adv_pred = adversary(hidden)
print(f"Adversary prediction shape: {adv_pred.shape}")

print("\n✅ All shapes compatible!")
