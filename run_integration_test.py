"""
Quick Integration Test
Run this to verify the complete adversarial fairness training with LangChain orchestrator.
"""

import torch
from torch.utils.data import DataLoader
from integration_example import train_with_orchestrator

print("="*70)
print("ADVERSARIAL FAIRNESS TRAINING WITH LANGCHAIN ORCHESTRATOR")
print("="*70)

# Generate synthetic data
print("\nGenerating synthetic data...")
n_samples = 1000
n_features = 20
n_sensitive = 2
batch_size = 64

X_train = torch.randn(n_samples, n_features)
y_train = torch.randint(0, 2, (n_samples,)).float()
Z_train = torch.randint(0, 2, (n_samples, n_sensitive)).float()

X_test = torch.randn(200, n_features)
y_test = torch.randint(0, 2, (200,)).float()
Z_test = torch.randint(0, 2, (200, n_sensitive)).float()

train_dataset = torch.utils.data.TensorDataset(X_train, y_train, Z_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test, Z_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(f"[OK] Data ready: {len(train_dataset)} train, {len(test_dataset)} test")

# Run training with orchestrator (reduced iterations for testing)
print("\nStarting training with LangChain orchestrator...")
classifier, adversary, metrics = train_with_orchestrator(
    train_loader=train_loader,
    test_loader=test_loader,
    n_features=n_features,
    n_sensitive=n_sensitive,
    max_iterations=3,  # Quick test with 3 iterations
    epochs_per_iteration=2,  # 2 epochs per iteration
    initial_lambda=[0.5, 0.3, 0.2],
    initial_adv_steps=3,
    memory_file="test_memory.json",
    target_p_rule=0.75,
    target_accuracy=0.70
)

print("\n" + "="*70)
print("[OK] Integration test completed successfully!")
print("="*70)
print(f"\nFinal Results:")
print(f"  Accuracy: {metrics['accuracy']:.4f}")
print(f"  P-rule:   {metrics['p_rule']:.4f}")
print(f"\nYou can now integrate this into your notebook!")
