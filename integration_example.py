"""
Integration Example: LangChain Orchestrator with Adversarial Training

This script demonstrates how to integrate the LangChain orchestrator
into an existing PyTorch adversarial training loop.

Key Points:
- Orchestrator is called ONLY at iteration boundaries (not per batch)
- Core PyTorch training logic remains unchanged
- Orchestrator operates as external meta-controller
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any
import numpy as np

# Import the LangChain orchestrator
from langchain_orchestrator import create_orchestrator


# ============================================================================
# MOCK PYTORCH MODELS (Replace with your actual models)
# ============================================================================

class MockClassifier(nn.Module):
    """Placeholder - use your actual Classifier class."""
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
    """Placeholder - use your actual Adversary class."""
    def __init__(self, n_sensitive: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 16),  # Takes classifier's hidden layer
            nn.ReLU(),
            nn.Linear(16, n_sensitive)
        )
    
    def forward(self, hidden):
        return self.net(hidden)


# ============================================================================
# TRAINING FUNCTIONS (Simplified - adapt to your actual implementation)
# ============================================================================

def train_adversary_step(
    adversary: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    adv_optimizer: optim.Optimizer,
    n_steps: int,
    device: torch.device
) -> float:
    """
    Train adversary for n_steps iterations.
    
    Args:
        adversary: Adversary model
        classifier: Classifier model (frozen during adversary training)
        train_loader: DataLoader
        adv_optimizer: Adversary optimizer
        n_steps: Number of gradient steps
        device: torch.device
    
    Returns:
        Average adversary loss
    """
    adversary.train()
    classifier.eval()  # Freeze classifier
    
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    
    for step in range(n_steps):
        for X, y, Z in train_loader:
            X, Z = X.to(device), Z.to(device)
            
            # Get classifier's hidden representation
            with torch.no_grad():
                hidden = classifier.extract_features(X)
            
            # Adversary tries to predict sensitive attributes
            adv_pred = adversary(hidden)
            
            # Loss: adversary wants to maximize its ability to detect bias
            adv_loss = criterion(adv_pred, Z)
            
            adv_optimizer.zero_grad()
            adv_loss.backward()
            adv_optimizer.step()
            
            total_loss += adv_loss.item()
            break  # One batch per step (modify as needed)
    
    return total_loss / max(n_steps, 1)


def train_learner_step(
    classifier: nn.Module,
    adversary: nn.Module,
    train_loader: DataLoader,
    clf_optimizer: optim.Optimizer,
    lambda_vector: List[float],
    device: torch.device
) -> float:
    """
    Train classifier for one epoch with fairness penalty.
    
    Args:
        classifier: Classifier model
        adversary: Adversary model (frozen during classifier training)
        train_loader: DataLoader
        clf_optimizer: Classifier optimizer
        lambda_vector: Fairness weights [L1, L2, L3, ...]
        device: torch.device
    
    Returns:
        Average task loss
    """
    classifier.train()
    adversary.eval()  # Freeze adversary
    
    criterion = nn.BCELoss()
    total_loss = 0.0
    
    for X, y, Z in train_loader:
        X, y, Z = X.to(device), y.to(device), Z.to(device)
        
        # Forward pass
        pred = classifier(X).squeeze()
        
        # Task loss (standard classification)
        task_loss = criterion(pred, y)
        
        # Fairness penalty: adversary should NOT be able to predict sensitive attrs
        # Get hidden representation
        hidden = classifier.extract_features(X)
        adv_pred = adversary(hidden)
        
        # Fairness loss: minimize adversary's prediction accuracy
        # (We want representations to be uninformative about Z)
        fairness_loss = -nn.BCEWithLogitsLoss()(adv_pred, Z)
        
        # Weighted total loss
        # lambda_vector can have multiple components for different fairness metrics
        # Here simplified: lambda_vector[0] weights the adversarial term
        total = task_loss + lambda_vector[0] * fairness_loss
        
        clf_optimizer.zero_grad()
        total.backward()
        clf_optimizer.step()
        
        total_loss += task_loss.item()
    
    return total_loss / len(train_loader)


def evaluate_metrics(
    classifier: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate classifier on test set.
    
    Returns:
        Dict with accuracy and p-rule (fairness metric)
    """
    classifier.eval()
    
    all_preds = []
    all_labels = []
    all_sensitive = []
    
    with torch.no_grad():
        for X, y, Z in test_loader:
            X = X.to(device)
            pred = classifier(X).squeeze()
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.numpy())
            all_sensitive.append(Z.numpy())
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    sensitive = np.concatenate(all_sensitive)
    
    # Binary predictions
    pred_binary = (preds > 0.5).astype(int)
    
    # Accuracy
    accuracy = (pred_binary == labels).mean()
    
    # P-rule (simplified for binary sensitive attribute)
    # P-rule = P(Y=1|Z=0) / P(Y=1|Z=1)
    # Should be close to 1.0 for fairness
    if sensitive.ndim == 1:
        group_0 = pred_binary[sensitive == 0]
        group_1 = pred_binary[sensitive == 1]
    else:
        # If multiple sensitive attributes, take first
        group_0 = pred_binary[sensitive[:, 0] == 0]
        group_1 = pred_binary[sensitive[:, 0] == 1]
    
    rate_0 = group_0.mean() if len(group_0) > 0 else 0.5
    rate_1 = group_1.mean() if len(group_1) > 0 else 0.5
    
    p_rule = min(rate_0 / (rate_1 + 1e-10), rate_1 / (rate_0 + 1e-10))
    
    return {
        "accuracy": float(accuracy),
        "p_rule": float(p_rule)
    }


# ============================================================================
# MAIN TRAINING LOOP WITH LANGCHAIN ORCHESTRATOR
# ============================================================================

def train_with_orchestrator(
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_features: int,
    n_sensitive: int,
    max_iterations: int = 10,
    epochs_per_iteration: int = 10,
    initial_lambda: List[float] = None,
    initial_adv_steps: int = 3,
    device: torch.device = None,
    memory_file: str = "orchestrator_memory.json",
    target_p_rule: float = 0.80,
    target_accuracy: float = 0.80
):
    """
    Main training loop with LangChain orchestrator integration.
    
    Args:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        n_features: Number of input features
        n_sensitive: Number of sensitive attributes
        max_iterations: Maximum outer iterations
        epochs_per_iteration: Epochs per iteration before consulting orchestrator
        initial_lambda: Initial lambda vector
        initial_adv_steps: Initial adversary steps
        device: torch.device
        memory_file: Path to experience log
        target_p_rule: Target p-rule threshold
        target_accuracy: Target accuracy threshold
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if initial_lambda is None:
        initial_lambda = [0.5, 0.3, 0.2]  # Default weights
    
    print("\n" + "=" * 70)
    print("ADVERSARIAL FAIRNESS TRAINING WITH LANGCHAIN ORCHESTRATOR")
    print("=" * 70)
    
    # ========== 1. Initialize Models ==========
    classifier = MockClassifier(n_features).to(device)
    adversary = MockAdversary(n_sensitive).to(device)
    
    clf_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    adv_optimizer = optim.Adam(adversary.parameters(), lr=1e-3)
    
    print(f"\n[OK] Models initialized")
    print(f"   Device: {device}")
    print(f"   Classifier params: {sum(p.numel() for p in classifier.parameters())}")
    print(f"   Adversary params: {sum(p.numel() for p in adversary.parameters())}")
    
    # ========== 2. Initialize LangChain Orchestrator ==========
    orchestrator = create_orchestrator(
        memory_file=memory_file,
        target_p_rule=target_p_rule,
        target_accuracy=target_accuracy,
        verbose=True
    )
    
    # Reset short-term memory for this run
    orchestrator.reset_short_term_memory()
    
    # Current hyperparameters
    current_lambda = initial_lambda.copy()
    current_adv_steps = initial_adv_steps
    
    print(f"\n[CONFIG] Initial configuration:")
    print(f"   Lambda: {current_lambda}")
    print(f"   Adversary steps: {current_adv_steps}")
    
    # ========== 3. Training Loop ==========
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    initial_metrics = None
    
    for iteration in range(max_iterations):
        print(f"\n{'-' * 70}")
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print(f"{'-' * 70}")
        
        # Train for multiple epochs before consulting orchestrator
        epoch_losses = []
        
        for epoch in range(epochs_per_iteration):
            # ===== Step 1: Train Adversary =====
            adv_loss = train_adversary_step(
                adversary=adversary,
                classifier=classifier,
                train_loader=train_loader,
                adv_optimizer=adv_optimizer,
                n_steps=current_adv_steps,
                device=device
            )
            
            # ===== Step 2: Train Learner =====
            task_loss = train_learner_step(
                classifier=classifier,
                adversary=adversary,
                train_loader=train_loader,
                clf_optimizer=clf_optimizer,
                lambda_vector=current_lambda,
                device=device
            )
            
            epoch_losses.append((task_loss, adv_loss))
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{epochs_per_iteration}: "
                      f"Task Loss={task_loss:.4f}, Adv Loss={adv_loss:.4f}")
        
        # ===== Step 3: Evaluate Metrics =====
        metrics = evaluate_metrics(classifier, test_loader, device)
        
        # Add loss to metrics
        avg_task_loss = np.mean([l[0] for l in epoch_losses])
        avg_adv_loss = np.mean([l[1] for l in epoch_losses])
        
        current_state = {
            "epoch": iteration + 1,
            "loss_task": avg_task_loss,
            "loss_adv": avg_adv_loss,
            "p_rule": metrics["p_rule"],
            "accuracy": metrics["accuracy"]
        }
        
        print(f"\n[METRICS] After iteration {iteration + 1}:")
        print(f"   Accuracy: {metrics['accuracy']:.4f} (target: >={target_accuracy:.2f})")
        print(f"   P-rule:   {metrics['p_rule']:.4f} (target: >={target_p_rule:.2f})")
        print(f"   Task Loss: {avg_task_loss:.4f}")
        print(f"   Adv Loss:  {avg_adv_loss:.4f}")
        
        # Store initial metrics
        if initial_metrics is None:
            initial_metrics = current_state.copy()
        
        # ===== Step 4: Consult LangChain Orchestrator =====
        decision = orchestrator.decide(
            current_state=current_state,
            current_lambda=current_lambda,
            current_adv_steps=current_adv_steps
        )
        
        # ===== Step 5: Apply Orchestrator's Decision =====
        current_lambda = decision.lambda_vector
        current_adv_steps = decision.adv_steps
        
        print(f"\n[UPDATE] Applied new configuration:")
        print(f"   Lambda: {current_lambda}")
        print(f"   Adversary steps: {current_adv_steps}")
        
        # ===== Step 6: Check Stopping Criteria =====
        if decision.stop:
            print(f"\n[STOP] TRAINING STOPPED BY ORCHESTRATOR")
            print(f"   Reason: {decision.reason}")
            
            # Record successful convergence
            orchestrator.record_experience(
                iteration=iteration + 1,
                initial_metrics=initial_metrics,
                final_metrics=current_state,
                lambda_vector=current_lambda,
                adv_steps=current_adv_steps,
                converged=True
            )
            break
    else:
        # Max iterations reached without convergence
        print(f"\n[WARNING] Max iterations reached without full convergence")
        orchestrator.record_experience(
            iteration=max_iterations,
            initial_metrics=initial_metrics,
            final_metrics=current_state,
            lambda_vector=current_lambda,
            adv_steps=current_adv_steps,
            converged=False
        )
    
    # ========== 4. Final Evaluation ==========
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    final_metrics = evaluate_metrics(classifier, test_loader, device)
    
    print(f"\n[RESULTS] Final Performance:")
    print(f"   Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"   P-rule:   {final_metrics['p_rule']:.4f}")
    print(f"   Iterations completed: {min(iteration + 1, max_iterations)}")
    
    # Success check
    success = (
        final_metrics['accuracy'] >= target_accuracy and 
        final_metrics['p_rule'] >= target_p_rule
    )
    
    if success:
        print(f"\n[SUCCESS] Both objectives achieved!")
    else:
        print(f"\n[WARNING] Targets not fully met. Consider:")
        if final_metrics['accuracy'] < target_accuracy:
            print(f"   - Accuracy below target: try reducing lambda weights")
        if final_metrics['p_rule'] < target_p_rule:
            print(f"   - P-rule below target: try increasing lambda weights or more iterations")
    
    print("\n" + "=" * 70)
    
    return classifier, adversary, final_metrics


# ============================================================================
# DEMO USAGE
# ============================================================================

if __name__ == "__main__":
    # Generate synthetic data for demo
    print("Generating synthetic data...")
    
    n_samples = 1000
    n_features = 20
    n_sensitive = 2
    batch_size = 64
    
    # Create fake train/test data
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
    
    print(f"[OK] Data ready: {len(train_dataset)} train, {len(test_dataset)} test\n")
    
    # Run training with orchestrator
    classifier, adversary, metrics = train_with_orchestrator(
        train_loader=train_loader,
        test_loader=test_loader,
        n_features=n_features,
        n_sensitive=n_sensitive,
        max_iterations=5,
        epochs_per_iteration=2,  # Reduced for faster testing
        initial_lambda=[0.5, 0.3, 0.2],
        initial_adv_steps=3,
        memory_file="demo_memory.json",
        target_p_rule=0.75,
        target_accuracy=0.70
    )
    
    print("\n[OK] Demo completed!")
