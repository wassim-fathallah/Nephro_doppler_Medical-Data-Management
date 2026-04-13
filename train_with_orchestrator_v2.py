"""
COMPLETE ADVERSARIAL FAIRNESS TRAINING WITH ORCHESTRATOR
==========================================================

Using Classifier and Adversary from integration_example.py
With OrchestratorAdaptive for smart lambda decisions
Target: Accuracy >= 80% AND P-rule >= 80%

This script:
1. Pre-trains classifier (Phase 1)
2. Pre-trains adversary (Phase 2)
3. Runs adaptive adversarial loop with orchestrator guidance (Phase 3)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# Import orchestrator
from orchestrator_adaptive import OrchestratorAdaptive

# ============================================================================
# MODELS (From integration_example.py)
# ============================================================================

class Classifier(nn.Module):
    """Classifier that tries to predict target while hiding sensitive attributes."""
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


class Adversary(nn.Module):
    """Adversary that tries to detect sensitive attributes from classifier's hidden layer."""
    def __init__(self, n_sensitive_attrs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 16),  # Takes classifier's hidden layer (32-dim)
            nn.ReLU(),
            nn.Linear(16, n_sensitive_attrs)
        )
    
    def forward(self, hidden):
        return self.net(hidden)


# ============================================================================
# DATA GENERATION (Synthetic fair classification task)
# ============================================================================

def generate_synthetic_data(n_samples: int = 1000, n_features: int = 20):
    """
    Generate synthetic data for fair classification.
    
    Returns:
        X_train, y_train, Z_train: Training features, labels, sensitive attributes
        X_test, y_test, Z_test: Test features, labels, sensitive attributes
    """
    np.random.seed(42)
    
    # Features
    X = np.random.randn(n_samples, n_features)
    
    # Two sensitive attributes
    Z1 = np.random.binomial(1, 0.5, n_samples)  # e.g., race (0/1)
    Z2 = np.random.binomial(1, 0.5, n_samples)  # e.g., gender (0/1)
    Z = np.column_stack([Z1, Z2])
    
    # Target: influenced by features and sensitive attributes
    y = (X[:, 0] + X[:, 1] + 0.2 * Z1 + 0.1 * Z2 > 0).astype(int)
    
    # Train/test split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split].astype(np.float32), X[split:].astype(np.float32)
    y_train, y_test = y[:split].astype(np.float32), y[split:].astype(np.float32)
    Z_train, Z_test = Z[:split].astype(np.float32), Z[split:].astype(np.float32)
    
    return X_train, y_train, Z_train, X_test, y_test, Z_test


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_classifier_epoch(
    classifier: nn.Module,
    adversary: nn.Module,
    train_loader: DataLoader,
    clf_optimizer: optim.Optimizer,
    lambda_vec: List[float],
    device: torch.device
) -> float:
    """
    Train classifier for 1 epoch.
    Loss = L_task - Σ(λᵢ * L_adv_individual)
    """
    classifier.train()
    adversary.eval()  # Frozen
    
    task_criterion = nn.BCELoss()
    adv_criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    total_loss = 0.0
    
    for X, y, Z in train_loader:
        X, y, Z = X.to(device), y.to(device), Z.to(device)
        
        # Forward
        hidden = classifier.extract_features(X)
        y_pred = classifier.classifier_head(hidden)
        
        # Task loss
        L_task = task_criterion(y_pred, y.unsqueeze(1))
        
        # Adversary loss (per sample)
        Z_pred = adversary(hidden)
        L_adv_individual = adv_criterion(Z_pred, Z)  # Shape: (batch, n_attrs)
        
        # Weighted fairness loss
        L_fair = 0.0
        for i in range(Z.shape[1]):
            L_fair += lambda_vec[i] * L_adv_individual[:, i].mean()
        
        # Total loss
        loss = L_task - L_fair
        
        # Backward
        clf_optimizer.zero_grad()
        loss.backward()
        clf_optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def train_adversary_epoch(
    adversary: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    adv_optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train adversary for 1 epoch.
    Tries to predict sensitive attributes from classifier's hidden layer.
    """
    adversary.train()
    classifier.eval()  # Frozen
    
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    
    for X, y, Z in train_loader:
        X, Z = X.to(device), Z.to(device)
        
        # Get hidden representation
        with torch.no_grad():
            hidden = classifier.extract_features(X)
        
        # Predict sensitive attributes
        Z_pred = adversary(hidden)
        
        # Loss
        loss = criterion(Z_pred, Z)
        
        # Backward
        adv_optimizer.zero_grad()
        loss.backward()
        adv_optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(
    classifier: nn.Module,
    adversary: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate classifier and compute fairness metrics.
    
    Returns:
        Dict with accuracy, p-rule for each attribute
    """
    classifier.eval()
    adversary.eval()
    
    y_true = []
    y_pred = []
    Z_true = []
    Z_pred_adv = []
    
    with torch.no_grad():
        for X, y, Z in test_loader:
            X, y, Z = X.to(device), y.to(device), Z.to(device)
            
            # Predictions
            hidden = classifier.extract_features(X)
            y_prob = classifier.classifier_head(hidden).squeeze()
            Z_pred = adversary(hidden)
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend((y_prob > 0.5).cpu().numpy())
            Z_true.extend(Z.cpu().numpy())
            Z_pred_adv.extend(Z_pred.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    Z_true = np.array(Z_true)
    Z_pred_adv = np.array(Z_pred_adv)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # P-rule per attribute
    p_rules = {}
    for attr_idx in range(Z_true.shape[1]):
        # P-rule = min(P(y=1|Z=0)/P(y=1|Z=1), P(y=1|Z=1)/P(y=1|Z=0))
        mask0 = Z_true[:, attr_idx] == 0
        mask1 = Z_true[:, attr_idx] == 1
        
        if mask0.sum() > 0 and mask1.sum() > 0:
            p_y0 = y_pred[mask0].mean()
            p_y1 = y_pred[mask1].mean()
            p_rule = min(p_y0 / (p_y1 + 1e-6), p_y1 / (p_y0 + 1e-6))
        else:
            p_rule = 1.0
        
        p_rules[f'attr_{attr_idx}'] = p_rule
    
    return {
        'accuracy': accuracy,
        'p_rules': p_rules
    }


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_with_orchestrator(
    X_train, y_train, Z_train,
    X_test, y_test, Z_test,
    epochs_phase1: int = 10,
    epochs_phase2: int = 10,
    epochs_per_step: int = 5,
    max_steps: int = 100,
    batch_size: int = 32,
    device: torch.device = torch.device('cpu')
):
    """
    Full training with orchestrator-guided lambda adaptation.
    
    Phase 1: Pre-train classifier (no adversary)
    Phase 2: Pre-train adversary (frozen classifier)
    Phase 3: Adversarial loop with orchestrator decisions
    """
    
    print("\n" + "="*70)
    print("ADVERSARIAL FAIRNESS TRAINING WITH ORCHESTRATOR")
    print("="*70)
    
    # Prepare data
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.FloatTensor(Z_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test),
        torch.FloatTensor(Z_test)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize models
    n_features = X_train.shape[1]
    n_attrs = Z_train.shape[1]
    
    classifier = Classifier(n_features).to(device)
    adversary = Adversary(n_attrs).to(device)
    
    clf_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    adv_optimizer = optim.Adam(adversary.parameters(), lr=0.001)
    
    # Initialize lambda and momentum
    lambda_vec = [0.5] * n_attrs
    momentum_vec = [0.0] * n_attrs
    
    print(f"\n[INIT] Models initialized")
    print(f"  Classifier params: {sum(p.numel() for p in classifier.parameters())}")
    print(f"  Adversary params: {sum(p.numel() for p in adversary.parameters())}")
    print(f"  Device: {device}")
    print(f"  Lambda: {lambda_vec}")
    print(f"  Target: Accuracy >= 0.80, P-rule >= 0.80")
    
    # Initialize orchestrator
    orchestrator = OrchestratorAdaptive(memory_file="orchestrator_training.json")
    
    # ========================================================================
    # PHASE 1: Pre-train classifier
    # ========================================================================
    print(f"\n[PHASE 1] Pre-training classifier ({epochs_phase1} epochs)")
    for epoch in range(epochs_phase1):
        loss = train_classifier_epoch(classifier, adversary, train_loader, clf_optimizer, lambda_vec, device)
    print(f"  ✓ Classifier pre-trained")
    
    # ========================================================================
    # PHASE 2: Pre-train adversary
    # ========================================================================
    print(f"\n[PHASE 2] Pre-training adversary ({epochs_phase2} epochs)")
    for epoch in range(epochs_phase2):
        loss = train_adversary_epoch(adversary, classifier, train_loader, adv_optimizer, device)
    print(f"  ✓ Adversary pre-trained")
    
    # ========================================================================
    # PHASE 3: Adaptive adversarial loop with orchestrator
    # ========================================================================
    print(f"\n[PHASE 3] Adversarial training with orchestrator ({max_steps} steps max, {epochs_per_step} epochs/step)")
    print("-" * 70)
    
    best_metrics = None
    
    for step in range(max_steps):
        # Train adversary and classifier
        for epoch in range(epochs_per_step):
            train_adversary_epoch(adversary, classifier, train_loader, adv_optimizer, device)
            train_classifier_epoch(classifier, adversary, train_loader, clf_optimizer, lambda_vec, device)
        
        # Evaluate
        metrics = evaluate(classifier, adversary, test_loader, device)
        accuracy = metrics['accuracy']
        p_rules = metrics['p_rules']
        
        print(f"\n[STEP {step+1}] Accuracy: {accuracy:.4f}, P-rules: {p_rules}")
        
        # ORCHESTRATOR DECISION
        lambda_vec, momentum_vec, reason = orchestrator.update_lambda(
            step=step,
            accuracy=accuracy,
            p_rule_values=p_rules,
            current_lambda=lambda_vec,
            current_momentum=momentum_vec,
            target_p_rule=0.80,
            target_accuracy=0.80,
            lambda_learning_rate=2.5,
            lambda_max=20.0
        )
        
        print(f"  Reason: {reason}")
        
        # Check convergence
        if accuracy >= 0.80 and min(p_rules.values()) >= 0.80:
            print(f"\n✓ CONVERGED at step {step+1}!")
            print(f"  Final Accuracy: {accuracy:.4f}")
            print(f"  Final P-rules: {p_rules}")
            best_metrics = metrics
            break
        
        best_metrics = metrics
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Final Accuracy: {best_metrics['accuracy']:.4f} (target: 0.80)")
    print(f"Final P-rules: {best_metrics['p_rules']} (target: 0.80)")
    
    return classifier, adversary, best_metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Generate data
    print("Generating synthetic data...")
    X_train, y_train, Z_train, X_test, y_test, Z_test = generate_synthetic_data(n_samples=1000)
    
    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier, adversary, metrics = train_with_orchestrator(
        X_train, y_train, Z_train,
        X_test, y_test, Z_test,
        epochs_phase1=10,
        epochs_phase2=10,
        epochs_per_step=5,
        max_steps=100,
        batch_size=32,
        device=device
    )
    
    print("\n✓ Done!")
