"""
COMPLETE ADVERSARIAL FAIRNESS TRAINING WITH LLM ORCHESTRATOR
==============================================================

Uses:
- Classifier & Adversary from CODE_FINAL_COMPLETE
- LLM-based orchestrator (Llama3.1 via Ollama + LangChain)
- Adaptive lambda learning with momentum
- Target: Accuracy >= 80% AND P-rule >= 80%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Import LLM orchestrator
from orchestrator_llm import LLMOrchestrator

# ============================================================================
# MODELS FROM CODE_FINAL_COMPLETE
# ============================================================================

class Classifier(nn.Module):
    """Neural network for binary classification (from CODE_FINAL_COMPLETE)."""
    def __init__(self, n_features: int, n_hidden: int = 32, p_dropout: float = 0.2):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(p_dropout),
            nn.Linear(n_hidden, 1),
        )
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))


class Adversary(nn.Module):
    """Adversarial network to predict sensitive attributes from predictions (from CODE_FINAL_COMPLETE)."""
    def __init__(self, n_sensitive: int, n_hidden: int = 32):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# DATA HANDLING
# ============================================================================

def load_adult_data(filepath: str = "adult.data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                            np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Adult dataset and prepare for training.
    """
    import pandas as pd
    
    print(f"Loading dataset from {filepath}...")
    
    # Column names for Adult dataset
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    
    df = pd.read_csv(filepath, names=columns, sep=',', skipinitialspace=True)
    
    # Target: income >50K
    y = (df['income'] == '>50K').astype(int).values
    
    # Sensitive attributes
    Z_race = (df['race'] != 'White').astype(int).values  # 1 if non-white
    Z_sex = (df['sex'] != 'Male').astype(int).values     # 1 if female
    Z = np.column_stack([Z_race, Z_sex])
    
    # Features: numerical only for simplicity
    X = df[['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].values
    
    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split].astype(np.float32), X[split:].astype(np.float32)
    y_train, y_test = y[:split].astype(np.float32), y[split:].astype(np.float32)
    Z_train, Z_test = Z[:split].astype(np.float32), Z[split:].astype(np.float32)
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Sensitive attributes: 2 (race, sex)")
    
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
    """Train classifier for 1 epoch with fairness penalty."""
    classifier.train()
    adversary.eval()  # Frozen
    
    task_criterion = nn.BCELoss()
    adv_criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    total_loss = 0.0
    
    for X, y, Z in train_loader:
        X, y, Z = X.to(device), y.to(device), Z.to(device)
        
        # Classifier prediction
        y_pred = classifier(X)
        
        # Task loss
        L_task = task_criterion(y_pred, y.unsqueeze(1))
        
        # Adversary loss (per sample per attribute)
        Z_pred = adversary(y_pred)
        L_adv_individual = adv_criterion(Z_pred, Z)  # Shape: (batch, n_attrs)
        
        # Weighted fairness loss
        L_fair = 0.0
        for i in range(Z.shape[1]):
            L_fair += lambda_vec[i] * L_adv_individual[:, i].mean()
        
        # Total loss: maximize task performance while minimizing fairness violation
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
    """Train adversary for 1 epoch to detect sensitive attributes."""
    adversary.train()
    classifier.eval()  # Frozen
    
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    
    for X, y, Z in train_loader:
        X, Z = X.to(device), Z.to(device)
        
        # Get classifier predictions
        with torch.no_grad():
            y_pred = classifier(X)
        
        # Adversary tries to predict sensitive attributes from predictions
        Z_pred = adversary(y_pred)
        
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
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Evaluate classifier accuracy and compute p-rule fairness metrics."""
    classifier.eval()
    
    y_true_all = []
    y_pred_all = []
    Z_true_all = []
    
    with torch.no_grad():
        # Collect all predictions
        for X, y, Z in test_loader:
            X, y, Z = X.to(device), y.to(device), Z.to(device)
            
            y_prob = classifier(X).squeeze()
            
            y_true_all.extend(y.cpu().numpy())
            y_pred_all.extend((y_prob > 0.5).cpu().numpy())
            Z_true_all.extend(Z.cpu().numpy())
    
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    Z_true = np.array(Z_true_all)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # P-rule per attribute
    p_rules = {}
    attr_names = ['race', 'sex']
    
    for attr_idx in range(Z_true.shape[1]):
        mask0 = Z_true[:, attr_idx] == 0
        mask1 = Z_true[:, attr_idx] == 1
        
        if mask0.sum() > 0 and mask1.sum() > 0:
            p_y0 = y_pred[mask0].mean()
            p_y1 = y_pred[mask1].mean()
            p_rule = min(p_y0 / (p_y1 + 1e-6), p_y1 / (p_y0 + 1e-6))
        else:
            p_rule = 1.0
        
        attr_name = attr_names[attr_idx] if attr_idx < len(attr_names) else f'attr_{attr_idx}'
        p_rules[attr_name] = p_rule
    
    return accuracy, p_rules


# ============================================================================
# MAIN TRAINING WITH LLM ORCHESTRATOR
# ============================================================================

def train_with_llm_orchestrator(
    X_train, y_train, Z_train,
    X_test, y_test, Z_test,
    n_features: int = 5,
    n_sensitive_attrs: int = 2,
    epochs_init_clf: int = 10,
    epochs_init_adv: int = 10,
    epochs_per_step: int = 5,
    max_steps: int = 100,
    batch_size: int = 32,
    llm_model: str = "llama3.1",
    device: torch.device = torch.device('cpu')
):
    """
    Full training loop with LLM-based orchestrator.
    
    Phase 1: Pre-train classifier
    Phase 2: Pre-train adversary
    Phase 3: Adaptive adversarial loop with LLM decisions
    """
    
    print("\n" + "="*70)
    print("ADVERSARIAL FAIRNESS TRAINING WITH LLM ORCHESTRATOR")
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
    classifier = Classifier(n_features=n_features).to(device)
    adversary = Adversary(n_sensitive=n_sensitive_attrs).to(device)
    
    clf_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    adv_optimizer = optim.Adam(adversary.parameters(), lr=0.001)
    
    # Initialize lambda and momentum
    lambda_vec = [0.5] * n_sensitive_attrs
    momentum_vec = [0.0] * n_sensitive_attrs
    
    print(f"\n[INIT] Models initialized")
    print(f"  Classifier params: {sum(p.numel() for p in classifier.parameters())}")
    print(f"  Adversary params: {sum(p.numel() for p in adversary.parameters())}")
    print(f"  Device: {device}")
    print(f"  Initial Lambda: {lambda_vec}")
    print(f"  Target: Accuracy >= 0.80, P-rule >= 0.80")
    
    # Initialize LLM orchestrator
    print(f"\n[LLM-ORCHESTRATOR] Initializing {llm_model}...")
    try:
        orchestrator = LLMOrchestrator(model_name=llm_model, temperature=0.3)
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM orchestrator: {e}")
        print("Make sure Ollama is running: ollama serve")
        print("And model is available: ollama pull llama3.1")
        raise
    
    # ========================================================================
    # PHASE 1: Pre-train classifier
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"[PHASE 1] Pre-training classifier ({epochs_init_clf} epochs)")
    print(f"{'='*70}")
    for epoch in range(epochs_init_clf):
        loss = train_classifier_epoch(classifier, adversary, train_loader, clf_optimizer, lambda_vec, device)
        if (epoch + 1) % 5 == 0:
            acc, p_rules = evaluate(classifier, train_loader, test_loader, device)
            print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}, P-rules={p_rules}")
    print(f"[OK] Classifier pre-trained")
    
    # ========================================================================
    # PHASE 2: Pre-train adversary
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"[PHASE 2] Pre-training adversary ({epochs_init_adv} epochs)")
    print(f"{'='*70}")
    for epoch in range(epochs_init_adv):
        loss = train_adversary_epoch(adversary, classifier, train_loader, adv_optimizer, device)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Adv Loss={loss:.4f}")
    print(f"[OK] Adversary pre-trained")
    
    # ========================================================================
    # PHASE 3: Adaptive adversarial loop with LLM orchestrator
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"[PHASE 3] Adversarial training with LLM orchestrator")
    print(f"{'='*70}")
    print(f"Running up to {max_steps} steps, {epochs_per_step} epochs per step")
    print(f"Target: Accuracy >= 0.80, Min P-rule >= 0.80")
    
    best_accuracy = 0.0
    best_p_rule = 0.0
    best_metrics = None
    metrics_history = []  # Store metrics for plotting
    
    for step in range(max_steps):
        print(f"\n{'-'*70}")
        print(f"STEP {step + 1}/{max_steps}")
        print(f"{'-'*70}")
        
        # Train for epochs_per_step
        for epoch in range(epochs_per_step):
            train_adversary_epoch(adversary, classifier, train_loader, adv_optimizer, device)
            train_classifier_epoch(classifier, adversary, train_loader, clf_optimizer, lambda_vec, device)
        
        # Evaluate
        accuracy, p_rules = evaluate(classifier, train_loader, test_loader, device)
        min_p_rule = min(p_rules.values())
        
        print(f"\n[METRICS]")
        print(f"  Accuracy: {accuracy:.4f} (target: 0.80)")
        print(f"  P-rules: {p_rules}")
        print(f"  Min P-rule: {min_p_rule:.4f} (target: 0.80)")
        
        # Save metrics for plotting
        metrics_history.append({
            'step': step + 1,
            'accuracy': float(accuracy),
            'min_p_rule': float(min_p_rule),
            'p_rules': {k: float(v) for k, v in p_rules.items()}
        })
        
        # LLM ORCHESTRATOR DECISION
        print(f"\n[ORCHESTRATION]")
        try:
            lambda_vec, momentum_vec, reasoning = orchestrator.decide(
                step=step,
                accuracy=accuracy,
                p_rule_values=p_rules,
                current_lambda=lambda_vec,
                current_momentum=momentum_vec,
                target_p_rule=0.80,
                target_accuracy=0.80,
                lambda_max=20.0
            )
        except Exception as e:
            print(f"[WARN]  LLM orchestrator error: {e}")
            print("Continuing with fallback strategy...")
        
        best_metrics = (accuracy, p_rules)
        best_accuracy = max(best_accuracy, accuracy)
        best_p_rule = max(best_p_rule, min_p_rule)
        
        # Check if both targets reached (but continue to 100 iterations)
        if accuracy >= 0.80 and min_p_rule >= 0.80:
            print(f"\n✅ TARGETS MET at step {step+1}!")
            print(f"   Accuracy: {accuracy:.4f} (target: 0.80)")
            print(f"   P-rule: {min_p_rule:.4f} (target: 0.80)")
            print(f"   (Continuing to complete 100 iterations...)")
    
    # Save metrics to file for plotting
    metrics_file = "training_metrics.json"
    import json
    with open(metrics_file, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    print(f"\n[SAVED] Training metrics saved to: {metrics_file}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    if best_metrics:
        acc, p_rules = best_metrics
        min_p_rule = min(p_rules.values())
        print(f"Best Accuracy: {best_accuracy:.4f} (target: 0.80)")
        print(f"Best P-rule: {best_p_rule:.4f} (target: 0.80)")
        print(f"Final P-rules: {p_rules}")
    
    return classifier, adversary, best_metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Load Adult dataset
    try:
        X_train, y_train, Z_train, X_test, y_test, Z_test = load_adult_data()
    except FileNotFoundError:
        print("[ERROR] adult.data not found!")
        print("Generating synthetic data instead...")
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=5, n_informative=5, 
                                   n_redundant=0, random_state=42)
        Z = np.random.binomial(1, 0.5, (X.shape[0], 2))
        
        split = int(0.8 * len(X))
        X_train, X_test = X[:split].astype(np.float32), X[split:].astype(np.float32)
        y_train, y_test = y[:split].astype(np.float32), y[split:].astype(np.float32)
        Z_train, Z_test = Z[:split].astype(np.float32), Z[split:].astype(np.float32)
        
        print(f"[OK] Synthetic data generated: {X_train.shape}")
    
    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        classifier, adversary, metrics = train_with_llm_orchestrator(
            X_train, y_train, Z_train,
            X_test, y_test, Z_test,
            n_features=X_train.shape[1],
            n_sensitive_attrs=Z_train.shape[1],
            epochs_init_clf=10,
            epochs_init_adv=10,
            epochs_per_step=5,
            max_steps=100,
            batch_size=32,
            llm_model="llama3.1",
            device=device
        )
        print("\n✅ Training completed!")
    except KeyboardInterrupt:
        print("\n[WARN]  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
