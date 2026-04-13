#!/usr/bin/env python3
"""
IMPROVED TRAINING: Use smart FALLBACK ORCHESTRATOR (no LLM dependency)
The LLM was causing timeouts and instability. This version uses a rule-based orchestrator instead.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
import json

warnings.filterwarnings('ignore')

# ============================================================================
# MODELS FROM CODE_FINAL_COMPLETE
# ============================================================================

class Classifier(nn.Module):
    """Neural network for binary classification."""
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
    """Adversarial network."""
    def __init__(self, n_sensitive: int, n_hidden: int = 32):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))


class SmartOrchestrator:
    """Smart rule-based orchestrator (no LLM)."""
    
    def __init__(self):
        self.history = []
        self.best_state = {"accuracy": 0.0, "p_rule": 0.0, "lambda": [0.5, 0.5]}
    
    def decide(self, step, accuracy, p_rule_values, current_lambda, target_accuracy=0.80, target_p_rule=0.80):
        """Make a decision based on rules."""
        
        min_p_rule = min(p_rule_values.values()) if p_rule_values else 0.5
        
        # Record history
        self.history.append({
            "step": step,
            "accuracy": accuracy,
            "p_rule": min_p_rule,
            "lambda": list(current_lambda)
        })
        
        # Converged
        if accuracy >= target_accuracy and min_p_rule >= target_p_rule:
            return list(current_lambda), "CONVERGED"
        
        # Decision rules
        adjustment = 0.0
        reasoning = ""
        
        if min_p_rule < 0.70:
            # P-rule critically low - boost fairness aggressively
            adjustment = 0.5  # Increase lambda
            reasoning = "P-rule critically low, increasing fairness weight"
        elif min_p_rule < 0.80:
            # P-rule improving but below target - moderate increase
            adjustment = 0.2
            reasoning = "P-rule below target, moderate increase"
        elif accuracy < target_accuracy:
            # Prioritize accuracy
            adjustment = -0.3
            reasoning = "Accuracy below target, prioritizing performance"
        else:
            # Both close to target - fine-tune with momentum
            adjustment = 0.05
            reasoning = "Fine-tuning both metrics"
        
        # Apply adjustment with momentum damping
        new_lambda = [max(0.0, min(curr + adjustment * 0.7, 2.0)) for curr in current_lambda]
        
        return new_lambda, reasoning


def load_data():
    """Load Adult dataset."""
    try:
        data = np.loadtxt('adult.data', dtype=str, delimiter=',')
        X = data[:, :5].astype(float)
        y = (data[:, 5] == ' >50K').astype(int)
        sensitive = np.column_stack([
            (data[:, 6] == ' White').astype(int),
            (data[:, 7] == ' Male').astype(int)
        ])
        
        n_train = int(0.8 * len(X))
        return (X[:n_train], y[:n_train], sensitive[:n_train]), \
               (X[n_train:], y[n_train:], sensitive[n_train:])
    except:
        print("[ERROR] adult.data not found")
        return None, None


def train():
    """Run 100-iteration training with smart orchestrator."""
    
    print("=" * 60)
    print("ADVERSARIAL FAIRNESS TRAINING - SMART ORCHESTRATOR")
    print("=" * 60)
    
    # Load data
    (X_train, y_train, s_train), (X_test, y_test, s_test) = load_data()
    if X_train is None:
        return
    
    print(f"\n[DATASET] Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    device = torch.device('cpu')
    n_features = X_train.shape[1]
    
    # Initialize models
    classifier = Classifier(n_features=n_features).to(device)
    adversary = Adversary(n_sensitive=2).to(device)
    
    print(f"[MODELS] Classifier params: {sum(p.numel() for p in classifier.parameters())}")
    print(f"         Adversary params: {sum(p.numel() for p in adversary.parameters())}")
    
    # Optimizers
    opt_clf = optim.Adam(classifier.parameters(), lr=0.01)
    opt_adv = optim.Adam(adversary.parameters(), lr=0.01)
    
    # Orchestrator
    orchestrator = SmartOrchestrator()
    
    # Data
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().view(-1, 1).to(device)
    s_train_t = torch.from_numpy(s_train).float().to(device)
    
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().view(-1, 1).to(device)
    s_test_t = torch.from_numpy(s_test).float().to(device)
    
    metrics = []
    current_lambda = [0.5, 0.5]
    
    print(f"\n[TRAINING] Starting 100 iterations...")
    print("-" * 60)
    
    for iteration in range(1, 101):
        # Train adversary
        for _ in range(2):
            classifier.eval()
            y_pred = classifier(X_train_t)
            adversary.train()
            s_pred = adversary(y_pred)
            adv_loss = nn.BCELoss()(s_pred, s_train_t)
            opt_adv.zero_grad()
            adv_loss.backward()
            opt_adv.step()
        
        # Train classifier
        for _ in range(2):
            adversary.eval()
            classifier.train()
            y_pred = classifier(X_train_t)
            clf_loss = nn.BCELoss()(y_pred, y_train_t)
            s_pred = adversary(y_pred)
            fair_loss = nn.BCELoss()(s_pred, s_train_t)
            total_loss = current_lambda[0] * clf_loss + current_lambda[1] * fair_loss
            opt_clf.zero_grad()
            total_loss.backward()
            opt_clf.step()
        
        # Evaluate
        classifier.eval()
        with torch.no_grad():
            y_pred = (classifier(X_test_t) > 0.5).cpu().numpy().flatten()
            accuracy = accuracy_score(y_test.flatten(), y_pred)
            
            # P-rule per attribute
            p_rules = {}
            for attr_idx, attr_name in enumerate(['race', 'sex']):
                group_0 = y_pred[s_test[:, attr_idx] == 0]
                group_1 = y_pred[s_test[:, attr_idx] == 1]
                if len(group_0) > 0 and len(group_1) > 0:
                    p_rules[attr_name] = min(
                        group_1.mean() / (group_0.mean() + 1e-10),
                        group_0.mean() / (group_1.mean() + 1e-10)
                    )
                else:
                    p_rules[attr_name] = 0.5
            
            min_p_rule = min(p_rules.values())
        
        # Orchestrator decision
        new_lambda, reasoning = orchestrator.decide(
            iteration, accuracy, p_rules, current_lambda
        )
        current_lambda = new_lambda
        
        # Record metrics
        metrics.append({
            "step": iteration,
            "accuracy": accuracy,
            "min_p_rule": min_p_rule,
            "p_rules": p_rules
        })
        
        # Print
        status = ""
        if accuracy >= 0.80:
            status += "Acc[OK]"
        else:
            status += f"Acc[{accuracy:.2f}]"
        
        if min_p_rule >= 0.80:
            status += " PRule[OK]"
        else:
            status += f" PRule[{min_p_rule:.2f}]"
        
        print(f"Step {iteration:3d}/100: {status}  λ={[f'{x:.2f}' for x in current_lambda]}")
        
        # Check convergence
        if accuracy >= 0.80 and min_p_rule >= 0.80:
            print(f"\n[SUCCESS] CONVERGED at step {iteration}!")
            break
    
    # Save metrics
    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"[FINAL] Accuracy: {metrics[-1]['accuracy']:.4f}, P-rule: {metrics[-1]['min_p_rule']:.4f}")
    print(f"[SAVED] Results in training_metrics.json")
    
    return metrics


if __name__ == "__main__":
    train()
