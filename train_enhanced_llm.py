"""
COMPLETE ADVERSARIAL FAIRNESS TRAINING WITH ENHANCED LLM ORCHESTRATOR
======================================================================

Uses:
- Classifier & Adversary models from CODE_FINAL_COMPLETE
- Enhanced LLM orchestrator (Llama3.1 with pattern detection)
- 100 iterations to track convergence point
- Target: Accuracy >= 80% AND P-rule >= 80%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
import json
import time

warnings.filterwarnings('ignore')

from orchestrator_enhanced import EnhancedLLMOrchestrator

# ============================================================================
# MODELS FROM CODE_FINAL_COMPLETE
# ============================================================================

class Classifier(nn.Module):
    """Neural network for binary classification with representation output."""
    def __init__(self, n_features: int, n_hidden: int = 64, p_dropout: float = 0.2):
        super(Classifier, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(p_dropout),
        )
        self.output_layer = nn.Linear(n_hidden, 1)
    
    def forward(self, x):
        hidden = self.hidden_layers(x)
        logits = self.output_layer(hidden)
        return torch.sigmoid(logits)
    
    def get_representation(self, x):
        """Get hidden layer representation for adversary."""
        return self.hidden_layers(x)


class Adversary(nn.Module):
    """Adversarial network to predict sensitive attributes from representations."""
    def __init__(self, n_hidden: int = 64, n_sensitive: int = 2):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )
    
    def forward(self, representation):
        """Input is classifier's hidden representation, not final prediction."""
        return torch.sigmoid(self.network(representation))


def load_data():
    """Load Adult dataset with proper normalization."""
    try:
        data = np.loadtxt('adult.data', dtype=str, delimiter=',')
        
        # Extract features (numerical): age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
        X = data[:, [0, 2, 4, 10, 11, 12]].astype(float)
        
        # NORMALIZE features to [0, 1] - CRITICAL for neural networks
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X = (X - X_min) / (X_max - X_min + 1e-8)
        
        # Target: income (<=50K = 0, >50K = 1)
        y = np.array([1 if val.strip() == '>50K' else 0 for val in data[:, 14]]).astype(int)
        
        # Sensitive attributes: race and sex (columns 8, 9)
        race = data[:, 8]
        sex = data[:, 9]
        
        sensitive = np.column_stack([
            np.array([1 if r.strip() == 'White' else 0 for r in race]),
            np.array([1 if s.strip() == 'Male' else 0 for s in sex])
        ]).astype(int)
        
        n_train = int(0.8 * len(X))
        print(f"[DATA] Train: {n_train} samples, Test: {len(X)-n_train} samples")
        print(f"[DATA] Features: {X.shape[1]}, Normalized to [0,1]")
        print(f"[DATA] Positive class rate: {y.mean():.2%}")
        
        return (X[:n_train], y[:n_train], sensitive[:n_train]), \
               (X[n_train:], y[n_train:], sensitive[n_train:])
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compute_p_rule(y_pred, sensitive_attr):
    """Compute P-rule (fairness metric) for each sensitive attribute."""
    p_rules = {}
    for attr_idx, attr_name in enumerate(['race', 'sex']):
        group_0 = y_pred[sensitive_attr[:, attr_idx] == 0]
        group_1 = y_pred[sensitive_attr[:, attr_idx] == 1]
        
        if len(group_0) > 0 and len(group_1) > 0:
            # P-rule = min(P(Y=1|S=0)/P(Y=1|S=1), P(Y=1|S=1)/P(Y=1|S=0))
            rate_0 = group_0.mean()
            rate_1 = group_1.mean()
            p_rule = min(rate_1 / (rate_0 + 1e-10), rate_0 / (rate_1 + 1e-10))
            p_rules[attr_name] = max(0.0, min(1.0, p_rule))
        else:
            p_rules[attr_name] = 0.5
    
    return p_rules


def train_with_enhanced_orchestrator():
    """Run 100-iteration adversarial training with Llama orchestrator."""
    
    print("=" * 70)
    print("ADVERSARIAL FAIRNESS TRAINING - ENHANCED LLM ORCHESTRATOR")
    print("=" * 70)
    print()
    
    # Load data
    (X_train, y_train, s_train), (X_test, y_test, s_test) = load_data()
    if X_train is None:
        return
    
    device = torch.device('cpu')
    n_features = X_train.shape[1]
    
    # Initialize models
    print("\n[MODELS]")
    classifier = Classifier(n_features=n_features).to(device)
    adversary = Adversary(n_sensitive=2).to(device)
    print(f"  Classifier params: {sum(p.numel() for p in classifier.parameters())}")
    print(f"  Adversary params: {sum(p.numel() for p in adversary.parameters())}")
    print(f"  Device: {device}")
    
    # Optimizers
    opt_clf = optim.Adam(classifier.parameters(), lr=0.01)
    opt_adv = optim.Adam(adversary.parameters(), lr=0.01)
    
    # Torch datasets
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().view(-1, 1).to(device)
    s_train_t = torch.from_numpy(s_train).float().to(device)
    
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().view(-1, 1).to(device)
    
    # Initialize orchestrator
    print("\n[ORCHESTRATOR]")
    try:
        orchestrator = EnhancedLLMOrchestrator(model_name="llama3.1", temperature=0.2)
    except Exception as e:
        print(f"[ERROR] Failed to initialize orchestrator: {e}")
        return
    
    # Initial lambda
    current_lambda = [0.5, 0.5]
    
    # Metrics tracking
    metrics = []
    convergence_step = None
    
    print("\n" + "=" * 70)
    print("STARTING 100-ITERATION TRAINING")
    print("=" * 70)
    
    for iteration in range(1, 101):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}/100")
        print(f"{'='*70}")
        
        # PHASE 1: Train Adversary (only, freeze classifier)
        # Goal: Learn to predict S from hidden representations
        classifier.eval()
        for adv_step in range(3):
            with torch.no_grad():
                hidden = classifier.get_representation(X_train_t)
            s_pred = adversary(hidden)
            adv_loss = nn.BCELoss()(s_pred, s_train_t)
            
            opt_adv.zero_grad()
            adv_loss.backward()
            opt_adv.step()
        
        # PHASE 2: Train Classifier (only, freeze adversary)
        # Goal: Predict Y well AND confuse the adversary
        # Strategy: Alternate between task-only and fairness steps
        adversary.eval()
        
        # Sub-phase A: Task-focused steps (MORE emphasis on Y prediction)
        for clf_step in range(4):
            hidden = classifier.get_representation(X_train_t)
            logits = classifier.output_layer(hidden)
            y_pred = torch.sigmoid(logits)
            
            task_loss = nn.BCELoss()(y_pred, y_train_t)
            
            # High emphasis on task accuracy first
            loss = task_loss
            
            opt_clf.zero_grad()
            loss.backward()
            opt_clf.step()
        
        # Sub-phase B: Fairness-focused steps (if task is reasonable)
        current_acc = (y_pred > 0.5).float().mean().item()
        if current_acc >= 0.65:  # Only apply fairness if task is doing ok
            for clf_step in range(2):
                hidden = classifier.get_representation(X_train_t)
                logits = classifier.output_layer(hidden)
                y_pred = torch.sigmoid(logits)
                
                task_loss = nn.BCELoss()(y_pred, y_train_t)
                s_pred = adversary(hidden)
                adv_loss = nn.BCELoss()(s_pred, s_train_t)
                
                # Minimize task + maximize adv confusion (push adv loss UP)
                # Current lambda controls how much we care about fairness
                loss = task_loss - current_lambda[1] * adv_loss
                
                opt_clf.zero_grad()
                loss.backward()
                opt_clf.step()
        
        # PHASE 3: Evaluate on test set
        classifier.eval()
        with torch.no_grad():
            y_pred_prob = classifier(X_test_t).cpu().numpy().flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test.flatten(), y_pred)
            p_rule_values = compute_p_rule(y_pred, s_test)
            min_p_rule = min(p_rule_values.values())
        
        # PHASE 4: Orchestrator decision
        try:
            new_lambda, reasoning = orchestrator.decide(
                step=iteration,
                accuracy=accuracy,
                p_rule_values=p_rule_values,
                current_lambda=current_lambda,
                target_p_rule=0.80,
                target_accuracy=0.80,
                lambda_max=2.0
            )
            current_lambda = new_lambda
        except Exception as e:
            print(f"[ERROR] Orchestrator failed: {e}")
            current_lambda = current_lambda  # Keep same
            reasoning = str(e)
        
        # Record metrics
        metrics.append({
            'step': iteration,
            'accuracy': float(accuracy),
            'min_p_rule': float(min_p_rule),
            'p_rules': {str(k): float(v) for k, v in p_rule_values.items()},
            'lambda': [float(x) for x in current_lambda],
            'reasoning': reasoning
        })
        
        # Print status
        print(f"\n[METRICS]")
        print(f"  Accuracy:  {accuracy:.4f} {'[OK]' if accuracy >= 0.80 else '[BELOW]'}")
        print(f"  Min P-rule: {min_p_rule:.4f} {'[OK]' if min_p_rule >= 0.80 else '[BELOW]'}")
        print(f"  P-rules: race={p_rule_values['race']:.4f}, sex={p_rule_values['sex']:.4f}")
        print(f"  Lambda: {[f'{x:.2f}' for x in current_lambda]}")
        print(f"\n[DECISION]")
        print(f"  {reasoning}")
        
        # Check convergence
        if accuracy >= 0.80 and min_p_rule >= 0.80:
            if convergence_step is None:
                convergence_step = iteration
                print(f"\n[SUCCESS] CONVERGENCE ACHIEVED AT STEP {iteration}!")
                print(f"  Accuracy: {accuracy:.4f} >= 0.80")
                print(f"  P-rule:   {min_p_rule:.4f} >= 0.80")
    
    # Save metrics
    with open("training_metrics_enhanced.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVED] Metrics to training_metrics_enhanced.json")
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - FINAL RESULTS")
    print("=" * 70)
    
    if convergence_step:
        print(f"\n[SUCCESS] Converged at step {convergence_step}")
        conv_metrics = metrics[convergence_step - 1]
        print(f"  Accuracy: {conv_metrics['accuracy']:.4f}")
        print(f"  P-rule:   {conv_metrics['min_p_rule']:.4f}")
    else:
        print(f"\n[INFO] Did not reach 80/80 convergence in 100 steps")
    
    # Final metrics
    final = metrics[-1]
    print(f"\n[FINAL STEP {len(metrics)}]")
    print(f"  Accuracy: {final['accuracy']:.4f}")
    print(f"  P-rule:   {final['min_p_rule']:.4f}")
    print(f"  Lambda:   {[f'{x:.2f}' for x in final['lambda']]}")
    
    # Statistics
    accuracies = [m['accuracy'] for m in metrics]
    p_rules = [m['min_p_rule'] for m in metrics]
    
    print(f"\n[STATISTICS]")
    print(f"  Accuracy:  min={min(accuracies):.4f}, max={max(accuracies):.4f}, mean={np.mean(accuracies):.4f}")
    print(f"  P-rule:    min={min(p_rules):.4f}, max={max(p_rules):.4f}, mean={np.mean(p_rules):.4f}")
    
    # Steps meeting each target
    steps_acc_ok = sum(1 for a in accuracies if a >= 0.80)
    steps_prule_ok = sum(1 for p in p_rules if p >= 0.80)
    steps_both_ok = sum(1 for m in metrics if m['accuracy'] >= 0.80 and m['min_p_rule'] >= 0.80)
    
    print(f"\n[CONVERGENCE TRACKING]")
    print(f"  Steps with Accuracy >= 0.80: {steps_acc_ok}/100")
    print(f"  Steps with P-rule >= 0.80:   {steps_prule_ok}/100")
    print(f"  Steps with BOTH >= 0.80:     {steps_both_ok}/100")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    train_with_enhanced_orchestrator()
