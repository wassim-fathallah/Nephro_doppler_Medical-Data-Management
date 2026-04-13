"""
Test: Run with FIXED HIGH LAMBDA to see what P-rule ceiling we can reach
WITHOUT Llama interference (to isolate architecture limits)
"""

import sys
sys.path.insert(0, 'C:\\Users\\MSI\\Desktop\\MEMOIRE')

from train_enhanced_llm import load_data, Classifier, Adversary, compute_p_rule
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import json

print("=" * 70)
print("TESTING LAMBDA IMPACT ON P-RULE")
print("=" * 70)

(X_train, y_train, s_train), (X_test, y_test, s_test) = load_data()

device = torch.device('cpu')
metrics_per_lambda = {}

# Test different fixed lambda values
for test_lambda in [1.0, 2.0, 3.0, 4.0, 5.0]:
    print(f"\n{'='*70}")
    print(f"Testing with FIXED LAMBDA = {test_lambda}")
    print(f"{'='*70}")
    
    classifier = Classifier(n_features=X_train.shape[1]).to(device)
    adversary = Adversary(n_hidden=32, n_sensitive=2).to(device)
    
    opt_clf = optim.Adam(classifier.parameters(), lr=0.01)
    opt_adv = optim.Adam(adversary.parameters(), lr=0.01)
    
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().view(-1, 1).to(device)
    s_train_t = torch.from_numpy(s_train).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    
    results = []
    
    for iteration in range(1, 51):  # 50 iterations
        # Train Adversary
        classifier.eval()
        for _ in range(3):
            with torch.no_grad():
                hidden = classifier.get_representation(X_train_t)
            s_pred = adversary(hidden)
            adv_loss = nn.BCELoss()(s_pred, s_train_t)
            
            opt_adv.zero_grad()
            adv_loss.backward()
            opt_adv.step()
        
        # Train Classifier - Task-focused first
        adversary.eval()
        for _ in range(4):
            hidden = classifier.get_representation(X_train_t)
            logits = classifier.output_layer(hidden)
            y_pred = torch.sigmoid(logits)
            task_loss = nn.BCELoss()(y_pred, y_train_t)
            
            opt_clf.zero_grad()
            task_loss.backward()
            opt_clf.step()
        
        # Fairness steps
        current_acc = (y_pred > 0.5).float().mean().item()
        if current_acc >= 0.65:
            for _ in range(2):
                hidden = classifier.get_representation(X_train_t)
                logits = classifier.output_layer(hidden)
                y_pred = torch.sigmoid(logits)
                
                task_loss = nn.BCELoss()(y_pred, y_train_t)
                s_pred = adversary(hidden)
                adv_loss = nn.BCELoss()(s_pred, s_train_t)
                
                loss = task_loss - test_lambda * adv_loss
                
                opt_clf.zero_grad()
                loss.backward()
                opt_clf.step()
        
        # Evaluate
        classifier.eval()
        with torch.no_grad():
            y_pred_prob = classifier(X_test_t).cpu().numpy().flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            accuracy = accuracy_score(y_test.flatten(), y_pred)
            p_rule_values = compute_p_rule(y_pred, s_test)
            min_p_rule = min(p_rule_values.values())
        
        results.append({
            'iter': iteration,
            'acc': accuracy,
            'p_rule': min_p_rule,
            'race': p_rule_values['race'],
            'sex': p_rule_values['sex']
        })
        
        if iteration % 10 == 0:
            print(f"Iter {iteration:2d}: Acc={accuracy:.4f}, P-rule={min_p_rule:.4f}, Race={p_rule_values['race']:.4f}, Sex={p_rule_values['sex']:.4f}")
    
    # Store results
    metrics_per_lambda[test_lambda] = {
        'final': results[-1],
        'max_p_rule': max(r['p_rule'] for r in results),
        'mean_p_rule': np.mean([r['p_rule'] for r in results]),
        'final_acc': results[-1]['acc'],
    }
    
    print(f"\nSummary for lambda={test_lambda}:")
    print(f"  Final Accuracy: {results[-1]['acc']:.4f}")
    print(f"  Final P-rule: {results[-1]['p_rule']:.4f}")
    print(f"  Max P-rule reached: {metrics_per_lambda[test_lambda]['max_p_rule']:.4f}")
    print(f"  Mean P-rule: {metrics_per_lambda[test_lambda]['mean_p_rule']:.4f}")

# Summary
print("\n" + "=" * 70)
print("LAMBDA IMPACT SUMMARY")
print("=" * 70)
for lam, metrics in metrics_per_lambda.items():
    print(f"Lambda={lam}: Final Acc={metrics['final_acc']:.4f}, Final P-rule={metrics['final']['p_rule']:.4f}, Max P-rule={metrics['max_p_rule']:.4f}")

# Find best lambda
best_lambda = max(metrics_per_lambda.keys(), key=lambda x: metrics_per_lambda[x]['max_p_rule'])
print(f"\nBest lambda for P-rule: {best_lambda} (reaches {metrics_per_lambda[best_lambda]['max_p_rule']:.4f})")

if metrics_per_lambda[best_lambda]['max_p_rule'] >= 0.80:
    print("[SUCCESS] Can reach P-rule >= 0.80!")
else:
    ceiling = max(m['max_p_rule'] for m in metrics_per_lambda.values())
    print(f"[LIMIT] Architecture ceiling appears to be P-rule <= {ceiling:.4f}")
    print("Need to improve architecture or explore different approach")
