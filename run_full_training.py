
import sys
sys.path.insert(0, 'C:\\Users\\MSI\\Desktop\\MEMOIRE')

from train_enhanced_llm import train_with_enhanced_orchestrator
import json

# Run training
print("[STARTING] Full 100-iteration training...")
train_with_enhanced_orchestrator()

# Save final metrics
print("\n[COMPLETE] Training finished. Analyzing results...")

try:
    with open('training_metrics_enhanced.json') as f:
        data = json.load(f)
    
    # Find convergence point
    converged_at = None
    for i, entry in enumerate(data, 1):
        if entry['accuracy'] >= 0.80 and entry['min_p_rule'] >= 0.80:
            converged_at = i
            break
    
    if converged_at:
        print(f"\n[SUCCESS] Converged at iteration {converged_at}!")
        print(f"  Accuracy: {data[converged_at-1]['accuracy']:.4f}")
        print(f"  P-rule: {data[converged_at-1]['min_p_rule']:.4f}")
    else:
        print("\n[NO CONVERGENCE] Did not reach both targets within 100 iterations")
        print(f"Final iteration:")
        last = data[-1]
        print(f"  Accuracy: {last['accuracy']:.4f}")
        print(f"  P-rule: {last['min_p_rule']:.4f}")
except Exception as e:
    print(f"Error reading results: {e}")
