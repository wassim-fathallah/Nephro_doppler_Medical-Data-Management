"""
Plot training metrics: Accuracy and P-rule vs Iterations
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_training_metrics(metrics_file="training_metrics.json", output_file="training_progress.png"):
    """
    Create a plot with 2 curves:
    - X-axis: Number of iterations
    - Y-axis: Accuracy and P-rule (both on same plot)
    """
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Extract data
    steps = [m['step'] for m in metrics]
    accuracy = [m['accuracy'] for m in metrics]
    p_rule = [m['min_p_rule'] for m in metrics]
    
    # Create figure with 2 subplots or 2 lines
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both metrics on same axis
    ax.plot(steps, accuracy, 'b-o', linewidth=2, markersize=4, label='Accuracy', alpha=0.8)
    ax.plot(steps, p_rule, 'r-s', linewidth=2, markersize=4, label='P-rule', alpha=0.8)
    
    # Add target lines
    ax.axhline(y=0.80, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (0.80)')
    
    # Labels and title
    ax.set_xlabel('Iteration (Step)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Training Progress: Accuracy vs P-rule', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.0, 1.05])
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_file}")
    
    # Show stats
    print(f"\n📊 TRAINING STATISTICS")
    print(f"{'='*50}")
    print(f"Total iterations: {len(steps)}")
    print(f"Final Accuracy: {accuracy[-1]:.4f}")
    print(f"Final P-rule: {p_rule[-1]:.4f}")
    print(f"Best Accuracy: {max(accuracy):.4f} at step {steps[accuracy.index(max(accuracy))]}")
    print(f"Best P-rule: {max(p_rule):.4f} at step {steps[p_rule.index(max(p_rule))]}")
    print(f"{'='*50}")
    
    # Show plot
    plt.show()
    
    return fig, ax

if __name__ == "__main__":
    try:
        fig, ax = plot_training_metrics()
        print("\n✅ Plotting complete!")
    except FileNotFoundError:
        print("❌ training_metrics.json not found!")
        print("   Make sure you ran: python train_final_with_llm.py")
