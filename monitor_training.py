#!/usr/bin/env python3
"""Monitor training progress in real-time."""

import json
import time
import os
from pathlib import Path

metrics_file = Path("C:\\Users\\MSI\\Desktop\\MEMOIRE\\training_metrics.json")

def monitor():
    last_checked = 0
    last_step = -1
    
    print("[MONITOR] Starting to monitor training progress...")
    print("[MONITOR] Checking for metrics file every 10 seconds...")
    
    while True:
        time.sleep(10)
        
        if not metrics_file.exists():
            if time.time() - last_checked > 30:
                print("[MONITOR] Waiting for training_metrics.json to be created...")
                last_checked = time.time()
            continue
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            if len(metrics) > last_step:
                # New data available
                for step_data in metrics[max(0, last_step):]:
                    step = step_data['step']
                    acc = step_data['accuracy']
                    prule = step_data['min_p_rule']
                    
                    # Check if converged
                    if acc >= 0.80 and prule >= 0.80:
                        print(f"\n[CONVERGED] Step {step}: Accuracy={acc:.4f}, P-rule={prule:.4f}")
                        print("[SUCCESS] BOTH TARGETS REACHED!")
                        return True
                    else:
                        print(f"Step {step}: Accuracy={acc:.4f}, P-rule={prule:.4f}")
                
                last_step = len(metrics) - 1
        except Exception as e:
            print(f"[ERROR] Failed to read metrics: {e}")

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n[MONITOR] Stopped by user")
