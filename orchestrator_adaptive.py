"""
Orchestrator for Adaptive Lambda Learning with Momentum
Learns to adjust lambda between training steps
Designed for CODE_FINAL_COMPLETE integration
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class LambdaDecision:
    """Decision returned by orchestrator"""
    def __init__(self, lambda_new: List[float], momentum_new: List[float], reason: str):
        self.lambda_new = lambda_new
        self.momentum_new = momentum_new
        self.reason = reason


class OrchestratorAdaptive:
    """
    Makes adaptive lambda decisions between training steps.
    Learns from memory of what lambda values worked.
    Uses momentum for smooth transitions.
    """
    
    def __init__(self, memory_file: str = "adaptive_memory.json"):
        self.memory_file = Path(memory_file)
        self.memory = []
        self.momentum_beta = 0.7  # How much to keep previous momentum
        
        self._load_memory()
    
    def _load_memory(self):
        """Load past experiences"""
        if self.memory_file.exists():
            with open(self.memory_file, 'r') as f:
                self.memory = json.load(f)
            print(f"[ORCHESTRATOR] Loaded {len(self.memory)} past step experiences")
        else:
            self.memory = []
            print("[ORCHESTRATOR] Starting with fresh memory")
    
    def _save_memory(self):
        """Save experiences"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def decide(
        self,
        step: int,
        accuracy: float,
        p_rule_values: Dict[str, float],
        current_lambda: List[float],
        current_momentum: List[float],
        target_p_rule: float = 0.80,
        target_accuracy: float = 0.80,
        lambda_learning_rate: float = 2.5,
        lambda_max: float = 20.0
    ) -> LambdaDecision:
        """
        Decide lambda adjustments for NEXT step based on current metrics and memory.
        
        Args:
            step: Current step number
            accuracy: Current classifier accuracy
            p_rule_values: P-rule for each sensitive attribute
            current_lambda: Current lambda values (per attribute)
            current_momentum: Current momentum values
            target_p_rule: Target p-rule (default 0.80 = 80%)
            target_accuracy: Target accuracy (default 0.80 = 80%)
            lambda_learning_rate: How aggressively to update lambda
            lambda_max: Maximum lambda value
        
        Returns:
            LambdaDecision with new lambda, momentum, and reasoning
        """
        
        # Compute min p-rule (bottleneck fairness)
        min_p_rule = min(p_rule_values.values()) if p_rule_values else 0.0
        
        # Check convergence
        if accuracy >= target_accuracy and min_p_rule >= target_p_rule:
            return LambdaDecision(
                lambda_new=current_lambda,
                momentum_new=current_momentum,
                reason=f"CONVERGENCE! Acc={accuracy:.3f}, min P-rule={min_p_rule:.3f}"
            )
        
        # LEARN from memory: what worked in similar situations?
        similar = self._find_similar_experiences(accuracy, min_p_rule)
        
        if similar:
            # Use LLM-guided decision based on similar past steps
            new_lambda, new_momentum = self._learn_from_similar(
                current_lambda, current_momentum, p_rule_values, similar,
                lambda_learning_rate, lambda_max, target_p_rule
            )
            reason = f"Learning from {len(similar)} similar past steps"
        else:
            # Fallback: use adaptive heuristic
            new_lambda, new_momentum = self._compute_adaptive_update(
                current_lambda, current_momentum, p_rule_values,
                lambda_learning_rate, lambda_max, target_p_rule
            )
            reason = "No similar experiences; using adaptive heuristic"
        
        # Record this step
        self._record_experience(
            step=step,
            accuracy=accuracy,
            min_p_rule=min_p_rule,
            p_rule_values=p_rule_values,
            lambda_old=current_lambda,
            momentum_old=current_momentum,
            lambda_new=new_lambda,
            momentum_new=new_momentum
        )
        
        print(f"[ORCHESTRATOR] Step {step}: Acc={accuracy:.3f}, min P-rule={min_p_rule:.3f}")
        print(f"  → Old λ: {[f'{x:.3f}' for x in current_lambda]}")
        print(f"  → New λ: {[f'{x:.3f}' for x in new_lambda]}")
        print(f"  Reason: {reason}")
        
        return LambdaDecision(
            lambda_new=new_lambda,
            momentum_new=new_momentum,
            reason=reason
        )
    
    def _find_similar_experiences(self, accuracy: float, min_p_rule: float) -> List[Dict]:
        """Find past steps with similar metrics"""
        if not self.memory:
            return []
        
        similar = []
        for exp in self.memory:
            acc_diff = abs(exp['accuracy'] - accuracy)
            p_rule_diff = abs(exp['min_p_rule'] - min_p_rule)
            distance = acc_diff + p_rule_diff
            
            if distance < 0.10:  # Close enough?
                similar.append((distance, exp))
        
        similar.sort(key=lambda x: x[0])
        return [exp for _, exp in similar[:3]]
    
    def _learn_from_similar(
        self, current_lambda, current_momentum, p_rule_values, similar_exps,
        lambda_learning_rate, lambda_max, target_p_rule
    ):
        """Learn what worked in similar past situations"""
        
        # Aggregate successful lambda increments from similar situations
        n_attrs = len(current_lambda)
        successful_increments = [[] for _ in range(n_attrs)]
        
        for exp in similar_exps:
            for i in range(n_attrs):
                old_lambda = exp['lambda_old'][i] if i < len(exp['lambda_old']) else current_lambda[i]
                new_lambda = exp['lambda_new'][i] if i < len(exp['lambda_new']) else current_lambda[i]
                increment = new_lambda - old_lambda
                
                # Did it help fairness? (was p-rule gap reducing?)
                if 'p_rule_values' in exp and i < len(exp['p_rule_values']):
                    old_p_rule = exp['p_rule_values'].get(f'attr_{i}', 0.5)
                    if old_p_rule < target_p_rule:  # Was below target before
                        # If we increased lambda, record it as helpful
                        if increment > 0:
                            successful_increments[i].append(increment)
        
        # Use average of successful increments, or fall back to heuristic
        new_lambda = list(current_lambda)
        new_momentum = list(current_momentum)
        
        for i in range(n_attrs):
            if successful_increments[i]:
                # Use learned increment
                avg_increment = np.mean(successful_increments[i])
                new_momentum[i] = self.momentum_beta * new_momentum[i] + (1 - self.momentum_beta) * avg_increment
            else:
                # Compute heuristic increment
                p_rule_gap = (target_p_rule - p_rule_values.get(f'attr_{i}', 0.5)) / 100.0
                increment = lambda_learning_rate * p_rule_gap
                new_momentum[i] = self.momentum_beta * new_momentum[i] + (1 - self.momentum_beta) * increment
            
            new_lambda[i] = max(0.0, min(new_lambda[i] + new_momentum[i], lambda_max))
        
        return new_lambda, new_momentum
    
    def _compute_adaptive_update(
        self, current_lambda, current_momentum, p_rule_values,
        lambda_learning_rate, lambda_max, target_p_rule
    ):
        """Compute lambda update using adaptive heuristic (original algorithm)"""
        
        new_lambda = list(current_lambda)
        new_momentum = list(current_momentum)
        
        for i in range(len(current_lambda)):
            # Compute gap for this attribute
            attr_key = f'attr_{i}' if f'attr_{i}' in p_rule_values else list(p_rule_values.keys())[i] if p_rule_values else f'attr_{i}'
            current_p_rule = p_rule_values.get(attr_key, 0.5) if p_rule_values else 0.5
            
            p_rule_gap = (target_p_rule - current_p_rule) / 100.0
            increment = lambda_learning_rate * p_rule_gap
            
            # Apply momentum
            new_momentum[i] = self.momentum_beta * new_momentum[i] + (1 - self.momentum_beta) * increment
            
            # Update lambda
            new_lambda[i] = max(0.0, min(new_lambda[i] + new_momentum[i], lambda_max))
        
        return new_lambda, new_momentum
    
    def _record_experience(
        self, step, accuracy, min_p_rule, p_rule_values,
        lambda_old, momentum_old, lambda_new, momentum_new
    ):
        """Record this step for future learning"""
        
        exp = {
            'step': step,
            'accuracy': float(accuracy),
            'min_p_rule': float(min_p_rule),
            'p_rule_values': {str(k): float(v) for k, v in p_rule_values.items()},
            'lambda_old': [float(x) for x in lambda_old],
            'momentum_old': [float(x) for x in momentum_old],
            'lambda_new': [float(x) for x in lambda_new],
            'momentum_new': [float(x) for x in momentum_new]
        }
        
        self.memory.append(exp)
        self._save_memory()
