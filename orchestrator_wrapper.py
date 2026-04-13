"""
Wrapper to integrate OrchestratorAdaptive into CODE_FINAL_COMPLETE pattern
Replaces the existing lambda update logic with orchestrator decisions
"""

from typing import Dict, List, Tuple
from orchestrator_adaptive import OrchestratorAdaptive


class CodeFinalOrchestrator:
    """
    Drop-in replacement for lambda update logic in CODE_FINAL_COMPLETE
    
    Usage in your notebook:
        orchestrator = CodeFinalOrchestrator(
            target_p_rule=0.80,
            target_accuracy=0.80,
            lambda_learning_rate=2.5,
            lambda_max=20.0
        )
        
        for step in range(max_iterations):
            # Train for 50 epochs...
            # Evaluate metrics...
            
            # NEW: Replace old lambda update with:
            new_lambda, new_momentum, reason = orchestrator.update_lambda(
                step=step,
                accuracy=accuracy,
                p_rule_values=p_rule_dict,  # {attr_name: value, ...}
                current_lambda=lambda_vector,
                current_momentum=momentum_vector
            )
            
            lambda_vector = new_lambda
            momentum_vector = new_momentum
    """
    
    def __init__(self, 
                 target_p_rule: float = 0.80,
                 target_accuracy: float = 0.80,
                 lambda_learning_rate: float = 2.5,
                 lambda_max: float = 20.0,
                 momentum_beta: float = 0.7,
                 memory_file: str = "adaptive_memory.json"):
        
        self.orchestrator = OrchestratorAdaptive(memory_file=memory_file)
        self.target_p_rule = target_p_rule
        self.target_accuracy = target_accuracy
        self.lambda_learning_rate = lambda_learning_rate
        self.lambda_max = lambda_max
        self.momentum_beta = momentum_beta
    
    def update_lambda(
        self,
        step: int,
        accuracy: float,
        p_rule_values: Dict[str, float],
        current_lambda: List[float],
        current_momentum: List[float]
    ) -> Tuple[List[float], List[float], str]:
        """
        Replace this OLD code in CODE_FINAL_COMPLETE:
        
            for i, attr in enumerate(attributes):
                p_rule_gap = (P_RULE_THRESHOLD - p_rule[attr]) / 100
                lambda_increment = LAMBDA_LEARNING_RATE * p_rule_gap
                lambda_momentum[i] = 0.7 * lambda_momentum[i] + 0.3 * lambda_increment
                lambda_vector[i] = max(0.0, min(lambda_vector[i] + lambda_momentum[i], LAMBDA_MAX))
        
        With this NEW code:
        
            new_lambda, new_momentum, reason = orchestrator.update_lambda(
                step=step,
                accuracy=accuracy,
                p_rule_values=p_rule,
                current_lambda=lambda_vector,
                current_momentum=lambda_momentum
            )
            lambda_vector = new_lambda
            lambda_momentum = new_momentum
        
        Args:
            step: Current step number (for tracking)
            accuracy: Current classifier accuracy
            p_rule_values: Dict like {'race': 0.75, 'gender': 0.82, ...}
            current_lambda: Current lambda vector [λ1, λ2, λ3]
            current_momentum: Current momentum vector [m1, m2, m3]
        
        Returns:
            Tuple of (new_lambda, new_momentum, reason_string)
        """
        
        decision = self.orchestrator.decide(
            step=step,
            accuracy=accuracy,
            p_rule_values=p_rule_values,
            current_lambda=current_lambda,
            current_momentum=current_momentum,
            target_p_rule=self.target_p_rule,
            target_accuracy=self.target_accuracy,
            lambda_learning_rate=self.lambda_learning_rate,
            lambda_max=self.lambda_max
        )
        
        return decision.lambda_new, decision.momentum_new, decision.reason
