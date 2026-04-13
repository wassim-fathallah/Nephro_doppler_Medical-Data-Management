"""
Enhanced LLM-Based Orchestrator for P-rule Convergence
- Detects oscillation patterns
- Analyzes trade-offs between accuracy and P-rule
- Learns from past mistakes and successes
- Llama3.1 makes intelligent decisions based on history
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    from langchain_ollama import OllamaLLM
    HAS_LANGCHAIN = True
except ImportError:
    try:
        from langchain_community.llms import Ollama as OllamaLLM
        HAS_LANGCHAIN = True
    except ImportError as e:
        HAS_LANGCHAIN = False
        LANGCHAIN_ERROR = str(e)
        OllamaLLM = None


class EnhancedMemory:
    """Track trajectory history with oscillation and trade-off detection."""
    
    def __init__(self, memory_file: str = "orchestrator_enhanced_memory.json"):
        self.memory_file = Path(memory_file)
        self.trajectory = []  # Full history
        self._load_trajectory()
    
    def _load_trajectory(self):
        """Load historical trajectory."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    self.trajectory = json.load(f)
                print(f"[MEMORY] Loaded {len(self.trajectory)} past steps")
            except:
                self.trajectory = []
        else:
            self.trajectory = []
    
    def _save_trajectory(self):
        """Persist trajectory."""
        with open(self.memory_file, 'w') as f:
            json.dump(self.trajectory, f, indent=2)
    
    def add_step(self, step: int, accuracy: float, min_p_rule: float, 
                 p_rules: Dict, lambda_vals: List[float]):
        """Record a training step."""
        entry = {
            'step': step,
            'accuracy': float(accuracy),
            'min_p_rule': float(min_p_rule),
            'p_rules': {str(k): float(v) for k, v in p_rules.items()},
            'lambda': [float(x) for x in lambda_vals],
        }
        self.trajectory.append(entry)
        self._save_trajectory()
    
    def detect_oscillation(self, window: int = 8) -> Tuple[bool, str, float]:
        """
        Detect if P-rule is oscillating.
        
        Returns:
            (is_oscillating, description, avg_swing_magnitude)
        """
        if len(self.trajectory) < window:
            return False, "Not enough history", 0.0
        
        recent = self.trajectory[-window:]
        p_rules = [s['min_p_rule'] for s in recent]
        
        # Calculate swings
        swings = []
        for i in range(1, len(p_rules)):
            swings.append(abs(p_rules[i] - p_rules[i-1]))
        
        avg_swing = np.mean(swings)
        max_swing = np.max(swings)
        
        # Oscillation = frequent large swings
        large_swings = sum(1 for s in swings if s > 0.15)
        
        if large_swings >= 2:  # At least 2 large swings in window
            desc = f"Oscillating: {large_swings} large swings (avg {avg_swing:.3f}, max {max_swing:.3f})"
            return True, desc, avg_swing
        
        return False, "Stable", avg_swing
    
    def analyze_trade_off(self, window: int = 5) -> Dict:
        """
        Analyze correlation between lambda changes and metric changes.
        
        Returns dict with:
        - lambda_trend: how lambda is changing
        - p_rule_trend: how P-rule is responding
        - accuracy_trend: how accuracy is responding
        - recommendations: what to do next
        """
        if len(self.trajectory) < window + 1:
            return {
                'correlation': 'insufficient_history',
                'lambda_trend': 'unknown',
                'p_rule_trend': 'unknown',
                'accuracy_trend': 'unknown'
            }
        
        recent = self.trajectory[-(window+1):]
        
        # Calculate changes
        lambda_changes = []
        p_rule_changes = []
        accuracy_changes = []
        
        for i in range(1, len(recent)):
            lam_delta = recent[i]['lambda'][1] - recent[i-1]['lambda'][1]  # fairness weight
            p_delta = recent[i]['min_p_rule'] - recent[i-1]['min_p_rule']
            acc_delta = recent[i]['accuracy'] - recent[i-1]['accuracy']
            
            lambda_changes.append(lam_delta)
            p_rule_changes.append(p_delta)
            accuracy_changes.append(acc_delta)
        
        # Trend analysis
        avg_lambda_change = np.mean(lambda_changes)
        avg_p_rule_change = np.mean(p_rule_changes)
        avg_accuracy_change = np.mean(accuracy_changes)
        
        # Correlation
        if avg_lambda_change > 0.05:
            lambda_trend = f"INCREASING (avg +{avg_lambda_change:.3f} per step)"
        elif avg_lambda_change < -0.05:
            lambda_trend = f"DECREASING (avg {avg_lambda_change:.3f} per step)"
        else:
            lambda_trend = "STABLE"
        
        if avg_p_rule_change > 0.03:
            p_rule_trend = f"IMPROVING (avg +{avg_p_rule_change:.3f} per step)"
        elif avg_p_rule_change < -0.03:
            p_rule_trend = f"DEGRADING (avg {avg_p_rule_change:.3f} per step)"
        else:
            p_rule_trend = "STABLE"
        
        if avg_accuracy_change > 0.01:
            accuracy_trend = f"IMPROVING (avg +{avg_accuracy_change:.3f} per step)"
        elif avg_accuracy_change < -0.01:
            accuracy_trend = f"DEGRADING (avg {avg_accuracy_change:.3f} per step)"
        else:
            accuracy_trend = "STABLE"
        
        return {
            'lambda_trend': lambda_trend,
            'p_rule_trend': p_rule_trend,
            'accuracy_trend': accuracy_trend,
            'avg_lambda_delta': float(avg_lambda_change),
            'avg_p_rule_delta': float(avg_p_rule_change),
            'avg_accuracy_delta': float(avg_accuracy_change),
        }
    
    def get_recent_context(self, steps: int = 5) -> str:
        """Get formatted recent history for LLM prompt."""
        if len(self.trajectory) < steps:
            return "History: Not enough past steps yet"
        
        recent = self.trajectory[-steps:]
        context = f"Recent history (last {steps} steps):\n"
        
        for entry in recent:
            context += f"  Step {entry['step']}: Acc={entry['accuracy']:.3f}, P-rule={entry['min_p_rule']:.3f}, λ={[f'{x:.2f}' for x in entry['lambda']]}\n"
        
        return context
    
    def find_best_state(self) -> Dict:
        """Find best step in history (closest to 80/80)."""
        if not self.trajectory:
            return None
        
        best = None
        best_score = -1
        
        for entry in self.trajectory:
            # Score: how close to 80/80 (both targets)
            acc_gap = max(0, 0.80 - entry['accuracy']) * 10  # penalty for being below 0.80
            prule_gap = max(0, 0.80 - entry['min_p_rule']) * 10
            score = 1.0 / (1.0 + acc_gap + prule_gap)
            
            if score > best_score:
                best_score = score
                best = entry
        
        return best
    
    def check_convergence(self, accuracy: float, min_p_rule: float) -> bool:
        """Check if both targets are met."""
        return accuracy >= 0.80 and min_p_rule >= 0.80


class EnhancedLLMOrchestrator:
    """
    Intelligent LLM orchestrator that learns from history and adapts decisions.
    """
    
    def __init__(self, model_name: str = "llama3.1", temperature: float = 0.2):
        """Initialize orchestrator with Llama3.1."""
        self.memory = EnhancedMemory()
        self.conversation_history = []  # For multi-turn if needed
        
        if not HAS_LANGCHAIN:
            raise ImportError(f"LangChain not installed: {LANGCHAIN_ERROR}")
        
        print(f"[ORCHESTRATOR] Initializing Llama3.1 (temperature={temperature})...")
        try:
            self.llm = OllamaLLM(model=model_name, temperature=temperature)
            self._test_llm()
        except Exception as e:
            print(f"[ERROR] Failed to initialize Llama: {e}")
            raise
    
    def _test_llm(self):
        """Test LLM connection."""
        try:
            response = self.llm.invoke("Say OK if you understand.")
            print("[OK] Llama3.1 connected and responding")
        except Exception as e:
            raise RuntimeError(f"Llama connection failed: {e}")
    
    def decide(
        self,
        step: int,
        accuracy: float,
        p_rule_values: Dict[str, float],
        current_lambda: List[float],
        target_p_rule: float = 0.80,
        target_accuracy: float = 0.80,
        lambda_max: float = 2.0
    ) -> Tuple[List[float], str]:
        """
        Intelligent decision: Llama analyzes patterns and decides lambda adjustment.
        """
        min_p_rule = min(p_rule_values.values()) if p_rule_values else 0.0
        
        # Record this step
        self.memory.add_step(step, accuracy, min_p_rule, p_rule_values, current_lambda)
        
        # Check convergence
        if self.memory.check_convergence(accuracy, min_p_rule):
            return current_lambda, f"[CONVERGED] Acc={accuracy:.3f}, P-rule={min_p_rule:.3f}"
        
        # Detect patterns
        is_oscillating, oscillation_desc, swing_mag = self.memory.detect_oscillation()
        trade_off_analysis = self.memory.analyze_trade_off()
        recent_history = self.memory.get_recent_context(steps=5)
        best_state = self.memory.find_best_state()
        
        # Build smart prompt
        prompt = self._build_smart_prompt(
            step, accuracy, min_p_rule, p_rule_values, current_lambda,
            target_accuracy, target_p_rule,
            is_oscillating, oscillation_desc, trade_off_analysis,
            recent_history, best_state, lambda_max
        )
        
        print(f"\n[STEP {step}] Consulting Llama3.1...")
        try:
            llm_response = self.llm.invoke(prompt)
            print(f"[LLAMA] {llm_response[:200]}...")
        except Exception as e:
            print(f"[WARN] Llama error: {e}, using fallback")
            return self._fallback_decision(
                current_lambda, accuracy, min_p_rule, is_oscillating, 
                trade_off_analysis, lambda_max
            )
        
        # Parse Llama response
        new_lambda, reasoning = self._parse_response(
            llm_response, current_lambda, lambda_max
        )
        
        return new_lambda, reasoning
    
    def _build_smart_prompt(
        self, step, accuracy, min_p_rule, p_rule_values, current_lambda,
        target_accuracy, target_p_rule, is_oscillating, oscillation_desc,
        trade_off_analysis, recent_history, best_state, lambda_max
    ) -> str:
        """Build intelligent prompt with full context."""
        
        prompt = f"""You are an expert fairness ML orchestrator. Your job: maximize P-rule to >=80% while keeping accuracy >=80%.

CURRENT METRICS (Step {step}):
  Accuracy: {accuracy:.3f} (target: {target_accuracy:.2f})
  Min P-rule: {min_p_rule:.3f} (target: {target_p_rule:.2f})
  Current Lambda: {[f'{x:.2f}' for x in current_lambda]}
  P-rules by attribute: {{{', '.join(f'{k}:{v:.3f}' for k, v in p_rule_values.items())}}}

PATTERN ANALYSIS:
  Oscillation Status: {oscillation_desc}
  Lambda Trend: {trade_off_analysis.get('lambda_trend', 'unknown')}
  P-rule Trend: {trade_off_analysis.get('p_rule_trend', 'unknown')}
  Accuracy Trend: {trade_off_analysis.get('accuracy_trend', 'unknown')}

RECENT HISTORY:
{recent_history}

BEST STATE ACHIEVED:
  Step {best_state['step']}: Acc={best_state['accuracy']:.3f}, P-rule={best_state['min_p_rule']:.3f}

YOUR TASK:
1. Analyze the pattern: Is P-rule converging or oscillating?
2. Assess trade-off: How does changing lambda affect accuracy vs fairness?
3. Learn from history: What worked in the past when metrics were similar?
4. Decide: Adjust lambda to maximize P-rule while keeping accuracy >=0.80

OUTPUT FORMAT (ONLY JSON, no other text):
{{
  "adjustment": <float between -1.5 and +1.5>,
  "confidence": <float 0-1>,
  "reasoning": "<explain decision based on patterns>",
  "expected_outcome": "<what you expect this adjustment to achieve>"
}}

GUIDELINES:
- If oscillating: Small adjustments (0.1-0.3) to stabilize
- If P-rule low: Aggressive increase (0.5-1.5) to push fairness
- If approaching 80/80: Tiny adjustments (0.05-0.2) to fine-tune
- Protect accuracy: Never let it drop below 0.70 intentionally
- Maximize P-rule: This is primary goal once accuracy >=0.80"""
        
        return prompt
    
    def _parse_response(self, response: str, current_lambda: List[float], 
                       lambda_max: float) -> Tuple[List[float], str]:
        """Parse Llama JSON response."""
        try:
            import json
            
            # Extract JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start == -1 or end == 0:
                raise ValueError("No JSON found")
            
            json_str = response[start:end]
            decision = json.loads(json_str)
            
            adjustment = float(decision.get('adjustment', 0.0))
            reasoning = decision.get('reasoning', 'Llama decision')
            
            # Clamp adjustment
            adjustment = max(-1.5, min(1.5, adjustment))
            
            # Apply to all lambda values
            new_lambda = [max(0.0, min(curr + adjustment, lambda_max)) 
                         for curr in current_lambda]
            
            return new_lambda, reasoning
        
        except Exception as e:
            print(f"[WARN] Parse error: {e}")
            return self._fallback_decision(
                current_lambda, 0.5, 0.5, False, {}, lambda_max
            )
    
    def _fallback_decision(self, current_lambda, accuracy, min_p_rule,
                          is_oscillating, trade_off, lambda_max):
        """Fallback rule-based decision."""
        
        # If oscillating, dampen changes
        if is_oscillating:
            adjustment = 0.1 if min_p_rule < 0.80 else 0.0
            reasoning = "Oscillation detected - dampening changes"
        # If P-rule critical
        elif min_p_rule < 0.70:
            adjustment = 0.5
            reasoning = "P-rule critical - aggressive increase"
        # If P-rule improving
        elif min_p_rule < 0.80:
            adjustment = 0.2
            reasoning = "P-rule below target - moderate increase"
        # If both near target
        elif accuracy >= 0.80:
            adjustment = 0.05
            reasoning = "Fine-tuning both metrics"
        # If accuracy low
        else:
            adjustment = -0.3
            reasoning = "Accuracy below target - protect performance"
        
        new_lambda = [max(0.0, min(curr + adjustment, lambda_max))
                     for curr in current_lambda]
        
        return new_lambda, reasoning
