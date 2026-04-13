"""
LangChain-based Orchestrator for Adversarial Fairness Learning

This module provides an intelligent meta-controller that operates OUTSIDE the 
PyTorch training loop to adaptively adjust hyperparameters (lambda_vector, 
adversary steps, etc.) based on observed metrics and past experiences.

Key Features:
- Memory-based decision making using experience_log JSON
- Structured JSON output with safe fallbacks
- No modification to core PyTorch training logic
"""

import json
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

# LangChain imports (modern package layout)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, validator


# ============================================================================
# OUTPUT SCHEMA - Ensures structured JSON responses
# ============================================================================

class OrchestratorDecision(BaseModel):
    """Structured output from the LangChain orchestrator agent."""
    
    lambda_vector: List[float] = Field(
        description="Fairness weight vector [demographic_parity, equalized_odds, calibration]",
        min_items=1,
        max_items=10
    )
    
    adv_steps: int = Field(
        description="Number of adversary update steps per iteration",
        ge=1,
        le=5
    )
    
    stop: bool = Field(
        description="Whether to stop training (True if objectives met or no progress)",
        default=False
    )
    
    reason: str = Field(
        description="Brief explanation of the decision (1-2 sentences)"
    )
    
    @validator('lambda_vector')
    def validate_lambda(cls, v):
        """Ensure lambda values are in valid range [0, 1]."""
        for val in v:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Lambda value {val} out of bounds [0, 1]")
        return v
    
    @validator('adv_steps')
    def validate_adv_steps(cls, v):
        """Ensure adversary steps are in valid range [1, 5]."""
        if not (1 <= v <= 5):
            raise ValueError(f"Adversary steps {v} out of bounds [1, 5]")
        return v


# ============================================================================
# LANGCHAIN ORCHESTRATOR AGENT
# ============================================================================

class LangChainOrchestrator:
    """
    Intelligent meta-controller for adversarial fairness training.
    
    This orchestrator operates OUTSIDE the training loop, observing metrics
    after each iteration and making strategic decisions about:
    - Lambda vector (fairness weights)
    - Adversary update frequency
    - Early stopping
    
    It uses:
    - Short-term memory: Last N iterations (in prompt)
    - Long-term memory: experience_log.json (past runs)
    - LLM reasoning: Pattern recognition from similar scenarios
    """
    
    def __init__(
        self,
        memory_file: str = "orchestrator_memory.json",
        llm_model: str = "llama3.1",
        temperature: float = 0.3,
        target_p_rule: float = 0.80,
        target_accuracy: float = 0.80,
        short_term_window: int = 5,
        verbose: bool = True
    ):
        """
        Initialize LangChain Orchestrator.
        
        Args:
            memory_file: Path to JSON file storing past experiences
            llm_model: Ollama model name (e.g., "llama3.1", "mistral")
            temperature: LLM temperature (0.0 = deterministic, 1.0 = creative)
            target_p_rule: Target p-rule threshold (e.g., 0.80 = 80% parity)
            target_accuracy: Minimum acceptable accuracy
            short_term_window: Number of recent iterations to include in prompt
            verbose: Whether to print debug information
        """
        self.memory_file = Path(memory_file)
        self.target_p_rule = target_p_rule
        self.target_accuracy = target_accuracy
        self.short_term_window = short_term_window
        self.verbose = verbose
        
        # Short-term memory (last N iterations)
        self.short_term_memory: List[Dict[str, Any]] = []
        
        # Long-term memory (loaded from JSON)
        self.long_term_memory: List[Dict[str, Any]] = []
        self._load_long_term_memory()
        
        # Initialize LangChain components
        self.llm = ChatOllama(
            model=llm_model,
            temperature=temperature,
            format="json"  # Force JSON output
        )
        
        # Output parser for structured responses
        self.output_parser = PydanticOutputParser(pydantic_object=OrchestratorDecision)
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Compose prompt -> model -> parser using modern Runnable API
        self.chain = self.prompt_template | self.llm | self.output_parser
        
        if self.verbose:
            print("\n[INIT] LangChain Orchestrator initialized")
            print(f"   Memory file: {self.memory_file}")
            print(f"   Long-term experiences: {len(self.long_term_memory)}")
            print(f"   Target p-rule: {self.target_p_rule:.2f}")
            print(f"   Target accuracy: {self.target_accuracy:.2f}")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the system prompt for the orchestrator agent."""
        
        template = """You are an expert optimization controller for adversarial fairness learning.

# YOUR ROLE
You observe training metrics and decide adaptive hyperparameters to:
1. Maximize task accuracy (target >= {target_accuracy})
2. Achieve fairness (p-rule >= {target_p_rule})
3. Minimize training time (fast convergence)

# CURRENT STATE
Iteration: {epoch}
Metrics:
  - Task Loss: {loss_task:.4f}
  - Adversary Loss: {loss_adv:.4f}
  - P-rule (fairness): {p_rule:.4f} (target: >={target_p_rule})
  - Accuracy: {accuracy:.4f} (target: >={target_accuracy})

Current Settings:
  - Lambda vector: {current_lambda}
  - Adversary steps: {current_adv_steps}

# SHORT-TERM HISTORY (Last {short_term_window} iterations)
{short_term_history}

# RELEVANT PAST EXPERIENCES
{relevant_experiences}

# DECISION GUIDELINES
- If p-rule < target: Increase lambda weights to prioritize fairness
- If accuracy drops too much: Decrease lambda to restore performance
- If training is unstable (oscillating losses): Reduce adversary steps
- If converging slowly: Try increasing adversary steps
- If objectives met (p-rule >= {target_p_rule} AND accuracy >= {target_accuracy}): Set stop=True

# CONSTRAINTS
- lambda_vector: Each value in [0.0, 1.0]
- adv_steps: Integer in [1, 5]
- Use past experiences to avoid repeating mistakes
- Prefer smooth, gradual adjustments over large jumps

# REQUIRED OUTPUT
{format_instructions}

Think step-by-step:
1. Analyze current metrics vs targets
2. Check if similar scenarios exist in past experiences
3. Decide lambda adjustments (prioritize fairness if p-rule low)
4. Decide adversary steps (balance stability vs convergence)
5. Determine if stopping criteria met

Provide your decision as valid JSON:"""
        
        return PromptTemplate(
            input_variables=[
                "epoch",
                "loss_task",
                "loss_adv",
                "p_rule",
                "accuracy",
                "current_lambda",
                "current_adv_steps",
                "short_term_history",
                "relevant_experiences",
                "target_p_rule",
                "target_accuracy",
                "short_term_window"
            ],
            template=template,
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
    
    def _load_long_term_memory(self):
        """Load past experiences from JSON file."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    self.long_term_memory = json.load(f)
                if self.verbose:
                    print(f"   [OK] Loaded {len(self.long_term_memory)} past experiences")
            except Exception as e:
                print(f"   [WARNING] Could not load memory file: {e}")
                self.long_term_memory = []
        else:
            self.long_term_memory = []
            if self.verbose:
                print(f"   [INFO] Memory file not found, starting fresh")
    
    def _save_long_term_memory(self):
        """Persist experiences to JSON file."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.long_term_memory, f, indent=2)
        except Exception as e:
            print(f"   [WARNING] Could not save memory: {e}")
    
    def _retrieve_relevant_experiences(
        self,
        current_state: Dict[str, Any],
        top_k: int = 3
    ) -> str:
        """
        Retrieve most relevant past experiences based on similarity.
        
        Simple heuristic: Find experiences with similar p-rule values.
        (Can be extended with embeddings/vector search later)
        """
        if not self.long_term_memory:
            return "No past experiences available."
        
        current_p_rule = current_state.get("p_rule", 0.0)
        
        # Score by similarity in p-rule + success
        scored_experiences = []
        for exp in self.long_term_memory:
            # Extract final metrics
            final_metrics = exp.get("final_metrics", {})
            exp_p_rule = final_metrics.get("p_rule", 0.0)
            exp_accuracy = final_metrics.get("accuracy", 0.0)
            
            # Similarity score: inverse of p-rule difference
            p_rule_diff = abs(current_p_rule - exp_p_rule)
            similarity = 1.0 / (1.0 + p_rule_diff)
            
            # Bonus if experience was successful
            success_bonus = 0.0
            if exp_p_rule >= self.target_p_rule and exp_accuracy >= self.target_accuracy:
                success_bonus = 0.5
            
            score = similarity + success_bonus
            scored_experiences.append((score, exp))
        
        # Sort by score (highest first)
        scored_experiences.sort(key=lambda x: x[0], reverse=True)
        
        # Format top-k experiences
        relevant = []
        for score, exp in scored_experiences[:top_k]:
            summary = {
                "iteration": exp.get("iteration", "?"),
                "lambda_used": exp.get("lambda_vector", []),
                "adv_steps": exp.get("adv_steps", "?"),
                "initial_p_rule": exp.get("initial_metrics", {}).get("p_rule", "?"),
                "final_p_rule": exp.get("final_metrics", {}).get("p_rule", "?"),
                "final_accuracy": exp.get("final_metrics", {}).get("accuracy", "?"),
                "converged": exp.get("converged", False)
            }
            relevant.append(json.dumps(summary, indent=2))
        
        return "\n\n".join(relevant) if relevant else "No relevant experiences found."
    
    def _format_short_term_history(self) -> str:
        """Format recent iterations for the prompt."""
        if not self.short_term_memory:
            return "No history yet (first iteration)."
        
        history_lines = []
        for i, state in enumerate(self.short_term_memory[-self.short_term_window:]):
            history_lines.append(
                f"  Iter {state['epoch']}: "
                f"p-rule={state['p_rule']:.3f}, "
                f"acc={state['accuracy']:.3f}, "
                f"lambda={state.get('lambda_vector', 'N/A')}"
            )
        return "\n".join(history_lines)
    
    def decide(
        self,
        current_state: Dict[str, Any],
        current_lambda: List[float],
        current_adv_steps: int
    ) -> OrchestratorDecision:
        """
        Main decision function - called after each training iteration.
        
        Args:
            current_state: Dict containing:
                - epoch: int
                - loss_task: float
                - loss_adv: float
                - p_rule: float
                - accuracy: float
            current_lambda: Current lambda vector
            current_adv_steps: Current adversary steps
        
        Returns:
            OrchestratorDecision with new hyperparameters
        """
        # Add current state to short-term memory
        state_with_config = {
            **current_state,
            "lambda_vector": current_lambda,
            "adv_steps": current_adv_steps
        }
        self.short_term_memory.append(state_with_config)
        
        # Retrieve relevant past experiences
        relevant_exp = self._retrieve_relevant_experiences(current_state)
        
        # Prepare prompt inputs
        prompt_inputs = {
            "epoch": current_state["epoch"],
            "loss_task": current_state["loss_task"],
            "loss_adv": current_state["loss_adv"],
            "p_rule": current_state["p_rule"],
            "accuracy": current_state["accuracy"],
            "current_lambda": current_lambda,
            "current_adv_steps": current_adv_steps,
            "short_term_history": self._format_short_term_history(),
            "relevant_experiences": relevant_exp,
            "target_p_rule": self.target_p_rule,
            "target_accuracy": self.target_accuracy,
            "short_term_window": self.short_term_window
        }
        
        # Call LLM chain
        try:
            decision = self.chain.invoke(prompt_inputs)
            
            if self.verbose:
                print(f"\n[DECISION] Orchestrator Decision (Iteration {current_state['epoch']}):")
                print(f"   Lambda: {decision.lambda_vector}")
                print(f"   Adv steps: {decision.adv_steps}")
                print(f"   Stop: {decision.stop}")
                print(f"   Reason: {decision.reason}")
            
            return decision
        
        except Exception as e:
            print(f"   [WARNING] LLM decision failed: {e}")
            # Fallback to safe heuristic
            return self._fallback_decision(current_state, current_lambda, current_adv_steps)
    
    def _fallback_decision(
        self,
        current_state: Dict[str, Any],
        current_lambda: List[float],
        current_adv_steps: int
    ) -> OrchestratorDecision:
        """
        Fallback heuristic if LLM fails.
        
        Simple rule-based controller:
        - If p-rule too low: increase lambda
        - If accuracy too low: decrease lambda
        - Keep adv_steps stable
        """
        p_rule = current_state["p_rule"]
        accuracy = current_state["accuracy"]
        
        new_lambda = current_lambda.copy()
        
        # Adjust lambda based on targets
        if p_rule < self.target_p_rule:
            # Need more fairness - increase lambda
            new_lambda = [min(1.0, l * 1.2) for l in new_lambda]
            reason = f"Fallback: Increased lambda (p-rule {p_rule:.3f} < {self.target_p_rule})"
        elif accuracy < self.target_accuracy:
            # Need better accuracy - decrease lambda
            new_lambda = [max(0.0, l * 0.8) for l in new_lambda]
            reason = f"Fallback: Decreased lambda (accuracy {accuracy:.3f} < {self.target_accuracy})"
        else:
            reason = "Fallback: Maintained current settings"
        
        # Check stopping criteria
        stop = (p_rule >= self.target_p_rule and accuracy >= self.target_accuracy)
        
        return OrchestratorDecision(
            lambda_vector=new_lambda,
            adv_steps=current_adv_steps,
            stop=stop,
            reason=reason
        )
    
    def record_experience(
        self,
        iteration: int,
        initial_metrics: Dict[str, Any],
        final_metrics: Dict[str, Any],
        lambda_vector: List[float],
        adv_steps: int,
        converged: bool
    ):
        """
        Record a completed training run in long-term memory.
        
        Args:
            iteration: Training iteration number
            initial_metrics: Metrics at start of iteration
            final_metrics: Metrics at end of iteration
            lambda_vector: Lambda vector used
            adv_steps: Adversary steps used
            converged: Whether training converged successfully
        """
        experience = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "lambda_vector": lambda_vector,
            "adv_steps": adv_steps,
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "converged": converged,
            "improvement": {
                "p_rule_delta": final_metrics.get("p_rule", 0) - initial_metrics.get("p_rule", 0),
                "accuracy_delta": final_metrics.get("accuracy", 0) - initial_metrics.get("accuracy", 0)
            }
        }
        
        self.long_term_memory.append(experience)
        self._save_long_term_memory()
        
        if self.verbose:
            print(f"   [SAVED] Experience recorded (total: {len(self.long_term_memory)})")
    
    def reset_short_term_memory(self):
        """Clear short-term memory (call at start of new training run)."""
        self.short_term_memory = []


# ============================================================================
# INTEGRATION HELPER FUNCTION
# ============================================================================

def create_orchestrator(
    memory_file: str = "orchestrator_memory.json",
    target_p_rule: float = 0.80,
    target_accuracy: float = 0.80,
    llm_model: str = "llama3.1",
    verbose: bool = True
) -> LangChainOrchestrator:
    """
    Factory function to create orchestrator with common defaults.
    
    Example:
        orchestrator = create_orchestrator(
            memory_file="my_experiences.json",
            target_p_rule=0.85,
            llm_model="llama3.1"
        )
    """
    return LangChainOrchestrator(
        memory_file=memory_file,
        llm_model=llm_model,
        target_p_rule=target_p_rule,
        target_accuracy=target_accuracy,
        verbose=verbose
    )


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("LangChain Orchestrator Demo")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = create_orchestrator(
        memory_file="test_memory.json",
        target_p_rule=0.80,
        target_accuracy=0.85
    )
    
    # Simulate training iterations
    current_lambda = [0.5, 0.3, 0.2]
    current_adv_steps = 3
    
    for epoch in range(1, 6):
        # Simulate metrics (in real scenario, these come from PyTorch training)
        current_state = {
            "epoch": epoch,
            "loss_task": 0.5 - epoch * 0.05,
            "loss_adv": 0.3 + epoch * 0.02,
            "p_rule": 0.60 + epoch * 0.05,  # Gradually improving
            "accuracy": 0.82 + epoch * 0.01
        }
        
        # Get decision from orchestrator
        decision = orchestrator.decide(
            current_state=current_state,
            current_lambda=current_lambda,
            current_adv_steps=current_adv_steps
        )
        
        # Apply decision
        current_lambda = decision.lambda_vector
        current_adv_steps = decision.adv_steps
        
        if decision.stop:
            print("\n[OK] Training stopped by orchestrator")
            break
    
    # Record final experience
    orchestrator.record_experience(
        iteration=1,
        initial_metrics={"p_rule": 0.60, "accuracy": 0.82},
        final_metrics={"p_rule": 0.85, "accuracy": 0.87},
        lambda_vector=current_lambda,
        adv_steps=current_adv_steps,
        converged=True
    )
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
