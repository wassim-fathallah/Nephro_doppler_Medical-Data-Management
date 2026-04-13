"""
Adapter for Integrating LangChain Orchestrator into Existing Code

This file shows EXACTLY what to change in your existing
CODE_FINAL_COMPLETE.ipynb to use the LangChain orchestrator.

IMPORTANT: This is a MINIMAL modification guide.
Copy the relevant sections into your notebook.
"""

# ============================================================================
# STEP 1: Import the LangChain Orchestrator
# ============================================================================
# Add this at the top of your notebook, after existing imports

from langchain_orchestrator import create_orchestrator


# ============================================================================
# STEP 2: Modify LLMOrchestrator.__init__() or create wrapper
# ============================================================================
# Option A: Replace your existing LLMOrchestrator entirely
# Option B: Add a new method to use LangChain (shown below)

class LLMOrchestratorWithLangChain:
    """
    Extended version of your existing LLMOrchestrator that delegates
    decision-making to the LangChain agent.
    
    This keeps all your existing data loading, evaluation, and reporting
    logic intact while replacing only the lambda update mechanism.
    """
    
    def __init__(self, original_orchestrator):
        """
        Wrap your existing orchestrator.
        
        Args:
            original_orchestrator: Your existing LLMOrchestrator instance
        """
        self.original = original_orchestrator
        
        # Create LangChain orchestrator
        self.langchain_orchestrator = create_orchestrator(
            memory_file=original_orchestrator.memory_file,
            target_p_rule=original_orchestrator.global_objective["min_prule"] / 100.0,
            target_accuracy=0.80,  # Set based on your requirements
            llm_model="llama3",
            verbose=True
        )
        
        print("✅ LangChain orchestrator integrated!")
    
    def __getattr__(self, name):
        """Delegate all other methods to original orchestrator."""
        return getattr(self.original, name)
    
    def suggest_initial_lambda(self):
        """
        Use LangChain to suggest initial lambda instead of heuristic.
        
        This replaces your existing suggest_initial_lambda() method.
        """
        print("\n🤖 Using LangChain to suggest initial lambda...")
        
        # Initial dummy state (before any training)
        initial_state = {
            "epoch": 0,
            "loss_task": 1.0,
            "loss_adv": 1.0,
            "p_rule": 0.5,  # Assume worst case
            "accuracy": 0.5
        }
        
        # Get LangChain's suggestion
        decision = self.langchain_orchestrator.decide(
            current_state=initial_state,
            current_lambda=[0.5, 0.3, 0.2],  # Default starting point
            current_adv_steps=3
        )
        
        print(f"   Suggested lambda: {decision.lambda_vector}")
        return decision.lambda_vector
    
    def adversarial_training_with_langchain(
        self,
        agent_2,  # ClassifierAgent
        agent_3,  # AdversaryAgent
        max_iterations: int = 10,
        epochs_per_iteration: int = 50
    ):
        """
        Modified adversarial training loop using LangChain orchestrator.
        
        This replaces the core of your _adversarial_training_loop() method.
        The PyTorch training logic (train_adversary, train_learner) remains
        COMPLETELY UNCHANGED.
        """
        print("\n" + "="*70)
        print("ADVERSARIAL TRAINING WITH LANGCHAIN ORCHESTRATOR")
        print("="*70)
        
        # Reset LangChain short-term memory for this run
        self.langchain_orchestrator.reset_short_term_memory()
        
        # Initialize hyperparameters
        current_lambda = self.suggest_initial_lambda()
        current_adv_steps = 3  # Default
        
        # Track initial metrics
        initial_metrics = None
        
        for iteration in range(max_iterations):
            print(f"\n{'─'*70}")
            print(f"### ITERATION {iteration + 1}/{max_iterations} ###")
            print(f"{'─'*70}")
            
            # ============================================================
            # UNCHANGED: Your existing PyTorch training code goes here
            # ============================================================
            # Example structure (adapt to your actual code):
            
            # 1. Train adversary (n_steps = current_adv_steps)
            print(f"   Training adversary ({current_adv_steps} steps)...")
            avg_adv_loss = 0.0
            for step in range(current_adv_steps):
                # Your existing adversary training code
                # adv_loss = agent_3.train_step(...)
                pass
            
            # 2. Train learner (with lambda_vector = current_lambda)
            print(f"   Training learner (lambda={current_lambda})...")
            avg_task_loss = 0.0
            for epoch in range(epochs_per_iteration):
                # Your existing learner training code
                # task_loss = agent_2.train_step(lambda_vector=current_lambda)
                pass
            
            # 3. Evaluate metrics
            print(f"   Evaluating...")
            metrics = self.original.evaluate(agent_2, agent_3)
            # Assuming evaluate() returns dict with 'accuracy' and 'p_rule'
            
            # ============================================================
            # NEW: Consult LangChain orchestrator
            # ============================================================
            
            # Prepare current state
            current_state = {
                "epoch": iteration + 1,
                "loss_task": avg_task_loss,
                "loss_adv": avg_adv_loss,
                "p_rule": metrics.get("p_rule", 0.0),
                "accuracy": metrics.get("accuracy", 0.0)
            }
            
            print(f"\n📊 Current Metrics:")
            print(f"   Accuracy: {current_state['accuracy']:.4f}")
            print(f"   P-rule: {current_state['p_rule']:.4f}")
            
            # Store initial metrics
            if initial_metrics is None:
                initial_metrics = current_state.copy()
            
            # Get LangChain decision
            decision = self.langchain_orchestrator.decide(
                current_state=current_state,
                current_lambda=current_lambda,
                current_adv_steps=current_adv_steps
            )
            
            # Apply decision
            current_lambda = decision.lambda_vector
            current_adv_steps = decision.adv_steps
            
            print(f"\n🔧 Updated Configuration:")
            print(f"   Lambda: {current_lambda}")
            print(f"   Adversary steps: {current_adv_steps}")
            
            # Check stopping criteria
            if decision.stop:
                print(f"\n✅ Training stopped by LangChain orchestrator")
                print(f"   Reason: {decision.reason}")
                
                # Record successful experience
                self.langchain_orchestrator.record_experience(
                    iteration=iteration + 1,
                    initial_metrics=initial_metrics,
                    final_metrics=current_state,
                    lambda_vector=current_lambda,
                    adv_steps=current_adv_steps,
                    converged=True
                )
                break
        else:
            # Max iterations reached
            print(f"\n⚠️ Max iterations reached")
            self.langchain_orchestrator.record_experience(
                iteration=max_iterations,
                initial_metrics=initial_metrics,
                final_metrics=current_state,
                lambda_vector=current_lambda,
                adv_steps=current_adv_steps,
                converged=False
            )
        
        print("\n" + "="*70)
        print("ADVERSARIAL TRAINING COMPLETE")
        print("="*70)
        
        return current_state


# ============================================================================
# STEP 3: Minimal Integration in Your Notebook
# ============================================================================

"""
In your Jupyter notebook, modify the main execution block:

# BEFORE (original code):
orchestrator = LLMOrchestrator(
    data_file="adult.data",
    target_column="income",
    columns_to_ignore=["fnlwgt"]
)
orchestrator.set_global_objective(...)
orchestrator.run_full_pipeline(classifier_agent, adversary_agent)


# AFTER (with LangChain):
original_orchestrator = LLMOrchestrator(
    data_file="adult.data",
    target_column="income",
    columns_to_ignore=["fnlwgt"]
)
original_orchestrator.set_global_objective(...)

# Wrap with LangChain
orchestrator = LLMOrchestratorWithLangChain(original_orchestrator)

# Use new training method
orchestrator.adversarial_training_with_langchain(
    agent_2=classifier_agent,
    agent_3=adversary_agent,
    max_iterations=10,
    epochs_per_iteration=50
)
"""


# ============================================================================
# STEP 4: Alternative - Minimal Patch
# ============================================================================

def patch_existing_orchestrator(orchestrator_instance):
    """
    MINIMAL modification: Just replace the lambda update logic.
    
    This function patches your existing orchestrator instance to use
    LangChain for lambda decisions while keeping everything else the same.
    
    Usage:
        orchestrator = LLMOrchestrator(...)
        patch_existing_orchestrator(orchestrator)
        # Now orchestrator.run_full_pipeline() will use LangChain
    """
    from langchain_orchestrator import create_orchestrator
    
    # Create LangChain agent
    lc_orchestrator = create_orchestrator(
        memory_file=orchestrator_instance.memory_file,
        target_p_rule=orchestrator_instance.global_objective["min_prule"] / 100.0,
        verbose=True
    )
    
    # Store as attribute
    orchestrator_instance._langchain = lc_orchestrator
    
    # Save original method
    original_update_lambda = orchestrator_instance._update_lambda_vector
    
    # Replace with LangChain version
    def langchain_update_lambda(self, metrics_dict, current_lambda):
        """Use LangChain instead of heuristic."""
        
        current_state = {
            "epoch": self.total_epochs_run,
            "loss_task": metrics_dict.get("task_loss", 0.0),
            "loss_adv": metrics_dict.get("adv_loss", 0.0),
            "p_rule": metrics_dict.get("p_rule_min", 0.5),
            "accuracy": metrics_dict.get("accuracy", 0.5)
        }
        
        decision = self._langchain.decide(
            current_state=current_state,
            current_lambda=current_lambda,
            current_adv_steps=3  # Or extract from your config
        )
        
        return decision.lambda_vector, decision.stop
    
    # Monkey-patch the method
    orchestrator_instance._update_lambda_vector = lambda *args, **kwargs: langchain_update_lambda(orchestrator_instance, *args, **kwargs)
    
    print("✅ Orchestrator patched with LangChain!")
    return orchestrator_instance


# ============================================================================
# EXAMPLE USAGE IN NOTEBOOK
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("INTEGRATION GUIDE SUMMARY")
    print("="*70)
    print("""
    Three integration options:
    
    1. FULL REPLACEMENT:
       - Create LLMOrchestratorWithLangChain
       - Use adversarial_training_with_langchain()
       - Most control, cleanest separation
    
    2. MINIMAL PATCH:
       - Use patch_existing_orchestrator()
       - Keeps all existing code
       - Just replaces lambda updates
    
    3. CUSTOM HYBRID:
       - Use langchain_orchestrator.decide() directly
       - Call it where you currently update lambda
       - Maximum flexibility
    
    Recommended: Start with option 2 (minimal patch) for fastest results.
    """)
    print("="*70)
