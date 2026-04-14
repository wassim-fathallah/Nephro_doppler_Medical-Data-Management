"""
LangChain tool + chain: decide_lambda

The LLM reads:
  - Short-term memory (current run trajectory: all past iterations)
  - Long-term memory (past successful runs on same dataset/attrs)
  - Current metrics (p_rules, accuracy, adversary_loss)
  - Current lambda vector

And decides the NEXT lambda vector with reasoning.

This replaces the fixed momentum formula from the original code with
an LLM-driven adaptive control strategy.

Also exposed as a standalone LCEL chain (lambda_chain) for direct calls
inside the training loop without going through the full ReAct agent.
"""

import json
import re
import numpy as np
from typing import List, Dict, Any

from langchain.tools import tool
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import state
from memory.short_term import ShortTermMemory

llm = OllamaLLM(model="llama3.1", temperature=0.2)


# ─────────────────────────────────────────────────────────────────────────────
# LCEL Lambda Decision Chain
# Used directly in the training loop (not via ReAct agent overhead)
# ─────────────────────────────────────────────────────────────────────────────

LAMBDA_PROMPT = PromptTemplate.from_template("""You are a fairness optimization controller for an adversarial machine learning system.

The system penalizes the classifier with: L_total = L_task - λ * L_adversary
Higher λ → more fairness pressure (classifier hides sensitive info better) but may reduce accuracy.
Each sensitive attribute has its own λ.

=== CURRENT STATE (iteration {iteration}/{max_iterations}) ===
Sensitive attributes: {sensitive_attrs}
Current λ vector: {current_lambda}
Current metrics:
  - Accuracy: {accuracy:.4f}
  - P-rule per attribute: {p_rules}
  - Adversary loss: {adv_loss:.4f}
  - Target P-rule threshold: {threshold}%

=== SHORT-TERM MEMORY (this run's trajectory) ===
{short_term_history}

=== YOUR TASK ===
Decide the next λ vector. Rules:
- If P-rule for an attribute is far below {threshold}%, increase its λ aggressively
- If P-rule is near threshold but stalling, increase λ moderately
- If P-rule is above threshold, you can decrease λ slightly to recover accuracy
- If accuracy drops below 0.70, consider decreasing all λ by 10-20%
- Max allowed λ: {lambda_max}
- λ must always be >= 0

Respond with ONLY this JSON:
{{
  "reasoning": "2-3 sentence explanation of your decision",
  "new_lambda": [{{"attr": "attr_name", "value": 1.5}}, ...]
}}
""")

lambda_chain = LAMBDA_PROMPT | llm | StrOutputParser()


def _extract_lambda_json(text: str, n_attrs: int, attr_names: List[str]) -> List[float]:
    """Parse LLM response into a list of lambda values."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            lambdas_raw = parsed.get("new_lambda", [])
            reasoning = parsed.get("reasoning", "")
            if reasoning:
                print(f"  [λ-LLM] {reasoning}")
            # Support both list of floats and list of {attr, value} dicts
            if lambdas_raw and isinstance(lambdas_raw[0], dict):
                result = []
                for attr in attr_names:
                    entry = next((x for x in lambdas_raw if x.get("attr") == attr), None)
                    result.append(float(entry["value"]) if entry else state.lambda_vector[len(result)])
                return result
            elif lambdas_raw:
                return [float(x) for x in lambdas_raw[:n_attrs]]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    # Fallback: bump all lambdas by 10% if parsing fails
    return [min(l * 1.1, 20.0) for l in state.lambda_vector]


def decide_lambda_for_iteration(
    short_term: ShortTermMemory,
    current_metrics: Dict[str, Any],
    iteration: int,
    max_iterations: int,
    lambda_max: float = 20.0,
) -> List[float]:
    """
    Core function called inside the training loop.
    Invokes the LCEL chain and returns the new lambda vector.
    """
    response = lambda_chain.invoke({
        "iteration": iteration,
        "max_iterations": max_iterations,
        "sensitive_attrs": state.sensitive_attrs,
        "current_lambda": state.lambda_vector,
        "accuracy": current_metrics["accuracy"],
        "p_rules": current_metrics["p_rules"],
        "adv_loss": current_metrics.get("adversary_loss", 0.0),
        "threshold": state.p_rule_threshold,
        "short_term_history": short_term.to_prompt_str(n=6),
        "lambda_max": lambda_max,
    })

    new_lambda = _extract_lambda_json(response, len(state.sensitive_attrs), state.sensitive_attrs)
    # Clip to valid range
    new_lambda = [float(np.clip(l, 0.0, lambda_max)) for l in new_lambda]
    return new_lambda


# ─────────────────────────────────────────────────────────────────────────────
# LangChain Tool version (used by ReAct agent for initial lambda suggestion)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def decide_initial_lambda(n_sensitive: int = 2) -> str:
    """
    Decides the INITIAL lambda vector before training starts.
    Uses only dataset schema (name, target, sensitive attrs, size) — no past runs.
    Returns a JSON string with the initial lambda values and reasoning.
    Call this exactly ONCE before pretraining.
    """
    if state.lambda_vector:
        return json.dumps({"initial_lambda": state.lambda_vector, "reasoning": "Already set, returning cached."}, indent=2)

    n_sensitive = len(state.sensitive_attrs) if state.sensitive_attrs else n_sensitive

    # Build dataset size info from state if data is already loaded
    size_info = ""
    if state.X_train is not None:
        n_samples = state.X_train.shape[0]
        n_features = state.X_train.shape[1]
        size_info = f"Training samples: {n_samples}, Features: {n_features}"

    prompt = f"""You are initializing lambda (λ) values for adversarial fairness training.

The system trains a classifier with a fairness penalty:
  L_total = L_task - λ * L_adversary
Higher λ → stronger pressure to hide sensitive attributes from the classifier's embeddings.
Each sensitive attribute has its own λ. They will be adapted at every training iteration.

=== DATASET SCHEMA ===
Dataset: {state.dataset_name}
Target column: {state.target_col}
Sensitive attributes: {state.sensitive_attrs}
{size_info}

=== YOUR TASK ===
Choose a conservative initial λ vector. This is just the starting point — the LLM will
adjust λ at every iteration based on real metrics. Rules:
- Start small (0.1–1.0). Too high risks destroying accuracy before training stabilizes.
- If a sensitive attribute is known to be highly correlated with the target in this dataset
  (e.g. sex in adult income, race in COMPAS), you can start slightly higher (0.5–1.0).
- All λ values must be >= 0.

Respond with ONLY this JSON:
{{"reasoning": "2-3 sentence explanation", "initial_lambda": [0.3, 0.2]}}
"""
    response = llm.invoke(prompt)
    try:
        parsed = json.loads(re.search(r"\{.*\}", response, re.DOTALL).group())
        lambdas = [float(x) for x in parsed["initial_lambda"]][:n_sensitive]
        reasoning = parsed.get("reasoning", "")
    except Exception:
        lambdas = [0.1] * n_sensitive
        reasoning = "LLM parsing failed — falling back to default small λ=0.1."

    state.lambda_vector = lambdas
    result = {"initial_lambda": lambdas, "reasoning": reasoning}
    print(f"[λ-init] {lambdas} — {reasoning}")
    return json.dumps(result, indent=2)
