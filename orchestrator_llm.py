"""
LangChain-based Orchestrator for Adaptive Lambda Learning

This module provides a LangChain meta-controller for adversarial fairness
training. It keeps the existing external training-loop interface used by
train_final_with_llm.py, but replaces the old direct Ollama prompting with:
- LangChain prompt composition
- llama3.1 via ChatOllama
- short-term memory
- long-term JSON experience memory
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator

from langchain_core.prompts import PromptTemplate

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None

try:
    from langchain_community.llms import Ollama as CommunityOllama
except Exception:
    CommunityOllama = None


class OrchestratorDecision(BaseModel):
    """Structured output from the LangChain orchestrator."""

    adjustment: float = Field(
        description="Single lambda adjustment applied to all lambda values",
    )
    reasoning: str = Field(
        description="Brief explanation of the decision",
    )

    @validator("adjustment")
    def validate_adjustment(cls, value: float) -> float:
        if not (-1.5 <= value <= 1.5):
            raise ValueError(f"Adjustment {value} out of bounds [-1.5, 1.5]")
        return value


class LLMOrchestratorMemory:
    """Store and retrieve past training experiences for learning."""

    def __init__(self, memory_file: str = "orchestrator_memory.json"):
        self.memory_file = Path(memory_file)
        self.memory: List[Dict[str, Any]] = []
        self._load_memory()

    def _load_memory(self) -> None:
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r", encoding="utf-8") as handle:
                    self.memory = json.load(handle)
                print(f"[LLM-ORCHESTRATOR] Loaded {len(self.memory)} past experiences from memory")
            except Exception:
                self.memory = []
        else:
            self.memory = []
            print("[LLM-ORCHESTRATOR] Starting with fresh memory")

    def _save_memory(self) -> None:
        with open(self.memory_file, "w", encoding="utf-8") as handle:
            json.dump(self.memory, handle, indent=2)

    def add_experience(
        self,
        step: int,
        accuracy: float,
        p_rule_dict: Dict[str, float],
        lambda_old: List[float],
        lambda_new: List[float],
        success: bool,
    ) -> None:
        experience = {
            "timestamp": datetime.now().isoformat(),
            "step": int(step),
            "accuracy": float(accuracy),
            "min_p_rule": float(min(p_rule_dict.values())) if p_rule_dict else 0.0,
            "p_rules": {str(key): float(value) for key, value in p_rule_dict.items()},
            "lambda_old": [float(value) for value in lambda_old],
            "lambda_new": [float(value) for value in lambda_new],
            "success": bool(success),
        }
        self.memory.append(experience)
        self._save_memory()

    def get_similar_experiences(self, accuracy: float, min_p_rule: float, topk: int = 3) -> List[Dict[str, Any]]:
        if not self.memory:
            return []

        similar: List[Tuple[float, Dict[str, Any]]] = []
        for experience in self.memory:
            acc_diff = abs(experience.get("accuracy", 0.0) - accuracy)
            p_rule_diff = abs(experience.get("min_p_rule", 0.0) - min_p_rule)
            distance = acc_diff + p_rule_diff
            if distance < 0.15:
                similar.append((distance, experience))

        similar.sort(key=lambda item: item[0])
        return [experience for _, experience in similar[:topk]]

    def format_memory_context(self, topk: int = 3) -> str:
        if not self.memory:
            return "No past experiences available."

        successful = [experience for experience in self.memory[-10:] if experience.get("success", False)]
        if not successful:
            successful = self.memory[-topk:]

        context_lines = ["Past training experiences:"]
        for index, experience in enumerate(successful[-topk:], 1):
            context_lines.append(f"{index}. Step {experience.get('step', '?')}")
            context_lines.append(f"   - Accuracy: {experience.get('accuracy', 0.0):.3f}")
            context_lines.append(f"   - Min P-rule: {experience.get('min_p_rule', 0.0):.3f}")
            context_lines.append(f"   - Lambda old: {experience.get('lambda_old', [])}")
            context_lines.append(f"   - Lambda new: {experience.get('lambda_new', [])}")

        return "\n".join(context_lines)


class LLMOrchestrator:
    """Orchestrator that uses llama3.1 via LangChain to make lambda decisions."""

    def __init__(
        self,
        model_name: str = "llama3.1",
        temperature: float = 0.3,
        memory_file: str = "orchestrator_memory.json",
        target_p_rule: float = 0.80,
        target_accuracy: float = 0.80,
        short_term_window: int = 5,
    ):
        self.memory = LLMOrchestratorMemory(memory_file=memory_file)
        self.momentum_beta = 0.7
        self.target_p_rule = target_p_rule
        self.target_accuracy = target_accuracy
        self.short_term_window = short_term_window
        self.short_term_memory: List[Dict[str, Any]] = []

        if ChatOllama is None and CommunityOllama is None:
            raise ImportError(
                "LangChain Ollama not available. Install with: pip install langchain-ollama langchain-community langchain"
            )

        llm_cls = ChatOllama if ChatOllama is not None else CommunityOllama
        print(f"[LLM-ORCHESTRATOR] Initializing Ollama with model: {model_name}")
        try:
            self.llm = llm_cls(model=model_name, temperature=temperature, format="json")
        except TypeError:
            self.llm = llm_cls(model=model_name, temperature=temperature)

        self.prompt = self._build_prompt_template()
        self.chain = self.prompt | self.llm
        self._test_llm()

    def _test_llm(self) -> None:
        try:
            result = self.chain.invoke(
                {
                    "step": 0,
                    "accuracy": 0.0,
                    "p_rule": 0.0,
                    "current_lambda": [0.1],
                    "current_momentum": [0.0],
                    "target_p_rule": 0.8,
                    "target_accuracy": 0.8,
                    "similar_experiences": "No past experiences available.",
                    "memory_context": "No past experiences available.",
                    "short_term_history": "No history yet.",
                    "category": "TEST",
                    "decision_hint": "Return a tiny adjustment.",
                }
            )
            _ = self._extract_text(result)
            print("[OK] LLM connection successful")
        except Exception as exc:
            raise RuntimeError(f"LLM test failed: {exc}")

    def _build_prompt_template(self) -> PromptTemplate:
        template = """You are a trained fairness orchestrator making a hyperparameter decision.

METRICS (Step {step}):
- Accuracy: {accuracy:.4f} (target: {target_accuracy:.4f})
- Min P-rule: {p_rule:.4f} (target: {target_p_rule:.4f})
- Current Lambda: {current_lambda}
- Current Momentum: {current_momentum}

SITUATION ANALYSIS: {category}
{decision_hint}

SHORT-TERM HISTORY:
{short_term_history}

RELEVANT PAST EXPERIENCES:
{similar_experiences}

MEMORY CONTEXT:
{memory_context}

OUTPUT INSTRUCTIONS:
Return ONLY valid JSON with these keys:
{{
  "adjustment": <single float between -1.5 and 1.5>,
  "reasoning": "<one sentence>"
}}

RULES:
1. Keep adjustments smooth and gradual.
2. Increase lambda when fairness is below target.
3. Decrease lambda when accuracy is below target and fairness is acceptable.
4. Use memory to avoid repeating bad decisions.
"""
        return PromptTemplate(
            input_variables=[
                "step",
                "accuracy",
                "p_rule",
                "current_lambda",
                "current_momentum",
                "target_p_rule",
                "target_accuracy",
                "similar_experiences",
                "memory_context",
                "short_term_history",
                "category",
                "decision_hint",
            ],
            template=template,
        )

    def _extract_text(self, response: Any) -> str:
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    def _extract_json(self, response_text: str) -> Dict[str, Any]:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON found in response")
        return json.loads(response_text[start : end + 1])

    def _format_short_term_history(self) -> str:
        if not self.short_term_memory:
            return "No history yet."

        history_lines = []
        for state in self.short_term_memory[-self.short_term_window :]:
            history_lines.append(
                f"  Iter {state['step']}: p-rule={state['p_rule']:.3f}, acc={state['accuracy']:.3f}, lambda={state.get('lambda_vector', 'N/A')}"
            )
        return "\n".join(history_lines)

    def _build_decision_context(
        self,
        step: int,
        accuracy: float,
        p_rule_values: Dict[str, float],
        current_lambda: List[float],
        current_momentum: List[float],
        target_p_rule: float,
        target_accuracy: float,
    ) -> Dict[str, Any]:
        min_p_rule = min(p_rule_values.values()) if p_rule_values else 0.0

        if accuracy >= target_accuracy and min_p_rule >= target_p_rule:
            category = "BOTH_MET"
            decision_hint = "Both targets are met. Keep lambda stable with tiny adjustments only."
        elif min_p_rule < 0.70:
            category = "P_CRITICAL"
            decision_hint = "P-rule is critically low. Increase lambda moderately to enforce fairness."
        elif min_p_rule < target_p_rule:
            category = "P_MODERATE"
            decision_hint = "P-rule is below target. Increase lambda gradually."
        elif accuracy < target_accuracy:
            category = "ACC_LOW"
            decision_hint = "Accuracy is below target. Decrease lambda slightly to recover performance."
        else:
            category = "STABLE"
            decision_hint = "Metrics are close to target. Make only tiny adjustments."

        similar_experiences = self.memory.format_memory_context()
        current_state = {
            "step": step,
            "p_rule": min_p_rule,
            "accuracy": accuracy,
            "lambda_vector": current_lambda,
        }
        self.short_term_memory.append(current_state)

        return {
            "step": step,
            "accuracy": accuracy,
            "p_rule": min_p_rule,
            "current_lambda": current_lambda,
            "current_momentum": current_momentum,
            "target_p_rule": target_p_rule,
            "target_accuracy": target_accuracy,
            "similar_experiences": similar_experiences,
            "memory_context": self.memory.format_memory_context(),
            "short_term_history": self._format_short_term_history(),
            "category": category,
            "decision_hint": decision_hint,
        }

    def decide(
        self,
        step: int,
        accuracy: float,
        p_rule_values: Dict[str, float],
        current_lambda: List[float],
        current_momentum: List[float],
        target_p_rule: float = 0.80,
        target_accuracy: float = 0.80,
        lambda_max: float = 20.0,
    ) -> Tuple[List[float], List[float], str]:
        min_p_rule = min(p_rule_values.values()) if p_rule_values else 0.0

        if accuracy >= target_accuracy and min_p_rule >= target_p_rule:
            return current_lambda, current_momentum, f"CONVERGED! Acc={accuracy:.3f}, P-rule={min_p_rule:.3f}"

        prompt_inputs = self._build_decision_context(
            step=step,
            accuracy=accuracy,
            p_rule_values=p_rule_values,
            current_lambda=current_lambda,
            current_momentum=current_momentum,
            target_p_rule=target_p_rule,
            target_accuracy=target_accuracy,
        )

        print(f"\n[LLM-ORCHESTRATOR] Consulting llama3.1 for step {step}...")
        try:
            response = self.chain.invoke(prompt_inputs)
            response_text = self._extract_text(response)
            print(f"[LLM] Response:\n{response_text[:500]}...")
            decision_json = self._extract_json(response_text)
        except Exception as exc:
            print(f"[WARN] LLM error: {exc}. Using fallback strategy.")
            return self._fallback_decision(current_lambda, current_momentum, p_rule_values, target_p_rule, lambda_max)

        try:
            decision = OrchestratorDecision(**decision_json)
        except Exception as exc:
            print(f"[WARN] Failed to parse LLM response: {exc}. Using fallback strategy.")
            return self._fallback_decision(current_lambda, current_momentum, p_rule_values, target_p_rule, lambda_max)

        adjustment = max(-1.5, min(1.5, float(decision.adjustment)))
        reasoning = decision.reasoning

        new_lambda = [max(0.0, min(value + adjustment, lambda_max)) for value in current_lambda]
        new_momentum = [0.7 * momentum + 0.3 * adjustment for momentum in current_momentum]

        success = accuracy >= target_accuracy and min_p_rule >= target_p_rule
        self.memory.add_experience(step, accuracy, p_rule_values, current_lambda, new_lambda, success)

        print(f"[LLM-ORCHESTRATOR] Step {step}: Acc={accuracy:.3f}, min P-rule={min_p_rule:.3f}")
        print(f"  Old λ: {[f'{x:.3f}' for x in current_lambda]}")
        print(f"  New λ: {[f'{x:.3f}' for x in new_lambda]}")
        print(f"  Reasoning: {reasoning[:200]}...")

        return new_lambda, new_momentum, reasoning

    def _fallback_decision(
        self,
        current_lambda: List[float],
        current_momentum: List[float],
        p_rule_values: Dict[str, float],
        target_p_rule: float,
        lambda_max: float,
    ) -> Tuple[List[float], List[float], str]:
        new_lambda = list(current_lambda)
        new_momentum = list(current_momentum)

        for index in range(len(current_lambda)):
            attr_key = list(p_rule_values.keys())[index] if index < len(p_rule_values) else f"attr_{index}"
            current_p_rule = p_rule_values.get(attr_key, 0.5)
            p_rule_gap = target_p_rule - current_p_rule

            if p_rule_gap < -0.10:
                increment = -0.5
            elif p_rule_gap < 0:
                increment = -0.2
            elif p_rule_gap < 0.05:
                increment = 0.1
            elif p_rule_gap < 0.10:
                increment = 0.5
            else:
                increment = 1.0

            new_momentum[index] = 0.85 * new_momentum[index] + 0.15 * increment
            new_lambda[index] = max(0.0, min(new_lambda[index] + new_momentum[index], lambda_max))
            print(f"[FALLBACK] Attr {index}: p-rule={current_p_rule:.3f}, gap={p_rule_gap:.3f}, increment={increment:.3f}")

        return new_lambda, new_momentum, "Fallback: Adaptive strategy (LLM unavailable)"

    def record_experience(
        self,
        iteration: int,
        initial_metrics: Dict[str, Any],
        final_metrics: Dict[str, Any],
        lambda_vector: List[float],
        adv_steps: int,
        converged: bool,
    ) -> None:
        experience = {
            "timestamp": datetime.now().isoformat(),
            "iteration": int(iteration),
            "lambda_vector": [float(value) for value in lambda_vector],
            "adv_steps": int(adv_steps),
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "converged": bool(converged),
            "improvement": {
                "p_rule_delta": final_metrics.get("p_rule", 0) - initial_metrics.get("p_rule", 0),
                "accuracy_delta": final_metrics.get("accuracy", 0) - initial_metrics.get("accuracy", 0),
            },
        }
        self.memory.memory.append(experience)
        self.memory._save_memory()
        print(f"   [SAVED] Experience recorded (total: {len(self.memory.memory)})")

    def reset_short_term_memory(self) -> None:
        self.short_term_memory = []


def create_orchestrator(
    memory_file: str = "orchestrator_memory.json",
    target_p_rule: float = 0.80,
    target_accuracy: float = 0.80,
    llm_model: str = "llama3.1",
    temperature: float = 0.3,
    verbose: bool = True,
) -> LLMOrchestrator:
    """Factory function to create an orchestrator with common defaults."""
    _ = verbose
    return LLMOrchestrator(
        model_name=llm_model,
        temperature=temperature,
        memory_file=memory_file,
        target_p_rule=target_p_rule,
        target_accuracy=target_accuracy,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("LangChain Orchestrator Demo")
    print("=" * 60)

    orchestrator = create_orchestrator(
        memory_file="test_memory.json",
        target_p_rule=0.80,
        target_accuracy=0.85,
    )

    current_lambda = [0.5, 0.3, 0.2]
    current_adv_steps = 3

    for step in range(1, 6):
        current_state = {
            "step": step,
            "accuracy": 0.82 + step * 0.01,
            "p_rule": 0.60 + step * 0.05,
        }

        decision = orchestrator.decide(
            step=step,
            accuracy=current_state["accuracy"],
            p_rule_values={"race": current_state["p_rule"], "sex": current_state["p_rule"]},
            current_lambda=current_lambda,
            current_momentum=[0.0, 0.0, 0.0],
            target_p_rule=0.80,
            target_accuracy=0.85,
        )

        current_lambda, _, _ = decision

    orchestrator.record_experience(
        iteration=1,
        initial_metrics={"p_rule": 0.60, "accuracy": 0.82},
        final_metrics={"p_rule": 0.85, "accuracy": 0.87},
        lambda_vector=current_lambda,
        adv_steps=current_adv_steps,
        converged=True,
    )

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
