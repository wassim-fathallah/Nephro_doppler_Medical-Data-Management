#!/usr/bin/env python3
"""Quick test of the improved orchestrator."""

from orchestrator_llm import LLMOrchestrator

print("[TEST] Initializing LLM Orchestrator...")
try:
    orchestrator = LLMOrchestrator(model_name="llama3.1", temperature=0.1)
    print("[OK] Orchestrator initialized")
except Exception as e:
    print(f"[ERROR] Failed to initialize: {e}")
    exit(1)

# Test decision making
print("\n[TEST] Testing orchestrator decision making...")

test_cases = [
    {
        "name": "P-rule critically low",
        "accuracy": 0.82,
        "p_rule_values": {"race": 0.65, "sex": 0.60},
        "current_lambda": [0.5, 0.5],
        "current_momentum": [0.0, 0.0],
    },
    {
        "name": "P-rule improving",
        "accuracy": 0.81,
        "p_rule_values": {"race": 0.78, "sex": 0.75},
        "current_lambda": [0.6, 0.6],
        "current_momentum": [0.1, 0.1],
    },
    {
        "name": "Both targets close",
        "accuracy": 0.81,
        "p_rule_values": {"race": 0.82, "sex": 0.80},
        "current_lambda": [0.7, 0.7],
        "current_momentum": [0.2, 0.2],
    },
]

for test in test_cases:
    print(f"\nTest: {test['name']}")
    print(f"  Input: Accuracy={test['accuracy']:.3f}, P-rule={min(test['p_rule_values'].values()):.3f}")
    
    try:
        new_lambda, new_momentum, reasoning = orchestrator.decide(
            step=1,
            accuracy=test['accuracy'],
            p_rule_values=test['p_rule_values'],
            current_lambda=test['current_lambda'],
            current_momentum=test['current_momentum'],
            target_accuracy=0.80,
            target_p_rule=0.80,
            lambda_max=2.0
        )
        
        print(f"  Output: λ={[f'{x:.3f}' for x in new_lambda]}")
        print(f"  Reasoning: {reasoning}")
        
    except Exception as e:
        print(f"  [ERROR] {e}")

print("\n[SUCCESS] Orchestrator test completed!")
