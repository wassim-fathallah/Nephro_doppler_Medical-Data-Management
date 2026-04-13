"""
Final Validation Script
Quick check to ensure all components are working before running full training.
"""

import sys

print("="*70)
print("FINAL VALIDATION - CHECKING ALL COMPONENTS")
print("="*70)

# 1. Check Ollama
print("\n1. Checking Ollama...")
try:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.1", temperature=0.1, timeout=10)
    response = llm.invoke("Say OK")
    print(f"   [OK] Ollama is working: {response.content[:30]}")
    ollama_ok = True
except Exception as e:
    print(f"   [ERROR] Ollama failed: {e}")
    ollama_ok = False

# 2. Check PyTorch and models
print("\n2. Checking PyTorch models...")
try:
    import torch
    from integration_example import MockClassifier, MockAdversary
    
    classifier = MockClassifier(20)
    adversary = MockAdversary(2)
    
    # Test forward pass
    x = torch.randn(32, 20)
    hidden = classifier.extract_features(x)
    output = classifier(x)
    adv_out = adversary(hidden)
    
    print(f"   [OK] Models work correctly")
    print(f"        Classifier: {x.shape} -> {hidden.shape} -> {output.shape}")
    print(f"        Adversary: {hidden.shape} -> {adv_out.shape}")
    models_ok = True
except Exception as e:
    print(f"   [ERROR] Models failed: {e}")
    models_ok = False

# 3. Check LangChain Orchestrator
print("\n3. Checking LangChain Orchestrator...")
try:
    from langchain_orchestrator import create_orchestrator
    
    orchestrator = create_orchestrator(
        memory_file="validation_test.json",
        verbose=False
    )
    print(f"   [OK] Orchestrator created successfully")
    orchestrator_ok = True
except Exception as e:
    print(f"   [ERROR] Orchestrator failed: {e}")
    import traceback
    traceback.print_exc()
    orchestrator_ok = False

# 4. Check training functions
print("\n4. Checking training functions...")
try:
    from integration_example import train_adversary_step
    from torch.utils.data import DataLoader
    
    X = torch.randn(100, 20)
    y = torch.randint(0, 2, (100,)).float()
    Z = torch.randint(0, 2, (100, 2)).float()
    
    dataset = torch.utils.data.TensorDataset(X, y, Z)
    loader = DataLoader(dataset, batch_size=32)
    
    optimizer = torch.optim.Adam(adversary.parameters(), lr=1e-3)
    
    loss = train_adversary_step(
        adversary=adversary,
        classifier=classifier,
        train_loader=loader,
        adv_optimizer=optimizer,
        n_steps=1,
        device=torch.device("cpu")
    )
    
    print(f"   [OK] Training functions work (loss: {loss:.4f})")
    training_ok = True
except Exception as e:
    print(f"   [ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
    training_ok = False

# Summary
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

results = {
    "Ollama LLM": ollama_ok,
    "PyTorch Models": models_ok,
    "LangChain Orchestrator": orchestrator_ok,
    "Training Functions": training_ok
}

for component, status in results.items():
    status_str = "[OK]" if status else "[FAILED]"
    print(f"{component:25s}: {status_str}")

all_ok = all(results.values())
print("\n" + "="*70)
if all_ok:
    print("[SUCCESS] All components are working!")
    print("\nYou can now run:")
    print("  python run_integration_test.py")
    print("or:")
    print("  python integration_example.py")
else:
    print("[WARNING] Some components failed. Check errors above.")
print("="*70)
