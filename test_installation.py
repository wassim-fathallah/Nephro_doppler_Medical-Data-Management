"""
Installation Validation Script
Run this to verify everything is set up correctly before integrating.
"""

import sys
import importlib
from pathlib import Path


OLLAMA_MODELS_TO_TRY = ["llama3.1", "llama3"]

def test_imports():
    """Test all required imports."""
    print("\n" + "="*70)
    print("TESTING PYTHON IMPORTS")
    print("="*70)
    
    required_modules = [
        ("torch", "PyTorch"),
        ("langchain", "LangChain"),
        ("langchain_core", "LangChain Core"),
        ("langchain_ollama", "LangChain Ollama"),
        ("langchain_community", "LangChain Community"),
        ("pydantic", "Pydantic"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn")
    ]
    
    failed = []
    
    for module_name, display_name in required_modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {display_name:20s} - version {version}")
        except ImportError as e:
            print(f"❌ {display_name:20s} - NOT FOUND")
            failed.append(module_name)
    
    if failed:
        print(f"\n⚠️  Missing modules: {', '.join(failed)}")
        print("Install with: pip install " + " ".join(failed))
        return False
    
    print("\n✅ All Python dependencies installed!")
    return True


def test_ollama():
    """Test Ollama availability."""
    print("\n" + "="*70)
    print("TESTING OLLAMA LLM")
    print("="*70)
    
    try:
        from langchain_ollama import ChatOllama

        for model_name in OLLAMA_MODELS_TO_TRY:
            print(f"Attempting to connect to Ollama ({model_name} model)...")
            llm = ChatOllama(model=model_name, temperature=0.1)
            try:
                # Simple test query
                response = llm.invoke("Say 'Hello World' in JSON format: {\"message\": \"...\"}")
                print(f"✅ Ollama responded with {model_name}: {response.content[:100]}...")
                return True
            except Exception as model_error:
                if "status code 404" in str(model_error):
                    continue
                raise

        print("❌ Ollama connection failed: none of the expected models were found")
        print(f"   Models tried: {', '.join(OLLAMA_MODELS_TO_TRY)}")
        print("   Install one with: ollama pull llama3.1")
        return False
        
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Install Ollama from https://ollama.ai")
        print("2. Run: ollama pull llama3.1")
        print("3. Ensure Ollama is running (check system tray)")
        return False


def test_orchestrator():
    """Test LangChain orchestrator creation."""
    print("\n" + "="*70)
    print("TESTING LANGCHAIN ORCHESTRATOR")
    print("="*70)
    
    try:
        from langchain_orchestrator import create_orchestrator
        
        print("Creating test orchestrator...")
        orchestrator = create_orchestrator(
            memory_file="test_memory_temp.json",
            target_p_rule=0.80,
            target_accuracy=0.80,
            llm_model="llama3.1",
            verbose=False
        )
        
        print("✅ Orchestrator created successfully!")
        
        # Test decision making
        print("\nTesting decision making...")
        test_state = {
            "epoch": 1,
            "loss_task": 0.5,
            "loss_adv": 0.3,
            "p_rule": 0.65,
            "accuracy": 0.82
        }
        
        decision = orchestrator.decide(
            current_state=test_state,
            current_lambda=[0.5, 0.3, 0.2],
            current_adv_steps=3
        )
        
        print(f"✅ Decision received:")
        print(f"   Lambda: {decision.lambda_vector}")
        print(f"   Adv steps: {decision.adv_steps}")
        print(f"   Stop: {decision.stop}")
        print(f"   Reason: {decision.reason}")
        
        # Clean up temp file
        Path("test_memory_temp.json").unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_files():
    """Verify all required files exist."""
    print("\n" + "="*70)
    print("TESTING FILE STRUCTURE")
    print("="*70)
    
    required_files = [
        "langchain_orchestrator.py",
        "integration_adapter.py",
        "integration_example.py",
        "requirements_langchain.txt",
        "README_LANGCHAIN.md",
        "QUICK_REFERENCE.md",
        "START_HERE.md"
    ]
    
    missing = []
    cwd = Path.cwd()
    
    for filename in required_files:
        filepath = cwd / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✅ {filename:35s} ({size:,} bytes)")
        else:
            print(f"❌ {filename:35s} - MISSING")
            missing.append(filename)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    
    print("\n✅ All required files present!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("LANGCHAIN ORCHESTRATOR - INSTALLATION VALIDATION")
    print("="*70)
    print("\nThis script will verify your setup is ready to use.")
    
    results = {
        "Files": test_files(),
        "Python Imports": test_imports(),
        "Ollama LLM": test_ollama(),
        "Orchestrator": test_orchestrator()
    }
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Your system is ready to use the LangChain orchestrator!")
        print("\nNext steps:")
        print("1. Read QUICK_REFERENCE.md for copy-paste integration code")
        print("2. Run: python integration_example.py (optional demo)")
        print("3. Integrate into your notebook using one of the 3 methods")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Missing Python packages: pip install -r requirements_langchain.txt")
        print("- Ollama issues: Install from https://ollama.ai, then run 'ollama pull llama3.1'")
        print("- Missing files: Re-download the complete package")
    
    print("\n" + "="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
