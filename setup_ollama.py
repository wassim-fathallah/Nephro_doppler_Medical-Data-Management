"""
Ollama Setup and Configuration Helper
This script verifies Ollama setup and pulls required models.
"""

import subprocess
import sys

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        print(f"✅ Ollama is installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ Ollama is NOT installed")
        print("\nInstallation steps:")
        print("1. Visit: https://ollama.ai/download")
        print("2. Download and install for Windows")
        print("3. Restart your terminal")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def list_available_models():
    """List all available Ollama models."""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        print("\n📦 Available Ollama models:")
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("   (No models installed)")
        return result.stdout
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return ""

def pull_model(model_name: str):
    """Pull a model from Ollama registry."""
    print(f"\n📥 Pulling model: {model_name}")
    print("   This may take several minutes (models are typically 4-7GB)...")
    
    try:
        # Use Popen to show progress
        process = subprocess.Popen(
            ['ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(f"   {line.rstrip()}")
        
        process.wait()
        
        if process.returncode == 0:
            print(f"✅ Successfully pulled {model_name}")
            return True
        else:
            print(f"❌ Failed to pull {model_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False

def test_model(model_name: str):
    """Test if a model works."""
    print(f"\n🧪 Testing model: {model_name}")
    
    try:
        result = subprocess.run(
            ['ollama', 'run', model_name, 'Say "OK" in one word.'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"✅ Model {model_name} works!")
            print(f"   Response: {result.stdout.strip()[:100]}")
            return True
        else:
            print(f"❌ Model {model_name} failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⚠️  Model {model_name} timed out (may still work)")
        return False
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def main():
    """Main setup process."""
    print("="*70)
    print("OLLAMA SETUP AND CONFIGURATION")
    print("="*70)
    
    # Step 1: Check if Ollama is installed
    if not check_ollama_installed():
        print("\n❌ Please install Ollama first, then run this script again.")
        return 1
    
    # Step 2: List available models
    models_output = list_available_models()
    
    # Step 3: Check for required models
    required_models = ['llama3.1']
    available_models = []
    
    for model in required_models:
        if model in models_output or f"{model}:latest" in models_output:
            available_models.append(model)
            print(f"✅ Required model '{model}' is available")
        else:
            print(f"⚠️  Required model '{model}' is NOT available")
    
    # Step 4: Offer to pull missing models
    missing_models = [m for m in required_models if m not in available_models]
    
    if missing_models:
        print(f"\n⚠️  Missing models: {', '.join(missing_models)}")
        response = input("\nWould you like to pull the missing models? (y/n): ").strip().lower()
        
        if response == 'y':
            for model in missing_models:
                success = pull_model(model)
                if success:
                    available_models.append(model)
        else:
            print("Skipping model download.")
    
    # Step 5: Test at least one model
    if available_models:
        print("\n" + "="*70)
        print("TESTING MODEL")
        print("="*70)
        test_model(available_models[0])
    
    # Step 6: Summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    
    if available_models:
        print(f"✅ Ollama is configured with models: {', '.join(available_models)}")
        print("\n✅ You can now run: python integration_example.py")
        return 0
    else:
        print("❌ No models are available. Please pull at least one model:")
        print("   ollama pull llama3.1")
        return 1

if __name__ == "__main__":
    sys.exit(main())
