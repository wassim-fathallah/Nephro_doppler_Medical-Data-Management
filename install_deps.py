#!/usr/bin/env python
"""Install missing dependencies"""

import subprocess
import sys

print("Installing LangChain dependencies...")

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_langchain.txt", "-q"], 
                         timeout=120)
    print("✅ LangChain installed successfully")
    
    # Verify
    import langchain
    print(f"✅ Verified: langchain {langchain.__version__}")
    
except subprocess.TimeoutExpired:
    print("⚠️  Installation timed out")
    sys.exit(1)
except Exception as e:
    print(f"❌ Installation failed: {e}")
    sys.exit(1)

print("\n✅ All dependencies ready! You can now run: python train_final_with_llm.py")
