import subprocess
import sys

print("Fixing numpy/pandas compatibility issue...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "pandas", "numpy"])
print("✓ Fixed! Try running test_installation.py again.")
