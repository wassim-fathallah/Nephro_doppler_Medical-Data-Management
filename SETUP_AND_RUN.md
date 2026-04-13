# 🔧 SETUP GUIDE: Install Dependencies & Run Training

## Current Status

✅ Training script ready: `train_final_with_llm.py`
✅ LLM orchestrator ready: `orchestrator_llm.py`
❌ LangChain not installed yet

## Step-by-Step Fix

### Step 1: Install Dependencies

**Option A: Using Batch File (Windows, Easiest)**
```
Double-click: install_dependencies.bat
Wait for completion
```

**Option B: Using Command Prompt**
```
Open Command Prompt
Run: pip install langchain-community langchain torch numpy pandas scikit-learn
Wait for "Successfully installed" message
```

**Option C: Using Python**
```
Open Command Prompt
Run: python install_deps.py
```

### Step 2: Verify Installation

```
python -c "import langchain; print('✅ LangChain installed')"
```

Should output: `✅ LangChain installed`

### Step 3: Ensure Ollama is Running

In a separate Command Prompt or PowerShell:
```
ollama serve
```

Wait for: "Listening on 127.0.0.1:11434"

### Step 4: Run Training

```
cd c:\Users\MSI\Desktop\MEMOIRE
python train_final_with_llm.py
```

---

## Installation Details

### What Gets Installed

| Package | Purpose |
|---------|---------|
| `langchain-community` | LLM integrations (including Ollama) |
| `langchain` | Core LangChain framework |
| `torch` | PyTorch (deep learning framework) |
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scikit-learn` | Machine learning utilities |

### Installation Time

- **First install**: 2-5 minutes (downloads packages)
- **Already installed**: 10 seconds (verification only)
- **Size**: ~500MB total

---

## Troubleshooting Installation

### Error: "pip: command not found"

Make sure Python is in PATH:
```
python -m pip install langchain-community langchain
```

### Error: "Unable to establish SSL connection"

Network/proxy issue. Try:
```
pip install --trusted-host pypi.python.org langchain-community langchain
```

### Error: "ModuleNotFoundError: No module named 'langchain'"

Installation didn't work. Retry:
```
python -m pip install --upgrade langchain-community langchain
```

### Installation takes too long

Normal on slow connections. Just wait. Or try a specific index:
```
pip install -i https://pypi.org/simple/ langchain-community langchain
```

---

## Before Running Training

Check all prerequisites:

```bash
# 1. Python version
python --version           # Should be 3.8+

# 2. LangChain installed
python -c "import langchain; print('OK')"

# 3. PyTorch installed  
python -c "import torch; print(torch.__version__)"

# 4. Ollama running
curl http://localhost:11434  # Should respond

# 5. Ollama model available
ollama list | grep llama3.1  # Should show llama3.1
```

If all show `OK` or similar, you're ready!

---

## Running Training

### Command
```bash
cd c:\Users\MSI\Desktop\MEMOIRE
python train_final_with_llm.py
```

### Expected Output (First Few Lines)
```
Loading dataset from adult.data...
  Train: 26048 samples
  Test: 6513 samples

[INIT] Models initialized
  Classifier params: 2337
  Adversary params: 1186
  Device: cpu
  
[LLM-ORCHESTRATOR] Initializing llama3.1...
✓ LLM connection successful
```

### What to Expect During Training

- Phase 1: "Pre-training classifier" - ~30 seconds
- Phase 2: "Pre-training adversary" - ~30 seconds
- Phase 3: Adaptive loop - Multiple steps until convergence
  - Each step shows: [METRICS], [ORCHESTRATION], [LLM] output
  - Look for: Accuracy and P-rule metrics
  - Success: `✅ CONVERGED` message

---

## Monitoring Progress

### Key Metrics to Watch

```
[METRICS]
  Accuracy: 0.7300 (target: >=0.80)    ← Getting closer?
  P-rules: {'race': 0.78, 'sex': 0.75} ← Both improving?

[LLM] Response:
  [LLM reasoning about decision]       ← What's it thinking?

[LLM-ORCHESTRATOR] Step X:
  Old λ: ['0.500', '0.500']            ← Previous weights
  New λ: ['0.650', '0.620']            ← LLM's decision
```

### Success Indicator

When you see:
```
✅ CONVERGED at step 25!
   Final Accuracy: 0.8120
   Final P-rules: {'race': 0.8150, 'sex': 0.8320}
```

Both targets (≥0.80) have been met! 🎉

---

## After Training Completes

### Check Results
- Final accuracy and p-rule shown in console
- Should both be ≥ 0.80 (80%)

### Analyze LLM Learning
```bash
# View memory of LLM decisions
cat orchestrator_training.json
```

Shows all training steps and what LLM decided.

### Save Results
```bash
# Manually copy or rename if desired
copy orchestrator_training.json orchestrator_training_backup.json
```

---

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| LangChain import error | Run: `pip install langchain-community` |
| Ollama connection error | Make sure: `ollama serve` is running |
| Model not found | Run: `ollama pull llama3.1` |
| Out of memory | Reduce: `batch_size=16` in script |
| Training very slow | This is normal on CPU; use GPU or reduce `epochs_per_step` |

---

## Next Commands to Run

```bash
# 1. Install dependencies (one time)
pip install langchain-community langchain torch numpy pandas scikit-learn

# 2. Make sure Ollama is running (in another terminal)
ollama serve

# 3. Run training
cd c:\Users\MSI\Desktop\MEMOIRE
python train_final_with_llm.py
```

---

## Ready?

1. ✅ Install dependencies (use batch file or pip)
2. ✅ Verify with: `python -c "import langchain; print('OK')"`
3. ✅ Start Ollama: `ollama serve` (in another terminal)
4. ✅ Run training: `python train_final_with_llm.py`

**Then watch your LLM orchestrate the training!** 🚀
