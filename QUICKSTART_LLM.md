# Quick Start: LLM-Orchestrated Training

## ⚡ 30 Second Setup

### 1. Start Ollama (in a terminal)
```bash
ollama serve
```

### 2. In Another Terminal, Pull Model
```bash
ollama pull llama3.1
```

### 3. Install Dependencies
```bash
pip install langchain-community langchain torch numpy pandas scikit-learn
```

### 4. Run Training
```bash
cd c:\Users\MSI\Desktop\MEMOIRE
python train_final_with_llm.py
```

## 📊 What You'll See

The script will:
1. Load Adult dataset (or generate synthetic data)
2. Pre-train classifier for 10 epochs
3. Pre-train adversary for 10 epochs
4. Run adaptive loop where **Llama3.1 decides lambda values** to reach:
   - ✅ Accuracy ≥ 80%
   - ✅ P-rule ≥ 80%

Each step shows:
- Current accuracy & p-rule metrics
- What Llama3.1 decided (lambda adjustments)
- Why (reasoning from LLM)

## 🎯 What's Different

### Old Approach (integration_example.py)
- Fixed hardcoded heuristic for lambda updates
- Result: Only reached ~52% accuracy, 86% p-rule
- **Problem**: Couldn't balance both metrics

### New Approach (train_final_with_llm.py)
- **LLM-powered orchestrator** makes intelligent decisions
- Learns from memory of past steps
- Balances accuracy & fairness simultaneously
- **Target**: ≥ 80% accuracy AND ≥ 80% p-rule

## 📁 Key Files

| File | Purpose |
|------|---------|
| `train_final_with_llm.py` | ⭐ **Main script - RUN THIS** |
| `orchestrator_llm.py` | LLM orchestrator with Ollama integration |
| `README_LLM_TRAINING.md` | Detailed documentation |

## ⚙️ Architecture

```
3-Phase Training:
  Phase 1: Pre-train Classifier (10 epochs)
  Phase 2: Pre-train Adversary (10 epochs)
  Phase 3: Adaptive Loop (up to 100 steps)
           - Each step: Train Adv + Clf
           - Evaluate metrics
           - LLM decides next lambda
           - Until 80/80 ✓
```

## 🔍 How LLM Makes Decisions

Llama3.1 gets this prompt each step:

```
Current: Accuracy=0.73, P-rule=0.75 (targets: 0.80 both)
Current Lambda: [0.5, 0.5]
Past successful similar steps: [...]

Decide: What lambda values for NEXT step?
```

Llama3.1 outputs:

```json
{
  "lambda_adjustments": [0.65, 0.62],
  "momentum_new": [0.10, 0.09],
  "reasoning": "Increase fairness penalty to reach 80% p-rule while maintaining accuracy"
}
```

## 📈 Expected Progress

```
Step 1:  Acc=0.73, P-rule=0.75 → Increase fairness penalty
Step 2:  Acc=0.74, P-rule=0.79 → Fine-tune balance
Step 3:  Acc=0.76, P-rule=0.81 → Good progress!
...
Step 25: Acc=0.81, P-rule=0.82 → CONVERGED ✓
```

## ✅ Success Criteria

Training stops when:
- Accuracy ≥ 0.80 (80%) **AND**
- Min P-rule ≥ 0.80 (80%)

For both sensitive attributes (race, gender).

## 🐛 Troubleshooting

### Error: "Failed to connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve  # in another terminal
```

### Error: "Model llama3.1 not found"
```bash
ollama pull llama3.1
```

### Training is slow
- Reduce epochs_per_step: `epochs_per_step=3`
- Reduce max_steps: `max_steps=50`
- Use GPU (auto-detected)

### Want to try different Ollama model
```python
# In train_final_with_llm.py, change:
llm_model="mistral"  # or llama2, neural-chat, etc.
```

## 📝 Memory Learning

After each step, orchestrator saves to `orchestrator_training.json`:
```json
{
  "step": 1,
  "accuracy": 0.73,
  "p_rules": {"race": 0.78, "sex": 0.75},
  "lambda_old": [0.5, 0.5],
  "lambda_new": [0.65, 0.62],
  "success": false
}
```

On future runs, Llama3.1 uses this memory to learn: *"Last time accuracy was 0.73 and p-rule was 0.75, I increased lambda to [0.65, 0.62]. Did it work? Can I use a similar strategy?"*

## 🚀 Next: Advanced Customization

### Change target metrics
```python
train_with_llm_orchestrator(
    ...,
    # Modify in orchestrator.decide() call:
    target_p_rule=0.90,  # stricter fairness
    target_accuracy=0.85  # higher accuracy
)
```

### Adjust training schedule
```python
train_with_llm_orchestrator(
    ...,
    epochs_init_clf=20,   # longer pre-training
    epochs_per_step=10,   # more epochs per step
    max_steps=200         # more iterations allowed
)
```

### Use different LLM
```python
llm_model="mistral"   # smaller, faster
# or
llm_model="neural-chat"
```

## 📞 Support

- Check `README_LLM_TRAINING.md` for detailed docs
- Review `orchestrator_training.json` to see LLM's learning history
- Models are from CODE_FINAL_COMPLETE.ipynb
- Fairness metric is standard P-rule (demographic parity)

---

**Ready?** Just run: `python train_final_with_llm.py` 🚀
