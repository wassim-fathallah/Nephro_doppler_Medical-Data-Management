# LLM-Orchestrated Adversarial Fairness Training

## Overview

Complete training pipeline that combines:
- **Models**: Classifier & Adversary from CODE_FINAL_COMPLETE
- **Orchestrator**: LLM-based (Llama3.1 via Ollama) intelligent lambda decision-making
- **Target**: Reach **Accuracy ≥ 80%** AND **P-rule ≥ 80%** simultaneously

## Files

- `train_final_with_llm.py` - **Main training script** (RUN THIS)
- `orchestrator_llm.py` - LLM orchestrator with memory & decision logic
- `orchestrator_adaptive.py` - Fallback adaptive orchestrator (for testing)
- `orchestrator_wrapper.py` - Wrapper utility (optional)

## Setup

### 1. Install Dependencies

```bash
# PyTorch
pip install torch torchvision torchaudio

# LangChain for Ollama integration
pip install langchain-community langchain

# ML libraries
pip install numpy pandas scikit-learn
```

### 2. Setup Ollama

```bash
# Install Ollama from https://ollama.ai/

# Start Ollama service
ollama serve

# In another terminal, pull model
ollama pull llama3.1
```

### 3. Verify Llama3.1 is Available

```bash
ollama list  # Should show llama3.1
```

## Running Training

### Option 1: Using Adult Dataset

```bash
# Make sure adult.data is in the same directory
python train_final_with_llm.py
```

### Option 2: Using Synthetic Data

If `adult.data` is not found, the script auto-generates synthetic data.

## Training Architecture

```
PHASE 1: Pre-train Classifier (10 epochs)
    ↓
PHASE 2: Pre-train Adversary (10 epochs)
    ↓
PHASE 3: Adaptive Adversarial Loop (up to 100 steps)
    Each step:
    1. Train Adversary for 5 epochs (frozen classifier)
    2. Train Classifier for 5 epochs (frozen adversary)
    3. Evaluate: Accuracy & P-rule per sensitive attribute
    4. LLM ORCHESTRATOR DECIDES:
       - Analyze current metrics vs targets
       - Query memory of past successful steps
       - Use Llama3.1 to decide lambda adjustments
       - Apply momentum for stability
    5. If Acc ≥ 80% AND min(P-rule) ≥ 80%: CONVERGE ✓
```

## How LLM Orchestrator Works

### Memory & Learning
- Stores all past training steps (accuracy, p-rule, lambda changes, success/failure)
- Retrieves similar past experiences when making decisions
- Learns from what worked before

### LLM Decision Making
Prompts Llama3.1 with:
- Current accuracy and p-rule metrics
- Target values (80% for both)
- Current lambda values and momentum
- Similar past successful steps
- P-rule breakdown per sensitive attribute

Llama3.1 outputs:
- **New lambda values** (per attribute) - Higher λ = prioritize fairness more
- **New momentum** - For smooth, stable transitions
- **Reasoning** - Why this decision was made

### Fallback Heuristic
If LLM fails (Ollama not running, parsing error), falls back to:
```
λ_new = λ_old + β·m_old + (1-β)·increment
where increment = 2.5 × (target_p_rule - current_p_rule) / 100
      β = 0.7 (momentum coefficient)
```

## Key Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `epochs_init_clf` | 10 | Classifier pre-training epochs |
| `epochs_init_adv` | 10 | Adversary pre-training epochs |
| `epochs_per_step` | 5 | Epochs per adaptive step |
| `max_steps` | 100 | Maximum adaptive steps |
| `lambda_max` | 20.0 | Maximum lambda value |
| `temperature` | 0.3 | LLM temperature (0.3 = deterministic) |

## Expected Output

```
======================================================================
ADVERSARIAL FAIRNESS TRAINING WITH LLM ORCHESTRATOR
======================================================================

[INIT] Models initialized
  Classifier params: 3457
  Adversary params: 562
  Device: cpu
  Initial Lambda: [0.5, 0.5]
  Target: Accuracy >= 0.80, P-rule >= 0.80

[LLM-ORCHESTRATOR] Initializing llama3.1...
✓ LLM connection successful

======================================================================
[PHASE 1] Pre-training classifier (10 epochs)
======================================================================
  Epoch 5: Loss=0.4321, Acc=0.6850, P-rules={'race': 0.72, 'sex': 0.68}
  Epoch 10: Loss=0.3920, Acc=0.7150, P-rules={'race': 0.75, 'sex': 0.71}
✓ Classifier pre-trained

======================================================================
[PHASE 2] Pre-training adversary (10 epochs)
======================================================================
  Epoch 5: Adv Loss=0.6123
  Epoch 10: Adv Loss=0.5876
✓ Adversary pre-trained

======================================================================
[PHASE 3] Adversarial training with LLM orchestrator
======================================================================

----------------------------------------------------------------------
STEP 1/100
----------------------------------------------------------------------

[METRICS]
  Accuracy: 0.7300 (target: 0.80)
  P-rules: {'race': 0.78, 'sex': 0.75}
  Min P-rule: 0.7500 (target: 0.80)

[ORCHESTRATION]
[LLM-ORCHESTRATOR] Consulting Llama3.1 for step 1...
[LLM] Response:
Looking at the current metrics:
- Accuracy is 0.73 (below 0.80 target)
- Min P-rule is 0.75 (below 0.80 target)

Both need improvement. I'll increase lambda values to encourage the classifier
to hide the sensitive attributes better, which should improve fairness.
...

[LLM-ORCHESTRATOR] Step 1: Acc=0.7300, min P-rule=0.7500
  Old λ: ['0.500', '0.500']
  New λ: ['0.650', '0.620']
  Reasoning: Increase fairness penalty to reach 80% P-rule target...

----------------------------------------------------------------------
STEP 2/100
----------------------------------------------------------------------
[METRICS]
  Accuracy: 0.7450
  P-rules: {'race': 0.79, 'sex': 0.81}
  Min P-rule: 0.7900

[ORCHESTRATION]
[LLM-ORCHESTRATOR] Consulting Llama3.1 for step 2...
[LLM] Response: Learning from 1 similar past steps...

...

✅ CONVERGED at step 25!
   Final Accuracy: 0.8120
   Final P-rules: {'race': 0.8150, 'sex': 0.8320}

======================================================================
TRAINING COMPLETE
======================================================================
Best Accuracy: 0.8120 (target: 0.80)
Best P-rule: 0.8150 (target: 0.80)
Final P-rules: {'race': 0.8150, 'sex': 0.8320}
```

## Troubleshooting

### "Ollama not running"
```bash
# Start Ollama in a terminal
ollama serve
```

### "Model llama3.1 not found"
```bash
# Pull the model
ollama pull llama3.1
```

### "LangChain import error"
```bash
pip install langchain-community langchain
```

### Training is slow
- Reduce `epochs_per_step` (e.g., 3 instead of 5)
- Use GPU: runs on CUDA if available
- Reduce dataset size
- Reduce `max_steps` to stop earlier

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LOADING                                  │
│  Adult Dataset → Features (X) + Target (y) + Sensitive (Z)     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 1: CLASSIFIER PRE-TRAINING                    │
│  Minimize: Task Loss = BCELoss(y_pred, y)                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 2: ADVERSARY PRE-TRAINING                     │
│  Maximize: Adv Loss = BCELoss(Z_pred, Z)                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│         PHASE 3: ADAPTIVE ADVERSARIAL LOOP (up to 100 steps)    │
│                                                                  │
│  FOR each step:                                                 │
│    ├─ Train Adversary:                                          │
│    │  Maximize: BCELoss(Z_pred, Z)                              │
│    │                                                             │
│    ├─ Train Classifier:                                         │
│    │  Minimize: L_task - Σ(λᵢ × L_adv_individual)              │
│    │                                                             │
│    ├─ Evaluate:                                                 │
│    │  → Accuracy = correct predictions / total                  │
│    │  → P-rule = min(P(y=1|Z=0)/P(y=1|Z=1), ...)              │
│    │                                                             │
│    └─ ORCHESTRATOR DECISION:                                    │
│       ├─ Query LLM Memory (similar past steps)                  │
│       ├─ Prompt Llama3.1:                                       │
│       │  "Based on metrics + memory, decide new lambda..."     │
│       ├─ Parse LLM response:                                    │
│       │  {lambda_adjustments: [...], momentum_new: [...]}      │
│       ├─ Apply momentum:                                        │
│       │  m_new = 0.7·m_old + 0.3·increment                     │
│       ├─ Update lambda:                                         │
│       │  λ_new = max(0, min(λ + m, 20.0))                      │
│       └─ Save to memory (for future learning)                   │
│                                                                  │
│  IF (accuracy ≥ 0.80 AND min_p_rule ≥ 0.80):                   │
│    → CONVERGENCE ✓                                              │
│    → Save models & stop                                         │
└─────────────────────────────────────────────────────────────────┘
```

## References

- **Classifier & Adversary**: From CODE_FINAL_COMPLETE.ipynb
- **Fairness Metric**: P-rule (also called demographic parity)
  - P-rule = min(P(y=1|S=s1)/P(y=1|S=s2), P(y=1|S=s2)/P(y=1|S=s1))
  - Closer to 1.0 = more fair
  - ≥ 0.80 = commonly accepted threshold
- **Adversarial Training**: Standard approach for fairness-aware ML
  - Adversary learns to detect bias → Classifier learns to hide bias
  - Creates dynamic equilibrium balancing accuracy and fairness

## Next Steps

1. Run training and observe LLM decisions
2. Check memory file (`orchestrator_training.json`) to see what LLM learned
3. Adjust hyperparameters if needed
4. Try different Ollama models (llama2, mistral, etc.)
