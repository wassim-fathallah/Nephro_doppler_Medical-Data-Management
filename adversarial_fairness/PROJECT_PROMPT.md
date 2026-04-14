# Full Project Prompt — Adversarial Fairness System

## Context

We are building a fairness-aware adversarial machine learning system for a Master's thesis,
with potential paper publication. The project is at:
  C:\Users\MSI\Desktop\MEMOIRE\adversarial_fairness\
Datasets are at:
  C:\Users\MSI\Desktop\MEMOIRE\datasets\datasets\

Environment:
- Python 3.11.9
- Virtual env: C:\Users\MSI\Desktop\MEMOIRE\.venv
- Activate: C:\Users\MSI\Desktop\MEMOIRE\.venv\Scripts\activate
- Ollama running locally with llama3.1 pulled
- Packages installed: torch 2.11, langchain 1.2.15, langchain-ollama 1.1.0,
  langchain-core 1.2.26, langchain-community 0.4.1, pandas 3.0.2, numpy 2.4.4,
  scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8

Run commands:
  python main.py --dataset german  --iterations 20 --epochs 50
  python main.py --dataset compas  --iterations 25 --epochs 50
  python main.py --dataset adult   --iterations 25 --epochs 50
  python main.py --dataset utkface --iterations 20 --epochs 50
  python main.py --dataset kdd     --iterations 25 --epochs 50

---

## Core Concept

Two neural networks compete in a min-max game:

- Classifier (Learner): predicts a target label (income, recidivism, credit risk...).
  Objective: minimize task loss AND minimize the adversary's ability to recover
  sensitive attributes from its internal 32-dim embedding representation.

- Adversary: tries to predict sensitive attributes (race, sex, age...) from the
  classifier's intermediate embedding — NOT the final scalar output. This is the
  key design choice: operating on embeddings enables detecting proxy features.

Training objective:
  L_classifier = L_task - sum_s(lambda_s * L_adversary_s)
  L_adversary  = L_adversary  (maximized, trained independently)

Lambda (one per sensitive attribute) controls the fairness-accuracy tradeoff.
The LLM (llama3.1) decides lambda at every training iteration using both
short-term and long-term memory. This replaces a fixed momentum formula
with an adaptive LLM-driven control strategy.

---

## File Structure

adversarial_fairness/
├── main.py                  # entry point with argparse
├── orchestrator.py          # sequential pipeline (5 fixed steps, direct tool calls)
├── state.py                 # global mutable state shared between all LangChain tools
├── models/
│   ├── classifier.py        # 3-stage NN: encoder -> embedding_layer(32) -> head
│   └── adversary.py         # NN: takes 32-dim embedding, outputs logits per sensitive attr
├── memory/
│   ├── short_term.py        # in-run trajectory: lambda/p_rule/acc per iteration
│   └── long_term.py         # JSON across runs, keyed by dataset+target+attrs
├── tools/                   # LangChain @tool decorated functions
│   ├── data_tools.py        # identify_sensitive, load_dataset
│   ├── training_tools.py    # pretrain, run_full_training
│   ├── lambda_tools.py      # decide_initial_lambda + LCEL lambda_chain
│   └── proxy_tools.py       # detect_proxies via embedding-feature correlation
└── utils/
    ├── metrics.py           # all metrics (see below)
    └── plotting.py          # 5-panel PNG saved after training

---

## LangChain Integration

- LLM: llama3.1 via local Ollama
  - OllamaLLM for direct calls (tools, LCEL chains)
  - ChatOllama for agent (if needed)
- All 5 pipeline functions are LangChain @tool decorated
- Pipeline is called sequentially in orchestrator.py (NOT via ReAct agent —
  local LLMs are unreliable for autonomous tool-calling loops)
- Lambda decisions use a LangChain LCEL chain:
    PromptTemplate | OllamaLLM | StrOutputParser
  called at every training iteration inside run_full_training

LLM makes real decisions for:
  1. Identifying sensitive attributes + binarization rules from column names/samples
  2. Choosing initial lambda from long-term memory (warm-start)
  3. Choosing next lambda at each iteration (reads short-term + long-term memory)
  4. Interpreting proxy features semantically (real-world explanation)

---

## Memory System

Short-term memory (memory/short_term.py):
- Tracks current run trajectory: per iteration stores lambda, p_rules per attr,
  accuracy, adversary_loss, clf_task_loss
- Passed as JSON to the LCEL lambda chain at every iteration
- Detects stalling: p_rule not improving by >1% over last 3 iterations

Long-term memory (memory/long_term.py):
- JSON file (long_term_memory.json) persisted across all runs
- Key: "{dataset_name}|{target_col}|{sorted_sensitive_attrs}"
- Stores last 10 runs per key: lambda_final, p_rules_final, accuracy_final,
  epochs, iterations, success flag, lambda_trajectory
- suggest_initial_lambda(): returns 50% of best successful run's final lambda
- Both memories formatted as JSON strings injected into LLM prompts

---

## Lambda Decision (LCEL Chain)

At every training iteration the LLM receives:
- Current iteration / max iterations
- Current lambda vector
- Current metrics (accuracy, p_rules per attr, adversary loss)
- Last 6 entries from short-term memory (full trajectory)
- Last 3 relevant runs from long-term memory
- Constraints: lambda_max=20, must be >= 0

LLM outputs: {"reasoning": "...", "new_lambda": [{"attr": "sex", "value": 2.5}, ...]}

Rules embedded in prompt:
- P-rule far below 80% -> increase lambda aggressively
- P-rule stalling -> increase moderately
- P-rule above threshold -> decrease slightly to recover accuracy
- Accuracy below 0.70 -> decrease all lambdas 10-20%

---

## Proxy Detection

Key insight: adversary operates on the 32-dim embedding, not the final scalar.
This lets the system detect proxy features — columns that carry sensitive
information indirectly through the learned representation
(e.g. zip_code encoding race, occupation encoding sex).

Algorithm (tools/proxy_tools.py):
  1. Get embeddings E (N, 32) for all training samples
  2. Get adversary logits A (N, n_sensitive) from those embeddings
  3. For each input feature j and sensitive attr s:
       proxy_score[j,s] = |PearsonCorr(X[:,j], A[:,s])|
  4. Report top-5 proxy features per sensitive attribute
  5. LLM interprets them semantically

Proxy detection runs automatically after pretraining.
Results stored in state.proxy_report.
Always-on (not conditional).

---

## Neural Network Architecture

Classifier (models/classifier.py):
  Input(n_features)
    -> Linear(64) -> ReLU -> Dropout(0.2)   [encoder part 1]
    -> Linear(32) -> ReLU -> Dropout(0.2)   [encoder part 2]
    -> Linear(32) -> ReLU                   [embedding_layer — exposed to adversary]
    -> Linear(1)  -> Sigmoid                [prediction head]

  get_embedding(x): returns the 32-dim vector
  forward(x): returns (prediction, embedding) tuple

Adversary (models/adversary.py):
  Embedding(32) -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(n_sensitive)
  Outputs raw logits per sensitive attribute.
  Loss: BCEWithLogitsLoss

---

## Training Loop

Pretraining (default 10 epochs, independent):
  - Classifier: task loss only (BCELoss)
  - Adversary: on detached embeddings (no gradient back to classifier)

Adversarial loop (default 25 iterations x 50 epochs each):
  - Each epoch: adversary step first (detached emb), classifier step second (fresh emb)
  - Classifier loss: L_task - sum_s(lambda_s * L_adv_s)
  - After each iteration: evaluate all metrics -> LLM decides lambda
    -> update short-term memory -> check convergence
  - No early stopping: always runs all max_iterations even if P-rule >= 80%
    (logs "[target reached]" but continues to optimize; best_metrics tracked separately)
  - Early stop: iterations exhausted OR all lambda at max (20.0)

---

## Metrics

Performance:
  - Accuracy
  - Precision
  - F1 Score
  - ROC-AUC (area under ROC curve, threshold-independent)
  - AP / Average Precision (area under Precision-Recall curve,
    better than ROC-AUC on imbalanced datasets)

Fairness (computed per sensitive attribute):
  - P-rule (Disparate Impact): min(P(y=1|z=0)/P(y=1|z=1), reverse) * 100
    Target: >= 80% (the 80% rule)
  - Equalized Odds: TPR gap + FPR gap across groups (lower = fairer)
  - Equalized Opportunity: TPR gap only (lower = fairer)
  - Demographic Parity Difference: |P(y=1|z=0) - P(y=1|z=1)|
  - ABCC (Area Between Calibration Curves): measures how differently
    the model is calibrated between groups. Lower = fairer.
    Computed via sklearn.calibration.calibration_curve per group.

Comparison with baseline paper (wandb.ai/fair_benchmark/exp1.adv_gr):
  Common metrics: Accuracy, Precision, ROC-AUC, AP, P-rule,
                  Demographic Parity, ABCC (their test/ddss)
  Our extras:     F1, Equalized Odds, Equalized Opportunity

---

## Dataset Handling (Generic — Dataset Agnostic)

The LLM reads column names + sample values and decides:
  - Which column is the target
  - Which 2 columns are the most important sensitive attributes
  - Binarization rule per sensitive attr:
      {"positive_value": "Male"} or {"threshold": 25}
  - Which columns to drop (IDs, dates, leakage, duplicates)

Preprocessing pipeline (load_dataset tool):
  1. Auto-detect separator, handle ?/NA missing values
  2. Drop LLM-identified columns
  3. Drop high-cardinality string columns (>50 unique OR >10% of rows)
     catches: dates, case numbers, charge descriptions
  4. Binarize target (binary: sort+pick; multiclass: median split)
  5. Binarize sensitive attributes per LLM rules
  6. One-hot encode remaining categoricals
  7. Drop near-constant columns (std < 1e-6) to prevent NaN in StandardScaler
  8. StandardScaler normalize + nan_to_num safety net
  9. 80/20 stratified train/test split, seed=42

Supported datasets:
  - Adult       : adult/raw/adult.data             | target: income       | sensitive: sex, race
  - German      : german/raw/german_credit_risk.csv | target: Class        | sensitive: Sex, Age
  - COMPAS      : compas/raw/compas-scores-two-years.csv | target: two_year_recid | sensitive: race, sex
  - UTKFace     : utkface/raw/age_gender.csv        | target: age/gender   | sensitive: ethnicity, gender
  - KDD Census  : census_income_kdd/raw/census-income.data | target: income | sensitive: race, sex

Hardcoded fallbacks exist per dataset if LLM fails to return valid JSON.

---

## Output

Console: step-by-step logs per iteration (all metrics + current lambda + LLM reasoning)

Plot ({dataset_name}_training_curves.png — unique filename per dataset), 5 panels saved after training:
  1. Accuracy, F1, Precision, ROC-AUC vs iteration
  2. P-rule per sensitive attribute vs iteration (80% threshold line)
  3. Equalized Odds (TPR gap + FPR gap) per attribute vs iteration
  4. Equalized Opportunity (TPR gap) per attribute vs iteration
  5. Lambda decisions by LLM per attribute vs iteration

Long-term memory: long_term_memory.json updated after every run

---

## Planned: Streamlit Dashboard (Next Step)

A local Streamlit web app for thesis presentation and paper visualization.
Strategy: training loop writes metrics to a JSON file after each iteration.
Streamlit runs as a SEPARATE PROCESS reading that file — zero impact on training.

Planned interfaces:

Interface 1 — Live Training Dashboard (during training):
  - Real-time loss curves (task loss, adversary loss per epoch)
  - P-rule per attribute updating each iteration
  - Equalized Odds / Opportunity updating each iteration
  - Lambda decisions per iteration with LLM reasoning text displayed
  - Proxy features bar chart (shown after pretrain)
  - Current iteration progress bar

Interface 2 — Final Results (after training):
  - Full metrics table: accuracy, F1, AUC, AP, P-rule, Demographic Parity,
    Equalized Odds, Equalized Opportunity, ABCC
  - Comparison bar charts: our results vs baseline paper per dataset
  - Radar chart: all metrics at once per dataset
  - Convergence speed: epochs needed to reach 80% P-rule

Interface 3 — Proxy Analysis:
  - Bar chart: top proxy features per sensitive attribute
  - Heatmap: feature x sensitive attribute proxy score matrix
  - LLM semantic interpretation text
  - Before/after: proxy scores before vs after adversarial correction

Interface 4 — Memory Explorer:
  - Long-term memory viewer: all past runs per dataset
  - Lambda trajectory across runs (did LLM learn better init over time?)
  - Best run highlighter per dataset

Interface 5 — Dataset Explorer:
  - Positive rate distribution per sensitive group (before/after correction)
  - Feature correlation heatmap
  - Sensitive attribute group sizes and balance

---

## Known Limitations / Open Issues

- UTKFace: pixels column (raw image data as string) is dropped. Model uses
  only age, ethnicity, gender as features (very few). Proper approach:
  PCA on pixel values or pretrained image embeddings.
- KDD: no header row — needs column names assigned from .names file
  before LLM can analyze schema properly.
- Local llama3.1 via Ollama cannot reliably do autonomous tool-calling
  in a ReAct loop — pipeline orchestration is therefore done via direct
  sequential Python calls, not via LangChain agent.
- Each LLM call (lambda decision, proxy interpretation) takes 10-30s on CPU.
  Training is slow without GPU.

---

## Comparison Goal

We are comparing against: https://wandb.ai/fair_benchmark/exp1.adv_gr
That work tracks: test/acc, test/precision, test/roc, train/AP, test/dp,
test/prule, test/ddss (likely ABCC) across adult, bank_marketing, german, compas.

Our comparison strategy:
  - Same datasets (adult, german, compas minimum)
  - Same metrics (all 7 in common)
  - Best run vs best run comparison (accepted in fairness ML literature)
  - Our extras: F1, Equalized Odds, Equalized Opportunity (advantages over baseline)
  - Visualized via Streamlit for thesis demo
