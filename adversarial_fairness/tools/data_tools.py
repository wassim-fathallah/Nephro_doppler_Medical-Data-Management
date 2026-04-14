"""
LangChain tools for dataset loading and sensitive attribute identification.

load_dataset      — generic CSV loader; LLM decides drops, binarization, target
identify_sensitive — LLM reads column names/types and returns sensitive attrs
                     + binarization rules for each
"""

import json
import re
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from langchain.tools import tool
from langchain_ollama import OllamaLLM

from state import state

# ── LLM shared instance (tools import this too) ───────────────────────────────
llm = OllamaLLM(model="llama3.1", temperature=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract first JSON object from LLM prose response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in LLM response:\n{text}")


def _binarize_column(series: pd.Series, rule: dict) -> pd.Series:
    """
    Binarize a column according to an LLM-provided rule.
    rule = {"positive_value": "Male"}  → Male=1, else=0
    rule = {"threshold": 25}           → >25 = 1, else 0  (numeric only)
    Falls back to sort-and-pick if rule doesn't match column dtype.
    """
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if "positive_value" in rule:
        pv = rule["positive_value"]
        # Cast positive_value to match series dtype
        if is_numeric:
            try:
                pv = type(series.dropna().iloc[0])(pv)
            except Exception:
                pass
        return (series == pv).astype(int)

    elif "threshold" in rule and is_numeric:
        return (series > rule["threshold"]).astype(int)

    else:
        # String column with threshold rule, or unknown rule — sort and pick
        try:
            vals = sorted(series.dropna().unique())
        except TypeError:
            vals = list(series.dropna().unique())
        pos = vals[1] if len(vals) > 1 else vals[0]
        return (series == pos).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — identify_sensitive
# ─────────────────────────────────────────────────────────────────────────────

@tool
def identify_sensitive(dataset_path: str, dataset_name: str = "") -> str:
    """
    Reads dataset column names and sample values, then asks the LLM which
    columns are sensitive attributes (race, gender, age, etc.) and how to
    binarize them. Also asks the LLM to identify the target column and
    columns to drop (IDs, duplicates, leakage).

    Returns a JSON string with keys:
      - sensitive_attrs: list of column names
      - binarization_rules: {col: {rule}} per sensitive attr
      - target_col: name of the prediction target
      - columns_to_drop: list of cols to exclude from features
      - justification: LLM reasoning
    """
    # Read a small sample for schema inference
    try:
        df = pd.read_csv(dataset_path, nrows=5, skipinitialspace=True, na_values=["?", "NA", "N/A", ""])
    except Exception:
        df = pd.read_csv(dataset_path, nrows=5, sep=r"\s*,\s*", engine="python",
                         skipinitialspace=True, na_values=["?", "NA", ""])

    schema_str = ""
    for col in df.columns:
        sample = df[col].dropna().head(3).tolist()
        dtype = str(df[col].dtype)
        schema_str += f"  - {col} ({dtype}): sample={sample}\n"

    prompt = f"""You are a fairness-aware ML expert analyzing a dataset for bias correction.

Dataset name: {dataset_name or 'unknown'}
Columns and sample values:
{schema_str}

Your task:
1. Identify all columns that represent sensitive/protected attributes (race, sex/gender, age, ethnicity, nationality, religion, disability, etc.)
2. Choose the 2 most important sensitive attributes for fairness correction
3. For each sensitive attr, define a binarization rule: either {{"positive_value": "<value>"}} for categorical or {{"threshold": <number>}} for numeric
4. Identify the target/label column (what we want to predict)
5. Identify columns to drop: IDs, row numbers, duplicates, direct leakage of target

Respond with ONLY this JSON (no prose):
{{
  "sensitive_attrs": ["col1", "col2"],
  "binarization_rules": {{
    "col1": {{"positive_value": "Male"}},
    "col2": {{"threshold": 25}}
  }},
  "target_col": "column_name",
  "columns_to_drop": ["id_col", "name_col"],
  "justification": "brief reason"
}}"""

    response = llm.invoke(prompt)
    try:
        result = _extract_json(response)
    except ValueError:
        # Hardcoded fallbacks per known dataset
        fallbacks = {
            "adult":   {"sensitive_attrs": ["sex", "race"], "binarization_rules": {"sex": {"positive_value": "Male"}, "race": {"positive_value": "White"}}, "target_col": "income", "columns_to_drop": ["fnlwgt", "education-num"], "justification": "fallback"},
            "german":  {"sensitive_attrs": ["Sex", "Age"], "binarization_rules": {"Sex": {"positive_value": "male"}, "Age": {"threshold": 25}}, "target_col": "Class", "columns_to_drop": ["Unnamed: 0"], "justification": "fallback"},
            "compas":  {"sensitive_attrs": ["race", "sex"], "binarization_rules": {"race": {"positive_value": "Caucasian"}, "sex": {"positive_value": "Male"}}, "target_col": "two_year_recid", "columns_to_drop": ["id", "name", "first", "last", "dob", "c_case_number", "r_case_number", "vr_case_number", "compas_screening_date", "c_jail_in", "c_jail_out", "r_offense_date", "r_jail_in", "r_jail_out", "vr_offense_date", "screening_date", "v_screening_date", "in_custody", "out_custody", "c_offense_date", "c_arrest_date"], "justification": "fallback"},
            "utkface": {"sensitive_attrs": ["ethnicity", "gender"], "binarization_rules": {"ethnicity": {"positive_value": 0}, "gender": {"positive_value": 0}}, "target_col": "age", "columns_to_drop": ["img_name", "pixels"], "justification": "fallback"},
            "kdd":     {"sensitive_attrs": ["race", "sex"], "binarization_rules": {"race": {"positive_value": "White"}, "sex": {"positive_value": "Male"}}, "target_col": "income", "columns_to_drop": ["instance_weight"], "justification": "fallback"},
        }
        name_lower = dataset_name.lower()
        matched = next((v for k, v in fallbacks.items() if k in name_lower), None)
        if matched:
            result = matched
        else:
            raise RuntimeError(f"LLM failed to return valid JSON and no fallback for '{dataset_name}':\n{response}")

    # Store in global state
    state.sensitive_attrs = result["sensitive_attrs"]
    state.binarization_rules = result["binarization_rules"]
    state.target_col = result["target_col"]
    state.columns_to_drop = result.get("columns_to_drop", [])
    state.dataset_name = dataset_name
    state.dataset_path = dataset_path

    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — load_dataset
# ─────────────────────────────────────────────────────────────────────────────

@tool
def load_dataset(dataset_path: str, dataset_name: str = "") -> str:
    """
    Generic dataset loader. Requires identify_sensitive to have been called first
    (to know target, sensitive attrs, binarization rules, columns to drop).

    Handles:
    - Multiple separators (comma, semicolon, whitespace)
    - Missing value markers (?, NA, N/A, empty)
    - Numeric and categorical columns
    - Binarization of sensitive attributes per LLM rules
    - One-hot encoding of remaining categoricals
    - Standard scaling of features
    - 80/20 train/test split

    Returns a summary string of the loaded dataset.
    """
    if not state.target_col:
        return "ERROR: Call identify_sensitive first to set target_col and sensitive_attrs."

    # ── Load raw CSV ──────────────────────────────────────────────────────────
    load_kwargs = dict(skipinitialspace=True, na_values=["?", "NA", "N/A", ""])
    try:
        df = pd.read_csv(dataset_path, **load_kwargs)
    except Exception:
        df = pd.read_csv(dataset_path, sep=r"\s*,\s*", engine="python", header=None, **load_kwargs)
        # Try to assign column names from .names file if available
        # (KDD case — we'll let the LLM handle naming via identify_sensitive)

    df.columns = df.columns.str.strip()

    # ── Drop irrelevant columns ───────────────────────────────────────────────
    drop_cols = [c for c in state.columns_to_drop if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # ── Drop rows with missing target ─────────────────────────────────────────
    df.dropna(subset=[state.target_col], inplace=True)

    # ── Binarize target ───────────────────────────────────────────────────────
    target_series = df[state.target_col].str.strip() if df[state.target_col].dtype == object else df[state.target_col]
    unique_targets = sorted(target_series.dropna().unique())
    if len(unique_targets) == 2:
        # Binary: higher/second value = 1 (e.g., ">50K", "2", True)
        pos_target = unique_targets[1]
        y = (target_series == pos_target).astype(int).values
    else:
        # Multiclass: try numeric median, else label-encode then split at median
        numeric_target = pd.to_numeric(target_series, errors="coerce")
        if numeric_target.notna().sum() > len(target_series) * 0.9:
            med = numeric_target.median()
            y = (numeric_target > med).astype(int).values
        else:
            # Categorical with >2 classes: encode and split at median code
            codes = pd.Categorical(target_series).codes
            med = np.median(codes)
            y = (codes > med).astype(int)
            print(f"[warn] Target '{state.target_col}' has {len(unique_targets)} string classes — "
                  f"binarized by label-code median ({med:.0f})")

    df.drop(columns=[state.target_col], inplace=True)

    # ── Extract and binarize sensitive attributes ─────────────────────────────
    sensitive_cols = []
    for attr in state.sensitive_attrs:
        if attr not in df.columns:
            print(f"[warn] sensitive attr '{attr}' not found in columns: {df.columns.tolist()}")
            continue
        rule = state.binarization_rules.get(attr, {})
        s = df[attr]
        if df[attr].dtype == object:
            s = s.str.strip()
        s_bin = _binarize_column(s, rule)
        sensitive_cols.append(s_bin.values)
        df.drop(columns=[attr], inplace=True)

    sensitive_matrix = np.stack(sensitive_cols, axis=1)  # (N, n_sensitive)

    # ── Drop high-cardinality string columns (IDs, free text, charge descriptions)
    # Use two thresholds:
    #   > MAX_UNIQUE_ABS  unique values → always drop (e.g. case numbers, descriptions)
    #   > MAX_UNIQUE_FRAC of rows       → drop (e.g. dates unique per row)
    MAX_UNIQUE_ABS  = 50
    MAX_UNIQUE_FRAC = 0.1
    for col in df.select_dtypes(include="object").columns:
        n_unique = df[col].nunique()
        if n_unique > MAX_UNIQUE_ABS or n_unique > MAX_UNIQUE_FRAC * len(df):
            df.drop(columns=[col], inplace=True)
            print(f"[info] Dropped high-cardinality column: {col} ({n_unique} unique)")

    # ── Fill remaining missing values ─────────────────────────────────────────
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().fillna("Unknown")
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    # ── One-hot encode categoricals ───────────────────────────────────────────
    df = pd.get_dummies(df, drop_first=True)

    X = df.values.astype(np.float32)
    state.feature_names = df.columns.tolist()

    # ── Train/test split ──────────────────────────────────────────────────────
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    s_tr, s_te = sensitive_matrix[train_idx], sensitive_matrix[test_idx]

    # Drop near-constant columns before scaling (std ~= 0 causes NaN after scale)
    col_stds = X_tr.std(axis=0)
    valid_cols = col_stds > 1e-6
    if not valid_cols.all():
        n_dropped = (~valid_cols).sum()
        print(f"[info] Dropped {n_dropped} near-constant columns (std ~= 0)")
        X_tr = X_tr[:, valid_cols]
        X_te = X_te[:, valid_cols]
        state.feature_names = [n for n, v in zip(state.feature_names, valid_cols) if v]

    # ── Normalize features ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    # Replace any residual NaN/Inf with 0 (safety net)
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    # Store raw numpy for proxy detection (before tensor conversion)
    state.X_train_raw = X_tr.copy()

    # ── Convert to tensors ────────────────────────────────────────────────────
    dev = state.device
    state.X_train = torch.tensor(X_tr, dtype=torch.float32).to(dev)
    state.X_test  = torch.tensor(X_te, dtype=torch.float32).to(dev)
    state.y_train = torch.tensor(y_tr, dtype=torch.float32).to(dev)
    state.y_test  = torch.tensor(y_te, dtype=torch.float32).to(dev)
    state.sensitive_train = torch.tensor(s_tr, dtype=torch.float32).to(dev)
    state.sensitive_test  = torch.tensor(s_te, dtype=torch.float32).to(dev)

    summary = (
        f"Dataset loaded: {len(X)} samples, {X.shape[1]} features\n"
        f"Train: {len(X_tr)} | Test: {len(X_te)}\n"
        f"Target: '{state.target_col}' | Positive rate: {y.mean():.2%}\n"
        f"Sensitive attrs: {state.sensitive_attrs}\n"
        f"Device: {state.device}"
    )
    print(f"[data] {summary}")
    return summary
