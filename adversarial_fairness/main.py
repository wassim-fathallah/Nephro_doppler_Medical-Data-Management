# -*- coding: utf-8 -*-
import sys, io
# Force UTF-8 output on Windows to avoid encoding crashes
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
Entry point for the adversarial fairness pipeline.

Usage:
    python main.py --dataset path/to/data.csv --name german --iterations 20
"""

import argparse
import os
from orchestrator import run_pipeline


DATASET_PRESETS = {
    "adult":   "datasets/datasets/adult/raw/adult.data",
    "german":  "datasets/datasets/german/raw/german_credit_risk.csv",
    "compas":  "datasets/datasets/compas/raw/compas-scores-two-years.csv",
    "utkface": "datasets/datasets/utkface/raw/age_gender.csv",
    "kdd":     "datasets/datasets/census_income_kdd/raw/census-income.data",
}


def main():
    parser = argparse.ArgumentParser(description="Adversarial Fairness Pipeline")
    parser.add_argument("--dataset", type=str, help="Path to CSV or preset name (adult/german/compas/utkface/kdd)")
    parser.add_argument("--name",    type=str, default="", help="Dataset name for memory key")
    parser.add_argument("--iterations", type=int, default=25, help="Max adversarial iterations")
    parser.add_argument("--epochs",     type=int, default=50, help="Epochs per adversarial step")
    parser.add_argument("--threshold",  type=float, default=80.0, help="P-rule target %%")
    parser.add_argument("--pretrain",   type=int, default=10, help="Pre-training epochs")
    parser.add_argument("--device",     type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()

    # Resolve dataset path
    dataset_path = args.dataset
    dataset_name = args.name
    if dataset_path in DATASET_PRESETS:
        dataset_name = dataset_name or dataset_path
        # Build absolute path relative to this file's parent
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(base, DATASET_PRESETS[dataset_path])

    if not dataset_path:
        parser.error("Provide --dataset (path or preset: adult/german/compas/utkface/kdd)")

    print(f"\n{'='*60}")
    print(f"  Adversarial Fairness Pipeline")
    print(f"  Dataset : {dataset_path}")
    print(f"  Name    : {dataset_name or '(auto)'}")
    print(f"{'='*60}\n")

    result = run_pipeline(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        max_iterations=args.iterations,
        epochs_per_step=args.epochs,
        p_rule_threshold=args.threshold,
        initial_epochs=args.pretrain,
        device=args.device,
    )

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE — {result.get('status', 'unknown')}")
    print(f"{'='*60}")
    fm = result.get("final_metrics", {})
    if fm:
        print(f"  Accuracy  : {fm.get('accuracy', 'n/a')}")
        print(f"  F1        : {fm.get('f1_score', 'n/a')}")
        print(f"  ROC-AUC   : {fm.get('roc_auc', 'n/a')}")
        print(f"  P-rules   : {fm.get('p_rules', {})}")
    if result.get("plot_saved"):
        print(f"  Plot      : {result['plot_saved']}")
    if result.get("proxy_report"):
        print("\n  Proxy features detected:")
        for attr, feats in result["proxy_report"].get("proxy_features", {}).items():
            names = [f["feature"] for f in feats[:3]]
            print(f"    {attr}: {names}")


if __name__ == "__main__":
    main()
