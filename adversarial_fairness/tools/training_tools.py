"""
LangChain tools for model training:

  pretrain          — trains classifier and adversary independently (no fairness pressure yet)
  run_full_training — the adversarial loop; LLM decides lambda at each iteration
                      via lambda_tools.decide_lambda_for_iteration
"""

import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from langchain.tools import tool

from state import state
from models.classifier import Classifier
from models.adversary import Adversary
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from utils.metrics import evaluate
from utils.plotting import TrainingPlotter
from tools.lambda_tools import decide_lambda_for_iteration
from tools.proxy_tools import detect_proxies


# ─────────────────────────────────────────────────────────────────────────────
# Internal training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(X, y, sensitive, batch_size=64):
    dataset = TensorDataset(X, y.unsqueeze(1), sensitive)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _train_one_epoch_pretrain(classifier, adversary, clf_opt, adv_opt, loader, device):
    """Pre-train: classifier on task loss, adversary on embedding."""
    clf_loss_fn = nn.BCELoss()
    adv_loss_fn = nn.BCEWithLogitsLoss()

    classifier.train()
    adversary.train()

    total_clf, total_adv = 0.0, 0.0
    for X_b, y_b, s_b in loader:
        X_b, y_b, s_b = X_b.to(device), y_b.to(device), s_b.to(device)

        # Classifier step
        clf_opt.zero_grad()
        pred, emb = classifier(X_b)
        loss_clf = clf_loss_fn(pred, y_b)
        loss_clf.backward()
        clf_opt.step()
        total_clf += loss_clf.item()

        # Adversary step (on detached embedding — adversary trains independently)
        adv_opt.zero_grad()
        with torch.no_grad():
            _, emb_det = classifier(X_b)
        adv_logits = adversary(emb_det.detach())
        loss_adv = adv_loss_fn(adv_logits, s_b)
        loss_adv.backward()
        adv_opt.step()
        total_adv += loss_adv.item()

    n = len(loader)
    return total_clf / n, total_adv / n


def _train_one_epoch_adversarial(
    classifier, adversary, clf_opt, adv_opt, loader, lambda_vector, device
):
    """
    Adversarial training epoch.
    Classifier objective: L_task - sum_s(lambda_s * L_adv_s)
    Adversary objective:  L_adv  (maximized = adversary trains to predict sensitive)
    """
    clf_loss_fn = nn.BCELoss()
    adv_loss_fn = nn.BCEWithLogitsLoss()
    lambdas = torch.tensor(lambda_vector, dtype=torch.float32).to(device)

    classifier.train()
    adversary.train()

    total_clf, total_adv, total_task = 0.0, 0.0, 0.0
    for X_b, y_b, s_b in loader:
        X_b, y_b, s_b = X_b.to(device), y_b.to(device), s_b.to(device)

        # ── Adversary step ────────────────────────────────────────────────────
        adv_opt.zero_grad()
        with torch.no_grad():
            _, emb = classifier(X_b)
        adv_logits = adversary(emb.detach())
        loss_adv = adv_loss_fn(adv_logits, s_b)
        loss_adv.backward()
        adv_opt.step()

        # ── Classifier step ───────────────────────────────────────────────────
        clf_opt.zero_grad()
        pred, emb = classifier(X_b)
        loss_task = clf_loss_fn(pred, y_b)

        # Adversary loss on FRESH embedding (not detached) — gradients flow back
        adv_logits_fresh = adversary(emb)
        # Per-attribute adversary losses
        adv_losses = [
            adv_loss_fn(adv_logits_fresh[:, s:s+1], s_b[:, s:s+1])
            for s in range(s_b.shape[1])
        ]
        # Weighted adversarial penalty
        penalty = sum(lambdas[s] * adv_losses[s] for s in range(len(adv_losses)))
        loss_clf = loss_task - penalty   # classifier MINIMIZES this
        loss_clf.backward()
        clf_opt.step()

        total_task += loss_task.item()
        total_adv  += loss_adv.item()
        total_clf  += loss_clf.item()

    n = len(loader)
    return total_task / n, total_adv / n, total_clf / n


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — pretrain
# ─────────────────────────────────────────────────────────────────────────────

@tool
def pretrain(n_epochs: int = 10) -> str:
    """
    Pre-trains the classifier and adversary independently for n_epochs.
    This gives both models a warm start before the adversarial competition begins.
    Also detects proxy features after pretraining is complete.
    Returns a JSON summary of initial metrics.
    """
    if state.X_train is None:
        return "ERROR: Load dataset first."

    device = state.device
    n_features  = state.X_train.shape[1]
    n_sensitive = state.sensitive_train.shape[1]

    # ── Instantiate models ────────────────────────────────────────────────────
    torch.manual_seed(42)
    clf = Classifier(n_features, embedding_dim=state.embedding_dim).to(device)
    adv = Adversary(
        embedding_dim=state.embedding_dim,
        n_sensitive=n_sensitive
    ).to(device)

    clf_opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    adv_opt = torch.optim.Adam(adv.parameters(), lr=1e-3)

    state.classifier    = clf
    state.adversary     = adv
    state.clf_optimizer = clf_opt
    state.adv_optimizer = adv_opt

    loader = _make_loader(state.X_train, state.y_train, state.sensitive_train)

    print(f"\n[pretrain] Pre-training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        clf_loss, adv_loss = _train_one_epoch_pretrain(
            clf, adv, clf_opt, adv_opt, loader, device
        )
        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch+1:3d}/{n_epochs} | clf_loss={clf_loss:.4f} | adv_loss={adv_loss:.4f}")

    # ── Initial evaluation ────────────────────────────────────────────────────
    metrics = evaluate(
        clf, state.X_test, state.y_test, state.sensitive_test,
        state.sensitive_attrs, device
    )
    print(f"[pretrain] Initial → acc={metrics['accuracy']:.4f} | p_rules={metrics['p_rules']}")

    state.total_epochs_run += n_epochs

    # ── Proxy detection after pretraining ────────────────────────────────────
    print("\n[pretrain] Running proxy detection on pretrained embeddings...")
    detect_proxies.invoke({"top_k": 5})

    return json.dumps({
        "status": "pretrain_complete",
        "initial_metrics": metrics,
        "epochs_run": n_epochs,
    }, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — run_full_training
# ─────────────────────────────────────────────────────────────────────────────

@tool
def run_full_training(
    max_iterations: int = 25,
    epochs_per_step: int = 50,
    p_rule_threshold: float = 80.0,
    lambda_max: float = 20.0,
) -> str:
    """
    Runs the full adversarial training loop.

    At EACH ITERATION:
      1. Train classifier + adversary for epochs_per_step epochs
      2. Evaluate fairness metrics (P-rule per sensitive attr, accuracy)
      3. Print step summary to console
      4. LLM reads short-term trajectory + long-term memory → decides new lambda
      5. Update short-term memory with this iteration's results
      6. Check convergence: stop if all P-rules >= threshold

    After training:
      - Saves run to long-term memory
      - Saves training curves plot (p_rule + accuracy + lambda per iteration)

    Returns a JSON summary of the final outcome.
    """
    if state.classifier is None:
        return "ERROR: Call pretrain first."

    state.p_rule_threshold = p_rule_threshold
    state.max_iterations   = max_iterations

    if not state.lambda_vector:
        state.lambda_vector = [0.1] * len(state.sensitive_attrs)

    short_term = ShortTermMemory()
    long_term  = LongTermMemory()
    plotter    = TrainingPlotter(state.sensitive_attrs)
    loader     = _make_loader(state.X_train, state.y_train, state.sensitive_train)

    device = state.device
    clf    = state.classifier
    adv    = state.adversary
    clf_opt = state.clf_optimizer
    adv_opt = state.adv_optimizer

    print(f"\n{'='*60}")
    print(f"[train] Starting adversarial training loop")
    print(f"  Max iterations : {max_iterations}")
    print(f"  Epochs/step    : {epochs_per_step}")
    print(f"  P-rule target  : {p_rule_threshold}%")
    print(f"  Initial λ      : {state.lambda_vector}")
    print(f"{'='*60}")

    final_metrics = {}
    lambda_trajectory = []

    for iteration in range(max_iterations):
        state.current_iteration = iteration

        # ── Train for epochs_per_step epochs ──────────────────────────────────
        epoch_task_losses, epoch_adv_losses = [], []
        for _ in range(epochs_per_step):
            task_loss, adv_loss, _ = _train_one_epoch_adversarial(
                clf, adv, clf_opt, adv_opt, loader, state.lambda_vector, device
            )
            epoch_task_losses.append(task_loss)
            epoch_adv_losses.append(adv_loss)

        state.total_epochs_run += epochs_per_step

        avg_task_loss = np.mean(epoch_task_losses)
        avg_adv_loss  = np.mean(epoch_adv_losses)

        # ── Evaluate ──────────────────────────────────────────────────────────
        metrics = evaluate(
            clf, state.X_test, state.y_test, state.sensitive_test,
            state.sensitive_attrs, device
        )
        metrics["adversary_loss"] = float(avg_adv_loss)
        metrics["clf_task_loss"]  = float(avg_task_loss)

        # ── Console log ───────────────────────────────────────────────────────
        p_rule_str = " | ".join(f"{a}={v:.1f}%" for a, v in metrics["p_rules"].items())
        print(
            f"[iter {iteration+1:2d}/{max_iterations}] "
            f"acc={metrics['accuracy']:.4f} | f1={metrics.get('f1_score',0):.4f} | "
            f"auc={metrics.get('roc_auc',0):.4f} | prec={metrics.get('precision',0):.4f}\n"
            f"  fairness: {p_rule_str} | "
            f"adv_loss={avg_adv_loss:.4f} | λ={[round(l,3) for l in state.lambda_vector]}"
        )

        # ── Update plotter ────────────────────────────────────────────────────
        plotter.update(
            iteration=iteration + 1,
            accuracy=metrics["accuracy"],
            p_rules=metrics["p_rules"],
            lambdas=state.lambda_vector,
            adv_loss=avg_adv_loss,
            precision=metrics.get("precision", 0.0),
            f1=metrics.get("f1_score", 0.0),
            roc_auc=metrics.get("roc_auc", 0.0),
            fairness=metrics.get("fairness", {}),
        )

        # ── Check convergence ─────────────────────────────────────────────────
        if metrics["min_p_rule"] >= p_rule_threshold:
            print(f"\n[train] Fairness target reached at iteration {iteration+1}!")
            final_metrics = metrics
            lambda_trajectory.append(state.lambda_vector.copy())
            break

        # ── LLM decides next lambda ───────────────────────────────────────────
        short_term.add(
            iteration=iteration + 1,
            lambda_vector=state.lambda_vector,
            p_rules=metrics["p_rules"],
            accuracy=metrics["accuracy"],
            adversary_loss=avg_adv_loss,
            clf_task_loss=avg_task_loss,
        )

        new_lambda = decide_lambda_for_iteration(
            short_term=short_term,
            long_term=long_term,
            current_metrics=metrics,
            iteration=iteration + 1,
            max_iterations=max_iterations,
            lambda_max=lambda_max,
        )
        lambda_trajectory.append(state.lambda_vector.copy())
        state.lambda_vector = new_lambda
        final_metrics = metrics

    # ── Post-training ─────────────────────────────────────────────────────────
    success = final_metrics.get("min_p_rule", 0) >= p_rule_threshold

    long_term.save_run(
        dataset_name=state.dataset_name,
        target_col=state.target_col,
        sensitive_attrs=state.sensitive_attrs,
        lambda_final=state.lambda_vector,
        p_rules_final=final_metrics.get("p_rules", {}),
        accuracy_final=final_metrics.get("accuracy", 0.0),
        total_epochs=state.total_epochs_run,
        iterations=state.current_iteration + 1,
        success=success,
        lambda_trajectory=lambda_trajectory,
    )

    plot_path = plotter.save("training_curves.png")
    state.training_done = True

    summary = {
        "status": "success" if success else "max_iterations_reached",
        "iterations_run": state.current_iteration + 1,
        "total_epochs": state.total_epochs_run,
        "final_metrics": final_metrics,
        "final_lambda": state.lambda_vector,
        "plot_saved": plot_path,
        "long_term_memory_updated": True,
    }

    print(f"\n{'='*60}")
    print(f"[train] Done: {summary['status']}")
    print(f"  Final acc   : {final_metrics.get('accuracy', 0):.4f}")
    print(f"  Final p_rule: {final_metrics.get('p_rules', {})}")
    print(f"  Plot        : {plot_path}")
    print(f"{'='*60}\n")

    return json.dumps(summary, indent=2)
