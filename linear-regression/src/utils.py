"""
Plotting and Visualization Utilities for Linear Regression Diagnostics and Analysis.

Includes:
- plot_loss_convergence: Plots loss/cost curve over iterations (Problem 1)
- plot_predictions_vs_actual: Scatter plot of predicted vs ground-truth targets with stats box
- plot_coefficient_paths: Plots weight trajectories across regularizing alphas with best alpha marker
- plot_residuals: Residuals vs fitted values and residual distribution histogram with summary statistics
- plot_model_comparison: Clean bar chart comparing metrics across models with concise non-overlapping column notes
- plot_multi_metric_comparison: Side-by-side 3-panel comparison for RMSE, MAE, and R2 scores
"""

import os
from typing import List, Optional, Dict, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def _ensure_dir(save_path: str):
    """Utility function to ensure directory exists before saving figure."""
    directory = os.path.dirname(os.path.abspath(save_path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def format_short_model_name(name: str) -> str:
    """
    Converts verbose model names into concise, multi-line labels
    to ensure x-axis notes never overlap horizontally.

    Examples:
        'OLS (Raw Features)'       -> 'OLS\\n(Raw)'
        'GD (Raw Features)'        -> 'GD\\n(Raw)'
        'OLS (Basis Expansion)'    -> 'OLS\\n(Expanded)'
        'Ridge (Basis Expansion)'  -> 'Ridge\\n(Expanded)'
        'LASSO (Basis Expansion)'  -> 'LASSO\\n(Expanded)'
    """
    clean = str(name).strip()
    mapping = {
        "OLS (Raw Features)": "OLS\n(Raw)",
        "GD (Raw Features)": "GD\n(Raw)",
        "OLS (Basis Expansion)": "OLS\n(Expanded)",
        "Ridge (Basis Expansion)": "Ridge\n(Expanded)",
        "LASSO (Basis Expansion)": "LASSO\n(Expanded)",
        "OLS (Raw)": "OLS\n(Raw)",
        "GD (Raw)": "GD\n(Raw)",
        "OLS (Expanded)": "OLS\n(Expanded)",
        "Ridge (Expanded)": "Ridge\n(Expanded)",
        "LASSO (Expanded)": "LASSO\n(Expanded)",
    }
    if clean in mapping:
        return mapping[clean]

    # Generalized clean abbreviation
    if " (" in clean:
        parts = clean.split(" (", 1)
        base = parts[0]
        desc = parts[1].rstrip(")")
        desc = desc.replace("Features", "").replace("Basis Expansion", "Expanded").strip()
        return f"{base}\n({desc})" if desc else base

    return clean


def plot_loss_convergence(
    cost_history: List[float],
    cost_name: str = "MSE",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plots training cost/loss over iterations with concise summary notes.
    """
    cost_label = cost_name.upper() if isinstance(cost_name, str) else "Loss"
    if title is None:
        title = f"Gradient Descent Convergence ({cost_label})"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_history, color="#1f77b4", linewidth=2.2, label=f"Cost ({cost_label})")

    # Concise Callout / Stat Box
    if len(cost_history) > 0:
        initial_cost = cost_history[0]
        final_cost = cost_history[-1]
        reduction_pct = (1.0 - final_cost / initial_cost) * 100 if initial_cost != 0 else 0
        info_text = (
            f"Start: {initial_cost:,.1f}\n"
            f"Final: {final_cost:,.1f}\n"
            f"Reduced: {reduction_pct:.1f}%"
        )
        ax.text(
            0.96, 0.78,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9)
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Iteration (x100)", fontsize=11, fontweight="medium")
    ax.set_ylabel(f"Cost Value ({cost_label})", fontsize=11, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Target",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Scatter plot of actual vs predicted values with reference line and concise metric card.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Calculate concise summary stats
    calc_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    calc_mae = np.mean(np.abs(y_true - y_pred))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    calc_r2 = 1.0 - (np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, color="#2ca02c", edgecolors="none", s=25, label="Predictions")

    # 45-degree reference line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y = x)")

    # Concise Metric Summary Box
    metrics_box = (
        f"RMSE : {calc_rmse:.2f}\n"
        f"MAE  : {calc_mae:.2f}\n"
        f"R²   : {calc_r2:.3f}"
    )
    ax.text(
        0.05, 0.85,
        metrics_box,
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9)
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Actual Values (y_true)", fontsize=11)
    ax.set_ylabel("Predicted Values (y_pred)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_coefficient_paths(
    alphas: np.ndarray,
    coefs: np.ndarray,
    best_alpha: Optional[float] = None,
    feature_names: Optional[List[str]] = None,
    title: str = "Regularization Coefficient Paths",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plots coefficient paths with optional highlight of best tuned alpha.
    """
    alphas = np.asarray(alphas)
    coefs = np.asarray(coefs)
    n_features = coefs.shape[1]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for j in range(n_features):
        label = feature_names[j] if feature_names and j < len(feature_names) else None
        ax.plot(alphas, coefs[:, j], linewidth=1.6, label=label)

    ax.set_xscale("log")
    ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.7)

    # Highlight best alpha if provided
    if best_alpha is not None:
        ax.axvline(best_alpha, color="#d62728", linestyle="--", linewidth=1.8, label=f"Best α = {best_alpha:.4f}")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Regularization Strength (α) [log scale]", fontsize=11)
    ax.set_ylabel("Coefficient Weight (w_j)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    if feature_names and len(feature_names) <= 10:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5, framealpha=0.9)
    elif best_alpha is not None:
        ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plots residual scatter plot and distribution histogram with concise stat notes.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred

    mean_res = np.mean(residuals)
    std_res = np.std(residuals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Residuals vs Fitted
    ax1.scatter(y_pred, residuals, alpha=0.35, color="#1f77b4", edgecolors="none", s=25)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1.8)
    ax1.set_title("Residuals vs Fitted Values", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Fitted Values (y_pred)", fontsize=11)
    ax1.set_ylabel("Residuals (y_true - y_pred)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Note on Panel 1
    ax1.text(
        0.05, 0.92,
        f"Mean: {mean_res:+.2f}\nStd:  {std_res:.2f}",
        transform=ax1.transAxes,
        fontsize=9.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85)
    )

    # Panel 2: Residual Histogram
    ax2.hist(residuals, bins=30, color="#ff7f0e", edgecolor="black", alpha=0.75, density=True)
    ax2.axvline(mean_res, color="red", linestyle="--", linewidth=1.8, label=f"Mean μ = {mean_res:.1f}")

    # Gaussian approximation curve
    x_grid = np.linspace(np.min(residuals), np.max(residuals), 200)
    pdf = (1.0 / (std_res * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - mean_res) / std_res) ** 2)
    ax2.plot(x_grid, pdf, color="#333333", linewidth=1.8, label="Normal Fit")

    ax2.set_title("Residual Distribution & Density", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Residual Error (y_true - y_pred)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_model_comparison(
    metrics_dict: Dict[str, float],
    metric_name: str = "RMSE",
    title: str = "Model Performance Comparison",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Bar chart comparing evaluation metrics across models.
    Each bar is annotated with a concise, non-overlapping label and value note.

    Key visual features:
    - Shortened, multi-line column names to eliminate horizontal overlap
    - Value annotations directly above each bar with ample headroom
    - Color distinction between Raw vs Expanded models
    - Best performing model highlighted
    """
    models = list(metrics_dict.keys())
    values = [float(v) for v in metrics_dict.values()]
    short_labels = [format_short_model_name(m) for m in models]

    # Determine best model index (lower is better for RMSE/MAE, higher for R2)
    is_r2 = "R2" in metric_name.upper() or "R^2" in metric_name.upper()
    best_idx = int(np.argmax(values) if is_r2 else np.argmin(values))

    # Color palette: Raw models (blueish), Expanded models (greenish), Best model (accent)
    colors = []
    for i, m in enumerate(models):
        if i == best_idx:
            colors.append("#2b8a3e" if is_r2 else "#2b8a3e")  # Green for champion
        elif "Raw" in m:
            colors.append("#4c72b0")  # Slate Blue
        else:
            colors.append("#7097c2")  # Light Slate

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(range(len(models)), values, color=colors, edgecolor="#212529", width=0.48, linewidth=1.2)

    # Set concise x-axis tick labels
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_labels, fontsize=10.5, fontweight="semibold")

    # Ample y-axis headroom so top value notes never touch or overlap top borders
    max_val = max(values)
    min_val = min(values)
    y_top = max_val * 1.18 if max_val > 0 else max_val * 0.85
    y_bottom = 0 if min_val >= 0 else min_val * 1.15
    ax.set_ylim(y_bottom, y_top)

    # Annotate each column with clear, concise numerical notes
    for i, bar in enumerate(bars):
        height = bar.get_height()
        is_best = (i == best_idx)
        val_str = f"{height:.2f}" if abs(height) >= 10 else f"{height:.3f}"
        note_str = f"{val_str}\n★ Best" if is_best else val_str
        
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (max_val * 0.02),
            note_str,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if is_best else "medium",
            color="#1b5e20" if is_best else "#212529"
        )

    # Legend / Info note
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#4c72b0", edgecolor="black", label="Raw Features"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#7097c2", edgecolor="black", label="Basis Expansion"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#2b8a3e", edgecolor="black", label="Best Model (★)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9.5, framealpha=0.9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Model Variant", fontsize=11, fontweight="semibold", labelpad=8)
    ax.set_ylabel(metric_name, fontsize=11, fontweight="semibold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_multi_metric_comparison(
    summary_df,
    title: str = "Test Set Performance Comparison Across All Metrics",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Creates a 3-panel subplots comparing RMSE, MAE, and R2 side-by-side with concise column notes.
    """
    import pandas as pd
    if isinstance(summary_df, list):
        summary_df = pd.DataFrame(summary_df)

    models = summary_df["Model"].tolist()
    # Compact labels for subplots
    compact_labels = []
    for m in models:
        short = format_short_model_name(m)
        compact_labels.append(short.replace("(Expanded)", "(Exp)"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    metrics = [
        ("RMSE", "Test RMSE (Lower is Better)", False),
        ("MAE", "Test MAE (Lower is Better)", False),
        ("R2", "Test R² Score (Higher is Better)", True)
    ]

    for ax, (col, ylabel, is_higher_better) in zip(axes, metrics):
        if col not in summary_df.columns:
            continue

        vals = summary_df[col].values
        best_i = int(np.argmax(vals) if is_higher_better else np.argmin(vals))

        bar_colors = ["#2b8a3e" if i == best_i else ("#4c72b0" if "Raw" in models[i] else "#7097c2") for i in range(len(models))]
        bars = ax.bar(range(len(models)), vals, color=bar_colors, edgecolor="#212529", width=0.52, linewidth=1.1)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(compact_labels, fontsize=10, fontweight="semibold")
        ax.set_title(col, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=10.5, fontweight="medium")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        max_v = max(vals)
        ax.set_ylim(0, max_v * 1.22 if max_v > 0 else 1.0)

        for i, bar in enumerate(bars):
            h = bar.get_height()
            is_best = (i == best_i)
            v_str = f"{h:.2f}" if abs(h) >= 10 else f"{h:.3f}"
            lbl = f"{v_str}\n★ Best" if is_best else v_str
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + (max_v * 0.02),
                lbl,
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold" if is_best else "medium",
                color="#1b5e20" if is_best else "#212529"
            )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
