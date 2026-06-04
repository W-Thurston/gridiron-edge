# src/gridiron_edge/evaluation/diagnostics.py

"""Model diagnostic plots and comparative evaluation.

Generates PNG plots saved to ``data/output/evaluation/`` for both
single-model inspection and multi-model comparison.

Single-model diagnostics (``plot_single_model``):
    calibration_curve       Predicted prob vs actual win rate
    confidence_distribution Histogram of predicted probabilities
    roc_curve               ROC curve with AUC score
    reliability_diagram     Calibration curve with confidence intervals
    brier_decomposition     Bar chart: reliability / resolution / uncertainty
    feature_importance      Coefficients (logistic) or importance (tree)
    performance_by_context  Accuracy broken down by week, season, margin

Multi-model comparison (``plot_model_comparison``):
    calibration_overlay     All models on one calibration plot
    roc_overlay             All models on one ROC plot
    metric_comparison       Bar chart of Brier / log-loss / AUC across models
    agreement_matrix        How often pairs of models agree on game outcome

All functions return the Path to the written PNG file.
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Final, Literal

from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from numpy import dtype, float64, ndarray
import pandas as pd
from pandas import DataFrame, Index, Series

from gridiron_edge.evaluation.metrics import (
    brier_decomposition,
    calibration_table,
    expected_calibration_error,
    roc_auc,
)
from gridiron_edge.models.artifact import ModelMetadata

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_EVAL_DIR: Final[str] = "data/output/evaluation"
_DPI: Final[int] = 150
_FIGSIZE_SINGLE: Final[tuple[int, int]] = (8, 6)
_FIGSIZE_WIDE: Final[tuple[int, int]] = (12, 6)
_FIGSIZE_GRID: Final[tuple[int, int]] = (14, 10)

# Consistent colour palette for model versions.
# Each registered model version has a distinct colour so multi-model
# comparison plots remain readable. Add new entries here when registering
# a new model variant.
_MODEL_COLORS: Final[dict[str, str]] = {
    "elo_v1": "#2563eb",  # blue
    "elo_v2": "#7c3aed",  # purple
    "elo_v3": "#db2777",  # pink
    # Champion models (unversioned)
    "random_forest": "#0891b2",  # cyan
    "xgboost": "#15803d",  # green
    "logistic": "#d97706",  # amber
}
_DEFAULT_COLOR: Final[str] = "#6b7280"  # gray fallback


def _model_color(model_version: str) -> str:
    """Return consistent colour for a model version."""
    return _MODEL_COLORS.get(model_version, _DEFAULT_COLOR)


def _output_dir(repo: Path, model_version: str | None = None) -> Path:
    """Return and create the output directory for evaluation plots."""
    base: Path = repo / _EVAL_DIR
    directory: Path = base / model_version if model_version else base
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save figure and close it."""
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Single-model diagnostics
# ---------------------------------------------------------------------------


def plot_calibration_curve(
    eval_df: pd.DataFrame,
    *,
    repo: Path,
    n_buckets: int = 10,
) -> Path:
    """Plot predicted probability vs actual win rate (calibration curve).

    A perfectly calibrated model follows the diagonal. Points above the
    diagonal mean the model is underconfident; below means overconfident.

    Args:
        eval_df: Output of ``build_evaluation_df()`` for one model.
        repo: Repository root (determines output path).
        n_buckets: Number of probability buckets.

    Returns:
        Path to the written PNG file.
    """
    model = eval_df["model_version"].iloc[0]
    cal: DataFrame = calibration_table(eval_df, n_buckets=n_buckets)
    ece: float = expected_calibration_error(eval_df["away_win_prob"], eval_df["away_team_won"])

    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    color: str = _model_color(model)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Perfect calibration")

    # Calibration curve
    ax.plot(
        cal["mean_predicted_prob"],
        cal["actual_win_rate"],
        "o-",
        color=color,
        linewidth=2,
        markersize=6,
        label=f"{model}  (ECE={ece:.4f})",
    )

    # Shade the error region
    ax.fill_between(
        cal["mean_predicted_prob"],
        cal["actual_win_rate"],
        cal["mean_predicted_prob"],
        alpha=0.15,
        color=color,
    )

    # Rug plot: bucket sizes as bar heights at bottom
    ax2: Axes = ax.twinx()
    ax2.bar(
        cal["bucket_mid"],
        cal["n_games"],
        width=0.08,
        alpha=0.2,
        color=color,
    )
    ax2.set_ylabel("Games per bucket", fontsize=9, alpha=0.6)
    ax2.tick_params(axis="y", labelsize=8, colors="gray")

    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Actual win rate", fontsize=11)
    ax.set_title(f"Calibration Curve — {model}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    out: Path = _output_dir(repo, model) / "calibration_curve.png"
    return _save(fig, out)


def plot_confidence_distribution(
    eval_df: pd.DataFrame,
    *,
    repo: Path,
    n_bins: int = 20,
) -> Path:
    """Plot histogram of predicted probabilities.

    Shows how confident the model is across all games. A healthy
    distribution is roughly bell-shaped centred near 0.6 (home
    advantage). A spike at 0.5 suggests the model is underconfident.

    Args:
        eval_df: Output of ``build_evaluation_df()`` for one model.
        repo: Repository root.
        n_bins: Number of histogram bins.

    Returns:
        Path to the written PNG file.
    """
    model = eval_df["model_version"].iloc[0]

    # Split by correct / incorrect predictions (exclude ties)
    no_ties: DataFrame = eval_df.loc[eval_df["away_team_won"] != 0.5, :].copy()
    correct: DataFrame = no_ties.loc[
        ((no_ties["away_win_prob"] > 0.5) & (no_ties["away_team_won"] == 1.0))
        | ((no_ties["away_win_prob"] < 0.5) & (no_ties["away_team_won"] == 0.0)),
        :,
    ]
    # pyrefly: ignore [unsupported-operation]
    incorrect: DataFrame = no_ties.loc[~no_ties.index.isin(correct.index), :]

    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)

    bins_list: list[float] = list(np.linspace(0, 1, n_bins + 1))
    ax.hist(
        correct["away_win_prob"],
        bins=bins_list,
        alpha=0.6,
        color="#16a34a",
        label=f"Correct ({len(correct):,})",
        edgecolor="white",
    )
    ax.hist(
        incorrect["away_win_prob"],
        bins=bins_list,
        alpha=0.6,
        color="#dc2626",
        label=f"Incorrect ({len(incorrect):,})",
        edgecolor="white",
    )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Predicted away-win probability", fontsize=11)
    ax.set_ylabel("Number of games", fontsize=11)
    ax.set_title(f"Confidence Distribution — {model}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    accuracy: Literal[0] | float = len(correct) / len(no_ties) if len(no_ties) > 0 else 0
    ax.text(
        0.98,
        0.97,
        f"Accuracy: {accuracy:.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    out: Path = _output_dir(repo, model) / "confidence_distribution.png"
    return _save(fig, out)


def plot_roc_curve(
    eval_df: pd.DataFrame,
    *,
    repo: Path,
) -> Path:
    """Plot ROC curve with AUC score.

    Shows the tradeoff between true positive rate (sensitivity) and
    false positive rate (1 - specificity) at all thresholds. AUC = 1.0
    is perfect; AUC = 0.5 is random.

    Args:
        eval_df: Output of ``build_evaluation_df()`` for one model.
        repo: Repository root.

    Returns:
        Path to the written PNG file.
    """
    model = eval_df["model_version"].iloc[0]
    color: str = _model_color(model)

    no_ties = eval_df.loc[eval_df["away_team_won"] != 0.5, :]
    p = no_ties["away_win_prob"].to_numpy()
    y = no_ties["away_team_won"].to_numpy()

    # Build ROC curve
    order = p.argsort()[::-1]
    y_sorted = y[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos

    tpr: ndarray = np.concatenate([[0], y_sorted.cumsum() / n_pos])
    fpr: ndarray = np.concatenate([[0], (1 - y_sorted).cumsum() / n_neg])
    auc = float(np.trapezoid(tpr, fpr))

    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random (AUC=0.50)")
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{model}  (AUC={auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.1, color=color)

    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title(f"ROC Curve — {model}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    out: Path = _output_dir(repo, model) / "roc_curve.png"
    return _save(fig, out)


def plot_brier_decomposition(
    eval_df: pd.DataFrame,
    *,
    repo: Path,
) -> Path:
    """Bar chart of Brier score decomposition: reliability, resolution, uncertainty.

    Reliability (lower is better): calibration error component.
    Resolution (higher is better): sharpness — how much predictions deviate
        from base rate.
    Uncertainty: inherent unpredictability. Fixed for the dataset.

    Args:
        eval_df: Output of ``build_evaluation_df()`` for one model.
        repo: Repository root.

    Returns:
        Path to the written PNG file.
    """
    model = eval_df["model_version"].iloc[0]
    decomp: dict[str, float] = brier_decomposition(
        eval_df["away_win_prob"], eval_df["away_team_won"]
    )

    components: list[str] = ["reliability", "resolution", "uncertainty"]
    values: list[float] = [decomp[c] for c in components]
    colors: list[str] = ["#dc2626", "#16a34a", "#6b7280"]
    labels: list[str] = [
        f"Reliability\n(lower=better)\n{decomp['reliability']:.5f}",
        f"Resolution\n(higher=better)\n{decomp['resolution']:.5f}",
        f"Uncertainty\n(fixed)\n{decomp['uncertainty']:.5f}",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars: BarContainer = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")

    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.5f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Value", fontsize=11)
    ax.set_title(
        f"Brier Score Decomposition — {model}\n"
        f"BS = Reliability - Resolution + Uncertainty = {decomp['brier_score']:.5f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")

    out: Path = _output_dir(repo, model) / "brier_decomposition.png"
    return _save(fig, out)


def plot_feature_importance(
    model_version: str,
    *,
    repo: Path,
) -> Path | None:
    """Plot feature coefficients (logistic) or importance (tree models).

    For logistic regression: plots signed coefficients showing direction
    and magnitude of each feature's influence.
    For XGBoost/tree models: plots gain-based feature importance.

    Args:
        model_version: Registered model version string.
        repo: Repository root.

    Returns:
        Path to the written PNG file, or None if the model type
        does not support feature importance.
    """
    from gridiron_edge.models.artifact import ArtifactStore

    store = ArtifactStore(repo)
    if not store.is_trained(model_version):
        logger.warning("%s: no artifact found.", model_version)
        return None

    pipeline = store.load(model_version)
    metadata: ModelMetadata = store.read_metadata(model_version)
    feature_names: list[str] = metadata.feature_columns

    # Try logistic regression coefficients
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        logger.warning("%s: no 'clf' step in pipeline.", model_version)
        return None

    if hasattr(clf, "coef_"):
        # Logistic regression — single coefficient per feature
        coefs = clf.coef_[0]
        df_imp: DataFrame = pd.DataFrame(
            {"feature": feature_names, "coefficient": coefs}
        ).sort_values("coefficient")

        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
        colors: list[str] = ["#dc2626" if c < 0 else "#2563eb" for c in df_imp["coefficient"]]
        ax.barh(df_imp["feature"], df_imp["coefficient"], color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient (positive = helps away team win)", fontsize=10)
        ax.set_title(
            f"Logistic Regression Coefficients — {model_version}",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

    elif hasattr(clf, "feature_importances_"):
        # Tree-based model — feature importance
        importances = clf.feature_importances_
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance"
        )

        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
        ax.barh(
            df_imp["feature"],
            df_imp["importance"],
            color=_model_color(model_version),
            edgecolor="white",
        )
        ax.set_xlabel("Feature importance (gain)", fontsize=10)
        ax.set_title(
            f"Feature Importance — {model_version}",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

    else:
        logger.info("%s: model type does not support feature importance plots.", model_version)
        return None

    out: Path = _output_dir(repo, model_version) / "feature_importance.png"
    return _save(fig, out)


def plot_performance_by_context(
    eval_df: pd.DataFrame,
    *,
    repo: Path,
) -> Path:
    """Grid of accuracy broken down by week, season, and confidence bucket.

    Reveals where the model is struggling: early-season weeks, specific
    seasons, or at particular confidence levels.

    Args:
        eval_df: Output of ``build_evaluation_df()`` for one model.
        repo: Repository root.

    Returns:
        Path to the written PNG file.
    """
    from gridiron_edge.evaluation.metrics import accuracy

    model = eval_df["model_version"].iloc[0]
    color: str = _model_color(model)
    no_ties = eval_df.loc[eval_df["away_team_won"] != 0.5, :].copy()

    fig, axes = plt.subplots(1, 3, figsize=_FIGSIZE_GRID)
    fig.suptitle(
        f"Performance by Context — {model}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    # --- By week ---
    ax = axes[0]
    week_acc: DataFrame = (
        no_ties.groupby("week")[["away_win_prob", "away_team_won"]]
        .apply(lambda d: accuracy(d["away_win_prob"], d["away_team_won"]))
        .reset_index()
    )
    week_acc.columns = ["week", "accuracy"]
    ax.bar(week_acc["week"], week_acc["accuracy"], color=color, alpha=0.8, edgecolor="white")
    ax.axhline(
        no_ties.pipe(lambda d: accuracy(d["away_win_prob"], d["away_team_won"])),
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Overall",
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Week", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("By Week", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- By season (last 10 seasons for readability) ---
    ax = axes[1]
    season_acc: DataFrame = (
        no_ties.groupby("season")[["away_win_prob", "away_team_won"]]
        .apply(lambda d: accuracy(d["away_win_prob"], d["away_team_won"]))
        .reset_index()
        .tail(10)
    )
    season_acc.columns = ["season", "accuracy"]
    ax.bar(
        range(len(season_acc)),
        season_acc["accuracy"],
        color=color,
        alpha=0.8,
        edgecolor="white",
    )
    ax.set_xticks(range(len(season_acc)))
    ax.set_xticklabels([s[:4] for s in season_acc["season"]], rotation=45, ha="right", fontsize=8)
    ax.axhline(
        no_ties.pipe(lambda d: accuracy(d["away_win_prob"], d["away_team_won"])),
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Season (start year)", fontsize=10)
    ax.set_title("By Season (last 10)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # --- By confidence bucket ---
    ax = axes[2]
    no_ties["conf_bucket"] = pd.cut(
        no_ties["away_win_prob"].clip(0, 1),
        bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0],
        labels=["<40%", "40-50%", "50-60%", "60-70%", ">70%"],
    )
    bucket_acc: DataFrame = (
        no_ties.groupby("conf_bucket", observed=True)[["away_win_prob", "away_team_won"]]
        .apply(
            lambda d: pd.Series(
                {
                    "accuracy": accuracy(d["away_win_prob"], d["away_team_won"]),
                    "n": len(d),
                }
            )
        )
        .reset_index()
    )
    bars = ax.bar(
        bucket_acc["conf_bucket"].astype(str),
        bucket_acc["accuracy"],
        color=color,
        alpha=0.8,
        edgecolor="white",
    )
    for bar, n in zip(bars, bucket_acc["n"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"n={int(n)}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Confidence bucket", fontsize=10)
    ax.set_title("By Confidence", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out: Path = _output_dir(repo, model) / "performance_by_context.png"
    return _save(fig, out)


def plot_single_model(
    eval_df: pd.DataFrame,
    *,
    repo: Path,
) -> list[Path]:
    """Generate all single-model diagnostic plots.

    Convenience wrapper that calls all individual plot functions and
    returns all output paths.

    Args:
        eval_df: Output of ``build_evaluation_df()`` for one model.
        repo: Repository root.

    Returns:
        List of paths to all written PNG files.
    """
    model = eval_df["model_version"].iloc[0]
    paths: list[Path] = []

    paths.append(plot_calibration_curve(eval_df, repo=repo))
    paths.append(plot_confidence_distribution(eval_df, repo=repo))
    paths.append(plot_roc_curve(eval_df, repo=repo))
    paths.append(plot_brier_decomposition(eval_df, repo=repo))
    paths.append(plot_performance_by_context(eval_df, repo=repo))

    fi_path: Path | None = plot_feature_importance(model, repo=repo)
    if fi_path is not None:
        paths.append(fi_path)

    logger.info("Single-model diagnostics complete: %d plots for %s", len(paths), model)
    return paths


# ---------------------------------------------------------------------------
# Multi-model comparison plots
# ---------------------------------------------------------------------------


def plot_calibration_overlay(
    eval_dfs: dict[str, pd.DataFrame],
    *,
    repo: Path,
    n_buckets: int = 10,
) -> Path:
    """Overlay calibration curves for all models on one plot.

    Args:
        eval_dfs: Dict mapping model_version string to its eval DataFrame.
        repo: Repository root.
        n_buckets: Number of probability buckets for calibration.

    Returns:
        Path to the written PNG file.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Perfect")

    for model, eval_df in eval_dfs.items():
        cal: DataFrame = calibration_table(eval_df, n_buckets=n_buckets)
        ece: float = expected_calibration_error(eval_df["away_win_prob"], eval_df["away_team_won"])
        auc: float = roc_auc(eval_df["away_win_prob"], eval_df["away_team_won"])
        ax.plot(
            cal["mean_predicted_prob"],
            cal["actual_win_rate"],
            "o-",
            color=_model_color(model),
            linewidth=2,
            markersize=5,
            label=f"{model}  ECE={ece:.4f}  AUC={auc:.4f}",
        )

    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Actual win rate", fontsize=11)
    ax.set_title("Calibration Curve Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    out: Path = _output_dir(repo) / "comparison_calibration.png"
    return _save(fig, out)


def plot_roc_overlay(
    eval_dfs: dict[str, pd.DataFrame],
    *,
    repo: Path,
) -> Path:
    """Overlay ROC curves for all models on one plot.

    Args:
        eval_dfs: Dict mapping model_version string to its eval DataFrame.
        repo: Repository root.

    Returns:
        Path to the written PNG file.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random (0.50)")

    for model, eval_df in eval_dfs.items():
        no_ties = eval_df.loc[eval_df["away_team_won"] != 0.5, :].copy()
        p = no_ties["away_win_prob"].to_numpy()
        y = no_ties["away_team_won"].to_numpy()

        order = p.argsort()[::-1]
        y_sorted = y[order]
        n_pos = y_sorted.sum()
        n_neg = len(y_sorted) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue

        tpr: ndarray = np.concatenate([[0], y_sorted.cumsum() / n_pos])
        fpr: ndarray = np.concatenate([[0], (1 - y_sorted).cumsum() / n_neg])
        auc = float(np.trapezoid(tpr, fpr))

        ax.plot(
            fpr,
            tpr,
            color=_model_color(model),
            linewidth=2,
            label=f"{model}  (AUC={auc:.4f})",
        )

    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    out: Path = _output_dir(repo) / "comparison_roc.png"
    return _save(fig, out)


def plot_metric_comparison(
    eval_dfs: dict[str, pd.DataFrame],
    *,
    repo: Path,
) -> Path:
    """Bar chart comparing Brier score, log loss, and AUC across models.

    Args:
        eval_dfs: Dict mapping model_version string to its eval DataFrame.
        repo: Repository root.

    Returns:
        Path to the written PNG file.
    """
    from gridiron_edge.evaluation.metrics import brier_score, log_loss

    rows: list[dict[str, float | str]] = []
    for model, eval_df in eval_dfs.items():
        p: Series = eval_df["away_win_prob"]
        y: Series = eval_df["away_team_won"]
        rows.append(
            {
                "model": model,
                "Brier Score": round(brier_score(p, y), 4),
                "Log Loss": round(log_loss(p, y), 4),
                "ROC-AUC": round(roc_auc(p, y), 4),
            }
        )

    df: DataFrame = pd.DataFrame(rows).set_index("model")

    fig, axes = plt.subplots(1, 3, figsize=_FIGSIZE_GRID)
    fig.suptitle("Model Comparison", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, ["Brier Score", "Log Loss", "ROC-AUC"], strict=True):
        vals: Series = df[metric]
        colors: list[str] = [_model_color(m) for m in vals.index]
        bars = ax.bar(vals.index, vals.values, color=colors, edgecolor="white")
        for bar, val in zip(bars, vals.values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(metric, fontsize=11)
        ax.set_xticklabels(vals.index, rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        lower_better: bool = metric != "ROC-AUC"
        ax.set_ylabel("Lower is better" if lower_better else "Higher is better", fontsize=9)

    plt.tight_layout()
    out: Path = _output_dir(repo) / "comparison_metrics.png"
    return _save(fig, out)


def plot_agreement_matrix(
    eval_dfs: dict[str, pd.DataFrame],
    *,
    repo: Path,
) -> Path:
    """Heatmap showing how often pairs of models agree on game outcome.

    Agreement = both models predict the same winner (away_win_prob > 0.5
    for the same team). High agreement means models are learning similar
    things; low agreement reveals where models diverge.

    Args:
        eval_dfs: Dict mapping model_version string to its eval DataFrame.
        repo: Repository root.

    Returns:
        Path to the written PNG file.
    """
    models: list[str] = list(eval_dfs.keys())
    n: int = len(models)

    # Build prediction (1 = predict away wins, 0 = predict home wins) per game
    preds: dict[str, pd.Series] = {}
    for model, eval_df in eval_dfs.items():
        game_preds = (
            eval_df[["game_id", "away_win_prob"]]
            .set_index("game_id")["away_win_prob"]
            .gt(0.5)
            .astype(int)
        )
        preds[model] = game_preds

    # Compute pairwise agreement on common games
    matrix: ndarray[tuple[int, int], dtype[float64]] = np.zeros((n, n))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            common: Index = preds[m1].index.intersection(preds[m2].index)
            if len(common) == 0:
                matrix[i, j] = float("nan")
            else:
                agreement = (preds[m1][common] == preds[m2][common]).mean()
                matrix[i, j] = agreement

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n)))
    im: AxesImage = ax.imshow(matrix, vmin=0.5, vmax=1.0, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Agreement rate")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(models, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black" if val > 0.65 else "white",
                )

    ax.set_title("Model Agreement Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out: Path = _output_dir(repo) / "comparison_agreement.png"
    return _save(fig, out)


def plot_model_comparison(
    eval_dfs: dict[str, pd.DataFrame],
    *,
    repo: Path,
) -> list[Path]:
    """Generate all multi-model comparison plots.

    Args:
        eval_dfs: Dict mapping model_version string to its eval DataFrame.
        repo: Repository root.

    Returns:
        List of paths to all written PNG files.
    """
    paths: list[Path] = [
        plot_calibration_overlay(eval_dfs, repo=repo),
        plot_roc_overlay(eval_dfs, repo=repo),
        plot_metric_comparison(eval_dfs, repo=repo),
        plot_agreement_matrix(eval_dfs, repo=repo),
    ]
    logger.info("Multi-model comparison complete: %d plots", len(paths))
    return paths
