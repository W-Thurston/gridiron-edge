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
from gridiron_edge.models.artifact import BaseModelMetadata

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_EVAL_DIR: Final[str] = "data/output/evaluation"
_DPI: Final[int] = 150
_FIGSIZE_SINGLE: Final[tuple[int, int]] = (8, 6)
_FIGSIZE_WIDE: Final[tuple[int, int]] = (12, 6)
_FIGSIZE_GRID: Final[tuple[int, int]] = (14, 10)

# Consistent colour palette for composite model keys. Keys match the
# PredictorRegistry composite-key format f"{model_name}_{model_type}".
# Add new entries here when registering a new model variant.
_MODEL_COLORS: Final[dict[str, str]] = {
    "win_prob_elo": "#2563eb",  # blue
    "win_prob_logistic": "#d97706",  # amber
    "win_prob_random_forest": "#0891b2",  # cyan
    "win_prob_xgboost": "#15803d",  # green
    "total_random_forest": "#7c3aed",  # purple
    "total_xgboost": "#db2777",  # pink
}
_DEFAULT_COLOR: Final[str] = "#6b7280"  # gray fallback


def _model_color(model_key: str) -> str:
    """Return consistent colour for a composite model key."""
    return _MODEL_COLORS.get(model_key, _DEFAULT_COLOR)


def _model_key(eval_df: pd.DataFrame) -> str:
    """Build composite model key from eval_df's canonical schema.

    The prediction archive uses (model_name, model_type) as the composite
    identity. This helper assembles the key for display labels and output
    paths.

    Args:
        eval_df: Evaluation DataFrame from build_evaluation_df.

    Returns:
        Composite key f"{model_name}_{model_type}".

    Raises:
        KeyError: If eval_df is missing model_name or model_type columns.
    """
    name: str = str(eval_df["model_name"].iloc[0])
    type_: str = str(eval_df["model_type"].iloc[0])
    return f"{name}_{type_}"


def _output_dir(repo: Path, model_key: str | None = None) -> Path:
    """Return and create the output directory for evaluation plots."""
    base: Path = repo / _EVAL_DIR
    directory: Path = base / model_key if model_key else base
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
    model: str = _model_key(eval_df)
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
    model: str = _model_key(eval_df)

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
    model: str = _model_key(eval_df)
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
    model: str = _model_key(eval_df)
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
    model_name: str,
    model_type: str,
    *,
    repo: Path,
) -> Path | None:
    """Plot feature coefficients (logistic) or importance (tree models).

    Handles the artifact shapes the codebase actually produces:
        - LogisticRegressionCV (bare): use .coef_
        - LogisticRegression (bare, wrapped in Pipeline): unwrap via named_steps
        - CalibratedClassifierCV: unwrap to base estimator
        - Pipeline([..., ("clf", CalibratedClassifierCV(...))]): unwrap both
        - RandomForestClassifier / RandomForestRegressor: use .feature_importances_
        - XGBClassifier / XGBRegressor: use .feature_importances_

    Args:
        model_name: Model purpose (e.g. ``"win_prob"``).
        model_type: Model algorithm (e.g. ``"random_forest"``).
        repo: Repository root.

    Returns:
        Path to the written PNG file, or None if the model artifact does
        not expose coefficients or importances.
    """
    from gridiron_edge.models.artifact import ArtifactStore

    store = ArtifactStore(repo)
    display: str = f"{model_name}/{model_type}"
    if not store.is_trained(model_name, model_type):
        logger.warning("%s: no artifact found.", display)
        return None

    artifact = store.load(model_name, model_type)
    metadata: BaseModelMetadata = store.read_metadata(model_name, model_type)
    feature_names: list[str] = metadata.feature_columns

    # Unwrap nested artifact structures to find the underlying estimator.
    estimator = _unwrap_estimator(artifact)
    if estimator is None:
        logger.info(
            "%s: could not unwrap estimator from artifact type %s.",
            display,
            type(artifact).__name__,
        )
        return None

    # Use the right attribute for this estimator type.
    coefs_or_importance, kind = _extract_importance(estimator, feature_names)
    if coefs_or_importance is None:
        logger.info(
            "%s: estimator %s does not expose coef_ or feature_importances_.",
            display,
            type(estimator).__name__,
        )
        return None

    df_imp: DataFrame = pd.DataFrame(
        {"feature": feature_names, "value": coefs_or_importance}
    ).sort_values("value")

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    if kind == "coefficient":
        colors: list[str] = ["#dc2626" if v < 0 else "#2563eb" for v in df_imp["value"]]
        ax.barh(df_imp["feature"], df_imp["value"], color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient (positive = helps away team win)", fontsize=10)
        ax.set_title(
            f"Logistic Regression Coefficients — {display}",
            fontsize=12,
            fontweight="bold",
        )
    else:  # importance
        ax.barh(
            df_imp["feature"],
            df_imp["value"],
            color=_model_color(f"{model_name}_{model_type}"),
            edgecolor="white",
        )
        ax.set_xlabel("Feature importance (gain)", fontsize=10)
        ax.set_title(
            f"Feature Importance — {display}",
            fontsize=12,
            fontweight="bold",
        )
    ax.grid(True, alpha=0.3, axis="x")

    out: Path = _output_dir(repo, f"{model_name}_{model_type}") / "feature_importance.png"
    return _save(fig, out)


def _unwrap_estimator(artifact: object) -> object | None:
    """Unwrap nested wrappers to find the underlying fittable estimator.

    Handles Pipeline (extracts the final 'clf' step) and
    CalibratedClassifierCV (extracts the calibrated_classifiers_[0].estimator).
    Returns the artifact unchanged if it's already a bare estimator.
    Returns None if no recognized structure is found.
    """
    # pyrefly: ignore [missing-import]
    from sklearn.calibration import CalibratedClassifierCV

    # pyrefly: ignore [missing-import]
    from sklearn.pipeline import Pipeline

    # Pipeline: unwrap to the 'clf' step (or last step if no clf)
    if isinstance(artifact, Pipeline):
        if "clf" in artifact.named_steps:
            inner = artifact.named_steps["clf"]
        else:
            # Last step is conventionally the estimator
            inner = artifact.steps[-1][1]
        # Recurse — the inner could itself be CalibratedClassifierCV
        return _unwrap_estimator(inner)

    # CalibratedClassifierCV: unwrap to the underlying estimator. After
    # fitting, calibrated_classifiers_ holds the fitted folds; their
    # .estimator attribute is the actual model. We use the first fold's
    # estimator as representative.
    if isinstance(artifact, CalibratedClassifierCV):
        if hasattr(artifact, "calibrated_classifiers_") and artifact.calibrated_classifiers_:
            return artifact.calibrated_classifiers_[0].estimator
        # Pre-fit (shouldn't happen for a loaded artifact), fall through
        # pyrefly: ignore [missing-attribute]
        return artifact.estimator

    # Bare estimator
    return artifact


def _extract_importance(
    estimator: object,
    feature_names: list[str],
) -> tuple[list[float] | None, str | None]:
    """Extract coefficients or feature importances from an unwrapped estimator.

    Returns:
        Tuple of (values, kind). ``kind`` is ``"coefficient"`` for linear
        models or ``"importance"`` for tree models. Both are ``None`` if
        the estimator type isn't supported.
    """
    # Linear models: coef_
    if hasattr(estimator, "coef_"):
        coef = estimator.coef_
        # For binary classification, coef_ is shape (1, n_features); flatten
        if coef.ndim == 2:
            coef = coef[0]
        if len(coef) != len(feature_names):
            logger.warning(
                "Coefficient length (%d) does not match feature_names (%d); skipping.",
                len(coef),
                len(feature_names),
            )
            return None, None
        return list(coef), "coefficient"

    # Tree models: feature_importances_
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        if len(importances) != len(feature_names):
            logger.warning(
                "Importance length (%d) does not match feature_names (%d); skipping.",
                len(importances),
                len(feature_names),
            )
            return None, None
        return list(importances), "importance"

    return None, None


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

    model: str = _model_key(eval_df)
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
    model: str = _model_key(eval_df)
    paths: list[Path] = []

    paths.append(plot_calibration_curve(eval_df, repo=repo))
    paths.append(plot_confidence_distribution(eval_df, repo=repo))
    paths.append(plot_roc_curve(eval_df, repo=repo))
    paths.append(plot_brier_decomposition(eval_df, repo=repo))
    paths.append(plot_performance_by_context(eval_df, repo=repo))

    # Feature importance plot (best-effort; returns None if model type
    # doesn't expose coefficients or importances).
    m_name: str = str(eval_df["model_name"].iloc[0])
    m_type: str = str(eval_df["model_type"].iloc[0])
    fi_path: Path | None = plot_feature_importance(m_name, m_type, repo=repo)
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
        eval_dfs: Dict mapping composite model_key string to its eval DataFrame.
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
        eval_dfs: Dict mapping composite model_key string to its eval DataFrame.
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
        eval_dfs: Dict mapping composite model_key string to its eval DataFrame.
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
        eval_dfs: Dict mapping composite model_key string to its eval DataFrame.
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
        eval_dfs: Dict mapping composite model_key string to its eval DataFrame.
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
