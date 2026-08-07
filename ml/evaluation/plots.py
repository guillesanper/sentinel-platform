"""
Figures for both Sentinel ML tracks.

**This module is never imported from a package ``__init__``, and never at module
level from anything on the serving path.** It pulls in matplotlib and seaborn,
which live in ``requirements-dev.txt``; ``api/Dockerfile`` copies ``ml/``
wholesale while installing only ``requirements.txt``, so an import chain that
reaches this file from ``api.main`` would either break the production image or
add ~120 MB of plotting stack and CVE surface to a security product. Import it
inside the branch that actually needs a figure::

    if plots:
        from ml.evaluation import plots
        plots.plot_confusion_matrix(...)

Ported from the CICIDS2017 TFA (``evaluation/visualization.py``). The
duplicated, non-parameterised ``_plot_*`` helpers that used to live in
``utils/loader.py`` were dropped in favour of the parameterised versions here —
that is what lets :mod:`ml.flow.loader` be imported without matplotlib.
``plot_split_distribution`` is new: it makes the temporal partition auditable,
which is the one thing in the flow track that fails silently.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

# Headless backend, selected before pyplot is imported: these figures are
# written to disk from CI runners and containers with no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import structlog
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import LabelBinarizer

logger = structlog.get_logger()


# ============================================================================
# Shared style
# ============================================================================

_PALETTE: str = "tab10"
_FIG_DPI: int = 150
_CMAP_CM: str = "Blues"
_CMAP_COR: str = "coolwarm"

# Column names used by the flow track. Passed as arguments rather than imported
# from ml.flow.loader: this package is shared by both tracks and must not depend
# on either one.
_DEFAULT_LABEL_COL: str = "Label"
_DEFAULT_DAY_COL: str = "capture_day"

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def _save_fig(fig: Figure, path: Path) -> None:
    """Tighten, write and close a figure, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("figure_saved", path=str(path))


def _as_dataframe(X: "pd.DataFrame | np.ndarray") -> pd.DataFrame:
    """Wrap a bare ndarray in a DataFrame with positional column names."""
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return X


def _flat_axes(axes: Any) -> np.ndarray:
    """
    Normalise whatever ``plt.subplots`` returned into a 1-D array of Axes.

    A 1x1 grid yields a bare Axes rather than an array, which every caller here
    would otherwise have to special-case.
    """
    flat: np.ndarray = np.atleast_1d(np.asarray(axes)).ravel()
    return flat


# ============================================================================
# EDA
# ============================================================================


def plot_class_distribution(
    y: "pd.Series | np.ndarray",
    output_dir: Path,
    *,
    title: str = "Class Distribution",
) -> None:
    """
    Bar chart of sample counts per class → ``class_distribution.png``.

    Args:
        y: 1-D array-like of string class labels.
        output_dir: Directory where the figure is written.
        title: Figure title.
    """
    y_s = y if isinstance(y, pd.Series) else pd.Series(y)
    counts = y_s.value_counts().sort_index()
    colors = sns.color_palette(_PALETTE, n_colors=len(counts))

    fig, ax = plt.subplots(figsize=(9, 5))
    values = counts.to_numpy()
    bars = ax.bar(counts.index, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + float(values.max()) * 0.01,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Sample Count")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="x", rotation=25)

    _save_fig(fig, Path(output_dir) / "class_distribution.png")


def plot_feature_correlation(
    X: "pd.DataFrame | np.ndarray",
    output_dir: Path,
    *,
    top_n: int = 20,
    title: str = "Feature Correlation Matrix (Top Variance Features)",
) -> None:
    """
    Pearson correlation heatmap of the highest-variance features →
    ``feature_correlation.png``.

    Args:
        X: Feature matrix (DataFrame with column names, or ndarray).
        output_dir: Output directory.
        top_n: Number of features selected by descending variance.
        title: Figure title.
    """
    X = _as_dataframe(X)
    top_cols = X.var().nlargest(top_n).index.tolist()
    corr = X[top_cols].corr()

    size = max(8, top_n * 0.55)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(
        corr,
        ax=ax,
        cmap=_CMAP_COR,
        center=0,
        vmin=-1,
        vmax=1,
        annot=(top_n <= 15),
        fmt=".2f",
        annot_kws={"size": 7},
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    _save_fig(fig, Path(output_dir) / "feature_correlation.png")


def plot_feature_distributions(
    X: "pd.DataFrame | np.ndarray",
    y: "pd.Series | np.ndarray",
    output_dir: Path,
    *,
    top_n: int = 6,
    title: str = "Feature Distributions by Class",
) -> None:
    """
    Grid of per-class KDE curves for the highest-variance features →
    ``feature_distributions.png``.

    Args:
        X: Feature matrix aligned with ``y``.
        y: Class labels, shape (n,).
        output_dir: Output directory.
        top_n: Number of features to display.
        title: Figure suptitle.
    """
    X = _as_dataframe(X)
    y_arr = np.asarray(y)
    top_cols = X.var().nlargest(top_n).index.tolist()
    classes = np.unique(y_arr)
    palette = dict(zip(classes, sns.color_palette(_PALETTE, n_colors=len(classes)), strict=True))

    ncols = min(3, top_n)
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = _flat_axes(axes)

    for i, col in enumerate(top_cols):
        ax = axes_flat[i]
        col_vals = X[col].to_numpy()
        for cls in classes:
            data = col_vals[y_arr == cls]
            try:
                sns.kdeplot(
                    data,
                    ax=ax,
                    label=str(cls),
                    color=palette[cls],
                    fill=True,
                    alpha=0.3,
                    linewidth=1.2,
                )
            # A degenerate class (constant or single-sample) has no KDE;
            # seaborn's failure mode varies by version, hence the broad catch.
            except Exception:
                ax.hist(data, bins=30, alpha=0.4, label=str(cls), color=palette[cls], density=True)
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("Density")
        ax.set_xlabel("")

    for j in range(len(top_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles, lbls = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8, title="Class")
    fig.suptitle(title, fontsize=12, y=1.01)

    _save_fig(fig, Path(output_dir) / "feature_distributions.png")


def run_eda(
    X: "pd.DataFrame | np.ndarray",
    y: "pd.Series | np.ndarray",
    output_dir: Path,
    *,
    corr_top_n: int = 20,
    dist_top_n: int = 6,
) -> None:
    """
    Generate the three exploratory figures in one call.

    Convenience wrapper so the training entry point has a single call inside its
    ``if eda:`` branch instead of three.

    Args:
        X: Feature matrix — build it with
            ``ml.flow.loader.get_feature_columns`` so the provenance columns
            never reach the figures.
        y: Matching class labels.
        output_dir: Directory for the three PNGs.
        corr_top_n: Features in the correlation heatmap.
        dist_top_n: Features in the distribution grid.
    """
    output_dir = Path(output_dir)
    logger.info("eda_started", output_dir=str(output_dir))

    plot_class_distribution(y, output_dir)
    plot_feature_correlation(X, output_dir, top_n=corr_top_n)
    plot_feature_distributions(X, y, output_dir, top_n=dist_top_n)

    logger.info("eda_finished", output_dir=str(output_dir))


def plot_split_distribution(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    *,
    label_col: str = _DEFAULT_LABEL_COL,
    day_col: str = _DEFAULT_DAY_COL,
    title: str = "Dataset partition — rows per (capture day, class)",
) -> None:
    """
    Rows per split for every ``(capture_day, class)`` block →
    ``split_distribution.png``.

    Makes the temporal partition auditable at a glance: a class missing from a
    split, or a block too small to yield a meaningful test slice, is visible
    here and nowhere else — neither condition raises an error, and both quietly
    invalidate the per-class metrics.

    Args:
        train: Training split.
        val: Validation split.
        test: Test split.
        output_dir: Output directory.
        label_col: Name of the label column.
        day_col: Name of the capture-day column. When absent (a frame that never
            went through ``load_cicids2017``), blocks degrade to classes.
        title: Figure title.
    """
    splits = {"train": train, "val": val, "test": test}
    group_cols = [day_col, label_col] if day_col in train.columns else [label_col]

    counts = {name: frame.groupby(group_cols).size() for name, frame in splits.items()}
    keys = sorted({key for series in counts.values() for key in series.index})
    if not keys:
        logger.warning("split_distribution_empty", hint="no rows in any split; figure skipped")
        return

    def _key_label(key: Any) -> str:
        return " / ".join(str(part) for part in key) if isinstance(key, tuple) else str(key)

    tick_labels = [_key_label(key) for key in keys]
    y_pos = np.arange(len(keys))
    height = 0.26
    palette = sns.color_palette(_PALETTE, n_colors=len(splits))

    fig, ax = plt.subplots(figsize=(10, max(4.0, len(keys) * 0.45)))
    for offset, ((name, series), color) in enumerate(zip(counts.items(), palette, strict=True)):
        block_counts = series.to_dict()
        values = [int(block_counts.get(key, 0)) for key in keys]
        # The y axis is inverted below, so the offset decreases with the split
        # order to keep train/val/test reading top to bottom.
        bars = ax.barh(
            y_pos + (offset - 1) * height,
            values,
            height=height,
            label=name,
            color=color,
            edgecolor="white",
        )
        # Exact counts, not just bar lengths: this figure exists to be audited,
        # and "how many test rows does Botnet actually have" is the question.
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {value:,}",
                va="center",
                fontsize=7,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Rows (log scale)")
    ax.set_xscale("symlog")
    ax.set_title(title)
    # Lower right: the top of the chart is where the longest (train) bars and
    # their count labels sit.
    ax.legend(fontsize=9, loc="lower right")
    ax.invert_yaxis()

    _save_fig(fig, Path(output_dir) / "split_distribution.png")


# ============================================================================
# Shared: confusion matrix
# ============================================================================


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    output_dir: Path,
    *,
    title: str = "Confusion Matrix",
    normalize: str = "true",
) -> None:
    """
    Row-normalised confusion matrix → ``confusion_matrix.png``.

    Args:
        y_true: Ground-truth string labels.
        y_pred: Predicted string labels.
        labels: Canonical class order (rows and columns).
        output_dir: Output directory.
        title: Figure title.
        normalize: ``'true'`` (row), ``'pred'`` (column) or ``'all'``.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(labels), normalize=normalize)
    size = max(6, len(labels) * 0.95)

    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    disp.plot(ax=ax, cmap=_CMAP_CM, colorbar=True, values_format=".2f")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    ax.set_title(title)

    _save_fig(fig, Path(output_dir) / "confusion_matrix.png")


# ============================================================================
# Stage 2: feature importance
# ============================================================================


def plot_feature_importance(
    feature_importance: dict[str, float],
    output_dir: Path,
    *,
    top_n: int = 20,
    title: str = "XGBoost Feature Importance",
) -> None:
    """
    Horizontal bar chart of the top-n feature importances →
    ``feature_importance.png``.

    Args:
        feature_importance: ``{feature_name: score}``, already sorted descending.
        output_dir: Output directory.
        top_n: Number of features to display.
        title: Figure title.
    """
    items = list(feature_importance.items())[:top_n]
    names = [k for k, _ in items][::-1]  # reversed so the highest sits on top
    scores = [v for _, v in items][::-1]
    max_score = max(scores) if scores else 1.0

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    color = sns.color_palette(_PALETTE, 1)[0]
    bars = ax.barh(names, scores, color=color, edgecolor="white")

    for bar, score in zip(bars, scores, strict=True):
        ax.text(
            bar.get_width() + max_score * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=7,
        )

    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    _save_fig(fig, Path(output_dir) / "feature_importance.png")


# ============================================================================
# Stage 1: threshold selection
# ============================================================================


def plot_threshold_curve(
    scores: np.ndarray,
    y_binary: np.ndarray,
    optimal_threshold: float,
    output_dir: Path,
    *,
    beta: float = 1.0,
    n_points: int = 300,
    title: str = "Isolation Forest — Threshold Optimisation",
) -> None:
    """
    Precision, recall and F-beta against the decision threshold →
    ``threshold_curve.png``.

    Args:
        scores: ``decision_function`` output; lower means more anomalous.
        y_binary: Binary ground truth (1 = attack, 0 = benign).
        optimal_threshold: Threshold maximising F-beta on validation.
        output_dir: Output directory.
        beta: Beta used during threshold optimisation.
        n_points: Number of threshold candidates swept.
        title: Figure title.
    """
    thresholds = np.linspace(scores.min(), scores.max(), num=n_points)
    precisions, recalls, fbetas = [], [], []

    for t in thresholds:
        y_pred = (scores < t).astype(int)
        if y_pred.sum() == 0:
            precisions.append(1.0)
            recalls.append(0.0)
            fbetas.append(0.0)
        else:
            precisions.append(float(precision_score(y_binary, y_pred, zero_division=0)))
            recalls.append(float(recall_score(y_binary, y_pred, zero_division=0)))
            fbetas.append(float(fbeta_score(y_binary, y_pred, beta=beta, zero_division=0)))

    beta_str = f"F{beta:.1g}".rstrip("0").rstrip(".")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, linewidth=1.8, label="Precision")
    ax.plot(thresholds, recalls, linewidth=1.8, label="Recall")
    ax.plot(thresholds, fbetas, linewidth=2.0, linestyle="--", label=beta_str)
    ax.axvline(
        optimal_threshold,
        color="red",
        linewidth=1.5,
        linestyle=":",
        label=f"Optimal  t = {optimal_threshold:.4f}",
    )
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9)

    _save_fig(fig, Path(output_dir) / "threshold_curve.png")


def plot_pr_tradeoff(
    scores: np.ndarray,
    y_binary: np.ndarray,
    optimal_threshold: float,
    output_dir: Path,
    *,
    beta: float = 1.0,
    title: str = "Isolation Forest — Precision-Recall Tradeoff",
) -> None:
    """
    PR curve of the Stage 1 detector with its operating point →
    ``pr_tradeoff.png``.

    An iso-F-beta contour through the operating point illustrates the FP/FN
    trade-off that ``beta`` controls.

    Args:
        scores: ``decision_function`` output; lower means more anomalous.
        y_binary: Binary ground truth (1 = attack, 0 = benign).
        optimal_threshold: Threshold maximising F-beta on validation.
        output_dir: Output directory.
        beta: Beta used when selecting the operating point.
        title: Figure title.
    """
    # sklearn wants "higher score = more positive"; negate so that a lower
    # Isolation Forest score reads as a higher anomaly score.
    prec_curve, rec_curve, _ = precision_recall_curve(y_binary, -scores)

    y_op = (scores < optimal_threshold).astype(int)
    op_prec = float(precision_score(y_binary, y_op, zero_division=0))
    op_rec = float(recall_score(y_binary, y_op, zero_division=0))
    op_fb = float(fbeta_score(y_binary, y_op, beta=beta, zero_division=0))

    # Iso-F-beta contour: P = F*R / ((1 + beta^2)*R - F*beta^2)
    beta2 = beta**2
    r_space = np.linspace(0.01, 1.0, 300)
    denom = (1.0 + beta2) * r_space - op_fb * beta2
    with np.errstate(divide="ignore", invalid="ignore"):
        iso_p = np.where(np.abs(denom) > 1e-10, op_fb * r_space / denom, np.nan)
    valid_mask = (iso_p > 0) & (iso_p <= 1.0) & ~np.isnan(iso_p)

    beta_str = f"{beta:.1g}".rstrip("0").rstrip(".")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec_curve, prec_curve, linewidth=1.8, label="PR curve")
    ax.scatter(
        [op_rec],
        [op_prec],
        s=120,
        zorder=5,
        color="red",
        label=f"Operating point (β={beta_str})\nP={op_prec:.3f}  R={op_rec:.3f}",
    )
    if valid_mask.any():
        ax.plot(
            r_space[valid_mask],
            iso_p[valid_mask],
            linestyle=":",
            color="grey",
            linewidth=1.0,
            label=f"Iso-F{beta_str} = {op_fb:.3f}",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8)

    _save_fig(fig, Path(output_dir) / "pr_tradeoff.png")


# ============================================================================
# Neural network training
# ============================================================================


def plot_training_curves(
    history: dict[str, list[float]],
    output_dir: Path,
    *,
    title: str = "Neural Network — Training Curves",
) -> None:
    """
    Loss and accuracy against epoch → ``training_curves.png``.

    A dotted vertical line marks the best-validation-loss epoch, i.e. where
    early stopping restored the weights from.

    Args:
        history: Dict with ``train_loss``, ``val_loss``, ``train_acc`` and
            ``val_acc``, as produced by ``NeuralNetwork.history``.
        output_dir: Output directory.
        title: Figure suptitle.
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    best_epoch = int(np.argmin(history["val_loss"])) + 1  # 1-indexed

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["train_loss"], linewidth=1.8, label="Train")
    ax_loss.plot(epochs, history["val_loss"], linewidth=1.8, linestyle="--", label="Val")
    ax_loss.axvline(
        best_epoch, color="red", linewidth=1.2, linestyle=":", label=f"Best epoch ({best_epoch})"
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(fontsize=9)

    ax_acc.plot(epochs, history["train_acc"], linewidth=1.8, label="Train")
    ax_acc.plot(epochs, history["val_acc"], linewidth=1.8, linestyle="--", label="Val")
    ax_acc.axvline(
        best_epoch, color="red", linewidth=1.2, linestyle=":", label=f"Best epoch ({best_epoch})"
    )
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend(fontsize=9)
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    fig.suptitle(title, fontsize=13)
    _save_fig(fig, Path(output_dir) / "training_curves.png")


# ============================================================================
# Model comparison
# ============================================================================

# Default legend names. The metrics module speaks of model_a / model_b since
# both tracks use it; in the flow track those are the two cascades.
_DEFAULT_LABEL_A: str = "Cascade XGB"
_DEFAULT_LABEL_B: str = "Cascade MLP"


def plot_metrics_bar(
    comparison_result: dict[str, Any],
    output_dir: Path,
    *,
    label_a: str = _DEFAULT_LABEL_A,
    label_b: str = _DEFAULT_LABEL_B,
    title: str = "Global Metrics",
) -> None:
    """
    Grouped bars for accuracy, F1-macro and F1-weighted → ``metrics_bar.png``.

    Args:
        comparison_result: Dict returned by
            :func:`ml.evaluation.metrics.compare_models` (keys ``model_a`` /
            ``model_b``).
        output_dir: Output directory.
        label_a: Legend name for model A.
        label_b: Legend name for model B.
        title: Figure title.
    """
    metrics_a = comparison_result["model_a"]
    metrics_b = comparison_result["model_b"]

    keys = ["accuracy", "f1_macro", "f1_weighted"]
    labels_nice = ["Accuracy", "F1-Macro", "F1-Weighted"]
    vals_a = [metrics_a.get(k, 0.0) for k in keys]
    vals_b = [metrics_b.get(k, 0.0) for k in keys]

    x = np.arange(len(keys))
    width = 0.35
    palette = sns.color_palette(_PALETTE, 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_a = ax.bar(
        x - width / 2, vals_a, width, label=label_a, color=palette[0], edgecolor="white"
    )
    bars_b = ax.bar(
        x + width / 2, vals_b, width, label=label_b, color=palette[1], edgecolor="white"
    )

    for bar in list(bars_a) + list(bars_b):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.005,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_nice)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(fontsize=9)

    _save_fig(fig, Path(output_dir) / "metrics_bar.png")


def plot_f1_per_class(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    output_dir: Path,
    *,
    label_a: str = _DEFAULT_LABEL_A,
    label_b: str = _DEFAULT_LABEL_B,
    title: str = "Per-Class F1 Score",
) -> None:
    """
    Grouped bars of per-class F1 for two models → ``f1_per_class.png``.

    Args:
        metrics_a: Metrics dict from
            :func:`ml.evaluation.metrics.compute_metrics` for model A.
        metrics_b: Same, for model B.
        output_dir: Output directory.
        label_a: Legend name for model A.
        label_b: Legend name for model B.
        title: Figure title.
    """
    f1_a = metrics_a.get("f1_per_class", {})
    f1_b = metrics_b.get("f1_per_class", {})
    classes = sorted(set(f1_a.keys()) | set(f1_b.keys()))

    vals_a = [f1_a.get(c, 0.0) for c in classes]
    vals_b = [f1_b.get(c, 0.0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35
    palette = sns.color_palette(_PALETTE, 2)

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.3), 5))
    bars_a = ax.bar(
        x - width / 2, vals_a, width, label=label_a, color=palette[0], edgecolor="white"
    )
    bars_b = ax.bar(
        x + width / 2, vals_b, width, label=label_b, color=palette[1], edgecolor="white"
    )

    for bar in list(bars_a) + list(bars_b):
        h = bar.get_height()
        if h > 0.02:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        else:
            # An explicit "0" instead of an invisible bar: a class the model
            # never predicts is a result, not missing data.
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.015,
                "0",
                ha="center",
                va="bottom",
                fontsize=6,
                color="#999999",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.legend(fontsize=9)

    _save_fig(fig, Path(output_dir) / "f1_per_class.png")


def plot_roc_curves(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    labels: Sequence[str],
    output_dir: Path,
    *,
    label_a: str = _DEFAULT_LABEL_A,
    label_b: str = _DEFAULT_LABEL_B,
    title: str = "ROC Curves",
) -> None:
    """
    One-vs-Rest ROC curve per class, both models overlaid → ``roc_curves.png``.

    Args:
        y_true: Ground-truth string labels, shape (n,).
        y_score_a: Class probability matrix for model A, shape (n, n_classes).
        y_score_b: Same, for model B.
        labels: Canonical class order matching the score column order.
        output_dir: Output directory.
        label_a: Legend name for model A.
        label_b: Legend name for model B.
        title: Figure suptitle.
    """
    _plot_ovr_curves(
        y_true,
        y_score_a,
        y_score_b,
        labels,
        output_dir,
        kind="roc",
        label_a=label_a,
        label_b=label_b,
        title=title,
    )


def plot_pr_curves(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    labels: Sequence[str],
    output_dir: Path,
    *,
    label_a: str = _DEFAULT_LABEL_A,
    label_b: str = _DEFAULT_LABEL_B,
    title: str = "Precision-Recall Curves",
) -> None:
    """
    One-vs-Rest PR curve per class, both models overlaid → ``pr_curves.png``.

    A dotted horizontal line marks the random-classifier baseline (the class
    prevalence), which for the rare classes sits close to zero.

    Args:
        y_true: Ground-truth string labels, shape (n,).
        y_score_a: Class probability matrix for model A, shape (n, n_classes).
        y_score_b: Same, for model B.
        labels: Canonical class order matching the score column order.
        output_dir: Output directory.
        label_a: Legend name for model A.
        label_b: Legend name for model B.
        title: Figure suptitle.
    """
    _plot_ovr_curves(
        y_true,
        y_score_a,
        y_score_b,
        labels,
        output_dir,
        kind="pr",
        label_a=label_a,
        label_b=label_b,
        title=title,
    )


def _plot_ovr_curves(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    labels: Sequence[str],
    output_dir: Path,
    *,
    kind: str,
    label_a: str,
    label_b: str,
    title: str,
) -> None:
    """Shared body of :func:`plot_roc_curves` and :func:`plot_pr_curves`."""
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_true)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    n_cls = len(labels)
    ncols = min(4, n_cls)
    nrows = (n_cls + ncols - 1) // ncols
    palette = sns.color_palette(_PALETTE, 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = _flat_axes(axes)

    for i, cls in enumerate(labels):
        ax = axes_flat[i]
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k:", linewidth=0.8)  # chance line, unlabelled
        else:
            baseline = float(y_bin[:, i].mean())
            # Label the baseline once, otherwise it repeats in the legend.
            ax.axhline(
                baseline,
                color="grey",
                linewidth=0.8,
                linestyle=":",
                label="Baseline" if i == 0 else "_nolegend_",
            )

        for scores, color, name, style in (
            (y_score_a, palette[0], label_a, "-"),
            (y_score_b, palette[1], label_b, "--"),
        ):
            col = min(i, scores.shape[1] - 1)
            try:
                if kind == "roc":
                    xs, ys, _ = roc_curve(y_bin[:, i], scores[:, col])
                else:
                    ys, xs, _ = precision_recall_curve(y_bin[:, i], scores[:, col])
            except (ValueError, IndexError) as exc:
                logger.debug("ovr_curve_skipped", kind=kind, label=str(cls), error=str(exc))
                continue
            ax.plot(xs, ys, color=color, linewidth=1.8, linestyle=style, label=name)

        ax.set_title(str(cls), fontsize=9)
        ax.set_xlabel("FPR" if kind == "roc" else "Recall")
        ax.set_ylabel("TPR" if kind == "roc" else "Precision")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)

    for j in range(n_cls, len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles, lbls = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=9)
    fig.suptitle(title, fontsize=13)

    filename = "roc_curves.png" if kind == "roc" else "pr_curves.png"
    _save_fig(fig, Path(output_dir) / filename)
