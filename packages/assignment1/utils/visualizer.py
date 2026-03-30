"""All plotting functions for the Jena EDA notebook."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    target: str = "T (degC)",
    figsize: tuple = (12, 10),
    save_path: Path | None = None,
) -> None:
    """Heatmap of Pearson correlations for all numeric columns."""
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"Correlation matrix (target: {target})", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()

    # Print top correlators with the target
    if target in corr:
        top = corr[target].drop(target).abs().sort_values(ascending=False).head(5)
        print(f"\nTop 5 correlations with {target}:\n{top}")


def plot_histograms(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    bins: int = 50,
    figsize: tuple = (16, 12),
    save_path: Path | None = None,
) -> None:
    """Distribution histograms for numeric columns."""
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()

    n = len(cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=bins, edgecolor="none", alpha=0.8)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature distributions", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_time_series(
    df: pd.DataFrame,
    col: str = "T (degC)",
    start: str | None = None,
    end: str | None = None,
    figsize: tuple = (16, 5),
    save_path: Path | None = None,
) -> None:
    """Raw series + 24 h and 7 d rolling means."""
    subset = df[col]
    if start or end:
        subset = subset.loc[start:end]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(subset, alpha=0.4, linewidth=0.6, label="raw")
    ax.plot(subset.rolling("24h").mean(), linewidth=1.2, label="24 h mean")
    ax.plot(subset.rolling("7D").mean(), linewidth=1.8, label="7 d mean")
    ax.set_title(f"{col} over time", fontsize=13)
    ax.set_ylabel(col)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_temperature_heatmap(
    df: pd.DataFrame,
    target: str = "T (degC)",
    figsize: tuple = (10, 6),
    save_path: Path | None = None,
) -> None:
    """Mean temperature heatmap: hour-of-day × day-of-week."""
    if "hour" not in df.columns or "day_of_week" not in df.columns:
        raise ValueError("Run extract_time_features() first.")

    pivot = df.pivot_table(values=target, index="hour", columns="day_of_week", aggfunc="mean")
    pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(pivot.columns)]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, cmap="RdYlBu_r", annot=False, ax=ax, cbar_kws={"label": target})
    ax.set_title(f"Mean {target}: hour × weekday", fontsize=13)
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Hour of day")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
