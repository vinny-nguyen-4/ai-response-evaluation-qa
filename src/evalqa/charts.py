"""Report charts (PNG) for the README and the Excel dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ = ["#e8f1fc", "#b9d4f5", "#86b6ef", "#4f93e3", "#2a78d6", "#1d5ca8", "#123f78"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False,
})
DEFECT_LABELS = {"factual_error": "Factual error", "hallucinated_source": "Made-up source",
                 "missing_requirement": "Missing requirement", "unsafe": "Unsafe advice", "padding": "Off-topic padding"}
DIM_LABELS = {"accuracy": "Accuracy", "instruction_following": "Instruction following", "relevance": "Relevance",
              "safety": "Safety", "clarity": "Clarity"}


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def reviewer_scorecard(sc: pd.DataFrame, min_detection: float, max_bias: float, out: Path) -> Path:
    s = sc.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].barh(s["rater_id"], s["detection_rate"], color=SERIES[0], height=0.6)
    axes[0].axvline(min_detection, color=INK2, lw=1, ls="--")
    axes[0].annotate(f"target {min_detection:.0%}", (min_detection, len(s) - 0.4), xytext=(4, 0),
                     textcoords="offset points", fontsize=8, color=INK2)
    axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    axes[0].set_xlim(0, 1.15)
    axes[0].set_title("Planted defects caught")
    axes[1].barh(s["rater_id"], s["bias"], color=SERIES[1], height=0.6)
    for x in (-max_bias, max_bias):
        axes[1].axvline(x, color=INK2, lw=1, ls="--")
    axes[1].axvline(0, color=INK2, lw=1)
    axes[1].set_xlim(-0.8, 0.8)
    axes[1].set_title("Scoring bias vs gold (points)")
    for i, (_, r) in enumerate(s.iterrows()):
        axes[0].annotate(f"{r['detection_rate']:.0%}", (r["detection_rate"], i), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=9)
        if r["status"] == "Recalibrate":
            axes[1].annotate("Recalibrate", (0.78, i), ha="right", va="center", fontsize=8.5, color=INK, fontweight="bold")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    return _save(fig, out)


def detection_heatmap(sc: pd.DataFrame, out: Path) -> Path:
    cols = [f"caught_{d}" for d in DEFECT_LABELS]
    m = sc.set_index("rater_id")[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), list(DEFECT_LABELS.values()))
    ax.set_yticks(range(len(sc)), sc["rater_id"])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if (not np.isnan(v) and v > 0.6) else INK)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Share of each planted defect type caught, by reviewer")
    return _save(fig, out)


def agreement_chart(agr: pd.DataFrame, out: Path) -> Path:
    a = agr.sort_values("krippendorff_alpha")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.barh([DIM_LABELS[d] for d in a["dimension"]], a["krippendorff_alpha"], color=SERIES[0], height=0.6)
    ax.axvline(0.667, color=INK2, lw=1, ls="--")
    for i, v in enumerate(a["krippendorff_alpha"]):
        ax.annotate(f"{v:.2f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9,
                    bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_title("Agreement between reviewers by dimension (Krippendorff's alpha; dashed line = 0.667 minimum)")
    return _save(fig, out)


def root_cause_chart(rc: pd.DataFrame, out: Path) -> Path:
    r = rc.sort_values("severity_points")
    labels = list(r["root_cause"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.barh(labels, r["severity_points"], color=SERIES[1], height=0.6)
    for i, (_, row) in enumerate(r.iterrows()):
        extra = f" ({row['critical']} critical)" if row["critical"] else ""
        ax.annotate(f"{row['findings']} finding{'s' if row['findings'] != 1 else ''}{extra}", (row["severity_points"], i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5)
    ax.set_xlim(0, r["severity_points"].max() * 1.45)
    ax.xaxis.set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_title("Root causes of QA findings, weighted by severity (critical 3, major 2, minor 1)")
    return _save(fig, out)
