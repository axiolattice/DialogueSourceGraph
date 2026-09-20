#!/usr/bin/env python3
"""Create publication figures for RQ3 reliability evidence.

The script reads the fixed Validation sensitivity and Test error-analysis
outputs. It does not fit a model, select parameters, or modify experiment
results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SENSITIVITY_DIR = ROOT / "robustness" / "fcn_parameter_sensitivity"
DEFAULT_ERROR_DIR = ROOT / "analysis" / "fcn_error_analysis"
DEFAULT_OUTDIR = ROOT / "figures" / "rq3_reliability"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
DARK = "#222222"
MID_GRAY = "#666666"
LIGHT_GRAY = "#F2F2F2"
FN_FILL = "#EAF2F8"
FP_FILL = "#FFF3E5"
VALIDATION_F1_BLUE = LinearSegmentedColormap.from_list(
    "validation_f1_blue",
    ["#F2F7FB", "#C9E0F2", "#77B6DD", "#0072B2", "#003B73"],
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required result file is missing: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_validation_sensitivity(
    topk: pd.DataFrame,
    grid: pd.DataFrame,
    outdir: Path,
) -> None:
    """Create a two-panel Validation sensitivity figure."""
    expected_topk = {"TopK", "Feasible", "Recall", "RetainedEdges"}
    expected_grid = {"lambda_0", "lambda_t", "F1", "selected_row"}
    if not expected_topk.issubset(topk.columns):
        raise ValueError("Stage 1 sensitivity file has unexpected columns.")
    if not expected_grid.issubset(grid.columns):
        raise ValueError("Stage 2 sensitivity file has unexpected columns.")

    topk = topk.sort_values("TopK").copy()
    total_candidates = 8_379
    topk["Compression"] = 1.0 - topk["RetainedEdges"] / total_candidates
    selected_topk = topk.loc[topk["is_selected"].astype(bool)]
    if len(selected_topk) != 1:
        raise ValueError("Expected exactly one selected Stage 1 configuration.")
    selected_grid = grid.loc[grid["selected_row"].astype(bool)]
    if len(selected_grid) != 1:
        raise ValueError("Expected exactly one selected Stage 2 configuration.")

    fig, (ax_left, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(10.4, 3.55),
        gridspec_kw={"width_ratios": [1.0, 1.22]},
        constrained_layout=True,
    )

    x = topk["TopK"].to_numpy(float)
    recall = topk["Recall"].to_numpy(float)
    compression = topk["Compression"].to_numpy(float)
    feasible = topk["Feasible"].astype(bool).to_numpy()

    ax_left.axhspan(0.98, 1.005, color="#E8F4EA", zorder=0)
    ax_left.axhline(0.98, color=GREEN, linestyle="--", linewidth=1.15, zorder=8)
    ax_left.plot(x, recall, color=BLUE, linewidth=1.8, zorder=9)
    ax_left.scatter(
        x[~feasible], recall[~feasible], s=38, facecolors="white",
        edgecolors=BLUE, linewidths=1.4, zorder=10,
    )
    ax_left.scatter(
        x[feasible], recall[feasible], s=54, color=ORANGE,
        edgecolors="white", linewidths=0.8, zorder=11,
    )
    ax_left.set_xlim(0.7, 5.3)
    ax_left.set_ylim(0.55, 1.01)
    ax_left.set_xticks(x)
    ax_left.set_xlabel("Top-$K$ candidate cap")
    ax_left.set_ylabel("Validation recall", color=BLUE)
    ax_left.tick_params(axis="y", colors=BLUE)
    ax_left.set_title("(a) Candidate retention", loc="left", fontweight="bold")
    ax_left.text(
        1.02,
        0.982,
        r"Recall requirement $\rho=0.98$",
        color=GREEN,
        fontsize=7.6,
        va="bottom",
    )

    ax_right = ax_left.twinx()
    # Keep compression bars in the background of the recall curve.
    ax_right.set_zorder(ax_left.get_zorder() - 1)
    ax_right.patch.set_visible(False)
    ax_right.bar(
        x,
        compression * 100.0,
        width=0.52,
        color="#C9DDEB",
        edgecolor="white",
        alpha=0.62,
        zorder=1,
    )
    ax_right.set_ylim(0, 100)
    ax_right.set_ylabel("Candidates removed (%)", color=MID_GRAY)
    ax_right.tick_params(axis="y", colors=MID_GRAY)
    ax_right.spines["top"].set_visible(False)
    ax_left.set_zorder(ax_right.get_zorder() + 1)
    ax_left.patch.set_visible(False)
    
    selected_k = int(selected_topk["TopK"].iloc[0])
    ax_left.annotate(
        f"Selected $K={selected_k}$",
        xy=(selected_k, float(selected_topk["Recall"].iloc[0])),
        xytext=(3.36, 0.69),
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1.0},
        color=ORANGE,
        fontsize=8,
        ha="left",
    )

    lambda0 = np.sort(grid["lambda_0"].unique())
    lambdat = np.sort(grid["lambda_t"].unique())
    matrix = (
        grid.pivot(index="lambda_t", columns="lambda_0", values="F1")
        .reindex(index=lambdat, columns=lambda0)
        .to_numpy(float)
    )
    image = ax_heat.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap=VALIDATION_F1_BLUE,
        interpolation="nearest",
    )
    x_ticks = np.arange(0, len(lambda0), 4)
    y_ticks = np.arange(0, len(lambdat), 2)
    ax_heat.set_xticks(x_ticks, [f"{lambda0[idx]:.1f}" for idx in x_ticks])
    ax_heat.set_yticks(y_ticks, [f"{lambdat[idx]:.1f}" for idx in y_ticks])
    ax_heat.set_xlabel(r"$\lambda_0$")
    ax_heat.set_ylabel(r"$\lambda_t$")
    ax_heat.set_title("(b) Structured recovery", loc="left", fontweight="bold")

    selected = selected_grid.iloc[0]
    selected_x = int(np.where(np.isclose(lambda0, selected["lambda_0"]))[0][0])
    selected_y = int(np.where(np.isclose(lambdat, selected["lambda_t"]))[0][0])
    tied_maximum_settings = [(0.20, 0.20), (0.25, 0.10), (0.25, 0.15), (0.25, 0.20)]
    for tied_lambda_0, tied_lambda_t in tied_maximum_settings:
        tied_x = int(np.where(np.isclose(lambda0, tied_lambda_0))[0][0])
        tied_y = int(np.where(np.isclose(lambdat, tied_lambda_t))[0][0])
        ax_heat.plot(
            tied_x,
            tied_y,
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=BLUE,
            markeredgewidth=0.8,
            linestyle="None",
            zorder=4,
        )
    ax_heat.add_patch(
        Rectangle(
            (selected_x - 0.46, selected_y - 0.46),
            0.92,
            0.92,
            fill=False,
            edgecolor=ORANGE,
            linewidth=1.7,
        )
    )
    selected_lambda_0 = 0.20
    selected_lambda_t = 0.15
    selected_f1 = 0.3526
    ax_heat.annotate(
        rf"Selected $({selected_lambda_0:.2f}, {selected_lambda_t:.2f})$"
        "\n"
        rf"maximum Validation F1 = {selected_f1:.4f}",
        xy=(selected_x, selected_y),
        xytext=(selected_x + 1.85, selected_y + 2.15),
        color=DARK,
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
        arrowprops={"arrowstyle": "-", "color": DARK, "lw": 0.9},
        zorder=6,
    )
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.045, pad=0.03)
    colorbar.set_ticks([])
    colorbar.set_label("Validation F1\n(darker blue indicates higher F1)")
    ax_heat.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="white",
                markeredgecolor=BLUE, markersize=5, label="other maximum-F1 settings",
            ),
            Rectangle((0, 0), 1, 1, fill=False, edgecolor=ORANGE, linewidth=1.5, label="selected setting"),
        ],
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#BDBDBD",
        framealpha=0.92,
        fontsize=6.8,
        borderpad=0.35,
        handlelength=1.1,
    )

    save_figure(fig, outdir / "rq3_validation_sensitivity")


def plot_error_boundaries(
    fn_attribution: pd.DataFrame,
    fp_taxonomy: pd.DataFrame,
    outdir: Path,
) -> None:
    """Create a compact publication table for error attribution."""
    fn_counts = dict(zip(fn_attribution["metric"], fn_attribution["count"]))
    fp_counts = dict(zip(fp_taxonomy["category"], fp_taxonomy["count"]))
    required_fn = {"stage1_recall_miss", "stage2_graph_pruning_drop", "total_fn"}
    required_fp = {
        "long_distance_carryover",
        "vague_answer_false_trigger",
        "multiple_source_over_recovery",
        "boundary_question_form",
        "other_manual_review",
    }
    if not required_fn.issubset(fn_counts) or not required_fp.issubset(fp_counts):
        raise ValueError("Error-analysis files do not contain the expected categories.")

    total_fn = int(fn_counts["total_fn"])
    total_fp = int(sum(fp_counts.values()))
    rows = [
        ("False negative", "Not retained by Stage 1", int(fn_counts["stage1_recall_miss"]), total_fn, FN_FILL),
        ("", "Not selected by Stage 2B", int(fn_counts["stage2_graph_pruning_drop"]), total_fn, FN_FILL),
        ("False positive", "Carryover across distant turns", int(fp_counts["long_distance_carryover"]), total_fp, FP_FILL),
        ("", "False triggers from vague answers", int(fp_counts["vague_answer_false_trigger"]), total_fp, FP_FILL),
        ("", "Excessive recovery of multiple sources", int(fp_counts["multiple_source_over_recovery"]), total_fp, FP_FILL),
        ("", "Questions at the annotation boundary", int(fp_counts["boundary_question_form"]), total_fp, FP_FILL),
        ("", "Other observed patterns", int(fp_counts["other_manual_review"]), total_fp, FP_FILL),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 3.35))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    left, right = 0.045, 0.955
    bottom, top = 0.14, 0.89
    widths = np.array([0.24, 0.50, 0.12, 0.14])
    x_edges = left + (right - left) * np.concatenate(([0.0], np.cumsum(widths)))
    row_height = (top - bottom) / (len(rows) + 1)
    headers = ["Error outcome", "Stage or interaction pattern", "Count", "Share"]

    ax.add_patch(Rectangle((left, top - row_height), right - left, row_height, facecolor=DARK, edgecolor=DARK))
    for index, header in enumerate(headers):
        x_center = (x_edges[index] + x_edges[index + 1]) / 2
        ax.text(x_center, top - row_height / 2, header, ha="center", va="center", color="white", fontweight="bold", fontsize=8.7)

    for row_index, (outcome, pattern, count, denominator, fill) in enumerate(rows):
        y_top = top - (row_index + 1) * row_height
        y_bottom = y_top - row_height
        ax.add_patch(Rectangle((left, y_bottom), right - left, row_height, facecolor=fill, edgecolor="white", linewidth=0.8))
        values = [outcome, pattern, f"{count}", f"{100.0 * count / denominator:.1f}%"]
        alignments = ["left", "left", "center", "center"]
        for index, (value, alignment) in enumerate(zip(values, alignments)):
            x_position = x_edges[index] + 0.012 if alignment == "left" else (x_edges[index] + x_edges[index + 1]) / 2
            ax.text(x_position, (y_top + y_bottom) / 2, value, ha=alignment, va="center", color=DARK, fontsize=8.5)

    for x_value in x_edges:
        ax.plot([x_value, x_value], [bottom, top], color="white", linewidth=0.8)
    ax.text(
        left,
        0.055,
        "Shares are calculated within false negatives (n=265) or false positives (n=162). "
        "False-positive categories are assigned through rule-assisted classification.",
        ha="left",
        va="center",
        color=MID_GRAY,
        fontsize=7.5,
    )
    save_figure(fig, outdir / "rq3_error_boundaries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity-dir", type=Path, default=DEFAULT_SENSITIVITY_DIR)
    parser.add_argument("--error-dir", type=Path, default=DEFAULT_ERROR_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Generate only the Validation-sensitivity figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    topk = read_csv(args.sensitivity_dir / "stage1_topk_sensitivity_summary.csv")
    grid = read_csv(args.sensitivity_dir / "stage2_validation_grid_enriched.csv")
    plot_validation_sensitivity(topk, grid, args.outdir)
    if not args.validation_only:
        fn_attribution = read_csv(args.error_dir / "error_fn_stage_attribution.csv")
        fp_taxonomy = read_csv(args.error_dir / "error_fp_taxonomy.csv")
        plot_error_boundaries(fn_attribution, fp_taxonomy, args.outdir)
    print(f"Wrote figures to: {args.outdir}")


if __name__ == "__main__":
    main()
