"""Render the qualitative source-attribution case used in Section 6.2."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE2_DIR = ROOT / "fcn_main_run_seed42" / "fcn_outputs_stage2"
PAPER_FIGURES = ROOT.parent / "paper" / "figures"
OUTPUT_DIR = ROOT / "experiments" / "figures" / "discussion"

SESSION_ID = "002002_2019"
TARGET_ROW = 5

BLUE = "#0072B2"
BLUE_FILL = "#DCEEF7"
GRAY = "#8A8A8A"
LIGHT_GRAY = "#F2F2F2"
MID_GRAY = "#E2E2E2"
TEXT = "#222222"


def load_case() -> pd.DataFrame:
    scored = pd.read_csv(STAGE2_DIR / "stage2a_test_scored.csv")
    final = pd.read_csv(STAGE2_DIR / "stage2_test_final_graph.csv")
    case = scored.loc[
        (scored["session_id"].astype(str) == SESSION_ID)
        & (scored["Q2_row"] == TARGET_ROW)
    ].copy()
    final_case = final.loc[
        (final["session_id"].astype(str) == SESSION_ID)
        & (final["Q2_row"] == TARGET_ROW),
        ["edge_id", "final_graph_pred_edge"],
    ]
    case = case.merge(final_case, on="edge_id", how="left", validate="one_to_one")
    case = case.sort_values("Q1_row").reset_index(drop=True)

    if len(case) != 5:
        raise ValueError(f"Expected five W=5 candidates, found {len(case)}.")
    if int(case["stage1_pred_edge"].sum()) != 3:
        raise ValueError("The case no longer has three retained candidates.")
    if int(case["final_graph_pred_edge"].sum()) != 1:
        raise ValueError("The case no longer has one final source edge.")
    return case


def draw_box(ax, x, y, width, height, title, detail, fill, edge, weight="normal"):
    box = FancyBboxPatch(
        (x, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.3,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + 0.025, y + 0.052, title, fontsize=8.5, color=TEXT, weight=weight,
            va="center", ha="left", zorder=4)
    ax.text(x + 0.025, y - 0.038, detail, fontsize=7.2, color="#4B4B4B",
            va="center", ha="left", linespacing=1.3, zorder=4)


def arrow(ax, start, end, color, style="solid", width=1.4, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=width,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(patch)


def render(case: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, (ax_left, ax_right) = plt.subplots(2, 1, figsize=(6.8, 6.8),
                                             gridspec_kw={"height_ratios": [1.12, 1.0]})
    fig.subplots_adjust(left=0.055, right=0.985, top=0.97, bottom=0.065, hspace=0.08)

    labels = {
        0: ("Q&A 1  |  $d=5$", "Hydrogen strategy\nand liquid-hydrogen project"),
        1: ("Q&A 2  |  $d=4$", "Mask exports\nand disclosure"),
        2: ("Q&A 3  |  $d=3$", "Shareholder reduction\nvia exchangeable bonds"),
        3: ("Q&A 4  |  $d=2$", "Convertible-bond projects\nand private placement"),
        4: ("Q&A 5  |  $d=1$", "Management share purchases\nand confidence"),
    }
    ys = [0.80, 0.64, 0.48, 0.32, 0.16]

    for ax in (ax_left, ax_right):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax_left.text(0.02, 0.98, "(a) Candidate window", fontsize=10.5, weight="bold", va="top")
    ax_left.text(0.02, 0.925, "Five earlier Q&A turns within $W=5$ for the later shareholder-reduction query",
                 fontsize=7.5, color="#4B4B4B", va="top")
    ax_left.add_patch(FancyBboxPatch((0.02, 0.06), 0.94, 0.79,
                                     boxstyle="round,pad=0.015,rounding_size=0.025",
                                     facecolor="#FBFBFB", edgecolor="#BFBFBF", linewidth=0.8))
    for idx, y in enumerate(ys):
        retained = bool(case.loc[idx, "stage1_pred_edge"])
        selected = bool(case.loc[idx, "final_graph_pred_edge"])
        fill = BLUE_FILL if selected else ("#FFFFFF" if retained else LIGHT_GRAY)
        edge = BLUE if selected else (GRAY if retained else "#BDBDBD")
        title, detail = labels[idx]
        draw_box(ax_left, 0.08, y, 0.58, 0.115, title, detail, fill, edge,
                 weight="bold" if selected else "normal")
        status = "Final source" if selected else ("Retained" if retained else "Removed")
        status_color = BLUE if selected else ("#555555" if retained else "#8A8A8A")
        ax_left.text(0.74, y, status, fontsize=7.8, color=status_color,
                     weight="bold" if selected else "normal", va="center")
    ax_left.text(0.08, 0.045, "Stage 1 retains 3 of 5 window candidates.", fontsize=7.5, color="#4B4B4B")

    ax_right.text(0.02, 0.98, "(b) Source attribution", fontsize=10.5, weight="bold", va="top")
    ax_right.text(0.02, 0.925, "Local evidence and target-wise selection for the retained candidates",
                  fontsize=7.5, color="#4B4B4B", va="top")
    target_x, target_y, target_w, target_h = 0.69, 0.50, 0.26, 0.20
    draw_box(ax_right, target_x, target_y, target_w, target_h,
             "Later query", "Shareholder reduction,\nshare recipients,\nand disclosure", "#FFF7E6", "#B09C85", weight="bold")

    retained_rows = case.loc[case["stage1_pred_edge"] == 1].copy()
    retained_y = [0.76, 0.50, 0.24]
    for (_, row), y in zip(retained_rows.iterrows(), retained_y):
        selected = bool(row["final_graph_pred_edge"])
        source_idx = int(row["Q1_row"])
        title, detail = labels[source_idx]
        evidence = float(row["stage2a_local_prob"])
        evidence_text = f"Local evidence: {evidence:.3f}" if evidence >= 0.001 else "Local evidence: < 0.001"
        fill = BLUE_FILL if selected else "#FFFFFF"
        edge = BLUE if selected else GRAY
        draw_box(ax_right, 0.05, y, 0.51, 0.13, title, f"{detail}\n{evidence_text}", fill, edge,
                 weight="bold" if selected else "normal")
        line_style = "solid" if selected else (0, (2, 2))
        arrow(ax_right, (0.56, y), (target_x, target_y), BLUE if selected else GRAY,
              style=line_style, width=2.1 if selected else 1.0, rad=0.0)

    ax_right.text(0.05, 0.06, "Solid blue: final source dependency edge     Dashed gray: retained but not selected",
                  fontsize=7.0, color="#4B4B4B")
    fig.text(0.5, 0.02,
             "Case from session 002002_2019. The diagram summarizes Test outputs for the target query at row 5.",
             ha="center", va="bottom", fontsize=7.1, color="#555555")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for directory in (OUTPUT_DIR, PAPER_FIGURES):
        fig.savefig(directory / "source_attribution_case.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(directory / "source_attribution_case.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    render(load_case())
