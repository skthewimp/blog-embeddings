#!/usr/bin/env python3
"""Ternary plot animation showing how blogging shifted across 3 macro themes.

Groups 15 embedding clusters into 3 broad themes:
  - Personal: campus life, reflections, relationships, family, culture
  - Analytical: markets, politics, data science, work
  - Lifestyle: food, travel, urban life, cricket, football, writing

Generates a GIF showing the trajectory year by year on an equilateral triangle.

Usage:
    python triangle.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio.v2 as imageio

import analysis

ROOT = Path(__file__).parent
CHARTS_DIR = ROOT / "charts"

# Style
BG_COLOR = "#eae4db"
PRIMARY = "#490000"
SECONDARY = "#d4cbaa"
GRAY = "#888888"
TEXT_COLOR = "#333333"
TRAIL_COLOR = "#8b4513"

# Macro theme groupings (cluster labels → macro theme)
MACRO_THEMES = {
    "Personal": [
        "Campus life", "Personal reflections", "Relationships & marriage",
        "Family & parenting", "Movies, culture & religion",
    ],
    "Analytical": [
        "Markets & economics", "India, politics & policy",
        "Data science & AI", "Work & careers",
    ],
    "Lifestyle": [
        "Food & coffee", "Travel", "Urban life & transport",
        "Cricket", "Football", "Writing, books & life",
    ],
}


def ternary_to_cartesian(p: np.ndarray) -> tuple[float, float]:
    """Convert ternary coordinates [a, b, c] to 2D cartesian for plotting.

    a = Personal (top), b = Analytical (bottom-left), c = Lifestyle (bottom-right)
    """
    a, b, c = p / p.sum()
    x = 0.5 * (2 * c + a)
    y = (np.sqrt(3) / 2) * a
    return x, y


def draw_triangle(ax):
    """Draw the equilateral triangle frame."""
    # Vertices
    top = ternary_to_cartesian(np.array([1, 0, 0]))
    bl = ternary_to_cartesian(np.array([0, 1, 0]))
    br = ternary_to_cartesian(np.array([0, 0, 1]))

    triangle = plt.Polygon([top, bl, br], fill=False, edgecolor=GRAY,
                           linewidth=1, zorder=1)
    ax.add_patch(triangle)

    # Labels at vertices
    pad = 0.04
    ax.text(top[0], top[1] + pad, "Personal", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=TEXT_COLOR)
    ax.text(bl[0] - pad, bl[1] - pad, "Analytical", ha="center", va="top",
            fontsize=11, fontweight="bold", color=TEXT_COLOR)
    ax.text(br[0] + pad, br[1] - pad, "Lifestyle", ha="center", va="top",
            fontsize=11, fontweight="bold", color=TEXT_COLOR)

    # Subtle descriptions
    ax.text(top[0], top[1] + pad - 0.03, "campus, relationships,\nfamily, culture",
            ha="center", va="top", fontsize=7, color=GRAY, style="italic")
    ax.text(bl[0] - pad, bl[1] - pad - 0.03, "markets, policy,\ndata science, work",
            ha="center", va="top", fontsize=7, color=GRAY, style="italic")
    ax.text(br[0] + pad, br[1] - pad - 0.03, "food, travel, sport,\nurban life, books",
            ha="center", va="top", fontsize=7, color=GRAY, style="italic")

    # Grid lines (ternary grid at 25%, 50%, 75%)
    for frac in [0.25, 0.5, 0.75]:
        for i in range(3):
            p1 = np.zeros(3)
            p2 = np.zeros(3)
            p1[i] = frac
            p1[(i + 1) % 3] = 1 - frac
            p2[i] = frac
            p2[(i + 2) % 3] = 1 - frac
            x1, y1 = ternary_to_cartesian(p1)
            x2, y2 = ternary_to_cartesian(p2)
            ax.plot([x1, x2], [y1, y2], color=SECONDARY, linewidth=0.4, zorder=0)


def compute_yearly_coords(df: pd.DataFrame, theme_labels: list[str]) -> pd.DataFrame:
    """Compute ternary coordinates for each year."""
    # Map cluster labels to macro themes
    cluster_to_macro = {}
    for macro, labels in MACRO_THEMES.items():
        for label in labels:
            cluster_to_macro[label] = macro

    # Assign macro theme to each post
    df = df.copy()
    df["macro"] = df["cluster"].map(lambda c: cluster_to_macro.get(theme_labels[c], "Lifestyle"))

    years = sorted(df["year"].unique())
    rows = []
    for year in years:
        year_df = df[df["year"] == year]
        counts = year_df["macro"].value_counts()
        total = len(year_df)
        personal = counts.get("Personal", 0) / total
        analytical = counts.get("Analytical", 0) / total
        lifestyle = counts.get("Lifestyle", 0) / total
        rows.append({
            "year": year,
            "personal": personal,
            "analytical": analytical,
            "lifestyle": lifestyle,
            "n_posts": total,
        })

    return pd.DataFrame(rows)


def make_frame(coords: pd.DataFrame, up_to_year: int, frame_path: Path):
    """Render one frame of the animation."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "font.family": "serif",
        "font.size": 9,
        "text.color": TEXT_COLOR,
        "savefig.facecolor": BG_COLOR,
    })

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_triangle(ax)

    # Title
    fig.text(0.5, 0.95, "22 years of blogging", ha="center",
             fontsize=16, fontweight="bold", color=TEXT_COLOR,
             transform=fig.transFigure)

    # Year label - large
    fig.text(0.5, 0.90, str(up_to_year), ha="center",
             fontsize=28, fontweight="bold", color=PRIMARY, alpha=0.7,
             transform=fig.transFigure)

    visible = coords[coords["year"] <= up_to_year]

    if len(visible) < 1:
        fig.savefig(frame_path, dpi=120, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        return

    # Compute cartesian coordinates
    xs, ys = [], []
    for _, row in visible.iterrows():
        p = np.array([row["personal"], row["analytical"], row["lifestyle"]])
        x, y = ternary_to_cartesian(p)
        xs.append(x)
        ys.append(y)

    # Trail (fading line)
    if len(xs) > 1:
        for i in range(1, len(xs)):
            alpha = 0.15 + 0.6 * (i / len(xs))
            lw = 0.8 + 1.5 * (i / len(xs))
            ax.plot([xs[i-1], xs[i]], [ys[i-1], ys[i]],
                    color=TRAIL_COLOR, linewidth=lw, alpha=alpha, zorder=2)

    # Past points (fading dots)
    for i in range(len(xs) - 1):
        alpha = 0.1 + 0.4 * (i / len(xs))
        size = 15 + 20 * (i / len(xs))
        ax.scatter(xs[i], ys[i], s=size, color=TRAIL_COLOR, alpha=alpha,
                   edgecolors="none", zorder=3)

    # Current point (prominent)
    ax.scatter(xs[-1], ys[-1], s=120, color=PRIMARY, edgecolors="white",
               linewidth=1.5, zorder=5)

    # Label current year's percentages
    row = visible.iloc[-1]
    label = f"P:{row['personal']:.0%}  A:{row['analytical']:.0%}  L:{row['lifestyle']:.0%}"
    ax.text(xs[-1], ys[-1] - 0.04, label, ha="center", va="top",
            fontsize=7, color=PRIMARY, fontweight="bold")

    # Start and end markers
    if up_to_year >= 2005:
        ax.text(xs[0], ys[0] + 0.03, "2004", ha="center", va="bottom",
                fontsize=7, color=GRAY, fontweight="bold")

    fig.savefig(frame_path, dpi=120, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def make_static_summary(coords: pd.DataFrame, out_path: Path):
    """Static chart with all years plotted and labeled on the triangle."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "font.family": "serif",
        "font.size": 9,
        "text.color": TEXT_COLOR,
        "savefig.facecolor": BG_COLOR,
    })

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.22, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_triangle(ax)

    # Title and subtitle
    fig.text(0.5, 0.96, "22 years of blogging, in three dimensions",
             ha="center", fontsize=15, fontweight="bold", color=TEXT_COLOR,
             transform=fig.transFigure)
    fig.text(0.5, 0.93,
             "Each dot is one year. 3,010 posts clustered by semantic embeddings into Personal, Analytical, and Lifestyle.",
             ha="center", fontsize=8, color=GRAY, transform=fig.transFigure)

    # Compute all coordinates
    xs, ys, years_list = [], [], []
    for _, row in coords.iterrows():
        p = np.array([row["personal"], row["analytical"], row["lifestyle"]])
        x, y = ternary_to_cartesian(p)
        xs.append(x)
        ys.append(y)
        years_list.append(int(row["year"]))

    # Trail line
    for i in range(1, len(xs)):
        progress = i / len(xs)
        alpha = 0.2 + 0.6 * progress
        lw = 0.8 + 1.2 * progress
        ax.plot([xs[i-1], xs[i]], [ys[i-1], ys[i]],
                color=TRAIL_COLOR, linewidth=lw, alpha=alpha, zorder=2)

    # Dots sized by post count
    sizes = coords["n_posts"].values
    size_scaled = 30 + (sizes / sizes.max()) * 150

    # Color gradient: early years lighter, recent darker
    n = len(xs)
    for i in range(n):
        progress = i / (n - 1)
        alpha = 0.35 + 0.65 * progress
        ax.scatter(xs[i], ys[i], s=size_scaled[i], color=PRIMARY, alpha=alpha,
                   edgecolors="white", linewidth=0.8, zorder=4)

    # Label every year — offset to avoid overlaps
    # Use a simple nudge strategy
    for i in range(n):
        year = years_list[i]
        x, y = xs[i], ys[i]

        # Default offset
        ha, va = "left", "bottom"
        dx, dy = 0.015, 0.015

        # Nudge specific years to reduce overlap
        if year in (2005, 2009, 2012):
            ha, va, dx, dy = "right", "bottom", -0.015, 0.015
        elif year in (2010, 2020):
            ha, va, dx, dy = "left", "top", 0.015, -0.015
        elif year in (2008, 2019):
            ha, va, dx, dy = "right", "top", -0.015, -0.015
        elif year in (2004,):
            ha, va, dx, dy = "center", "bottom", 0, 0.02

        fontsize = 8 if year % 5 == 0 or year == 2004 or year == 2026 else 6.5
        weight = "bold" if year % 5 == 0 or year == 2004 or year == 2026 else "normal"

        ax.text(x + dx, y + dy, str(year), ha=ha, va=va,
                fontsize=fontsize, fontweight=weight, color=TEXT_COLOR, zorder=5)

    # Annotations for key moments
    annotations = [
        (2004, "Started blogging\nat IIT Madras", "right", 0.06, 0.02),
        (2013, "Mint columns,\npolicy writing", "left", -0.08, 0.03),
        (2026, "Data science\ntakes over", "right", 0.04, 0.02),
    ]
    for year, text, ha_ann, offx, offy in annotations:
        idx = years_list.index(year)
        ax.annotate(
            text, xy=(xs[idx], ys[idx]),
            xytext=(xs[idx] + offx, ys[idx] + offy),
            fontsize=7.5, color=PRIMARY, fontweight="bold", fontstyle="italic",
            ha=ha_ann,
            arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.5),
            zorder=6,
        )

    # Legend for dot size
    fig.text(0.82, 0.08, "Dot size = posts that year", ha="center",
             fontsize=7, color=GRAY, transform=fig.transFigure)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main():
    """Generate the triangle GIF."""
    CHARTS_DIR.mkdir(exist_ok=True)
    frames_dir = CHARTS_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)

    print("Loading data and clustering...")
    df = analysis.load_data()
    df, theme_info = analysis.cluster_posts(df, n_clusters=15)
    theme_labels = analysis.assign_theme_labels(theme_info)

    print("Computing ternary coordinates...")
    coords = compute_yearly_coords(df, theme_labels)
    print(coords.to_string(index=False))

    print("\nRendering frames...")
    frame_paths = []
    years = sorted(coords["year"].unique())

    for year in years:
        frame_path = frames_dir / f"frame_{year}.png"
        make_frame(coords, year, frame_path)
        frame_paths.append(frame_path)
        print(f"  {year}")

    # Hold last frame longer
    for _ in range(20):
        frame_paths.append(frame_paths[-1])

    print("Assembling GIF...")
    frames = [imageio.imread(str(p)) for p in frame_paths]
    gif_path = CHARTS_DIR / "blog_triangle.gif"
    imageio.mimsave(str(gif_path), frames, duration=1.2, loop=0)
    print(f"Saved: {gif_path}")

    # Static summary with all years labeled
    final_path = CHARTS_DIR / "blog_triangle_final.png"
    make_static_summary(coords, final_path)
    print(f"Saved: {final_path}")


if __name__ == "__main__":
    main()
