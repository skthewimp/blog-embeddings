#!/usr/bin/env python3
"""Analyze 22 years of blog posts using Gemini embeddings.

Clusters posts by semantic similarity, extracts themes via TF-IDF,
and generates static Tufte-style charts showing how topics have
evolved over time.

Usage:
    python analysis.py                # run full analysis + generate charts
    python analysis.py --clusters 10  # override cluster count
"""

import re
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).parent
POSTS_PARQUET = ROOT / "posts.parquet"
DB_PATH = ROOT / "embeddings.duckdb"
CHARTS_DIR = ROOT / "charts"

# Style constants — matches Bangalore weather chart aesthetic
BG_COLOR = "#eae4db"
PRIMARY = "#490000"       # dark maroon
SECONDARY = "#d4cbaa"     # warm beige
GRAY = "#888888"
TEXT_COLOR = "#333333"
ACCENT_COLORS = [
    "#490000", "#8b4513", "#2e5a3c", "#1a4a6e", "#6b3a6b",
    "#8b6914", "#c25a3c", "#3d6b5a", "#5a3d8b", "#6b8b3d",
    "#8b3d5a", "#3d5a8b", "#5a8b6b", "#8b6b3d", "#3d8b8b",
]


def setup_style():
    """Configure matplotlib to match ggplot/Tufte aesthetic."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": "none",
        "axes.linewidth": 0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": False,
        "axes.labelcolor": TEXT_COLOR,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "text.color": TEXT_COLOR,
        "font.family": "serif",
        "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
        "savefig.facecolor": BG_COLOR,
    })


def load_data() -> pd.DataFrame:
    """Load posts and embeddings, return merged dataframe."""
    posts = pd.read_parquet(POSTS_PARQUET)
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    emb = conn.execute("SELECT slug, embedding FROM embeddings").df()
    conn.close()

    df = posts.merge(emb, on="slug", how="inner")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df


def find_optimal_clusters(X: np.ndarray, k_range: range) -> dict:
    """Evaluate clustering quality across k values."""
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(2000, len(X)))
        results[k] = {"score": score, "labels": labels, "model": km}
        print(f"  k={k:2d}  silhouette={score:.4f}")
    return results


# Extended stopwords for TF-IDF
EXTRA_STOPS = {
    "ve", "don", "ll", "didn", "doesn", "isn", "wasn", "wouldn", "couldn",
    "shouldn", "haven", "hasn", "aren", "won", "hadn", "let", "got",
    "like", "just", "know", "think", "going", "want", "really", "thing",
    "things", "way", "good", "time", "lot", "said", "people", "make",
    "go", "come", "say", "see", "get", "take", "one", "two", "also",
    "much", "well", "even", "would", "could", "since", "back", "went",
    "quite", "rather", "made", "many", "new", "first", "last", "long",
    "right", "still", "day", "days", "year", "years", "today", "week",
    "need", "doing", "stuff", "given", "end", "started", "write",
    "written", "read", "look", "used", "use", "point", "kind",
}
_URL_RE = re.compile(r"https?://\S+")
_HTML_RE = re.compile(r"<[^>]+>")


def _clean_text(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = _HTML_RE.sub("", text)
    return text.lower()


def extract_theme_names(df: pd.DataFrame, labels: np.ndarray, n_clusters: int) -> list[dict]:
    """Extract human-readable theme names from cluster content using TF-IDF."""
    all_stops = list(ENGLISH_STOP_WORDS | EXTRA_STOPS)
    results = []

    for i in range(n_clusters):
        mask = labels == i
        cluster_docs = df.loc[mask, "content"].tolist()

        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=all_stops,
            min_df=3,
            max_df=0.7,
            ngram_range=(1, 2),
            preprocessor=_clean_text,
        )
        tfidf = vectorizer.fit_transform(cluster_docs)
        mean_tfidf = tfidf.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[-8:][::-1]
        features = vectorizer.get_feature_names_out()
        top_terms = [features[j] for j in top_indices]

        # Representative titles (closest to cluster center)
        center = mean_tfidf
        feature_tfidf = tfidf.toarray()
        dists = np.linalg.norm(feature_tfidf - center, axis=1)
        closest = dists.argsort()[:5]
        sample_titles = df.loc[mask].iloc[closest]["title"].tolist()

        results.append({
            "id": i,
            "top_terms": top_terms,
            "sample_titles": sample_titles,
            "count": int(mask.sum()),
        })

    return results


def assign_theme_labels(theme_info: list[dict]) -> list[str]:
    """Map TF-IDF terms to concise, unique human-readable labels.

    Uses a scoring approach: each rule has required terms and anti-terms
    that prevent false matches. More specific rules are checked first.
    """
    # (label, required_terms, anti_terms) — order matters: specific first
    rules = [
        ("Football", {"liverpool", "league", "club"}, set()),
        ("Cricket", {"cricket", "batting"}, {"liverpool"}),
        ("Family & parenting", {"daughter", "baby", "berry", "children"}, set()),
        ("Relationships & marriage", {"wedding", "arranged", "marriage", "relationship"}, set()),
        ("Campus life", {"iimb", "iit", "quiz"}, set()),
        ("Data science & AI", {"code", "ai", "data science", "science"}, set()),
        ("Travel", {"hotel", "trip", "flight"}, set()),
        ("Food & coffee", {"coffee", "food", "restaurant", "eat", "rice"}, {"hotel", "trip", "flight"}),
        ("Urban life & transport", {"bus", "road", "traffic", "buses", "walk"}, set()),
        ("Movies, culture & religion", {"movie", "movies", "kannada", "music", "temple"}, set()),
        ("India, politics & policy", {"government", "india", "states"}, {"wedding", "daughter", "cricket"}),
        ("Markets & economics", {"market", "price", "uber", "stock", "investors", "cost"}, {"wedding", "arranged"}),
        ("Writing, books & life", {"book", "blog", "twitter", "writing", "books", "media"}, {"daughter", "berry"}),
        ("Personal reflections", {"feel", "feeling", "remember", "yesterday", "trying"}, set()),
        ("Work & careers", {"job", "company", "jobs", "career"}, {"iimb", "iit", "code", "ai", "feel", "feeling"}),
    ]

    used_labels = set()
    labels = []

    for t in theme_info:
        top = set(w.lower() for w in t["top_terms"][:8])
        label = None

        for rule_label, pos_terms, anti_terms in rules:
            if (top & pos_terms) and not (top & anti_terms):
                label = rule_label
                break

        if label is None:
            label = ", ".join(t["top_terms"][:3]).title()

        # Disambiguate: if label already used, find a better variant
        if label in used_labels:
            # Try combining with a distinguishing term
            for term in t["top_terms"][:4]:
                term_l = term.lower()
                if term_l not in label.lower() and len(term_l) > 2:
                    candidate = f"{label} ({term})"
                    if candidate not in used_labels:
                        label = candidate
                        break

        used_labels.add(label)
        labels.append(label)

    return labels


def cluster_posts(df: pd.DataFrame, n_clusters: int | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Cluster posts by embedding similarity, return labeled dataframe."""
    X = np.stack(df["embedding"].values)

    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    X_norm = normalize(X_pca)
    print(f"PCA: {X.shape[1]}d → {X_pca.shape[1]}d ({pca.explained_variance_ratio_.sum():.1%} variance)")

    if n_clusters is None:
        print("\nSearching for optimal cluster count...")
        results = find_optimal_clusters(X_norm, range(6, 16))
        best_k = max(results, key=lambda k: results[k]["score"])
        print(f"\nBest k={best_k} (silhouette={results[best_k]['score']:.4f})")
        n_clusters = best_k
        labels = results[best_k]["labels"]
    else:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_norm)
        score = silhouette_score(X_norm, labels, sample_size=min(2000, len(X_norm)))
        print(f"k={n_clusters}  silhouette={score:.4f}")

    df = df.copy()
    df["cluster"] = labels

    print("\nExtracting themes via TF-IDF...")
    theme_info = extract_theme_names(df, labels, n_clusters)
    return df, theme_info


def print_cluster_report(df: pd.DataFrame, theme_info: list[dict]):
    """Print detailed cluster descriptions."""
    print("\n" + "=" * 70)
    print("CLUSTER REPORT")
    print("=" * 70)

    for t in sorted(theme_info, key=lambda x: x["count"], reverse=True):
        cluster_df = df[df["cluster"] == t["id"]]
        year_range = f"{cluster_df['year'].min()}-{cluster_df['year'].max()}"
        peak_year = cluster_df["year"].value_counts().idxmax()

        print(f"\n--- Cluster {t['id']} ({t['count']} posts, {year_range}, peak: {peak_year}) ---")
        print(f"  Top terms: {', '.join(t['top_terms'])}")
        print(f"  Sample titles:")
        for title in t["sample_titles"]:
            print(f"    - {title}")


def temporal_analysis(df: pd.DataFrame, theme_info: list[dict]) -> pd.DataFrame:
    """Compute topic proportions by year."""
    n_clusters = len(theme_info)
    years = sorted(df["year"].unique())

    counts = pd.crosstab(df["year"], df["cluster"])
    counts = counts.reindex(index=years, columns=range(n_clusters), fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0)
    return proportions


# ---------------------------------------------------------------------------
# Charts — all follow the Bangalore weather aesthetic
# ---------------------------------------------------------------------------

def _add_y_axis_line(ax):
    """Add a subtle y-axis line."""
    ax.axvline(ax.get_xlim()[0], color=TEXT_COLOR, linewidth=0.3, zorder=0)


def plot_volume_timeline(df: pd.DataFrame):
    """Vertical segments showing posts per year."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 2.5))

    counts = df.groupby("year").size()

    ax.bar(counts.index, counts.values, color=PRIMARY, alpha=0.8, width=0.7,
           edgecolor="none")

    # Direct labels on notable years
    for year in counts.index:
        val = counts[year]
        ax.text(year, val + 3, str(val), ha="center", va="bottom",
                fontsize=6.5, color=TEXT_COLOR)

    ax.set_xlim(counts.index[0] - 0.8, counts.index[-1] + 0.8)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="y", left=False, labelleft=False)

    fig.text(0.12, 0.92, "Posts per year", fontsize=12, fontweight="bold",
             transform=fig.transFigure)

    fig.savefig(CHARTS_DIR / "posts_per_year.png")
    plt.close(fig)
    print(f"Saved: {CHARTS_DIR / 'posts_per_year.png'}")


def plot_streamgraph(proportions: pd.DataFrame, theme_labels: list[str]):
    """Stacked area showing topic evolution, with direct labels."""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 7))

    years = proportions.index.values
    smoothed = proportions.rolling(3, center=True, min_periods=1).mean()
    data = smoothed.values.T

    # Sort by peak year
    peak_years = [years[np.argmax(row)] for row in data]
    order = np.argsort(peak_years)
    data_sorted = data[order]
    labels_sorted = [theme_labels[i] for i in order]
    colors = [ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in range(len(order))]

    ax.stackplot(years, data_sorted, colors=colors, alpha=0.8,
                 edgecolor=BG_COLOR, linewidth=0.5)

    # Direct annotations instead of legend
    cumulative = np.zeros(len(years))
    for i, (row, label, color) in enumerate(zip(data_sorted, labels_sorted, colors)):
        midpoints = cumulative + row / 2
        cumulative += row

        # Find the year where this topic is most prominent
        peak_idx = np.argmax(row)
        if row[peak_idx] > 0.04:  # only label visible topics
            ax.text(years[peak_idx], midpoints[peak_idx], label,
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    color="white", alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=color,
                              edgecolor="none", alpha=0.7))

    ax.set_xlim(years[0], years[-1])
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axhline(y=0, color=TEXT_COLOR, linewidth=0.3)

    fig.text(0.12, 0.94, "What I wrote about, 2004-2026", fontsize=14,
             fontweight="bold", transform=fig.transFigure)
    fig.text(0.12, 0.91, "Share of posts by topic, 3-year rolling average",
             fontsize=9, color=GRAY, transform=fig.transFigure)

    fig.savefig(CHARTS_DIR / "topic_evolution.png")
    plt.close(fig)
    print(f"Saved: {CHARTS_DIR / 'topic_evolution.png'}")


def plot_small_multiples(proportions: pd.DataFrame, theme_labels: list[str]):
    """Small multiples — one panel per topic showing rise and fall."""
    setup_style()
    n = len(theme_labels)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.0), sharex=True)
    axes = axes.flatten()

    years = proportions.index.values
    smoothed = proportions.rolling(3, center=True, min_periods=1).mean()

    # Sort by peak year
    peak_years = [(np.argmax(smoothed[c].values), c) for c in smoothed.columns]
    order = [c for _, c in sorted(peak_years)]

    for idx, cluster_id in enumerate(order):
        ax = axes[idx]
        y = smoothed[cluster_id].values

        ax.fill_between(years, y, alpha=0.25, color=PRIMARY)
        ax.plot(years, y, color=PRIMARY, linewidth=1.2)

        ax.set_title(theme_labels[cluster_id], fontsize=8.5, fontweight="bold",
                     pad=4, color=TEXT_COLOR)
        ax.set_ylim(0, max(0.30, y.max() * 1.3))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
        ax.tick_params(axis="both", labelsize=6.5)
        ax.tick_params(left=False, bottom=False)

        # Mark peak
        peak_idx = np.argmax(y)
        ax.plot(years[peak_idx], y[peak_idx], "o", color=PRIMARY, markersize=3)
        ax.text(years[peak_idx], y[peak_idx] * 1.08, str(years[peak_idx]),
                fontsize=6.5, ha="center", va="bottom", color=PRIMARY,
                fontweight="bold")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.text(0.08, 0.98, "Each topic's share of writing over time",
             fontsize=13, fontweight="bold", transform=fig.transFigure)
    fig.text(0.08, 0.96, "3-year rolling average of post proportions",
             fontsize=9, color=GRAY, transform=fig.transFigure)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(CHARTS_DIR / "topic_small_multiples.png")
    plt.close(fig)
    print(f"Saved: {CHARTS_DIR / 'topic_small_multiples.png'}")


def plot_era_heatmap(proportions: pd.DataFrame, theme_labels: list[str]):
    """Heatmap showing topic intensity across eras."""
    setup_style()

    eras = {
        "2004-07": (2004, 2007),
        "2008-11": (2008, 2011),
        "2012-15": (2012, 2015),
        "2016-19": (2016, 2019),
        "2020-23": (2020, 2023),
        "2024-26": (2024, 2027),
    }

    era_data = {}
    for era_name, (start, end) in eras.items():
        era_years = [y for y in proportions.index if start <= y < end]
        if era_years:
            era_data[era_name] = proportions.loc[era_years].mean()

    era_df = pd.DataFrame(era_data)

    # Sort by peak era
    peak_era_idx = era_df.values.argmax(axis=1)
    order = np.argsort(peak_era_idx)

    fig, ax = plt.subplots(figsize=(9, max(5, len(theme_labels) * 0.42)))

    data = era_df.iloc[order].values
    ylabels = [theme_labels[i] for i in order]

    # Custom colormap: beige to dark maroon
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blog", [BG_COLOR, SECONDARY, "#8b4513", PRIMARY])

    ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(range(len(eras)))
    ax.set_xticklabels(eras.keys(), fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.tick_params(length=0)
    ax.xaxis.set_ticks_position("top")

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val > data.max() * 0.45 else TEXT_COLOR
            if val >= 0.005:
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=7.5, color=color, fontweight="bold" if val > 0.15 else "normal")

    fig.text(0.13, 0.96, "Topic intensity across eras", fontsize=13,
             fontweight="bold", transform=fig.transFigure)
    fig.text(0.13, 0.93, "Darker cells = higher share of posts in that period",
             fontsize=9, color=GRAY, transform=fig.transFigure)

    fig.savefig(CHARTS_DIR / "era_heatmap.png")
    plt.close(fig)
    print(f"Saved: {CHARTS_DIR / 'era_heatmap.png'}")


def plot_topic_trajectory(df: pd.DataFrame, theme_labels: list[str]):
    """Slope chart showing the rise and fall of major topics across eras."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    eras = [("2004-07", 2004, 2007), ("2008-11", 2008, 2011),
            ("2012-15", 2012, 2015), ("2016-19", 2016, 2019),
            ("2020-23", 2020, 2023), ("2024-26", 2024, 2026)]
    era_names = [e[0] for e in eras]
    x_positions = range(len(eras))

    # Compute proportions per era
    era_props = {}
    for name, start, end in eras:
        era_df = df[(df["year"] >= start) & (df["year"] <= end)]
        props = era_df["cluster"].value_counts(normalize=True)
        era_props[name] = props

    props_df = pd.DataFrame(era_props).fillna(0)

    # Only plot topics that ever exceed 8%
    notable = props_df.index[props_df.max(axis=1) > 0.08]

    for cluster_id in notable:
        y = [props_df.loc[cluster_id, name] if cluster_id in props_df.index else 0
             for name in era_names]
        color = ACCENT_COLORS[int(cluster_id) % len(ACCENT_COLORS)]

        ax.plot(x_positions, y, "o-", color=color, linewidth=2, markersize=5, alpha=0.8)

        # Label at the end where it's most prominent
        peak_idx = np.argmax(y)
        label = theme_labels[cluster_id]
        offset = 0.008
        ax.text(peak_idx + 0.15, y[peak_idx] + offset, label,
                fontsize=7.5, color=color, fontweight="bold", va="bottom")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(era_names, fontsize=10, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.set_ylim(0, props_df.max().max() * 1.2)
    ax.axhline(y=0, color=TEXT_COLOR, linewidth=0.3)

    # Subtle vertical lines at each era
    for x in x_positions:
        ax.axvline(x, color=SECONDARY, linewidth=0.5, zorder=0)

    fig.text(0.12, 0.94, "Rise and fall of major topics", fontsize=14,
             fontweight="bold", transform=fig.transFigure)
    fig.text(0.12, 0.91, "Topics exceeding 8% of posts in any era",
             fontsize=9, color=GRAY, transform=fig.transFigure)

    fig.savefig(CHARTS_DIR / "topic_trajectory.png")
    plt.close(fig)
    print(f"Saved: {CHARTS_DIR / 'topic_trajectory.png'}")


def main(clusters: int | None = None):
    """Run full analysis pipeline."""
    CHARTS_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} posts with embeddings, {df['year'].min()}-{df['year'].max()}")

    df, theme_info = cluster_posts(df, n_clusters=clusters)
    print_cluster_report(df, theme_info)

    theme_labels = assign_theme_labels(theme_info)

    # Print label mapping
    print("\n" + "=" * 70)
    print("THEME LABELS")
    print("=" * 70)
    for t, label in zip(theme_info, theme_labels):
        print(f"  Cluster {t['id']:2d} ({t['count']:3d} posts): {label}")

    proportions = temporal_analysis(df, theme_info)

    # Era summaries
    print("\n" + "=" * 70)
    print("ERA ANALYSIS")
    print("=" * 70)
    eras = [
        ("Early blogging (2004-07)", 2004, 2007),
        ("Post-MBA / early career (2008-11)", 2008, 2011),
        ("Analytical peak (2012-15)", 2012, 2015),
        ("Mid-career (2016-19)", 2016, 2019),
        ("Pandemic + startup (2020-23)", 2020, 2023),
        ("Recent (2024-26)", 2024, 2026),
    ]
    for era_name, start, end in eras:
        era_df = df[(df["year"] >= start) & (df["year"] <= end)]
        if len(era_df) == 0:
            continue
        era_props = era_df["cluster"].value_counts(normalize=True).head(3)
        print(f"\n{era_name} ({len(era_df)} posts):")
        for cluster_id, prop in era_props.items():
            print(f"  {prop:5.1%}  {theme_labels[cluster_id]}")

    # Generate charts
    print("\n" + "=" * 70)
    print("GENERATING CHARTS")
    print("=" * 70)
    plot_volume_timeline(df)
    plot_streamgraph(proportions, theme_labels)
    plot_small_multiples(proportions, theme_labels)
    plot_era_heatmap(proportions, theme_labels)
    plot_topic_trajectory(df, theme_labels)

    # Save clustered data
    output = df[["slug", "title", "date", "year", "source", "cluster"]].copy()
    output.to_parquet(ROOT / "posts_clustered.parquet", index=False)
    print(f"\nSaved clustered posts → {ROOT / 'posts_clustered.parquet'}")


if __name__ == "__main__":
    import typer
    typer.run(main)
