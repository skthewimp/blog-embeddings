#!/usr/bin/env python3
"""Generate blogmap/blogmap.json for the UMAP visualization.

Reads posts.parquet + embeddings.parquet, runs PCA → UMAP → KMeans,
and exports a compact JSON file consumed by blogmap/index.html.

Usage:
    python blogmap.py              # generate blogmap.json
    python blogmap.py --refit      # force refit all models
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).parent
POSTS_PARQUET = ROOT / "posts.parquet"
EMBEDDINGS_PARQUET = ROOT / "embeddings.parquet"
BLOGMAP_DIR = ROOT / "blogmap"
JSON_OUT = BLOGMAP_DIR / "blogmap.json"
MODELS_DIR = ROOT / "models"

# Hyperparameters
N_CLUSTERS = 12
N_PCA = 50
UMAP_NEIGHBORS = 35
UMAP_MIN_DIST = 0.18


def reorder_labels(raw: np.ndarray) -> np.ndarray:
    """Relabel clusters by descending size."""
    counts = pd.Series(raw).value_counts().sort_values(ascending=False)
    mapping = {int(old): new for new, old in enumerate(counts.index)}
    return np.array([mapping[k] for k in raw], dtype=np.int32)


def top_terms_by_cluster(texts: pd.Series, clusters: np.ndarray, n_terms: int = 3) -> dict[int, list[str]]:
    """Find distinctive terms for each cluster using TF-IDF."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=3,
        max_df=0.30,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())
    global_mean = np.asarray(matrix.mean(axis=0)).ravel() + 1e-9

    cluster_terms: dict[int, list[str]] = {}
    for cluster_id in sorted(set(clusters)):
        mask = clusters == cluster_id
        cluster_mean = np.asarray(matrix[mask].mean(axis=0)).ravel()
        score = cluster_mean / global_mean
        order = np.argsort(score)[::-1]
        chosen = []
        for idx in order:
            term = terms[idx]
            if len(term) < 3:
                continue
            if any(tok.isdigit() for tok in term.split()):
                continue
            chosen.append(term)
            if len(chosen) == n_terms:
                break
        cluster_terms[int(cluster_id)] = chosen
    return cluster_terms


def main(refit: bool = False) -> None:
    BLOGMAP_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading posts and embeddings...")
    posts = pd.read_parquet(POSTS_PARQUET)
    emb = pd.read_parquet(EMBEDDINGS_PARQUET)

    # Join on slug
    frame = posts.merge(emb, on="slug", how="inner")
    print(f"  {len(frame)} posts with embeddings")

    if len(frame) == 0:
        print("No posts with embeddings found. Run embeddings.py first.")
        return

    # Build vectors
    vectors = normalize(np.vstack(frame["embedding"].tolist()).astype(np.float32))

    # PCA
    print("Fitting PCA...")
    n_pca = min(N_PCA, len(frame) - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    pca_feat = pca.fit_transform(vectors)

    # UMAP (lazy import - slow due to numba JIT)
    print("Fitting UMAP (this takes a minute)...")
    import umap as umap_mod
    reducer = umap_mod.UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    umap_coords = reducer.fit_transform(vectors)

    # KMeans
    print(f"Fitting KMeans ({N_CLUSTERS} clusters)...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
    raw_labels = km.fit_predict(pca_feat[:, :20])
    labels = reorder_labels(raw_labels)

    # Get cluster names from TF-IDF
    cluster_terms = top_terms_by_cluster(frame["content"], labels)
    cluster_names = {k: ", ".join(v) for k, v in cluster_terms.items()}
    print("  Cluster names:")
    for k, name in sorted(cluster_names.items()):
        count = (labels == k).sum()
        print(f"    {k}: {name} ({count} posts)")

    # Filter outliers (3x IQR)
    u1, u2 = umap_coords[:, 0], umap_coords[:, 1]
    bounds = {}
    for col, vals in [("u1", u1), ("u2", u2)]:
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        bounds[f"{col}_min"] = float(q1 - 3 * iqr)
        bounds[f"{col}_max"] = float(q3 + 3 * iqr)

    # Build records
    records = []
    for idx in range(len(frame)):
        x = float(umap_coords[idx, 0])
        y = float(umap_coords[idx, 1])
        if x < bounds["u1_min"] or x > bounds["u1_max"]:
            continue
        if y < bounds["u2_min"] or y > bounds["u2_max"]:
            continue

        row = frame.iloc[idx]
        date = row["date"]
        if pd.isna(date):
            continue

        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        records.append({
            "s": row["slug"],
            "t": row["title"],
            "d": date_str,
            "y": int(date_str[:4]),
            "c": row["category"],
            "src": row["source"],
            "k": int(labels[idx]),
            "u1": round(x, 4),
            "u2": round(y, 4),
        })

    cat_counts = pd.Series([r["c"] for r in records]).value_counts()
    u1_vals = [r["u1"] for r in records]
    u2_vals = [r["u2"] for r in records]

    output = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": records,
        "cats": cat_counts.index.tolist(),
        "clusterNames": cluster_names,
        "sources": sorted(frame["source"].unique().tolist()),
        "minYear": min(r["y"] for r in records),
        "maxYear": max(r["y"] for r in records),
        "xDom": [round(min(u1_vals), 2), round(max(u1_vals), 2)],
        "yDom": [round(min(u2_vals), 2), round(max(u2_vals), 2)],
    }

    with open(JSON_OUT, "w") as f:
        json.dump(output, f, separators=(",", ":"), ensure_ascii=False)

    size_kb = JSON_OUT.stat().st_size / 1024
    print(f"\nWrote {len(records)} posts → {JSON_OUT} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    import typer
    typer.run(main)
