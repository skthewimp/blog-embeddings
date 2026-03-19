#!/usr/bin/env python3
"""Generate Gemini embeddings for blog posts, stored in DuckDB + Parquet.

Reads posts.parquet (from parse_sources.py), generates 768-dim embeddings
using Gemini, and stores results incrementally in DuckDB.

Usage:
    python embeddings.py                  # embed all new/changed posts
    python embeddings.py --limit 10       # test run: embed at most 10
    python embeddings.py --force          # re-embed everything
"""

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

load_dotenv(override=True)

ROOT = Path(__file__).parent
POSTS_PARQUET = ROOT / "posts.parquet"
DB_PATH = ROOT / "embeddings.duckdb"
PARQUET_PATH = ROOT / "embeddings.parquet"
MODEL = "gemini-embedding-001"
DIMENSIONS = 768
CHUNK_SIZE = 5        # contents per embed_content call
BATCH_DELAY = 10      # seconds between batches to avoid rate limits
MAX_CHARS = 20_000    # truncate each post before embedding
INPUT_VERSION = "v1"  # bump to force re-embedding after content changes

console = Console()

# Regex for cleaning
URL_RE = re.compile(r"https?://\S+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def get_db() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(DB_PATH))
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            slug       TEXT PRIMARY KEY,
            hash       TEXT NOT NULL,
            embedding  FLOAT[{DIMENSIONS}],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn


def content_hash(title: str, content: str) -> str:
    payload = f"{title}\0{content}\0{INPUT_VERSION}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def prepare_text(title: str, content: str) -> str:
    """Clean and truncate content for embedding."""
    text = f"{title}\n\n{content}" if title else content
    text = URL_RE.sub("", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()[:MAX_CHARS]


def embed_with_backoff(client, texts: list[str]) -> list[list[float]]:
    """Call embed_content with exponential back-off on rate-limit errors."""
    delay = 60
    for attempt in range(50):
        try:
            result = client.models.embed_content(
                model=MODEL,
                contents=texts,
                config={"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": DIMENSIONS},
            )
            return [list(e.values) for e in result.embeddings]
        except genai_errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                # Parse retry delay from error if available
                retry_match = re.search(r"retry in (\d+)", str(e), re.IGNORECASE)
                wait = int(retry_match.group(1)) + 5 if retry_match else delay
                wait = max(wait, 60)  # always wait at least 60s on rate limit
                console.print(f"[yellow]Rate-limited — waiting {wait}s (attempt {attempt + 1}/50)[/yellow]")
                time.sleep(wait)
            else:
                raise
        except genai_errors.ServerError:
            console.print(f"[yellow]Server error — retrying in {delay}s (attempt {attempt + 1})[/yellow]")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("Embedding retries exhausted")


def export_parquet(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(f"COPY embeddings TO '{PARQUET_PATH}' (FORMAT PARQUET)")
    n = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    console.print(f"[green]Exported {n} embeddings → {PARQUET_PATH}[/green]")


def main(
    limit: Optional[int] = None,
    force: bool = False,
) -> None:
    """Generate Gemini embeddings for all posts."""
    import typer
    app = typer.Typer()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]Set GEMINI_API_KEY in .env or environment[/red]")
        raise SystemExit(1)

    client = genai.Client(api_key=api_key)
    conn = get_db()

    # Load posts
    posts = pd.read_parquet(POSTS_PARQUET)
    console.print(f"Loaded {len(posts)} posts from {POSTS_PARQUET.name}")

    # Load existing hashes
    existing: dict[str, str] = {}
    if not force:
        existing = dict(conn.execute("SELECT slug, hash FROM embeddings").fetchall())

    # Find posts needing embedding
    to_embed = []
    hash_hits = 0
    for _, row in posts.iterrows():
        h = content_hash(row["title"], row["content"])
        if existing.get(row["slug"]) == h:
            hash_hits += 1
            continue
        to_embed.append((row["slug"], row["title"], row["content"], h))

    if limit:
        to_embed = to_embed[:limit]

    console.print(f"Posts: {len(posts)} total, {hash_hits} hash-skipped, {len(to_embed)} to embed")

    if not to_embed:
        console.print("Nothing new to embed.")
        export_parquet(conn)
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding...", total=len(to_embed))

        for i in range(0, len(to_embed), CHUNK_SIZE):
            chunk = to_embed[i : i + CHUNK_SIZE]
            texts = [prepare_text(title, content) for _, title, content, _ in chunk]
            vectors = embed_with_backoff(client, texts)

            for (slug, _, _, h), vec in zip(chunk, vectors):
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings (slug, hash, embedding) VALUES (?, ?, ?)",
                    [slug, h, vec],
                )
            progress.advance(task, len(chunk))
            if i + CHUNK_SIZE < len(to_embed):
                time.sleep(BATCH_DELAY)

    export_parquet(conn)


if __name__ == "__main__":
    import typer
    typer.run(main)
