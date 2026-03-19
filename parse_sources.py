#!/usr/bin/env python3
"""Parse WordPress XML export and Substack CSV exports into a unified posts parquet file.

Sources:
  - wordpress-export.xml (WordPress WXR format)
  - noenthuda-substack-posts.csv + noenthuda-substack-html/*.html
  - artofdatascience-substack-posts.csv + artofdatascience-substack-html/*.html

Output:
  - posts.parquet with columns: slug, title, date, category, source, content
"""

import csv
import re
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
WP_XML = ROOT / "wordpress-export.xml"
NOENTHUDA_CSV = ROOT / "noenthuda-substack-posts.csv"
NOENTHUDA_HTML = ROOT / "noenthuda-substack-html"
AODS_CSV = ROOT / "artofdatascience-substack-posts.csv"
AODS_HTML = ROOT / "artofdatascience-substack-html"
OUTPUT = ROOT / "posts.parquet"

# WordPress namespaces
NS = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "wp": "http://wordpress.org/export/1.2/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

# Regex for stripping HTML tags
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t]+")


def strip_html(html: str) -> str:
    """Convert HTML to plain text."""
    if not html:
        return ""
    # Replace block-level tags with newlines
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|div|h[1-6]|li|tr|blockquote)>", "\n", text, flags=re.IGNORECASE)
    text = HTML_TAG_RE.sub("", text)
    text = unescape(text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def parse_wordpress(xml_path: Path) -> list[dict]:
    """Parse WordPress WXR XML export."""
    print(f"Parsing WordPress XML: {xml_path.name}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    channel = root.find("channel")

    rows = []
    for item in channel.findall("item"):
        post_type = item.find("wp:post_type", NS)
        if post_type is None or post_type.text != "post":
            continue

        status = item.find("wp:status", NS)
        if status is not None and status.text != "publish":
            continue

        title_el = item.find("title")
        title = title_el.text if title_el is not None and title_el.text else ""

        content_el = item.find("content:encoded", NS)
        raw_content = content_el.text if content_el is not None and content_el.text else ""

        slug_el = item.find("wp:post_name", NS)
        slug = slug_el.text if slug_el is not None and slug_el.text else ""

        date_el = item.find("wp:post_date", NS)
        date_str = date_el.text if date_el is not None and date_el.text else None

        # Get first category
        categories = []
        for cat in item.findall("category"):
            if cat.get("domain") == "category" and cat.text:
                categories.append(cat.text)

        content = strip_html(raw_content)
        if not content or len(content) < 50:
            continue

        rows.append({
            "slug": slug or title.lower().replace(" ", "-")[:60],
            "title": title,
            "date": pd.to_datetime(date_str, errors="coerce"),
            "category": categories[0] if categories else "(none)",
            "source": "noenthuda.com",
            "content": content,
        })

    print(f"  Found {len(rows)} published posts")
    return rows


def parse_substack(csv_path: Path, html_dir: Path, source_name: str) -> list[dict]:
    """Parse Substack CSV + HTML exports."""
    print(f"Parsing Substack: {source_name}")
    rows = []

    # Read CSV for metadata
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        posts_meta = list(reader)

    for meta in posts_meta:
        if meta.get("is_published") != "true":
            continue

        post_id_slug = meta.get("post_id", "")
        title = meta.get("title", "")
        date_str = meta.get("post_date", "")
        subtitle = meta.get("subtitle", "")

        # Find matching HTML file
        html_file = html_dir / f"{post_id_slug}.html"
        if not html_file.exists():
            # Try finding by prefix
            prefix = post_id_slug.split(".")[0]
            matches = list(html_dir.glob(f"{prefix}.*.html"))
            html_file = matches[0] if matches else None

        content = ""
        if html_file and html_file.exists():
            raw_html = html_file.read_text(encoding="utf-8", errors="replace")
            content = strip_html(raw_html)

        if not content or len(content) < 50:
            continue

        # Extract slug from post_id (format: "12345.slug-name")
        slug = post_id_slug.split(".", 1)[1] if "." in post_id_slug else post_id_slug

        rows.append({
            "slug": slug,
            "title": title,
            "date": pd.to_datetime(date_str, errors="coerce"),
            "category": source_name,  # Use substack name as category
            "source": source_name,
            "content": content,
        })

    print(f"  Found {len(rows)} published posts")
    return rows


def main():
    all_rows = []

    # WordPress
    if WP_XML.exists():
        all_rows.extend(parse_wordpress(WP_XML))

    # Substack - noenthuda
    if NOENTHUDA_CSV.exists():
        all_rows.extend(parse_substack(NOENTHUDA_CSV, NOENTHUDA_HTML, "noenthuda.substack"))

    # Substack - art of data science
    if AODS_CSV.exists():
        all_rows.extend(parse_substack(AODS_CSV, AODS_HTML, "artofdatascience.substack"))

    df = pd.DataFrame(all_rows)

    # Normalize all dates to tz-naive UTC
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

    # Drop duplicates by title (WordPress and Substack may overlap)
    before = len(df)
    df = df.drop_duplicates(subset="title", keep="first")
    if before != len(df):
        print(f"  Removed {before - len(df)} duplicate titles")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Drop rows with no date
    no_date = df["date"].isna().sum()
    if no_date:
        print(f"  Dropping {no_date} posts with no date")
        df = df.dropna(subset=["date"])

    df.to_parquet(OUTPUT, index=False)
    print(f"\nTotal: {len(df)} posts → {OUTPUT}")
    print(f"  Sources: {df['source'].value_counts().to_dict()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
