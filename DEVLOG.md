# Development Log — Blog Embeddings Map

---

## 2026-03-16 — Initial build: parsing, embeddings pipeline, UMAP visualization

### User prompts this session

> "I found this fairly nice blog post on creating an embeddings thing from blog. Can you try and implement this for my blog somewhere on my computer? You should find something called 'Nonthuda' - that's the name of my blog and XML file related to that. I also have a Substack I can export the XML stuff from there as well. They follow the instructions in this blog post and the GitHub to replicate this stuff here. Maybe you can pull the repo here and work on it. https://www.s-anand.net/blog/blog-embeddings-map/"
>
> "it is noenthuda"
>
> "among all the xml outputs use the latest one"
>
> "ok use claude rather than gemini. and there should be a later blog export - nov 2023 or something. i also have another blog (on substack) that i started then. i need you to include that as well. let me know the process of extracting that (actually i want you to extract from 2 substacks - noenthuda.substack.com and artofdatascience.substack.com, both of which I write)."
>
> "ok just use gemini. tell me where to get the key. and i'm positive i've exported my noenthuda.com posts in late 2023 - that's when i moved to substack"
>
> "this must be the file: pertinentobservations.WordPress.2023-09-29-2.xml in downloads"
>
> "Ok, I have got 2 new exports of my two substacks in the Downloads folder now. Pick them up and let me search for the main XML and get back to you."
>
> "ok made a fresh download. check downloads."
>
> "i'll paste the key here. you take it from there"
>
> "let's abandon for now and resume tomorrow"
>
> "yes document the whole thing properly"

### What changed

**Project created** at `/Users/Karthik/Documents/work/data work/blog-embeddings/`. Adapted from S. Anand's blog embeddings map (https://github.com/sanand0/blog/tree/main/analysis/embeddings) but rewritten to work with WordPress XML + Substack CSV/HTML exports instead of markdown files.

**Source parsing (`parse_sources.py`)** — Parses three sources into a unified `posts.parquet`:
- WordPress WXR XML export (`wordpress-export.xml`, fresh export from 2026-03-16, 2,796 published posts with content > 50 chars)
- Substack noenthuda export (162 published posts with HTML content)
- Substack artofdatascience export (97 published posts)
- Deduplicates by title (45 removed, mostly WordPress→Substack migration overlap)
- Final output: 3,010 posts spanning 2004-07-27 to 2026-03-10

**Embeddings pipeline (`embeddings.py`)** — Generates 768-dim embeddings using Gemini `gemini-embedding-001`. Features:
- Content hashing for incremental re-runs (skips unchanged posts)
- DuckDB storage with per-batch commits for crash resilience
- Exponential backoff on rate limits (429/RESOURCE_EXHAUSTED)
- Batch size of 10 with 2s inter-batch delay to stay under free tier limits
- **Status: 985/3,010 posts embedded before rate limit exhaustion. Resumable.**

**UMAP visualization (`blogmap.py`)** — Reads posts + embeddings, runs PCA(50) → UMAP(2D) → KMeans(12 clusters), outputs `blogmap/blogmap.json`. Cluster names auto-generated via TF-IDF term extraction (the reference project hard-coded them). Outliers filtered by 3x IQR.

**Interactive HTML (`blogmap/index.html`)** — Adapted from reference with additions:
- "Source" as a 4th color-by and filter dimension (alongside Category, Cluster, Year)
- Dynamic cluster names loaded from JSON (not hard-coded)
- URL builder per source (noenthuda.com, noenthuda.substack, artofdatascience.substack)
- Canvas-rendered scatter plot, D3 brush selection, quadtree tooltip, bar chart race, date range slider with play/pause

### Decisions rejected

**Claude for embeddings** — User initially wanted Claude, but Anthropic doesn't offer an embeddings API. Suggested Voyage AI as the Claude ecosystem alternative, but user opted for Gemini (free tier, simpler setup).

**2023-09-29 WordPress export** — User thought this was the latest export. Turned out to be a part-2 file containing only 400 attachments (images), zero posts. Used fresh 2026-03-16 export instead (2,838 posts).

**2023-01-02 WordPress export** — Was the most comprehensive historical export (2,766 posts, 21MB) but superseded by the fresh download.

### Technical choices

**WordPress XML parsing** — Used `xml.etree.ElementTree` with WXR namespaces. Filters to `post_type=post` and `status=publish`. Strips HTML to plain text via regex (block tags → newlines, then strip all tags, then `html.unescape`). Posts with < 50 chars content are dropped.

**Substack parsing** — CSV provides metadata (title, date, publish status). HTML content is in separate files named `{post_id}.{slug}.html`. Joined by post_id prefix. Source name used as the category for Substack posts (WordPress posts retain their original categories).

**Timezone normalization** — WordPress dates are tz-naive, Substack dates are UTC. Pandas `sort_values` crashes on mixed tz. Fixed by converting all to tz-naive UTC via `pd.to_datetime(utc=True).dt.tz_localize(None)`.

**Embedding model** — `gemini-embedding-001` (not the experimental `gemini-embedding-exp-03-07` which isn't available). 768 dimensions, `RETRIEVAL_DOCUMENT` task type.

### Files created

| File | Purpose |
|------|---------|
| `parse_sources.py` | WordPress XML + Substack CSV/HTML → `posts.parquet` |
| `embeddings.py` | Gemini embeddings → DuckDB + `embeddings.parquet` |
| `blogmap.py` | PCA + UMAP + KMeans → `blogmap/blogmap.json` |
| `blogmap/index.html` | Interactive UMAP scatter plot visualization |
| `.env` | Gemini API key (gitignored) |
| `.env.example` | Template for API key |
| `.gitignore` | Excludes data files, models, API keys |
| `posts.parquet` | 3,010 parsed posts (generated) |
| `embeddings.duckdb` | 985 embeddings so far (generated) |

### Next steps

1. Resume `python embeddings.py` (will pick up from 985/3,010)
2. Run `python blogmap.py` to generate UMAP coordinates + clusters
3. Serve `blogmap/` directory locally to view visualization
4. Initialize git repo, create README, push to GitHub
