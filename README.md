# Blog Embeddings Map

Interactive UMAP visualization of ~3,000 blog posts from [Pertinent Observations](https://www.noenthuda.com) (WordPress, 2004-present), [noenthuda.substack.com](https://noenthuda.substack.com), and [artofdatascience.substack.com](https://artofdatascience.substack.com).

## Attribution

This project is a fork/adaptation of [S. Anand's blog embeddings map](https://github.com/sanand0/blog/tree/main/analysis/embeddings). Anand wrote an excellent [blog post](https://www.s-anand.net/blog/blog-embeddings-map/) describing how he built an interactive map of his own blog posts using Gemini embeddings and UMAP. I liked the idea and adapted his code to work with my WordPress + Substack exports instead of markdown files. The visualization (blogmap) is heavily based on his original work. Thanks, Anand!

## What it does

- Generates semantic embeddings for every blog post using Gemini
- Reduces to 2D with UMAP and clusters with KMeans
- Renders an interactive scatter plot where you can:
  - Color by category, cluster, source, or year
  - Filter by any dimension
  - Brush-select regions to see post lists
  - Animate through time with the date slider

## Setup

```bash
# Install dependencies (into your existing venv)
uv pip install google-genai duckdb python-dotenv rich typer pandas pyarrow scikit-learn umap-learn pyyaml

# Get a Gemini API key at https://aistudio.google.com/apikey
echo "GEMINI_API_KEY=your-key" > .env
```

## Data preparation

Place these files in the project root:

- `wordpress-export.xml` — WordPress WXR export from Tools > Export
- `noenthuda-substack-posts.csv` + `noenthuda-substack-html/` — Substack export (Settings > Export)
- `artofdatascience-substack-posts.csv` + `artofdatascience-substack-html/` — Second Substack export

## Running the pipeline

```bash
# 1. Parse all sources into posts.parquet
python parse_sources.py

# 2. Generate embeddings (resumable, takes ~15 min on free tier)
python embeddings.py

# 3. Generate UMAP visualization data
python blogmap.py

# 4. View the visualization
cd blogmap && python -m http.server 8000
# Open http://localhost:8000
```

## Project structure

```
├── parse_sources.py      # WordPress XML + Substack → posts.parquet
├── embeddings.py         # Gemini embeddings → embeddings.parquet
├── blogmap.py            # PCA + UMAP + KMeans → blogmap.json
├── blogmap/
│   ├── index.html        # Interactive visualization
│   └── blogmap.json      # Generated data (gitignored)
├── .env                  # GEMINI_API_KEY (gitignored)
├── DEVLOG.md             # Development log
└── README.md
```
