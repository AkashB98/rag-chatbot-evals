#!/usr/bin/env python3
"""
Build the sqlite-vec index for grounded-rag-chat.

    data/sample_docs/*.md  ->  chunk (~120 words, 20 overlap)
                           ->  embed (fastembed, local, no API keys)
                           ->  data/rag.db (sqlite-vec, single file)

Idempotent: every run rebuilds the DB from scratch.
Swap in your own markdown docs and re-run — no code changes needed.

Usage:
    python ingest.py
"""

from pathlib import Path

from rag import VectorStore, chunk_text, default_embed_fn

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "data" / "sample_docs"
DB_PATH = ROOT / "data" / "rag.db"


def title_of(path, text):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("-", " ").title()


def clean_doc(text):
    """Drop the per-doc 'fictional sample data' disclaimer lines.

    Every sample doc carries one for honesty; it adds no topical signal and
    would otherwise leak into quoted answers. The corpus-level fiction notice
    lives in the README and the chat UI footer instead.
    """
    return "\n".join(
        line for line in text.splitlines()
        if "fictional sample data for demo purposes" not in line
    )


def main():
    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"no markdown docs in {DOCS_DIR}")

    chunks = []
    for path in md_files:
        text = clean_doc(path.read_text(encoding="utf-8"))
        title = title_of(path, text)
        doc_id = path.stem  # evals match expected sources on this stem
        for chunk in chunk_text(text):
            chunks.append((doc_id, title, chunk))

    store = VectorStore.build(DB_PATH, chunks, default_embed_fn)
    print(f"indexed {store.doc_count()} chunks from {len(md_files)} docs -> {DB_PATH}")
    store.close()


if __name__ == "__main__":
    main()
