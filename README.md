# Book Oracle — Offline Book RAG System

Zero-cost, fully offline RAG system for personal book collections. Built on innerstellar (WSL2, RTX 2080).

## Architecture

```
Books (PDF/EPUB/MOBI) → Extract → Meilisearch (full-text) → FAISS (semantic) → spaCy (entities) → RAG (llama.cpp)
```

## Phases

| Phase | What | Status |
|-------|------|--------|
| 0 | Scaffold + GitHub | ✅ |
| 1 | Pipeline (extract + control) | ✅ |
| 2 | Search (Meilisearch) | ✅ |
| 3 | Web UI (FastAPI) | ✅ |
| 4 | Entity Graph (spaCy) | ✅ |
| 5 | Embeddings + Concepts | ✅ |
| 6 | RAG Oracle | ✅ |

## Quick Start

```bash
pip install -r requirements.txt
python pipeline/run.py --books /path/to/books
python run.py
```

Open http://localhost:8000

## Data Model

- `books` table: id, path, format, filename, size_bytes, title, author, isbn, language, text_extracted, text_path, text_chars, indexed, ner_done, embedded, error_log
- `entities` table: id, book_id, entity_text, entity_type, context_snippet, char_offset
- `entity_links` table: entity_id, book_id, count
- `segments` table: segment_id, book_id, chapter_idx, start_char, end_char
- `concepts` table: concept_id, parent_id, label, centroid, member_count, level
