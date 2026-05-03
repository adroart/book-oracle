# Book Oracle - Build Context

## Project
Offline, zero-cost RAG system for book collection. Built on innerstellar.

## Directory Layout
```
pipeline/           # Core pipeline engine
  __init__.py
  control.py        # SQLite control plane
  extract.py        # Text extraction (PDF, EPUB, MOBI, TXT)
  index.py          # Meilisearch indexing
  ner.py            # spaCy NER
  entity_graph.py   # Entity linking
  embed.py          # Sentence embeddings + FAISS
  concept_space.py  # HDBSCAN clustering
web/                # Web interface
  __init__.py
  main.py           # FastAPI server
  search.py         # Search endpoints
  templates/        # Jinja2 HTML
  static/           # CSS, JS
rag/                # RAG engine
  __init__.py
  retriever.py      # Multi-source retrieval
  generator.py      # LLM prompt + cite
  chain.py          # Full RAG chain
  llm.py            # llama.cpp wrapper
run.py              # Startup command
```

## Data
- Books: P:\Files\3 Resources\Books\ (pCloud, Windows)
- Extracted: /home/krimez/repos/book-oracle/data/extracted/
- DB: /home/krimez/repos/book-oracle/data/library.db
- Meilisearch data: /home/krimez/repos/book-oracle/data/meili/
- FAISS index: /home/krimez/repos/book-oracle/data/faiss.index
- Models: /home/krimez/repos/book-oracle/data/models/

## Stack
Python 3.12 | SQLite | Meilisearch | FAISS | Sentence-Transformers | spaCy | llama.cpp | FastAPI

## Key Commands
- `python run.py` - Start everything (Meilisearch + FastAPI)
- `python pipeline/run.py` - Run extraction pipeline only
