"""Search API endpoints — Meilisearch + FAISS (semantic)."""

import os
import json
from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/search", tags=["search"])


def _get_client():
    from pipeline.index import get_client
    return get_client()


@router.get("")
async def search_text(q: str = Query(..., description="Search query"), limit: int = Query(20, le=100)):
    try:
        import pipeline.index as idx
        client = _get_client()
        result = idx.search(q, limit=limit)
        return {"query": q, "results": result.get("hits", []), "total": result.get("estimatedTotalHits", 0)}
    except Exception as e:
        return {"query": q, "error": str(e), "results": []}


@router.get("/semantic")
async def search_semantic(q: str = Query(...), limit: int = Query(20, le=100)):
    """FAISS semantic search — returns book segments matching the query concept."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import faiss

        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models", "mini-lm")
        index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss.index")

        if not os.path.exists(index_path):
            return {"query": q, "error": "No FAISS index built yet. Run pipeline/embed.py first.", "results": []}

        model = SentenceTransformer(model_path)
        index = faiss.read_index(index_path)

        vec = model.encode([q])
        distances, indices = index.search(vec, limit)

        # Load segment metadata
        from pipeline.control import init_db
        conn, _ = init_db()
        c = conn.cursor()

        results = []
        for i, idx_val in enumerate(indices[0]):
            if idx_val < 0:
                continue
            c.execute("""
                SELECT s.id, s.book_id, s.chapter_idx, s.start_char, s.end_char, b.filename, b.title, b.author
                FROM segments s JOIN books b ON s.book_id = b.id WHERE s.id=?
            """, (int(idx_val) + 1,))
            row = c.fetchone()
            if row:
                results.append({
                    "segment_id": row[0], "book_id": row[1], "chapter": row[2],
                    "filename": row[5], "title": row[6], "author": row[7],
                    "score": float(distances[0][i]),
                })

        conn.close()
        return {"query": q, "results": results}
    except Exception as e:
        return {"query": q, "error": str(e), "results": []}


@router.get("/hybrid")
async def search_hybrid(q: str = Query(...), limit: int = Query(20, le=50)):
    """Combined full-text + semantic search."""
    import asyncio
    text_results = await search_text(q, limit)
    semantic_results = await search_semantic(q, limit)
    return {
        "query": q,
        "fulltext": text_results.get("results", []),
        "semantic": semantic_results.get("results", []),
    }
