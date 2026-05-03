"""Multi-source retriever — Meilisearch + FAISS + entities."""

import os
import numpy as np


def retrieve_meili(query, top_k=10):
    """Retrieve from Meilisearch full-text index."""
    try:
        from pipeline.index import search
        result = search(query, limit=top_k)
        hits = result.get("hits", [])
        return [{
            "id": h.get("id"),
            "title": h.get("title", ""),
            "author": h.get("author", ""),
            "text": h.get("text", "")[:2000],
            "format": h.get("format", ""),
            "score": h.get("_rankingScore", 1.0),
            "source": "fulltext"
        } for h in hits]
    except Exception as e:
        return []


def retrieve_faiss(query, top_k=10):
    """Retrieve from FAISS semantic index."""
    try:
        from sentence_transformers import SentenceTransformer
        import faiss

        base = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base, "data", "models", "mini-lm")
        index_path = os.path.join(base, "data", "faiss.index")

        if not os.path.exists(index_path):
            return []

        model = SentenceTransformer(model_path)
        index = faiss.read_index(index_path)
        vec = model.encode([query])
        distances, indices = index.search(vec, top_k)

        from pipeline.control import init_db
        conn, _ = init_db()
        c = conn.cursor()

        results = []
        for i, idx_val in enumerate(indices[0]):
            if idx_val < 0:
                continue
            c.execute("""
                SELECT s.book_id, s.chapter_idx, s.start_char, s.end_char, b.filename, b.title, b.author
                FROM segments s JOIN books b ON s.book_id = b.id WHERE s.id=?
            """, (int(idx_val) + 1,))
            row = c.fetchone()
            if row:
                # Read the segment text
                text_path = os.path.join(base, "data", "extracted", f"{row[0]}.txt")
                text = ""
                if os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8") as f:
                        f.seek(row[2] if row[2] else 0)
                        chunk_size = (row[3] or 2000) - (row[2] or 0)
                        text = f.read(min(chunk_size, 2000) or 2000)

                results.append({
                    "id": str(row[0]),
                    "title": row[5] or row[4] or "Unknown",
                    "author": row[6] or "",
                    "text": text[:2000],
                    "score": float(distances[0][i]),
                    "source": "semantic"
                })
        conn.close()
        return results
    except Exception as e:
        return []


def retrieve(query, top_k=20):
    """Multi-source retrieval with fusion."""
    meili_results = retrieve_meili(query, top_k // 2)
    faiss_results = retrieve_faiss(query, top_k // 2)

    # Fusion: alternating merge by score
    all_results = meili_results + faiss_results
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Dedup by id
    seen = set()
    unique = []
    for r in all_results:
        rid = str(r.get("id", ""))
        if rid not in seen:
            seen.add(rid)
            unique.append(r)

    return unique[:top_k]
