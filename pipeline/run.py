"""Pipeline runner — orchestrates extraction, indexing, NER, embeddings, RAG."""

import os
import sys
import json
import argparse
import time
import subprocess

from pipeline import control as ctrl
from pipeline import extract as ext


def run_extraction(books_dir, workers=4, limit=None):
    """Stage 1: Register books and extract text."""
    conn, _ = ctrl.init_db()
    result = ctrl.register_books(books_dir, conn)
    print(f"[pipeline] Registered {result['registered']} new books ({result['total_files']} total found)")

    pending = ctrl.get_pending(conn, stage="extract", limit=limit or 100000)
    print(f"[pipeline] Extracting {len(pending)} books (workers={workers})...")

    batch = [(b["id"], b["path"], b["format"]) for b in pending]
    start = time.time()
    results = ext.extract_batch(batch, workers=workers)
    elapsed = time.time() - start

    ok = err = 0
    for book_id, chars, error in results:
        if error:
            ctrl.mark_error(book_id, "extract", error, conn)
            err += 1
        else:
            ctrl.mark_done(book_id, "extract", conn)
            ok += 1

    print(f"[pipeline] Extraction done: {ok} OK, {err} errors in {elapsed:.1f}s")
    conn.close()
    return {"ok": ok, "errors": err, "elapsed": elapsed}


def run_graph():
    """Stage 3: Entity graph via spaCy NER."""
    from pipeline import graph
    count = graph.run_graph()
    print(f"[pipeline] Entity graph: {count} mentions extracted")
    return count


def run_embeddings():
    """Stage 4: Generate embeddings and FAISS index."""
    from pipeline import embeddings
    n = embeddings.run_embeddings()
    print(f"[pipeline] Embeddings: {n} vectors indexed")
    return n


def run_rag(query):
    """Stage 5: RAG query."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from rag.oracle import BookOracle
    oracle = BookOracle()
    result = oracle.answer(query)
    print()
    print(result["answer"])
    if result.get("sources"):
        print("\n📖 Sources:")
        for s in result["sources"]:
            print(f"  • {s['title']} (relevance: {s['relevance']})")
    return result


def main():
    parser = argparse.ArgumentParser(description="Book Oracle Pipeline")
    parser.add_argument("--books", help="Path to book collection directory (for extraction)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Max books to process")
    parser.add_argument("--stats-only", action="store_true", help="Just show stats")
    parser.add_argument("--extract", action="store_true", help="Extract books")
    parser.add_argument("--graph", action="store_true", help="Run entity graph (NER)")
    parser.add_argument("--embeddings", action="store_true", help="Generate embeddings + FAISS index")
    parser.add_argument("--rag", type=str, nargs="*", help="Run RAG query (use '--rag query here')")
    parser.add_argument("--reset", action="store_true", help="Reset database")
    parser.add_argument("--web", action="store_true", help="Start web UI server")

    args = parser.parse_args()

    if args.reset:
        if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "books.db")):
            os.remove(os.path.join(os.path.dirname(__file__), "..", "data", "books.db"))
            print("[pipeline] Database reset.")
        conn, _ = ctrl.init_db()
        conn.close()
        return

    if args.web:
        web_dir = os.path.join(os.path.dirname(__file__), "..", "web")
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        print("[pipeline] Starting web UI on http://0.0.0.0:8000")
        subprocess.run([sys.executable, "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"])
        return

    if args.stats_only:
        conn, _ = ctrl.init_db()
        stats = ctrl.get_stats(conn)
        print(json.dumps(stats, indent=2))
        conn.close()
        return

    if args.extract:
        if not args.books:
            print("[pipeline] --books required for extraction")
            sys.exit(1)
        run_extraction(args.books, workers=args.workers, limit=args.limit)

    if args.graph:
        run_graph()

    if args.embeddings:
        run_embeddings()

    if args.rag:
        query = " ".join(args.rag)
        run_rag(query)


if __name__ == "__main__":
    main()
