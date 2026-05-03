"""Pipeline runner — orchestrates extraction, indexing, NER, embeddings."""

import os
import sys
import json
import argparse
import time

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


def main():
    parser = argparse.ArgumentParser(description="Book Oracle Pipeline")
    parser.add_argument("--books", required=True, help="Path to book collection directory")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Max books to process")
    parser.add_argument("--stats-only", action="store_true", help="Just show stats")
    args = parser.parse_args()

    if args.stats_only:
        conn, _ = ctrl.init_db()
        stats = ctrl.get_stats(conn)
        print(json.dumps(stats, indent=2))
        conn.close()
        return

    run_extraction(args.books, workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
