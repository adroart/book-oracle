#!/usr/bin/env python3
"""Book Oracle — single startup command.

Usage:
    python run.py                          # Start web UI only
    python run.py --pipeline /path/books   # Run extraction + start web
    python run.py --llm /path/to/model.gguf  # Start with LLM
"""

import os
import sys
import subprocess
import time
import argparse

BASE_DIR = os.path.dirname(__file__)


def start_meilisearch():
    """Start Meilisearch if not already running."""
    import httpx
    try:
        r = httpx.get("http://localhost:7700/health", timeout=2)
        if r.status_code == 200:
            print("[boot] Meilisearch already running")
            return None
    except Exception:
        pass

    data_dir = os.path.join(BASE_DIR, "data", "meili")
    os.makedirs(data_dir, exist_ok=True)
    log_path = os.path.join(BASE_DIR, "data", "meili.log")

    print("[boot] Starting Meilisearch...")
    proc = subprocess.Popen(
        ["meilisearch", "--db-path", data_dir, "--http-addr", "127.0.0.1:7700", "--no-analytics"],
        stdout=open(log_path, "a"), stderr=subprocess.STDOUT
    )
    time.sleep(2)
    return proc


def start_web():
    """Start the FastAPI web server."""
    from web.main import main
    print("[boot] Starting web UI at http://localhost:8000")
    main()


def main():
    parser = argparse.ArgumentParser(description="Book Oracle")
    parser.add_argument("--pipeline", help="Run extraction pipeline on this book directory before starting UI")
    parser.add_argument("--llm", help="Path to GGUF model file for RAG")
    parser.add_argument("--workers", type=int, default=4, help="Pipeline workers")
    parser.add_argument("--limit", type=int, default=None, help="Max books to process")
    args = parser.parse_args()

    # Run extraction pipeline first if requested
    if args.pipeline:
        print(f"[boot] Running extraction pipeline on: {args.pipeline}")
        from pipeline.run import run_extraction
        result = run_extraction(args.pipeline, workers=args.workers, limit=args.limit)
        print(f"[boot] Pipeline complete: {result}")

    # Start LLM if requested
    if args.llm:
        if not os.path.exists(args.llm):
            print(f"[boot] ERROR: Model file not found: {args.llm}")
            sys.exit(1)
        from rag.llm import start_llama_server
        print(f"[boot] Starting llama-server with: {args.llm}")
        start_llama_server(args.llm)
        print("[boot] LLM ready on port 8080")

    # Start Meilisearch
    start_meilisearch()

    # Start web UI
    start_web()


if __name__ == "__main__":
    main()
