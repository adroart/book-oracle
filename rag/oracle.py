#!/usr/bin/env python3
"""RAG Oracle — semantic question answering over book collection.

Pipeline:
  1. Embed user query → FAISS nearest-neighbor search (top-5 chunks)
  2. Retrieve chunk text + book metadata from SQLite
  3. Format RAG prompt with context + question
  4. Generate answer via llama.cpp (Qwen2.5-1.5B GGUF)
  5. Return answer with citations (book title, author)
"""

import json
import sys
import csv
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

DB_PATH = Path(__file__).parent.parent / "data" / "books.db"
INDEX_PATH = Path(__file__).parent.parent / "data" / "embeddings" / "books.index"
MODEL_PATH = Path(__file__).parent.parent / "models" / "gguf" / "qwen.gguf"
MAPPING_PATH = Path(__file__).parent.parent / "data" / "embeddings" / "chunk_mapping.csv"


class BookOracle:
    def __init__(self):
        print("[oracle] Loading sentence-transformer model...", file=sys.stderr)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[oracle] Loading FAISS index...", file=sys.stderr)
        self.index = faiss.read_index(str(INDEX_PATH))
        print(f"[oracle] Index loaded: {self.index.ntotal} vectors, {self.index.d} dims", file=sys.stderr)

        print("[oracle] Loading mapping...", file=sys.stderr)
        self.mapping = self._load_mapping()

        print("[oracle] Loading LLM (Qwen2.5-1.5B GGUF)...", file=sys.stderr)
        self.llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
        )
        print("[oracle] Ready!", file=sys.stderr)

    def _load_mapping(self):
        """Load chunk_id -> (book_id, chunk_index) mapping from CSV."""
        mapping = []
        with open(MAPPING_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping.append((int(row["book_id"]), int(row["chunk_index"])))
        return mapping

    def _get_chunk_text(self, book_id, chunk_index):
        """Retrieve chunk text from SQLite."""
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            "SELECT chunk_text FROM book_chunks WHERE book_id = ? AND chunk_index = ?",
            (book_id, chunk_index),
        )
        row = c.fetchone()
        c.execute(
            "SELECT title, author FROM extracted_books WHERE id = ?",
            (book_id,),
        )
        book_info = c.fetchone()
        conn.close()

        text = row[0] if row else ""
        title = book_info[0] if book_info else "Unknown"
        author = book_info[1] if book_info else "Unknown"
        return text, title, author

    def search(self, query, top_k=5):
        """Search the book index and return top-k chunks with book metadata."""
        q_vec = self.embedder.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(np.array(q_vec, dtype=np.float32), top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.mapping):
                continue
            book_id, chunk_index = self.mapping[idx]
            text, title, author = self._get_chunk_text(book_id, chunk_index)
            results.append({
                "book_id": book_id,
                "title": title,
                "author": author,
                "chunk_index": chunk_index,
                "text": text,
                "score": float(dist),
            })
        return results

    def answer(self, query, top_k=5):
        """Answer a question about the book collection using RAG."""
        results = self.search(query, top_k=top_k)
        if not results:
            return {"answer": "No relevant content found in the book collection.", "citations": []}

        # Build context from top chunks
        context_parts = []
        citations = []
        seen_books = set()
        for i, r in enumerate(results):
            header = f"[Source {i+1}] {r['title']} by {r['author']}"
            context_parts.append(f"{header}\n{r['text']}")
            key = (r["title"], r["author"])
            if key not in seen_books:
                seen_books.add(key)
                citations.append({"title": r["title"], "author": r["author"]})

        context = "\n\n".join(context_parts)

        prompt = f"""<|im_start|>system
You are a scholarly research assistant specialized in book analysis. Answer the user's question based ONLY on the provided book excerpts. Cite the specific books you reference. If the excerpts don't contain enough information to answer, say so.
<|im_end|>
<|im_start|>user
Context from books:
{context}

Question: {query}

Please provide a detailed answer with specific references to the books and authors cited.
<|im_end|>
<|im_start|>assistant"""

        output = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.3,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
        )

        answer = output["choices"][0]["text"].strip()

        return {
            "answer": answer,
            "citations": citations,
            "sources": [
                {
                    "title": r["title"],
                    "author": r["author"],
                    "relevance": round(r["score"], 4),
                }
                for r in results[:3]
            ],
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python rag/oracle.py <query>")
        print("       python rag/oracle.py --interactive")
        sys.exit(1)

    oracle = BookOracle()

    if sys.argv[1] == "--interactive":
        print("\n📚 Book Oracle — interactive mode (type 'quit' to exit)\n")
        while True:
            try:
                q = input("Ask > ").strip()
                if not q or q.lower() in ("quit", "exit", "q"):
                    break
                result = oracle.answer(q)
                print(f"\n{result['answer']}\n")
                print("📖 Sources:")
                for s in result["sources"]:
                    print(f"  • {s['title']} (relevance: {s['relevance']})")
                print()
            except (KeyboardInterrupt, EOFError):
                break
    else:
        query = " ".join(sys.argv[1:])
        result = oracle.answer(query)
        print(result["answer"])
        print()
        print("📖 Sources:")
        for s in result["sources"]:
            print(f"  • {s['title']} (relevance: {s['relevance']})")


if __name__ == "__main__":
    main()
