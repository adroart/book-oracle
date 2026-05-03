"""Embeddings — sentence-transformers + FAISS index for semantic search.
Splits book texts into overlapping chunks, generates embeddings, stores index."""

import os
import sys
import sqlite3
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "books.db")
EMBEDDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "embeddings")
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "books.index")

# Chunking parameters
CHUNK_SIZE = 512  # tokens (approximated as words)
OVERLAP = 0.10     # 10% overlap


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks by word count."""
    words = text.split()
    step = int(chunk_size * (1 - overlap))
    if step < 1:
        step = 1

    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        chunk_text_str = " ".join(chunk_words)
        # Calculate char positions (approximate)
        start_char = len(" ".join(words[:i])) if i > 0 else 0
        # Reconstruct to get exact positions
        chunk_join = " ".join(chunk_words)
        end_char = start_char + len(chunk_join)
        chunks.append({
            "text": chunk_join,
            "start_char": start_char,
            "end_char": end_char,
            "chunk_index": len(chunks)
        })

        if len(chunks) > 50000:  # Safety limit
            break

    return chunks


def init_chunks_table(conn):
    """Create book_chunks table if it doesn't exist."""
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS book_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_char INTEGER DEFAULT 0,
            end_char INTEGER DEFAULT 0,
            FOREIGN KEY(book_id) REFERENCES extracted_books(id)
        );
        CREATE INDEX IF NOT EXISTS idx_book_chunks_book ON book_chunks(book_id);
    """)
    conn.commit()


def run_embeddings(limit=None):
    """Generate embeddings for all books and build FAISS index."""
    import faiss
    from sentence_transformers import SentenceTransformer

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    print("[embeddings] Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    conn = sqlite3.connect(DB_PATH)
    init_chunks_table(conn)
    c = conn.cursor()

    # Clear existing chunks
    c.execute("DELETE FROM book_chunks")

    # Get books
    c.execute("SELECT id, title, text FROM extracted_books ORDER BY id")
    if limit:
        books = c.fetchmany(limit)
    else:
        books = c.fetchall()

    all_chunks = []
    all_embeddings = []

    for book_id, title, text in books:
        if not text or len(text.strip()) < 100:
            print(f"[embeddings] Skipping book {book_id} ({title}) — no text")
            continue

        print(f"[embeddings] Chunking book {book_id}: {title} ({len(text)} chars)...")
        chunks = chunk_text(text)
        print(f"  Generated {len(chunks)} chunks")

        # Store chunk metadata in DB
        for chunk in chunks:
            c.execute(
                "INSERT INTO book_chunks (book_id, chunk_text, chunk_index, start_char, end_char) VALUES (?, ?, ?, ?, ?)",
                (book_id, chunk["text"], chunk["chunk_index"], chunk["start_char"], chunk["end_char"])
            )
        conn.commit()

        # Generate embeddings
        print(f"  Generating embeddings for {len(chunks)} chunks...")
        texts = [ch["text"] for chunk in chunks for ch in [chunk]]
        # Actually just use chunk.text
        texts = [ch["text"] for ch in chunks]

        # Process in batches to avoid OOM
        batch_size = 64
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_emb = model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_emb)
            print(f"  Encoded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        if embeddings:
            book_embeddings = np.vstack(embeddings)
            all_embeddings.append(book_embeddings)
            all_chunks.extend([(book_id, ch["chunk_index"]) for ch in chunks])

        print(f"  Book done. Embedding shape: {book_embeddings.shape}")

    if not all_embeddings:
        print("[embeddings] No embeddings generated!")
        conn.close()
        return 0

    # Concatenate all embeddings
    all_embeddings_np = np.vstack(all_embeddings)
    dim = all_embeddings_np.shape[1]
    print(f"[embeddings] Total embeddings: {all_embeddings_np.shape}")

    # Build and save FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(all_embeddings_np.astype(np.float32))
    faiss.write_index(index, INDEX_PATH)
    print(f"[embeddings] FAISS index saved to {INDEX_PATH}")

    # Store chunk_id mapping for retrieval
    # We store book_id + chunk_index in the same order as embeddings
    mapping_path = os.path.join(EMBEDDINGS_DIR, "chunk_mapping.csv")
    with open(mapping_path, "w") as f:
        f.write("emb_index,book_id,chunk_index\n")
        for i, (bid, cidx) in enumerate(all_chunks):
            f.write(f"{i},{bid},{cidx}\n")
    print(f"[embeddings] Chunk mapping saved to {mapping_path}")

    conn.close()
    print(f"[embeddings] Done. Index contains {index.ntotal} vectors")
    return index.ntotal


def search_similar(query, k=5):
    """Search for similar chunks using FAISS."""
    import faiss
    from sentence_transformers import SentenceTransformer

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(INDEX_PATH)

    # Encode query
    query_vec = model.encode([query]).astype(np.float32)

    # Search
    distances, indices = index.search(query_vec, k)

    # Load mapping
    mapping = []
    mapping_path = os.path.join(EMBEDDINGS_DIR, "chunk_mapping.csv")
    with open(mapping_path, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                mapping.append((int(parts[1]), int(parts[2])))

    # Load chunk text from DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    results = []
    for i, idx_val in enumerate(indices[0]):
        if idx_val < 0 or idx_val >= len(mapping):
            continue
        book_id, chunk_index = mapping[idx_val]

        c.execute(
            "SELECT chunk_text FROM book_chunks WHERE book_id=? AND chunk_index=?",
            (book_id, chunk_index)
        )
        row = c.fetchone()
        c.execute("SELECT title, author FROM extracted_books WHERE id=?", (book_id,))
        book_info = c.fetchone()

        if row and book_info:
            results.append({
                "book_id": book_id,
                "title": book_info[0],
                "author": book_info[1],
                "chunk_index": chunk_index,
                "text": row[0][:500],
                "score": float(distances[0][i]),
            })

    conn.close()
    return results


if __name__ == "__main__":
    n = run_embeddings()
    if n:
        print(f"\n=== Verification ===")
        results = search_similar("jitterbug perfume plot")
        for r in results:
            print(f"  [{r['title']}] score={r['score']:.2f}: {r['text'][:100]}...")
