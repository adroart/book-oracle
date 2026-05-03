"""Entity Graph — spaCy NER extraction from book texts.
Stores entity mentions in SQLite table `entity_mentions`."""

import sqlite3
import os
import sys
from collections import defaultdict
import spacy

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "books.db")


def init_entity_table(conn):
    """Create the entity_mentions table if it doesn't exist."""
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS entity_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER NOT NULL,
            entity_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            count INTEGER DEFAULT 1,
            sentences TEXT DEFAULT '',
            FOREIGN KEY(book_id) REFERENCES extracted_books(id)
        );
        CREATE INDEX IF NOT EXISTS idx_entity_mentions_book ON entity_mentions(book_id);
        CREATE INDEX IF NOT EXISTS idx_entity_mentions_name ON entity_mentions(entity_name);
    """)
    conn.commit()


def extract_entities_for_book(book_id, text, nlp):
    """Extract named entities from text, aggregate by name+type, collect sample sentences."""
    doc = nlp(text[:500000])  # Limit to 500K chars per book for speed
    entities = defaultdict(lambda: {"count": 0, "sentences": []})

    for ent in doc.ents:
        # Normalize entity name
        name = ent.text.strip()
        if not name or len(name) < 2:
            continue
        key = (name, ent.label_)
        entities[key]["count"] += 1

        # Collect a sample sentence (at most 3 per entity)
        if entities[key]["count"] <= 3:
            sent = ent.sent.text.strip() if ent.sent else ""
            entities[key]["sentences"].append(sent[:300])  # Truncate long sentences

    return entities


def run_entity_graph(limit=None):
    """Extract entities from all books and store in entity_mentions table."""
    print("[graph] Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    conn = sqlite3.connect(DB_PATH)
    init_entity_table(conn)
    c = conn.cursor()

    # Clear existing data
    c.execute("DELETE FROM entity_mentions")

    # Get books
    c.execute("SELECT id, title, text FROM extracted_books ORDER BY id")
    if limit:
        books = c.fetchmany(limit)
    else:
        books = c.fetchall()

    total_entities = 0
    for book_id, title, text in books:
        if not text or len(text.strip()) < 100:
            print(f"[graph] Skipping book {book_id} ({title}) — no text")
            continue

        print(f"[graph] Processing book {book_id}: {title} ({len(text)} chars)...")
        entities = extract_entities_for_book(book_id, text, nlp)
        print(f"  Found {len(entities)} unique entity types")

        # Insert into DB
        inserted = 0
        for (name, etype), data in entities.items():
            sentences_joined = " | ".join(data["sentences"])
            c.execute(
                "INSERT INTO entity_mentions (book_id, entity_name, entity_type, count, sentences) VALUES (?, ?, ?, ?, ?)",
                (book_id, name, etype, data["count"], sentences_joined)
            )
            inserted += 1
            total_entities += 1

        conn.commit()
        print(f"  Inserted {inserted} entity rows")

    conn.close()
    print(f"[graph] Done. Total entity rows: {total_entities}")
    return total_entities


def get_entity_summary(conn=None):
    """Return summary stats about entity mentions."""
    close = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        close = True
    try:
        c = conn.cursor()

        # Per-book entity counts
        c.execute("""
            SELECT eb.title, COUNT(DISTINCT em.entity_name) as uniq_entities, SUM(em.count) as total_mentions
            FROM entity_mentions em
            JOIN extracted_books eb ON em.book_id = eb.id
            GROUP BY em.book_id
            ORDER BY eb.id
        """)
        per_book = c.fetchall()

        # Top entities overall
        c.execute("""
            SELECT entity_name, entity_type, SUM(count) as total
            FROM entity_mentions
            GROUP BY entity_name, entity_type
            ORDER BY total DESC
            LIMIT 20
        """)
        top = c.fetchall()

        # Entity type distribution
        c.execute("""
            SELECT entity_type, COUNT(DISTINCT entity_name) as uniq, SUM(count) as total
            FROM entity_mentions
            GROUP BY entity_type
            ORDER BY total DESC
        """)
        types = c.fetchall()

        return {"per_book": per_book, "top_entities": top, "type_distribution": types}
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    run_entity_graph()
    conn = sqlite3.connect(DB_PATH)
    summary = get_entity_summary(conn)
    conn.close()
    print("\n=== Entity Summary ===")
    for title, uniq, total in summary["per_book"]:
        print(f"  {title}: {uniq} unique entities, {total} total mentions")
    print(f"\nTop Entities: {summary['top_entities'][:5]}")
    print(f"Entity Types: {len(summary['type_distribution'])} types found")
