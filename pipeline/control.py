"""SQLite control plane — database schema, CRUD, status tracking."""

import sqlite3
import os
import time
from pathlib import Path

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "library.db")


def init_db(db_path=None):
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            format TEXT NOT NULL,
            filename TEXT NOT NULL,
            size_bytes INTEGER DEFAULT 0,
            title TEXT DEFAULT '',
            author TEXT DEFAULT '',
            isbn TEXT DEFAULT '',
            language TEXT DEFAULT '',
            text_extracted INTEGER DEFAULT 0,
            text_path TEXT DEFAULT '',
            text_chars INTEGER DEFAULT 0,
            indexed INTEGER DEFAULT 0,
            ner_done INTEGER DEFAULT 0,
            embedded INTEGER DEFAULT 0,
            error_log TEXT DEFAULT '',
            last_updated TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER NOT NULL,
            entity_text TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            context_snippet TEXT DEFAULT '',
            char_offset INTEGER DEFAULT 0,
            FOREIGN KEY(book_id) REFERENCES books(id)
        );
        CREATE TABLE IF NOT EXISTS entity_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            book_id INTEGER NOT NULL,
            count INTEGER DEFAULT 1,
            FOREIGN KEY(entity_id) REFERENCES entities(id),
            FOREIGN KEY(book_id) REFERENCES books(id)
        );
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER NOT NULL,
            chapter_idx INTEGER DEFAULT 0,
            start_char INTEGER DEFAULT 0,
            end_char INTEGER DEFAULT 0,
            FOREIGN KEY(book_id) REFERENCES books(id)
        );
        CREATE TABLE IF NOT EXISTS concepts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER DEFAULT NULL,
            label TEXT DEFAULT '',
            centroid TEXT DEFAULT '',
            member_count INTEGER DEFAULT 0,
            level INTEGER DEFAULT 0,
            FOREIGN KEY(parent_id) REFERENCES concepts(id)
        );
        CREATE INDEX IF NOT EXISTS idx_books_path ON books(path);
        CREATE INDEX IF NOT EXISTS idx_books_text_extracted ON books(text_extracted);
        CREATE INDEX IF NOT EXISTS idx_books_indexed ON books(indexed);
        CREATE INDEX IF NOT EXISTS idx_entities_book ON entities(book_id);
        CREATE INDEX IF NOT EXISTS idx_entities_text ON entities(entity_text);
        CREATE INDEX IF NOT EXISTS idx_segments_book ON segments(book_id);
    """)
    conn.commit()
    return conn, path


def register_books(folder_path, conn=None):
    """Walk a folder and register all supported book files in the DB."""
    close = False
    if conn is None:
        conn, _ = init_db()
        close = True
    try:
        supported = {'.pdf', '.epub', '.mobi', '.txt'}
        books = []
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in supported:
                    full_path = os.path.join(root, f)
                    size = os.path.getsize(full_path)
                    books.append((full_path, ext[1:], f, size))
        c = conn.cursor()
        registered = 0
        for path, fmt, filename, size in books:
            try:
                c.execute(
                    "INSERT OR IGNORE INTO books (path, format, filename, size_bytes) VALUES (?, ?, ?, ?)",
                    (path, fmt, filename, size)
                )
                if c.rowcount > 0:
                    registered += 1
            except Exception:
                pass
        conn.commit()
        return {"registered": registered, "total_files": len(books)}
    finally:
        if close:
            conn.close()


def get_pending(conn=None, stage="extract", limit=100):
    """Get books pending a given stage."""
    close = False
    if conn is None:
        conn, _ = init_db()
        close = True
    try:
        c = conn.cursor()
        stage_map = {
            "extract": "text_extracted",
            "index": "indexed",
            "ner": "ner_done",
            "embed": "embedded",
        }
        col = stage_map.get(stage, "text_extracted")
        c.execute(f"SELECT id, path, format, filename FROM books WHERE {col}=0 ORDER BY id LIMIT ?", (limit,))
        rows = [{"id": r[0], "path": r[1], "format": r[2], "filename": r[3]} for r in c.fetchall()]
        return rows
    finally:
        if close:
            conn.close()


def mark_done(book_id, stage, conn=None):
    stage_map = {"extract": "text_extracted", "index": "indexed", "ner": "ner_done", "embed": "embedded"}
    col = stage_map.get(stage, "text_extracted")
    close = False
    if conn is None:
        conn, _ = init_db()
        close = True
    try:
        conn.execute(f"UPDATE books SET {col}=1, last_updated=datetime('now') WHERE id=?", (book_id,))
        conn.commit()
    finally:
        if close:
            conn.close()


def mark_error(book_id, stage, error, conn=None):
    stage_map = {"extract": "text_extracted", "index": "indexed", "ner": "ner_done", "embed": "embedded"}
    col = stage_map.get(stage, "extract")
    close = False
    if conn is None:
        conn, _ = init_db()
        close = True
    try:
        conn.execute(
            f"UPDATE books SET {col}=-1, error_log=?, last_updated=datetime('now') WHERE id=?",
            (str(error)[:500], book_id)
        )
        conn.commit()
    finally:
        if close:
            conn.close()


def get_stats(conn=None):
    close = False
    if conn is None:
        conn, _ = init_db()
        close = True
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM books")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM books WHERE text_extracted=1")
        extracted = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM books WHERE indexed=1")
        indexed = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM books WHERE ner_done=1")
        ner_done = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM books WHERE embedded=1")
        embedded = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM books WHERE text_extracted=-1")
        errors = c.fetchone()[0]
        return {
            "total": total, "extracted": extracted, "indexed": indexed,
            "ner_done": ner_done, "embedded": embedded, "errors": errors
        }
    finally:
        if close:
            conn.close()


def get_books(conn=None, offset=0, limit=50):
    close = False
    if conn is None:
        conn, _ = init_db()
        close = True
    try:
        c = conn.cursor()
        c.execute("SELECT id, path, format, filename, size_bytes, title, author, text_chars, text_extracted, error_log FROM books ORDER BY id LIMIT ? OFFSET ?", (limit, offset))
        return [{"id": r[0], "path": r[1], "format": r[2], "filename": r[3], "size_bytes": r[4], "title": r[5], "author": r[6], "text_chars": r[7], "text_extracted": r[8], "error_log": r[9]} for r in c.fetchall()]
    finally:
        if close:
            conn.close()
