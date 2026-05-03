"""Text extraction — PDF, EPUB, MOBI, TXT → plaintext."""

import os
import sys
import json
import tempfile
import subprocess
import multiprocessing
from pathlib import Path

EXTRACTED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "extracted")


def ensure_dir():
    os.makedirs(EXTRACTED_DIR, exist_ok=True)


def _extract_txt(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_pdf(path):
    try:
        import fitz  # pymupdf
    except ImportError:
        return "(ERROR: pymupdf not installed)"
    text_parts = []
    try:
        doc = fitz.open(path)
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"(ERROR extracting PDF: {e})"


def _extract_epub(path):
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        return "(ERROR: ebooklib not installed)"
    try:
        book = epub.read_epub(path)
        texts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            texts.append(soup.get_text(separator="\n"))
        return "\n\n".join(texts)
    except Exception as e:
        return f"(ERROR extracting EPUB: {e})"


def _extract_mobi(path):
    """Convert MOBI to text via calibre's ebook-convert CLI."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
            tmp_path = tmp.name
        subprocess.run(
            ["ebook-convert", path, tmp_path],
            capture_output=True, timeout=120
        )
        with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        os.unlink(tmp_path)
        return text
    except FileNotFoundError:
        return "(ERROR: calibre ebook-convert not installed)"
    except Exception as e:
        return f"(ERROR converting MOBI: {e})"


EXTRACTORS = {
    ".txt": _extract_txt,
    ".pdf": _extract_pdf,
    ".epub": _extract_epub,
    ".mobi": _extract_mobi,
}


def extract_one(book_id, path, fmt):
    """Extract a single book to plaintext. Returns (book_id, text_chars, error_or_None)."""
    ensure_dir()
    ext = f".{fmt}".lower()
    extractor = EXTRACTORS.get(ext)
    if not extractor:
        return book_id, 0, f"Unsupported format: {fmt}"

    text = extractor(path)
    if text.startswith("(ERROR"):
        return book_id, 0, text

    out_path = os.path.join(EXTRACTED_DIR, f"{book_id}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    return book_id, len(text), None


def extract_batch(books, workers=4):
    """Extract a list of books in parallel.
    books: list of (id, path, format) tuples.
    Returns: list of (book_id, text_chars, error_or_None)
    """
    with multiprocessing.Pool(workers) as pool:
        results = pool.starmap(extract_one, [(bid, p, f) for bid, p, f in books])
    return results


if __name__ == "__main__":
    # CLI usage: python pipeline/extract.py <book_id> <path> <format>
    if len(sys.argv) == 4:
        bid, path, fmt = int(sys.argv[1]), sys.argv[2], sys.argv[3]
        result = extract_one(bid, path, fmt)
        print(json.dumps({"book_id": result[0], "chars": result[1], "error": result[2]}))
    else:
        print("Usage: python pipeline/extract.py <book_id> <path> <format>")
