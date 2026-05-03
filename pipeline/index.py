"""Meilisearch indexing — push extracted text to search index."""

import os
import json
import time

try:
    import meilisearch
except ImportError:
    meilisearch = None

SEARCH_INDEX = "books"
MEILI_URL = "http://localhost:7700"


def get_client(url=None):
    if meilisearch is None:
        raise ImportError("meilisearch package not installed")
    return meilisearch.Client(url or MEILI_URL)


def setup_index(client=None):
    if client is None:
        client = get_client()
    try:
        client.create_index(SEARCH_INDEX, {"primaryKey": "id"})
    except Exception:
        pass
    try:
        index = client.index(SEARCH_INDEX)
        index.update_searchable_attributes(["title", "author", "text"])
        index.update_filterable_attributes(["format", "language", "author"])
    except Exception:
        pass
    return client.index(SEARCH_INDEX)


def push_books(batch, client=None):
    """Push a batch of books to Meilisearch.
    batch: list of dicts with id, title, author, text, format, language
    """
    if client is None:
        client = get_client()
    index = client.index(SEARCH_INDEX)
    docs = []
    for b in batch:
        doc = {
            "id": str(b["id"]),
            "title": b.get("title", b.get("filename", "Unknown")),
            "author": b.get("author", ""),
            "text": b.get("text", ""),
            "format": b.get("format", ""),
            "language": b.get("language", ""),
        }
        docs.append(doc)
    result = index.add_documents(docs)
    return result


def search(query, limit=20, client=None):
    if client is None:
        client = get_client()
    index = client.index(SEARCH_INDEX)
    return index.search(query, {"limit": limit})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = search(query)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python pipeline/index.py <search query>")
