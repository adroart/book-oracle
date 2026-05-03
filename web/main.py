"""FastAPI server — Book Oracle web UI.
Inline HTML only (avoids Jinja2 cache bug with starlette 1.0).
"""

import os
import sys
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import meilisearch

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "books.db")

app = FastAPI(title="Book Oracle")
meili = meilisearch.Client("http://localhost:7700", "")


HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Book Oracle</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f0f14;color:#e0e0e8;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.5}
nav{background:#1a1a24;padding:12px 24px;display:flex;gap:20px;align-items:center;border-bottom:1px solid #2a2a3a}
nav a{color:#8888cc;text-decoration:none;font-weight:500}
nav a:hover{color:#aaaaff}
.logo{font-size:1.2em;color:#d4d4ff!important}
main{max-width:960px;margin:0 auto;padding:24px}
h1{font-size:1.6em;margin-bottom:20px;color:#c8c8f0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:16px;margin-bottom:28px}
.stat-card{background:#1a1a24;border-radius:12px;padding:20px;text-align:center;border:1px solid #2a2a3a}
.stat-num{font-size:2em;font-weight:700;color:#a0a0ff;display:block}
.stat-label{font-size:.85em;color:#8888aa;margin-top:4px}
.actions{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px}
.btn{background:#2a2a40;color:#c0c0e0;padding:10px 24px;border-radius:8px;text-decoration:none;font-weight:500;border:1px solid #3a3a50}
.btn:hover{background:#3a3a50}
.book-table{width:100%;border-collapse:collapse}
.book-table th,.book-table td{padding:10px 14px;text-align:left;border-bottom:1px solid #2a2a3a}
.book-table th{color:#8888aa;font-size:.85em;text-transform:uppercase}
.search-box{width:100%;padding:12px 16px;border-radius:8px;border:1px solid #3a3a50;background:#1a1a24;color:#e0e0e8;font-size:1em;margin-bottom:16px}
.search-box:focus{outline:none;border-color:#6666aa}
.result-item{padding:14px;margin-bottom:8px;background:#1a1a24;border-radius:8px;border:1px solid #2a2a3a}
.result-item h3{color:#a0a0ff;margin-bottom:4px}
.result-item .meta{color:#8888aa;font-size:.85em}
.hit-text{color:#c0c0d0;font-size:.9em;margin-top:6px}
em{color:#b0b0ff;font-style:normal;background:#2a2a4a;padding:0 2px}
</style></head><body>
<nav><a href="/" class="logo">📚 Book Oracle</a>
<a href="/">Dashboard</a><a href="/search">Search</a><a href="/books">Browse</a><a href="/chat">Chat</a></nav>
<main>"""

HTML_FOOT = "</main></body></html>"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_stats():
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*), COALESCE(SUM(text_chars), 0) FROM extracted_books")
        total_books, total_chars = c.fetchone()
        c.execute("SELECT COUNT(*) FROM entity_mentions")
        total_entities = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM book_chunks")
        total_chunks = c.fetchone()[0]
    except Exception:
        total_books = total_chars = total_entities = total_chunks = 0
    conn.close()
    return {"total_books": total_books, "total_chars": total_chars,
            "total_entities": total_entities, "total_chunks": total_chunks}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    s = get_stats()
    html = HTML_HEAD + f"""
<h1>📚 Book Oracle</h1>
<div class="stats">
<div class="stat-card"><span class="stat-num">{s['total_books']}</span><span class="stat-label">Books</span></div>
<div class="stat-card"><span class="stat-num">{s['total_chars']:,}</span><span class="stat-label">Characters</span></div>
<div class="stat-card"><span class="stat-num">{s['total_chunks']:,}</span><span class="stat-label">Chunks</span></div>
<div class="stat-card"><span class="stat-num">{s['total_entities']:,}</span><span class="stat-label">Entities</span></div>
</div>
<div class="actions">
<a href="/search" class="btn">🔍 Search Books</a>
<a href="/books" class="btn">📖 Browse</a>
<a href="/chat" class="btn">💬 Chat with Oracle</a>
</div>
<p>10 Tom Robbins books loaded — extracted, indexed, embedded, and RAG-ready.</p>
""" + HTML_FOOT
    return HTMLResponse(html)


@app.get("/search", response_class=HTMLResponse)
async def search(q: str = ""):
    results_html = ""
    if q.strip():
        try:
            result = meili.index("books").search(q, {"limit": 20, "attributesToHighlight": ["text"]})
            hits = result.get("hits", [])
            conn = get_db()
            for hit in hits:
                bid = hit.get("book_id", hit.get("id"))
                row = conn.execute("SELECT title, author FROM extracted_books WHERE id=?", (bid,)).fetchone()
                title = hit.get("title", row["title"] if row else "Unknown")
                author = hit.get("author", row["author"] if row else "Unknown")
                txt = hit.get("_formatted", {}).get("text", hit.get("text", ""))[:300]
                results_html += f'<div class="result-item"><h3>{title}</h3><div class="meta">{author}</div><div class="hit-text">{txt}...</div></div>'
            conn.close()
        except Exception as e:
            results_html = f'<p>Search error: {e}</p>'

    html = HTML_HEAD + f"""
<h1>🔍 Search Books</h1>
<form><input type="text" name="q" class="search-box" placeholder="Search full text..." value="{q}" autofocus></form>
{results_html}
""" + HTML_FOOT
    return HTMLResponse(html)


@app.get("/books", response_class=HTMLResponse)
async def books():
    conn = get_db()
    rows = conn.execute("SELECT id, title, author, text_chars FROM extracted_books ORDER BY id").fetchall()
    conn.close()
    rows_html = ""
    for r in rows:
        rows_html += f'<tr><td><a href="/books/{r["id"]}" style="color:#a0a0ff">{r["title"] or r["id"]}</a></td><td>{r["author"] or "—"}</td><td>{r["text_chars"]:,} chars</td></tr>'
    html = HTML_HEAD + f"""
<h1>📖 Books</h1>
<table class="book-table"><thead><tr><th>Title</th><th>Author</th><th>Size</th></tr></thead><tbody>
{rows_html}</tbody></table>
""" + HTML_FOOT
    return HTMLResponse(html)


@app.get("/books/{book_id}", response_class=HTMLResponse)
async def book_detail(book_id: int):
    conn = get_db()
    row = conn.execute("SELECT id, title, author, filename, text_chars, substr(text,1,3000) as preview FROM extracted_books WHERE id=?", (book_id,)).fetchone()
    conn.close()
    if not row:
        return HTMLResponse("Book not found", 404)
    html = HTML_HEAD + f"""
<h1>{row['title'] or row['filename']}</h1>
<p><strong>Author:</strong> {row['author'] or 'Unknown'} | <strong>Size:</strong> {row['text_chars']:,} chars</p>
<pre style="background:#1a1a24;padding:16px;border-radius:8px;border:1px solid #2a2a3a;overflow-x:auto;font-size:.9em;white-space:pre-wrap;margin-top:16px">{row['preview']}</pre>
""" + HTML_FOOT
    return HTMLResponse(html)


@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    html = HTML_HEAD + """
<h1>💬 Chat with the Oracle</h1>
<div style="background:#1a1a24;border-radius:12px;border:1px solid #2a2a3a;padding:20px;margin-bottom:16px;min-height:200px" id="response">
<p style="color:#8888aa">Ask anything about your book collection...</p>
</div>
<div style="display:flex;gap:8px">
<input type="text" id="query" class="search-box" placeholder="e.g., What is Jitterbug Perfume about?" style="flex:1">
<button class="btn" onclick="ask()">Ask</button>
</div>
<script>
async function ask(){const q=document.getElementById('query').value;if(!q)return;
document.getElementById('response').innerHTML='<p style="color:#8888aa">Thinking...</p>';
const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})});
const d=await r.json();
let html='<div style="margin-bottom:12px">'+d.answer+'</div>';
if(d.sources&&d.sources.length){html+='<div style="font-size:.85em;color:#8888aa"><strong>Sources:</strong><br>'+d.sources.map(s=>'• '+s.title+' (relevance: '+s.relevance+')').join('<br>')+'</div>'}
if(d.citations&&d.citations.length){html+='<div style="font-size:.85em;color:#8888aa;margin-top:8px"><strong>Books cited:</strong><br>'+d.citations.map(c=>'• '+c.title+' by '+c.author).join('<br>')+'</div>'}
document.getElementById('response').innerHTML=html;}
document.getElementById('query').addEventListener('keypress',e=>{if(e.key==='Enter')ask()});
</script>
""" + HTML_FOOT
    return HTMLResponse(html)


class ChatRequest(BaseModel):
    query: str


@app.post("/api/chat")
async def chat_api(req: ChatRequest):
    try:
        from rag.oracle import BookOracle
        oracle = BookOracle()
        result = oracle.answer(req.query)
        return result
    except Exception as e:
        import traceback
        return {"answer": f"Error: {str(e)}", "citations": [], "traceback": traceback.format_exc()}


@app.get("/api/stats")
async def api_stats():
    return get_stats()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "book-oracle"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
