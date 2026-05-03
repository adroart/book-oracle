"""FastAPI server — Research Engine.
Single shared RAG instance. Persistent model. Deep research interface."""

import os
import sys
import sqlite3
import threading
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import meilisearch

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "books.db")
app = FastAPI(title="Research Engine")
meili = meilisearch.Client("http://localhost:7700", "")

# ── Shared RAG Engine (load once, serve many) ──────────────────────
_rag_engine = None
_rag_lock = threading.Lock()

def get_rag():
    global _rag_engine
    if _rag_engine is None:
        with _rag_lock:
            if _rag_engine is None:
                from rag.oracle import BookOracle
                _rag_engine = BookOracle()
    return _rag_engine

# ── HTML ─────────────────────────────────────────────────────────────
STYLES = """<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#c8c8d4;font-family:system-ui,-apple-system,sans-serif;line-height:1.6}
nav{background:#111118;padding:14px 24px;display:flex;gap:24px;align-items:center;border-bottom:1px solid #1e1e2a;position:sticky;top:0;z-index:100}
nav a{color:#7878a0;text-decoration:none;font-size:.9em;letter-spacing:.5px}
nav a:hover{color:#b0b0d0}
.logo{font-size:1.1em;color:#c0c0e8!important;font-weight:600;letter-spacing:0}
.logo span{color:#6666aa}
main{max-width:900px;margin:0 auto;padding:32px 24px}
h1{font-size:1.5em;font-weight:600;color:#d0d0e8;margin-bottom:8px}
h2{font-size:1.1em;font-weight:500;color:#8888b0;margin-bottom:20px}
.sub{color:#686888;font-size:.9em;margin-bottom:24px}
.card{background:#111118;border:1px solid #1e1e2a;border-radius:10px;padding:20px;margin-bottom:16px}
.card h3{color:#b0b0d0;font-size:.95em;margin-bottom:8px;font-weight:500}
.card .val{font-size:1.8em;font-weight:600;color:#d0d0ff}
.card .unit{color:#686888;font-size:.8em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:28px}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:28px}
.btn{background:#181825;color:#b8b8d8;padding:9px 20px;border-radius:6px;text-decoration:none;font-size:.85em;border:1px solid #2a2a3a;transition:all .15s}
.btn:hover{background:#222238;border-color:#3a3a5a;color:#d0d0f0}
.search-wrap{display:flex;gap:8px;margin-bottom:20px}
.search-wrap input{flex:1;padding:12px 16px;border-radius:6px;border:1px solid #2a2a3a;background:#0f0f18;color:#d0d0e0;font-size:.95em}
.search-wrap input:focus{outline:none;border-color:#4848aa}
.search-wrap button{padding:12px 24px;border-radius:6px;border:none;background:#2a2a55;color:#d0d0f0;font-size:.85em;font-weight:500;cursor:pointer}
.search-wrap button:hover{background:#3a3a77}
.result{background:#0f0f18;border:1px solid #1a1a28;border-radius:8px;padding:16px;margin-bottom:10px}
.result h3{color:#a0a0d0;margin-bottom:4px}
.result .meta{color:#686888;font-size:.8em;margin-bottom:6px}
.result .snippet{color:#aaaabc;font-size:.9em;line-height:1.5}
.result .snippet em{color:#c0c0ff;font-style:normal;background:#181830;padding:0 3px;border-radius:2px}
#chat-log{min-height:200px;margin-bottom:16px}
.msg{padding:16px;border-radius:8px;margin-bottom:12px;line-height:1.6;font-size:.92em}
.msg-q{background:#12122a;border:1px solid #2a2a50;color:#d0d0e8}
.msg-a{background:#111118;border:1px solid #1e1e2a;color:#c8c8d4}
.msg .sources{font-size:.8em;color:#6868a0;margin-top:10px;padding-top:10px;border-top:1px solid #1e1e28}
.msg .sources span{color:#8888aa;cursor:pointer;text-decoration:underline}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid #3a3a5a;border-top-color:#8888cc;border-radius:50%;animation:spin .8s linear infinite;margin-right:8px}
@keyframes spin{to{transform:rotate(360deg)}
}
.thinking{color:#6868a0;font-size:.85em;display:flex;align-items:center;gap:6px;padding:12px;background:#0f0f18;border-radius:8px;margin-bottom:12px}
.book-row{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #1a1a24;font-size:.9em}
.book-row:last-child{border:none}
.book-row .title{color:#b0b0d0}
.book-row .size{color:#686888;font-size:.8em}
</style>"""

def page(title, body, nav_active=""):
    nav_links = [
        ('Dashboard', '/'),
        ('Research', '/research'),
        ('Browse', '/books'),
        ('Search', '/search'),
    ]
    nav_html = "".join(f'<a href="{u}"{" style=color:#c0c0e0" if l==nav_active else ""}>● {l}</a>' for l,u in nav_links)
    return HTMLResponse(f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{'Research Engine — ' + nav_active if nav_active != 'Dashboard' else ''}Research Engine</title>
{STYLES}</head><body>
<nav><a href="/" class="logo">⧩ <span>Research</span> Engine</a>{nav_html}</nav>
<main>{body}</main></body></html>""")

# ── DB ────────────────────────────────────────────────────────────────
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

# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    s = get_stats()
    body = f"""
<div class="grid">
<div class="card"><h3>Books</h3><div class="val">{s['total_books']}</div></div>
<div class="card"><h3>Characters</h3><div class="val">{s['total_chars']:,}</div></div>
<div class="card"><h3>Chunks</h3><div class="val">{s['total_chunks']:,}</div></div>
<div class="card"><h3>Entities</h3><div class="val">{s['total_entities']:,}</div></div>
</div>
<div class="actions">
<a href="/research" class="btn">Start Research</a>
<a href="/search" class="btn">Full-Text Search</a>
<a href="/books" class="btn">Browse Collection</a>
</div>
<p class="sub">Tom Robbins collection · 10 books · FAISS + Qwen2.5 RAG</p>"""
    return page("Dashboard", body, "Dashboard")

@app.get("/research", response_class=HTMLResponse)
async def research():
    body = """<h1>Deep Research</h1>
<p class="sub">Ask questions across the full collection. Results cite specific sources.</p>

<div id="chat-log"></div>

<div class="thinking" id="thinking" style="display:none">
<div class="spinner"></div><span>Retrieving and analyzing...</span>
</div>

<div class="search-wrap">
<input type="text" id="query" placeholder='e.g., "What are the untraditional beliefs explored in Tom Robbins' work?"' autofocus>
<button onclick="ask()">Research →</button>
</div>

<script>
const q = document.getElementById('query');
q.addEventListener('keypress', e => { if(e.key === 'Enter') ask() });

async function ask() {
    const query = q.value.trim();
    if (!query) return;
    const log = document.getElementById('chat-log');
    const thinking = document.getElementById('thinking');

    log.innerHTML += '<div class="msg msg-q">' + escapeHtml(query) + '</div>';
    q.value = '';
    thinking.style.display = 'flex';

    try {
        const r = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query})
        });
        const d = await r.json();
        thinking.style.display = 'none';

        let html = '<div class="msg msg-a">' + escapeHtml(d.answer || d.error || 'No response');

        if (d.sources && d.sources.length) {
            html += '<div class="sources"><strong>Sources:</strong><br>';
            d.sources.forEach(s => {
                html += '<span onclick="searchSource(\\'' + s.title.replace(/'/g, "\\'") + '\\')">' + escapeHtml(s.title) + '</span> (' + s.relevance + ')<br>';
            });
            html += '</div>';
        }
        if (d.citations && d.citations.length) {
            html += '<div class="sources"><strong>Cited Books:</strong> ';
            html += d.citations.map(c => escapeHtml(c.title)).join(', ');
            html += '</div>';
        }
        html += '</div>';
        log.innerHTML += html;
        log.scrollTop = log.scrollHeight;
    } catch(e) {
        thinking.style.display = 'none';
        log.innerHTML += '<div class="msg msg-a">Error: ' + e.message + '</div>';
    }
}

function searchSource(title) {
    window.location.href = '/search?q="' + encodeURIComponent(title) + '"';
}

function escapeHtml(s) {
    if (!s) return '';
    return s.replace(/[&<>]/g, function(c) {
        return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c] || c;
    }).replace(/\\n/g, '<br>');
}
</script>"""
    return page("Research", body, "Research")

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
                txt = hit.get("_formatted", {}).get("text", hit.get("text", ""))[:400]
                results_html += f'<div class="result"><h3>{title}</h3><div class="meta">{author}</div><div class="snippet">{txt}...</div></div>'
            conn.close()
        except Exception as e:
            results_html = f'<div class="result"><p style="color:#887766">Search error: {e}</p></div>'

    body = f"""<h1>Full-Text Search</h1>
<p class="sub">Meilisearch-powered — searches every word in the collection.</p>
<div class="search-wrap">
<form style="display:contents;flex:1">
<input type="text" name="q" class="search-input" placeholder="Search words or phrases..." value="{q}" autofocus style="flex:1">
</form>
</div>
{results_html}"""
    return page("Search", body, "Search")

@app.get("/books", response_class=HTMLResponse)
async def books():
    conn = get_db()
    rows = conn.execute("SELECT id, title, author, text_chars FROM extracted_books ORDER BY id").fetchall()
    conn.close()
    books_html = ""
    for r in rows:
        books_html += f'<div class="book-row"><span class="title"><a href="/books/{r["id"]}" style="color:#a0a0d0;text-decoration:none">{r["title"] or "Untitled"}</a></span><span class="size">{r["author"] or "—"} · {r["text_chars"]:,} chars</span></div>'
    body = f"""<h1>Collection</h1><p class="sub">All books in the research library.</p>
{books_html}"""
    return page("Browse", body, "Browse")

@app.get("/books/{book_id}", response_class=HTMLResponse)
async def book_detail(book_id: int):
    conn = get_db()
    row = conn.execute("SELECT id, title, author, filename, text_chars, substr(text,1,4000) as preview FROM extracted_books WHERE id=?", (book_id,)).fetchone()
    conn.close()
    if not row:
        return HTMLResponse("Not found", 404)
    body = f"""<h1>{row['title'] or row['filename']}</h1>
<p class="sub">{row['author'] or 'Unknown'} · {row['text_chars']:,} chars</p>
<pre style="background:#0f0f18;border:1px solid #1a1a28;border-radius:8px;padding:16px;overflow-x:auto;font-size:.85em;white-space:pre-wrap;line-height:1.5">{row['preview']}</pre>"""
    return page("Book Detail", body)

# ── API ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_api(req: ChatRequest):
    try:
        oracle = get_rag()
        result = oracle.answer(req.query)
        return result
    except Exception as e:
        import traceback
        return {"answer": f"Research error: {str(e)}", "citations": [], "traceback": traceback.format_exc()}

@app.get("/api/stats")
async def api_stats():
    return get_stats()

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "research-engine", "model_loaded": _rag_engine is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
