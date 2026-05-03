"""FastAPI server — main app with all routes."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from pipeline import control as ctrl
from web.search import router as search_router

app = FastAPI(title="Book Oracle")

# Static files & templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Search API
app.include_router(search_router)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    conn, _ = ctrl.init_db()
    stats = ctrl.get_stats(conn)
    books = ctrl.get_books(conn, limit=20)
    conn.close()
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "stats": stats, "books": books
    })


@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, q: str = ""):
    results = []
    if q.strip():
        try:
            from pipeline.index import search
            result = search(q.strip())
            results = result.get("hits", [])
        except Exception as e:
            results = []
    return templates.TemplateResponse("search.html", {
        "request": request, "query": q, "results": results
    })


@app.get("/books", response_class=HTMLResponse)
async def browse_books(request: Request, offset: int = 0, limit: int = 50):
    conn, _ = ctrl.init_db()
    books = ctrl.get_books(conn, offset=offset, limit=limit + 1)
    conn.close()
    has_more = len(books) > limit
    if has_more:
        books = books[:limit]
    return templates.TemplateResponse("books.html", {
        "request": request, "books": books, "offset": offset, "limit": limit,
        "has_more": has_more
    })


@app.get("/books/{book_id}", response_class=HTMLResponse)
async def book_detail(request: Request, book_id: int):
    conn, _ = ctrl.init_db()
    c = conn.cursor()
    c.execute("SELECT * FROM books WHERE id=?", (book_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return HTMLResponse("Book not found", status_code=404)
    columns = [d[0] for d in c.description]
    book = dict(zip(columns, row))
    # Read extracted text if available
    text_preview = ""
    if book.get("text_extracted") == 1 and book.get("text_path"):
        try:
            with open(book["text_path"], "r", encoding="utf-8") as f:
                text_preview = f.read(2000)
        except Exception:
            pass
    return templates.TemplateResponse("book_detail.html", {
        "request": request, "book": book, "text_preview": text_preview
    })


@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_page(request: Request):
    conn, _ = ctrl.init_db()
    stats = ctrl.get_stats(conn)
    books = ctrl.get_books(conn, limit=50)
    conn.close()
    return templates.TemplateResponse("pipeline.html", {
        "request": request, "stats": stats, "books": books
    })


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


class ChatRequest(BaseModel):
    query: str


@app.post("/api/chat")
async def chat_api(req: ChatRequest):
    try:
        from rag.chain import answer_query
        result = answer_query(req.query)
        return result
    except ImportError as e:
        return {"answer": f"RAG engine not ready yet: {e}", "citations": []}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "citations": []}


@app.get("/api/stats")
async def api_stats():
    conn, _ = ctrl.init_db()
    stats = ctrl.get_stats(conn)
    conn.close()
    return stats


@app.get("/health")
async def health():
    return {"status": "ok", "service": "book-oracle"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
