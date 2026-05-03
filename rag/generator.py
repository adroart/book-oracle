"""RAG generator — format prompt with context, cite sources."""

import json
import re


def build_prompt(query: str, contexts: list) -> str:
    """Build a prompt with retrieved context for the LLM."""
    sources = []
    for i, ctx in enumerate(contexts):
        title = ctx.get("title", "Unknown")
        author = ctx.get("author", "")
        text = ctx.get("text", "")[:1000]
        source_tag = f"[{i + 1}]"
        sources.append({
            "tag": source_tag,
            "title": title,
            "author": author,
            "id": ctx.get("id", ""),
        })
        ctx["_tag"] = source_tag

    context_str = "\n\n".join([
        f"{s['tag']} From \"{s['title']}\" by {s['author']}:\n{c.get('text', '')[:800]}"
        for s, c in zip(sources, contexts)
    ])

    prompt = f"""You are a research assistant answering questions based on a book library.
Answer concisely using the provided sources. Cite sources using [1], [2], etc.

Context:
{context_str}

Question: {query}

Answer:"""
    return prompt


def parse_citations(contexts: list) -> list:
    """Extract unique citation metadata."""
    seen = set()
    citations = []
    for ctx in contexts:
        cid = str(ctx.get("id", ""))
        if cid not in seen:
            seen.add(cid)
            citations.append({
                "book_id": cid,
                "title": ctx.get("title", "Unknown"),
                "author": ctx.get("author", ""),
            })
    return citations


def generate_answer(query: str, contexts: list, llm_func=None) -> dict:
    """Generate answer with citations from contexts."""
    prompt = build_prompt(query, contexts)
    citations = parse_citations(contexts)

    if llm_func:
        answer = llm_func(prompt)
    else:
        # Fallback: simple answer from context
        snippets = [c.get("text", "")[:200] for c in contexts[:3]]
        answer = f"Based on your library, here's what I found:\n\n"
        for s, ctx in zip(snippets, contexts[:3]):
            answer += f"- From \"{ctx.get('title', 'Unknown')}\": {s}...\n"

    return {"answer": answer, "citations": citations}
