"""Full RAG chain — query → retrieve → fuse → generate → cite."""

from rag.retriever import retrieve
from rag.generator import generate_answer
from rag.llm import generate, is_running


def answer_query(query: str, use_llm: bool = True) -> dict:
    """Full RAG chain: retrieve contexts, generate answer with citations."""
    contexts = retrieve(query, top_k=10)

    if not contexts:
        return {
            "answer": "No relevant books found in your library for this query.",
            "citations": []
        }

    if use_llm and is_running():
        result = generate_answer(query, contexts, llm_func=generate)
    else:
        result = generate_answer(query, contexts, llm_func=None)

    return result
