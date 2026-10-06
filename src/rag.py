"""Retrieval-only RAG module: local embeddings + FAISS vector search.

No OpenAI / Groq / Gemini / any LLM or API key is required. Everything runs
locally with sentence-transformers/all-MiniLM-L6-v2.

Score semantics (verified against the installed langchain-community source,
``langchain_community/vectorstores/faiss.py``):
    * The default FAISS index is ``faiss.IndexFlatL2`` with
      ``DistanceStrategy.EUCLIDEAN_DISTANCE``.
    * ``similarity_search_with_score`` returns the **squared L2 distance**
      between the query and chunk embeddings: lower score = more similar.

Therefore MIN_SCORE is an **upper bound on the squared L2 distance**: chunks
scoring <= MIN_SCORE are kept, anything larger is discarded as irrelevant.

Public API used by the Streamlit app:
    retrieve_context(question, history=None) -> str   ("" if nothing relevant)

Lower-level API:
    retrieve(query) -> list of hit dicts
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS

import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # for `python -m src.rag`

from src import ingest  # noqa: F401

try:  # works both as `python src/rag.py` and as `python -m src.rag`
    from .ingest import DOCS_DIR, INDEX_DIR, build_index, get_embeddings
except ImportError:
    from ingest import DOCS_DIR, INDEX_DIR, build_index, get_embeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Maximum acceptable FAISS score (squared L2 distance, lower = more similar).
# Calibrated empirically on the test corpus by measuring raw scores:
#   relevant chunks for the 3 college queries: 0.22 - 1.30
#   off-topic college chunks:                 >= 1.42
#   "capital of Japan" (no relevant doc):     >= 1.78
# 1.40 sits in the observed gap. Re-calibrate if the corpus changes.
MIN_SCORE = 1.40

TOP_K = 4  # candidates fetched from FAISS before MIN_SCORE filtering
CONTEXT_K = 6  # candidates fetched when building context for the LLM

# --- Strict relevance gate (applied on top of MIN_SCORE) --------------------
# A candidate is kept only if it has meaningful keyword overlap with the
# question OR is a very strong semantic match. It never pads the result list:
# if only 2 chunks are relevant, only 2 are returned.
#   * STRONG_SEMANTIC_SCORE: on-topic top hits for the verified queries
#     measured 0.43-0.83; clearly unrelated chunks measured >= 1.29.
#   * MIN_KEYWORD_OVERLAP: at least half of the question's content words
#     must appear in the retrieved chunk.
STRONG_SEMANTIC_SCORE = 0.90
MIN_KEYWORD_OVERLAP = 0.5

_STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to for from by with as is are was were
    be been being do does did doing can could should would will shall may
    might must i me my we our you your he she it its this that these those
    what which who whom whose when where why how not no nor so than too very
    just about into over under please tell ask give show list any all
    """.split()
)


def _keywords(text: str) -> List[str]:
    """Lowercased content words of a question (stopwords and 1-char dropped)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) >= 2 and w not in _STOPWORDS]


def is_relevant(query: str, content: str, score: float) -> bool:
    """Strict relevance gate for one candidate hit.

    Keeps the hit if either:
      * it has meaningful keyword overlap with the question (at least
        ``MIN_KEYWORD_OVERLAP`` of the question's content words appear in
        the chunk), or
      * it is a very strong semantic match (``score <= STRONG_SEMANTIC_SCORE``).
    """
    if score <= STRONG_SEMANTIC_SCORE:
        return True  # very strong semantic match
    kws = _keywords(query)
    if not kws:
        return False
    content_words = set(re.findall(r"[a-z0-9]+", content.lower()))
    overlap = sum(1 for kw in kws if kw in content_words)
    return (overlap / len(kws)) >= MIN_KEYWORD_OVERLAP


# ---------------------------------------------------------------------------
# get_store()
# ---------------------------------------------------------------------------

_store: Optional[FAISS] = None


def get_store(index_dir: Path | str = INDEX_DIR, rebuild: bool = False) -> FAISS:
    """Return the FAISS vector store, loading it from disk on first use.

    If the index does not exist yet (or ``rebuild=True``), it is built from
    ``data/college_docs/`` first. The store is cached for the process lifetime.
    """
    global _store
    index_dir = Path(index_dir)

    if _store is not None and not rebuild:
        return _store

    if rebuild or not (index_dir / "index.faiss").exists():
        build_index(index_dir=index_dir)

    store = FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,  # index written by this project
    )
    _store = store
    return store


def reset_store() -> None:
    """Drop the cached store (mainly for tests)."""
    global _store
    _store = None


# ---------------------------------------------------------------------------
# retrieve() with MIN_SCORE filtering and source metadata
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int = TOP_K,
    min_score: float = MIN_SCORE,
    store: Optional[FAISS] = None,
) -> List[Dict[str, Any]]:
    """Retrieve the most relevant chunks for ``query``.

    Returns a list of hit dicts sorted best (lowest score) first::

        {
            "content": str,      # the chunk text
            "score": float,      # squared L2 distance; lower = better
            "source": str,       # e.g. "student_services.md"
            "file_type": str,    # "md" | "txt" | "pdf" | "docx"
            "chunk": int,        # chunk index within the corpus
            "doc_id": str,       # "source#page"
            "metadata": dict,    # full LangChain metadata
        }

    An empty list means nothing in the knowledge base was close enough.
    """
    query = (query or "").strip()
    if not query:
        return []

    vs = store if store is not None else get_store()
    pairs = vs.similarity_search_with_score(query, k=k)

    hits: List[Dict[str, Any]] = []
    for doc, score in pairs:
        if score > min_score:  # squared L2: keep only score <= min_score
            continue
        if not is_relevant(query, doc.page_content, score):
            continue  # strict gate: no keyword overlap and not a strong match
        meta = dict(doc.metadata)
        hits.append(
            {
                "content": doc.page_content,
                "score": float(score),
                "source": meta.get("source", "unknown"),
                "file_type": meta.get("file_type", ""),
                "chunk": meta.get("chunk", -1),
                "doc_id": meta.get("doc_id", ""),
                "metadata": meta,
            }
        )
    hits.sort(key=lambda h: h["score"])  # best first (lower distance)
    return hits


# ---------------------------------------------------------------------------
# retrieve_context(): the function app.py calls
# ---------------------------------------------------------------------------

def retrieve_context(question: str, history: Optional[list] = None) -> str:
    """Return retrieved chunks as one context string for the LLM.

    * Returns "" when nothing relevant is found, so the workflow can answer
      "not available in the knowledge base" instead of guessing.
    * Follow-ups: if the question is very short (e.g. "what about its fee?")
      and there is chat history, the previous user question is prepended to
      the search query so retrieval still finds the right topic.
    """
    query = question
    if history and len(_keywords(question)) <= 3:
        previous = [m["content"] for m in history if m.get("role") == "user"]
        if previous:
            query = f"{previous[-1]} {question}"

    hits = retrieve(query, k=CONTEXT_K)
    if not hits:
        return ""
    return "\n\n---\n\n".join(f"(Source: {h['source']})\n{h['content']}" for h in hits)


# ---------------------------------------------------------------------------
# Interactive CLI test:  python -m src.rag
# ---------------------------------------------------------------------------

def main() -> None:
    """Interactive command-line retrieval test (type exit / quit / q to stop)."""
    # cp1252 consoles cannot encode some corpus characters (e.g. U+20B9).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

    store = get_store()  # loads the existing index (builds it only if missing)
    print("RAG Retrieval Test")

    while True:
        try:
            query = input("Enter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        hits = retrieve(query, store=store)
        if not hits:
            print("No relevant information found.")
            continue

        for i, h in enumerate(hits, start=1):
            print(f"\nResult {i}")
            print(f"Source: {h['source']}")
            print(f"Score: {h['score']:.4f}")
            print(f"Content: {h['content']}")
        print()


if __name__ == "__main__":
    main()