"""Document ingestion: loading, preprocessing, chunking and FAISS index building.

This module is retrieval-only. It uses local embeddings
(sentence-transformers/all-MiniLM-L6-v2) and requires no API key or LLM.

Usage:
    python src/ingest.py            # build the index from data/college_docs/
    or from code: from ingest import build_index
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "college_docs"
INDEX_DIR = PROJECT_ROOT / "data" / "faiss_index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}

# LangChain recursive character chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ---------------------------------------------------------------------------
# 1. Document loading
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> List[Document]:
    """Load a PDF with pypdf (via LangChain's PyPDFLoader)."""
    docs = PyPDFLoader(str(path)).load()
    # PyPDFLoader sets metadata["page"]; normalise source for all pages.
    for d in docs:
        d.metadata["source"] = path.name
        d.metadata["file_type"] = "pdf"
    return docs


def _load_docx(path: Path) -> List[Document]:
    """Load a DOCX with python-docx (paragraphs + table cells)."""
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    parts: List[str] = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = "\n".join(parts)
    return [Document(page_content=text, metadata={"source": path.name, "file_type": "docx"})]


def _load_text(path: Path, loader_cls=TextLoader) -> List[Document]:
    """Load a plain-text file (.md / .txt) as UTF-8."""
    docs = loader_cls(str(path), encoding="utf-8").load()
    for d in docs:
        d.metadata["source"] = path.name
        d.metadata["file_type"] = path.suffix.lstrip(".").lower()
    return docs


def load_documents(docs_dir: Path | str = DOCS_DIR) -> List[Document]:
    """Load every supported file (PDF, DOCX, MD, TXT) from a directory.

    Returns a list of LangChain Documents, one per PDF page and one per
    other file, each with ``source`` and ``file_type`` metadata.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    documents: List[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if ext == ".pdf":
                loaded = _load_pdf(path)
            elif ext == ".docx":
                loaded = _load_docx(path)
            else:
                loaded = _load_text(path)
        except Exception as exc:  # keep ingestion resilient to one bad file
            print(f"  [warn] failed to load {path.name}: {exc}")
            continue
        documents.extend(loaded)
        print(f"  loaded {path.name} ({ext[1:]}) -> {len(loaded)} doc(s)")
    return documents


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

_MULTI_BLANK = re.compile(r"\n{3,}")


def preprocess(documents: List[Document]) -> List[Document]:
    """Clean text while keeping the ``source``/``file_type`` metadata.

    - Normalise unicode (NFKC) so smart quotes / ligatures don't fragment chunks.
    - Strip null bytes and zero-width characters.
    - Collapse runs of 3+ newlines (PDF extraction artefacts).
    - Drop documents that are empty after cleaning.
    """
    cleaned: List[Document] = []
    for doc in documents:
        text = doc.page_content
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\x00", "")
        text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
        text = _MULTI_BLANK.sub("\n\n", text).strip()
        if not text:
            continue
        doc.page_content = text
        cleaned.append(doc)
    return cleaned


# ---------------------------------------------------------------------------
# 3. Chunking (LangChain text splitters)
# ---------------------------------------------------------------------------

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks with LangChain's
    RecursiveCharacterTextSplitter. Source metadata is propagated to
    every chunk, plus a ``chunk`` index and ``doc_id``."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i
        chunk.metadata["doc_id"] = f"{chunk.metadata.get('source', '?')}#{chunk.metadata.get('page', 0)}"
    return chunks


# ---------------------------------------------------------------------------
# 5. build_index()
# ---------------------------------------------------------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the local embedding model (cached; runs fully offline)."""
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": False},
    )


def build_index(
    docs_dir: Path | str = DOCS_DIR,
    index_dir: Path | str = INDEX_DIR,
) -> FAISS:
    """Load -> preprocess -> chunk -> embed -> build FAISS index and save it.

    Returns the in-memory FAISS store (also persisted to ``index_dir`` so
    ``get_store()`` can reload it without re-embedding).
    """
    docs_dir = Path(docs_dir)
    index_dir = Path(index_dir)

    print(f"[ingest] loading documents from {docs_dir}")
    docs = load_documents(docs_dir)
    docs = preprocess(docs)
    chunks = chunk_documents(docs)
    if not chunks:
        raise RuntimeError(f"No usable content found in {docs_dir}")
    print(f"[ingest] {len(docs)} document(s) -> {len(chunks)} chunk(s) "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    embeddings = get_embeddings()
    print(f"[ingest] embedding with {MODEL_NAME}")
    store = FAISS.from_documents(chunks, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    print(f"[ingest] index saved to {index_dir}")
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FAISS index for college docs.")
    parser.add_argument("--docs", default=str(DOCS_DIR), help="Documents directory")
    parser.add_argument("--index", default=str(INDEX_DIR), help="Index output directory")
    args = parser.parse_args()
    build_index(args.docs, args.index)


if __name__ == "__main__":
    main()
