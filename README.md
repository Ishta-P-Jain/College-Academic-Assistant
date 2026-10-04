# Member 1 — Data & RAG Module (Retrieval Only)

Local, offline Retrieval-Augmented Generation **ingestion + retrieval** layer.
No OpenAI / Groq / Gemini / any LLM or API key is required — this module only
*retrieves* relevant college knowledge-base chunks; answer generation (if any)
belongs to another member's module.

## Features

1. **Document loading & preprocessing** — PDF, DOCX, MD, TXT; unicode
   normalisation, null/zero-width stripping, blank-run collapsing.
2. **Chunking with LangChain** — `RecursiveCharacterTextSplitter`
   (chunk_size=500, overlap=50).
3. **Local embeddings** — `sentence-transformers/all-MiniLM-L6-v2`
   (runs offline; model is cached locally).
4. **FAISS vector database** — `faiss-cpu`, `IndexFlatL2`, persisted to
   `data/faiss_index/`.
5. **`build_index()`** — load → preprocess → chunk → embed → save.
6. **`get_store()`** — lazily loads the persisted index (auto-builds if missing).
7. **`retrieve()`** — ranked chunks for a query.
8. **`MIN_SCORE` filtering** — drops chunks that are not close enough.
9. **Source metadata** — every hit carries `source`, `file_type`, `chunk`,
   `doc_id` and the full metadata dict.
10. **Simple retrieval test** — `src/test_retrieval.py` (46 assertions).

## Project layout

```
data/college_docs/     # NMAMIT knowledge base: 13 source files
                       #   NMAMIT_CSE_General_Information.pdf  (original)
                       #   NMAMIT_CSE_Faculty_Name_Designation_Joining_Date.docx (original)
                       #   11 supplied .md files (regulations, syllabus, calendar,
                       #   exams, attendance, internships, fees, services, rules,
                       #   FAQs, sources)
data/faiss_index/      # generated FAISS index (git-ignored; rebuild anytime)
src/ingest.py          # loading, preprocessing, chunking, build_index()
src/rag.py             # get_store(), retrieve(), MIN_SCORE, metadata
src/test_retrieval.py  # the retrieval test
requirements.txt
```

The loader supports **PDF** (pypdf), **DOCX** (python-docx), **Markdown** and
**TXT** (UTF-8). Files are indexed exactly as provided — no summaries,
no invented content.

## Install

```bash
pip install -r requirements.txt
```

All versions are pinned to what this environment was verified against.
The embedding model downloads once (or is used from the local HF cache) and
never contacts an LLM provider — **no API keys needed, ever.**

## Usage

```bash
# 1. Build the index (13 source files -> 20 documents -> 77 chunks)
python src/ingest.py

# 2. Run the retrieval test (exit code 0 = all passed)
python src/test_retrieval.py

# 3. Optional interactive prompt
python src/rag.py
```

From code:

```python
from ingest import build_index
from rag import retrieve, get_store

build_index()                      # (re)build from data/college_docs/
store = get_store()                # load persisted index (auto-builds if absent)

hits = retrieve("what are the library timings?")
for h in hits:
    print(h["score"], h["source"], h["content"][:80])
```

`retrieve()` returns hit dicts sorted best-first (lowest score first):

```python
{
    "content":  str,    # chunk text
    "score":    float,  # FAISS score (see below)
    "source":   str,    # e.g. "library_timings.md"
    "file_type":str,    # "md" | "txt" | "pdf" | "docx"
    "chunk":    int,    # chunk index
    "doc_id":   str,    # "source#page"
    "metadata": dict,   # full LangChain metadata
}
```

## How scores work (verified, not guessed)

Checked directly against the **installed** `langchain-community 0.4.2`
source (`langchain_community/vectorstores/faiss.py`):

- The default store builds `faiss.IndexFlatL2` with
  `DistanceStrategy.EUCLIDEAN_DISTANCE`.
- `similarity_search_with_score()` therefore returns the
  **squared L2 distance** between query and chunk embeddings — the source
  docstring states: *"L2 distance in float. Lower score represents more
  similarity."*
- The library's own `score_threshold` for euclidean distance applies with
  `operator.le` (keep results where `score <= threshold`).

So **`MIN_SCORE` is a maximum distance, not a minimum similarity**: a chunk is
kept only when `score <= MIN_SCORE`. Lower is better.

### Why `MIN_SCORE = 1.40`

Measured raw scores on the NMAMIT corpus (squared L2):

| Query type | Observed score range |
|---|---|
| Relevant top hits for all 9 verification topics | **0.43 – 1.31** |
| No-relevant-document query ("capital of Japan") | **≥ 1.69** |

1.40 sits inside the observed gap, so every relevant top hit passes and
queries with no matching knowledge-base content are filtered to zero results.
**Re-calibrate if you change the corpus or the model.**

## Test results (NMAMIT knowledge base, 13 source files)

| Verification query | Top-1 source | Score |
|---|---|---|
| CSE department information | NMAMIT_CSE_General_Information.pdf | 0.83 |
| Faculty name/designation/joining date | NMAMIT_CSE_Faculty_Name_Designation_Joining_Date.docx | 0.61 |
| Library timings | student_services.md | 1.25 |
| Hostel information | student_services.md | 0.72 |
| Internship information | NMAMIT_CSE_General_Information.pdf | 0.51 |
| Academic regulations | NMAMIT_CSE_General_Information.pdf | 0.75 |
| Examination information | NMAMIT_CSE_General_Information.pdf | 0.81 |
| Fees / scholarships | admissions_fees_scholarships.md | 0.78 |
| CSE 2025–2029 syllabus | NMAMIT_CSE_General_Information.pdf | 0.43 |
| "capital of Japan" (must be empty) | no hits — filtered by MIN_SCORE | — |

```
RESULT: 46 passed, 0 failed
```

## Notes

- Supported extensions: `.pdf` (pypdf), `.docx` (python-docx),
  `.md` / `.txt` (UTF-8 stdlib).
- Files that fail to load are skipped with a warning; ingestion never
  crashes on one bad file.
- The FAISS index is a generated artefact (git-ignored). Deleting
  `data/faiss_index/` is safe — `get_store()` rebuilds it automatically.
- Earlier placeholder demo files were moved to
  `data/_placeholder_samples_archive/` (outside the indexed corpus) so the
  index contains only the official package.
- `langchain-community` prints a deprecation warning (the package is being
  sunset); the FAISS/PyPDF APIs used here remain functional in the pinned
  version.
