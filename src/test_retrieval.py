"""Simple retrieval test for the Member 1 Data & RAG module.

Runs the required verification queries against the FAISS index built from
data/college_docs/ (the NMAMIT knowledge-base package) and checks that:
  1. Each topic retrieves relevant college chunks ranked first, with correct
     source metadata and ascending score order.
  2. The unrelated question ("capital of Japan") retrieves nothing —
     MIN_SCORE filtering rejects every candidate.

Run:  python src/test_retrieval.py     (exit code 0 = all passed)

No LLM / API key required — retrieval only.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles/files default to cp1252, which cannot encode characters
# present in the corpus (e.g. the rupee sign U+20B9). Print output as UTF-8
# so displaying a hit can never abort the test run.
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import MIN_SCORE, get_store, retrieve  # noqa: E402

# (query, expected top-1 sources, keyword that must appear in the top hit)
COLLEGE_CASES = [
    # 1. CSE department information
    (
        "information about the computer science engineering department",
        {"NMAMIT_CSE_General_Information.pdf", "cse_syllabus_2025_29.md",
         "sources.md"},
        "computer science",
    ),
    # 2. faculty name / designation / joining date
    (
        "name designation and joining date of CSE faculty",
        {"NMAMIT_CSE_Faculty_Name_Designation_Joining_Date.docx"},
        "designat",
    ),
    # 3. library timings
    (
        "library timings",
        {"student_services.md", "student_faqs.md"},
        "librar",
    ),
    # 4. hostel information
    (
        "hostel facilities",
        {"student_services.md", "student_faqs.md"},
        "hostel",
    ),
    # 5. internship information
    (
        "summer research internship program",
        {"NMAMIT_CSE_General_Information.pdf", "internship_guidelines.md"},
        "internship",
    ),
    # 6. academic regulations
    (
        "academic regulations",
        {"NMAMIT_CSE_General_Information.pdf", "academic_regulations.md"},
        "regulation",
    ),
    # 7. examination information
    (
        "examination information",
        {"NMAMIT_CSE_General_Information.pdf", "examination_information.md",
         "student_faqs.md"},
        "examination",
    ),
    # 8. fees / scholarships
    (
        "admission fees and scholarships",
        {"admissions_fees_scholarships.md", "NMAMIT_CSE_General_Information.pdf"},
        "fee",
    ),
    # 9. CSE 2025-2029 syllabus information
    (
        "CSE 2025-2029 syllabus",
        {"NMAMIT_CSE_General_Information.pdf", "cse_syllabus_2025_29.md"},
        "syllabus",
    ),
]

IRRELEVANT_QUERY = "what is the capital of Japan?"

_passed = 0
_failed = 0


def check(description: str, ok: bool, detail: str = "") -> None:
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  PASS  {description}")
    else:
        _failed += 1
        print(f"  FAIL  {description}" + (f"  [{detail}]" if detail else ""))


def show(hits) -> None:
    if not hits:
        print("        (no hits — everything filtered by MIN_SCORE)")
        return
    for h in hits:
        print(f"        score={h['score']:.4f}  source={h['source']}  "
              f"type={h['file_type']}  chunk={h['chunk']}")
        print(f"        content: {h['content'][:110]!r}")


def main() -> int:
    store = get_store()
    n_vectors = store.index.ntotal
    print(f"FAISS index: {n_vectors} vectors | MIN_SCORE={MIN_SCORE} "
          f"(max accepted squared-L2 distance)\n")

    for query, expected_sources, keyword in COLLEGE_CASES:
        print(f'Q: "{query}"')
        hits = retrieve(query, store=store)
        show(hits)
        check("retrieves at least one chunk", bool(hits),
              "no chunk passed MIN_SCORE")
        if hits:
            top = hits[0]
            check(
                f"top-1 source in {sorted(expected_sources)}",
                top["source"] in expected_sources,
                f"got {top['source']}",
            )
            check(
                f"top-1 content contains {keyword!r}",
                keyword in top["content"].lower(),
                top["content"][:60],
            )
            check(
                "hits are sorted by ascending score",
                all(hits[i]["score"] <= hits[i + 1]["score"]
                    for i in range(len(hits) - 1)),
            )
            check(
                "every hit carries source metadata",
                all(h["source"] and h["source"] != "unknown" for h in hits),
            )
        print()

    # --- unrelated query must retrieve nothing ------------------------------
    print(f'Q: "{IRRELEVANT_QUERY}"')
    hits = retrieve(IRRELEVANT_QUERY, store=store)
    show(hits)
    check("retrieves NO college chunks (all filtered by MIN_SCORE)",
          len(hits) == 0, f"{len(hits)} hit(s) leaked through")

    print(f"\n{'=' * 60}")
    print(f"RESULT: {_passed} passed, {_failed} failed")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
