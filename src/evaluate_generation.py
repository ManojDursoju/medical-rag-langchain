from __future__ import annotations

import os
import argparse
import json
import re
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from evidence_selector import EvidenceSelector
from reranker import MedicalReranker


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation"

RESULTS_FILE = (
    RESULTS_DIR / "generation_evaluation.json"
)

LLM_MODEL = "gemini-3.6-flash"

FINAL_EVIDENCE_K = 5


# ============================================================
# Evaluation questions
# ============================================================

EVALUATION_QUERIES = [

    {
        "id": "Q01",
        "question": (
            "What MRI characteristics are associated "
            "with glioblastoma?"
        ),
    },

    {
        "id": "Q02",
        "question": (
            "How is deep learning used for "
            "brain tumor segmentation?"
        ),
    },

    {
        "id": "Q03",
        "question": (
            "What is the role of radiomics "
            "in glioma diagnosis?"
        ),
    },

    {
        "id": "Q04",
        "question": (
            "How can MRI be used to predict "
            "treatment response in glioblastoma?"
        ),
    },

    {
        "id": "Q05",
        "question": (
            "What imaging features are associated "
            "with glioma molecular characteristics?"
        ),
    },

    {
        "id": "Q06",
        "question": (
            "What deep learning methods are used "
            "for glioblastoma classification?"
        ),
    },

    {
        "id": "Q07",
        "question": (
            "How is radiogenomics applied to "
            "brain tumor imaging?"
        ),
    },

    {
        "id": "Q08",
        "question": (
            "What MRI techniques are useful for "
            "monitoring glioblastoma treatment response?"
        ),
    },

    {
        "id": "Q09",
        "question": (
            "What role does IDH status play in "
            "glioma imaging research?"
        ),
    },

    {
        "id": "Q10",
        "question": (
            "What machine learning approaches are "
            "used in brain neoplasm research?"
        ),
    },
]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
You are a Medical Research RAG Copilot.

Answer the research question using ONLY the supplied
PubMed evidence.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not invent numerical results.
4. Do not fabricate citations.
5. Only cite PMIDs present in the supplied evidence.
6. Only cite DOIs present in the supplied evidence.
7. Every substantive scientific claim should have a PMID citation
   when supported by a specific publication.
8. Distinguish study findings from synthesis or interpretation.
9. Preserve numerical values accurately.
10. Mention limitations when supported by the evidence.
11. If the evidence is insufficient, say:

"The retrieved PubMed evidence is insufficient to answer this reliably."

12. Do not diagnose individual patients.
13. Do not recommend treatment for individual patients.

Citation format:

[PMID: 12345678]

or:

[PMID: 12345678; DOI: 10.xxxx/xxxxx]

Never invent a PMID or DOI.
"""


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT,
        ),
        (
            "human",
            """
Research question:

{question}

Retrieved PubMed evidence:

{context}

Answer using ONLY the retrieved evidence.

Use this structure:

## Answer

## Evidence

## Limitations

## Sources

Only include sources actually used.

Every PMID and DOI must come from the supplied evidence.
""",
        ),
    ]
)


# ============================================================
# Evidence formatting
# ============================================================

def format_evidence(
    evidence: list[dict],
) -> str:

    sections = []

    for i, item in enumerate(
        evidence,
        start=1,
    ):

        sections.append(
            f"""
--- EVIDENCE {i} ---

PMID: {item.get("pmid", "")}
Title: {item.get("title", "")}
Journal: {item.get("journal", "")}
Year: {item.get("year", "")}
DOI: {item.get("doi", "")}
Topic: {item.get("topic", "")}

Evidence:
{item.get("page_content", "")}
""".strip()
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# Gemini response extraction
# ============================================================

def clean_response(
    content,
) -> str:

    if content is None:
        return ""

    if isinstance(
        content,
        str,
    ):
        return content.strip()

    if isinstance(
        content,
        list,
    ):

        parts = []

        for block in content:

            if isinstance(
                block,
                dict,
            ):

                text = block.get(
                    "text"
                )

                if text:
                    parts.append(
                        str(text)
                    )

            else:

                text = getattr(
                    block,
                    "text",
                    None,
                )

                if text:
                    parts.append(
                        str(text)
                    )

        return "\n".join(
            parts
        ).strip()

    return str(
        content
    ).strip()


# ============================================================
# Citation extraction
# ============================================================

def extract_pmids(
    text: str,
) -> list[str]:

    matches = re.findall(
        r"\[PMID:\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    return list(
        dict.fromkeys(
            matches
        )
    )


def extract_dois(
    text: str,
) -> list[str]:

    matches = re.findall(
        r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
        text,
        flags=re.IGNORECASE,
    )

    cleaned = []

    for doi in matches:

        doi = doi.rstrip(
            ".,;)"
        )

        if doi not in cleaned:
            cleaned.append(
                doi
            )

    return cleaned


# ============================================================
# Citation validation
# ============================================================

def validate_citations(
    answer: str,
    evidence: list[dict],
) -> dict:

    valid_pmids = {
        str(
            item.get(
                "pmid",
                "",
            )
        ).strip()
        for item in evidence
        if item.get("pmid")
    }

    valid_dois = {
        str(
            item.get(
                "doi",
                "",
            )
        ).strip().lower()
        for item in evidence
        if item.get("doi")
    }

    cited_pmids = extract_pmids(
        answer
    )

    cited_dois = extract_dois(
        answer
    )

    invalid_pmids = [
        pmid
        for pmid in cited_pmids
        if pmid not in valid_pmids
    ]

    invalid_dois = [
        doi
        for doi in cited_dois
        if doi.lower()
        not in valid_dois
    ]

    return {
        "cited_pmids": cited_pmids,
        "valid_pmids": [
            x
            for x in cited_pmids
            if x in valid_pmids
        ],
        "invalid_pmids": invalid_pmids,
        "cited_dois": cited_dois,
        "valid_dois": [
            x
            for x in cited_dois
            if x.lower()
            in valid_dois
        ],
        "invalid_dois": invalid_dois,
        "citation_integrity": (
            len(invalid_pmids) == 0
            and len(invalid_dois) == 0
        ),
    }


# ============================================================
# Structure validation
# ============================================================

def validate_structure(
    answer: str,
) -> dict:

    required_sections = [
        "## Answer",
        "## Evidence",
        "## Limitations",
        "## Sources",
    ]

    section_results = {
        section: section in answer
        for section in required_sections
    }

    return {
        "sections": section_results,
        "structure_complete": all(
            section_results.values()
        ),
    }


# ============================================================
# Source usage
# ============================================================

def source_usage(
    answer: str,
    evidence: list[dict],
) -> dict:

    cited_pmids = set(
        extract_pmids(
            answer
        )
    )

    retrieved_pmids = {
        str(
            item.get(
                "pmid",
                "",
            )
        ).strip()
        for item in evidence
        if item.get("pmid")
    }

    used = (
        cited_pmids
        & retrieved_pmids
    )

    return {
        "retrieved_sources": len(
            retrieved_pmids
        ),
        "cited_sources": len(
            cited_pmids
        ),
        "retrieved_sources_used": len(
            used
        ),
        "source_utilization": (
            len(used)
            / len(retrieved_pmids)
            if retrieved_pmids
            else 0.0
        ),
    }


# ============================================================
# Evaluate one query
# ============================================================

def evaluate_query(
    selector: EvidenceSelector,
    llm,
    chain,
    item: dict,
) -> dict:

    question = item[
        "question"
    ]

    print(
        "\n" + "=" * 90
    )

    print(
        f"{item['id']}: {question}"
    )

    print(
        "=" * 90
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval_start = (
        time.perf_counter()
    )

    evidence = selector.select(
        question
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    print(
        f"Evidence selected: "
        f"{len(evidence)}"
    )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    context = format_evidence(
        evidence
    )

    print(
        "\nGenerating with Gemini..."
    )

    generation_start = (
        time.perf_counter()
    )

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    answer = clean_response(
        response.content
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    citation_result = (
        validate_citations(
            answer,
            evidence,
        )
    )

    structure_result = (
        validate_structure(
            answer
        )
    )

    usage_result = (
        source_usage(
            answer,
            evidence,
        )
    )

    # --------------------------------------------------------
    # Overall pass
    # --------------------------------------------------------

    passed = (
        citation_result[
            "citation_integrity"
        ]
        and structure_result[
            "structure_complete"
        ]
        and len(answer) > 0
    )

    result = {
        "id": item["id"],
        "question": question,
        "retrieval_time_seconds": (
            retrieval_time
        ),
        "generation_time_seconds": (
            generation_time
        ),
        "answer_length": len(
            answer
        ),
        "citation_validation": (
            citation_result
        ),
        "structure_validation": (
            structure_result
        ),
        "source_usage": (
            usage_result
        ),
        "passed": passed,
        "retrieved_sources": [
            {
                "pmid": x.get(
                    "pmid",
                    "",
                ),
                "title": x.get(
                    "title",
                    "",
                ),
                "doi": x.get(
                    "doi",
                    "",
                ),
                "year": x.get(
                    "year",
                    "",
                ),
            }
            for x in evidence
        ],
        "answer": answer,
    }

    # --------------------------------------------------------
    # Console result
    # --------------------------------------------------------

    print(
        f"\nRetrieval time: "
        f"{retrieval_time:.2f}s"
    )

    print(
        f"Generation time: "
        f"{generation_time:.2f}s"
    )

    print(
        f"Answer length: "
        f"{len(answer)} characters"
    )

    print(
        f"Citation integrity: "
        f"{citation_result['citation_integrity']}"
    )

    print(
        f"Structure complete: "
        f"{structure_result['structure_complete']}"
    )

    print(
        f"Sources used: "
        f"{usage_result['retrieved_sources_used']}/"
        f"{usage_result['retrieved_sources']}"
    )

    print(
        f"PASS: {passed}"
    )

    print(
        "\nGenerated answer:"
    )

    print(
        answer
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N queries."
        ),
    )

    args = parser.parse_args()

    queries = EVALUATION_QUERIES

    if args.limit:

        queries = queries[
            :args.limit
        ]

    print(
        "=" * 90
    )

    print(
        "MEDICAL RAG LLM GENERATION EVALUATION"
    )

    print(
        "=" * 90
    )

    print(
        f"Queries: {len(queries)}"
    )

    print(
        f"Gemini model: {LLM_MODEL}"
    )

    # --------------------------------------------------------
    # API key
    # --------------------------------------------------------

    api_key = os.getenv(
        "GOOGLE_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GOOGLE_API_KEY is not configured."
        )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    reranker = MedicalReranker()

    selector = EvidenceSelector(
        reranker,
        final_k=FINAL_EVIDENCE_K,
    )

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
    )

    chain = (
        PROMPT
        | llm
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    for item in queries:

        result = evaluate_query(
            selector,
            llm,
            chain,
            item,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    total = len(
        results
    )

    passed = sum(
        1
        for x in results
        if x["passed"]
    )

    citation_passed = sum(
        1
        for x in results
        if x[
            "citation_validation"
        ][
            "citation_integrity"
        ]
    )

    structure_passed = sum(
        1
        for x in results
        if x[
            "structure_validation"
        ][
            "structure_complete"
        ]
    )

    avg_retrieval = (
        sum(
            x[
                "retrieval_time_seconds"
            ]
            for x in results
        )
        / total
        if total
        else 0
    )

    avg_generation = (
        sum(
            x[
                "generation_time_seconds"
            ]
            for x in results
        )
        / total
        if total
        else 0
    )

    avg_source_usage = (
        sum(
            x[
                "source_usage"
            ][
                "source_utilization"
            ]
            for x in results
        )
        / total
        if total
        else 0
    )

    summary = {
        "queries_evaluated": total,
        "generation_pass_rate": (
            passed / total
            if total
            else 0
        ),
        "citation_integrity_rate": (
            citation_passed / total
            if total
            else 0
        ),
        "structure_completion_rate": (
            structure_passed / total
            if total
            else 0
        ),
        "average_retrieval_seconds": (
            avg_retrieval
        ),
        "average_generation_seconds": (
            avg_generation
        ),
        "average_source_utilization": (
            avg_source_usage
        ),
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "model": LLM_MODEL,
        "summary": summary,
        "results": results,
    }

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "LLM GENERATION EVALUATION SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        f"Queries evaluated:        "
        f"{total}"
    )

    print(
        f"Generation pass rate:     "
        f"{passed}/{total} "
        f"({summary['generation_pass_rate']:.2%})"
    )

    print(
        f"Citation integrity:       "
        f"{citation_passed}/{total} "
        f"({summary['citation_integrity_rate']:.2%})"
    )

    print(
        f"Structure completion:      "
        f"{structure_passed}/{total} "
        f"({summary['structure_completion_rate']:.2%})"
    )

    print(
        f"Average retrieval time:    "
        f"{avg_retrieval:.2f}s"
    )

    print(
        f"Average generation time:   "
        f"{avg_generation:.2f}s"
    )

    print(
        f"Average source utilization:"
        f" {avg_source_usage:.2%}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_FILE
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "LLM GENERATION EVALUATION COMPLETED"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()