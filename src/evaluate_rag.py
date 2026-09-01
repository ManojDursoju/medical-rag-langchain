from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evidence_selector import EvidenceSelector
from reranker import MedicalReranker


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation"

RESULTS_FILE = RESULTS_DIR / "retrieval_evaluation.json"


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
        "required_terms": [
            "glioblastoma",
            "mri",
        ],
    },

    {
        "id": "Q02",
        "question": (
            "How is deep learning used for "
            "brain tumor segmentation?"
        ),
        "required_terms": [
            "deep learning",
            "segmentation",
        ],
    },

    {
        "id": "Q03",
        "question": (
            "What is the role of radiomics "
            "in glioma diagnosis?"
        ),
        "required_terms": [
            "radiomics",
            "glioma",
        ],
    },

    {
        "id": "Q04",
        "question": (
            "How can MRI be used to predict "
            "treatment response in glioblastoma?"
        ),
        "required_terms": [
            "mri",
            "glioblastoma",
            "treatment",
        ],
    },

    {
        "id": "Q05",
        "question": (
            "What imaging features are associated "
            "with glioma molecular characteristics?"
        ),
        "required_terms": [
            "glioma",
            "molecular",
        ],
    },

    {
        "id": "Q06",
        "question": (
            "What deep learning methods are used "
            "for glioblastoma classification?"
        ),
        "required_terms": [
            "deep learning",
            "glioblastoma",
            "classification",
        ],
    },

    {
        "id": "Q07",
        "question": (
            "How is radiogenomics applied to "
            "brain tumor imaging?"
        ),
        "required_terms": [
            "radiogenomics",
            "brain tumor",
        ],
    },

    {
        "id": "Q08",
        "question": (
            "What MRI techniques are useful for "
            "monitoring glioblastoma treatment response?"
        ),
        "required_terms": [
            "mri",
            "glioblastoma",
            "treatment",
        ],
    },

    {
        "id": "Q09",
        "question": (
            "What role does IDH status play in "
            "glioma imaging research?"
        ),
        "required_terms": [
            "idh",
            "glioma",
        ],
    },

    {
        "id": "Q10",
        "question": (
            "What machine learning approaches are "
            "used in brain neoplasm research?"
        ),
        "required_terms": [
            "machine learning",
            "brain",
        ],
    },
]


# ============================================================
# Utility
# ============================================================

def normalize(text: str) -> str:

    return str(text).lower()


# ============================================================
# Evaluate one query
# ============================================================

def evaluate_query(
    selector: EvidenceSelector,
    item: dict,
) -> dict:

    question = item["question"]

    print(
        "\n" + "=" * 90
    )

    print(
        f"{item['id']}: {question}"
    )

    print(
        "=" * 90
    )

    start = time.perf_counter()

    evidence = selector.select(
        question
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Collect evidence text
    # --------------------------------------------------------

    combined_text = " ".join(
        normalize(
            x.get(
                "page_content",
                "",
            )
        )
        for x in evidence
    )

    # --------------------------------------------------------
    # Required concept coverage
    # --------------------------------------------------------

    required_terms = item[
        "required_terms"
    ]

    term_results = {}

    for term in required_terms:

        term_results[term] = (
            normalize(term)
            in combined_text
        )

    covered_terms = sum(
        term_results.values()
    )

    coverage = (
        covered_terms
        / len(required_terms)
        if required_terms
        else 0.0
    )

    # --------------------------------------------------------
    # PMID diversity
    # --------------------------------------------------------

    pmids = [
        str(
            x.get(
                "pmid",
                "",
            )
        ).strip()
        for x in evidence
    ]

    unique_pmids = set(
        x
        for x in pmids
        if x
    )

    diversity = (
        len(unique_pmids)
        / len(pmids)
        if pmids
        else 0.0
    )

    # --------------------------------------------------------
    # Domain compatibility
    # --------------------------------------------------------

    compatible = sum(
        1
        for x in evidence
        if x.get(
            "topic_compatible",
            True,
        )
    )

    compatibility = (
        compatible
        / len(evidence)
        if evidence
        else 0.0
    )

    # --------------------------------------------------------
    # Average ranking score
    # --------------------------------------------------------

    scores = [
        float(
            x.get(
                "final_score",
                0.0,
            )
        )
        for x in evidence
    ]

    average_score = (
        sum(scores)
        / len(scores)
        if scores
        else 0.0
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "id": item["id"],
        "question": question,
        "evidence_count": len(evidence),
        "unique_pmids": len(
            unique_pmids
        ),
        "term_coverage": coverage,
        "term_results": term_results,
        "domain_compatibility": compatibility,
        "average_final_score": average_score,
        "retrieval_seconds": elapsed,
        "sources": [
            {
                "pmid": x.get(
                    "pmid",
                    "",
                ),
                "title": x.get(
                    "title",
                    "",
                ),
                "year": x.get(
                    "year",
                    "",
                ),
                "doi": x.get(
                    "doi",
                    "",
                ),
                "topic": x.get(
                    "topic",
                    "",
                ),
                "score": x.get(
                    "final_score",
                    0.0,
                ),
            }
            for x in evidence
        ],
    }

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print(
        f"Evidence:             "
        f"{len(evidence)}"
    )

    print(
        f"Unique PMIDs:         "
        f"{len(unique_pmids)}"
    )

    print(
        f"Term coverage:        "
        f"{coverage:.2%}"
    )

    print(
        f"Domain compatibility: "
        f"{compatibility:.2%}"
    )

    print(
        f"Average score:        "
        f"{average_score:.4f}"
    )

    print(
        f"Retrieval time:       "
        f"{elapsed:.2f}s"
    )

    print(
        "\nSources:"
    )

    for source in result[
        "sources"
    ]:

        print(
            f"  PMID {source['pmid']} | "
            f"{source['title']}"
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
        help="Evaluate only the first N queries.",
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
        "MEDICAL RAG RETRIEVAL EVALUATION"
    )

    print(
        "=" * 90
    )

    print(
        f"Queries: {len(queries)}"
    )

    print(
        "Evaluating retrieval + reranking + evidence selection."
    )

    print(
        "Gemini generation is NOT called."
    )

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    reranker = MedicalReranker()

    selector = EvidenceSelector(
        reranker,
        final_k=5,
    )

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    results = []

    for item in queries:

        result = evaluate_query(
            selector,
            item,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    if results:

        avg_coverage = sum(
            x["term_coverage"]
            for x in results
        ) / len(results)

        avg_compatibility = sum(
            x["domain_compatibility"]
            for x in results
        ) / len(results)

        avg_score = sum(
            x["average_final_score"]
            for x in results
        ) / len(results)

        avg_time = sum(
            x["retrieval_seconds"]
            for x in results
        ) / len(results)

        duplicate_free = sum(
            1
            for x in results
            if x["unique_pmids"]
            == x["evidence_count"]
        )

    else:

        avg_coverage = 0
        avg_compatibility = 0
        avg_score = 0
        avg_time = 0
        duplicate_free = 0

    summary = {
        "queries_evaluated": len(
            results
        ),
        "average_term_coverage": (
            avg_coverage
        ),
        "average_domain_compatibility": (
            avg_compatibility
        ),
        "average_final_score": (
            avg_score
        ),
        "average_retrieval_seconds": (
            avg_time
        ),
        "duplicate_free_queries": (
            duplicate_free
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
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        f"Queries evaluated:          "
        f"{len(results)}"
    )

    print(
        f"Average term coverage:      "
        f"{avg_coverage:.2%}"
    )

    print(
        f"Average domain compatibility:"
        f" {avg_compatibility:.2%}"
    )

    print(
        f"Average final score:         "
        f"{avg_score:.4f}"
    )

    print(
        f"Average retrieval time:      "
        f"{avg_time:.2f}s"
    )

    print(
        f"Duplicate-free queries:      "
        f"{duplicate_free}/{len(results)}"
    )

    print(
        "\nEvaluation saved to:"
    )

    print(
        RESULTS_FILE
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "RAG RETRIEVAL EVALUATION COMPLETED"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()