from __future__ import annotations

from reranker import (
    MedicalReranker,
)
from retriever import (
    MedicalRetriever,
)


# ============================================================
# Configuration
# ============================================================

CANDIDATE_K = 30

RERANK_K = 15

FINAL_EVIDENCE_K = 5

MAX_CHUNKS_PER_PMID = 1


# ============================================================
# Evidence selector
# ============================================================

class EvidenceSelector:

    def __init__(
        self,
        reranker: MedicalReranker,
        final_k: int = FINAL_EVIDENCE_K,
    ):

        self.reranker = reranker

        self.final_k = final_k

    # ========================================================
    # Select evidence
    # ========================================================

    def select(
        self,
        query: str,
    ) -> list[dict]:

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        candidates = (
            self.reranker.retriever.retrieve(
                query,
                top_k=CANDIDATE_K,
            )
        )

        # ----------------------------------------------------
        # Rerank
        # ----------------------------------------------------

        reranked = (
            self.reranker.rerank(
                query,
                candidates,
                top_k=RERANK_K,
            )
        )

        # ----------------------------------------------------
        # Diversity selection
        # ----------------------------------------------------

        selected = []

        pmid_counts = {}

        for item in reranked:

            pmid = str(
                item.get(
                    "pmid",
                    "",
                )
            ).strip()

            if pmid:

                count = pmid_counts.get(
                    pmid,
                    0,
                )

                if (
                    count
                    >= MAX_CHUNKS_PER_PMID
                ):
                    continue

                pmid_counts[
                    pmid
                ] = count + 1

            selected.append(
                item
            )

            if (
                len(selected)
                >= self.final_k
            ):
                break

        return selected


# ============================================================
# Evidence formatting
# ============================================================

def format_evidence(
    evidence: list[dict],
) -> str:

    blocks = []

    for i, item in enumerate(
        evidence,
        start=1,
    ):

        block = f"""
SOURCE {i}

PMID: {item.get('pmid', '')}
Title: {item.get('title', '')}
Journal: {item.get('journal', '')}
Year: {item.get('year', '')}
DOI: {item.get('doi', '')}
Topic: {item.get('topic', '')}

Evidence:
{item.get('page_content', '')}
""".strip()

        blocks.append(
            block
        )

    return "\n\n" + (
        "\n\n".join(blocks)
    )


# ============================================================
# Display
# ============================================================

def display_evidence(
    query: str,
    evidence: list[dict],
):

    print(
        "\n" + "=" * 90
    )

    print(
        "FINAL SELECTED EVIDENCE"
    )

    print(
        "=" * 90
    )

    print(
        f"\nQuestion:\n{query}"
    )

    for i, item in enumerate(
        evidence,
        start=1,
    ):

        print(
            "\n" + "-" * 90
        )

        print(
            f"Evidence #{i}"
        )

        print(
            f"PMID:        "
            f"{item.get('pmid', '')}"
        )

        print(
            f"Title:       "
            f"{item.get('title', '')}"
        )

        print(
            f"Year:        "
            f"{item.get('year', '')}"
        )

        print(
            f"Journal:     "
            f"{item.get('journal', '')}"
        )

        print(
            f"DOI:         "
            f"{item.get('doi', '')}"
        )

        print(
            f"Topic:       "
            f"{item.get('topic', '')}"
        )

        print(
            f"Final score: "
            f"{item.get('final_score', 0):.4f}"
        )

        print(
            "\nEvidence text:"
        )

        print(
            item.get(
                "page_content",
                "",
            )
        )


# ============================================================
# Test queries
# ============================================================

TEST_QUERIES = [

    "What MRI characteristics are associated with glioblastoma?",

    "How is deep learning used for brain tumor segmentation?",

    "What is the role of radiomics in glioma diagnosis?",

    "How can MRI be used to predict treatment response in glioblastoma?",

    "What imaging features are associated with glioma molecular characteristics?",
]


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n" + "=" * 90
    )

    print(
        "PUBMED EVIDENCE SELECTION"
    )

    print(
        "=" * 90
    )

    retriever = MedicalRetriever()

    reranker = MedicalReranker(
        retriever
    )

    selector = EvidenceSelector(
        reranker,
        final_k=FINAL_EVIDENCE_K,
    )

    for query in TEST_QUERIES:

        evidence = selector.select(
            query
        )

        display_evidence(
            query,
            evidence,
        )

    print(
        "\n" + "=" * 90
    )

    print(
        "EVIDENCE SELECTION COMPLETED"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()