"""
Medical Evidence Selector

Purpose:
    Select diverse, high-quality evidence from reranked candidates.

Pipeline:
    FAISS + BM25
        ↓
    RRF
        ↓
    Reranker
        ↓
    Evidence Selector
        ↓
    Diverse evidence for LLM
"""

from __future__ import annotations

from pathlib import Path
import pickle

from reranker import MedicalReranker


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CANDIDATE_K = 20
FINAL_EVIDENCE_K = 5

# Prefer different publications.
MAX_CHUNKS_PER_PMID = 1


# ---------------------------------------------------------------------
# Evidence Selector
# ---------------------------------------------------------------------

class EvidenceSelector:

    def __init__(
        self,
        reranker: MedicalReranker,
        final_k: int = FINAL_EVIDENCE_K,
    ):

        self.reranker = reranker
        self.final_k = final_k

    # -----------------------------------------------------------------
    # Select diverse evidence
    # -----------------------------------------------------------------

    def select(
        self,
        query: str,
    ) -> list[dict]:

        # -------------------------------------------------------------
        # Retrieve candidates
        # -------------------------------------------------------------

        candidates = (
            self.reranker.retrieve_candidates(
                query,
                top_k=CANDIDATE_K,
            )
        )

        # -------------------------------------------------------------
        # Rerank
        # -------------------------------------------------------------

        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=CANDIDATE_K,
        )

        # -------------------------------------------------------------
        # Diversity selection
        # -------------------------------------------------------------

        selected = []

        seen_pmids = set()

        for item in reranked:

            pmid = str(
                item.get(
                    "pmid",
                    "",
                )
            ).strip()

            # Skip duplicate publication.
            if pmid and pmid in seen_pmids:
                continue

            selected.append(item)

            if pmid:
                seen_pmids.add(pmid)

            if len(selected) >= self.final_k:
                break

        return selected


# ---------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------

def display_evidence(
    query: str,
    evidence: list[dict],
):

    print("\n" + "=" * 90)
    print("FINAL SELECTED EVIDENCE")
    print("=" * 90)

    print(f"\nQuestion:\n{query}")

    for i, item in enumerate(
        evidence,
        start=1,
    ):

        print("\n" + "-" * 90)

        print(f"Evidence #{i}")

        print(
            f"PMID:        {item['pmid']}"
        )

        print(
            f"Title:       {item['title']}"
        )

        print(
            f"Year:        {item['year']}"
        )

        print(
            f"Journal:     {item['journal']}"
        )

        print(
            f"DOI:         {item['doi']}"
        )

        print(
            f"Topic:       {item['topic']}"
        )

        print(
            f"Reranker:    "
            f"{item['reranker_score']:.4f}"
        )

        print(
            f"Medical:     "
            f"{item['medical_relevance']:.4f}"
        )

        print("\nEvidence text:")

        print(
            item["page_content"]
        )


# ---------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------

TEST_QUERIES = [

    "What MRI characteristics are associated with glioblastoma?",

    "How is deep learning used for brain tumor segmentation?",

    "What is the role of radiomics in glioma diagnosis?",

    "How can MRI be used to predict treatment response in glioblastoma?",

    (
        "What imaging features are associated with "
        "IDH1 MGMT EGFR and other glioma molecular characteristics?"
    ),
]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("\n" + "=" * 90)
    print("PUBMED EVIDENCE SELECTION")
    print("=" * 90)

    reranker = MedicalReranker()

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

    print("\n" + "=" * 90)
    print("EVIDENCE SELECTION COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()