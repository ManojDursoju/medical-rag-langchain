from __future__ import annotations

import re

from sentence_transformers import CrossEncoder

from retriever import MedicalRetriever


# ============================================================
# Configuration
# ============================================================

RERANKER_MODEL = "BAAI/bge-reranker-base"

CANDIDATE_K = 30
FINAL_RERANK_K = 15


# ============================================================
# Query concept detection
# ============================================================

QUERY_GROUPS = {
    "glioblastoma": [
        "glioblastoma",
        "glioblastoma multiforme",
        "gbm",
    ],
    "glioma": [
        "glioma",
        "astrocytoma",
        "oligodendroglioma",
        "diffuse glioma",
    ],
    "brain": [
        "brain tumor",
        "brain tumour",
        "brain neoplasm",
        "intracranial tumor",
        "intracranial tumour",
    ],
    "mri": [
        "mri",
        "magnetic resonance",
        "t1-weighted",
        "t2-weighted",
        "flair",
        "diffusion",
        "perfusion",
    ],
    "segmentation": [
        "segmentation",
        "segment",
        "segmented",
        "tumor boundary",
        "tumour boundary",
        "u-net",
        "unet",
        "voxel",
        "delineation",
    ],
    "radiomics": [
        "radiomics",
        "radiomic",
        "radiomic features",
        "texture features",
    ],
    "treatment": [
        "treatment response",
        "response assessment",
        "response monitoring",
        "therapy response",
        "treatment outcome",
        "progression",
        "pseudoprogression",
        "radionecrosis",
        "recurrence",
        "temozolomide",
        "radiotherapy",
    ],
    "molecular": [
        "idh",
        "idh1",
        "idh2",
        "mgmt",
        "egfr",
        "1p/19q",
        "cdkn2a",
        "tp53",
        "nf1",
        "pdgfra",
        "mutation",
        "methylation",
        "molecular",
        "genomic",
    ],
    "deep_learning": [
        "deep learning",
        "convolutional neural network",
        "cnn",
        "neural network",
        "u-net",
        "unet",
        "transformer",
        "machine learning",
    ],
}


# ============================================================
# Normalization
# ============================================================

def normalize(text: str) -> str:

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# Detect concepts
# ============================================================

def detect_concepts(
    text: str,
) -> set[str]:

    text = normalize(text)

    concepts = set()

    for group, terms in QUERY_GROUPS.items():

        for term in terms:

            if term in text:
                concepts.add(group)
                break

    return concepts


# ============================================================
# Topic compatibility
# ============================================================

def topic_compatibility(
    query: str,
    item: dict,
) -> float:

    query_concepts = detect_concepts(
        query
    )

    title = item.get(
        "title",
        "",
    )

    content = item.get(
        "page_content",
        "",
    )

    text_concepts = detect_concepts(
        title + " " + content
    )

    if not query_concepts:
        return 0.5

    matched = len(
        query_concepts
        & text_concepts
    )

    return matched / len(
        query_concepts
    )


# ============================================================
# Strong domain compatibility
# ============================================================

def is_strongly_compatible(
    query: str,
    item: dict,
) -> bool:

    query_concepts = detect_concepts(
        query
    )

    title = normalize(
        item.get("title", "")
    )

    topic = normalize(
        item.get("topic", "")
    )

    content = normalize(
        item.get("page_content", "")
    )

    # --------------------------------------------------------
    # Brain/glioma queries
    # --------------------------------------------------------

    brain_query = bool(
        query_concepts
        & {
            "brain",
            "glioma",
            "glioblastoma",
        }
    )

    if brain_query:

        brain_evidence = any(
            term in title
            or term in content[:4000]
            for term in [
                "brain tumor",
                "brain tumour",
                "brain neoplasm",
                "glioma",
                "glioblastoma",
                "glioblastoma multiforme",
                "gbm",
            ]
        )

        if not brain_evidence:
            return False

        # Topic-level protection.
        unrelated_topics = [
            "meningioma",
            "cervical",
            "prostate",
            "breast",
            "lung",
            "colorectal",
            "rectal",
            "renal",
            "shoulder",
            "hip",
            "rotator",
        ]

        # If topic contains an unrelated domain and
        # the title does not explicitly establish brain
        # tumor relevance, reject it.
        for term in unrelated_topics:

            if term in title:
                return False

    # --------------------------------------------------------
    # Radiomics + glioma
    # --------------------------------------------------------

    if (
        "radiomics" in query_concepts
        and (
            "glioma" in query_concepts
            or "glioblastoma" in query_concepts
            or "brain" in query_concepts
        )
    ):

        glioma_present = any(
            term in title
            or term in content[:5000]
            for term in [
                "glioma",
                "glioblastoma",
                "gbm",
                "brain tumor",
                "brain tumour",
            ]
        )

        if not glioma_present:
            return False

    return True


# ============================================================
# Reranker
# ============================================================

class MedicalReranker:

    def __init__(
        self,
        retriever: MedicalRetriever | None = None,
    ):

        print("=" * 80)
        print("MEDICAL RERANKER")
        print("=" * 80)

        self.retriever = (
            retriever
            if retriever is not None
            else MedicalRetriever()
        )

        print(
            f"Reranker model:   "
            f"{RERANKER_MODEL}"
        )

        self.reranker = CrossEncoder(
            RERANKER_MODEL,
            device="cpu",
        )

        print(
            "Reranker ready."
        )

    # ========================================================
    # Rerank
    # ========================================================

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = FINAL_RERANK_K,
    ) -> list[dict]:

        # ----------------------------------------------------
        # Remove duplicate PMID
        # ----------------------------------------------------

        unique = []

        seen_pmids = set()

        for item in candidates:

            pmid = str(
                item.get(
                    "pmid",
                    "",
                )
            ).strip()

            if pmid and pmid in seen_pmids:
                continue

            if pmid:
                seen_pmids.add(pmid)

            unique.append(item)

        # ----------------------------------------------------
        # Domain filtering
        # ----------------------------------------------------

        compatible = []

        for item in unique:

            compatible_flag = (
                is_strongly_compatible(
                    query,
                    item,
                )
            )

            item[
                "topic_compatible"
            ] = compatible_flag

            if compatible_flag:
                compatible.append(item)

        # Don't return nothing if a query is unusual.
        if not compatible:
            compatible = unique

        # ----------------------------------------------------
        # CrossEncoder
        # ----------------------------------------------------

        pairs = [
            (
                query,
                item.get(
                    "page_content",
                    "",
                ),
            )
            for item in compatible
        ]

        scores = self.reranker.predict(
            pairs,
            show_progress_bar=True,
        )

        # ----------------------------------------------------
        # Combine scores
        # ----------------------------------------------------

        for item, score in zip(
            compatible,
            scores,
        ):

            reranker_score = float(
                score
            )

            retrieval_score = float(
                item.get(
                    "retrieval_score",
                    item.get(
                        "faiss_score",
                        0.0,
                    ),
                )
            )

            domain_score = float(
                item.get(
                    "domain_score",
                    0.5,
                )
            )

            # CrossEncoder is the strongest signal.
            item[
                "reranker_score"
            ] = reranker_score

            item[
                "final_score"
            ] = (
                0.60 * reranker_score
                + 0.25 * retrieval_score
                + 0.15 * domain_score
            )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        ranked = sorted(
            compatible,
            key=lambda x: x[
                "final_score"
            ],
            reverse=True,
        )

        return ranked[:top_k]


# ============================================================
# Display
# ============================================================

def display_results(
    query: str,
    results: list[dict],
):

    print(
        "\n" + "=" * 90
    )

    print(
        "RERANKED MEDICAL EVIDENCE"
    )

    print(
        "=" * 90
    )

    print(
        f"\nQuery:\n{query}"
    )

    for i, item in enumerate(
        results,
        start=1,
    ):

        print(
            "\n" + "-" * 90
        )

        print(
            f"Result #{i}"
        )

        print(
            f"Final score:       "
            f"{item['final_score']:.4f}"
        )

        print(
            f"Reranker score:    "
            f"{item['reranker_score']:.4f}"
        )

        print(
            f"Retrieval score:   "
            f"{item.get('retrieval_score', 0):.4f}"
        )

        print(
            f"Domain score:      "
            f"{item.get('domain_score', 0):.4f}"
        )

        print(
            f"PMID:              "
            f"{item['pmid']}"
        )

        print(
            f"Title:             "
            f"{item['title']}"
        )

        print(
            f"Year:              "
            f"{item['year']}"
        )

        print(
            f"Journal:           "
            f"{item['journal']}"
        )

        print(
            f"DOI:               "
            f"{item['doi']}"
        )

        print(
            f"Topic:              "
            f"{item['topic']}"
        )


# ============================================================
# Tests
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

    retriever = MedicalRetriever()

    reranker = MedicalReranker(
        retriever
    )

    for query in TEST_QUERIES:

        candidates = retriever.retrieve(
            query,
            top_k=CANDIDATE_K,
        )

        results = reranker.rerank(
            query,
            candidates,
            top_k=5,
        )

        display_results(
            query,
            results,
        )

    print(
        "\n" + "=" * 90
    )

    print(
        "RERANKING TEST COMPLETED"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()