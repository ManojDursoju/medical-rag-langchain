from __future__ import annotations

import pickle
import re
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB = PROJECT_ROOT / "data" / "vector_db"

INDEX_FILE = VECTOR_DB / "index.faiss"
METADATA_FILE = VECTOR_DB / "metadata.pkl"
CONFIG_FILE = VECTOR_DB / "config.pkl"


# ------------------------------------------------------------
# Medical relevance terms
# ------------------------------------------------------------

BRAIN_TERMS = {
    "brain tumor",
    "brain tumour",
    "brain neoplasm",
    "glioma",
    "glioblastoma",
    "astrocytoma",
    "oligodendroglioma",
    "glioblastoma multiforme",
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
    "mri",
    "magnetic resonance imaging",
    "flair",
    "diffusion",
    "perfusion",
    "radiomics",
    "radiogenomics",
}


def normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()


def medical_relevance_score(text: str) -> float:
    """
    Lightweight domain relevance score.

    This is a ranking signal, not a hard filter.
    """

    text = normalize(text)

    matches = sum(
        1 for term in BRAIN_TERMS
        if term in text
    )

    return min(
        matches / 8.0,
        1.0,
    )


# ------------------------------------------------------------
# Retriever + reranker
# ------------------------------------------------------------

class MedicalReranker:

    def __init__(self):

        print("=" * 80)
        print("MEDICAL RERANKER")
        print("=" * 80)

        # Load FAISS
        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        # Load metadata
        with METADATA_FILE.open("rb") as f:
            self.metadata = pickle.load(f)

        # Load configuration
        with CONFIG_FILE.open("rb") as f:
            self.config = pickle.load(f)

        embedding_model = self.config[
            "embedding_model"
        ]

        print(
            f"Embedding model: {embedding_model}"
        )

        self.embedding_model = SentenceTransformer(
            embedding_model,
            device="cpu",
        )

        # Cross encoder
        reranker_name = (
            "BAAI/bge-reranker-base"
        )

        print(
            f"Reranker model:   {reranker_name}"
        )

        self.reranker = CrossEncoder(
            reranker_name,
            device="cpu",
        )

        print("Reranker ready.")

    # --------------------------------------------------------
    # Candidate retrieval
    # --------------------------------------------------------

    def retrieve_candidates(
        self,
        query: str,
        top_k: int = 20,
    ):

        embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            embedding,
            top_k,
        )

        candidates = []

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):

            if idx < 0:
                continue

            item = self.metadata[idx].copy()

            item["faiss_score"] = float(score)
            item["faiss_rank"] = rank
            item["faiss_id"] = int(idx)

            candidates.append(item)

        return candidates

    # --------------------------------------------------------
    # Rerank
    # --------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ):

        pairs = []

        for item in candidates:

            pairs.append(
                (
                    query,
                    item["page_content"],
                )
            )

        scores = self.reranker.predict(
            pairs,
            show_progress_bar=True,
        )

        for item, score in zip(
            candidates,
            scores,
        ):

            item["reranker_score"] = float(
                score
            )

            item[
                "medical_relevance"
            ] = medical_relevance_score(
                item["page_content"]
            )

            # Combined ranking signal.
            item["final_score"] = (
                0.75 * item["reranker_score"]
                + 0.25 * item[
                    "medical_relevance"
                ]
            )

        ranked = sorted(
            candidates,
            key=lambda x: x["final_score"],
            reverse=True,
        )

        return ranked[:top_k]


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

def display_results(
    query: str,
    results: list[dict],
):

    print("\n" + "=" * 90)
    print("RERANKED MEDICAL EVIDENCE")
    print("=" * 90)

    print(f"\nQuery:\n{query}")

    for i, result in enumerate(
        results,
        start=1,
    ):

        print("\n" + "-" * 90)

        print(f"Result #{i}")
        print(
            f"Final score:       "
            f"{result['final_score']:.4f}"
        )

        print(
            f"Reranker score:    "
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"Medical relevance: "
            f"{result['medical_relevance']:.4f}"
        )

        print(
            f"FAISS score:       "
            f"{result['faiss_score']:.4f}"
        )

        print(
            f"PMID:               "
            f"{result['pmid']}"
        )

        print(
            f"Title:              "
            f"{result['title']}"
        )

        print(
            f"Year:               "
            f"{result['year']}"
        )

        print(
            f"DOI:                "
            f"{result['doi']}"
        )

        print(
            f"Topic:              "
            f"{result['topic']}"
        )

        print("\nEvidence:")

        content = result["page_content"]

        if len(content) > 1200:
            content = content[:1200] + "..."

        print(content)


# ------------------------------------------------------------
# Test questions
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    reranker = MedicalReranker()

    for query in TEST_QUERIES:

        print(
            "\n\n"
            + "#" * 90
        )

        candidates = (
            reranker.retrieve_candidates(
                query,
                top_k=20,
            )
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

    print("\n" + "=" * 90)
    print("RERANKING TEST COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()