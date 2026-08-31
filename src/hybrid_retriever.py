"""
Hybrid PubMed Retriever

Combines:
    1. FAISS semantic retrieval
    2. BM25 lexical retrieval
    3. Reciprocal Rank Fusion (RRF)

Input:
    data/vector_db/index.faiss
    data/vector_db/metadata.pkl
    data/vector_db/config.pkl

Output:
    Ranked hybrid retrieval results
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"

INDEX_FILE = VECTOR_DB_DIR / "index.faiss"
METADATA_FILE = VECTOR_DB_DIR / "metadata.pkl"
CONFIG_FILE = VECTOR_DB_DIR / "config.pkl"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FAISS_TOP_K = 20
BM25_TOP_K = 20
FINAL_TOP_K = 10

RRF_K = 60


# ---------------------------------------------------------------------
# Text tokenizer
# ---------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """
    Simple medical/scientific tokenizer.

    Keeps terms such as:
        IDH1
        MGMT
        EGFR
        MRI
        T1-weighted
        T2-weighted
    """

    return re.findall(
        r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*",
        text.lower(),
    )


# ---------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------

class HybridPubMedRetriever:

    def __init__(
        self,
        faiss_top_k: int = FAISS_TOP_K,
        bm25_top_k: int = BM25_TOP_K,
        final_top_k: int = FINAL_TOP_K,
    ):

        self.faiss_top_k = faiss_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k

        # -------------------------------------------------------------
        # Load configuration
        # -------------------------------------------------------------

        with CONFIG_FILE.open("rb") as file:
            self.config = pickle.load(file)

        # -------------------------------------------------------------
        # Load FAISS
        # -------------------------------------------------------------

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        # -------------------------------------------------------------
        # Load metadata
        # -------------------------------------------------------------

        with METADATA_FILE.open("rb") as file:
            self.metadata = pickle.load(file)

        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                "FAISS index and metadata counts do not match."
            )

        # -------------------------------------------------------------
        # Load embedding model
        # -------------------------------------------------------------

        model_name = self.config[
            "embedding_model"
        ]

        print(
            f"Loading embedding model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name,
            device="cpu",
        )

        # -------------------------------------------------------------
        # Build BM25 corpus
        # -------------------------------------------------------------

        print("Building BM25 index...")

        corpus = [
            tokenize(
                item["page_content"]
            )
            for item in self.metadata
        ]

        self.bm25 = BM25Okapi(
            corpus
        )

        print("Hybrid retriever ready.")

    # -----------------------------------------------------------------
    # FAISS search
    # -----------------------------------------------------------------

    def faiss_search(
        self,
        query: str,
    ) -> list[dict]:

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = query_embedding.astype(
            "float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            self.faiss_top_k,
        )

        results = []

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):

            if idx < 0:
                continue

            results.append(
                {
                    "faiss_id": int(idx),
                    "rank": rank,
                    "score": float(score),
                }
            )

        return results

    # -----------------------------------------------------------------
    # BM25 search
    # -----------------------------------------------------------------

    def bm25_search(
        self,
        query: str,
    ) -> list[dict]:

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for rank, idx in enumerate(
            ranked_indices[
                : self.bm25_top_k
            ],
            start=1,
        ):

            results.append(
                {
                    "faiss_id": int(idx),
                    "rank": rank,
                    "score": float(scores[idx]),
                }
            )

        return results

    # -----------------------------------------------------------------
    # Reciprocal Rank Fusion
    # -----------------------------------------------------------------

    def reciprocal_rank_fusion(
        self,
        faiss_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:

        fused_scores = {}

        # -------------------------------------------------------------
        # FAISS contribution
        # -------------------------------------------------------------

        for result in faiss_results:

            idx = result["faiss_id"]

            fused_scores.setdefault(
                idx,
                {
                    "rrf_score": 0.0,
                    "faiss_rank": None,
                    "bm25_rank": None,
                    "faiss_score": None,
                    "bm25_score": None,
                },
            )

            fused_scores[idx][
                "rrf_score"
            ] += 1 / (
                RRF_K + result["rank"]
            )

            fused_scores[idx][
                "faiss_rank"
            ] = result["rank"]

            fused_scores[idx][
                "faiss_score"
            ] = result["score"]

        # -------------------------------------------------------------
        # BM25 contribution
        # -------------------------------------------------------------

        for result in bm25_results:

            idx = result["faiss_id"]

            fused_scores.setdefault(
                idx,
                {
                    "rrf_score": 0.0,
                    "faiss_rank": None,
                    "bm25_rank": None,
                    "faiss_score": None,
                    "bm25_score": None,
                },
            )

            fused_scores[idx][
                "rrf_score"
            ] += 1 / (
                RRF_K + result["rank"]
            )

            fused_scores[idx][
                "bm25_rank"
            ] = result["rank"]

            fused_scores[idx][
                "bm25_score"
            ] = result["score"]

        # -------------------------------------------------------------
        # Sort by fused score
        # -------------------------------------------------------------

        ranked = sorted(
            fused_scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True,
        )

        results = []

        for idx, fusion_data in ranked[
            : self.final_top_k
        ]:

            item = self.metadata[idx].copy()

            item.update(
                {
                    "faiss_id": idx,
                    **fusion_data,
                }
            )

            results.append(item)

        return results

    # -----------------------------------------------------------------
    # Public search method
    # -----------------------------------------------------------------

    def search(
        self,
        query: str,
    ) -> list[dict]:

        faiss_results = self.faiss_search(
            query
        )

        bm25_results = self.bm25_search(
            query
        )

        return self.reciprocal_rank_fusion(
            faiss_results,
            bm25_results,
        )


# ---------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------

def display_results(
    query: str,
    results: list[dict],
):

    print("\n" + "=" * 90)
    print("HYBRID RETRIEVAL")
    print("=" * 90)

    print(f"\nQuery:")
    print(query)

    for i, result in enumerate(
        results,
        start=1,
    ):

        print("\n" + "-" * 90)

        print(f"Result #{i}")
        print(
            f"RRF score:    "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"FAISS rank:   "
            f"{result['faiss_rank']}"
        )

        print(
            f"BM25 rank:    "
            f"{result['bm25_rank']}"
        )

        print(
            f"FAISS score:  "
            f"{result['faiss_score']}"
        )

        print(
            f"BM25 score:   "
            f"{result['bm25_score']}"
        )

        print(
            f"PMID:         "
            f"{result['pmid']}"
        )

        print(
            f"Title:        "
            f"{result['title']}"
        )

        print(
            f"Year:         "
            f"{result['year']}"
        )

        print(
            f"DOI:          "
            f"{result['doi']}"
        )

        print(
            f"Topic:        "
            f"{result['topic']}"
        )

        print("\nEvidence:")

        content = result[
            "page_content"
        ]

        if len(content) > 800:
            content = content[:800] + "..."

        print(content)


# ---------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------

TEST_QUERIES = [

    "What MRI characteristics are associated with glioblastoma?",

    "How is deep learning used for brain tumor segmentation?",

    "What is the role of radiomics in glioma diagnosis?",

    "How can MRI be used to predict treatment response in glioblastoma?",

    "What imaging features are associated with IDH1 MGMT EGFR and other glioma molecular characteristics?",
]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("\n" + "=" * 90)
    print("PUBMED HYBRID RETRIEVER")
    print("=" * 90)

    retriever = HybridPubMedRetriever()

    for query in TEST_QUERIES:

        results = retriever.search(
            query
        )

        display_results(
            query,
            results,
        )

    print("\n" + "=" * 90)
    print("HYBRID RETRIEVAL TEST COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()