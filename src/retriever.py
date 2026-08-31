"""
PubMed FAISS Retriever

Loads:
    data/vector_db/index.faiss
    data/vector_db/metadata.pkl
    data/vector_db/config.pkl

Uses the same embedding model to perform semantic search.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
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
# Retriever
# ---------------------------------------------------------------------

class PubMedRetriever:

    def __init__(self, top_k: int = 5):

        self.top_k = top_k

        # -------------------------------------------------------------
        # Load configuration
        # -------------------------------------------------------------

        if not CONFIG_FILE.exists():
            raise FileNotFoundError(
                f"Vector configuration not found:\n{CONFIG_FILE}"
            )

        with CONFIG_FILE.open("rb") as file:
            self.config = pickle.load(file)

        model_name = self.config["embedding_model"]

        # -------------------------------------------------------------
        # Load FAISS
        # -------------------------------------------------------------

        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found:\n{INDEX_FILE}"
            )

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        # -------------------------------------------------------------
        # Load metadata
        # -------------------------------------------------------------

        if not METADATA_FILE.exists():
            raise FileNotFoundError(
                f"Metadata file not found:\n{METADATA_FILE}"
            )

        with METADATA_FILE.open("rb") as file:
            self.metadata = pickle.load(file)

        # -------------------------------------------------------------
        # Load embedding model
        # -------------------------------------------------------------

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(
            model_name,
            device="cpu",
        )

        print("Retriever ready.")

        # -------------------------------------------------------------
        # Safety checks
        # -------------------------------------------------------------

        if self.index.ntotal != len(self.metadata):

            raise RuntimeError(
                "FAISS index count does not match metadata count."
            )

    # -----------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:

        if not query.strip():
            return []

        k = top_k or self.top_k

        # Don't request more vectors than exist.
        k = min(
            k,
            self.index.ntotal,
        )

        # -------------------------------------------------------------
        # Embed query
        # -------------------------------------------------------------

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # FAISS expects float32.
        query_embedding = query_embedding.astype(
            "float32"
        )

        # -------------------------------------------------------------
        # Search
        # -------------------------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0],
        ):

            if index_id < 0:
                continue

            item = self.metadata[index_id].copy()

            item["score"] = float(score)
            item["faiss_id"] = int(index_id)

            results.append(item)

        return results


# ---------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------

def display_results(
    query: str,
    results: list[dict],
) -> None:

    print("\n" + "=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)

    print(f"\nQuery:")
    print(query)

    if not results:
        print("\nNo results found.")
        return

    for i, result in enumerate(
        results,
        start=1,
    ):

        print("\n" + "-" * 80)

        print(f"Result #{i}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"PMID:      {result['pmid']}")
        print(f"Title:     {result['title']}")
        print(f"Journal:   {result['journal']}")
        print(f"Year:      {result['year']}")
        print(f"DOI:       {result['doi']}")
        print(f"Topic:     {result['topic']}")

        print("\nEvidence:")

        content = result["page_content"]

        if len(content) > 1000:
            content = content[:1000] + "..."

        print(content)


# ---------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------

TEST_QUERIES = [
    "What MRI characteristics are associated with glioblastoma?",
    "How is deep learning used for brain tumor segmentation?",
    "What is the role of radiomics in glioma diagnosis?",
    "How can MRI be used to predict treatment response in glioblastoma?",
    "What imaging features are associated with glioma molecular characteristics?",
]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("\n" + "=" * 80)
    print("PUBMED SEMANTIC RETRIEVER")
    print("=" * 80)

    retriever = PubMedRetriever(
        top_k=5
    )

    for query in TEST_QUERIES:

        results = retriever.search(
            query,
            top_k=5,
        )

        display_results(
            query,
            results,
        )

    print("\n" + "=" * 80)
    print("RETRIEVAL TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()