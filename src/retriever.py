from __future__ import annotations

import pickle
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB = PROJECT_ROOT / "data" / "vector_db"

INDEX_FILE = VECTOR_DB / "index.faiss"
METADATA_FILE = VECTOR_DB / "metadata.pkl"
CONFIG_FILE = VECTOR_DB / "config.pkl"


# ============================================================
# Medical concept groups
# ============================================================

CONCEPTS = {
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
        "dwi",
        "adc",
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
        "feature extraction",
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
        "chemotherapy",
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

    text = text.replace(
        "\u2013",
        "-",
    )

    text = text.replace(
        "\u2014",
        "-",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# Concept detection
# ============================================================

def detect_query_concepts(
    query: str,
) -> set[str]:

    query = normalize(query)

    detected = set()

    for concept, terms in CONCEPTS.items():

        for term in terms:

            if term in query:
                detected.add(concept)
                break

    return detected


# ============================================================
# Text concept detection
# ============================================================

def detect_text_concepts(
    text: str,
) -> set[str]:

    text = normalize(text)

    detected = set()

    for concept, terms in CONCEPTS.items():

        for term in terms:

            if term in text:
                detected.add(concept)
                break

    return detected


# ============================================================
# Query-document compatibility
# ============================================================

def calculate_domain_score(
    query: str,
    title: str,
    content: str,
) -> tuple[float, set[str], set[str]]:

    query_concepts = detect_query_concepts(
        query
    )

    title_text = normalize(title)
    content_text = normalize(content)

    # Title receives more importance.
    title_concepts = detect_text_concepts(
        title_text
    )

    content_concepts = detect_text_concepts(
        content_text
    )

    if not query_concepts:
        return (
            0.5,
            set(),
            content_concepts,
        )

    matched = 0

    for concept in query_concepts:

        if concept in title_concepts:

            matched += 1

        elif concept in content_concepts:

            matched += 0.5

    score = matched / len(
        query_concepts
    )

    return (
        min(score, 1.0),
        query_concepts,
        content_concepts,
    )


# ============================================================
# Obvious noise detection
# ============================================================

NON_BRAIN_DOMAINS = [
    "rotator cuff",
    "shoulder",
    "hip",
    "rectal cancer",
    "renal cell carcinoma",
    "lung cancer",
    "nsclc",
    "prostate cancer",
    "breast cancer",
    "colorectal cancer",
]


def is_obvious_noise(
    query: str,
    title: str,
    content: str,
) -> bool:

    query_concepts = detect_query_concepts(
        query
    )

    text = normalize(
        title + " " + content[:2500]
    )

    # If query is explicitly brain/glioma-related,
    # reject clearly unrelated anatomical/cancer domains.
    brain_query = bool(
        query_concepts
        & {
            "glioblastoma",
            "glioma",
            "brain",
        }
    )

    if not brain_query:
        return False

    for domain in NON_BRAIN_DOMAINS:

        if domain in text:

            # Allow it only when brain/glioma
            # evidence is also explicitly present.
            brain_hits = 0

            for term in (
                CONCEPTS["glioblastoma"]
                + CONCEPTS["glioma"]
                + CONCEPTS["brain"]
            ):

                if term in text:
                    brain_hits += 1

            if brain_hits == 0:
                return True

    return False


# ============================================================
# Retriever
# ============================================================

class MedicalRetriever:

    def __init__(self):

        print("=" * 80)
        print("PUBMED SEMANTIC RETRIEVER")
        print("=" * 80)

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        with METADATA_FILE.open(
            "rb"
        ) as f:
            self.metadata = pickle.load(f)

        with CONFIG_FILE.open(
            "rb"
        ) as f:
            self.config = pickle.load(f)

        model_name = self.config[
            "embedding_model"
        ]

        print(
            f"Loading embedding model: "
            f"{model_name}"
        )

        self.embedding_model = (
            SentenceTransformer(
                model_name,
                device="cpu",
            )
        )

        print(
            f"FAISS vectors: "
            f"{self.index.ntotal}"
        )

        print("Retriever ready.")

    # ========================================================
    # Retrieve
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
    ) -> list[dict]:

        embedding = (
            self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            .astype("float32")
        )

        # Retrieve a larger pool first.
        search_k = min(
            max(top_k * 4, 60),
            self.index.ntotal,
        )

        scores, indices = (
            self.index.search(
                embedding,
                search_k,
            )
        )

        candidates = []

        for rank, (
            similarity,
            idx,
        ) in enumerate(
            zip(
                scores[0],
                indices[0],
            ),
            start=1,
        ):

            if idx < 0:
                continue

            item = self.metadata[
                int(idx)
            ].copy()

            title = item.get(
                "title",
                "",
            )

            content = item.get(
                "page_content",
                "",
            )

            domain_score, query_concepts, text_concepts = (
                calculate_domain_score(
                    query,
                    title,
                    content,
                )
            )

            noise = is_obvious_noise(
                query,
                title,
                content,
            )

            item["faiss_score"] = float(
                similarity
            )

            item["faiss_rank"] = rank

            item["faiss_id"] = int(
                idx
            )

            item["domain_score"] = (
                float(domain_score)
            )

            item["query_concepts"] = sorted(
                query_concepts
            )

            item["text_concepts"] = sorted(
                text_concepts
            )

            item["obvious_noise"] = noise

            # ----------------------------------------------------
            # Final retrieval score
            # ----------------------------------------------------

            item["retrieval_score"] = (
                0.65 * float(similarity)
                + 0.35 * float(domain_score)
            )

            # Explicitly penalize obvious unrelated papers.
            if noise:
                item["retrieval_score"] -= 0.50

            candidates.append(item)

        candidates.sort(
            key=lambda x: x[
                "retrieval_score"
            ],
            reverse=True,
        )

        # Remove obvious noise.
        filtered = [
            x
            for x in candidates
            if not x["obvious_noise"]
        ]

        # If filtering became too aggressive,
        # retain the strongest candidates.
        if len(filtered) < top_k:
            filtered = candidates

        return filtered[:top_k]


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
        "RETRIEVAL RESULTS"
    )

    print(
        "=" * 90
    )

    print(
        f"\nQuery:\n{query}"
    )

    for i, result in enumerate(
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
            f"Retrieval score: "
            f"{result['retrieval_score']:.4f}"
        )

        print(
            f"FAISS similarity: "
            f"{result['faiss_score']:.4f}"
        )

        print(
            f"Domain score:    "
            f"{result['domain_score']:.4f}"
        )

        print(
            f"PMID:            "
            f"{result['pmid']}"
        )

        print(
            f"Title:           "
            f"{result['title']}"
        )

        print(
            f"Journal:         "
            f"{result['journal']}"
        )

        print(
            f"Year:            "
            f"{result['year']}"
        )

        print(
            f"DOI:             "
            f"{result['doi']}"
        )

        print(
            f"Topic:           "
            f"{result['topic']}"
        )

        print(
            f"Concepts:        "
            f"{', '.join(result['text_concepts'])}"
        )

        print(
            "\nEvidence:"
        )

        evidence = result[
            "page_content"
        ]

        if len(evidence) > 1200:
            evidence = (
                evidence[:1200]
                + "..."
            )

        print(evidence)


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

    for query in TEST_QUERIES:

        results = retriever.retrieve(
            query,
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
        "RETRIEVAL TEST COMPLETED"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()