"""
CPU-optimized PubMed embedding + FAISS vector store.

Input:
    data/processed/chunks.pkl

Output:
    data/vector_db/
        index.faiss
        metadata.pkl
        config.pkl
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.pkl"
)

VECTOR_DB_DIR = (
    PROJECT_ROOT
    / "data"
    / "vector_db"
)

EMBEDDINGS_FILE = (
    VECTOR_DB_DIR
    / "embeddings.npy"
)

INDEX_FILE = (
    VECTOR_DB_DIR
    / "index.faiss"
)

METADATA_FILE = (
    VECTOR_DB_DIR
    / "metadata.pkl"
)

CONFIG_FILE = (
    VECTOR_DB_DIR
    / "config.pkl"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Your machine has 8 CPU cores.
CPU_THREADS = 8

# Start conservatively.
BATCH_SIZE = 16

# Save progress after every N batches.
CHECKPOINT_EVERY = 25


# ---------------------------------------------------------------------
# CPU configuration
# ---------------------------------------------------------------------

def configure_cpu():
    """Configure PyTorch for CPU execution."""

    torch.set_num_threads(CPU_THREADS)

    try:
        torch.set_num_interop_threads(
            min(4, CPU_THREADS)
        )
    except RuntimeError:
        pass

    print(f"CPU threads: {torch.get_num_threads()}")


# ---------------------------------------------------------------------
# Load chunks
# ---------------------------------------------------------------------

def load_chunks():
    """Load LangChain chunks."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found:\n{INPUT_FILE}"
        )

    with INPUT_FILE.open("rb") as file:
        chunks = pickle.load(file)

    return chunks


# ---------------------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------------------

def load_model():

    print("\nLoading embedding model:")
    print(MODEL_NAME)

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print("Embedding model loaded.")

    return model


# ---------------------------------------------------------------------
# Create metadata
# ---------------------------------------------------------------------

def create_metadata(chunks):

    metadata = []

    for chunk in chunks:

        metadata.append(
            {
                "pmid": chunk.metadata.get(
                    "pmid",
                    "",
                ),
                "title": chunk.metadata.get(
                    "title",
                    "",
                ),
                "journal": chunk.metadata.get(
                    "journal",
                    "",
                ),
                "year": chunk.metadata.get(
                    "year",
                    "",
                ),
                "doi": chunk.metadata.get(
                    "doi",
                    "",
                ),
                "pmcid": chunk.metadata.get(
                    "pmcid",
                    "",
                ),
                "topic": chunk.metadata.get(
                    "topic",
                    "",
                ),
                "publication_status": chunk.metadata.get(
                    "publication_status",
                    "",
                ),
                "source_file": chunk.metadata.get(
                    "source_file",
                    "",
                ),
                "source": chunk.metadata.get(
                    "source",
                    "PubMed",
                ),
                "page_content": chunk.page_content,
            }
        )

    return metadata


# ---------------------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------------------

def create_embeddings(model, chunks):

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    total = len(texts)

    # -------------------------------------------------------------
    # Resume support
    # -------------------------------------------------------------

    if EMBEDDINGS_FILE.exists():

        existing = np.load(
            EMBEDDINGS_FILE,
        )

        if (
            len(existing) < total
            and existing.ndim == 2
        ):
            print(
                f"\nFound checkpoint:"
                f" {len(existing):,}/{total:,}"
            )

            embeddings = existing.tolist()

            start_index = len(existing)

        else:
            print("\nExisting embedding file found.")

            if len(existing) == total:
                return existing.astype(
                    "float32"
                )

            # Incompatible checkpoint.
            print(
                "Existing checkpoint is incompatible."
            )

            EMBEDDINGS_FILE.unlink()

            embeddings = []
            start_index = 0

    else:

        embeddings = []
        start_index = 0

    # -------------------------------------------------------------
    # Generate remaining embeddings
    # -------------------------------------------------------------

    print("\nCreating embeddings...")
    print(f"Total chunks:   {total:,}")
    print(f"Starting from:  {start_index:,}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"CPU threads:    {CPU_THREADS}")

    for start in range(
        start_index,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch_texts = texts[start:end]

        batch_embeddings = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        batch_embeddings = (
            batch_embeddings.astype(
                "float32"
            )
        )

        embeddings.extend(
            batch_embeddings.tolist()
        )

        completed = end

        print(
            f"\rEmbedded "
            f"{completed:,}/{total:,} "
            f"({completed / total * 100:.1f}%)",
            end="",
            flush=True,
        )

        # ---------------------------------------------------------
        # Checkpoint
        # ---------------------------------------------------------

        batch_number = (
            start // BATCH_SIZE
        ) + 1

        if (
            batch_number % CHECKPOINT_EVERY == 0
            or completed == total
        ):

            checkpoint_array = np.asarray(
                embeddings,
                dtype="float32",
            )

            np.save(
                EMBEDDINGS_FILE,
                checkpoint_array,
            )

            print(
                f"\nCheckpoint saved: "
                f"{completed:,}/{total:,}"
            )

    print()

    return np.asarray(
        embeddings,
        dtype="float32",
    )


# ---------------------------------------------------------------------
# Build FAISS
# ---------------------------------------------------------------------

def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    print("\nBuilding FAISS index...")
    print(f"Embedding dimension: {dimension}")

    # Normalized embeddings + inner product
    # = cosine similarity.
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ---------------------------------------------------------------------
# Save vector database
# ---------------------------------------------------------------------

def save_vector_store(
    index,
    metadata,
    dimension,
):

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # FAISS
    faiss.write_index(
        index,
        str(INDEX_FILE),
    )

    # Metadata
    with METADATA_FILE.open(
        "wb"
    ) as file:

        pickle.dump(
            metadata,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Configuration
    config = {
        "embedding_model": MODEL_NAME,
        "embedding_dimension": dimension,
        "index_type": "FAISS IndexFlatIP",
        "similarity": "cosine",
        "normalized_embeddings": True,
        "chunk_count": len(metadata),
        "device": "cpu",
        "cpu_threads": CPU_THREADS,
    }

    with CONFIG_FILE.open(
        "wb"
    ) as file:

        pickle.dump(
            config,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("\nVector store saved:")
    print(f"  {INDEX_FILE}")
    print(f"  {METADATA_FILE}")
    print(f"  {CONFIG_FILE}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("PUBMED CPU VECTOR STORE")
    print("=" * 70)

    configure_cpu()

    chunks = load_chunks()

    print(
        f"\nChunks loaded: "
        f"{len(chunks):,}"
    )

    model = load_model()

    embeddings = create_embeddings(
        model,
        chunks,
    )

    print(
        f"\nFinal embedding matrix: "
        f"{embeddings.shape}"
    )

    # Safety check
    if len(embeddings) != len(chunks):

        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    index = build_faiss_index(
        embeddings
    )

    print(
        f"FAISS vectors: "
        f"{index.ntotal:,}"
    )

    metadata = create_metadata(
        chunks
    )

    save_vector_store(
        index,
        metadata,
        embeddings.shape[1],
    )

    print("\n" + "=" * 70)
    print("VECTOR STORE CREATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()