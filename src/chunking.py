"""
LangChain Document Chunking

Input:
    data/processed/langchain_documents.pkl

Output:
    data/processed/chunks.pkl
"""

from __future__ import annotations

import pickle
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "langchain_documents.pkl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.pkl"
)


# ---------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def load_documents():
    """Load LangChain Documents."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    with INPUT_FILE.open("rb") as file:
        documents = pickle.load(file)

    return documents


def create_chunks(documents):
    """Split documents into retrieval-friendly chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
        ],
    )

    chunks = splitter.split_documents(documents)

    return chunks


def save_chunks(chunks):
    """Save chunked LangChain Documents."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open("wb") as file:
        pickle.dump(
            chunks,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("\nChunks saved to:")
    print(OUTPUT_FILE)


def main():

    print("\n" + "=" * 70)
    print("LANGCHAIN DOCUMENT CHUNKING")
    print("=" * 70)

    documents = load_documents()

    print(f"\nDocuments loaded: {len(documents):,}")

    chunks = create_chunks(documents)

    print(f"Chunks created:    {len(chunks):,}")

    if documents:
        print(
            f"Average chunks/document: "
            f"{len(chunks) / len(documents):.2f}"
        )

    save_chunks(chunks)

    # -------------------------------------------------------------
    # Show examples
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE CHUNKS")
    print("=" * 70)

    for i, chunk in enumerate(chunks[:3], start=1):

        print(f"\n--- Chunk {i} ---")

        print("PMID:", chunk.metadata.get("pmid"))
        print("Topic:", chunk.metadata.get("topic"))
        print("Year:", chunk.metadata.get("year"))

        print("\nContent:")
        print(chunk.page_content[:800])

    print("\n" + "=" * 70)
    print("CHUNKING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()