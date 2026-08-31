"""
PubMed CSV -> LangChain Documents

Input:
    data/processed/pubmed_clean.csv

Output:
    data/processed/langchain_documents.pkl
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

from langchain_core.documents import Document


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pubmed_clean.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "langchain_documents.pkl"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MIN_ABSTRACT_LENGTH = 50


# ---------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------

def load_pubmed_records() -> list[dict]:
    """Load validated PubMed records from CSV."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        records = list(reader)

    return records


# ---------------------------------------------------------------------
# Convert to LangChain Documents
# ---------------------------------------------------------------------

def create_documents(
    records: list[dict],
) -> list[Document]:
    """
    Convert PubMed records into LangChain Documents.

    Page content:
        Title + Abstract

    Metadata:
        PMID, journal, year, DOI, PMCID, topic, etc.
    """

    documents = []

    skipped = 0

    for record in records:

        pmid = record.get("PMID", "").strip()
        title = record.get("Title", "").strip()
        abstract = record.get("Abstract", "").strip()

        if not pmid:
            skipped += 1
            continue

        if not title:
            skipped += 1
            continue

        if len(abstract) < MIN_ABSTRACT_LENGTH:
            skipped += 1
            continue

        # -------------------------------------------------------------
        # Searchable content
        # -------------------------------------------------------------

        page_content = (
            f"Title: {title}\n\n"
            f"Abstract: {abstract}"
        )

        # -------------------------------------------------------------
        # Metadata
        # -------------------------------------------------------------

        metadata = {
            "pmid": pmid,
            "title": title,
            "journal": record.get("Journal", "").strip(),
            "year": record.get("Year", "").strip(),
            "doi": record.get("DOI", "").strip(),
            "pmcid": record.get("PMCID", "").strip(),
            "topic": record.get("Topic", "").strip(),
            "publication_status": record.get(
                "Publication_Status",
                "",
            ).strip(),
            "source_file": record.get(
                "Source_File",
                "",
            ).strip(),
            "source": "PubMed",
        }

        document = Document(
            page_content=page_content,
            metadata=metadata,
        )

        documents.append(document)

    print(f"Records loaded:       {len(records):,}")
    print(f"Documents created:    {len(documents):,}")
    print(f"Records skipped:      {skipped:,}")

    return documents


# ---------------------------------------------------------------------
# Save Documents
# ---------------------------------------------------------------------

def save_documents(
    documents: list[Document],
) -> None:
    """Save LangChain documents."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open("wb") as file:
        pickle.dump(
            documents,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("\nDocuments saved to:")
    print(OUTPUT_FILE)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("\n" + "=" * 70)
    print("PUBMED → LANGCHAIN INGESTION")
    print("=" * 70)

    records = load_pubmed_records()

    documents = create_documents(records)

    save_documents(documents)

    # -------------------------------------------------------------
    # Show example
    # -------------------------------------------------------------

    if documents:

        example = documents[0]

        print("\n" + "=" * 70)
        print("EXAMPLE LANGCHAIN DOCUMENT")
        print("=" * 70)

        print("\nMetadata:")
        for key, value in example.metadata.items():
            print(f"{key}: {value}")

        print("\nPage content:")
        print(example.page_content[:1000])

    print("\n" + "=" * 70)
    print("INGESTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()