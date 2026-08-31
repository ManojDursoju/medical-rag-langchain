"""
Validate the cleaned PubMed dataset.

Input:
    data/processed/pubmed_clean.csv

Checks:
    - Required columns
    - Total records
    - Duplicate PMIDs
    - Missing values
    - Retracted/withdrawn records
    - Abstract quality
    - Topic distribution
    - Year distribution
"""

from pathlib import Path
import csv
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_ROOT / "data" / "processed" / "pubmed_clean.csv"


REQUIRED_COLUMNS = [
    "PMID",
    "Title",
    "Abstract",
    "Journal",
    "Year",
    "DOI",
    "PMCID",
    "Topic",
    "Publication_Status",
    "Source_File",
]


def main():

    print("\n" + "=" * 70)
    print("PUBMED CLEAN DATASET VALIDATION")
    print("=" * 70)

    if not CSV_FILE.exists():
        print(f"\nERROR: File not found:")
        print(CSV_FILE)
        return

    print(f"\nFile:")
    print(CSV_FILE)

    with CSV_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        columns = reader.fieldnames or []

        print("\nColumns:")
        for column in columns:
            print(f"  ✓ {column}")

        # ---------------------------------------------------------
        # Column validation
        # ---------------------------------------------------------

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in columns
        ]

        if missing_columns:
            print("\n❌ Missing required columns:")
            for column in missing_columns:
                print(f"  - {column}")
            return

        print("\n✓ All required columns are present.")

        rows = list(reader)

    total = len(rows)

    print(f"\nTotal records: {total:,}")

    # -------------------------------------------------------------
    # PMID validation
    # -------------------------------------------------------------

    pmids = [
        row["PMID"].strip()
        for row in rows
        if row["PMID"].strip()
    ]

    duplicate_pmids = [
        pmid
        for pmid, count in Counter(pmids).items()
        if count > 1
    ]

    print("\nPMID validation")
    print("-" * 40)
    print(f"Records with PMID:      {len(pmids):,}")
    print(f"Unique PMIDs:           {len(set(pmids)):,}")
    print(f"Duplicate PMID values:  {len(duplicate_pmids):,}")

    if duplicate_pmids:
        print("❌ Duplicate PMIDs found.")
    else:
        print("✓ No duplicate PMIDs.")

    # -------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------

    print("\nMissing-value validation")
    print("-" * 40)

    for column in REQUIRED_COLUMNS:
        missing = sum(
            1
            for row in rows
            if not row[column].strip()
        )

        percentage = (
            missing / total * 100
            if total
            else 0
        )

        print(
            f"{column:22} {missing:6,} "
            f"({percentage:.2f}%)"
        )

    # -------------------------------------------------------------
    # Abstract quality
    # -------------------------------------------------------------

    abstract_lengths = [
        len(row["Abstract"].strip())
        for row in rows
    ]

    short_abstracts = sum(
        1
        for length in abstract_lengths
        if length < 50
    )

    good_abstracts = sum(
        1
        for length in abstract_lengths
        if length >= 50
    )

    print("\nAbstract quality")
    print("-" * 40)
    print(f"Valid abstracts (>=50 chars): {good_abstracts:,}")
    print(f"Short abstracts (<50 chars):  {short_abstracts:,}")

    if abstract_lengths:
        print(
            f"Shortest abstract:           "
            f"{min(abstract_lengths)} chars"
        )
        print(
            f"Longest abstract:            "
            f"{max(abstract_lengths)} chars"
        )
        print(
            f"Average abstract length:     "
            f"{sum(abstract_lengths) / len(abstract_lengths):.0f} chars"
        )

    # -------------------------------------------------------------
    # Publication status
    # -------------------------------------------------------------

    status_counts = Counter(
        row["Publication_Status"].strip()
        for row in rows
    )

    print("\nPublication status")
    print("-" * 40)

    for status, count in status_counts.most_common():
        print(f"{status:25} {count:,}")

    # -------------------------------------------------------------
    # Topic distribution
    # -------------------------------------------------------------

    topic_counts = Counter(
        row["Topic"].strip()
        for row in rows
    )

    print("\nTopic distribution")
    print("-" * 40)

    for topic, count in topic_counts.most_common():
        print(f"{topic:45} {count:,}")

    # -------------------------------------------------------------
    # Year distribution
    # -------------------------------------------------------------

    year_counts = Counter(
        row["Year"].strip()
        for row in rows
        if row["Year"].strip()
    )

    print("\nPublication years")
    print("-" * 40)

    for year, count in sorted(
        year_counts.items(),
        reverse=True
    )[:15]:

        print(f"{year}: {count:,}")

    # -------------------------------------------------------------
    # Sample records
    # -------------------------------------------------------------

    print("\nSample records")
    print("=" * 70)

    for index, row in enumerate(rows[:3], start=1):

        print(f"\nRecord {index}")
        print(f"PMID:     {row['PMID']}")
        print(f"Title:    {row['Title']}")
        print(f"Journal:  {row['Journal']}")
        print(f"Year:     {row['Year']}")
        print(f"DOI:      {row['DOI']}")
        print(f"Topic:    {row['Topic']}")
        print(
            "Abstract: "
            + row["Abstract"][:500]
            + ("..." if len(row["Abstract"]) > 500 else "")
        )

    # -------------------------------------------------------------
    # Final assessment
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    checks = []

    checks.append(
        ("Required columns", not missing_columns)
    )

    checks.append(
        ("PMID completeness", len(pmids) == total)
    )

    checks.append(
        ("No duplicate PMIDs", len(duplicate_pmids) == 0)
    )

    checks.append(
        ("Abstract quality", short_abstracts == 0)
    )

    bad_statuses = {
        "RETRACTED",
        "WITHDRAWN",
    }

    remaining_bad = sum(
        count
        for status, count in status_counts.items()
        if status in bad_statuses
    )

    checks.append(
        ("No retracted/withdrawn records", remaining_bad == 0)
    )

    all_passed = True

    for name, passed in checks:

        if passed:
            print(f"✓ {name}")
        else:
            print(f"❌ {name}")
            all_passed = False

    if all_passed:
        print("\n🎉 DATASET VALIDATION PASSED")
        print("The dataset is ready for the LangChain ingestion stage.")
    else:
        print("\n⚠ DATASET NEEDS ATTENTION")
        print("Do not build the vector database yet.")

    print("=" * 70)


if __name__ == "__main__":
    main()