"""
PubMed raw TXT -> cleaned CSV pipeline

Project:
Medical Research RAG & LLM Copilot

Input:
    data/raw/*.txt

Output:
    data/processed/pubmed_clean.csv
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "pubmed_clean.csv"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REQUIRED_FIELDS = [
    "PMID",
    "Title",
    "Abstract",
]

RETRACTED_MARKERS = [
    "RETRACTED ARTICLE",
    "RETRACTED PUBLICATION",
    "WITHDRAWN ARTICLE",
    "WITHDRAWN PUBLICATION",
]

PROBLEM_MARKERS = [
    "EXPRESSION OF CONCERN",
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Normalize spaces and line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Clean extracted text while preserving readable content."""
    text = normalize_whitespace(text)
    return text.strip()


def extract_pmid(record: str) -> Optional[str]:
    """Extract PubMed ID."""
    match = re.search(
        r"\bPMID:\s*(\d+)",
        record,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None


def extract_doi(record: str) -> str:
    """Extract DOI."""
    match = re.search(
        r"\bDOI:\s*(10\.\S+)",
        record,
        flags=re.IGNORECASE,
    )

    if not match:
        return ""

    doi = match.group(1).strip()

    # Remove trailing punctuation commonly present in exported text.
    doi = doi.rstrip(".,;)")

    return doi


def extract_title(record: str) -> str:
    """Extract the article title from a PubMed TXT record."""

    lines = record.splitlines()

    header_found = False
    metadata_finished = False

    for i, raw_line in enumerate(lines):

        line = raw_line.strip()

        if not line:
            if header_found:
                metadata_finished = True
            continue

        # Publication/journal header
        if re.match(
            r"^\d+\.\s+.+\.\s+(?:19|20)\d{2}\b",
            line
        ):
            header_found = True
            metadata_finished = False
            continue

        if not header_found:
            continue

        # DOI / Epub metadata belongs to the publication header.
        if re.search(
            r"\bdoi:\s*|Epub\s+\d{4}",
            line,
            flags=re.IGNORECASE
        ):
            metadata_finished = False
            continue

        # After the metadata blank line, the first valid
        # non-metadata line is the title.
        if metadata_finished:

            # Reject dates.
            if re.fullmatch(
                r"\d{4}\s+[A-Za-z]{3}\s+\d{1,2}\.?",
                line
            ):
                metadata_finished = False
                continue

            # Reject DOI-only lines.
            if re.match(
                r"^10\.\d{4,9}/",
                line
            ):
                continue

            # Reject author-information markers.
            if line.startswith(
                (
                    "Author information:",
                    "Authors:",
                )
            ):
                return ""

            return clean_text(line)

    return ""


def extract_journal(record: str) -> str:
    """Extract journal name from PubMed publication header."""

    lines = record.splitlines()

    for line in lines:

        line = line.strip()

        match = re.match(
            r"^\d+\.\s+(.+?)\.\s+(?:19|20)\d{2}\b",
            line
        )

        if match:

            journal = match.group(1).strip()

            # Remove PubMed numbering if present.
            journal = re.sub(
                r"^\d+\.\s*",
                "",
                journal
            )

            return clean_text(journal)

    return ""


def extract_year(record: str) -> str:
    """Extract publication year from PubMed publication header."""

    lines = record.splitlines()

    for line in lines:

        line = line.strip()

        # Example:
        # 2. J Digit Imaging. 2023 Oct;36(5):2075-2087.
        match = re.search(
            r"\b((?:19|20)\d{2})\b",
            line
        )

        if re.match(
            r"^\d+\.\s+",
            line
        ) and match:

            return match.group(1)

    return ""


def extract_abstract(record: str) -> str:
    """
    Extract the complete scientific abstract from a PubMed TXT record.

    Expected PubMed structure:

        PMID:
        Journal/date header
        DOI/Epub metadata

        Article title

        Authors

        Author information:
        affiliations...

        Abstract text...

        PMID: ...
    """

    lines = record.splitlines()

    # ----------------------------------------------------------
    # 1. Locate Author information
    # ----------------------------------------------------------

    author_info_index = -1

    for i, line in enumerate(lines):

        if line.strip().lower() == "author information:":
            author_info_index = i
            break

    if author_info_index == -1:
        return ""

    # ----------------------------------------------------------
    # 2. Find the end of the affiliation section
    # ----------------------------------------------------------

    abstract_start = -1

    for i in range(
        author_info_index + 1,
        len(lines)
    ):

        line = lines[i].strip()

        if not line:
            continue

        # Affiliation entries:
        # (1)Department...
        # (2)University...
        if re.match(r"^\(\d+\)", line):
            continue

        # Continuation of affiliations.
        # Skip lines that clearly belong to institutional
        # information.
        if re.search(
            r"\b("
            r"University|"
            r"Department|"
            r"School|"
            r"Hospital|"
            r"Institute|"
            r"Center|"
            r"Centre|"
            r"College|"
            r"Faculty|"
            r"Medical|"
            r"Laboratory|"
            r"Laboratories"
            r")\b",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        # Contact/email lines.
        if "@" in line:
            continue

        # ------------------------------------------------------
        # This is the first actual scientific abstract line.
        # Do NOT require 80 characters.
        # ------------------------------------------------------

        abstract_start = i
        break

    if abstract_start == -1:
        return ""

    # ----------------------------------------------------------
    # 3. Collect complete abstract
    # ----------------------------------------------------------

    abstract_lines = []

    for line in lines[abstract_start:]:

        line = line.strip()

        if not line:
            continue

        # NEVER allow the next PubMed record into the abstract.
        if re.match(
            r"^PMID:\s*\d+",
            line,
            flags=re.IGNORECASE,
        ):
            break

        # Stop at common post-abstract sections.
        if line.startswith(
            (
                "Keywords:",
                "MeSH terms:",
                "Publication types:",
                "Grant support:",
                "Conflict of interest statement:",
                "References",
            )
        ):
            break

        abstract_lines.append(line)

    return clean_text(
        " ".join(abstract_lines)
    )


def extract_pmcid(record: str) -> str:
    """Extract PMCID when present."""
    match = re.search(
        r"\bPMCID:\s*(PMC\d+)",
        record,
        flags=re.IGNORECASE,
    )

    return match.group(1) if match else ""


def detect_retracted(record: str) -> bool:
    """Detect explicitly retracted records."""
    upper_record = record.upper()

    return any(
        marker in upper_record
        for marker in RETRACTED_MARKERS
    )


def detect_problem_status(record: str) -> str:
    """Detect special publication status."""
    upper_record = record.upper()

    if detect_retracted(record):
        return "RETRACTED"

    if any(marker in upper_record for marker in PROBLEM_MARKERS):
        return "EXPRESSION_OF_CONCERN"

    if "WITHDRAWN" in upper_record:
        return "WITHDRAWN"

    if "ERRATUM" in upper_record or "CORRIGENDUM" in upper_record:
        return "ERRATUM_OR_CORRECTION"

    return "NORMAL"


def split_pubmed_records(text: str) -> List[str]:
    """
    Split a PubMed export into individual records.

    Records in the supplied files begin with:
        1. Journal...
        2. Journal...
        3. Journal...
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Start of numbered PubMed records.
    starts = list(
        re.finditer(
            r"(?m)^\s*(\d+)\.\s+",
            text,
        )
    )

    if not starts:
        return []

    records = []

    for i, match in enumerate(starts):
        start = match.start()

        if i + 1 < len(starts):
            end = starts[i + 1].start()
        else:
            end = len(text)

        record = text[start:end].strip()

        if record:
            records.append(record)

    return records


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------

def parse_record(
    record: str,
    source_file: str,
) -> Optional[Dict[str, str]]:
    """Parse one PubMed record."""

    pmid = extract_pmid(record)

    # PMID is the primary identifier for deduplication.
    if not pmid:
        return None

    title = extract_title(record)
    abstract = extract_abstract(record)

    status = detect_problem_status(record)

    # Topic/category comes from the source filename.
    category = Path(source_file).stem

    return {
        "PMID": pmid,
        "Title": title,
        "Abstract": abstract,
        "Journal": extract_journal(record),
        "Year": extract_year(record),
        "DOI": extract_doi(record),
        "PMCID": extract_pmcid(record),
        "Topic": category,
        "Publication_Status": status,
        "Source_File": source_file,
    }


# ---------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------

def process_all_files() -> List[Dict[str, str]]:
    """Read and parse every TXT file in data/raw/."""

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {RAW_DIR}"
        )

    txt_files = sorted(RAW_DIR.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in: {RAW_DIR}"
        )

    all_records: List[Dict[str, str]] = []

    print("\n" + "=" * 70)
    print("PUBMED DATA PROCESSING")
    print("=" * 70)
    print(f"Raw directory: {RAW_DIR}")
    print(f"TXT files found: {len(txt_files)}")
    print()

    for file_path in txt_files:
        print(f"Reading: {file_path.name}")

        text = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        records = split_pubmed_records(text)

        print(f"  Raw records detected: {len(records)}")

        parsed_count = 0

        for record in records:
            parsed = parse_record(
                record,
                file_path.name,
            )

            if parsed:
                all_records.append(parsed)
                parsed_count += 1

        print(f"  Records parsed: {parsed_count}")
        print()

    return all_records


# ---------------------------------------------------------------------
# Deduplication and quality filtering
# ---------------------------------------------------------------------

def deduplicate_records(
    records: List[Dict[str, str]],
) -> tuple[List[Dict[str, str]], int]:
    """Deduplicate using PMID."""

    unique_records: Dict[str, Dict[str, str]] = {}

    duplicates = 0

    for record in records:
        pmid = record["PMID"]

        if pmid in unique_records:
            duplicates += 1

            # Prefer the record with a longer abstract.
            existing = unique_records[pmid]

            if len(record["Abstract"]) > len(existing["Abstract"]):
                unique_records[pmid] = record

        else:
            unique_records[pmid] = record

    return list(unique_records.values()), duplicates


def filter_records(
    records: List[Dict[str, str]],
) -> tuple[List[Dict[str, str]], Dict[str, int]]:
    """Apply basic quality filters."""

    stats = {
        "missing_title": 0,
        "missing_abstract": 0,
        "retracted": 0,
        "withdrawn": 0,
    }

    clean_records = []

    for record in records:

        if not record["Title"]:
            stats["missing_title"] += 1
            continue

        if not record["Abstract"] or len(record["Abstract"]) < 50:
            stats["missing_abstract"] += 1
            continue

        if record["Publication_Status"] == "RETRACTED":
            stats["retracted"] += 1
            continue

        if record["Publication_Status"] == "WITHDRAWN":
            stats["withdrawn"] += 1
            continue

        clean_records.append(record)

    return clean_records, stats


# ---------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------

def save_csv(records: List[Dict[str, str]]) -> None:
    """Save cleaned records to CSV."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
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

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"\nClean dataset saved to:")
    print(OUTPUT_FILE)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    records = process_all_files()

    raw_count = len(records)

    records, duplicate_count = deduplicate_records(records)

    unique_count = len(records)

    records, filter_stats = filter_records(records)

    final_count = len(records)

    save_csv(records)

    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)

    print(f"Parsed records:              {raw_count}")
    print(f"Duplicate PMIDs removed:    {duplicate_count}")
    print(f"Unique records:             {unique_count}")
    print(f"Missing title removed:      {filter_stats['missing_title']}")
    print(f"Missing/short abstract:     {filter_stats['missing_abstract']}")
    print(f"Retracted removed:          {filter_stats['retracted']}")
    print(f"Withdrawn removed:          {filter_stats['withdrawn']}")
    print(f"FINAL CLEAN RECORDS:        {final_count}")

    print("=" * 70)
    print("Data processing completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()