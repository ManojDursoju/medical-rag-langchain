"""
Medical Research RAG & LLM Copilot

Pipeline:

Question
   ↓
FAISS + BM25
   ↓
RRF
   ↓
BGE Reranker
   ↓
Evidence Selector
   ↓
Gemini LLM
   ↓
Grounded research answer
   ↓
PMID / DOI citations
"""

from __future__ import annotations

import argparse
import os
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from evidence_selector import EvidenceSelector
from reranker import MedicalReranker


# ==============================================================
# Configuration
# ==============================================================

FINAL_EVIDENCE_K = 5

LLM_MODEL = "gemini-3.6-flash"


# ==============================================================
# System Prompt
# ==============================================================

SYSTEM_PROMPT = """
You are a Medical Research RAG Copilot.

Your job is to answer scientific and medical research questions
using ONLY the PubMed evidence supplied in the user message.

The retrieved evidence is the authoritative source for this answer.

STRICT GROUNDING RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not invent numerical results.
4. Do not fabricate citations.
5. Only cite PMIDs that appear in the supplied evidence.
6. Only provide DOIs that appear in the supplied evidence.
7. Do not attribute a finding to a paper unless the supplied evidence
   supports that finding.
8. Distinguish reported study findings from your synthesis or interpretation.
9. Preserve numerical values accurately.
10. Mention uncertainty, limitations, or conflicting findings when supported
    by the supplied evidence.
11. If the evidence is insufficient, explicitly state:

"The retrieved PubMed evidence is insufficient to answer this reliably."

12. Do not provide an individual medical diagnosis.
13. Do not recommend treatment for an individual patient.
14. This system is intended for research and educational purposes.

CITATION RULES:

Use:

[PMID: 12345678]

or:

[PMID: 12345678; DOI: 10.xxxx/xxxxx]

Never create a PMID or DOI.

Every PMID and DOI in the answer must exist in the supplied evidence.
"""


# ==============================================================
# Prompt
# ==============================================================

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT,
        ),
        (
            "human",
            """
Research question:

{question}

Retrieved PubMed evidence:

{context}

Answer the research question using ONLY the retrieved PubMed evidence.

Use this structure:

## Answer

Provide a concise but scientifically useful synthesis of the evidence.

Cite claims using the supplied PMID.

## Evidence

Explain the important findings from the retrieved publications.

Use PMID citations for individual findings.

## Limitations

Describe limitations, uncertainty, or conflicting evidence
when supported by the retrieved publications.

## Sources

List only the publications actually used in the answer.

For each source provide:

- Title
- PMID
- DOI if available

Do not introduce information that is not supported by the supplied evidence.
""",
        ),
    ]
)


# ==============================================================
# Format evidence
# ==============================================================

def format_evidence(
    evidence: list[dict],
) -> str:

    sections = []

    for i, item in enumerate(
        evidence,
        start=1,
    ):

        pmid = str(
            item.get(
                "pmid",
                "",
            )
        ).strip()

        title = str(
            item.get(
                "title",
                "",
            )
        ).strip()

        journal = str(
            item.get(
                "journal",
                "",
            )
        ).strip()

        year = str(
            item.get(
                "year",
                "",
            )
        ).strip()

        doi = str(
            item.get(
                "doi",
                "",
            )
        ).strip()

        topic = str(
            item.get(
                "topic",
                "",
            )
        ).strip()

        page_content = str(
            item.get(
                "page_content",
                "",
            )
        ).strip()

        sections.append(
            f"""
--- EVIDENCE {i} ---

PMID: {pmid}
Title: {title}
Journal: {journal}
Year: {year}
DOI: {doi}
Topic: {topic}

Evidence:
{page_content}
""".strip()
        )

    return "\n\n".join(
        sections
    )


# ==============================================================
# Clean Gemini response
# ==============================================================

def clean_llm_response(
    content,
) -> str:
    """
    Gemini 3.6 Flash may return structured content such as:

    [
        {
            "type": "text",
            "text": "..."
        }
    ]

    Convert it into clean plain text.
    """

    if content is None:
        return ""

    # ----------------------------------------------------------
    # Normal string
    # ----------------------------------------------------------

    if isinstance(
        content,
        str,
    ):

        return content.strip()

    # ----------------------------------------------------------
    # Structured list
    # ----------------------------------------------------------

    if isinstance(
        content,
        list,
    ):

        text_parts = []

        for block in content:

            # Dictionary block
            if isinstance(
                block,
                dict,
            ):

                text = block.get(
                    "text"
                )

                if text:
                    text_parts.append(
                        str(text)
                    )

                continue

            # Object with .text
            text = getattr(
                block,
                "text",
                None,
            )

            if text:
                text_parts.append(
                    str(text)
                )

        return "\n".join(
            text_parts
        ).strip()

    # ----------------------------------------------------------
    # Fallback
    # ----------------------------------------------------------

    return str(
        content
    ).strip()


# ==============================================================
# Validate citations
# ==============================================================

def validate_citations(
    answer: str,
    evidence: list[dict],
) -> str:
    """
    Remove accidental PMID/DOI references that were not present
    in the retrieved evidence.

    This is a safety layer, not a substitute for the grounding prompt.
    """

    valid_pmids = {
        str(
            item.get(
                "pmid",
                "",
            )
        ).strip()
        for item in evidence
        if item.get("pmid")
    }

    valid_dois = {
        str(
            item.get(
                "doi",
                "",
            )
        ).strip().lower()
        for item in evidence
        if item.get("doi")
    }

    # ----------------------------------------------------------
    # PMID validation
    # ----------------------------------------------------------

    def replace_pmid(
        match,
    ):

        pmid = match.group(
            1
        )

        if pmid in valid_pmids:
            return match.group(0)

        return "[PMID: unavailable]"

    answer = re.sub(
        r"\[PMID:\s*(\d+)\]",
        replace_pmid,
        answer,
        flags=re.IGNORECASE,
    )

    # ----------------------------------------------------------
    # DOI validation
    # ----------------------------------------------------------

    def replace_doi(
        match,
    ):

        doi = match.group(
            1
        ).rstrip(
            ".,;)"
        )

        if doi.lower() in valid_dois:
            return match.group(0)

        return "DOI: unavailable"

    answer = re.sub(
        r"(?:DOI:\s*)(10\.\S+)",
        replace_doi,
        answer,
        flags=re.IGNORECASE,
    )

    return answer


# ==============================================================
# Medical RAG Copilot
# ==============================================================

class MedicalRAGCopilot:

    def __init__(
        self,
        retrieval_only: bool = False,
    ):

        self.retrieval_only = (
            retrieval_only
        )

        print(
            "=" * 80
        )

        print(
            "MEDICAL RESEARCH RAG & LLM COPILOT"
        )

        print(
            "=" * 80
        )

        # ------------------------------------------------------
        # Retrieval
        # ------------------------------------------------------

        print(
            "\nInitializing retrieval system..."
        )

        reranker = MedicalReranker()

        self.selector = EvidenceSelector(
            reranker,
            final_k=FINAL_EVIDENCE_K,
        )

        # ------------------------------------------------------
        # LLM
        # ------------------------------------------------------

        self.llm = None

        self.chain = None

        if retrieval_only:

            print(
                "\nMode: RETRIEVAL ONLY"
            )

            print(
                "Gemini initialization skipped."
            )

            print(
                "No LLM API call will be made."
            )

        else:

            api_key = os.getenv(
                "GOOGLE_API_KEY"
            )

            if not api_key:

                raise RuntimeError(
                    "GOOGLE_API_KEY is not configured."
                )

            print(
                f"\nLoading Gemini model: "
                f"{LLM_MODEL}"
            )

            # IMPORTANT:
            # gemini-3.6-flash uses fixed sampling defaults.
            # Do not pass temperature.
            self.llm = (
                ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    google_api_key=api_key,
                )
            )

            self.chain = (
                PROMPT
                | self.llm
            )

            print(
                "\nMode: FULL RAG"
            )

            print(
                "Gemini LLM initialized."
            )

    # ==========================================================
    # Retrieve
    # ==========================================================

    def retrieve(
        self,
        question: str,
    ) -> list[dict]:

        print(
            "\nRetrieving PubMed evidence..."
        )

        evidence = self.selector.select(
            question
        )

        print(
            f"Evidence selected: "
            f"{len(evidence)}"
        )

        return evidence

    # ==========================================================
    # Ask
    # ==========================================================

    def ask(
        self,
        question: str,
    ):

        evidence = self.retrieve(
            question
        )

        # ------------------------------------------------------
        # Retrieval-only
        # ------------------------------------------------------

        if self.retrieval_only:

            return None, evidence

        # ------------------------------------------------------
        # No evidence
        # ------------------------------------------------------

        if not evidence:

            return (
                "The retrieved PubMed evidence is insufficient "
                "to answer this reliably.",
                evidence,
            )

        # ------------------------------------------------------
        # Format evidence
        # ------------------------------------------------------

        context = format_evidence(
            evidence
        )

        print(
            "\nGenerating grounded answer with Gemini..."
        )

        # ------------------------------------------------------
        # Generate
        # ------------------------------------------------------

        response = self.chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        # ------------------------------------------------------
        # Clean structured Gemini output
        # ------------------------------------------------------

        answer = clean_llm_response(
            response.content
        )

        # ------------------------------------------------------
        # Citation validation
        # ------------------------------------------------------

        answer = validate_citations(
            answer,
            evidence,
        )

        return (
            answer,
            evidence,
        )


# ==============================================================
# Display sources
# ==============================================================

def display_sources(
    evidence: list[dict],
):

    print(
        "\n" + "=" * 80
    )

    print(
        "PUBMED SOURCES"
    )

    print(
        "=" * 80
    )

    for i, item in enumerate(
        evidence,
        start=1,
    ):

        title = item.get(
            "title",
            "",
        )

        pmid = item.get(
            "pmid",
            "",
        )

        year = item.get(
            "year",
            "",
        )

        doi = item.get(
            "doi",
            "",
        )

        print(
            f"\n{i}. {title}"
        )

        print(
            f"   PMID:   {pmid}"
        )

        print(
            f"   Year:   {year}"
        )

        if doi:

            print(
                f"   DOI:    {doi}"
            )


# ==============================================================
# Main
# ==============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Medical Research RAG & LLM Copilot"
        )
    )

    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Run retrieval and evidence "
            "selection without Gemini."
        ),
    )

    parser.add_argument(
        "--question",
        type=str,
        default=(
            "How is deep learning used "
            "for brain tumor segmentation?"
        ),
        help=(
            "Research question to answer."
        ),
    )

    args = parser.parse_args()

    # ----------------------------------------------------------
    # Initialize
    # ----------------------------------------------------------

    copilot = MedicalRAGCopilot(
        retrieval_only=(
            args.retrieval_only
        )
    )

    # ----------------------------------------------------------
    # Run
    # ----------------------------------------------------------

    answer, evidence = copilot.ask(
        args.question
    )

    # ----------------------------------------------------------
    # Retrieval-only
    # ----------------------------------------------------------

    if args.retrieval_only:

        display_sources(
            evidence
        )

        print(
            "\n" + "=" * 80
        )

        print(
            "RETRIEVAL-ONLY TEST COMPLETED"
        )

        print(
            "No Gemini API call was made."
        )

        print(
            "=" * 80
        )

        return

    # ----------------------------------------------------------
    # Full RAG
    # ----------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "MEDICAL RAG ANSWER"
    )

    print(
        "=" * 80
    )

    print(
        answer
    )

    display_sources(
        evidence
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FULL RAG QUERY COMPLETED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()