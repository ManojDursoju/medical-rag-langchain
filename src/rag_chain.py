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

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from evidence_selector import EvidenceSelector
from reranker import MedicalReranker


# ==============================================================
# Configuration
# ==============================================================

FINAL_EVIDENCE_K = 5

# Use a Gemini model available to your API account.
LLM_MODEL = "gemini-3.6-flash"


# ==============================================================
# System Prompt
# ==============================================================

SYSTEM_PROMPT = """
You are a Medical Research RAG Copilot.

You answer scientific and medical research questions using ONLY
the PubMed evidence supplied by the retrieval system.

STRICT GROUNDING RULES:

1. Do not invent facts.
2. Do not use unsupported information.
3. Do not fabricate citations.
4. Only cite PMIDs present in the supplied evidence.
5. Only provide DOIs present in the supplied evidence.
6. Distinguish reported study findings from interpretation.
7. Preserve numerical results accurately.
8. Mention limitations and uncertainty when supported by the papers.
9. If the retrieved evidence is insufficient, explicitly say:

   "The retrieved PubMed evidence is insufficient to answer this reliably."

10. Do not provide an individual medical diagnosis.
11. Do not recommend treatment for an individual patient.
12. This system is for research and educational purposes.

Citation format:

[PMID: 12345678]

or:

[PMID: 12345678; DOI: 10.xxxx/xxxxx]

Never create a PMID or DOI that is not present in the evidence.
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

Answer the question using ONLY the retrieved evidence.

Use this structure:

## Answer

Provide a clear scientific synthesis.

## Evidence

Explain the important findings and cite the relevant PMID.

## Limitations

Describe limitations, uncertainty, or conflicting evidence
when supported by the retrieved publications.

## Sources

List the PubMed publications actually used.

Do not introduce information that is not supported by the evidence.
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

        sections.append(
            f"""
--- EVIDENCE {i} ---

PMID: {item.get("pmid", "")}
Title: {item.get("title", "")}
Journal: {item.get("journal", "")}
Year: {item.get("year", "")}
DOI: {item.get("doi", "")}
Topic: {item.get("topic", "")}

Evidence:
{item.get("page_content", "")}
"""
        )

    return "\n".join(sections)


# ==============================================================
# Medical RAG Copilot
# ==============================================================

class MedicalRAGCopilot:

    def __init__(
        self,
        retrieval_only: bool = False,
    ):

        self.retrieval_only = retrieval_only

        print("=" * 80)
        print("MEDICAL RESEARCH RAG & LLM COPILOT")
        print("=" * 80)

        # ------------------------------------------------------
        # Retrieval
        # ------------------------------------------------------

        print("\nInitializing retrieval system...")

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

            print("\nMode: RETRIEVAL ONLY")
            print("Gemini initialization skipped.")
            print("No LLM API call will be made.")

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

            self.llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                temperature=0,
                google_api_key=api_key,
            )

            self.chain = (
                PROMPT
                | self.llm
            )

            print("\nMode: FULL RAG")
            print("Gemini LLM initialized.")

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
        # Gemini generation
        # ------------------------------------------------------

        context = format_evidence(
            evidence
        )

        print(
            "\nGenerating grounded answer with Gemini..."
        )

        response = self.chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        return (
            response.content,
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

        print(
            f"\n{i}. {item.get('title', '')}"
        )

        print(
            f"   PMID:   {item.get('pmid', '')}"
        )

        print(
            f"   Year:   {item.get('year', '')}"
        )

        doi = item.get(
            "doi",
            "",
        )

        if doi:
            print(
                f"   DOI:    {doi}"
            )


# ==============================================================
# Main
# ==============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--retrieval-only",
        action="store_true",
    )

    parser.add_argument(
        "--question",
        type=str,
        default=(
            "How is deep learning used "
            "for brain tumor segmentation?"
        ),
    )

    args = parser.parse_args()

    copilot = MedicalRAGCopilot(
        retrieval_only=args.retrieval_only
    )

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