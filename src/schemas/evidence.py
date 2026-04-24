from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


SourceType = Literal["pdf", "web", "unknown"]


class Evidence(BaseModel):
    """
    Standardized evidence object used across the retrieval,
    research, synthesis, citation, critic, and evaluation layers.
    """

    evidence_id: str = Field(
        description="Stable unique ID for this evidence item, such as doc_1_chunk_4 or web_2"
    )

    source_type: SourceType = Field(
        default="unknown",
        description="Where the evidence came from: pdf, web, or unknown"
    )

    title: str = Field(
        default="Untitled Source",
        description="Human-readable source title"
    )

    content: str = Field(
        description="The actual evidence text used by agents"
    )

    source_path: Optional[str] = Field(
        default=None,
        description="Local file path for documents, if available"
    )

    url: Optional[str] = Field(
        default=None,
        description="URL for web sources, if available"
    )

    page: Optional[int] = Field(
        default=None,
        description="PDF page number, if available"
    )

    chunk_id: Optional[int] = Field(
        default=None,
        description="Chunk index from the source document, if available"
    )

    relevance_score: Optional[float] = Field(
        default=None,
        description="Retriever or reranker relevance score"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata preserved from retrievers/tools"
    )

    used_in_answer: bool = Field(
        default=False,
        description="Whether this evidence was cited or used in the final answer"
    )


def format_evidence_for_prompt(evidence: Evidence) -> str:
    """
    Converts one evidence object into a citation-friendly prompt block.
    """

    source_bits = []

    if evidence.title:
        source_bits.append(f"Title: {evidence.title}")

    if evidence.source_type:
        source_bits.append(f"Type: {evidence.source_type}")

    if evidence.page is not None:
        source_bits.append(f"Page: {evidence.page}")

    if evidence.url:
        source_bits.append(f"URL: {evidence.url}")

    source_info = " | ".join(source_bits)

    return f"""
[{evidence.evidence_id}]
{source_info}

{evidence.content}
""".strip()


def format_evidence_list_for_prompt(evidence_items: list[Evidence]) -> str:
    """
    Converts a list of evidence objects into a single prompt-ready block.
    """

    return "\n\n---\n\n".join(
        format_evidence_for_prompt(item)
        for item in evidence_items
    )