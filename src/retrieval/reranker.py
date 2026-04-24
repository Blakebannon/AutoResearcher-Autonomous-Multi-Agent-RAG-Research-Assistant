from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
import json

from src.schemas.evidence import Evidence


def get_reranker_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1200,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def rerank_evidence(
    query: str,
    evidence_items: List[Evidence],
    top_k: int = 8,
) -> List[Evidence]:
    """
    LLM-based reranker.

    Scores evidence for relevance to the original user query.
    Returns the top_k evidence items with relevance_score populated.
    """

    if not evidence_items:
        return []

    compact_evidence = [
        {
            "evidence_id": item.evidence_id,
            "title": item.title,
            "source_type": item.source_type,
            "content_preview": item.content[:700],
        }
        for item in evidence_items
    ]

    system_prompt = """
You are an evidence reranker for a research assistant.

Your job is to score each evidence item by how useful it is for answering the original user query.

Score from 0.0 to 1.0:
- 1.0 = directly answers the query
- 0.7 = strongly relevant
- 0.4 = somewhat relevant
- 0.1 = barely relevant
- 0.0 = irrelevant

Return ONLY valid JSON in this format:

{
  "scores": [
    {
      "evidence_id": "doc_1",
      "relevance_score": 0.92,
      "reason": "Brief reason"
    }
  ]
}
"""

    human_prompt = f"""
Original User Query:
{query}

Evidence Items:
{json.dumps(compact_evidence, indent=2)}
"""

    llm = get_reranker_llm()
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    try:
        parsed = json.loads(response.content.strip())
        scores = parsed.get("scores", [])
    except Exception:
        return evidence_items[:top_k]

    score_lookup = {
        item["evidence_id"]: float(item.get("relevance_score", 0.0))
        for item in scores
        if "evidence_id" in item
    }

    scored_items = []

    for item in evidence_items:
        score = score_lookup.get(item.evidence_id, 0.0)
        scored_items.append(
            item.model_copy(update={"relevance_score": score})
        )

    scored_items.sort(
        key=lambda item: item.relevance_score or 0.0,
        reverse=True,
    )

    return scored_items[:top_k]