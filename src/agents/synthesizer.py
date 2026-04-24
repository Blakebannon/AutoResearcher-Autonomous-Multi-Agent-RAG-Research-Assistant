import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.graph.state import AgentState
from src.schemas.evidence import Evidence

load_dotenv()


def get_synthesizer_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=2200,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def build_source_list(evidence: list[Evidence]) -> str:
    if not evidence:
        return "No sources available."

    source_lines = []

    for item in evidence:
        parts = []

        if item.title:
            parts.append(item.title)

        if item.page is not None:
            parts.append(f"page {item.page}")

        if item.url:
            parts.append(item.url)

        source_detail = " | ".join(parts) if parts else "Unknown source"

        source_lines.append(f"- [{item.evidence_id}] {source_detail}")

    return "\n".join(source_lines)


def synthesizer_node(state: AgentState) -> dict:
    llm = get_synthesizer_llm()

    query = state["query"]
    research_summary = state.get("research_summary", "")
    evidence_context = state.get("evidence_context", "")
    evidence = state.get("evidence", [])
    errors = state.get("errors", [])

    source_list = build_source_list(evidence)

    system_prompt = """
You are the synthesizer agent in an autonomous multi-agent research system.

Your job is to produce the final user-facing answer using the research summary and evidence.

Rules:
- Use ONLY the provided research summary and evidence.
- Do NOT invent facts.
- Every major factual claim must include a citation ID, such as [doc_1] or [web_2].
- Use citation IDs exactly as provided.
- If the evidence does not support a claim, do not include that claim.
- If evidence is incomplete or conflicting, clearly say so.
- Write in a clear, concise, professional style.
- Do not cite sources that are not relevant to the specific claim.
- Include a Sources section at the end.

Required output format:

## Final Answer

Your answer with inline citations.

## Sources

- [doc_1] Source title or URL
- [web_1] Source title or URL
"""

    human_prompt = f"""
Original User Query:
{query}

Research Summary:
{research_summary}

Evidence:
{evidence_context}

Available Sources:
{source_list}

Retrieval Errors:
{errors}
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    return {
        "final_answer": response.content,
    }