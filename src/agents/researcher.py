import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.graph.state import AgentState

load_dotenv()


def get_researcher_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1800,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def researcher_node(state: AgentState) -> dict:
    llm = get_researcher_llm()

    query = state["query"]
    tasks = state.get("tasks", [])
    evidence_context = state.get("evidence_context", "")
    errors = state.get("errors", [])

    system_prompt = """
You are the research agent in a multi-agent RAG system.

Your job is to analyze the provided evidence and produce a structured research summary.

Rules:
- Use ONLY the provided evidence.
- Do NOT invent facts.
- Preserve citation IDs exactly as shown, such as [doc_1] or [web_2].
- Every major factual claim should include at least one evidence ID.
- If evidence is weak, incomplete, or conflicting, say so.
- Do not produce the final answer. Produce research notes for the synthesizer.

Output format:

## Research Summary

## Key Findings
- Finding with citation ID
- Finding with citation ID

## Evidence Gaps
- Missing or insufficient evidence

## Recommended Answer Direction
Briefly explain what the final answer should emphasize.
"""

    human_prompt = f"""
Original User Query:
{query}

Research Tasks:
{tasks}

Evidence:
{evidence_context}

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
        "research_summary": response.content,
    }