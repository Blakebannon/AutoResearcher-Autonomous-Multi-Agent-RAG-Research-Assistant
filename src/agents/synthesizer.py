import json
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1800,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def synthesizer_node(state):
    llm = get_llm()

    query = state["query"]
    research_summary = state["research_summary"]
    evidence = state["evidence"]

    # Keep evidence volume under control
    compact_evidence = []
    for item in evidence[:12]:
        compact_evidence.append(
            {
                "source": item.get("source"),
                "chunk_id": item.get("chunk_id"),
                "tool_used": item.get("tool_used"),
                "url": item.get("url"),
            }
        )

    system_prompt = """
You are the final synthesis agent for an autonomous research system.

Your job is to produce a polished, grounded answer to the user's query.

Rules:
- Use the research summary as your primary analysis input
- Ground your answer in the provided evidence metadata
- Do NOT invent facts, sources, or citations
- If evidence is incomplete or mixed, say so clearly
- When referencing sources, mention them naturally in the answer
- Prefer clarity and structure over hype
- Produce the final answer only
"""

    human_prompt = f"""
User Query:
{query}

Research Summary:
{research_summary}

Evidence Metadata:
{json.dumps(compact_evidence, indent=2)}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])

    return {
        "final_answer": response.content
    }