def is_summary_query(query: str) -> bool:
    return any(
        keyword in query.lower()
        for keyword in ["summarize", "summary", "overview", "high level"]
    )


def researcher_node(state: AgentState) -> dict:
    llm = get_researcher_llm()

    query = state["query"]
    tasks = state.get("tasks", [])
    evidence_context = state.get("evidence_context", "")
    errors = state.get("errors", [])

    summary_mode = is_summary_query(query)

    if summary_mode:
        system_prompt = """
You are the research agent in a multi-agent RAG system.

The user is asking for a FULL DOCUMENT SUMMARY.

Your job:
- Synthesize ALL available evidence into a cohesive understanding of the entire document.
- Do NOT focus on a single section.
- Identify major themes, topics, and conclusions across the document.

Rules:
- Use ONLY the provided evidence.
- Do NOT invent facts.
- Preserve citation IDs exactly (e.g., [doc_1], [web_2]).
- Combine information across multiple chunks.
- If the document appears incomplete, say so.

Output format:

## Research Summary

## Document Overview
High-level description of the document’s purpose and scope.

## Key Themes
- Theme with citations
- Theme with citations

## Key Findings
- Important insight with citations
- Important insight with citations

## Evidence Gaps
- Missing or insufficient coverage

## Recommended Answer Direction
Explain how the final summary should be structured.
"""
    else:
        system_prompt = """
You are the research agent in a multi-agent RAG system.

Your job is to analyze the provided evidence and produce structured research notes.

Rules:
- Use ONLY the provided evidence.
- Do NOT invent facts.
- Preserve citation IDs exactly as shown.
- Every major factual claim should include at least one evidence ID.
- If evidence is weak, incomplete, or conflicting, say so.
- Do not produce the final answer.

Output format:

## Research Summary

## Key Findings
- Finding with citation ID
- Finding with citation ID

## Evidence Gaps
- Missing or insufficient evidence

## Recommended Answer Direction
Explain what the final answer should emphasize.
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