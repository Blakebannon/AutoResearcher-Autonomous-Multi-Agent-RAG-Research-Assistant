import re
from typing import List

from src.graph.state import AgentState, EvidenceItem
from src.tools.tools import (
    create_document_retriever_tool,
    create_web_search_tool,
)


def parse_document_retriever_output(raw_text: str) -> List[EvidenceItem]:
    """
    Parse document retriever output of the form:

    [SOURCE: file.pdf | CHUNK: 23]
    some text here
    """
    evidence_items: List[EvidenceItem] = []

    if not raw_text or not raw_text.strip():
        return evidence_items

    pattern = r"\[SOURCE:\s*(.*?)\s*\|\s*CHUNK:\s*(.*?)\]\n(.*?)(?=\n\[SOURCE:|\Z)"
    matches = re.findall(pattern, raw_text, re.DOTALL)

    for source, chunk_id, content in matches:
        evidence_items.append(
            {
                "content": content.strip(),
                "source": source.strip(),
                "chunk_id": chunk_id.strip(),
                "tool_used": "document_retriever",
                "url": None,
            }
        )

    return evidence_items


def parse_web_search_output(raw_result) -> List[EvidenceItem]:
    """
    Normalize Tavily output into EvidenceItem format.

    Tavily may return either:
    - a dict-like object
    - a list
    - a string
    depending on version/config
    """
    evidence_items: List[EvidenceItem] = []

    if not raw_result:
        return evidence_items

    if isinstance(raw_result, str):
        evidence_items.append(
            {
                "content": raw_result.strip(),
                "source": "web_search",
                "chunk_id": None,
                "tool_used": "web_search_tool",
                "url": None,
            }
        )
        return evidence_items

    if isinstance(raw_result, dict):
        results = raw_result.get("results", [])
        if results:
            for item in results:
                evidence_items.append(
                    {
                        "content": item.get("content", "").strip() or item.get("snippet", "").strip(),
                        "source": item.get("title", "web_result"),
                        "chunk_id": None,
                        "tool_used": "web_search_tool",
                        "url": item.get("url"),
                    }
                )
            return evidence_items

        answer = raw_result.get("answer")
        if answer:
            evidence_items.append(
                {
                    "content": answer.strip(),
                    "source": "tavily_answer",
                    "chunk_id": None,
                    "tool_used": "web_search_tool",
                    "url": None,
                }
            )
            return evidence_items

    if isinstance(raw_result, list):
        for item in raw_result:
            if isinstance(item, dict):
                evidence_items.append(
                    {
                        "content": item.get("content", "").strip() or item.get("snippet", "").strip(),
                        "source": item.get("title", "web_result"),
                        "chunk_id": None,
                        "tool_used": "web_search_tool",
                        "url": item.get("url"),
                    }
                )
            else:
                evidence_items.append(
                    {
                        "content": str(item).strip(),
                        "source": "web_result",
                        "chunk_id": None,
                        "tool_used": "web_search_tool",
                        "url": None,
                    }
                )

    return evidence_items


def dedupe_evidence(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
    """
    Remove duplicate evidence items based on content + source.
    """
    seen = set()
    unique_items: List[EvidenceItem] = []

    for item in evidence:
        key = (item["source"], item["content"])
        if key not in seen:
            seen.add(key)
            unique_items.append(item)

    return unique_items


def retrieval_node(state: AgentState, retriever) -> dict:
    """
    Execute retrieval based on planner tasks and normalize all outputs.
    """
    tasks = state.get("tasks", [])
    all_evidence: List[EvidenceItem] = []
    route_log = list(state.get("route_log", []))
    errors = list(state.get("errors", []))

    document_tool = create_document_retriever_tool(retriever)
    web_tool = create_web_search_tool()

    for task in tasks:
        subquestion = task["subquestion"]
        source_preference = task["source_preference"]

        try:
            if source_preference == "local":
                route_log.append(f"LOCAL -> {subquestion}")
                raw_docs = document_tool.invoke(subquestion)
                all_evidence.extend(parse_document_retriever_output(raw_docs))

            elif source_preference == "web":
                route_log.append(f"WEB -> {subquestion}")
                raw_web = web_tool.invoke({"query": subquestion})
                all_evidence.extend(parse_web_search_output(raw_web))

            elif source_preference == "hybrid":
                route_log.append(f"HYBRID -> {subquestion}")

                raw_docs = document_tool.invoke(subquestion)
                all_evidence.extend(parse_document_retriever_output(raw_docs))

                raw_web = web_tool.invoke({"query": subquestion})
                all_evidence.extend(parse_web_search_output(raw_web))

            else:
                errors.append(f"Unknown source_preference '{source_preference}' for task: {subquestion}")

        except Exception as e:
            errors.append(f"Retrieval failed for '{subquestion}': {str(e)}")

    all_evidence = dedupe_evidence(all_evidence)

    return {
        "evidence": all_evidence,
        "route_log": route_log,
        "errors": errors,
    }

from langgraph.graph import StateGraph, START, END

from src.agents.planner import planner_node
from src.agents.researcher import researcher_node
from src.agents.synthesizer import synthesizer_node
from src.graph.state import AgentState


def build_workflow(retriever):
    """
    Build and compile the AutoResearcher LangGraph workflow.
    """

    graph = StateGraph(AgentState)

    def retrieval_step(state: AgentState):
        return retrieval_node(state, retriever)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieve", retrieval_step)
    graph.add_node("research", researcher_node)
    graph.add_node("synthesize", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "research")
    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", END)

    app = graph.compile()
    return app