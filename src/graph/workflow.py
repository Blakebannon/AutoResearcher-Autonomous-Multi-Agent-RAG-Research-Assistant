import json
import os
import re
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from src.schemas.evidence import Evidence, format_evidence_list_for_prompt
from src.agents.critic import critic_node
from src.agents.planner import planner_node
from src.agents.researcher import researcher_node
from src.agents.synthesizer import synthesizer_node
from src.retrieval.reranker import rerank_evidence
from src.evaluation.judge import judge_node
from src.graph.state import AgentState
from src.tools.tools import (
    create_document_retriever_tool,
    create_web_search_tool,
)

load_dotenv()


def parse_document_retriever_output(raw_text: str) -> List[Evidence]:
    evidence_items: List[Evidence] = []

    if not raw_text or not raw_text.strip():
        return evidence_items

    pattern = r"\[SOURCE:\s*(.*?)\s*\|\s*CHUNK:\s*(.*?)\]\n(.*?)(?=\n\[SOURCE:|\Z)"
    matches = re.findall(pattern, raw_text, re.DOTALL)

    for index, (source, chunk_id, content) in enumerate(matches, start=1):
        source = source.strip()
        chunk_id_clean = chunk_id.strip()
        content = content.strip()

        evidence_items.append(
            Evidence(
                evidence_id=f"doc_{index}",
                source_type="pdf",
                title=source or "Local Document",
                content=content,
                source_path=source,
                chunk_id=int(chunk_id_clean) if chunk_id_clean.isdigit() else None,
                metadata={
                    "source": source,
                    "chunk_id": chunk_id_clean,
                    "tool_used": "document_retriever",
                },
            )
        )

    return evidence_items


def parse_web_search_output(raw_result) -> List[Evidence]:
    evidence_items: List[Evidence] = []

    if not raw_result:
        return evidence_items

    if isinstance(raw_result, str):
        evidence_items.append(
            Evidence(
                evidence_id="web_1",
                source_type="web",
                title="Web Search Result",
                content=raw_result.strip(),
                metadata={
                    "source": "web_search",
                    "tool_used": "web_search_tool",
                },
            )
        )
        return evidence_items

    if isinstance(raw_result, dict):
        results = raw_result.get("results", [])

        if results:
            for index, item in enumerate(results, start=1):
                content = (
                    item.get("content", "").strip()
                    or item.get("snippet", "").strip()
                )

                if not content:
                    continue

                evidence_items.append(
                    Evidence(
                        evidence_id=f"web_{index}",
                        source_type="web",
                        title=item.get("title", "Web Result"),
                        content=content,
                        url=item.get("url"),
                        metadata={
                            **item,
                            "tool_used": "web_search_tool",
                        },
                    )
                )

            return evidence_items

        answer = raw_result.get("answer")
        if answer:
            evidence_items.append(
                Evidence(
                    evidence_id="web_answer_1",
                    source_type="web",
                    title="Tavily Answer",
                    content=answer.strip(),
                    metadata={
                        "source": "tavily_answer",
                        "tool_used": "web_search_tool",
                    },
                )
            )

    if isinstance(raw_result, list):
        for index, item in enumerate(raw_result, start=1):
            if isinstance(item, dict):
                content = (
                    item.get("content", "").strip()
                    or item.get("snippet", "").strip()
                )

                if not content:
                    continue

                evidence_items.append(
                    Evidence(
                        evidence_id=f"web_{index}",
                        source_type="web",
                        title=item.get("title", "Web Result"),
                        content=content,
                        url=item.get("url"),
                        metadata={
                            **item,
                            "tool_used": "web_search_tool",
                        },
                    )
                )
            else:
                evidence_items.append(
                    Evidence(
                        evidence_id=f"web_{index}",
                        source_type="web",
                        title="Web Result",
                        content=str(item).strip(),
                        metadata={
                            "source": "web_result",
                            "tool_used": "web_search_tool",
                        },
                    )
                )

    return evidence_items


def dedupe_evidence(evidence: List[Evidence]) -> List[Evidence]:
    seen = set()
    unique_items: List[Evidence] = []

    for item in evidence:
        key = (item.title, item.content)

        if key not in seen:
            seen.add(key)
            unique_items.append(item)

    return unique_items


def renumber_evidence(evidence: List[Evidence]) -> List[Evidence]:
    renumbered: List[Evidence] = []

    doc_count = 0
    web_count = 0
    other_count = 0

    for item in evidence:
        if item.source_type == "pdf":
            doc_count += 1
            new_id = f"doc_{doc_count}"
        elif item.source_type == "web":
            web_count += 1
            new_id = f"web_{web_count}"
        else:
            other_count += 1
            new_id = f"evidence_{other_count}"

        renumbered.append(
            item.model_copy(update={"evidence_id": new_id})
        )

    return renumbered


def retrieval_node(state: AgentState, retriever) -> dict:
    tasks = state.get("tasks", [])
    all_evidence: List[Evidence] = []
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
                errors.append(
                    f"Unknown source_preference '{source_preference}' for task: {subquestion}"
                )

        except Exception as e:
            errors.append(f"Retrieval failed for '{subquestion}': {str(e)}")

    all_evidence = dedupe_evidence(all_evidence)
    all_evidence = renumber_evidence(all_evidence)

    #Week 5: rerank evidence before passing it downstream
    try:
        all_evidence = rerank_evidence(
            query=state["query"],
            evidence_items=all_evidence,
            top_k=8,
        )
    except Exception as e:
        errors.append(f"Reranking failed: {str(e)}")

    evidence_context = format_evidence_list_for_prompt(all_evidence)

    return {
        "evidence": all_evidence,
        "evidence_context": evidence_context,
        "route_log": route_log,
        "errors": errors,
    }


def get_revision_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1800,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def revision_node(state: AgentState) -> dict:
    llm = get_revision_llm()

    query = state["query"]
    final_answer = state["final_answer"]
    critic_feedback = state["critic_feedback"]
    evidence_context = state.get("evidence_context", "")
    evidence = state.get("evidence", [])
    iteration = state.get("iteration", 0)

    source_lines = []

    for item in evidence:
        if isinstance(item, Evidence):
            parts = []

            if item.title:
                parts.append(item.title)

            if item.page is not None:
                parts.append(f"page {item.page}")

            if item.url:
                parts.append(item.url)

            source_detail = " | ".join(parts) if parts else "Unknown source"
            source_lines.append(f"- [{item.evidence_id}] {source_detail}")

    source_list = "\n".join(source_lines) if source_lines else "No sources available."

    system_prompt = """
You are a revision agent for an autonomous research system.

Your job is to fix only the specific issues identified by the critic.

Rules:
- Use ONLY the provided evidence.
- Do NOT introduce new claims or sources.
- Do NOT rewrite the entire answer unnecessarily.
- Preserve correct content from the original answer.
- Preserve citation IDs exactly, such as [doc_1] or [web_2].
- Every major factual claim must include a citation ID.
- Keep or restore the "## Sources" section.
- If evidence is insufficient, clearly say so.
- Remove any claim that is not supported by the evidence.

Goal:
Produce a corrected final answer that is grounded, citation-supported, and responsive to the original query.
"""

    human_prompt = f"""
Original User Query:
{query}

Original Final Answer:
{final_answer}

Critic Feedback:
{critic_feedback}

Evidence:
{evidence_context}

Available Sources:
{source_list}
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    return {
        "final_answer": response.content,
        "iteration": iteration + 1,
    }


def route_after_critic(state: AgentState) -> str:
    needs_revision = state.get("needs_revision", False)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 1)

    if needs_revision and iteration < max_iterations:
        return "revise"

    return "end"


def build_workflow(retriever):
    graph = StateGraph(AgentState)

    def retrieval_step(state: AgentState):
        return retrieval_node(state, retriever)

    graph.add_node("planner", planner_node)
    graph.add_node("retrieve", retrieval_step)
    graph.add_node("research", researcher_node)
    graph.add_node("synthesize", synthesizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("revise", revision_node)
    graph.add_node("judge", judge_node)
    

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "research")
    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", "critic")

    graph.add_conditional_edges(
        "critic",
    route_after_critic,
    {
        "revise": "revise",
        "end": "judge",
    },
)

    graph.add_edge("revise", "critic")
    graph.add_edge("judge", END)
    

    return graph.compile()