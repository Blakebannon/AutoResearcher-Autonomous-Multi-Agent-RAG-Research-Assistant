# src/services/research_service.py

from src.graph.workflow import build_workflow
from src.graph.state import AgentState
from src.rag_pipeline import load_vectorstore


def research(query: str) -> dict:
    """
    Runs the full LangGraph AutoResearcher workflow and returns structured output
    for the Streamlit UI layer.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    graph = build_workflow(retriever)

    initial_state: AgentState = {
        "query": query,
        "tasks": [],
        "route_log": [],
        "evidence": [],
        "evidence_context": "",
        "research_summary": "",
        "final_answer": "",
        "critic_feedback": "",
        "needs_revision": False,
        "iteration": 0,
        "max_iterations": 1,
        "evaluation": {},
        "errors": [],
    }

    final_state = graph.invoke(initial_state)

    return {
        "answer": final_state.get("final_answer", ""),
        "final_answer": final_state.get("final_answer", ""),
        "tasks": final_state.get("tasks", []),
        "route_log": final_state.get("route_log", []),
        "evidence": final_state.get("evidence", []),
        "evidence_context": final_state.get("evidence_context", ""),
        "research_summary": final_state.get("research_summary", ""),
        "critic": {
            "needs_revision": final_state.get("needs_revision", False),
            "critic_feedback": final_state.get("critic_feedback", ""),
            "iteration": final_state.get("iteration", 0),
        },
        "evaluation": final_state.get("evaluation", {}),
        "errors": final_state.get("errors", []),
        "raw_state": final_state,
    }