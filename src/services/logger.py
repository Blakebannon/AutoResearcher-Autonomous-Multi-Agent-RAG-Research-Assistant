import csv
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("research_logs.csv")


def log_research(query: str, state: dict):
    """
    Logs multi-agent workflow data from the final state.
    Includes Week 5 evaluation metrics.
    """

    tasks = state.get("tasks", [])
    route_log = state.get("route_log", [])
    evidence = state.get("evidence", [])
    errors = state.get("errors", [])
    final_answer = state.get("final_answer", "")
    research_summary = state.get("research_summary", "")

    needs_revision = state.get("needs_revision", False)
    iteration = state.get("iteration", 0)
    critic_feedback = state.get("critic_feedback", "")

    evaluation = state.get("evaluation", {})

    # Extract evaluation metrics
    groundedness_score = evaluation.get("groundedness_score", 0)
    citation_score = evaluation.get("citation_score", 0)
    completeness_score = evaluation.get("completeness_score", 0)
    clarity_score = evaluation.get("clarity_score", 0)
    overall_score = evaluation.get("overall_score", 0)
    judge_feedback = evaluation.get("judge_feedback", "")

    # Extract tool usage from route log
    tools_used = set()
    for route in route_log:
        if route.startswith("WEB"):
            tools_used.add("web_search_tool")
        elif route.startswith("LOCAL"):
            tools_used.add("document_retriever")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,

        # Planner
        "planner_task_count": len(tasks),
        "planner_tasks": " | ".join([t["subquestion"] for t in tasks]),

        # Routing + tools
        "route_log": " | ".join(route_log),
        "tools_used": ",".join(tools_used) if tools_used else "none",

        # Retrieval
        "evidence_count": len(evidence),

        # Outputs
        "research_summary_length": len(research_summary),
        "final_answer_length": len(final_answer),

        # Critic
        "needs_revision": needs_revision,
        "iteration": iteration,
        "critic_feedback": critic_feedback[:500],  # prevent CSV overflow

        # Evaluation (Week 5)
        "groundedness_score": groundedness_score,
        "citation_score": citation_score,
        "completeness_score": completeness_score,
        "clarity_score": clarity_score,
        "overall_score": overall_score,
        "judge_feedback": judge_feedback[:500],  # truncate long text

        # Errors
        "error_count": len(errors),
    }

    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(log_entry)