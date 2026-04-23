import csv
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("research_logs.csv")


def log_research(query: str, state: dict):
    """
    Logs multi-agent workflow data from the final state.
    """

    tasks = state.get("tasks", [])
    route_log = state.get("route_log", [])
    evidence = state.get("evidence", [])
    errors = state.get("errors", [])
    final_answer = state.get("final_answer", "")
    research_summary = state.get("research_summary", "")

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
        "planner_task_count": len(tasks),
        "planner_tasks": " | ".join([t["subquestion"] for t in tasks]),
        "route_log": " | ".join(route_log),
        "tools_used": ",".join(tools_used) if tools_used else "none",
        "evidence_count": len(evidence),
        "research_summary_length": len(research_summary),
        "final_answer_length": len(final_answer),
        "error_count": len(errors),
    }

    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(log_entry)