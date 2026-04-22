import csv
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("research_logs.csv")


def log_research(query: str, result: dict):
    """
    Logs each research query and metadata to a CSV file.
    Supports multiple tool calls per run.
    """

    messages = result.get("messages", [])

    # Collect ALL tools used (not just the first one)
    tools_used = []

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                tools_used.append(call.get("name", "unknown_tool"))

    tool_used_str = ",".join(tools_used) if tools_used else "none"

    # Get final answer safely
    final_answer = ""
    if messages and hasattr(messages[-1], "content"):
        final_answer = messages[-1].content or ""

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "tool_used": tool_used_str,
        "response_length": len(final_answer),
    }

    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(log_entry)