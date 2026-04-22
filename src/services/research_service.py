from src.agents.react_agent import run_research_query, get_final_answer


def research(query: str) -> dict:
    """
    Runs the agentic research workflow and returns structured output
    for the UI layer.
    """
    result = run_research_query(query)

    messages = result.get("messages", [])
    final_answer = get_final_answer(result)

    tools_used = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                tools_used.append(call.get("name", "unknown_tool"))

    return {
        "answer": final_answer,
        "tools_used": tools_used,
        "raw_result": result,
    }