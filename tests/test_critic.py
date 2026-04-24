from src.agents.critic import critic_node


test_state = {
    "query": "What are the latest AI regulations in the United States in 2026?",
    "tasks": [],
    "evidence": [
        {
            "content": "There is no comprehensive federal AI law in the United States. AI governance is currently shaped by executive orders, agency enforcement, and state-level legislation.",
            "source": "Example policy source",
            "chunk_id": None,
            "tool_used": "web_search_tool",
            "url": "https://example.com",
        }
    ],
    "research_summary": "The US does not currently have a comprehensive federal AI law. State-level AI regulation is increasing, and federal policy is shaped by executive orders and agency activity.",
    "final_answer": "The United States has a comprehensive federal AI law that fully regulates all AI systems nationwide.",
    "route_log": [],
    "errors": [],
    "iteration": 0,
    "max_iterations": 1,
    "needs_revision": False,
    "critic_feedback": "",
}

result = critic_node(test_state)

print(result)