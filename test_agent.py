from src.agents.react_agent import run_research_query, get_final_answer

query = "Summarize the main ideas in my uploaded documents."

result = run_research_query(query)

print("\nFULL MESSAGE TRACE:\n")
for i, message in enumerate(result["messages"], start=1):
    print(f"\n--- MESSAGE {i} ---")
    print(f"TYPE: {type(message).__name__}")
    
    if hasattr(message, "content"):
        print("CONTENT:")
        print(message.content)

    if hasattr(message, "tool_calls") and message.tool_calls:
        print("TOOL CALLS:")
        print(message.tool_calls)

print("\n" + "=" * 80)
print("FINAL ANSWER:\n")
print(get_final_answer(result))