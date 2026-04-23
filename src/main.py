from src.graph.workflow import build_workflow
from src.rag_pipeline import load_vectorstore
from src.services.logger import log_research


def main():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    app = build_workflow(retriever)

    query = "What are the latest AI regulations in the United States in 2026?"

    initial_state = {
        "query": query,
        "tasks": [],
        "evidence": [],
        "research_summary": "",
        "final_answer": "",
        "route_log": [],
        "errors": [],
        "iteration": 0,
    }

    result = app.invoke(initial_state)

    log_research(query, result)

    print("\n" + "=" * 80)
    print("PLANNER TASKS")
    print("=" * 80)
    for i, task in enumerate(result.get("tasks", []), start=1):
        print(f"{i}. {task['subquestion']}")
        print(f"   source_preference: {task['source_preference']}")
        print(f"   rationale: {task['rationale']}")

    print("\n" + "=" * 80)
    print("EVIDENCE COUNT")
    print("=" * 80)
    print(len(result.get("evidence", [])))

    print("\n" + "=" * 80)
    print("RESEARCH SUMMARY PREVIEW")
    print("=" * 80)
    summary = result.get("research_summary", "")
    print(summary[:1200] + ("..." if len(summary) > 1200 else ""))

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result.get("final_answer", ""))

    print("\n" + "=" * 80)
    print("ROUTE LOG")
    print("=" * 80)
    for item in result.get("route_log", []):
        print("-", item)

    errors = result.get("errors", [])
    if errors:
        print("\n" + "=" * 80)
        print("ERRORS")
        print("=" * 80)
        for err in errors:
            print("-", err)


if __name__ == "__main__":
    main()