from src.graph.workflow import build_workflow
from src.rag_pipeline import load_vectorstore
from src.services.logger import log_research


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_tasks(tasks: list[dict]) -> None:
    if not tasks:
        print("No tasks generated.")
        return

    for i, task in enumerate(tasks, start=1):
        print(f"{i}. {task['subquestion']}")
        print(f"   source_preference: {task['source_preference']}")
        print(f"   rationale: {task['rationale']}")


def print_route_log(route_log: list[str]) -> None:
    if not route_log:
        print("No route log entries.")
        return

    for item in route_log:
        print(f"- {item}")


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

    section("STREAMING WORKFLOW")

    for chunk in app.stream(initial_state):
        for node_name, node_output in chunk.items():
            print(f"\n[{node_name.upper()} COMPLETED]")

            if "tasks" in node_output:
                print("Tasks:")
                print_tasks(node_output["tasks"])

            if "evidence" in node_output:
                print(f"Evidence collected: {len(node_output['evidence'])}")

            if "research_summary" in node_output:
                summary = node_output["research_summary"]
                preview = summary[:500] + ("..." if len(summary) > 500 else "")
                print("Research summary preview:")
                print(preview)

            if "final_answer" in node_output:
                answer = node_output["final_answer"]
                preview = answer[:700] + ("..." if len(answer) > 700 else "")
                print("Final answer preview:")
                print(preview)

            if "route_log" in node_output:
                print("Route log:")
                print_route_log(node_output["route_log"])

            if "errors" in node_output and node_output["errors"]:
                print("Errors:")
                for err in node_output["errors"]:
                    print(f"- {err}")

    # Run once more for clean final state + logging.
    result = app.invoke(initial_state)
    log_research(query, result)

    section("FINAL RESULT")

    section("PLANNER TASKS")
    print_tasks(result.get("tasks", []))

    section("EVIDENCE COUNT")
    print(len(result.get("evidence", [])))

    section("RESEARCH SUMMARY PREVIEW")
    summary = result.get("research_summary", "")
    print(summary[:1200] + ("..." if len(summary) > 1200 else ""))

    section("FINAL ANSWER")
    print(result.get("final_answer", ""))

    section("ROUTE LOG")
    print_route_log(result.get("route_log", []))

    errors = result.get("errors", [])
    if errors:
        section("ERRORS")
        for err in errors:
            print(f"- {err}")


if __name__ == "__main__":
    main()