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


def print_evaluation(evaluation: dict) -> None:
    if not evaluation:
        print("No evaluation available.")
        return

    print(f"Groundedness: {evaluation.get('groundedness_score')}")
    print(f"Citations: {evaluation.get('citation_score')}")
    print(f"Completeness: {evaluation.get('completeness_score')}")
    print(f"Clarity: {evaluation.get('clarity_score')}")
    print(f"Overall: {evaluation.get('overall_score')}")
    print("Judge feedback:")
    print(evaluation.get("judge_feedback", ""))


def main():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    app = build_workflow(retriever)

    query = "How will AI regulations in the US impact startups, consumers, and long-term innovation?"

    initial_state = {
        "query": query,
        "tasks": [],
        "evidence": [],
        "evidence_context": "",
        "research_summary": "",
        "final_answer": "",
        "route_log": [],
        "errors": [],
        "iteration": 0,
        "max_iterations": 1,
        "needs_revision": False,
        "critic_feedback": "",
        "evaluation": {},
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

                for item in node_output["evidence"]:
                    score = getattr(item, "relevance_score", None)
                    title = getattr(item, "title", "Unknown source")
                    evidence_id = getattr(item, "evidence_id", "unknown")

                    print(f"- [{evidence_id}] score={score} | {title}")

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

            if "needs_revision" in node_output:
                print(f"Needs revision: {node_output['needs_revision']}")

            if "critic_feedback" in node_output:
                print("Critic feedback:")
                print(node_output["critic_feedback"])

            if "iteration" in node_output:
                print(f"Iteration: {node_output['iteration']}")

            if "evaluation" in node_output:
                print("Evaluation:")
                print_evaluation(node_output["evaluation"])

    # Run once more for clean final state + logging.
    result = app.invoke(initial_state)
    log_research(query, result)

    section("FINAL RESULT")

    section("PLANNER TASKS")
    print_tasks(result.get("tasks", []))

    section("EVIDENCE COUNT")
    evidence = result.get("evidence", [])
    print(len(evidence))

    if evidence:
        section("EVIDENCE USED")
        for item in evidence:
            score = getattr(item, "relevance_score", None)
            title = getattr(item, "title", "Unknown source")
            evidence_id = getattr(item, "evidence_id", "unknown")
            source_type = getattr(item, "source_type", "unknown")
            url = getattr(item, "url", None)

            print(f"- [{evidence_id}] {source_type} | score={score} | {title}")
            if url:
                print(f"  URL: {url}")

    section("RESEARCH SUMMARY PREVIEW")
    summary = result.get("research_summary", "")
    print(summary[:1200] + ("..." if len(summary) > 1200 else ""))

    section("FINAL ANSWER")
    print(result.get("final_answer", ""))

    section("ROUTE LOG")
    print_route_log(result.get("route_log", []))

    section("CRITIC REVIEW")
    print(f"Needs revision: {result.get('needs_revision', False)}")
    print(f"Iteration count: {result.get('iteration', 0)}")
    print("Critic feedback:")
    print(result.get("critic_feedback", ""))

    section("LLM-AS-JUDGE EVALUATION")
    print_evaluation(result.get("evaluation", {}))

    errors = result.get("errors", [])
    if errors:
        section("ERRORS")
        for err in errors:
            print(f"- {err}")


if __name__ == "__main__":
    main()