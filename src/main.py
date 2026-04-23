from src.graph.workflow import build_workflow
from src.rag_pipeline import load_vectorstore


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

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result["final_answer"])

    print("\n" + "=" * 80)
    print("ROUTE LOG")
    print("=" * 80)
    for item in result["route_log"]:
        print("-", item)

    if result["errors"]:
        print("\n" + "=" * 80)
        print("ERRORS")
        print("=" * 80)
        for err in result["errors"]:
            print("-", err)


if __name__ == "__main__":
    main()