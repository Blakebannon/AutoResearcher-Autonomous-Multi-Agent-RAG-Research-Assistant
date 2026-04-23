from src.graph.workflow import retrieval_node
from src.rag_pipeline import load_vectorstore

# Load your existing retriever from Week 1/2
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

test_state = {
    "query": "What are the latest AI regulations in the US?",
    "tasks": [
        {
            "subquestion": "What are the current federal AI regulations in the US?",
            "source_preference": "web",
            "rationale": "Current regulations are best found on the web.",
        },
        {
            "subquestion": "How do my uploaded documents discuss AI governance?",
            "source_preference": "local",
            "rationale": "This should come from local docs.",
        },
        {
            "subquestion": "How do existing laws and recent policy updates apply to AI?",
            "source_preference": "hybrid",
            "rationale": "Needs both local and web evidence.",
        },
    ],
    "evidence": [],
    "research_summary": "",
    "final_answer": "",
    "route_log": [],
    "errors": [],
    "iteration": 0,
}

result = retrieval_node(test_state, retriever)

print("\nROUTE LOG:")
for item in result["route_log"]:
    print("-", item)

print("\nERRORS:")
for err in result["errors"]:
    print("-", err)

print("\nEVIDENCE COUNT:", len(result["evidence"]))

print("\nFIRST 3 EVIDENCE ITEMS:")
for item in result["evidence"][:3]:
    print(item)
    print("-" * 80)