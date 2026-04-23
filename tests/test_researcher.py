from src.agents.researcher import researcher_node
from src.graph.workflow import retrieval_node
from src.rag_pipeline import load_vectorstore

# Load retriever
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Simulated graph state
test_state = {
    "query": "What are the latest AI regulations in the US?",
    "tasks": [
        {
            "subquestion": "What are the current federal AI regulations in the US?",
            "source_preference": "web",
            "rationale": "Current regulations are best found on the web.",
        },
        {
            "subquestion": "What are the state-level AI regulations in the US?",
            "source_preference": "web",
            "rationale": "State policy is likely freshest on the web.",
        },
    ],
    "evidence": [],
    "research_summary": "",
    "final_answer": "",
    "route_log": [],
    "errors": [],
    "iteration": 0,
}

# First run retrieval
retrieval_result = retrieval_node(test_state, retriever)
test_state.update(retrieval_result)

# Then run researcher
research_result = researcher_node(test_state)
test_state.update(research_result)

print("\nRESEARCH SUMMARY:\n")
print(test_state["research_summary"])