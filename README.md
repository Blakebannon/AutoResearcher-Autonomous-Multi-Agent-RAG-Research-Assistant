# AutoResearcher

🔗 Live Demo: https://autoresearcher-autonomous-multi-agent-rag-research-assistant-6.streamlit.app/

AutoResearcher is an autonomous multi-agent research system built with LangGraph, hybrid retrieval (local + web), evidence tracking, self-critique, and LLM-as-judge evaluation.

## Key Features

- Multi-agent workflow (Planner → Retrieval → Researcher → Synthesizer → Critic → Judge)
- Hybrid retrieval (local documents + web search)
- Evidence-based answers with citations
- Self-correcting reasoning loop
- LLM-as-judge evaluation (groundedness, clarity, completeness)
- Full workflow trace + observability UI

## System Architecture

User Query  
→ Planner  
→ Retrieval (Local / Web / Hybrid)  
→ Reranker  
→ Researcher  
→ Synthesizer  
→ Critic  
→ Revision Loop (if needed)  
→ Judge  
→ Final Answer + Metrics

## Tech Stack

- LangChain + LangGraph
- ChromaDB (vector store)
- Sentence Transformers (embeddings)
- Tavily (web search)
- Streamlit (UI + deployment)

## Local Setup

```bash
git clone <repo>
cd AutoResearcher
pip install -r requirements.txt

# create .env
GROQ_API_KEY=...
TAVILY_API_KEY=...

streamlit run streamlit_app.py

