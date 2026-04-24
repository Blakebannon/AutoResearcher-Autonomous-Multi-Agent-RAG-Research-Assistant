## AutoResearcher  
Autonomous Multi-Agent Research System (RAG + LangGraph + Streamlit)

---

### Overview

AutoResearcher is a production-style autonomous research assistant that combines Retrieval-Augmented Generation (RAG), multi-agent orchestration, self-critique, and evaluation loops to produce reliable, source-backed answers.

Rather than simply generating responses, the system plans how to answer a query, retrieves relevant information from documents and the web, synthesizes a response, critiques its own output, and evaluates the final result.

---

### Key Features

- Document Intelligence  
  PDF ingestion, chunking, and embedding with a persistent vector database (Chroma)

- Hybrid Retrieval  
  Combines local document retrieval with web search using Tavily

- Multi-Agent Workflow  
  Planner → Retrieval → Reranker → Researcher → Synthesizer → Critic → Judge

- Self-Correcting System  
  The critic agent identifies weaknesses and can trigger revisions

- LLM-as-Judge Evaluation  
  Measures groundedness, citation quality, completeness, and clarity

- Source Transparency  
  All answers are backed by traceable evidence with source metadata and content previews

- Production-Style Interface  
  Streamlit UI with file upload, indexing, workflow trace, report export, and session history

---

### Architecture

User Query  
→ Planner Agent  
→ Retrieval Layer (Documents + Web)  
→ Reranker  
→ Researcher Agent  
→ Synthesizer Agent  
→ Critic Agent (self-evaluation loop)  
→ Judge Agent (scoring and feedback)  
→ Final Answer with Sources and Metrics

---

### Tech Stack

- LLM: Groq (LLaMA 3)  
- Frameworks: LangChain, LangGraph  
- Vector Database: ChromaDB  
- Embeddings: HuggingFace MiniLM  
- Frontend: Streamlit  
- Search: Tavily API  

---

### Key Engineering Challenges Solved

- Prevented empty vectorstore failures through runtime guardrails  
- Handled Streamlit ephemeral storage and deployment constraints  
- Designed a self-correcting multi-agent workflow  
- Implemented reliable evidence tracking and citation handling  
- Built LLM-based evaluation metrics for answer quality  
- Resolved dependency conflicts across Python, NumPy, and protobuf  

---

### Running Locally

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run streamlit_app.py


Author: Blake Bannon (Red Rocks Technology Group, LLC)