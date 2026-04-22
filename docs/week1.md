# Week 1 Progress – AutoResearcher

## What I Built
This week, I built the first working prototype of AutoResearcher, an AI-powered research assistant that uses Retrieval-Augmented Generation (RAG) to answer questions over PDF documents.

The prototype currently supports:
- Loading PDF documents from the `data/` folder
- Splitting documents into chunks
- Creating embeddings using HuggingFace
- Storing vectors in ChromaDB
- Retrieving relevant context for user questions
- Generating grounded answers using Groq
- Interacting through a Streamlit UI

## Challenges Faced
- Virtual environment setup issues in VS Code
- Python interpreter mismatch between 3.10 and 3.14
- PowerShell execution policy blocking venv activation
- Dependency installation issues across LangChain, Chroma, and embedding libraries
- Understanding how Streamlit runs as a local browser-hosted app

## Lessons Learned
- LangChain dependency management requires careful package installation
- RAG systems rely on multiple layers: loaders, splitters, embeddings, vector stores, and LLMs
- Streamlit is a fast way to turn backend AI logic into a usable prototype
- Debugging environment issues is a major part of real-world AI engineering

## Current Status
The Week 1 prototype is working locally:
- PDFs can be indexed
- Questions can be asked through the UI
- Answers are returned using retrieved document context

## Next Steps
- Improve prompt quality and answer formatting
- Add source citations in responses
- Add better error handling
- Expand into a ReAct-style agent with web search in Week 2