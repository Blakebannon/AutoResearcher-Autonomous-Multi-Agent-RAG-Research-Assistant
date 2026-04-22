# AutoResearcher: Autonomous Multi-Agent RAG Research Assistant - Day 0 Plan and Roadmap

**Tagline:** Ask any question → get a fully cited, hallucination-resistant research report with custom documents + live web fallback.

Built with LangGraph, LangChain, Chroma, and Groq (or Ollama).  
Part of my journey into Generative AI Engineering (IBM Data Science + IBM RAG & Agentic AI Certified).

## Features (planned)
- Multi-agent workflow (Planner, Retriever, Researcher, Synthesizer, Critic)
- Advanced hybrid RAG + web search
- Self-critique loop for reliability
- Streamlit UI + report export

## Current Status (Week 1)
- Basic RAG pipeline with document upload and querying working
- Live demo: [add link after deployment]

See `docs/` for architecture and weekly progress.

# AutoResearcher: Autonomous Multi-Agent RAG Research Assistant - Day 6 Project update after 1 Week.

## Overview
AutoResearcher is an AI-powered research assistant designed to summarize and extract insights from PDF documents using Retrieval-Augmented Generation (RAG).

## Current Capabilities
- Upload and index PDF documents
- Ask questions about indexed documents
- Retrieve relevant chunks using ChromaDB
- Generate answers with Groq + LangChain
- Run a local UI with Streamlit

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

GROQ_API_KEY=your_key_here

