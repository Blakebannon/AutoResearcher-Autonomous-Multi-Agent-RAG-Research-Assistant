from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from src.utils import get_llm
from src.rag_pipeline import get_retriever
from src.tools.tools import get_tools
from src.services.logger import log_research


SYSTEM_PROMPT = """
You are AutoResearcher, an expert autonomous research assistant.

You have access to two tools:
1. document_retriever -> searches the user's local uploaded documents
2. tavily_search -> searches the web for current or missing information

Rules:
- Use document_retriever first when the question may be answered from local documents.
- Use tavily_search when the answer requires recent or live information, or when the local documents are insufficient.
- Always ground your answers in tool outputs.
- When using document_retriever:
  - explicitly cite sources using the [SOURCE: ... | CHUNK: ...] format when available
  - mention that the answer comes from retrieved document chunks
  - do not guess document titles or source names
- When using tavily_search:
  - include citations with URLs
- Do NOT make up information not supported by tools.
- If uncertain, say so clearly.

Output format:
- Provide a clear, structured answer
- Use bullet points when helpful
- Include a "Sources" section at the end
"""


def build_agent():
    """
    Create and return the ReAct agent for AutoResearcher.
    """
    llm = get_llm()
    retriever = get_retriever(k=5)
    tools = get_tools(retriever)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    return agent


def run_research_query(query: str):
    agent = build_agent()

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]}
    )

    # Log the query + result
    log_research(query, result)

    return result


def get_final_answer(result: dict) -> str:
    """
    Extract the final assistant response from the agent result.
    """
    messages = result["messages"]
    return messages[-1].content