import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1500,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def researcher_node(state):
    llm = get_llm()

    query = state["query"]
    tasks = state["tasks"]
    evidence = state["evidence"]

    # Convert to compact JSON to control tokens
    evidence_json = json.dumps(evidence[:10])  # cap to avoid token explosion

    system_prompt = """
You are a research analyst.

Your job is to analyze retrieved evidence and produce a structured research summary.

Instructions:
- Organize findings by subquestion
- Highlight key insights
- Remove redundant information
- Note if evidence is weak or missing
- DO NOT make up information not present in the evidence
- DO NOT produce a final answer to the user

Return a clean, well-structured research summary.
"""

    human_prompt = f"""
User Query:
{query}

Research Tasks:
{json.dumps(tasks, indent=2)}

Evidence:
{evidence_json}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])

    return {
        "research_summary": response.content
    }