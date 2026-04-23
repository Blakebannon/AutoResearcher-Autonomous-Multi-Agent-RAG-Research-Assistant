import os
from dotenv import load_dotenv

load_dotenv()

import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1000,
        api_key=os.getenv("GROQ_API_KEY")
    )

def planner_node(state):
    llm = get_llm()

    query = state["query"]

    system_prompt = """
You are a research planner.

Break the user query into 1 to 3 focused research subquestions.

For each subquestion:
- Assign a source_preference: "local", "web", or "hybrid"
- Provide a short rationale

Rules:
- Keep total subquestions <= 3
- Use "web" for recent/current topics
- Use "local" if documents are likely sufficient
- Use "hybrid" if both may be needed

Return ONLY valid JSON in this format:

[
  {
    "subquestion": "...",
    "source_preference": "...",
    "rationale": "..."
  }
]
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    response = llm.invoke(messages)

    try:
        tasks = json.loads(response.content)
    except Exception as e:
        tasks = []
        print("Planner JSON parse error:", e)

    return {
        "tasks": tasks
    }