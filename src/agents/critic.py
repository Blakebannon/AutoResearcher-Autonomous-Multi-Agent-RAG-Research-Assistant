import json
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=800,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def critic_node(state):
    llm = get_llm()

    query = state["query"]
    research_summary = state["research_summary"]
    final_answer = state["final_answer"]
    evidence = state["evidence"]

    compact_evidence = []
    for item in evidence[:8]:
        compact_evidence.append(
            {
                "source": item.get("source"),
                "chunk_id": item.get("chunk_id"),
                "tool_used": item.get("tool_used"),
                "url": item.get("url"),
                "content_preview": item.get("content", "")[:400],
            }
        )

    system_prompt = """
    You are a strict but practical reviewer for an autonomous research assistant.

    Your job is to determine whether the final answer is acceptable or needs revision.

    Evaluate based on:
    1. Groundedness — Are claims supported by the evidence?
    2. Completeness — Does it sufficiently answer the query?
    3. Accuracy — Any contradictions or incorrect claims?
    4. Clarity — Is the answer understandable and logically structured?

    Rules:
    - DO NOT request revision for minor wording or style issues
    - DO NOT request revision if the answer is already useful and mostly correct
    - ONLY request revision if there is a meaningful problem

    Decision criteria:
    - If the answer has unsupported claims → needs_revision = true
    - If the answer misses major parts of the query → needs_revision = true
    - Otherwise → needs_revision = false

    Return ONLY valid JSON:

    {
     "needs_revision": true or false,
     "critic_feedback": "Short, precise explanation of what should be fixed (if any)."
    }
    """

    human_prompt = f"""
Original Query:
{query}

Research Summary:
{research_summary}

Final Answer:
{final_answer}

Evidence:
{json.dumps(compact_evidence, indent=2)}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])

    try:
        parsed = json.loads(response.content)
        needs_revision = bool(parsed.get("needs_revision", False))
        critic_feedback = parsed.get("critic_feedback", "")
    except Exception as e:
        needs_revision = False
        critic_feedback = f"Critic JSON parse failed; defaulting to no revision. Error: {str(e)}"

    return {
        "needs_revision": needs_revision,
        "critic_feedback": critic_feedback,
    }