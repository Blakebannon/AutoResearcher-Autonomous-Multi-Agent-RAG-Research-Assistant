import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.graph.state import AgentState

load_dotenv()


def get_critic_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1200,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def parse_critic_response(raw_text: str) -> dict:
    try:
        cleaned = raw_text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```").removesuffix("```").strip()

        parsed = json.loads(cleaned)

        return {
            "needs_revision": bool(parsed.get("needs_revision", True)),
            "critic_feedback": parsed.get(
                "critic_feedback",
                "Critic response did not include feedback.",
            ),
        }

    except Exception:
        return {
            "needs_revision": True,
            "critic_feedback": (
                "Critic response could not be parsed as valid JSON. "
                "Raw response: " + raw_text[:800]
            ),
        }


def critic_node(state: AgentState) -> dict:
    llm = get_critic_llm()

    query = state["query"]
    final_answer = state.get("final_answer", "")
    research_summary = state.get("research_summary", "")
    evidence_context = state.get("evidence_context", "")
    errors = state.get("errors", [])

    system_prompt = """
You are the critic/reviewer agent in an autonomous research system.

Your job is to evaluate whether the final answer is sufficiently grounded,
complete, and properly cited.

Evaluate these criteria:

1. Groundedness
- Does the final answer use only the provided evidence?
- Are there unsupported claims?

2. Citation Integrity
- Does every major factual claim include a citation ID such as [doc_1] or [web_2]?
- Do the cited evidence blocks actually support the claims?
- Are citations used accurately and not decoratively?

3. Completeness
- Does the answer address the user's original query?
- Are important caveats or evidence gaps mentioned?

4. Safety Against Hallucination
- Does the answer avoid adding facts that are not present in the evidence?
- Does it avoid overclaiming beyond what the evidence supports?

Return ONLY valid JSON with this exact schema:

{
  "needs_revision": true,
  "critic_feedback": "Specific explanation of what must be fixed."
}

Set "needs_revision" to false only if the answer is grounded, adequately complete,
and citation-supported.

If revision is needed, be specific and concise. Mention exactly what the revision
agent should fix.
"""

    human_prompt = f"""
Original User Query:
{query}

Research Summary:
{research_summary}

Final Answer:
{final_answer}

Evidence:
{evidence_context}

Retrieval Errors:
{errors}
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    return parse_critic_response(response.content)