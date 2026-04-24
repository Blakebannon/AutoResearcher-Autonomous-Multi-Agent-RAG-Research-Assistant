import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.graph.state import AgentState

load_dotenv()


def get_judge_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1200,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def parse_judge_response(raw_text: str) -> dict:
    try:
        cleaned = raw_text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```").removesuffix("```").strip()

        parsed = json.loads(cleaned)

        return {
            "evaluation": {
                "groundedness_score": float(parsed.get("groundedness_score", 0)),
                "citation_score": float(parsed.get("citation_score", 0)),
                "completeness_score": float(parsed.get("completeness_score", 0)),
                "clarity_score": float(parsed.get("clarity_score", 0)),
                "overall_score": float(parsed.get("overall_score", 0)),
                "judge_feedback": parsed.get("judge_feedback", ""),
            }
        }

    except Exception:
        return {
            "evaluation": {
                "groundedness_score": 0.0,
                "citation_score": 0.0,
                "completeness_score": 0.0,
                "clarity_score": 0.0,
                "overall_score": 0.0,
                "judge_feedback": f"Judge response could not be parsed. Raw response: {raw_text[:500]}",
            }
        }


def judge_node(state: AgentState) -> dict:
    llm = get_judge_llm()

    query = state["query"]
    final_answer = state.get("final_answer", "")
    research_summary = state.get("research_summary", "")
    evidence_context = state.get("evidence_context", "")
    critic_feedback = state.get("critic_feedback", "")
    needs_revision = state.get("needs_revision", False)

    system_prompt = """
You are an LLM-as-judge evaluator for an autonomous research assistant.

Your job is NOT to revise the answer.
Your job is to score the final answer for quality and observability.

Evaluate the answer using the provided evidence only.

Scoring:
- 1.0 = excellent
- 0.8 = good
- 0.6 = acceptable
- 0.4 = weak
- 0.2 = poor
- 0.0 = failing

Metrics:

1. groundedness_score
Does the answer stay supported by the provided evidence?

2. citation_score
Are major factual claims cited with evidence IDs like [doc_1] or [web_2]?
Do citations appear relevant?

3. completeness_score
Does the answer address the original query sufficiently?

4. clarity_score
Is the answer clear, organized, and useful?

5. overall_score
Holistic score based on all criteria.

Return ONLY valid JSON with this exact schema:

{
  "groundedness_score": 0.0,
  "citation_score": 0.0,
  "completeness_score": 0.0,
  "clarity_score": 0.0,
  "overall_score": 0.0,
  "judge_feedback": "Brief explanation of the scores."
}
"""

    human_prompt = f"""
Original User Query:
{query}

Final Answer:
{final_answer}

Research Summary:
{research_summary}

Evidence:
{evidence_context}

Critic Decision:
needs_revision = {needs_revision}

Critic Feedback:
{critic_feedback}
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    return parse_judge_response(response.content)