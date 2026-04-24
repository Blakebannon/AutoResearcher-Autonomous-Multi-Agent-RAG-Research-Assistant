from typing import TypedDict, List, Optional, Literal

from src.schemas.evidence import Evidence


class ResearchTask(TypedDict):
    subquestion: str
    source_preference: Literal["local", "web", "hybrid"]
    rationale: str


class EvidenceItem(TypedDict):
    """
    Legacy evidence dict shape.

    Kept temporarily for backward compatibility.
    We are now migrating toward the Evidence Pydantic model.
    """

    content: str
    source: str
    chunk_id: Optional[str]
    tool_used: str
    url: Optional[str]


class AgentState(TypedDict):
    query: str
    tasks: List[ResearchTask]

    evidence: List[Evidence]
    evidence_context: str

    research_summary: str
    final_answer: str

    route_log: List[str]
    errors: List[str]

    iteration: int
    max_iterations: int

    needs_revision: bool
    critic_feedback: str

    evaluation: dict