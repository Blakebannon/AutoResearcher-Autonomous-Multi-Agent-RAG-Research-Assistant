from typing import TypedDict, List, Optional, Literal

class ResearchTask(TypedDict):
    subquestion: str
    source_preference: Literal["local", "web", "hybrid"]
    rationale: str

class EvidenceItem(TypedDict):
    content: str
    source: str
    chunk_id: Optional[str]
    tool_used: str
    url: Optional[str]

class AgentState(TypedDict):
    query: str
    tasks: List[ResearchTask]
    evidence: List[EvidenceItem]
    research_summary: str
    final_answer: str
    route_log: List[str]
    errors: List[str]
    iteration: int