
"""
State Definitions for Multi-Agent Research Supervisor

This module defines the state objects and tools used for the multi-agent
research supervisor workflow, including coordination state and research tools.
"""

import operator
from typing_extensions import Annotated, Literal, TypedDict, Sequence

from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from src.evidence_utils import (
    merge_evidence_ledgers,
    merge_gap_cards,
    merge_gap_ledger,
    merge_observability_events,
    merge_retrieval_events,
)

AgendaStatus = Literal["active", "partial", "completed", "deferred", "not_answerable"]
AgendaPriority = Literal["high", "medium", "low"]
SearchType = Literal["internal", "external", "both"]
WorkMode = Literal[
    "normal_research",
    "boundary_with_artifact_check",
    "limitation_to_draft",
    "close_unavailable",
]
ArtifactAvailability = Literal["available", "unavailable", "unknown"]
PriorityAction = Literal["research", "refine_draft", "finalize_candidate"]
ProgressAction = Literal["continue_research", "finalize"]

class SupervisorState(TypedDict):
    """
    State for the multi-agent research supervisor.

    Manages coordination between supervisor and research agents, tracking
    research progress and accumulating findings from multiple sub-agents.
    """

    # Messages exchanged with supervisor for coordination and decision-making
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Detailed research brief that guides the overall research direction
    research_brief: str
    # Article summary from document ingestion
    article_summary: str
    # Selected newsletter template type (research_article or commentary)
    newsletter_template: str
    # Processed and structured notes ready for final report generation
    notes: Annotated[list[str], operator.add] = []
    # Explicit unresolved gap backlog used by router.
    gap_cards: Annotated[list[dict], merge_gap_cards] = []
    # Gap lifecycle ledger used for evidence-driven convergence.
    gap_ledger: Annotated[list[dict], merge_gap_ledger] = []
    # Minimal structured evidence returned from ConductResearch rounds
    evidence_ledger: Annotated[list[dict], merge_evidence_ledgers] = []
    # Structured retrieval telemetry returned from ConductResearch rounds
    retrieval_events: Annotated[list[dict], merge_retrieval_events] = []
    # Structured routing/stop telemetry
    observability_events: Annotated[list[dict], merge_observability_events] = []
    # Counter tracking the number of research iterations performed
    research_iterations: int = 0
    # Counter tracking the number of STORM research rounds (ConductResearch calls)
    storm_rounds: int = 0
    # Counter tracking the number of draft editing rounds (refine_draft_report calls)
    draft_editing_rounds: int = 0
    # Raw unprocessed research notes collected from sub-agent research
    raw_notes: Annotated[list[str], operator.add] = []
    # Draft report
    draft_report: str
    # Novelty summaries from recent research rounds
    research_novelty_history: Annotated[list[dict], operator.add] = []
    # Compact per-round summaries that become the primary working findings payload
    research_round_summaries: Annotated[list[dict], operator.add] = []
    # Number of research round summaries already merged into the draft
    merged_round_count: int = 0
    # Number of agenda update log entries already merged into the draft
    merged_agenda_update_count: int = 0
    # Reused STORM perspectives carried across supervisor rounds
    storm_perspectives: list[str] = []
    # Perspective plans/memory that evolve across follow-up rounds
    storm_perspective_research_plans: dict = {}
    # Latest summary of whether the most recent round materially improved research state
    last_round_impact_summary: str = ""
    # Flag for the latest round-level material improvement judgment
    last_round_material_improvement: bool = False
    # Source-mix telemetry surfaced to the model
    source_mix_summary: str = ""
    internal_rounds: int = 0
    external_rounds: int = 0
    external_grounding_considered: bool = False
    external_grounding_rationale: str = ""
    # Only populated when the supervisor proposes another burst before merging findings
    why_follow_up_before_merge: str = ""
    # Explicit stop / convergence reason for observability
    supervisor_stop_reason: str = ""
    # Explicit latest route reason for observability
    supervisor_route_reason: str = ""
    # Completion/degraded status for downstream observability
    supervisor_completion_status: str = ""
    # Current deterministic supervisor phase for observability/debugging
    supervisor_phase: str = ""
    # Fixed perspectives chosen once for the run
    research_perspectives: list[dict] = []
    # Perspective bios/profiles keyed by perspective name
    perspective_profiles: dict = {}
    # Perspective-specific upfront proposals used to shape the agenda
    perspective_proposals: Annotated[list[dict], operator.add] = []
    # Persistent global research agenda maintained by the agenda manager only
    research_agenda: dict = {}
    # Small delta log of agenda updates across the run
    agenda_update_log: Annotated[list[dict], operator.add] = []
    # Detailed history of executed research assignments/tasks
    task_history: Annotated[list[dict], operator.add] = []
    # Latest prioritization decision
    latest_priority_decision: dict = {}
    # Latest assignment/routing decision for research workers
    latest_assignment: dict = {}
    # Latest aggregated research round payload
    latest_research_round: dict = {}
    # Whether at least one true external-grounding round executed
    external_grounding_completed: bool = False


class FixedPerspective(BaseModel):
    """A perspective chosen once and reused for the whole run."""

    name: str = Field(description="Short stable perspective label.")
    description: str = Field(
        description="What this perspective cares about and why it matters for this article."
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Compact list of concepts or risks this perspective should emphasize.",
    )


class FixedPerspectiveRoster(BaseModel):
    """Upfront fixed perspective selection."""

    perspectives: list[FixedPerspective] = Field(
        min_length=1,
        max_length=3,
        description="Up to three fixed research perspectives for the run.",
    )
    rationale: str = Field(
        description="Why these perspectives cover the article's highest-value questions."
    )


class PerspectiveProposal(BaseModel):
    """Perspective-specific suggestions for the initial global agenda."""

    perspective_name: str = Field(description="Perspective proposing these items.")
    proposed_questions: list[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=4,
        description="High-value questions this perspective believes are worth investigating.",
    )
    external_grounding_needs: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="External context this perspective thinks would materially improve the output.",
    )
    high_value_risks: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Risks, caveats, or blind spots this perspective wants the agenda to track.",
    )


class AgendaItem(BaseModel):
    """Single tracked item in the global agenda."""

    item_id: str = Field(description="Stable short identifier.")
    title: str = Field(description="Short human-readable title.")
    research_question: str = Field(description="Question or task the system is trying to resolve.")
    status: AgendaStatus = Field(
        description="One of: active, partial, completed, deferred, or not_answerable."
    )
    priority: AgendaPriority = Field(description="high, medium, or low.")
    why_it_matters: str = Field(description="Why this matters for the newsletter.")
    completion_criteria: str = Field(description="What would count as enough evidence.")
    recommended_search_type: SearchType = Field(
        description="internal, external, or both depending on where the answer should come from."
    )
    assigned_perspectives: list[str] = Field(
        default_factory=list,
        description="Perspectives best suited to investigate this item.",
    )
    evidence_summary: str = Field(
        default="",
        description="Compact current state of evidence or remaining uncertainty.",
    )
    execution_focus: str = Field(
        default="",
        description="Single executable fact cluster or sub-question to target in the next round.",
    )
    work_mode: WorkMode = Field(
        default="normal_research",
        description=(
            "normal_research, boundary_with_artifact_check, limitation_to_draft, or close_unavailable."
        ),
    )
    internal_focus: str = Field(
        default="",
        description="Article-contained closure task when the item is in a bounded execution mode.",
    )
    external_focus: str = Field(
        default="",
        description="Single bounded public-artifact or external-context step when needed.",
    )
    closure_condition: str = Field(
        default="",
        description="Concrete condition that would count as progress or closure for the next round.",
    )
    artifact_state: dict[str, ArtifactAvailability] = Field(
        default_factory=dict,
        description=(
            "Known availability state for required artifacts or dependencies. "
            "Values should be available, unavailable, or unknown."
        ),
    )
    closure_reason: str = Field(
        default="",
        description="Why the item is currently bounded, deferred, or closed.",
    )
    reopen_only_if: str = Field(
        default="",
        description="Condition required before the item can move back into active work.",
    )
    attempt_count: int = Field(
        default=0,
        description="How many explicit research attempts have been made on this item.",
    )


class GlobalResearchAgenda(BaseModel):
    """Persistent agenda for the run."""

    overall_goals: list[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=5,
        description="Top-level goals for the run.",
    )
    active_items: list[AgendaItem] = Field(default_factory=list)
    partial_items: list[AgendaItem] = Field(default_factory=list)
    completed_items: list[AgendaItem] = Field(default_factory=list)
    deferred_items: list[AgendaItem] = Field(default_factory=list)
    external_grounding_goals: list[str] = Field(
        default_factory=list,
        description="Specific outside-context questions that must be addressed before stopping.",
    )
    agenda_notes: str = Field(
        default="",
        description="Compact narrative note about the current scope and boundaries.",
    )


class PriorityDecision(BaseModel):
    """Choose the next best action from the current agenda."""

    action: PriorityAction = Field(
        description="One of: research, refine_draft, or finalize_candidate."
    )
    item_id: str = Field(
        default="",
        description="Agenda item to work next when action is research.",
    )
    rationale: str = Field(description="Why this is the next best action.")


class PerspectiveAssignment(BaseModel):
    """Assignment for a worker perspective."""

    perspective_name: str = Field(description="Perspective to execute the task.")
    worker_brief: str = Field(
        description="Compact assigned brief for this worker. This should not be a full plan rewrite."
    )


class AssignmentDecision(BaseModel):
    """Route one agenda item to the most relevant perspectives."""

    item_id: str = Field(description="Agenda item being assigned.")
    search_type: SearchType = Field(description="internal, external, or both.")
    assignments: list[PerspectiveAssignment] = Field(
        min_length=1,
        max_length=3,
        description="Perspective-level assignments for this round.",
    )
    rationale: str = Field(description="Why these perspectives and this source mix were chosen.")


class AgendaItemUpdate(BaseModel):
    """Small update to an existing agenda item."""

    item_id: str = Field(description="Agenda item identifier.")
    new_status: AgendaStatus = Field(description="active, partial, completed, deferred, or not_answerable.")
    evidence_summary: str = Field(
        default="",
        description="Compact updated evidence state for the item.",
    )
    recommended_search_type: SearchType | Literal[""] = Field(
        default="",
        description="Optional replacement search type for future work.",
    )
    assigned_perspectives: list[str] = Field(
        default_factory=list,
        description="Optional replacement perspective list.",
    )
    execution_focus: str = Field(
        default="",
        description="Optional replacement execution focus for the next round.",
    )
    work_mode: WorkMode | Literal[""] = Field(
        default="",
        description=(
            "Optional replacement work mode: normal_research, boundary_with_artifact_check, limitation_to_draft, or close_unavailable."
        ),
    )
    internal_focus: str = Field(
        default="",
        description="Optional replacement article-contained closure task.",
    )
    external_focus: str = Field(
        default="",
        description="Optional replacement bounded external step.",
    )
    closure_condition: str = Field(
        default="",
        description="Optional replacement closure condition for the next round.",
    )
    artifact_state: dict[str, ArtifactAvailability] = Field(
        default_factory=dict,
        description="Optional replacement artifact/dependency state map.",
    )
    closure_reason: str = Field(
        default="",
        description="Optional updated boundary or closure reason.",
    )
    reopen_only_if: str = Field(
        default="",
        description="Optional updated condition required to reopen the item.",
    )


class AgendaUpdateDelta(BaseModel):
    """Small, deliberate agenda update generated after a research round."""

    updates: list[AgendaItemUpdate] = Field(default_factory=list)
    add_items: list[AgendaItem] = Field(default_factory=list, max_length=3)
    completed_item_ids: list[str] = Field(default_factory=list)
    deferred_item_ids: list[str] = Field(default_factory=list)
    agenda_note: str = Field(
        default="",
        description="Short summary of what changed in the agenda.",
    )
    external_grounding_completed: bool = Field(
        default=False,
        description="True only if this round included real external grounding worth counting.",
    )


class ProgressGateDecision(BaseModel):
    """Decision for whether the research program should continue."""

    should_continue: bool = Field(
        description="Whether additional research/refinement should continue."
    )
    recommended_action: ProgressAction = Field(
        description="One of: continue_research or finalize."
    )
    rationale: str = Field(
        description="Why the run should continue or stop."
    )

@tool(parse_docstring=True)
def ConductResearch(
    research_topic: str,
    search_type: str = "both",
    gap_id: str = "",
    perspectives: list[str] | None = None,
) -> str:
    """Tool for delegating a research task to a specialized sub-agent.
    
    This tool is used by the supervisor to delegate research tasks to specialized
    sub-agents. When called, it triggers a dedicated research agent to conduct
    comprehensive research on the specified topic. The research agent will gather
    information, analyze sources, and return compressed research findings.
    
    The supervisor should use this tool when:
    - A specific research topic needs investigation
    - Multiple independent topics need parallel research (make multiple calls)
    - Additional information is needed to fill gaps in the current research
    
    Args:
        research_topic: The gap to research. Keep it compact and decision-oriented. Prefer a small gap card over a long extraction essay: name the gap, say why it matters to the newsletter, note where the answer is likely to live, and state what would count as closure.
        search_type: The source policy to prioritize. Valid values: "internal" for article-grounded extraction, "external" for true web grounding/discovery, or "both" for questions that may bridge the supplied article and external context. Defaults to "both".
        gap_id: Optional gap identifier selected by the router for traceable convergence.
        perspectives: Optional perspective subset for this gap. If omitted, existing persisted perspectives are reused.
    
    Returns:
        Confirmation message that research has been delegated
    """
    # This tool serves as a trigger - actual execution happens in supervisor_tools node
    # The research_topic and search_type are extracted from tool call args and passed to researcher_agent
    selected = ",".join(perspectives or [])
    return (
        f"Research delegated for topic: {research_topic[:100]}... "
        f"(search_type: {search_type}, gap_id: {gap_id or '(none)'}, perspectives: {selected or '(default)'})"
    )

@tool(parse_docstring=True)
def ResearchComplete() -> str:
    """Tool for indicating that the research process is complete.
    
    Call this tool when you have completed all necessary research and are satisfied
    with the comprehensiveness of the research findings. This signals to the system
    that no further research iterations are needed and the process should proceed
    to final report generation.
    
    Use this tool when:
    - All research topics have been thoroughly investigated
    - The research findings are comprehensive and complete
    - You have sufficient information to generate a high-quality report
    - No gaps remain in the research that need additional investigation
    
    Returns:
        Confirmation message that research is complete
    """
    return "Research process marked as complete."
