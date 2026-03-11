
"""State Definitions and Pydantic Schemas for Research Scoping.

This defines the state objects and structured schemas used for
the research agent scoping workflow, including researcher state management and output schemas.
"""

import operator
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from src.evidence_utils import (
    merge_evidence_ledgers,
    merge_gap_cards,
    merge_gap_ledger,
    merge_observability_events,
    merge_retrieval_events,
)

# ===== STATE DEFINITIONS =====

class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    # Full absolute path to PDF file if provided (None if no PDF)
    pdf_path: Optional[str] = None

class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.

    Design Considerations:
    - All fields are defined upfront for type safety, IDE support, and LangGraph
      reducer requirements (e.g., add_messages must be specified at definition time).
    - This explicit approach provides clear documentation of data flow but creates
      some duplication with SupervisorState (e.g., supervisor_messages, draft_report).
    - Alternative approach: Keep only cross-cutting fields here and let subgraphs
      define subgraph-specific fields (LangGraph automatically maps compatible fields).
    - Current trade-off: Explicitness and type safety vs. modular separation of concerns.
      For complex multi-agent systems, the current approach is reasonable and maintainable.
    """

    # Selected newsletter template type (must be set by select_newsletter_template node)
    newsletter_template: Optional[str] = None
    # Research brief generated from user conversation history
    research_brief: Optional[str] = None
    # Article summary from document ingestion (replaces research_brief in draft generation)
    article_summary: Optional[str] = None
    # Full raw article content from document ingestion (for draft generation and other uses)
    article_content: Optional[str] = None
    # Messages exchanged with the supervisor agent for coordination
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages] = []
    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[list[str], operator.add] = []
    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add] = []
    # Explicit gap backlog used by the lightweight router.
    gap_cards: Annotated[list[dict], merge_gap_cards] = []
    # Gap lifecycle ledger used for convergence decisions.
    gap_ledger: Annotated[list[dict], merge_gap_ledger] = []
    # Minimal structured evidence carried across handoffs for traceability/export
    evidence_ledger: Annotated[list[dict], merge_evidence_ledgers] = []
    # Structured retrieval telemetry for observability/debugging
    retrieval_events: Annotated[list[dict], merge_retrieval_events] = []
    # Structured routing/clipping/stop telemetry
    observability_events: Annotated[list[dict], merge_observability_events] = []
    # Compact per-round research summaries used as the primary downstream findings payload
    research_round_summaries: Annotated[list[dict], operator.add] = []
    # Number of round summaries already merged into the working draft
    merged_round_count: int = 0
    # Number of agenda update entries already merged into the working draft
    merged_agenda_update_count: int = 0
    # STORM perspectives/plans carried across follow-up rounds
    storm_perspectives: list[str] = []
    storm_perspective_research_plans: dict = {}
    # Latest round-level impact and source-mix state
    last_round_impact_summary: str = ""
    last_round_material_improvement: bool = False
    source_mix_summary: str = ""
    internal_rounds: int = 0
    external_rounds: int = 0
    external_grounding_considered: Optional[bool] = None
    external_grounding_rationale: str = ""
    external_grounding_completed: bool = False
    why_follow_up_before_merge: str = ""
    supervisor_completion_status: str = ""
    supervisor_phase: str = ""
    # Fixed perspective roster and persistent agenda state from the research program
    research_perspectives: list[dict] = []
    perspective_profiles: dict = {}
    perspective_proposals: Annotated[list[dict], operator.add] = []
    research_agenda: dict = {}
    agenda_update_log: Annotated[list[dict], operator.add] = []
    task_history: Annotated[list[dict], operator.add] = []
    latest_priority_decision: dict = {}
    latest_assignment: dict = {}
    latest_research_round: dict = {}
    # Draft research report
    draft_report: str = ""
    # Final formatted research report
    final_report: str = ""
    # Stable newsletter title tracked across rewrite/polish stages
    newsletter_title: str = ""
    # Structured source-paper/article metadata extracted upstream for deterministic export
    source_metadata: Optional[dict] = None
    # Critique-Rewrite loop state
    critique_iterations: int = 0  # Counter for critique-rewrite loops (max 5)
    critique_feedback: str = ""   # Latest critique feedback for rewrite guidance
    # Internal critique fields for routing (cleared after each rewrite)
    _critique_is_complete: Optional[bool] = None
    _critique_strengths: str = ""
    _critique_issues: str = ""
    _critique_actionable: str = ""
    # Structured newsletter output for website import (JSON-serializable)
    structured_newsletter: Optional[dict] = None
    # Final claim-to-source map derived from rendered markdown.
    claim_source_map: Optional[dict] = None
    # Export status for the final artifact writer
    export_status: Optional[str] = None
    # Absolute path to the structured JSON export when available
    structured_output_path: Optional[str] = None
    # Absolute path to the raw markdown export when available
    raw_markdown_path: Optional[str] = None
    # Export error message if structured export fails
    export_error: str = ""
    # Source document filename (for output naming)
    source_filename: Optional[str] = None
    # Full absolute path to PDF file if provided (None if no PDF)
    pdf_path: Optional[str] = None

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class DraftReport(BaseModel):
    """Draft report generation with the specified structure."""

    draft_report: str = Field(
        description="A draft newsletter with the specified structure in markdown format.",
    )


class TemplateSelection(BaseModel):
    """Schema for newsletter template selection."""

    selected_template: str = Field(
        description="The selected newsletter template identifier. Must be one of the available template names (e.g., 'research_article' or 'commentary').",
    )
    reasoning: str = Field(
        description="Brief explanation of why this template was selected based on the article content.",
    )


class CritiqueReflection(BaseModel):
    """Schema for critique reflection on the final report."""

    is_complete: bool = Field(
        description="True if the newsletter is ready for final polish (no major issues). False if it needs another rewrite."
    )
    quality_score: int = Field(
        description="Quality score from 1-10. 8+ typically indicates ready for polish."
    )
    strengths: str = Field(
        description="Brief summary of what the newsletter does well."
    )
    issues: str = Field(
        description="Specific issues that need to be addressed. Empty string if is_complete is True."
    )
    actionable_feedback: str = Field(
        description="Concrete, actionable feedback for the rewriter. Focus on the most impactful improvements. Empty string if is_complete is True."
    )


class GapCard(BaseModel):
    """A single unresolved information gap tracked by the router."""

    gap_id: str = Field(description="Stable identifier for the gap.")
    priority: str = Field(description="Priority label: high, medium, or low.")
    question_intent: str = Field(description="Plain-language gap statement.")
    closure_criteria: str = Field(description="What evidence is needed to close this gap.")
    preferred_search_type: str = Field(
        default="both",
        description="Preferred source mode for this gap: internal, external, or both.",
    )
    preferred_perspectives: List[str] = Field(
        default_factory=list,
        description="Optional perspective hints for this gap.",
    )


class GapLedgerEntry(BaseModel):
    """Lifecycle state for one gap."""

    gap_id: str = Field(description="Gap identifier matching GapCard.gap_id.")
    status: str = Field(
        description="Current status: supported, not_in_source, conflicted, deferred_low_value, or open."
    )
    attempt_count: int = Field(default=0, description="How many times this gap was actively researched.")
    non_progress_count: int = Field(
        default=0,
        description="Consecutive attempts that failed to materially improve this gap.",
    )
    last_search_type: str = Field(default="", description="Search mode used in the latest attempt.")
    last_reason: str = Field(default="", description="Latest routing/closure rationale.")


# ===== STRUCTURED NEWSLETTER OUTPUT =====

class NewsletterSection(BaseModel):
    """A single section of the newsletter."""
    
    name: str = Field(
        description="The section heading (e.g., 'Quick Take', 'Why It Matters', 'What They Did')"
    )
    content: str = Field(
        description="The content of the section (may include markdown formatting like bullets)"
    )


class NewsletterSource(BaseModel):
    """A citation source used in the newsletter."""
    
    id: int = Field(description="Citation number (e.g., 1, 2, 3)")
    title: str = Field(description="Title or description of the source")
    url: Optional[str] = Field(default=None, description="URL of the source if available")


class StructuredNewsletter(BaseModel):
    """
    Structured newsletter output for website import.
    
    Works for both research_article and commentary newsletter structures.
    """
    
    newsletter_structure: str = Field(
        description="Type of newsletter: 'research_article' or 'commentary'"
    )
    title: str = Field(
        description="The newsletter title (H1 heading)"
    )
    sections: List[NewsletterSection] = Field(
        default_factory=list,
        description="Ordered list of newsletter sections"
    )
    sources: List[NewsletterSource] = Field(
        default_factory=list,
        description="List of citation sources"
    )
    raw_markdown: str = Field(
        default="",
        description="The original markdown content for reference"
    )
