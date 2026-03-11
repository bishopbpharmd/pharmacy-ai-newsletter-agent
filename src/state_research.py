
"""
State Definitions and Pydantic Schemas for Research Agent

This module defines the state objects and structured schemas used for
the research agent workflow, including researcher state management and output schemas.
"""

import operator
from typing import Any
from typing_extensions import TypedDict, Annotated, List, Sequence, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from src.evidence_utils import (
    merge_evidence_ledgers,
    merge_observability_events,
    merge_retrieval_events,
)

# ===== CUSTOM REDUCERS FOR DICT MERGING =====

def merge_dicts(left: Dict, right: Dict) -> Dict:
    """Merge two dicts, preserving all keys from both.
    
    This reducer implements a DELTA approach:
    - Nodes return only NEW data (deltas)
    - Reducer accumulates/extends existing data
    
    For dict values (nested), merge recursively (preserve all keys).
    For list values, EXTEND (accumulate) - nodes return only new items.
    For other values, right takes precedence.
    
    This ensures:
    1. Nodes return only deltas (new questions/messages)
    2. Reducer accumulates them into existing state
    3. All perspectives are preserved
    """
    if not isinstance(left, dict):
        return right if isinstance(right, dict) else {}
    if not isinstance(right, dict):
        return left
    
    # Start with left (existing state)
    result = left.copy()
    
    # Update with right (delta from node), merging nested dicts and extending lists
    for key, right_value in right.items():
        if key in result:
            left_value = result[key]
            # If both are dicts, merge recursively (preserve all keys)
            if isinstance(left_value, dict) and isinstance(right_value, dict):
                result[key] = merge_dicts(left_value, right_value)
            # If both are lists, EXTEND (accumulate) - node returns only new items
            elif isinstance(left_value, list) and isinstance(right_value, list):
                result[key] = left_value + right_value  # Accumulate: existing + new
            # For other types, right takes precedence
            else:
                result[key] = right_value
        else:
            # New key, add it
            result[key] = right_value
    return result

# ===== STATE DEFINITIONS =====

class ResearcherState(TypedDict):
    """
    State for the research agent containing message history and research metadata.

    This state tracks the researcher's conversation, iteration count for limiting
    tool calls, the research topic being investigated, compressed findings,
    and raw research notes for detailed analysis.
    """
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    research_brief: str
    article_summary: str
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    # STORM-specific fields
    perspectives: List[str]  # List of discovered perspectives/personas
    current_perspective: str  # Current perspective being used for questioning
    conversation_round: int  # Current round in conversation simulation
    expert_responses: Annotated[List[str], operator.add]  # Accumulated expert responses
    should_continue_conversation: bool  # Whether writer wants to continue asking questions
    draft_report: str  # Current draft report (passed from supervisor, used throughout STORM)
    search_type: str  # Source policy guidance: "internal", "external", or "both"
    perspective_messages: Annotated[dict, merge_dicts]  # Dict[str, List[BaseMessage]] - Isolated messages per perspective for LangSmith separation
    perspective_research_plans: dict  # Dict[str, str] - Research plan for each perspective (set by generate_research_plans node)
    perspective_profiles: dict  # Dict[str, str] - Bio / behavioral profile for each perspective
    forced_perspectives: List[str]  # Optional perspective subset selected by router for this gap
    active_gap_id: str  # Active gap id being researched in this invocation
    reuse_existing_research_plans: bool  # When true, skip plan regeneration if plans already exist
    # Q&A Reflection and Retry fields
    is_retry_attempt: bool  # Whether current writer_node call is a retry (uses retry_query instead of generating new question)
    retry_query: str  # The alternative query to use for retry attempt
    original_qa: dict  # Stores {"question": ..., "answer": ...} from first attempt for synthesis
    current_qa_id: str  # Stable identifier for the active Q&A, preserved across retry/synthesis
    last_question: str  # The most recent question asked (for reflection evaluation)
    last_answer: str  # The most recent answer received (for reflection evaluation)
    retry_tool_name: str  # Optional tool override for the next retry attempt
    current_retrieval_plan: dict  # Structured plan for the current retrieval attempt
    last_retrieval_metadata: dict  # Structured metadata about the last retrieval execution
    retrieval_events: Annotated[List[Dict[str, Any]], merge_retrieval_events]  # Execution telemetry for observability
    conversation_route_reason: str  # Latest route decision within the STORM subgraph
    observability_events: Annotated[List[Dict[str, Any]], merge_observability_events]  # Structured routing/clipping/stop telemetry
    last_evidence: List[Dict[str, Any]]  # Structured evidence extracted from the most recent tool results
    evidence_ledger: Annotated[List[Dict[str, Any]], merge_evidence_ledgers]  # Finalized Q&A + evidence records

class ResearcherOutputState(TypedDict):
    """
    Output state for the research agent containing final research results.

    This represents the final output of the research process with compressed
    research findings and all raw notes from the research process.
    """
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    evidence_ledger: Annotated[List[Dict[str, Any]], merge_evidence_ledgers]
    retrieval_events: Annotated[List[Dict[str, Any]], merge_retrieval_events]
    observability_events: Annotated[List[Dict[str, Any]], merge_observability_events]
    perspectives: List[str]
    perspective_research_plans: dict
    perspective_profiles: dict

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ResearchQuestion(BaseModel):
    """Schema for research brief generation."""
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class Summary(BaseModel):
    """Schema for webpage content summarization."""
    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")

class PerspectiveList(BaseModel):
    """Schema for perspective discovery output."""
    perspectives: List[str] = Field(
        description="List of exactly 3 distinct perspectives/personas to guide questioning. Should include 2 diverse perspectives plus one 'basic fact' perspective.",
        min_length=3,
        max_length=3
    )

class WriterQuestion(BaseModel):
    """Schema for writer node question generation."""
    question: str = Field(
        description="A specific question grounded in the current perspective to ask the expert"
    )
    should_continue: bool = Field(
        description="Whether to continue asking more questions or conclude research for this perspective"
    )

class ResearchPlan(BaseModel):
    """Schema for perspective-specific research plan generation."""
    research_plan: str = Field(
        description="A focused research plan (3-5 sentences) identifying specific topics and information to seek for this perspective"
    )

class QAReflection(BaseModel):
    """Schema for Q&A reflection and retry decision."""
    answer_quality: str = Field(
        description="Brief assessment of the answer quality: 'sufficient' (answer addresses the question adequately), 'insufficient' (answer is missing key information or failed to retrieve relevant data), or 'off_target' (answer doesn't address the question's intent)"
    )
    needs_retry: bool = Field(
        description="Whether a retry with a different query is needed. True only if the answer is insufficient/off_target AND a different query approach could yield better results."
    )
    retry_query: str = Field(
        default="",
        description="If needs_retry is True, provide exactly one alternative query on a single line that captures the same intent with different phrasing/keywords. Max 25 words. Leave empty if no retry is needed."
    )
    suggested_tool: str = Field(
        default="",
        description="If the retry should use a different tool, return either 'retrieve_document_chunks' or 'tavily_search'. Leave empty if the same tool should be reused."
    )
    rewrite_reason: str = Field(
        default="",
        description="Short explanation of what should change about the query if a retry is recommended."
    )
    reasoning: str = Field(
        description="Brief explanation of the assessment and decision."
    )
