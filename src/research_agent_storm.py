
"""STORM-based Research Agent Implementation.

This module implements a STORM (Structured Organization of Research and Multi-perspective) 
research agent that uses perspective discovery and conversation simulation for comprehensive research.

STORM Architecture:
1. Perspective Discovery: Identifies diverse viewpoints/personas
2. Conversation Simulation: Multi-turn conversation between Writer (asks questions) and Expert (researches answers)
"""

import asyncio
import os
import re
import time
from typing_extensions import Literal
from typing import Any, List, Optional, Dict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage

from src.model_config import (
    PIPELINE_MODEL_SETTINGS,
    build_chat_model,
)
from src.state_research import (
    ResearcherState, 
    ResearcherOutputState,
    PerspectiveList,
    QAReflection
)
from src.utils import (
    DEFAULT_DOC_ID,
    tavily_search, 
    get_today_str, 
    retrieve_document_chunks
)
from src.prompts import (
    storm_perspective_discovery_prompt,
    storm_research_plan_prompt,
    storm_writer_prompt,
    compress_research_system_prompt,
    compress_research_human_message,
    expert_synthesize_prompt,
    qa_reflection_prompt,
    answer_synthesis_prompt
)
from src.evidence_utils import (
    canonicalize_answer_summary,
    canonicalize_evidence_ledger,
    merge_evidence_items as merge_canonical_evidence_items,
    source_mix_report,
)
from src.document_profile import infer_document_profile
from src.logging_config import (
    get_logger, 
    summarize_message, 
    log_token_usage, 
    get_global_tracker
)

# ===== LOGGING SETUP =====

logger = get_logger("deep_research.research_agent_storm")

# ===== CONFIGURATION =====

# Set up tools for writer node (Writer asks questions and calls tools)
writer_tools = [tavily_search, retrieve_document_chunks]
writer_tools_by_name = {tool.name: tool for tool in writer_tools}

# Initialize models
perspective_model = build_chat_model(PIPELINE_MODEL_SETTINGS.storm_perspective_discovery_model)
research_plan_model = build_chat_model(PIPELINE_MODEL_SETTINGS.storm_research_plan_model)
writer_model = build_chat_model(PIPELINE_MODEL_SETTINGS.storm_writer_model)
writer_model_with_tools = writer_model.bind_tools(writer_tools)  # Writer calls tools
expert_model = build_chat_model(PIPELINE_MODEL_SETTINGS.storm_expert_synthesis_model)
reflection_model = build_chat_model(PIPELINE_MODEL_SETTINGS.storm_retry_reflection_model)
synthesis_model = build_chat_model(PIPELINE_MODEL_SETTINGS.storm_answer_synthesis_model)
compress_model = build_chat_model(PIPELINE_MODEL_SETTINGS.storm_compression_model)

# STORM configuration
MAX_CONVERSATION_ROUNDS = int(
    os.environ.get("DEEP_RESEARCH_STORM_MAX_CONVERSATION_ROUNDS", "2")
)  # Short but real iterative questioning per perspective
MAX_PERSPECTIVES = 3  # Number of perspectives to explore
MAX_STORM_ROUNDS = 3  # Maximum number of STORM research rounds (perspective discovery + Q&A cycles)
MAX_DRAFT_EDITING_ROUNDS = 3  # Maximum number of draft editing rounds

logger.info("STORM research agent initialized with tools: %s", [t.name for t in writer_tools])

# ===== HELPER FUNCTIONS =====

def _truncate_text(text: str, max_chars: int = 280) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")


def _normalize_question_text(text: str) -> str:
    if not text:
        return ""
    for line in str(text).splitlines():
        stripped = re.sub(r"\s+", " ", line).strip()
        if stripped:
            return stripped
    return ""


def _extract_supervisor_focus_query(research_topic: str, research_plan: str = "") -> str:
    for text, prefixes in (
        (research_plan, ("Priority question:", "Primary evidence gap:")),
        (research_topic, ("Research question:",)),
    ):
        for raw_line in str(text or "").splitlines():
            stripped = re.sub(r"\s+", " ", raw_line).strip()
            for prefix in prefixes:
                if stripped.lower().startswith(prefix.lower()):
                    candidate = stripped[len(prefix):].strip()
                    if candidate:
                        if candidate[-1] not in {"?", "."}:
                            candidate += "?"
                        return candidate
    return ""


def _normalize_retry_query(text: str, max_chars: int = 220) -> str:
    """Reduce model-generated retry instructions to one tool-ready query line."""
    if not text:
        return ""

    candidate = ""
    for raw_line in str(text).splitlines():
        stripped = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", raw_line).strip()
        if stripped:
            candidate = stripped
            break

    if not candidate:
        candidate = str(text).strip()

    candidate = re.sub(
        r"^(?:retry query|alternative query|query)\s*:\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    candidate = re.sub(r"\s+", " ", candidate)
    candidate = re.split(r"\s+(?:\d+[.)]|[-*])\s+", candidate, maxsplit=1)[0].strip()

    sentence_match = re.match(rf"(.{{1,{max_chars}}}?[?.!])(?:\s|$)", candidate)
    if sentence_match:
        candidate = sentence_match.group(1).strip()

    if len(candidate) > max_chars:
        candidate = candidate[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")

    return candidate.strip(' "\'')


def _query_fingerprint(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _find_duplicate_query_event(
    retrieval_events: List[Dict[str, Any]] | None,
    retrieval_query: str,
    tool_name: str,
    active_gap_id: str = "",
) -> Optional[Dict[str, Any]]:
    fingerprint = _query_fingerprint(retrieval_query)
    if not fingerprint or not tool_name:
        return None

    normalized_gap_id = str(active_gap_id or "").strip()
    for event in reversed(list(retrieval_events or [])):
        stage = str(event.get("stage", "") or "").strip().lower()
        if stage and stage not in {"planned", "executed", "executed_fallback", "skipped_duplicate"}:
            continue
        event_tool = str(event.get("tool_name", "") or "").strip()
        if event_tool != tool_name:
            continue
        event_gap = str(event.get("gap_id", "") or "").strip()
        if normalized_gap_id and event_gap and event_gap != normalized_gap_id:
            continue
        prior_query = str(
            event.get("retrieval_query", "")
            or event.get("query", "")
            or event.get("original_question", "")
            or ""
        ).strip()
        if _query_fingerprint(prior_query) == fingerprint:
            return dict(event)
    return None


_EXTERNAL_SEARCH_HINTS = {
    "alert fatigue",
    "author background",
    "best practice",
    "best practices",
    "benchmark",
    "benchmarks",
    "broader context",
    "broader landscape",
    "cms",
    "comparator study",
    "comparator studies",
    "current practice",
    "deployment",
    "fda",
    "field benchmark",
    "governance",
    "guideline",
    "guidelines",
    "external literature",
    "institution background",
    "monitoring",
    "operational literature",
    "policy",
    "prior studies",
    "prior work",
    "regulation",
    "regulatory",
    "recent studies",
    "recent study",
    "related literature",
    "related work",
    "similar studies",
    "validation data",
    "warning",
    "website",
}

_EXTERNAL_ARTIFACT_HINTS = {
    "artifact",
    "code",
    "data availability",
    "dataset download",
    "download link",
    "github",
    "project page",
    "repo",
    "repository",
    "source code",
    "supplement link",
    "supplementary file",
    "supplementary materials",
    "where can i find",
}

_ARTICLE_INTERNAL_HINTS = {
    "appendix",
    "author affiliations",
    "authors",
    "block duration",
    "cohort",
    "counting",
    "cross-validation",
    "dataset",
    "debounce",
    "definition",
    "definitions",
    "false positive",
    "figure",
    "framework",
    "in the article",
    "in the paper",
    "in the study",
    "inclusion",
    "leakage",
    "limitations",
    "methods",
    "model",
    "notification",
    "operational",
    "patients",
    "results",
    "sample size",
    "split",
    "supplementary",
    "table",
    "threshold",
    "title",
    "tp",
    "true positive",
    "window",
}

_RESEARCH_TOPIC_INTERNAL_ONLY_HINTS = (
    "do not search the web",
    "extract from the pre-loaded user_article",
    "only using internal retrieval",
    "pre-loaded user_article",
    "retrieve_document_chunks only",
    "use retrieve_document_chunks only",
)

_RESEARCH_TOPIC_EXTERNAL_CONTEXT_HINTS = (
    "add purposeful external grounding",
    "broader professional context",
    "current guidelines",
    "current practice",
    "deployment/governance",
    "external context points",
    "field context",
    "recent comparator studies",
    "recent studies",
)

_LOW_INFORMATION_PATTERNS = (
    re.compile(r"^\s*in\s+[A-Z][A-Za-z-]+\s+et\s+al\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:this|the)\s+(?:article|study|paper)\s*$", re.IGNORECASE),
)

_LOCAL_FACT_CLUSTERS = (
    (
        "performance_metrics",
        (
            "auroc",
            "auprc",
            "c-recall",
            "c-precision",
            "c-specificity",
            "c-npv",
            "mcs",
            "performance metric",
            "metrics",
        ),
        "What exact performance metrics does the article report?",
    ),
    (
        "denominators",
        (
            "sample size",
            "sample sizes",
            "denominator",
            "denominators",
            "patient stays",
            "events",
            "positive blocks",
            "negative blocks",
            "counts",
        ),
        "What exact sample sizes, event counts, or denominators does the article report?",
    ),
    (
        "definitions",
        (
            "definition",
            "definitions",
            "define",
            "defined",
            "formula",
            "formulas",
            "formal definition",
        ),
        "How does the article define the relevant method or metric?",
    ),
    (
        "notification_logic",
        (
            "notification",
            "threshold",
            "suppression",
            "repeat-alert",
            "collapse",
            "window",
            "block length",
            "prediction block",
        ),
        "What exact notification or threshold logic does the article report?",
    ),
    (
        "splits_leakage",
        (
            "split",
            "splits",
            "validation",
            "cross-validation",
            "fold",
            "folds",
            "partition",
            "leakage",
        ),
        "How were the data splits, cross-validation, and leakage controls defined?",
    ),
)


def _query_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9/_-]*", str(text or "")))


def _normalize_scope_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _text_contains_hint(text: str, hint: str) -> bool:
    normalized_text = _normalize_scope_text(text)
    normalized_hint = _normalize_scope_text(hint)
    if not normalized_text or not normalized_hint:
        return False

    if " " in normalized_hint or "-" in normalized_hint:
        return normalized_hint in normalized_text

    return re.search(
        rf"(?<![a-z0-9]){re.escape(normalized_hint)}(?![a-z0-9])",
        normalized_text,
    ) is not None


def _matching_hints(text: str, hints) -> List[str]:
    normalized_text = _normalize_scope_text(text)
    return [hint for hint in hints if _text_contains_hint(normalized_text, hint)]


def _research_topic_enforces_internal_only(research_topic: str) -> bool:
    normalized_topic = _normalize_scope_text(research_topic)
    return any(
        _text_contains_hint(normalized_topic, hint)
        for hint in _RESEARCH_TOPIC_INTERNAL_ONLY_HINTS
    )


def _research_topic_prefers_external_context(research_topic: str) -> bool:
    normalized_topic = _normalize_scope_text(research_topic)
    return any(
        _text_contains_hint(normalized_topic, hint)
        for hint in _RESEARCH_TOPIC_EXTERNAL_CONTEXT_HINTS
    )


def _resolve_effective_search_type(search_type: str, research_topic: str) -> tuple[str, str]:
    normalized_search_type = str(search_type or "both").strip().lower() or "both"
    if normalized_search_type not in {"internal", "external", "both"}:
        normalized_search_type = "both"

    if _research_topic_enforces_internal_only(research_topic):
        return "internal", "research_topic_internal_only"

    return normalized_search_type, "requested_search_type"


def _looks_explicit_external_artifact_query(text: str) -> bool:
    normalized = _normalize_scope_text(text)
    artifact_hints = _matching_hints(normalized, _EXTERNAL_ARTIFACT_HINTS)
    if not artifact_hints:
        return False

    explicit_artifact_markers = (
        "data availability",
        "download",
        "download link",
        "github",
        "link",
        "project page",
        "repo",
        "repository",
        "source code",
        "supplement link",
        "supplementary materials",
        "where can i find",
    )
    return any(_text_contains_hint(normalized, marker) for marker in explicit_artifact_markers)


def _looks_external_search_need(text: str) -> bool:
    normalized = _normalize_scope_text(text)
    return bool(
        _matching_hints(normalized, _EXTERNAL_SEARCH_HINTS)
        or _looks_explicit_external_artifact_query(normalized)
    )


def _looks_article_scoped_query(text: str) -> bool:
    normalized = _normalize_scope_text(text)
    if not normalized:
        return False

    if _looks_explicit_external_artifact_query(normalized):
        return False

    if _matching_hints(normalized, _ARTICLE_INTERNAL_HINTS):
        return True

    if re.search(
        r"\b(?:authors?|study|paper|article)\b.{0,40}\b"
        r"(?:use|used|report|reported|define|defined|describe|described|"
        r"test|tested|evaluate|evaluated|measure|measured|count|counted)\b",
        normalized,
    ):
        return True

    if re.search(r"\b(?:methods?|results?|supplement(?:ary)?|appendix)\b", normalized) and re.search(
        r"\b(?:notification|debounce|threshold|block duration|tp|fp|"
        r"true positive|false positive|split|cohort|dataset|framework|operational)\b",
        normalized,
    ):
        return True

    return False


def _classify_question_scope(question_text: str, research_topic: str = "") -> Dict[str, Any]:
    normalized_question = _normalize_scope_text(question_text)
    article_hints = _matching_hints(normalized_question, _ARTICLE_INTERNAL_HINTS)
    external_artifact_hints = _matching_hints(normalized_question, _EXTERNAL_ARTIFACT_HINTS)
    external_context_hints = _matching_hints(normalized_question, _EXTERNAL_SEARCH_HINTS)
    research_topic_internal_only = _research_topic_enforces_internal_only(research_topic)
    explicit_external_artifact = _looks_explicit_external_artifact_query(normalized_question)
    article_scoped = _looks_article_scoped_query(normalized_question)
    article_protocol_intent = bool(
        re.search(
            r"\b(?:authors?|study|paper|article)\b.{0,40}\b"
            r"(?:use|used|report|reported|define|defined|describe|described|"
            r"test|tested|evaluate|evaluated|measure|measured|count|counted)\b",
            normalized_question,
        )
    )

    internal_score = 0
    external_score = 0
    internal_reasons: List[str] = []
    external_reasons: List[str] = []

    if research_topic_internal_only:
        internal_score += 3
        internal_reasons.append("research_topic_internal_only")

    if article_protocol_intent:
        internal_score += 4
        internal_reasons.append("article_protocol_intent")
    elif article_scoped:
        internal_score += 2
        internal_reasons.append("article_scoped_query")

    if article_hints:
        article_hint_score = min(2, len(article_hints))
        internal_score += article_hint_score
        internal_reasons.append(f"article_hints={article_hint_score}")

    if explicit_external_artifact:
        external_score += 4
        external_reasons.append("explicit_external_artifact")
    elif external_artifact_hints:
        artifact_hint_score = min(2, len(external_artifact_hints))
        external_score += artifact_hint_score
        external_reasons.append(f"external_artifact_hints={artifact_hint_score}")

    if external_context_hints:
        external_context_score = min(2, len(external_context_hints))
        external_score += external_context_score
        external_reasons.append(f"external_context_hints={external_context_score}")

    # Questions about what the paper reported should generally stay local even if they
    # contain contextual terms like guidelines/comparison.
    if article_protocol_intent and external_context_hints:
        internal_score += 1
        internal_reasons.append("article_protocol_outweighs_context_term")

    score_gap = internal_score - external_score
    scope = "ambiguous"
    reason = "no_clear_scope_signal"
    if research_topic_internal_only and not explicit_external_artifact and external_score <= 1:
        scope = "article_internal"
        reason = "research_topic_internal_only"
    elif explicit_external_artifact and external_score >= internal_score + 1:
        scope = "external_artifact"
        reason = "external_artifact_discovery"
    elif internal_score >= external_score + 2 and internal_score >= 3:
        scope = "article_internal"
        if article_protocol_intent:
            reason = "article_protocol_intent"
        else:
            reason = "article_content_extraction"
    elif external_score >= internal_score + 2 and external_score >= 2:
        scope = "external_context"
        reason = "external_grounding_gap"
    elif internal_score >= 2 and external_score >= 2:
        scope = "mixed"
        reason = "mixed_article_and_external_signals"

    confidence = "low"
    if explicit_external_artifact or article_protocol_intent or research_topic_internal_only:
        confidence = "strong"
    elif abs(score_gap) >= 3:
        confidence = "strong"
    elif abs(score_gap) >= 1 or scope == "mixed":
        confidence = "medium"

    return {
        "scope": scope,
        "reason": reason,
        "confidence": confidence,
        "internal_score": internal_score,
        "external_score": external_score,
        "score_gap": score_gap,
        "internal_reasons": internal_reasons[:4],
        "external_reasons": external_reasons[:4],
        "article_hints": article_hints[:6],
        "external_artifact_hints": external_artifact_hints[:6],
        "external_context_hints": external_context_hints[:6],
        "research_topic_internal_only": research_topic_internal_only,
    }


def _clean_retrieval_query(text: str, max_chars: int = 220) -> tuple[str, str]:
    original = re.sub(r"\s+", " ", str(text or "")).strip().strip(' "\'')
    if not original:
        return "", ""

    cleaned = original
    cleaned = re.sub(r"^\s*From\s+doc_id\s*=\s*user_article[:,]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*From\s+user_article[:,]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*In\s+the\s+article[:,]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*From\s+the\s+article[:,]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(' "\'')

    sentence_match = re.match(rf"(.{{1,{max_chars}}}?[?.!])(?:\s|$)", cleaned)
    if sentence_match:
        cleaned = sentence_match.group(1).strip()
    elif len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")

    rewrite_reason = ""
    if cleaned != original:
        rewrite_reason = "removed_wrapper_or_clipped"

    return cleaned, rewrite_reason


def _query_quality_flags(query: str, tool_name: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", str(query or "")).strip()
    lowered = normalized.lower()
    flags: List[str] = []
    word_count = _query_word_count(normalized)
    question_scope = _classify_question_scope(normalized).get("scope", "ambiguous")

    if word_count < 4:
        flags.append("too_short")
    if tool_name == "retrieve_document_chunks" and word_count > 32:
        flags.append("too_long_for_local_rerank")
    if tool_name == "tavily_search" and word_count > 40:
        flags.append("too_long_for_web_query")
    if any(pattern.match(normalized) for pattern in _LOW_INFORMATION_PATTERNS):
        flags.append("too_vague")
    if re.search(r"\b(?:extract|locate|retrieve|provide|list|find)\b", lowered):
        flags.append("instruction_heavy")
    if normalized.count(";") >= 1 or normalized.count("?") > 1 or len(re.findall(r"\b(?:and|or)\b", lowered)) >= 3:
        flags.append("multipart")
    if tool_name == "retrieve_document_chunks" and question_scope in {"external_artifact", "external_context"}:
        flags.append("corpus_misaligned")
    if tool_name == "tavily_search" and question_scope == "article_internal":
        flags.append("article_internal_offload")

    return flags


def _is_low_information_query(query: str) -> bool:
    flags = _query_quality_flags(query, "")
    return "too_short" in flags or "too_vague" in flags


def _choose_tool_name(
    question_text: str,
    model_tool_name: str,
    search_type: str,
    retry_tool_name: str = "",
    research_topic: str = "",
) -> tuple[str, str]:
    scope_info = _classify_question_scope(question_text, research_topic)
    scope = scope_info.get("scope", "ambiguous")
    confidence = scope_info.get("confidence", "low")
    prefers_external_context = _research_topic_prefers_external_context(research_topic)
    internal_only_topic = _research_topic_enforces_internal_only(research_topic)
    normalized_question = re.sub(r"\s+", " ", str(question_text or "")).strip().lower()
    strong_external_override = (
        _looks_paper_specific_external_need(research_topic)
        or _has_broad_external_grounding_intent(research_topic)
    )
    article_reporting_request = bool(
        re.search(
            r"\b(?:does|did|what|which|where|how)\s+(?:the\s+)?(?:article|paper|study|supplement|appendix)\s+"
            r"(?:report|describe|show|state|list|include|provide)\b",
            normalized_question,
        )
    )

    if retry_tool_name in writer_tools_by_name:
        if internal_only_topic and retry_tool_name == "tavily_search":
            return "retrieve_document_chunks", "retry_tool_blocked_by_internal_only_topic"
        return retry_tool_name, "retry_tool_override"

    if article_reporting_request and search_type in {"internal", "external", "both"} and not strong_external_override:
        return "retrieve_document_chunks", "article_reporting_request"

    if scope == "article_internal" and confidence in {"strong", "medium"}:
        if search_type == "external" and prefers_external_context and strong_external_override:
            return "tavily_search", "research_topic_external_context_priority"
        if search_type == "external" and strong_external_override and not internal_only_topic:
            return "tavily_search", "explicit_external_need_override"
        return "retrieve_document_chunks", scope_info.get("reason", "article_content_extraction")

    if search_type == "internal":
        return "retrieve_document_chunks", "search_type_internal"
    if search_type == "external":
        if internal_only_topic:
            return "retrieve_document_chunks", "research_topic_internal_only"
        if prefers_external_context and scope != "external_artifact":
            return "tavily_search", "research_topic_external_context_priority"
        return "tavily_search", "search_type_external"

    if prefers_external_context and search_type == "both" and scope in {"ambiguous", "mixed", "external_context"}:
        return "tavily_search", "research_topic_external_context_priority"
    if scope in {"external_artifact", "external_context"} and confidence == "strong":
        return "tavily_search", "external_artifact_or_context_gap"
    if scope == "mixed":
        if model_tool_name in writer_tools_by_name:
            return model_tool_name, "model_selected_for_mixed_scope"
        return "retrieve_document_chunks", "mixed_scope_article_first"
    if model_tool_name in writer_tools_by_name:
        return model_tool_name, "model_selected"
    if scope in {"article_internal", "external_artifact", "external_context"}:
        return (
            ("retrieve_document_chunks", "article_evidence_gap")
            if scope == "article_internal"
            else ("tavily_search", "external_artifact_or_context_gap")
        )
    return "retrieve_document_chunks", "default_article_first"


def _extract_article_title(article_summary: str) -> str:
    summary_text = str(article_summary or "")
    quoted_match = re.search(r"“([^”]+)”", summary_text)
    if quoted_match:
        return re.sub(r"[*_`]+", "", quoted_match.group(1)).strip().strip('"')

    title_match = re.search(r"\*\*Title/Topic\*\*\s+(.+?)(?:\n|$)", summary_text, re.DOTALL)
    if title_match:
        title_text = re.sub(r"[*_`]+", "", title_match.group(1)).strip()
        if title_text:
            return title_text.strip('"')

    return ""


def _extract_article_doi(article_summary: str) -> str:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", str(article_summary or ""))
    return match.group(0).strip().rstrip(").,;") if match else ""


def _extract_external_artifact_terms(question_text: str) -> List[str]:
    normalized = _normalize_scope_text(question_text)
    ordered_terms = (
        ("data availability", ("data availability",)),
        ("supplement", ("supplement", "supplementary", "appendix")),
        ("repository", ("repository", "repo", "github", "gitlab", "bitbucket")),
        ("code", ("code", "source code", "pseudocode", "notebook")),
        ("csv", ("csv", "table", "tables", "dataset")),
        ("time series", ("time series", "per-hour", "per hour", "timestamp")),
        ("notification logic", ("notification", "collapse", "aggregation", "debounce", "block length", "window")),
        ("fold definitions", ("cross-validation", "cross validation", "fold", "split", "labeling")),
    )
    extracted: List[str] = []
    for label, hints in ordered_terms:
        if any(_text_contains_hint(normalized, hint) for hint in hints):
            extracted.append(label)
    return extracted[:4]


def _extract_metric_terms(question_text: str) -> List[str]:
    ordered_terms = (
        "AUROC",
        "AUPRC",
        "MCS",
        "C-Precision",
        "C-Recall",
        "median lead time",
        "lead time",
        "sample size",
        "denominator",
        "threshold",
    )
    extracted: List[str] = []
    for term in ordered_terms:
        if _text_contains_hint(question_text, term):
            extracted.append(term)
    return extracted[:4]


def _build_title_anchored_external_query(question_text: str, article_summary: str) -> str:
    article_title = _extract_article_title(article_summary)
    article_doi = _extract_article_doi(article_summary)
    terms = _extract_external_artifact_terms(question_text)
    if not article_title:
        return ""

    base = f"\"{article_title}\""
    if article_doi:
        base = f"{base} OR \"{article_doi}\""

    artifact_scope = "\"supplement\" OR appendix OR repository OR GitHub OR code OR \"data availability\""
    if terms:
        keyword_clause = " OR ".join(
            f"\"{term}\"" if " " in term else term
            for term in terms
        )
        return f"({base}) ({artifact_scope}) ({keyword_clause})"
    return f"({base}) ({artifact_scope})"


def _has_broad_external_grounding_intent(text: str) -> bool:
    normalized = _normalize_scope_text(text)
    if not normalized:
        return False
    broad_hints = (
        "best practice",
        "best practices",
        "comparator",
        "compare with",
        "current",
        "deployment",
        "field",
        "governance",
        "guideline",
        "guidelines",
        "implementation",
        "latest",
        "news",
        "operational lesson",
        "operational lessons",
        "practice trend",
        "practice trends",
        "prior work",
        "recent",
        "related work",
        "state of the field",
        "trend",
        "trends",
    )
    return any(_text_contains_hint(normalized, hint) for hint in broad_hints)


def _looks_paper_specific_external_need(text: str) -> bool:
    normalized = _normalize_scope_text(text)
    if not normalized:
        return False
    if _looks_explicit_external_artifact_query(normalized):
        return True
    paper_specific_hints = (
        "author request",
        "author response",
        "contact authors",
        "data availability",
        "download",
        "downloadable supplement",
        "downloadable supplementary",
        "github",
        "gitlab",
        "public repository",
        "publicly available",
        "publicly accessible",
        "repository",
        "repo",
        "doi",
        "source code",
    )
    return any(_text_contains_hint(normalized, hint) for hint in paper_specific_hints)


def _strip_article_reference_for_broad_external_query(text: str) -> str:
    cleaned = str(text or "")
    replacements = (
        (r"\bthis article'?s\b", ""),
        (r"\bthis paper'?s\b", ""),
        (r"\bthis study'?s\b", ""),
        (r"\bfor this article\b", ""),
        (r"\bfor this paper\b", ""),
        (r"\bfor this study\b", ""),
        (r"\bin this article\b", ""),
        (r"\bin this paper\b", ""),
        (r"\brelative to this article\b", ""),
        (r"\brelative to this paper\b", ""),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:")
    return cleaned


def _extract_quoted_terms(text: str) -> List[str]:
    terms = re.findall(r"'([^']+)'|\"([^\"]+)\"", str(text or ""))
    flattened = [first or second for first, second in terms]
    unique_terms: List[str] = []
    for term in flattened:
        normalized_term = re.sub(r"\s+", " ", term).strip()
        if normalized_term and normalized_term not in unique_terms:
            unique_terms.append(normalized_term)
    return unique_terms


def _has_methodology_intent(text: str) -> bool:
    normalized = _normalize_scope_text(text)
    if not normalized:
        return False
    intent_hints = (
        "aggregate",
        "aggregation",
        "computed",
        "counted",
        "define",
        "defined",
        "denominator",
        "fold",
        "groupby",
        "leakage",
        "logic",
        "patient_id",
        "split",
        "stay_id",
        "stratif",
        "validation",
        "window",
    )
    intent_verbs = (
        "how",
        "whether",
        "did",
        "does",
        "was",
        "were",
        "specify",
        "specified",
        "using",
        "used",
    )
    return any(hint in normalized for hint in intent_hints) and any(
        re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalized)
        for term in intent_verbs
    )


def _preferred_methodology_cluster(
    cluster_hits: List[tuple[int, str, str]],
    query: str,
) -> tuple[int, str, str] | None:
    if not _has_methodology_intent(query):
        return None
    preferred_order = (
        "splits_leakage",
        "notification_logic",
        "definitions",
        "denominators",
    )
    hit_lookup = {name: hit for hit in cluster_hits for _, name, _ in [hit]}
    for cluster_name in preferred_order:
        hit = hit_lookup.get(cluster_name)
        if hit and hit[0] >= 1:
            return hit
    return None


def _shape_local_retrieval_query(
    question_text: str,
    article_summary: str = "",
) -> tuple[str, str]:
    query, base_reason = _clean_retrieval_query(question_text)
    if not query:
        return "", base_reason

    rewrite_reasons: List[str] = [base_reason] if base_reason else []

    def _cluster_hits(text: str) -> list[tuple[int, str, str]]:
        normalized = _normalize_scope_text(text)
        hits: list[tuple[int, str, str]] = []
        for cluster_name, keywords, canonical_question in _LOCAL_FACT_CLUSTERS:
            score = sum(1 for keyword in keywords if _text_contains_hint(normalized, keyword))
            if score > 0:
                hits.append((score, cluster_name, canonical_question))
        hits.sort(reverse=True)
        return hits

    def _extract_context_suffix(text: str) -> str:
        lowered = text.lower()
        if lowered.startswith("for "):
            suffix = text.strip()
            suffix = re.sub(r"^[Ff]or\s+", "for ", suffix)
            suffix = re.sub(
                r",?\s*what\b.*$",
                "",
                suffix,
                flags=re.IGNORECASE,
            ).rstrip(" ,;:")
            if 0 < len(suffix) <= 120:
                return suffix
        for marker in (" for ", " in ", " on "):
            idx = lowered.find(marker)
            if idx >= 0:
                suffix = text[idx:].strip()
                suffix = re.sub(
                    r"\s+(?:and|with)\s+(?:where|which|cite|including)\b.*$",
                    "",
                    suffix,
                    flags=re.IGNORECASE,
                ).rstrip(" ,;:")
                if 0 < len(suffix) <= 120:
                    return suffix
        return ""

    def _specialized_cluster_question(cluster_name: str, text: str, context_suffix: str) -> str:
        if cluster_name != "performance_metrics":
            return ""
        metric_terms = _extract_metric_terms(text)
        if not metric_terms:
            return ""
        metrics = ", ".join(metric_terms[:3])
        if len(metric_terms) > 3:
            metrics += ", or related performance metrics"
        else:
            metrics += ", or related performance metrics"
        if context_suffix:
            return f"What {metrics} does the article report {context_suffix.lstrip()}?"
        return f"What {metrics} does the article report?"

    trimmed = re.sub(
        r"^\s*(?:methods?|supplement(?:ary)?(?: materials?)?|appendix)"
        r"(?:\s+or\s+(?:methods?|supplement(?:ary)?(?: materials?)?|appendix))?\s*:\s*",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()
    if trimmed != query:
        query = trimmed
        rewrite_reasons.append("removed_section_prefix")

    normalized_query = _normalize_scope_text(query or question_text)
    thesis_intent = any(
        phrase in normalized_query
        for phrase in (
            "main contribution",
            "primary contribution",
            "real thesis",
            "main thesis",
            "core thesis",
            "main point",
            "real point",
            "paper contributes",
            "article contributes",
            "paper proposes",
            "article proposes",
            "what do the authors say the paper contributes",
        )
    )
    framework_intent = any(
        phrase in normalized_query
        for phrase in (
            "framework",
            "evaluation approach",
            "evaluation method",
            "method",
            "approach",
            "case study",
            "example",
        )
    )
    profile = infer_document_profile(article_summary)
    source_kind = str(profile.get("source_kind", "") or "")
    has_subgroup_intent = any(
        phrase in normalized_query
        for phrase in (
            "subgroup",
            "cohort",
            "worse performance",
            "failure pattern",
            "onset-timing",
        )
    )
    has_split_intent = any(
        phrase in normalized_query
        for phrase in (
            "cross-validation",
            "data split",
            "patient or stay level",
            "leakage",
            "folds",
        )
    )

    if thesis_intent:
        if source_kind == "regulatory_guidance":
            query = "What change, criteria, or scope boundary does the guidance define, and which examples make that boundary clear?"
        elif source_kind in {"commentary_analysis", "commentary_product_update"}:
            query = "What is the article's main claim, and which named examples or comparisons support it?"
        elif source_kind == "research_clinical_or_comparative":
            query = "What intervention, comparator, and outcomes define the main claim, and what results support it?"
        elif source_kind == "research_real_world_evaluation":
            query = "What intervention or service is being evaluated, in what setting, and what real-world outcomes support the main claim?"
        else:
            query = (
                "What framework, method, or evaluation approach does the article propose?"
                if framework_intent
                else "What do the abstract and introduction say the paper proposes?"
            )
        rewrite_reasons.append("thesis_contribution_rewrite")
    elif source_kind == "regulatory_guidance" and (framework_intent or has_subgroup_intent or has_split_intent):
        query = "What exact criteria, exclusions, examples, or implementation boundaries does the guidance give?"
        rewrite_reasons.append("document_profile_regulatory_reframe")
    elif source_kind in {"commentary_analysis", "commentary_product_update"} and (framework_intent or has_subgroup_intent or has_split_intent):
        query = "What named examples, product contrasts, or recommendations does the article use to support its main argument?"
        rewrite_reasons.append("document_profile_commentary_reframe")
    elif source_kind == "research_real_world_evaluation" and (framework_intent or has_split_intent):
        query = "What study design, setting, measured outcomes, and reported limitations define this real-world evaluation?"
        rewrite_reasons.append("document_profile_real_world_reframe")

    keyword_terms = _extract_quoted_terms(query or question_text)
    looks_like_keyword_bag = bool(
        keyword_terms
        and "?" not in query
        and not re.search(r"\b(?:what|which|how|where|who|when|did|does|is|are|was|were)\b", query.lower())
    )
    if looks_like_keyword_bag:
        joined_terms = ", ".join(keyword_terms[:6])
        if re.search(r"\b(?:methods?|supplement(?:ary)?|appendix)\b", question_text, re.IGNORECASE):
            query = f"What do the article methods or supplementary materials say about {joined_terms}?"
        else:
            query = f"What does the article report about {joined_terms}?"
        rewrite_reasons.append("keyword_bag_to_standalone_question")

    for prefix, replacement in (
        ("report the ", "What are the "),
        ("extract the ", "What are the "),
        ("list the ", "What are the "),
        ("locate the ", "What are the "),
        ("find the ", "What are the "),
    ):
        if query.lower().startswith(prefix):
            query = replacement + query[len(prefix):]
            rewrite_reasons.append("local_instruction_to_question")
            break

    query = re.sub(
        r"\s+(?:and|with)\s+(?:where|which table|which figure|cite|including page|include page|with page|with section)\b.*$",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()

    cluster_hits = _cluster_hits(query or question_text)
    connector_count = len(re.findall(r"\b(?:and|or)\b", query.lower()))
    is_dense_multipart_query = (
        _query_word_count(query) > 20
        or query.count(",") >= 4
        or connector_count >= 2
    )
    if (
        is_dense_multipart_query
        and len(cluster_hits) >= 2
        and cluster_hits[0][0] >= 2
        and (cluster_hits[1][0] >= 2 or connector_count >= 3 or query.count(",") >= 4)
    ):
        preferred_hit = _preferred_methodology_cluster(cluster_hits, query or question_text)
        selected_hit = preferred_hit or cluster_hits[0]
        _, cluster_name, canonical_question = selected_hit
        context_suffix = _extract_context_suffix(query)
        specialized_question = _specialized_cluster_question(cluster_name, query or question_text, context_suffix)
        if specialized_question:
            query = specialized_question
        elif context_suffix:
            query = f"{canonical_question[:-1]} {context_suffix.lstrip()}?"
        else:
            query = canonical_question
        rewrite_reasons.append(
            "fact_cluster_decomposition"
            if not preferred_hit
            else f"fact_cluster_decomposition:{cluster_name}"
        )

    query = query.strip()
    if query and query[-1] not in {"?", "!"}:
        query += "?"

    return query, ";".join(reason for reason in rewrite_reasons if reason)


def _shape_external_search_query(question_text: str, article_summary: str) -> tuple[str, str]:
    query, base_reason = _clean_retrieval_query(question_text, max_chars=180)
    if not query:
        return "", base_reason

    rewrite_reasons: List[str] = [base_reason] if base_reason else []
    article_title = _extract_article_title(article_summary)
    article_doi = _extract_article_doi(article_summary)
    lowered = query.lower()
    broad_external_grounding = _has_broad_external_grounding_intent(query)
    paper_specific_need = _looks_paper_specific_external_need(query)
    profile = infer_document_profile(article_summary)
    source_kind = str(profile.get("source_kind", "") or "")
    anchor_terms = [str(item).strip() for item in (profile.get("anchor_terms", []) or []) if str(item).strip()]
    preferred_domains = [str(item).strip() for item in (profile.get("preferred_domains", []) or []) if str(item).strip()]
    explicit_external_context_prompt = (
        lowered.startswith("what public external context")
        or "materially change how a pharmacy leader interprets this source" in lowered
    )

    if (
        source_kind == "regulatory_guidance"
        and (
            broad_external_grounding
            or explicit_external_context_prompt
            or "external context" in lowered
            or "public external context" in lowered
            or "interpret" in lowered
        )
    ):
        if article_title and preferred_domains:
            query = f"site:{preferred_domains[0]} \"{article_title}\" OR FAQ OR town hall OR guidance"
            rewrite_reasons.append("issuer_first_official_grounding")
        elif preferred_domains:
            query = f"site:{preferred_domains[0]} guidance FAQ town hall official interpretation"
            rewrite_reasons.append("issuer_first_official_grounding")
    elif (broad_external_grounding or explicit_external_context_prompt) and source_kind in {"commentary_analysis", "commentary_product_update"} and anchor_terms:
        query = " ".join(f"\"{term}\"" for term in anchor_terms[:3]) + " release notes OR product page OR comparison OR review"
        rewrite_reasons.append("entity_anchored_external_grounding")
    elif (broad_external_grounding or explicit_external_context_prompt) and source_kind in {
        "research_real_world_evaluation",
        "research_clinical_or_comparative",
        "research_general",
    } and anchor_terms:
        query = " ".join(f"\"{term}\"" for term in anchor_terms[:2]) + " real-world study OR comparator OR scoping review"
        rewrite_reasons.append("study_anchored_external_grounding")
    elif broad_external_grounding:
        stripped = _strip_article_reference_for_broad_external_query(query)
        if stripped and stripped != query:
            query = stripped
            rewrite_reasons.append("broad_external_grounding_query")
    elif article_title and any(
        _text_contains_hint(lowered, token)
        for token in ("repository", "repo", "github", "source code", "code")
    ):
        query = f"\"{article_title}\" repository OR GitHub OR source code"
        rewrite_reasons.append("title_anchored_artifact_search")
    elif article_title and any(
        _text_contains_hint(lowered, token)
        for token in ("supplement", "supplementary materials", "supplement link", "appendix")
    ):
        query = f"\"{article_title}\" supplementary materials OR supplement OR appendix"
        rewrite_reasons.append("title_anchored_artifact_search")
    elif article_title and paper_specific_need:
        anchored = _build_title_anchored_external_query(question_text, article_summary)
        if anchored:
            query = anchored
            rewrite_reasons.append("title_anchored_external_discovery")
    elif article_doi and any(
        _text_contains_hint(lowered, token)
        for token in ("repository", "supplement", "code", "data", "appendix")
    ):
        query = f"\"{article_doi}\" repository OR supplement OR appendix OR code"
        rewrite_reasons.append("doi_anchored_artifact_search")

    return query, ";".join(reason for reason in rewrite_reasons if reason)


def _shape_query_for_tool(
    question_text: str,
    tool_name: str,
    article_summary: str = "",
) -> tuple[str, str, str]:
    original = re.sub(r"\s+", " ", str(question_text or "")).strip().strip(' "\'')
    if tool_name == "retrieve_document_chunks":
        query, shape_reason = _shape_local_retrieval_query(question_text, article_summary=article_summary)
    elif tool_name == "tavily_search":
        query, shape_reason = _shape_external_search_query(question_text, article_summary)
    else:
        query, shape_reason = _clean_retrieval_query(question_text)

    cleaned_query = query or question_text
    rewrite_reason = "removed_wrapper_or_clipped" if cleaned_query != original else ""
    return cleaned_query, rewrite_reason, shape_reason


def _parse_tool_result_metadata(tool_name: str, result: str) -> Dict[str, Any]:
    content = str(result or "")
    metadata: Dict[str, Any] = {
        "tool_name": tool_name,
        "raw_result_len": len(content),
        "status": "ok",
    }

    if tool_name == "retrieve_document_chunks":
        chunk_matches = list(
            re.finditer(
                r"\[chunk\s+\d+\]\s+(?P<locator>.*?)\s+\|\s+score=(?P<score>\d+\.\d+)",
                content,
            )
        )
        scores = [float(match.group("score")) for match in chunk_matches]
        locators = [match.group("locator").strip() for match in chunk_matches]
        metadata.update(
            {
                "matched_chunks": len(chunk_matches),
                "chunk_locators": locators,
                "best_score": max(scores) if scores else 0.0,
            }
        )
        if "No chunks met confidence threshold" in content or "No relevant chunks found" in content:
            metadata["status"] = "no_match"
            best_match = re.search(r"Best:\s*(\d+\.\d+)", content)
            if best_match:
                metadata["best_score"] = float(best_match.group(1))
        elif "Error during reranking" in content or "No document found" in content:
            metadata["status"] = "error"
        return metadata

    if tool_name == "tavily_search":
        source_count = len(re.findall(r"--- SOURCE \d+:", content))
        metadata["source_count"] = source_count
        if source_count == 0:
            metadata["status"] = "empty"
        return metadata

    return metadata


_MISSING_ANSWER_PHRASES = (
    "not found in retrieved context",
    "not specified in text",
    "unable to find information",
    "document may not contain this specific information",
    "document does not appear to contain information semantically related",
    "no information was retrieved",
    "not reported in the retrieved excerpts",
)


def _answer_token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9/_-]*", str(text or "").lower())
        if len(token) >= 4
    }


def _has_substantive_answer_content(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(answer or "")).strip().lower()
    if not normalized:
        return False

    cleaned = normalized
    for phrase in _MISSING_ANSWER_PHRASES:
        cleaned = cleaned.replace(phrase, " ")
    cleaned = re.sub(r"sources used.*$", "", cleaned, flags=re.IGNORECASE)
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9/_-]*", cleaned)
        if len(token) >= 3
    ]
    # Heuristic: at least a short factual clause remains after stripping missingness boilerplate.
    return len(tokens) >= 12


def _answers_materially_diverge(first_answer: str, second_answer: str) -> bool:
    first_tokens = _answer_token_set(first_answer)
    second_tokens = _answer_token_set(second_answer)
    if not first_tokens or not second_tokens:
        return False
    overlap = len(first_tokens & second_tokens) / max(min(len(first_tokens), len(second_tokens)), 1)
    return overlap < 0.35


def _infer_answer_status(
    answer: str,
    answer_quality: str = "",
    retrieval_metadata: Optional[Dict[str, Any]] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, str]:
    normalized_answer = re.sub(r"\s+", " ", str(answer or "")).strip().lower()
    retrieval_metadata = retrieval_metadata or {}
    evidence = evidence or []

    has_missing_phrase = any(phrase in normalized_answer for phrase in _MISSING_ANSWER_PHRASES)
    has_substantive_content = _has_substantive_answer_content(answer)
    has_good_local_retrieval = (
        retrieval_metadata.get("tool_name") == "retrieve_document_chunks"
        and retrieval_metadata.get("status") == "ok"
        and float(retrieval_metadata.get("best_score") or 0.0) >= 0.8
        and int(retrieval_metadata.get("matched_chunks") or 0) >= 2
    )

    if has_missing_phrase:
        if (evidence or has_good_local_retrieval) and has_substantive_content:
            return "needs_review", "partial_missing_language_with_supported_content"
        if evidence or has_good_local_retrieval:
            return "not_in_source", "answer_indicates_not_reported_in_relevant_context"
        if retrieval_metadata.get("tool_name") == "retrieve_document_chunks":
            return "premise_mismatch", "answer_indicates_missing_data_after_low_value_local_retrieval"
        return "missing", "answer_indicates_missing_data"
    if retrieval_metadata.get("status") in {"no_match", "empty", "error"} and not evidence:
        if retrieval_metadata.get("tool_name") == "retrieve_document_chunks":
            return "premise_mismatch", f"retrieval_status={retrieval_metadata.get('status')}"
        return "missing", f"retrieval_status={retrieval_metadata.get('status')}"
    quality = str(answer_quality or "").strip().lower()
    if quality in {"insufficient", "off_target"}:
        if (evidence or has_good_local_retrieval) and has_substantive_content:
            return "needs_review", f"answer_quality={quality}_but_evidence_present"
        if not evidence:
            if retrieval_metadata.get("tool_name") == "retrieve_document_chunks":
                return "premise_mismatch", f"answer_quality={quality}"
            return "missing", f"answer_quality={quality}"
    return "supported", ""


def _compare_retry_outcome(
    original_answer: str,
    retry_answer: str,
    original_metadata: Optional[Dict[str, Any]],
    retry_metadata: Optional[Dict[str, Any]],
    original_evidence: List[Dict[str, Any]],
    retry_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    status_rank = {"error": 0, "empty": 0, "no_match": 1, "ok": 3}
    original_metadata = original_metadata or {}
    retry_metadata = retry_metadata or {}

    original_status = str(original_metadata.get("status", "ok") or "ok")
    retry_status = str(retry_metadata.get("status", "ok") or "ok")
    original_rank = status_rank.get(original_status, 2)
    retry_rank = status_rank.get(retry_status, 2)

    original_best = float(original_metadata.get("best_score") or 0.0)
    retry_best = float(retry_metadata.get("best_score") or 0.0)
    original_locators = {item.get("locator", "") for item in original_evidence or [] if item.get("locator")}
    retry_locators = {item.get("locator", "") for item in retry_evidence or [] if item.get("locator")}
    novel_retry_locators = retry_locators - original_locators

    improved = False
    if retry_rank > original_rank:
        improved = True
    elif retry_rank == original_rank and retry_best >= original_best + 0.05:
        improved = True
    elif retry_status == "ok" and novel_retry_locators:
        improved = True
    elif retry_metadata.get("fallback_tool_name") == "tavily_search" and int(retry_metadata.get("fallback_source_count") or 0) > 0:
        improved = True

    regressed = False
    if original_rank > retry_rank:
        regressed = True
    elif original_best >= 0.85 and retry_best + 0.1 < original_best and not novel_retry_locators:
        regressed = True

    complementary = bool(novel_retry_locators) and retry_rank >= original_rank
    materially_divergent = _answers_materially_diverge(original_answer, retry_answer)
    prefer_retry_only = retry_rank > original_rank and original_rank <= 1 and retry_status == "ok"

    if improved:
        decision_reason = "retry_added_or_improved_evidence"
    elif regressed:
        decision_reason = "retry_regressed"
    elif complementary:
        decision_reason = "retry_added_complementary_evidence"
    else:
        decision_reason = "retry_did_not_improve_evidence"

    return {
        "improved": improved,
        "regressed": regressed,
        "complementary": complementary,
        "materially_divergent": materially_divergent,
        "prefer_retry_only": prefer_retry_only,
        "accept_retry": improved or complementary,
        "decision_reason": decision_reason,
        "original_status": original_status,
        "retry_status": retry_status,
        "original_best_score": round(original_best, 3),
        "retry_best_score": round(retry_best, 3),
        "novel_retry_locators": len(novel_retry_locators),
    }


def _coerce_retry_decision(
    answer_quality: str,
    needs_retry: bool,
    retry_query: str,
    last_question: str,
    last_retrieval_metadata: Optional[Dict[str, Any]] = None,
    suggested_tool_name: str = "",
    article_summary: str = "",
    retrieval_events: Optional[List[Dict[str, Any]]] = None,
    active_gap_id: str = "",
) -> tuple[bool, str, str]:
    """Apply deterministic retry guardrails to structured reflection output."""
    if not needs_retry:
        return False, "", "model_declined_retry"

    normalized_quality = str(answer_quality or "").strip().lower()
    if normalized_quality not in {"insufficient", "off_target"}:
        return False, "", f"answer_quality={normalized_quality or 'unknown'}"

    normalized_retry_query = _normalize_retry_query(retry_query)
    if not normalized_retry_query:
        return False, "", "empty_retry_query"

    if _is_low_information_query(normalized_retry_query):
        return False, "", "low_information_retry_query"

    if _query_fingerprint(normalized_retry_query) == _query_fingerprint(last_question):
        return False, "", "duplicate_retry_query"

    retrieval_metadata = last_retrieval_metadata or {}
    if (
        retrieval_metadata.get("tool_name") == "retrieve_document_chunks"
        and retrieval_metadata.get("status") == "ok"
        and float(retrieval_metadata.get("best_score") or 0.0) >= 0.85
        and int(retrieval_metadata.get("matched_chunks") or 0) >= 2
    ):
        return False, "", "high_confidence_initial_retrieval"

    retry_tool_name = str(suggested_tool_name or retrieval_metadata.get("tool_name") or "").strip()
    if retry_tool_name in writer_tools_by_name:
        shaped_retry_query, _, _ = _shape_query_for_tool(
            question_text=normalized_retry_query,
            tool_name=retry_tool_name,
            article_summary=article_summary,
        )
        if _query_fingerprint(shaped_retry_query) == _query_fingerprint(str(retrieval_metadata.get("query", "") or "")):
            return False, "", "duplicate_shaped_retry_query"
        duplicate_event = _find_duplicate_query_event(
            retrieval_events=retrieval_events,
            retrieval_query=shaped_retry_query,
            tool_name=retry_tool_name,
            active_gap_id=active_gap_id,
        )
        if duplicate_event:
            return False, "", "duplicate_shaped_retry_query"

    return True, normalized_retry_query, "accepted"


def _extract_question_from_message(msg: BaseMessage) -> str:
    if isinstance(msg, AIMessage):
        question_text = _normalize_question_text(str(msg.content) if msg.content else "")
        if question_text:
            return question_text

        for tool_call in getattr(msg, "tool_calls", []) or []:
            query = _normalize_question_text(str((tool_call.get("args") or {}).get("query", "")))
            if query:
                return query

    if isinstance(msg, HumanMessage):
        content = str(msg.content or "")
        if "]: " in content:
            return _normalize_question_text(content.split("]: ", 1)[-1])
        return _normalize_question_text(content)

    return ""


def _parse_document_chunk_evidence(content: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not content:
        return items

    for chunk in content.split("\n\n---\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("[chunk "):
            continue

        lines = chunk.splitlines()
        if not lines:
            continue

        header = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        locator_match = re.search(r"\]\s+(.*?)\s+\|\s+score=", header)
        locator = locator_match.group(1).strip() if locator_match else header
        page_match = re.search(r"page=(\d+)", header)
        page_suffix = f" page {page_match.group(1)}" if page_match else ""

        items.append(
            {
                "source_type": "article_chunk",
                "title": f"Provided article (user-supplied text){page_suffix}",
                "locator": locator,
                "snippet": _truncate_text(body, 320),
            }
        )

    if items:
        return items

    locator_match = re.search(r"Closest match locator:\s*(.+)", content)
    text_match = re.search(r'Closest match text:\s*"(.+?)"', content, flags=re.DOTALL)
    if locator_match:
        items.append(
            {
                "source_type": "article_chunk",
                "title": "Provided article (user-supplied text)",
                "locator": locator_match.group(1).strip(),
                "snippet": _truncate_text(text_match.group(1) if text_match else content, 320),
            }
        )

    return items


def _parse_tavily_evidence(content: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not content:
        return items

    pattern = re.compile(
        r"--- SOURCE \d+: (?P<title>.*?) ---\s*URL:\s*(?P<url>\S+)\s*(?P<body>.*?)(?=\n-{20,}|\Z)",
        flags=re.DOTALL,
    )

    for match in pattern.finditer(content):
        title = _truncate_text(match.group("title"), 160)
        url = match.group("url").strip().rstrip(").,;")
        body = match.group("body").strip()
        summary_match = re.search(r"<summary>\s*(.*?)\s*</summary>", body, flags=re.DOTALL)
        snippet = summary_match.group(1).strip() if summary_match else body

        items.append(
            {
                "source_type": "web",
                "title": title or url,
                "locator": url,
                "snippet": _truncate_text(snippet, 320),
            }
        )

    return items


def extract_evidence_from_tool_messages(messages: List[BaseMessage], max_items: int = 4) -> List[Dict[str, str]]:
    """Extract a compact structured evidence list from current tool results."""
    evidence: List[Dict[str, str]] = []
    seen_keys: set[str] = set()

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        tool_name = getattr(msg, "name", "")
        content = str(getattr(msg, "content", "") or "")
        if not content.strip():
            continue

        if tool_name == "retrieve_document_chunks":
            candidates = _parse_document_chunk_evidence(content)
        elif tool_name == "tavily_search":
            candidates = _parse_tavily_evidence(content)
        else:
            candidates = []

        for item in candidates:
            key = f"{item.get('source_type','')}|{item.get('locator','')}|{item.get('snippet','')}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            evidence.append(item)
            if len(evidence) >= max_items:
                return evidence

    return evidence


def _build_ledger_entry(
    state: ResearcherState,
    question: str,
    answer: str,
    evidence: List[Dict[str, Any]],
    answer_origin: str,
    answer_status: str = "supported",
    missing_reason: str = "",
    review_note: str = "",
) -> Dict[str, Any]:
    current_perspective = state.get("current_perspective", "")
    perspective_research_plans = state.get("perspective_research_plans", {})
    canonical_answer = canonicalize_answer_summary(
        answer,
        answer_status=answer_status,
    )
    entry = {
        "qa_id": state.get("current_qa_id", ""),
        "gap_id": state.get("active_gap_id", ""),
        "perspective": current_perspective,
        "research_topic": state.get("research_topic", ""),
        "research_plan": perspective_research_plans.get(current_perspective, ""),
        "search_type": state.get("search_type", "both"),
        "question": question,
        "answer": canonical_answer,
        "answer_origin": answer_origin,
        "answer_status": answer_status,
        "evidence": evidence[:4],
    }
    if missing_reason:
        entry["missing_reason"] = missing_reason
    if review_note:
        entry["review_note"] = review_note
    return entry


def _merge_evidence_items(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return merge_canonical_evidence_items(*groups)[:4]


def _select_primary_tool_call(tool_calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for tool_call in tool_calls or []:
        if tool_call.get("name") in writer_tools_by_name:
            return tool_call
    return None


def _sanitize_writer_response(
    response: AIMessage,
    question_text: str,
) -> AIMessage:
    """Legacy shim retained for compatibility."""
    sanitized_response, _ = _build_retrieval_plan(
        response=response,
        question_text=question_text,
        search_type="both",
        research_topic="",
        article_summary="",
    )
    return sanitized_response


def _build_retrieval_plan(
    response: AIMessage,
    question_text: str,
    search_type: str,
    retry_tool_name: str = "",
    is_retry_attempt: bool = False,
    research_topic: str = "",
    article_summary: str = "",
    retrieval_events: Optional[List[Dict[str, Any]]] = None,
    active_gap_id: str = "",
) -> tuple[AIMessage, Dict[str, Any]]:
    """Enforce the one-question / one-tool contract and emit structured plan metadata."""
    tool_calls = list(getattr(response, "tool_calls", []) or [])
    primary_tool_call = _select_primary_tool_call(tool_calls)
    model_tool_name = primary_tool_call.get("name", "") if primary_tool_call else ""
    effective_search_type, search_type_reason = _resolve_effective_search_type(
        search_type,
        research_topic,
    )
    scope_info = _classify_question_scope(question_text, research_topic)

    chosen_tool_name, tool_reason = _choose_tool_name(
        question_text=question_text,
        model_tool_name=model_tool_name,
        search_type=effective_search_type,
        retry_tool_name=retry_tool_name,
        research_topic=research_topic,
    )
    retrieval_query, rewrite_reason, query_shape_reason = _shape_query_for_tool(
        question_text=question_text,
        tool_name=chosen_tool_name,
        article_summary=article_summary,
    )
    if not retrieval_query:
        retrieval_query = question_text

    query_flags = _query_quality_flags(retrieval_query, chosen_tool_name)
    if (
        chosen_tool_name == "retrieve_document_chunks"
        and "corpus_misaligned" in query_flags
        and effective_search_type in {"both", "external"}
        and scope_info.get("scope") in {"external_artifact", "external_context"}
        and scope_info.get("confidence") == "strong"
    ):
        chosen_tool_name = "tavily_search"
        tool_reason = "external_override_for_corpus_mismatch"
        retrieval_query, rewrite_reason, query_shape_reason = _shape_query_for_tool(
            question_text=question_text,
            tool_name=chosen_tool_name,
            article_summary=article_summary,
        )
        query_flags = _query_quality_flags(retrieval_query, chosen_tool_name)

    duplicate_event = _find_duplicate_query_event(
        retrieval_events=retrieval_events,
        retrieval_query=retrieval_query,
        tool_name=chosen_tool_name,
        active_gap_id=active_gap_id,
    )

    sanitized_tool_calls: List[Dict[str, Any]] = []
    if primary_tool_call or not tool_calls:
        args = {"query": retrieval_query}
        if chosen_tool_name == "retrieve_document_chunks":
            args["doc_id"] = DEFAULT_DOC_ID
        raw_tool_call_id = str((primary_tool_call or {}).get("id") or "").strip()
        sanitized_tool_call_id = raw_tool_call_id or f"{chosen_tool_name}_call"
        if model_tool_name and model_tool_name != chosen_tool_name:
            sanitized_tool_call_id = f"{sanitized_tool_call_id}__sanitized__{chosen_tool_name}"
        sanitized_tool_calls = [
            {
                "id": sanitized_tool_call_id,
                "name": chosen_tool_name,
                "args": args,
            }
        ]

        if len(tool_calls) > 1:
            logger.warning(
                "WRITER_NODE: Dropped %d extra tool calls to enforce single-tool execution",
                len(tool_calls) - 1,
            )
    elif tool_calls:
        logger.warning("WRITER_NODE: Tool calls present but none matched allowed tools; dropping all")

    sanitized_response = AIMessage(content=question_text, tool_calls=sanitized_tool_calls)
    if model_tool_name and model_tool_name != chosen_tool_name:
        logger.info(
            "WRITER_NODE: rewrote tool call after scope validation | raw_tool=%s | sanitized_tool=%s | reason=%s",
            model_tool_name,
            chosen_tool_name,
            tool_reason,
        )
    retrieval_plan = {
        "stage": "planned",
        "tool_name": chosen_tool_name,
        "model_tool_name": model_tool_name,
        "original_question": question_text,
        "retrieval_query": retrieval_query,
        "search_type": search_type,
        "effective_search_type": effective_search_type,
        "search_type_reason": search_type_reason,
        "is_retry_attempt": is_retry_attempt,
        "tool_selection_reason": tool_reason,
        "question_scope": scope_info.get("scope", "ambiguous"),
        "scope_reason": scope_info.get("reason", ""),
        "scope_confidence": scope_info.get("confidence", "low"),
        "scope_scores": {
            "internal": scope_info.get("internal_score", 0),
            "external": scope_info.get("external_score", 0),
            "gap": scope_info.get("score_gap", 0),
        },
        "scope_signals": {
            "article": scope_info.get("article_hints", []),
            "external_artifact": scope_info.get("external_artifact_hints", []),
            "external_context": scope_info.get("external_context_hints", []),
            "internal_reasons": scope_info.get("internal_reasons", []),
            "external_reasons": scope_info.get("external_reasons", []),
        },
        "model_tool_overridden": bool(model_tool_name and model_tool_name != chosen_tool_name),
        "tool_handoff_mode": (
            "rewritten_after_scope_validation"
            if model_tool_name and model_tool_name != chosen_tool_name
            else "model_tool_preserved"
        ),
        "raw_tool_call_id": str((primary_tool_call or {}).get("id", "") or ""),
        "sanitized_tool_call_id": str((sanitized_tool_calls[0] or {}).get("id", "") if sanitized_tool_calls else ""),
        "query_rewrite_reason": rewrite_reason,
        "query_shape_reason": query_shape_reason,
        "query_quality_flags": query_flags,
        "query_fingerprint": _query_fingerprint(retrieval_query),
        "duplicate_query": bool(duplicate_event),
        "duplicate_of_event_key": str((duplicate_event or {}).get("event_key", "") or ""),
        "duplicate_of_query": str(
            (duplicate_event or {}).get("retrieval_query", "")
            or (duplicate_event or {}).get("query", "")
            or ""
        ),
    }
    return sanitized_response, retrieval_plan


def _get_perspective_profile(state: ResearcherState, perspective_name: str) -> str:
    profiles = state.get("perspective_profiles", {}) or {}
    return str(profiles.get(perspective_name, "") or "").strip()

def initialize_storm_state(state: ResearcherState) -> dict:
    """Initialize STORM-specific state fields if missing.
    
    Also initializes base ResearcherState fields that supervisor may not provide.
    """
    updates = {}
    
    # STORM-specific fields
    if "perspectives" not in state:
        updates["perspectives"] = []
    if "current_perspective" not in state:
        updates["current_perspective"] = ""
    if "conversation_round" not in state:
        updates["conversation_round"] = 0
    if "expert_responses" not in state:
        updates["expert_responses"] = []
    if "should_continue_conversation" not in state:
        updates["should_continue_conversation"] = False
    if "perspective_messages" not in state:
        updates["perspective_messages"] = {}  # Store messages per perspective for isolation
    if "perspective_research_plans" not in state:
        updates["perspective_research_plans"] = {}  # Research plans per perspective (set by generate_research_plans)
    if "perspective_profiles" not in state:
        updates["perspective_profiles"] = {}  # Bio / behavioral profile for each perspective
    if "draft_report" not in state:
        updates["draft_report"] = ""  # Will be passed from supervisor
    if "research_brief" not in state:
        updates["research_brief"] = ""
    if "article_summary" not in state:
        updates["article_summary"] = ""
    if "search_type" not in state:
        inferred_search_type, _ = _resolve_effective_search_type("both", state.get("research_topic", ""))
        updates["search_type"] = inferred_search_type
    if "forced_perspectives" not in state:
        updates["forced_perspectives"] = []
    if "active_gap_id" not in state:
        updates["active_gap_id"] = ""
    # Q&A Reflection and Retry fields
    if "is_retry_attempt" not in state:
        updates["is_retry_attempt"] = False
    if "retry_query" not in state:
        updates["retry_query"] = ""
    if "original_qa" not in state:
        updates["original_qa"] = {}
    if "current_qa_id" not in state:
        updates["current_qa_id"] = ""
    if "last_question" not in state:
        updates["last_question"] = ""
    if "last_answer" not in state:
        updates["last_answer"] = ""
    if "retry_tool_name" not in state:
        updates["retry_tool_name"] = ""
    if "current_retrieval_plan" not in state:
        updates["current_retrieval_plan"] = {}
    if "last_retrieval_metadata" not in state:
        updates["last_retrieval_metadata"] = {}
    if "retrieval_events" not in state:
        updates["retrieval_events"] = []
    if "conversation_route_reason" not in state:
        updates["conversation_route_reason"] = ""
    if "observability_events" not in state:
        updates["observability_events"] = []
    if "last_evidence" not in state:
        updates["last_evidence"] = []
    if "evidence_ledger" not in state:
        updates["evidence_ledger"] = []
    
    # Base ResearcherState fields (supervisor may not provide these)
    if "tool_call_iterations" not in state:
        updates["tool_call_iterations"] = 0
    if "compressed_research" not in state:
        updates["compressed_research"] = ""
    if "raw_notes" not in state:
        updates["raw_notes"] = []
    
    return updates

def get_draft_report(state: ResearcherState) -> str:
    """Get draft report from state.
    
    Args:
        state: Current researcher state
        
    Returns:
        Draft report string (may be empty if not yet created)
    """
    return state.get("draft_report", "")

def extract_tool_content(messages: List[BaseMessage]) -> str:
    """Extract content from all ToolMessages as plain text.
    
    This is used to avoid OpenAI's strict tool_calls/ToolMessage pairing requirements.
    Instead of passing ToolMessages (which require strict pairing with AIMessages that have tool_calls),
    we extract their content as plain text and inject it into the prompt.
    
    Args:
        messages: List of messages to extract tool content from
        
    Returns:
        Plain text string containing all tool results, separated by double newlines.
        Returns empty string if no ToolMessages found.
    """
    from langchain_core.messages import ToolMessage
    
    tool_contents = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = getattr(msg, 'content', '')
            if content:
                tool_contents.append(str(content))
    
    return "\n\n".join(tool_contents) if tool_contents else ""

def build_conversation_history(state: ResearcherState, current_perspective: str) -> str:
    """Build conversation history string for current perspective only.
    
    Prefer the finalized evidence ledger so retries replace earlier answers in
    downstream context. Fall back to transcript parsing only if needed.
    """
    active_gap_id = str(state.get("active_gap_id", "") or "").strip()
    ledger_entries = [
        entry
        for entry in canonicalize_evidence_ledger(state.get("evidence_ledger", []) or [])
        if entry.get("perspective") == current_perspective
    ]
    if active_gap_id:
        gap_specific_entries = [
            entry
            for entry in ledger_entries
            if str(entry.get("gap_id", "") or "").strip() == active_gap_id
        ]
        if gap_specific_entries:
            ledger_entries = gap_specific_entries

    if ledger_entries:
        history_parts = [
            f"Q: {entry.get('question', '')}\nA: {_truncate_text(entry.get('answer', ''), 500)}"
            for entry in ledger_entries[-MAX_CONVERSATION_ROUNDS:]
            if entry.get("question") and entry.get("answer")
        ]
        if history_parts:
            history_text = "\n\n".join(history_parts)
            logger.info(
                "BUILD_CONVERSATION_HISTORY: using evidence ledger for '%s' (%d entries, %d chars)",
                current_perspective[:100],
                len(history_parts),
                len(history_text),
            )
            return history_text

    perspective_messages = get_perspective_messages(state, current_perspective)
    logger.info(
        "BUILD_CONVERSATION_HISTORY: fallback transcript parsing for '%s' (%d messages)",
        current_perspective[:100],
        len(perspective_messages),
    )

    history_parts = []
    current_question = ""

    for msg in perspective_messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            current_question = _extract_question_from_message(msg)
        elif isinstance(msg, AIMessage) and current_question and not getattr(msg, "tool_calls", None):
            answer = _truncate_text(str(msg.content or ""), 500)
            if answer:
                history_parts.append(f"Q: {current_question}\nA: {answer}")
            current_question = ""

    if history_parts:
        history_text = "\n\n".join(history_parts[-MAX_CONVERSATION_ROUNDS:])
        logger.info("BUILD_CONVERSATION_HISTORY: transcript fallback built %d entries", len(history_parts))
        return history_text

    logger.info("BUILD_CONVERSATION_HISTORY: no prior finalized Q&A for '%s'", current_perspective[:100])
    return "No previous conversation for this perspective."

# ===== STORM NODES =====

def perspective_discovery(state: ResearcherState):
    """Discover diverse perspectives for comprehensive research.
    
    Identifies N distinct perspectives/personas that would ask different questions
    about the research topic.
    """
    # Initialize STORM state fields if missing (critical for first invocation)
    state_updates = initialize_storm_state(state)
    if state_updates:
        state = {**state, **state_updates}
        logger.info("PERSPECTIVE_DISCOVERY: Initialized missing state fields: %s", list(state_updates.keys()))
    
    research_topic = state.get("research_topic", "Unknown")
    research_brief = state.get("research_brief", "")
    article_summary = state.get("article_summary", "")
    draft_report = get_draft_report(state)
    search_type = state.get("search_type", "both")
    forced_perspectives = [
        str(p).strip()
        for p in (state.get("forced_perspectives", []) or [])
        if str(p).strip()
    ]
    logger.info("PERSPECTIVE_DISCOVERY: search_type='%s' (from supervisor)", search_type)
    
    start_time = time.perf_counter()
    
    logger.info("="*60)
    logger.info("PERSPECTIVE_DISCOVERY START | topic='%s'", research_topic[:100])

    existing_perspectives = [p for p in state.get("perspectives", []) or [] if str(p).strip()]
    if existing_perspectives or forced_perspectives:
        selected_perspectives = list(existing_perspectives)
        if forced_perspectives:
            forced_lower = {f.lower() for f in forced_perspectives}
            selected_perspectives = [
                p for p in existing_perspectives if p.lower() in forced_lower
            ] or forced_perspectives
        logger.info(
            "PERSPECTIVE_DISCOVERY: Using %d router/persisted perspectives instead of rediscovering",
            len(selected_perspectives),
        )
        existing_messages = state.get("researcher_messages", [])
        perspective_messages = state.get("perspective_messages", {}) or {}
        for perspective in selected_perspectives:
            perspective_messages.setdefault(perspective, [])

        return {
            "perspectives": selected_perspectives[:MAX_PERSPECTIVES],
            "current_perspective": selected_perspectives[0],
            "conversation_round": 0,
            "expert_responses": [],
            "should_continue_conversation": False,
            "draft_report": state.get("draft_report", ""),
            "research_brief": state.get("research_brief", ""),
            "article_summary": state.get("article_summary", ""),
            "search_type": state.get("search_type", "both"),
            "forced_perspectives": forced_perspectives,
            "active_gap_id": state.get("active_gap_id", ""),
            "tool_call_iterations": state.get("tool_call_iterations", 0),
            "compressed_research": state.get("compressed_research", ""),
            "raw_notes": state.get("raw_notes", []),
            "evidence_ledger": state.get("evidence_ledger", []),
            "perspective_messages": perspective_messages,
            "perspective_research_plans": state.get("perspective_research_plans", {}),
            "perspective_profiles": state.get("perspective_profiles", {}),
            "is_retry_attempt": False,
            "retry_query": "",
            "original_qa": {},
            "current_qa_id": "",
            "last_question": "",
            "last_answer": "",
            "retry_tool_name": "",
            "current_retrieval_plan": {},
            "last_retrieval_metadata": {},
            "retrieval_events": state.get("retrieval_events", []),
            "conversation_route_reason": "reused_router_selected_perspectives",
            "observability_events": state.get("observability_events", []),
            "last_evidence": [],
            "researcher_messages": existing_messages
            + [SystemMessage(content=f"Using {len(selected_perspectives)} selected perspectives.")],
        }

    # Use structured output for perspective discovery
    structured_model = perspective_model.with_structured_output(PerspectiveList)
    
    prompt = storm_perspective_discovery_prompt.format(
        research_topic=research_topic,
        research_brief=research_brief,
        article_summary=article_summary,
        draft_report=draft_report,
        date=get_today_str()
    )
    
    llm_start = time.perf_counter()
    try:
        result = structured_model.invoke([HumanMessage(content=prompt)])
        llm_elapsed = time.perf_counter() - llm_start
        
        perspectives = result.perspectives[:MAX_PERSPECTIVES]  # Limit perspectives
        
        # Validate perspectives list
        if not perspectives or len(perspectives) == 0:
            logger.error("PERSPECTIVE_DISCOVERY ERROR: No perspectives discovered")
            raise ValueError("Perspective discovery failed: no perspectives returned")
        
        if len(perspectives) < MAX_PERSPECTIVES:
            logger.warning("PERSPECTIVE_DISCOVERY: Only %d perspectives found (expected %d)",
                         len(perspectives), MAX_PERSPECTIVES)
        
        logger.info("PERSPECTIVE_DISCOVERY COMPLETE | found=%d perspectives | time=%.2fs",
                   len(perspectives), llm_elapsed)
        for i, p in enumerate(perspectives, 1):
            logger.info("  perspective[%d]: %s", i, p[:100])
        
        # Initialize state with defaults if not present
        existing_messages = state.get("researcher_messages", [])
        
        # CRITICAL: Initialize ALL required ResearcherState fields
        # Supervisor only passes researcher_messages and research_topic,
        # so we must initialize the rest here
        return {
            "perspectives": perspectives,
            "current_perspective": perspectives[0],
            "conversation_round": 0,
            "expert_responses": [],
            "should_continue_conversation": False,
            "draft_report": state.get("draft_report", ""),  # Pass through from supervisor
            "research_brief": state.get("research_brief", ""),
            "article_summary": state.get("article_summary", ""),
            "search_type": state.get("search_type", "both"),  # Initialize search_type (default to "both")
            "forced_perspectives": forced_perspectives,
            "active_gap_id": state.get("active_gap_id", ""),
            "tool_call_iterations": state.get("tool_call_iterations", 0),  # Initialize if missing
            "compressed_research": state.get("compressed_research", ""),  # Initialize if missing
            "raw_notes": state.get("raw_notes", []),  # Initialize if missing
            "evidence_ledger": state.get("evidence_ledger", []),
            "perspective_messages": {p: [] for p in perspectives},  # Initialize empty message lists for each perspective
            "perspective_research_plans": {},  # Initialize empty, will be populated by generate_research_plans
            "perspective_profiles": state.get("perspective_profiles", {}),
            # Initialize Q&A reflection and retry fields
            "is_retry_attempt": False,
            "retry_query": "",
            "original_qa": {},
            "current_qa_id": "",
            "last_question": "",
            "last_answer": "",
            "retry_tool_name": "",
            "current_retrieval_plan": {},
            "last_retrieval_metadata": {},
            "retrieval_events": state.get("retrieval_events", []),
            "conversation_route_reason": "",
            "observability_events": state.get("observability_events", []),
            "last_evidence": [],
            "researcher_messages": existing_messages + [
                SystemMessage(content=f"Discovered {len(perspectives)} perspectives for research.")
            ]
        }
        
    except Exception as e:
        llm_elapsed = time.perf_counter() - llm_start
        logger.error("PERSPECTIVE_DISCOVERY ERROR after %.2fs: %s", llm_elapsed, e)
        logger.exception("Full traceback:")
        raise

async def generate_research_plans(state: ResearcherState):
    """Generate perspective-specific research plans in parallel.
    
    For each perspective discovered, generates a focused research plan that will
    guide the Writer's questions during the Q&A session. Runs LLM calls in parallel
    for efficiency.
    """
    perspectives = state.get("perspectives", [])
    research_topic = state.get("research_topic", "")
    research_brief = state.get("research_brief", "")
    article_summary = state.get("article_summary", "")
    draft_report = get_draft_report(state)
    
    start_time = time.perf_counter()
    
    logger.info("="*60)
    logger.info("GENERATE_RESEARCH_PLANS START | perspectives=%d", len(perspectives))
    
    if not perspectives:
        logger.warning("GENERATE_RESEARCH_PLANS: No perspectives found, returning empty plans")
        return {"perspective_research_plans": {}}
    
    existing_plans = state.get("perspective_research_plans", {}) or {}
    reuse_existing_plans = bool(state.get("reuse_existing_research_plans", False))

    if reuse_existing_plans:
        reusable = {
            perspective: str(existing_plans.get(perspective, "") or "").strip()
            for perspective in perspectives
            if str(existing_plans.get(perspective, "") or "").strip()
        }
        if len(reusable) == len(perspectives):
            logger.info(
                "GENERATE_RESEARCH_PLANS: reusing %d existing plans from supervisor assignment",
                len(reusable),
            )
            return {"perspective_research_plans": reusable}
    
    async def generate_plan_for_perspective(perspective: str) -> tuple[str, str]:
        """Generate research plan for a single perspective."""
        previous_plan = existing_plans.get(perspective, "")
        perspective_memory = build_conversation_history(state, perspective)
        prompt = storm_research_plan_prompt.format(
            research_topic=research_topic,
            research_brief=research_brief,
            article_summary=article_summary,
            perspective=perspective,
            perspective_profile=_get_perspective_profile(state, perspective) or "(not provided)",
            draft_report=draft_report,
            previous_plan=previous_plan or "(none yet)",
            perspective_memory=perspective_memory,
        )
        
        try:
            response = await research_plan_model.ainvoke([HumanMessage(content=prompt)])
            plan = re.sub(r"\s+", " ", str(getattr(response, "content", "") or "")).strip()
            if not plan:
                raise ValueError("empty_plan")
            plan = plan[:520].strip()
            logger.info("  Generated plan for '%s': %s", perspective[:40], plan[:100])
            return (perspective, plan)
        except Exception as e:
            logger.error("  Failed to generate plan for '%s': %s", perspective[:40], e)
            # Fallback to a generic plan
            fallback = f"Investigate key facts and context relevant to {perspective[:50]}."
            return (perspective, fallback)
    
    # Run all plan generations in parallel
    llm_start = time.perf_counter()
    tasks = [generate_plan_for_perspective(p) for p in perspectives]
    results = await asyncio.gather(*tasks)
    llm_elapsed = time.perf_counter() - llm_start
    
    # Build the research plans dict
    perspective_research_plans: Dict[str, str] = dict(results)
    
    total_elapsed = time.perf_counter() - start_time
    logger.info("GENERATE_RESEARCH_PLANS COMPLETE | plans=%d | llm_time=%.2fs | total_time=%.2fs",
               len(perspective_research_plans), llm_elapsed, total_elapsed)
    for perspective, plan in perspective_research_plans.items():
        logger.info("  [%s]: %s", perspective[:30], plan[:80])
    
    return {
        "perspective_research_plans": perspective_research_plans
    }


def writer_node(state: ResearcherState):
    """Writer node: Generate questions and call tools to retrieve information.
    
    The Writer persona asks questions based on their perspective and conversation history,
    then decides which tool to use (tavily_search or retrieve_document_chunks) to retrieve
    information to answer the question.
    
    In retry mode (is_retry_attempt=True), uses the retry_query directly instead of
    generating a new question.
    """
    # Ensure state is initialized
    state_updates = initialize_storm_state(state)
    if state_updates:
        state = {**state, **state_updates}
    
    current_perspective = state.get("current_perspective", "")
    research_topic = state.get("research_topic", "")
    research_brief = state.get("research_brief", "")
    article_summary = state.get("article_summary", "")
    conversation_round = state.get("conversation_round", 0)
    is_retry_attempt = state.get("is_retry_attempt", False)
    retry_query = _normalize_retry_query(state.get("retry_query", ""))
    retry_tool_name = state.get("retry_tool_name", "")
    
    if not current_perspective:
        logger.error("WRITER_NODE ERROR: No current_perspective set")
        raise ValueError("current_perspective must be set before writer_node")
    
    # Get perspective-specific research plan (set by generate_research_plans node)
    perspective_research_plans = state.get("perspective_research_plans", {})
    research_plan = perspective_research_plans.get(current_perspective, "Investigate key facts relevant to this perspective.")
    perspective_profile = _get_perspective_profile(state, current_perspective)
    
    # SAFEGUARD: Prevent writer_node from running if we've already exceeded max rounds
    # This should not happen if routing is correct, but provides defense in depth
    # NOTE: Retry attempts do NOT count against conversation_round
    if conversation_round >= MAX_CONVERSATION_ROUNDS and not is_retry_attempt:
        logger.error("WRITER_NODE ERROR: conversation_round=%d >= MAX_CONVERSATION_ROUNDS=%d - this should not happen!",
                    conversation_round, MAX_CONVERSATION_ROUNDS)
        logger.error("  This indicates a routing bug - writer_node should not be called when max rounds reached")
        # Don't raise - let should_continue_conversation handle routing
        # But log error so we can debug
    
    draft_report = get_draft_report(state)
    conversation_history = build_conversation_history(state, current_perspective)
    
    # Get search_type guidance from state and tighten it when the research topic explicitly forbids web search.
    search_type = state.get("search_type", "both")
    effective_search_type, search_type_reason = _resolve_effective_search_type(search_type, research_topic)
    logger.info(
        "WRITER_NODE: search_type='%s' | effective_search_type='%s' | reason=%s",
        search_type,
        effective_search_type,
        search_type_reason,
    )
    if effective_search_type == "internal":
        search_type_guidance = (
            "PRIORITIZE retrieve_document_chunks for this research. "
            "Keep questions about the supplied article's methods, definitions, counts, thresholds, frameworks, tables, results, and ingested supplementary/appendix content in local retrieval. "
            "Do NOT use tavily_search to offload article reading."
        )
    elif effective_search_type == "external":
        search_type_guidance = (
            "PRIORITIZE tavily_search for this research. Focus on external context, guidelines, related work, author/institution background, or artifact discovery. "
            "Do NOT use tavily_search to answer the article's own methods or results."
        )
    else:  # "both" or default
        search_type_guidance = (
            "Use both tools as needed. Start with retrieve_document_chunks for the supplied article and any ingested appendix/supplementary content. "
            "Use tavily_search only for true external grounding or artifact discovery such as related work, broader context, institutions, repositories, or linked supplements that are not in the local corpus."
        )
    
    start_time = time.perf_counter()
    
    logger.info("="*60)
    logger.info("WRITER_NODE START | perspective='%s' | round=%d | is_retry=%s",
               current_perspective[:50], conversation_round, is_retry_attempt)
    
    existing_entry_count = sum(
        1
        for entry in state.get("evidence_ledger", []) or []
        if entry.get("perspective") == current_perspective
    )
    current_qa_id = state.get("current_qa_id", "") or f"{current_perspective}::q{existing_entry_count + 1}"

    if is_retry_attempt and retry_query:
        logger.info("WRITER_NODE: ⚡ RETRY MODE - using retry_query: %s", retry_query[:80])
        prompt = (
            "You are retrying one failed-or-weak research question.\n\n"
            f"<Supervisor Research Topic>\n{research_topic}\n</Supervisor Research Topic>\n\n"
            f"<Research Brief>\n{research_brief}\n</Research Brief>\n\n"
            f"<Article Summary>\n{article_summary}\n</Article Summary>\n\n"
            f"<Perspective>\n{current_perspective}\n</Perspective>\n\n"
            f"<Perspective Profile>\n{perspective_profile or '(not provided)'}\n</Perspective Profile>\n\n"
            f"<Research Plan>\n{research_plan}\n</Research Plan>\n\n"
            f"<Retry Query>\n{retry_query}\n</Retry Query>\n\n"
            f"<Search Type Guidance>\n{search_type_guidance}\n</Search Type Guidance>\n\n"
            "Rules:\n"
            "- Output ONLY the retry query in your message content.\n"
            "- Make exactly ONE tool call.\n"
            "- The system may rewrite the retry query into a tighter retrieval query before execution.\n"
            f"- If using retrieve_document_chunks, set doc_id to \"{DEFAULT_DOC_ID}\".\n"
            "- Do not ask a new question.\n"
            "- Do not call more than one tool.\n"
        )
    else:
        prompt = storm_writer_prompt.format(
            research_topic=research_topic,
            research_brief=research_brief,
            article_summary=article_summary,
            perspective=current_perspective,
            perspective_profile=perspective_profile or "(not provided)",
            research_plan=research_plan,
            draft_report=draft_report,
            conversation_history=conversation_history,
            search_type_guidance=search_type_guidance,
            conversation_round=conversation_round + 1,  # +1 because round is incremented after asking
            max_conversation_rounds=MAX_CONVERSATION_ROUNDS,
        )
    
    logger.info("  research_plan: %s", research_plan[:100] if research_plan else "(none)")
    
    # Use LangSmith run name to separate traces per perspective
    run_name = f"storm_writer_perspective_{current_perspective[:30].replace(' ', '_')}"
    
    llm_start = time.perf_counter()
    try:
        # Writer asks question and calls tools in one LLM call
        response = writer_model_with_tools.invoke(
            [HumanMessage(content=prompt)],
            config={
                "run_name": run_name,
                "tags": [f"perspective_{current_perspective[:30].replace(' ', '_')}", "storm", "writer"]
            }
        )
        llm_elapsed = time.perf_counter() - llm_start
        
        forced_question = retry_query.strip() if is_retry_attempt and retry_query else ""
        deterministic_first_query = ""
        if (
            not forced_question
            and conversation_round == 0
            and search_type == "internal"
            and not state.get("evidence_ledger")
        ):
            candidate_first_query = _extract_supervisor_focus_query(
                research_topic=research_topic,
                research_plan=research_plan,
            )
            candidate_flags = set(_query_quality_flags(candidate_first_query, "retrieve_document_chunks"))
            if (
                candidate_first_query
                and candidate_first_query.count(",") < 2
                and not (candidate_flags & {"multipart", "too_long_for_local_rerank"})
            ):
                deterministic_first_query = candidate_first_query
        question_text = (
            forced_question
            or deterministic_first_query
            or _extract_question_from_message(response)
            or _normalize_question_text(research_topic)
            or "What specific missing detail should be researched?"
        )
        sanitized_response, retrieval_plan = _build_retrieval_plan(
            response=response,
            question_text=question_text,
            search_type=search_type,
            retry_tool_name=retry_tool_name,
            is_retry_attempt=is_retry_attempt,
            research_topic=research_topic,
            article_summary=article_summary,
            retrieval_events=state.get("retrieval_events", []) or [],
            active_gap_id=state.get("active_gap_id", ""),
        )
        tool_calls = getattr(sanitized_response, 'tool_calls', [])
        log_token_usage(logger, response, "writer_node")
        get_global_tracker().add_usage(response, "writer_node")
        
        # Validate that Writer called tools (required for this design)
        if not tool_calls:
            logger.warning("WRITER_NODE: Writer did not call any tools. This may cause issues in expert_synthesize.")
        
        # Determine should_continue based on conversation_round
        should_continue = conversation_round + 1 < MAX_CONVERSATION_ROUNDS
        
        logger.info("WRITER_NODE: tool_calls=%d | should_continue=%s | time=%.2fs",
                   len(tool_calls) if tool_calls else 0, should_continue, llm_elapsed)
        
        if tool_calls:
            for tc in tool_calls:
                logger.info("  tool_call: %s (id=%s)", tc.get('name'), tc.get('id'))
        logger.info(
            "  retrieval_plan: tool=%s | query='%s' | flags=%s | selection=%s",
            retrieval_plan.get("tool_name", ""),
            retrieval_plan.get("retrieval_query", "")[:120],
            retrieval_plan.get("query_quality_flags", []),
            retrieval_plan.get("tool_selection_reason", ""),
        )
        if retrieval_plan.get("duplicate_query"):
            logger.info(
                "  retrieval_plan_duplicate: prior_event=%s | prior_query='%s'",
                retrieval_plan.get("duplicate_of_event_key", ""),
                retrieval_plan.get("duplicate_of_query", "")[:120],
            )
        
        logger.info("WRITER_NODE COMPLETE | tool_calls=%d | should_continue=%s | is_retry=%s | total_time=%.2fs",
                   len(tool_calls) if tool_calls else 0, should_continue, is_retry_attempt,
                   time.perf_counter() - start_time)
        
        # Return delta update - conversation_history tracks Q&A naturally
        # NOTE: Retry attempts do NOT increment conversation_round
        result = {
            "researcher_messages": [sanitized_response],
            "perspective_messages": {current_perspective: [sanitized_response]},
            "should_continue_conversation": should_continue,
            "current_qa_id": current_qa_id,
            "current_retrieval_plan": {
                **retrieval_plan,
                "qa_id": current_qa_id,
                "event_key": f"{current_qa_id}:{'retry' if is_retry_attempt else 'initial'}:planned",
            },
            "retrieval_events": [
                {
                    **retrieval_plan,
                    "qa_id": current_qa_id,
                    "event_key": f"{current_qa_id}:{'retry' if is_retry_attempt else 'initial'}:planned",
                }
            ],
            "observability_events": [
                {
                    "category": "scope_routing",
                    "node": "writer_node",
                    "event_key": f"{current_qa_id}:{'retry' if is_retry_attempt else 'initial'}:scope_routing",
                    "requested_search_type": search_type,
                    "effective_search_type": retrieval_plan.get("effective_search_type", effective_search_type),
                    "question_scope": retrieval_plan.get("question_scope", ""),
                    "scope_reason": retrieval_plan.get("scope_reason", ""),
                    "scope_confidence": retrieval_plan.get("scope_confidence", "low"),
                    "selected_tool": retrieval_plan.get("tool_name", ""),
                    "tool_selection_reason": retrieval_plan.get("tool_selection_reason", ""),
                    "model_tool_overridden": retrieval_plan.get("model_tool_overridden", False),
                }
            ],
        }
        
        # Only increment conversation_round for non-retry attempts
        if not is_retry_attempt:
            result["conversation_round"] = conversation_round + 1
        else:
            logger.info("  RETRY: Not incrementing conversation_round (stays at %d)", conversation_round)
        
        return result
        
    except Exception as e:
        llm_elapsed = time.perf_counter() - llm_start
        logger.error("WRITER_NODE ERROR after %.2fs: %s", llm_elapsed, e)
        logger.exception("Full traceback:")
        raise

# expert_node removed - Writer now calls tools directly, Expert only synthesizes

def writer_tool_node(state: ResearcherState):
    """Execute tool calls from the Writer node.
    
    Executes tavily_search and retrieve_document_chunks calls that the Writer requested.
    
    CRITICAL: Stores tool results in perspective-isolated storage.
    """
    start_time = time.perf_counter()
    
    current_perspective = state.get("current_perspective", "")
    search_type = state.get("search_type", "both")
    current_retrieval_plan = state.get("current_retrieval_plan", {}) or {}
    current_qa_id = state.get("current_qa_id", "")
    effective_search_type = current_retrieval_plan.get("effective_search_type", search_type)
    
    # Get the last message from perspective-isolated messages (should be the writer's tool selection with tool_calls)
    messages = state.get("researcher_messages", [])  # Still check main for tool calls
    if not messages:
        logger.error("WRITER_TOOL_NODE ERROR: No messages in state")
        return {"researcher_messages": []}
    
    last_message = messages[-1]
    tool_calls = getattr(last_message, 'tool_calls', None)
    
    if not tool_calls:
        logger.warning("WRITER_TOOL_NODE: No tool calls found in last message")
        return {"researcher_messages": []}
    
    tool_calls = list(tool_calls)
    if len(tool_calls) > 1:
        logger.warning("WRITER_TOOL_NODE: Received %d tool calls, executing only the first", len(tool_calls))
        tool_calls = tool_calls[:1]
    
    logger.info("="*60)
    logger.info("WRITER_TOOL_NODE START | executing %d tool calls", len(tool_calls))

    tool_outputs: List[ToolMessage] = []
    retrieval_events: List[Dict[str, Any]] = []
    total_result_chars = 0
    combined_retrieval_metadata: Dict[str, Any] = {}

    if current_retrieval_plan.get("duplicate_query"):
        duplicate_tool = str(current_retrieval_plan.get("tool_name", "") or "tool")
        duplicate_query = str(current_retrieval_plan.get("retrieval_query", "") or "")
        duplicate_of_query = str(current_retrieval_plan.get("duplicate_of_query", "") or "")
        duplicate_of_event_key = str(current_retrieval_plan.get("duplicate_of_event_key", "") or "")
        logger.info(
            "WRITER_TOOL_NODE: skipping duplicate retrieval | tool=%s | query='%s' | prior_event=%s",
            duplicate_tool,
            duplicate_query[:120],
            duplicate_of_event_key or "(unknown)",
        )
        note = (
            f"Skipped duplicate {duplicate_tool} retrieval for this agenda item because the same query was already attempted. "
            f"Current query: {duplicate_query or '(empty)'}."
        )
        if duplicate_of_query and duplicate_of_query != duplicate_query:
            note += f" Prior query: {duplicate_of_query}."
        synthetic_tool_message = ToolMessage(
            content=note,
            name=duplicate_tool,
            tool_call_id=f"{current_qa_id or 'qa'}__duplicate_skip",
        )
        return {
            "researcher_messages": [synthetic_tool_message],
            "perspective_messages": {current_perspective: [synthetic_tool_message]},
            "last_retrieval_metadata": {
                "tool_name": duplicate_tool,
                "status": "duplicate_skipped",
                "query": duplicate_query,
                "duplicate_of_event_key": duplicate_of_event_key,
            },
            "retrieval_events": [
                {
                    **current_retrieval_plan,
                    "stage": "skipped_duplicate",
                    "qa_id": current_qa_id,
                    "event_key": f"{current_qa_id}:{duplicate_tool}:skipped_duplicate",
                    "tool_name": duplicate_tool,
                    "retrieval_query": duplicate_query,
                    "result_status": "duplicate_skipped",
                    "duplicate_of_event_key": duplicate_of_event_key,
                }
            ],
        }
    
    for i, tool_call in enumerate(tool_calls):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        logger.info("  executing tool[%d]: %s", i, tool_name)
        logger.info("    args: %s", str(tool_args)[:200])
        
        tool_start = time.perf_counter()
        try:
            tool = writer_tools_by_name[tool_name]
            result = tool.invoke(tool_args)
            tool_elapsed = time.perf_counter() - tool_start

            result_metadata = _parse_tool_result_metadata(tool_name, str(result))
            result_metadata.update(
                {
                    "qa_id": current_qa_id,
                    "query": tool_args.get("query", ""),
                    "original_question": current_retrieval_plan.get("original_question", ""),
                }
            )

            tool_outputs.append(
                ToolMessage(
                    content=str(result),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )

            fallback_used = False
            if (
                tool_name == "retrieve_document_chunks"
                and effective_search_type in {"both", "external"}
                and result_metadata.get("status") == "no_match"
                and current_retrieval_plan.get("question_scope") in {"external_artifact", "external_context"}
                and current_retrieval_plan.get("scope_confidence") == "strong"
            ):
                fallback_query = (
                    _shape_external_search_query(
                        current_retrieval_plan.get("original_question", ""),
                        state.get("article_summary", ""),
                    )[0]
                    or tool_args.get("query", "")
                )
                logger.info(
                    "    ADAPTIVE FALLBACK: local miss on external-looking query -> tavily_search('%s')",
                    fallback_query[:100],
                )
                fallback_result = tavily_search.invoke({"query": fallback_query})
                fallback_metadata = _parse_tool_result_metadata("tavily_search", str(fallback_result))
                fallback_metadata.update(
                    {
                        "qa_id": current_qa_id,
                        "query": fallback_query,
                        "fallback_reason": "local_no_match_external_gap",
                    }
                )
                tool_outputs.append(
                    ToolMessage(
                        content=str(fallback_result),
                        name="tavily_search",
                        tool_call_id=f"{tool_call['id']}__adaptive_tavily",
                    )
                )
                retrieval_events.append(
                    {
                        **current_retrieval_plan,
                        "stage": "executed_fallback",
                        "qa_id": current_qa_id,
                        "event_key": f"{current_qa_id}:adaptive_tavily",
                        "tool_name": "tavily_search",
                        "retrieval_query": fallback_query,
                        "result_status": fallback_metadata.get("status", ""),
                        "source_count": fallback_metadata.get("source_count", 0),
                        "fallback_reason": "local_no_match_external_gap",
                    }
                )
                result_metadata["fallback_tool_name"] = "tavily_search"
                result_metadata["fallback_status"] = fallback_metadata.get("status", "")
                result_metadata["fallback_source_count"] = fallback_metadata.get("source_count", 0)
                fallback_used = True

            retrieval_events.append(
                {
                    **current_retrieval_plan,
                    "stage": "executed",
                    "qa_id": current_qa_id,
                    "event_key": f"{current_qa_id}:{tool_name}:executed",
                    "tool_name": tool_name,
                    "retrieval_query": tool_args.get("query", ""),
                    "result_status": result_metadata.get("status", ""),
                    "best_score": result_metadata.get("best_score", 0.0),
                    "matched_chunks": result_metadata.get("matched_chunks", 0),
                    "chunk_locators": result_metadata.get("chunk_locators", []),
                    "fallback_used": fallback_used,
                }
            )

            combined_retrieval_metadata = result_metadata
            result_len = len(str(result))
            total_result_chars += result_len
            logger.info(
                "    result_len: %d chars in %.2fs | status=%s | best_score=%s",
                result_len,
                tool_elapsed,
                result_metadata.get("status", ""),
                result_metadata.get("best_score", ""),
            )
                    
        except Exception as e:
            tool_elapsed = time.perf_counter() - tool_start
            logger.error("    TOOL ERROR after %.2fs: %s", tool_elapsed, e)
            logger.exception("    Full traceback:")
            error_text = f"Error executing {tool_name}: {str(e)}"
            tool_outputs.append(
                ToolMessage(
                    content=error_text,
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
            retrieval_events.append(
                {
                    **current_retrieval_plan,
                    "stage": "executed",
                    "qa_id": current_qa_id,
                    "event_key": f"{current_qa_id}:{tool_name}:error",
                    "tool_name": tool_name,
                    "retrieval_query": tool_args.get("query", ""),
                    "result_status": "error",
                }
            )

    # Return delta - only new tool outputs
    total_elapsed = time.perf_counter() - start_time
    logger.info("WRITER_TOOL_NODE COMPLETE | messages=%d | total_chars=%d | time=%.2fs", 
               len(tool_outputs), total_result_chars, total_elapsed)

    return {
        "researcher_messages": tool_outputs,
        "perspective_messages": {current_perspective: tool_outputs},
        "last_retrieval_metadata": combined_retrieval_metadata,
        "retrieval_events": retrieval_events,
    }

def get_perspective_messages(state: ResearcherState, perspective: str) -> List:
    """Get messages isolated to a specific perspective.
    
    Returns only messages from the current perspective to ensure isolation.
    This prevents perspectives from seeing each other's conversations.
    
    CRITICAL: Never falls back to researcher_messages (which contains ALL perspectives).
    Returns empty list if perspective_messages is not initialized or perspective not found.
    """
    perspective_messages_dict = state.get("perspective_messages", {})
    
    logger.debug("get_perspective_messages: Looking for perspective '%s' (len=%d) | dict_keys=%s",
                perspective[:100], len(perspective),
                list(perspective_messages_dict.keys())[:3] if isinstance(perspective_messages_dict, dict) else "not_a_dict")
    
    # Ensure it's a dict (not None or other type)
    if not isinstance(perspective_messages_dict, dict):
        logger.warning("get_perspective_messages: ❌ perspective_messages is not a dict (type=%s), initializing empty dict",
                      type(perspective_messages_dict).__name__)
        perspective_messages_dict = {}
    
    # Get messages for this perspective if they exist
    if perspective in perspective_messages_dict:
        messages = perspective_messages_dict[perspective]
        if isinstance(messages, list):
            logger.debug("get_perspective_messages: ✅ Found %d messages for perspective '%s'",
                        len(messages), perspective[:100])
            if messages:
                msg_types = [type(m).__name__ for m in messages]
                logger.debug("  Message types: %s", msg_types)
            return messages
        else:
            logger.warning("get_perspective_messages: ❌ Messages for perspective '%s' is not a list (type=%s), returning empty list",
                          perspective[:100], type(messages).__name__)
            return []
    
    # If perspective not found, return empty list (don't fall back to researcher_messages)
    # This ensures we don't mix perspectives and maintains isolation
    logger.warning("get_perspective_messages: ❌ Perspective '%s' not found in perspective_messages", perspective[:100])
    if isinstance(perspective_messages_dict, dict) and perspective_messages_dict:
        logger.warning("  Available keys: %s", [k[:100] for k in list(perspective_messages_dict.keys())[:3]])
        # Check for partial match (in case of truncation issues)
        for key in perspective_messages_dict.keys():
            if perspective[:50] in key[:50] or key[:50] in perspective[:50]:
                logger.warning("  ⚠️ Possible partial match found: '%s' (requested: '%s')", key[:100], perspective[:100])
    else:
        logger.warning("  perspective_messages dict is empty")
    return []

def expert_synthesize(state: ResearcherState):
    """Expert synthesizes final answer after tool execution.
    
    After tools are executed, the Expert synthesizes a comprehensive answer.
    If no tools were called, the Expert provides an answer based on available context.
    
    CRITICAL: Only uses messages from the current perspective to ensure isolation.
    
    NOTE: During retry attempts, we do NOT add to expert_responses here.
    The answer_synthesis node will add the final synthesized answer instead.
    This prevents duplicate entries in expert_responses.
    """
    research_topic = state.get("research_topic", "Unknown")
    current_perspective = state.get("current_perspective", "")
    is_retry_attempt = state.get("is_retry_attempt", False)
    
    # ISOLATION: Only get messages from current perspective
    perspective_messages = get_perspective_messages(state, current_perspective)
    
    # DEBUG: Log what we're seeing
    logger.info("EXPERT_SYNTHESIZE: perspective_messages count=%d for '%s'", 
               len(perspective_messages), current_perspective[:50])
    if perspective_messages:
        logger.info("  Message types: %s", [type(m).__name__ for m in perspective_messages])
        # Log the most recent messages
        for i, msg in enumerate(perspective_messages[-3:], start=max(0, len(perspective_messages)-3)):
            msg_type = type(msg).__name__
            if isinstance(msg, AIMessage):
                has_tool_calls = bool(getattr(msg, 'tool_calls', None))
                content_preview = str(msg.content)[:100] if msg.content else "(empty)"
                logger.info("    [%d] %s (tool_calls=%s): %s", i, msg_type, has_tool_calls, content_preview)
            elif isinstance(msg, ToolMessage):
                content_preview = str(msg.content)[:100] if msg.content else "(empty)"
                logger.info("    [%d] %s: %s", i, msg_type, content_preview)
            else:
                logger.info("    [%d] %s", i, msg_type)
    else:
        logger.warning("EXPERT_SYNTHESIZE: perspective_messages is EMPTY - this will cause question extraction to fail!")
    
    # Extract question from Writer's response (Writer now asks question and calls tools)
    # The writer's question is in an AIMessage with tool_calls that precedes the ToolMessages
    current_question = ""
    
    # Strategy 1: Find the most recent AIMessage with tool_calls (Writer's question)
    logger.info("EXPERT_SYNTHESIZE: Question extraction Strategy 1 | scanning %d messages backwards",
                len(perspective_messages))
    for idx, msg in enumerate(reversed(perspective_messages)):
        if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
            logger.info("EXPERT_SYNTHESIZE: Strategy 1 - Found AIMessage with tool_calls at index %d (from end)",
                       idx)
            # CRITICAL: Extract question from content OR tool call query
            # When models call tools, content may be empty, but question is in tool call's query argument
            question_text = str(msg.content) if msg.content else ""
            extraction_source = "msg.content" if question_text else "none"
            
            logger.info("EXPERT_SYNTHESIZE: Strategy 1 - msg.content='%s' (len=%d) | tool_calls=%d",
                       str(msg.content)[:100] if msg.content else "(empty)",
                       len(str(msg.content)) if msg.content else 0,
                       len(getattr(msg, 'tool_calls', [])))
            
            # If content is empty, extract from tool call query
            if not question_text or not question_text.strip():
                tool_calls = getattr(msg, 'tool_calls', [])
                for tc_idx, tc in enumerate(tool_calls):
                    tool_args = tc.get('args', {})
                    query = tool_args.get('query', '')
                    if query:
                        question_text = query
                        extraction_source = f"tool_calls[{tc_idx}].args.query"
                        logger.info("EXPERT_SYNTHESIZE: ✅ Strategy 1 - Extracted question from tool call query (len=%d): '%s'", 
                                   len(question_text), question_text[:150])
                        break
            
            if question_text and question_text.strip():
                current_question = question_text.strip()
                logger.info("EXPERT_SYNTHESIZE: ✅ Strategy 1 SUCCESS - Extracted question (source=%s, len=%d): '%s'", 
                           extraction_source, len(current_question), current_question[:150])
                break
            else:
                logger.warning("EXPERT_SYNTHESIZE: Strategy 1 - Could not extract question from AIMessage (source=%s)", 
                             extraction_source)
    
    # Strategy 2: If no AIMessage with tool_calls found, look for most recent ToolMessage and find preceding AIMessage
    if not current_question:
        # Find the most recent ToolMessage (should be from this round) and look backwards for the AIMessage that triggered it
        for i in range(len(perspective_messages) - 1, -1, -1):
            msg = perspective_messages[i]
            if isinstance(msg, ToolMessage):
                # Look backwards from this ToolMessage for the AIMessage with tool_calls that triggered it
                for j in range(i - 1, -1, -1):
                    prev_msg = perspective_messages[j]
                    if isinstance(prev_msg, AIMessage) and getattr(prev_msg, 'tool_calls', None):
                        # CRITICAL: Extract question from content OR tool call query
                        question_text = str(prev_msg.content) if prev_msg.content else ""
                        
                        # If content is empty, extract from tool call query
                        if not question_text or not question_text.strip():
                            tool_calls = getattr(prev_msg, 'tool_calls', [])
                            for tc in tool_calls:
                                tool_args = tc.get('args', {})
                                query = tool_args.get('query', '')
                                if query:
                                    question_text = query
                                    break
                        
                        if question_text and question_text.strip():
                            current_question = question_text.strip()
                            logger.info("EXPERT_SYNTHESIZE: Extracted question from AIMessage preceding ToolMessage: '%s'", 
                                       current_question[:100])
                            break
                if current_question:
                    break
    
    # Strategy 3: Fallback to old format (HumanMessage with [Writer])
    if not current_question:
        for msg in reversed(perspective_messages):
            if isinstance(msg, HumanMessage) and "[Writer" in str(msg.content):
                question_text = str(msg.content).split("]: ", 1)[-1] if "]: " in str(msg.content) else str(msg.content)
                if question_text and question_text.strip():
                    current_question = question_text.strip()
                    logger.info("EXPERT_SYNTHESIZE: Extracted question from old format HumanMessage: '%s'", 
                               current_question[:100])
                    break
    
    # Strategy 4: Last resort - use a generic message (NOT research_topic, which is the supervisor's call)
    if not current_question:
        current_question = "the current research question"  # Generic fallback, not research_topic
        logger.warning("EXPERT_SYNTHESIZE: Could not extract question from messages, using generic fallback")
    
    # Check if there are tool results (ToolMessages) from Writer's tool calls
    has_tool_results = any(isinstance(msg, ToolMessage) for msg in perspective_messages)
    
    # Handle case where Writer didn't call tools (edge case)
    if not has_tool_results:
        logger.warning("EXPERT_SYNTHESIZE: No tool results found. Writer may not have called tools.")
        # Provide a response acknowledging this limitation
        fallback_answer = (
            f"I notice that no information was retrieved for the question: '{current_question[:200]}'. "
            f"Without retrieved information, I cannot provide a comprehensive answer. "
            f"Please ensure tools are called to retrieve relevant information before asking for synthesis."
        )
        
        # CRITICAL: DELTA APPROACH - Return only NEW fallback answer, not full accumulated state
        expert_message = AIMessage(content=fallback_answer)
        
        result = {
            "researcher_messages": [expert_message],  # Delta: only new message
            "perspective_messages": {current_perspective: [expert_message]},  # Delta: only new message for current perspective
            "last_question": current_question,  # For reflection node
            "last_answer": fallback_answer,  # For reflection node
            "last_evidence": [],
        }
        
        return result
    
    # ISOLATION: Get only messages from the CURRENT question's tool execution
    # Find the most recent AIMessage with tool_calls (the current question's Writer message)
    # and only include ToolMessages that come AFTER it
    current_question_start_idx = -1
    for i in range(len(perspective_messages) - 1, -1, -1):
        msg = perspective_messages[i]
        if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
            current_question_start_idx = i
            break
    
    if current_question_start_idx >= 0:
        # Only get messages from the current question onwards
        current_question_messages = perspective_messages[current_question_start_idx:]
        logger.info("EXPERT_SYNTHESIZE: Isolated current question messages (idx %d onwards, %d messages)",
                   current_question_start_idx, len(current_question_messages))
    else:
        # Fallback: use last 15 messages if no AIMessage with tool_calls found
        current_question_messages = perspective_messages[-15:] if len(perspective_messages) > 15 else perspective_messages
        logger.warning("EXPERT_SYNTHESIZE: No AIMessage with tool_calls found, using fallback (last %d messages)",
                      len(current_question_messages))
    
    # CRITICAL: Extract tool content as plain text to avoid OpenAI's strict tool_calls/ToolMessage pairing
    # OpenAI API requires strict pairing, so we extract tool results as plain text instead
    # Only extract from current question's messages to avoid including previous Q&A's tool results
    tool_content = extract_tool_content(current_question_messages)
    current_tool_messages = [msg for msg in current_question_messages if isinstance(msg, ToolMessage)]
    evidence_items = extract_evidence_from_tool_messages(current_tool_messages)
    
    # Clean history: Remove ALL ToolMessages and ALL AIMessages with tool_calls
    # Keep only SystemMessage, HumanMessage, and AIMessage without tool_calls
    clean_messages = []
    for msg in current_question_messages:
        # Skip ToolMessages (we've extracted their content)
        if isinstance(msg, ToolMessage):
            continue
        # Skip AIMessages with tool_calls (Writer's tool selection - we don't need it)
        if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
            continue
        # Keep everything else (SystemMessage, HumanMessage, AIMessage without tool_calls)
        clean_messages.append(msg)
    
    logger.info("EXPERT_SYNTHESIZE: Extracted tool content (%d chars), cleaned messages (%d -> %d)",
               len(tool_content), len(current_question_messages), len(clean_messages))
    logger.info("EXPERT_SYNTHESIZE: extracted %d structured evidence items", len(evidence_items))
    
    start_time = time.perf_counter()
    
    logger.info("="*60)
    logger.info("EXPERT_SYNTHESIZE START | perspective='%s' | question='%s' | has_tool_results=%s | clean_messages=%d",
               current_perspective[:50], current_question[:100], has_tool_results, len(clean_messages))
    
    # Build expert prompt - simple and focused
    # Don't truncate article summary - provide full context
    formatted_prompt = expert_synthesize_prompt.format(question=current_question)
    
    # Inject tool content as plain text to avoid OpenAI's strict tool_calls/ToolMessage pairing
    # Append tool results to the prompt as plain text
    if tool_content:
        formatted_prompt += f"\n\n<Retrieved Context>\n{tool_content}\n</Retrieved Context>"
    
    # Build clean message list: System prompt + clean messages (no ToolMessages, no AIMessages with tool_calls)
    # This ensures we never trigger OpenAI's strict validation
    synthesis_messages = [
        SystemMessage(content=formatted_prompt)
    ] + clean_messages
    
    # Retry logic with exponential backoff
    max_retries = 3
    retry_delay = 1.0  # Start with 1 second
    
    # Use LangSmith run name to separate traces per perspective
    run_name = f"storm_expert_synthesize_perspective_{current_perspective[:30].replace(' ', '_')}"
    
    for attempt in range(max_retries):
        llm_start = time.perf_counter()
        try:
            # Invoke with metadata for LangSmith separation
            response = expert_model.invoke(
                synthesis_messages,
                config={
                    "run_name": run_name,
                    "tags": [f"perspective_{current_perspective[:30].replace(' ', '_')}", "storm", "expert_synthesize"]
                }
            )
            llm_elapsed = time.perf_counter() - llm_start
            
            answer = response.content if response.content else ""
            
            log_token_usage(logger, response, "expert_synthesize")
            get_global_tracker().add_usage(response, "expert_synthesize")
            
            logger.info("EXPERT_SYNTHESIZE COMPLETE | answer_len=%d | time=%.2fs | attempt=%d",
                       len(answer), llm_elapsed, attempt + 1)
            logger.info("  answer_preview: %s", answer[:300])
            
            # CRITICAL: DELTA APPROACH - Return only NEW answer, not full accumulated state
            # The reducers (merge_dicts, operator.add) will correctly merge these deltas
            # This prevents exponential growth from returning full state + new items
            expert_message = AIMessage(content=answer)
            
            logger.info("EXPERT_SYNTHESIZE: 📤 Returning DELTA:")
            logger.info("  - researcher_messages: 1 new answer message")
            logger.info("  - perspective_messages: 1 new answer for '%s'", current_perspective[:100])
            
            result = {
                "researcher_messages": [expert_message],  # Delta: only new message
                "perspective_messages": {current_perspective: [expert_message]},  # Delta: only new message for current perspective
                "last_question": current_question,  # For reflection node
                "last_answer": answer,  # For reflection node
                "last_evidence": evidence_items,
            }
            
            return result
            
        except Exception as e:
            llm_elapsed = time.perf_counter() - llm_start
            error_msg = str(e)
            
            # Check if it's the specific ToolMessage error
            is_tool_message_error = "messages with role 'tool' must be a response" in error_msg or \
                                   "tool_calls" in error_msg.lower()
            
            if is_tool_message_error and attempt < max_retries - 1:
                # This shouldn't happen with our new approach, but if it does, use only the System prompt
                logger.warning("EXPERT_SYNTHESIZE: Unexpected tool message error (attempt %d/%d), using system prompt only",
                             attempt + 1, max_retries)
                
                # Use only the System prompt (which already contains tool content as plain text)
                synthesis_messages = [
                    SystemMessage(content=formatted_prompt)
                ]
                
                retry_delay *= 2  # Exponential backoff
                time.sleep(retry_delay)
                continue
            
            # For other errors or final attempt, log and handle gracefully
            logger.error("EXPERT_SYNTHESIZE ERROR after %.2fs (attempt %d/%d): %s", 
                       llm_elapsed, attempt + 1, max_retries, error_msg)
            
            if attempt == max_retries - 1:
                # Final attempt failed - provide a graceful fallback answer
                logger.error("EXPERT_SYNTHESIZE: All retries exhausted, providing fallback answer")
                logger.exception("Full traceback:")
                
                # Provide a fallback answer based on available context
                fallback_answer = (
                    f"I encountered an error while synthesizing the answer to: '{current_question[:200]}'. "
                    f"Based on the available context from previous research, I can provide the following: "
                    f"The information gathered from previous expert responses should be available in the conversation history. "
                    f"Please refer to the earlier expert responses for details on this topic."
                )
                
                # CRITICAL: DELTA APPROACH - Return only NEW fallback answer, not full accumulated state
                logger.warning("EXPERT_SYNTHESIZE: Returning fallback answer to continue pipeline")
                
                result = {
                    "researcher_messages": [AIMessage(content=fallback_answer)],  # Delta: only new message
                    "perspective_messages": {current_perspective: [AIMessage(content=fallback_answer)]},  # Delta: only new message for current perspective
                    "last_question": current_question,  # For reflection node
                    "last_answer": fallback_answer,  # For reflection node
                    "last_evidence": evidence_items,
                }
                
                return result
            
            # Wait before retry
            retry_delay *= 2
            time.sleep(retry_delay)

def compress_research(state: ResearcherState) -> dict:
    """Compile a stable, structured findings packet for downstream handoff.

    Default behavior is deterministic compilation for better long-running context
    stability. Optional LLM compression is available behind
    DEEP_RESEARCH_STORM_LLM_COMPRESSION=1.
    """
    start_time = time.perf_counter()

    research_topic = state.get("research_topic", "Unknown topic")
    evidence_ledger = canonicalize_evidence_ledger(state.get("evidence_ledger", []) or [])
    expert_responses = state.get("expert_responses", [])
    retrieval_events = state.get("retrieval_events", []) or []
    observability_events = state.get("observability_events", []) or []
    perspectives = state.get("perspectives", [])
    perspective_research_plans = state.get("perspective_research_plans", {})

    logger.info("=" * 60)
    logger.info("COMPRESS_RESEARCH START | topic='%s'", research_topic[:100])
    logger.info("  perspectives: %d", len(perspectives))
    logger.info("  evidence_ledger: %d", len(evidence_ledger))
    logger.info("  expert_responses: %d", len(expert_responses))
    logger.info("  perspective_research_plans: %d", len(perspective_research_plans))

    if not evidence_ledger and not expert_responses:
        logger.warning("COMPRESS_RESEARCH: No expert responses or perspective messages found, returning empty research")
        return {
            "compressed_research": "No research findings were generated.",
            "raw_notes": [],
            "evidence_ledger": [],
            "retrieval_events": retrieval_events,
            "observability_events": observability_events,
            "perspectives": perspectives,
            "perspective_research_plans": perspective_research_plans,
            "perspective_profiles": state.get("perspective_profiles", {}) or {},
        }

    def _entry_line(entry: dict) -> str:
        question = _truncate_text(str(entry.get("question", "") or ""), 150)
        answer = _truncate_text(str(entry.get("answer", "") or ""), 280)
        refs = [
            _truncate_text(str(item.get("locator", "") or ""), 100)
            for item in (entry.get("evidence", []) or [])[:2]
            if str(item.get("locator", "") or "").strip()
        ]
        line = f"- Q: {question} | A: {answer}"
        if refs:
            line += " | refs: " + ", ".join(refs)
        return line

    supported = [entry for entry in evidence_ledger if str(entry.get("answer_status", "supported") or "supported") == "supported"]
    unresolved = [
        entry for entry in evidence_ledger
        if str(entry.get("answer_status", "") or "") in {"missing", "needs_review", "not_in_source", "premise_mismatch"}
    ]
    conflicted = [entry for entry in evidence_ledger if str(entry.get("answer_status", "") or "") == "conflicted"]
    mix = source_mix_report(evidence_ledger, retrieval_events)

    deterministic_lines = [
        "<ResearchCompression>",
        f"Topic: {_truncate_text(str(research_topic or ''), 240)}",
        (
            "Source mix: local_queries={local} external_queries={external} article_sources={article} external_sources={external_sources}".format(
                local=mix.get("article_queries", 0),
                external=mix.get("external_queries", 0),
                article=mix.get("article_sources", 0),
                external_sources=mix.get("external_sources", 0),
            )
        ),
        f"Status counts: supported={len(supported)} unresolved={len(unresolved)} conflicted={len(conflicted)}",
    ]
    if supported:
        deterministic_lines.append("Supported:")
        deterministic_lines.extend(_entry_line(entry) for entry in supported[:6])
    if unresolved:
        deterministic_lines.append("Unresolved:")
        deterministic_lines.extend(_entry_line(entry) for entry in unresolved[:4])
    if conflicted:
        deterministic_lines.append("Conflicted:")
        deterministic_lines.extend(_entry_line(entry) for entry in conflicted[:3])
    deterministic_lines.append("</ResearchCompression>")
    deterministic_summary = "\n".join(deterministic_lines)

    compressed_output = deterministic_summary
    use_llm_compression = str(os.environ.get("DEEP_RESEARCH_STORM_LLM_COMPRESSION", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if use_llm_compression:
        logger.info("  LLM compression enabled via DEEP_RESEARCH_STORM_LLM_COMPRESSION")
        perspectives_with_plans_parts = []
        for i, perspective_name in enumerate(perspectives, 1):
            research_plan = perspective_research_plans.get(perspective_name, "No specific research plan")
            perspectives_with_plans_parts.append(
                f"**{i}. {perspective_name}**\n   Research Plan: {research_plan}"
            )
        perspectives_with_plans = "\n\n".join(perspectives_with_plans_parts)
        system_message = compress_research_system_prompt.format(date=get_today_str())
        human_message = compress_research_human_message.format(
            research_topic=research_topic,
            perspectives_with_plans=perspectives_with_plans,
        )
        llm_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Deterministic compression seed:\n{deterministic_summary}"),
            HumanMessage(content=human_message),
        ]
        llm_start = time.perf_counter()
        try:
            response = compress_model.invoke(llm_messages)
            llm_elapsed = time.perf_counter() - llm_start
            log_token_usage(logger, response, "compress_research")
            get_global_tracker().add_usage(response, "compress_research")
            compressed_output = str(response.content or deterministic_summary)
            logger.info("  LLM compression complete in %.2fs", llm_elapsed)
        except Exception as exc:
            logger.warning("  LLM compression failed, falling back to deterministic summary: %s", exc)

    raw_notes = [str(entry.get("answer", "") or "") for entry in evidence_ledger if entry.get("answer")]
    if not raw_notes:
        raw_notes = expert_responses.copy()
    raw_notes_chars = sum(len(n) for n in raw_notes)
    logger.info("  raw_notes: count=%d, total_chars=%d", len(raw_notes), raw_notes_chars)

    total_elapsed = time.perf_counter() - start_time
    logger.info("COMPRESS_RESEARCH COMPLETE | output_len=%d | time=%.2fs", len(compressed_output), total_elapsed)

    return {
        "compressed_research": compressed_output,
        "raw_notes": raw_notes,
        "evidence_ledger": evidence_ledger,
        "retrieval_events": retrieval_events,
        "observability_events": observability_events,
        "perspectives": perspectives,
        "perspective_research_plans": perspective_research_plans,
        "perspective_profiles": state.get("perspective_profiles", {}) or {},
        "researcher_messages": state.get("researcher_messages", []),
    }

# ===== Q&A REFLECTION AND RETRY =====

def qa_reflection(state: ResearcherState):
    """Reflect on the Q&A exchange and decide if a retry is needed.
    
    Evaluates the quality of the answer and determines if:
    1. The answer is sufficient → proceed to next question
    2. A retry with a different query would help → trigger retry (max 1)
    
    Can also update the research plan based on learnings.
    """
    current_perspective = state.get("current_perspective", "")
    last_question = state.get("last_question", "")
    last_answer = state.get("last_answer", "")
    last_evidence = state.get("last_evidence", []) or []
    is_retry_attempt = state.get("is_retry_attempt", False)
    current_retrieval_plan = state.get("current_retrieval_plan", {}) or {}
    last_retrieval_metadata = state.get("last_retrieval_metadata", {}) or {}
    perspective_research_plans = state.get("perspective_research_plans", {})
    research_plan = perspective_research_plans.get(current_perspective, "")
    perspective_profile = _get_perspective_profile(state, current_perspective)
    
    start_time = time.perf_counter()
    
    logger.info("="*60)
    logger.info("QA_REFLECTION START | perspective='%s' | is_retry=%s", 
               current_perspective[:50], is_retry_attempt)
    logger.info("  last_question: %s", last_question[:100] if last_question else "(empty)")
    logger.info("  last_answer_len: %d chars", len(last_answer))
    
    # If this was already a retry attempt, we need to synthesize both answers
    if is_retry_attempt:
        logger.info("QA_REFLECTION: This was a retry attempt -> route to answer_synthesis")
        return {
            "is_retry_attempt": False,  # Reset for next Q&A
            "retry_tool_name": "",
            "observability_events": [
                {
                    "category": "routing",
                    "node": "qa_reflection",
                    "event_key": f"{state.get('current_qa_id', '')}:qa_reflection:answer_synthesis",
                    "route": "answer_synthesis",
                    "reason": "completed_retry_attempt",
                }
            ],
            # Keep original_qa for synthesis node
        }
    
    # Skip reflection if no Q&A to evaluate
    if not last_question or not last_answer:
        logger.warning("QA_REFLECTION: No Q&A to evaluate, proceeding normally")
        return {
            "is_retry_attempt": False,
            "original_qa": {},
            "retry_query": "",
            "retry_tool_name": "",
            "current_retrieval_plan": {},
            "last_retrieval_metadata": {},
            "last_evidence": [],
            "observability_events": [
                {
                    "category": "routing",
                    "node": "qa_reflection",
                    "event_key": f"{state.get('current_qa_id', '')}:qa_reflection:no_qa",
                    "route": "should_continue_conversation_node",
                    "reason": "missing_qa_payload",
                }
            ],
        }
    
    # Set up structured output model
    structured_model = reflection_model.with_structured_output(QAReflection)
    
    prompt = qa_reflection_prompt.format(
        perspective=current_perspective,
        perspective_profile=perspective_profile or "(not provided)",
        research_plan=research_plan,
        question=last_question,
        answer=last_answer,
        original_question=current_retrieval_plan.get("original_question", last_question),
        retrieval_query=current_retrieval_plan.get("retrieval_query", last_question),
        tool_name=last_retrieval_metadata.get("tool_name", current_retrieval_plan.get("tool_name", "")),
        question_scope=current_retrieval_plan.get("question_scope", "ambiguous"),
        scope_reason=current_retrieval_plan.get("scope_reason", "(none)"),
        query_quality=", ".join(current_retrieval_plan.get("query_quality_flags", []) or []) or "(none)",
        rewrite_reason=current_retrieval_plan.get("query_rewrite_reason", "") or "(none)",
        query_shape_reason=current_retrieval_plan.get("query_shape_reason", "") or "(none)",
        retrieval_status=last_retrieval_metadata.get("status", "(unknown)"),
        retrieval_best_score=last_retrieval_metadata.get("best_score", "(n/a)"),
        retrieval_matches=last_retrieval_metadata.get("matched_chunks", 0),
        fallback_status=last_retrieval_metadata.get("fallback_status", "(none)"),
    )
    
    llm_start = time.perf_counter()
    try:
        response = structured_model.invoke([HumanMessage(content=prompt)])
        llm_elapsed = time.perf_counter() - llm_start
        
        logger.info("QA_REFLECTION: quality='%s' | needs_retry=%s | time=%.2fs",
                   response.answer_quality, response.needs_retry, llm_elapsed)
        logger.info("  reasoning: %s", response.reasoning[:150] if response.reasoning else "(none)")
        
        if response.needs_retry:
            logger.info("  retry_query: %s", response.retry_query[:100] if response.retry_query else "(none)")
            logger.info("  suggested_tool: %s", response.suggested_tool[:60] if response.suggested_tool else "(same)")
            logger.info("  rewrite_reason: %s", response.rewrite_reason[:100] if response.rewrite_reason else "(none)")

        allow_retry, normalized_retry_query, retry_decision_reason = _coerce_retry_decision(
            answer_quality=response.answer_quality,
            needs_retry=bool(response.needs_retry),
            retry_query=response.retry_query,
            last_question=last_question,
            last_retrieval_metadata=last_retrieval_metadata,
            suggested_tool_name=response.suggested_tool,
            article_summary=state.get("article_summary", ""),
            retrieval_events=state.get("retrieval_events", []) or [],
            active_gap_id=state.get("active_gap_id", ""),
        )
        if response.needs_retry and not allow_retry:
            logger.info(
                "QA_REFLECTION: Retry suppressed by guardrail (%s)",
                retry_decision_reason,
            )
        
        # Build return state
        result = {
            "is_retry_attempt": False,
            "original_qa": {},
            "retry_query": "",
            "retry_tool_name": "",
            "current_retrieval_plan": {},
            "last_retrieval_metadata": {},
            "last_evidence": [],
            "retrieval_events": [
                {
                    **current_retrieval_plan,
                    "stage": "reflection",
                    "qa_id": state.get("current_qa_id", ""),
                    "event_key": f"{state.get('current_qa_id', '')}:reflection",
                    "tool_name": last_retrieval_metadata.get("tool_name", current_retrieval_plan.get("tool_name", "")),
                    "answer_quality": response.answer_quality,
                    "rewrite_reason": response.rewrite_reason,
                    "retry_recommended": bool(response.needs_retry),
                    "retry_decision_reason": retry_decision_reason,
                }
            ],
            "observability_events": [],
        }
        
        if allow_retry:
            current_tool_name = str(
                last_retrieval_metadata.get(
                    "tool_name",
                    current_retrieval_plan.get("tool_name", ""),
                )
                or ""
            ).strip()
            candidate_tool_name = (
                response.suggested_tool
                if response.suggested_tool in writer_tools_by_name
                else current_tool_name
            )
            query_flags = set(current_retrieval_plan.get("query_quality_flags", []) or [])
            retrieval_status = str(last_retrieval_metadata.get("status", "") or "")
            question_scope = str(current_retrieval_plan.get("question_scope", "") or "")
            has_alternate_path = (
                (candidate_tool_name and candidate_tool_name != current_tool_name)
                or retrieval_status in {"no_match", "empty", "error"}
                or bool(query_flags & {"corpus_misaligned", "article_internal_offload"})
                or question_scope in {"external_artifact", "external_context"}
            )
            if not has_alternate_path:
                allow_retry = False
                retry_decision_reason = "no_alternate_retrieval_path"

        if allow_retry:
            # Store original Q&A and set up retry
            result["is_retry_attempt"] = True  # Will be True for next writer_node call
            result["retry_query"] = normalized_retry_query
            if response.suggested_tool in writer_tools_by_name:
                result["retry_tool_name"] = response.suggested_tool
            result["original_qa"] = {
                "qa_id": state.get("current_qa_id", ""),
                "question": last_question,
                "answer": last_answer,
                "evidence": last_evidence,
                "retrieval_metadata": last_retrieval_metadata,
                "retrieval_plan": current_retrieval_plan,
            }
            result["observability_events"] = [
                {
                    "category": "routing",
                    "node": "qa_reflection",
                    "event_key": f"{state.get('current_qa_id', '')}:qa_reflection:retry",
                    "route": "writer_node",
                    "reason": retry_decision_reason,
                    "retry_query": normalized_retry_query,
                    "retry_tool_name": result.get("retry_tool_name", ""),
                }
            ]
            logger.info(
                "QA_REFLECTION: ⚡ Retry triggered with normalized query: %s",
                normalized_retry_query[:80],
            )
        else:
            answer_status, missing_reason = _infer_answer_status(
                last_answer,
                response.answer_quality,
                last_retrieval_metadata,
                last_evidence,
            )
            canonical_answer = canonicalize_answer_summary(
                last_answer,
                answer_status=answer_status,
            )
            result["expert_responses"] = [canonical_answer]
            result["evidence_ledger"] = [
                _build_ledger_entry(
                    state=state,
                    question=last_question,
                    answer=last_answer,
                    evidence=last_evidence,
                    answer_origin="single_pass",
                    answer_status=answer_status,
                    missing_reason=missing_reason,
                    review_note=retry_decision_reason if answer_status != "supported" else "",
                )
            ]
            result["observability_events"] = [
                {
                    "category": "routing",
                    "node": "qa_reflection",
                    "event_key": f"{state.get('current_qa_id', '')}:qa_reflection:continue",
                    "route": "should_continue_conversation_node",
                    "reason": retry_decision_reason,
                    "answer_status": answer_status,
                }
            ]
            logger.info("QA_REFLECTION: Finalized Q&A into evidence ledger (items=%d)", len(last_evidence))
        
        total_elapsed = time.perf_counter() - start_time
        logger.info("QA_REFLECTION COMPLETE | time=%.2fs", total_elapsed)
        
        return result
        
    except Exception as e:
        llm_elapsed = time.perf_counter() - llm_start
        logger.error("QA_REFLECTION ERROR after %.2fs: %s", llm_elapsed, e)
        logger.exception("Full traceback:")
        # On error, proceed without retry
        fallback_status, fallback_reason = _infer_answer_status(
            last_answer,
            "insufficient",
            last_retrieval_metadata,
            last_evidence,
        )
        return {
            "is_retry_attempt": False,
            "original_qa": {},
            "retry_query": "",
            "last_evidence": [],
            "expert_responses": [
                canonicalize_answer_summary(last_answer, answer_status=fallback_status)
            ]
            if last_answer
            else [],
            "evidence_ledger": [
                _build_ledger_entry(
                    state=state,
                    question=last_question,
                    answer=last_answer,
                    evidence=last_evidence,
                    answer_origin="single_pass",
                    answer_status=fallback_status,
                    missing_reason=fallback_reason,
                    review_note="qa_reflection_error",
                )
            ] if last_question and last_answer else [],
            "retry_tool_name": "",
            "current_retrieval_plan": {},
            "last_retrieval_metadata": {},
            "observability_events": [
                {
                    "category": "routing",
                    "node": "qa_reflection",
                    "event_key": f"{state.get('current_qa_id', '')}:qa_reflection:error",
                    "route": "should_continue_conversation_node",
                    "reason": "qa_reflection_error",
                }
            ],
        }


def answer_synthesis(state: ResearcherState):
    """Synthesize the original and retry answers into a final comprehensive answer.
    
    Called after a retry attempt to combine both answers.
    """
    current_perspective = state.get("current_perspective", "")
    original_qa = state.get("original_qa", {})
    last_question = state.get("last_question", "")
    last_answer = state.get("last_answer", "")  # This is the retry answer
    last_evidence = state.get("last_evidence", []) or []
    retry_query = state.get("retry_query", "")
    retry_metadata = state.get("last_retrieval_metadata", {}) or {}
    
    original_question = original_qa.get("question", "")
    original_answer = original_qa.get("answer", "")
    original_evidence = original_qa.get("evidence", []) or []
    original_metadata = original_qa.get("retrieval_metadata", {}) or {}
    
    start_time = time.perf_counter()
    
    logger.info("="*60)
    logger.info("ANSWER_SYNTHESIS START | perspective='%s'", current_perspective[:50])
    logger.info("  original_answer_len: %d chars", len(original_answer))
    logger.info("  retry_answer_len: %d chars", len(last_answer))
    
    # If no original Q&A (shouldn't happen), just return
    if not original_qa or not original_answer:
        logger.warning("ANSWER_SYNTHESIS: No original Q&A to synthesize, using retry answer only")
        if last_question and last_answer:
            answer_status, missing_reason = _infer_answer_status(
                last_answer,
                "insufficient",
                retry_metadata,
                last_evidence,
            )
            canonical_answer = canonicalize_answer_summary(
                last_answer,
                answer_status=answer_status,
            )
            return {
                "expert_responses": [canonical_answer],
                "evidence_ledger": [
                    _build_ledger_entry(
                        state=state,
                        question=last_question,
                        answer=last_answer,
                        evidence=last_evidence,
                        answer_origin="retry_only",
                        answer_status=answer_status,
                        missing_reason=missing_reason,
                    )
                ],
                "original_qa": {},
                "retry_query": "",
                "retry_tool_name": "",
                "current_retrieval_plan": {},
                "last_retrieval_metadata": {},
                "last_evidence": [],
                "observability_events": [
                    {
                        "category": "retry_comparison",
                        "node": "answer_synthesis",
                        "event_key": f"{state.get('current_qa_id', '')}:retry_only",
                        "decision": "retry_only_no_original_qa",
                    }
                ],
            }
        return {}

    comparison = _compare_retry_outcome(
        original_answer=original_answer,
        retry_answer=last_answer,
        original_metadata=original_metadata,
        retry_metadata=retry_metadata,
        original_evidence=original_evidence,
        retry_evidence=last_evidence,
    )
    logger.info(
        "ANSWER_SYNTHESIS COMPARISON | accept_retry=%s | improved=%s | complementary=%s | regressed=%s | decision=%s",
        comparison["accept_retry"],
        comparison["improved"],
        comparison["complementary"],
        comparison["regressed"],
        comparison["decision_reason"],
    )

    retry_comparison_event = {
        "stage": "retry_comparison",
        "qa_id": state.get("current_qa_id", ""),
        "event_key": f"{state.get('current_qa_id', '')}:retry_comparison",
        "tool_name": retry_metadata.get("tool_name", ""),
        "retry_query": retry_query,
        **comparison,
    }
    retry_observability_event = {
        "category": "retry_comparison",
        "node": "answer_synthesis",
        "event_key": f"{state.get('current_qa_id', '')}:retry_comparison",
        "decision": comparison["decision_reason"],
        "accept_retry": comparison["accept_retry"],
        "improved": comparison["improved"],
        "regressed": comparison["regressed"],
        "complementary": comparison["complementary"],
    }

    if not comparison["accept_retry"]:
        answer_status, missing_reason = _infer_answer_status(
            original_answer,
            "sufficient",
            original_metadata,
            original_evidence,
        )
        canonical_answer = canonicalize_answer_summary(
            original_answer,
            answer_status=answer_status,
        )
        chosen_message = AIMessage(content=f"[Retry rejected; kept original] {original_answer}")
        return {
            "researcher_messages": [chosen_message],
            "perspective_messages": {current_perspective: [chosen_message]},
            "expert_responses": [canonical_answer],
            "evidence_ledger": [
                _build_ledger_entry(
                    state=state,
                    question=original_question or last_question,
                    answer=original_answer,
                    evidence=original_evidence,
                    answer_origin="retry_rejected_original_kept",
                    answer_status=answer_status,
                    missing_reason=missing_reason,
                    review_note=comparison["decision_reason"],
                )
            ],
            "last_answer": original_answer,
            "original_qa": {},
            "retry_query": "",
            "retry_tool_name": "",
            "current_retrieval_plan": {},
            "last_retrieval_metadata": {},
            "last_evidence": [],
            "retrieval_events": [retry_comparison_event],
            "observability_events": [retry_observability_event],
        }

    merged_evidence = _merge_evidence_items(original_evidence, last_evidence)
    if comparison["prefer_retry_only"]:
        synthesized_answer = last_answer
        answer_origin = "retry_accepted_retry_only"
        synthesized_message = AIMessage(content=f"[Retry accepted] {synthesized_answer}")
    else:
        prompt = answer_synthesis_prompt.format(
            original_question=original_question,
            original_answer=original_answer,
            retry_query=retry_query,
            retry_answer=last_answer
        )

        llm_start = time.perf_counter()
        try:
            response = synthesis_model.invoke([HumanMessage(content=prompt)])
            llm_elapsed = time.perf_counter() - llm_start

            synthesized_answer = response.content if response.content else last_answer

            log_token_usage(logger, response, "answer_synthesis")
            get_global_tracker().add_usage(response, "answer_synthesis")

            logger.info("ANSWER_SYNTHESIS COMPLETE | synthesized_len=%d | time=%.2fs",
                       len(synthesized_answer), llm_elapsed)
            synthesized_message = AIMessage(content=f"[Synthesized from retry] {synthesized_answer}")
            answer_origin = "retry_synthesized"

        except Exception as e:
            llm_elapsed = time.perf_counter() - llm_start
            logger.error("ANSWER_SYNTHESIS ERROR after %.2fs: %s", llm_elapsed, e)
            logger.exception("Full traceback:")
            synthesized_answer = last_answer if comparison["improved"] else original_answer
            synthesized_message = AIMessage(content=f"[Retry fallback] {synthesized_answer}")
            answer_origin = "retry_synthesis_error_fallback"

    answer_status, missing_reason = _infer_answer_status(
        synthesized_answer,
        "sufficient",
        retry_metadata if answer_origin != "retry_rejected_original_kept" else original_metadata,
        merged_evidence,
    )
    canonical_answer = canonicalize_answer_summary(
        synthesized_answer,
        answer_status=answer_status,
    )
    if comparison["materially_divergent"] and answer_status == "supported":
        answer_status = "conflicted"
    review_note = ""
    if comparison["materially_divergent"]:
        review_note = "Original and retry answers diverged; merged answer retained for review."

    total_elapsed = time.perf_counter() - start_time
    logger.info("  total_time: %.2fs", total_elapsed)

    return {
        "researcher_messages": [synthesized_message],
        "perspective_messages": {current_perspective: [synthesized_message]},
        "expert_responses": [canonical_answer],
        "evidence_ledger": [
            _build_ledger_entry(
                state=state,
                question=original_question or last_question,
                answer=synthesized_answer,
                evidence=merged_evidence,
                answer_origin=answer_origin,
                answer_status=answer_status,
                missing_reason=missing_reason,
                review_note=review_note or comparison["decision_reason"],
            )
        ],
        "last_answer": synthesized_answer,
        "original_qa": {},
        "retry_query": "",
        "retry_tool_name": "",
        "current_retrieval_plan": {},
        "last_retrieval_metadata": {},
        "last_evidence": [],
        "retrieval_events": [retry_comparison_event],
        "observability_events": [retry_observability_event],
    }


# ===== ROUTING LOGIC =====

def should_continue_writer(state: ResearcherState) -> Literal["writer_tool_node", "expert_synthesize"]:
    """Determine if Writer should execute tools or go to Expert for synthesis."""
    messages = state.get("researcher_messages", [])
    if not messages:
        logger.warning("SHOULD_CONTINUE_WRITER: No messages, defaulting to expert_synthesize")
        return "expert_synthesize"
    
    last_message = messages[-1]
    tool_calls = getattr(last_message, 'tool_calls', None)
    has_tool_calls = bool(tool_calls)
    
    if has_tool_calls:
        tool_names = [tc.get('name', '?') for tc in tool_calls]
        logger.info("SHOULD_CONTINUE_WRITER: has_tool_calls=True -> writer_tool_node (tools: %s)", tool_names)
        return "writer_tool_node"
    else:
        logger.info("SHOULD_CONTINUE_WRITER: has_tool_calls=False -> expert_synthesize")
        return "expert_synthesize"

def route_after_reflection(state: ResearcherState) -> Literal["writer_node", "answer_synthesis", "should_continue_conversation_node"]:
    """Route after Q&A reflection - decide if retry, synthesis, or normal continuation.
    
    Returns:
        - "writer_node": If retry is needed (is_retry_attempt was set to True)
        - "answer_synthesis": If this was a retry attempt and we need to synthesize
        - "should_continue_conversation_node": Normal flow - proceed to next question/perspective
    """
    is_retry_attempt = state.get("is_retry_attempt", False)
    original_qa = state.get("original_qa", {})
    
    logger.info("ROUTE_AFTER_REFLECTION: is_retry_attempt=%s | has_original_qa=%s",
               is_retry_attempt, bool(original_qa))
    
    # If reflection triggered a retry (is_retry_attempt was just set to True)
    if is_retry_attempt:
        logger.info("ROUTE_AFTER_REFLECTION: ⚡ Retry triggered -> writer_node")
        return "writer_node"
    
    # If we have original_qa, this means we just finished a retry and need to synthesize
    # (is_retry_attempt was reset to False by qa_reflection after detecting retry attempt)
    if original_qa and original_qa.get("answer"):
        logger.info("ROUTE_AFTER_REFLECTION: Post-retry -> answer_synthesis")
        return "answer_synthesis"
    
    # Normal flow - no retry needed
    logger.info("ROUTE_AFTER_REFLECTION: Normal flow -> should_continue_conversation_node")
    return "should_continue_conversation_node"


def _determine_conversation_route(state: ResearcherState) -> tuple[str, str]:
    should_continue = state.get("should_continue_conversation", False)
    conversation_round = state.get("conversation_round", 0)
    perspectives = state.get("perspectives", [])
    current_perspective = state.get("current_perspective", "")

    if conversation_round >= MAX_CONVERSATION_ROUNDS:
        return "next_perspective", "max_rounds_reached"
    if should_continue and conversation_round < MAX_CONVERSATION_ROUNDS:
        return "writer_node", "writer_requested_more_questions"
    current_idx = perspectives.index(current_perspective) if current_perspective in perspectives else -1
    if current_idx >= 0 and current_idx < len(perspectives) - 1:
        return "next_perspective", "move_to_next_perspective"
    return "compress_research", "all_perspectives_complete"


def should_continue_conversation_node(state: ResearcherState):
    """Node wrapper for should_continue_conversation routing.
    
    This node exists to provide a clean routing point after qa_reflection
    or answer_synthesis before continuing to the next question/perspective.
    """
    route, reason = _determine_conversation_route(state)
    logger.info("SHOULD_CONTINUE_CONVERSATION_NODE: prepared route=%s reason=%s", route, reason)
    return {
        "conversation_route_reason": reason,
        "observability_events": [
            {
                "category": "routing",
                "node": "should_continue_conversation_node",
                "event_key": f"{state.get('current_qa_id', '')}:conversation_route",
                "route": route,
                "reason": reason,
            }
        ],
    }


def should_continue_conversation(state: ResearcherState) -> Literal["writer_node", "next_perspective", "compress_research"]:
    """Determine if conversation should continue, move to next perspective, or compress."""
    should_continue = state.get("should_continue_conversation", False)
    conversation_round = state.get("conversation_round", 0)
    perspectives = state.get("perspectives", [])
    current_perspective = state.get("current_perspective", "")
    
    logger.info("SHOULD_CONTINUE_CONVERSATION | round=%d/%d | should_continue=%s | current_perspective_idx=%d/%d",
               conversation_round, MAX_CONVERSATION_ROUNDS, should_continue,
               perspectives.index(current_perspective) if current_perspective in perspectives else -1,
               len(perspectives))
    route, reason = _determine_conversation_route(state)
    logger.info("SHOULD_CONTINUE_CONVERSATION: route=%s reason=%s", route, reason)
    return route  # type: ignore[return-value]

def next_perspective(state: ResearcherState):
    """Move to the next perspective."""
    perspectives = state.get("perspectives", [])
    current_perspective = state.get("current_perspective", "")
    
    current_idx = perspectives.index(current_perspective) if current_perspective in perspectives else -1
    next_idx = current_idx + 1
    
    if next_idx < len(perspectives):
        next_perspective_name = perspectives[next_idx]
        logger.info("NEXT_PERSPECTIVE: moving from '%s' (idx=%d) to '%s' (idx=%d)",
                   current_perspective[:50], current_idx, next_perspective_name[:50], next_idx)
        return {
            "current_perspective": next_perspective_name,
            "conversation_round": 0,  # Reset conversation round for new perspective
            "is_retry_attempt": False,  # Reset retry state for new perspective
            "retry_query": "",
            "retry_tool_name": "",
            "original_qa": {},
            "current_qa_id": "",
            "current_retrieval_plan": {},
            "last_retrieval_metadata": {},
            "last_evidence": [],
            "conversation_route_reason": "switched_perspective",
            "observability_events": [
                {
                    "category": "routing",
                    "node": "next_perspective",
                    "event_key": f"{next_perspective_name}:entered",
                    "route": "writer_node",
                    "reason": "advance_to_next_perspective",
                }
            ],
            "researcher_messages": [
                SystemMessage(content=f"Switching to perspective: {next_perspective_name}")
            ]
        }
    else:
        # No more perspectives - clear current_perspective so route_after_next_perspective routes to compress_research
        logger.info("NEXT_PERSPECTIVE: no more perspectives (current_idx=%d, total=%d) - clearing current_perspective",
                   current_idx, len(perspectives))
        return {
            "current_perspective": "",  # Clear to signal we're done with all perspectives
            "conversation_round": 0,  # Reset for clarity
            "is_retry_attempt": False,
            "retry_query": "",
            "retry_tool_name": "",
            "original_qa": {},
            "current_qa_id": "",
            "current_retrieval_plan": {},
            "last_retrieval_metadata": {},
            "last_evidence": [],
            "conversation_route_reason": "all_perspectives_complete",
            "observability_events": [
                {
                    "category": "routing",
                    "node": "next_perspective",
                    "event_key": "all_perspectives_complete",
                    "route": "compress_research",
                    "reason": "no_more_perspectives",
                }
            ],
        }

# ===== GRAPH CONSTRUCTION =====

# Build the STORM agent workflow
storm_agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes
storm_agent_builder.add_node("perspective_discovery", perspective_discovery)
storm_agent_builder.add_node("generate_research_plans", generate_research_plans)
storm_agent_builder.add_node("writer_node", writer_node)
storm_agent_builder.add_node("writer_tool_node", writer_tool_node)
storm_agent_builder.add_node("expert_synthesize", expert_synthesize)
storm_agent_builder.add_node("qa_reflection", qa_reflection)
storm_agent_builder.add_node("answer_synthesis", answer_synthesis)
storm_agent_builder.add_node("should_continue_conversation_node", should_continue_conversation_node)
storm_agent_builder.add_node("next_perspective", next_perspective)
storm_agent_builder.add_node("compress_research", compress_research)

# Add edges
# Flow: perspective_discovery -> generate_research_plans -> writer_node -> ...
storm_agent_builder.add_edge(START, "perspective_discovery")
storm_agent_builder.add_edge("perspective_discovery", "generate_research_plans")
storm_agent_builder.add_edge("generate_research_plans", "writer_node")
storm_agent_builder.add_conditional_edges(
    "writer_node",
    should_continue_writer,
    {
        "writer_tool_node": "writer_tool_node",
        "expert_synthesize": "expert_synthesize",
    },
)
storm_agent_builder.add_edge("writer_tool_node", "expert_synthesize")

# After expert_synthesize, go to qa_reflection for evaluation
storm_agent_builder.add_edge("expert_synthesize", "qa_reflection")

# qa_reflection routes to: retry (writer_node), synthesis (answer_synthesis), or continue
storm_agent_builder.add_conditional_edges(
    "qa_reflection",
    route_after_reflection,
    {
        "writer_node": "writer_node",  # Retry with new query
        "answer_synthesis": "answer_synthesis",  # Synthesize original + retry answers
        "should_continue_conversation_node": "should_continue_conversation_node",  # Normal continuation
    },
)

# After answer_synthesis, proceed to normal continuation
storm_agent_builder.add_edge("answer_synthesis", "should_continue_conversation_node")

# should_continue_conversation_node routes to next question, next perspective, or compress
storm_agent_builder.add_conditional_edges(
    "should_continue_conversation_node",
    should_continue_conversation,
    {
        "writer_node": "writer_node",
        "next_perspective": "next_perspective",
        "compress_research": "compress_research",
    },
)
def route_after_next_perspective(state: ResearcherState) -> Literal["writer_node", "compress_research"]:
    """Route after moving to next perspective."""
    current_perspective = state.get("current_perspective", "")
    perspectives = state.get("perspectives", [])
    conversation_round = state.get("conversation_round", 0)
    
    logger.info("ROUTE_AFTER_NEXT_PERSPECTIVE: current_perspective='%s' | conversation_round=%d | perspectives=%d",
               current_perspective[:100] if current_perspective else "(empty)",
               conversation_round, len(perspectives))
    
    if current_perspective:
        # We have a perspective to work on
        logger.info("ROUTE_AFTER_NEXT_PERSPECTIVE: ✅ has_perspective -> writer_node")
        return "writer_node"
    else:
        # No more perspectives - all done, compress research
        logger.info("ROUTE_AFTER_NEXT_PERSPECTIVE: ✅ no_perspective -> compress_research")
        return "compress_research"

storm_agent_builder.add_conditional_edges(
    "next_perspective",
    route_after_next_perspective,
    {
        "writer_node": "writer_node",
        "compress_research": "compress_research",
    },
)
storm_agent_builder.add_edge("compress_research", END)

# Compile the agent
storm_researcher_agent = storm_agent_builder.compile()
