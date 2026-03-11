
"""Multi-agent supervisor for coordinating research across multiple specialized agents.

This module implements a supervisor pattern where:
1. A supervisor agent coordinates research activities and delegates tasks
2. Multiple researcher agents work on specific sub-topics independently
3. Results are aggregated and compressed for final reporting

The supervisor uses parallel research execution to improve efficiency while
maintaining isolated context windows for each research topic.
"""

import asyncio
import os
import re
import time
from typing_extensions import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage, 
    BaseMessage, 
    SystemMessage, 
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from src.model_config import (
    PIPELINE_MODEL_SETTINGS,
    build_chat_model,
)
from src.prompts import lead_researcher_with_multiple_steps_diffusion_double_check_prompt
from src.research_agent_storm import storm_researcher_agent as researcher_agent
from src.state_multi_agent_supervisor import (
    SupervisorState, 
    ConductResearch,
    ResearchComplete
)
from src.templates import get_research_guidance
from src.utils import get_today_str, think_tool, refine_draft_report
from src.logging_config import get_logger, log_token_usage, get_global_tracker
from src.evidence_utils import (
    canonicalize_evidence_ledger,
    compute_evidence_novelty,
    compute_research_round_impact,
    evidence_item_signature,
    ledger_entry_signature,
    retrieval_event_signature,
    source_mix_report,
)

# ===== LOGGING SETUP =====

logger = get_logger("deep_research.supervisor")

def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


_CONTEXT_STOPWORDS = {
    "about",
    "after",
    "against",
    "article",
    "context",
    "draft",
    "external",
    "find",
    "from",
    "guidance",
    "have",
    "into",
    "leaders",
    "more",
    "pharmacy",
    "research",
    "should",
    "that",
    "their",
    "these",
    "they",
    "this",
    "topic",
    "what",
    "which",
    "with",
}


def _keyword_set(text: str, min_len: int = 4) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9/_-]*", str(text or "").lower())
        if len(token) >= min_len and token not in _CONTEXT_STOPWORDS and not token.isdigit()
    }


def _split_markdown_sections(markdown: str) -> list[tuple[str, str]]:
    text = str(markdown or "").strip()
    if not text:
        return []

    sections: list[tuple[str, str]] = []
    current_header = "(preamble)"
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = line.strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_header, "\n".join(current_lines).strip()))

    return [(header, content) for header, content in sections if content.strip()]


def _build_targeted_draft_context(
    research_topic: str,
    draft_report: str,
    max_chars: int,
) -> tuple[str, list[str]]:
    """Keep only the most relevant draft sections for the current research gap."""
    draft_text = str(draft_report or "").strip()
    if not draft_text:
        return "", []
    if max_chars <= 0:
        return draft_text, ["full_draft"]

    sections = _split_markdown_sections(draft_text)
    if len(sections) <= 1:
        return _truncate(draft_text, max_chars), ["full_draft"]

    topic_tokens = _keyword_set(research_topic, min_len=3)
    if not topic_tokens:
        return _truncate(draft_text, max_chars), ["full_draft"]

    ranked_sections: list[tuple[int, int, str, str]] = []
    for idx, (header, content) in enumerate(sections):
        header_overlap = len(topic_tokens & _keyword_set(header, min_len=3))
        body_overlap = len(topic_tokens & _keyword_set(content))
        score = header_overlap * 3 + body_overlap
        ranked_sections.append((score, idx, header, content))

    matched = [item for item in ranked_sections if item[0] > 0]
    if not matched:
        return _truncate(draft_text, max_chars), ["full_draft"]

    selected_indices = {0}
    for _, idx, _, _ in sorted(matched, key=lambda item: (item[0], -item[1]), reverse=True)[:2]:
        selected_indices.add(idx)

    selected_headers = [sections[idx][0] for idx in sorted(selected_indices)]
    assembled_sections: list[str] = []
    remaining_chars = max_chars

    for idx in sorted(selected_indices):
        section_text = sections[idx][1]
        separator_cost = 2 if assembled_sections else 0
        if remaining_chars <= separator_cost:
            break
        remaining_chars -= separator_cost
        if len(section_text) > remaining_chars:
            section_text = _truncate(section_text, remaining_chars)
        assembled_sections.append(section_text)
        remaining_chars -= len(section_text)
        if remaining_chars <= 0:
            break

    targeted_context = "\n\n".join(assembled_sections).strip()
    if not targeted_context:
        return _truncate(draft_text, max_chars), ["full_draft"]

    return targeted_context, selected_headers

def _pending_findings_since_last_refine(
    supervisor_messages: list[BaseMessage],
    max_items: int = 2,
    max_chars_each: int = 8000,
) -> str:
    """Return ConductResearch ToolMessage contents since last refine_draft_report.

    Returned as plain text for safe, compact supervisor context (no tool message pairing).
    """
    last_refine_idx = -1
    for idx, msg in enumerate(supervisor_messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "refine_draft_report":
            last_refine_idx = idx

    pending: list[str] = []
    for idx, msg in enumerate(supervisor_messages):
        if idx <= last_refine_idx:
            continue
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "ConductResearch":
            content = str(getattr(msg, "content", "") or "")
            if content.strip():
                pending.append(_truncate(content, max_chars_each))

    if not pending:
        return ""
    return "\n\n".join(pending[-max_items:])


def _pending_round_count(state: SupervisorState) -> int:
    round_summaries = state.get("research_round_summaries", []) or []
    merged_round_count = int(state.get("merged_round_count", 0) or 0)
    return max(0, len(round_summaries) - merged_round_count)


def _has_pending_round_summaries(state: SupervisorState) -> bool:
    return _pending_round_count(state) > 0


def _make_tool_call(name: str, args: dict | None = None, call_id: str = "") -> dict:
    return {
        "id": call_id or f"auto_{name.lower()}",
        "name": name,
        "args": args or {},
    }


GAP_STAGNATION_THRESHOLD = int(
    os.environ.get("DEEP_RESEARCH_GAP_STAGNATION_THRESHOLD", "2")
)
_PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
_CLOSED_GAP_STATUSES = {"supported", "not_in_source", "conflicted", "deferred_low_value"}


def _normalize_gap_id(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    if not slug:
        return fallback
    return slug[:72]


def _normalize_gap_status(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"missing", "not_specified", "not_found"}:
        return "open"
    if normalized in {"needs_review"}:
        return "open"
    if normalized in {"supported", "not_in_source", "conflicted", "deferred_low_value", "open"}:
        return normalized
    return "open"


def _status_is_closed(status: str) -> bool:
    return _normalize_gap_status(status) in _CLOSED_GAP_STATUSES


def _merge_gap_cards(*groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for raw in group or []:
            card = dict(raw or {})
            gap_id = str(card.get("gap_id", "") or "").strip()
            if not gap_id or gap_id in seen:
                continue
            seen.add(gap_id)
            merged.append(card)
    return merged


def _build_seed_gap_cards(state: SupervisorState) -> list[dict]:
    cards: list[dict] = []
    existing_cards = [dict(item or {}) for item in (state.get("gap_cards", []) or [])]
    if existing_cards:
        return []
    existing_entries = canonicalize_evidence_ledger(state.get("evidence_ledger", []) or [])

    if not existing_entries:
        focus_text = _extract_focus_text(state, max_chars=700)
        cards.append(
            {
                "gap_id": "article_anchor_core_claims",
                "priority": "high",
                "question_intent": (
                    "Anchor the newsletter in the article with decisive quantitative findings and explicit limitations."
                ),
                "closure_criteria": (
                    "Extract concrete findings from the supplied article or explicitly state not_in_source."
                ),
                "preferred_search_type": "internal",
                "preferred_perspectives": [],
                "context_hint": focus_text,
            }
        )
        return cards

    for idx, entry in enumerate(existing_entries, start=1):
        status = _normalize_gap_status(str(entry.get("answer_status", "") or "open"))
        if status != "open":
            continue
        question = str(entry.get("question", "") or "").strip()
        if not question:
            continue
        gap_id = _normalize_gap_id(question, f"gap_{idx}")
        previous_mode = str(entry.get("search_type", "both") or "both").lower()
        preferred_search_type = "external" if previous_mode == "internal" else "internal"
        cards.append(
            {
                "gap_id": gap_id,
                "priority": "high",
                "question_intent": question,
                "closure_criteria": "Resolve this gap with direct evidence or mark not_in_source.",
                "preferred_search_type": preferred_search_type,
                "preferred_perspectives": [str(entry.get("perspective", "") or "").strip()],
            }
        )

    if cards:
        return cards

    if int(state.get("internal_rounds", 0) or 0) > 0 and int(state.get("external_rounds", 0) or 0) == 0:
        cards.append(
            {
                "gap_id": "external_context_validation",
                "priority": "medium",
                "question_intent": (
                    "Add compact external grounding that materially changes interpretation, trust, or actionability."
                ),
                "closure_criteria": "Identify 1-2 external context points or mark deferred_low_value.",
                "preferred_search_type": "external",
                "preferred_perspectives": [],
            }
        )

    return cards


def _select_gap_perspectives(gap_card: dict, state: SupervisorState, max_items: int = 1) -> list[str]:
    persisted = [p for p in state.get("storm_perspectives", []) or [] if str(p).strip()]
    hinted = [p for p in gap_card.get("preferred_perspectives", []) or [] if str(p).strip()]
    if not persisted:
        if hinted:
            return hinted[:max_items]
        return ["Basic Facts Researcher"][:max_items]

    hinted_set = {h.lower() for h in hinted}
    if hinted_set:
        direct = [p for p in persisted if p.lower() in hinted_set]
        if direct:
            return direct[:max_items]

    gap_tokens = _keyword_set(gap_card.get("question_intent", ""), min_len=4)
    if not gap_tokens:
        return persisted[:max_items]

    ranked = sorted(
        persisted,
        key=lambda p: len(gap_tokens & _keyword_set(p, min_len=4)),
        reverse=True,
    )
    return ranked[:max_items]


def _resolve_gap_search_type(gap_card: dict, gap_ledger: list[dict]) -> str:
    preferred = str(gap_card.get("preferred_search_type", "both") or "both").lower()
    gap_id = str(gap_card.get("gap_id", "") or "")
    ledger_row = next((row for row in gap_ledger if str(row.get("gap_id", "")) == gap_id), {})
    non_progress_count = int(ledger_row.get("non_progress_count", 0) or 0)
    last_search_type = str(ledger_row.get("last_search_type", "") or "").lower()

    if preferred not in {"internal", "external", "both"}:
        preferred = "both"

    if preferred == "both":
        if non_progress_count > 0 and last_search_type in {"internal", "external"}:
            return "external" if last_search_type == "internal" else "internal"
        return "external"

    if non_progress_count > 0:
        return "external" if preferred == "internal" else "internal"

    return preferred


def _upsert_gap_ledger_entry(
    gap_ledger: list[dict],
    *,
    gap_id: str,
    search_type: str,
    reason: str,
) -> list[dict]:
    updated: list[dict] = []
    found = False
    for row in gap_ledger or []:
        item = dict(row or {})
        if str(item.get("gap_id", "")) == gap_id:
            found = True
            item["attempt_count"] = int(item.get("attempt_count", 0) or 0) + 1
            item["last_search_type"] = search_type
            item["last_reason"] = reason
        updated.append(item)

    if not found:
        updated.append(
            {
                "gap_id": gap_id,
                "status": "open",
                "attempt_count": 1,
                "non_progress_count": 0,
                "last_search_type": search_type,
                "last_reason": reason,
            }
        )
    return updated


def _apply_gap_round_result(
    gap_ledger: list[dict],
    *,
    gap_id: str,
    impact: dict,
) -> list[dict]:
    if not gap_id:
        return gap_ledger

    status_counts = impact.get("status_counts", {}) or {}
    material_improvement = bool(impact.get("material_improvement", False))
    non_material_reason = str(impact.get("non_material_reason", "") or "")
    updated: list[dict] = []

    for row in gap_ledger or []:
        item = dict(row or {})
        if str(item.get("gap_id", "")) != gap_id:
            updated.append(item)
            continue

        non_progress_count = int(item.get("non_progress_count", 0) or 0)
        if material_improvement:
            if int(status_counts.get("conflicted", 0)) > 0:
                item["status"] = "conflicted"
                item["last_reason"] = "conflicting_evidence_found"
            elif int(status_counts.get("supported", 0)) > 0:
                item["status"] = "supported"
                item["last_reason"] = "evidence_supports_gap_closure"
            elif int(status_counts.get("needs_review", 0)) > 0:
                item["status"] = "deferred_low_value"
                item["last_reason"] = "needs_review_deferred"
            else:
                item["status"] = "open"
            item["non_progress_count"] = 0
        else:
            non_progress_count += 1
            item["non_progress_count"] = non_progress_count
            item["last_reason"] = non_material_reason or "no_material_change"
            if non_progress_count >= GAP_STAGNATION_THRESHOLD:
                item["status"] = "not_in_source"
                item["last_reason"] = (
                    f"stagnated_after_{GAP_STAGNATION_THRESHOLD}_attempts"
                )
        updated.append(item)

    return updated


def _build_gap_router_response(
    state: SupervisorState,
) -> tuple[AIMessage, str, str, list[dict], list[dict]]:
    gap_cards = _merge_gap_cards(
        state.get("gap_cards", []) or [],
        _build_seed_gap_cards(state),
    )
    gap_ledger = [dict(item or {}) for item in (state.get("gap_ledger", []) or [])]

    if _has_pending_round_summaries(state):
        return (
            AIMessage(
                content="Merge pending findings into the draft before requesting more research.",
                tool_calls=[_make_tool_call("refine_draft_report", call_id="auto_refine_pending_findings")],
            ),
            "merge_pending_findings",
            "pending_findings_not_merged",
            gap_cards,
            gap_ledger,
        )

    open_cards: list[dict] = []
    for card in gap_cards:
        gap_id = str(card.get("gap_id", "") or "").strip()
        if not gap_id:
            continue
        ledger_row = next((row for row in gap_ledger if str(row.get("gap_id", "")) == gap_id), {})
        status = _normalize_gap_status(str(ledger_row.get("status", "open") or "open"))
        if _status_is_closed(status):
            continue
        open_cards.append(card)

    if not open_cards:
        return (
            AIMessage(
                content="All priority gaps are closed or explicitly marked not_in_source/deferred. Finalize research.",
                tool_calls=[_make_tool_call("ResearchComplete", call_id="auto_research_complete")],
            ),
            "finalize",
            "gap_ledger_closed",
            gap_cards,
            gap_ledger,
        )

    selected_gap = sorted(
        open_cards,
        key=lambda card: (
            _PRIORITY_RANK.get(str(card.get("priority", "medium")).lower(), 1),
            int(
                next(
                    (
                        row.get("attempt_count", 0)
                        for row in gap_ledger
                        if str(row.get("gap_id", "")) == str(card.get("gap_id", ""))
                    ),
                    0,
                )
                or 0
            ),
        ),
    )[0]

    gap_id = str(selected_gap.get("gap_id", "") or "")
    search_type = _resolve_gap_search_type(selected_gap, gap_ledger)
    selected_perspectives = _select_gap_perspectives(selected_gap, state, max_items=1)

    gap_ledger = _upsert_gap_ledger_entry(
        gap_ledger,
        gap_id=gap_id,
        search_type=search_type,
        reason="router_selected_next_gap",
    )

    gap_statement = _truncate(str(selected_gap.get("question_intent", "") or ""), 360)
    closure_criteria = _truncate(str(selected_gap.get("closure_criteria", "") or ""), 260)
    focus_hint = _truncate(str(selected_gap.get("context_hint", "") or ""), 500)
    research_topic = (
        f"Gap: {gap_statement}\n"
        f"Why it matters: This is currently unresolved and blocks confident newsletter conclusions.\n"
        f"Closure criteria: {closure_criteria}\n"
        + (f"Current focus hint:\n{focus_hint}\n" if focus_hint else "")
        + "Return concise evidence that closes this exact gap or explicitly states not_in_source."
    )

    return (
        AIMessage(
            content=f"Research next unresolved gap: {gap_id}",
            tool_calls=[_make_tool_call(
                "ConductResearch",
                {
                    "research_topic": research_topic,
                    "search_type": search_type,
                    "gap_id": gap_id,
                    **({"perspectives": selected_perspectives} if selected_perspectives else {}),
                },
                call_id=f"auto_gap_{gap_id[:40] or 'unknown'}",
            )],
        ),
        "gap_router",
        "next_open_gap_selected",
        gap_cards,
        gap_ledger,
    )


def _extract_focus_text(state: SupervisorState, max_chars: int = 1400) -> str:
    draft_report = str(state.get("draft_report", "") or "").strip()
    if draft_report:
        draft_focus, _ = _build_targeted_draft_context(
            "main implication alert burden workflow governance best practices comparison external context",
            draft_report,
            max_chars,
        )
        if draft_focus.strip():
            return draft_focus

    article_summary = str(state.get("article_summary", "") or "").strip()
    if article_summary:
        return _truncate(article_summary, max_chars)

    research_brief = str(state.get("research_brief", "") or "").strip()
    return _truncate(research_brief, max_chars) if research_brief else "(none)"


def _build_initial_internal_research_topic(state: SupervisorState) -> str:
    focus_text = _extract_focus_text(state, max_chars=900)
    return (
        "Use only the supplied article. Close the highest-value factual gaps needed for the newsletter: "
        "exact article title/publication timing if stated, decisive quantitative findings, major limitations/caveats, "
        "and the operational implication a pharmacy leader should not miss. "
        "Prefer facts that materially affect interpretation, trust, or actionability over exhaustive extraction. "
        f"Current focus:\n{focus_text}\n"
        "Closure: a compact evidence-backed summary with explicit 'Not specified in text' for any still-missing high-priority fact."
    )


def _build_external_grounding_research_topic(state: SupervisorState) -> str:
    focus_text = _extract_focus_text(state, max_chars=1100)
    last_round = _truncate(str(state.get("last_round_impact_summary", "") or ""), 900)
    return (
        "Add purposeful external grounding for this article. Find 1-2 high-value external context points that materially improve "
        "how a pharmacy leader should interpret the article's main implication. "
        "Prioritize current guidelines, best practices, recent comparator studies, deployment/governance expectations, "
        "or operational literature most relevant to the article's core claim. "
        "Do not restate the article and do not broaden into generic web browsing. "
        f"Current draft focus:\n{focus_text}\n"
        f"Latest internal findings:\n{last_round or '(none)'}\n"
        "Closure: concise external grounding that confirms, qualifies, or reframes the article without overshadowing it."
    )


def _determine_supervisor_phase(state: SupervisorState) -> tuple[str, str]:
    storm_rounds = int(state.get("storm_rounds", 0) or 0)
    internal_rounds = int(state.get("internal_rounds", 0) or 0)
    external_rounds = int(state.get("external_rounds", 0) or 0)
    last_round_material_improvement = bool(state.get("last_round_material_improvement", False))

    if _has_pending_round_summaries(state):
        return "merge_pending_findings", "pending_findings_not_merged"
    if storm_rounds == 0:
        return "initial_internal_research", "no_research_rounds_yet"
    if internal_rounds > 0 and external_rounds == 0 and storm_rounds < MAX_STORM_ROUNDS:
        return "required_external_grounding", "external_context_missing"
    if storm_rounds >= MAX_STORM_ROUNDS:
        return "finalize", "research_budget_exhausted"
    if external_rounds > 0 and not last_round_material_improvement:
        return "finalize", "last_round_not_material"
    return "llm_decide", "optional_targeted_follow_up_or_finalize"


def _build_phase_response(state: SupervisorState, phase: str) -> AIMessage | None:
    if phase == "merge_pending_findings":
        return AIMessage(
            content="Merge pending findings into the draft before requesting more research.",
            tool_calls=[_make_tool_call("refine_draft_report", call_id="auto_refine_pending_findings")],
        )
    if phase == "initial_internal_research":
        return AIMessage(
            content="Run one compact internal extraction pass to anchor the newsletter in the article.",
            tool_calls=[
                _make_tool_call(
                    "ConductResearch",
                    {
                        "research_topic": _build_initial_internal_research_topic(state),
                        "search_type": "internal",
                    },
                    call_id="auto_initial_internal_research",
                )
            ],
        )
    if phase == "required_external_grounding":
        return AIMessage(
            content="Run one compact external grounding pass to place the article in broader professional context.",
            tool_calls=[
                _make_tool_call(
                    "ConductResearch",
                    {
                        "research_topic": _build_external_grounding_research_topic(state),
                        "search_type": "external",
                    },
                    call_id="auto_external_grounding_research",
                )
            ],
        )
    if phase == "finalize":
        return AIMessage(
            content="The current research is sufficient for synthesis. Finalize instead of opening another research burst.",
            tool_calls=[_make_tool_call("ResearchComplete", call_id="auto_research_complete")],
        )
    return None


def _format_novelty_history(history: list[dict], max_items: int = 3) -> str:
    if not history:
        return "(none)"

    parts: list[str] = []
    for item in history[-max_items:]:
        parts.append(
            "round={round} novel_entries={novel_entries}/{new_entries} novel_locators={novel_locators}/{new_locators} "
            "ratio={novelty_ratio} low_novelty={is_low_novelty} topic={topic}".format(
                round=item.get("storm_round", "?"),
                novel_entries=item.get("novel_entries", 0),
                new_entries=item.get("new_entries", 0),
                novel_locators=item.get("novel_locators", 0),
                new_locators=item.get("new_locators", 0),
                novelty_ratio=item.get("novelty_ratio", 0),
                is_low_novelty=item.get("is_low_novelty", False),
                topic=_truncate(str(item.get("topic", "") or ""), 160),
            )
        )
    return "\n".join(parts)


def _format_round_summary_history(history: list[dict], max_items: int = 3) -> str:
    if not history:
        return "(none)"

    parts: list[str] = []
    for item in history[-max_items:]:
        summary = _truncate(str(item.get("summary", "") or ""), 400)
        parts.append(
            "round={round} material_improvement={material_improvement} search_type={search_type} summary={summary}".format(
                round=item.get("storm_round", "?"),
                material_improvement=item.get("material_improvement", False),
                search_type=item.get("search_type", "both"),
                summary=summary,
            )
        )
    return "\n".join(parts)


def _pending_round_summaries(state: SupervisorState, max_items: int = 2, max_chars_each: int = 2000) -> str:
    round_summaries = state.get("research_round_summaries", []) or []
    merged_round_count = int(state.get("merged_round_count", 0) or 0)
    pending = [item for item in round_summaries[merged_round_count:] if isinstance(item, dict)]
    if not pending:
        return ""
    return _compile_pending_findings_packet(
        pending[-max_items:],
        max_rounds=max_items,
        max_chars=max_items * max_chars_each,
    )


def _format_source_mix_status(state: SupervisorState) -> str:
    summary = str(state.get("source_mix_summary", "") or "").strip()
    if not summary:
        return "(none yet)"
    rationale = str(state.get("external_grounding_rationale", "") or "").strip()
    if not rationale:
        return summary
    return f"{summary}\nRationale: {rationale}"


def _derive_external_grounding_rationale(
    *,
    mix: dict,
    external_grounding_considered: bool,
) -> str:
    if mix.get("external_queries", 0) or mix.get("external_sources", 0):
        return (
            f"External grounding was used with {mix.get('external_queries', 0)} external retrieval executions "
            f"and {mix.get('external_sources', 0)} external evidence sources."
        )
    if external_grounding_considered:
        return (
            "External grounding was considered but the executed questions stayed article-contained, "
            "so the run remains article-only."
        )
    if mix.get("article_queries", 0) or mix.get("article_sources", 0):
        return "All completed research rounds stayed article-contained, so no external grounding was used."
    return "No research evidence has been gathered yet."


def _format_round_examples(entries: list[dict], status: str, max_items: int = 2) -> list[str]:
    examples: list[str] = []
    for entry in entries:
        if str(entry.get("answer_status", "supported") or "supported") != status:
            continue
        question = _truncate(str(entry.get("question", "") or ""), 140)
        answer = _truncate(str(entry.get("answer", "") or ""), 260)
        if question and answer:
            examples.append(f"- {question}: {answer}")
        if len(examples) >= max_items:
            break
    return examples


def _entry_brief_claim(entry: dict, max_claim_chars: int = 240) -> dict:
    question = _truncate(str(entry.get("question", "") or ""), 180)
    answer = _truncate(str(entry.get("answer", "") or ""), max_claim_chars)
    evidence_refs = [
        _truncate(str(item.get("locator", "") or ""), 120)
        for item in (entry.get("evidence", []) or [])[:2]
        if str(item.get("locator", "") or "").strip()
    ]
    return {
        "question": question,
        "claim": answer,
        "evidence_refs": evidence_refs,
    }


def _build_round_claim_delta(
    entries: list[dict],
    *,
    max_supported: int = 3,
    max_unresolved: int = 3,
) -> dict:
    supported_claims: list[dict] = []
    unresolved_claims: list[dict] = []
    conflicted_claims: list[dict] = []

    for entry in entries:
        status = str(entry.get("answer_status", "supported") or "supported").strip().lower()
        claim = _entry_brief_claim(entry)
        if status == "supported":
            if len(supported_claims) < max_supported:
                supported_claims.append(claim)
            continue
        if status == "conflicted":
            if len(conflicted_claims) < max_unresolved:
                conflicted_claims.append(claim)
            continue
        if status in {"missing", "needs_review"}:
            if len(unresolved_claims) < max_unresolved:
                unresolved_claims.append(claim)

    return {
        "supported_claims": supported_claims,
        "unresolved_claims": unresolved_claims,
        "conflicted_claims": conflicted_claims,
    }


def _format_round_findings_packet(round_summary: dict) -> str:
    lines = [
        "<RoundDelta>",
        (
            "round={round} gap={gap_id} search_type={search_type} material_improvement={material}".format(
                round=round_summary.get("storm_round", "?"),
                gap_id=round_summary.get("gap_id", "(none)"),
                search_type=round_summary.get("search_type", "both"),
                material=bool(round_summary.get("material_improvement", False)),
            )
        ),
    ]
    status_counts = round_summary.get("status_counts", {}) or {}
    lines.append(
        "status_counts supported={supported} conflicted={conflicted} needs_review={needs_review} missing={missing}".format(
            supported=status_counts.get("supported", 0),
            conflicted=status_counts.get("conflicted", 0),
            needs_review=status_counts.get("needs_review", 0),
            missing=status_counts.get("missing", 0),
        )
    )

    for label, key in (
        ("SupportedClaims", "supported_claims"),
        ("UnresolvedClaims", "unresolved_claims"),
        ("ConflictedClaims", "conflicted_claims"),
    ):
        claims = round_summary.get(key, []) or []
        if not claims:
            continue
        lines.append(f"{label}:")
        for claim in claims:
            line = f"- Q: {_truncate(str(claim.get('question', '') or ''), 140)} | A: {_truncate(str(claim.get('claim', '') or ''), 220)}"
            refs = claim.get("evidence_refs", []) or []
            if refs:
                line += " | refs: " + ", ".join(_truncate(str(ref), 80) for ref in refs[:2])
            lines.append(line)

    lines.append("</RoundDelta>")
    return "\n".join(lines).strip()


def _compile_pending_findings_packet(
    round_summaries: list[dict],
    *,
    max_rounds: int = 3,
    max_chars: int = 4000,
) -> str:
    pending = [dict(item or {}) for item in (round_summaries or []) if isinstance(item, dict)]
    if not pending:
        return ""
    packets = [_format_round_findings_packet(item) for item in pending[-max_rounds:]]
    return _truncate("\n\n".join(packet for packet in packets if packet), max_chars)


def _diff_round_entries(existing_entries: list[dict], candidate_entries: list[dict]) -> list[dict]:
    """Return only the net-new or materially updated evidence entries for this round."""
    existing_canonical = canonicalize_evidence_ledger(existing_entries or [])
    candidate_canonical = canonicalize_evidence_ledger(candidate_entries or [])
    if not candidate_canonical:
        return []
    if not existing_canonical:
        return candidate_canonical

    existing_by_signature = {
        ledger_entry_signature(entry): dict(entry or {})
        for entry in existing_canonical
    }
    delta_entries: list[dict] = []
    for entry in candidate_canonical:
        signature = ledger_entry_signature(entry)
        existing = existing_by_signature.get(signature)
        if existing is None:
            delta_entries.append(entry)
            continue

        existing_evidence = {
            evidence_item_signature(item)
            for item in (existing.get("evidence", []) or [])
        }
        new_evidence = [
            dict(item or {})
            for item in (entry.get("evidence", []) or [])
            if evidence_item_signature(item) not in existing_evidence
        ]
        if new_evidence:
            updated = dict(entry or {})
            updated["evidence"] = new_evidence[:4]
            delta_entries.append(updated)

    return canonicalize_evidence_ledger(delta_entries)


def _diff_round_retrieval_events(existing_events: list[dict], candidate_events: list[dict]) -> list[dict]:
    existing_signatures = {
        retrieval_event_signature(event)
        for event in (existing_events or [])
    }
    delta = [
        dict(event or {})
        for event in (candidate_events or [])
        if retrieval_event_signature(event) not in existing_signatures
    ]
    return delta


def _build_round_summary(
    *,
    storm_round: int,
    gap_id: str,
    research_topic: str,
    search_type: str,
    new_entries: list[dict],
    impact: dict,
) -> dict:
    claim_delta = _build_round_claim_delta(new_entries)
    supported_examples = _format_round_examples(new_entries, "supported")
    conflicted_examples = _format_round_examples(new_entries, "conflicted", max_items=1)
    missing_examples = _format_round_examples(new_entries, "missing", max_items=1)

    if impact.get("material_improvement"):
        impact_line = (
            "Material improvement: yes"
            + (
                f" ({', '.join(impact.get('material_reasons', []))})"
                if impact.get("material_reasons")
                else ""
            )
        )
    else:
        impact_line = (
            "Material improvement: no"
            + (
                f" ({impact.get('non_material_reason', 'no_material_change')})"
                if impact.get("non_material_reason")
                else ""
            )
        )

    status_counts = impact.get("status_counts", {}) or {}
    source_mix = impact.get("source_mix", {}) or {}
    lines = [
        f"Round {storm_round}: {_truncate(research_topic, 220)}",
        f"Search type: {search_type}",
        impact_line,
        (
            "Status delta: supported={supported}, conflicted={conflicted}, needs_review={needs_review}, missing={missing}".format(
                supported=status_counts.get("supported", 0),
                conflicted=status_counts.get("conflicted", 0),
                needs_review=status_counts.get("needs_review", 0),
                missing=status_counts.get("missing", 0),
            )
        ),
        (
            "Source mix: local queries={local}, external queries={external}, article_only={article_only}".format(
                local=source_mix.get("article_queries", 0),
                external=source_mix.get("external_queries", 0),
                article_only=source_mix.get("article_only", True),
            )
        ),
    ]
    if supported_examples:
        lines.append("Key supported findings:")
        lines.extend(supported_examples)
    if conflicted_examples:
        lines.append("Needs review:")
        lines.extend(conflicted_examples)
    if missing_examples:
        lines.append("Still missing:")
        lines.extend(missing_examples)

    summary = {
        "storm_round": storm_round,
        "gap_id": gap_id,
        "research_topic": research_topic,
        "search_type": search_type,
        "material_improvement": bool(impact.get("material_improvement")),
        "material_reasons": impact.get("material_reasons", []),
        "non_material_reason": impact.get("non_material_reason", ""),
        "status_counts": status_counts,
        "source_mix": source_mix,
        "supported_claims": claim_delta.get("supported_claims", []),
        "unresolved_claims": claim_delta.get("unresolved_claims", []),
        "conflicted_claims": claim_delta.get("conflicted_claims", []),
        "summary": "\n".join(lines).strip(),
        "findings_packet": "",
    }
    summary["findings_packet"] = _format_round_findings_packet(summary)
    return summary

def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """Extract research notes from ConductResearch ToolMessages only.

    This function retrieves the compressed research findings that sub-agents
    return as ToolMessage content. Only extracts content from ConductResearch
    tool calls - excludes refine_draft_report (which contains full drafts)
    and other tools to prevent token explosion in final_report_generation.

    Args:
        messages: List of messages from supervisor's conversation history

    Returns:
        List of research note strings extracted from ConductResearch ToolMessages
    """
    return [
        tool_msg.content 
        for tool_msg in filter_messages(messages, include_types="tool")
        if getattr(tool_msg, 'name', '') == 'ConductResearch'
    ]

# Ensure async compatibility for Jupyter environments
try:
    import nest_asyncio
    # Only apply if running in Jupyter/IPython environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Not in Jupyter, no need for nest_asyncio
except ImportError:
    pass  # nest_asyncio not available, proceed without it


# ===== CONFIGURATION =====

supervisor_tools = [ConductResearch, ResearchComplete, think_tool,refine_draft_report]
supervisor_model = build_chat_model(PIPELINE_MODEL_SETTINGS.supervisor_orchestration_model)
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

# System constants
# Maximum number of tool call iterations for individual researcher agents
# This prevents infinite loops and controls research depth per topic
max_researcher_iterations = 15 # Calls to think_tool + ConductResearch + refine_draft_report

# Maximum number of STORM research rounds (ConductResearch calls that trigger STORM).
# This is a safety guard, not a primary convergence mechanism.
MAX_STORM_ROUNDS = int(os.environ.get("DEEP_RESEARCH_MAX_STORM_ROUNDS", "12"))

# Maximum number of draft editing rounds (refine_draft_report calls)
MAX_DRAFT_EDITING_ROUNDS = 3

# Maximum number of concurrent research agents the supervisor can launch
# This is passed to the lead_researcher_prompt to limit parallel research tasks
max_concurrent_researchers = 3

# ===== SUPERVISOR NODES =====

async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """Coordinate research activities.

    Analyzes the research brief and current progress to decide:
    - What research topics need investigation
    - Whether to conduct parallel research
    - When research is complete

    Args:
        state: Current supervisor state with messages and research progress

    Returns:
        Command to proceed to supervisor_tools node with updated state
    """
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    
    start_time = time.perf_counter()
    
    logger.info("="*80)
    logger.info("SUPERVISOR START | iteration=%d | message_count=%d", 
               research_iterations, len(supervisor_messages))
    
    # Calculate total content size for debugging
    total_content_size = sum(len(str(getattr(msg, 'content', ''))) for msg in supervisor_messages)
    logger.info("  total_content_size: %d chars (%.1f KB)", total_content_size, total_content_size/1024)
    
    # Log message summary
    for i, msg in enumerate(supervisor_messages):
        msg_type = type(msg).__name__
        content_len = len(str(getattr(msg, 'content', '')))
        tool_calls = getattr(msg, 'tool_calls', None)
        logger.info("  msg[%d]: %s content_len=%d tool_calls=%s", 
                   i, msg_type, content_len, 
                   len(tool_calls) if tool_calls else 0)

    # Keep template validation so downstream tools have required context.
    newsletter_template = state.get("newsletter_template")
    if not newsletter_template:
        raise ValueError("newsletter_template not set in state. Ensure select_newsletter_template runs before supervisor.")
    logger.info("  newsletter_template: %s", newsletter_template)

    response, supervisor_phase, phase_reason, gap_cards, gap_ledger = _build_gap_router_response(state)
    tool_calls = getattr(response, "tool_calls", [])
    logger.info(
        "SUPERVISOR: Lightweight gap router selected action | phase=%s | reason=%s | tool_calls=%d",
        supervisor_phase,
        phase_reason,
        len(tool_calls) if tool_calls else 0,
    )

    total_elapsed = time.perf_counter() - start_time
    logger.info("SUPERVISOR COMPLETE | total_time=%.2fs", total_elapsed)

    return Command(
        goto="supervisor_tools",
        update={
            # Only pass NEW messages - the add_messages reducer handles appending
            # to existing conversation history automatically
            "supervisor_messages": [response],
            "research_iterations": research_iterations + 1,
            "why_follow_up_before_merge": "",
            "supervisor_phase": supervisor_phase,
            "gap_cards": gap_cards,
            "gap_ledger": gap_ledger,
        }
    )

async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """Execute supervisor decisions - either conduct research or end the process.

    Handles:
    - Executing think_tool calls for strategic reflection
    - Launching parallel research agents for different topics
    - Aggregating research results
    - Determining when research is complete

    Args:
        state: Current supervisor state with messages and iteration count

    Returns:
        Command to continue supervision, end process, or handle errors
    """
    start_time = time.perf_counter()
    
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    newsletter_template = state.get("newsletter_template", "")
    storm_rounds = int(state.get("storm_rounds", 0) or 0)
    draft_editing_rounds = int(state.get("draft_editing_rounds", 0) or 0)
    
    logger.info("="*80)
    logger.info("SUPERVISOR_TOOLS START | iteration=%d | message_count=%d", 
               research_iterations, len(supervisor_messages))
    
    # The last message should be the AIMessage from supervisor() containing tool_calls
    most_recent_message = supervisor_messages[-1]
    
    tool_calls = getattr(most_recent_message, 'tool_calls', [])
    logger.info("  most_recent_message: %s | tool_calls=%d", 
               type(most_recent_message).__name__, 
               len(tool_calls) if tool_calls else 0)

    # Initialize variables for single return pattern
    tool_messages = []
    all_raw_notes = []
    all_evidence_ledger = []
    all_retrieval_events = []
    research_novelty_updates = []
    round_summary_updates = []
    round_note_updates = []
    observability_updates = []
    draft_report = ""
    next_step = "supervisor"  # Default next step
    should_end = False
    stop_reason = ""
    route_reason = "continue_supervision"
    last_round_impact_summary = str(state.get("last_round_impact_summary", "") or "")
    last_round_material_improvement = bool(state.get("last_round_material_improvement", False))
    source_mix_summary_text = str(state.get("source_mix_summary", "") or "")
    internal_rounds = int(state.get("internal_rounds", 0) or 0)
    external_rounds = int(state.get("external_rounds", 0) or 0)
    external_grounding_considered = bool(state.get("external_grounding_considered", False))
    external_grounding_rationale = str(state.get("external_grounding_rationale", "") or "")
    completion_status = str(state.get("supervisor_completion_status", "") or "")
    merged_round_count = int(state.get("merged_round_count", 0) or 0)
    supervisor_phase = str(state.get("supervisor_phase", "") or "")
    gap_cards = [dict(item or {}) for item in (state.get("gap_cards", []) or [])]
    gap_ledger = [dict(item or {}) for item in (state.get("gap_ledger", []) or [])]

    # Check exit criteria first
    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    ) if most_recent_message.tool_calls else False
    
    logger.info("  exit_check: exceeded_iterations=%s, no_tool_calls=%s, research_complete=%s",
               exceeded_iterations, no_tool_calls, research_complete)
    
    if exceeded_iterations:
        logger.warning("  ITERATION LIMIT REACHED: %d >= %d", research_iterations, max_researcher_iterations)

    if exceeded_iterations or no_tool_calls or research_complete:
        should_end = True
        next_step = END
        stop_reason = (
            "exceeded_iterations"
            if exceeded_iterations
            else "no_tool_calls"
            if no_tool_calls
            else "research_complete"
        )
        route_reason = stop_reason
        completion_status = (
            "stopped_due_to_limits"
            if exceeded_iterations
            else "completed_cleanly"
            if research_complete
            else "completed_without_explicit_research_complete"
        )
        logger.info("  ENDING: should_end=True")

    else:
        # Execute ALL tool calls before deciding next step
        try:
            # Separate think_tool calls from ConductResearch calls
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls 
                if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls 
                if tool_call["name"] == "ConductResearch"
            ]

            refine_report_calls = [
                tool_call for tool_call in most_recent_message.tool_calls 
                if tool_call["name"] == "refine_draft_report"
            ]
            
            logger.info("  tool_breakdown: think_tool=%d, ConductResearch=%d, refine_draft_report=%d",
                       len(think_tool_calls), len(conduct_research_calls), len(refine_report_calls))

            if conduct_research_calls and _has_pending_round_summaries(state) and not refine_report_calls:
                logger.info(
                    "  MERGE_GUARD: pending round summaries detected (%d). Inserting refine_draft_report before more research.",
                    _pending_round_count(state),
                )
                refine_report_calls = [
                    _make_tool_call("refine_draft_report", call_id="auto_refine_before_more_research")
                ]
                conduct_research_calls = []
                route_reason = "merge_guard_inserted_refine"
                observability_updates.append(
                    {
                        "category": "routing",
                        "node": "supervisor_tools",
                        "event_key": "supervisor_tools:merge_guard",
                        "route": "refine_draft_report",
                        "reason": "pending_findings_not_merged",
                    }
                )
            
            # Handle think_tool calls - just acknowledge without invoking
            # The reflection content is already captured in the AIMessage's tool_call arguments
            for i, tool_call in enumerate(think_tool_calls):
                logger.info("  skipping think_tool[%d] (content already in tool_call args)", i)
                tool_messages.append(
                    ToolMessage(
                        content="Acknowledged.",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # Handle ConductResearch calls (asynchronous)
            if conduct_research_calls:
                for tool_call in conduct_research_calls:
                    args = tool_call.get("args", {})
                    research_topic = args.get("research_topic", "")
                    gap_id = str(args.get("gap_id", "") or "").strip()
                    if gap_id:
                        args["gap_id"] = gap_id
                    perspectives = args.get("perspectives")
                    if isinstance(perspectives, list):
                        args["perspectives"] = [str(p).strip() for p in perspectives if str(p).strip()]
                    else:
                        args["perspectives"] = []

                    if "search_type" not in args or not args.get("search_type"):
                        if "Search type:" in research_topic or "search type:" in research_topic:
                            match = re.search(r"[Ss]earch\s+type:\s*(\w+)", research_topic)
                            if match:
                                extracted_type = match.group(1).lower()
                                if extracted_type in ["internal", "external", "both"]:
                                    args["search_type"] = extracted_type
                                    logger.info("  Extracted search_type='%s' from research_topic", extracted_type)
                                    args["research_topic"] = re.sub(
                                        r"\s*[Ss]earch\s+type:\s*\w+\.?\s*",
                                        "",
                                        research_topic,
                                    ).strip()
                                else:
                                    args["search_type"] = "both"
                            else:
                                normalized_topic = research_topic.lower()
                                if (
                                    "only using internal retrieval" in normalized_topic
                                    or "do not search the web" in normalized_topic
                                    or "retrieve_document_chunks" in normalized_topic and "user_article" in normalized_topic
                                ):
                                    args["search_type"] = "internal"
                                else:
                                    args["search_type"] = "both"
                        else:
                            normalized_topic = research_topic.lower()
                            if (
                                "only using internal retrieval" in normalized_topic
                                or "do not search the web" in normalized_topic
                                or "retrieve_document_chunks" in normalized_topic and "user_article" in normalized_topic
                            ):
                                args["search_type"] = "internal"
                            else:
                                args["search_type"] = "both"
                    else:
                        logger.info("  search_type already specified: '%s'", args.get("search_type"))

                current_storm_rounds = storm_rounds
                new_storm_rounds = current_storm_rounds + len(conduct_research_calls)

                if new_storm_rounds > MAX_STORM_ROUNDS:
                    logger.warning(
                        "  STORM_ROUND_LIMIT: Requested %d ConductResearch calls would exceed max (%d). Limiting to %d.",
                        len(conduct_research_calls),
                        MAX_STORM_ROUNDS,
                        max(0, MAX_STORM_ROUNDS - current_storm_rounds),
                    )
                    remaining_rounds = max(0, MAX_STORM_ROUNDS - current_storm_rounds)
                    conduct_research_calls = conduct_research_calls[:remaining_rounds]
                    if not conduct_research_calls:
                        logger.warning(
                            "  STORM_ROUND_LIMIT: All ConductResearch calls blocked. Max STORM rounds (%d) reached.",
                            MAX_STORM_ROUNDS,
                        )
                        should_end = True
                        next_step = END
                        stop_reason = "research_budget_exhausted"
                        route_reason = "conduct_research_limit_reached"
                        completion_status = "completed_after_budget_exhausted"
                        observability_updates.append(
                            {
                                "category": "routing",
                                "node": "supervisor_tools",
                                "event_key": "conduct_research:round_limit_reached",
                                "route": str(END),
                                "reason": "conduct_research_limit_reached",
                            }
                        )

                if len(conduct_research_calls) > 1:
                    logger.info(
                        "  Limiting to 1 ConductResearch call (requested %d, will process remaining in subsequent turns)",
                        len(conduct_research_calls),
                    )
                    conduct_research_calls = conduct_research_calls[:1]

                if conduct_research_calls and not should_end:
                    logger.info(
                        "  LAUNCHING %d ConductResearch agents... (STORM rounds: %d -> %d/%d)",
                        len(conduct_research_calls),
                        current_storm_rounds,
                        current_storm_rounds + len(conduct_research_calls),
                        MAX_STORM_ROUNDS,
                    )

                    for i, tc in enumerate(conduct_research_calls):
                        topic = tc["args"]["research_topic"][:100]
                        search_type = tc["args"].get("search_type", "both")
                        gap_id = tc["args"].get("gap_id", "")
                        selected_perspectives = tc["args"].get("perspectives", [])
                        logger.info(
                            "    agent[%d] topic: %s... | search_type: %s | gap_id=%s | perspectives=%s",
                            i,
                            topic,
                            search_type,
                            gap_id or "(none)",
                            selected_perspectives or ["(default)"],
                        )

                    draft_report = state.get("draft_report", "")
                    max_draft_context_chars = int(
                        os.environ.get("DEEP_RESEARCH_STORM_DRAFT_CONTEXT_CHARS", "16000")
                    )
                    draft_context_by_tool_call: dict[str, str] = {}
                    for tc in conduct_research_calls:
                        research_topic = tc["args"]["research_topic"]
                        draft_report_context, selected_headers = _build_targeted_draft_context(
                            research_topic=research_topic,
                            draft_report=str(draft_report or ""),
                            max_chars=max_draft_context_chars,
                        )
                        draft_context_by_tool_call[tc["id"]] = draft_report_context
                        logger.info(
                            "  Draft context for topic '%s': %d chars -> %d chars | sections=%s",
                            research_topic[:80],
                            len(str(draft_report or "")),
                            len(draft_report_context),
                            selected_headers or ["(empty)"],
                        )

                    coros = [
                        researcher_agent.ainvoke(
                            {
                                "researcher_messages": [
                                    HumanMessage(content=tool_call["args"]["research_topic"])
                                ],
                                "research_topic": tool_call["args"]["research_topic"],
                                "research_brief": state.get("research_brief", ""),
                                "article_summary": state.get("article_summary", ""),
                                "draft_report": draft_context_by_tool_call.get(tool_call["id"], ""),
                                "search_type": tool_call["args"].get("search_type", "both"),
                                "perspectives": (
                                    tool_call["args"].get("perspectives", [])
                                    or state.get("storm_perspectives", [])
                                    or []
                                ),
                                "forced_perspectives": tool_call["args"].get("perspectives", []) or [],
                                "active_gap_id": tool_call["args"].get("gap_id", ""),
                                "perspective_research_plans": state.get("storm_perspective_research_plans", {}) or {},
                                "evidence_ledger": state.get("evidence_ledger", []) or [],
                            },
                            config={
                                "run_name": f"storm_research_agent_{i}_{tool_call['args']['research_topic'][:50].replace(' ', '_')}",
                                "tags": ["storm", "research_agent", "conduct_research"],
                            },
                        )
                        for i, tool_call in enumerate(conduct_research_calls)
                    ]

                    research_start = time.perf_counter()
                    logger.info("  Waiting for %d research agents to complete...", len(coros))
                    tool_results = await asyncio.gather(*coros) if coros else []
                    research_elapsed = time.perf_counter() - research_start
                    avg_elapsed = research_elapsed / len(tool_results) if tool_results else 0.0
                    logger.info(
                        "  All %d research agents completed in %.2fs (avg %.2fs each)",
                        len(tool_results),
                        research_elapsed,
                        avg_elapsed,
                    )

                    total_compressed = 0
                    total_raw_notes = 0
                    for i, result in enumerate(tool_results):
                        compressed = result.get("compressed_research", "")
                        raw_notes = result.get("raw_notes", [])
                        total_compressed += len(compressed)
                        total_raw_notes += len(raw_notes)
                        logger.info(
                            "    agent[%d] result: compressed_len=%d chars, raw_notes=%d items",
                            i,
                            len(compressed),
                            len(raw_notes),
                        )

                    logger.info(
                        "  RESEARCH TOTALS: compressed=%d chars, raw_notes=%d items",
                        total_compressed,
                        total_raw_notes,
                    )

                    running_entries = canonicalize_evidence_ledger(state.get("evidence_ledger", []) or [])
                    running_retrieval_events = [dict(item or {}) for item in (state.get("retrieval_events", []) or [])]
                    for idx, (result, tool_call) in enumerate(zip(tool_results, conduct_research_calls), start=1):
                        candidate_entries = canonicalize_evidence_ledger(result.get("evidence_ledger", []) or [])
                        candidate_events = [dict(item or {}) for item in (result.get("retrieval_events", []) or [])]
                        new_entries = _diff_round_entries(running_entries, candidate_entries)
                        new_events = _diff_round_retrieval_events(running_retrieval_events, candidate_events)
                        round_number = current_storm_rounds + idx

                        novelty_summary = compute_evidence_novelty(running_entries, new_entries)
                        novelty_summary["storm_round"] = round_number
                        novelty_summary["topic"] = tool_call["args"].get("research_topic", "")
                        research_novelty_updates.append(novelty_summary)
                        observability_updates.append(
                            {
                                "category": "novelty",
                                "node": "supervisor_tools",
                                "event_key": f"storm_round_{round_number}",
                                "reason": "post_conduct_research",
                                **novelty_summary,
                            }
                        )

                        impact = compute_research_round_impact(running_entries, new_entries, new_events)
                        gap_id = str(tool_call["args"].get("gap_id", "") or "").strip()
                        round_summary = _build_round_summary(
                            storm_round=round_number,
                            gap_id=gap_id,
                            research_topic=tool_call["args"].get("research_topic", ""),
                            search_type=tool_call["args"].get("search_type", "both"),
                            new_entries=new_entries,
                            impact=impact,
                        )
                        round_summary_updates.append(round_summary)
                        round_note_updates.append(round_summary.get("findings_packet", "") or round_summary["summary"])
                        last_round_impact_summary = round_summary["summary"]
                        last_round_material_improvement = round_summary["material_improvement"]

                        tool_messages.append(
                            ToolMessage(
                                content=round_summary.get("findings_packet", "") or round_summary["summary"],
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

                        delta_raw_notes = [
                            str(entry.get("answer", "") or "").strip()
                            for entry in new_entries
                            if str(entry.get("answer", "") or "").strip()
                        ]
                        if not delta_raw_notes:
                            delta_raw_notes = [
                                str(note).strip()
                                for note in (result.get("raw_notes", []) or [])
                                if str(note).strip()
                            ][:1]
                        if delta_raw_notes:
                            all_raw_notes.extend(delta_raw_notes[:3])

                        if new_entries:
                            all_evidence_ledger.extend(new_entries)
                        if new_events:
                            all_retrieval_events.extend(new_events)

                        round_mix = impact.get("source_mix", {}) or {}
                        if round_mix.get("article_queries", 0):
                            internal_rounds += 1
                        if round_mix.get("external_queries", 0):
                            external_rounds += 1
                        if tool_call["args"].get("search_type", "both") != "internal":
                            external_grounding_considered = True

                        observability_updates.append(
                            {
                                "category": "round_impact",
                                "node": "supervisor_tools",
                                "event_key": f"storm_round_{round_number}:impact",
                                "reason": "post_conduct_research",
                                "material_improvement": round_summary["material_improvement"],
                                "material_reasons": round_summary.get("material_reasons", []),
                                "non_material_reason": round_summary.get("non_material_reason", ""),
                                "search_type": round_summary.get("search_type", "both"),
                            }
                        )

                        if gap_id:
                            gap_ledger = _apply_gap_round_result(
                                gap_ledger,
                                gap_id=gap_id,
                                impact=impact,
                            )
                            observability_updates.append(
                                {
                                    "category": "gap_ledger",
                                    "node": "supervisor_tools",
                                    "event_key": f"{gap_id}:post_round",
                                    "reason": "post_conduct_research",
                                    "gap_id": gap_id,
                                    "material_improvement": round_summary["material_improvement"],
                                    "non_material_reason": round_summary.get("non_material_reason", ""),
                                    "status_counts": round_summary.get("status_counts", {}),
                                }
                            )

                        running_entries = canonicalize_evidence_ledger([*running_entries, *new_entries])
                        running_retrieval_events = [*running_retrieval_events, *new_events]

                    storm_rounds = current_storm_rounds + len(conduct_research_calls)
                    route_reason = "conduct_research_completed"

                    if tool_results:
                        latest_result = tool_results[-1]
                        state = {
                            **state,
                            "storm_perspectives": latest_result.get("perspectives", state.get("storm_perspectives", []) or []),
                            "storm_perspective_research_plans": latest_result.get(
                                "perspective_research_plans",
                                state.get("storm_perspective_research_plans", {}) or {},
                            ),
                        }

                    overall_mix = source_mix_report(
                        canonicalize_evidence_ledger([
                            *(state.get("evidence_ledger", []) or []),
                            *all_evidence_ledger,
                        ]),
                        [
                            *(state.get("retrieval_events", []) or []),
                            *all_retrieval_events,
                        ],
                    )
                    source_mix_summary_text = overall_mix.get("summary", "")
                    external_grounding_rationale = _derive_external_grounding_rationale(
                        mix=overall_mix,
                        external_grounding_considered=external_grounding_considered,
                    )

            # Collect findings from the most recent ConductResearch calls (since last refine_draft_report)
            # This handles both same-turn and separate-turn scenarios
            current_round_findings = ""
            if conduct_research_calls:
                # Same turn: pass structured round deltas instead of prose summaries.
                current_round_findings = _compile_pending_findings_packet(
                    round_summary_updates,
                    max_rounds=max(1, len(round_summary_updates)),
                    max_chars=7000,
                )
                logger.info("  current_round_findings (same turn): %d chars from %d research calls", 
                           len(current_round_findings), len(conduct_research_calls))
            elif refine_report_calls:
                current_round_findings = _pending_round_summaries(state, max_items=4, max_chars_each=2200)
                logger.info(
                    "  current_round_findings (from round summaries): %d chars",
                    len(current_round_findings),
                )

            # Check draft editing round limit before processing refine_draft_report calls
            current_draft_editing_rounds = draft_editing_rounds
            if refine_report_calls and current_draft_editing_rounds >= MAX_DRAFT_EDITING_ROUNDS:
                logger.warning("  DRAFT_EDITING_LIMIT: %d refine_draft_report calls blocked. Max draft editing rounds (%d) reached.",
                              len(refine_report_calls), MAX_DRAFT_EDITING_ROUNDS)
                refine_report_calls = []  # Block all refine_draft_report calls
            
            if refine_report_calls and current_draft_editing_rounds + len(refine_report_calls) > MAX_DRAFT_EDITING_ROUNDS:
                remaining_rounds = max(0, MAX_DRAFT_EDITING_ROUNDS - current_draft_editing_rounds)
                logger.warning("  DRAFT_EDITING_LIMIT: Requested %d refine_draft_report calls would exceed max (%d). Limiting to %d.",
                              len(refine_report_calls), MAX_DRAFT_EDITING_ROUNDS, remaining_rounds)
                refine_report_calls = refine_report_calls[:remaining_rounds]
            
            # Initialize current_draft from state for iterative refinement within same turn
            current_draft = state.get("draft_report", "")
            
            for i, tool_call in enumerate(refine_report_calls):
              logger.info("  executing refine_draft_report[%d] (draft editing rounds: %d -> %d/%d)", 
                         i, current_draft_editing_rounds, 
                         current_draft_editing_rounds + i + 1, MAX_DRAFT_EDITING_ROUNDS)
              # Use only current round's findings, not all accumulated findings
              # The draft_report already contains prior research integrated from previous rounds
              findings = current_round_findings
              logger.info("    findings_len=%d (current round only)", len(findings))
              logger.info("    input_draft_len=%d", len(current_draft))

              draft_report = refine_draft_report.invoke({
                    "research_brief": state.get("research_brief", ""),
                    "article_summary": state.get("article_summary", ""),
                    "findings": findings,
                    "draft_report": current_draft,  # Use iteratively updated draft
                    "newsletter_template": newsletter_template
              })
              
              # Update current_draft for next iteration in same turn
              current_draft = draft_report

              tool_messages.append(
                  ToolMessage(
                      content=draft_report,
                      name=tool_call["name"],
                      tool_call_id=tool_call["id"]
                  )
              )
              
              # Update draft editing rounds counter after each refine_draft_report call
              current_draft_editing_rounds += 1
              draft_editing_rounds = current_draft_editing_rounds
              merged_round_count = len((state.get("research_round_summaries", []) or [])) + len(round_summary_updates)
              state = {
                  **state,
                  "draft_editing_rounds": draft_editing_rounds,
                  "draft_report": draft_report,
                  "merged_round_count": merged_round_count,
              }
              logger.info("    output_draft_len=%d", len(str(draft_report)))

        except Exception as e:
            logger.exception("SUPERVISOR_TOOLS ERROR: %s", e)
            should_end = True
            next_step = END
            stop_reason = "error"
            route_reason = "error"
            completion_status = "completed_after_degraded_recovery"

    # Single return point with appropriate state updates
    total_elapsed = time.perf_counter() - start_time
    
    if should_end:
        total_notes = [*(state.get("notes", []) or []), *round_note_updates]
        total_notes_chars = sum(len(n) for n in total_notes)
        logger.info("="*60)
        logger.info("SUPERVISOR_TOOLS ENDING")
        logger.info("  reason: %s", 
                   "exceeded_iterations" if exceeded_iterations else 
                   "no_tool_calls" if no_tool_calls else 
                   "research_complete" if research_complete else stop_reason or "error")
        logger.info("  notes_count: %d", len(total_notes))
        logger.info("  notes_total_chars: %d", total_notes_chars)
        logger.info("  total_time: %.2fs", total_elapsed)
        logger.info("="*60)
        return Command(
            goto=next_step,
            update={
                "notes": round_note_updates,
                "raw_notes": all_raw_notes,
                "storm_rounds": storm_rounds,
                "draft_editing_rounds": draft_editing_rounds,
                "evidence_ledger": all_evidence_ledger,
                "retrieval_events": all_retrieval_events,
                "research_novelty_history": research_novelty_updates,
                "research_round_summaries": round_summary_updates,
                "merged_round_count": merged_round_count,
                "storm_perspectives": state.get("storm_perspectives", []) or [],
                "storm_perspective_research_plans": state.get("storm_perspective_research_plans", {}) or {},
                "last_round_impact_summary": last_round_impact_summary,
                "last_round_material_improvement": last_round_material_improvement,
                "source_mix_summary": source_mix_summary_text,
                "internal_rounds": internal_rounds,
                "external_rounds": external_rounds,
                "external_grounding_considered": external_grounding_considered,
                "external_grounding_rationale": external_grounding_rationale,
                "why_follow_up_before_merge": "",
                "supervisor_stop_reason": stop_reason,
                "supervisor_route_reason": route_reason,
                "supervisor_completion_status": completion_status,
                "supervisor_phase": supervisor_phase,
                "gap_cards": gap_cards,
                "gap_ledger": gap_ledger,
                "observability_events": observability_updates
                + [
                    {
                        "category": "routing",
                        "node": "supervisor_tools",
                        "event_key": f"supervisor_tools:end:{stop_reason}",
                        "route": str(next_step),
                        "reason": route_reason,
                        "stop_reason": stop_reason,
                        "completion_status": completion_status,
                    }
                ],
            }
        )
    elif len(refine_report_calls) > 0:
        logger.info("SUPERVISOR_TOOLS CONTINUE (with draft) | tool_messages=%d | next=%s | time=%.2fs", 
                   len(tool_messages), next_step, total_elapsed)
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "notes": round_note_updates,
                "raw_notes": all_raw_notes,
                "draft_report": draft_report,
                "storm_rounds": storm_rounds,
                "draft_editing_rounds": draft_editing_rounds,
                "evidence_ledger": all_evidence_ledger,
                "retrieval_events": all_retrieval_events,
                "research_novelty_history": research_novelty_updates,
                "research_round_summaries": round_summary_updates,
                "merged_round_count": merged_round_count,
                "storm_perspectives": state.get("storm_perspectives", []) or [],
                "storm_perspective_research_plans": state.get("storm_perspective_research_plans", {}) or {},
                "last_round_impact_summary": last_round_impact_summary,
                "last_round_material_improvement": last_round_material_improvement,
                "source_mix_summary": source_mix_summary_text,
                "internal_rounds": internal_rounds,
                "external_rounds": external_rounds,
                "external_grounding_considered": external_grounding_considered,
                "external_grounding_rationale": external_grounding_rationale,
                "why_follow_up_before_merge": "",
                "supervisor_stop_reason": stop_reason,
                "supervisor_route_reason": "continue_with_refined_draft",
                "supervisor_completion_status": completion_status,
                "supervisor_phase": supervisor_phase,
                "gap_cards": gap_cards,
                "gap_ledger": gap_ledger,
                "observability_events": observability_updates
                + [
                    {
                        "category": "routing",
                        "node": "supervisor_tools",
                        "event_key": "supervisor_tools:continue:refine",
                        "route": next_step,
                        "reason": "refine_draft_then_continue",
                    }
                ],
            }
        )
    else:
        logger.info("SUPERVISOR_TOOLS CONTINUE | tool_messages=%d | next=%s | time=%.2fs", 
                   len(tool_messages), next_step, total_elapsed)
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "notes": round_note_updates,
                "raw_notes": all_raw_notes,
                "storm_rounds": storm_rounds,
                "draft_editing_rounds": draft_editing_rounds,
                "evidence_ledger": all_evidence_ledger,
                "retrieval_events": all_retrieval_events,
                "research_novelty_history": research_novelty_updates,
                "research_round_summaries": round_summary_updates,
                "merged_round_count": merged_round_count,
                "storm_perspectives": state.get("storm_perspectives", []) or [],
                "storm_perspective_research_plans": state.get("storm_perspective_research_plans", {}) or {},
                "last_round_impact_summary": last_round_impact_summary,
                "last_round_material_improvement": last_round_material_improvement,
                "source_mix_summary": source_mix_summary_text,
                "internal_rounds": internal_rounds,
                "external_rounds": external_rounds,
                "external_grounding_considered": external_grounding_considered,
                "external_grounding_rationale": external_grounding_rationale,
                "supervisor_stop_reason": stop_reason,
                "supervisor_route_reason": route_reason,
                "supervisor_completion_status": completion_status,
                "supervisor_phase": supervisor_phase,
                "gap_cards": gap_cards,
                "gap_ledger": gap_ledger,
                "observability_events": observability_updates
                + [
                    {
                        "category": "routing",
                        "node": "supervisor_tools",
                        "event_key": "supervisor_tools:continue",
                        "route": next_step,
                        "reason": route_reason,
                    }
                ],
            }
        )


# ===== GRAPH CONSTRUCTION =====

# Build supervisor graph
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_agent = supervisor_builder.compile()
