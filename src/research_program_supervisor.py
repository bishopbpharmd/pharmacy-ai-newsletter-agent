"""Research-program orchestration for long-running deep research.

This module replaces the old lightweight gap-router loop with a more explicit
multi-agent program:
1. Select fixed perspectives once for the run.
2. Collect perspective proposals once.
3. Initialize one persistent global research agenda.
4. Repeatedly prioritize, assign, execute, update agenda, refine draft, and gate.

The agenda manager is the only component that updates the agenda. Other agents
read it, act on it, and emit bounded deltas or decisions.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any
from typing_extensions import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from src.evidence_utils import (
    canonicalize_evidence_ledger,
    compute_evidence_novelty,
    compute_research_round_impact,
    source_mix_report,
)
from src.document_profile import format_document_profile, infer_document_profile
from src.logging_config import get_logger
from src.model_config import PIPELINE_MODEL_SETTINGS, build_chat_model
from src.prompts import (
    agenda_update_prompt,
    fixed_perspective_selection_prompt,
    global_research_agenda_initialization_prompt,
    perspective_agenda_proposal_prompt,
    progress_gate_prompt,
    research_assignment_prompt,
    research_priority_planner_prompt,
)
from src.research_agent_storm import (
    _classify_question_scope,
    _extract_external_artifact_terms,
    _has_broad_external_grounding_intent,
    _looks_paper_specific_external_need,
    _query_word_count,
    _shape_local_retrieval_query,
    storm_researcher_agent,
)
from src.state_multi_agent_supervisor import (
    AgendaUpdateDelta,
    AssignmentDecision,
    FixedPerspectiveRoster,
    GlobalResearchAgenda,
    PerspectiveProposal,
    PriorityDecision,
    ProgressGateDecision,
    SupervisorState,
)
from src.utils import get_today_str, refine_draft_report
from src.multi_agent_supervisor import (
    _build_round_summary,
    _build_targeted_draft_context,
    _compile_pending_findings_packet,
    _derive_external_grounding_rationale,
    _diff_round_entries,
    _diff_round_retrieval_events,
)

logger = get_logger("deep_research.research_program")

agenda_model = build_chat_model(PIPELINE_MODEL_SETTINGS.supervisor_orchestration_model)
MAX_RESEARCH_PROGRAM_CYCLES = int(
    os.environ.get("DEEP_RESEARCH_MAX_PROGRAM_CYCLES", "10")
)
DETERMINISTIC_SUPERVISOR = os.environ.get(
    "DEEP_RESEARCH_DETERMINISTIC_SUPERVISOR",
    "true",
).strip().lower() not in {"0", "false", "no"}
AGENDA_PROPOSAL_TIMEOUT_SECONDS = float(
    os.environ.get("DEEP_RESEARCH_AGENDA_PROPOSAL_TIMEOUT_SECONDS", "45")
)
AGENDA_PROPOSAL_RETRY_TIMEOUT_SECONDS = float(
    os.environ.get("DEEP_RESEARCH_AGENDA_PROPOSAL_RETRY_TIMEOUT_SECONDS", "20")
)
SUPERVISOR_FIXED_PERSPECTIVE_TIMEOUT_SECONDS = float(
    os.environ.get("DEEP_RESEARCH_SUPERVISOR_FIXED_PERSPECTIVE_TIMEOUT_SECONDS", "45")
)
SUPERVISOR_AGENDA_INIT_TIMEOUT_SECONDS = float(
    os.environ.get("DEEP_RESEARCH_SUPERVISOR_AGENDA_INIT_TIMEOUT_SECONDS", "60")
)
SUPERVISOR_PRIORITY_TIMEOUT_SECONDS = float(
    os.environ.get("DEEP_RESEARCH_SUPERVISOR_PRIORITY_TIMEOUT_SECONDS", "45")
)
SUPERVISOR_ASSIGNMENT_TIMEOUT_SECONDS = float(
    os.environ.get("DEEP_RESEARCH_SUPERVISOR_ASSIGNMENT_TIMEOUT_SECONDS", "60")
)
SUPERVISOR_AGENDA_UPDATE_TIMEOUT_SECONDS = float(
    os.environ.get("DEEP_RESEARCH_SUPERVISOR_AGENDA_UPDATE_TIMEOUT_SECONDS", "60")
)
SUPERVISOR_PROGRESS_GATE_TIMEOUT_SECONDS = float(
    os.environ.get("DEEP_RESEARCH_SUPERVISOR_PROGRESS_GATE_TIMEOUT_SECONDS", "45")
)
SUPERVISOR_STRUCTURED_TIMEOUT_RETRIES = int(
    os.environ.get("DEEP_RESEARCH_SUPERVISOR_STRUCTURED_TIMEOUT_RETRIES", "1")
)

_ARTICLE_THESIS_HINTS = (
    "framework",
    "evaluation framework",
    "evaluation method",
    "we propose",
    "we describe",
    "our contributions",
    "novel",
    "retrospective evaluation",
    "assessment",
)

_NON_AUTONOMOUS_HINTS = (
    "our facility",
    "our hospital",
    "local retrospective cohort",
    "vendor",
    "request",
    "contact author",
    "email authors",
    "reproduce",
    "simulate missing",
    "obtain local",
)


def _truncate(text: Any, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    sentence_parts = re.findall(r"[^.!?]+[.!?]?", value)
    kept: list[str] = []
    for part in sentence_parts:
        candidate = re.sub(r"\s+", " ", part).strip()
        if not candidate:
            continue
        projected = " ".join([*kept, candidate]).strip()
        if projected and len(projected) <= max_chars:
            kept.append(candidate)
        else:
            break
    if kept:
        return " ".join(kept).strip()
    return value[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")


def _perspective_names(state: SupervisorState) -> list[str]:
    names: list[str] = []
    for entry in state.get("research_perspectives", []) or []:
        if isinstance(entry, dict):
            name = str(entry.get("name", "") or "").strip()
        else:
            name = str(entry or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _best_perspective_name(state: SupervisorState, *keywords: str) -> str:
    names = _perspective_names(state)
    lowered = [(name, name.lower()) for name in names]
    for keyword in keywords:
        for name, lowered_name in lowered:
            if keyword.lower() in lowered_name:
                return name
    return names[0] if names else ""


def _is_framework_or_method_paper(article_summary: str) -> bool:
    summary = str(article_summary or "").lower()
    return any(hint in summary for hint in _ARTICLE_THESIS_HINTS)


def _thesis_focus_text(article_summary: str) -> str:
    if _is_framework_or_method_paper(article_summary):
        return (
            "Identify the source's primary contribution first. "
            "If datasets, case studies, or examples are only demonstrations of a broader framework or evaluation method, keep them subordinate."
        )
    return (
        "Identify the source's main claim, intervention, or finding first, then separate supporting examples from the central result."
    )


def _document_profile_from_state(state: SupervisorState) -> dict[str, Any]:
    return infer_document_profile(
        str(state.get("article_summary", "") or ""),
        newsletter_template=str(state.get("newsletter_template", "") or ""),
        draft_report=str(state.get("draft_report", "") or ""),
    )


def _compact_agenda_text(text: str, max_chars: int = 220) -> str:
    return _truncate(re.sub(r"\s+", " ", str(text or "")).strip(), max_chars)


def _strip_non_autonomous_scope(text: str) -> str:
    cleaned = str(text or "")
    lowered = cleaned.lower()
    if not any(hint in lowered for hint in _NON_AUTONOMOUS_HINTS):
        return cleaned
    return (
        "Stay within what the article, public sources, and the current run can actually support. "
        "Do not depend on local site data, non-public artifacts, author outreach, or reproducing unavailable outputs."
    )


def _build_source_grounded_initial_agenda(state: SupervisorState) -> dict:
    profile = _document_profile_from_state(state)
    source_kind = str(profile.get("source_kind", "") or "")
    title = str(profile.get("title", "") or "the source").strip()
    anchor_terms = [str(item).strip() for item in (profile.get("anchor_terms", []) or []) if str(item).strip()]
    anchor_text = ", ".join(anchor_terms[:4]) or "the source's named entities and decisive facts"
    external_strategy = str(profile.get("external_strategy", "") or "bounded external grounding")
    paper_evaluator = _best_perspective_name(state, "paper evaluator", "methods", "evidence", "study")
    informatics = _best_perspective_name(state, "informatics")
    director = _best_perspective_name(state, "director", "operations", "clinical", "icu")

    core_assignees = [name for name in (paper_evaluator, director, informatics) if name][:2]
    if not core_assignees:
        core_assignees = _perspective_names(state)[:2]
    limits_assignees = [name for name in (paper_evaluator, informatics, director) if name][:2]
    if not limits_assignees:
        limits_assignees = _perspective_names(state)[:2]
    external_assignees = [name for name in (director, informatics, paper_evaluator) if name][:2]
    if not external_assignees:
        external_assignees = _perspective_names(state)[:2]

    if source_kind == "regulatory_guidance":
        core_title = "Anchor the operative rule, boundary, and decisive examples"
        core_question = (
            f"According to {title}, what changed or what scope boundary is the guidance defining, and which statutory criteria or examples most directly support that interpretation? "
            "Explicitly distinguish what is clearly supported from what remains ambiguous."
        )
        core_focus = "What changed, what boundary does the guidance define, and which criteria or examples make that boundary real?"
        detail_title = "Extract the criteria, scope limits, and edge-case examples"
        detail_question = (
            "What exact criteria, exclusions, examples, or carve-outs determine when the software remains outside device oversight versus when FDA still treats it as a device function?"
        )
        detail_focus = "Which exact criteria, exclusions, and examples determine what remains regulated versus excluded?"
        trust_title = "Capture interpretation limits and implementation guardrails"
        trust_question = (
            "What nonbinding caveats, issuer-side clarifications, implementation constraints, or interpretive ambiguities should temper how an operator reads this guidance?"
        )
        trust_focus = "What nonbinding caveats, interpretive ambiguities, or implementation guardrails matter most?"
        external_question = (
            "What official issuer materials, title-anchored pages, town halls, FAQs, or closely related standards would materially improve interpretation of this guidance without broadening into generic policy commentary?"
        )
        external_focus = (
            f"Start with the issuer and title: use {title} plus official materials before secondary commentary."
        )
        external_turn_query = "What official issuer material or title-anchored context most improves interpretation of this guidance?"
    elif source_kind in {"commentary_analysis", "commentary_product_update"}:
        core_title = "Anchor the article's main claim and decisive named examples"
        core_question = (
            f"According to the article itself, what is the central shift, argument, or recommendation, and which named examples or comparisons ({anchor_text}) most directly support it? "
            "Mark details that are commentary, recommendation, or uncertainty rather than hard evidence."
        )
        core_focus = "What is the article's main claim, and which named examples or comparisons make that claim concrete?"
        detail_title = "Extract the definitions, contrasts, and concrete examples"
        detail_question = (
            "How does the article define its key categories, product layers, or decision framework, and which examples best show why those distinctions matter in practice?"
        )
        detail_focus = "How does the article define its key categories, and which examples or contrasts best illustrate them?"
        trust_title = "Capture caveats, uncertainty, and operational caution"
        trust_question = (
            "What caveats, missing evidence, security or workflow constraints, or explicit limitations should keep a pharmacy leader from taking the commentary as stronger proof than it is?"
        )
        trust_focus = "What caveats, uncertainty, or operational caution should temper the article's claims?"
        external_question = (
            "What surrounding product, issuer, comparator, or timeline context would materially sharpen interpretation of the article's main claim without drowning the newsletter in generic external commentary?"
        )
        external_focus = (
            f"Use named entities from the article first ({anchor_text}) and prefer issuer or comparator sources tied to those entities."
        )
        external_turn_query = "What issuer, comparator, or timeline context most sharpens the article's main claim?"
    elif source_kind == "research_methods_or_model":
        thesis_focus = _thesis_focus_text(str(state.get("article_summary", "") or ""))
        core_title = "Anchor the source's real thesis and decisive evidence"
        core_question = (
            f"According to the article itself, {thesis_focus.lower()} Extract the specific results, framework elements, and quantitative findings that most directly support that thesis. "
            "Explicitly mark details that are not reported in the source instead of inventing them."
        )
        core_focus = "What framework, method, or evaluation approach does the article propose?"
        detail_title = "Extract the decisive evaluation setup and metrics"
        detail_question = (
            "Which evaluation setup, benchmark details, decisive metrics, or failure patterns most directly determine whether the method is convincing?"
        )
        detail_focus = "Which evaluation setup, decisive metrics, or failure patterns most directly determine whether the method is convincing?"
        trust_title = "Extract methodological caveats that should temper trust"
        trust_question = (
            "What methodological caveats, leakage risks, threshold choices, retrospective-design limits, or feature-design issues should temper trust for a pharmacy audience?"
        )
        trust_focus = "What methodological caveats, leakage risks, or threshold choices should temper trust?"
        external_question = (
            "What related comparator studies, benchmarks, or governance expectations would materially change how a pharmacy leader interprets this method paper?"
        )
        external_focus = "Look for title-anchored comparators or governance context that changes interpretation, not generic best-practice filler."
        external_turn_query = "What comparator studies, benchmarks, or governance context most change interpretation of this method paper?"
    elif source_kind == "research_clinical_or_comparative":
        core_title = "Anchor the intervention, comparator, and decisive results"
        core_question = (
            "According to the article itself, what intervention, comparator, population, and outcome define the main claim, and which quantitative results most directly support it?"
        )
        core_focus = "What intervention, comparator, and outcomes define the main claim, and what results support it?"
        detail_title = "Extract decision-relevant outcomes and subgroup signals"
        detail_question = (
            "Which outcomes, subgroup findings, or comparison results would most change how a pharmacy leader interprets the intervention's real-world relevance?"
        )
        detail_focus = "Which outcomes, subgroup findings, or comparison results most change operational interpretation?"
        trust_title = "Extract trust boundaries and transferability limits"
        trust_question = (
            "What design limits, generalizability issues, operational caveats, or missing follow-up should temper trust in this study?"
        )
        trust_focus = "What design limits, generalizability issues, or missing follow-up temper trust?"
        external_question = (
            "What related comparator studies, practice context, or guideline-level framing would materially improve interpretation of this study?"
        )
        external_focus = "Seek comparator or practice context that changes interpretation of the main result."
        external_turn_query = "What comparator studies or practice context most improve interpretation of this study?"
    else:
        core_title = "Anchor the intervention, setting, and decisive real-world outcomes"
        core_question = (
            "According to the article itself, what intervention, service, or workflow is being evaluated, in what setting, and which measured outcomes most directly support the main claim?"
        )
        core_focus = "What intervention or service is being evaluated, in what setting, and what outcomes support the main claim?"
        detail_title = "Extract the measured outcomes and operationally meaningful results"
        detail_question = (
            "Which measured outcomes, care-path changes, appropriateness findings, or quantitative results most change how a pharmacy leader should interpret this study?"
        )
        detail_focus = "Which measured outcomes or quantitative results most change operational interpretation?"
        trust_title = "Extract trust boundaries and generalizability limits"
        trust_question = (
            "What study-design limits, missing follow-up, setting-specific constraints, or transferability issues should temper trust in these findings?"
        )
        trust_focus = "What study-design limits, missing follow-up, or transferability issues temper trust?"
        external_question = (
            "What surrounding real-world evidence, adoption context, or comparator material would materially improve interpretation of this study without overpowering the article itself?"
        )
        external_focus = "Use title-anchored comparators and surrounding real-world context that change interpretation."
        external_turn_query = "What surrounding real-world evidence or comparator context most improves interpretation of this study?"

    agenda = {
        "overall_goals": [
            "Publish a concise pharmacy-leader newsletter that reflects the source's real thesis, strongest evidence, and practical limitations.",
            "Keep the run grounded in article-supported facts plus one bounded layer of external context that materially changes interpretation, trust, or actionability.",
            "Converge when remaining uncertainties depend on unavailable or out-of-scope information; carry those limitations into the draft instead of inventing new work.",
        ],
        "active_items": [
            {
                "item_id": "source_core_claim_and_decisive_evidence",
                "title": core_title,
                "research_question": core_question,
                "status": "active",
                "priority": "high",
                "why_it_matters": (
                    "The newsletter will drift if it centers an illustrative example or secondary metric instead of the source's actual contribution."
                ),
                "completion_criteria": (
                    "The draft states the primary contribution first, uses article-supported numbers to back it up, and clearly labels missing details as not_in_source."
                ),
                "recommended_search_type": "internal",
                "assigned_perspectives": core_assignees,
                "evidence_summary": "",
                "execution_focus": core_focus,
                "first_turn_query": core_focus,
                "closure_condition": (
                    "Close when the article's central contribution, decisive numbers, and example-vs-thesis framing are all explicitly grounded."
                ),
                "attempt_count": 0,
            },
            {
                "item_id": "source_structure_scope_and_decisive_details",
                "title": detail_title,
                "research_question": detail_question,
                "status": "active",
                "priority": "high",
                "why_it_matters": (
                    "The newsletter should surface the document's actual controlling facts, definitions, or quantitative details rather than generic abstractions."
                ),
                "completion_criteria": (
                    "The draft names the highest-value source-native details with article-supported evidence and explains why they matter."
                ),
                "recommended_search_type": "internal",
                "assigned_perspectives": limits_assignees,
                "evidence_summary": "",
                "execution_focus": detail_focus,
                "first_turn_query": detail_focus,
                "closure_condition": (
                    "Close when the draft includes the most decision-relevant source-native details without forcing the wrong ontology onto the document."
                ),
                "attempt_count": 0,
            },
            {
                "item_id": "source_trust_boundaries_and_operational_limits",
                "title": trust_title,
                "research_question": trust_question,
                "status": "active",
                "priority": "high",
                "why_it_matters": (
                    "Pharmacy readers need the trust boundaries and evidence limits, not just the main headline."
                ),
                "completion_criteria": (
                    "The draft includes the highest-value article-supported trust caveats and translates them into pharmacy-relevant consequences without inventing local analyses."
                ),
                "recommended_search_type": "internal",
                "assigned_perspectives": limits_assignees,
                "evidence_summary": "",
                "execution_focus": trust_focus,
                "first_turn_query": trust_focus,
                "closure_condition": (
                    "Close when the draft's trust-boundary framing is article-grounded and no longer overclaims deployment readiness or operational certainty."
                ),
                "attempt_count": 0,
            },
            {
                "item_id": "external_context_grounding",
                "title": "Add one bounded layer of external context",
                "research_question": external_question,
                "status": "active",
                "priority": "medium",
                "why_it_matters": (
                    "The newsletter should place the source in broader context, but that context must stay bounded and publicly obtainable."
                ),
                "completion_criteria": (
                    "Add 1-2 compact external points that materially improve interpretation or actionability, or explicitly conclude that further external searching is low-yield."
                ),
                "recommended_search_type": "external",
                "assigned_perspectives": external_assignees,
                "evidence_summary": "",
                "execution_focus": external_focus,
                "first_turn_query": external_turn_query,
                "closure_condition": (
                    "Close when one bounded external pass materially improves the draft or when the public web adds no higher-value context."
                ),
                "attempt_count": 0,
            },
        ],
        "partial_items": [],
        "completed_items": [],
        "deferred_items": [],
        "external_grounding_goals": [
            f"Add publicly obtainable outside context that changes interpretation, trust, or actionability using {external_strategy}.",
        ],
        "agenda_notes": _strip_non_autonomous_scope(
            "Keep the agenda compact and source-grounded. Use the inferred document profile instead of default paper habits. "
            f"Profile: {format_document_profile(profile)}. "
            "Do not create work that depends on local data, non-public artifacts, author outreach, or reproducing unavailable outputs."
        ),
    }
    return _sanitize_initial_agenda(agenda, [], state.get("research_perspectives", []) or [])


def _researchable_agenda_items(agenda: dict) -> list[dict]:
    items: list[dict] = []
    for item in [*(agenda.get("active_items", []) or []), *(agenda.get("partial_items", []) or [])]:
        compact_item = _compact_agenda_item(item)
        if _normalize_work_mode(compact_item.get("work_mode", "")) in {"limitation_to_draft", "close_unavailable"}:
            continue
        if str(compact_item.get("status", "") or "").lower() in {"completed", "deferred", "not_answerable"}:
            continue
        items.append(compact_item)
    return items


def _external_grounding_exhausted(agenda: dict) -> bool:
    all_items = [
        *(agenda.get("active_items", []) or []),
        *(agenda.get("partial_items", []) or []),
        *(agenda.get("completed_items", []) or []),
        *(agenda.get("deferred_items", []) or []),
    ]
    relevant = [
        _compact_agenda_item(item)
        for item in all_items
        if str(item.get("recommended_search_type", "") or "").lower() in {"external", "both"}
        or "external" in str(item.get("item_id", "") or "").lower()
    ]
    if not relevant:
        return False
    return not any(item.get("status") in {"active", "partial"} for item in relevant)


def _deterministic_priority_decision(state: SupervisorState) -> dict:
    if len(state.get("research_round_summaries", []) or []) > int(state.get("merged_round_count", 0) or 0):
        return {
            "action": "refine_draft",
            "item_id": "",
            "rationale": "Pending findings should be merged into the draft before more research.",
        }

    agenda = _normalize_agenda(state.get("research_agenda", {}))
    item = _select_highest_value_item(agenda)
    if not item:
        return {
            "action": "finalize_candidate",
            "item_id": "",
            "rationale": "No researchable agenda items remain; ask the progress gate whether finalization is justified.",
        }

    override = _blocked_item_priority_override(item)
    if override:
        return override

    return {
        "action": "research",
        "item_id": str(item.get("item_id", "") or ""),
        "rationale": "Continue with the highest-value unresolved agenda item from the compact source-grounded agenda.",
    }


def _deterministic_agenda_delta(state: SupervisorState, latest_round: dict, focused_item: dict) -> dict:
    round_summary = latest_round.get("round_summary", {}) or {}
    impact = latest_round.get("impact", {}) or {}
    material = bool(round_summary.get("material_improvement", False))
    item = _compact_agenda_item(focused_item)
    item_id = str(item.get("item_id", "") or latest_round.get("item_id", "") or "").strip()
    current_attempt = int(item.get("attempt_count", 0) or 0) + 1
    summary = _truncate(round_summary.get("summary", "") or latest_round.get("findings_packet", "") or "(none)", 280)
    search_type = str(latest_round.get("search_type", item.get("recommended_search_type", "both")) or item.get("recommended_search_type", "both")).lower()

    new_status = "completed" if material else "active"
    work_mode = item.get("work_mode", "normal_research") or "normal_research"
    closure_reason = ""
    agenda_note = "Updated agenda deterministically from the latest research round."

    if item_id == "external_context_grounding":
        if material and impact.get("source_mix", {}).get("external_queries", 0):
            new_status = "completed"
            closure_reason = "external_grounding_completed"
            agenda_note = "Completed the bounded external-context pass because it materially improved interpretation."
        elif current_attempt >= 1:
            new_status = "deferred"
            work_mode = "limitation_to_draft"
            closure_reason = "external_grounding_low_yield"
            agenda_note = "Deferred further external searching because the bounded public-context pass was low-yield."
    elif material:
        new_status = "completed"
        closure_reason = "supported_enough_for_newsletter"
        agenda_note = "Completed the agenda item because the latest round materially improved the article-grounded draft."
    elif current_attempt >= 2:
        new_status = "deferred"
        work_mode = "limitation_to_draft"
        closure_reason = "deferred_low_yield_item"
        agenda_note = "Deferred the item because repeated searching added limited value; carry the limitation into the draft."

    completed_ids = [item_id] if new_status == "completed" else []
    deferred_ids = [item_id] if new_status == "deferred" else []
    return {
        "updates": [
            {
                "item_id": item_id,
                "new_status": new_status,
                "evidence_summary": summary,
                "recommended_search_type": search_type if search_type in {"internal", "external", "both"} else item.get("recommended_search_type", "both"),
                "assigned_perspectives": latest_round.get("perspectives", []) or item.get("assigned_perspectives", []),
                "execution_focus": item.get("execution_focus", ""),
                "work_mode": work_mode,
                "internal_focus": item.get("internal_focus", ""),
                "external_focus": item.get("external_focus", ""),
                "closure_condition": item.get("closure_condition", ""),
                "artifact_state": item.get("artifact_state", {}),
                "closure_reason": closure_reason,
                "reopen_only_if": item.get("reopen_only_if", ""),
            }
        ],
        "add_items": [],
        "completed_item_ids": completed_ids,
        "deferred_item_ids": deferred_ids,
        "agenda_note": agenda_note,
        "external_grounding_completed": bool(
            item_id == "external_context_grounding"
            and material
            and impact.get("source_mix", {}).get("external_queries", 0)
        ),
    }


def _allowed_acute_care_role_pool() -> list[dict]:
    return [
        {
            "name": "Informatics Pharmacist",
            "description": (
                "Focuses on alert logic, EHR integration, data quality, implementation guardrails, "
                "and how continuous-model outputs would behave inside inpatient medication-use workflows."
            ),
            "focus_areas": ["alert burden", "integration", "data quality", "workflow reliability"],
            "keywords": (
                "alert", "alerting", "algorithm", "cds", "decision support", "ehr",
                "integration", "informatics", "model", "prediction", "reliability", "threshold",
            ),
        },
        {
            "name": "ICU / Critical Care Pharmacist",
            "description": (
                "Focuses on bedside relevance for unstable inpatients, sepsis recognition, treatment timing, "
                "false positives in critical illness, and whether model outputs would help or distract critical care teams."
            ),
            "focus_areas": ["sepsis", "ICU workflow", "time-to-treatment", "clinical relevance"],
            "keywords": (
                "critical care", "icu", "intensive care", "sepsis", "shock", "vasopressor",
                "ventilator", "rapid deterioration", "bedside", "acute care",
            ),
        },
        {
            "name": "Clinical Pharmacist",
            "description": (
                "Focuses on clinical actionability, therapeutic consequences, patient-safety tradeoffs, "
                "and whether the evidence would change day-to-day inpatient pharmacy practice."
            ),
            "focus_areas": ["clinical actionability", "patient safety", "therapy decisions", "adoption"],
            "keywords": (
                "clinical", "patient", "therapy", "treatment", "adverse", "stewardship",
                "medication", "harm", "safety",
            ),
        },
        {
            "name": "Pharmacy Director",
            "description": (
                "Focuses on service-line feasibility, staffing, governance, pilot design, vendor diligence, "
                "and whether the intervention is worth operational attention from inpatient pharmacy leadership."
            ),
            "focus_areas": ["staffing", "pilot design", "governance", "operational feasibility"],
            "keywords": (
                "budget", "director", "feasibility", "governance", "implementation", "operations",
                "pilot", "resource", "staffing", "vendor", "workflow",
            ),
        },
    ]


def _select_acute_care_roles(article_summary: str, draft_report: str) -> list[dict]:
    context = f"{article_summary}\n{draft_report}".lower()
    scored_roles: list[tuple[int, int, dict]] = []
    for index, role in enumerate(_allowed_acute_care_role_pool()):
        score = sum(1 for keyword in role.get("keywords", ()) if keyword in context)
        scored_roles.append((score, -index, role))

    ranked = [role for _, _, role in sorted(scored_roles, reverse=True)]
    selected: list[dict] = []

    for preferred_name in ("Informatics Pharmacist", "Pharmacy Director", "ICU / Critical Care Pharmacist", "Clinical Pharmacist"):
        role = next((entry for entry in ranked if entry["name"] == preferred_name), None)
        if role and role not in selected:
            if preferred_name == "Pharmacy Director" and any("ICU" in item["name"] for item in selected):
                continue
            selected.append(role)
        if len(selected) >= 2:
            break

    if len(selected) < 2:
        for role in ranked:
            if role not in selected:
                selected.append(role)
            if len(selected) >= 2:
                break

    return selected[:2]


def _primary_analyst_for_profile(profile: dict[str, Any]) -> dict[str, Any]:
    source_kind = str(profile.get("source_kind", "") or "")
    if source_kind == "regulatory_guidance":
        return {
            "name": "Regulatory / Governance Pharmacist",
            "description": (
                "Focuses on what the guidance changes, where the scope boundaries really sit, "
                "which examples control interpretation, and what governance or compliance implications matter."
            ),
            "focus_areas": ["scope boundary", "official interpretation", "governance", "implementation guardrails"],
        }
    if source_kind in {"commentary_analysis", "commentary_product_update"}:
        return {
            "name": "Digital Strategy Pharmacist",
            "description": (
                "Interrogates market or commentary claims as operational signals: what is actually happening, "
                "which named examples are decisive, and where product or workflow claims outrun evidence."
            ),
            "focus_areas": ["named examples", "claim strength", "product fit", "operational signal"],
        }
    return {
        "name": "Evidence Evaluator",
        "description": (
            "Evaluates the source as an evidence artifact: what is actually being studied or proposed, "
            "how strong the support is, and which limitations should temper trust."
        ),
        "focus_areas": ["study design", "results", "limitations", "evidence strength"],
    }


def _secondary_roles_for_profile(
    profile: dict[str, Any],
    article_summary: str,
    draft_report: str,
) -> list[dict]:
    selected = _select_acute_care_roles(article_summary, draft_report)
    source_kind = str(profile.get("source_kind", "") or "")
    if source_kind == "regulatory_guidance":
        preferred = ("Informatics Pharmacist", "Pharmacy Director")
    elif source_kind in {"commentary_analysis", "commentary_product_update"}:
        preferred = ("Informatics Pharmacist", "Pharmacy Director")
    else:
        preferred = ("Informatics Pharmacist", "Pharmacy Director")

    ordered: list[dict] = []
    for name in preferred:
        role = next((entry for entry in selected if entry["name"] == name), None)
        if role and role not in ordered:
            ordered.append(role)
    for role in selected:
        if role not in ordered:
            ordered.append(role)
    return ordered[:2]


def _default_perspectives(
    article_summary: str,
    draft_report: str,
    newsletter_template: str = "",
) -> list[dict]:
    profile = infer_document_profile(
        article_summary,
        newsletter_template=newsletter_template,
        draft_report=draft_report,
    )
    return [
        _primary_analyst_for_profile(profile),
        *[
            {
                "name": role["name"],
                "description": role["description"],
                "focus_areas": role["focus_areas"],
            }
            for role in _secondary_roles_for_profile(profile, article_summary, draft_report)
        ],
    ]


def _format_perspectives_for_prompt(perspectives: list[dict]) -> str:
    if not perspectives:
        return "(none)"
    parts: list[str] = []
    for idx, perspective in enumerate(perspectives, start=1):
        focus = ", ".join(str(item) for item in (perspective.get("focus_areas", []) or [])[:5])
        parts.append(
            f"{idx}. {perspective.get('name', '')}\n"
            f"   Description: {perspective.get('description', '')}\n"
            f"   Focus areas: {focus or '(none)'}"
        )
    return "\n\n".join(parts)


def _perspective_profile_map(perspectives: list[dict]) -> dict[str, str]:
    profiles: dict[str, str] = {}
    for perspective in perspectives:
        name = str(perspective.get("name", "") or "").strip()
        if not name:
            continue
        description = str(perspective.get("description", "") or "").strip()
        focus = ", ".join(str(item) for item in (perspective.get("focus_areas", []) or [])[:6])
        profile = description
        if focus:
            profile = f"{description} Focus areas: {focus}."
        profiles[name] = profile.strip()
    return profiles


def _format_proposals_for_prompt(proposals: list[dict]) -> str:
    if not proposals:
        return "(none)"
    parts: list[str] = []
    for proposal in proposals:
        parts.append(
            "Perspective: {name}\nQuestions: {questions}\nExternal grounding: {external}\nRisks: {risks}".format(
                name=proposal.get("perspective_name", ""),
                questions="; ".join(proposal.get("proposed_questions", []) or []) or "(none)",
                external="; ".join(proposal.get("external_grounding_needs", []) or []) or "(none)",
                risks="; ".join(proposal.get("high_value_risks", []) or []) or "(none)",
            )
        )
    return "\n\n".join(parts)


def _compact_text(value: Any, max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    text = re.sub(r"(?i)\b(?:immediate next steps?|deliverables?|progress\s*=|next step:)\b.*$", "", text).strip(" ,;:")
    if len(text) <= max_chars:
        return text
    sentence_parts = re.findall(r"[^.!?]+[.!?]?", text)
    kept: list[str] = []
    for part in sentence_parts:
        candidate = re.sub(r"\s+", " ", part).strip()
        if not candidate:
            continue
        projected = " ".join([*kept, candidate]).strip()
        if projected and len(projected) <= max_chars:
            kept.append(candidate)
        else:
            break
    if kept:
        return " ".join(kept).strip()
    clipped = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return (clipped or text[:max_chars]).strip()


def _compact_question_focus(question: str, max_chars: int = 220) -> str:
    text = _compact_text(question, max_chars=max_chars * 2)
    if not text:
        return ""
    text = re.sub(r"(?i)\bif not\b.*$", "", text).strip(" ,;:")
    text = re.sub(
        r"(?i)\b(?:request|produce|deliver|draft|prepare|run|re-run|recompute|re-compute)\b.*$",
        "",
        text,
    ).strip(" ,;:")
    if len(text) > max_chars:
        text = _compact_text(text, max_chars)
    if text and text[-1] not in {"?", "."}:
        text += "?"
    return text


def _normalize_artifact_state_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for raw_key, raw_status in value.items():
        key = _compact_text(raw_key, 48).lower()
        status = str(raw_status or "").strip().lower()
        if not key or status not in _ARTIFACT_STATE_VALUES:
            continue
        normalized[key] = status
    return normalized


def _artifact_state_text(item: dict) -> str:
    artifact_state = _normalize_artifact_state_map(item.get("artifact_state", {}))
    if not artifact_state:
        return "(none)"
    return ", ".join(f"{key}:{status}" for key, status in artifact_state.items())


def _structured_supervisor_model(schema: Any):
    try:
        return agenda_model.with_structured_output(schema, method="function_calling")
    except TypeError:
        return agenda_model.with_structured_output(schema)


async def _invoke_supervisor_structured(
    schema: Any,
    prompt: str,
    timeout_seconds: float,
    validation_retries: int = 1,
):
    attempts = max(1, SUPERVISOR_STRUCTURED_TIMEOUT_RETRIES + 1)
    last_exc: BaseException | None = None
    repair_suffix = ""
    validation_attempts_left = max(0, validation_retries)
    for attempt_index in range(1, attempts + 1):
        model = _structured_supervisor_model(schema)
        messages = [HumanMessage(content=f"{prompt}{repair_suffix}")]
        try:
            if hasattr(model, "ainvoke"):
                return await asyncio.wait_for(model.ainvoke(messages), timeout=timeout_seconds)
            return await asyncio.wait_for(
                asyncio.to_thread(model.invoke, messages),
                timeout=timeout_seconds,
            )
        except ValidationError as exc:
            last_exc = exc
            logger.warning(
                "SUPERVISOR_STRUCTURED_CALL: %s failed validation on attempt %d/%d: %s",
                getattr(schema, "__name__", str(schema)),
                attempt_index,
                attempts,
                exc,
            )
            if validation_attempts_left <= 0:
                raise
            validation_attempts_left -= 1
            repair_suffix = (
                "\n\nValidation repair required:\n"
                f"{exc}\n"
                "Return the same schema again. "
                "Keep enum fields exact, keep object-typed fields as JSON objects, and do not change unrelated fields."
            )
            continue
        except asyncio.TimeoutError as exc:
            last_exc = exc
            logger.warning(
                "SUPERVISOR_STRUCTURED_CALL: %s timed out after %.1fs on attempt %d/%d",
                getattr(schema, "__name__", str(schema)),
                timeout_seconds,
                attempt_index,
                attempts,
            )
            if attempt_index >= attempts:
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Structured supervisor invocation exited without result or exception.")


def _infer_item_search_type(item: dict, fallback: str = "both") -> str:
    search_type = str(item.get("recommended_search_type", fallback) or fallback).lower()
    if search_type not in {"internal", "external", "both"}:
        search_type = fallback if fallback in {"internal", "external", "both"} else "both"
    work_mode = _normalize_work_mode(item.get("work_mode", ""))

    question_text = " ".join(
        str(item.get(key, "") or "")
        for key in ("execution_focus", "research_question", "title", "completion_criteria")
    ).strip()
    if not question_text:
        return "both" if work_mode == "boundary_with_artifact_check" else search_type

    scope_info = _classify_question_scope(question_text)
    scope = str(scope_info.get("scope", "ambiguous") or "ambiguous")
    confidence = str(scope_info.get("confidence", "low") or "low")
    requires_public_artifact = _item_requires_public_artifact(item)
    has_external_need = (
        requires_public_artifact
        or _looks_paper_specific_external_need(question_text)
        or _has_broad_external_grounding_intent(question_text)
    )
    explicit_article_first = (
        search_type == "internal"
        or question_text.lower().startswith("according to the article")
        or question_text.lower().startswith("what does the article")
    )

    if work_mode == "boundary_with_artifact_check":
        return "both"
    if work_mode in {"limitation_to_draft", "close_unavailable"}:
        return search_type if search_type in {"internal", "external", "both"} else "both"
    if explicit_article_first and not requires_public_artifact:
        if scope in {"article_internal", "mixed", "ambiguous"} or confidence != "strong":
            return "internal"

    if _has_broad_external_grounding_intent(question_text):
        return "external"

    if scope == "article_internal" and confidence in {"strong", "medium"} and not has_external_need:
        if search_type == "external":
            return "both"
        return search_type if search_type in {"internal", "both"} else "both"

    if scope in {"external_artifact", "external_context"} and confidence == "strong":
        return "external"

    if scope == "mixed":
        return "both"

    if requires_public_artifact and search_type == "internal":
        return "both"

    return search_type


def _execution_focus_from_item(item: dict) -> str:
    existing = _compact_question_focus(item.get("execution_focus", ""), max_chars=180)
    if existing:
        return existing
    search_type = str(item.get("recommended_search_type", "both") or "both").lower()
    question = str(item.get("research_question", "") or item.get("title", "")).strip()
    if not question:
        return ""
    if search_type in {"internal", "both"}:
        shaped, _ = _shape_local_retrieval_query(question)
        compacted = _compact_question_focus(shaped, max_chars=180)
        if compacted:
            return compacted
    return _compact_question_focus(question, max_chars=180)


def _closure_condition_from_item(item: dict) -> str:
    existing = _compact_text(item.get("closure_condition", ""), 220)
    if existing:
        return existing
    completion = _compact_text(item.get("completion_criteria", ""), 220)
    if completion:
        return completion
    focus = _execution_focus_from_item(item)
    if not focus:
        return ""
    artifact_state = _normalize_artifact_state_map(item.get("artifact_state", {}))
    if artifact_state:
        return (
            "Either find direct evidence answering this focus or establish cleanly that the required artifact "
            "is not reported or not publicly available."
        )
    return (
        "Either find direct evidence answering this focus or establish cleanly that the paper does not report it."
    )


def _default_artifact_state(item: dict) -> dict[str, str]:
    return _normalize_artifact_state_map(item.get("artifact_state", {}))


def _default_reopen_only_if(item: dict) -> str:
    existing = _compact_text(item.get("reopen_only_if", ""), 220)
    if existing:
        return existing
    artifact_state = _normalize_artifact_state_map(item.get("artifact_state", {}))
    if artifact_state:
        return (
            "Reopen only if a previously missing artifact becomes available or new supported evidence materially changes the answer path."
        )
    return (
        "Reopen only if a new supported evidence path materially changes the current boundary."
    )


_WORK_MODES = {
    "normal_research",
    "boundary_with_artifact_check",
    "limitation_to_draft",
    "close_unavailable",
}


def _normalize_work_mode(value: str) -> str:
    clean = str(value or "").strip()
    return clean if clean in _WORK_MODES else "normal_research"


def _item_is_closed(item: dict) -> bool:
    return str(item.get("status", "") or "").lower() in {"completed", "deferred", "not_answerable"}


def _item_all_artifacts_unavailable(item: dict) -> bool:
    artifact_state = _normalize_artifact_state_map(item.get("artifact_state", {}))
    return bool(artifact_state) and all(status == "unavailable" for status in artifact_state.values())


def _item_requires_public_artifact(item: dict) -> bool:
    return bool(_normalize_artifact_state_map(item.get("artifact_state", {})))


def _item_is_blocked_by_unavailable_dependencies(item: dict) -> bool:
    return _item_requires_public_artifact(item) and _item_all_artifacts_unavailable(item)


def _round_supports_reopen(
    item: dict,
    latest_round: dict,
    proposed_artifact_state: dict[str, str] | None = None,
) -> bool:
    impact = latest_round.get("impact", {}) or {}
    if int(impact.get("newly_supported", 0) or 0) <= 0 and int(impact.get("status_upgrades", 0) or 0) <= 0:
        return False
    if _item_all_artifacts_unavailable(item):
        round_artifacts = _normalize_artifact_state_map(
            proposed_artifact_state if proposed_artifact_state is not None else item.get("artifact_state", {})
        )
        return any(status == "available" for status in round_artifacts.values())
    return True

def _build_executable_worker_brief(item: dict, perspective_name: str, search_type: str) -> str:
    first_turn_query = _compact_question_focus(
        item.get("first_turn_query", "") or item.get("execution_focus", "") or item.get("research_question", "") or item.get("title", ""),
        max_chars=150,
    )
    if not first_turn_query:
        return ""
    return _compact_worker_brief(f"Priority question: {first_turn_query}", max_chars=220, max_sentences=1)


def _enforce_worker_brief_contract(item: dict, perspective_name: str, search_type: str, worker_brief: str) -> str:
    return _build_executable_worker_brief(item, perspective_name, search_type)


def _normalize_agenda(agenda: dict | None) -> dict:
    normalized = dict(agenda or {})
    for key in (
        "overall_goals",
        "active_items",
        "partial_items",
        "completed_items",
        "deferred_items",
        "external_grounding_goals",
    ):
        value = normalized.get(key)
        if not isinstance(value, list):
            normalized[key] = []
    if not isinstance(normalized.get("agenda_notes"), str):
        normalized["agenda_notes"] = ""
    return normalized


def _compact_agenda_item(item: dict) -> dict:
    normalized = dict(item or {})
    normalized["title"] = _compact_text(normalized.get("title", ""), 96) or str(normalized.get("item_id", "") or "Agenda item")
    normalized["research_question"] = _compact_question_focus(
        normalized.get("research_question", "") or normalized.get("title", ""),
        max_chars=240,
    )
    normalized["why_it_matters"] = _compact_text(normalized.get("why_it_matters", ""), 220)
    normalized["completion_criteria"] = _compact_text(normalized.get("completion_criteria", ""), 220)
    normalized["evidence_summary"] = _compact_text(normalized.get("evidence_summary", ""), 260)
    normalized["first_turn_query"] = _compact_question_focus(
        normalized.get("first_turn_query", "") or normalized.get("execution_focus", ""),
        max_chars=180,
    )
    normalized["artifact_state"] = _default_artifact_state(normalized)
    normalized["recommended_search_type"] = _infer_item_search_type(
        normalized,
        fallback=str(normalized.get("recommended_search_type", "both") or "both").lower(),
    )
    normalized["work_mode"] = _normalize_work_mode(normalized.get("work_mode", ""))
    normalized["execution_focus"] = _execution_focus_from_item(normalized)
    normalized["internal_focus"] = _compact_text(normalized.get("internal_focus", ""), 220)
    normalized["external_focus"] = _compact_text(normalized.get("external_focus", ""), 220)
    normalized["closure_condition"] = _closure_condition_from_item(normalized)
    normalized["closure_reason"] = _compact_text(normalized.get("closure_reason", ""), 220)
    normalized["reopen_only_if"] = _default_reopen_only_if(normalized)
    normalized["recommended_search_type"] = _infer_item_search_type(
        normalized,
        fallback=str(normalized.get("recommended_search_type", "both") or "both").lower(),
    )
    assigned: list[str] = []
    for name in normalized.get("assigned_perspectives", []) or []:
        clean = str(name or "").strip()
        if clean and clean not in assigned:
            assigned.append(clean)
    normalized["assigned_perspectives"] = assigned[:3]
    normalized["attempt_count"] = int(normalized.get("attempt_count", 0) or 0)
    return normalized


def _dedupe_items(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in items:
        normalized = _compact_agenda_item(item)
        item_id = str(normalized.get("item_id", "") or "").strip()
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        deduped.append(normalized)
    return deduped


def _ensure_external_grounding_item(agenda: dict, perspectives: list[dict]) -> dict:
    agenda = _normalize_agenda(agenda)
    all_items = [
        *agenda.get("active_items", []),
        *agenda.get("partial_items", []),
        *agenda.get("completed_items", []),
        *agenda.get("deferred_items", []),
    ]
    has_external_item = any(
        str(item.get("recommended_search_type", "") or "").lower() in {"external", "both"}
        or "external" in str(item.get("item_id", "") or "")
        for item in all_items
    )
    if has_external_item:
        return agenda

    perspective_names = [str(entry.get("name", "") or "").strip() for entry in perspectives if str(entry.get("name", "") or "").strip()]
    agenda.setdefault("external_grounding_goals", []).append(
        "Add outside context that materially improves interpretation, trust, or actionability."
    )
    agenda.setdefault("active_items", []).append(
        {
            "item_id": "external_context_grounding",
            "title": "External Context Grounding",
            "research_question": (
                "What related work, current field state, governance expectations, or operational evidence from outside the article "
                "would materially change how a pharmacy leader interprets this piece?"
            ),
            "status": "active",
            "priority": "high",
            "why_it_matters": "The run should not finalize without true external grounding.",
            "completion_criteria": "At least one real external research round materially informs the draft.",
            "recommended_search_type": "external",
            "assigned_perspectives": perspective_names or ["Informatics Governance Researcher"],
            "evidence_summary": "",
            "attempt_count": 0,
        }
    )
    agenda["active_items"] = _dedupe_items(agenda.get("active_items", []))
    return agenda


_DELIVERABLE_ONLY_HINTS = (
    "350-550",
    "350–550",
    "create the final newsletter",
    "draft final",
    "newsletter",
    "template constraints",
    "word-count",
    "word count",
)

_ARTIFACT_STATE_VALUES = {"available", "unavailable", "unknown"}


def _looks_like_deliverable_item(item: dict) -> bool:
    text = " ".join(
        str(item.get(key, "") or "")
        for key in ("item_id", "title", "research_question", "completion_criteria", "evidence_summary")
    ).lower()
    return any(hint in text for hint in _DELIVERABLE_ONLY_HINTS)


def _seed_agenda_items_from_proposals(proposals: list[dict]) -> list[dict]:
    items: list[dict] = []
    seen_ids: set[str] = set()
    for index, proposal in enumerate(proposals):
        perspective_name = str(proposal.get("perspective_name", "") or "").strip()
        for question in proposal.get("proposed_questions", []) or []:
            question = _compact_question_focus(question, 220)
            if not question:
                continue
            item_id = re.sub(r"[^a-z0-9]+", "_", str(question).lower()).strip("_")[:64] or f"agenda_item_{index+1}"
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            seeded_item = {
                "item_id": item_id,
                "title": _truncate(question, 84),
                "research_question": question,
                "status": "active",
                "priority": "high" if len(items) < 2 else "medium",
                "why_it_matters": "Seeded from fixed-perspective agenda proposals.",
                "completion_criteria": "Gather enough evidence to materially improve or close this item.",
                "recommended_search_type": "external" if proposal.get("external_grounding_needs") else "internal",
                "assigned_perspectives": [perspective_name] if perspective_name else [],
                "evidence_summary": "",
                "attempt_count": 0,
            }
            items.append(_compact_agenda_item(seeded_item))
            break
        if len(items) >= 3:
            break
    return items


def _sanitize_initial_agenda(agenda: dict, proposals: list[dict], perspectives: list[dict]) -> dict:
    cleaned = _normalize_agenda(agenda)
    for section in ("active_items", "partial_items", "completed_items", "deferred_items"):
        cleaned[section] = _dedupe_items(
            [
                dict(item or {})
                for item in (cleaned.get(section, []) or [])
                if isinstance(item, dict) and not _looks_like_deliverable_item(item)
            ]
        )

    if not cleaned.get("active_items"):
        cleaned["active_items"] = _seed_agenda_items_from_proposals(proposals)

    for section in ("active_items", "partial_items", "completed_items", "deferred_items"):
        cleaned[section] = [_compact_agenda_item(item) for item in (cleaned.get(section, []) or [])]

    cleaned = _ensure_external_grounding_item(cleaned, perspectives)
    return cleaned


def _agenda_snapshot_text(agenda: dict, max_items_per_section: int = 6) -> str:
    agenda = _normalize_agenda(agenda)
    lines: list[str] = []
    goals = agenda.get("overall_goals", []) or []
    if goals:
        lines.append("Overall goals:")
        lines.extend(f"- {_truncate(goal, 180)}" for goal in goals[:5])
    for label, key in (
        ("Active", "active_items"),
        ("Partial", "partial_items"),
        ("Completed", "completed_items"),
        ("Deferred", "deferred_items"),
    ):
        items = agenda.get(key, []) or []
        if not items:
            continue
        lines.append(f"{label}:")
        for item in items[:max_items_per_section]:
            lines.append(
                "- {item_id} | {priority} | {search_type} | {title} | {question} | focus={focus} | artifacts={artifacts} | evidence={evidence}".format(
                    item_id=item.get("item_id", ""),
                    priority=item.get("priority", "medium"),
                    search_type=item.get("recommended_search_type", "both"),
                    title=_truncate(item.get("title", ""), 80),
                    question=_truncate(item.get("research_question", ""), 180),
                    focus=_truncate(
                        item.get("internal_focus", "")
                        or item.get("execution_focus", ""),
                        120,
                    ) or "(none)",
                    artifacts=_artifact_state_text(item),
                    evidence=_truncate(
                        f"mode={item.get('work_mode', 'normal_research')} | {item.get('evidence_summary', '')}",
                        160,
                    ) or "(none)",
                )
            )
    external_goals = agenda.get("external_grounding_goals", []) or []
    if external_goals:
        lines.append("External grounding goals:")
        lines.extend(f"- {_truncate(goal, 180)}" for goal in external_goals[:5])
    if agenda.get("agenda_notes"):
        lines.append("Agenda notes:")
        lines.append(_truncate(agenda.get("agenda_notes", ""), 400))
    return "\n".join(lines).strip() or "(empty agenda)"


def _task_history_text(task_history: list[dict], max_items: int = 8) -> str:
    if not task_history:
        return "(none)"
    lines: list[str] = []
    for entry in task_history[-max_items:]:
        lines.append(
            "- round={round} item={item_id} search_type={search_type} perspectives={perspectives} material={material} summary={summary}".format(
                round=entry.get("round", "?"),
                item_id=entry.get("item_id", ""),
                search_type=entry.get("search_type", "both"),
                perspectives=",".join(entry.get("perspectives", []) or []) or "(none)",
                material=entry.get("material_improvement", False),
                summary=_truncate(entry.get("summary", ""), 220),
            )
        )
    return "\n".join(lines)


def _build_boundary_mode_assignment(item: dict, state: SupervisorState) -> dict:
    compact_item = _compact_agenda_item(item)
    available_names = _perspective_names(state)
    ordered_names = [
        name for name in (compact_item.get("assigned_perspectives", []) or [])
        if name in available_names
    ]
    if not ordered_names:
        ordered_names = available_names[:2]
    internal_name = ordered_names[0] if ordered_names else ""
    external_name = ordered_names[1] if len(ordered_names) > 1 else ""
    if internal_name and not external_name:
        external_name = next(
            (name for name in available_names if name != internal_name),
            "",
        )

    internal_focus = _compact_text(
        compact_item.get("internal_focus", "") or compact_item.get("execution_focus", "") or compact_item.get("research_question", ""),
        220,
    )
    artifact_targets = list(_normalize_artifact_state_map(compact_item.get("artifact_state", {})).keys())
    target_phrase = ", ".join(artifact_targets[:3]) if artifact_targets else "linked public artifact"
    external_focus = _compact_text(
        compact_item.get("external_focus", "") or f"Check whether any public artifact for {target_phrase} closes the missing boundary.",
        220,
    )
    assignments: list[dict] = []
    if internal_name:
        assignments.append(
            {
                "perspective_name": internal_name,
                "worker_brief": _compact_worker_brief(
                    "Primary evidence gap: {focus}. "
                    "Preferred source mode: internal — use only the supplied article and ingested appendix/supplementary content. "
                    "Progress = {closure}.".format(
                        focus=internal_focus or compact_item.get("execution_focus", "") or compact_item.get("research_question", ""),
                        closure="Cite the reported article evidence and note the exact missing pieces that must remain an explicit limitation",
                    ),
                    max_chars=520,
                ),
            }
        )
    if external_name and external_focus:
        assignments.append(
            {
                "perspective_name": external_name,
                "worker_brief": _compact_worker_brief(
                    "Primary evidence gap: whether any public artifact for {target_phrase} closes the missing boundary. "
                    "Preferred source mode: article-first with one tightly scoped public-artifact search only. "
                    "Progress = {closure}.".format(
                        target_phrase=external_focus or target_phrase,
                        closure="Either cite a public artifact that closes the gap or establish cleanly that none is available",
                    ),
                    max_chars=420,
                ),
            }
        )

    if not assignments:
        return _deterministic_assignment_fallback(compact_item, state)

    search_type = "both" if len(assignments) > 1 else "internal"
    return {
        "item_id": str(compact_item.get("item_id", "") or ""),
        "search_type": search_type,
        "assignments": assignments[:2],
        "rationale": (
            "Boundary-mode assignment: one perspective closes the article-reported boundary and one perspective performs a single public-artifact availability check."
            if len(assignments) > 1
            else "Boundary-mode assignment: close the article-reported boundary before reopening broader research."
        ),
    }


def _should_use_boundary_assignment_fallback(item: dict) -> bool:
    compact_item = _compact_agenda_item(item)
    return _normalize_work_mode(compact_item.get("work_mode", "")) == "boundary_with_artifact_check"


def _recent_item_history(task_history: list[dict], item_id: str, max_items: int = 4) -> list[dict]:
    clean_item_id = str(item_id or "").strip()
    if not clean_item_id:
        return []
    return [
        dict(entry or {})
        for entry in (task_history or [])
        if str((entry or {}).get("item_id", "") or "").strip() == clean_item_id
    ][-max_items:]


def _consecutive_item_repeat_count(task_history: list[dict], item_id: str) -> int:
    clean_item_id = str(item_id or "").strip()
    count = 0
    for entry in reversed(task_history or []):
        if str((entry or {}).get("item_id", "") or "").strip() != clean_item_id:
            break
        count += 1
    return count


def _agenda_item_lookup(agenda: dict) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    agenda = _normalize_agenda(agenda)
    for key in ("active_items", "partial_items", "completed_items", "deferred_items"):
        for item in agenda.get(key, []) or []:
            item_id = str(item.get("item_id", "") or "").strip()
            if item_id:
                lookup[item_id] = dict(item)
    return lookup


def _move_item_to_section(agenda: dict, item: dict, section: str) -> None:
    for key in ("active_items", "partial_items", "completed_items", "deferred_items"):
        agenda[key] = [existing for existing in agenda.get(key, []) or [] if str(existing.get("item_id", "") or "") != str(item.get("item_id", "") or "")]
    agenda.setdefault(section, []).append(item)
    agenda[section] = _dedupe_items(agenda.get(section, []) or [])


def _apply_agenda_delta(agenda: dict, delta: dict, assignment: dict | None = None) -> dict:
    updated = _normalize_agenda(agenda)
    lookup = _agenda_item_lookup(updated)
    assigned_item_id = str((assignment or {}).get("item_id", "") or "").strip()
    if assigned_item_id and assigned_item_id in lookup:
        lookup[assigned_item_id]["attempt_count"] = int(lookup[assigned_item_id].get("attempt_count", 0) or 0) + 1

    for update in delta.get("updates", []) or []:
        item_id = str(update.get("item_id", "") or "").strip()
        if not item_id:
            continue
        existing = dict(lookup.get(item_id, {"item_id": item_id, "title": item_id, "research_question": item_id}))
        if update.get("evidence_summary"):
            existing["evidence_summary"] = update["evidence_summary"]
        if update.get("recommended_search_type"):
            existing["recommended_search_type"] = update["recommended_search_type"]
        if update.get("assigned_perspectives"):
            existing["assigned_perspectives"] = update["assigned_perspectives"]
        if update.get("execution_focus"):
            existing["execution_focus"] = update["execution_focus"]
        if update.get("first_turn_query"):
            existing["first_turn_query"] = update["first_turn_query"]
        if update.get("work_mode"):
            existing["work_mode"] = update["work_mode"]
        if update.get("internal_focus"):
            existing["internal_focus"] = update["internal_focus"]
        if update.get("external_focus"):
            existing["external_focus"] = update["external_focus"]
        if update.get("closure_condition"):
            existing["closure_condition"] = update["closure_condition"]
        if update.get("artifact_state"):
            existing["artifact_state"] = _normalize_artifact_state_map(update["artifact_state"])
        if update.get("closure_reason"):
            existing["closure_reason"] = update["closure_reason"]
        if update.get("reopen_only_if"):
            existing["reopen_only_if"] = update["reopen_only_if"]
        new_status = str(update.get("new_status", existing.get("status", "active")) or existing.get("status", "active")).strip().lower()
        existing["status"] = new_status
        lookup[item_id] = _compact_agenda_item(existing)

    for raw_item in delta.get("add_items", []) or []:
        item = _compact_agenda_item(raw_item or {})
        item_id = str(item.get("item_id", "") or "").strip()
        if not item_id:
            continue
        lookup[item_id] = item

    for item_id in delta.get("completed_item_ids", []) or []:
        if item_id in lookup:
            lookup[item_id]["status"] = "completed"
    for item_id in delta.get("deferred_item_ids", []) or []:
        if item_id in lookup:
            lookup[item_id]["status"] = "deferred"

    rebuilt = {
        "overall_goals": updated.get("overall_goals", []),
        "active_items": [],
        "partial_items": [],
        "completed_items": [],
        "deferred_items": [],
        "external_grounding_goals": updated.get("external_grounding_goals", []),
        "agenda_notes": delta.get("agenda_note") or updated.get("agenda_notes", ""),
    }
    status_to_section = {
        "active": "active_items",
        "partial": "partial_items",
        "completed": "completed_items",
        "deferred": "deferred_items",
        "not_answerable": "deferred_items",
    }
    for item in lookup.values():
        section = status_to_section.get(str(item.get("status", "active") or "active").lower(), "active_items")
        _move_item_to_section(rebuilt, item, section)
    return _normalize_agenda(rebuilt)


def _select_highest_value_item(agenda: dict) -> dict | None:
    agenda = _normalize_agenda(agenda)
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    candidates = [*(agenda.get("active_items", []) or []), *(agenda.get("partial_items", []) or [])]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            priority_rank.get(str(item.get("priority", "medium") or "medium").lower(), 1),
            int(item.get("attempt_count", 0) or 0),
        ),
    )[0]


def _deterministic_priority_fallback(state: SupervisorState) -> dict:
    agenda = _normalize_agenda(state.get("research_agenda", {}))
    if len(state.get("research_round_summaries", []) or []) > int(state.get("merged_round_count", 0) or 0):
        return {
            "action": "refine_draft",
            "item_id": "",
            "rationale": "Pending findings should be merged into the draft before more research.",
        }
    item = _select_highest_value_item(agenda)
    if item:
        return {
            "action": "research",
            "item_id": str(item.get("item_id", "") or ""),
            "rationale": "Continue with the highest-value unresolved agenda item.",
        }
    return {
        "action": "finalize_candidate",
        "item_id": "",
        "rationale": "Agenda appears largely resolved; ask the progress gate to confirm finalization.",
    }


def _deterministic_assignment_fallback(item: dict, state: SupervisorState) -> dict:
    compact_item = _compact_agenda_item(item)
    if _normalize_work_mode(compact_item.get("work_mode", "")) == "boundary_with_artifact_check":
        return _build_boundary_mode_assignment(compact_item, state)
    perspective_names = _perspective_names(state)
    assigned = [
        name for name in (compact_item.get("assigned_perspectives", []) or []) if name in perspective_names
    ]
    if not assigned:
        assigned = perspective_names[:2] or ["Basic Facts Researcher"]
    recommended_search_type = str(compact_item.get("recommended_search_type", "both") or "both").lower()
    if recommended_search_type == "internal":
        assigned = assigned[:1] or perspective_names[:1] or ["Basic Facts Researcher"]
    return {
        "item_id": str(compact_item.get("item_id", "") or ""),
        "search_type": recommended_search_type if recommended_search_type in {"internal", "external", "both"} else "both",
        "assignments": [
            {
                "perspective_name": name,
                "worker_brief": _build_executable_worker_brief(compact_item, name, recommended_search_type),
            }
            for name in assigned[:2]
        ],
        "rationale": "Fallback assignment based on agenda metadata.",
    }


def _format_assignment_for_prompt(assignment: dict) -> str:
    if not assignment:
        return "(none)"
    lines = [f"item_id={assignment.get('item_id', '')}", f"search_type={assignment.get('search_type', '')}"]
    for worker in assignment.get("assignments", []) or []:
        lines.append(
            f"- {worker.get('perspective_name', '')}: {_truncate(worker.get('worker_brief', ''), 260)}"
        )
    if assignment.get("rationale"):
        lines.append(f"rationale={assignment.get('rationale', '')}")
    return "\n".join(lines)


def _format_agenda_item_for_assignment(item: dict) -> str:
    item = _compact_agenda_item(item)
    return "\n".join(
        [
            f"item_id={item.get('item_id', '')}",
            f"title={item.get('title', '')}",
            f"priority={item.get('priority', 'medium')}",
            f"search_type={item.get('recommended_search_type', 'both')}",
            f"work_mode={item.get('work_mode', 'normal_research')}",
            f"question={item.get('research_question', '')}",
            f"execution_focus={item.get('execution_focus', '') or '(none)'}",
            f"first_turn_query={item.get('first_turn_query', '') or '(none)'}",
            f"internal_focus={item.get('internal_focus', '') or '(none)'}",
            f"external_focus={item.get('external_focus', '') or '(none)'}",
            f"why_it_matters={item.get('why_it_matters', '') or '(none)'}",
            f"completion={item.get('completion_criteria', '') or '(none)'}",
            f"closure_condition={item.get('closure_condition', '') or '(none)'}",
            f"evidence_state={item.get('evidence_summary', '') or '(none)'}",
            f"artifact_state={_artifact_state_text(item)}",
            f"closure_reason={item.get('closure_reason', '') or '(none)'}",
            f"reopen_only_if={item.get('reopen_only_if', '') or '(none)'}",
            f"assigned_perspectives={', '.join(item.get('assigned_perspectives', []) or []) or '(none)'}",
            f"attempt_count={item.get('attempt_count', 0)}",
        ]
    )


def _format_recent_item_history(task_history: list[dict], item_id: str, max_items: int = 3) -> str:
    recent = _recent_item_history(task_history, item_id, max_items=max_items)
    if not recent:
        return "(none)"
    lines: list[str] = []
    for entry in recent:
        lines.append(
            "- search_type={search_type} perspectives={perspectives} material={material} novelty={novelty} summary={summary}".format(
                search_type=entry.get("search_type", "both"),
                perspectives=",".join(entry.get("perspectives", []) or []) or "(none)",
                material=entry.get("material_improvement", False),
                novelty=entry.get("novelty", {}).get("novel_entries", 0) if isinstance(entry.get("novelty"), dict) else "?",
                summary=_compact_text(entry.get("summary", ""), 220),
            )
        )
    return "\n".join(lines)


def _compact_worker_brief(text: str, max_chars: int = 420, max_sentences: int = 3) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    normalized = re.sub(
        r"(?i)\b(?:deliverables?|deliver:|ignore\b|do not\b|immediate next steps?:)\b.*$",
        "",
        normalized,
    ).strip(" ,;:")
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    kept: list[str] = []
    total = 0
    for sentence in sentences:
        clean = sentence.strip()
        if not clean:
            continue
        projected = total + len(clean) + (1 if kept else 0)
        if projected > max_chars:
            break
        kept.append(clean)
        total = projected
        if len(kept) >= max_sentences:
            break
    compacted = " ".join(kept) or _compact_text(normalized, max_chars)
    return _compact_text(compacted, max_chars)


def _postprocess_assignment(item: dict, assignment: dict, state: SupervisorState) -> dict:
    normalized = dict(assignment or {})
    compact_item = _compact_agenda_item(item)
    preserve_boundary_briefs = _should_use_boundary_assignment_fallback(compact_item)
    normalized["search_type"] = _infer_item_search_type(
        {**compact_item, "recommended_search_type": normalized.get("search_type", compact_item.get("recommended_search_type", "both"))},
        fallback=str(compact_item.get("recommended_search_type", "both") or "both").lower(),
    )
    available_names = _perspective_names(state)
    item_assigned = [
        name for name in (compact_item.get("assigned_perspectives", []) or [])
        if name in available_names
    ]
    recent_history = _recent_item_history(state.get("task_history", []) or [], item.get("item_id", ""), max_items=4)
    recently_used = {
        str(name or "").strip()
        for entry in recent_history
        for name in (entry.get("perspectives", []) or [])
        if str(name or "").strip()
    }

    cleaned_assignments: list[dict] = []
    seen: set[str] = set()
    for raw_worker in normalized.get("assignments", []) or []:
        perspective_name = str((raw_worker or {}).get("perspective_name", "") or "").strip()
        if not perspective_name or perspective_name in seen or perspective_name not in available_names:
            continue
        seen.add(perspective_name)
        search_type = str(normalized.get("search_type", compact_item.get("recommended_search_type", "both")) or "both").lower()
        raw_brief = str((raw_worker or {}).get("worker_brief", "") or "")
        worker_brief = (
            _compact_worker_brief(raw_brief, max_chars=520, max_sentences=4)
            if preserve_boundary_briefs and raw_brief.strip()
            else _enforce_worker_brief_contract(
                compact_item,
                perspective_name,
                search_type,
                raw_brief,
            )
        )
        cleaned_assignments.append(
            {
                "perspective_name": perspective_name,
                "worker_brief": worker_brief,
            }
        )

    if not cleaned_assignments:
        fallback = _deterministic_assignment_fallback(item, state)
        cleaned_assignments = fallback.get("assignments", []) or []

    attempt_count = int(compact_item.get("attempt_count", 0) or 0)
    if attempt_count >= 2 and len(cleaned_assignments) < 2:
        candidate_pool = item_assigned or available_names
        candidate = next(
            (
                name
                for name in candidate_pool
                if name not in seen and name not in recently_used
            ),
            None,
        ) or next((name for name in candidate_pool if name not in seen), None)
        if candidate:
            cleaned_assignments.append(
                {
                    "perspective_name": candidate,
                    "worker_brief": _build_executable_worker_brief(
                        compact_item,
                        candidate,
                        str(normalized.get("search_type", compact_item.get("recommended_search_type", "both")) or "both").lower(),
                    ),
                }
            )

    normalized["assignments"] = cleaned_assignments[:2]
    normalized["rationale"] = _compact_text(normalized.get("rationale", ""), 260)
    return normalized


def _compose_research_topic(item: dict) -> str:
    item = _compact_agenda_item(item)
    research_focus = _compact_question_focus(
        item.get("execution_focus", "") or item.get("research_question", ""),
        max_chars=180,
    ) or item.get("execution_focus", "") or item.get("research_question", "")
    closure = _compact_text(
        item.get("closure_condition", "") or item.get("completion_criteria", ""),
        140,
    )
    parts = [
        f"Agenda item: {item.get('title', '')}",
        f"Research question: {research_focus}",
    ]
    if closure:
        parts.append(f"Close when: {closure}")
    parts.append("Return only evidence that closes this question.")
    return "\n".join(parts)


def _constrain_agenda_delta(state: SupervisorState, delta: dict, latest_round: dict) -> dict:
    constrained = dict(delta or {})
    impact = (latest_round or {}).get("impact", {}) or {}
    agenda = _normalize_agenda(state.get("research_agenda", {}))
    open_items = len(agenda.get("active_items", []) or []) + len(agenda.get("partial_items", []) or [])
    adds = [dict(item or {}) for item in (constrained.get("add_items", []) or []) if isinstance(item, dict)]

    should_drop_adds = (
        open_items >= 5
        or (impact.get("newly_supported", 0) == 0 and impact.get("status_upgrades", 0) == 0)
    )
    if should_drop_adds and adds:
        constrained["add_items"] = []
        note = str(constrained.get("agenda_note", "") or "").strip()
        drop_note = "Dropped low-confidence agenda expansion to keep the agenda converging."
        constrained["agenda_note"] = f"{note} {drop_note}".strip() if note else drop_note

    constrained["updates"] = [dict(update or {}) for update in (constrained.get("updates", []) or []) if isinstance(update, dict)]
    return constrained


def _stabilize_agenda_delta(state: SupervisorState, delta: dict, latest_round: dict) -> dict:
    stabilized = dict(delta or {})
    lookup = _agenda_item_lookup(state.get("research_agenda", {}))
    latest_item_id = str((latest_round or {}).get("item_id", "") or "").strip()
    stabilized_updates: list[dict] = []
    for raw_update in stabilized.get("updates", []) or []:
        update = dict(raw_update or {})
        item_id = str(update.get("item_id", "") or "").strip()
        existing = dict(lookup.get(item_id) or {})
        if not existing:
            stabilized_updates.append(update)
            continue
        merged_artifact_state = {
            **_normalize_artifact_state_map(existing.get("artifact_state", {})),
            **_normalize_artifact_state_map(update.get("artifact_state", {})),
        }
        if merged_artifact_state:
            update["artifact_state"] = merged_artifact_state
        if not update.get("execution_focus"):
            update["execution_focus"] = existing.get("execution_focus", "") or _execution_focus_from_item(existing)
        if not update.get("work_mode"):
            update["work_mode"] = existing.get("work_mode", "") or "normal_research"
        if not update.get("internal_focus"):
            update["internal_focus"] = existing.get("internal_focus", "")
        if not update.get("external_focus"):
            update["external_focus"] = existing.get("external_focus", "")
        if not update.get("closure_condition"):
            update["closure_condition"] = existing.get("closure_condition", "") or _closure_condition_from_item(existing)
        if not update.get("reopen_only_if"):
            update["reopen_only_if"] = existing.get("reopen_only_if", "") or _default_reopen_only_if(existing)
        proposed_status = str(update.get("new_status", existing.get("status", "active")) or existing.get("status", "active")).lower()
        if (
            _item_is_closed(existing)
            and proposed_status in {"active", "partial"}
            and item_id == latest_item_id
            and not _round_supports_reopen(existing, latest_round, merged_artifact_state)
        ):
            update["new_status"] = existing.get("status", "completed")
            update["closure_reason"] = existing.get("closure_reason", "") or "reopen_blocked_without_new_evidence"
            update["reopen_only_if"] = existing.get("reopen_only_if", "") or _default_reopen_only_if(existing)
        stabilized_updates.append(update)
    stabilized["updates"] = stabilized_updates
    return stabilized


def _blocked_item_priority_override(item: dict) -> dict | None:
    compact_item = _compact_agenda_item(item)
    work_mode = _normalize_work_mode(compact_item.get("work_mode", ""))
    if work_mode in {"limitation_to_draft", "close_unavailable"}:
        return {
            "action": "refine_draft",
            "item_id": str(compact_item.get("item_id", "") or ""),
            "rationale": (
                "The agenda marks this item as ready to carry into the draft rather than continue research. "
                "Refine the newsletter with the bounded limitation instead of reopening the item."
            ),
        }
    if int(compact_item.get("attempt_count", 0) or 0) < 2:
        return None
    if not _item_is_blocked_by_unavailable_dependencies(compact_item):
        return None
    return {
        "action": "refine_draft",
        "item_id": str(compact_item.get("item_id", "") or ""),
        "rationale": (
            "Further autonomous searching is blocked by unavailable artifacts or dependencies. "
            "Carry the limitation into the draft instead of reopening the same dependency chase."
        ),
    }


def _apply_convergence_policy(state: SupervisorState) -> tuple[dict, str]:
    latest_round = state.get("latest_research_round", {}) or {}
    item_id = str(latest_round.get("item_id", "") or "").strip()
    if not item_id:
        return _normalize_agenda(state.get("research_agenda", {})), ""

    agenda = _normalize_agenda(state.get("research_agenda", {}))
    lookup = _agenda_item_lookup(agenda)
    item = dict(lookup.get(item_id) or {})
    if not item:
        return agenda, ""

    impact = latest_round.get("impact", {}) or {}
    attempts = int(item.get("attempt_count", 0) or 0)
    recent_history = _recent_item_history(state.get("task_history", []) or [], item_id, max_items=4)
    summary = _compact_text(
        latest_round.get("round_summary", {}).get("summary", "") or latest_round.get("findings_packet", ""),
        280,
    )
    work_mode = _normalize_work_mode(item.get("work_mode", ""))
    if work_mode == "close_unavailable" or (item.get("status") == "completed" and _item_all_artifacts_unavailable(item)):
        item["status"] = "completed"
        item["evidence_summary"] = summary or item.get("evidence_summary", "") or "Resolved as unavailable in public sources."
        item["artifact_state"] = _default_artifact_state(item)
        item["closure_reason"] = item.get("closure_reason", "") or "resolved_as_unavailable"
        item["reopen_only_if"] = item.get("reopen_only_if", "") or _default_reopen_only_if(item)
        rebuilt = _apply_agenda_delta(
            agenda,
            {
                "updates": [{
                    "item_id": item_id,
                    "new_status": "completed",
                    "evidence_summary": item["evidence_summary"],
                    "recommended_search_type": item.get("recommended_search_type", "external"),
                    "assigned_perspectives": item.get("assigned_perspectives", []),
                    "artifact_state": item.get("artifact_state", {}),
                    "closure_reason": item.get("closure_reason", ""),
                    "reopen_only_if": item.get("reopen_only_if", ""),
                }],
                "completed_item_ids": [item_id],
                "deferred_item_ids": [],
                "add_items": [],
                "agenda_note": "Closed availability item because explicit agenda state marked the required public artifact unavailable.",
            },
            {"item_id": item_id},
        )
        return rebuilt, "resolved_as_unavailable"

    if work_mode == "limitation_to_draft" or item.get("status") == "deferred":
        item["status"] = "deferred"
        item["evidence_summary"] = summary or item.get("evidence_summary", "") or (
            "Carry this as an explicit unresolved limitation unless a new supported evidence path appears."
        )
        item["artifact_state"] = _default_artifact_state(item)
        item["closure_reason"] = item.get("closure_reason", "") or "reported_boundary_reached"
        item["reopen_only_if"] = item.get("reopen_only_if", "") or _default_reopen_only_if(item)
        rebuilt = _apply_agenda_delta(
            agenda,
            {
                "updates": [{
                    "item_id": item_id,
                    "new_status": "deferred",
                    "evidence_summary": item["evidence_summary"],
                    "recommended_search_type": item.get("recommended_search_type", "both"),
                    "assigned_perspectives": item.get("assigned_perspectives", []),
                    "artifact_state": item.get("artifact_state", {}),
                    "closure_reason": item.get("closure_reason", ""),
                    "reopen_only_if": item.get("reopen_only_if", ""),
                }],
                "completed_item_ids": [],
                "deferred_item_ids": [item_id],
                "add_items": [],
                "agenda_note": "Deferred item because explicit agenda state says the remaining gap should now be carried into the draft.",
            },
            {"item_id": item_id},
        )
        return rebuilt, item.get("closure_reason", "") or "reported_boundary_reached"

    low_progress = (
        attempts >= 3
        and impact.get("newly_supported", 0) == 0
        and impact.get("status_upgrades", 0) == 0
    )
    repeated_same_path = len(recent_history) >= 3 and len({entry.get("search_type", "") for entry in recent_history[-3:]}) == 1
    if low_progress or repeated_same_path:
        item["status"] = "deferred"
        item["evidence_summary"] = summary or (
            "Further autonomous searching appears low-yield; keep this as an explicit unresolved limitation unless a new artifact becomes available."
        )
        item["artifact_state"] = _default_artifact_state(item)
        item["closure_reason"] = "deferred_low_yield_item"
        item["reopen_only_if"] = _default_reopen_only_if(item)
        rebuilt = _apply_agenda_delta(
            agenda,
            {
                "updates": [{
                    "item_id": item_id,
                    "new_status": "deferred",
                    "evidence_summary": item["evidence_summary"],
                    "recommended_search_type": item.get("recommended_search_type", "external"),
                    "assigned_perspectives": item.get("assigned_perspectives", []),
                    "artifact_state": item.get("artifact_state", {}),
                    "closure_reason": item.get("closure_reason", ""),
                    "reopen_only_if": item.get("reopen_only_if", ""),
                }],
                "completed_item_ids": [],
                "deferred_item_ids": [item_id],
                "add_items": [],
                "agenda_note": "Deferred repeated low-yield item so the run can converge and carry the uncertainty into the draft.",
            },
            {"item_id": item_id},
        )
        return rebuilt, "deferred_low_yield_item"

    return agenda, ""


def _compile_pending_closure_notes(
    agenda_update_log: list[dict],
    agenda: dict,
    max_items: int = 4,
    max_chars: int = 1600,
) -> str:
    lookup = _agenda_item_lookup(agenda)
    lines: list[str] = []
    seen: set[str] = set()

    for entry in agenda_update_log or []:
        item_id = str(entry.get("item_id", "") or "").strip()
        if not item_id or item_id in seen:
            continue
        note = _compact_text(entry.get("agenda_note", ""), 220)
        completed = set(str(value or "").strip() for value in (entry.get("completed_item_ids", []) or []))
        deferred = set(str(value or "").strip() for value in (entry.get("deferred_item_ids", []) or []))
        if not note and item_id not in completed and item_id not in deferred:
            continue

        item = dict(lookup.get(item_id) or {})
        title = _compact_text(item.get("title", "") or item_id, 96) or item_id
        evidence_summary = _compact_text(item.get("evidence_summary", ""), 220)

        if item_id in completed or "resolved_as_unavailable" in note:
            status_text = "closed as unavailable or sufficiently resolved"
        elif item_id in deferred or "deferred" in note:
            status_text = "deferred as low-yield"
        else:
            status_text = "agenda state updated"

        detail = evidence_summary or note or "Carry this as an explicit limitation rather than continuing autonomous search."
        lines.append(f"- {title}: {status_text}. {detail}")
        seen.add(item_id)
        if len(lines) >= max_items:
            break

    if not lines:
        return ""

    packet = "Closure / convergence notes:\n" + "\n".join(lines)
    return _truncate(packet, max_chars)


async def select_fixed_perspectives(state: SupervisorState):
    if state.get("research_perspectives"):
        logger.info("SELECT_FIXED_PERSPECTIVES: reusing existing perspectives")
        return {"supervisor_phase": "perspectives_selected"}

    logger.info("=" * 80)
    logger.info("SELECT_FIXED_PERSPECTIVES START")
    start_time = time.perf_counter()

    research_brief = str(state.get("research_brief", "") or "")
    article_summary = str(state.get("article_summary", "") or "")
    draft_report = str(state.get("draft_report", "") or "")
    document_profile = _document_profile_from_state(state)

    if DETERMINISTIC_SUPERVISOR:
        perspectives = _default_perspectives(
            article_summary,
            draft_report,
            newsletter_template=str(state.get("newsletter_template", "") or ""),
        )
        elapsed = time.perf_counter() - start_time
        logger.info(
            "SELECT_FIXED_PERSPECTIVES: using deterministic perspective roster | perspectives=%s | time=%.2fs",
            [entry.get("name", "") for entry in perspectives],
            elapsed,
        )
        logger.info("=" * 80)
        return {
            "research_perspectives": perspectives,
            "storm_perspectives": [entry.get("name", "") for entry in perspectives if entry.get("name")],
            "perspective_profiles": _perspective_profile_map(perspectives),
            "storm_perspective_research_plans": {
                entry.get("name", ""): entry.get("description", "")
                for entry in perspectives
                if entry.get("name")
            },
            "supervisor_phase": "perspectives_selected",
        }

    prompt = fixed_perspective_selection_prompt.format(
        research_brief=research_brief,
        article_summary=article_summary,
        document_profile=format_document_profile(document_profile),
        draft_report=draft_report,
        date=get_today_str(),
    )

    try:
        roster = await _invoke_supervisor_structured(
            FixedPerspectiveRoster,
            prompt,
            SUPERVISOR_FIXED_PERSPECTIVE_TIMEOUT_SECONDS,
        )
        perspectives = [entry.model_dump() for entry in roster.perspectives][:3]
        if len(perspectives) != 3:
            raise ValueError(f"Expected 3 perspectives, got {len(perspectives)}")
    except asyncio.TimeoutError:
        logger.warning(
            "SELECT_FIXED_PERSPECTIVES: timed out after %.1fs, using fallback",
            SUPERVISOR_FIXED_PERSPECTIVE_TIMEOUT_SECONDS,
        )
        perspectives = _default_perspectives(
            article_summary,
            draft_report,
            newsletter_template=str(state.get("newsletter_template", "") or ""),
        )
    except Exception as exc:
        logger.warning("SELECT_FIXED_PERSPECTIVES: model selection failed, using fallback: %s", exc)
        perspectives = _default_perspectives(
            article_summary,
            draft_report,
            newsletter_template=str(state.get("newsletter_template", "") or ""),
        )

    elapsed = time.perf_counter() - start_time
    logger.info(
        "SELECT_FIXED_PERSPECTIVES COMPLETE | perspectives=%s | time=%.2fs",
        [entry.get("name", "") for entry in perspectives],
        elapsed,
    )
    logger.info("=" * 80)

    return {
        "research_perspectives": perspectives,
        "storm_perspectives": [entry.get("name", "") for entry in perspectives if entry.get("name")],
        "perspective_profiles": _perspective_profile_map(perspectives),
        "storm_perspective_research_plans": {
            entry.get("name", ""): entry.get("description", "")
            for entry in perspectives
            if entry.get("name")
        },
        "supervisor_phase": "perspectives_selected",
    }


async def initialize_research_agenda(state: SupervisorState):
    if state.get("research_agenda"):
        logger.info("INITIALIZE_RESEARCH_AGENDA: reusing existing agenda")
        return {"supervisor_phase": "research_program_initialized"}
    if not state.get("research_perspectives"):
        raise ValueError("research_perspectives must be set before initialize_research_agenda")

    if DETERMINISTIC_SUPERVISOR:
        logger.info("INITIALIZE_RESEARCH_AGENDA: using deterministic source-grounded agenda")
        agenda = _build_source_grounded_initial_agenda(state)
        return {
            "perspective_proposals": [],
            "research_agenda": agenda,
            "supervisor_phase": "research_program_initialized",
            "source_mix_summary": state.get("source_mix_summary", "") or "No meaningful research evidence has been recorded yet.",
        }

    logger.info("=" * 80)
    logger.info("INITIALIZE_RESEARCH_AGENDA START")
    start_time = time.perf_counter()

    research_brief = str(state.get("research_brief", "") or "")
    article_summary = str(state.get("article_summary", "") or "")
    draft_report = str(state.get("draft_report", "") or "")
    document_profile = _document_profile_from_state(state)
    perspectives = [dict(entry or {}) for entry in (state.get("research_perspectives", []) or [])]

    proposal_model = _structured_supervisor_model(PerspectiveProposal)

    def _proposal_prompt(perspective: dict, compact: bool = False) -> str:
        return perspective_agenda_proposal_prompt.format(
            perspective_name=perspective.get("name", ""),
            perspective_description=_truncate(perspective.get("description", ""), 260 if compact else 520),
            focus_areas=", ".join(perspective.get("focus_areas", []) or []) or "(none)",
            research_brief=_truncate(research_brief, 1800 if compact else 4200),
            article_summary=_truncate(article_summary, 2200 if compact else 5200),
            document_profile=format_document_profile(document_profile),
            draft_report=_truncate(draft_report, 1400 if compact else 3600),
            date=get_today_str(),
        )

    async def _proposal_for(perspective: dict) -> dict:
        for attempt_index, compact in enumerate((False, True), start=1):
            try:
                response = await asyncio.wait_for(
                    proposal_model.ainvoke([HumanMessage(content=_proposal_prompt(perspective, compact=compact))]),
                    timeout=AGENDA_PROPOSAL_TIMEOUT_SECONDS if not compact else min(
                        AGENDA_PROPOSAL_TIMEOUT_SECONDS,
                        AGENDA_PROPOSAL_RETRY_TIMEOUT_SECONDS,
                    ),
                )
                return response.model_dump()
            except asyncio.TimeoutError:
                logger.warning(
                    "INITIALIZE_RESEARCH_AGENDA: proposal timed out for %s on attempt %d%s",
                    perspective.get("name", ""),
                    attempt_index,
                    " (compact retry)" if compact else "",
                )
                continue
            except Exception as exc:
                logger.warning(
                    "INITIALIZE_RESEARCH_AGENDA: proposal failed for %s on attempt %d: %s",
                    perspective.get("name", ""),
                    attempt_index,
                    exc,
                )
                break

        return {
            "perspective_name": perspective.get("name", ""),
            "proposed_questions": [
                f"What unresolved question matters most to {perspective.get('name', '')}?"
            ],
            "external_grounding_needs": [],
            "high_value_risks": list(perspective.get("focus_areas", []) or [])[:2],
        }

    proposals = []
    for perspective in perspectives:
        proposals.append(await _proposal_for(perspective))

    agenda_prompt = global_research_agenda_initialization_prompt.format(
        research_brief=research_brief,
        article_summary=article_summary,
        document_profile=format_document_profile(document_profile),
        draft_report=draft_report,
        perspectives=_format_perspectives_for_prompt(perspectives),
        perspective_proposals=_format_proposals_for_prompt(proposals),
        date=get_today_str(),
    )
    try:
        agenda = (
            await _invoke_supervisor_structured(
                GlobalResearchAgenda,
                agenda_prompt,
                SUPERVISOR_AGENDA_INIT_TIMEOUT_SECONDS,
            )
        ).model_dump()
    except asyncio.TimeoutError:
        logger.warning(
            "INITIALIZE_RESEARCH_AGENDA: agenda init timed out after %.1fs, using fallback",
            SUPERVISOR_AGENDA_INIT_TIMEOUT_SECONDS,
        )
        agenda = {
            "overall_goals": [
                "Strengthen the newsletter with article-grounded facts and real external context.",
            ],
            "active_items": _seed_agenda_items_from_proposals(proposals),
            "partial_items": [],
            "completed_items": [],
            "deferred_items": [],
            "external_grounding_goals": [],
            "agenda_notes": "Fallback agenda initialized from fixed-perspective proposals.",
        }
    except Exception as exc:
        logger.warning("INITIALIZE_RESEARCH_AGENDA: agenda init failed, using fallback: %s", exc)
        agenda = {
            "overall_goals": [
                "Strengthen the newsletter with article-grounded facts and real external context.",
            ],
            "active_items": _seed_agenda_items_from_proposals(proposals),
            "partial_items": [],
            "completed_items": [],
            "deferred_items": [],
            "external_grounding_goals": [],
            "agenda_notes": "Fallback agenda initialized from fixed-perspective proposals.",
        }

    agenda = _sanitize_initial_agenda(agenda, proposals, perspectives)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "INITIALIZE_RESEARCH_AGENDA COMPLETE | active_items=%d | time=%.2fs",
        len(agenda.get("active_items", []) or []),
        elapsed,
    )
    logger.info("=" * 80)

    return {
        "perspective_proposals": proposals,
        "research_agenda": agenda,
        "supervisor_phase": "research_program_initialized",
        "source_mix_summary": state.get("source_mix_summary", "") or "No meaningful research evidence has been recorded yet.",
    }


async def prioritize_research_work(state: SupervisorState):
    logger.info("PRIORITIZE_RESEARCH_WORK START")
    if not state.get("research_agenda"):
        raise ValueError("research_agenda must be initialized before prioritize_research_work")

    if DETERMINISTIC_SUPERVISOR:
        decision = _deterministic_priority_decision(state)
        logger.info("PRIORITIZE_RESEARCH_WORK COMPLETE | action=%s | item_id=%s", decision.get("action", ""), decision.get("item_id", ""))
        return {
            "latest_priority_decision": decision,
            "supervisor_phase": f"priority:{decision.get('action', '')}",
        }

    fallback = _deterministic_priority_fallback(state)
    pending_findings = len(state.get("research_round_summaries", []) or []) > int(state.get("merged_round_count", 0) or 0)

    agenda_snapshot = _agenda_snapshot_text(state.get("research_agenda", {}))
    task_history_text = _task_history_text(state.get("task_history", []) or [])
    draft_report = _truncate(state.get("draft_report", ""), 3500)
    source_mix_summary = state.get("source_mix_summary", "") or "(none)"

    prompt = research_priority_planner_prompt.format(
        agenda_snapshot=agenda_snapshot,
        task_history=task_history_text,
        source_mix_summary=source_mix_summary,
        draft_report=draft_report,
        date=get_today_str(),
    )
    try:
        decision = (
            await _invoke_supervisor_structured(
                PriorityDecision,
                prompt,
                SUPERVISOR_PRIORITY_TIMEOUT_SECONDS,
            )
        ).model_dump()
    except asyncio.TimeoutError:
        logger.warning(
            "PRIORITIZE_RESEARCH_WORK: planner timed out after %.1fs, using fallback",
            SUPERVISOR_PRIORITY_TIMEOUT_SECONDS,
        )
        decision = fallback
    except Exception as exc:
        logger.warning("PRIORITIZE_RESEARCH_WORK: planner failed, using fallback: %s", exc)
        decision = fallback

    if pending_findings:
        decision = {
            "action": "refine_draft",
            "item_id": "",
            "rationale": "Pending findings should be merged into the draft before more research.",
        }
    elif decision.get("action") == "refine_draft":
        item = _select_highest_value_item(state.get("research_agenda", {}))
        if item:
            decision = {
                "action": "research",
                "item_id": str(item.get("item_id", "") or ""),
                "rationale": "No pending findings exist, so the next action must dispatch research work.",
            }
        else:
            decision = {
                "action": "finalize_candidate",
                "item_id": "",
                "rationale": "No unresolved research items remain; ask the progress gate whether finalization is justified.",
            }

    if decision.get("action") == "research" and not str(decision.get("item_id", "") or "").strip():
        item = _select_highest_value_item(state.get("research_agenda", {}))
        if item:
            decision["item_id"] = str(item.get("item_id", "") or "")

    if decision.get("action") == "research":
        selected_item = _agenda_item_lookup(state.get("research_agenda", {})).get(str(decision.get("item_id", "") or "").strip())
        if selected_item:
            override = _blocked_item_priority_override(selected_item)
            if override:
                decision = override

    repeated_item_id = str(decision.get("item_id", "") or "").strip()
    if (
        not pending_findings
        and not state.get("external_grounding_completed", False)
        and repeated_item_id
        and repeated_item_id != "external_context_grounding"
    ):
        repeated_count = _consecutive_item_repeat_count(state.get("task_history", []) or [], repeated_item_id)
        external_item = _agenda_item_lookup(state.get("research_agenda", {})).get("external_context_grounding")
        if repeated_count >= 2 and external_item:
            decision = {
                "action": "research",
                "item_id": "external_context_grounding",
                "rationale": (
                    "Repeated internal work on one item is no longer the highest-leverage next step. "
                    "Do a true external-grounding round now so the run gains outside context instead of re-asking the same internal question."
                ),
            }

    if not state.get("external_grounding_completed", False) and decision.get("action") == "finalize_candidate":
        external_item = _agenda_item_lookup(state.get("research_agenda", {})).get("external_context_grounding")
        if external_item:
            decision = {
                "action": "research",
                "item_id": "external_context_grounding",
                "rationale": "External grounding is mandatory before finalization.",
            }
        else:
            decision = fallback

    logger.info("PRIORITIZE_RESEARCH_WORK COMPLETE | action=%s | item_id=%s", decision.get("action", ""), decision.get("item_id", ""))
    return {
        "latest_priority_decision": decision,
        "supervisor_phase": f"priority:{decision.get('action', '')}",
    }


def route_after_priority(state: SupervisorState) -> Literal["prepare_research_assignment", "draft_refinement_agent", "progress_gate"]:
    action = str((state.get("latest_priority_decision", {}) or {}).get("action", "research") or "research")
    if action == "refine_draft":
        return "draft_refinement_agent"
    if action == "finalize_candidate":
        return "progress_gate"
    return "prepare_research_assignment"


async def prepare_research_assignment(state: SupervisorState):
    logger.info("PREPARE_RESEARCH_ASSIGNMENT START")
    decision = state.get("latest_priority_decision", {}) or {}
    item_id = str(decision.get("item_id", "") or "").strip()
    lookup = _agenda_item_lookup(state.get("research_agenda", {}))
    item = lookup.get(item_id) or _select_highest_value_item(state.get("research_agenda", {}))
    if not item:
        logger.warning("PREPARE_RESEARCH_ASSIGNMENT: no agenda item available")
        return {"latest_assignment": {}, "supervisor_phase": "assignment:none"}
    item = _compact_agenda_item(item)
    item_work_mode = _normalize_work_mode(item.get("work_mode", ""))
    if item.get("status") in {"completed", "deferred"} or item_work_mode in {"limitation_to_draft", "close_unavailable"}:
        logger.info(
            "PREPARE_RESEARCH_ASSIGNMENT: skipping non-researchable item | item_id=%s | status=%s | work_mode=%s",
            item.get("item_id", ""),
            item.get("status", ""),
            item_work_mode,
        )
        return {
            "latest_assignment": {},
            "supervisor_phase": f"assignment:skipped:{item.get('item_id', '')}",
        }

    task_history_text = _task_history_text(state.get("task_history", []) or [])
    recent_item_history = _format_recent_item_history(state.get("task_history", []) or [], item.get("item_id", ""))
    perspectives_text = _format_perspectives_for_prompt(state.get("research_perspectives", []) or [])
    draft_context, _ = _build_targeted_draft_context(
        item.get("research_question", ""),
        str(state.get("draft_report", "") or ""),
        2600,
    )

    if DETERMINISTIC_SUPERVISOR:
        assignment = _deterministic_assignment_fallback(item, state)
    elif _should_use_boundary_assignment_fallback(item):
        logger.info(
            "PREPARE_RESEARCH_ASSIGNMENT: using boundary-mode deterministic assignment | item_id=%s",
            item.get("item_id", ""),
        )
        assignment = _build_boundary_mode_assignment(item, state)
    else:
        prompt = research_assignment_prompt.format(
            agenda_item=_format_agenda_item_for_assignment(item),
            document_profile=format_document_profile(_document_profile_from_state(state)),
            perspectives=perspectives_text,
            task_history=task_history_text,
            recent_item_history=recent_item_history,
            draft_report=draft_context,
            date=get_today_str(),
        )
        try:
            assignment = (
                await _invoke_supervisor_structured(
                    AssignmentDecision,
                    prompt,
                    SUPERVISOR_ASSIGNMENT_TIMEOUT_SECONDS,
                )
            ).model_dump()
        except asyncio.TimeoutError:
            logger.warning(
                "PREPARE_RESEARCH_ASSIGNMENT: timed out after %.1fs, using fallback",
                SUPERVISOR_ASSIGNMENT_TIMEOUT_SECONDS,
            )
            assignment = _deterministic_assignment_fallback(item, state)
        except Exception as exc:
            logger.warning("PREPARE_RESEARCH_ASSIGNMENT: model failed, using fallback: %s", exc)
            assignment = _deterministic_assignment_fallback(item, state)

    search_type = str(assignment.get("search_type", item.get("recommended_search_type", "both")) or item.get("recommended_search_type", "both")).lower()
    if search_type not in {"internal", "external", "both"}:
        search_type = str(item.get("recommended_search_type", "both") or "both").lower()
    item_search_type = _infer_item_search_type(item, fallback=str(item.get("recommended_search_type", "both") or "both").lower())
    if item_search_type == "external" and search_type == "internal" and _item_requires_public_artifact(_compact_agenda_item(item)):
        search_type = "both"
    search_type = _infer_item_search_type({**dict(item or {}), "recommended_search_type": search_type}, fallback=item_search_type)
    assignment["search_type"] = search_type
    assignment["item_id"] = str(item.get("item_id", "") or "")

    if not assignment.get("assignments"):
        assignment = _deterministic_assignment_fallback(item, state)
    assignment = _postprocess_assignment(item, assignment, state)

    logger.info(
        "PREPARE_RESEARCH_ASSIGNMENT COMPLETE | item_id=%s | search_type=%s | workers=%d",
        assignment.get("item_id", ""),
        assignment.get("search_type", ""),
        len(assignment.get("assignments", []) or []),
    )
    return {
        "latest_assignment": assignment,
        "supervisor_phase": f"assignment:{assignment.get('item_id', '')}",
    }


async def execute_research_assignments(state: SupervisorState):
    assignment = state.get("latest_assignment", {}) or {}
    item_id = str(assignment.get("item_id", "") or "").strip()
    lookup = _agenda_item_lookup(state.get("research_agenda", {}))
    item = lookup.get(item_id)
    if not assignment or not item:
        logger.warning("EXECUTE_RESEARCH_ASSIGNMENTS: no assignment/item available")
        return {"latest_research_round": {}, "supervisor_phase": "research:none"}

    search_type = str(assignment.get("search_type", item.get("recommended_search_type", "both")) or item.get("recommended_search_type", "both")).lower()
    workers = [dict(worker or {}) for worker in (assignment.get("assignments", []) or [])[:3]]
    if not workers:
        logger.warning("EXECUTE_RESEARCH_ASSIGNMENTS: assignment has no workers")
        return {"latest_research_round": {}, "supervisor_phase": "research:none"}

    logger.info("=" * 80)
    logger.info(
        "EXECUTE_RESEARCH_ASSIGNMENTS START | item_id=%s | search_type=%s | workers=%d",
        item_id,
        search_type,
        len(workers),
    )
    start_time = time.perf_counter()

    existing_entries = canonicalize_evidence_ledger(state.get("evidence_ledger", []) or [])
    existing_events = [dict(entry or {}) for entry in (state.get("retrieval_events", []) or [])]
    running_entries = list(existing_entries)
    running_events = list(existing_events)

    draft_context, selected_headers = _build_targeted_draft_context(
        item.get("research_question", ""),
        str(state.get("draft_report", "") or ""),
        12000,
    )
    logger.info(
        "  targeted_draft_context: %d chars | sections=%s",
        len(draft_context),
        selected_headers or ["(empty)"],
    )
    scoped_evidence_ledger = [
        entry
        for entry in canonicalize_evidence_ledger(state.get("evidence_ledger", []) or [])
        if str(entry.get("gap_id", "") or "").strip() == item_id
    ][-8:]
    scoped_retrieval_events = [
        dict(event or {})
        for event in (state.get("retrieval_events", []) or [])
        if str((event or {}).get("gap_id", "") or "").strip() == item_id
    ][-16:]

    async def _run_worker(worker: dict) -> dict:
        perspective_name = str(worker.get("perspective_name", "") or "").strip()
        worker_brief = str(worker.get("worker_brief", "") or "").strip()
        research_topic = _compose_research_topic(item)
        result = await storm_researcher_agent.ainvoke(
            {
                "researcher_messages": [HumanMessage(content=research_topic)],
                "research_topic": research_topic,
                "research_brief": state.get("research_brief", ""),
                "article_summary": state.get("article_summary", ""),
                "draft_report": draft_context,
                "search_type": search_type,
                "perspectives": [perspective_name],
                "forced_perspectives": [perspective_name],
                "perspective_profiles": state.get("perspective_profiles", {}) or _perspective_profile_map(state.get("research_perspectives", []) or []),
                "active_gap_id": item_id,
                "perspective_research_plans": {perspective_name: worker_brief},
                "reuse_existing_research_plans": True,
                "evidence_ledger": scoped_evidence_ledger,
                "retrieval_events": scoped_retrieval_events,
            },
            config={
                "run_name": f"research_worker_{item_id}_{perspective_name[:24].replace(' ', '_')}",
                "tags": ["research_program", "worker", item_id],
            },
        )
        return {"worker": worker, "result": result}

    worker_results = await asyncio.gather(*[_run_worker(worker) for worker in workers])

    all_new_entries: list[dict] = []
    all_new_events: list[dict] = []
    raw_notes: list[str] = []
    perspectives_used: list[str] = []
    for payload in worker_results:
        worker = payload["worker"]
        result = payload["result"]
        perspective_name = str(worker.get("perspective_name", "") or "").strip()
        if perspective_name:
            perspectives_used.append(perspective_name)

        candidate_entries = canonicalize_evidence_ledger(result.get("evidence_ledger", []) or [])
        candidate_events = [dict(entry or {}) for entry in (result.get("retrieval_events", []) or [])]
        new_entries = _diff_round_entries(running_entries, candidate_entries)
        new_events = _diff_round_retrieval_events(running_events, candidate_events)
        if new_entries:
            all_new_entries.extend(new_entries)
            running_entries = canonicalize_evidence_ledger([*running_entries, *new_entries])
        if new_events:
            all_new_events.extend(new_events)
            running_events = [*running_events, *new_events]

        delta_notes = [str(entry.get("answer", "") or "").strip() for entry in new_entries if str(entry.get("answer", "") or "").strip()]
        if not delta_notes:
            delta_notes = [str(note).strip() for note in (result.get("raw_notes", []) or []) if str(note).strip()][:2]
        raw_notes.extend(delta_notes[:2])

    round_number = int(state.get("storm_rounds", 0) or 0) + 1
    impact = compute_research_round_impact(existing_entries, all_new_entries, all_new_events)
    novelty = compute_evidence_novelty(existing_entries, all_new_entries)
    round_summary = _build_round_summary(
        storm_round=round_number,
        gap_id=item_id,
        research_topic=item.get("research_question", ""),
        search_type=search_type,
        new_entries=all_new_entries,
        impact=impact,
    )
    round_summary["perspectives"] = perspectives_used
    round_summary["title"] = item.get("title", "")
    round_summary["assignment_rationale"] = assignment.get("rationale", "")

    overall_mix = source_mix_report(
        canonicalize_evidence_ledger([*(state.get("evidence_ledger", []) or []), *all_new_entries]),
        [*(state.get("retrieval_events", []) or []), *all_new_events],
    )
    external_completed = bool(state.get("external_grounding_completed", False)) or (
        bool(impact.get("source_mix", {}).get("external_queries", 0))
        and bool(round_summary.get("material_improvement", False))
    )
    external_considered = bool(state.get("external_grounding_considered", False)) or search_type in {"external", "both"}
    external_rationale = _derive_external_grounding_rationale(
        mix=overall_mix,
        external_grounding_considered=external_considered,
    )

    task_entry = {
        "round": round_number,
        "item_id": item_id,
        "title": item.get("title", ""),
        "search_type": search_type,
        "perspectives": perspectives_used,
        "material_improvement": round_summary.get("material_improvement", False),
        "summary": round_summary.get("summary", ""),
        "status_counts": round_summary.get("status_counts", {}),
        "novelty": novelty,
    }

    elapsed = time.perf_counter() - start_time
    logger.info(
        "EXECUTE_RESEARCH_ASSIGNMENTS COMPLETE | new_entries=%d | new_events=%d | material=%s | time=%.2fs",
        len(all_new_entries),
        len(all_new_events),
        round_summary.get("material_improvement", False),
        elapsed,
    )
    logger.info("=" * 80)

    return {
        "research_iterations": int(state.get("research_iterations", 0) or 0) + 1,
        "storm_rounds": round_number,
        "notes": [round_summary.get("findings_packet", "") or round_summary.get("summary", "")],
        "raw_notes": raw_notes,
        "evidence_ledger": all_new_entries,
        "retrieval_events": all_new_events,
        "research_round_summaries": [round_summary],
        "research_novelty_history": [{**novelty, "storm_round": round_number, "topic": item.get("research_question", "")}],
        "last_round_impact_summary": round_summary.get("summary", ""),
        "last_round_material_improvement": round_summary.get("material_improvement", False),
        "source_mix_summary": overall_mix.get("summary", ""),
        "internal_rounds": int(state.get("internal_rounds", 0) or 0) + (1 if impact.get("source_mix", {}).get("article_queries", 0) else 0),
        "external_rounds": int(state.get("external_rounds", 0) or 0) + (1 if impact.get("source_mix", {}).get("external_queries", 0) else 0),
        "external_grounding_considered": external_considered,
        "external_grounding_completed": external_completed,
        "external_grounding_rationale": external_rationale,
        "latest_research_round": {
            "item_id": item_id,
            "item_title": item.get("title", ""),
            "search_type": search_type,
            "perspectives": perspectives_used,
            "new_entries": all_new_entries,
            "new_events": all_new_events,
            "round_summary": round_summary,
            "impact": impact,
            "novelty": novelty,
            "findings_packet": round_summary.get("findings_packet", "") or round_summary.get("summary", ""),
        },
        "task_history": [task_entry],
        "storm_perspective_research_plans": {
            **(state.get("storm_perspective_research_plans", {}) or {}),
            **{
                str(worker.get("perspective_name", "") or "").strip(): str(worker.get("worker_brief", "") or "").strip()
                for worker in workers
                if str(worker.get("perspective_name", "") or "").strip()
            },
        },
        "supervisor_phase": f"research_executed:{item_id}",
    }


def _agenda_update_fallback_delta(
    latest_round: dict,
    focused_item: dict,
    *,
    agenda_note: str,
    material_improvement: bool,
) -> dict:
    round_summary = latest_round.get("round_summary", {}) or {}
    item_id = str(latest_round.get("item_id", "") or "")
    fallback_search_type = round_summary.get("search_type", "both")
    if not material_improvement and fallback_search_type == "internal":
        fallback_search_type = "external"
    update = {
        "item_id": item_id,
        "new_status": "partial",
        "evidence_summary": _truncate(round_summary.get("summary", ""), 280),
        "recommended_search_type": fallback_search_type,
        "assigned_perspectives": latest_round.get("perspectives", []) or focused_item.get("assigned_perspectives", []),
    }
    for key in (
        "work_mode",
        "execution_focus",
        "internal_focus",
        "external_focus",
        "closure_condition",
        "artifact_state",
        "closure_reason",
        "reopen_only_if",
    ):
        if focused_item.get(key):
            update[key] = focused_item.get(key)
    return {
        "updates": [update],
        "add_items": [],
        "completed_item_ids": [],
        "deferred_item_ids": [],
        "agenda_note": agenda_note,
        "external_grounding_completed": bool(latest_round.get("impact", {}).get("source_mix", {}).get("external_queries", 0)) and material_improvement,
    }


def _agenda_delta_contract_issues(delta: dict) -> list[str]:
    issues: list[str] = []
    for update in delta.get("updates", []) or []:
        if not isinstance(update, dict):
            continue
        work_mode = _normalize_work_mode(update.get("work_mode", ""))
        new_status = str(update.get("new_status", "") or "").strip().lower()
        artifact_state = _normalize_artifact_state_map(update.get("artifact_state", {}))
        if work_mode == "boundary_with_artifact_check":
            if not str(update.get("internal_focus", "") or "").strip():
                issues.append("boundary_with_artifact_check requires non-empty internal_focus")
            if not str(update.get("external_focus", "") or "").strip():
                issues.append("boundary_with_artifact_check requires non-empty external_focus")
            if not artifact_state:
                issues.append("boundary_with_artifact_check requires non-empty artifact_state")
        if work_mode == "limitation_to_draft" and new_status != "deferred":
            issues.append("limitation_to_draft requires new_status=deferred")
        if work_mode == "close_unavailable":
            if new_status != "completed":
                issues.append("close_unavailable requires new_status=completed")
            if not artifact_state:
                issues.append("close_unavailable requires non-empty artifact_state")
    return issues


async def update_research_agenda(state: SupervisorState):
    latest_round = state.get("latest_research_round", {}) or {}
    if not latest_round:
        logger.info("UPDATE_RESEARCH_AGENDA: no latest round, skipping")
        return {}

    agenda = _normalize_agenda(state.get("research_agenda", {}))
    agenda_snapshot = _agenda_snapshot_text(agenda)
    latest_findings = latest_round.get("findings_packet", "") or latest_round.get("round_summary", {}).get("summary", "") or "(none)"
    task_history_text = _task_history_text(state.get("task_history", []) or [])
    assignment_text = _format_assignment_for_prompt(state.get("latest_assignment", {}) or {})
    item_id = str(latest_round.get("item_id", "") or "")
    focused_item = dict(_agenda_item_lookup(agenda).get(item_id) or {})
    focused_item_text = (
        json.dumps(_compact_agenda_item(focused_item), ensure_ascii=False, indent=2)
        if focused_item
        else "(none)"
    )
    recent_item_history = _format_recent_item_history(state.get("task_history", []) or [], item_id)

    if DETERMINISTIC_SUPERVISOR:
        delta = _deterministic_agenda_delta(state, latest_round, focused_item)
    else:
        prompt = agenda_update_prompt.format(
            agenda_snapshot=agenda_snapshot,
            assignment=assignment_text,
            latest_research_round=json.dumps(latest_round.get("round_summary", {}), ensure_ascii=False, indent=2),
            latest_findings=latest_findings,
            focused_item=focused_item_text,
            recent_item_history=recent_item_history,
            task_history=task_history_text,
            date=get_today_str(),
        )
        try:
            delta = (
                await _invoke_supervisor_structured(
                    AgendaUpdateDelta,
                    prompt,
                    SUPERVISOR_AGENDA_UPDATE_TIMEOUT_SECONDS,
                )
            ).model_dump()
            issues = _agenda_delta_contract_issues(delta)
            if issues:
                logger.warning(
                    "UPDATE_RESEARCH_AGENDA: contract repair retry required: %s",
                    "; ".join(issues),
                )
                repair_prompt = (
                    f"{prompt}\n\nContract repair required:\n- " + "\n- ".join(issues) +
                    "\nReturn the same agenda delta schema again. Keep enum fields exact and keep artifact_state as a JSON object."
                )
                delta = (
                    await _invoke_supervisor_structured(
                        AgendaUpdateDelta,
                        repair_prompt,
                        SUPERVISOR_AGENDA_UPDATE_TIMEOUT_SECONDS,
                        validation_retries=0,
                    )
                ).model_dump()
        except asyncio.TimeoutError:
            logger.warning(
                "UPDATE_RESEARCH_AGENDA: timed out after %.1fs, using deterministic fallback",
                SUPERVISOR_AGENDA_UPDATE_TIMEOUT_SECONDS,
            )
            round_summary = latest_round.get("round_summary", {}) or {}
            delta = _agenda_update_fallback_delta(
                latest_round,
                focused_item,
                agenda_note=(
                    "Agenda updated deterministically from latest research round."
                    if round_summary.get("material_improvement", False)
                    else "Latest round added limited value; keep the item open with its current explicit boundaries."
                ),
                material_improvement=bool(round_summary.get("material_improvement", False)),
            )
        except Exception as exc:
            logger.warning("UPDATE_RESEARCH_AGENDA: agenda delta failed, using deterministic fallback: %s", exc)
            round_summary = latest_round.get("round_summary", {}) or {}
            delta = _agenda_update_fallback_delta(
                latest_round,
                focused_item,
                agenda_note=(
                    "Agenda updated deterministically from latest research round."
                    if round_summary.get("material_improvement", False)
                    else "Latest round added limited value; keep the item open with its current explicit boundaries."
                ),
                material_improvement=bool(round_summary.get("material_improvement", False)),
            )

    delta = _constrain_agenda_delta(state, delta, latest_round)
    delta = _stabilize_agenda_delta(state, delta, latest_round)
    updated_agenda = _apply_agenda_delta(
        state.get("research_agenda", {}),
        delta,
        state.get("latest_assignment", {}) or {},
    )
    updated_agenda = _ensure_external_grounding_item(updated_agenda, state.get("research_perspectives", []) or [])

    return {
        "research_agenda": updated_agenda,
        "agenda_update_log": [{
            "item_id": latest_round.get("item_id", ""),
            "agenda_note": delta.get("agenda_note", ""),
            "completed_item_ids": delta.get("completed_item_ids", []),
            "deferred_item_ids": delta.get("deferred_item_ids", []),
        }],
        "external_grounding_completed": bool(state.get("external_grounding_completed", False)) or bool(delta.get("external_grounding_completed", False)),
        "supervisor_phase": f"agenda_updated:{latest_round.get('item_id', '')}",
    }


async def draft_refinement_agent(state: SupervisorState):
    pending_count = len(state.get("research_round_summaries", []) or []) - int(state.get("merged_round_count", 0) or 0)
    pending_agenda_count = len(state.get("agenda_update_log", []) or []) - int(state.get("merged_agenda_update_count", 0) or 0)
    if pending_count <= 0 and pending_agenda_count <= 0:
        logger.info("DRAFT_REFINEMENT_AGENT: no pending findings to merge")
        return {}

    findings_sections: list[str] = []
    if pending_count > 0:
        findings_sections.append(
            _compile_pending_findings_packet(
                (state.get("research_round_summaries", []) or [])[int(state.get("merged_round_count", 0) or 0):],
                max_rounds=max(1, pending_count),
                max_chars=7000,
            )
        )
    if pending_agenda_count > 0:
        closure_notes = _compile_pending_closure_notes(
            (state.get("agenda_update_log", []) or [])[int(state.get("merged_agenda_update_count", 0) or 0):],
            state.get("research_agenda", {}),
        )
        if closure_notes:
            findings_sections.append(closure_notes)

    findings = "\n\n".join(section for section in findings_sections if str(section or "").strip())
    if not findings.strip():
        logger.info("DRAFT_REFINEMENT_AGENT: pending updates produced no usable findings packet")
        return {
            "merged_round_count": len(state.get("research_round_summaries", []) or []),
            "merged_agenda_update_count": len(state.get("agenda_update_log", []) or []),
            "supervisor_phase": "draft_refined:no_material_findings",
        }

    draft_report = refine_draft_report.invoke(
        {
            "research_brief": state.get("research_brief", ""),
            "article_summary": state.get("article_summary", ""),
            "findings": findings,
            "draft_report": state.get("draft_report", ""),
            "newsletter_template": state.get("newsletter_template", ""),
        }
    )
    return {
        "draft_report": draft_report,
        "draft_editing_rounds": int(state.get("draft_editing_rounds", 0) or 0) + 1,
        "merged_round_count": len(state.get("research_round_summaries", []) or []),
        "merged_agenda_update_count": len(state.get("agenda_update_log", []) or []),
        "supervisor_phase": "draft_refined",
    }


async def convergence_review(state: SupervisorState):
    logger.info("CONVERGENCE_REVIEW START")
    updated_agenda, decision = _apply_convergence_policy(state)
    if not decision:
        logger.info("CONVERGENCE_REVIEW COMPLETE | decision=none")
        return {"supervisor_phase": "convergence_review:none"}

    logger.info("CONVERGENCE_REVIEW COMPLETE | decision=%s", decision)
    latest_item_id = (state.get("latest_research_round", {}) or {}).get("item_id", "")
    completed_item_ids = [latest_item_id] if decision == "resolved_as_unavailable" and latest_item_id else []
    deferred_item_ids = [latest_item_id] if decision == "deferred_low_yield_item" and latest_item_id else []
    return {
        "research_agenda": updated_agenda,
        "agenda_update_log": [{
            "item_id": latest_item_id,
            "agenda_note": decision,
            "completed_item_ids": completed_item_ids,
            "deferred_item_ids": deferred_item_ids,
        }],
        "supervisor_phase": f"convergence_review:{decision}",
    }


async def progress_gate(state: SupervisorState):
    logger.info("PROGRESS_GATE START")
    agenda = _normalize_agenda(state.get("research_agenda", {}))
    external_completed = bool(state.get("external_grounding_completed", False))
    researchable_items = _researchable_agenda_items(agenda)
    pending_rounds = max(
        0,
        len(state.get("research_round_summaries", []) or []) - int(state.get("merged_round_count", 0) or 0),
    )
    pending_agenda_updates = max(
        0,
        len(state.get("agenda_update_log", []) or []) - int(state.get("merged_agenda_update_count", 0) or 0),
    )

    if DETERMINISTIC_SUPERVISOR:
        if pending_rounds > 0 or pending_agenda_updates > 0:
            decision = {
                "should_continue": True,
                "recommended_action": "continue_research",
                "rationale": "Merge the latest findings into the draft before deciding whether to stop.",
            }
        elif researchable_items:
            if not external_completed and not any(
                str(item.get("recommended_search_type", "") or "").lower() == "external"
                or "external" in str(item.get("item_id", "") or "").lower()
                for item in researchable_items
            ):
                decision = {
                    "should_continue": False,
                    "recommended_action": "finalize",
                    "rationale": (
                        "No researchable external-grounding item remains. Finalize with explicit limitations instead of reopening impossible work."
                    ),
                }
            else:
                decision = {
                    "should_continue": True,
                    "recommended_action": "continue_research",
                    "rationale": "Researchable agenda items remain and can still materially improve the newsletter.",
                }
        elif external_completed or _external_grounding_exhausted(agenda):
            decision = {
                "should_continue": False,
                "recommended_action": "finalize",
                "rationale": (
                    "No researchable agenda items remain. Finalize and carry any deferred limitations into the newsletter."
                ),
            }
        else:
            decision = {
                "should_continue": True,
                "recommended_action": "continue_research",
                "rationale": "Continue because true external grounding has not completed yet.",
            }

        logger.info("PROGRESS_GATE COMPLETE | should_continue=%s | action=%s", decision.get("should_continue", False), decision.get("recommended_action", ""))
        return {
            "supervisor_route_reason": decision.get("rationale", ""),
            "supervisor_stop_reason": "research_complete" if not decision.get("should_continue", True) else "",
            "supervisor_completion_status": (
                "completed_cleanly"
                if not decision.get("should_continue", True) and external_completed
                else "completed_with_limitations"
                if not decision.get("should_continue", True)
                else ""
            ),
            "supervisor_phase": f"progress_gate:{decision.get('recommended_action', '')}",
            "latest_priority_decision": {**(state.get("latest_priority_decision", {}) or {}), "progress_gate": decision},
        }

    agenda_snapshot = _agenda_snapshot_text(state.get("research_agenda", {}))
    task_history_text = _task_history_text(state.get("task_history", []) or [])
    draft_report = _truncate(state.get("draft_report", ""), 5000)
    source_mix_summary = state.get("source_mix_summary", "") or "(none)"
    external_rationale = state.get("external_grounding_rationale", "") or "(none)"

    prompt = progress_gate_prompt.format(
        agenda_snapshot=agenda_snapshot,
        source_mix_summary=source_mix_summary,
        external_grounding_completed=external_completed,
        external_grounding_rationale=external_rationale,
        task_history=task_history_text,
        draft_report=draft_report,
        date=get_today_str(),
    )
    try:
        decision = (
            await _invoke_supervisor_structured(
                ProgressGateDecision,
                prompt,
                SUPERVISOR_PROGRESS_GATE_TIMEOUT_SECONDS,
            )
        ).model_dump()
    except asyncio.TimeoutError:
        logger.warning(
            "PROGRESS_GATE: timed out after %.1fs, using fallback",
            SUPERVISOR_PROGRESS_GATE_TIMEOUT_SECONDS,
        )
        decision = {
            "should_continue": bool((agenda.get("active_items", []) or []) or (agenda.get("partial_items", []) or []) or not external_completed),
            "recommended_action": "continue_research" if ((agenda.get("active_items", []) or []) or (agenda.get("partial_items", []) or []) or not external_completed) else "finalize",
            "rationale": "Fallback progress decision from agenda and external grounding state.",
        }
    except Exception as exc:
        logger.warning("PROGRESS_GATE: model failed, using fallback: %s", exc)
        decision = {
            "should_continue": bool((agenda.get("active_items", []) or []) or (agenda.get("partial_items", []) or []) or not external_completed),
            "recommended_action": "continue_research" if ((agenda.get("active_items", []) or []) or (agenda.get("partial_items", []) or []) or not external_completed) else "finalize",
            "rationale": "Fallback progress decision from agenda and external grounding state.",
        }

    research_iterations = int(state.get("research_iterations", 0) or 0)

    if not external_completed:
        decision = {
            "should_continue": True,
            "recommended_action": "continue_research",
            "rationale": "Continue because true external grounding has not completed yet.",
        }
    elif research_iterations >= MAX_RESEARCH_PROGRAM_CYCLES and decision.get("recommended_action") != "finalize":
        decision = {
            "should_continue": False,
            "recommended_action": "finalize",
            "rationale": (
                "Finalize because the research program reached its safety budget after completing external grounding. "
                "This cap is a fail-safe, not the primary stopping rule."
            ),
        }

    if decision.get("recommended_action") == "finalize" and decision.get("should_continue"):
        decision["recommended_action"] = "continue_research"

    logger.info("PROGRESS_GATE COMPLETE | should_continue=%s | action=%s", decision.get("should_continue", False), decision.get("recommended_action", ""))
    return {
        "supervisor_route_reason": decision.get("rationale", ""),
        "supervisor_stop_reason": "research_complete" if not decision.get("should_continue", True) else "",
        "supervisor_completion_status": "completed_cleanly" if not decision.get("should_continue", True) else "",
        "supervisor_phase": f"progress_gate:{decision.get('recommended_action', '')}",
        "latest_priority_decision": {**(state.get("latest_priority_decision", {}) or {}), "progress_gate": decision},
    }


def route_after_progress_gate(state: SupervisorState) -> Literal["prioritize_research_work", "__end__"]:
    progress_gate_decision = ((state.get("latest_priority_decision", {}) or {}).get("progress_gate", {}) or {})
    if progress_gate_decision.get("should_continue", True):
        return "prioritize_research_work"
    return END


research_program_builder = StateGraph(SupervisorState)
research_program_builder.add_node("prioritize_research_work", prioritize_research_work)
research_program_builder.add_node("prepare_research_assignment", prepare_research_assignment)
research_program_builder.add_node("execute_research_assignments", execute_research_assignments)
research_program_builder.add_node("update_research_agenda", update_research_agenda)
research_program_builder.add_node("convergence_review", convergence_review)
research_program_builder.add_node("draft_refinement_agent", draft_refinement_agent)
research_program_builder.add_node("progress_gate", progress_gate)

research_program_builder.add_edge(START, "prioritize_research_work")
research_program_builder.add_conditional_edges(
    "prioritize_research_work",
    route_after_priority,
    {
        "prepare_research_assignment": "prepare_research_assignment",
        "draft_refinement_agent": "draft_refinement_agent",
        "progress_gate": "progress_gate",
    },
)
research_program_builder.add_edge("prepare_research_assignment", "execute_research_assignments")
research_program_builder.add_edge("execute_research_assignments", "update_research_agenda")
research_program_builder.add_edge("update_research_agenda", "convergence_review")
research_program_builder.add_edge("convergence_review", "draft_refinement_agent")
research_program_builder.add_edge("draft_refinement_agent", "progress_gate")
research_program_builder.add_conditional_edges(
    "progress_gate",
    route_after_progress_gate,
    {
        "prioritize_research_work": "prioritize_research_work",
        END: END,
    },
)

supervisor_agent = research_program_builder.compile()
