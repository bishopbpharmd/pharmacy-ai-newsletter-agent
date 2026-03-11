
"""User Clarification and Research Brief Generation.

This module implements the scoping phase of the research workflow, where we:
1. Assess if the user's request needs clarification
2. Generate a detailed research brief from the conversation

The workflow uses structured output to make deterministic decisions about
whether sufficient context exists to proceed with research.
"""

import time
from datetime import datetime
from typing_extensions import Literal
import os
import re

from langchain_core.messages import HumanMessage, get_buffer_string
from langgraph.types import Command

from src.model_config import (
    PIPELINE_MODEL_SETTINGS,
    build_chat_model,
)
from src.prompts import (
    draft_report_generation_prompt,
    select_newsletter_template_prompt,
    transform_messages_into_research_topic_human_msg_prompt,
)
from src.state_scope import (
    AgentState,
    AgentInputState,
    TemplateSelection,
)
from src.templates import get_available_templates_info, get_valid_template_names, format_template_for_prompt
from src.logging_config import get_logger, log_token_usage, get_global_tracker

# ===== LOGGING SETUP =====

logger = get_logger("deep_research.scope")

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

# ===== CONFIGURATION =====

model = build_chat_model(PIPELINE_MODEL_SETTINGS.scope_template_selection_and_brief_model)
creative_model = build_chat_model(PIPELINE_MODEL_SETTINGS.scope_draft_generation_model)


def _compact_complete_units(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= max_chars:
        return normalized
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
    if kept:
        return " ".join(kept)
    return normalized[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")


def _compact_research_brief(text: str, max_chars: int = 1000) -> str:
    """Normalize the research brief so it stays lightweight across the workflow."""
    return _compact_complete_units(text, max_chars)


def _compress_article_content_for_draft(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\r\n?", "\n", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if not paragraphs:
        return _compact_complete_units(cleaned, max_chars)

    selected: list[str] = []
    last_paragraph = paragraphs[-1] if len(paragraphs) > 1 else ""
    reserved_tail = len(last_paragraph) + 2 if last_paragraph else 0
    total = 0

    def _append(paragraph: str) -> bool:
        nonlocal total
        projected = total + len(paragraph) + (2 if selected else 0)
        if projected > max_chars:
            return False
        selected.append(paragraph)
        total = projected
        return True

    front_target = max_chars * 3 // 5
    for paragraph in paragraphs:
        if paragraph == last_paragraph:
            continue
        if total >= front_target:
            break
        projected_with_tail = total + len(paragraph) + (2 if selected else 0) + reserved_tail
        if last_paragraph and projected_with_tail > max_chars:
            break
        if not _append(paragraph):
            break

    if last_paragraph and last_paragraph not in selected:
        _append(last_paragraph)

    if selected:
        ordered = []
        seen = set()
        for paragraph in paragraphs:
            if paragraph in selected and paragraph not in seen:
                ordered.append(paragraph)
                seen.add(paragraph)
        return "\n\n".join(ordered)
    return _compact_complete_units(cleaned, max_chars)

# ===== WORKFLOW NODES =====

def select_newsletter_template(state: AgentState) -> Command[Literal["write_research_brief"]]:
    """
    Select the appropriate newsletter template based on article content.

    Analyzes the article summary and metadata to determine whether to use
    the research_article or commentary template. This selection guides
    the research and writing phases.
    
    Raises:
        ValueError: If the model fails to select a valid template
    """
    start_time = time.perf_counter()
    
    logger.info("="*80)
    logger.info("SELECT_NEWSLETTER_TEMPLATE START")
    
    # Get article summary from state (set by ingest_document_node)
    article_summary = state.get("article_summary", "")
    logger.info("  article_summary_len: %d chars", len(article_summary))
    
    if not article_summary or not str(article_summary).strip():
        # Micro fallback: when we have no document summary, don't burn tokens or guess.
        # Default to the most robust template for short/uncertain inputs.
        default_template = "commentary"
        logger.warning(
            "  No article_summary in state - defaulting newsletter_template='%s'",
            default_template,
        )
        return Command(
            goto="write_research_brief",
            update={"newsletter_template": default_template},
        )
    
    # Get available templates info for the prompt
    available_templates = get_available_templates_info()
    valid_templates = get_valid_template_names()
    
    logger.info("  available_templates: %s", valid_templates)
    
    # Set up structured output model
    structured_output_model = model.with_structured_output(TemplateSelection)
    
    llm_start = time.perf_counter()
    # Invoke the model with template selection prompt
    response = structured_output_model.invoke([
        HumanMessage(content=select_newsletter_template_prompt.format(
            article_summary=article_summary,
            date=get_today_str(),
            available_templates=available_templates
        ))
    ])
    llm_elapsed = time.perf_counter() - llm_start
    
    selected = response.selected_template
    reasoning = response.reasoning
    
    logger.info("  selected_template: %s", selected)
    logger.info("  reasoning: %s", reasoning[:150] if reasoning else "None")
    logger.info("  llm_time: %.2fs", llm_elapsed)
    
    # Validate the selection
    if selected not in valid_templates:
        error_msg = f"Invalid template selection '{selected}'. Must be one of: {valid_templates}"
        logger.error("  TEMPLATE SELECTION ERROR: %s", error_msg)
        raise ValueError(error_msg)
    
    total_elapsed = time.perf_counter() - start_time
    logger.info("SELECT_NEWSLETTER_TEMPLATE COMPLETE | template=%s | time=%.2fs", selected, total_elapsed)
    
    return Command(
        goto="write_research_brief",
        update={"newsletter_template": selected}
    )


def write_research_brief(state: AgentState) -> Command[Literal["write_draft_report"]]:
    """Generate a concise brief that preserves user/editorial intent only."""
    start_time = time.perf_counter()

    logger.info("=" * 80)
    logger.info("WRITE_RESEARCH_BRIEF START")

    newsletter_template = state.get("newsletter_template") or "commentary"
    message_history = get_buffer_string(state.get("messages", []))

    system_context: list[str] = []
    if state.get("article_summary"):
        system_context.append(
            "[SYSTEM CONTEXT] A document is pre-loaded as doc_id=user_article. "
            "A separate article summary is already available to downstream research steps."
        )
    if state.get("source_filename"):
        system_context.append(f"[SYSTEM CONTEXT] Source filename: {state.get('source_filename')}")

    prompt_messages = "\n".join(
        part for part in [*system_context, message_history] if part and str(part).strip()
    ).strip()

    if not prompt_messages:
        fallback_brief = (
            f"I want a concise {newsletter_template} newsletter for a hospital pharmacy leader. "
            "Keep it article-first, skimmable, and grounded in the provided source."
        )
        logger.warning("  No message history available; using deterministic fallback brief")
        return Command(goto="write_draft_report", update={"research_brief": fallback_brief})

    llm_start = time.perf_counter()
    response = model.invoke(
        [
            HumanMessage(
                content=transform_messages_into_research_topic_human_msg_prompt.format(
                    messages=prompt_messages,
                    date=get_today_str(),
                    newsletter_template=newsletter_template,
                )
            )
        ]
    )
    llm_elapsed = time.perf_counter() - llm_start

    research_brief = _compact_research_brief(getattr(response, "content", "") or "")

    logger.info("  newsletter_template: %s", newsletter_template)
    logger.info("  research_brief_len: %d chars", len(research_brief))
    logger.info("  research_brief: %s", research_brief[:300])
    logger.info("  llm_time: %.2fs", llm_elapsed)
    logger.info("WRITE_RESEARCH_BRIEF COMPLETE | time=%.2fs", time.perf_counter() - start_time)

    return Command(
        goto="write_draft_report",
        update={"research_brief": research_brief},
    )


def write_draft_report(state: AgentState) -> dict:
    """
    Initial draft report generation.

    Creates an initial draft using the selected template structure.
    """
    start_time = time.perf_counter()
    
    logger.info("="*80)
    logger.info("WRITE_DRAFT_REPORT START")
    
    research_brief = state.get("research_brief", "")
    article_summary = state.get("article_summary", "")
    article_content = state.get("article_content", "")
    
    logger.info("  research_brief_len: %d chars", len(research_brief))
    logger.info("  article_summary_len: %d chars", len(article_summary))
    logger.info("  article_content_len: %d chars", len(article_content))
    
    # Validate newsletter_template - fail fast if not set
    newsletter_template = state.get("newsletter_template")
    if not newsletter_template:
        raise ValueError("newsletter_template not set in state. Ensure select_newsletter_template runs before write_draft_report.")
    
    logger.info("  newsletter_template: %s", newsletter_template)
    
    # Get formatted template for prompt injection
    template_content = format_template_for_prompt(newsletter_template)

    # Context hygiene: strip OCR page markers (kept for chunk metadata) and cap raw content
    # passed into the first draft prompt to avoid token bloat.
    cleaned_article_content = re.sub(
        r"^<!--\s*page:\s*\d+\s*-->\s*$\n?",
        "",
        str(article_content or ""),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    max_article_chars = int(os.environ.get("DEEP_RESEARCH_DRAFT_ARTICLE_CHARS", "80000"))
    if max_article_chars > 0 and len(cleaned_article_content) > max_article_chars:
        cleaned_article_content = _compress_article_content_for_draft(cleaned_article_content, max_article_chars)
    
    draft_report_prompt = draft_report_generation_prompt.format(
        research_brief=research_brief,
        article_content=cleaned_article_content,
        date=get_today_str(),
        template_content=template_content
    )
    
    logger.info("  prompt_len: %d chars", len(draft_report_prompt))

    llm_start = time.perf_counter()
    response = creative_model.invoke([HumanMessage(content=draft_report_prompt)])
    llm_elapsed = time.perf_counter() - llm_start

    draft_report = str(getattr(response, "content", "") or "").strip()
    draft_len = len(draft_report)
    logger.info("  draft_report_len: %d chars", draft_len)
    logger.info("  draft_report_preview: %s", draft_report[:300])
    logger.info("  llm_time: %.2fs", llm_elapsed)
    
    total_elapsed = time.perf_counter() - start_time
    logger.info("WRITE_DRAFT_REPORT COMPLETE | time=%.2fs -> supervisor_subgraph", total_elapsed)

    return {
        "draft_report": draft_report
    }
