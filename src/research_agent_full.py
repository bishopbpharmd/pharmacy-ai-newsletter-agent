
"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation  
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""

import json
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from src.utils import (
    get_today_str,
    parse_newsletter_to_structured,
    sources_from_evidence_ledger,
    structured_newsletter_to_dict,
    _sanitize_filename,
    _write_json_atomic,
    _write_text_atomic,
)
from src.evidence_utils import (
    claim_source_map_from_markdown,
    citation_rendering_policy,
    canonicalize_evidence_ledger,
    clean_prompt_artifacts,
    dedupe_text_list,
    source_mix_report,
)
from src.document_profile import extract_plausible_title_lines, extract_title_from_summary
from src.prompts import (
    final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt,
    copywriter_polish_prompt,
    critique_reflection_prompt,
    rewrite_with_critique_prompt
)
from src.templates import format_template_for_prompt, get_template_sections_list
from src.state_scope import AgentState, AgentInputState, CritiqueReflection
from src.research_agent_scope import (
    select_newsletter_template,
    write_draft_report,
    write_research_brief,
)
from src.research_program_supervisor import (
    initialize_research_agenda,
    select_fixed_perspectives,
    supervisor_agent,
)
from src.logging_config import get_logger, log_token_usage, get_global_tracker
from src.model_config import (
    PIPELINE_MODEL_SETTINGS,
    build_chat_model,
)

# ===== LOGGING SETUP =====

logger = get_logger("deep_research.full_agent")

# ===== Config =====

writer_model = build_chat_model(PIPELINE_MODEL_SETTINGS.final_report_generation_model)
copywriter_model = build_chat_model(PIPELINE_MODEL_SETTINGS.final_copywriter_polish_model)
critique_model = build_chat_model(PIPELINE_MODEL_SETTINGS.final_critique_reflection_model)

# Maximum critique-rewrite iterations before forcing exit to copywriter
MAX_CRITIQUE_ITERATIONS = 5

logger.info("Full agent initialized with writer, copywriter, and critique models")


def _truncate(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip()


def _count_truncation_markers(value: object) -> int:
    return str(value or "").count("[...truncated...]")


_PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*\d+\s*-->\s*", re.IGNORECASE)
_SOURCE_METADATA_BLOCK_RE = re.compile(
    r"<!-- SOURCE_METADATA_START -->.*?<!-- SOURCE_METADATA_END -->\n*",
    re.DOTALL,
)
_PLACEHOLDER_TITLE_RE = re.compile(r"^[#>*\-\s]*t?itle[:\s]*$", re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_AUTHOR_SUFFIX_RE = re.compile(
    r"(?:,\s*|\s+)(?:pharmd|ph\.?d\.?|md|m\.d\.?|do|mba|mph|ms|m\.s\.?|msc|rn|rph|bpharm|dnp|facc|facs|facp|fccp)\b\.?",
    re.IGNORECASE,
)
_AUTHOR_PREFIX_RE = re.compile(r"^(?:dr|prof|professor)\.?\s+", re.IGNORECASE)
_GENERIC_SOURCE_TITLE_RE = re.compile(
    r"^(?:original article|preface|abstract|introduction|contains nonbinding recommendations|guidance for industry and food and drug administration staff)$",
    re.IGNORECASE,
)


def _strip_page_markers(text: str) -> str:
    return _PAGE_MARKER_RE.sub("", str(text or ""))


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_newsletter_title(markdown: str) -> str:
    text = str(markdown or "")
    match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def _sanitize_author_name(name: str) -> str:
    cleaned = _AUTHOR_PREFIX_RE.sub("", str(name or "").strip())
    cleaned = _AUTHOR_SUFFIX_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;")
    return cleaned


def _split_author_names(raw_line: str) -> list[str]:
    normalized = _normalize_space(raw_line)
    if not normalized:
        return []
    normalized = normalized.replace(" and ", ", ")
    parts = [_sanitize_author_name(part) for part in re.split(r"\s*[,;]\s*", normalized)]
    authors: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        word_like = [
            re.sub(r"[^A-Za-z-]", "", token)
            for token in part.split()
            if re.search(r"[A-Za-z]", token)
        ]
        if len([token for token in word_like if len(token) >= 2]) < 2:
            continue
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        authors.append(part)
    return authors


def _normalize_title_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _looks_like_duplicate_title(line: str, title: str) -> bool:
    candidate = re.sub(r"^[#>*\\-\\s]+", "", str(line or "")).strip()
    if not candidate or not title:
        return False
    candidate_key = _normalize_title_key(candidate)
    title_key = _normalize_title_key(title)
    if not candidate_key or not title_key:
        return False
    if candidate_key == title_key:
        return True
    return SequenceMatcher(None, candidate_key, title_key).ratio() >= 0.96


def _is_placeholder_title_line(line: str) -> bool:
    candidate = str(line or "").strip()
    return bool(candidate) and bool(_PLACEHOLDER_TITLE_RE.match(candidate))


def _extract_title_candidate_from_body(body: str) -> str:
    for raw_line in str(body or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            break
        if stripped.startswith("<!--"):
            continue
        if _is_placeholder_title_line(stripped):
            continue
        bullet_match = re.match(r"^(?:[-*]|\d+[.)])\s+(.+)$", stripped)
        if bullet_match:
            candidate = bullet_match.group(1).strip()
            if candidate and not _is_placeholder_title_line(candidate):
                return candidate
            continue
        if not stripped.startswith("#"):
            return stripped
    return ""


def _is_generic_source_title(value: str) -> bool:
    candidate = _normalize_space(value)
    return bool(candidate) and bool(_GENERIC_SOURCE_TITLE_RE.match(candidate))


def _date_parts_to_iso(parts: Any) -> str:
    if not isinstance(parts, list) or not parts:
        return ""
    row = parts[0] if isinstance(parts[0], list) else parts
    if not isinstance(row, list) or not row:
        return ""
    year = int(row[0])
    month = int(row[1]) if len(row) > 1 else 1
    day = int(row[2]) if len(row) > 2 else 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def _looks_like_scholarly_article(article_content: str, article_summary: str) -> bool:
    combined = f"{article_content}\n{article_summary}"
    if _DOI_RE.search(combined):
        return True
    has_abstract = bool(re.search(r"\babstract\b", combined, flags=re.IGNORECASE))
    has_results_stack = bool(
        re.search(r"\b(?:methods|results|conclusions)\b", combined, flags=re.IGNORECASE)
    )
    has_submission_cues = bool(
        re.search(r"\b(?:received|accepted)\s*:", combined, flags=re.IGNORECASE)
    )
    return has_abstract or (has_results_stack and has_submission_cues) or has_submission_cues


def _extract_title_and_authors_from_content(article_content: str, article_summary: str) -> tuple[str, list[str]]:
    cleaned_content = _strip_page_markers(article_content)
    raw_lines = [line.strip() for line in cleaned_content.splitlines() if line.strip()]
    best_title = ""
    best_idx = -1
    best_score = -1

    for idx, raw_line in enumerate(raw_lines[:10]):
        if not raw_line.startswith("#"):
            continue
        candidate = _normalize_space(raw_line.lstrip("#").strip())
        if not candidate or _is_generic_source_title(candidate):
            continue
        if len(candidate) < 12 or len(candidate) > 220:
            continue
        if re.search(r"\b(?:doi|received|accepted|published|document issued on)\b", candidate, flags=re.IGNORECASE):
            continue
        score = 0
        if raw_line.startswith("#"):
            score += 4
        if 5 <= len(candidate.split()) <= 28:
            score += 2
        if re.search(r"[—:]", candidate):
            score += 1
        for follower in raw_lines[idx + 1 : idx + 5]:
            if follower.startswith("#"):
                break
            if (
                len(follower) <= 180
                and "http" not in follower.lower()
                and "@" not in follower
                and not re.search(r"\b(?:doi|received|accepted|published|document issued)\b", follower, flags=re.IGNORECASE)
                and _split_author_names(follower)
            ):
                score += 2
                break
        if score > best_score:
            best_score = score
            best_title = candidate
            best_idx = idx

    if not best_title:
        summary_title = extract_title_from_summary(article_summary)
        if summary_title and not _is_generic_source_title(summary_title):
            best_title = summary_title

    authors: list[str] = []
    if best_idx >= 0:
        for candidate in raw_lines[best_idx + 1 : best_idx + 5]:
            if candidate.startswith("#"):
                break
            if (
                len(candidate) > 180
                or "http" in candidate.lower()
                or "@" in candidate
                or re.search(r"\b(?:doi|received|accepted|published|document issued)\b", candidate, flags=re.IGNORECASE)
            ):
                continue
            authors = _split_author_names(candidate)
            if authors:
                break

    return best_title, authors


@lru_cache(maxsize=128)
def _crossref_metadata_lookup(article_title: str) -> dict[str, Any]:
    title = _normalize_space(article_title)
    if not title:
        return {}

    params = urlencode(
        {
            "query.title": title,
            "rows": 5,
            "select": "DOI,title,container-title,published-online,published-print,created,author",
        }
    )
    request = Request(
        f"https://api.crossref.org/works?{params}",
        headers={"User-Agent": "deep-research-newsletter/1.0"},
    )
    try:
        with urlopen(request, timeout=6) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.debug("Crossref lookup failed for %s: %s", title, exc)
        return {}
    except Exception as exc:  # pragma: no cover - network failures are best-effort
        logger.debug("Unexpected Crossref lookup failure for %s: %s", title, exc)
        return {}

    items = ((payload or {}).get("message") or {}).get("items") or []
    if not isinstance(items, list):
        return {}

    target = _normalize_title_key(title)
    best_item: dict[str, Any] | None = None
    best_score = 0.0
    for item in items:
        candidate_titles = item.get("title") or []
        candidate_title = candidate_titles[0] if candidate_titles else ""
        normalized_candidate = _normalize_title_key(candidate_title)
        if not normalized_candidate:
            continue
        score = 1.0 if normalized_candidate == target else SequenceMatcher(None, target, normalized_candidate).ratio()
        if score > best_score:
            best_score = score
            best_item = item

    if not best_item or best_score < 0.75:
        return {}

    authors = []
    for author in best_item.get("author") or []:
        if not isinstance(author, dict):
            continue
        full_name = " ".join(
            piece.strip()
            for piece in [str(author.get("given", "")).strip(), str(author.get("family", "")).strip()]
            if piece and str(piece).strip()
        ).strip()
        clean_name = _sanitize_author_name(full_name)
        if clean_name:
            authors.append(clean_name)

    return {
        "article_title": _normalize_space((best_item.get("title") or [""])[0]),
        "authors": authors,
        "journal": _normalize_space((best_item.get("container-title") or [""])[0]),
        "doi": _normalize_space(best_item.get("DOI", "")),
        "publication_date": (
            _date_parts_to_iso((best_item.get("published-online") or {}).get("date-parts"))
            or _date_parts_to_iso((best_item.get("published-print") or {}).get("date-parts"))
            or _date_parts_to_iso((best_item.get("created") or {}).get("date-parts"))
        ),
    }


def _extract_source_metadata(article_content: str, article_summary: str) -> dict[str, Any]:
    cleaned_content = _strip_page_markers(article_content)
    article_title, authors = _extract_title_and_authors_from_content(article_content, article_summary)
    scholarly_source = _looks_like_scholarly_article(article_content, article_summary)

    doi_match = _DOI_RE.search(cleaned_content)
    if not doi_match and scholarly_source:
        doi_match = _DOI_RE.search(str(article_summary or ""))
    publication_match = re.search(
        r"(?:Published(?: online)?|Document issued on|Issued on)\s*:?\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        f"{cleaned_content}\n{article_summary}",
        flags=re.IGNORECASE,
    )
    journal_match = re.search(
        r"^\s*(?:Journal|Published in)\s*:\s*(.+?)\s*$",
        cleaned_content,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    metadata: dict[str, Any] = {
        "article_title": _normalize_space(article_title),
        "authors": authors,
        "publication_date": _normalize_space(publication_match.group(1)) if publication_match else "",
        "journal": _normalize_space(journal_match.group(1)) if journal_match else "",
        "doi": doi_match.group(0) if doi_match else "",
    }

    if metadata.get("publication_date"):
        parsed_date = None
        for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                parsed_date = datetime.strptime(metadata["publication_date"], fmt)
                break
            except ValueError:
                continue
        if parsed_date:
            metadata["publication_date"] = parsed_date.strftime("%Y-%m-%d")

    if metadata.get("article_title") and scholarly_source:
        resolved = _crossref_metadata_lookup(metadata["article_title"])
        for key in ("article_title", "authors", "publication_date", "journal", "doi"):
            if metadata.get(key):
                continue
            value = resolved.get(key)
            if value:
                metadata[key] = value

    if not metadata.get("authors"):
        metadata["authors"] = []

    return metadata


def _format_source_metadata_block(source_metadata: dict[str, Any] | None) -> str:
    metadata = dict(source_metadata or {})
    article_title = _normalize_space(metadata.get("article_title", ""))
    authors = [name for name in metadata.get("authors", []) or [] if _normalize_space(name)]
    journal = _normalize_space(metadata.get("journal", ""))
    publication_date = _normalize_space(metadata.get("publication_date", ""))
    doi = _normalize_space(metadata.get("doi", ""))

    lines: list[str] = []
    if article_title:
        lines.append(f"> **Source article:** {article_title}")
    if authors:
        lines.append(f"> **Authors:** {', '.join(authors)}")
    if journal:
        lines.append(f"> **Journal:** {journal}")
    if publication_date:
        lines.append(f"> **Published:** {publication_date}")
    if doi:
        lines.append(f"> **DOI:** {doi}")

    if not lines:
        return ""

    return "\n".join(
        ["<!-- SOURCE_METADATA_START -->", *lines, "<!-- SOURCE_METADATA_END -->"]
    )


def _finalize_newsletter_markdown(
    markdown: str,
    *,
    fallback_title: str = "",
    source_metadata: dict[str, Any] | None = None,
) -> str:
    text = clean_prompt_artifacts(str(markdown or "")).strip()
    if not text:
        return ""

    current_title = _extract_newsletter_title(text)
    title = current_title or _normalize_space(fallback_title)
    body = text
    if current_title:
        body = re.sub(r"^#\s+.+?\n*", "", text, count=1, flags=re.MULTILINE).lstrip()
    body = _SOURCE_METADATA_BLOCK_RE.sub("", body).lstrip()
    if not title or _is_placeholder_title_line(title):
        recovered_title = _extract_title_candidate_from_body(body)
        if recovered_title:
            title = recovered_title
        elif _is_placeholder_title_line(title):
            title = ""
    if title and body:
        body_lines = body.splitlines()
        while body_lines and (
            _is_placeholder_title_line(body_lines[0])
            or _looks_like_duplicate_title(body_lines[0], title)
        ):
            body_lines.pop(0)
            while body_lines and not body_lines[0].strip():
                body_lines.pop(0)
        body = "\n".join(body_lines).lstrip()

    parts: list[str] = []
    if title:
        parts.append(f"# {title}")
    metadata_block = _format_source_metadata_block(source_metadata)
    if metadata_block:
        parts.append(metadata_block)
    if body:
        parts.append(body)
    return "\n\n".join(part for part in parts if part).strip() + "\n"


def _compact_findings(notes: list[str], max_chars: int = 24000) -> str:
    cleaned_notes = [
        clean_prompt_artifacts(note)
        for note in dedupe_text_list(notes)
        if clean_prompt_artifacts(note)
    ]
    joined = "\n\n".join(cleaned_notes)
    return _truncate(joined, max_chars) if max_chars > 0 else joined


def _format_claim_line(claim: dict) -> str:
    question = _truncate(str(claim.get("question", "") or ""), 140)
    answer = _truncate(str(claim.get("claim", "") or ""), 220)
    refs = [str(ref).strip() for ref in (claim.get("evidence_refs", []) or []) if str(ref).strip()]
    base = f"- Q: {question} | A: {answer}"
    if refs:
        base += " | refs: " + ", ".join(_truncate(ref, 90) for ref in refs[:2])
    return base


def _format_round_delta(round_summary: dict) -> str:
    packet = str(round_summary.get("findings_packet", "") or "").strip()
    if packet:
        return clean_prompt_artifacts(packet)

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
        lines.extend(_format_claim_line(claim) for claim in claims[:4])
    lines.append("</RoundDelta>")
    return "\n".join(lines)


def _format_gap_ledger_snapshot(gap_ledger: list[dict], max_items: int = 8) -> str:
    rows = [dict(item or {}) for item in (gap_ledger or []) if isinstance(item, dict)]
    if not rows:
        return "(none)"
    parts: list[str] = []
    for row in rows[:max_items]:
        parts.append(
            "- gap_id={gap_id} status={status} attempts={attempts} non_progress={non_progress} reason={reason}".format(
                gap_id=_truncate(str(row.get("gap_id", "") or ""), 64),
                status=str(row.get("status", "open") or "open"),
                attempts=int(row.get("attempt_count", 0) or 0),
                non_progress=int(row.get("non_progress_count", 0) or 0),
                reason=_truncate(str(row.get("last_reason", "") or ""), 120),
            )
        )
    if len(rows) > max_items:
        parts.append(f"- ... {len(rows) - max_items} more gaps omitted ...")
    return "\n".join(parts)


def _compile_findings_context(state: AgentState, max_chars: int = 24000) -> tuple[str, dict]:
    round_summaries = [
        dict(item or {})
        for item in (state.get("research_round_summaries", []) or [])
        if isinstance(item, dict)
    ]
    gap_ledger = [dict(item or {}) for item in (state.get("gap_ledger", []) or []) if isinstance(item, dict)]

    if round_summaries:
        blocks = [_format_round_delta(item) for item in round_summaries[-8:]]
        ledger_snapshot = _format_gap_ledger_snapshot(gap_ledger, max_items=10)
        payload = (
            "<CompiledFindings>\n"
            f"round_count={len(round_summaries)} merged_round_count={int(state.get('merged_round_count', 0) or 0)}\n"
            "GapLedgerSnapshot:\n"
            f"{ledger_snapshot}\n\n"
            + "\n\n".join(block for block in blocks if str(block).strip())
            + "\n</CompiledFindings>"
        )
        return _truncate(clean_prompt_artifacts(payload), max_chars), {
            "mode": "round_summaries",
            "round_count": len(round_summaries),
            "gap_count": len(gap_ledger),
        }

    notes = state.get("notes", []) or []
    fallback = _compact_findings(notes, max_chars=max_chars)
    return fallback, {
        "mode": "notes_fallback",
        "round_count": 0,
        "gap_count": len(gap_ledger),
        "notes_count": len(notes),
    }


def _format_evidence_entry(entry: dict, idx: int) -> str:
    part_lines = [
        f"{idx}. Perspective: {entry.get('perspective', '')}",
        f"   Topic: {_truncate(str(entry.get('research_topic', '') or ''), 240)}",
        f"   Question: {_truncate(str(entry.get('question', '') or ''), 240)}",
        f"   Answer: {_truncate(str(entry.get('answer', '') or ''), 360)}",
    ]
    if entry.get("review_note"):
        part_lines.append(f"   Review note: {_truncate(str(entry.get('review_note', '') or ''), 220)}")
    if entry.get("missing_reason"):
        part_lines.append(f"   Missing reason: {_truncate(str(entry.get('missing_reason', '') or ''), 160)}")
    evidence_items = entry.get("evidence", []) or []
    for evidence in evidence_items[:2]:
        part_lines.append(
            "   Source: "
            f"{evidence.get('source_type', 'source')} | "
            f"{_truncate(str(evidence.get('title', '') or ''), 120)} | "
            f"{_truncate(str(evidence.get('locator', '') or ''), 160)} | "
            f"{_truncate(str(evidence.get('snippet', '') or ''), 220)}"
        )
    return "\n".join(part_lines)


def _format_curated_evidence_packet(
    evidence_ledger: list[dict],
    max_supported: int = 14,
    max_watchlist: int = 6,
    max_missing: int = 6,
) -> tuple[str, dict]:
    canonical_ledger = canonicalize_evidence_ledger(evidence_ledger)
    if not canonical_ledger:
        return "(none)", {
            "supported_count": 0,
            "watchlist_count": 0,
            "missing_count": 0,
            "omitted_supported": 0,
            "omitted_watchlist": 0,
            "omitted_missing": 0,
        }

    supported = [entry for entry in canonical_ledger if entry.get("answer_status", "supported") == "supported"]
    watchlist = [
        entry
        for entry in canonical_ledger
        if entry.get("answer_status", "") in {"conflicted", "needs_review"}
    ]
    missing = [
        entry
        for entry in canonical_ledger
        if entry.get("answer_status", "") in {"missing", "not_in_source", "premise_mismatch"}
    ]

    sections: list[str] = []
    if supported:
        supported_lines = ["Supported findings:"]
        for idx, entry in enumerate(supported[:max_supported], 1):
            supported_lines.append(_format_evidence_entry(entry, idx))
        if len(supported) > max_supported:
            supported_lines.append(f"... {len(supported) - max_supported} more supported findings omitted ...")
        sections.append("\n\n".join(supported_lines))
    if watchlist:
        watchlist_lines = ["Needs review / possible contradictions:"]
        for idx, entry in enumerate(watchlist[:max_watchlist], 1):
            watchlist_lines.append(_format_evidence_entry(entry, idx))
        if len(watchlist) > max_watchlist:
            watchlist_lines.append(f"... {len(watchlist) - max_watchlist} more review items omitted ...")
        sections.append("\n\n".join(watchlist_lines))
    if missing:
        missing_lines = ["Confirmed missing or unavailable items:"]
        for idx, entry in enumerate(missing[:max_missing], 1):
            missing_lines.append(_format_evidence_entry(entry, idx))
        if len(missing) > max_missing:
            missing_lines.append(f"... {len(missing) - max_missing} more missing items omitted ...")
        sections.append("\n\n".join(missing_lines))

    return "\n\n".join(sections), {
        "supported_count": len(supported),
        "watchlist_count": len(watchlist),
        "missing_count": len(missing),
        "omitted_supported": max(0, len(supported) - max_supported),
        "omitted_watchlist": max(0, len(watchlist) - max_watchlist),
        "omitted_missing": max(0, len(missing) - max_missing),
    }

# ===== DOCUMENT INGESTION =====

async def ingest_document_node(state: AgentState):
    """
    Ingest document and generate article summary.
    
    Extracts document from state (PDF path or text from messages) and generates
    a global summary that will be used throughout the pipeline.
    """
    logger.info("="*80)
    logger.info("INGEST_DOCUMENT_NODE START")
    
    start_time = time.perf_counter()
    
    # Get PDF path from state (preferred method)
    pdf_path = state.get("pdf_path")
    text = ""
    url = ""
    derived_source_filename: str | None = None
    
    # If no PDF in state, check if text or a URL was provided in messages
    if not pdf_path:
        messages = state.get("messages", [])
        # Extract URL or text from first user message
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                content = str(msg.content)
                content_stripped = content.strip()

                # URL ingestion path (runner often provides a URL as the query)
                if re.match(r"^https?://\S+$", content_stripped):
                    url = content_stripped
                    logger.info("  Found URL in message: %s", url)
                    break

                # Check if it's a long text (likely article content)
                if len(content) > 500:  # Reasonable threshold for article text
                    text = content
                    logger.info("  Found text content in message (%d chars)", len(text))
                    break
    else:
        logger.info("  PDF path from state: %s", pdf_path)

    # If we got a URL, fetch + extract readable text (best-effort)
    if url and not text and not pdf_path:
        try:
            from urllib.parse import urlparse
            from src.utils import fetch_url_as_text

            extracted = fetch_url_as_text(url)
            if extracted and len(extracted.strip()) > 500:
                text = f"Source URL: {url}\n\n{extracted}"
                logger.info("  URL extracted text length: %d chars", len(text))

                # If the user didn't provide a filename, derive one from the URL for output naming.
                if not state.get("source_filename"):
                    parsed = urlparse(url)
                    candidate = Path(parsed.path).stem or parsed.netloc or "url_article"
                    derived_source_filename = _sanitize_filename(candidate)
            else:
                logger.warning("  URL fetch/extraction returned insufficient text; continuing without ingestion")
        except Exception as e:
            logger.warning("  URL ingestion failed (%s). Continuing without ingestion.", e)
    
    if not pdf_path and not text:
        logger.warning("  No document found in input - skipping ingestion")
        return {"article_summary": "", "article_content": ""}
    
    # Import ingest_document tool
    from src.utils import ingest_document, DEFAULT_DOC_ID
    
    # Call ingest_document
    if pdf_path:
        logger.info("  Ingesting PDF: %s", pdf_path)
        ingest_result = ingest_document.invoke({
            "doc_id": DEFAULT_DOC_ID,
            "pdf_path": pdf_path,
            "text": ""
        })
    else:
        logger.info("  Ingesting text (%d chars)", len(text))
        ingest_result = ingest_document.invoke({
            "doc_id": DEFAULT_DOC_ID,
            "pdf_path": "",
            "text": text
        })
    
    # Extract global_summary and article_content from result
    if isinstance(ingest_result, dict) and ingest_result.get("status") == "success":
        global_summary = ingest_result.get("global_summary", "")
        article_content = ingest_result.get("article_content", "")
        chunk_count = ingest_result.get("chunk_count", 0)
        source_metadata = _extract_source_metadata(article_content, global_summary)
        logger.info("  Ingestion successful: %d chunks, summary=%d words, content=%d chars", 
                   chunk_count, 
                   len(global_summary.split()) if global_summary else 0,
                   len(article_content) if article_content else 0)
        logger.info(
            "  source_metadata: title=%s | authors=%d | journal=%s | publication_date=%s | doi=%s",
            _truncate(source_metadata.get("article_title", ""), 90),
            len(source_metadata.get("authors", []) or []),
            _truncate(source_metadata.get("journal", ""), 60),
            source_metadata.get("publication_date", ""),
            source_metadata.get("doi", ""),
        )
    else:
        logger.warning("  Ingestion failed: %s", ingest_result.get("message", "Unknown error"))
        global_summary = ""
        article_content = ""
        source_metadata = {}
    
    elapsed = time.perf_counter() - start_time
    logger.info("INGEST_DOCUMENT_NODE COMPLETE | time=%.2fs", elapsed)
    logger.info("="*80)
    
    update: dict = {
        "article_summary": global_summary,
        "article_content": article_content,
        "source_metadata": source_metadata,
    }
    if derived_source_filename:
        update["source_filename"] = derived_source_filename
    return update

# ===== FINAL REPORT GENERATION =====

from src.state_scope import AgentState

async def final_report_generation(state: AgentState):
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """
    logger.info("="*80)
    logger.info("FINAL_REPORT_GENERATION START")
    
    start_time = time.perf_counter()

    notes = state.get("notes", [])
    research_brief = state.get("research_brief", "")
    raw_article_summary = state.get("article_summary", "")
    raw_draft_report = state.get("draft_report", "")
    article_summary = _truncate(clean_prompt_artifacts(raw_article_summary), 8000)
    draft_report = _truncate(clean_prompt_artifacts(raw_draft_report), 20000)
    evidence_ledger = canonicalize_evidence_ledger(state.get("evidence_ledger", []) or [])
    source_mix = source_mix_report(evidence_ledger, state.get("retrieval_events", []) or [])
    citation_policy = citation_rendering_policy(evidence_ledger)
    
    # Validate newsletter_template - fail fast if not set
    newsletter_template = state.get("newsletter_template")
    if not newsletter_template:
        raise ValueError("newsletter_template not set in state. Ensure select_newsletter_template runs before final_report_generation.")
    
    logger.info("  notes_count: %d", len(notes))
    logger.info("  research_brief_len: %d chars", len(research_brief))
    logger.info("  article_summary_len: %d chars", len(article_summary))
    logger.info("  draft_report_len: %d chars", len(draft_report))
    logger.info("  evidence_ledger_count: %d", len(evidence_ledger))
    logger.info("  newsletter_template: %s", newsletter_template)
    logger.info("  source_mix: %s", source_mix.get("summary", ""))
    logger.info("  citation_policy: %s", citation_policy.get("guidance", ""))

    findings, findings_stats = _compile_findings_context(state)
    evidence_packet_text, evidence_packet_stats = _format_curated_evidence_packet(evidence_ledger)
    truncation_markers_removed = (
        _count_truncation_markers(raw_article_summary)
        + _count_truncation_markers(raw_draft_report)
        + sum(_count_truncation_markers(note) for note in notes)
        + sum(
            _count_truncation_markers(item.get("summary", ""))
            + _count_truncation_markers(item.get("findings_packet", ""))
            for item in (state.get("research_round_summaries", []) or [])
            if isinstance(item, dict)
        )
    )
    logger.info("  total_findings_len: %d chars | mode=%s", len(findings), findings_stats.get("mode", "unknown"))
    
    # Get formatted template for prompt injection
    template_content = format_template_for_prompt(newsletter_template)

    try:
        final_report_prompt = final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt.format(
            research_brief=research_brief,
            article_summary=article_summary,
            findings=findings,
            evidence_ledger=evidence_packet_text,
            source_mix_summary=state.get("source_mix_summary", "") or source_mix.get("summary", ""),
            citation_guidance=citation_policy.get("guidance", ""),
            date=get_today_str(),
            draft_report=draft_report,
            template_content=template_content
        )
        
        logger.info("  prompt_len: %d chars", len(final_report_prompt))
        logger.info("  Invoking writer model for final report...")

        final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])
        
        elapsed = time.perf_counter() - start_time
        report_len = len(final_report.content) if final_report.content else 0
        
        # Log token usage
        log_token_usage(logger, final_report, "final_report_generation")
        get_global_tracker().add_usage(final_report, "final_report_generation")
        
        logger.info("FINAL_REPORT_GENERATION COMPLETE")
        logger.info("  output_len: %d chars", report_len)
        logger.info("  time_elapsed: %.2fs", elapsed)
        logger.info("="*80)

        generated_markdown = _finalize_newsletter_markdown(
            str(final_report.content or ""),
            source_metadata=state.get("source_metadata") or {},
        )
        return {
            "final_report": generated_markdown,
            "newsletter_title": _extract_newsletter_title(generated_markdown),
            "messages": ["Here is the final report: " + generated_markdown],
            "observability_events": [
                {
                    "category": "prompt_assembly",
                    "node": "final_report_generation",
                    "event_key": "final_report_generation:prompt_assembly",
                    "removed_truncation_markers": truncation_markers_removed,
                    "canonical_evidence_entries": len(evidence_ledger),
                    **evidence_packet_stats,
                }
            ],
        }
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error("FINAL_REPORT_GENERATION ERROR after %.2fs: %s", elapsed, e)
        logger.exception("Full traceback:")
        raise


# ===== COPYWRITER POLISH =====

async def copywriter_polish(state: AgentState):
    """
    Final copywriting polish node.
    
    Applies light editing for clarity, flow, and precision without changing content.
    """
    logger.info("="*80)
    logger.info("COPYWRITER_POLISH START")
    
    start_time = time.perf_counter()
    
    raw_final_report = state.get("final_report", "")
    raw_article_summary = state.get("article_summary", "")
    final_report = clean_prompt_artifacts(raw_final_report)
    incoming_title = _extract_newsletter_title(final_report) or _normalize_space(state.get("newsletter_title", ""))
    research_brief = state.get("research_brief", "")
    article_summary = _truncate(clean_prompt_artifacts(raw_article_summary), 8000)
    notes = state.get("notes", [])
    findings, findings_stats = _compile_findings_context(state)
    
    # Validate newsletter_template - fail fast if not set
    newsletter_template = state.get("newsletter_template")
    if not newsletter_template:
        raise ValueError("newsletter_template not set in state. Ensure select_newsletter_template runs before copywriter_polish.")
    
    logger.info("  input_report_len: %d chars", len(final_report))
    logger.info("  research_brief_len: %d chars", len(research_brief))
    logger.info("  article_summary_len: %d chars", len(article_summary))
    logger.info("  findings_len: %d chars | mode=%s", len(findings), findings_stats.get("mode", "unknown"))
    logger.info("  newsletter_template: %s", newsletter_template)
    
    if not final_report:
        logger.warning("  No final report to polish, skipping")
        return {}
    
    # Get formatted template for section compliance reference
    template_content = format_template_for_prompt(newsletter_template)
    
    try:
        polish_prompt = copywriter_polish_prompt.format(
            newsletter=final_report,
            research_brief=research_brief,
            article_summary=article_summary,
            findings=findings,
            template_content=template_content
        )
        logger.info("  prompt_len: %d chars", len(polish_prompt))
        logger.info("  Invoking copywriter model for final polish...")
        
        polished = await copywriter_model.ainvoke([HumanMessage(content=polish_prompt)])
        
        elapsed = time.perf_counter() - start_time
        output_len = len(polished.content) if polished.content else 0
        
        # Log token usage
        log_token_usage(logger, polished, "copywriter_polish")
        get_global_tracker().add_usage(polished, "copywriter_polish")
        
        logger.info("COPYWRITER_POLISH COMPLETE")
        logger.info("  output_len: %d chars (delta: %d)", output_len, output_len - len(final_report))
        logger.info("  time_elapsed: %.2fs", elapsed)
        logger.info("="*80)

        polished_markdown = _finalize_newsletter_markdown(
            str(polished.content or ""),
            fallback_title=incoming_title,
            source_metadata=state.get("source_metadata") or {},
        )
        
        return {
            "final_report": polished_markdown,
            "newsletter_title": _extract_newsletter_title(polished_markdown) or incoming_title,
            "messages": ["Here is the polished newsletter: " + polished_markdown],
            "observability_events": [
                {
                    "category": "prompt_assembly",
                    "node": "copywriter_polish",
                    "event_key": "copywriter_polish:prompt_assembly",
                    "removed_truncation_markers": _count_truncation_markers(raw_final_report)
                    + _count_truncation_markers(raw_article_summary)
                    + sum(_count_truncation_markers(note) for note in notes),
                }
            ],
        }
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error("COPYWRITER_POLISH ERROR after %.2fs: %s", elapsed, e)
        logger.exception("Full traceback:")
        # On error, return unpolished report rather than failing
        logger.warning("  Returning unpolished report due to error")
        return {}


# ===== CRITIQUE-REWRITE LOOP =====

def _deterministic_structure_issues(
    markdown: str,
    newsletter_template: str,
    *,
    require_sources: bool,
    enforce_article_only_cleanup: bool = False,
) -> list[str]:
    """Lightweight, template-aware checks to prevent brittle export failures."""
    if not markdown or not str(markdown).strip():
        return ["Empty newsletter markdown."]

    issues: list[str] = []
    text = str(markdown)

    if not re.search(r"^#\s+\S+", text, flags=re.MULTILINE):
        issues.append("Missing top-level title header (`# Title`).")

    try:
        required_sections = get_template_sections_list(newsletter_template)
    except Exception:
        required_sections = []

    for header in required_sections:
        if not re.search(rf"^##\s+{re.escape(header)}\s*$", text, flags=re.MULTILINE):
            issues.append(f"Missing required section header: `## {header}`.")

    sources_header_match = re.search(r"^##\s+Sources\s*$", text, flags=re.MULTILINE)
    if require_sources and not sources_header_match:
        issues.append("Missing required `## Sources` section.")
    elif sources_header_match:
        start = sources_header_match.end()
        next_header_match = re.search(r"^##\s+.+$", text[start:], flags=re.MULTILINE)
        end = start + next_header_match.start() if next_header_match else len(text)
        sources_block = text[start:end]
        if not re.search(r"^\s*(?:-\s*)?\[\d+\]\s+\S+", sources_block, flags=re.MULTILINE):
            issues.append("`## Sources` exists but has no numbered items like `[1] ...`.")

    if not require_sources and enforce_article_only_cleanup:
        if sources_header_match:
            issues.append("Article-only runs should omit the `## Sources` section.")
        if re.search(r"\[\d+\]", text):
            issues.append("Article-only runs should omit inline numeric citations.")

    return issues


def _deterministic_source_metadata_issues(
    source_metadata: dict[str, Any],
    article_content: str,
    article_summary: str,
) -> list[str]:
    issues: list[str] = []
    metadata = dict(source_metadata or {})
    title = _normalize_space(metadata.get("article_title", ""))
    doi = _normalize_space(metadata.get("doi", ""))
    expected_title, _ = _extract_title_and_authors_from_content(article_content, article_summary)
    expected_doi_match = _DOI_RE.search(f"{article_content}\n{article_summary}")
    expected_doi = expected_doi_match.group(0) if expected_doi_match else ""

    if title and _is_generic_source_title(title):
        issues.append("Source metadata title is a generic placeholder rather than the real source title.")
    if expected_title and not title:
        issues.append("Source metadata is missing an extractable source title.")
    elif expected_title and title:
        similarity = SequenceMatcher(None, _normalize_title_key(expected_title), _normalize_title_key(title)).ratio()
        if similarity < 0.55:
            issues.append(
                f"Source metadata title appears mismatched. Expected something close to `{expected_title}`, got `{title}`."
            )

    if expected_doi and not doi:
        issues.append("Source metadata is missing a DOI that is present in the source.")
    elif expected_doi and doi and expected_doi.lower() != doi.lower():
        issues.append(
            f"Source metadata DOI appears mismatched. Expected `{expected_doi}`, got `{doi}`."
        )

    return issues


async def critique_reflection(state: AgentState):
    """
    Critique reflection node.
    
    Evaluates the final report quality and determines if it needs another
    rewrite or is ready for final polish. Returns structured critique feedback.
    """
    logger.info("="*80)
    logger.info("CRITIQUE_REFLECTION START")
    
    start_time = time.perf_counter()
    
    raw_final_report = state.get("final_report", "")
    raw_article_summary = state.get("article_summary", "")
    final_report = clean_prompt_artifacts(raw_final_report)
    incoming_title = _extract_newsletter_title(final_report) or _normalize_space(state.get("newsletter_title", ""))
    research_brief = state.get("research_brief", "")
    article_summary = _truncate(clean_prompt_artifacts(raw_article_summary), 8000)
    article_content = state.get("article_content", "") or ""
    source_metadata = state.get("source_metadata") or {}
    notes = state.get("notes", [])
    findings, findings_stats = _compile_findings_context(state)
    critique_iterations = state.get("critique_iterations", 0)
    critique_view = _finalize_newsletter_markdown(
        final_report,
        fallback_title=incoming_title,
        source_metadata=source_metadata,
    )
    
    logger.info("  critique_iteration: %d/%d", critique_iterations + 1, MAX_CRITIQUE_ITERATIONS)
    logger.info("  final_report_len: %d chars", len(final_report))
    logger.info("  findings_len: %d chars | mode=%s", len(findings), findings_stats.get("mode", "unknown"))
    
    if not final_report:
        logger.warning("  No final report to critique, marking as complete")
        return {
            "critique_iterations": critique_iterations + 1,
            "critique_feedback": ""
        }
    
    try:
        # Build critique prompt
        critique_prompt = critique_reflection_prompt.format(
            date=get_today_str(),
            research_brief=research_brief,
            article_summary=article_summary,
            findings=findings,
            newsletter=critique_view
        )
        
        logger.info("  prompt_len: %d chars", len(critique_prompt))
        logger.info("  Invoking critique model...")
        
        # Use structured output for reliable parsing
        critique_model_structured = critique_model.with_structured_output(CritiqueReflection)
        critique_result = await critique_model_structured.ainvoke([HumanMessage(content=critique_prompt)])
        
        elapsed = time.perf_counter() - start_time
        
        logger.info("CRITIQUE_REFLECTION COMPLETE")
        logger.info("  is_complete: %s", critique_result.is_complete)
        logger.info("  quality_score: %d/10", critique_result.quality_score)
        logger.info("  time_elapsed: %.2fs", elapsed)
        
        # Deterministic structural checks (micro-validator) to avoid "looks good" but unexportable output.
        newsletter_template = state.get("newsletter_template") or ""
        evidence_ledger = canonicalize_evidence_ledger(state.get("evidence_ledger", []) or [])
        claim_source_map = claim_source_map_from_markdown(
            critique_view,
            evidence_ledger,
        )
        structural_issues = _deterministic_structure_issues(
            critique_view,
            newsletter_template,
            require_sources=bool(claim_source_map.get("requires_sources_section", False)),
            enforce_article_only_cleanup=bool(claim_source_map.get("article_only_claims", True)),
        )
        structural_issues.extend(
            _deterministic_source_metadata_issues(
                source_metadata,
                article_content,
                article_summary,
            )
        )
        if structural_issues:
            critique_result.is_complete = False
            critique_result.issues = (
                (critique_result.issues + "\n\n" if critique_result.issues else "")
                + "Deterministic structural issues:\n"
                + "\n".join(f"- {i}" for i in structural_issues)
            )
            critique_result.actionable_feedback = (
                (critique_result.actionable_feedback + "\n\n" if critique_result.actionable_feedback else "")
                + "Fix the structure and citation mode exactly (title, required sections, and sources only when external citations are actually in play) before any stylistic edits."
            )

        if critique_result.is_complete:
            logger.info("  -> Newsletter deemed complete, proceeding to polish")
        else:
            logger.info("  -> Issues found, will trigger rewrite")
            logger.info("  issues: %s", critique_result.issues[:200] + "..." if len(critique_result.issues) > 200 else critique_result.issues)
        
        logger.info("="*80)
        
        # Store critique feedback for rewrite node
        critique_feedback = f"Strengths: {critique_result.strengths}\n\nIssues: {critique_result.issues}\n\nActionable: {critique_result.actionable_feedback}"
        
        return {
            "critique_iterations": critique_iterations + 1,
            "critique_feedback": critique_feedback,
            # Store structured fields for routing decision
            "_critique_is_complete": critique_result.is_complete,
            "_critique_strengths": critique_result.strengths,
            "_critique_issues": critique_result.issues,
            "_critique_actionable": critique_result.actionable_feedback,
            "claim_source_map": claim_source_map,
            "observability_events": [
                {
                    "category": "prompt_assembly",
                    "node": "critique_reflection",
                    "event_key": f"critique_reflection:{critique_iterations + 1}",
                    "removed_truncation_markers": _count_truncation_markers(raw_final_report)
                    + _count_truncation_markers(raw_article_summary)
                    + sum(_count_truncation_markers(note) for note in notes),
                }
            ],
        }
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error("CRITIQUE_REFLECTION ERROR after %.2fs: %s", elapsed, e)
        logger.exception("Full traceback:")
        # On error, skip critique and proceed to polish
        return {
            "critique_iterations": critique_iterations + 1,
            "critique_feedback": "",
            "_critique_is_complete": True  # Explicit: skip to polish on error
        }


async def rewrite_report(state: AgentState):
    """
    Rewrite report node.
    
    Rewrites the final report based on critique feedback.
    """
    logger.info("="*80)
    logger.info("REWRITE_REPORT START")
    
    start_time = time.perf_counter()
    
    raw_final_report = state.get("final_report", "")
    raw_article_summary = state.get("article_summary", "")
    final_report = clean_prompt_artifacts(raw_final_report)
    incoming_title = _extract_newsletter_title(final_report) or _normalize_space(state.get("newsletter_title", ""))
    research_brief = state.get("research_brief", "")
    article_summary = _truncate(clean_prompt_artifacts(raw_article_summary), 8000)
    notes = state.get("notes", [])
    findings, findings_stats = _compile_findings_context(state)
    critique_iterations = state.get("critique_iterations", 0)
    
    # Get critique details from state
    strengths = state.get("_critique_strengths", "")
    issues = state.get("_critique_issues", "")
    actionable_feedback = state.get("_critique_actionable", "")
    
    # Validate newsletter_template
    newsletter_template = state.get("newsletter_template")
    if not newsletter_template:
        raise ValueError("newsletter_template not set in state.")
    
    logger.info("  rewrite_iteration: %d/%d", critique_iterations, MAX_CRITIQUE_ITERATIONS)
    logger.info("  input_report_len: %d chars", len(final_report))
    logger.info("  findings_len: %d chars | mode=%s", len(findings), findings_stats.get("mode", "unknown"))
    logger.info("  strengths_len: %d chars", len(strengths))
    logger.info("  issues_len: %d chars", len(issues))
    
    # Get formatted template
    template_content = format_template_for_prompt(newsletter_template)
    
    try:
        rewrite_prompt = rewrite_with_critique_prompt.format(
            date=get_today_str(),
            research_brief=research_brief,
            article_summary=article_summary,
            findings=findings,
            newsletter=final_report,
            strengths=strengths,
            issues=issues,
            actionable_feedback=actionable_feedback,
            template_content=template_content
        )
        
        logger.info("  prompt_len: %d chars", len(rewrite_prompt))
        logger.info("  Invoking writer model for rewrite...")
        
        rewritten = await writer_model.ainvoke([HumanMessage(content=rewrite_prompt)])
        
        elapsed = time.perf_counter() - start_time
        output_len = len(rewritten.content) if rewritten.content else 0
        
        # Log token usage
        log_token_usage(logger, rewritten, "rewrite_report")
        get_global_tracker().add_usage(rewritten, "rewrite_report")
        
        logger.info("REWRITE_REPORT COMPLETE")
        logger.info("  output_len: %d chars (delta: %d)", output_len, output_len - len(final_report))
        logger.info("  time_elapsed: %.2fs", elapsed)
        logger.info("="*80)

        rewritten_markdown = _finalize_newsletter_markdown(
            str(rewritten.content or ""),
            fallback_title=incoming_title,
            source_metadata=state.get("source_metadata") or {},
        )
        
        return {
            "final_report": rewritten_markdown,
            "newsletter_title": _extract_newsletter_title(rewritten_markdown) or incoming_title,
            # Clear critique details after use
            "_critique_is_complete": None,
            "_critique_strengths": "",
            "_critique_issues": "",
            "_critique_actionable": "",
            "observability_events": [
                {
                    "category": "prompt_assembly",
                    "node": "rewrite_report",
                    "event_key": f"rewrite_report:{critique_iterations}",
                    "removed_truncation_markers": _count_truncation_markers(raw_final_report)
                    + _count_truncation_markers(raw_article_summary)
                    + sum(_count_truncation_markers(note) for note in notes),
                }
            ],
        }
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error("REWRITE_REPORT ERROR after %.2fs: %s", elapsed, e)
        logger.exception("Full traceback:")
        # On error, keep existing report
        return {}


def route_after_critique(state: AgentState) -> str:
    """
    Route after critique reflection.
    
    Returns:
        - "rewrite_report" if critique found issues and we haven't hit max iterations
        - "copywriter_polish" if critique approved or max iterations reached
    """
    critique_iterations = state.get("critique_iterations", 0)
    is_complete = state.get("_critique_is_complete", True)  # Default to complete if not set
    
    # Force exit if max iterations reached
    if critique_iterations >= MAX_CRITIQUE_ITERATIONS:
        logger.info("ROUTE_AFTER_CRITIQUE: Max iterations (%d) reached, forcing exit to polish", 
                   MAX_CRITIQUE_ITERATIONS)
        return "copywriter_polish"
    
    # Exit if critique approved
    if is_complete:
        logger.info("ROUTE_AFTER_CRITIQUE: Critique approved (iteration %d), proceeding to polish", 
                   critique_iterations)
        return "copywriter_polish"
    
    # Otherwise, rewrite
    logger.info("ROUTE_AFTER_CRITIQUE: Issues found (iteration %d), triggering rewrite", 
               critique_iterations)
    return "rewrite_report"


# ===== FINAL MARKDOWN NORMALIZATION =====

async def finalize_newsletter_markdown(state: AgentState):
    """Deterministically restore the title and inject structured source metadata."""
    logger.info("=" * 80)
    logger.info("FINALIZE_NEWSLETTER_MARKDOWN START")

    start_time = time.perf_counter()
    final_report = clean_prompt_artifacts(state.get("final_report", ""))
    newsletter_title = _extract_newsletter_title(final_report) or _normalize_space(state.get("newsletter_title", ""))
    source_metadata = state.get("source_metadata") or {}

    finalized_markdown = _finalize_newsletter_markdown(
        final_report,
        fallback_title=newsletter_title,
        source_metadata=source_metadata,
    )

    logger.info(
        "FINALIZE_NEWSLETTER_MARKDOWN COMPLETE | output_len=%d chars | title=%s | metadata_fields=%d | time=%.2fs",
        len(finalized_markdown),
        _truncate(_extract_newsletter_title(finalized_markdown), 90),
        sum(1 for key in ("article_title", "authors", "journal", "publication_date", "doi") if source_metadata.get(key)),
        time.perf_counter() - start_time,
    )
    logger.info("=" * 80)

    return {
        "final_report": finalized_markdown,
        "newsletter_title": _extract_newsletter_title(finalized_markdown) or newsletter_title,
    }


# ===== STRUCTURE OUTPUT =====

# Output directory for JSON exports
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


def _extract_source_filename(state: AgentState) -> str:
    """
    Extract the source filename from state.
    
    Looks for:
    1. state["source_filename"] if set explicitly
    2. state["pdf_path"] - extracts filename from PDF path
    3. Falls back to timestamp-based name
    
    All filenames are sanitized to be safe for file system use.
    """
    # Check if explicitly set
    source_filename = state.get("source_filename")
    if source_filename:
        return _sanitize_filename(source_filename)
    
    # Extract from PDF path in state
    pdf_path = state.get("pdf_path")
    if pdf_path:
        # Extract just the filename without extension and sanitize
        return _sanitize_filename(Path(pdf_path).stem)
    
    # Fallback to timestamp (already safe, but sanitize for consistency)
    return f"newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _export_newsletter_json(structured_dict: dict, filename: str) -> str:
    """
    Export the structured newsletter to a JSON file in the outputs folder.
    
    Args:
        structured_dict: The structured newsletter dictionary
        filename: Base filename (without extension)
    
    Returns:
        Path to the exported JSON file
    """
    # Ensure outputs directory exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create output path
    output_path = OUTPUTS_DIR / f"{filename}.json"
    
    # Write JSON atomically to avoid partial/corrupt artifacts on failure.
    _write_json_atomic(output_path, structured_dict)

    return str(output_path)


def _export_newsletter_markdown(markdown: str, filename: str) -> str:
    """Export the raw newsletter markdown alongside the structured JSON."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"{filename}.md"
    normalized_markdown = str(markdown or "")
    if normalized_markdown and not normalized_markdown.endswith("\n"):
        normalized_markdown += "\n"
    _write_text_atomic(output_path, normalized_markdown)
    return str(output_path)


def _build_export_result(
    *,
    source_filename: str,
    structured_newsletter: dict | None = None,
    claim_source_map: dict | None = None,
    structured_output_path: str | None = None,
    raw_markdown_path: str | None = None,
    export_status: str,
    export_error: str = "",
) -> dict:
    result = {
        "source_filename": source_filename,
        "export_status": export_status,
        "structured_output_path": structured_output_path,
        "raw_markdown_path": raw_markdown_path,
        "export_error": export_error,
    }
    if structured_newsletter is not None:
        result["structured_newsletter"] = structured_newsletter
    if claim_source_map is not None:
        result["claim_source_map"] = claim_source_map
    return result


async def structure_newsletter_output(state: AgentState):
    """
    Parse the final newsletter markdown into a structured format for website import.
    
    This creates a JSON-serializable structure with sections array that works
    for both research_article and commentary newsletter types.
    
    Exports both a structured JSON artifact and a raw markdown backup in `outputs/`.
    """
    logger.info("="*80)
    logger.info("STRUCTURE_NEWSLETTER_OUTPUT START")
    
    start_time = time.perf_counter()
    
    source_metadata = state.get("source_metadata") or {}
    final_report = _finalize_newsletter_markdown(
        state.get("final_report", ""),
        fallback_title=_normalize_space(state.get("newsletter_title", "")),
        source_metadata=source_metadata,
    )
    source_filename = _extract_source_filename(state)
    
    # Validate newsletter_template - fail fast if not set
    newsletter_template = state.get("newsletter_template")
    
    logger.info("  input_report_len: %d chars", len(final_report))
    logger.info("  newsletter_template: %s", newsletter_template)
    logger.info("  source_filename: %s", source_filename)
    
    if not final_report:
        logger.warning("  No final report to structure, skipping")
        return _build_export_result(
            source_filename=source_filename,
            export_status="skipped",
            export_error="No final report available for export.",
        )
    
    raw_markdown_path: str | None = None

    try:
        raw_markdown_path = _export_newsletter_markdown(final_report, source_filename)
        logger.info("  raw_markdown_exported_to: %s", raw_markdown_path)

        if not newsletter_template:
            raise ValueError(
                "newsletter_template not set in state. Ensure select_newsletter_template runs before structure_newsletter_output."
            )

        evidence_ledger = canonicalize_evidence_ledger(state.get("evidence_ledger", []) or [])
        citation_policy = citation_rendering_policy(evidence_ledger)
        claim_source_map = claim_source_map_from_markdown(final_report, evidence_ledger)
        structural_issues = _deterministic_structure_issues(
            final_report,
            newsletter_template,
            require_sources=bool(claim_source_map.get("requires_sources_section", False)),
            enforce_article_only_cleanup=False,
        )
        if structural_issues:
            raise ValueError(" | ".join(structural_issues))

        # Parse markdown to structured format (template used for tagging)
        structured = parse_newsletter_to_structured(final_report, newsletter_template)
        structured_dict = structured_newsletter_to_dict(structured)
        if source_metadata:
            structured_dict.setdefault("metadata", {})["source_article"] = {
                "title": source_metadata.get("article_title", ""),
                "authors": list(source_metadata.get("authors", []) or []),
                "publication_date": source_metadata.get("publication_date", ""),
                "journal": source_metadata.get("journal", ""),
                "doi": source_metadata.get("doi", ""),
            }
        
        if (
            not structured_dict.get("sources")
            and evidence_ledger
            and bool(claim_source_map.get("requires_sources_section", False))
        ):
            structured_dict["sources"] = sources_from_evidence_ledger(
                evidence_ledger,
                include_article_sources=bool(citation_policy.get("requires_sources_section", False)),
            )
        if not bool(claim_source_map.get("requires_sources_section", False)):
            structured_dict["sources"] = []
        if evidence_ledger:
            structured_dict["evidence_ledger"] = evidence_ledger
            structured_dict.setdefault("metadata", {})["evidence_ledger_entries"] = len(evidence_ledger)
        structured_dict.setdefault("metadata", {})["claim_source_mode"] = claim_source_map.get("mode", "")
        structured_dict.setdefault("metadata", {})["claim_requires_sources"] = bool(
            claim_source_map.get("requires_sources_section", False)
        )
        
        # Export to JSON file
        output_path = _export_newsletter_json(structured_dict, source_filename)
        logger.info("  exported_to: %s", output_path)
        
        elapsed = time.perf_counter() - start_time
        
        logger.info("STRUCTURE_NEWSLETTER_OUTPUT COMPLETE")
        logger.info("  newsletter_structure: %s", structured_dict.get("newsletter_structure"))
        logger.info("  title: %s", structured_dict.get("title", "")[:50])
        logger.info("  sections_count: %d", len(structured_dict.get("sections", [])))
        logger.info("  sources_count: %d", len(structured_dict.get("sources", [])))
        logger.info("  output_file: %s", output_path)
        logger.info("  time_elapsed: %.2fs", elapsed)
        logger.info("="*80)
        
        return _build_export_result(
            source_filename=source_filename,
            structured_newsletter=structured_dict,
            claim_source_map=claim_source_map,
            structured_output_path=output_path,
            raw_markdown_path=raw_markdown_path,
            export_status="structured_json",
        )
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error("STRUCTURE_NEWSLETTER_OUTPUT ERROR after %.2fs: %s", elapsed, e)
        logger.exception("Full traceback:")
        if not raw_markdown_path:
            try:
                raw_markdown_path = _export_newsletter_markdown(final_report, source_filename)
                logger.warning("  Fallback raw markdown exported to: %s", raw_markdown_path)
            except Exception as fallback_exc:
                logger.error("  Raw markdown fallback export failed: %s", fallback_exc)
                logger.exception("Full fallback traceback:")
                return _build_export_result(
                    source_filename=source_filename,
                    export_status="failed",
                    export_error=(
                        f"Structured export failed: {e}. "
                        f"Raw markdown fallback failed: {fallback_exc}"
                    ),
                )

        return _build_export_result(
            source_filename=source_filename,
            claim_source_map=claim_source_map_from_markdown(
                final_report,
                canonicalize_evidence_ledger(state.get("evidence_ledger", []) or []),
            ),
            raw_markdown_path=raw_markdown_path,
            export_status="fallback_raw_markdown",
            export_error=f"Structured export failed: {e}",
        )


# ===== GRAPH CONSTRUCTION =====
# Build the overall workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("ingest_document", ingest_document_node)
deep_researcher_builder.add_node("select_newsletter_template", select_newsletter_template)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("write_draft_report", write_draft_report)
deep_researcher_builder.add_node("select_fixed_perspectives", select_fixed_perspectives)
deep_researcher_builder.add_node("initialize_research_agenda", initialize_research_agenda)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)
deep_researcher_builder.add_node("critique_reflection", critique_reflection)
deep_researcher_builder.add_node("rewrite_report", rewrite_report)
deep_researcher_builder.add_node("copywriter_polish", copywriter_polish)
deep_researcher_builder.add_node("finalize_newsletter_markdown", finalize_newsletter_markdown)
deep_researcher_builder.add_node("structure_newsletter_output", structure_newsletter_output)

# Add workflow edges
# Flow: ingest -> select_template -> write_research_brief -> write_draft -> supervisor -> final_report
#       -> critique_reflection <-> rewrite_report (loop up to 5x)
#       -> copywriter_polish -> finalize_newsletter_markdown -> structure_output
deep_researcher_builder.add_edge(START, "ingest_document")
deep_researcher_builder.add_edge("ingest_document", "select_newsletter_template")
deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")
deep_researcher_builder.add_edge("write_draft_report", "select_fixed_perspectives")
deep_researcher_builder.add_edge("select_fixed_perspectives", "initialize_research_agenda")
deep_researcher_builder.add_edge("initialize_research_agenda", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", "critique_reflection")

# Critique-rewrite loop with conditional routing
deep_researcher_builder.add_conditional_edges(
    "critique_reflection",
    route_after_critique,
    {
        "rewrite_report": "rewrite_report",
        "copywriter_polish": "copywriter_polish"
    }
)
deep_researcher_builder.add_edge("rewrite_report", "critique_reflection")  # Loop back

# Final polish and output
deep_researcher_builder.add_edge("copywriter_polish", "finalize_newsletter_markdown")
deep_researcher_builder.add_edge("finalize_newsletter_markdown", "structure_newsletter_output")
deep_researcher_builder.add_edge("structure_newsletter_output", END)

# Compile the full workflow
agent = deep_researcher_builder.compile()
