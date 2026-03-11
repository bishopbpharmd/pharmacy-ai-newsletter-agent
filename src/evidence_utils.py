"""Utilities for canonical evidence, answer shaping, and observability handling."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_prompt_artifacts(text: Any) -> str:
    value = str(text or "")
    value = value.replace("\n[...truncated...]", "")
    value = value.replace("\n\n[...truncated...]\n", "\n")
    value = value.replace("[...truncated...]", "")
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _truncate_compact(text: Any, max_chars: int) -> str:
    value = normalize_text(text)
    if len(value) <= max_chars:
        return value
    sentences = re.split(r"(?<=[.!?])\s+", value)
    kept: List[str] = []
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
    return value[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")


def strip_sources_block(text: Any) -> str:
    lines = str(text or "").splitlines()
    cleaned_lines: List[str] = []
    skipping_sources = False

    for raw_line in lines:
        line = raw_line.strip()
        if re.match(r"^(?:#+\s*)?sources(?:\s+used)?\s*:?\s*$", line, flags=re.IGNORECASE):
            skipping_sources = True
            continue

        if skipping_sources:
            if not line:
                continue
            if re.match(r"^(?:[-*]|\[\d+\])\s+", line):
                continue
            if "http://" in line or "https://" in line or "id=user_article" in line:
                continue
            skipping_sources = False

        cleaned_lines.append(raw_line)

    return "\n".join(cleaned_lines).strip()


def _answer_units(text: Any) -> List[str]:
    stripped_text = strip_sources_block(clean_prompt_artifacts(text))
    units: List[str] = []
    paragraph_buffer: List[str] = []

    for raw_line in stripped_text.splitlines():
        line = raw_line.strip()
        if not line:
            if paragraph_buffer:
                units.append(normalize_text(" ".join(paragraph_buffer)))
                paragraph_buffer = []
            continue

        if re.match(r"^\[\d+\]\s+", line):
            continue

        bullet_match = re.match(r"^(?:[-*]|\d+[.)])\s+(.*)$", line)
        if bullet_match:
            if paragraph_buffer:
                units.append(normalize_text(" ".join(paragraph_buffer)))
                paragraph_buffer = []
            candidate = normalize_text(bullet_match.group(1))
            if candidate:
                units.append(candidate)
            continue

        paragraph_buffer.append(line)

    if paragraph_buffer:
        units.append(normalize_text(" ".join(paragraph_buffer)))

    return [unit for unit in units if unit]


def is_external_evidence_item(item: Dict[str, Any]) -> bool:
    source_type = normalize_text(item.get("source_type", ""))
    locator = str(item.get("locator", "") or "").strip()
    if source_type == "article_chunk":
        return False
    return locator.startswith("http://") or locator.startswith("https://") or source_type == "web"


def citation_rendering_policy(evidence_ledger: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    article_locators: set[str] = set()
    external_locators: set[str] = set()

    for entry in canonicalize_evidence_ledger(evidence_ledger or []):
        for evidence in entry.get("evidence", []) or []:
            locator = normalize_text(evidence.get("locator", "")) or normalize_text(
                evidence.get("title", "")
            )
            if not locator:
                continue
            if is_external_evidence_item(evidence):
                external_locators.add(locator)
            else:
                article_locators.add(locator)

    has_external_sources = bool(external_locators)
    article_only = bool(article_locators) and not has_external_sources

    if has_external_sources:
        guidance = (
            "If external-backed claims survive in the final newsletter, render inline citations and a ## Sources section. "
            "If the final draft stays article-only, omit numeric citation chrome."
        )
    else:
        guidance = (
            "This run is article-only. Do not render inline numeric citations or a ## Sources section "
            "unless external grounding actually appears."
        )

    return {
        "article_source_count": len(article_locators),
        "external_source_count": len(external_locators),
        "article_only": article_only or not has_external_sources,
        "has_external_sources": has_external_sources,
        "requires_sources_section": has_external_sources,
        "render_inline_citations": has_external_sources,
        "guidance": guidance,
    }


_MARKDOWN_URL_PATTERN = re.compile(r"https?://[^\s)>\]]+", flags=re.IGNORECASE)
_SOURCE_LINE_PATTERN = re.compile(r"^\s*(?:[-*]\s*)?\[(\d+)\]\s+(.+?)\s*$")


def claim_source_map_from_markdown(
    markdown: Any,
    evidence_ledger: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Derive claim-level citation requirements from final markdown, not retrieval history alone."""
    text = str(markdown or "")
    header_match = re.search(r"^##\s+Sources?\s*$", text, flags=re.MULTILINE)
    has_sources_section = header_match is not None
    sources_block = ""
    body_text = text

    if header_match:
        start = header_match.end()
        next_header = re.search(r"^##\s+.+$", text[start:], flags=re.MULTILINE)
        end = start + next_header.start() if next_header else len(text)
        sources_block = text[start:end].strip()
        body_text = text[: header_match.start()].strip()

    inline_citation_ids = sorted(
        {int(match.group(1)) for match in re.finditer(r"\[(\d+)\]", body_text)}
    )
    inline_external_urls = [url.rstrip(").,;") for url in _MARKDOWN_URL_PATTERN.findall(body_text)]

    external_source_ids: set[int] = set()
    article_source_ids: set[int] = set()
    unknown_source_ids: set[int] = set()

    for raw_line in sources_block.splitlines():
        match = _SOURCE_LINE_PATTERN.match(raw_line.strip())
        if not match:
            continue
        source_id = int(match.group(1))
        source_text = match.group(2).strip().lower()
        has_url = bool(_MARKDOWN_URL_PATTERN.search(source_text))
        is_article_hint = "provided article" in source_text or "user_article" in source_text
        if has_url:
            external_source_ids.add(source_id)
        elif is_article_hint:
            article_source_ids.add(source_id)
        else:
            unknown_source_ids.add(source_id)

    cited_ids = set(inline_citation_ids)
    cited_external_ids = sorted(cited_ids & external_source_ids)
    cited_unknown_ids = sorted(cited_ids & unknown_source_ids)
    cited_unresolved_ids = sorted(
        cited_ids - external_source_ids - article_source_ids - unknown_source_ids
    )

    external_claims_present = bool(
        cited_external_ids
        or cited_unknown_ids
        or cited_unresolved_ids
        or inline_external_urls
    )
    requires_sources_section = external_claims_present
    claim_mode = "external_backed" if external_claims_present else "article_only"

    run_policy = citation_rendering_policy(evidence_ledger or [])
    external_evidence_available = bool(run_policy.get("has_external_sources", False))

    return {
        "mode": claim_mode,
        "requires_sources_section": requires_sources_section,
        "has_sources_section": has_sources_section,
        "inline_citation_ids": inline_citation_ids,
        "inline_external_url_count": len(inline_external_urls),
        "inline_external_urls": inline_external_urls[:12],
        "sources_external_ids": sorted(external_source_ids),
        "sources_article_ids": sorted(article_source_ids),
        "sources_unknown_ids": sorted(unknown_source_ids),
        "cited_external_ids": cited_external_ids,
        "cited_unknown_ids": cited_unknown_ids,
        "cited_unresolved_ids": cited_unresolved_ids,
        "external_claims_present": external_claims_present,
        "article_only_claims": not external_claims_present,
        "external_evidence_available": external_evidence_available,
    }


def source_mix_report(
    evidence_ledger: Iterable[Dict[str, Any]],
    retrieval_events: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    article_queries = 0
    external_queries = 0
    fallback_external_queries = 0

    for event in retrieval_events or []:
        stage = normalize_text(event.get("stage", ""))
        tool_name = normalize_text(event.get("tool_name", ""))
        if stage not in {"executed", "executed_fallback"}:
            continue
        if tool_name == "retrieve_document_chunks":
            article_queries += 1
        elif tool_name == "tavily_search":
            external_queries += 1
            if stage == "executed_fallback":
                fallback_external_queries += 1

    policy = citation_rendering_policy(evidence_ledger)
    article_sources = int(policy.get("article_source_count", 0))
    external_sources = int(policy.get("external_source_count", 0))

    if external_queries or external_sources:
        summary = (
            f"Local retrieval executions: {article_queries}; external retrieval executions: {external_queries}; "
            f"article evidence sources: {article_sources}; external evidence sources: {external_sources}."
        )
    elif article_queries or article_sources:
        summary = (
            f"Article-only so far: {article_queries} local retrieval executions, "
            f"{article_sources} article evidence sources, and no external grounding executed."
        )
    else:
        summary = "No meaningful research evidence has been recorded yet."

    return {
        "article_queries": article_queries,
        "external_queries": external_queries,
        "fallback_external_queries": fallback_external_queries,
        "article_sources": article_sources,
        "external_sources": external_sources,
        "article_only": bool(policy.get("article_only", True)),
        "summary": summary,
    }


def canonicalize_answer_summary(
    answer: Any,
    *,
    answer_status: str = "",
    max_chars: int = 750,
    max_units: int = 3,
) -> str:
    units = _answer_units(answer)
    normalized_status = normalize_text(answer_status).lower()

    if not units:
        return _truncate_compact(answer, max_chars)

    if normalized_status in {"missing", "not_in_source", "premise_mismatch"}:
        missing_units = [
            unit
            for unit in units
            if any(phrase in unit.lower() for phrase in _MISSING_ANSWER_PHRASES)
        ]
        selected = missing_units[:2] or units[:2]
        if len(selected) == 1:
            for unit in units:
                if unit not in selected:
                    selected.append(unit)
                    break
        return _truncate_compact(" ".join(selected), min(max_chars, 320))

    selected: List[str] = []
    for unit in units:
        selected.append(unit)
        if len(selected) >= max_units:
            break

    if len(selected) == 1:
        return _truncate_compact(selected[0], max_chars)

    primary = selected[0]
    supporting = [f"- {unit}" for unit in selected[1:]]
    return _truncate_compact("\n".join([primary, *supporting]), max_chars)


def evidence_item_signature(item: Dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_text(item.get("source_type", "")),
        normalize_text(item.get("locator", "")),
        normalize_text(item.get("snippet", ""))[:240],
    )


def merge_evidence_items(*groups: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for group in groups:
        for raw_item in group or []:
            item = dict(raw_item or {})
            key = evidence_item_signature(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def ledger_entry_signature(entry: Dict[str, Any]) -> tuple[str, str, str, str, str]:
    qa_id = normalize_text(entry.get("qa_id", ""))
    perspective = normalize_text(entry.get("perspective", ""))
    question = normalize_text(entry.get("question", ""))
    answer_origin = normalize_text(entry.get("answer_origin", ""))
    answer = normalize_text(entry.get("answer", ""))[:1000]
    return (
        qa_id or perspective,
        question,
        answer_origin,
        answer,
        perspective,
    )


def canonicalize_evidence_ledger(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    index_by_key: Dict[tuple[str, str, str, str, str], int] = {}

    for raw_entry in entries or []:
        entry = dict(raw_entry or {})
        entry["evidence"] = merge_evidence_items(entry.get("evidence", []) or [])
        key = ledger_entry_signature(entry)

        if key in index_by_key:
            existing_idx = index_by_key[key]
            existing = ordered[existing_idx]
            existing["evidence"] = merge_evidence_items(
                existing.get("evidence", []) or [],
                entry.get("evidence", []) or [],
            )
            for field in (
                "research_topic",
                "research_plan",
                "search_type",
                "answer_origin",
            ):
                if entry.get(field):
                    existing[field] = entry[field]
            continue

        index_by_key[key] = len(ordered)
        ordered.append(entry)

    return ordered


def merge_evidence_ledgers(left: Iterable[Dict[str, Any]], right: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return canonicalize_evidence_ledger([*(left or []), *(right or [])])


def retrieval_event_signature(event: Dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        normalize_text(event.get("stage", "")),
        normalize_text(event.get("qa_id", "")),
        normalize_text(event.get("tool_name", "")),
        normalize_text(event.get("retrieval_query", "")),
        normalize_text(event.get("event_key", "")),
    )


def merge_retrieval_events(left: Iterable[Dict[str, Any]], right: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for group in (left or [], right or []):
        for raw_event in group:
            event = dict(raw_event or {})
            key = retrieval_event_signature(event)
            if key in seen:
                continue
            seen.add(key)
            merged.append(event)
    return merged


def observability_event_signature(event: Dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_text(event.get("category", "")),
        normalize_text(event.get("event_key", "")),
        normalize_text(event.get("node", "")),
    )


def merge_observability_events(left: Iterable[Dict[str, Any]], right: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for group in (left or [], right or []):
        for raw_event in group:
            event = dict(raw_event or {})
            key = observability_event_signature(event)
            if key in seen:
                continue
            seen.add(key)
            merged.append(event)
    return merged


def merge_gap_cards(left: Iterable[Dict[str, Any]], right: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    by_id: Dict[str, int] = {}
    for group in (left or [], right or []):
        for raw_card in group:
            card = dict(raw_card or {})
            gap_id = normalize_text(card.get("gap_id", ""))
            if not gap_id:
                continue
            card["gap_id"] = gap_id
            if gap_id in by_id:
                merged[by_id[gap_id]] = {**merged[by_id[gap_id]], **card}
            else:
                by_id[gap_id] = len(merged)
                merged.append(card)
    return merged


def merge_gap_ledger(left: Iterable[Dict[str, Any]], right: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    by_id: Dict[str, int] = {}
    for group in (left or [], right or []):
        for raw_entry in group:
            entry = dict(raw_entry or {})
            gap_id = normalize_text(entry.get("gap_id", ""))
            if not gap_id:
                continue
            entry["gap_id"] = gap_id
            if gap_id in by_id:
                merged[by_id[gap_id]] = {**merged[by_id[gap_id]], **entry}
            else:
                by_id[gap_id] = len(merged)
                merged.append(entry)
    return merged


def dedupe_text_list(items: Iterable[Any]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for item in items or []:
        normalized = normalize_text(clean_prompt_artifacts(item))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(str(item).strip())
    return deduped


def evidence_locator_set(entries: Iterable[Dict[str, Any]]) -> set[str]:
    locators: set[str] = set()
    for entry in entries or []:
        for evidence in entry.get("evidence", []) or []:
            locator = normalize_text(evidence.get("locator", ""))
            if locator:
                locators.add(locator)
    return locators


def compute_evidence_novelty(
    existing_entries: Iterable[Dict[str, Any]],
    new_entries: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    existing_canonical = canonicalize_evidence_ledger(existing_entries or [])
    new_canonical = canonicalize_evidence_ledger(new_entries or [])

    existing_keys = {ledger_entry_signature(entry) for entry in existing_canonical}
    new_keys = {ledger_entry_signature(entry) for entry in new_canonical}
    novel_keys = new_keys - existing_keys

    existing_locators = evidence_locator_set(existing_canonical)
    new_locators = evidence_locator_set(new_canonical)
    novel_locators = new_locators - existing_locators

    total_new_items = max(len(new_canonical), 1)
    novelty_ratio = len(novel_keys) / total_new_items

    return {
        "new_entries": len(new_canonical),
        "novel_entries": len(novel_keys),
        "new_locators": len(new_locators),
        "novel_locators": len(novel_locators),
        "novelty_ratio": round(novelty_ratio, 3),
        "is_low_novelty": len(novel_keys) == 0 and len(novel_locators) <= 1,
    }


_STATUS_RANK = {
    "premise_mismatch": 0,
    "missing": 0,
    "not_in_source": 1,
    "needs_review": 2,
    "conflicted": 3,
    "supported": 4,
}


def _research_state_key(entry: Dict[str, Any]) -> tuple[str, str]:
    return (
        normalize_text(entry.get("perspective", "")) or normalize_text(entry.get("qa_id", "")),
        normalize_text(entry.get("question", "")),
    )


def compute_research_round_impact(
    existing_entries: Iterable[Dict[str, Any]],
    new_entries: Iterable[Dict[str, Any]],
    retrieval_events: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    existing_canonical = canonicalize_evidence_ledger(existing_entries or [])
    new_canonical = canonicalize_evidence_ledger(new_entries or [])
    retrieval_events = list(retrieval_events or [])

    existing_by_key = {_research_state_key(entry): entry for entry in existing_canonical}

    status_counts = {
        "supported": 0,
        "conflicted": 0,
        "needs_review": 0,
        "not_in_source": 0,
        "premise_mismatch": 0,
        "missing": 0,
    }
    status_upgrades = 0
    newly_supported = 0
    newly_bounded_absence = 0

    for entry in new_canonical:
        status = normalize_text(entry.get("answer_status", "supported")).lower() or "supported"
        if status not in status_counts:
            status = "needs_review"
        status_counts[status] += 1

        key = _research_state_key(entry)
        previous = existing_by_key.get(key)
        previous_rank = _STATUS_RANK.get(
            normalize_text((previous or {}).get("answer_status", "missing")).lower(),
            0,
        )
        current_rank = _STATUS_RANK.get(status, 1)
        if previous and current_rank > previous_rank and status in {"supported", "conflicted", "needs_review"}:
            status_upgrades += 1
        if status == "supported" and previous_rank < _STATUS_RANK["supported"]:
            newly_supported += 1
        if status == "not_in_source" and previous_rank < _STATUS_RANK["not_in_source"]:
            newly_bounded_absence += 1

    mix = source_mix_report(new_canonical, retrieval_events)

    external_events = [
        dict(event or {})
        for event in retrieval_events
        if str((event or {}).get("tool_name", "") or "").strip() == "tavily_search"
    ]
    external_query_flags = {
        str(flag or "")
        for event in external_events
        for flag in ((event or {}).get("query_quality_flags", []) or [])
        if str(flag or "")
    }
    external_shape_reasons = " ".join(
        str((event or {}).get("query_shape_reason", "") or "")
        for event in external_events
    ).lower()
    meaningful_external_grounding = bool(external_events) and (
        "title_anchored" in external_shape_reasons
        or "doi_anchored" in external_shape_reasons
        or "article_internal_offload" not in external_query_flags
    ) and bool(status_counts["supported"] or status_upgrades or newly_supported)

    material_reasons: List[str] = []
    if newly_supported:
        material_reasons.append("closed_supported_gap")
    if status_upgrades:
        material_reasons.append("upgraded_existing_gap")
    if meaningful_external_grounding and mix["external_queries"] and mix["external_sources"]:
        material_reasons.append("added_external_grounding")

    if material_reasons:
        non_material_reason = ""
    elif not new_canonical:
        non_material_reason = "no_new_evidence_entries"
    elif status_counts["missing"] + status_counts["premise_mismatch"] == len(new_canonical):
        non_material_reason = "missing_only"
    elif status_counts["not_in_source"] + status_counts["missing"] + status_counts["premise_mismatch"] == len(new_canonical):
        non_material_reason = "absence_or_premise_mismatch_only"
    else:
        non_material_reason = "restated_known_state_without_useful_change"

    return {
        "new_entries": len(new_canonical),
        "status_counts": status_counts,
        "status_upgrades": status_upgrades,
        "newly_supported": newly_supported,
        "newly_bounded_absence": newly_bounded_absence,
        "material_improvement": bool(material_reasons),
        "material_reasons": material_reasons,
        "non_material_reason": non_material_reason,
        "source_mix": mix,
    }


_MISSING_ANSWER_PHRASES = (
    "not found in retrieved context",
    "not specified in text",
    "unable to find information",
    "document may not contain this specific information",
    "document does not appear to contain information semantically related",
    "no information was retrieved",
    "not reported in the retrieved excerpts",
)
