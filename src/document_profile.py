"""Document profiling helpers for document-aware research routing."""

from __future__ import annotations

import re
from typing import Any


_GENERIC_TITLE_LINES = {
    "original article",
    "preface",
    "abstract",
    "introduction",
    "contains nonbinding recommendations",
    "guidance for industry and food and drug administration staff",
}

_ENTITY_STOPWORDS = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "Today",
    "Methods",
    "Results",
    "Conclusions",
    "Background",
    "Introduction",
    "Abstract",
    "Clinical",
    "Guidance",
    "Software",
    "Models",
    "Apps",
    "Harnesses",
    "Title",
    "Topic",
    "Key",
    "Claims",
    "Findings",
    "January",
}


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_title_from_summary(article_summary: str) -> str:
    summary = str(article_summary or "")
    title_window_match = re.search(r"Title/Topic.*?(?:\n\n|Key Claims|Key claims|$)", summary, flags=re.IGNORECASE | re.DOTALL)
    title_window = title_window_match.group(0) if title_window_match else summary[:400]
    normalized_window = normalize_space(re.sub(r"^\*\*?Title/Topic\*\*?\s*", "", title_window, flags=re.IGNORECASE))

    if normalized_window.lower().startswith(("this article is", "the article is", "this source is")):
        return ""

    quoted = re.search(r"[“\"]([^”\"]{5,160})[”\"]", title_window[:140], flags=re.IGNORECASE)
    if quoted:
        return normalize_space(quoted.group(1))

    titled = re.search(r"Title/Topic\s*[:\*]*\s*([^\n]+)", summary, flags=re.IGNORECASE)
    if titled:
        line = normalize_space(titled.group(1))
        line = re.sub(r"^[\-\*\s]+", "", line)
        if line.lower().startswith(("this article is", "the article is", "this source is")):
            return ""
        if line and len(line) <= 220:
            return line

    return ""


def extract_plausible_title_lines(article_content: str) -> list[str]:
    cleaned = re.sub(r"<!--\s*page:\s*\d+\s*-->", "", str(article_content or ""))
    lines = [normalize_space(line.lstrip("#").strip()) for line in cleaned.splitlines() if normalize_space(line)]
    titles: list[str] = []
    for line in lines[:40]:
        lowered = line.lower()
        if lowered in _GENERIC_TITLE_LINES:
            continue
        if len(line) < 12 or len(line) > 220:
            continue
        if re.fullmatch(r"[A-Z][A-Z\s/&\-]+", line) and len(line.split()) <= 5:
            continue
        if re.search(r"\b(?:doi|received|accepted|published)\b", lowered):
            continue
        titles.append(line)
    return titles


def _detect_issuer(article_summary: str) -> str:
    text = str(article_summary or "")
    lowered = text.lower()
    issuer_map = {
        "fda": ("Food and Drug Administration", (" food and drug administration", " fda ")),
        "cms": ("Centers for Medicare & Medicaid Services", (" cms ", " centers for medicare & medicaid services")),
        "nih": ("National Institutes of Health", (" national institutes of health", " nih ")),
        "nejm": ("NEJM", (" nejm ", "new england journal of medicine")),
    }
    padded = f" {lowered} "
    for issuer_name, hints in issuer_map.values():
        if any(hint in padded for hint in hints):
            return issuer_name
    return ""


def _extract_anchor_terms(article_summary: str, title: str) -> list[str]:
    text = f"{title}\n{article_summary}"
    candidates: list[str] = []

    for match in re.finditer(r"[“\"]([^”\"]{3,90})[”\"]", text):
        phrase = normalize_space(match.group(1))
        if phrase and phrase not in candidates:
            candidates.append(phrase)

    entity_pattern = re.compile(
        r"\b(?:[A-Z][a-z]+|[A-Z]{2,}|[A-Z][a-zA-Z0-9.-]+)"
        r"(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|[A-Z][a-zA-Z0-9.-]+|of|for|and|the|to)){0,6}\b"
    )
    for match in entity_pattern.finditer(text):
        phrase = normalize_space(match.group(0).strip(" ,.;:()[]"))
        if not phrase or len(phrase) < 4 or len(phrase) > 80:
            continue
        normalized_phrase = phrase.strip(" ,.;:").lower()
        if phrase.split()[0] in _ENTITY_STOPWORDS:
            continue
        if normalized_phrase in {"title/topic", "key claims/findings", "title", "topic", "key claims", "key findings"}:
            continue
        if phrase not in candidates:
            candidates.append(phrase)

    return candidates[:10]


def _looks_regulatory(text: str) -> bool:
    return any(
        hint in text
        for hint in (
            "guidance for industry",
            "nonbinding recommendations",
            "section 520(o)",
            "supersedes",
            "food and drug administration",
            "fda guidance",
            " fda ",
            "final rule",
            "draft guidance",
            "guidance document",
            "public comment",
        )
    )


def _looks_product_or_vendor(text: str) -> bool:
    return any(
        hint in text
        for hint in (
            "product",
            "vendor",
            "launch",
            "preview",
            "pricing",
            "app",
            "apps",
            "models",
            "harnesses",
            "chatbot",
            "agentic",
            "openai",
            "anthropic",
            "google",
            "claude",
            "gpt-",
            "gemini",
        )
    )


def _looks_methods_model_paper(text: str) -> bool:
    return any(
        hint in text
        for hint in (
            "benchmark",
            "cross-validation",
            "dataset",
            "model performance",
            "auc",
            "predictive model",
            "framework",
            "we propose",
            "algorithm",
            "classifier",
        )
    )


def _looks_scholarly_cues(text: str) -> bool:
    return any(
        hint in text
        for hint in (
            "abstract",
            "methods",
            "results",
            "conclusions",
            "doi",
            "prospective",
            "observational",
            "trial",
            "quality improvement",
            "cohort",
            "randomized",
        )
    )


def _looks_trial_or_comparative(text: str) -> bool:
    return any(
        hint in text
        for hint in (
            "randomized",
            "controlled trial",
            "comparative effectiveness",
            "versus",
            "compared with",
            "trial",
            "intervention group",
            "control group",
        )
    )


def _looks_real_world_evaluation(text: str) -> bool:
    return any(
        hint in text
        for hint in (
            "real-world",
            "quality improvement",
            "prospective",
            "observational",
            "service evaluation",
            "implementation study",
            "embedded in",
            "actual behavior",
        )
    )


def infer_document_profile(
    article_summary: str,
    newsletter_template: str = "",
    draft_report: str = "",
) -> dict[str, Any]:
    summary = str(article_summary or "")
    combined = normalize_space(f"{summary}\n{draft_report}").lower()
    template = normalize_space(newsletter_template).lower()

    if _looks_regulatory(combined):
        source_kind = "regulatory_guidance"
    elif template == "commentary":
        source_kind = "commentary_product_update" if _looks_product_or_vendor(combined) else "commentary_analysis"
    elif template == "research_article":
        if _looks_real_world_evaluation(combined):
            source_kind = "research_real_world_evaluation"
        elif _looks_trial_or_comparative(combined):
            source_kind = "research_clinical_or_comparative"
        elif _looks_methods_model_paper(combined):
            source_kind = "research_methods_or_model"
        else:
            source_kind = "research_general"
    elif _looks_product_or_vendor(combined) and not _looks_scholarly_cues(combined):
        source_kind = "commentary_product_update"
    elif _looks_real_world_evaluation(combined):
        source_kind = "research_real_world_evaluation"
    elif _looks_trial_or_comparative(combined):
        source_kind = "research_clinical_or_comparative"
    elif _looks_methods_model_paper(combined):
        source_kind = "research_methods_or_model"
    else:
        source_kind = "research_general"

    title = extract_title_from_summary(summary)
    issuer = _detect_issuer(summary)
    anchors = _extract_anchor_terms(summary, title)

    domains: list[str] = []
    if issuer == "Food and Drug Administration":
        domains.append("fda.gov")
    elif issuer == "Centers for Medicare & Medicaid Services":
        domains.append("cms.gov")
    elif issuer == "National Institutes of Health":
        domains.append("nih.gov")

    if source_kind == "regulatory_guidance":
        external_strategy = "issuer_first_official_context"
        internal_focuses = [
            "what changed or what boundary the guidance defines",
            "which exact criteria, scope limits, and examples control interpretation",
            "what implementation caveats or nonbinding limits matter for operators",
        ]
        external_focuses = [
            "official issuer pages, FAQs, town halls, or linked standards",
            "closely related public guidance or implementation context",
        ]
    elif source_kind == "commentary_product_update":
        external_strategy = "entity_and_comparator_grounding"
        internal_focuses = [
            "the article's main claim and named examples",
            "how the article defines its key categories or product layers",
            "which caveats or recommendations matter most for adoption",
        ]
        external_focuses = [
            "issuer statements, release notes, or product pages",
            "credible comparators or surrounding market reactions",
        ]
    elif source_kind == "commentary_analysis":
        external_strategy = "named_examples_and_surrounding_context"
        internal_focuses = [
            "the article's thesis and supporting examples",
            "the definitions, contrasts, or recommendations that make the thesis concrete",
            "which caveats, missing evidence, or uncertainty temper the argument",
        ]
        external_focuses = [
            "named entities, comparators, reactions, or timeline context",
            "broader field developments that sharpen interpretation",
        ]
    elif source_kind == "research_methods_or_model":
        external_strategy = "comparator_and_field_context"
        internal_focuses = [
            "the main method or framework contribution",
            "the reported evaluation setup and decisive metrics",
            "the methodological caveats and deployment limits",
        ]
        external_focuses = [
            "related comparator studies, benchmarks, or governance expectations",
        ]
    elif source_kind == "research_clinical_or_comparative":
        external_strategy = "clinical_comparator_context"
        internal_focuses = [
            "the intervention, comparator, population, and outcome",
            "the decisive quantitative results and operationally relevant subgroup findings",
            "the trust boundaries and generalizability limits",
        ]
        external_focuses = [
            "related studies, practice context, or comparator standards",
        ]
    else:
        external_strategy = "real_world_and_comparator_context"
        internal_focuses = [
            "the intervention or service being evaluated",
            "the real-world outcomes and decisive numbers",
            "the study design limits and transferability constraints",
        ]
        external_focuses = [
            "surrounding real-world evidence, adoption context, or comparators",
        ]

    return {
        "source_kind": source_kind,
        "title": title,
        "issuer": issuer,
        "anchor_terms": anchors,
        "preferred_domains": domains,
        "external_strategy": external_strategy,
        "internal_focuses": internal_focuses,
        "external_focuses": external_focuses,
    }


def format_document_profile(profile: dict[str, Any]) -> str:
    if not profile:
        return "(unavailable)"
    parts = [
        f"source_kind={profile.get('source_kind', '') or 'unknown'}",
    ]
    if profile.get("title"):
        parts.append(f"title={profile['title']}")
    if profile.get("issuer"):
        parts.append(f"issuer={profile['issuer']}")
    anchors = ", ".join(profile.get("anchor_terms", [])[:5])
    if anchors:
        parts.append(f"anchors={anchors}")
    if profile.get("external_strategy"):
        parts.append(f"external_strategy={profile['external_strategy']}")
    return " | ".join(parts)
